"""How long each arm of a run would take on this machine, from a few fits of its model on the run's
data, so that a slow arm can be put to the user before the run starts.

    $CORE experiments/runners/run_experiment.py --config <config> --estimate

**What is measured, per arm.** Its pool is built on the fitting rows as a Stage 2 cell builds it:
the enumerated sources and the covariates in full, and each source a language model writes as
stand-in columns of the width a proposal has (`stand_in_width`), since what the model writes cannot
be had without calling it. The arm's model is then fitted on one thread, as each Stage 2 process
fits, on a sample of the fitting rows that doubles until one fit takes `ENOUGH_SECONDS` or uses
every row, and an arm stops probing once its next fit would run past `PROBE_SECONDS`. A pool wider
than `PROBE_COLUMNS` is fitted on that many of its columns, drawn at random, and the time scaled to
its width, except for PySR and DEFT's tree (below), which are given every column. The last fit's
time is scaled to every fitting row along the line through the last two fits, so that what a fit
costs whatever its rows is not multiplied with them, or in proportion to the rows where the first
fit was the last (`scale_rows`).
- **PySR**'s first search in a process starts Julia, so a search of one iteration goes first,
  untimed. The timed search runs `PYSR_PROBE_ITERATIONS` iterations in place of a full search's
  count, and its time is scaled to the full count, since each iteration evolves every population
  for as long. The search evolves on batches of rows above `PYSR_BATCH_ROWS`
  (`seqmaestro.models.symbolic`), so it is timed once, on at most that many rows, and its time is
  scaled to the rows only up to that many.
- **DEFT's tree** is timed on DEFT's side (`experiments/adapters/deft_cart.py time`), growing CART
  on the pool's positions and handed columns. On a pool with `llm_generated_joint` this leaves out
  the candidates DEFT's language model writes at each node, which are few beside the positional
  ones.
- **IGSR**'s time is its language model's, so nothing of it is measured.

**What is estimated from it** (`estimate_arm`):
- **Stage 2**: for every labels and seed, building the pool's columns once and fitting the model
  once, or `1 + feedback_rounds` times on the iterative source;
- **the reduction**: one fit per step of `eliminate.schedule`, from the columns the probe's fit used
  down to N, and the top-N control. A fit's time scales with its columns for every model but PySR,
  whose search costs about the same at any width;
- **the panel**: `2 * pairs` selections, each fitting the whole pool on half of the training and
  validation rows and eliminating from what that fit used, three sweeps of them on the check arm
  (the real labels, the null and the planted signal), and none for an arm the panel leaves out;
- **the wall time** of each of these stages: its CPU time spread over the run's threads, and never
  less than its longest single process. An arm's wall time adds its three stages, as they follow
  one another, and supposes the arm has the machine to itself, which no arm has, so the run's total
  is given beside it.

**Not estimated:** the time a language model takes to answer, which waits on the network and costs
calls to measure. The table marks each arm whose time also has it. Nor is the work each process does
whatever its model, such as starting, scoring a cell, and the panel's permutations of the target,
which costs little beside the fits wherever an arm's time runs to hours.

**Slow.** An arm whose wall time exceeds `--slow-hours` is flagged, with what can be done about it.
The driver prints the table and leaves the choice to whoever launches the run (`process/suite.md`).
Inside a run directory the estimate is also written to `logs/estimate.json`, so a finished run can
be checked against it.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seqmaestro import arms as registry
from seqmaestro.execution import thread_environment
from seqmaestro.features import LLM_GENERATED, LLM_ITERATIVE, LLM_JOINT, parse_pool, pool_name
from seqmaestro.harness.runner import subsample_training_rows
from seqmaestro.models import make_model
from seqmaestro.pipeline import rows as run_rows
from seqmaestro.pipeline.config import ExperimentConfig
from seqmaestro.pipeline.layout import LOGS_DIR
from seqmaestro.pipeline.repository import ADAPTERS
from seqmaestro.pipeline.stage2 import pool_featuriser
from seqmaestro.pipeline.vendored import deft
from seqmaestro.reduction.eliminate import schedule

#: A fit this long is timed well enough to be scaled to every row.
ENOUGH_SECONDS = 2.0
#: How long one arm's probing may run before its last fit is taken as it is.
PROBE_SECONDS = 60.0
#: The smallest sample of fitting rows an arm is probed on.
FIRST_ROWS = 250
#: The most columns a probe fits on: a wider pool is sampled, and the time scaled to its width.
PROBE_COLUMNS = 500
#: Above about this many rows PySR evolves on batches of them, so its search costs no more.
PYSR_BATCH_ROWS = 1000
#: The iterations of PySR's timed search, scaled to a full search's.
PYSR_PROBE_ITERATIONS = 10
#: The models whose fit costs about the same at any number of columns.
WIDTH_FLAT = frozenset({"pysr"})
#: The default threshold for a slow arm, in hours of wall time, Stage 2 and Stage 3 together.
SLOW_HOURS = 2.0
#: Where a run keeps its estimate.
ESTIMATE_FILE = "estimate.json"


@dataclass(frozen=True)
class Probe:
    """What was measured for one arm."""

    model: str
    pool: str
    #: The columns the model is handed, stand-ins for the written sources included.
    width: int
    #: The columns the probe's last fit used, scaled to the pool's width where it fitted on a sample
    #: of the columns.
    used: int
    #: One fit on every fitting row, on one thread, in seconds, scaled from `probed_rows`.
    fit_seconds: float
    #: Building the pool's columns for one cell: the fitting, validation and training rows.
    featurise_seconds: float
    probed_rows: int
    fitting_rows: int
    #: Whether the arm also waits on a language model, whose time is not estimated.
    language_model: bool


@dataclass(frozen=True)
class ArmEstimate:
    """One arm's estimated cost: CPU seconds per stage, and the wall seconds of all three."""

    model: str
    pool: str
    width: int
    fit_seconds: float
    stage2: float
    reduction: float
    panel: float
    wall: float
    language_model: bool
    slow: bool = False

    @property
    def arm(self) -> str:
        return f"{self.model}/{self.pool}"


# ------------------------------------------------------------------------------------ arithmetic


def stand_in_width(member: str, config: ExperimentConfig) -> int:
    """How many columns a written source hands the model: a proposal's features, over every round
    for the iterative source. The joint source is written inside its model's fit, so it hands none
    that this side builds."""
    proposer = config.sources.proposer
    if member == LLM_GENERATED:
        return proposer.proposal_rounds * proposer.n_per_round
    if member == LLM_ITERATIVE:
        return (1 + config.sources.feedback_rounds) * proposer.proposal_rounds * proposer.n_per_round
    return 0


def elimination_widths(start: int, n_target: int) -> list[int]:
    """The columns of each fit an elimination from `start` columns down to `n_target` makes, the
    first and the last included (`eliminate.schedule`)."""
    widths = [start]
    while widths[-1] > n_target:
        widths.append(widths[-1] - schedule(widths[-1], n_target))
    return widths


def fit_at(probe: Probe, width: int, rows: float) -> float:
    """The seconds of one fit of the arm's model on `width` of its columns and `rows` rows."""
    if probe.model == "pysr":
        row_share = min(rows, PYSR_BATCH_ROWS) / min(probe.fitting_rows, PYSR_BATCH_ROWS)
    else:
        row_share = rows / probe.fitting_rows
    width_share = 1.0 if probe.model in WIDTH_FLAT else width / max(probe.width, 1)
    return probe.fit_seconds * row_share * width_share


def elimination_seconds(probe: Probe, start: int, n_target: int, rows: float) -> float:
    """An elimination from `start` columns down to `n_target`, and the top-N control."""
    return (sum(fit_at(probe, width, rows) for width in elimination_widths(start, n_target))
            + fit_at(probe, n_target, rows))


def estimate_arm(probe: Probe, config: ExperimentConfig, threads: int, panel_rows: int,
                 slow_hours: float = SLOW_HOURS) -> ArmEstimate:
    """One arm's cost from what its probe measured (the module docstring says how)."""
    arm = (probe.model, probe.pool)
    cells = len(config.grid.labels) * len(config.grid.seeds)
    fits = 1 + config.sources.feedback_rounds if LLM_ITERATIVE in parse_pool(probe.pool) else 1
    cell = probe.featurise_seconds + fits * probe.fit_seconds
    stage2 = cells * cell
    wall = max(stage2 / threads, cell)

    reduction = panel = 0.0
    if config.reduction is not None and arm in config.reduced_arms():
        n_target = config.reduction.n_features
        start = max(min(probe.used, probe.width), n_target)
        reduction = elimination_seconds(probe, start, n_target, probe.fitting_rows)
        wall += reduction
        if arm in config.profiled_arms():
            selection = (fit_at(probe, probe.width, panel_rows)
                         + elimination_seconds(probe, start, n_target, panel_rows))
            sweeps = 3 if arm == config.reliability.check_arm else 1
            panel = 2 * config.reliability.panel.pairs * sweeps * selection
            wall += max(panel / threads, selection)
    return ArmEstimate(model=probe.model, pool=probe.pool, width=probe.width,
                       fit_seconds=probe.fit_seconds, stage2=stage2, reduction=reduction,
                       panel=panel, wall=wall, language_model=probe.language_model,
                       slow=wall > slow_hours * 3600)


def options(estimate: ArmEstimate, config: ExperimentConfig) -> list[str]:
    """What can be done about a slow arm, each with what it saves."""
    found = [f"leave {estimate.model} out of `arms.{estimate.pool}.models`"]
    if estimate.panel:
        found.append(f"leave it out of the panel, `not_profiled: {{{estimate.model}: <reason>}}` in "
                     f"`arms.{estimate.pool}`, which saves {_hours(estimate.panel)} of CPU")
        found.append(f"lower `reliability.pairs` from {config.reliability.panel.pairs}, which the "
                     f"panel's time follows")
    if estimate.reduction:
        found.append(f"leave it out of Stage 3, `not_reduced: {{{estimate.model}: <reason>}}`, which "
                     f"saves {_hours(estimate.reduction + estimate.panel)} of CPU")
    return found


# ---------------------------------------------------------------------------------------- probes


@contextlib.contextmanager
def one_thread():
    """Fits held to one thread, as each Stage 2 process fits: the models' jobs and the numeric
    libraries' pools."""
    from threadpoolctl import threadpool_limits

    saved = {name: os.environ.get(name) for name in thread_environment(1)}
    os.environ.update(thread_environment(1))
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class Rows:
    """The run's rows, as a Stage 2 cell of its first seed fits on them."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        read = run_rows.Rows.of(config)
        self.dataset, self.train, self.val = read.dataset, read.train, read.val
        self.fit_rows, self.y_fit = subsample_training_rows(
            self.train, seed=config.grid.seeds[0], columns=self.dataset.columns,
            train_fraction=config.grid.train_fraction, permute_labels=False,
            permutation_key=None)
        weight = self.dataset.columns.weight
        self.weights = None if weight is None else self.fit_rows[weight].to_numpy(dtype=float)
        self._built: dict[str, tuple[pd.DataFrame, float]] = {}

    @property
    def panel_rows(self) -> int:
        """Half of the training and validation rows, as each of Tier 1's selections fits on."""
        return (len(self.train) + len(self.val)) // 2

    def columns(self, pool: str) -> tuple[pd.DataFrame, float]:
        """The pool's columns on the fitting rows, stand-ins included, and the seconds it takes a
        cell to build them. Built once per pool, for every model fitted on it."""
        if pool not in self._built:
            members = parse_pool(pool)
            built = [member for member in members if member not in (LLM_GENERATED, LLM_ITERATIVE,
                                                                    LLM_JOINT)]
            parts, seconds = [], 0.0
            if built:
                featuriser = pool_featuriser(self.config, self.dataset, pool_name(built))
                columns = self.dataset.columns_for(pool)
                started = time.perf_counter()
                parts.append(featuriser.fit_transform(columns.inputs(self.fit_rows), self.y_fit)
                             .reset_index(drop=True))
                # A cell also builds them on the validation rows and on every training row.
                seconds = ((time.perf_counter() - started)
                           * (len(self.fit_rows) + len(self.val) + len(self.train)) / len(self.fit_rows))
            rng = np.random.default_rng(0)
            for member in members:
                width = stand_in_width(member, self.config)
                if width:
                    parts.append(pd.DataFrame(rng.normal(size=(len(self.fit_rows), width)),
                                              columns=[f"{member}_{i}" for i in range(width)]))
            self._built[pool] = (pd.concat(parts, axis=1) if parts else pd.DataFrame(), seconds)
        return self._built[pool]


def sample_sizes(total: int) -> list[int]:
    """The samples of fitting rows a probe tries, smallest first, ending at every row."""
    sizes = [min(FIRST_ROWS, total)]
    while sizes[-1] < total:
        sizes.append(min(2 * sizes[-1], total))
    return sizes


def timed_fits(fit, sizes: list[int]) -> list[tuple[int, float, int]]:
    """Fit on growing samples until one fit takes `ENOUGH_SECONDS`, uses every row, or the next
    would run past `PROBE_SECONDS`. `fit(rows)` fits on the first `rows` rows and returns how many
    columns it used. Returns each fit's rows, seconds and columns used."""
    fits, spent = [], 0.0
    for rows in sizes:
        started = time.perf_counter()
        used = fit(rows)
        seconds = time.perf_counter() - started
        spent += seconds
        fits.append((rows, seconds, used))
        if seconds >= ENOUGH_SECONDS or rows == sizes[-1] or spent + 2 * seconds > PROBE_SECONDS:
            return fits
    raise AssertionError("unreachable: the last size always returns")


def scale_rows(fits: list[tuple[int, float, int]], total: int) -> float:
    """One fit's seconds on `total` rows, from fits on growing samples (`timed_fits`): along the
    line through the last two, or in proportion to the rows where there was one fit. The line's
    slope is never taken below zero, since a fit on more rows does not take less time."""
    rows, seconds, _ = fits[-1]
    if rows == total:
        return seconds
    if len(fits) == 1:
        return seconds * total / rows
    before_rows, before_seconds, _ = fits[-2]
    return seconds + max(0.0, (seconds - before_seconds) / (rows - before_rows)) * (total - rows)


def probe_core(model: str, pool: str, rows: Rows) -> Probe:
    """One of Core's models, fitted in this process on one thread."""
    X, featurise_seconds = rows.columns(pool)
    width = X.shape[1]
    if width > PROBE_COLUMNS and model not in WIDTH_FLAT:
        X = X.iloc[:, np.sort(np.random.default_rng(0).choice(width, PROBE_COLUMNS, replace=False))]
    values, y = X.to_numpy(dtype=float), np.asarray(rows.y_fit)
    kind = rows.dataset.target_kind
    weighting = {} if rows.config.grid.class_weight is None else {
        "class_weight": rows.config.grid.class_weight}

    def fit(n: int, **settings) -> int:
        fitted = make_model(model, target_kind=kind, seed=rows.config.grid.seeds[0], **weighting,
                            **settings)
        fitted.fit(pd.DataFrame(values[:n], columns=X.columns), y[:n],
                   sample_weight=None if rows.weights is None else rows.weights[:n])
        return len(fitted.importances())

    with one_thread():
        if model == "pysr":
            # The first search starts Julia, which the run pays once per process and not per fit.
            # Past `PYSR_BATCH_ROWS` rows a search evolves on batches, so it is timed on at most
            # that many, and its time follows its iterations.
            fit(min(FIRST_ROWS, len(values)), niterations=1)
            full = make_model(model, target_kind=kind).niterations
            fits = timed_fits(lambda n: fit(n, niterations=PYSR_PROBE_ITERATIONS),
                              [min(len(values), PYSR_BATCH_ROWS)])
            seconds = fits[-1][1] * full / PYSR_PROBE_ITERATIONS
        else:
            fits = timed_fits(fit, sample_sizes(len(values)))
            seconds = scale_rows(fits, len(values)) * width / X.shape[1]
    return Probe(model=model, pool=pool, width=width,
                 used=min(width, round(fits[-1][2] * width / X.shape[1])), fit_seconds=seconds,
                 featurise_seconds=featurise_seconds, probed_rows=fits[-1][0],
                 fitting_rows=len(values), language_model=registry.calls_language_model(pool))


def probe_tree(pool: str, rows: Rows) -> Probe:
    """DEFT's tree, grown with CART on DEFT's side on the pool's positions and handed columns."""
    cart = pool_name(member for member in parse_pool(pool) if member != LLM_JOINT)
    kmer = rows.config.sources.kmer
    dataset = rows.dataset
    with tempfile.TemporaryDirectory() as scratch:
        inputs = Path(scratch) / "inputs"
        started = time.perf_counter()
        deft.write_inputs(cart, rows.fit_rows, rows.y_fit, rows.train, rows.val, inputs,
                          columns=dataset.columns_for(cart),
                          descriptions=dataset.covariates_for(cart),
                          window=dataset.positional_window,
                          k_range=None if kmer is None else kmer.k_range,
                          kmer_normalise=None if kmer is None else kmer.normalise)
        featurise_seconds = time.perf_counter() - started
        out = Path(scratch) / "timings.json"
        overrides = deft.overrides(rows.config, cart)
        command = [str(deft.DEFT_PYTHON), str(ADAPTERS / "deft_cart.py"),
                   "time", "--inputs", str(inputs), "--target-column", dataset.columns.target,
                   "--target-kind", dataset.target_kind, "--sequence-column", dataset.columns.sequence,
                   "--first-rows", str(FIRST_ROWS), "--enough", str(ENOUGH_SECONDS),
                   "--budget", str(PROBE_SECONDS), "--out", str(out),
                   *(["--weight-column", dataset.columns.weight] if dataset.columns.weight else []),
                   *(part for item in overrides for part in ("--config-override", item))]
        completed = subprocess.run(command, capture_output=True, text=True, check=False,
                                   env={**os.environ, **thread_environment(1)})
        if completed.returncode != 0:
            raise RuntimeError(f"DEFT's side could not time the tree:\n{completed.stderr[-2000:]}")
        timings = json.loads(out.read_text())
    fits = [(fit["rows"], fit["seconds"], fit["used"]) for fit in timings["fits"]]
    return Probe(model=registry.DEFT_OR_CART, pool=pool, width=timings["candidates"],
                 used=fits[-1][2], fit_seconds=scale_rows(fits, len(rows.fit_rows)),
                 featurise_seconds=featurise_seconds, probed_rows=fits[-1][0],
                 fitting_rows=len(rows.fit_rows), language_model=registry.calls_language_model(pool))


def probe_arm(model: str, pool: str, rows: Rows) -> Probe:
    kind = registry.kind(model, pool)
    if kind == registry.DEFT:
        return probe_tree(pool, rows)
    if kind == registry.IGSR_KIND:
        return Probe(model=model, pool=pool, width=0, used=0, fit_seconds=0.0,
                     featurise_seconds=0.0, probed_rows=0, fitting_rows=len(rows.fit_rows),
                     language_model=True)
    return probe_core(model, pool, rows)


# ---------------------------------------------------------------------------------------- output


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 360 else f"{seconds / 60:.0f} min"


def estimate_run(config: ExperimentConfig, threads: int, slow_hours: float = SLOW_HOURS,
                 report=print) -> tuple[list[Probe], list[ArmEstimate]]:
    """Probe every arm of the config and estimate its cost, slowest first. `report` is told each
    arm as its probe finishes, so that a caller can show progress while the others are probed."""
    rows = Rows(config)
    probes, estimates = [], []
    for model, pool in config.arm_cells():
        started = time.perf_counter()
        probe = probe_arm(model, pool, rows)
        probes.append(probe)
        estimates.append(estimate_arm(probe, config, threads, rows.panel_rows, slow_hours))
        report(f"  probed {model}/{pool} in {time.perf_counter() - started:.0f} s")
    return probes, sorted(estimates, key=lambda estimate: -estimate.wall)


def table(estimates: list[ArmEstimate], config: ExperimentConfig, threads: int,
          slow_hours: float) -> str:
    """The estimate as the driver prints it."""
    lines = [f"Estimated on {threads} threads; CPU time per stage, and wall time if the arm had the "
             f"machine to itself.",
             f"{'arm':48} {'columns':>8} {'one fit':>8} {'stage 2':>8} {'reduce':>8} {'panel':>8} "
             f"{'wall':>8}"]
    for estimate in estimates:
        lines.append(
            f"{estimate.arm:48} {estimate.width:>8} {estimate.fit_seconds:>7.1f}s "
            f"{_hours(estimate.stage2):>8} {_hours(estimate.reduction):>8} "
            f"{_hours(estimate.panel):>8} {_hours(estimate.wall):>8}"
            + ("  + language model" if estimate.language_model else "")
            + ("  SLOW" if estimate.slow else ""))
    total = sum(estimate.stage2 + estimate.reduction + estimate.panel for estimate in estimates)
    lines.append(f"All arms: {_hours(total)} of CPU, about {_hours(total / threads)} on {threads} "
                 f"threads, before any language model's time.")
    slow = [estimate for estimate in estimates if estimate.slow]
    if slow:
        lines.append(f"\n{len(slow)} arm(s) over {slow_hours:g} h, each to be decided before the "
                     f"run starts:")
        for estimate in slow:
            lines.append(f"  {estimate.arm}, about {_hours(estimate.wall)}:")
            lines += [f"    - {option}" for option in options(estimate, config)]
    return "\n".join(lines)


def record(run_dir: Path, probes: list[Probe], estimates: list[ArmEstimate], threads: int,
           slow_hours: float) -> Path:
    """Keep the estimate in the run, so its processes' times can be checked against it."""
    logs = run_dir / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / ESTIMATE_FILE
    path.write_text(json.dumps({
        "threads": threads, "slow_hours": slow_hours,
        "probes": [asdict(probe) for probe in probes],
        "arms": [{**asdict(estimate), "arm": estimate.arm} for estimate in estimates],
    }, indent=2) + "\n")
    return path
