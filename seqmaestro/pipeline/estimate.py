"""How long each arm of a run would take on this machine, from a few fits of its model on the run's
data, so that a slow arm can be raised with the user before the run starts.

    $CORE experiments/runners/run_experiment.py --config <config> --estimate

**What is measured for each arm.** The arm's pool is built on the fitting rows as a Stage 2 cell
builds it: the fixed sources (the covariates among them) in full, and each written source as random
stand-in columns as wide as a proposal (`stand_in_width`), because the features a language model
would write cannot be had without calling it. The arm's model is then fitted on one thread, as in a
Stage 2 process, on a sample of the fitting rows that doubles until one fit takes `ENOUGH_SECONDS`
or uses every row. An arm stops probing once its next fit would run past `PROBE_SECONDS`. A pool
wider than `PROBE_COLUMNS` is fitted on that many of its columns, drawn at random, and the time is
scaled to its full width, except for PySR and DEFT's tree (below), which are given every column.
The last fit's time is scaled to every fitting row along the line through the last two fits, so
that a fit's fixed overhead is not multiplied with the rows, or in proportion to the rows where the
first fit was also the last (`scale_rows`).
- **A source the run makes in a step before the arms** (`SCANNED_ONCE`) is built once for all the
  arms whose pools hold it. Its time is scaled to the training and validation rows (and, for the
  discovered motifs, to every label condition and seed) and reported as that step's time instead
  of being charged to each cell, because each cell finds the scans and the discoveries in the run's
  store.
- **PySR**'s first search in a process starts Julia, so an untimed search of one iteration goes
  first. The timed search runs `PYSR_PROBE_ITERATIONS` iterations instead of a full search's count,
  and its time is scaled to the full count, because each iteration evolves every population for
  the same length of time. Above `PYSR_BATCH_ROWS` rows the search evolves on batches
  (`seqmaestro.models.symbolic`), so it is timed once, on at most that many rows, and its time is
  scaled with the rows only up to that many.
- **DEFT's tree** is timed on DEFT's side (`experiments/adapters/deft_cart.py time`), which grows
  CART on the pool's positions and the columns Core hands it. On a pool with `llm_generated_joint`,
  this leaves out the candidates DEFT's language model writes at each node, which are few compared
  with the positional ones. Handed motif columns are timed with their scan, which a run makes only
  once, so such an arm's Stage 2 time is overstated by about one scan per cell.
- **IGSR**'s time is spent waiting on its language model, so none of it is measured.

**What is estimated from the measurements** (`estimate_arm`):
- **Stage 2**: for every label condition and seed, building the pool's columns once and fitting
  the model once, or `1 + feedback_rounds` times on the iterative source;
- **the reduction**: one fit per step of `eliminate.schedule`, from the columns the probe's fit used
  down to N, and the top-N control. A fit's time scales with its columns for every model except
  PySR, whose search takes about the same time at any width;
- **the panel**: `2 * pairs` selections, each of which fits the whole pool on half of the training
  and validation rows and eliminates from the columns that fit used. The check arm takes three
  sweeps of them (the real labels, the null and the planted signal), and an arm the panel leaves
  out takes none;
- **the wall time** of each of these stages: its CPU time spread over the run's threads, and never
  less than its longest single process. An arm's wall time is the sum of its three stages, because
  they follow one another, and it supposes that the arm has the machine to itself. Since the arms
  share the machine, the run's total is given beside it.

**The language model's calls are estimated apart** (`seqmaestro.pipeline.calls`): how many each
arm makes, what they cost and how long they wait, measured with two small calls, and the load they
put on the provider. The driver prints them after this table, with the grid of every arm's time and
cost. **Not estimated:** the work each process does whatever its model (starting, scoring a cell,
and the panel's permutations of the target), because it is small compared with the fits wherever an
arm's time runs to hours.

**Slow arms.** An arm whose wall time exceeds `--slow-hours` is flagged, with the options for it.
The driver prints the table and leaves the choice to whoever launches the run (`process/suite.md`).
When a run directory is named, the estimate is also written to its `logs/estimate.json`, so that a
finished run can be checked against it.
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
from seqmaestro.features import (
    LLM_GENERATED,
    LLM_ITERATIVE,
    LLM_JOINT,
    MOTIF_DISCOVERED,
    MOTIF_KNOWN,
    parse_pool,
    pool_name,
)
from seqmaestro.harness.runner import subsample_training_rows
from seqmaestro.models import make_model
from seqmaestro.pipeline import rows as run_rows
from seqmaestro.pipeline.config import ExperimentConfig
from seqmaestro.pipeline.layout import LOGS_DIR
from seqmaestro.pipeline.repository import ADAPTERS
from seqmaestro.pipeline.sources import pool_featuriser
from seqmaestro.pipeline.vendored import deft
from seqmaestro.reduction.eliminate import schedule

#: A fit this long is timed well enough to be scaled to every row.
ENOUGH_SECONDS = 2.0
#: How long one arm's probing may run before its last fit is taken as it is.
PROBE_SECONDS = 60.0
#: The smallest sample of fitting rows an arm is probed on.
FIRST_ROWS = 250
#: The most columns a probe fits on: a wider pool is sampled, and the time is scaled to its width.
PROBE_COLUMNS = 500
#: Above about this many rows PySR evolves on batches of them, so its search time stops growing.
PYSR_BATCH_ROWS = 1000
#: The iterations of PySR's timed search, whose time is scaled to a full search's count.
PYSR_PROBE_ITERATIONS = 10
#: The models whose fit takes about the same time at any number of columns.
WIDTH_FLAT = frozenset({"pysr"})
#: The default threshold for a slow arm, in hours of wall time over Stage 2 and Stage 3 together.
SLOW_HOURS = 2.0
#: Where a run keeps its estimate.
ESTIMATE_FILE = "estimate.json"
#: The sources a run computes in a step before the arms, each with what the step does.
SCANNED_ONCE = {MOTIF_KNOWN: "scanning the training and validation rows with the motif collection",
                MOTIF_DISCOVERED: "discovering motifs with STREME, and scanning with them, for "
                                  "every label condition and seed"}


@dataclass(frozen=True)
class Probe:
    """What was measured for one arm."""

    model: str
    pool: str
    #: The number of columns the model is handed, including the stand-ins for the written sources.
    width: int
    #: The number of columns the probe's last fit used, scaled to the pool's width where it fitted
    #: on a sample of the columns.
    used: int
    #: The seconds of one fit on every fitting row on one thread, scaled from `probed_rows`.
    fit_seconds: float
    #: The seconds a cell takes to build the pool's columns on the fitting, validation and training
    #: rows.
    featurise_seconds: float
    probed_rows: int
    fitting_rows: int
    #: Whether the arm also waits on a language model, whose calls `seqmaestro.pipeline.calls`
    #: estimates.
    language_model: bool
    #: For DEFT's tree, the splits it made on the most rows probed, or None for another model.
    nodes: int | None = None


@dataclass(frozen=True)
class ArmEstimate:
    """One arm's estimated time: the CPU seconds of each stage and the wall seconds of all three."""

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
    """How many columns a written source hands the model: a proposal's features, summed over every
    round for the iterative source. The joint source is written inside its model's fit, so Core
    builds no columns for it."""
    proposer = config.sources.proposer
    if member == LLM_GENERATED:
        return proposer.proposal_rounds * proposer.n_per_round
    if member == LLM_ITERATIVE:
        return (1 + config.sources.feedback_rounds) * proposer.proposal_rounds * proposer.n_per_round
    return 0


def elimination_widths(start: int, n_target: int) -> list[int]:
    """The number of columns of each fit that an elimination from `start` columns down to
    `n_target` makes, including the first and the last (`eliminate.schedule`)."""
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
    """The seconds of an elimination from `start` columns down to `n_target`, and of the top-N
    control."""
    return (sum(fit_at(probe, width, rows) for width in elimination_widths(start, n_target))
            + fit_at(probe, n_target, rows))


def estimate_arm(probe: Probe, config: ExperimentConfig, threads: int, panel_rows: int,
                 slow_hours: float = SLOW_HOURS) -> ArmEstimate:
    """One arm's estimated time from what its probe measured (the module docstring says how)."""
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
    """The options for a slow arm, each with the time it saves."""
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
    """Hold fits to one thread, as in each Stage 2 process, for both the models' jobs and the
    numeric libraries' thread pools."""
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
    """The run's rows, with the fitting rows drawn as for a Stage 2 cell at the first seed."""

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
        self._once: dict[str, pd.DataFrame] = {}
        #: The CPU seconds of each step a run makes once before the arms, keyed by what it does.
        self.before: dict[str, float] = {}

    @property
    def panel_rows(self) -> int:
        """Half of the training and validation rows, the number each of Tier 1's selections fits
        on."""
        return (len(self.train) + len(self.val)) // 2

    def columns(self, pool: str) -> tuple[pd.DataFrame, float]:
        """The pool's columns on the fitting rows (stand-ins included) and the seconds a cell takes
        to build them. They are built once per pool for every model fitted on it."""
        if pool not in self._built:
            members = parse_pool(pool)
            built = [member for member in members if member not in (LLM_GENERATED, LLM_ITERATIVE,
                                                                    LLM_JOINT, *SCANNED_ONCE)]
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
            parts += [self.scanned_once(member) for member in members if member in SCANNED_ONCE]
            rng = np.random.default_rng(0)
            for member in members:
                width = stand_in_width(member, self.config)
                if width:
                    parts.append(pd.DataFrame(rng.normal(size=(len(self.fit_rows), width)),
                                              columns=[f"{member}_{i}" for i in range(width)]))
            self._built[pool] = (pd.concat(parts, axis=1) if parts else pd.DataFrame(), seconds)
        return self._built[pool]

    def scanned_once(self, member: str) -> pd.DataFrame:
        """The columns on the fitting rows of a source the run scans once, built once for all the
        arms. The time a run takes to scan the training and validation rows with it is added to
        `before`."""
        if member not in self._once:
            featuriser = pool_featuriser(self.config, self.dataset, member)
            started = time.perf_counter()
            self._once[member] = featuriser.fit_transform(
                self.dataset.columns_for(member).inputs(self.fit_rows), self.y_fit
            ).reset_index(drop=True)
            grid = self.config.grid
            times = len(grid.labels) * len(grid.seeds) if member == MOTIF_DISCOVERED else 1
            self.before[SCANNED_ONCE[member]] = ((time.perf_counter() - started) * times
                                                 * (len(self.train) + len(self.val))
                                                 / len(self.fit_rows))
        return self._once[member]


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
    """Probe one of Core's models, fitted in this process on one thread."""
    X, featurise_seconds = rows.columns(pool)
    width = X.shape[1]
    if width > PROBE_COLUMNS and model not in WIDTH_FLAT:
        X = X.iloc[:, np.sort(np.random.default_rng(0).choice(width, PROBE_COLUMNS, replace=False))]
    values, y = X.to_numpy(dtype=float), np.asarray(rows.y_fit)
    kind = rows.dataset.target_kind
    weighting = {} if rows.config.grid.class_weight is None else {
        "class_weight": rows.config.grid.class_weight}
    # The settings the run's config gives the model, which a probe's own setting (PySR's shortened
    # search) overrides.
    configured = rows.config.settings_for(model)

    def fit(n: int, **settings) -> int:
        fitted = make_model(model, target_kind=kind, seed=rows.config.grid.seeds[0], **weighting,
                            **{**configured, **settings})
        fitted.fit(pd.DataFrame(values[:n], columns=X.columns), y[:n],
                   sample_weight=None if rows.weights is None else rows.weights[:n])
        return len(fitted.importances())

    with one_thread():
        if model == "pysr":
            # The first search starts Julia, which happens once per process instead of once per
            # fit. Past `PYSR_BATCH_ROWS` rows a search evolves on batches, so it is timed on at
            # most that many, and its time is scaled by the number of iterations.
            fit(min(FIRST_ROWS, len(values)), niterations=1)
            full = make_model(model, target_kind=kind, **configured).niterations
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
    """Probe DEFT's tree, grown with CART on DEFT's side on the pool's positions and the columns
    Core hands it."""
    cart = pool_name(member for member in parse_pool(pool) if member != LLM_JOINT)
    dataset = rows.dataset
    with tempfile.TemporaryDirectory() as scratch:
        inputs = Path(scratch) / "inputs"
        started = time.perf_counter()
        deft.write_inputs(cart, rows.fit_rows, rows.y_fit, rows.train, rows.val, inputs,
                          columns=dataset.columns_for(cart),
                          descriptions=dataset.covariates_for(cart),
                          window=dataset.positional_window,
                          supplied=deft.supplied_featuriser(rows.config, dataset, cart, None))
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
                 nodes=timings["fits"][-1].get("nodes"),
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
                 report=print) -> tuple[list[Probe], list[ArmEstimate], dict[str, float]]:
    """Probe every arm of the config and estimate its time. Returns the probes, the estimates
    (slowest first) and the CPU seconds of each step the run makes once before the arms
    (`Rows.before`). `report` is given a line for each arm as its probe finishes, so that a caller
    can show progress while the others are probed."""
    rows = Rows(config)
    probes, estimates = [], []
    for model, pool in config.arm_cells():
        started = time.perf_counter()
        probe = probe_arm(model, pool, rows)
        probes.append(probe)
        estimates.append(estimate_arm(probe, config, threads, rows.panel_rows, slow_hours))
        report(f"  probed {model}/{pool} in {time.perf_counter() - started:.0f} s")
    return probes, sorted(estimates, key=lambda estimate: -estimate.wall), dict(rows.before)


def table(estimates: list[ArmEstimate], config: ExperimentConfig, threads: int,
          slow_hours: float, before: dict[str, float] | None = None) -> str:
    """The estimate as the driver prints it, with each step the run makes once before the arms
    (`before`, CPU seconds by what the step does)."""
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
    for step, seconds in (before or {}).items():
        lines.append(f"Once, before the arms: {step}, {_hours(seconds)} of CPU on one thread.")
    total = (sum(estimate.stage2 + estimate.reduction + estimate.panel for estimate in estimates)
             + sum((before or {}).values()))
    lines.append(f"All arms: {_hours(total)} of CPU, about {_hours(total / threads)} on {threads} "
                 f"threads, before the language model's waiting, which follows.")
    slow = [estimate for estimate in estimates if estimate.slow]
    if slow:
        lines.append(f"\n{len(slow)} arm(s) over {slow_hours:g} h, each to be decided before the "
                     f"run starts:")
        for estimate in slow:
            lines.append(f"  {estimate.arm}, about {_hours(estimate.wall)}:")
            lines += [f"    - {option}" for option in options(estimate, config)]
    return "\n".join(lines)


def record(run_dir: Path, probes: list[Probe], estimates: list[ArmEstimate], threads: int,
           slow_hours: float, before: dict[str, float] | None = None,
           language_model: dict | None = None) -> Path:
    """Keep the estimate in the run, so its processes' times can be checked against it, with the
    language model's part (`seqmaestro.pipeline.calls.as_record`) where it was estimated."""
    logs = run_dir / LOGS_DIR
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / ESTIMATE_FILE
    path.write_text(json.dumps({
        "threads": threads, "slow_hours": slow_hours, "before": dict(before or {}),
        "probes": [asdict(probe) for probe in probes],
        "arms": [{**asdict(estimate), "arm": estimate.arm} for estimate in estimates],
        "language_model": language_model,
    }, indent=2) + "\n")
    return path
