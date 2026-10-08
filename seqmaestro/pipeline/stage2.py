"""Stage 2: the cells of a run's arms.

    python experiments/runners/run_arm.py --run-dir experiments/runs/<date>_<name>
    python experiments/runners/run_arm.py --run-dir ... --arms ebm/positional --labels real --seeds 42
    python experiments/runners/run_arm.py --run-dir ... --arms l1_logistic/llm_generated --show-prompt

A cell is one model fitted on one pool, under one condition of the labels, at one seed
(`seqmaestro.harness.cell`). The driver starts this once per arm, labels and seed, many at once.

**Every setting comes from the run's recorded config**, `<run>/config.yaml`: the grid, how each
source is computed, each model's settings. The command line only narrows which of the config's
cells a call fits. A setting changed in the config therefore reaches every stage, and no call can
fit a cell under a setting the run does not record.

**How a cell is fitted depends on the kind of arm** (`seqmaestro.arms.kind`):

- A Core model on a fixed pool is fitted in this process, on columns that follow from the rows by
  rule. It needs no language model and no credentials.
- A Core model on a pool the proposer writes into is fitted in this process too, after a language
  model has proposed the pool's written features: in one proposal for `llm_generated`, and in
  further proposals, each shown a fit of the cell's model, for `llm_generated_iterative`
  (`seqmaestro.proposer`). A proposal is made for its arm, labels and seed, so two models compared
  on a written source are compared on separate draws. It is kept under `proposals/`, so a cell
  fitted again asks nothing twice.
- DEFT's tree and IGSR are fitted under their methods' interpreters, from files written here, and
  hand back a bundle (`seqmaestro.pipeline.vendored`).

**Credentials** are needed by an arm whose pool holds a written source: `set -a && source .env &&
set +a`. The dataset's description, which every language model is told, is
`experiments/datasets/<task>.json`, the one file the judge and the recurrence analysis also read.

A finished cell is skipped unless `--force` is given. A cell that fails is recorded in the call's
failures file and the others go on. A revealed run is refused.

A DEFT cell needs more care, because a second fit draws another tree, and on a permuted cell a
choice between two trees would choose which draw of the null is kept. An unfinished cell whose
bundle holds no grown tree is fitted again like any other. One whose tree was grown is refused:
`--reuse-tree` completes it from that tree, and only `--force` grows a new one.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from seqmaestro import arms as registry
from seqmaestro.arms import Arm, arm_name, split_arm
from seqmaestro.dataset import Dataset, load_dataset
from seqmaestro.features import (
    LLM_GENERATED,
    LLM_ITERATIVE,
    make_featuriser,
    parse_pool,
    pool_name,
)
from seqmaestro.features.base import Featuriser
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import refuse_if_revealed
from seqmaestro.harness.runner import (
    labels_to_choose_on,
    permutation_key,
    run_model,
    subsample_training_rows,
)
from seqmaestro.llm import LiteLLMClient, TaskContext, blinding_leaks, load_task
from seqmaestro.llm.columns import describe_columns
from seqmaestro.models import make_model
from seqmaestro.pipeline.config import ConfigError, ExperimentConfig, load_recorded
from seqmaestro.pipeline.layout import (
    PROPOSALS_DIR,
    call_name,
    cell_dir,
    target_line,
    validation_line,
)
from seqmaestro.pipeline.records import Failures, snapshot_environments
from seqmaestro.pipeline.repository import DESCRIPTIONS
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.vendored import deft, igsr
from seqmaestro.proposer import (
    IterativeGeneratedFeaturiser,
    ProposalCache,
    ProposalSettings,
    Proposer,
    render_request,
    sample_examples,
)
from seqmaestro.proposer.prompt import SYSTEM_PROMPT
from seqmaestro.provenance import missing_python, record_call

#: What a proposal cost and what became of it, beside the cell fitted on it.
TOKEN_USAGE_FILE = "token_usage.json"


# ------------------------------------------------------------------- what every cell shares

def cell_settings(config: ExperimentConfig, labels: str, seed: int) -> dict[str, Any]:
    """What a cell of the run is fitted and scored with, as `run_model` takes it: the grid's
    settings, and this cell's seed and condition of the labels."""
    grid = config.grid
    return {"seed": seed, "train_fraction": grid.train_fraction,
            "permute_labels": labels == cells.PERMUTED,
            "permutation_scope": grid.permutation_scope, "n_bootstrap": grid.n_bootstrap,
            "permutation_repeats": grid.permutation_repeats}


def pool_featuriser(config: ExperimentConfig, dataset: Dataset, pool: str,
                    proposer: Proposer | None = None) -> Featuriser:
    """The featuriser of `pool` as the run computes its sources: the k-mers as `sources.kmer`
    states, the positions over the dataset's declared window, and the dataset's covariates where
    the pool holds them."""
    kmer = config.sources.kmer
    return make_featuriser(pool, proposer=proposer, covariates=dataset.covariates_for(pool),
                           window=dataset.positional_window,
                           k_range=None if kmer is None else kmer.k_range,
                           kmer_normalise=None if kmer is None else kmer.normalise)


def cell_call(model: str, pool: str, labels: str, seed: int) -> str:
    """The name of the call that fits one cell (`layout.call_name`)."""
    return call_name(None, [(model, pool)], labels, f"seed_{seed}")


def run_cells(planned: Iterable[tuple[str, Path, Callable[[], dict]]], *, target_kind: str,
              force: bool, failures: Failures) -> None:
    """Fit each planned cell: its name for the log, its directory, and the call that fits it,
    writes it and returns its manifest. A finished cell is skipped unless `force`, and one that
    raises is recorded in `failures` while the rest go on."""
    planned = list(planned)
    for index, (name, directory, fit) in enumerate(planned, start=1):
        where = f"[{index}/{len(planned)}] {name}"
        if cells.is_finished(directory) and not force:
            print(f"{where}  already done", flush=True)
            continue
        started = time.time()
        cells.unfinish(directory)
        try:
            run = fit()
        except Exception:  # one cell must not take the others with it
            failures.record(name)
            continue
        print(f"{where}  {validation_line(run['metrics'], target_kind)}  "
              f"used {run['n_features_used']:>5}  {time.time() - started:6.1f}s", flush=True)


# ------------------------------------------------------------------------- the Core arms

class FixedArm:
    """A Core model on a fixed pool: fitted in this process, on columns that follow from the rows
    by rule."""

    def __init__(self, config: ExperimentConfig, rows: Rows, model: str, pool: str) -> None:
        self.config, self.rows = config, rows
        self.model, self.pool = model, pool
        self.dataset = rows.dataset
        #: The columns the arm reads. The covariates are among them only when its pool holds them.
        self.columns = rows.dataset.columns_for(pool)

    def new_model(self, seed: int):
        return make_model(self.model, target_kind=self.dataset.target_kind, seed=seed,
                          class_weight=self.config.grid.class_weight)

    def fit(self, cell: Path, labels: str, seed: int) -> dict:
        return self.fit_on(pool_featuriser(self.config, self.dataset, self.pool), cell, labels, seed)

    def fit_on(self, featuriser: Featuriser, cell: Path, labels: str, seed: int,
               reports: Callable[[], Mapping[str, Any]] | None = None) -> dict:
        return run_model(self.new_model(seed), featuriser, self.rows.train, self.rows.val,
                         columns=self.columns, out_dir=cell, reports=reports,
                         arm=(self.model, self.pool),
                         **cell_settings(self.config, labels, seed))


def built_pool(pool: str) -> str:
    """The pool a proposer cell's featuriser is built from: the cell's, with `llm_generated` in
    place of `llm_generated_iterative`, whose proposal the iterative featuriser then wraps."""
    return pool_name(tuple(LLM_GENERATED if member == LLM_ITERATIVE else member
                           for member in parse_pool(pool)))


def proposal_part(featuriser):
    """The part of an arm's featuriser that holds the proposal and its reports: the featuriser
    itself, or the member of its pool the proposer wrote when the pool holds more."""
    for part in getattr(featuriser, "parts", [featuriser]):
        if hasattr(part, "fit_report"):
            return part
    raise TypeError(f"{type(featuriser).__name__} holds no proposal.")


def proposal_settings(config: ExperimentConfig) -> ProposalSettings:
    """How the run's proposer writes: `sources.llm_generated` for the requests, and the grid's
    policy for a written feature that fails on the rows."""
    proposer, generated = config.sources.proposer, config.grid.generated_features
    return ProposalSettings(
        n_rounds=proposer.proposal_rounds, n_per_round=proposer.n_per_round,
        n_examples=proposer.n_examples, repair_rounds=generated.repair_rounds,
        max_undefined_share=generated.max_undefined_share,
        max_example_length=generated.max_example_length, max_seconds=generated.max_seconds)


class ProposerArm(FixedArm):
    """A Core model on a pool the proposer writes into, alone or beside fixed sources.

    The proposer is shown the same rows whatever else the pool holds. In an iterative pool each
    informed round is shown a fit over every column of the pool, the fixed ones included.
    """

    def __init__(self, config: ExperimentConfig, rows: Rows, run_dir: Path, model: str,
                 pool: str) -> None:
        super().__init__(config, rows, model, pool)
        self.context = load_task(DESCRIPTIONS, config.dataset.task)
        self.settings = proposal_settings(config)
        # Proposals live with the run, so a frozen run directory carries the features it was built
        # from as well as the results. Each is keyed by its arm, so no two arms share one.
        self.cache = ProposalCache(Path(run_dir) / PROPOSALS_DIR)

    def proposer(self, seed: int) -> Proposer:
        dataset = self.dataset
        return Proposer(
            LiteLLMClient(), self.context, settings=self.settings, cache=self.cache, seed=seed,
            sequence_column=dataset.sequence_column, target_column=dataset.target_column,
            target_kind=dataset.target_kind, covariates=dataset.covariates_for(self.pool),
            arm=arm_name(self.model, self.pool))

    def featuriser(self, proposer: Proposer, labels: str, seed: int) -> Featuriser:
        written = pool_featuriser(self.config, self.dataset, built_pool(self.pool), proposer)
        if LLM_ITERATIVE not in parse_pool(self.pool):
            return written
        # The pool holds the blind proposal, the first round, whose features count as `selected`
        # because this arm only chose among them. `rounds` counts the loop's proposals. The first
        # is the pool's, which the loop takes as its round one, so the number of informed rounds
        # equals `sources.llm_generated_iterative.feedback_rounds`.
        #
        # **The validation labels are permuted for a permuted cell.** `run_model` shuffles only the
        # training labels, and validation stays real so a control can be scored against real
        # labels. The loop, however, selects features against the validation labels, so leaving
        # them real would hand the control real signal and leave no null.
        grid = self.config.grid
        chosen_on = labels_to_choose_on(
            self.rows.val, seed=seed, columns=self.columns,
            permute_labels=labels == cells.PERMUTED,
            permutation_key=permutation_key(grid.permutation_scope, (self.model, self.pool)))
        return IterativeGeneratedFeaturiser(
            proposer, lambda: self.new_model(seed),
            validation=(self.columns.inputs(self.rows.val), chosen_on), pool=written,
            rounds=1 + self.config.sources.feedback_rounds)

    def fit(self, cell: Path, labels: str, seed: int) -> dict:
        proposer = self.proposer(seed)
        featuriser = self.featuriser(proposer, labels, seed)

        def reports() -> dict[str, Any]:
            # Kept with the cell and never inferred from the feature count, because a guard that
            # drops most of what the model wrote looks just like a model that had little to say.
            proposal = proposal_part(featuriser)
            return {TOKEN_USAGE_FILE: {
                "usage": proposer.usage(),
                # Every proposal the cell's proposer made, counted together: one for the one-round
                # source, and one a round for the iterative source.
                "proposal_stats": proposer.total.as_dict(),
                "fit_report": proposal.fit_report,
                "transform_failures": proposal.transform_failures,
                # Undefined values filled with the fit-time median at transform, by feature.
                "transform_undefined": proposal.transform_undefined,
                # Per round: proposed, already present, kept, dropped. Empty for the one-round
                # source.
                "rounds": getattr(featuriser, "round_report", []),
            }}

        run = self.fit_on(featuriser, cell, labels, seed, reports)
        cost = ("read from the run's proposals" if proposer.total.cache_hit
                else f"{proposer.usage().get('total_tokens', 0):,} tokens")
        print(f"    proposal of {proposal_part(featuriser).fit_report['n_proposed']} features: "
              f"{cost}", flush=True)
        return run

    def show_prompt(self, seed: int) -> None:
        """Print what the first request of a cell at `seed` would send. It draws the rows the cell
        would fit on, so a prompt inspected here is the prompt paid for later."""
        dataset, settings = self.dataset, self.settings
        fit_rows, y_fit = subsample_training_rows(
            self.rows.train, seed=seed, columns=dataset.columns,
            train_fraction=self.config.grid.train_fraction, permute_labels=False,
            permutation_key=None)
        shown = self.columns.inputs(fit_rows)
        examples = sample_examples(shown, y_fit, settings.n_examples, seed, dataset.target_kind)
        user = render_request(
            self.context, n_features=settings.n_per_round, examples=examples, already_proposed=[],
            sequence_column=dataset.sequence_column, target_column=dataset.target_column,
            target_kind=dataset.target_kind,
            columns=describe_columns(shown, dataset.sequence_column,
                                     dataset.covariates_for(self.pool)),
            max_example_length=settings.max_example_length)
        print(f"--- system ---\n{SYSTEM_PROMPT}\n\n--- user (round 1 of {settings.n_rounds}) ---\n"
              f"{user}")
        print(f"\n--- {len(fit_rows)} fit rows, {len(examples)} shown, "
              f"~{len(SYSTEM_PROMPT) + len(user)} prompt characters ---")


# -------------------------------------------------------------------------- choosing the arm

def arm_of(config: ExperimentConfig, rows: Rows, run_dir: Path, model: str, pool: str, *,
           reuse_tree: bool = False, force: bool = False):
    """What fits the cells of the arm `model` on `pool`, by its kind. Each has `fit(cell, labels,
    seed)`, which fits one cell, writes it and returns its manifest."""
    kind = registry.kind(model, pool)
    if reuse_tree and kind != registry.DEFT:
        raise ConfigError(f"--reuse-tree is for an arm of {registry.DEFT_OR_CART}, and "
                          f"{arm_name(model, pool)} is not one.")
    if kind == registry.DEFT:
        return deft.Arm(config, rows, model, pool, reuse_tree=reuse_tree, force=force)
    if kind == registry.IGSR_KIND:
        return igsr.Arm(config, rows, model, pool)
    if kind == registry.PROPOSER:
        return ProposerArm(config, rows, run_dir, model, pool)
    return FixedArm(config, rows, model, pool)


def check(config: ExperimentConfig) -> None:
    """Refuse a config under which some arm's cells could not be started: one whose settings for
    a vendored model state a value the grid states too. The driver calls this before it starts
    anything, so the conflict costs nothing."""
    for model, pool in config.arm_cells():
        kind = registry.kind(model, pool)
        if kind == registry.DEFT:
            deft.run_overrides(config, pool)
        elif kind == registry.IGSR_KIND:
            igsr.run_overrides(config)


def blinding_problems(config: ExperimentConfig, context: TaskContext) -> list[str]:
    """Why the run's blinding cannot stand, or nothing.

    `dataset.blinded` must agree with the task description: a blinded run withholds a fact in its
    `judge_only`, and a run that is not blinded withholds nothing. A blinded run must also not name
    what it withholds where the models can see it: in a column name, or in the declared description
    of a covariate any arm is handed (`seqmaestro.llm.blinding_leaks`).
    """
    task = config.dataset.task
    if config.dataset.blinded and not context.judge_only:
        return [f"`dataset.blinded` is true, and {task}.json withholds nothing: it has no "
                f"`judge_only`."]
    if not config.dataset.blinded:
        if context.judge_only:
            return [f"`dataset.blinded` is false, and {task}.json withholds a fact from the models in "
                    f"`judge_only`. Set `dataset.blinded: true`, or name a description that withholds "
                    f"nothing."]
        return []
    shown = load_dataset(config.dataset.directory).shown_columns(
        pool for _, pool in config.arm_cells())
    return [f"the run is blinded, and {leak}. Rename the column, or reword `judge_only` in {task}.json "
            f"so that it does not hold the word." for leak in blinding_leaks(context, shown)]


# ------------------------------------------------------------------------------ the command

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run, whose config.yaml states every setting")
    parser.add_argument("--arms", nargs="+", default=None, metavar="MODEL/POOL",
                        help="fit only these of the arms the config names")
    parser.add_argument("--labels", nargs="+", default=None, choices=list(cells.LABELS),
                        help="fit only these of the grid's label conditions")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="fit only these of the grid's seeds")
    parser.add_argument("--force", action="store_true",
                        help="fit a cell again although it is finished, or although a DEFT "
                             "tree was grown for it and the cell never finished")
    parser.add_argument("--reuse-tree", action="store_true",
                        help="for a DEFT arm, to recover a cell whose tree was grown and whose "
                             "bundle is incomplete: take the tree already in the bundle and grow "
                             "no new one")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the first request a proposer arm would send, and fit nothing")
    return parser


def narrowed(given: list | None, stated: Iterable, what: str) -> list:
    """What a call works on: every item the config has (`stated`, described by `what`), or those
    of them the command line names. A name the config does not have is refused."""
    stated = list(stated)
    if given is None:
        return stated
    unknown = [item for item in given if item not in stated]
    if unknown:
        raise ConfigError(f"{', '.join(map(_spelled, unknown))}: not among the {what}, which are "
                          f"{', '.join(map(_spelled, stated))}.")
    return [item for item in stated if item in given]


def _spelled(item) -> str:
    return arm_name(*item) if isinstance(item, tuple) else str(item)


def selected_arms(given: list[str] | None, stated: list[Arm], what: str) -> list[Arm]:
    """The arms a call works on, of those the config has for its stage (`stated`, described by
    `what`)."""
    try:
        wanted = None if given is None else [split_arm(name) for name in given]
    except ValueError as error:
        raise ConfigError(str(error)) from None
    return narrowed(wanted, stated, what)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    try:
        config = load_recorded(run_dir)
        arms = selected_arms(args.arms, config.arm_cells(), "arms the run's config names")
        labels = narrowed(args.labels, config.grid.labels, "label conditions of the run's grid")
        seeds = narrowed(args.seeds, config.grid.seeds, "seeds of the run's grid")
        rows = Rows.of(config)
        fitters = {arm: arm_of(config, rows, run_dir, *arm, reuse_tree=args.reuse_tree,
                               force=args.force)
                   for arm in arms}
        # A description that withholds a fact is a blinded run's, and its models must not be shown
        # the fact in a column name or a covariate's description either. Checked before a prompt
        # is printed, which shows what the models would be shown.
        problems = (blinding_problems(config, load_task(DESCRIPTIONS, config.dataset.task))
                    if any(registry.calls_language_model(pool) for _, pool in arms) else [])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2

    if args.show_prompt:
        proposing = [fitter for fitter in fitters.values() if isinstance(fitter, ProposerArm)]
        if not proposing:
            print("--show-prompt needs an arm whose pool the proposer writes into.", file=sys.stderr)
            return 2
        proposing[0].show_prompt(seeds[0])
        return 0

    missing = [problem for model in sorted({model for model, _ in arms})
               if (environment := registry.spec(model).environment)
               and (problem := missing_python(environment))]
    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 1
    refuse_if_revealed(run_dir, "run_arm")
    call = call_name(None if args.arms else "arms", arms if args.arms else (), "-".join(labels),
                     "seed_" + "-".join(str(seed) for seed in seeds))
    record_call(run_dir, call, vars(args))

    kind = rows.dataset.target_kind
    print(f"train {len(rows.train)}  val {len(rows.val)}  "
          f"{target_line(rows.train[rows.dataset.target_column], kind)}", flush=True)
    planned = [(str(cell_dir("", *arm, label, seed)), cell,
                partial(fitters[arm].fit, cell, label, seed))
               for arm in arms for label in labels for seed in seeds
               for cell in [cell_dir(run_dir, *arm, label, seed)]]
    print(f"{len(planned)} cells: {[arm_name(*arm) for arm in arms]} x {labels} x {seeds}\n",
          flush=True)
    failures = Failures(run_dir, call)
    run_cells(planned, target_kind=kind, force=args.force, failures=failures)
    snapshot_environments(run_dir, {model for model, _ in arms})
    return failures.close()
