"""The composition baseline: base fractions into a standardised sparse linear model, every seed.

    python experiments/runners/run_baseline.py --run-dir experiments/runs/<date>_<name>

The reference line every arm is judged against: what the proportion of each base, and nothing
else, achieves on the same rows. When a dataset's target goes with composition the baseline
predicts it, so an arm that does no better has not shown it found anything composition alone does
not give.

**The baseline is fixed and simple.** The features and the model are set here and nowhere else
(neither the config nor the dataset's preparation can change them), so every dataset gets the same
baseline, computed the same way, and it cannot be tuned towards a conclusion. The run's config
supplies only what the baseline must share with the arms for the comparison to hold: the rows each
seed fits on (the same stratified subsample, through the same harness), the class weighting of a
binary target and the per-row weights, and how a cell is scored.

The model is the arms' sparse linear model for the target's kind, on columns standardised on the
fitting rows as in every L1 fit: `L1Logistic` at a fixed C = 1 instead of the arms' default 0.1 for
a binary target, and `L1Linear` at a hundredth of the universal penalty the arms use for a
continuous one. Base fractions are a handful of columns with nothing to select
among, and the reference line should not be penalised towards chance.

Cells go to `<run>/baseline/seed_<n>/`, outside `results/`, in the standard cell format. They use
the real target only, because the arms' permuted controls already measure chance.

The baseline reads the sequences only, so a dataset's covariates never enter it, and the reference
line means the same thing on every dataset. A finished seed is skipped unless `--force` is given,
and a revealed run is refused.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from seqmaestro.features import CompositionFeaturiser
from seqmaestro.harness.cell import REAL
from seqmaestro.harness.heldout import refuse_if_revealed
from seqmaestro.harness.runner import run_model
from seqmaestro.models import make_model
from seqmaestro.pipeline.config import ConfigError, ExperimentConfig, load_recorded
from seqmaestro.pipeline.layout import baseline_dir
from seqmaestro.pipeline.records import Failures, snapshot_environments
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.stage2 import cell_settings, run_cells
from seqmaestro.provenance import record_call
from seqmaestro.target import BINARY, CONTINUOUS

#: What the baseline's cells report as their model and their pool. Neither is one an arm can have.
MODEL_NAME = "composition_baseline"
POOL = "composition"
#: The name of the baseline's call (`layout.call_name`).
CALL = "baseline"

#: The baseline's model for each kind of target, and its penalty: weak in both, so the few fraction
#: columns are essentially unpenalised, since there is nothing to select among. A C of 1 against the
#: arms' 0.1 for the logistic model, and a hundredth of the universal penalty the arms use for the
#: linear one.
MODEL = {BINARY: ("l1_logistic", {"C": 1.0}),
         CONTINUOUS: ("l1_linear", {"strength": 0.01})}


def baseline_model(seed: int, target_kind: str, class_weight: str | None):
    """The baseline's model for one seed."""
    name, settings = MODEL[target_kind]
    return make_model(name, target_kind=target_kind, seed=seed, class_weight=class_weight,
                      **settings)


def fit(config: ExperimentConfig, rows: Rows, cell: Path, seed: int) -> dict:
    """One seed's baseline cell, fitted on the rows the arms fit on at that seed."""
    dataset = rows.dataset
    return run_model(
        baseline_model(seed, dataset.target_kind, config.grid.class_weight),
        CompositionFeaturiser(), rows.train, rows.val, columns=dataset.columns, out_dir=cell,
        arm=(MODEL_NAME, POOL), **cell_settings(config, REAL, seed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run, whose config.yaml states the seeds and how a cell is scored")
    parser.add_argument("--force", action="store_true",
                        help="fit again a seed that is already finished")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    try:
        config = load_recorded(run_dir)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    refuse_if_revealed(run_dir, "run_baseline")
    record_call(run_dir, CALL, vars(args))
    rows = Rows.of(config)

    failures = Failures(run_dir, CALL)
    run_cells(((str(baseline_dir("", seed)), baseline_dir(run_dir, seed),
                partial(fit, config, rows, baseline_dir(run_dir, seed), seed))
               for seed in config.grid.seeds),
              target_kind=rows.dataset.target_kind, force=args.force, failures=failures)
    snapshot_environments(run_dir)
    return failures.close()
