"""The composition baseline: a standardised sparse linear model on base fractions, at every seed.

    python experiments/runners/run_baseline.py --run-dir experiments/runs/<date>_<name>

The features and the model are set here and nowhere else, so that every dataset gets the same
baseline, and the run's config supplies only what the baseline shares with the arms: the rows each
seed fits on, the class weighting or per-row weights, and how a cell is scored. The cells are
written to `<run>/baseline/seed_<n>/` in the standard cell format, on the real labels only. A
finished seed is skipped unless `--force` is given, and a revealed run is refused.
`docs/developer/baseline.md` gives the settings and the reasons for them.
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

#: The model and the pool the baseline's cells report. An arm cannot have either name.
MODEL_NAME = "composition_baseline"
POOL = "composition"
#: The name of the baseline's call (`layout.call_name`).
CALL = "baseline"

#: The baseline's model for each kind of target, with its penalty. Both penalties are weak, because
#: there is nothing to select among the few fraction columns, which are left essentially
#: unpenalised: C = 1 for the logistic model (the arms use 0.1), and a hundredth of the arms'
#: universal penalty for the linear one.
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
