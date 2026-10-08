"""Stage 3A: reduce each arm to a model of N features, on validation only.

    python experiments/runners/run_reduction.py --run-dir experiments/runs/<date>_<name>

For every arm the run's config reduces, the real-label cell with the best validation headline
(AUROC for a binary target, R-squared for a continuous one) is taken, and its model is eliminated
down to N of the features it used by the loop in `seqmaestro.reduction.eliminate`. The elimination
is made wherever the arm's model lives (`seqmaestro.pipeline.sides`): in this process for a Core
model, on DEFT's side for its tree, which evaluates its features there, and in this process for an
IGSR arm, on the values of its terms as IGSR's side evaluates them. Each arm leaves three models
(the full refit, the reduced one, and the top-N control) as standard cells under `reduction/`,
scored on validation. The reveal scores them on the test split with the rest. `docs/reduction.md`
says what is computed and why.

An arm with a `reduction.json` is skipped unless `--force` is given. A revealed run is refused.

The driver runs this once per arm (`--arms`), each call as soon as its arm's cells are there,
several at once. `reduction/summary.csv` is written from every arm reduced so far, so whichever
call writes it last leaves it whole.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from seqmaestro import files
from seqmaestro.arms import Arm, arm_name
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import refuse_if_revealed
from seqmaestro.pipeline.config import ConfigError, ExperimentConfig, load_recorded
from seqmaestro.pipeline.layout import (
    CONFIG_FILE,
    REDUCTION_DIR,
    call_name,
    cell_dir,
    reduction_dir,
)
from seqmaestro.pipeline.records import Failures, snapshot_environments
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.sides import side_of
from seqmaestro.pipeline.stage2 import selected_arms
from seqmaestro.provenance import record_call
from seqmaestro.reduction import outputs
from seqmaestro.reduction.arm import Source, best_seed
from seqmaestro.target import target_for

#: The stage's name in a call's name (`layout.call_name`).
STAGE = "reduction"
SUMMARY_FILE = "summary.csv"


def source_cell(run_dir: Path, arm: Arm, seeds, headline: str) -> Path:
    """The real-label cell an arm is reduced from: its best validation seed.

    Every seed of the grid must be there, since a best seed chosen among only some of them may not
    be the one the run would have chosen.
    """
    found = {seed: cell_dir(run_dir, *arm, cells.REAL, seed) for seed in seeds}
    missing = [str(cell) for cell in found.values() if not cells.is_finished(cell)]
    if missing:
        raise FileNotFoundError(f"cells not finished, so no best seed can be chosen: {missing}")
    validation = {seed: cells.read_metrics(cell)[cells.VALIDATION] for seed, cell in found.items()}
    return found[best_seed(validation, headline)]


def reduce_one(run_dir: Path, arm: Arm, config: ExperimentConfig, rows: Rows) -> dict:
    """Reduce one arm and write its outputs. Returns its record."""
    dataset, n_target = rows.dataset, config.reduction.n_features
    source = Source.read(source_cell(run_dir, arm, config.grid.seeds,
                                     target_for(dataset.target_kind).headline))
    arm_dir = reduction_dir(run_dir, *arm)
    outputs.unfinish(arm_dir)
    started = time.time()

    def log(step: dict) -> None:
        print(f"    n={step['n_features']:5d} used={step['n_used_by_model']:5d}  "
              f"{time.time() - started:7.1f}s", flush=True)

    reduction, save_model = side_of(source.run, source.cell).reduce(
        source, rows, n_target=n_target, out_dir=arm_dir, on_step=log)
    reduction.settings["wall_clock_seconds"] = round(time.time() - started, 1)
    return outputs.write(arm_dir, reduction, source=source, run_dir=run_dir, n_target=n_target,
                         y_val=rows.val[dataset.target_column].to_numpy(),
                         target_kind=dataset.target_kind, save_model=save_model)


def summary_row(run_dir: Path, arm: Arm, headline: str) -> dict:
    """One reduced arm's line in `reduction/summary.csv`."""
    arm_dir = reduction_dir(run_dir, *arm)
    record = outputs.read(arm_dir)
    roles = {role: cells.read_metrics(arm_dir / role)[cells.VALIDATION][headline]
             for role in record["roles"]}
    model, pool = arm
    return {"model": model, "pool": pool, "seed": record["seed"],
            "n_start": record["n_start"], "n_final": record["n_final"],
            "n_steps": len(record["steps"]),
            **{f"val_{headline}_{role}": value for role, value in roles.items()}}


def write_summary(run_dir: Path, config: ExperimentConfig, headline: str) -> None:
    """`reduction/summary.csv`, from every arm of the config that is reduced by now."""
    reduced = [arm for arm in config.reduced_arms()
               if outputs.is_reduced(reduction_dir(run_dir, *arm))]
    if reduced:
        out_dir = run_dir / REDUCTION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame([summary_row(run_dir, arm, headline) for arm in reduced])
        files.write_text(out_dir / SUMMARY_FILE, table.to_csv(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=None, metavar="MODEL/POOL",
                        help="reduce only these of the arms the config reduces")
    parser.add_argument("--force", action="store_true",
                        help="reduce again an arm that already has a reduction.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    try:
        config = load_recorded(run_dir)
        if config.reduction is None:
            raise ConfigError(f"{run_dir / CONFIG_FILE} has no `reduction` section, so there is "
                              f"nothing to reduce.")
        arms = selected_arms(args.arms, config.reduced_arms(), "arms the run's config reduces")
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    refuse_if_revealed(run_dir, "run_reduction")
    call = call_name(STAGE, arms if args.arms else ())
    record_call(run_dir, call, vars(args))

    rows = Rows.of(config)
    headline = target_for(rows.dataset.target_kind).headline
    print(f"reducing {len(arms)} arm(s) to {config.reduction.n_features} features; "
          f"train {len(rows.train)}  val {len(rows.val)}", flush=True)
    for name, why in config.not_reduced().items():
        print(f"  not reduced: {name} ({why})", flush=True)
    print(flush=True)

    failures = Failures(run_dir, call)
    for arm in arms:
        arm_dir = reduction_dir(run_dir, *arm)
        print(arm_name(*arm), flush=True)
        if outputs.is_reduced(arm_dir) and not args.force:
            record = outputs.read(arm_dir)
            print("    already reduced", flush=True)
        else:
            try:
                record = reduce_one(run_dir, arm, config, rows)
            except Exception:  # one arm must not take the others with it
                failures.record(arm_name(*arm))
                continue
        row = summary_row(run_dir, arm, headline)
        print(f"    seed {record['seed']}: {record['n_start']} -> {record['n_final']} features in "
              f"{len(record['steps'])} steps; val {headline} "
              + "  ".join(f"{role} {row[f'val_{headline}_{role}']:.3f}" for role in record["roles"]),
              flush=True)

    write_summary(run_dir, config, headline)
    # A model whose search ran in another environment (PySR's) has that environment recorded too.
    snapshot_environments(run_dir, {model for model, _ in arms})
    return failures.close()
