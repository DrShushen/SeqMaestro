"""Open the test split, once, and score every saved model on it.

    python experiments/runners/run_reveal.py --run-dir experiments/runs/<date>_<name>
    python experiments/runners/run_reveal.py --run-dir ... --check    # is it ready? reveals nothing

Every stage before this one works on the training and validation rows. This step opens the test
split:

1. **Check the run is finished.** Every cell the run's recorded config should have produced must
   exist, have been scored on validation, and have saved its model, and every arm the reliability
   panel profiles must have its record. Otherwise the reveal refuses and lists what is missing.
   With `--override-incomplete REASON` it goes ahead, and records the reason and what was missing
   in `reveal.json`, because a reveal over a partial run qualifies every number it produces.
2. **Seal it.** `reveal.json` is written before any test row is read, listing everything the reveal
   will score. From then on the stages that fit or select refuse to run in this directory.
3. **Read the test split**, checked against the digest Stage 1 recorded for it.
4. **Predict and score each sealed cell**: the arms', the baseline's, each reduced arm's three
   models and each profiled arm's leave-one-out refits. Each cell's model predicts wherever it
   lives (`seqmaestro.pipeline.sides`, by the `predict_with` its `run.json` records): a Core model
   in this process, and a vendored method's on its side of the environment boundary, which is
   handed the rows without their labels. Each cell's test block is scored with a binary target's
   threshold chosen on validation and the seed and bootstrap count the cell recorded, and written
   into the cell's `metrics.json` and `predictions.npz` (see `seqmaestro.harness.heldout`).
5. **Compute each profiled arm's Tier 2 on the test split**, the only test-side work that does not
   score a model. The arm's prefiltered pool is evaluated on the test rows by the featuriser, tree
   or terms it came from, and the association of each feature with the real labels, with a
   permutation of them and, on the check arm, with the planted labels is calibrated (see
   `seqmaestro.reliability.panel`).

Nothing is refitted. The reveal is resumable: run again after an interruption, it does whatever on
the sealed list is not done yet, and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from seqmaestro.arms import Arm
from seqmaestro.dataset import Dataset, DatasetError, load_dataset
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import (
    REVEAL_FILE,
    SCORING_KEYS,
    TEST_SPLIT,
    begin_reveal,
    finish_reveal,
    is_scored,
    open_test,
    reveal_record,
    score_cell,
)
from seqmaestro.pipeline.config import ConfigError, ExperimentConfig, load_recorded
from seqmaestro.pipeline.layout import (
    BASELINE_DIR,
    REDUCTION_DIR,
    RELIABILITY_DIR,
    RESULTS_DIR,
    reliability_dir,
)
from seqmaestro.pipeline.records import Failures
from seqmaestro.pipeline.reliability import panel_for
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.sides import side_of
from seqmaestro.provenance import record_call
from seqmaestro.reliability import panel as panels
from seqmaestro.reliability.files import LOCO_DIR, RECORD_FILE
from seqmaestro.target import target_for

#: The name of the reveal's call (`layout.call_name`).
CALL = "reveal"

#: How many missing cells a refusal lists before summarising the rest.
SHOWN = 12

#: What `reveal.json` counts a profiled arm's Tier 2 under, beside the cells counted by directory.
PANEL_KIND = "reliability_tier2"


def readiness(cell: Path, target_kind: str) -> str | None:
    """Why a cell cannot be revealed for a dataset of this kind of target, or None when it can.
    Everything scoring a cell needs is asked for here, so that no cell fails once the seal is
    written."""
    if not cells.is_finished(cell):
        return f"no {cells.METRICS_FILE}: the cell did not finish"
    if cells.VALIDATION not in cells.read_metrics(cell):
        return f"{cells.METRICS_FILE} has no validation block"
    run = cells.read_run(cell)
    missing = [key for key in SCORING_KEYS if key not in run]
    if missing:
        return f"{cells.RUN_FILE} does not record {missing}, which the test block is scored with"
    if run["target_kind"] != target_kind:
        return (f"it was fitted against a {run['target_kind']} target, and the dataset's is "
                f"{target_kind}")
    try:
        side = side_of(run)
    except ValueError as error:
        return str(error)
    if not side.is_saved(cell, run):
        return f"its saved model is missing: `predict_with` is {run['predict_with']!r}"
    return None


def cells_on_disk(run_dir: Path) -> set[str]:
    """Every finished cell and reliability panel under the run, as paths relative to it."""
    marker = cells.METRICS_FILE
    found = [*(run_dir / RESULTS_DIR).glob(f"*/*/*/seed_*/{marker}"),
             *(run_dir / BASELINE_DIR).glob(f"seed_*/{marker}"),
             *(run_dir / REDUCTION_DIR).glob(f"*/*/*/{marker}"),
             *(run_dir / RELIABILITY_DIR).glob(f"*/*/{LOCO_DIR}/*/{marker}"),
             *(run_dir / RELIABILITY_DIR).glob(f"*/*/{RECORD_FILE}")]
    return {str(path.parent.relative_to(run_dir)) for path in found}


def panels_of(config: ExperimentConfig) -> dict[str, Arm]:
    """Each arm the reliability panel profiles, by its directory relative to the run: the item the
    reveal seals for its Tier 2 on the test split."""
    return {str(reliability_dir("", *arm)): arm for arm in config.profiled_arms()}


def predict(run_dir: Path, cell: Path, test: pd.DataFrame, dataset: Dataset
            ) -> tuple[np.ndarray, int | None]:
    """The saved model's score for every test row, in row order, and how many rows took the
    model's constant prediction (`Side.predict`). The rows carry every covariate the dataset
    declares, and no label."""
    run = cells.read_run(cell)
    return side_of(run, cell).predict(run_dir, cell, run, dataset.inputs(test), dataset)


def survey(run_dir: Path, config: ExperimentConfig) -> tuple[list[str], dict[str, str], list[str]]:
    """What the run should hold, what of it is not ready and why, and what it holds besides.

    The config names the cells and the panels. A panel's leave-one-out cells depend on what the
    reduction and the panel found, so they are listed by its record; a panel without one is not
    ready, and nothing of it is sealed.
    """
    expected_cells = [str(cell) for cell in config.expected_cells()]
    not_ready = {}
    for relative in panels_of(config):
        if not panels.is_ready(run_dir / relative):
            not_ready[relative] = f"the reliability panel has not finished: no {RECORD_FILE}"
            continue
        expected_cells += [f"{relative}/{entry['cell']}"
                           for entry in panels.read_record(run_dir / relative)["loco"]]
    kind = config.dataset.target_kind
    not_ready.update({cell: why for cell in expected_cells
                      if (why := readiness(run_dir / cell, kind))})
    expected = expected_cells + list(panels_of(config))
    unsealed = sorted(cells_on_disk(run_dir) - set(expected))
    if unsealed:
        print(f"{len(unsealed)} item(s) on disk are not in the run's config and will not be "
              f"revealed: {', '.join(unsealed[:SHOWN])}", flush=True)
    return expected, not_ready, unsealed


def listing(not_ready: dict[str, str]) -> str:
    shown = "\n".join(f"  {cell}: {why}" for cell, why in list(not_ready.items())[:SHOWN])
    more = f"\n  ... and {len(not_ready) - SHOWN} more" if len(not_ready) > SHOWN else ""
    return shown + more


def seal(run_dir: Path, config: ExperimentConfig, dataset: Dataset, override: str | None
         ) -> dict | None:
    """Check the run and write `reveal.json`. Returns the record, or None after a refusal."""
    expected, not_ready, unsealed = survey(run_dir, config)
    if not_ready and not override:
        print(f"refusing to reveal {run_dir}: {len(not_ready)} of {len(expected)} items are not "
              f"ready.\n{listing(not_ready)}\nFinish them first (the driver resumes), or pass "
              f"--override-incomplete with the reason, which is recorded.", file=sys.stderr)
        return None
    sealed = [item for item in expected if item not in not_ready]
    record = begin_reveal(run_dir, dataset, sealed, override=override,
                          incomplete=[f"{item}: {why}" for item, why in not_ready.items()],
                          unsealed=unsealed)
    print(f"sealed {len(sealed)} items in {run_dir / REVEAL_FILE}", flush=True)
    return record


def summarise(run_dir: Path, sealed: list[str], headline: str) -> str:
    """Mean test headline per arm (AUROC for a binary target, R-squared for a continuous one), for
    the log: every label condition of the arms, and each reduced model. The cells hold the full
    blocks."""
    table = []
    for relative in sealed:
        parts = Path(relative).parts
        if parts[0] == RELIABILITY_DIR:
            continue  # the panel's tables report its refits and Tier 2
        metrics = cells.read_metrics(run_dir / relative)
        if parts[0] == BASELINE_DIR:
            arm, column = "composition baseline", cells.REAL
        elif parts[0] == REDUCTION_DIR:
            arm, column = f"{parts[1]} / {parts[2]}", f"3A {parts[3]}"
        else:
            arm, column = f"{parts[1]} / {parts[2]}", parts[3]
        table.append({"arm": arm, "column": column, headline: metrics[TEST_SPLIT][headline]})
    if not table:
        return "no cell was scored"
    frame = pd.DataFrame(table)
    return (frame.pivot_table(index="arm", columns="column", values=headline, aggfunc="mean")
            .round(3).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--override-incomplete", metavar="REASON", default=None,
                        help="reveal although some items are not ready, for the reason given; the "
                             "reason and what was missing are recorded in reveal.json")
    parser.add_argument("--check", action="store_true",
                        help="only report whether everything is ready; seal nothing, read nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir

    try:
        config = load_recorded(run_dir)
        dataset = load_dataset(config.dataset.directory)
    except (ConfigError, DatasetError) as error:
        print(error, file=sys.stderr)
        return 2

    record = reveal_record(run_dir)
    if args.check:
        if record is not None:
            print(f"{run_dir} was revealed {record['revealed_utc']}.")
            return 0
        expected, not_ready, _ = survey(run_dir, config)
        print(f"{len(expected) - len(not_ready)} of {len(expected)} items are ready to reveal."
              + (f"\n{listing(not_ready)}" if not_ready else ""))
        return 2 if not_ready else 0
    if record is None:
        record = seal(run_dir, config, dataset, args.override_incomplete)
        if record is None:
            return 2
    else:
        print(f"resuming the reveal begun {record['revealed_utc']}", flush=True)
    record_call(run_dir, CALL, vars(args))

    test = open_test(run_dir, dataset)
    y_test = test[dataset.target_column].to_numpy()
    panel_arms = panels_of(config)
    rows = None
    scored, failures = Counter(), Failures(run_dir, CALL)
    for index, relative in enumerate(record["sealed"], start=1):
        item = run_dir / relative
        is_panel = relative in panel_arms
        kind = PANEL_KIND if is_panel else Path(relative).parts[0]
        if panels.is_revealed(item) if is_panel else is_scored(item):
            scored[kind] += 1
            continue
        try:
            if is_panel:
                rows = rows or Rows.of(config)
                panel_for(run_dir, config, panel_arms[relative], rows).reveal_tier2(
                    dataset.inputs(test), y_test, log=print)
            else:
                scores, filled = predict(run_dir, item, test, dataset)
                score_cell(item, y_test, scores, target_kind=dataset.target_kind, filled=filled)
        except Exception:  # one item must not stop the rest; the reveal resumes
            failures.record(relative)
            continue
        scored[kind] += 1
        print(f"[{index}/{len(record['sealed'])}] {relative}", flush=True)

    if failures.close():
        print("Run this again to finish what failed.", file=sys.stderr)
        return 1
    finish_reveal(run_dir, dict(scored))
    headline = target_for(dataset.target_kind).headline
    print(f"\nrevealed: {dict(scored)}\n\nmean test {headline}\n"
          f"{summarise(run_dir, record['sealed'], headline)}")
    return 0
