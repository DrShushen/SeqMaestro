"""The held-out test split, closed until the reveal and opened only here.

**This module is the only reader of the test split** outside Stage 1, which creates it
(`seqmaestro.dataset`, and `seqmaestro.cli.prepare` for the command line).
`tests/test_heldout.py` fails if any other module in `seqmaestro/` or in
`experiments/{runners,analysis,adapters}/` names it. Code that reads test results after the reveal
takes the names from here (`TEST_SPLIT`, `TEST_ARRAYS`) and checks `reveal_record` first, so every
dependency on the test split is an import of this module.

**The reveal cannot be undone.** `reveal.json` is written into the run directory before the first
test row is read, and from then on every stage that fits or selects refuses to run in that
directory (`refuse_if_revealed`), so adding an arm afterwards means starting a new run.

**The reveal predicts and scores but never fits.** `score_cell` computes a cell's test block from
the predictions of the model the cell saved, with the metrics of the target's kind
(`seqmaestro.target`), a binary target's threshold chosen on validation, and the seed and number of
bootstrap resamples recorded in `run.json`, so the block is what scoring the test split at fit time
would have produced, and it goes into the same files.

`docs/developer/heldout.md` explains why the split stays closed, and what the rule guards against.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from seqmaestro import files
from seqmaestro.harness import cell as cells
from seqmaestro.models.base import FILLED_ROWS
from seqmaestro.provenance import git_is_dirty, git_sha
from seqmaestro.target import target_for

if TYPE_CHECKING:
    # Only for annotations. `open_test` imports what it calls from `seqmaestro.dataset` on use.
    from seqmaestro.dataset import Dataset

#: The split's name, as `metrics.json` keys it once a cell has been revealed.
TEST_SPLIT = "test"
#: The file Stage 1 writes the split to, beside `train.csv` and `val.csv`.
TEST_FILE = f"{TEST_SPLIT}.csv"
#: The labels and scores the reveal adds to a cell's `predictions.npz`.
TEST_ARRAYS = (f"{TEST_SPLIT}_true", f"{TEST_SPLIT}_score")

#: Written into the run directory when the reveal starts, before any test row is read.
REVEAL_FILE = "reveal.json"
REVEAL_VERSION = 1

#: What a cell must record for its test block to be scored as inline scoring would have scored it:
#: the seed and size of its bootstrap, and the kind of target it was fitted to, which must be the
#: dataset's. A binary target's threshold is in the validation block of `metrics.json`.
SCORING_KEYS = ("seed", "n_bootstrap", "target_kind")

REPO = Path(__file__).resolve().parents[2]


class SealError(RuntimeError):
    """Something the seal forbids: a fitting stage after the reveal, or opening the test split
    before it."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def reveal_record(run_dir: str | Path) -> dict[str, Any] | None:
    """The run's reveal record, or `None` while the test split is still closed."""
    path = Path(run_dir) / REVEAL_FILE
    return json.loads(path.read_text()) if path.exists() else None


def refuse_if_revealed(run_dir: str | Path, stage: str) -> None:
    """Stop a stage that fits or selects from running in a run that has been revealed.

    Called at the start of every such stage. After the reveal, anything fitted or chosen in the run
    directory could have been chosen with knowledge of the test numbers, and the files would not
    show it. The seal exists to rule that out.
    """
    record = reveal_record(run_dir)
    if record is not None:
        raise SealError(
            f"{run_dir} was revealed at {record['revealed_utc']}, so {stage} cannot run in it: "
            f"anything fitted or selected now could be steered by the test numbers. Start a new "
            f"run to add to it.")


def begin_reveal(run_dir: str | Path, dataset: Dataset, sealed: Iterable[str], *,
                 override: str | None = None, incomplete: Iterable[str] = (),
                 unsealed: Iterable[str] = ()) -> dict[str, Any]:
    """Write `reveal.json`, which commits the run to what it holds now. Returns the record.

    Called before `open_test`, so the record exists before the first test row is read.

    `sealed` are the items the reveal scores, as paths relative to the run directory. The list is
    fixed here, so a resumed reveal scores these and nothing added since. `override` is the reason
    given for revealing a run whose pre-reveal stages did not all finish, and `incomplete` is what
    was missing. Both are recorded, because a reveal over a partial run qualifies every number it
    produces. `unsealed` are items found in the run that its config does not account for, which are
    recorded and left unscored. The record also holds the time, the commit and whether the working
    tree had uncommitted changes, and the digest that Stage 1 recorded for the test file, which
    `open_test` checks.
    """
    run_dir = Path(run_dir)
    if reveal_record(run_dir) is not None:
        raise SealError(f"{run_dir} has already been revealed; resume the reveal instead.")
    incomplete = list(incomplete)
    if incomplete and not override:
        raise SealError(
            f"{len(incomplete)} item(s) of {run_dir} are not ready, and a partial reveal needs a "
            f"stated reason: {'; '.join(incomplete[:5])}")
    record = {
        "reveal_version": REVEAL_VERSION,
        "revealed_utc": _now(),
        "git_sha": git_sha(REPO),
        "git_dirty": git_is_dirty(REPO),
        "dataset": {"splits": str(dataset.directory),
                    "test_sha256": dataset.split_counts[TEST_SPLIT]["sha256"]},
        "sealed": sorted(sealed),
        "unsealed": sorted(unsealed),
        "override": {"reason": override, "incomplete": incomplete} if override else None,
        "completed_utc": None,
    }
    files.write_json(run_dir / REVEAL_FILE, record, default=str)
    return record


def open_test(run_dir: str | Path, dataset: Dataset) -> pd.DataFrame:
    """The test split, once the run has been revealed, checked against the digest Stage 1 recorded.

    The digest is checked here because the file is not opened earlier. Without the check, a test
    split edited since the dataset was finalised would be scored as though it were the split that
    every other number in the dataset's record describes.
    """
    from seqmaestro.dataset import file_digest  # imported on use, as the import block says

    if reveal_record(run_dir) is None:
        raise SealError(
            f"{run_dir} has not been revealed. The test split is opened only by the reveal, after "
            f"`begin_reveal` has recorded it.")
    path = dataset.directory / TEST_FILE
    recorded = dataset.split_counts[TEST_SPLIT]["sha256"]
    if file_digest(path) != recorded:
        raise SealError(
            f"{path} has changed since the dataset was finalised; its digest no longer matches "
            f"the one recorded in dataset.json. Restore it before revealing.")
    return pd.read_csv(path)


def is_scored(cell_dir: str | Path) -> bool:
    """Whether the reveal has already added this cell's test block."""
    return TEST_SPLIT in cells.read_metrics(cell_dir)


def score_cell(cell_dir: str | Path, y_test, p_test, *, target_kind: str,
               filled: int | None = None) -> dict[str, Any]:
    """Score one cell's test predictions and write them into its files. Returns the test block.

    The cell is scored by the metrics of the dataset's kind of target, and a cell that records
    another kind is refused. A binary target's threshold is the one chosen on validation, and the
    seed and bootstrap count are the cell's, so the block is what inline scoring would have written.
    The arrays are written before the metrics, because a cell counts as scored once its
    `metrics.json` has the block, so an interruption between the two is repaired by scoring the cell
    again.

    `filled` is how many test rows took the model's constant prediction, for a model that can be
    undefined on a row, and is written beside the validation split's count.
    """
    cell_dir = Path(cell_dir)
    run = cells.read_run(cell_dir)
    missing = [key for key in SCORING_KEYS if key not in run]
    if missing:
        raise ValueError(
            f"{cell_dir}/run.json has no {missing}, so its test block cannot be scored the way its "
            f"validation block was.")
    if run["target_kind"] != target_kind:
        raise ValueError(f"{cell_dir} was fitted against a {run['target_kind']} target, and the "
                         f"dataset's target is {target_kind}.")
    metrics = cells.read_metrics(cell_dir)
    if TEST_SPLIT in metrics:
        raise SealError(f"{cell_dir} already has a test block; a cell is scored once.")

    y_test = np.asarray(y_test)
    p_test = np.asarray(p_test, dtype=float)
    block = target_for(target_kind).metrics(
        y_test, p_test, threshold=metrics["val"].get("threshold"),
        n_bootstrap=run["n_bootstrap"], seed=run["seed"])

    cells.write_predictions(cell_dir, {**cells.read_predictions(cell_dir),
                                       **dict(zip(TEST_ARRAYS, (y_test, p_test), strict=True))})
    written = {**metrics, TEST_SPLIT: block}
    if filled is not None:
        written[FILLED_ROWS] = {**metrics.get(FILLED_ROWS, {}), TEST_SPLIT: int(filled)}
    cells.write_metrics(cell_dir, written)
    return block


def finish_reveal(run_dir: str | Path, scored: dict[str, int]) -> dict[str, Any]:
    """Mark the reveal complete, with how many items of each kind it scored."""
    run_dir = Path(run_dir)
    record = reveal_record(run_dir)
    if record is None:
        raise SealError(f"{run_dir} has no reveal to finish.")
    record.update(completed_utc=_now(), scored=scored)
    files.write_json(run_dir / REVEAL_FILE, record, default=str)
    return record


def revealed_metrics(cell_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """A cell's test block, for code that reads results after the reveal. Refused before it."""
    if reveal_record(run_dir) is None:
        raise SealError(f"{run_dir} has not been revealed; its cells have no test numbers yet.")
    metrics = cells.read_metrics(cell_dir)
    if TEST_SPLIT not in metrics:
        raise SealError(f"{cell_dir} was not scored by the reveal.")
    return metrics[TEST_SPLIT]
