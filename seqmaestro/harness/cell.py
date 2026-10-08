"""A cell: the directory one fitted model leaves behind, and the one way it is written.

    run.json             what was fitted, and what the reveal needs to score it
    metrics.json         the validation block, and after the reveal the block of the held-out split
    predictions.npz      the labels and the scores, so any metric can be computed again without a fit
    features.json        the features the model used, with importances and directions
    feature_values.npz   those features evaluated on the validation rows

A Stage 2 cell holds all five. A model fitted later from a Stage 2 cell (a reduced arm's three
models, a refit of the reliability panel) holds the first three, since its features are described
by the cell it came from. Beside them a cell keeps its saved model, in the form its `predict_with`
names, and whatever its arm reports about the fit.

**`metrics.json` is written last, and a cell that has one is finished.** A resumed run skips such
a cell, and the reveal scores it. Every file is written whole (`seqmaestro.files`), so a process
killed in the middle of a cell leaves no marker, and the cell is fitted again.

The two conditions a cell's labels can be in are named here too. A `real` cell was fitted on the
labels as they are. A `permuted` cell was fitted on the training labels shuffled, and is the
control its arm is compared with.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from seqmaestro import files
from seqmaestro.harness.features import save_feature_values, save_features

if TYPE_CHECKING:
    from seqmaestro.features.base import FeatureRecord

RUN_FILE = "run.json"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.npz"
FEATURES_FILE = "features.json"
FEATURE_VALUES_FILE = "feature_values.npz"

REAL, PERMUTED = "real", "permuted"
#: The label conditions, in the order a grid lists them.
LABELS = (REAL, PERMUTED)

#: The split every cell is scored on when it is fitted, as `metrics.json` and the arrays of
#: `predictions.npz` name it.
VALIDATION = "val"


@dataclass(frozen=True)
class Features:
    """What a Stage 2 cell records about the features its model used."""

    records: list[FeatureRecord]
    #: The records evaluated on the validation rows, one column per record, in their order.
    values: np.ndarray
    #: What `features.json` says about the fit above its records.
    header: dict[str, Any] = field(default_factory=dict)


def is_finished(cell: str | Path) -> bool:
    """Whether the cell was written to its end."""
    return (Path(cell) / METRICS_FILE).exists()


def unfinish(cell: str | Path) -> None:
    """Take a cell's marker away, before it is fitted again. Without this, a second fit that died
    part way would leave the files of two fits under a marker that says the cell is whole."""
    (Path(cell) / METRICS_FILE).unlink(missing_ok=True)


def read_run(cell: str | Path) -> dict[str, Any]:
    return json.loads((Path(cell) / RUN_FILE).read_text())


def read_metrics(cell: str | Path) -> dict[str, Any]:
    return json.loads((Path(cell) / METRICS_FILE).read_text())


def read_predictions(cell: str | Path) -> dict[str, np.ndarray]:
    """Every array of a cell's `predictions.npz`, by name."""
    with np.load(Path(cell) / PREDICTIONS_FILE) as saved:
        return {name: saved[name] for name in saved.files}


def validation_scores(cell: str | Path) -> np.ndarray:
    """The scores the cell's model gave the validation rows, in their order."""
    return np.asarray(read_predictions(cell)[f"{VALIDATION}_score"], dtype=float)


def write(cell: str | Path, *, run: Mapping[str, Any], metrics: Mapping[str, Any], y_val, scores,
          features: Features | None = None, reports: Mapping[str, Any] | None = None) -> Path:
    """Write a cell: its record, its validation arrays, its features where it has any, and its
    marker last.

    `reports` are further JSON files of the cell, by file name, such as what a language model's
    proposal cost. They are written before the marker, so a finished cell has them.
    """
    cell = Path(cell)
    cell.mkdir(parents=True, exist_ok=True)
    unfinish(cell)
    files.write_json(cell / RUN_FILE, run, default=str)
    write_predictions(cell, {f"{VALIDATION}_true": np.asarray(y_val),
                             f"{VALIDATION}_score": np.asarray(scores, dtype=float)})
    if features is not None:
        save_features(cell / FEATURES_FILE, features.records, **features.header)
        # Only the used features, in record order, so the two files line up by position.
        save_feature_values(cell / FEATURE_VALUES_FILE,
                            [record.name for record in features.records], features.values)
    for name, payload in (reports or {}).items():
        files.write_json(cell / name, payload)
    write_metrics(cell, metrics)
    return cell


def write_predictions(cell: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    files.write_arrays(Path(cell) / PREDICTIONS_FILE, **arrays)


def write_metrics(cell: str | Path, metrics: Mapping[str, Any]) -> None:
    files.write_json(Path(cell) / METRICS_FILE, metrics)
