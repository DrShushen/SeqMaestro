"""A check that rebuilt columns are the arm's columns. **numpy only**, like `eliminate.py`.

Every Stage 2 cell carries `feature_values.npz`: the features its model used, evaluated on the
validation rows when the cell was fitted. A later stage that evaluates those features again, from
the saved featuriser or on a vendored method's side, compares its validation matrix against that
file before fitting anything on it. This directly checks the step that could otherwise go wrong
silently (a column taken from the wrong featuriser, or a generated function evaluated against a
differently named column), and it costs one comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Largest relative disagreement tolerated. A cell stores its values as float32, so the rebuilt ones
#: are compared at that precision, where they are expected to be exactly equal. The tolerance admits
#: rounding and nothing else. It is relative because a count reaches the hundreds while an indicator
#: is 0 or 1.
TOLERANCE = 1e-5


def stored_values(cell_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """The names and validation values a Stage 2 cell recorded for the features it used."""
    with np.load(Path(cell_dir) / "feature_values.npz", allow_pickle=True) as payload:
        return [str(name) for name in payload["names"]], np.asarray(payload["values"])


def compare(rebuilt: np.ndarray, names: list[str], cell_dir: str | Path,
            tolerance: float = TOLERANCE) -> dict:
    """The largest disagreement between a rebuilt validation matrix and the cell's, or an
    `AssertionError` when any column is further out than `tolerance`.

    Two NaNs at the same place agree: a generated feature that could not be evaluated on a row
    failed the same way both times.
    """
    stored_names, stored = stored_values(cell_dir)
    if stored_names != list(names):
        differing = sorted(set(stored_names) ^ set(names))
        raise AssertionError(
            f"{cell_dir}: the cell recorded {len(stored_names)} features and the rebuild has "
            f"{len(names)}; {len(differing)} name(s) differ, first {differing[:5]}.")
    rebuilt = np.asarray(rebuilt, dtype=np.float32)
    if rebuilt.shape != stored.shape:
        raise AssertionError(f"{cell_dir}: shape {rebuilt.shape} against stored {stored.shape}.")
    if not rebuilt.size:
        return {"n_features": len(names), "max_relative_error": 0.0, "worst_feature": None,
                "n_columns_over_tolerance": 0}

    both_missing = np.isnan(rebuilt) & np.isnan(stored)
    difference = np.where(both_missing, 0.0, np.abs(rebuilt.astype(float) - stored.astype(float)))
    difference = np.where(np.isnan(difference), np.inf, difference)
    scale = np.maximum(np.nan_to_num(np.abs(stored).astype(float)).max(axis=0), 1.0)
    error = difference.max(axis=0) / scale
    worst = int(np.argmax(error))
    result = {
        "n_features": len(names),
        "max_relative_error": float(error[worst]),
        "worst_feature": names[worst],
        "n_columns_over_tolerance": int((error > tolerance).sum()),
    }
    if result["n_columns_over_tolerance"]:
        raise AssertionError(
            f"{cell_dir}: {result['n_columns_over_tolerance']} rebuilt column(s) differ from the "
            f"cell's own values, worst {result['worst_feature']} at "
            f"{result['max_relative_error']:.3g}.")
    return result
