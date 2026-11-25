"""What a reduced arm leaves behind, written once for every arm whichever side fitted it.

    reduction/<model>/<pool>/
        reduction.json    the source cell and seed; every step's features, importances, and
                          validation headline and its companion (AUROC and AUPRC, or R-squared
                          and Spearman's rho); the order in which features left; the survivors and
                          their importances; the top-N columns; the column check; the settings
        steps.npz         validation scores at every step, and the validation labels
        full/             the model refitted on every feature the cell used: the first step
        reduced/          the model the elimination arrives at, on N features
        top_n/            the N features the full model ranked highest, fitted once
            run.json, metrics.json, predictions.npz    in each of the three: a standard cell,
                                                       on validation only
            fitted.pkl                                 a Core arm's model and its columns
            model.dill                                 or the vendored tree, which its adapter loads
            model.pkl                                  or an IGSR arm's least-squares model
        elimination.json, elimination.npz              DEFT arms only: the elimination as their
                                                       side returned it
    reduction/summary.csv  one row per reduced arm: its model, pool and seed, the features it
                           started from and ended on, its steps, and the validation headline of the
                           three models (val_<metric>_<model>)

Each of the three models is saved as a standard cell, so the reveal scores it like any other: the
saved model predicts the held-out split, which is scored with the threshold chosen here on
validation, the seed the arm was reduced at, and the bootstrap count its Stage 2 cell used.
`reduction.json` is written last and whole, so an arm that has one is finished.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from seqmaestro import __version__, files
from seqmaestro.harness import cell as cells
from seqmaestro.harness.runner import score_validation
from seqmaestro.models.base import FILLED_ROWS
from seqmaestro.provenance import git_sha
from seqmaestro.reduction import eliminate
from seqmaestro.reduction.arm import ROLES, Reduction, Source
from seqmaestro.target import target_for

REPO = Path(__file__).resolve().parents[2]
REDUCTION_FILE = "reduction.json"
STEPS_FILE = "steps.npz"
REDUCTION_VERSION = 1


def role_columns(reduction: Reduction, source: Source) -> dict[str, list[str]]:
    """The columns each of the three models was fitted on."""
    return {"full": source.names, "reduced": reduction.final_columns,
            "top_n": reduction.top_n_columns}


def _step_scores(y: np.ndarray, scores: np.ndarray, target_kind: str) -> dict[str, float | None]:
    """One step's validation headline and companion, as `val_<metric>`: AUROC and AUPRC for a
    binary target, R-squared and Spearman's rho for a continuous one. None where undefined."""
    scored = target_for(target_kind).feedback_metrics(y, scores)
    return {f"val_{name}": float(value) if np.isfinite(value) else None
            for name, value in scored.items()}


def provenance(source: Source, run_dir: str | Path) -> dict[str, Any]:
    """What every record made from a Stage 2 cell says about where, when and from what it was
    made."""
    return {
        "seqmaestro_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
        "model": source.run["model"],
        "pool": source.run["pool"],
        "labels": cells.REAL,
        "seed": source.run["seed"],
        "source_cell": str(source.cell.relative_to(Path(run_dir))),
    }


def refit_run(source: Source, run_dir: str | Path, columns: list[str], n_fit_rows: int, *,
              target_kind: str, **about: Any) -> dict[str, Any]:
    """The `run.json` of a model refitted from a Stage 2 cell, on its fitting rows, over `columns`.

    It holds the source cell's seed and bootstrap count and the target's kind, which the reveal
    scores it with, and whatever `about` says of its stage, its role and how its model is saved.
    """
    return {**provenance(source, run_dir), "n_bootstrap": source.run["n_bootstrap"],
            "target_kind": target_kind,
            "reference_split": cells.VALIDATION, "n_features": len(columns),
            "columns": list(columns),
            "n_fit_rows": n_fit_rows, "train_fraction": source.run["train_fraction"],
            "weight_column": source.run["weight_column"], **about}


def write_cell(cell: str | Path, run: dict[str, Any], y_val, scores, filled: int | None = None
               ) -> None:
    """Write a standard cell for a model fitted after Stage 2: its `run.json`, and its validation
    block and arrays, scored with the seed and bootstrap count `run` records. The caller saves the
    model, and the reveal scores such a cell as it scores any other. `filled` is how many validation
    rows took the model's constant prediction, for a model that can be undefined on a row."""
    y_val, scores = np.asarray(y_val), np.asarray(scores, dtype=float)
    metrics = score_validation(y_val, scores, target_kind=run["target_kind"], seed=run["seed"],
                               n_bootstrap=run["n_bootstrap"])
    if filled is not None:
        metrics[FILLED_ROWS] = {cells.VALIDATION: int(filled)}
    cells.write(cell, run=run, metrics=metrics, y_val=y_val, scores=scores)


def write(arm_dir: str | Path, reduction: Reduction, *, source: Source, run_dir: str | Path,
          n_target: int, y_val, target_kind: str,
          save_model: Callable[[str, Path], dict[str, Any]]) -> dict:
    """Write the arm's record and its three cells. Returns the record.

    `save_model(role, cell_dir)` writes that role's fitted model into its cell and returns what the
    cell's `run.json` should say about it. For our arms it saves `fitted.pkl` and describes the
    model, whereas the vendored tree's side has already written its tree, so there it says how the
    reveal reaches the tree.
    """
    arm_dir, run_dir = Path(arm_dir), Path(run_dir)
    y_val = np.asarray(y_val)
    common = provenance(source, run_dir)

    for role, names in role_columns(reduction, source).items():
        cell = arm_dir / role
        cell.mkdir(parents=True, exist_ok=True)
        run = refit_run(source, run_dir, names, reduction.n_fit_rows, target_kind=target_kind,
                        stage="reduction", role=role,
                        role_means=ROLES[role], **save_model(role, cell))
        write_cell(cell, run, y_val, reduction.val_scores[role], filled=reduction.filled.get(role))

    files.write_arrays(
        arm_dir / STEPS_FILE, val_true=y_val,
        val_by_step=np.vstack([np.asarray(s, dtype=float) for s in reduction.val_by_step]),
        n_features_by_step=np.array([step["n_features"] for step in reduction.steps]))
    record = {
        "reduction_version": REDUCTION_VERSION,
        **common,
        "n_target": n_target,
        "n_start": len(source.names),
        "n_final": len(reduction.final_columns),
        "n_fit_rows": reduction.n_fit_rows,
        "keep_fraction": eliminate.KEEP_FRACTION,
        "fine_below": eliminate.FINE_BELOW,
        "featurisation_check": reduction.featurisation_check,
        "final_columns": reduction.final_columns,
        "final_importance": reduction.final_importance,
        "top_n_columns": reduction.top_n_columns,
        "roles": ROLES,
        "settings": reduction.settings,
        "steps": [{**step, **_step_scores(y_val, scores, target_kind)}
                  for step, scores in zip(reduction.steps, reduction.val_by_step, strict=True)],
        "removed_at": reduction.removed_at,
    }
    files.write_json(arm_dir / REDUCTION_FILE, record, default=str)
    return record


def is_reduced(arm_dir: str | Path) -> bool:
    return (Path(arm_dir) / REDUCTION_FILE).exists()


def unfinish(arm_dir: str | Path) -> None:
    """Take a reduced arm's record away, before the arm is reduced again. Without this, a second
    reduction that failed part way would leave the models of two reductions under a record that
    says the arm is whole."""
    (Path(arm_dir) / REDUCTION_FILE).unlink(missing_ok=True)


def read(arm_dir: str | Path) -> dict:
    """A reduced arm's record, refused when it is from another version of this layout."""
    record = json.loads((Path(arm_dir) / REDUCTION_FILE).read_text())
    if record.get("reduction_version") != REDUCTION_VERSION:
        raise ValueError(f"{arm_dir}: reduction_version {record.get('reduction_version')!r}, "
                         f"expected {REDUCTION_VERSION}.")
    return record
