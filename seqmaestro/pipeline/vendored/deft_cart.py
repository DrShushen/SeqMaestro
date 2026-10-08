"""The Core side of `experiments/adapters/deft_cart.py`: what the later stages ask of DEFT.

A DEFT arm's features are functions its saved tree carries, so they are evaluated in DEFT's
environment and never here. This module prepares what the adapter reads (the rows in the form the
arm's tree saw its own, numbered, with labels only on the rows it fits on), runs the adapter under
DEFT's interpreter, and reads back what it wrote. It never imports `deft`.

Each later stage has its entry point here: `reduce_arm` for Stage 3A, `DeftFits` for the
reliability panel (a DEFT arm's fits, behind the same interface as
`seqmaestro.reliability.fits.InProcess`), `predict_cell` for the reveal, and `export_tree` for the
report. `seqmaestro.pipeline.sides` calls them for a cell DEFT's side wrote.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqmaestro.dataset import Columns
from seqmaestro.features.base import Inputs
from seqmaestro.harness import cell as cells
from seqmaestro.pipeline.repository import ADAPTERS
from seqmaestro.pipeline.vendored.deft import DEFT_PYTHON
from seqmaestro.pipeline.vendored.handover import ROW_ID, input_frame, predict_numbered
from seqmaestro.reduction.arm import Reduction, Source
from seqmaestro.reliability.files import PREPARE_ARRAYS, loco_cell, selections_file
from seqmaestro.reliability.pool import Pool
from seqmaestro.report.internals import check_reproduces

ADAPTER = ADAPTERS / "deft_cart.py"
#: What a reduced or refitted DEFT cell's `run.json` says about how the reveal predicts it:
#: `predict_cell`.
PREDICT_WITH = "deft_cart"
#: The tree inside a reduced or refitted DEFT cell. It restates the adapter's `MODEL_FILE`, because
#: that module runs in the other environment, and `tests/test_reduction.py` checks the two agree.
MODEL_FILE = "model.dill"


def saved_tree(tree: dict[str, Any]) -> dict[str, Any]:
    """What the `run.json` of a cell says about the tree DEFT's side wrote into it: how the reveal
    reaches it, and the settings it was grown with."""
    return {"predict_with": PREDICT_WITH, "fitted_model": MODEL_FILE,
            "model_settings": {"model": "deft_cart_tree", **tree}}


def reduce_arm(source_cell: Path, fit_rows: pd.DataFrame, y_fit: np.ndarray, val: Inputs, *,
               columns: Columns, weight_column: str | None, n_target: int,
               out_dir: Path) -> Reduction:
    """Eliminate a DEFT arm on DEFT's side. The trees are written to `out_dir/<role>/model.dill`.

    The fitting rows carry their labels and, when the design has them, their weights under the
    dataset's column names, and the validation rows carry neither. `weight_column` is the one the
    Stage 2 cell recorded.
    """
    sequence_column, target_column = columns.sequence, columns.target
    fit = input_frame(source_cell, columns.inputs(fit_rows), sequence_column)
    fit.insert(0, ROW_ID, np.arange(len(fit)))
    for column in (target_column, weight_column):
        if column is not None and column in fit.columns:
            raise ValueError(f"{source_cell}: a feature column is already called {column!r}.")
    fit[target_column] = np.asarray(y_fit)
    if weight_column is not None:
        fit[weight_column] = fit_rows[weight_column].to_numpy(dtype=float)
    val = input_frame(source_cell, val, sequence_column)
    val.insert(0, ROW_ID, np.arange(len(val)))

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deft_reduce_") as scratch:
        fit_path, val_path = Path(scratch) / "fit.csv", Path(scratch) / "val.csv"
        fit.to_csv(fit_path, index=False)
        val.to_csv(val_path, index=False)
        command = [str(DEFT_PYTHON), str(ADAPTER), "reduce", "--cell", str(source_cell),
                   "--fit", str(fit_path), "--val", str(val_path), "--id-column", ROW_ID,
                   "--target-column", target_column, "--n-target", str(n_target),
                   "--out", str(out_dir)]
        if weight_column is not None:
            command += ["--weight-column", weight_column]
        completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{source_cell}: deft_cart.py reduce exited {completed.returncode}.")

    record = json.loads((out_dir / "elimination.json").read_text())
    with np.load(out_dir / "elimination.npz") as arrays:
        if not np.array_equal(arrays["row_id"], np.arange(len(val))):
            raise RuntimeError(f"{out_dir}: the validation predictions came back out of order.")
        val_by_step = list(np.asarray(arrays["val_by_step"], dtype=float))
        scores = {role: np.asarray(arrays[role], dtype=float)
                  for role in ("full", "reduced", "top_n")}
    return Reduction(
        steps=record["steps"], removed_at=record["removed_at"],
        final_columns=record["final_columns"], final_importance=record["final_importance"],
        top_n_columns=record["top_n_columns"], val_by_step=val_by_step, val_scores=scores,
        featurisation_check=record["featurisation_check"], n_fit_rows=record["n_fit_rows"],
        settings=record["settings"],
    )


#: The file a DEFT arm's pool over D is left in, beside its panel's other files.
POOL_FILE = "pool.npz"


def _adapter(*arguments: str) -> None:
    completed = subprocess.run([str(DEFT_PYTHON), str(ADAPTER), *arguments], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"deft_cart.py {arguments[0]} exited {completed.returncode}.")


def _numbered(frame: pd.DataFrame, path: Path) -> None:
    numbered = frame.copy()
    numbered.insert(0, ROW_ID, np.arange(len(numbered)))
    numbered.to_csv(path, index=False)


def read_pool(path: Path, n_rows: int) -> Pool:
    """What `deft_cart.py pool` wrote, checked to be every row it was handed, in order."""
    return Pool.load(path, n_rows)


class DeftFits:
    """A DEFT arm's panel fits, made on DEFT's side by `deft_cart.py` and read back here.

    The pool is evaluated there, over D and later over the test rows. Step 0 is decided here and
    handed back in `prepare.npz`. The trees are fitted there, on the pool the adapter evaluated. The
    fitting rows define the naive population, as they defined the root of the Stage 2 tree.
    """

    #: A tree is a deterministic function of its rows, weights and seed.
    reproducible = True

    def __init__(self, source: Source, discovery: pd.DataFrame, n_train: int, *,
                 columns: Columns) -> None:
        self.source = source
        self.sequence_column = columns.sequence
        self.discovery = columns.inputs(discovery)
        rows, _, _ = source.fitting_rows(discovery.iloc[:n_train], columns)
        self.fit_inputs = columns.inputs(rows)

    def _cell_args(self, arm_dir: Path) -> list[str]:
        return ["--cell", str(self.source.cell), "--pool", str(arm_dir / POOL_FILE),
                "--prepared", str(arm_dir / PREPARE_ARRAYS)]

    def _evaluate(self, inputs: Inputs, out: Path) -> Pool:
        with tempfile.TemporaryDirectory(prefix="deft_pool_") as scratch:
            fit, rows = Path(scratch) / "fit.csv", Path(scratch) / "rows.csv"
            _numbered(input_frame(self.source.cell, self.fit_inputs, self.sequence_column), fit)
            _numbered(input_frame(self.source.cell, inputs, self.sequence_column), rows)
            _adapter("pool", "--cell", str(self.source.cell), "--fit", str(fit), "--rows", str(rows),
                     "--id-column", ROW_ID, "--sequence-column", self.sequence_column,
                     "--out", str(out))
        return read_pool(out, len(inputs))

    def pool(self, arm_dir: Path) -> Pool:
        path = Path(arm_dir) / POOL_FILE
        if path.exists():
            return read_pool(path, len(self.discovery))
        Path(arm_dir).mkdir(parents=True, exist_ok=True)
        return self._evaluate(self.discovery, path)

    def reproduce(self, arm_dir, pool, prepared, recorded, n_target) -> dict[str, list[str]]:
        out = Path(arm_dir) / "reproduction.json"
        _adapter("reproduce", *self._cell_args(Path(arm_dir)), "--n-target", str(n_target),
                 "--out", str(out))
        found = json.loads(out.read_text())
        return {"from_pool": found["from_pool"], "from_recorded": found["from_recorded"]}

    def select(self, arm_dir, pool, prepared, label_sets, n_target, workers, force,
               memory=None, threads_file=None) -> dict[str, np.ndarray]:
        chosen = {}
        for label_set in label_sets:
            out = Path(arm_dir) / selections_file(label_set)
            _adapter("select", *self._cell_args(Path(arm_dir)), "--label-set", label_set,
                     "--n-target", str(n_target), "--workers", str(workers), "--out", str(out),
                     *(["--force"] if force else []),
                     *(["--memory", str(memory)] if memory is not None else []),
                     *(["--threads-file", str(threads_file)] if threads_file is not None else []))
            with np.load(out) as saved:
                if [str(name) for name in saved["names"]] != prepared.kept_names or not saved["done"].all():
                    raise RuntimeError(f"{out}: not every selection over the kept candidates finished.")
                chosen[label_set] = saved["selected"]
        return chosen

    def refit(self, arm_dir, pool, prepared, plan) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
        arm_dir = Path(arm_dir)
        with tempfile.TemporaryDirectory(prefix="deft_refit_") as scratch:
            plan_file, out = Path(scratch) / "plan.json", Path(scratch) / "scores.npz"
            plan_file.write_text(json.dumps([
                {"label": entry["label"], "columns": entry["columns"],
                 "cell": str(arm_dir / loco_cell(entry["label"]))} for entry in plan]))
            _adapter("refit", *self._cell_args(arm_dir), "--plan", str(plan_file), "--out", str(out))
            with np.load(out) as saved:
                labels = [str(label) for label in saved["labels"]]
                scores = np.asarray(saved["scores"], dtype=float)
                settings = json.loads(str(saved["settings"]))
        if labels != [entry["label"] for entry in plan]:
            raise RuntimeError(f"{arm_dir}: the refits came back for other labels, or in another order.")
        about = saved_tree(settings)
        return {label: (scores[index], about) for index, label in enumerate(labels)}

    def evaluate(self, names: list[str], inputs: Inputs) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="deft_rows_") as scratch:
            evaluated = self._evaluate(inputs, Path(scratch) / "pool.npz")
        return evaluated.values[:, evaluated.index_of(names)]


def export_tree(run_dir: Path, cell: Path, train: pd.DataFrame, val: pd.DataFrame, *,
                columns: Columns) -> dict[str, Any]:
    """A reduced DEFT cell's tree structure, for a report to draw, as `deft_cart.py tree` reads it.

    The fitting rows cross with their labels and weights, since a node's counts are counts of
    those rows. The validation rows cross without them, and the tree's predictions for them must
    match the ones the cell recorded, or the drawing would be of some other tree.
    """
    run = cells.read_run(run_dir / cell)
    source = Source.read(run_dir / run["source_cell"])
    sequence_column, target_column = columns.sequence, columns.target
    rows, y_fit, weights = source.fitting_rows(train, columns)
    fit = input_frame(source.cell, columns.inputs(rows), sequence_column)
    fit.insert(0, ROW_ID, np.arange(len(fit)))
    fit[target_column] = np.asarray(y_fit)
    weight_column = source.run["weight_column"]
    if weight_column is not None:
        fit[weight_column] = weights
    with tempfile.TemporaryDirectory(prefix="deft_tree_") as scratch:
        fit_path, val_path, out = (Path(scratch) / name for name in ("fit.csv", "val.csv", "tree.json"))
        fit.to_csv(fit_path, index=False)
        _numbered(input_frame(source.cell, columns.inputs(val), sequence_column), val_path)
        _adapter("tree", "--cell", str(source.cell), "--model", str(run_dir / cell),
                 "--fit", str(fit_path), "--val", str(val_path), "--id-column", ROW_ID,
                 "--target-column", target_column, "--out", str(out),
                 *(["--weight-column", weight_column] if weight_column else []))
        structure = json.loads(out.read_text())
    if structure.pop("val_row_id") != list(range(len(val))):
        raise RuntimeError(f"{cell}: the validation predictions came back out of order.")
    check_reproduces(cells.validation_scores(run_dir / cell),
                     np.asarray(structure.pop("val_score")), str(cell))
    return {**structure, "arm_cell": str(cell), "source_cell": run["source_cell"]}


def predict_cell(run_dir: Path, cell: Path, inputs: Inputs, sequence_column: str) -> np.ndarray:
    """A reduced or refitted DEFT cell's scores for these rows: the arm's features evaluated by its
    Stage 2 tree, narrowed to the cell's columns, predicted by the cell's tree. The reveal calls
    this, and the rows cross without their labels."""
    source = run_dir / cells.read_run(cell)["source_cell"]
    return predict_numbered(
        [str(DEFT_PYTHON), str(ADAPTER), "predict", "--cell", str(source), "--model", str(cell)],
        input_frame(source, inputs, sequence_column), cell)
