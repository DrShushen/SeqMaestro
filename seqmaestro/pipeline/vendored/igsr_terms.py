"""The Core side of an IGSR arm after Stage 2: what the later stages ask of it.

An IGSR arm's features are terms its language model wrote as functions of the rows, so they are
evaluated by IGSR's copy, under the guard and the timeout its search used, and never here:
`evaluate_terms` hands the rows to `igsr_bundle.py evaluate` under IGSR's interpreter, numbered and
without their target, and reads the values back. What is fitted on those values is the fit IGSR's
search made, which the Stage 2 bundle records (`igsr_fit`): least squares, plain or with IGSR's ridge
or lasso penalty. It needs nothing of IGSR, so it is made here
(`seqmaestro.models.sparse.LeastSquares`), with the elimination, the selections and the refits every
in-process arm uses.

Each later stage has its entry point here: `reduce_arm` for Stage 3A, `IgsrFits` for the reliability
panel (a `seqmaestro.reliability.fits.ColumnFits` whose pool IGSR evaluates), `predict_cell` for the
reveal of a reduced or refitted cell, and `load_model` for the report.
`seqmaestro.pipeline.sides` calls them for a cell IGSR's side wrote. Every term the arm's search
accepted is its pool, in `terms.json` of its Stage 2 bundle.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqmaestro.dataset import Columns
from seqmaestro.features.base import Inputs
from seqmaestro.harness import cell as cells
from seqmaestro.models.sparse import LeastSquares
from seqmaestro.pipeline.vendored.handover import BUNDLE, COLUMNS_MANIFEST, INPUTS, ROW_ID, input_frame
from seqmaestro.pipeline.vendored.igsr import ADAPTER, IGSR_PYTHON
from seqmaestro.reduction.arm import Reduction, Source, fit_step_with, reduce_columns
from seqmaestro.reliability.fits import ColumnFits
from seqmaestro.reliability.pool import RECORDED_FEATURES, SEARCH_HISTORY, Pool
from seqmaestro.target import CONTINUOUS

#: What a reduced or refitted IGSR cell's `run.json` says about how the reveal predicts it:
#: `predict_cell`.
PREDICT_WITH = "igsr_terms"
#: The least-squares model inside a reduced or refitted IGSR cell.
MODEL_FILE = "model.pkl"
#: The file an IGSR arm's pool over D is left in, beside its panel's other files.
POOL_FILE = "pool.npz"


def evaluate_terms(source_cell: Path, names: list[str] | None, inputs: Inputs, sequence_column: str,
                   source: str = RECORDED_FEATURES) -> Pool:
    """Terms of a Stage 2 IGSR cell's search on these rows, as IGSR evaluates them, with each
    term's recipe: the named ones, or every term the search accepted when `names` is None. The rows
    hold the columns the cell was handed (`input_frame`), numbered, and the order is checked on
    return."""
    frame = input_frame(source_cell, inputs, sequence_column)
    frame.insert(0, ROW_ID, np.arange(len(frame)))
    with tempfile.TemporaryDirectory(prefix="igsr_terms_") as scratch:
        rows, out, wanted = (Path(scratch) / name
                             for name in ("rows.csv", "values.npz", "names.json"))
        frame.to_csv(rows, index=False)
        command = [str(IGSR_PYTHON), str(ADAPTER), "evaluate",
                   "--bundle", str(source_cell / BUNDLE),
                   "--columns", str(source_cell / INPUTS / COLUMNS_MANIFEST),
                   "--rows", str(rows), "--id-column", ROW_ID, "--out", str(out)]
        if names is not None:
            wanted.write_text(json.dumps(list(names)))
            command += ["--names", str(wanted)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{source_cell}: igsr_bundle.py evaluate exited "
                               f"{completed.returncode}:\n{completed.stderr[-2000:]}")
        with np.load(out) as saved:
            if not np.array_equal(saved["row_id"], np.arange(len(frame))):
                raise RuntimeError(f"{source_cell}: the terms came back for other rows, or in "
                                   f"another order.")
            evaluated = Pool(names=[str(name) for name in saved["names"]],
                             kinds=[str(kind) for kind in saved["kinds"]],
                             params=[json.loads(str(params)) for params in saved["params"]],
                             descriptions=[str(text) for text in saved["descriptions"]],
                             values=np.asarray(saved["values"], dtype=float), source=source)
    if names is not None and evaluated.names != list(names):
        raise RuntimeError(f"{source_cell}: the terms came back under other names or in another "
                           f"order.")
    return evaluated


def igsr_fit(source_cell: Path) -> dict[str, Any]:
    """The fit a Stage 2 IGSR cell's search made, as its bundle records it: `LeastSquares`'s
    penalty, alpha and intercept."""
    arm = json.loads((Path(source_cell) / BUNDLE / "arm.json").read_text())
    if "fit" not in arm:
        raise ValueError(f"{source_cell}: its bundle does not record the fit IGSR's search made, so "
                         f"the arm cannot be refitted as IGSR fitted it. Run the cell again.")
    return arm["fit"]


def fit_step(fit: dict[str, Any], weights: np.ndarray | None):
    """The elimination's fit on an IGSR arm's terms: the fit its search made (`igsr_fit`)."""
    return fit_step_with(lambda: LeastSquares(**fit), weights, target_kind=CONTINUOUS)


def reduce_arm(source: Source, train: pd.DataFrame, val: pd.DataFrame, *, columns: Columns,
               n_target: int, on_step=None) -> Reduction:
    """Eliminate an IGSR arm's terms to `n_target`, by least squares on their values, which IGSR
    evaluates on the cell's fitting rows and on the validation rows."""
    rows, y_fit, weights = source.fitting_rows(train, columns)
    names = source.names
    X_fit = evaluate_terms(source.cell, names, columns.inputs(rows), columns.sequence).values
    X_val = evaluate_terms(source.cell, names, columns.inputs(val), columns.sequence).values
    return reduce_columns(source, X_fit, y_fit, X_val, fit_step(igsr_fit(source.cell), weights),
                          n_target, on_step=on_step)


def save_model(cell: Path, model) -> dict[str, Any]:
    """Write a model fitted on an IGSR arm's terms into its cell. Returns what the cell's
    `run.json` says about it, which is how the reveal finds `predict_cell`."""
    cell.mkdir(parents=True, exist_ok=True)
    with open(cell / MODEL_FILE, "wb") as handle:
        pickle.dump(model, handle)
    return {"predict_with": PREDICT_WITH, "fitted_model": MODEL_FILE,
            "model_settings": model.describe()}


def load_model(cell: Path):
    with open(Path(cell) / MODEL_FILE, "rb") as handle:
        return pickle.load(handle)


class IgsrFits(ColumnFits):
    """An IGSR arm's panel fits: its pool, every term its search accepted, evaluated over D by IGSR
    and saved beside the panel's other files, and the fit its search made, fitted here on it."""

    def __init__(self, source: Source, discovery: pd.DataFrame, *, columns: Columns) -> None:
        self.source = source
        self.discovery = columns.inputs(discovery)
        self.sequence_column = columns.sequence
        self.fit = igsr_fit(source.cell)

    def _fit_step(self, weights: np.ndarray | None, *, one_core: bool = False):
        return fit_step(self.fit, weights)

    def _save(self, cell: Path, model, columns: list[str]) -> dict[str, Any]:
        return save_model(cell, model)

    def pool(self, arm_dir: Path) -> Pool:
        path = Path(arm_dir) / POOL_FILE
        if path.exists():
            return Pool.load(path, len(self.discovery))
        pool = evaluate_terms(self.source.cell, None, self.discovery, self.sequence_column,
                              source=SEARCH_HISTORY)
        Path(arm_dir).mkdir(parents=True, exist_ok=True)
        pool.save(path)
        return pool

    def evaluate(self, names: list[str], inputs: Inputs) -> np.ndarray:
        return evaluate_terms(self.source.cell, names, inputs, self.sequence_column).values


def predict_cell(run_dir: Path, cell: Path, inputs: Inputs, sequence_column: str) -> np.ndarray:
    """A reduced or refitted IGSR cell's predictions for these rows: its columns, evaluated by IGSR
    from the Stage 2 cell's terms, and its least-squares model. The reveal calls this, and the rows
    cross without their target."""
    run = cells.read_run(cell)
    columns = run["columns"]
    values = evaluate_terms(Path(run_dir) / run["source_cell"], columns, inputs,
                            sequence_column).values
    return np.asarray(load_model(cell).predict(pd.DataFrame(values, columns=columns)), dtype=float)
