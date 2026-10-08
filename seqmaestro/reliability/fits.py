"""The panel's fits for one of our own arms, made in this process.

Everything is rebuilt from the Stage 2 cell the arm was reduced from, as the reduction rebuilds it.
The featuriser the cell saved computes the candidates, and the model is rebuilt from the
description the cell recorded (`model_settings`) and its seed, and fitted with the rows' weights. A
DEFT arm's fits are made on DEFT's side, through `DeftFits` in
`seqmaestro/pipeline/vendored/deft_cart.py`, behind the same interface (`panel.Fits`). An IGSR
arm's candidates are evaluated on IGSR's side and its fits are made here, by the same `ColumnFits`
as ours (`seqmaestro/pipeline/vendored/igsr_terms.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from seqmaestro.features.base import Inputs
from seqmaestro.harness.fitted import PREDICT_WITH as FITTED
from seqmaestro.harness.fitted import load_fitted, save_fitted
from seqmaestro.harness.runner import evaluate_columns
from seqmaestro.models import one_core as held_to_one_core
from seqmaestro.models import reproducible
from seqmaestro.models.base import FILLED_ROWS
from seqmaestro.reduction import eliminate
from seqmaestro.reduction.arm import SelectedColumns, Source, fit_step_for
from seqmaestro.reliability import pool as pools
from seqmaestro.reliability import selections
from seqmaestro.reliability.files import loco_cell, selections_file

if TYPE_CHECKING:
    from seqmaestro.dataset import Columns

def one_thread(fit_step):
    """`fit_step`, run with the numeric libraries' thread pools held to one thread.

    Tier 1's workers already run one selection per core, and each forked worker inherits a BLAS pool
    sized to the machine, so without the limit every core carries a pool's worth of threads. A
    sparse linear model's coordinate descent makes a BLAS call per coordinate, and on a large half
    of D it then runs many times slower. The limit changes how fast a fit runs. A sum taken on one
    thread can round differently from one split across threads, as between two machines.
    """
    def step(*args, **kwargs):
        with threadpool_limits(limits=1):
            return fit_step(*args, **kwargs)
    return step


class ColumnFits:
    """Fits on an arm's candidates, computed once over D, with a model fitted in this process.

    The selections and the refits need only the candidates' values, a fit step and a way to save a
    refitted model, so they are written once here. A subclass says how the candidates are computed
    (`pool`), which model is fitted and how a refit is saved: `InProcess` for our own arms, and the
    IGSR arm's side (`seqmaestro/pipeline/vendored/igsr_terms.py`), whose terms IGSR evaluates and
    whose model is fitted here.
    """

    #: Whether refitting on the same rows returns the same model, which the reproduction check needs.
    reproducible = True

    def _fit_step(self, weights: np.ndarray | None, *, one_core: bool = False):
        raise NotImplementedError

    def _save(self, cell: Path, model, columns: list[str]) -> dict[str, Any]:
        """Write a refitted model into its cell. Returns what the cell's `run.json` says about it."""
        raise NotImplementedError

    def reproduce(self, arm_dir, pool, prepared, recorded, n_target) -> dict[str, list[str]]:
        fit = prepared.fit_index
        y = prepared.labels["real"][fit]
        weights = None if prepared.weights is None else prepared.weights[fit]
        found = {}
        for start, names in (("from_pool", pool.names), ("from_recorded", recorded)):
            select = selections.selector(eliminate.run, names, n_target)
            found[start] = select(self._fit_step(weights),
                                  np.ascontiguousarray(pool.values[np.ix_(fit, pool.index_of(names))]),
                                  y)
        return found

    def select(self, arm_dir, pool, prepared, label_sets, n_target, workers, force,
               memory=None, threads_file=None) -> dict[str, np.ndarray]:
        names = prepared.kept_names
        X = np.ascontiguousarray(pool.values[:, prepared.kept_index])
        chosen = {}
        for label_set in label_sets:
            chosen[label_set], _ = selections.run(
                selections.selector(eliminate.run, names, n_target), X, prepared.labels[label_set],
                names, prepared.halves,
                make_fit_step=lambda weights: one_thread(self._fit_step(weights, one_core=True)),
                weights=prepared.weights, out_path=Path(arm_dir) / selections_file(label_set),
                workers=workers, force=force, memory=memory, threads_file=threads_file)
        return chosen

    def refit(self, arm_dir, pool, prepared, plan) -> dict[str, tuple[np.ndarray, dict]]:
        fit, val = prepared.fit_index, prepared.val_index
        weights = None if prepared.weights is None else prepared.weights[fit]
        fit_step = self._fit_step(weights)
        fitted = {}
        for entry in plan:
            columns = pool.index_of(entry["columns"])
            model = fit_step(pool.values[np.ix_(fit, columns)], prepared.labels["real"][fit],
                             entry["columns"])
            cell = Path(arm_dir) / loco_cell(entry["label"])
            cell.mkdir(parents=True, exist_ok=True)
            about = self._save(cell, model.model, entry["columns"])
            filled = model.filled(pool.values[np.ix_(val, columns)])
            if filled is not None:
                about[FILLED_ROWS] = filled
            fitted[entry["label"]] = (model.predict(pool.values[np.ix_(val, columns)]), about)
        return fitted


class InProcess(ColumnFits):
    """One of our arms: its featuriser and model loaded and fitted here."""

    def __init__(self, source: Source, discovery: pd.DataFrame, columns: Columns) -> None:
        self.source = source
        self.inputs = columns.inputs(discovery)
        self.sequence_column = columns.sequence
        self.target_kind = columns.target_kind
        self.featuriser = load_fitted(source.cell).featuriser
        self.description = source.run["model_settings"]
        self.seed = source.run["seed"]
        self.reproducible = reproducible(self.description["model"])

    def _fit_step(self, weights: np.ndarray | None, *, one_core: bool = False):
        # Tier 1's workers already run one fit per core, so each of their fits is held to one
        # (`seqmaestro.models.one_core`). Without that an EBM spreads its bags across threads and
        # competes with the process pool, and each worker's PySR search runs a thread per core.
        # The option changes how a fit runs and never which model it fits, so the reported models,
        # the LOCO refits, are fitted without it, as Stage 2 fitted them.
        execution = held_to_one_core(self.description["model"]) if one_core else None
        return fit_step_for(self.description, self.seed, weights, execution,
                            target_kind=self.target_kind)

    def _save(self, cell: Path, model, columns: list[str]) -> dict[str, Any]:
        save_fitted(cell, SelectedColumns(self.featuriser, columns), model,
                    sequence_column=self.sequence_column)
        return {"predict_with": FITTED, "model_settings": model.describe()}

    def pool(self, arm_dir: Path) -> pools.Pool:
        return pools.from_featuriser(self.featuriser, self.inputs)

    def evaluate(self, names: list[str], inputs: Inputs) -> np.ndarray:
        return evaluate_columns(self.featuriser, inputs, names)
