"""Where a fitted model lives, and what the stages after Stage 2 ask of it there.

A model fitted in Core is saved whole in its cell (`fitted.pkl`), and any Core process loads it.
DEFT's tree carries its features as functions, and an IGSR model's terms are code its language model
wrote, so both are evaluated only under their method's interpreter, through an adapter. Each of the
three is a *side*: `CORE`, `DEFT` and `IGSR`.

The stages after Stage 2 ask the same six things of an arm, whichever side its model is on:

- `reduce`: eliminate the arm's features to N (Stage 3A).
- `fits`: make the reliability panel's fits (Stage 3B).
- `is_saved`: say whether a cell holds the model its `predict_with` promises (the reveal, before it
  opens the test split).
- `predict`: score rows with a cell's saved model (the reveal).
- `reported_columns`: the reduced model's features over D, for the report's concepts.
- `internals`: what is inside the reduced model, for the report's drawing of it.

**A cell says which side wrote it**, in the `predict_with` of its `run.json`. Each side has one
value for its Stage 2 cells and one for the cells fitted later from them (a reduced arm's three
models, the panel's refits), because a later cell keeps a different model: a tree over the Stage 2
tree's features, or a linear model over the Stage 2 search's terms. `side_of` finds the side from
the value, so a stage never asks which method an arm belongs to.

A new vendored method adds a side here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqmaestro.arms import Arm
from seqmaestro.dataset import Columns, Dataset
from seqmaestro.features.base import Inputs
from seqmaestro.harness import cell as cells
from seqmaestro.harness.fitted import FILENAME as FITTED_FILE
from seqmaestro.harness.fitted import PREDICT_WITH as FITTED
from seqmaestro.harness.fitted import load_fitted, save_fitted
from seqmaestro.pipeline.layout import REDUCED_MODEL, reduction_dir, reliability_dir
from seqmaestro.pipeline.rows import Rows
from seqmaestro.pipeline.vendored import deft, deft_cart, igsr, igsr_terms
from seqmaestro.pipeline.vendored.handover import BUNDLE
from seqmaestro.reduction.arm import Reduction, SelectedColumns, Source, reduce_in_process
from seqmaestro.reliability.fits import InProcess
from seqmaestro.reliability.panel import Fits
from seqmaestro.report import internals as insides

#: Writes one of a reduced arm's three models into its cell and returns what the cell's `run.json`
#: says about it: `(role, cell)`.
SaveModel = Callable[[str, Path], dict[str, Any]]


class Side:
    """One place a model lives. A subclass implements each of the six requests for its place."""

    #: What a Stage 2 cell of this side records in `predict_with`, and what a cell fitted later
    #: from one records.
    stage2: str
    later: str

    def is_saved(self, cell: Path, run: Mapping[str, Any]) -> bool:
        """Whether the cell, whose `run.json` is `run`, holds the model its `predict_with` promises.
        The reveal asks before it opens the test split, so that a missing model stops it while it
        can still be stopped."""
        raise NotImplementedError

    def reduce(self, source: Source, rows: Rows, *, n_target: int, out_dir: Path,
               on_step: Callable[[dict], None]) -> tuple[Reduction, SaveModel]:
        """Eliminate the arm fitted in `source` to `n_target` features, on validation only."""
        raise NotImplementedError

    def fits(self, source: Source, discovery: pd.DataFrame, n_train: int, columns: Columns) -> Fits:
        """The panel's fits for the arm fitted in `source`, over D, whose first `n_train` rows are
        the training rows."""
        raise NotImplementedError

    def predict(self, run_dir: Path, cell: Path, run: Mapping[str, Any], inputs: Inputs,
                dataset: Dataset) -> tuple[np.ndarray, int | None]:
        """The saved model's score for every row of `inputs`, in their order: the probability of
        the positive class for a binary target, the predicted value for a continuous one. With it,
        how many rows took the model's constant prediction, for a model that can be undefined on a
        row, or None. The rows cross to a vendored method without their labels."""
        raise NotImplementedError

    def reported_columns(self, run_dir: Path, arm: Arm, names: list[str], source: Source,
                         rows: Rows) -> np.ndarray:
        """The reduced model's features `names` over D, one column each."""
        fits = self.fits(source, rows.discovery(), len(rows.train), rows.dataset.columns)
        pool = fits.pool(reliability_dir(run_dir, *arm))
        return pool.values[:, pool.index_of(names)]

    def internals(self, run_dir: Path, cell: Path, record: Mapping[str, Any], rows: Rows,
                  marginal: Mapping[str, float | None]) -> dict[str, Any]:
        """What is inside the reduced model in `cell` (relative to the run), as plain data a figure
        is drawn from. `record` is its arm's reduction record, and `marginal` each of its
        features' marginal effect."""
        raise NotImplementedError


def _loaded(run_dir: Path, cell: Path, record: Mapping[str, Any], rows: Rows,
            marginal: Mapping[str, float | None], *, model, columns: list[str],
            predicted: np.ndarray, values_on: Callable[[Inputs], np.ndarray]) -> dict[str, Any]:
    """The insides of a reduced model that this process loaded, once its validation predictions are
    seen to be the ones its cell recorded. `values_on` computes the model's columns on given rows."""
    dataset = rows.dataset
    insides.check_reproduces(cells.validation_scores(run_dir / cell), predicted, str(cell))
    source = Source.read(run_dir / record["source_cell"])
    fit_rows, _, _ = source.fitting_rows(rows.train, dataset.columns)
    placed = {"arm_cell": str(cell), "source_cell": record["source_cell"],
              "n_fit_rows": len(fit_rows)}
    empty = insides.no_features(model, predicted, str(cell))
    if empty is not None:
        return {**empty, **placed}
    read = insides.read(model, columns, lambda: values_on(dataset.inputs(fit_rows)), marginal)
    return {**read, **placed}


class CoreSide(Side):
    """A model fitted in this environment, saved with its featuriser as `fitted.pkl`."""

    stage2 = later = FITTED

    def is_saved(self, cell: Path, run: Mapping[str, Any]) -> bool:
        return (cell / FITTED_FILE).exists()

    def reduce(self, source, rows, *, n_target, out_dir, on_step):
        dataset = rows.dataset
        reduction = reduce_in_process(source, rows.train, rows.val, columns=dataset.columns,
                                      n_target=n_target, on_step=on_step)

        def save_model(role: str, cell: Path) -> dict[str, Any]:
            model, names = reduction.models[role]
            save_fitted(cell, SelectedColumns(reduction.featuriser, names), model,
                        sequence_column=dataset.sequence_column)
            return {"predict_with": FITTED, "model_settings": model.describe()}
        return reduction, save_model

    def fits(self, source, discovery, n_train, columns):
        return InProcess(source, discovery, columns)

    def predict(self, run_dir, cell, run, inputs, dataset):
        # The rows carry every covariate the dataset declares. A featuriser that reads covariates
        # picks out the ones it was fitted with, and refuses rows that lack one.
        fitted = load_fitted(cell)
        if fitted.sequence_column not in (None, dataset.sequence_column):
            raise ValueError(f"{cell} was fitted on column {fitted.sequence_column!r}, but the "
                             f"dataset's sequence column is {dataset.sequence_column!r}.")
        return fitted.predict_counting(inputs)

    def reported_columns(self, run_dir, arm, names, source, rows):
        # The reduced cell's saved featuriser computes exactly the reduced model's columns, so the
        # whole pool is not built for them.
        cell = reduction_dir(run_dir, *arm) / REDUCED_MODEL
        fitted = load_fitted(cell)
        if fitted.featuriser.feature_names != names:
            raise RuntimeError(f"{cell}: the saved model's columns are not the reduced ones.")
        return fitted.transform(rows.dataset.inputs(rows.discovery())).to_numpy(dtype=float)

    def internals(self, run_dir, cell, record, rows, marginal):
        fitted = load_fitted(run_dir / cell)
        return _loaded(
            run_dir, cell, record, rows, marginal, model=fitted.model,
            columns=fitted.featuriser.feature_names,
            predicted=fitted.predict(rows.dataset.inputs(rows.val)),
            values_on=lambda inputs: fitted.transform(inputs).to_numpy(dtype=float))


class VendoredSide(Side):
    """A method whose models its interpreter alone uses. A cell's `run.json` names the model's
    file in `fitted_model`: inside the bundle the method handed back for a Stage 2 cell, and inside
    the cell for one fitted later."""

    def is_saved(self, cell: Path, run: Mapping[str, Any]) -> bool:
        name = run.get("fitted_model")
        home = cell / BUNDLE if run["predict_with"] == self.stage2 else cell
        return isinstance(name, str) and (home / name).is_file()


class DeftSide(VendoredSide):
    """DEFT's tree. A Stage 2 cell keeps the tree in its bundle, and a later cell keeps a tree
    fitted over that tree's features. Both are read by DEFT's interpreter alone."""

    stage2, later = deft.PREDICT_WITH, deft_cart.PREDICT_WITH

    def reduce(self, source, rows, *, n_target, out_dir, on_step):
        columns = rows.dataset.columns
        fit_rows, y_fit, _ = source.fitting_rows(rows.train, columns)
        reduction = deft_cart.reduce_arm(
            source.cell, fit_rows, y_fit, columns.inputs(rows.val), columns=columns,
            weight_column=source.run["weight_column"], n_target=n_target, out_dir=out_dir)

        def save_model(role: str, cell: Path) -> dict[str, Any]:
            # The tree is already in the cell, written on DEFT's side.
            return deft_cart.saved_tree(reduction.settings["tree"])
        return reduction, save_model

    def fits(self, source, discovery, n_train, columns):
        return deft_cart.DeftFits(source, discovery, n_train, columns=columns)

    def predict(self, run_dir, cell, run, inputs, dataset):
        if run["predict_with"] == self.stage2:
            return deft.predict_cell(cell, inputs, dataset.sequence_column), None
        return deft_cart.predict_cell(run_dir, cell, inputs, dataset.sequence_column), None

    def internals(self, run_dir, cell, record, rows, marginal):
        return deft_cart.export_tree(run_dir, cell, rows.train, rows.val,
                                     columns=rows.dataset.columns)


class IgsrSide(VendoredSide):
    """IGSR. A Stage 2 cell keeps the model of its search in its bundle. A later cell keeps a
    least-squares model fitted here on terms of that search, which IGSR's interpreter evaluates."""

    stage2, later = igsr.PREDICT_WITH, igsr_terms.PREDICT_WITH

    def reduce(self, source, rows, *, n_target, out_dir, on_step):
        reduction = igsr_terms.reduce_arm(source, rows.train, rows.val,
                                          columns=rows.dataset.columns, n_target=n_target,
                                          on_step=on_step)

        def save_model(role: str, cell: Path) -> dict[str, Any]:
            model, _ = reduction.models[role]
            return igsr_terms.save_model(cell, model)
        return reduction, save_model

    def fits(self, source, discovery, n_train, columns):
        return igsr_terms.IgsrFits(source, discovery, columns=columns)

    def predict(self, run_dir, cell, run, inputs, dataset):
        if run["predict_with"] == self.stage2:
            return igsr.predict_cell(cell, inputs, dataset.sequence_column), None
        return igsr_terms.predict_cell(run_dir, cell, inputs, dataset.sequence_column), None

    def internals(self, run_dir, cell, record, rows, marginal):
        dataset, columns = rows.dataset, list(record["final_columns"])
        source_cell = run_dir / record["source_cell"]
        return _loaded(
            run_dir, cell, record, rows, marginal, model=igsr_terms.load_model(run_dir / cell),
            columns=columns,
            predicted=igsr_terms.predict_cell(run_dir, run_dir / cell, dataset.inputs(rows.val),
                                              dataset.sequence_column),
            values_on=lambda inputs: igsr_terms.evaluate_terms(
                source_cell, columns, inputs, dataset.sequence_column).values)


CORE, DEFT, IGSR = CoreSide(), DeftSide(), IgsrSide()
SIDES = (CORE, DEFT, IGSR)


def side_of(run: Mapping[str, Any], where: str | Path = "the cell") -> Side:
    """The side that wrote the cell whose `run.json` is `run`."""
    recorded = run.get("predict_with")
    for side in SIDES:
        if recorded in (side.stage2, side.later):
            return side
    raise ValueError(f"{where} records `predict_with: {recorded!r}`, which no side predicts. "
                     f"Expected one of "
                     f"{', '.join(sorted({v for s in SIDES for v in (s.stage2, s.later)}))}.")

