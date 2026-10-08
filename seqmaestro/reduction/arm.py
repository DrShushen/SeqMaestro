"""One arm, reduced in this process: its seed, its columns, its model, the elimination.

Everything is rebuilt from what the Stage 2 cell recorded, and no setting is declared a second
time. The seed is that of the real-label cell with the best validation headline. The rows are the
ones that cell was fitted on, from its `train_fraction`, seed and weight column. The columns come
from the featuriser the cell saved (the same values its model was fitted on) and are checked against
the values it recorded. The model is rebuilt from the description the cell recorded
(`model_settings`), and refused if the rebuild does not describe itself the same way. The reduction
therefore eliminates from the model the cell reported, and changes only its columns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from seqmaestro.features.base import FeatureRecord, Featuriser, Inputs
from seqmaestro.harness import cell as cells
from seqmaestro.harness.features import load_features
from seqmaestro.harness.fitted import load_fitted
from seqmaestro.harness.runner import evaluate_columns, subsample_training_rows
from seqmaestro.models import make_model
from seqmaestro.models.base import Model, filled_rows
from seqmaestro.models.weights import resolve_fit_weights
from seqmaestro.reduction import check, eliminate

if TYPE_CHECKING:
    from seqmaestro.dataset import Columns

#: The three models every reduced arm keeps, and what each one is.
ROLES = {
    "full": "the arm's model refitted on every feature its Stage 2 fit used: the first step",
    "reduced": "the model the elimination arrives at, on N features",
    "top_n": "the N features the full model ranked highest, fitted once: the control for whether "
             "the recursion earned its cost",
}


#: Description keys that a model fills in only once fitted, or that the rebuild sets itself.
_NOT_SETTINGS = frozenset({"per_row_weights", "max_bins", "alpha", "trees_kept", "seed", "fitted"})


def best_seed(validation: dict[int, dict], headline: str) -> int:
    """The seed with the highest validation `headline` (the kind of target's, `seqmaestro.target`);
    ties go to the smaller seed.

    Validation is used because it is the only split read before the reveal. For an arm on the
    iterative source it is also the split that steered its proposals, so its best seed is partly the
    one that was steered best. The report template has a report of such an arm say so
    (`docs/report/TEMPLATE.md`, Stage 3A).
    """
    if not validation:
        raise ValueError("No seeds to choose from.")
    return min(validation, key=lambda seed: (-validation[seed][headline], seed))


def rebuild_model(description: dict[str, Any], seed: int, target_kind: str,
                  execution: dict[str, Any] | None = None) -> Model:
    """A fresh model configured as the cell's was, refused if it would not be the same model.

    Stage 2 builds a model from a name, the target's kind, a seed and, for a binary target, a
    class weight, so those are what is passed. The fresh model's description is then compared with
    the cell's, because a setting the rebuild dropped would otherwise fit a different model under
    the arm's name, and a cell fitted against another kind of target is refused the same way. The
    comparison runs both ways, so a setting the model gained after the cell was fitted is refused
    too. `execution` holds options that change how a fit runs and not what it fits, such as how many
    cores it may use, and is left out of the comparison.
    """
    options: dict[str, Any] = {"seed": seed, **(execution or {})}
    if "class_weight" in description:
        options["class_weight"] = description["class_weight"]
    model = make_model(description["model"], target_kind=target_kind, **options)
    fresh = model.describe()
    differing = sorted(
        key for key in (set(description) | set(fresh)) - _NOT_SETTINGS - set(execution or {})
        if key not in description or key not in fresh or description[key] != fresh[key]
    )
    if differing:
        shown = ", ".join(f"{key}={_setting(description, key)} (rebuilt {_setting(fresh, key)})"
                          for key in differing)
        raise ValueError(f"The cell's {description['model']} was fitted with settings a rebuild "
                         f"does not reproduce: {shown}. The reduction would fit a different model "
                         f"under the arm's name.")
    return model


def _setting(description: dict[str, Any], key: str) -> str:
    return repr(description[key]) if key in description else "absent"


class SelectedColumns(Featuriser):
    """A fitted featuriser narrowed to the columns a reduced model was fitted on.

    A reduced model is saved with one, so what `load_fitted` returns turns rows into exactly that
    model's inputs, which is how the reveal predicts it.
    """

    def __init__(self, featuriser: Featuriser, names: list[str]) -> None:
        super().__init__()
        by_name = {record.name: record for record in featuriser.records}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"The featuriser has no column(s) {missing[:5]}.")
        self.featuriser = featuriser
        self._records = [by_name[name] for name in names]

    def _fit(self, inputs: Inputs, y=None, sample_weight=None) -> list[FeatureRecord]:
        raise TypeError("SelectedColumns is built from a fitted featuriser and is never refitted.")

    def _transform(self, inputs: Inputs) -> np.ndarray:
        return evaluate_columns(self.featuriser, inputs, self.feature_names)

    def describe(self) -> dict[str, Any]:
        # The featuriser it narrows is described in the Stage 2 cell it came from, and is not
        # described again here: a restored featuriser cannot always describe itself, since
        # fit-time machinery such as a proposer is not saved.
        return {"kind": "selected_columns", "columns": self.feature_names,
                "of": type(self.featuriser).__name__}


class Fitted:
    """A fitted model, reduced to the two things elimination needs."""

    def __init__(self, model: Model, names: list[str]) -> None:
        self.model = model
        self.columns = names

    def importances(self) -> np.ndarray:
        """Absolute importance per column, zero where the model did not use it.

        Absolute, because an L1 coefficient is signed and a large negative weight is as influential
        as a large positive one. `rank_by_importance` in the harness does the same.
        """
        reported = self.model.importances()
        return np.array([abs(float(reported.get(name, 0.0))) for name in self.columns])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(_frame(X, self.columns)), dtype=float)

    def filled(self, X: np.ndarray) -> int | None:
        """How many rows took the model's constant prediction (`models.base.filled_rows`)."""
        return filled_rows(self.model, _frame(X, self.columns))


def _frame(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, columns=names, copy=False)


class NoFeatures:
    """The model given no feature: every row gets the mean of the fitting targets, weighted as the
    model it stands in for weights them. For a binary target that mean is the positive rate.

    Tier 3 refits a model without each of its features, and without each cluster of them. Where one
    feature or one cluster is the whole model, the refit has nothing left to use, and all it can
    still do is predict the average the fit saw. The test then asks whether the model beats that
    constant. With balanced class weights the constant is one half.
    """

    name = "no_features"
    importance_kind = "none"
    direction_kind = "none"
    interpretable = True
    reproducible = True

    def __init__(self, target_kind: str, class_weight: str | None = None) -> None:
        self.target_kind = target_kind
        self.class_weight = class_weight
        self.prediction: float | None = None

    def fit(self, X: pd.DataFrame, y, sample_weight: np.ndarray | None = None) -> NoFeatures:
        y = np.asarray(y, dtype=float).ravel()
        self.prediction = float(np.average(
            y, weights=resolve_fit_weights(y, self.class_weight, sample_weight)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.prediction, dtype=float)

    def importances(self) -> dict[str, float]:
        return {}

    def directions(self, X: pd.DataFrame) -> dict[str, str]:
        return {}

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "class_weight": self.class_weight,
                "target_kind": self.target_kind, "prediction": self.prediction}

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        return {}


def fit_step_with(build: Callable[[], Model], sample_weight: np.ndarray | None, *,
                  target_kind: str, class_weight: str | None = None):
    """The elimination's `fit_step`: a fresh model from `build`, fitted with the rows' weights.
    Given no columns, it fits `NoFeatures`, weighted as the model it stands in for. A Tier 3 refit
    of a model without its only feature or cluster asks for that, and so does the reduction of an
    arm whose Stage 2 model used no feature."""
    def fit_step(X: np.ndarray, y: np.ndarray, names: list[str]) -> Fitted:
        model = build() if names else NoFeatures(target_kind, class_weight)
        model.fit(_frame(X, names), y, sample_weight=sample_weight)
        return Fitted(model, names)
    return fit_step


def fit_step_for(description: dict[str, Any], seed: int, sample_weight: np.ndarray | None,
                 execution: dict[str, Any] | None = None, *, target_kind: str):
    """The elimination's `fit_step` for one of our arms: its model, rebuilt from the cell's
    description (`model_settings`), with its seed and its fitting weights."""
    return fit_step_with(lambda: rebuild_model(description, seed, target_kind, execution),
                         sample_weight, target_kind=target_kind,
                         class_weight=description.get("class_weight"))


@dataclass
class Source:
    """The Stage 2 cell an arm is reduced from, and what it recorded."""

    cell: Path
    run: dict[str, Any]
    records: list[FeatureRecord]

    @classmethod
    def read(cls, cell: str | Path) -> Source:
        cell = Path(cell)
        records, _ = load_features(cell / cells.FEATURES_FILE)
        return cls(cell, cells.read_run(cell), records)

    @property
    def names(self) -> list[str]:
        return [record.name for record in self.records]

    def fitting_rows(self, train: pd.DataFrame, columns: Columns
                     ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        """The rows, real labels and weights the cell was fitted on."""
        if self.run.get("labels") != cells.REAL:
            raise ValueError(f"{self.cell} is a {self.run.get('labels')}-label cell; only real-label "
                             f"cells are reduced, since the permuted ones are the null.")
        rows, y = subsample_training_rows(
            train, seed=self.run["seed"], columns=columns,
            train_fraction=self.run["train_fraction"], permute_labels=False,
            permutation_key=None)
        weight_column = self.run["weight_column"]
        weights = rows[weight_column].to_numpy(dtype=float) if weight_column else None
        return rows, y, weights


@dataclass
class Reduction:
    """What eliminating one arm produced, in the shape both environments hand back."""

    steps: list[dict]
    removed_at: list[int]
    final_columns: list[str]
    final_importance: list[float]
    top_n_columns: list[str]
    val_by_step: list[np.ndarray]
    #: Validation scores of the three models, by role.
    val_scores: dict[str, np.ndarray]
    featurisation_check: dict
    n_fit_rows: int
    #: The fitted models by role, with their columns, for the side that can save them.
    models: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    #: The Stage 2 featuriser the columns came from, when they were built in this process.
    featuriser: Featuriser | None = None
    #: By role, how many validation rows took the model's constant prediction, for a model that can
    #: be undefined on a row (`models.base.filled_rows`), and None for any other. Empty where the
    #: arm was eliminated on DEFT's side.
    filled: dict[str, int | None] = field(default_factory=dict)


def reduce_in_process(source: Source, train: pd.DataFrame, val: pd.DataFrame, *,
                      columns: Columns, n_target: int, on_step=None) -> Reduction:
    """Eliminate one of our own arms to `n_target` features, and fit the top-N control."""
    featuriser = load_fitted(source.cell).featuriser
    names = source.names
    rows, y_fit, weights = source.fitting_rows(train, columns)
    X_fit = evaluate_columns(featuriser, columns.inputs(rows), names)
    X_val = evaluate_columns(featuriser, columns.inputs(val), names)
    fit_step = fit_step_for(source.run["model_settings"], source.run["seed"], weights,
                            target_kind=columns.target_kind)
    reduction = reduce_columns(source, X_fit, y_fit, X_val, fit_step, n_target, on_step=on_step)
    reduction.featuriser = featuriser
    return reduction


def reduce_columns(source: Source, X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray,
                   fit_step, n_target: int, on_step=None) -> Reduction:
    """Eliminate an arm's recorded features to `n_target`, and fit the top-N control, on their
    values over the fitting and validation rows, however those were computed. The validation
    values are checked against the ones the cell recorded first."""
    names = source.names
    checked = check.compare(X_val, names, source.cell)
    result = eliminate.run(fit_step, X_fit, y_fit, X_val, names, n_target, on_step=on_step)
    top = eliminate.top_n(result["steps"][0]["importance"], n_target)
    top_names = [names[i] for i in top]
    top_fitted = fit_step(X_fit[:, top], y_fit, top_names)
    final = [names.index(name) for name in result["final_columns"]]

    return Reduction(
        steps=result["steps"],
        removed_at=[int(v) for v in result["removed_at"]],
        final_columns=result["final_columns"],
        final_importance=[float(v) for v in result["final_importance"]],
        top_n_columns=top_names,
        val_by_step=[np.asarray(scores, dtype=float) for scores in result["val_scores_by_step"]],
        val_scores={"full": np.asarray(result["val_scores_by_step"][0], dtype=float),
                    "reduced": np.asarray(result["val_scores"], dtype=float),
                    "top_n": top_fitted.predict(X_val[:, top])},
        featurisation_check=checked,
        n_fit_rows=len(y_fit),
        models={"full": (result["first_fitted"].model, names),
                "reduced": (result["fitted"].model, result["final_columns"]),
                "top_n": (top_fitted.model, top_names)},
        filled={"full": result["first_fitted"].filled(X_val),
                "reduced": result["fitted"].filled(X_val[:, final]),
                "top_n": top_fitted.filled(X_val[:, top])},
    )
