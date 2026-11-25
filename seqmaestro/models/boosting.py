"""Gradient-boosted trees, the black-box reference.

They are reported as a **strong reference among models of tabular features**, not as the ceiling on
accuracy. On short sequence windows a convolutional network would plausibly beat gradient boosting,
and the DEFT paper compares against both a convolutional network and a transformer, so a true
ceiling would have to come from models like those.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from seqmaestro.execution import jobs
from seqmaestro.harness import direction
from seqmaestro.models.base import fit_summary, fitted_values
from seqmaestro.models.weights import resolve_class_weight, resolve_fit_weights
from seqmaestro.target import BINARY, CONTINUOUS, target_for


class _Trees:
    """What boosted trees of either kind report: gain, TreeSHAP directions, and cover beside gain.
    The subclasses fit and predict."""

    name = "xgboost"
    importance_kind = "xgb_gain"
    direction_kind = "treeshap_dependence"
    #: An ensemble of many trees, which can be interpreted only through its predictions.
    interpretable = False
    reproducible = True
    #: What a run's config may change (`seqmaestro.models.settings`): how many trees, how deep each
    #: may grow, and how much each corrects the trees before it.
    settings: ClassVar[dict[str, type]] = {"n_estimators": int, "max_depth": int, "learning_rate": float}
    target_kind: str
    _model: Any
    #: The columns the model was fitted on, in order, which every prediction is checked against.
    _columns: list[str]

    def _values(self, X: pd.DataFrame) -> np.ndarray:
        """`X` as the array the booster predicts from (`fitted_values`)."""
        return fitted_values(X, self._columns)

    def importances(self) -> dict[str, float]:
        """Gain, over the features that appear in at least one split.

        `get_score` omits unused features, which is the behaviour wanted here: a feature the model
        never split on was not discovered by it.
        """
        return {
            str(name): float(gain)
            for name, gain in self._model.get_booster().get_score(importance_type="gain").items()
        }

    def directions(self, X: pd.DataFrame) -> dict[str, str]:
        """Which way each split feature moves the model, from its TreeSHAP contributions.

        Gain is unsigned, so unlike the linear models and the EBM this one has no direction stored
        in the fitted object. TreeSHAP (the Shapley-value attribution for tree ensembles)
        supplies it: `pred_contribs` gives every row's signed contribution from every feature, in
        log-odds for a binary target and in the target's units for a continuous one, computed
        exactly and not by sampling. It ships inside XGBoost, so this takes one prediction pass and
        no new dependency.

        Grouping those contributions into bins of the feature turns them into a curve, which is
        then treated the way an additive model's shape function is. That is what lets a boosted
        model report `non_monotone` at all. Averaging the contributions instead would return nearly
        zero for a feature that helps at one end and hurts at the other, which is also what a
        feature the model ignores returns.
        """
        import xgboost as xgb

        used = self.importances()
        words = target_for(self.target_kind).directions
        # Column order comes from `X`, and `DMatrix` keeps its names, so the booster refuses a
        # frame whose columns are not the ones it was fitted on instead of misattributing them.
        contributions = self._model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)

        found = {}
        for name in used:
            if name not in X.columns:
                continue
            column = X.columns.get_loc(name)
            curve = direction.dependence_curve(X[name].to_numpy(dtype=float),
                                               contributions[:, column])
            if (word := direction.from_curve(curve, words)) is not None:
                found[name] = word
        return found

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        """The shared summary, plus gain against cover.

        Gain says how much a split improved the objective, whereas cover says how many rows it
        applied to. A feature with high gain and low cover fired on a handful of rows, so the next
        proposals should build *around* it instead of restating it, and the ranking alone cannot
        show that.
        """
        summary = fit_summary(self, X, y)
        booster = self._model.get_booster()
        cover = booster.get_score(importance_type="cover")
        summary["cover"] = {
            name: round(float(cover.get(name, 0.0)), 3)
            for name in (item["feature"] for item in summary["most_used"])
        }
        return summary


class GradientBoosting(_Trees):
    """XGBoost for a binary target, reporting gain over the features it split on."""

    target_kind = BINARY

    def __init__(
        self,
        seed: int = 0,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        class_weight: str | None = "balanced",
        **kwargs: Any,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.class_weight = resolve_class_weight(class_weight)
        self.kwargs = kwargs
        self._model = None
        self._row_weights = False

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> GradientBoosting:
        from xgboost import XGBClassifier

        self._columns = list(X.columns)
        self._row_weights = sample_weight is not None

        # `sample_weight` instead of `scale_pos_weight`: the arithmetic is the same here, and
        # `sample_weight` is the mechanism the other models and DEFT's tree also use.
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbosity=0,
            **jobs(self.kwargs),
            **self.kwargs,
        ).fit(X, y, sample_weight=resolve_fit_weights(y, self.class_weight, sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(self._values(X), validate_features=False)[:, 1]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "class_weight": self.class_weight,
            "target_kind": self.target_kind,
            "per_row_weights": self._row_weights,
            **self.kwargs,
        }


class GradientBoostingRegression(_Trees):
    """XGBoost for a continuous target. Its trees are fitted by squared error, so its TreeSHAP
    contributions are in the target's units. It takes no class weighting, and per-row weights from
    the design are used as for every model.

    **It stops early.** A fixed number of trees fits the noise of a heavy-tailed target, and squared
    error punishes that, whereas a ranking metric does not, so held-out R-squared can fall below
    zero. Each fit therefore sets aside `validation_fraction` of its fitting rows, drawn with the
    seed and stratified on the target's quantile bins, and grows trees on the rest until the squared
    error on the part set aside has not fallen for `early_stopping_rounds` rounds. The model keeps
    the trees up to the round where that error was lowest, at most `n_estimators` of them.

    The part set aside comes from the fitting rows, and not from the dataset's validation split, so
    that the fit depends on its fitting rows alone, as every model's does: the reduction's steps and
    the panel's refits stop the same way on whatever rows they are given, and the validation scores
    that choose an arm's seed are not the ones its trees were stopped on.
    """

    target_kind = CONTINUOUS

    def __init__(self, seed: int = 0, n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.1, validation_fraction: float = 0.15,
                 early_stopping_rounds: int = 20, **kwargs: Any) -> None:
        if not 0 < validation_fraction < 1:
            raise ValueError(f"validation_fraction is the share of the fitting rows set aside to "
                             f"stop on, strictly between 0 and 1, not {validation_fraction!r}.")
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs
        self._model = None
        self._row_weights = False
        self._trees_kept: int | None = None

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> GradientBoostingRegression:
        from xgboost import XGBRegressor
        from xgboost.callback import EarlyStopping

        self._columns = list(X.columns)
        self._row_weights = sample_weight is not None
        y = np.asarray(y, dtype=float).ravel()
        weights = resolve_fit_weights(y, None, sample_weight)
        grow, stop = train_test_split(np.arange(len(y)), test_size=self.validation_fraction,
                                      stratify=target_for(CONTINUOUS).strata(y),
                                      random_state=self.seed)
        # `save_best` drops the rounds grown after the best one, so the gain, the TreeSHAP
        # contributions and the predictions all come from the trees that were kept.
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            verbosity=0,
            callbacks=[EarlyStopping(rounds=self.early_stopping_rounds, save_best=True)],
            **jobs(self.kwargs),
            **self.kwargs,
        ).fit(X.iloc[grow], y[grow], sample_weight=None if weights is None else weights[grow],
              eval_set=[(X.iloc[stop], y[stop])],
              sample_weight_eval_set=None if weights is None else [weights[stop]],
              verbose=False)
        self._trees_kept = self._model.get_booster().num_boosted_rounds()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._values(X), validate_features=False)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "validation_fraction": self.validation_fraction,
            "early_stopping_rounds": self.early_stopping_rounds,
            "trees_kept": self._trees_kept,
            "seed": self.seed,
            "target_kind": self.target_kind,
            "per_row_weights": self._row_weights,
            **self.kwargs,
        }
