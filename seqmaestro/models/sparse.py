"""Linear models. The L1-regularised ones are the canonical sparse baseline: logistic for a binary
target, and least squares for a continuous one. `LeastSquares`, plain or with the ridge or lasso
penalty IGSR can fit, is the model IGSR fits over its terms, with which the later stages refit an
IGSR arm.

On the positional pool the sparse models' coefficients give one weight per base per position, in
the shape of a position-weight matrix over the window, which makes these arms directly comparable
to the motif literature. The selected features are the discovery output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from seqmaestro.harness import direction
from seqmaestro.models.base import fit_summary, fitted_values
from seqmaestro.models.weights import resolve_class_weight, resolve_fit_weights
from seqmaestro.target import BINARY, CONTINUOUS, target_for


class L1Logistic:
    """Sparse logistic regression. Smaller `C` selects fewer features.

    `liblinear` (coordinate descent) instead of `saga` (stochastic gradient descent), because
    `saga` is far slower on a wide k-mer matrix of unscaled integer counts, which coordinate
    descent copes with and stochastic gradient descent does not. This model standardises its
    columns (below), so that reason may not hold for it.

    `penalty="l1"` is deprecated in scikit-learn 1.8 and removed in 1.10, so this uses `l1_ratio`.

    **It fits on columns centred and scaled on the fitting rows**, so each coefficient is per
    standard deviation of its feature and the penalty treats every column alike. On raw columns the
    penalty depends on units: an indicator whose standard deviation is well below one needs a large
    coefficient to matter and is penalised for it, and a coefficient's size says as much about the
    column's scale as about the feature. The mean and standard deviation are unweighted, over the
    rows the model is fitted on. A column constant on those rows is left unscaled, and its
    coefficient is zero either way. At the same `C` this keeps more features than a fit on raw
    columns would on indicator and count pools, whose standard deviations are well below one.
    """

    name = "l1_logistic"
    target_kind = BINARY
    #: Signed, per standard deviation of the column on the fitting rows.
    importance_kind = "l1_coefficient_per_sd"
    direction_kind = "l1_coefficient_sign"
    interpretable = True
    reproducible = True

    def __init__(
        self, C: float = 0.1, seed: int = 0, max_iter: int = 2000,
        class_weight: str | None = "balanced",
    ) -> None:
        self.C = C
        self.seed = seed
        self.max_iter = max_iter
        self.class_weight = resolve_class_weight(class_weight)
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._columns: list[str] = []
        self._row_weights = False

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> L1Logistic:
        self._columns = list(X.columns)
        self._row_weights = sample_weight is not None
        values = X.to_numpy(dtype=float)
        self._scaler = StandardScaler().fit(values)
        inputs = self._scaler.transform(values)
        self._model = LogisticRegression(
            l1_ratio=1, solver="liblinear", C=self.C, max_iter=self.max_iter, random_state=self.seed
        ).fit(inputs, y, sample_weight=resolve_fit_weights(y, self.class_weight, sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(
            self._scaler.transform(fitted_values(X, self._columns)))[:, 1]

    def importances(self) -> dict[str, float]:
        """Signed coefficients of the selected features, per standard deviation of each column. The
        sign gives the direction, and the harness ranks by absolute value."""
        coefficients = self._model.coef_[0]
        return {
            name: float(coefficient)
            for name, coefficient in zip(self._columns, coefficients, strict=True)
            if coefficient != 0
        }

    def directions(self, X: pd.DataFrame | None = None) -> dict[str, str]:
        """The sign of each selected coefficient.

        Exact and complete for this model, and it needs no rows, because the fit is monotone in
        every feature. One sign describes everything the model does with a feature, since there is
        no range over which the relationship could reverse. `non_monotone` is unreachable here.
        """
        return _coefficient_directions(self.importances(), self.target_kind)

    def intercept(self) -> float:
        """The fitted intercept, in the model's scale."""
        return float(np.ravel(self._model.intercept_)[0])

    def column_scales(self) -> np.ndarray:
        """Each column's standard deviation on the fitting rows, in the order the columns were
        fitted. A weight per standard deviation divided by it is the weight per unit."""
        return np.asarray(self._scaler.scale_, dtype=float)

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        """The shared summary, plus what only a sparse linear fit knows: the sign of each weight
        (`_signed_feedback`)."""
        return _signed_feedback(self, X, y)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "C": self.C, "solver": "liblinear", "penalty": "l1",
                "seed": self.seed, "class_weight": self.class_weight,
                "target_kind": self.target_kind, "per_row_weights": self._row_weights}


class L1Linear:
    """Sparse linear regression, for a continuous target. Larger `strength` selects fewer features.

    The Lasso, fitted on columns centred and scaled on the fitting rows as `L1Logistic`'s are, so
    each coefficient is per standard deviation of its feature and the penalty treats every column
    alike.

    **The penalty is the universal one** (Donoho and Johnstone): sigma * sqrt(2 log p / n) on
    scikit-learn's scale, where sigma is the target's standard deviation on the fitting rows, p the
    number of columns and n the number of rows. It is the penalty at which a column of pure noise
    is kept with small probability, so one setting means the same on every dataset, whatever the
    target's units, the width of the pool or the row count. A fixed alpha would keep every feature
    of a target measured in large units and none of one measured in small units, and a fixed
    fraction of the penalty that zeroes every coefficient would keep hundreds of noise columns on a
    wide pool with few rows. Taking sigma as the target's whole spread, and not the residual's,
    errs towards keeping fewer features. `strength` scales it; the arms use 1.
    """

    name = "l1_linear"
    target_kind = CONTINUOUS
    #: Signed, per standard deviation of the column on the fitting rows.
    importance_kind = "l1_coefficient_per_sd"
    direction_kind = "l1_coefficient_sign"
    interpretable = True
    reproducible = True

    def __init__(self, strength: float = 1.0, seed: int = 0, max_iter: int = 10_000) -> None:
        if strength <= 0:
            raise ValueError(f"strength scales the universal penalty and must be positive, not "
                             f"{strength!r}.")
        self.strength = strength
        self.seed = seed
        self.max_iter = max_iter
        self._model: Lasso | None = None
        self._scaler: StandardScaler | None = None
        self._columns: list[str] = []
        self._alpha: float | None = None
        self._row_weights = False

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> L1Linear:
        self._columns = list(X.columns)
        self._row_weights = sample_weight is not None
        values = X.to_numpy(dtype=float)
        self._scaler = StandardScaler().fit(values)
        inputs = self._scaler.transform(values)
        y = np.asarray(y, dtype=float).ravel()
        weights = resolve_fit_weights(y, None, sample_weight)
        self._alpha = self.strength * _universal_penalty(y, weights, n_columns=inputs.shape[1])
        self._model = Lasso(alpha=self._alpha, max_iter=self.max_iter,
                            random_state=self.seed).fit(inputs, y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._scaler.transform(fitted_values(X, self._columns)))

    def importances(self) -> dict[str, float]:
        """Signed coefficients of the selected features, per standard deviation of each column, in
        the target's units. The sign gives the direction, and the harness ranks by absolute value."""
        return {name: float(coefficient)
                for name, coefficient in zip(self._columns, self._model.coef_, strict=True)
                if coefficient != 0}

    def directions(self, X: pd.DataFrame | None = None) -> dict[str, str]:
        """The sign of each selected coefficient, which says everything this model does with a
        feature, as for `L1Logistic`."""
        return _coefficient_directions(self.importances(), self.target_kind)

    def intercept(self) -> float:
        """The fitted intercept, in the model's scale."""
        return float(np.ravel(self._model.intercept_)[0])

    def column_scales(self) -> np.ndarray:
        """Each column's standard deviation on the fitting rows, in the order the columns were
        fitted. A weight per standard deviation divided by it is the weight per unit."""
        return np.asarray(self._scaler.scale_, dtype=float)

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        """The shared summary, plus the sign of each weight (`_signed_feedback`)."""
        return _signed_feedback(self, X, y)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "strength": self.strength, "alpha": self._alpha,
                "penalty": "universal", "solver": "coordinate_descent", "seed": self.seed,
                "target_kind": self.target_kind, "per_row_weights": self._row_weights}


#: The penalties `LeastSquares` fits with: none, the L1 penalty (lasso) or the L2 penalty (ridge).
LEAST_SQUARES_PENALTIES = (None, "l1", "l2")


class LeastSquares:
    """Least squares, for a continuous target: the fit IGSR makes over its terms, which the later
    stages refit an IGSR arm with.

    It is not among the Stage 2 models (`MODELS`), since an IGSR arm's Stage 2 model is IGSR's own,
    fitted in its environment. After Stage 2 the arm's model is this one over the arm's term values,
    which IGSR's copy computes, configured as the arm's search fitted (its bundle records how):
    ordinary least squares, or ridge (`penalty="l2"`) or lasso (`penalty="l1"`) at IGSR's `alpha`,
    with or without an intercept. The fit is made on the values as they are, as IGSR makes it,
    because a penalty depends on each column's scale; scikit-learn's estimators with their default
    settings are the ones IGSR calls, so the predictions are IGSR's. Each coefficient is then
    reported per standard deviation of its column on the fitting rows, as `L1Linear`'s are.
    """

    name = "least_squares"
    target_kind = CONTINUOUS
    #: Signed, per standard deviation of the column on the fitting rows.
    importance_kind = "least_squares_coefficient_per_sd"
    direction_kind = "least_squares_coefficient_sign"
    interpretable = True
    reproducible = True

    def __init__(self, penalty: str | None = None, alpha: float | None = None,
                 fit_intercept: bool = True) -> None:
        if penalty not in LEAST_SQUARES_PENALTIES:
            raise ValueError(f"penalty must be one of {LEAST_SQUARES_PENALTIES}, not {penalty!r}.")
        if (penalty is None) != (alpha is None) or (alpha is not None and alpha <= 0):
            raise ValueError("A penalised fit needs a positive alpha, and an unpenalised one takes "
                             f"none; given penalty {penalty!r} and alpha {alpha!r}.")
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._model: LinearRegression | Lasso | Ridge | None = None
        #: Fitted only to measure each column's standard deviation on the fitting rows.
        self._scaler: StandardScaler | None = None
        self._columns: list[str] = []
        self._row_weights = False

    def _estimator(self):
        if self.penalty is None:
            return LinearRegression(fit_intercept=self.fit_intercept)
        estimator = Lasso if self.penalty == "l1" else Ridge
        return estimator(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> LeastSquares:
        self._columns = list(X.columns)
        self._row_weights = sample_weight is not None
        values = X.to_numpy(dtype=float)
        self._scaler = StandardScaler().fit(values)
        y = np.asarray(y, dtype=float).ravel()
        self._model = self._estimator().fit(values, y, sample_weight=resolve_fit_weights(
            y, None, sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(fitted_values(X, self._columns))

    def importances(self) -> dict[str, float]:
        """Signed coefficients per standard deviation of each column, in the target's units, for
        every column with a coefficient other than zero."""
        per_sd = np.ravel(self._model.coef_) * self._scaler.scale_
        return {name: float(coefficient)
                for name, coefficient in zip(self._columns, per_sd, strict=True)
                if coefficient != 0}

    def directions(self, X: pd.DataFrame | None = None) -> dict[str, str]:
        """The sign of each coefficient, which says everything a linear model does with a column."""
        return _coefficient_directions(self.importances(), self.target_kind)

    def intercept(self) -> float:
        """The fitted intercept, in the model's scale."""
        return float(np.ravel(self._model.intercept_)[0])

    def column_scales(self) -> np.ndarray:
        """Each column's standard deviation on the fitting rows, in the order the columns were
        fitted. A weight per standard deviation divided by it is the weight per unit."""
        return np.asarray(self._scaler.scale_, dtype=float)

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        return _signed_feedback(self, X, y)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "penalty": self.penalty, "alpha": self.alpha,
                "fit_intercept": self.fit_intercept, "target_kind": self.target_kind,
                "per_row_weights": self._row_weights}


def _universal_penalty(y: np.ndarray, weights: np.ndarray | None, n_columns: int) -> float:
    """sigma * sqrt(2 log p / n), with sigma the target's standard deviation, weighted where the
    rows are. Weights are taken as `resolve_fit_weights` returns them, summing to the row count,
    which is how scikit-learn rescales them. A single column is treated as two, and a constant
    target given a penalty of one, so that the penalty is never zero, which would be no penalty at
    all."""
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    centre = np.average(y, weights=weights)
    sigma = float(np.sqrt(np.average(np.square(y - centre), weights=weights)))
    if sigma == 0:
        return 1.0
    return sigma * float(np.sqrt(2 * np.log(max(n_columns, 2)) / len(y)))


def _coefficient_directions(importances: dict[str, float], target_kind: str) -> dict[str, str]:
    """Each selected coefficient's sign, in the words of the target's kind. Exact and complete for a
    linear model, and it needs no rows: the fit is monotone in every feature, so no range exists
    over which the relationship could reverse, and `non_monotone` is unreachable."""
    words = target_for(target_kind).directions
    return {name: word for name, coefficient in importances.items()
            if (word := direction.from_sign(coefficient, words))}


def _signed_feedback(model, X: pd.DataFrame, y) -> dict[str, Any]:
    """`fit_summary`, plus the sign of each weight a sparse linear fit gives.

    Direction is the part a proposer can act on. "This feature counts against the outcome" is a
    different instruction from "this feature is unused", and an absolute importance conflates them.

    This says what `directions` says, in its own words, and does not call it. The feedback is
    digested into the proposal cache key, so rewording it would discard every cached proposal of
    the iterative source. Keep the two separate until there is a reason to pay that cost.
    """
    summary = fit_summary(model, X, y)
    weights = model.importances()
    summary["direction"] = {
        name: ("positive" if weights[name] > 0 else "negative")
        for name in (item["feature"] for item in summary["most_used"])
    }
    summary["n_features_zeroed"] = int(X.shape[1]) - len(weights)
    return summary
