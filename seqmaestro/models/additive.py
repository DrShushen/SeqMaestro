"""Explainable Boosting, the control the feature-source comparison depends on.

An EBM learns one shape function per input feature. **It does not create features**, so its
interpretability is inherited entirely from the pool it is fitted on. Positional indicators give
one trivial shape function per indicator, and *k*-mer counts give a shape function per *k*-mer,
which is a finding. That contrast is why the comparison fits it on more than one pool.

In the comparison it is the same downstream model on both sides of the feature source. Without an
EBM on a fixed vocabulary, a good result from an EBM on model-proposed features could not be
attributed to the proposal over the additive model or the featurisation.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from seqmaestro.execution import jobs
from seqmaestro.harness import direction
from seqmaestro.models.base import fit_summary, fitted_values
from seqmaestro.models.weights import resolve_class_weight, resolve_fit_weights
from seqmaestro.target import BINARY, CONTINUOUS, target_for


class _ShapeFunctions:
    """What an EBM of either kind reports: its terms' importances, the direction of each term's
    curve, and how far each curve swings. The subclasses fit and predict."""

    name = "ebm"
    importance_kind = "mean_abs_shape"
    direction_kind = "shape_function_trend"
    interpretable = True
    reproducible = True
    #: An EBM fits its bags on as many threads as it is given.
    one_core: ClassVar[dict[str, Any]] = {"n_jobs": 1}
    target_kind: str
    _model: Any

    @property
    def estimator(self) -> Any:
        """The fitted `interpret` model, for code that reads its terms and their curves."""
        return self._model

    def importances(self) -> dict[str, float]:
        """Mean absolute contribution of each term's shape function.

        A feature whose shape function is flat contributes nothing and is dropped, so the result is
        the set of features used, and not the whole input space with rankings attached.
        """
        return {
            str(name): float(importance)
            for name, importance in zip(
                self._model.term_names_, self._model.term_importances(), strict=True
            )
            if importance > 0
        }

    def directions(self, X: pd.DataFrame | None = None) -> dict[str, str]:
        """Which way each term's shape function runs, across the bins it was fitted with.

        This is the most complete direction any arm reports, because with `interactions=0` the
        model *is* the sum of these curves, in log-odds for a binary target and in the target's
        units for a continuous one. Each curve is everything the model does with its feature, and no
        rows are needed to read one back.

        A curve that turns is reported as `non_monotone` instead of by its net trend. A feature that
        raises the score at low values and lowers it at high ones makes a different claim from one
        that does neither, and only the curve can tell them apart.

        Terms over more than one feature are skipped. A pairwise term's shape belongs to the pair,
        and attributing it to either member would put a direction on a feature that does not have
        one on its own. `interactions=0` means there are none unless a caller asked for them.
        """
        used = self.importances()
        words = target_for(self.target_kind).directions
        found = {}
        for name, features, scores in zip(self._model.term_names_, self._model.term_features_,
                                          self._model.term_scores_, strict=True):
            name = str(name)
            if len(features) != 1 or name not in used:
                continue
            # `term_scores_` pads each univariate curve with a leading `missing` bin and a trailing
            # `unknown` bin. Both sit outside the observed range, so counting them as the ends of
            # the curve would add a step into or out of a bin no row is in.
            curve = np.asarray(scores, dtype=float).ravel()[1:-1]
            if (word := direction.from_curve(curve, words)) is not None:
                found[name] = word
        return found

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        """The shared summary, plus how far each shape function swings.

        An additive model's contribution per feature is a curve, and its range is the part a
        proposer can use: a feature whose shape is nearly flat was offered and effectively
        declined, whatever its mean absolute magnitude suggests.
        """
        summary = fit_summary(self, X, y)
        shapes = getattr(self._model, "term_scores_", None)
        if shapes is not None:
            names = list(self._model.term_names_)
            swing = {}
            for item in summary["most_used"]:
                if item["feature"] in names:
                    values = np.asarray(shapes[names.index(item["feature"])], dtype=float)
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        swing[item["feature"]] = round(float(finite.max() - finite.min()), 5)
            summary["shape_swing"] = swing
        return summary


class ExplainableBoosting(_ShapeFunctions):
    """EBM with main effects only by default, for a binary target.

    `interactions=0` keeps it a generalised additive model instead of a GA2M, which is the control
    the comparison needs: one shape function per feature, with nothing entangled between pairs. It
    is also the affordable setting. An EBM boosts round-robin over features, so its cost scales
    with the column count in a way the other models' does not, and pairwise interactions on
    thousands of columns are far more expensive.

    Uses `interpret-core` and not `interpret`, because the meta-package pulls in a dashboard stack
    that fitting a model and reading its shape functions do not need.
    """

    target_kind = BINARY

    #: Four outer bags instead of the library default of fourteen. Fitting time grows with the bag
    #: count, and the further bags smooth the shape functions and hardly change the validation
    #: score. Raise it if the shape functions look noisy.
    DEFAULT_OUTER_BAGS = 4

    def __init__(
        self, seed: int = 0, interactions: int = 0, outer_bags: int = DEFAULT_OUTER_BAGS,
        max_bins: int | None = None, class_weight: str | None = "balanced", **kwargs: Any
    ) -> None:
        self.seed = seed
        self.interactions = interactions
        self.outer_bags = outer_bags
        self.max_bins = max_bins
        self.class_weight = resolve_class_weight(class_weight)
        self.kwargs = kwargs
        self._model = None
        self._resolved_max_bins: int | None = None
        self._row_weights = False

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> ExplainableBoosting:
        from interpret.glassbox import ExplainableBoostingClassifier

        self._row_weights = sample_weight is not None
        self._resolved_max_bins = self.max_bins or _bins_for(X)
        self._model = ExplainableBoostingClassifier(
            interactions=self.interactions,
            outer_bags=self.outer_bags,
            max_bins=self._resolved_max_bins,
            random_state=self.seed,
            **jobs(self.kwargs),
            **self.kwargs,
        ).fit(X, y, sample_weight=resolve_fit_weights(y, self.class_weight, sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(fitted_values(X, self._model.feature_names_in_))[:, 1]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "interactions": self.interactions,
            "outer_bags": self.outer_bags,
            "max_bins": self._resolved_max_bins,
            "max_bins_source": "given" if self.max_bins else "derived_from_data",
            "seed": self.seed,
            "class_weight": self.class_weight,
            "target_kind": self.target_kind,
            "per_row_weights": self._row_weights,
            **self.kwargs,
        }


class ExplainableBoostingRegression(_ShapeFunctions):
    """EBM with main effects only by default, for a continuous target: the same additive model as
    `ExplainableBoosting`, fitted by least squares, with shape functions in the target's units.

    It takes no class weighting, since a continuous target has no classes. Per-row weights from the
    design are used as for every model.
    """

    target_kind = CONTINUOUS
    DEFAULT_OUTER_BAGS = ExplainableBoosting.DEFAULT_OUTER_BAGS

    def __init__(
        self, seed: int = 0, interactions: int = 0, outer_bags: int = DEFAULT_OUTER_BAGS,
        max_bins: int | None = None, **kwargs: Any
    ) -> None:
        self.seed = seed
        self.interactions = interactions
        self.outer_bags = outer_bags
        self.max_bins = max_bins
        self.kwargs = kwargs
        self._model = None
        self._resolved_max_bins: int | None = None
        self._row_weights = False

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> ExplainableBoostingRegression:
        from interpret.glassbox import ExplainableBoostingRegressor

        self._row_weights = sample_weight is not None
        self._resolved_max_bins = self.max_bins or _bins_for(X)
        y = np.asarray(y, dtype=float).ravel()
        self._model = ExplainableBoostingRegressor(
            interactions=self.interactions,
            outer_bags=self.outer_bags,
            max_bins=self._resolved_max_bins,
            random_state=self.seed,
            **jobs(self.kwargs),
            **self.kwargs,
        ).fit(X, y, sample_weight=resolve_fit_weights(y, None, sample_weight))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(fitted_values(X, self._model.feature_names_in_))

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "interactions": self.interactions,
            "outer_bags": self.outer_bags,
            "max_bins": self._resolved_max_bins,
            "max_bins_source": "given" if self.max_bins else "derived_from_data",
            "seed": self.seed,
            "target_kind": self.target_kind,
            "per_row_weights": self._row_weights,
            **self.kwargs,
        }


def _bins_for(X: pd.DataFrame, ceiling: int = 1024) -> int:
    """Bins for the range of values present, in place of the library's flat 1024.

    Positional indicators and k-mer counts are small non-negative integers (an indicator takes two
    values, and a count is bounded by the sequence length), so binning a 0/1 column into 1024
    buckets is pure cost. The count is taken from the data: the span of the whole matrix plus two,
    clipped to between 8 and `ceiling`. It follows the range of the values and not their
    resolution, so a matrix whose values all lie in [0, 1] gets the minimum of 8 bins.
    """
    values = X.to_numpy()
    span = int(np.nanmax(values) - np.nanmin(values)) + 2
    return int(np.clip(span, 8, ceiling))
