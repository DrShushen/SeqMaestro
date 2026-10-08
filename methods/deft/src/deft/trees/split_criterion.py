from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from deft.targets import BINARY, CONTINUOUS

"""
Note that we want to maximize the scores (hence the negative sign)
"""


def normalize_class_label(class_label: Any) -> Any:
    """Convert a NumPy scalar label to its Python equivalent.

    A `class_weight` mapping is written by a human -- `{0: 1.0, 1: 3.0}` -- while labels come
    out of a NumPy array as `np.int64`, and those do not hash to the same key. Shared by the
    splitting criteria and by the tree, which both look weights up by label.
    """
    if isinstance(class_label, np.generic):
        return class_label.item()
    return class_label


class SplitCriterion(ABC):
    #: The kinds of target (`deft.targets`) the criterion is written for. The tree refuses to grow
    #: with a criterion for another kind.
    target_kinds: tuple[str, ...] = ()

    @abstractmethod
    def __call__(
        self,
        y_left,
        y_right,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Run the instance as a callable."""

    def prepare(self, y, class_weight=None) -> None:
        """Called once by the tree before it grows, with every fitting row and their weighting. A
        criterion whose units depend on the data fixes them here, and the others need nothing."""
        return None

    @staticmethod
    def _row_weights(y, row_weights: pd.Series) -> np.ndarray:
        """The per-row weights of `y`'s rows, aligned by `y`'s index."""
        index = getattr(y, "index", None)
        if index is None:
            raise TypeError(
                "Per-row `sample_weight` needs labels that carry an index to align against; "
                "got a plain array. Pass the labels as a pandas Series or DataFrame."
            )
        weights = row_weights.reindex(index).to_numpy(dtype=float)
        if np.isnan(weights).any():
            raise ValueError("Per-row `sample_weight` does not cover every row of this node.")
        return weights

    def _weighted_class_counts(
        self,
        y,
        class_weight: "Mapping[Any, float] | pd.Series | None" = None,
    ) -> np.ndarray:
        """Return class counts after applying optional weights.

        Two kinds of weighting reduce to the same thing here -- a weight per class for this node:

        - a **mapping** from class label to weight, the usual case, including what
          ``class_weight="balanced"`` resolves to;
        - a **pandas Series of per-row weights**, aligned to ``y``'s index. This is how a matched
          design expressed as weights rather than as subsampling reaches the criterion: the weight
          differs row by row within a class, so no mapping can express it.

        A Series is not a ``Mapping``, so the two are distinguishable without a flag. Per-row
        weights require labels that carry an index, which every path through ``fit`` provides -- the
        dataset plugin hands over a DataFrame -- and are refused rather than silently mis-aligned if
        they do not.
        """
        labels = np.unique(y)
        if class_weight is None:
            _, counts = np.unique(y, return_counts=True)
            return counts.astype(float)

        if isinstance(class_weight, pd.Series):
            row_weights = self._row_weights(y, class_weight)
            values = np.asarray(y).ravel()
            return np.array([row_weights[values == label].sum() for label in labels], dtype=float)

        _, counts = np.unique(y, return_counts=True)
        weights = np.array(
            [
                float(class_weight.get(normalize_class_label(label), 1.0))
                for label in labels
            ],
            dtype=float,
        )
        return counts.astype(float) * weights

    def _weighted_total(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Return the class-weighted size of a node."""
        return float(np.sum(self._weighted_class_counts(y, class_weight)))


class InformationGainCriterion(SplitCriterion):
    target_kinds = (BINARY,)

    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)

    def _entropy(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Handle entropy."""
        counts = self._weighted_class_counts(y, class_weight)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        probabilities = counts / total
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def __call__(
        self,
        y_left,
        y_right,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Run the instance as a callable."""
        n_left = self._weighted_total(y_left, class_weight)
        n_right = self._weighted_total(y_right, class_weight)
        n_total = n_left + n_right
        if n_total <= 0:
            return -np.inf

        p_left = n_left / n_total
        p_right = n_right / n_total

        entropy_left = self._entropy(y_left, class_weight)
        entropy_right = self._entropy(y_right, class_weight)

        return -p_left * entropy_left - p_right * entropy_right + self.offset


class GiniCriterion(SplitCriterion):
    target_kinds = (BINARY,)

    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)

    def _gini(
        self,
        y,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Return the class-weighted Gini impurity of a node."""
        counts = self._weighted_class_counts(y, class_weight)
        total = np.sum(counts)
        if total <= 0:
            return 0.0
        probabilities = counts / total
        return float(1 - np.sum(probabilities**2))

    def __call__(
        self,
        y_left,
        y_right,
        class_weight: Mapping[Any, float] | None = None,
    ) -> float:
        """Run the instance as a callable."""
        n_left = self._weighted_total(y_left, class_weight)
        n_right = self._weighted_total(y_right, class_weight)
        n_total = n_left + n_right
        if n_total <= 0:
            return -np.inf

        p_left = n_left / n_total
        p_right = n_right / n_total

        gini_left = self._gini(y_left, class_weight)
        gini_right = self._gini(y_right, class_weight)

        return -p_left * gini_left - p_right * gini_right + self.offset


class SquaredErrorCriterion(SplitCriterion):
    """Variance reduction, for a continuous target, in units of the variance of the fitting rows.

    A split scores `offset - p_left * I(left) - p_right * I(right)`, the form of the classification
    criteria, so an empty side contributes nothing and `criterion(y, empty) == offset - I(y)`. Here
    I is a node's weighted variance divided by the variance of every row the tree is fitted on,
    which `prepare` fixes once, before the tree grows. Every candidate at every node is then scored
    in the same units whatever the target's scale, so a score compares at four decimals, and the
    impurity a split removes is a share of the variance the tree began with.

    Rows are weighted one by one (`sample_weight`) or not at all. A class weighting weights the
    classes of a binary target, and a continuous target has none.
    """

    target_kinds = (CONTINUOUS,)

    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)
        self.root_variance: float | None = None

    def prepare(self, y, class_weight=None) -> None:
        """Fix the unit of impurity: the weighted variance of every fitting row."""
        _, variance = self._weighted_variance(y, class_weight)
        # A constant target has nothing to explain and its root becomes a leaf; any positive unit
        # keeps the scores defined.
        self.root_variance = variance if variance > 0 else 1.0

    def _weighted_variance(self, y, class_weight=None) -> tuple[float, float]:
        """A node's total weight and its weighted variance, both 0 for a node with no rows."""
        values = np.asarray(y, dtype=float).ravel()
        if class_weight is None:
            weights = np.ones(values.size)
        elif isinstance(class_weight, pd.Series):
            weights = self._row_weights(y, class_weight)
        else:
            raise TypeError(
                "A continuous target is weighted row by row (`sample_weight`) or not at all. A "
                "class weighting has no classes to weight."
            )
        total = float(np.sum(weights))
        if total <= 0:
            return 0.0, 0.0
        mean = np.sum(weights * values) / total
        return total, float(np.sum(weights * (values - mean) ** 2) / total)

    def __call__(
        self,
        y_left,
        y_right,
        class_weight: pd.Series | None = None,
    ) -> float:
        """Run the instance as a callable."""
        if self.root_variance is None:
            raise RuntimeError(
                "The squared-error criterion measures impurity in units of the fitting rows' "
                "variance, which the tree fixes when it is fitted (`prepare`)."
            )
        n_left, variance_left = self._weighted_variance(y_left, class_weight)
        n_right, variance_right = self._weighted_variance(y_right, class_weight)
        n_total = n_left + n_right
        if n_total <= 0:
            return -np.inf
        return (-(n_left / n_total) * variance_left / self.root_variance
                - (n_right / n_total) * variance_right / self.root_variance + self.offset)
