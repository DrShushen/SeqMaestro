"""What kind of target a dataset has, and the arithmetic that differs between the kinds.

A dataset's target is **binary**, a label of 0 or 1 on every row with class 1 the positive class,
or **continuous**, one finite number on every row. Stage 1 declares the kind (`target.kind` in the
declaration, see `docs/dataset.md`) and `dataset.json` records it. A stage that treats the two
kinds differently asks the object `target_for` returns, so the arithmetic of each kind is written
once: how the target is read, summarised and stratified, what a split can detect, how a model's
predictions are scored and which metric ranks them, how one feature's association with the target
is measured, and the words a direction is given in.

Nothing guesses a kind from the values. A column of 0s and 1s can be a measured rate as well as a
label, and code written for labels runs on a continuous target without an error: a label cast to an
integer truncates it, a sample drawn class by class draws one row per distinct value, and an AUC of
a number against itself returns a number. So the kind is always the declared one, and a stage that
has not been told it has nothing to fall back on.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from seqmaestro.harness.direction import BINARY_WORDS, CONTINUOUS_WORDS
from seqmaestro.harness.effect import rank_biserial, rank_correlation
from seqmaestro.harness.metrics import (
    choose_threshold,
    classification_metrics,
    r_squared,
    regression_metrics,
)
from seqmaestro.harness.power import (
    auroc_half_width,
    contrast_effective_n,
    correlation_half_width,
    effective_sample_size,
    smallest_detectable_auroc,
    smallest_detectable_correlation,
)

BINARY, CONTINUOUS = "binary", "continuous"

#: Every kind of target, in the order they are listed.
TARGET_KINDS = (BINARY, CONTINUOUS)

#: How many quantile bins a continuous target is stratified on: enough that a stratified sample
#: keeps both tails, and few enough that every bin holds rows on a split of a few hundred.
STRATA = 10

#: AUROCs at which the half-width of the held-out interval is reported for a binary target.
AUROCS_AT = (0.52, 0.55, 0.60, 0.70)

#: Correlations at which the half-width of the held-out interval is reported for a continuous one.
CORRELATIONS_AT = (0.10, 0.20, 0.30, 0.50)


class Target(Protocol):
    """What every kind provides."""

    #: One of `TARGET_KINDS`.
    kind: str
    #: The values a label takes, in the order a breakdown by class reports them: `(1, 0)` for a
    #: binary target, and none for a continuous one, whose rows have no classes to break down by.
    labels: tuple[int, ...]
    #: The metric that ranks arms and seeds, and the one reported beside it wherever one is shown.
    headline: str
    companion: str
    #: What the headline scores for a model that has learned nothing, such as a permuted-label arm.
    chance: float
    #: The word for a feature whose rise raises the model's score, then the word for one whose rise
    #: lowers it (`seqmaestro.harness.direction`).
    directions: tuple[str, str]
    #: What `association` measures, as the records name it.
    association_kind: str
    #: Whether a model's scores are probabilities, which the metrics require for a binary target.
    probabilities: bool

    def values(self, y) -> np.ndarray:
        """Targets a stage has been handed, as this kind holds them, or `ValueError`."""

    def threshold(self, y, scores) -> float | None:
        """The cut a thresholded metric is scored at, chosen on these rows, or None for a kind
        without one."""

    def metrics(self, y, scores, *, threshold: float | None, n_bootstrap: int = 0,
                seed: int = 0) -> dict[str, Any]:
        """Every metric of this kind, for scores against targets."""

    def score(self, y, scores) -> float:
        """The headline alone, NaN where it is undefined."""

    def feedback_metrics(self, y, scores) -> dict[str, float]:
        """The headline and its companion, which need no threshold: what a model reports of its own
        fit to a proposer."""

    def association(self, values, y) -> np.ndarray:
        """How far each column of `values` goes with the target on its own, from -1 to 1."""

    def parse(self, column: pd.Series) -> tuple[np.ndarray, list[str]]:
        """A split file's target column as this kind holds it, and the raw values it cannot hold."""

    def summary(self, y) -> dict[str, Any]:
        """What `dataset.json` records about one split's targets."""

    def strata(self, y) -> np.ndarray:
        """What a stratified sample of these rows keeps in proportion, one value per row."""

    def held_out_power(self, y) -> dict[str, Any]:
        """What a held-out split with these targets can detect: the interval on its headline."""

    def training_information(self, y, weights=None) -> dict[str, Any]:
        """How much information a training split with these targets and fitting weights holds."""


class BinaryTarget:
    """A label of 0 or 1 on every row. Class 1 is the positive class.

    A model's score is the probability of class 1. AUROC ranks arms because it is free of any
    threshold and of where a model's scores happen to sit, and the thresholded metrics are scored
    at the cut chosen on validation.
    """

    kind = BINARY
    labels = (1, 0)
    headline, companion, chance = "auroc", "auprc", 0.5
    directions = BINARY_WORDS
    association_kind = "rank_biserial"
    probabilities = True

    def values(self, y) -> np.ndarray:
        y = np.asarray(y).ravel()
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"A binary target holds 0 and 1 only; found {sorted(set(np.unique(y)) - {0, 1})[:5]}.")
        return y.astype(int)

    def threshold(self, y, scores) -> float:
        return choose_threshold(y, scores)

    def metrics(self, y, scores, *, threshold: float | None, n_bootstrap: int = 0,
                seed: int = 0) -> dict[str, Any]:
        if threshold is None:
            raise ValueError("A binary target's metrics are scored at a threshold.")
        return classification_metrics(y, scores, threshold, n_bootstrap=n_bootstrap, seed=seed)

    def score(self, y, scores) -> float:
        y = np.asarray(y).ravel()
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, np.asarray(scores, dtype=float).ravel()))

    def feedback_metrics(self, y, scores) -> dict[str, float]:
        scored = classification_metrics(y, scores, threshold=0.5, n_bootstrap=0)
        return {self.headline: scored[self.headline], self.companion: scored[self.companion]}

    def association(self, values, y) -> np.ndarray:
        return rank_biserial(values, y)

    def parse(self, column: pd.Series) -> tuple[np.ndarray, list[str]]:
        """The labels as integers, and the raw values that are not 0 or 1, which the check reports.
        A value that is not a number comes back as -1, so it is counted in neither class."""
        values = pd.to_numeric(column, errors="coerce")
        bad = sorted({str(raw) for raw, value in zip(column, values, strict=True)
                      if pd.isna(value) or value not in (0, 1)})
        return values.fillna(-1).to_numpy().astype(int), bad

    def summary(self, y) -> dict[str, Any]:
        """The count of each class and the positive rate."""
        y = np.asarray(y)
        return {"positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
                "positive_rate": round(float((y == 1).mean()), 4) if len(y) else 0.0}

    def strata(self, y) -> np.ndarray:
        """The labels themselves."""
        return np.asarray(y)

    def held_out_power(self, y) -> dict[str, Any]:
        """The interval on a held-out AUROC, from the split's class counts. Metrics are computed on
        unweighted rows, so fitting weights do not enter it."""
        y = np.asarray(y)
        positives, negatives = float((y == 1).sum()), float((y == 0).sum())
        return {
            "n_positive": int(positives),
            "n_negative": int(negatives),
            "effective_n": (round(2 / (1 / positives + 1 / negatives), 1)
                            if positives and negatives else 0.0),
            "smallest_detectable_auroc": _finite(smallest_detectable_auroc(positives, negatives), 3),
            "half_width_at": {f"{auroc:.2f}": _finite(auroc_half_width(auroc, positives, negatives), 4)
                              for auroc in AUROCS_AT},
        }

    def training_information(self, y, weights=None) -> dict[str, Any]:
        """The effective size of the two-class contrast. For a weighted design it is Kish's
        effective size per class, combined harmonically, which can be far below the row count. Rows
        with zero weight are left out of it, as the split intends."""
        y = np.asarray(y)
        weights, kept = _kept(weights, len(y))
        kept_y = y[kept]
        n_positive, n_negative = int((kept_y == 1).sum()), int((kept_y == 0).sum())
        return {
            "rows": len(y),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "weighted": weights is not None,
            "effective_n": round(contrast_effective_n(None if weights is None else weights[kept],
                                                      kept_y), 1),
            "effective_n_counting_rows": round(2 / (1 / max(n_positive, 1) + 1 / max(n_negative, 1)), 1),
        }


class ContinuousTarget:
    """One finite number on every row.

    A model's score is its prediction of the number. R-squared ranks arms because it measures the
    squared error every model here minimises, and a model that predicts the target's mean, which is
    what a permuted-label arm learns, scores zero on it. There is no threshold.
    """

    kind = CONTINUOUS
    labels = ()
    headline, companion, chance = "r2", "spearman_rho", 0.0
    directions = CONTINUOUS_WORDS
    association_kind = "spearman"
    probabilities = False

    def values(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        if not np.isfinite(y).all():
            raise ValueError("A continuous target holds finite numbers only.")
        return y

    def threshold(self, y, scores) -> None:
        return None

    def metrics(self, y, scores, *, threshold: float | None, n_bootstrap: int = 0,
                seed: int = 0) -> dict[str, Any]:
        if threshold is not None:
            raise ValueError("A continuous target is scored without a threshold.")
        return regression_metrics(y, scores, n_bootstrap=n_bootstrap, seed=seed)

    def score(self, y, scores) -> float:
        return r_squared(y, scores)

    def feedback_metrics(self, y, scores) -> dict[str, float]:
        scored = regression_metrics(y, scores)
        return {self.headline: scored[self.headline], self.companion: scored[self.companion]}

    def association(self, values, y) -> np.ndarray:
        return rank_correlation(values, y)

    def parse(self, column: pd.Series) -> tuple[np.ndarray, list[str]]:
        """The values as floats, and the raw values that are not finite numbers, which come back as
        NaN. An empty field, a word, `NaN` and infinity are all refused, because every stage needs a
        number on every row."""
        values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float, copy=True)
        unusable = ~np.isfinite(values)
        bad = sorted({str(raw) for raw, refused in zip(column, unusable, strict=True) if refused})
        values[unusable] = np.nan
        return values, bad

    def summary(self, y) -> dict[str, Any]:
        """The target's centre, spread and quartiles, to six significant digits, since a target can
        be on any scale. A value `parse` could not read is NaN, and is left out."""
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if not len(y):
            return {"target": None}
        low, q25, median, q75, high = np.percentile(y, [0, 25, 50, 75, 100])
        return {"target": {"mean": _significant(y.mean()),
                           "sd": _significant(y.std(ddof=1)) if len(y) > 1 else None,
                           "min": _significant(low), "q25": _significant(q25),
                           "median": _significant(median), "q75": _significant(q75),
                           "max": _significant(high)}}

    def strata(self, y) -> np.ndarray:
        """`STRATA` quantile bins of the target, so a stratified sample keeps its whole range. Ties
        can make two bin edges coincide, and then the bins are fewer."""
        return pd.qcut(np.asarray(y, dtype=float), STRATA, labels=False, duplicates="drop")

    def held_out_power(self, y) -> dict[str, Any]:
        """The interval on a held-out correlation between predictions and targets, from the split's
        row count: the smallest correlation whose interval excludes zero, and its square, which is
        roughly the smallest R-squared the split can separate from zero for a model whose
        predictions are on the target's scale."""
        n = len(np.asarray(y))
        smallest = smallest_detectable_correlation(n)
        return {
            "rows": n,
            "smallest_detectable_r": _finite(smallest, 3),
            "smallest_detectable_r2": _finite(smallest ** 2, 4),
            "half_width_at": {f"{r:.2f}": _finite(correlation_half_width(r, n), 4)
                              for r in CORRELATIONS_AT},
        }

    def training_information(self, y, weights=None) -> dict[str, Any]:
        """Kish's effective size, which is the row count for an unweighted design. Rows with zero
        weight are left out of it, as the split intends."""
        y = np.asarray(y)
        weights, kept = _kept(weights, len(y))
        return {
            "rows": len(y),
            "weighted": weights is not None,
            "effective_n": round(effective_sample_size(None if weights is None else weights[kept],
                                                       n=int(kept.sum())), 1),
            "effective_n_counting_rows": int(kept.sum()),
        }


_TARGETS: dict[str, Target] = {BINARY: BinaryTarget(), CONTINUOUS: ContinuousTarget()}


def target_for(kind: str) -> Target:
    """The object for a declared kind, refusing any other."""
    if kind not in _TARGETS:
        raise ValueError(f"Unknown target kind {kind!r}; expected one of {', '.join(TARGET_KINDS)}.")
    return _TARGETS[kind]


def recorded_kind(record) -> str:
    """The kind a cell's `run.json` or a `features.json` header states under `target_kind`."""
    try:
        return record["target_kind"]
    except KeyError:
        raise ValueError("The record states no `target_kind`, so it cannot be said which kind of "
                         "target it was fitted against.") from None


def _kept(weights, n: int) -> tuple[np.ndarray | None, np.ndarray]:
    """The training weights as floats, and which rows carry any: every row when there are none."""
    if weights is None:
        return None, np.ones(n, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    return weights, weights > 0


def _finite(value: float, digits: int) -> float | None:
    """A rounded number, or None where the arithmetic has no answer, so the JSON holds null for it."""
    return round(float(value), digits) if np.isfinite(value) else None


def _significant(value: float, digits: int = 6) -> float:
    return float(f"{float(value):.{digits}g}")
