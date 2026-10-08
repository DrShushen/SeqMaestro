"""Binary classification and regression metrics, computed once and identically for every arm.

There are three reasons to compute these here instead of in each method:

**No method's metric set is complete.** DEFT reports accuracy and has no AUROC anywhere in it,
and the paper we compare against reports accuracy and AUPRC. Scoring on the SeqMaestro side
measures every arm the same way, including arms that compute no metric of their own.

**Accuracy alone does not separate these arms.** Two arms can tie on accuracy while differing in
calibration, ranking quality and the interpretability axes, so a single headline number would hide
the differences that matter. The set of metrics is therefore wide, and those in `BOOTSTRAP_METRICS`
carry a bootstrap interval on request, since two accuracies a point apart mean nothing without one.

**Anything not recorded is lost.** The raw confusion counts, the prevalence and the sample size are
returned alongside the derived metrics, so a metric derived from them can be computed later from a
stored `metrics.json` without re-running anything.

Left out because they add nothing to what is here: Youden's J, which is
`2 * balanced_accuracy - 1`, and Cohen's kappa, which on balanced binary data tracks MCC closely
enough not to need a column. Specificity is included, in the thresholded block.

**A continuous target has its own set** (`regression_metrics`): R-squared, the error in the
target's units (RMSE and MAE), and the two correlations of prediction with target. R-squared
measures the squared error a least-squares model minimises, and a permuted-label arm scores zero on
it. A heavy-tailed target makes it rest on its tails, which is why Spearman's rank correlation
always stands beside it. There is no threshold, so nothing is chosen on validation for this set.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn import metrics as skm

# The metrics reported with a bootstrap interval: the ones a comparison between arms turns on.
BOOTSTRAP_METRICS = ("auroc", "auprc", "accuracy", "balanced_accuracy", "f1", "mcc", "brier")

#: The same, for a continuous target.
REGRESSION_BOOTSTRAP_METRICS = ("r2", "rmse", "mae", "pearson_r", "spearman_rho")

#: Equal-width bins the expected calibration error is measured over.
CALIBRATION_BINS = 10

HIGHER_IS_BETTER = {
    "auroc": True, "auprc": True, "auprc_lift": True, "accuracy": True, "balanced_accuracy": True,
    "f1": True, "precision": True, "recall": True, "specificity": True, "mcc": True,
    "brier": False, "log_loss": False, "expected_calibration_error": False,
    "r2": True, "rmse": False, "mae": False, "pearson_r": True, "spearman_rho": True,
}

# The metrics that change when the threshold moves. The rest are ranking or probabilistic metrics,
# identical at every cut point, which is what makes `choose_threshold` cheap.
THRESHOLDED_METRICS = frozenset(
    {"accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity", "mcc"}
)


def classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict:
    """Every binary classification metric this project reports, from true labels and scores.

    `y_score` is the predicted probability of the positive class. Brier score, log loss and
    calibration are meaningless on a hard label or an unbounded decision function, so scores
    outside [0, 1] are refused: silently reporting a meaningless number is worse than failing.

    With `n_bootstrap > 0`, adds percentile confidence intervals under `"ci"`.
    """
    y_true, y_score = _check(y_true, y_score)

    result = {
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        **_metrics(y_true, y_score, threshold),
    }
    result.update(_counts(y_true, y_score >= threshold))

    if n_bootstrap:
        result["ci"] = _percentile_intervals(
            lambda take: _metrics(y_true[take], y_score[take], threshold),
            y_true.size, BOOTSTRAP_METRICS, n_bootstrap, seed)
        result["n_bootstrap"] = int(n_bootstrap)
    return result


def regression_metrics(y_true, y_pred, n_bootstrap: int = 0, seed: int = 0) -> dict:
    """Every regression metric this project reports, from true values and predictions.

    The centre and spread of both are recorded beside the metrics, so a stored `metrics.json` shows
    whether a low R-squared is a poor ranking or predictions on the wrong scale. A correlation is
    undefined when either side is constant, and is then NaN, as a ranking metric is on a resample
    holding one class.

    With `n_bootstrap > 0`, adds percentile confidence intervals under `"ci"`.
    """
    y_true, y_pred = _check_regression(y_true, y_pred)
    result = {
        "n": int(y_true.size),
        "mean_true": float(y_true.mean()),
        "sd_true": float(y_true.std(ddof=1)) if y_true.size > 1 else float("nan"),
        "mean_pred": float(y_pred.mean()),
        "sd_pred": float(y_pred.std(ddof=1)) if y_pred.size > 1 else float("nan"),
        **_regression(y_true, y_pred),
    }
    if n_bootstrap:
        result["ci"] = _percentile_intervals(
            lambda take: _regression(y_true[take], y_pred[take]),
            y_true.size, REGRESSION_BOOTSTRAP_METRICS, n_bootstrap, seed)
        result["n_bootstrap"] = int(n_bootstrap)
    return result


def r_squared(y_true, y_pred) -> float:
    """1 - SSE / SST, NaN where the target does not vary. Below zero for predictions worse than
    the target's mean."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total = float(np.square(y_true - y_true.mean()).sum())
    if total == 0:
        return float("nan")
    return 1.0 - float(np.square(y_true - y_pred).sum()) / total


def choose_threshold(y_true, y_score, metric: str = "balanced_accuracy") -> float:
    """The threshold that optimises `metric`, to be chosen on validation and then applied to test.

    A fixed 0.5 suits a model whose scores are calibrated probabilities. A shallow tree's scores are
    leaf proportions, which need not be calibrated, so an arm can get an unfair accuracy purely
    because of where its scores sit. Choosing on validation removes that, provided the chosen
    threshold is then *fixed* for test.

    All candidate cut points are evaluated at once. Calling the full set of metrics once per
    candidate would cost O(n^2): there are up to `n` distinct scores, and each call recomputes every
    metric over all `n` rows, including AUROC and AUPRC, which sort and do not depend on the
    threshold at all. Here the four confusion counts come from one sort and a `searchsorted`, and
    only the requested metric is formed, vectorised across candidates.
    """
    y_true, y_score = _check(y_true, y_score)
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(HIGHER_IS_BETTER)}.")
    if y_true.size == 0:
        raise ValueError("No samples to choose a threshold from.")

    # Cut points at the midpoints between distinct scores, where no score sits, plus 0 and 1.
    candidates = np.unique(y_score)
    candidates = np.concatenate([[0.0], (candidates[:-1] + candidates[1:]) / 2, [1.0]])

    if metric in THRESHOLDED_METRICS:
        values = _threshold_curve(y_true, y_score, candidates, metric)
    else:
        # The metric is identical at every cut point, so the choice is vacuous. It is computed once
        # and broadcast, so the tie-break below still picks the threshold, and an undefined metric
        # still falls back to 0.5.
        single = _metrics(y_true, y_score, 0.5).get(metric, np.nan)
        values = np.full(candidates.shape, single, dtype=float)

    finite = np.isfinite(values)
    if not finite.any():
        return 0.5

    # The tie-break decides which of a plateau of equally good cut points an arm is scored at, so
    # changing it changes scores. It follows the metric's direction: where higher is better the
    # largest tied threshold wins, and where lower is better the smallest.
    values, candidates = values[finite], candidates[finite]
    higher_is_better = HIGHER_IS_BETTER[metric]
    best = values.max() if higher_is_better else values.min()
    tied = candidates[values == best]
    return float(tied.max() if higher_is_better else tied.min())


# ------------------------------------------------------------------------------------ internals


def _threshold_curve(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray, metric: str
) -> np.ndarray:
    """One thresholded metric evaluated at every threshold, from a single sort.

    `y_score >= t` partitions the sorted scores, so the four confusion counts at every threshold
    follow from one `searchsorted` and one cumulative sum. Zero-denominator cases return 0.0, which
    matches the scikit-learn calls in `_metrics` under `zero_division=0`.
    """
    order = np.argsort(y_score, kind="mergesort")
    sorted_score, sorted_true = y_score[order], y_true[order]
    n = sorted_true.size
    positives = int(sorted_true.sum())
    negatives = n - positives

    # First index whose score is >= t; everything from there on is predicted positive.
    first = np.searchsorted(sorted_score, thresholds, side="left")
    positives_below = np.concatenate([[0], np.cumsum(sorted_true)])[first]

    predicted_positive = (n - first).astype(float)
    tp = float(positives) - positives_below
    fp = predicted_positive - tp
    fn = float(positives) - tp
    tn = float(negatives) - fp

    def ratio(numerator, denominator):
        """Zero where the denominator vanishes, matching `zero_division=0`."""
        denominator = np.asarray(denominator, dtype=float)
        return np.divide(
            numerator, denominator, out=np.zeros(np.shape(thresholds)), where=denominator > 0
        )

    if metric == "accuracy":
        return (tp + tn) / n
    if metric == "precision":
        return ratio(tp, predicted_positive)
    if metric == "recall":
        return ratio(tp, np.full_like(tp, positives))
    if metric == "specificity":
        return ratio(tn, np.full_like(tn, negatives))
    if metric == "f1":
        return ratio(2 * tp, 2 * tp + fp + fn)
    if metric == "balanced_accuracy":
        # scikit-learn averages recall over the classes *present in y_true*, so a single-class
        # y_true gives that class's recall alone rather than half of it.
        recall = ratio(tp, np.full_like(tp, positives))
        specificity = ratio(tn, np.full_like(tn, negatives))
        if positives and negatives:
            return (recall + specificity) / 2
        return recall if positives else specificity
    if metric == "mcc":
        # Zero when any margin is empty, since the correlation is undefined once a predicted or
        # true class is empty. `_metrics` gives the same result.
        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return np.divide(
            tp * tn - fp * fn,
            np.sqrt(denominator),
            out=np.zeros(np.shape(thresholds)),
            where=denominator > 0,
        )
    raise ValueError(f"{metric!r} is not a thresholded metric.")


def _check_regression(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError(f"{y_true.size} targets but {y_pred.size} predictions.")
    if y_true.size == 0:
        raise ValueError("No samples.")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Targets or predictions contain NaN or infinity.")
    return y_true, y_pred


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson's correlation, NaN where either side is constant."""
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt(np.square(a).sum() * np.square(b).sum())
    return float((a * b).sum() / scale) if scale > 0 else float("nan")


def _regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    return {
        "r2": r_squared(y_true, y_pred),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
        "pearson_r": _correlation(y_true, y_pred),
        # Average ranks, so tied targets (a value delivered to a few decimals) share a rank.
        "spearman_rho": _correlation(rankdata(y_true), rankdata(y_pred)),
    }


def _check(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    if y_true.size != y_score.size:
        raise ValueError(f"{y_true.size} labels but {y_score.size} scores.")
    if y_true.size == 0:
        raise ValueError("No samples.")

    classes = set(np.unique(y_true).tolist())
    if not classes <= {0, 1}:
        raise ValueError(
            f"Binary labels expected, encoded 0/1; found {sorted(classes)}. Multi-class is out of "
            f"scope. Map the labels before calling, so the mapping is explicit and recorded."
        )
    if not np.isfinite(y_score).all():
        raise ValueError("Scores contain NaN or infinity.")
    if y_score.min() < 0 or y_score.max() > 1:
        raise ValueError(
            f"Scores must be probabilities in [0, 1]; got [{y_score.min():.4g}, {y_score.max():.4g}]. "
            f"Brier score, log loss and calibration are meaningless on an unbounded decision "
            f"function. Apply a link function first, instead of reporting a number that looks "
            f"valid and is not."
        )
    return y_true.astype(int), y_score


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    prevalence = float(y_true.mean())
    # AUROC and AUPRC need both classes present, and a bootstrap resample can lose one.
    both_classes = 0 < y_true.sum() < y_true.size

    auprc = float(skm.average_precision_score(y_true, y_score)) if both_classes else np.nan

    return {
        # Ranking: threshold-free, so unaffected by where a model's scores happen to sit.
        "auroc": float(skm.roc_auc_score(y_true, y_score)) if both_classes else np.nan,
        "auprc": auprc,
        # A random ranker scores AUPRC == prevalence, so the raw value is not comparable across
        # datasets with different balance. The lift is.
        "auprc_baseline": prevalence,
        "auprc_lift": auprc / prevalence if both_classes and prevalence > 0 else np.nan,
        # Probabilistic: whether the score is a usable probability as well as a usable ranking.
        # This matters here because a tree's leaf proportions are offered as probabilities.
        "brier": float(np.mean((y_score - y_true) ** 2)),
        "log_loss": float(skm.log_loss(y_true, y_score, labels=[0, 1])),
        "expected_calibration_error": _expected_calibration_error(y_true, y_score, CALIBRATION_BINS),
        # Thresholded.
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(skm.recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        # The most informative single summary on binary data: unlike F1 it uses all four cells of
        # the confusion matrix, so it cannot be inflated by ignoring the negative class.
        "mcc": float(skm.matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """The four cells, so any thresholded metric can be recomputed from a stored result."""
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred.astype(int), labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int) -> float:
    """Mean gap between confidence and accuracy, weighted by bin occupancy.

    Uses equal-width bins, the common convention. The value depends on `bins`, so it is comparable
    across arms scored here but not with published numbers.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Digitised on the inner edges only: with `right=True` on the full edges, a score of exactly 0.0
    # would fall below the first bin.
    index = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, bins - 1)

    error = 0.0
    for b in range(bins):
        in_bin = index == b
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(y_true[in_bin].mean() - y_score[in_bin].mean())
    return float(error)


def _percentile_intervals(compute, size: int, names, n: int, seed: int) -> dict:
    """Percentile intervals by resampling rows with replacement.

    `compute(rows)` returns the metrics of a resample given as row indices. Seeded, so a rerun
    reproduces the interval exactly. A resample on which a metric is undefined (one that loses a
    class, or holds one target value) gives NaN, and those NaNs are left out of the percentiles.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in names}

    for _ in range(n):
        computed = compute(rng.integers(0, size, size))
        for name in names:
            draws[name].append(computed[name])

    intervals = {}
    for name, values in draws.items():
        finite = np.asarray([v for v in values if np.isfinite(v)])
        intervals[name] = (
            [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
            if finite.size
            else [np.nan, np.nan]
        )
    return intervals
