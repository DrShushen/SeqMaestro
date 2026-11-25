"""How strongly one feature is associated with the target on its own, measured directly on the data.

`rank_biserial` and `rank_correlation` compute the rank association that the harness records for
every feature as `marginal_effect`, the first for a binary target and the second for a continuous
one (`seqmaestro.target` assigns them). The other functions make up the direct effect test, which
contrasts two classes and so refuses a label with any other number of values:

- `match_within_strata` fixes the class ratio inside every stratum of a confounder (equal classes by
  default) by subsampling the more abundant class;
- `stratified_permutation_p` shuffles the labels within each stratum;
- `direct_effect` runs the whole set of features against one label vector and returns the table,
  with ratios and their intervals where one is defined (`proportion_ratio_interval`).

`docs/developer/analyses/direct_measurement.md` is the account of the test: why it exists beside
the models, why it needs both the matching and the permuting, and what its output can support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CONFIDENCE_Z = 1.96
DEFAULT_DRAWS = 10_000

#: How many matched samples to draw, to show how much the result depends on which sample was drawn.
#: See `matched_effect`.
DEFAULT_MATCH_REPEATS = 20

#: Permutation draws evaluated per matrix product (each batch is one `argsort` and one `matmul`).
#: It trades memory for speed and does not change the result.
DRAW_BATCH = 250


def _stratum_codes(strata, n: int) -> np.ndarray:
    """Strata as contiguous integer codes, or a single stratum when none is given."""
    if strata is None:
        return np.zeros(n, dtype=np.int64)
    codes = pd.factorize(pd.Series(np.asarray(strata)), sort=True)[0]
    if (codes < 0).any():
        raise ValueError("Missing values in the stratifying variable; a row belongs to a stratum "
                         "or it is not in the analysis.")
    return codes.astype(np.int64)


def _binary_label(y) -> np.ndarray:
    labels = np.asarray(y)
    present = np.unique(labels)
    if len(present) != 2:
        raise ValueError(f"A direct effect is a two-class contrast; found {len(present)} class(es).")
    return (labels == present.max()).astype(np.float64)


def match_within_strata(strata, y, *, seed: int = 0, ratio: int = 1) -> np.ndarray:
    """Row positions giving a **fixed class ratio** inside every stratum, by subsampling the
    abundant class.

    The fixed ratio makes stratum identity uninformative: every stratum then says the same thing
    about the label, so a model or a rate computed on these rows cannot recover the label from the
    stratum, whichever class is abundant overall.

    `ratio` is how many negative rows to keep per positive row. A ratio of 1 balances the classes
    and keeps the most of the scarce class, so only the abundant class loses rows. Where the
    negative class is the abundant one, a larger ratio keeps more rows in total, but the scarce class
    becomes a smaller share of them, and the scarce class is what limits a matched design. A stratum
    with fewer than `ratio` negative rows per positive row also loses positive rows, so that it keeps
    the ratio. A stratum that holds a single class contributes nothing and is dropped.

    One random generator serves all strata, instead of one seed reused inside each, because reusing
    a seed per stratum would make each stratum's draw a deterministic function of its row order.
    That would correlate the selections in a way that downstream code does not expect.
    """
    if ratio < 1:
        raise ValueError(f"ratio must be at least 1; got {ratio}.")
    label = _binary_label(y)
    codes = _stratum_codes(strata, len(label))
    rng = np.random.default_rng(seed)

    kept: list[np.ndarray] = []
    for code in np.unique(codes):
        inside = np.flatnonzero(codes == code)
        positive = inside[label[inside] == 1]
        negative = inside[label[inside] == 0]
        size = min(len(positive), len(negative) // ratio)
        if not size:
            continue
        kept.append(rng.choice(positive, size=size, replace=False))
        kept.append(rng.choice(negative, size=size * ratio, replace=False))
    return np.sort(np.concatenate(kept)) if kept else np.empty(0, dtype=np.int64)


def proportion_ratio_interval(k1: int, n1: int, k0: int, n0: int,
                              z: float = CONFIDENCE_Z) -> tuple[float, float, float]:
    """The ratio of two proportions with its 95% interval, by the delta method on the log scale.

    It is defined only for a *binary* feature, where each class mean is a proportion and its
    variance follows from the count. A continuous feature's ratio of means has no such variance, so
    `direct_effect` does not call this for one and leaves the interval empty.
    """
    if not (n1 and n0):
        return (float("nan"),) * 3
    p1, p0 = k1 / n1, k0 / n0
    if p1 <= 0 or p0 <= 0:
        return (float("nan"),) * 3
    error = np.sqrt((1 - p1) / (n1 * p1) + (1 - p0) / (n0 * p0))
    ratio = p1 / p0
    return float(ratio), float(np.exp(np.log(ratio) - z * error)), \
        float(np.exp(np.log(ratio) + z * error))


def _within_stratum_permutations(label: np.ndarray, codes: np.ndarray, batch: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """`batch` label vectors, each a permutation of `label` *inside* every stratum.

    Sorting on `stratum + U[0, 1)` orders the rows by stratum with a random order within each,
    because the codes are integers and the added noise never reaches 1. One `argsort` over the
    batch therefore gives every draw at once. Each stratum keeps its class counts, which lets
    `stratified_permutation_p` treat the class sizes as constants.
    """
    grouped = np.argsort(codes, kind="stable")
    keys = codes[None, :] + rng.random((batch, len(label)))
    order = np.argsort(keys, axis=1, kind="stable")
    drawn = np.empty((batch, len(label)), dtype=np.float64)
    drawn[:, grouped] = label[order]
    return drawn


def stratified_permutation_p(values: np.ndarray, y, strata=None, *,
                             draws: int = DEFAULT_DRAWS, seed: int = 0,
                             batch: int = DRAW_BATCH) -> np.ndarray:
    """Two-sided p-value per feature: how often shuffled labels move the class means at least this
    far apart.

    The labels are shuffled *within* each stratum, so every draw keeps the design's structure, and a
    difference the confounder produces appears in the null as readily as in the estimate. With no
    strata given, this is the ordinary label permutation.

    The p-value is `(hits + 1) / (draws + 1)`, counting the observed arrangement as one of the
    arrangements. This prevents a p-value of zero, which a permutation test cannot support.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if values.shape[0] != len(np.asarray(y)):
        values = values.T
    label = _binary_label(y)
    codes = _stratum_codes(strata, len(label))

    n_positive, n_negative = float(label.sum()), float((1 - label).sum())
    if draws <= 0 or not n_positive or not n_negative:
        # Zero draws means no p-value was asked for, which is NaN. Returning `1 / (0 + 1)` would give
        # a p-value of 1.0 backed by no evidence, which would look like a measured null result.
        return np.full(values.shape[1], float("nan"))
    totals = values.sum(axis=0)

    def contrast(indicator: np.ndarray) -> np.ndarray:
        carried = indicator @ values
        return carried / n_positive - (totals - carried) / n_negative

    observed = np.abs(contrast(label))
    rng = np.random.default_rng(seed)
    hits = np.zeros(values.shape[1], dtype=np.int64)
    done = 0
    while done < draws:
        size = min(batch, draws - done)
        drawn = _within_stratum_permutations(label, codes, size, rng)
        hits += (np.abs(contrast(drawn)) >= observed - 1e-12).sum(axis=0)
        done += size
    return (hits + 1) / (draws + 1)


def rank_biserial(values: np.ndarray, y) -> np.ndarray:
    """How far each feature separates the two classes, on one scale for every feature and arm.

    `2 * AUC - 1` per column, where AUC is that column used alone as a score for the positive
    class. It runs from -1 to 1 and has a direct meaning: 0.4 means that of one row drawn from each
    class, the positive one holds the higher value 70% of the time. A positive value means **higher
    in the positive class** (the larger of the two label values).

    **The statistic uses ranks.** A count and a fraction of the same thing over the same window are
    one feature under a monotone map, and only a rank statistic scores them alike
    (`seqmaestro.harness.stability` matches on Spearman correlation for the same reason). Ranks also
    make the number immune to a class imbalance, whereas a difference of means is not.

    **This is a marginal statement** about the feature and the label alone. What a model does with
    the feature, given the other features it holds, is a different quantity, recorded beside this
    one (see `seqmaestro.harness.direction`).

    A column with any non-finite value scores NaN instead of being quietly imputed, because the
    matrix should not contain one, and repairing it here would hide that it did.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    label = np.asarray(y).ravel()
    if len(label) != values.shape[0]:
        raise ValueError(f"{values.shape[0]} row(s) of values against {len(label)} label(s).")
    if len(np.unique(label)) != 2:
        # NaN, not zero: nothing was measured, which differs from measuring no separation.
        return np.full(values.shape[1], np.nan)

    positive = label == np.unique(label).max()
    n_positive, n_negative = float(positive.sum()), float((~positive).sum())

    scores = np.full(values.shape[1], np.nan)
    usable = np.isfinite(values).all(axis=0)
    if not usable.any():
        return scores

    ranked = rankdata(values[:, usable], axis=0)
    # Mann-Whitney U over the positive rows, normalised to an AUC. Ties take average ranks, which
    # counts a tie as half a win, as `roc_auc_score` does.
    auc = (ranked[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    scores[usable] = 2 * auc - 1
    return scores


def rank_correlation(values: np.ndarray, y) -> np.ndarray:
    """How strongly each feature is associated with a continuous target, on one scale for every
    feature and arm.

    Spearman's rho per column, with average ranks for ties: from -1 to 1, positive where higher
    values of the feature go with higher values of the target. It is the continuous counterpart of
    `rank_biserial` and uses ranks for the same reason, so that a count and a fraction of the same
    thing score alike. Ranks also keep a few rows of a heavy-tailed target from deciding the result.

    A constant column, or one with any non-finite value, scores NaN: nothing was measured there, and
    that differs from measuring no association.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    target = np.asarray(y, dtype=np.float64).ravel()
    if len(target) != values.shape[0]:
        raise ValueError(f"{values.shape[0]} row(s) of values against {len(target)} target(s).")

    scores = np.full(values.shape[1], np.nan)
    if len(target) < 2 or not np.isfinite(target).all() or np.ptp(target) == 0:
        return scores
    usable = np.isfinite(values).all(axis=0) & (np.ptp(values, axis=0) > 0)
    if not usable.any():
        return scores
    ranked = rankdata(values[:, usable], axis=0)
    ranked -= ranked.mean(axis=0)
    target_ranks = rankdata(target)
    target_ranks -= target_ranks.mean()
    scores[usable] = (target_ranks @ ranked) / np.sqrt(
        np.square(target_ranks).sum() * np.square(ranked).sum(axis=0))
    return scores


def direct_effect(values: np.ndarray, y, names: list[str], *, strata=None,
                  draws: int = DEFAULT_DRAWS, seed: int = 0) -> pd.DataFrame:
    """One row per feature: each class's mean (`mean_positive`, `mean_negative`), the difference
    between them (`difference`), the ratio of the two with its interval where one is defined
    (`ratio`, `ci_lo`, `ci_hi`), and a two-sided permutation p-value (`p_permutation`). `binary`
    says whether the feature takes only the values 0 and 1.

    `values` is rows by features, matching `names`. `strata` sets the structure of the permutation.
    To match the estimate as well, pass the rows through `match_within_strata` first.

    The ratio is reported because an enrichment is quoted that way, with its interval only where the
    delta method applies. A ratio with a small denominator has a sign but no reliable magnitude, so
    compare features on the difference column.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[1] != len(names):
        raise ValueError(f"{values.shape[1]} column(s) of values against {len(names)} name(s).")
    label = _binary_label(y)
    positive, negative = label == 1, label == 0

    p_values = stratified_permutation_p(values, label, strata, draws=draws, seed=seed)
    rows = []
    for position, name in enumerate(names):
        column = values[:, position]
        finite = np.unique(column[np.isfinite(column)])
        binary = bool(len(finite) <= 2 and np.isin(finite, (0.0, 1.0)).all())
        mean_positive = float(column[positive].mean())
        mean_negative = float(column[negative].mean())
        if binary:
            ratio, low, high = proportion_ratio_interval(
                int(column[positive].sum()), int(positive.sum()),
                int(column[negative].sum()), int(negative.sum()))
        else:
            # A ratio only where it means something. For a quantity that takes both signs, the ratio
            # of means swings with how close the denominator is to zero and flips sign as it crosses
            # zero, so it would look like a magnitude while having none.
            ratio = (mean_positive / mean_negative
                     if mean_negative > 0 and mean_positive >= 0 else float("nan"))
            low = high = float("nan")
        rows.append({
            "feature": name, "binary": binary,
            "mean_positive": mean_positive, "mean_negative": mean_negative,
            "difference": mean_positive - mean_negative,
            "ratio": ratio, "ci_lo": low, "ci_hi": high,
            "p_permutation": float(p_values[position]),
        })
    frame = pd.DataFrame(rows)
    frame.attrs["design"] = {
        "n": len(label), "n_positive": int(positive.sum()), "n_negative": int(negative.sum()),
        "strata": len(np.unique(_stratum_codes(strata, len(label)))),
        "draws": int(draws), "seed": int(seed),
    }
    return frame


def matched_effect(values: np.ndarray, y, strata, names: list[str], *,
                   repeats: int = DEFAULT_MATCH_REPEATS, draws: int = DEFAULT_DRAWS,
                   seed: int = 0) -> pd.DataFrame:
    """Match within strata, contrast the classes, and redraw the match to show how much the draw
    matters.

    **A matched estimate is a statistic of a random subsample.** Both the interval and the p-value
    condition on the subsample that was drawn, and neither says anything about the next draw. That
    hardly matters for a common feature, whereas for a rare one the abundant class is heavily
    subsampled and the estimate moves with the seed by an amount that can rival the stated interval.

    The estimate, its interval and its p-value all come from one matched sample, which keeps them a
    consistent analysis of one design. `ratio_draw_min`, `ratio_draw_max` and `difference_draw_sd`
    report how far the same quantities moved across `repeats` draws (the reported draw included).
    Treat them as an imprecision that the interval does not include. Where they are wide, matching
    loses more than it gains, and a weighted design is the better instrument.

    Averaging the draws would give a lower-variance estimator, but its interval and its p-value
    would then describe a quantity that nothing else in the table describes, so the draws are not
    averaged.
    """
    label = _binary_label(y)
    strata = np.asarray(strata)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]

    tables = []
    for offset in range(max(repeats, 1)):
        keep = match_within_strata(strata, label, seed=seed + offset)
        if not len(keep):
            return pd.DataFrame()
        # The null is computed on the reported draw only, because it is the slow half and the other
        # draws are needed only for their estimates.
        tables.append(direct_effect(values[keep], label[keep], names, strata=strata[keep],
                                    draws=draws if offset == 0 else 0, seed=seed))

    reported = tables[0].copy()
    spread = pd.concat(tables, ignore_index=True).groupby("feature", sort=False)
    reported["ratio_draw_min"] = reported.feature.map(spread.ratio.min())
    reported["ratio_draw_max"] = reported.feature.map(spread.ratio.max())
    reported["difference_draw_sd"] = reported.feature.map(spread.difference.std())
    reported.attrs["design"] = {**tables[0].attrs["design"], "match_repeats": max(repeats, 1)}
    return reported
