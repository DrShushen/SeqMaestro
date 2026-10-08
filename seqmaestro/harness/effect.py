"""How far one feature goes with the target on its own, measured directly on the data.

`docs/analyses/direct_measurement.md` is the maintained account of this. Keep the two in step.

The module holds two things. `rank_biserial` and `rank_correlation` are the rank association the
harness records for every feature as `marginal_effect`: the first for a binary target and the
second for a continuous one, as `seqmaestro.target` assigns them. The other functions make up the
direct effect test, which an analysis computes on demand. It is a contrast between two classes, so
it refuses a label holding any other number of values.

A fitted model answers "can these features predict the label together?". The direct effect test
answers the narrower question that is usually the one a collaborator asked: does *this* feature
differ between the classes, by how much, and could a difference that size have arisen by chance?

This is needed beside the models for two reasons. A model's score is a joint statement and cannot
be attributed to any one of its inputs. And an arm can report a feature because it helps
prediction while that feature's class difference, taken alone, is indistinguishable from noise.

**The test is stratified**, because the case that needs measuring is a dataset with a confounder
in it: a batch, a donor, a gene, whatever the split already groups on. There are three pieces, kept
separate because they are separate decisions:

- `match_within_strata` equalises the classes inside every stratum by subsampling the abundant one,
  so stratum identity carries no information about the label.
- `stratified_permutation_p` shuffles the labels *within* each stratum, so the null has the same
  structure as the design and the confounder cannot contribute to it either.
- `direct_effect` runs the whole set of features against one label vector and returns the table.

Matching and permuting are both needed. Matching removes the association from the *estimate*, and
permuting within strata removes it from the *null*. Matching alone leaves a p-value that a stratum
imbalance can inflate, and permuting alone leaves an estimate that stratum composition still moves.

**A stratum holding only one class adds nothing.** Matching drops it, and permuting inside it
changes nothing. It offers no contrast, so it says nothing about the label, and keeping it in the
estimate would only put the association back.

The statistic is the difference of class means, which is a difference of rates when the feature is
binary. Ratios are reported beside it because that is how an enrichment is usually quoted, with an
interval only where one is defined (see `proportion_ratio_interval`).

`rank_biserial` is the same contrast reduced to one scale-free number, and `rank_correlation` is
its counterpart for a continuous target. Both are unstratified and carry no null of their own,
because a per-feature p over a whole grid is a claim about the analysis rather than about the
feature. For a binary label that question belongs to the functions above.
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

#: Permutation draws evaluated per matrix product. It trades memory for speed and does not change
#: the result. Each batch is one `argsort` and one `matmul`.
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

    The fixed ratio makes stratum identity uninformative. Every stratum then says the same thing
    about the label, so a model or a rate computed on these rows cannot recover the label from the
    stratum, whichever class is abundant overall.

    `ratio` is how many negative rows to keep per positive row. A ratio of 1 balances the classes
    and keeps the most of the scarce class, so only the abundant class loses rows. Where the
    negative class is the abundant one, larger values keep more rows in total, of which the scarce
    class is a smaller share, and the scarce class is what limits a matched design. A stratum
    holding fewer than `ratio` negative rows per positive row loses positive rows too, so that it
    keeps the ratio. Strata holding a single class contribute nothing and are dropped.

    One generator is used across all strata, instead of one seed reused inside each. Reusing a seed
    per stratum would make each stratum's draw a deterministic function of its own row order, which
    correlates the selections in a way nothing downstream expects.
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
    """Ratio of two proportions with its 95% interval, by the delta method on the log.

    Defined only for a *binary* feature, where each class mean is a proportion and its variance
    follows from the count. A continuous feature's ratio of means has no such variance, so
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
    batch therefore gives every draw at once. Each stratum keeps its own class counts, which is
    what lets `stratified_permutation_p` treat the class sizes as constants.
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
    """Two-sided p per feature: how often shuffled labels move the class means this far apart.

    The labels are shuffled *within* each stratum, so every draw keeps the design's structure, and a
    difference the confounder produces appears in the null as readily as in the estimate. With no
    strata given, this is the ordinary label permutation.

    The p-value is `(hits + 1) / (draws + 1)`, counting the observed arrangement as one of the
    arrangements. This prevents a p of exactly zero, which no permutation test can support.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if values.shape[0] != len(np.asarray(y)):
        values = values.T
    label = _binary_label(y)
    codes = _stratum_codes(strata, len(label))

    n_positive, n_negative = float(label.sum()), float((1 - label).sum())
    if draws <= 0 or not n_positive or not n_negative:
        # Zero draws is "not asked for", which is NaN. Returning `1 / (0 + 1)` would be a p of 1.0
        # backed by no evidence at all, and would look like a measured null result.
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
    class. It runs from -1 to 1, and has a direct meaning: 0.4 means that of one row drawn from
    each class, the positive one holds the higher value 70% of the time. Positive is **higher in
    the positive class**, whichever way round the labels happen to be encoded.

    **Ranks, not means.** A count and a fraction of the same thing over the same window are the
    same feature under a monotone map, and only a rank statistic scores them alike.
    `seqmaestro.harness.stability` matches on Spearman correlation for the same reason. It also
    makes the number immune to a class imbalance that a difference of means is not.

    **This is a marginal statement**, about the feature and the label alone. What a model does with
    the feature once it holds the others is a different quantity, recorded beside this one (see
    `seqmaestro.harness.direction`).

    A column carrying any non-finite value scores NaN instead of being quietly imputed: the matrix
    should not contain one, and repairing it here would hide that it did.
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
    # counts a tie as half a win, the same convention `roc_auc_score` uses.
    auc = (ranked[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    scores[usable] = 2 * auc - 1
    return scores


def rank_correlation(values: np.ndarray, y) -> np.ndarray:
    """How far each feature goes with a continuous target, on one scale for every feature and arm.

    Spearman's rho per column, with average ranks for ties: from -1 to 1, positive where higher
    values of the feature go with higher values of the target. It is the continuous counterpart of
    `rank_biserial`, and a rank statistic for the same reason: a count and a fraction of the same
    thing score alike, and a heavy-tailed target cannot let a few rows decide it.

    A constant column, or a column carrying any non-finite value, scores NaN: nothing was measured
    there, which differs from measuring no association.
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
    """One row per feature: each class's mean, the difference between them, and a permutation p.

    `values` is rows by features, matching `names`. `strata` sets the structure of the permutation.
    To match the estimate as well, pass the rows through `match_within_strata` first.

    The ratio is reported because it is how an enrichment is quoted, and its interval only where the
    delta method applies. A ratio with a small denominator carries a sign but no reliable magnitude,
    so compare features on the difference column.
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
            # Only where a ratio means anything. For a quantity that takes both signs, the ratio of
            # means swings with how close the denominator sits to zero and flips sign through it,
            # so it would look like a magnitude while carrying none.
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
    condition on the subsample that was drawn, and neither says anything about the next draw. For a
    common feature that hardly matters. For a rare one, the abundant class is sampled down hard and
    the estimate moves with the seed by an amount that can rival the stated interval.

    So the estimate, its interval and its p-value all come from one matched sample, which keeps
    them an internally consistent analysis of one design. `ratio_draw_min`, `ratio_draw_max` and
    `difference_draw_sd` report how far the same quantities moved across `repeats` draws, the
    reported one included. Treat them as an imprecision the interval does not include. Where they
    are wide, matching costs more than it gains, and a weighted design is the better instrument.

    Averaging the draws would give a lower-variance estimator, but it is not done, because its
    interval and its p-value would then describe a quantity nothing else in the table is about.
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
        # The null is computed on the reported draw only. It is the expensive half, and the other
        # draws are needed only for their estimates, so running it on them would waste the cost.
        tables.append(direct_effect(values[keep], label[keep], names, strata=strata[keep],
                                    draws=draws if offset == 0 else 0, seed=seed))

    reported = tables[0].copy()
    spread = pd.concat(tables, ignore_index=True).groupby("feature", sort=False)
    reported["ratio_draw_min"] = reported.feature.map(spread.ratio.min())
    reported["ratio_draw_max"] = reported.feature.map(spread.ratio.max())
    reported["difference_draw_sd"] = reported.feature.map(spread.difference.std())
    reported.attrs["design"] = {**tables[0].attrs["design"], "match_repeats": max(repeats, 1)}
    return reported
