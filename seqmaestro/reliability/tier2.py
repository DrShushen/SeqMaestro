"""Tier 2: does the feature go with the target on its own, calibrated against permuted targets.

**One statistic for every kind of feature.** For a binary target, T_j = |AUC_j - 1/2|, where AUC_j
is the Mann-Whitney area under the curve of feature j used directly as a score, with average ranks
for ties. For a continuous target, T_j = |rho_j|, Spearman's rank correlation of feature j with the
target, with average ranks for ties on both sides. Indicators, counts and continuous ratios can all
be ranked, so one statistic covers every candidate and every kind of feature is measured the same
way. It detects monotone association only: a feature that separates the classes, or moves the
target, at both extremes but not in the middle can score zero.

**The null permutes the target.** Ranking the columns once and permuting `y` gives every null
statistic from one matrix product. With R the (n, p) matrix of within-column ranks and Y the (n, B2)
matrix of permuted label vectors, R'Y is every feature's positive-class rank sum under every
permutation, and

    AUC = (rank sum of the positives - n1 (n1 + 1) / 2) / (n1 n0)

turns each sum into an AUC, because permuting a binary label leaves n1 fixed. This makes 100,000
permutations over thousands of features affordable. A continuous target goes through the same
product with its doubled, centred ranks in place of the labels: R'r is n times the rank
covariance, and it becomes Spearman's rho on division by the two rank norms, which permuting the
target leaves fixed.

**The comparisons are exact.** The ranks are doubled and centred before the product (`2r - (n + 1)`
is an integer for average ranks), so every statistic is an integer, held exactly in float64 at
every size this is run on, and `T(b) >= T` is an exact comparison. This matters because a binary
indicator on a few hundred rows takes very few distinct values of the statistic, so much of its
permutation distribution sits on the observed statistic, and a comparison that rounded those ties
the wrong way would shrink the p-value.

**The p-values and their adjustments.** Each feature gets a raw permutation p-value, with the +1
correction of Phipson and Smyth; BH and BY q-values and a Holm-adjusted p-value (`multitest.py`
says what each assumes); and a Westfall-Young min-P adjusted p-value. Westfall-Young controls the
family-wise error rate using the dependence present among the candidates, so on positional
indicators and overlapping *k*-mers it is far less conservative than Bonferroni. The plug-in FDR
estimate is reported beside them: the average number of features a permuted run calls at a cutoff,
divided by the number the real run calls. Observed and null p-values share one scale because the
observed statistic and its B2 permuted counterparts are treated as B2 + 1 exchangeable draws.

**Model importances are never substituted for T_j.** Permuting `y` calibrates a multivariate
importance only under the global null, so a per-feature p-value computed from one is not valid.
For this reason Tier 2 is model-free, and Tiers 1 and 3, which use the model, make no per-feature
significance claim of this kind.

**An undefined value is ranked as its column's median over the defined ones**
(`impute_non_finite`), as the panel's clusters rank it. Our own featurisers fill such values before
their candidates are built, but a DEFT feature written for one node can return NaN on rows unlike
those that reach the node. The statistic is then that of the imputed column, and the panel records
which candidates were undefined and on what share of D.

**Unweighted, so a weighted design is refused.** Every row counts once here. A design that fits
with per-row weights would need a weighted statistic and a null that respects the weights, and
neither is implemented. `refuse_weighted` raises for such a design, and the panel records the
refusal instead of computing an unweighted number for it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from seqmaestro.harness.stability import impute_non_finite
from seqmaestro.reliability.multitest import benjamini_hochberg, benjamini_yekutieli, holm
from seqmaestro.target import BINARY, CONTINUOUS, target_for

#: Permutations per block of the label matrix, which bounds the (n, block) array on a large split.
PERMUTATION_BLOCK = 1000

#: The q-value levels every table carries.
Q_LEVELS = (0.01, 0.05, 0.1, 0.2)

#: The raw-p cutoffs the plug-in FDR curve is evaluated at.
ALPHA_LEVELS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

#: The signed association Tier 2 reports for each kind of target, and its value for a feature with
#: no association: the AUC for a binary target, and Spearman's rho for a continuous one.
ASSOCIATION = {BINARY: ("auc", 0.5), CONTINUOUS: ("rho", 0.0)}


class WeightedDesignError(ValueError):
    """Tier 2 was asked for on a design that fits with per-row weights."""


def refuse_weighted(weight_column: str | None) -> None:
    """Refuse a weighted design: Tier 2's statistic and its null count every row once."""
    if weight_column is not None:
        raise WeightedDesignError(
            f"Tier 2 is not computed: the design fits with per-row weights (`{weight_column}`), and "
            f"its statistic (a feature's association with the target on its own, calibrated by "
            f"permuting the labels) "
            f"counts every row once. A weighted statistic, with a null that respects the weights, "
            f"does not exist here yet. Tiers 1 and 3 fit with the weights and are unaffected.")


def doubled_centred_ranks(X: np.ndarray) -> np.ndarray:
    """`2r - (n + 1)` per column, an integer whenever `r` is an average rank."""
    return 2.0 * rankdata(X, axis=0) - (X.shape[0] + 1)


def statistic_to_auc(statistic: np.ndarray, n_positive: int, n_negative: int) -> np.ndarray:
    """`AUC - 1/2` from the doubled centred rank sum."""
    return statistic / (2.0 * n_positive * n_negative)


def profile(X: np.ndarray, y: np.ndarray, names: list[str], *, target_kind: str, seed: int,
            n_permutations: int, block: int = PERMUTATION_BLOCK, q_levels=Q_LEVELS,
            alpha_levels=ALPHA_LEVELS) -> dict:
    """Every candidate's marginal association with `y`, calibrated against permuted targets."""
    target = target_for(target_kind)
    y = target.values(y)
    n = len(y)
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows and y has {n}.")
    p = X.shape[1]
    X = impute_non_finite(X)
    ranks = doubled_centred_ranks(X)
    if target.labels:
        # The labels themselves: R'y is the positives' doubled, centred rank sum.
        scores = y.astype(float)
        n_positive = int(y.sum())
        n_negative = n - n_positive
        if n_positive == 0 or n_negative == 0:
            raise ValueError("The labels hold only one class.")
    else:
        scores = doubled_centred_ranks(y[:, None])[:, 0]
        if not scores.any():
            raise ValueError("The target takes one value only.")

    signed = ranks.T @ scores
    if target.labels:
        association = 0.5 + statistic_to_auc(signed, n_positive, n_negative)
    else:
        norms = np.sqrt(np.square(ranks).sum(axis=0) * np.square(scores).sum())
        association = np.divide(signed, norms, out=np.zeros_like(signed), where=norms > 0)
    # Kept in its exact integer form for the comparisons below, and turned into the association
    # only where it is reported.
    observed = np.abs(signed)

    null = np.empty((p, n_permutations))
    rng = np.random.default_rng(seed)
    for start in range(0, n_permutations, block):
        width = min(block, n_permutations - start)
        permuted = np.empty((n, width))
        for column in range(width):
            permuted[:, column] = rng.permutation(scores)
        null[:, start:start + width] = np.abs(ranks.T @ permuted)

    at_least = (null >= observed[:, None]).sum(axis=1)
    p_value = (1.0 + at_least) / (n_permutations + 1.0)

    wy, plug_in = _min_p_and_plug_in(null, observed, p_value, alpha_levels)
    q_bh, q_by, p_holm = benjamini_hochberg(p_value), benjamini_yekutieli(p_value), holm(p_value)

    name, centre = ASSOCIATION[target_kind]
    increasing, decreasing = target.directions
    return {
        "names": list(names),
        "target_kind": target_kind,
        "association": name,
        "n_rows": int(n),
        **({"n_positive": n_positive} if target.labels else {}),
        "n_permutations": int(n_permutations),
        "resolution": float(1.0 / (n_permutations + 1.0)),
        name: association,
        "statistic": np.abs(association - centre),
        "p_value": p_value,
        "q_bh": q_bh,
        "q_by": q_by,
        "p_holm": p_holm,
        "p_westfall_young": wy,
        "mean": X.mean(axis=0),
        "nonzero_rate": (X != 0).mean(axis=0),
        "direction": np.where(association > centre, increasing,
                              np.where(association < centre, decreasing, "tied")),
        "called": {float(level): {
            "bh": int((q_bh <= level).sum()),
            "by": int((q_by <= level).sum()),
            "holm": int((p_holm <= level).sum()),
            "westfall_young": int((wy <= level).sum()),
        } for level in q_levels},
        "plug_in_fdr": plug_in,
    }


#: The per-feature arrays `profile` returns, which are saved beside its summary, after the
#: association the kind of target reports (`ASSOCIATION`).
ARRAYS = ("statistic", "p_value", "q_bh", "q_by", "p_holm", "p_westfall_young", "mean",
          "nonzero_rate", "direction")


def arrays(target_kind: str) -> tuple[str, ...]:
    """The per-feature arrays of a result for this kind of target, its association first."""
    return (ASSOCIATION[target_kind][0], *ARRAYS)


def _min_p_and_plug_in(null: np.ndarray, observed: np.ndarray, p_value: np.ndarray,
                       alpha_levels) -> tuple[np.ndarray, list[dict]]:
    """Westfall-Young adjusted p-values and the plug-in FDR curve, from the same null matrix.

    A permutation's p-value for feature j is its rank inside that feature's null row, on the
    same B2 + 1 scale as the observed p-value. The observed statistic counts as one of the draws,
    so a null statistic that beats it is counted exactly once on both sides. The work is done in
    blocks of features, so nothing the size of the null matrix is held twice.
    """
    p, n_permutations = null.shape
    min_p = np.ones(n_permutations)
    counts = {alpha: np.zeros(n_permutations) for alpha in alpha_levels}

    chunk = max(1, int(1e7 // max(n_permutations, 1)))
    for start in range(0, p, chunk):
        block = null[start:start + chunk]
        # `max` ranking of the negated row is #{b' : T(b') >= T(b)} exactly, ties included.
        within = rankdata(-block, method="max", axis=1)
        beaten_by_observed = (observed[start:start + chunk, None] >= block).astype(np.float64)
        null_p = (within + beaten_by_observed) / (n_permutations + 1.0)
        np.minimum(min_p, null_p.min(axis=0), out=min_p)
        for alpha in alpha_levels:
            counts[alpha] += (null_p <= alpha).sum(axis=0)

    wy = np.array([(1.0 + (min_p <= value).sum()) / (n_permutations + 1.0) for value in p_value])
    plug_in = []
    for alpha in alpha_levels:
        called = int((p_value <= alpha).sum())
        expected = float(counts[alpha].mean())
        plug_in.append({
            "alpha": float(alpha),
            "n_called": called,
            "expected_null_called": expected,
            "plug_in_fdr": float(expected / max(1, called)),
        })
    return wy, plug_in
