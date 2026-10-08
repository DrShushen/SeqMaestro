"""Step 0, the label-blind work: which candidates to profile, and which are near-duplicates.

Both parts run on **D = train + val only**, and neither looks at a label. This is what lets the
resulting column set be carried unchanged onto the test split at the reveal. A filter or a
clustering recomputed there would be a decision made with held-out rows.

**The prefilter.** A feature that is constant, or nearly constant, has no association to measure and
no selection to be stable about. Kept, it would count as a hypothesis in Tier 2's multiplicity
corrections that can almost never be called, and it would raise p in Tier 1's bound while almost
never being selected, which makes the bound look better than it is. For a binary column the usual
rule is to drop zero variance, or a non-zero prevalence below 1% or above 99%. That rule would
delete every continuous feature that is never zero, so the prefilter uses a form that gives the
same result on a binary column and the intended one on the rest: **drop a feature whose single
commonest value covers more than the configured share of D, or whose variance is zero.** On a 0/1
column, with the share set to 0.99, the two rules are identical.

**The clustering.** *k*-mer counts of overlapping words are near-duplicates by construction, and two
generated features can compute the same function under different names. A feature that is unstable
only because its exact twin was chosen instead should not count as unreliable, and the
cluster-level view separates that case from real instability. The clustering is complete linkage on
1 - |Spearman rho|, cut so that **every pair inside a cluster has |rho| at least the configured
threshold**. Complete linkage is what makes this hold for every pair, and not only along a chain of
neighbours. Spearman is used because the kinds of feature are on different scales, and monotone
agreement is what makes one feature a stand-in for another. The sign is discarded because a
perfectly anti-correlated feature carries the same information.

**Position groups** are formed by construction. The indicators of one position are mutually
exclusive, so their correlations sit near -1/(alphabet - 1) and the clustering threshold never
groups them. Grouping them by position lets a report say "this position matters" as well as "this
base at this position matters".
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from seqmaestro.harness.stability import impute_non_finite

#: Rows used to estimate the correlation matrix: all of D, or a seeded subsample of this many rows
#: when D is larger. The cost of a p x p Spearman grows with the rows, its numbers are only compared
#: against the clustering threshold, and 20,000 rows pin a correlation to about +/- 0.01.
MAX_CORRELATION_ROWS = 20_000

#: Tie-break order for a cluster's representative: a plain indicator, then a *k*-mer count, then a
#: generated function. This is the order of how much a reader must accept about a feature before
#: acting on it. Any other kind comes last.
KIND_RANK = {"positional": 0, "kmer": 1, "llm_generated": 2}


def prefilter(X: np.ndarray, names: list[str], max_modal_share: float) -> dict:
    """Which columns of D survive, and why each of the rest did not."""
    n, p = X.shape
    kept = np.ones(p, dtype=bool)
    reasons: dict[str, str] = {}
    modal_share = np.empty(p)
    nonzero_rate = np.empty(p)

    for column in range(p):
        values = X[:, column]
        nonzero_rate[column] = float(np.count_nonzero(values) / n)
        _, counts = np.unique(values, return_counts=True)
        modal_share[column] = float(counts.max() / n)
        if len(counts) == 1:
            kept[column] = False
            reasons[names[column]] = "constant"
        elif modal_share[column] > max_modal_share:
            kept[column] = False
            reasons[names[column]] = f"modal value covers {modal_share[column]:.4f} of D"

    return {
        "kept": kept,
        "modal_share": modal_share,
        "nonzero_rate": nonzero_rate,
        "dropped": reasons,
        "n_before": int(p),
        "n_after": int(kept.sum()),
        "max_modal_share": float(max_modal_share),
    }


def undefined(X: np.ndarray, names: list[str]) -> dict[str, float]:
    """Each column undefined (NaN or infinite) on some rows, with the share of rows it is undefined
    on, for the record of what the rank statistics imputed."""
    share = (~np.isfinite(X)).mean(axis=0)
    return {names[column]: round(float(share[column]), 4) for column in np.flatnonzero(share > 0)}


def spearman_abs(X: np.ndarray, seed: int, max_rows: int = MAX_CORRELATION_ROWS) -> np.ndarray:
    """|Spearman rho| between every pair of columns, computed as the Pearson correlation of the
    ranked columns, with average ranks for ties.

    An undefined value (NaN or infinite) is ranked as its column's median over the defined ones,
    as the matching of features across repeats ranks one (`impute_non_finite`). Our own
    featurisers fill such values before their candidates are built, but a feature computed on a
    method's side can carry them: a DEFT feature written for one node can return NaN on rows unlike
    those that reach the node.
    """
    rows = np.arange(X.shape[0])
    if X.shape[0] > max_rows:
        rows = np.random.default_rng(seed).choice(X.shape[0], size=max_rows, replace=False)
        rows.sort()
    ranked = rankdata(impute_non_finite(X[rows]), axis=0)
    ranked -= ranked.mean(axis=0, keepdims=True)
    norm = np.sqrt((ranked ** 2).sum(axis=0))
    # A column constant on the sampled rows is given zero correlation with every other column,
    # because dividing by its zero norm would put a NaN into the linkage and break the clustering.
    safe = norm.copy()
    safe[safe == 0] = 1.0
    ranked /= safe
    rho = ranked.T @ ranked
    rho[norm == 0, :] = 0.0
    rho[:, norm == 0] = 0.0
    np.fill_diagonal(rho, 1.0)
    return np.abs(np.clip(rho, -1.0, 1.0))


def cluster(abs_rho: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster label per column: complete linkage on 1 - |rho|, cut at 1 - threshold."""
    p = abs_rho.shape[0]
    if p == 1:
        return np.zeros(1, dtype=np.int64)
    distance = 1.0 - abs_rho
    np.fill_diagonal(distance, 0.0)
    distance = np.maximum(distance, distance.T)
    tree = linkage(squareform(distance, checks=False), method="complete")
    return fcluster(tree, t=1.0 - threshold, criterion="distance").astype(np.int64)


def worst_within_cluster(abs_rho: np.ndarray, labels: np.ndarray) -> float:
    """The lowest |rho| of any pair sharing a cluster, so the clustering's guarantee is measured."""
    worst = 1.0
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) < 2:
            continue
        worst = min(worst, float(abs_rho[np.ix_(members, members)].min()))
    return worst


def representatives(labels: np.ndarray, pi_hat: np.ndarray, kinds: list[str],
                    names: list[str]) -> dict[int, int]:
    """One index per cluster: the highest Tier 1 selection frequency, ties by kind, then name."""
    chosen = {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        best = min(members, key=lambda i: (-pi_hat[i], KIND_RANK.get(kinds[i], len(KIND_RANK)),
                                           names[i]))
        chosen[int(label)] = int(best)
    return chosen


def position_group(kind: str, params: dict) -> str:
    """The position an indicator belongs to, for the position-level summary; empty for the rest."""
    return f"position_{params['position']}" if kind == "positional" else ""
