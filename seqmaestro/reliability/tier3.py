"""Tier 3: whether the reported model needs a feature, judged by leaving it out.

Tiers 1 and 2 ask about the feature. Tier 3 asks about M*, the arm's reduced model: refit M* on its
features minus one, with the same rows, weights and settings that produced it, and test whether its
predictions on the test split get worse. Nothing is re-selected and no setting changes, so the
refit is M* with one column removed, and a difference can only be put down to that column. M* itself
is not refitted: it is the reduction's saved model.

**The comparison is paired and per-row.** The loss of the target's kind is computed row by row for
M* and for the refit (`p_reduced` below) and differenced, so the test is on
`d_i = loss_i(refit) - loss_i(M*)` instead of on two summary numbers, whose difference has no
standard error. The loss is log-loss for a binary target and squared error for a continuous one,
the loss each kind's models are fitted by. `H0: E[d] <= 0` (the refit is no worse) is tested
one-sided, because a feature whose removal improves the model is not evidence that the feature is
needed. A sign-flip permutation test is reported beside the t-test, because per-row loss differences
are heavy-tailed and skewed, a squared error on a heavy-tailed target most of all, and a t-test on a
hundred of them may be poorly calibrated.

**Clusters are also dropped whole.** Two reported features that are near-duplicates can each look
unnecessary alone and be necessary together, which single-feature LOCO cannot show. Every cluster
holding two or more reported features gets its own refit with all of them removed. Features that
are individually substitutable and jointly necessary are a finding about their cluster, and the two
results do not contradict each other.

**"Necessary" means Holm-adjusted p < 0.05 across the whole family of refits.** It is a statement
about this model on this test split. It says nothing about the biology or about any other model that
might use the feature.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from seqmaestro.reliability.multitest import benjamini_hochberg, holm
from seqmaestro.target import BINARY, CONTINUOUS, target_for

#: Probabilities are clipped before the logarithm. A model confidently wrong about one row would
#: otherwise contribute an infinite loss and take the mean and its interval with it.
EPSILON = 1e-12

#: Sign flips behind the permutation check.
N_SIGN_FLIPS = 10_000

#: Family-wise level at which a feature is called necessary.
HOLM_ALPHA = 0.05


def refits(reported: list[str], cluster_of: dict) -> list[dict]:
    """Every refit: one per reported feature, and one per cluster holding several of them.

    `cluster_of` maps a reported feature to its cluster label. A cluster holding one reported
    feature would repeat that feature's refit, and is skipped.
    """
    plan = [{"label": name, "kind": "feature", "dropped": [name],
             "columns": [other for other in reported if other != name]} for name in reported]
    grouped: dict = {}
    for name in reported:
        grouped.setdefault(cluster_of[name], []).append(name)
    for label, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(members) < 2:
            continue
        plan.append({"label": f"cluster_{label}", "kind": "cluster", "dropped": list(members),
                     "columns": [other for other in reported if other not in set(members)]})
    return plan


def per_sample_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPSILON, 1.0 - EPSILON)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def per_sample_squared_error(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.square(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))


#: The per-row loss of each kind of target, and the name its keys are written under.
LOSS = {BINARY: ("log_loss", per_sample_log_loss),
        CONTINUOUS: ("squared_error", per_sample_squared_error)}


def compare(y: np.ndarray, p_model: np.ndarray, p_reduced: np.ndarray, *, target_kind: str,
            seed: int, n_sign_flips: int = N_SIGN_FLIPS) -> dict:
    """One refit (`p_reduced`) against M* (`p_model`), on the test split."""
    target = target_for(target_kind)
    loss_name, loss = LOSS[target_kind]
    y = np.asarray(y)
    p_model, p_reduced = np.asarray(p_model, dtype=float), np.asarray(p_reduced, dtype=float)
    d = loss(y, p_reduced) - loss(y, p_model)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    stderr = sd / np.sqrt(n) if n > 1 and sd > 0 else 0.0

    if stderr > 0:
        t_statistic = mean / stderr
        p_t = float(stats.t.sf(t_statistic, df=n - 1))
        half_width = float(stats.t.ppf(0.975, df=n - 1) * stderr)
    else:
        t_statistic = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
        p_t = 0.0 if mean > 0 else 1.0
        half_width = 0.0

    rng = np.random.default_rng(seed)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_sign_flips, n))
    null_means = (flips * d).mean(axis=1)
    p_flip = float((1 + (null_means >= mean).sum()) / (n_sign_flips + 1))

    headline = target.headline
    compared = {
        f"mean_delta_{loss_name}": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "t_statistic": float(t_statistic),
        "p_t_one_sided": p_t,
        "p_sign_flip": p_flip,
        f"delta_{headline}": target.score(y, p_model) - target.score(y, p_reduced),
        f"{headline}_reduced": target.score(y, p_reduced),
        "n_rows": int(n),
    }
    if target.probabilities:
        # Rows at the clip, in either model. A tree whose leaf is pure predicts 0 or 1, and its
        # log-loss on a row it gets wrong is bounded only by EPSILON, so a handful of rows can carry
        # the whole mean. They are counted and not smoothed away: when the count is not zero, go by
        # the sign-flip test.
        at_clip = ((p_model <= EPSILON) | (p_model >= 1 - EPSILON)
                   | (p_reduced <= EPSILON) | (p_reduced >= 1 - EPSILON))
        compared["n_at_clip"] = int(at_clip.sum())
    compared["max_abs_difference"] = float(np.abs(d).max()) if n else 0.0
    return compared


def summarise(y: np.ndarray, p_model: np.ndarray, predictions: dict[str, np.ndarray],
              plan: list[dict], *, target_kind: str, seed: int,
              holm_alpha: float = HOLM_ALPHA) -> dict:
    """Every refit against M*, with the family-wise and false-discovery adjustments across them."""
    rows = [{**{key: entry[key] for key in ("label", "kind", "dropped")},
             **compare(y, p_model, predictions[entry["label"]], target_kind=target_kind,
                       seed=seed)} for entry in plan]

    p_values = np.array([row["p_t_one_sided"] for row in rows])
    for row, adjusted, q, flip_adjusted in zip(
            rows, holm(p_values), benjamini_hochberg(p_values),
            holm(np.array([row["p_sign_flip"] for row in rows])), strict=True):
        row["p_holm"] = float(adjusted)
        row["q_bh"] = float(q)
        row["p_sign_flip_holm"] = float(flip_adjusted)
        row["verdict"] = "necessary" if adjusted < holm_alpha else "substitutable"

    target = target_for(target_kind)
    loss_name, loss = LOSS[target_kind]
    return {
        "target_kind": target_kind,
        f"{target.headline}_model": target.score(y, p_model),
        f"mean_{loss_name}_model": float(loss(y, p_model).mean()),
        "holm_alpha": float(holm_alpha),
        "n_tests": len(rows),
        "rows": rows,
    }
