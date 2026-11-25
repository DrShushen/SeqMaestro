"""The two sanity checks: labels with no signal, and labels with a known planted signal.

**Global null.** The labels are replaced by a random permutation of themselves, and Tier 2 is rerun
under them on every arm. The rerun is quick and directly tests the permutation calibration on that
arm's candidates: a correct null gives uniform p-values and about alpha * p features called at any
raw cutoff alpha. Tier 1 is rerun under the permuted labels on the check arm alone, because that
needs a second round of selections and tests what every arm shares (the half-samples and the
bound).

**Planted signal.** A target is drawn from a linear predictor on a few real columns of the check
arm's candidates, at a moderate strength: strong enough that the procedure should find them, but
not so strong that any procedure would. For a binary target the labels are drawn through the
logistic link, whereas a continuous target is the predictor plus unit Gaussian noise, so at a
predictor standard deviation of 1.5 its R-squared is about 0.69. The target is drawn on the real
X, so the planted features keep the candidates' correlation structure and their near-duplicates are
genuinely associated too. The check therefore measures recovery and says nothing about the false
discovery rate. The model is saved as a recipe, so that the same features, coefficients and
standardisation generate labels for D (the training and validation rows) and for the test split.
On D the labels feed Tier 1 and the in-sample Tier 2, which is the only place with enough rows to
call the planted features rather than merely rank them, and on the test split they feed Tier 2 at
the reveal.
"""

from __future__ import annotations

import numpy as np

from seqmaestro.target import CONTINUOUS

#: A planted feature is drawn from the columns whose commonest value covers less than this share of
#: D, so that it varies enough for a signal to be planted on it, or from all the columns when too
#: few qualify.
PLANTED_MAX_MODAL_SHARE = 0.9

#: The selection frequencies at which the summaries count features: the stability threshold a
#: report quotes, and a stricter one.
PI_LEVELS = (0.7, 0.9)

#: A planted feature counts as recovered when its rank among the candidates is at most this.
TOP_RANKS = 10


def planted_recipe(X: np.ndarray, names: list[str], modal_share: np.ndarray, *, seed: int,
                   n_planted: int, signal_sd: float, target_kind: str) -> dict:
    """A linear predictor on `n_planted` real columns of the candidates, as a reusable recipe for a
    target of kind `target_kind`.

    The coefficients alternate between +1 and -1 on the standardised columns, and the linear
    predictor is scaled so that its standard deviation on D is `signal_sd`.
    """
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(modal_share < PLANTED_MAX_MODAL_SHARE)
    if len(eligible) < n_planted:
        eligible = np.arange(X.shape[1])
    chosen = np.sort(rng.choice(eligible, size=n_planted, replace=False))
    block = np.asarray(X[:, chosen], dtype=float)
    mean, sd = block.mean(axis=0), np.maximum(block.std(axis=0), 1e-9)
    beta = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_planted)])
    raw = ((block - mean) / sd) @ beta
    return {"columns": chosen.tolist(), "names": [names[int(i)] for i in chosen],
            "beta": beta.tolist(), "mean": mean.tolist(), "sd": sd.tolist(),
            "scale": float(signal_sd / max(raw.std(), 1e-9)), "signal_sd": float(signal_sd),
            "target_kind": target_kind}


def draw_planted(X: np.ndarray, recipe: dict, *, seed: int) -> np.ndarray:
    """A target from a planted-signal recipe, for any rows over the same columns: labels through the
    logistic link for a binary target, and the predictor plus unit Gaussian noise for a continuous
    one."""
    block = np.asarray(X[:, np.asarray(recipe["columns"])], dtype=float)
    score = ((block - np.asarray(recipe["mean"])) / np.asarray(recipe["sd"])) @ np.asarray(recipe["beta"])
    score = score * recipe["scale"]
    rng = np.random.default_rng(seed)
    if recipe["target_kind"] == CONTINUOUS:
        return score + rng.normal(0.0, 1.0, len(score))
    return (rng.random(len(score)) < 1.0 / (1.0 + np.exp(-score))).astype(np.int8)


def permuted(y: np.ndarray, *, seed: int) -> np.ndarray:
    """The global null's labels: the real ones, permuted."""
    return np.random.default_rng(seed).permutation(np.asarray(y))


def _called(called: dict, level: float, method: str) -> int:
    """A count from `called`, whose keys come back from a JSON round trip as strings, so either form
    of key is accepted."""
    return int(called[level if level in called else str(level)][method])


def null_tier2(p_value: np.ndarray, called: dict) -> dict:
    """Tier 2 under permuted labels: how many features are called, and how far the p-values are from
    uniform."""
    p_value = np.asarray(p_value)
    grid = np.linspace(0, 1, 21)[1:]
    return {
        "p": len(p_value),
        "n_called_bh_0.05": _called(called, 0.05, "bh"),
        "n_called_by_0.05": _called(called, 0.05, "by"),
        "n_called_wy_0.05": _called(called, 0.05, "westfall_young"),
        "ks_statistic_vs_uniform": float(np.max(np.abs(
            np.searchsorted(np.sort(p_value), grid, side="right") / len(p_value) - grid))),
        "mean_p": float(p_value.mean()),
        "share_p_below_0.05": float((p_value <= 0.05).mean()),
        "quantiles": {str(q): float(np.quantile(p_value, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }


def null_tier1(q_hat: float, pi_hat: np.ndarray) -> dict:
    """Tier 1 under permuted labels, where the selection still chooses what its model uses (at most
    N features) and nothing should be stable."""
    pi_hat = np.asarray(pi_hat)
    return {"q_hat": q_hat,
            "max_pi_hat": float(pi_hat.max()),
            **{f"n_pi_ge_{level}": int((pi_hat >= level).sum()) for level in PI_LEVELS},
            "mean_pi_of_selected": float(pi_hat[pi_hat > 0].mean()) if (pi_hat > 0).any() else 0.0,
            "n_features_ever_selected": int((pi_hat > 0).sum())}


def _ranks(values: np.ndarray) -> dict[int, int]:
    order = np.argsort(-np.asarray(values), kind="stable")
    return {int(index): position + 1 for position, index in enumerate(order)}


def planted_tier1(q_hat: float, p: int, pi_hat: np.ndarray, names: list[str], recipe: dict) -> dict:
    """Tier 1 under planted labels: where each planted feature ranks by selection frequency, among
    the `p` candidates offered to the selection."""
    rank = _ranks(pi_hat)
    rows = [{"feature": name, "pi_hat": float(pi_hat[names.index(name)]),
             "rank": rank[names.index(name)]} for name in recipe["names"]]
    return {"q_hat": q_hat, "p": int(p), "planted": rows,
            f"n_planted_in_top_{TOP_RANKS}": sum(1 for row in rows if row["rank"] <= TOP_RANKS),
            f"n_planted_pi_ge_{PI_LEVELS[0]}": sum(1 for row in rows if row["pi_hat"] >= PI_LEVELS[0])}


def planted_tier2(result: dict, recipe: dict) -> dict:
    """Tier 2 under planted labels: whether each planted feature is called, and where it ranks."""
    names = list(result["names"])
    rank = _ranks(result["statistic"])
    association = result["association"]
    rows = [{"feature": name, association: float(result[association][names.index(name)]),
             "q_bh": float(result["q_bh"][names.index(name)]), "rank": rank[names.index(name)]}
            for name in recipe["names"]]
    return {"n_called_bh_0.05": _called(result["called"], 0.05, "bh"), "p": len(names),
            "n_rows": result["n_rows"], "planted": rows,
            "n_planted_called_bh_0.05": sum(1 for row in rows if row["q_bh"] <= 0.05),
            f"n_planted_in_top_{TOP_RANKS}": sum(1 for row in rows if row["rank"] <= TOP_RANKS)}
