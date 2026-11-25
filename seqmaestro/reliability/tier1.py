"""Tier 1: how stable the selection is, by complementary pairs subsampling.

Tier 1 measures whether the same procedure, run on a different half of the same data, would have
chosen the feature at all. Every half-sample runs the whole selection (fit on the candidates, keep
what the model used, eliminate to N), because re-ranking one fixed fit would measure the ranking
instead of the selection.

    D = train + val, n = |D|
    for b = 1..B:  A_b, A'_b are disjoint random subsets of size floor(n/2)
    pi_j = (number of the 2B half-samples that selected j) / 2B
    q    = mean size of a selected set

**The bound.** For a threshold tau in (1/2, 1], with S_tau = {j : pi_j >= tau},

    E[V_tau] <= q^2 / ((2 tau - 1) p)

where V_tau counts the features in S_tau whose *true* half-sample selection probability is no
better than chance, q/p. This is the Meinshausen-Bühlmann bound evaluated on the complementary-pairs
design of Shah and Samworth (2013). Their r-concavity refinements are tighter but are not used,
because a number from a different derivation would not be comparable with this one. The bound
assumes that the half-samples are exchangeable and that the procedure is no worse than random at
ranking.

The bound is an expected count, not a probability. At q = 10 and p = 800, tau = 0.8 gives
10^2 / (0.6 * 800) = 0.21, so a set of features each chosen in four half-samples out of five is
expected to hold about a fifth of one chance-level feature. At p = 30 the same arithmetic gives 5.6,
so the bound is printed for every arm but is informative for only some.

**Clusters and positions are treated in the same way**: a group counts as selected when any of its
members is, p is the number of groups, and q the mean number of groups selected. This is the level
to use when the candidates include near-duplicates. A position whose indicators share the selection
between them is stable as a position but unstable as a feature, and only the group-level number
shows it.
"""

from __future__ import annotations

import numpy as np

#: The thresholds every table reports. The lower end is just above 1/2, because the bound is
#: undefined at 1/2 and says nothing below it.
TAU_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def half_samples(n: int, n_pairs: int, seed: int) -> np.ndarray:
    """`2 * n_pairs` row-index sets: each pair is two disjoint subsets of size floor(n/2).

    They are returned as one (2B, floor(n/2)) array, with pair `b` in rows `2b` and `2b + 1`, so
    that the pairing is kept in the record and can be checked.
    """
    half = n // 2
    if half < 2:
        raise ValueError(f"n = {n} is too small to halve.")
    rng = np.random.default_rng(seed)
    out = np.empty((2 * n_pairs, half), dtype=np.int64)
    for pair in range(n_pairs):
        order = rng.permutation(n)
        out[2 * pair] = np.sort(order[:half])
        out[2 * pair + 1] = np.sort(order[half: 2 * half])
    return out


def check_half_samples(halves: np.ndarray, n: int) -> dict:
    """Check the bookkeeping the bound depends on: the halves of each pair are disjoint, equal in
    size and within range."""
    half = n // 2
    if halves.shape[1] != half:
        raise AssertionError(f"Half-samples are {halves.shape[1]} rows, expected {half}.")
    if halves.min() < 0 or halves.max() >= n:
        raise AssertionError("A half-sample indexes a row outside D.")
    for pair in range(halves.shape[0] // 2):
        a, b = halves[2 * pair], halves[2 * pair + 1]
        if len(np.intersect1d(a, b)):
            raise AssertionError(f"Pair {pair} is not disjoint.")
        if len(np.unique(a)) != half or len(np.unique(b)) != half:
            raise AssertionError(f"Pair {pair} repeats a row.")
    return {"n_pairs": int(halves.shape[0] // 2), "half_size": int(half), "n": int(n),
            "disjoint": True}


def expected_false_positives(q_hat: float, p: int, tau: float) -> float:
    """The bound E[V_tau] <= q^2 / ((2 tau - 1) p), which is infinite at or below tau = 1/2, where
    it says nothing."""
    if tau <= 0.5:
        return float("inf")
    return float(q_hat ** 2 / ((2.0 * tau - 1.0) * p))


def selection_frequency(selected: np.ndarray) -> np.ndarray:
    """The selection frequency pi per column, from a (2B, p) boolean selection matrix."""
    return selected.mean(axis=0)


def curve(selected: np.ndarray, p: int, tau_grid=TAU_GRID) -> list[dict]:
    """The size of S_tau and the bound at every threshold."""
    pi_hat = selection_frequency(selected)
    q_hat = float(selected.sum(axis=1).mean())
    return [
        {
            "tau": float(tau),
            "n_selected": int((pi_hat >= tau).sum()),
            "bound_expected_false": expected_false_positives(q_hat, p, tau),
        }
        for tau in tau_grid
    ]


def smallest_tau_with_bound(q_hat: float, p: int, ceiling: float,
                            grid=np.arange(0.505, 1.0001, 0.005)) -> float | None:
    """The lowest threshold whose bound is at or below `ceiling`, or `None` when there is none."""
    for tau in grid:
        if expected_false_positives(q_hat, p, float(tau)) <= ceiling:
            return round(float(tau), 3)
    return None


def collapse(selected: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The selection matrix collapsed to groups: a group is selected in a half-sample when any of
    its members is.

    Returns the (2B, n_groups) matrix and the group labels in the order of its columns.
    """
    unique = np.unique(labels)
    out = np.zeros((selected.shape[0], len(unique)), dtype=bool)
    for column, label in enumerate(unique):
        members = np.flatnonzero(labels == label)
        out[:, column] = selected[:, members].any(axis=1)
    return out, unique


def summarise(selected: np.ndarray, p: int, tau_grid=TAU_GRID) -> dict:
    """Everything Tier 1 reports about one level: features, clusters or positions."""
    pi_hat = selection_frequency(selected)
    sizes = selected.sum(axis=1)
    q_hat = float(sizes.mean())
    return {
        "p": int(p),
        "n_half_samples": int(selected.shape[0]),
        "q_hat": q_hat,
        "selected_set_sizes": {"min": int(sizes.min()), "max": int(sizes.max()), "mean": q_hat},
        "pi_hat": pi_hat,
        "curve": curve(selected, p, tau_grid),
        "tau_bound_1": smallest_tau_with_bound(q_hat, p, 1.0),
        "tau_bound_half": smallest_tau_with_bound(q_hat, p, 0.5),
    }
