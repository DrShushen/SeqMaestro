"""Multiplicity corrections, written out so a report can name the one it uses.

The three corrections control different things, so a report prints all three side by side:

| | controls | assumes |
|---|---|---|
| Benjamini-Hochberg | false discovery rate | independence, or positive regression dependence |
| Benjamini-Yekutieli | false discovery rate | nothing (valid under arbitrary dependence) |
| Holm | family-wise error rate | nothing |

Positional indicators and overlapping *k*-mer counts are dependent candidates, and their
dependence has not been shown to be positive, so BH is the number to quote, whereas BY is the one
with a guarantee. BY's q-values are BH's multiplied by the harmonic number of p (a factor of about
7.3 at p = 800), so BY calls fewer features in exchange for its guarantee.

The tests compare these functions with `statsmodels.stats.multitest` instead of hand-worked
examples, because the likely bug is an off-by-one in the step-up ordering, which a three-element
example would not reveal.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH q-values: the step-up procedure, made monotone from the largest p downwards."""
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return p_values.copy()
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q, 1.0)
    return out


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """BY q-values: BH inflated by the harmonic number of p, valid under arbitrary dependence."""
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return p_values.copy()
    # Summed exactly, because the approximation log(n) + gamma omits a 1/(2n) term, which is enough
    # to move a q-value across a threshold at the boundary. The sum takes negligible time at any p
    # this will see.
    harmonic = float(np.sum(1.0 / np.arange(1, n + 1)))
    return np.minimum(benjamini_hochberg(p_values) * harmonic, 1.0)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values: step-down, made monotone from the smallest p upwards."""
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return p_values.copy()
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = np.maximum.accumulate(ranked * (n - np.arange(n)))
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out
