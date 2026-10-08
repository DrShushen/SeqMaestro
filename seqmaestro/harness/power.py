"""What a design can detect, computed from its row counts alone before anything is run.

A full grid is expensive and takes hours. If the test set cannot separate the effect from chance,
the grid reports near-chance scores with wide intervals, which look the same as the effect being
absent. A little arithmetic beforehand avoids that, and its result is the number a report's opening
section quotes so that every arm can be judged against it.

Everything here is closed form, with no data and no fitting. For a binary target it follows from
how many rows of each class the split holds, and for a continuous target from how many rows it
holds. A weighted design is worth fewer rows than it has, and Kish's effective sample size says how
many (`effective_sample_size`, `contrast_effective_n`). `seqmaestro.target` applies all of it to a
prepared split when Stage 1 finalises it, and `experiments/prepare/design_power.py` is the
command-line tool for planning one.

The AUROC interval is Hanley and McNeil's, which assumes an exponential score distribution and is
the standard first approximation. The correlation interval is Fisher's, through the z-transform,
which assumes the predictions and the targets are close to bivariate normal. Both are close enough
to decide whether to run something. Once the run exists, report its own bootstrap interval.
"""

from __future__ import annotations

import numpy as np

CONFIDENCE_Z = 1.96


def auroc_standard_error(auroc: float, n_positive: float, n_negative: float) -> float:
    """Hanley and McNeil (1982), the standard closed form for the variance of an AUROC.

    NaN below two rows of either class: a design with one positive has no sampling distribution to
    describe, and returning a number there would invite quoting one.
    """
    if n_positive < 2 or n_negative < 2:
        return float("nan")
    q1 = auroc / (2 - auroc)
    q2 = 2 * auroc**2 / (1 + auroc)
    variance = (auroc * (1 - auroc)
                + (n_positive - 1) * (q1 - auroc**2)
                + (n_negative - 1) * (q2 - auroc**2)) / (n_positive * n_negative)
    return float(np.sqrt(max(variance, 0.0)))


def auroc_half_width(auroc: float, n_positive: float, n_negative: float,
                     z: float = CONFIDENCE_Z) -> float:
    """Half the 95% interval on a single measured AUROC."""
    return z * auroc_standard_error(auroc, n_positive, n_negative)


def smallest_detectable_auroc(n_positive: float, n_negative: float,
                              grid: np.ndarray | None = None) -> float:
    """The smallest AUROC whose interval excludes 0.5, which is the smallest this design can claim.

    **An arm below this has not been shown to be at chance.** It has only been shown to score below
    what the test set can separate from chance.

    Solved on a grid because the standard error depends on the AUROC being tested, so the condition
    is not linear in it.
    """
    grid = np.arange(0.500, 0.999, 0.001) if grid is None else grid
    for auroc in grid:
        if auroc - auroc_half_width(auroc, n_positive, n_negative) > 0.5:
            return float(auroc)
    return float("nan")


def correlation_interval(r: float, n: float, z: float = CONFIDENCE_Z) -> tuple[float, float]:
    """The 95% interval on a correlation `r` measured over `n` rows, by Fisher's z-transform.

    NaN below four rows, where the transform's standard error, 1 / sqrt(n - 3), is undefined.
    """
    if n <= 3:
        return float("nan"), float("nan")
    centre, spread = np.arctanh(r), z / np.sqrt(n - 3)
    return float(np.tanh(centre - spread)), float(np.tanh(centre + spread))


def correlation_half_width(r: float, n: float, z: float = CONFIDENCE_Z) -> float:
    """Half the width of the 95% interval on a correlation, which is asymmetric about `r`."""
    low, high = correlation_interval(r, n, z)
    return (high - low) / 2


def smallest_detectable_correlation(n: float, z: float = CONFIDENCE_Z) -> float:
    """The smallest correlation between predictions and targets whose interval excludes zero.

    **An arm below this may still predict something.** The test set is too small to separate it
    from an arm that predicts nothing. Under Fisher's transform the interval excludes zero when
    |arctanh(r)| exceeds z / sqrt(n - 3), so the answer is closed form.
    """
    if n <= 3:
        return float("nan")
    return float(np.tanh(z / np.sqrt(n - 3)))


def effective_sample_size(weights: np.ndarray | None, n: int | None = None) -> float:
    """How many equally-weighted rows a weighted design is worth.

    `(sum w)^2 / sum w^2`, Kish's formula. Equal weights give back the row count, and the more
    unequal the weights, the smaller the result.

    **Counting rows instead oversells a weighted design.** A stratum-balancing weight of
    `1 / n_class(stratum)` gives a stratum's 500 majority rows a total weight of 1.0 between them,
    the same as its 5 minority rows carry. Reported as 505 rows the design looks large, but as a
    two-class contrast (`contrast_effective_n`) it is worth about 10. Whether a weighted design
    beats a matched one depends on the weighting scheme, and only effective sizes, never row
    counts, can tell.
    """
    if weights is None:
        if n is None:
            raise ValueError("effective_sample_size needs either weights or n.")
        return float(n)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float(total * total / np.square(weights).sum())


def contrast_effective_n(weights: np.ndarray | None, y) -> float:
    """Effective sample size for a *two-class contrast*, which is what a label comparison estimates.

    The precision of a difference between two groups is set by the harmonic mean of their sizes, so
    a design with 500 effective majority rows and 5 effective minority rows is worth about 10, not
    505. This combines each class's Kish size by that harmonic mean, and is the number to quote for
    a matched or weighted design and to compare designs on.
    """
    labels = np.asarray(y)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError(f"contrast_effective_n expects two classes, found {len(classes)}.")
    sizes = []
    for label in classes:
        mask = labels == label
        sizes.append(effective_sample_size(None if weights is None else np.asarray(weights)[mask],
                                           n=int(mask.sum())))
    if min(sizes) <= 0:
        return 0.0
    return float(2.0 / (1.0 / sizes[0] + 1.0 / sizes[1]))
