"""Class weighting, defined once so every arm means the same thing by it.

On an imbalanced dataset (a 10% positive rate, say), every model left alone spends its capacity on
the majority class: a tree's impurity gain is dominated by it, and a boosted model's early rounds
chase it. The comparison is between *feature sources*, so any difference in how the models handle
imbalance is noise in the thing being measured.

**One definition.** `balanced` here is scikit-learn's, `n_samples / (n_classes * count)` per class,
which is also what DEFT's tree resolves `class_weight="balanced"` to. The agreement matters more
than the formula. The models fitted in Core and the vendored tree sit on opposite sides of an
environment boundary and cannot share code, so they share the arithmetic instead. If one side's
definition of "balanced" drifts, the arms stop being comparable and nothing reports it.

**Weights are for fitting only.** Metrics are computed on unweighted rows, validation and test
alike, because the prevalence a model is scored against is the one the data has. Weighting the
evaluation as well would report a score for a class-balanced population under the name AUPRC.
"""

from __future__ import annotations

import numpy as np

#: What a config's `grid.class_weight` may be. The config's reader takes its choices from here,
#: so the set a config may state and the set `resolve_class_weight` accepts cannot drift apart.
CLASS_WEIGHT_CHOICES = ("balanced", "none")


def resolve_class_weight(name: str | None) -> str | None:
    """Accept a class weighting as a config states it (`grid.class_weight`), or `None`, refusing
    anything else.

    `"none"` maps to `None`, so a config states either choice as a word.
    """
    if name is None or (isinstance(name, str) and name.lower() == "none"):
        return None
    if name in CLASS_WEIGHT_CHOICES:
        return name
    raise ValueError(
        f"Unknown class_weight {name!r}; expected one of {CLASS_WEIGHT_CHOICES}."
    )


def resolve_fit_weights(y, class_weight: str | None, given: np.ndarray | None) -> np.ndarray | None:
    """The weights a `fit` should use: the caller's per-row weights if given, else class weights.

    Per-row weights **replace** class weighting and are not combined with it. They exist for a
    matched design expressed as weights instead of subsampling. Such weights already equalise the
    classes within each stratum, so a class correction on top would correct twice. Every model
    resolves this here, in the same way.
    """
    if given is None:
        return sample_weight(y, class_weight)
    given = np.asarray(given, dtype=float)
    if len(given) != len(np.asarray(y)):
        raise ValueError(f"sample_weight has {len(given)} entries for {len(np.asarray(y))} rows.")
    if not np.all(np.isfinite(given)) or np.any(given < 0):
        raise ValueError("sample_weight must be finite and non-negative.")
    if given.sum() <= 0:
        raise ValueError("sample_weight sums to zero; nothing would be fitted.")

    # **Rescaled to mean 1, which changes the fit.** A regularised objective is
    # `C * sum_i w_i * loss_i + penalty`, so the *total* weight sets how much the data counts
    # against the penalty. Weights that balance strata, `1 / n_class(stratum)`, are small, and their
    # sum can be a small fraction of the row count. Unnormalised, they multiply the effective
    # regularisation by the ratio of row count to total weight, which can be enough for an L1 fit
    # to zero every coefficient. Normalising leaves every *relative* weight untouched, and matches
    # the convention `balanced` already follows, whose weights also sum to n.
    return given * (len(given) / given.sum())


def sample_weight(y, class_weight: str | None) -> np.ndarray | None:
    """Per-row fitting weights, or `None` when no weighting is asked for.

    `None` instead of an array of ones, because a model handed uniform weights is not always
    identical to one handed none (libraries take different code paths), and the unweighted
    configuration should be the library's unweighted fit.
    """
    if resolve_class_weight(class_weight) is None:
        return None

    labels = np.asarray(y)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        # One class present: every weight would be equal anyway, and the ratio is undefined.
        return None
    weights = {
        label: len(labels) / (len(classes) * count)
        for label, count in zip(classes, counts, strict=True)
    }
    return np.array([weights[label] for label in labels], dtype=float)
