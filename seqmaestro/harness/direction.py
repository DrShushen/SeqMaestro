"""Which way a feature points, in one vocabulary every model can speak.

`docs/analyses/direction.md` is the maintained account of this, including how each arm reaches
its answer and where the answer breaks down. Keep the two in step.

An importance says how much a model leaned on a feature. It does not say which class the feature
argues for, or which way it moves a continuous target, which is the first thing a biologist asks. A
claim such as "more upstream start codons" is plausible in either direction, so it can only be
checked once its direction is stated.

Every model can answer this, each from different machinery: a signed coefficient, a shape
function, a tree's split, an attribution binned by the feature. Those quantities are in different
units and are no more comparable with each other than the importances are. **Their signs are
comparable**, and a direction is a sign, so the words defined below are all that is compared
between arms.

Two things are recorded per feature, and they answer different questions:

    direction         what the fitted model does with the feature, given everything else it holds
    marginal_effect   what the feature does in the data, on its own (see `seqmaestro.harness.effect`)

They can disagree, and a disagreement is a finding, not a bug: a feature with no marginal
association that the model still relies on is one it uses to cancel something else.

**One primitive underneath.** A shape function is a curve, a boosted model's dependence is a curve,
and a tree's use of a feature is a set of splits. All three reduce to a sequence of signed
increments, which `from_increments` turns into a word. A new model therefore only has to produce
increments, and the vocabulary does not change.

**Two pairs of words, one per kind of target.** A binary target's directions name a class
(`higher_in_positive`, `higher_in_negative`), and a continuous target's name which way it moves
(`increases_target`, `decreases_target`). `non_monotone` serves both. The functions below take the
pair to answer in, which `seqmaestro.target` holds for each kind, so the arithmetic is shared and
only the words differ.
"""

from __future__ import annotations

import numpy as np

#: The model's score for the positive class rises with the feature.
HIGHER_IN_POSITIVE = "higher_in_positive"
#: The model's score for the positive class falls as the feature rises.
HIGHER_IN_NEGATIVE = "higher_in_negative"
#: The relationship reverses inside the observed range, so no single direction describes it. This
#: is an answer in its own right: a feature that helps at one end and hurts at the other makes a
#: different claim from a feature that does nothing (`None`), and reporting one as the other
#: misleads.
NON_MONOTONE = "non_monotone"

#: Higher values of the feature go with a higher continuous target.
INCREASES_TARGET = "increases_target"
#: Higher values of the feature go with a lower continuous target.
DECREASES_TARGET = "decreases_target"

#: Each kind's pair: the word for a feature whose rise raises the model's score, then the word for
#: one whose rise lowers it.
BINARY_WORDS = (HIGHER_IN_POSITIVE, HIGHER_IN_NEGATIVE)
CONTINUOUS_WORDS = (INCREASES_TARGET, DECREASES_TARGET)

DIRECTIONS = (*BINARY_WORDS, *CONTINUOUS_WORDS, NON_MONOTONE)

#: What each direction becomes on the feature's negation. Needed wherever two features are matched
#: on *absolute* correlation: a GC fraction and an AT fraction over one window are one property, so
#: comparing their directions without flipping one counts agreement as disagreement.
#: `non_monotone` is its own opposite, because the negation of a curve that reverses still reverses.
OPPOSITE = {HIGHER_IN_POSITIVE: HIGHER_IN_NEGATIVE,
            HIGHER_IN_NEGATIVE: HIGHER_IN_POSITIVE,
            INCREASES_TARGET: DECREASES_TARGET,
            DECREASES_TARGET: INCREASES_TARGET,
            NON_MONOTONE: NON_MONOTONE}

#: What produced a direction, in the same form as `seqmaestro.harness.features.IMPORTANCE_KINDS`:
#: free text, recorded once per file because every record in the file shares it. Never compare the
#: underlying magnitudes across kinds. Only the words above mean the same thing everywhere.
DIRECTION_KINDS = (
    "l1_coefficient_sign",   # l1_logistic, l1_linear: monotone in each feature, so the sign says everything
    "shape_function_trend",  # ebm: the additive term's curve, in log-odds or the target's units
    "treeshap_dependence",   # xgboost: mean TreeSHAP contribution across bins of the feature
    "tree_split_rates",      # deft_or_cart: class rates on either side of each split
    "tree_split_means",      # deft_or_cart: the target's mean on either side of each split
    "shapley_dependence",    # pysr: mean Shapley value of the equation's output across bins of the feature
    "least_squares_coefficient_sign",  # igsr, and the least-squares models fitted on its terms
)

#: The share of a curve's total movement that must be net movement one way before the curve is
#: given a direction. At 1.0 only a perfectly monotone curve would qualify, and no fitted shape
#: function is one. At 0 nothing would ever be `non_monotone`. At 0.5 the curve must move at least
#: three times as far one way as the other.
#:
#: A stated convention, like `DEFAULT_TOP_N`, and not fitted to any data.
MONOTONICITY_FLOOR = 0.5


def from_increments(increments, words: tuple[str, str]) -> str | None:
    """A direction from a sequence of signed steps, in `words`, or None when the steps give none.

    The steps are in whatever unit the model works in: log-odds between adjacent bins of a shape
    function, mean contribution between bins of a dependence curve, class rate across a split.
    Their scale is not comparable between models and is not used. Only their signs, and how far
    they agree with each other, decide the result.

    Returns `None` when the feature does not move the model at all, because a flat curve has no
    direction. Non-finite steps are dropped, so that a part of a curve the model could not
    evaluate is not counted as agreement.
    """
    increments = np.asarray(increments, dtype=float).ravel()
    increments = increments[np.isfinite(increments)]
    if increments.size == 0:
        return None

    net = float(increments.sum())
    travelled = float(np.abs(increments).sum())
    if travelled == 0:
        return None
    if abs(net) / travelled < MONOTONICITY_FLOOR:
        return NON_MONOTONE
    return words[0] if net > 0 else words[1]


def from_curve(levels, words: tuple[str, str]) -> str | None:
    """A direction from a curve given as levels instead of steps, such as a shape function."""
    levels = np.asarray(levels, dtype=float).ravel()
    return from_increments(np.diff(levels), words) if levels.size >= 2 else None


def from_sign(value, words: tuple[str, str]) -> str | None:
    """A direction from a single signed number: a coefficient, a correlation, an effect size.

    Zero gives `None`. For a coefficient, zero means the feature was not selected, and for a
    correlation or an effect size it means the feature does not go with the target on its own. Both
    are the absence of an answer, so neither is reported as a direction.
    """
    if value is None:
        return None
    value = float(value)
    return from_increments([value], words) if np.isfinite(value) else None


#: Number of bins of the feature used to turn per-row contributions into a curve. Enough to show a
#: reversal, and few enough that each bin still holds rows to average over on a split of a few
#: hundred rows.
DEPENDENCE_BINS = 8


def dependence_curve(values, contributions, bins: int = DEPENDENCE_BINS) -> np.ndarray:
    """A dependence curve: the mean contribution per bin of the feature, lowest bin first.

    This is the standard way to summarise a per-row attribution. Rows are grouped by their value of
    the feature, and each group's mean contribution is one point. **The bins are quantiles, so
    every point is backed by rows that exist.** The curve never asks the model about a value the
    data does not contain, which is where a partial-dependence sweep goes wrong on features as
    correlated as these.

    A row's bin follows from the share of rows whose value is below its own, so `bins` bins hold
    about as many rows each, and rows that share a value share a bin. A feature taking two values
    therefore yields two points, and one that is zero on most rows yields a point for its zeros
    and the remaining points for the rows above them. A bin no row falls in is skipped instead of
    leaving a gap.
    """
    values = np.asarray(values, dtype=float).ravel()
    contributions = np.asarray(contributions, dtype=float).ravel()
    if values.size != contributions.size:
        raise ValueError(f"{values.size} value(s) against {contributions.size} contribution(s).")

    usable = np.isfinite(values) & np.isfinite(contributions)
    values, contributions = values[usable], contributions[usable]
    if values.size == 0 or np.ptp(values) == 0:
        return np.empty(0)

    below = np.searchsorted(np.sort(values), values, side="left")
    index = below * max(bins, 1) // values.size
    return np.array([contributions[index == bin_].mean() for bin_ in np.unique(index)])
