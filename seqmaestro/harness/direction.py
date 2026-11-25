"""The direction of a feature, in one vocabulary shared by every model.

A binary target's directions name a class (`higher_in_positive`, `higher_in_negative`), and a
continuous target's say which way the feature moves it (`increases_target`, `decreases_target`),
with `non_monotone` for both. The functions below take the pair of words to answer in, which
`seqmaestro.target` holds for each kind, so the arithmetic is shared and only the words differ.
Every model's direction comes from `from_increments`, which turns a sequence of signed increments
(the steps of a curve, or a tree's splits) into a word, so a new model only has to produce
increments.

Each model takes its increments from what it offers, and its cell names the basis once, in the
header of `features.json`, as `direction_kind` (one of `DIRECTION_KINDS`):

- `l1_coefficient_sign`, the sparse linear models: the sign of each selected coefficient. The fit
  is monotone in every feature, so `non_monotone` cannot occur.
- `least_squares_coefficient_sign`, IGSR and the least-squares models fitted on its terms: the sign
  of each term's weight.
- `shape_function_trend`, the EBM: the steps of each term's shape function, in log-odds for a
  binary target and in the target's units for a continuous one. With no interaction terms the
  model is the sum of these curves, so a curve is everything the model does with its feature.
- `treeshap_dependence`, XGBoost: its gain has no sign, so each row's TreeSHAP contribution from
  the feature, which XGBoost computes exactly, is averaged within bins of the feature to give a
  curve (`dependence_curve`).
- `shapley_dependence`, PySR: the same curve, from the exact Shapley values of the equation's
  output. A product of two features shares their interaction equally between them, so each
  feature's curve includes half of it.
- `tree_split_rates` and `tree_split_means`, the decision tree (CART or DEFT): one step per split,
  the share of the positive class (or the mean of the target) above the threshold less that at or
  below it, weighted by the rows that reach the split. A split near the root therefore counts for
  more than one deep in a branch, and a feature used one way at the root and the other way inside
  a branch is `non_monotone`.

The quantities behind the increments are on scales of their own, so only the words are compared
across models.

`docs/developer/analyses/direction.md` is the account of what a direction means, how it differs
from the marginal effect (`seqmaestro.harness.effect`), and where it breaks down.
"""

from __future__ import annotations

import numpy as np

#: The model's score for the positive class rises with the feature.
HIGHER_IN_POSITIVE = "higher_in_positive"
#: The model's score for the positive class falls as the feature rises.
HIGHER_IN_NEGATIVE = "higher_in_negative"
#: The relationship reverses inside the observed range, so a single direction cannot describe it.
#: This is an answer in itself: a feature that helps at one end and hurts at the other makes a
#: different claim from a feature that does nothing (`None`), so reporting one as the other would
#: mislead.
NON_MONOTONE = "non_monotone"

#: Higher values of the feature go with a higher continuous target.
INCREASES_TARGET = "increases_target"
#: Higher values of the feature go with a lower continuous target.
DECREASES_TARGET = "decreases_target"

#: The pair of words for each kind of target: first the word for a feature whose rise raises the
#: model's score, then the word for one whose rise lowers it.
BINARY_WORDS = (HIGHER_IN_POSITIVE, HIGHER_IN_NEGATIVE)
CONTINUOUS_WORDS = (INCREASES_TARGET, DECREASES_TARGET)

DIRECTIONS = (*BINARY_WORDS, *CONTINUOUS_WORDS, NON_MONOTONE)

#: What each direction becomes when the feature is negated. It is needed wherever two features are
#: matched on *absolute* correlation: a GC fraction and an AT fraction over one window measure one
#: property, so comparing their directions without flipping one would count agreement as
#: disagreement. `non_monotone` is its own opposite, because a curve that reverses still reverses
#: when negated.
OPPOSITE = {HIGHER_IN_POSITIVE: HIGHER_IN_NEGATIVE,
            HIGHER_IN_NEGATIVE: HIGHER_IN_POSITIVE,
            INCREASES_TARGET: DECREASES_TARGET,
            DECREASES_TARGET: INCREASES_TARGET,
            NON_MONOTONE: NON_MONOTONE}

#: What produced a direction, in the same form as `seqmaestro.harness.features.IMPORTANCE_KINDS`:
#: free text, recorded once per file because every record in the file shares it. The module
#: docstring says what each kind is. Never compare the underlying magnitudes across kinds, because
#: only the words above mean the same thing everywhere.
DIRECTION_KINDS = (
    "l1_coefficient_sign",   # l1_logistic, l1_linear
    "shape_function_trend",  # ebm
    "treeshap_dependence",   # xgboost
    "tree_split_rates",      # deft_or_cart, binary target
    "tree_split_means",      # deft_or_cart, continuous target
    "shapley_dependence",    # pysr
    "least_squares_coefficient_sign",  # igsr, and the least-squares models fitted on its terms
)

#: The share of a curve's total movement that must be net movement one way before the curve is
#: given a direction. At 1.0 only a perfectly monotone curve would qualify, which a fitted shape
#: function rarely is, and at 0 nothing would ever be `non_monotone`. At 0.5 the curve must move at
#: least three times as far one way as the other.
#:
#: Like `DEFAULT_TOP_N`, this is a stated convention and is not fitted to any data.
MONOTONICITY_FLOOR = 0.5


def from_increments(increments, words: tuple[str, str]) -> str | None:
    """A direction from a sequence of signed steps, in `words`, or `None` when the steps give none.

    The steps are in whatever unit the model works in: log-odds between adjacent bins of a shape
    function, mean contribution between bins of a dependence curve, or class rate across a split.
    Their scale is not comparable between models, so it is not used: only their signs, and how far
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

    Zero gives `None`. For a coefficient, zero means the feature was not selected, whereas for a
    correlation or an effect size it means the feature is not associated with the target on its
    own. Both are the absence of an answer, so neither is reported as a direction.
    """
    if value is None:
        return None
    value = float(value)
    return from_increments([value], words) if np.isfinite(value) else None


#: The number of bins of the feature that turn per-row contributions into a curve: enough to show a
#: reversal, yet few enough that each bin still holds rows to average over on a split of a few
#: hundred rows.
DEPENDENCE_BINS = 8


def dependence_curve(values, contributions, bins: int = DEPENDENCE_BINS) -> np.ndarray:
    """A dependence curve: the mean contribution per bin of the feature, lowest bin first.

    This is the standard way to summarise a per-row attribution. Rows are grouped by their value of
    the feature, and each group's mean contribution is one point. A curve shows a feature that
    raises the score at one end of its range and lowers it at the other as `non_monotone`, whereas
    its mean contribution over all rows would be close to zero, as it is for a feature the model
    ignores. **The bins are quantiles, so
    every point is backed by rows that exist.** The curve therefore never evaluates the model at a
    value the data does not contain, which is where a partial-dependence sweep goes wrong on
    features as correlated as these.

    A row's bin follows from the share of rows whose value is below its own, so the `bins` bins hold
    about equal numbers of rows, and rows that share a value share a bin. A feature taking two
    values therefore yields two points, and one that is zero on most rows yields a point for its
    zeros and the remaining points for the rows above them. An empty bin is skipped instead of
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
