"""An importance that means the same thing for every arm.

Each model reports a model-specific importance: a mean absolute shape function, an L1
coefficient, a boosting gain, or a tree's impurity decrease. Those are the right numbers within one
arm, whereas the `Model` protocol states that they are never comparable between arms. That suffices
for an analysis that compares an arm only with itself, but an analysis of whether several arms found
the same thing needs more, because each arm's selection comes from a different quantity and their
agreement partly reflects what each happened to sort by.

Permutation importance is the model-agnostic answer: shuffle a feature's column, predict again, and
see how far the score falls. It needs only a fitted model, the rows and a metric, so it is computed
in the same way for every arm. This module computes it for a model fitted in this process, whereas
DEFT's and IGSR's adapters compute it on their side by the same rule. The metric is the headline
metric of the model's kind of target (`seqmaestro.target`): AUROC for a binary target and R-squared
for a continuous one.

**Its known weakness here is correlated columns.** Shuffling `pos_50_A` leaves `pos_50_C/G/T` in
place, and between them they largely determine the value that was removed, so the drop understates
the feature. For this reason one-hot and *k*-mer columns are deflated relative to standalone
continuous ones. Permuting the four indicators of a position together would fix this, but it would
produce an importance for the *position*, which a per-feature record cannot hold, so columns are
permuted one at a time. Ranks are computed within an arm, and most arms hold one kind of feature
throughout, so the deflation applies evenly and the ordering survives. The exception is an arm that
mixes indicators with continuous features, such as a generative tree over a positional pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqmaestro.target import target_for

#: Repeats per feature. Each is one prediction pass, so the time taken is linear in this and in the
#: number of features. A single draw ranks noisily, so this follows `scikit-learn` in using a few.
DEFAULT_REPEATS = 5

def permutation_importance(
    model,
    X: pd.DataFrame,
    y,
    *,
    columns: list[str] | None = None,
    repeats: int = DEFAULT_REPEATS,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """How far the headline metric of the model's kind of target falls when each column is shuffled.

    A positive value means the model relies on the feature, and a value around zero means it does
    not. A negative value arises when shuffling helps by chance, and it is kept as measured, because
    clipping at zero would turn noise into a weak positive signal for every unused feature.

    Returns an empty mapping when the score is undefined (one class present, or one target value)
    or no repeats were asked for, because a mapping of zeros would be indistinguishable from a model
    that ignores every feature.
    """
    y = np.asarray(y).ravel()
    target = target_for(model.target_kind)
    if repeats < 1:
        return {}
    rng = rng or np.random.default_rng(0)
    # One copy of the rows, shuffled one column at a time and restored. Each prediction is given a
    # frame built over that array, which a model turns back into the same array without copying
    # (`seqmaestro.models.base.fitted_values`), so a prediction takes only the model's arithmetic,
    # without converting the whole frame.
    names = list(X.columns)
    values = X.to_numpy(dtype=float, copy=True)

    def score() -> float:
        frame = pd.DataFrame(values, columns=names, copy=False)
        return target.score(y, np.asarray(model.predict(frame), dtype=float).ravel())

    baseline = score()
    if not np.isfinite(baseline):
        return {}

    position = {name: index for index, name in enumerate(names)}
    out: dict[str, float] = {}
    for name in (columns if columns is not None else names):
        if name not in position:
            continue
        column = position[name]
        original = values[:, column].copy()
        drops = []
        for _ in range(repeats):
            values[:, column] = rng.permutation(original)
            drops.append(baseline - score())
        values[:, column] = original
        out[name] = float(np.mean(drops))
    return out


def describe(repeats: int, split: str, metric: str) -> dict:
    """The metric, repeat count and split the importances were measured with, for the record.

    For a cell under the permuted label condition, the split's labels are the **real** ones, because
    the harness shuffles only the labels the model trains on. The drop then measures how far a model
    fitted on noise relies on each feature when scored against the true labels, which is the null the
    analyses need.
    """
    return {"metric": metric, "repeats": repeats, "split": split}
