"""Recursive feature elimination, written once for every arm. **numpy only.**

The loop is the textbook one: fit, rank by the model's importance, drop the weakest, refit. The
three decisions a textbook leaves open are made here, in one place, so every arm of every dataset is
eliminated by the same rules.

**The schedule.** Dropping one feature at a time from thousands would take thousands of fits of a
model that can be slow to fit. Dropping a fraction at each step instead costs a number of fits that
grows with the logarithm of the number of features, and the late steps, which drop one at a time,
fit few columns and are cheap. So a quarter of the remaining features go at each step while more
than `FINE_BELOW` are left, and one at a time after that, which keeps the part of the trajectory a
reader studies at full resolution.

**Features the model does not use.** A feature with zero importance is unused: an L1 coefficient
driven to zero, a term whose shape function is flat, a column no tree split on. All of them are
dropped together as soon as they appear, though never below `n_target` columns, instead of a
quarter at a time. This is faster, and it avoids an arbitrary choice, since "the weakest quarter of
the features the model ignores" is an arbitrary subset of them.

**Ties.** Many arms have long ties at zero, so the tie-break decides real outcomes, and it is
written down here instead of being left to `argsort`. Among features of equal importance, the first
to go is the one that last carried any importance longest ago, then the one whose importance was
smallest when it last had any, then the one later in the arm's record order. Every part is
deterministic and none of it looks at a label.

**Why numpy only.** The vendored tree runs in its own environment and cannot import `seqmaestro`,
but its arms must be eliminated by this same loop. `experiments/adapters/deft_cart.py` loads this
file by path and records its digest, and `tests/test_reduction.py` fails if it imports anything but
numpy and the standard library. The reliability panel's selection step calls it too, starting from
an arm's candidates instead of the features its model recorded.

The caller scores the elimination on validation only. Nothing here sees any labels except those of
the fitting rows.
"""

from __future__ import annotations

import numpy as np

#: Share of the surviving features kept at each coarse step.
KEEP_FRACTION = 0.75

#: At or below this many features, one goes at a time.
FINE_BELOW = 30


def schedule(n: int, n_target: int) -> int:
    """How many features to drop from `n`, never overshooting the target."""
    if n <= n_target:
        return 0
    if n <= FINE_BELOW:
        return 1
    return min(n - n_target, max(1, n - int(np.ceil(n * KEEP_FRACTION))))


def order_worst_first(active: np.ndarray, importance: np.ndarray, last_seen: np.ndarray,
                      last_value: np.ndarray) -> np.ndarray:
    """Indices of `active`, weakest first, under the tie-break the module docstring states."""
    keys = np.lexsort((
        -active,                     # last resort: the arm's record order, earlier kept
        last_value[active],          # how much it carried when it last carried anything
        last_seen[active],           # how recently it was used at all
        importance,                  # what it is worth now
    ))
    return keys


def run(
    fit_step,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    names: list[str],
    n_target: int,
    on_step=None,
) -> dict:
    """Eliminate down to `n_target`, returning the trajectory and the final model.

    `fit_step(X, y, columns) -> fitted` is the only thing that differs between arms. It is passed
    in, not imported, so the tree arms, which cannot import this project's model package, use the
    same loop. A fitted object has `importances()`, one non-negative number per column, and
    `predict(X)`, its score for each row. Anything else a fit needs, such as per-row weights for the
    fitting rows, the caller closes over.

    A set of features already no larger than `n_target` is fitted once and kept whole. The survivors
    are `final_columns`, and `final_used` names those the final model uses (`used_columns`).
    """
    n_features = len(names)
    active = np.arange(n_features)
    last_seen = np.zeros(n_features)
    last_value = np.zeros(n_features)
    removed_at = np.full(n_features, -1)

    steps, val_scores = [], []
    step = 0
    fitted = first_fitted = None
    while True:
        columns = [names[i] for i in active]
        # A step over every column passes the array itself, because fancy indexing copies, and on a
        # wide arm that copy can run to gigabytes for a selection that selects everything.
        whole = len(active) == n_features
        fit_matrix = X_fit if whole else X_fit[:, active]
        val_matrix = X_val if whole else X_val[:, active]
        fitted = fit_step(fit_matrix, y_fit, columns)
        importance = fitted.importances()
        val_scores.append(fitted.predict(val_matrix))

        used = importance > 0
        last_seen[active[used]] = step + 1
        last_value[active[used]] = importance[used]

        steps.append({
            "step": step,
            "n_features": len(active),
            "n_used_by_model": int(used.sum()),
            "features": list(columns),
            "importance": [float(v) for v in importance],
        })
        if first_fitted is None:
            first_fitted = fitted
        if on_step is not None:
            on_step(steps[-1])

        if len(active) <= n_target:
            break

        n_drop = schedule(len(active), n_target)
        n_zero = int((~used).sum())
        if n_zero:
            n_drop = min(len(active) - n_target, max(n_drop, n_zero))
        worst = order_worst_first(active, importance, last_seen, last_value)[:n_drop]
        removed_at[active[worst]] = step
        active = np.delete(active, worst)
        step += 1

    return {
        "steps": steps,
        "val_scores": np.asarray(val_scores[-1]),
        "val_scores_by_step": val_scores,
        "final_columns": [names[i] for i in active],
        "final_used": used_columns([names[i] for i in active], fitted.importances()),
        "final_importance": fitted.importances(),
        "fitted": fitted,
        "first_fitted": first_fitted,
        "removed_at": removed_at,
    }


def used_columns(columns: list[str], importance) -> list[str]:
    """The columns a fitted model uses, in their order: those whose importance is above zero.

    The elimination keeps `n_target` columns even when the model uses fewer. The columns it does not
    use then hold their places by the tie-break alone, which is the same on any rows that share the
    record order, so a selection chose only the columns its final model uses.
    """
    return [str(name) for name, value in zip(columns, importance, strict=True) if value > 0]


def top_n(first_importance, n_target: int) -> np.ndarray:
    """The `n_target` columns the full model ranked highest, in record order.

    This is the one-shot alternative to the recursion (keep what the first fit ranked highest and
    fit once), and so the control for whether the recursion earned its cost. The sort is stable, so
    ties keep record order.
    """
    return np.sort(np.argsort(-np.asarray(first_importance), kind="stable")[:n_target])
