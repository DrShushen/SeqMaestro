"""The kinds of target DEFT grows a tree for, as `dataset.target_kind` names them.

A **binary** target holds the labels 0 and 1, and a leaf predicts the share of 1s among its rows. A
**continuous** target holds any finite numbers, and a leaf predicts their mean. The rest follows
from the kind: the splitting criterion must be one written for it (`target_kinds` on each
criterion), the prompts must describe the task it sets (`params_generation.target_kind`), and a
class weighting applies to a binary target only, since a continuous one has no classes.
"""

from __future__ import annotations

import numpy as np

BINARY = "binary"
CONTINUOUS = "continuous"
TARGET_KINDS = (BINARY, CONTINUOUS)


def checked_kind(kind) -> str:
    """`kind`, refused unless it is one DEFT grows a tree for."""
    if kind not in TARGET_KINDS:
        raise ValueError(f"`target_kind` is {kind!r}; it must be one of {', '.join(TARGET_KINDS)}.")
    return kind


def check_target(y, kind: str) -> None:
    """Refuse a target that does not hold what its kind requires: 0 and 1 for a binary target,
    finite numbers for a continuous one.

    A binary tree grown on other labels would treat every distinct value as a class and predict the
    share of 1s among them, so a continuous target declared binary by omission would give a tree
    of nonsense and no error.
    """
    values = np.asarray(y).ravel()
    if checked_kind(kind) == BINARY:
        unexpected = sorted({value.item() if isinstance(value, np.generic) else value
                             for value in values} - {0, 1})
        if unexpected:
            raise ValueError(
                f"A binary target holds the labels 0 and 1, and this one also holds "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}. Declare "
                f"`target_kind: continuous` for a numeric target."
            )
        return
    try:
        numbers = values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"A continuous target holds numbers: {error}") from error
    if not np.isfinite(numbers).all():
        raise ValueError(
            f"A continuous target holds finite numbers, and {int((~np.isfinite(numbers)).sum())} "
            f"value(s) are missing or infinite."
        )
