"""How many independent nulls (permuted draws) a run has, which can be fewer than its permuted
cells.

Under seed scope (`permutation_scope: seed`) every arm at a seed shares one permuted draw. A draw
that happens to sit a little above chance then lifts every arm's permuted cell at that seed at
once, which looks like a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqmaestro.dataset import Columns
from seqmaestro.harness.runner import (
    PERMUTATION_SCOPES,
    labels_to_choose_on,
    permutation_key,
    permutation_rng,
    subsample_training_rows,
)
from seqmaestro.target import BINARY

ARMS = [("xgboost", "positional"), ("xgboost", "positional+kmer"), ("l1_logistic", "positional")]


def labels(n=200, positive=40):
    return np.concatenate([np.ones(positive), np.zeros(n - positive)])


def test_seed_scope_gives_every_arm_the_same_draw():
    """Under seed scope every arm of a seed is fitted against one shuffle of the labels."""
    draws = [permutation_rng(42, permutation_key("seed", arm)).permutation(labels()) for arm in ARMS]
    assert all(np.array_equal(draws[0], other) for other in draws[1:])


def test_cell_scope_gives_each_arm_its_own_draw():
    draws = [permutation_rng(42, permutation_key("cell", arm)).permutation(labels()) for arm in ARMS]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])


def test_cell_scope_is_reproducible_across_processes():
    """The key is hashed with `sha256`, because Python's `hash()` is salted per process and a rerun
    would not match."""
    key = permutation_key("cell", ARMS[0])
    assert key == "xgboost/positional"
    assert np.array_equal(permutation_rng(42, key).permutation(labels()),
                          permutation_rng(42, key).permutation(labels()))


def test_the_seed_still_moves_the_draw_under_cell_scope():
    key = permutation_key("cell", ARMS[0])
    assert not np.array_equal(permutation_rng(42, key).permutation(labels()),
                              permutation_rng(43, key).permutation(labels()))


def test_permuting_keeps_the_labels_and_only_reorders_them():
    y = labels()
    for key in (None, permutation_key("cell", ARMS[0])):
        shuffled = permutation_rng(42, key).permutation(y)
        assert sorted(shuffled) == sorted(y)


def test_cell_scope_refuses_to_run_without_something_to_tell_cells_apart():
    """Cell scope needs the cell's arm (a model's name and a pool's). Without it the draw would fall
    back to a shared one while claiming independent ones."""
    for arm in (None, ("xgboost", ""), ("", "positional")):
        with pytest.raises(ValueError, match="permutation_scope='cell' needs"):
            permutation_key("cell", arm)


def test_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="Unknown permutation_scope"):
        permutation_key("per-arm", ARMS[0])
    assert PERMUTATION_SCOPES == ("seed", "cell")


def test_subsample_threads_the_key_through():
    frame = pd.DataFrame({"sequence": ["ACGT"] * 200, "y": labels().astype(int)})
    common = {"train": frame, "seed": 42, "train_fraction": None,
              "columns": Columns("sequence", "y", None, (), BINARY), "permute_labels": True}
    _, shared_a = subsample_training_rows(**common, permutation_key=None)
    _, shared_b = subsample_training_rows(**common, permutation_key=None)
    _, per_cell = subsample_training_rows(**common, permutation_key="xgboost/positional")
    assert np.array_equal(shared_a, shared_b)
    assert not np.array_equal(shared_a, per_cell)


def test_unpermuted_rows_are_untouched_whatever_the_key():
    frame = pd.DataFrame({"sequence": ["ACGT"] * 200, "y": labels().astype(int)})
    common = {"train": frame, "seed": 42, "train_fraction": None,
              "columns": Columns("sequence", "y", None, (), BINARY)}
    _, real = subsample_training_rows(**common, permute_labels=False,
                                      permutation_key="xgboost/positional")
    assert np.array_equal(real, frame.y.to_numpy())


def test_a_real_cell_chooses_against_the_real_validation_labels_whatever_the_key():
    frame = pd.DataFrame({"sequence": ["ACGT"] * 200, "y": labels().astype(int)})
    columns = Columns("sequence", "y", None, (), BINARY)
    for key in (None, permutation_key("cell", ARMS[0])):
        chosen_on = labels_to_choose_on(frame, seed=42, columns=columns, permute_labels=False,
                                        permutation_key=key)
        assert np.array_equal(chosen_on, frame.y.to_numpy())


def test_a_control_chooses_against_the_validation_labels_shuffled_as_its_training_labels_are():
    """An arm that makes choices on the validation rows is scored against their real labels. In a
    control cell it chooses against those labels shuffled by the draw its training labels take, so
    the run's permutation scope decides which arms share the shuffle."""
    frame = pd.DataFrame({"sequence": ["ACGT"] * 200, "y": labels().astype(int)})
    columns = Columns("sequence", "y", None, (), BINARY)

    def chosen_on(scope, arm):
        return labels_to_choose_on(frame, seed=42, columns=columns, permute_labels=True,
                                   permutation_key=permutation_key(scope, arm))

    shuffled = chosen_on("seed", ARMS[0])
    assert sorted(shuffled) == sorted(frame.y) and not np.array_equal(shuffled, frame.y)
    assert np.array_equal(shuffled, permutation_rng(42).permutation(frame.y.to_numpy()))
    assert all(np.array_equal(shuffled, chosen_on("seed", arm)) for arm in ARMS[1:])
    assert not np.array_equal(chosen_on("cell", ARMS[0]), chosen_on("cell", ARMS[1]))
