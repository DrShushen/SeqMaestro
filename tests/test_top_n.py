"""The comparable view: each arm's most important features, capped at a shared N.

An EBM fits a shape function to every input, so it "selects" the whole k-mer vocabulary, whereas
L1 selects a small fraction of it and a shallow tree names a handful. Comparing those sets directly
would make stability a measure of set size, so the main comparison is over the top N of each. As in
`seqmaestro.harness.stability`, a run (`FeatureRun`) is one cell's features with their values.
"""

from __future__ import annotations

import numpy as np
import pytest

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.features import DEFAULT_TOP_N, top_features
from seqmaestro.harness.stability import FeatureRun, stability, stability_views

RNG = np.random.default_rng(0)


def records(importances: dict[str, float]) -> list[FeatureRecord]:
    return [
        FeatureRecord(name=name, description=f"{name} d", kind="kmer", importance=importance)
        for name, importance in importances.items()
    ]


def a_run(label, columns: dict[str, tuple[np.ndarray, float]]) -> FeatureRun:
    """`columns` maps name to (values, importance)."""
    return FeatureRun(
        label,
        records({name: importance for name, (_, importance) in columns.items()}),
        np.column_stack([values for values, _ in columns.values()]),
    )


# ------------------------------------------------------------------------------ top_features


def test_the_default_is_twenty():
    assert DEFAULT_TOP_N == 20


def test_the_most_important_come_first():
    chosen = top_features(records({"small": 0.1, "big": 0.9, "middling": 0.5}), n=2)

    assert [record.name for record in chosen] == ["big", "middling"]


def test_selection_is_by_absolute_importance():
    """A signed coefficient's sign gives its direction, so strength is ranked by absolute value."""
    chosen = top_features(records({"positive": 0.2, "negative": -0.9}), n=1)

    assert chosen[0].name == "negative"


def test_a_short_list_is_not_padded():
    """A model that used three features contributes three, with no blanks to make up twenty."""
    assert len(top_features(records({"a": 1.0, "b": 0.5, "c": 0.1}), n=20)) == 3


def test_features_without_importance_come_last():
    chosen = top_features(records({"unscored": None, "scored": 0.01}), n=1)

    assert chosen[0].name == "scored"


def test_n_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        top_features(records({"a": 1.0}), n=0)


# ----------------------------------------------------------------------- the run-level view


def test_restricting_a_run_keeps_values_in_step_with_records():
    """If the columns and the records fall out of order, matching compares the wrong features."""
    keep, drop = RNG.normal(size=50), RNG.normal(size=50)
    run = a_run("seed_1", {"important": (keep, 0.9), "trivial": (drop, 0.01)})

    top = run.top(1)

    assert top.names == ["important"]
    assert np.array_equal(top.values[:, 0], keep)


def test_a_model_that_selects_everything_is_stable_for_free():
    """Identical full sets score a perfect 1, which says nothing about whether the model ranks the
    same features highest. The top-N view exists to avoid this."""
    columns = {f"f{i}": RNG.normal(size=100) for i in range(30)}
    everything = [
        FeatureRun(
            f"seed_{seed}",
            records({name: 1.0 / (1 + (i + seed) % 30) for i, name in enumerate(columns)}),
            np.column_stack(list(columns.values())),
        )
        for seed in (0, 1)
    ]

    full = stability(everything)
    top = stability(everything, top_n=5)

    assert full["jaccard"]["mean_pairwise"] == 1.0, "every feature 'selected' in both runs"
    assert top["jaccard"]["mean_pairwise"] < 1.0, "but they do not agree on which matter most"


def test_the_view_is_recorded_with_the_numbers():
    runs = [a_run(f"seed_{i}", {"a": (RNG.normal(size=60), 1.0)}) for i in range(2)]

    assert stability(runs)["top_n"] is None
    assert stability(runs, top_n=5)["top_n"] == 5


# --------------------------------------------------------------------------------- both views


def test_both_views_are_returned_together():
    columns = {f"f{i}": RNG.normal(size=80) for i in range(25)}
    runs = [
        FeatureRun(
            f"seed_{seed}",
            records({name: RNG.random() for name in columns}),
            np.column_stack(list(columns.values())),
        )
        for seed in (0, 1)
    ]

    views = stability_views(runs)

    assert set(views) == {"full", "top_n"}
    assert views["full"]["top_n"] is None
    assert views["top_n"]["top_n"] == DEFAULT_TOP_N
    assert max(views["top_n"]["n_features_per_run"].values()) <= DEFAULT_TOP_N


def test_the_top_n_view_can_be_turned_off():
    runs = [a_run(f"seed_{i}", {"a": (RNG.normal(size=60), 1.0)}) for i in range(2)]

    assert set(stability_views(runs, top_n=None)) == {"full"}


def test_n_is_configurable():
    columns = {f"f{i}": RNG.normal(size=80) for i in range(10)}
    runs = [
        FeatureRun(f"seed_{seed}",
                   records({name: RNG.random() for name in columns}),
                   np.column_stack(list(columns.values())))
        for seed in (0, 1)
    ]

    assert max(stability_views(runs, top_n=3)["top_n"]["n_features_per_run"].values()) == 3
