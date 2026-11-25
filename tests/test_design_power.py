"""What a design can detect, checked against the properties the arithmetic has to have."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqmaestro.harness.power import (
    auroc_half_width,
    auroc_standard_error,
    smallest_detectable_auroc,
)

# The command-line front end is a script, so it is loaded by path. It only reads the splits and
# formats the result, whereas the arithmetic imported above is library code with its own tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "prepare"))
from design_power import describe, describe_rows, from_splits

from seqmaestro.target import BINARY, CONTINUOUS


def test_more_rows_narrow_the_interval():
    widths = [auroc_half_width(0.6, n, n) for n in (100, 400, 1600)]
    assert widths[0] > widths[1] > widths[2]


def test_the_interval_narrows_as_the_square_root_of_n():
    assert auroc_half_width(0.6, 400, 400) == pytest.approx(auroc_half_width(0.6, 100, 100) / 2,
                                                            rel=0.05)


def test_an_imbalanced_design_is_limited_by_its_smaller_side():
    """1,000 against 1,000 gives a narrower interval than 100 against 10,000, despite holding
    fewer rows."""
    assert auroc_half_width(0.6, 1000, 1000) < auroc_half_width(0.6, 100, 10_000)


def test_too_few_rows_to_say_anything_is_not_an_answer():
    assert np.isnan(auroc_standard_error(0.6, 1, 500))
    assert np.isnan(auroc_standard_error(0.6, 500, 0))


def test_a_stronger_effect_is_easier_to_establish():
    """The interval shrinks toward the extremes, so a large effect needs fewer rows."""
    assert auroc_half_width(0.95, 200, 200) < auroc_half_width(0.55, 200, 200)


def test_the_smallest_detectable_auroc_clears_chance_and_the_one_below_it_does_not():
    for n in (100, 334, 2000):
        found = smallest_detectable_auroc(n, n)
        assert found - auroc_half_width(found, n, n) > 0.5
        below = found - 0.001
        assert below - auroc_half_width(below, n, n) <= 0.5


def test_a_bigger_design_can_detect_a_smaller_effect():
    assert smallest_detectable_auroc(2000, 2000) < smallest_detectable_auroc(200, 200)


def test_a_few_hundred_rows_per_class_cannot_establish_a_small_effect():
    """A matched design can end up this size, and this floor then applies to every arm in the run."""
    assert smallest_detectable_auroc(334, 334) == pytest.approx(0.55, abs=0.02)


def test_describe_reports_the_harmonic_effective_n():
    row = describe("d", 10, 10_000)
    assert row["effective_n"] == pytest.approx(19.96, abs=0.1)


def _write_splits(directory: Path, weights=None):
    directory.mkdir(parents=True, exist_ok=True)
    for name, n in (("train", 400), ("test", 200)):
        frame = pd.DataFrame({"sequence": ["ACGT"] * n,
                              "is_positive": [1] * (n // 4) + [0] * (n - n // 4)})
        if weights is not None and name == "train":
            frame["fit_weight"] = weights(len(frame))
        frame.to_csv(directory / f"{name}.csv", index=False)


def test_from_splits_reports_test_precision_and_training_information(tmp_path):
    _write_splits(tmp_path / "plain")
    table = from_splits(tmp_path / "plain", "is_positive", BINARY, "fit_weight")
    assert len(table) == 2
    assert "what can be stated" in table.design.iloc[0]
    assert "what is there to state" in table.design.iloc[1]
    assert table.note.iloc[1] == "unweighted"


def test_unequal_weights_reduce_the_training_figure_and_say_so(tmp_path):
    """Many unequally weighted rows can amount to little information, and the tool exists to make
    that visible."""
    _write_splits(tmp_path / "weighted",
                  weights=lambda n: np.concatenate([np.ones(n // 4), np.full(n - n // 4, 1e-4)]))
    table = from_splits(tmp_path / "weighted", "is_positive", BINARY, "fit_weight")
    assert table.effective_n.iloc[1] < 200
    assert "ignoring the weights would say" in table.note.iloc[1]


def test_zero_weighted_rows_are_excluded_as_the_split_intends(tmp_path):
    _write_splits(tmp_path / "zeroed",
                  weights=lambda n: np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)]))
    table = from_splits(tmp_path / "zeroed", "is_positive", BINARY, "fit_weight")
    assert table.n_positive.iloc[1] + table.n_negative.iloc[1] == 200


def test_a_continuous_design_is_sized_by_its_rows(tmp_path):
    _write_splits(tmp_path / "continuous")
    table = from_splits(tmp_path / "continuous", "is_positive", CONTINUOUS, None)
    assert table.rows.iloc[0] == 200 and table.effective_n.iloc[1] == 400
    assert table.smallest_detectable_r.iloc[0] == describe_rows("d", 200)["smallest_detectable_r"]


def test_more_rows_detect_a_smaller_correlation():
    assert (describe_rows("big", 10_000)["smallest_detectable_r"]
            < describe_rows("small", 200)["smallest_detectable_r"])
