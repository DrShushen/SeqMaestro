"""Effective sample size, which is the number a weighted design has to be judged on.

A stratum-balancing design can be worth a small fraction of its row count, so reporting it at its
row count overstates it, and nothing else in the code would show that.
"""

from __future__ import annotations

import numpy as np
import pytest

from seqmaestro.harness.power import contrast_effective_n, effective_sample_size


def test_equal_weights_are_worth_their_row_count():
    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    assert effective_sample_size(np.full(100, 7.5)) == pytest.approx(100.0)


def test_scale_does_not_matter_only_inequality_does():
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    assert effective_sample_size(weights * 1000) == pytest.approx(effective_sample_size(weights))


def test_unweighted_falls_back_to_the_row_count():
    assert effective_sample_size(None, n=42) == 42.0
    with pytest.raises(ValueError, match="either weights or n"):
        effective_sample_size(None)


def test_zero_total_weight_is_worth_nothing():
    assert effective_sample_size(np.zeros(10)) == 0.0


def test_one_dominant_row_is_worth_about_one_row():
    # One row has almost all the weight, so the design is worth about one row, not 1000.
    weights = np.concatenate([[1.0], np.full(999, 1e-6)])
    assert effective_sample_size(weights) < 2.0


def test_the_stratum_balancing_case_this_was_written_for():
    # One stratum, 5 minority against 500 majority, weighted 1/n_class: 505 rows, worth about 10.
    weights = np.concatenate([np.full(5, 1 / 5), np.full(500, 1 / 500)])
    y = np.concatenate([np.ones(5), np.zeros(500)])
    assert len(weights) == 505
    assert contrast_effective_n(weights, y) == pytest.approx(9.9, abs=0.1)


def test_matching_the_same_stratum_beats_weighting_it():
    """Within one stratum, weighting by 1/n_class keeps every row of each class, so its contrast
    effective n (about 10) is above that of matching the stratum down to 5 against 5 (exactly 5)."""
    weighted_w = np.concatenate([np.full(5, 1 / 5), np.full(500, 1 / 500)])
    weighted_y = np.concatenate([np.ones(5), np.zeros(500)])
    matched_y = np.concatenate([np.ones(5), np.zeros(5)])
    assert contrast_effective_n(None, matched_y) == pytest.approx(5.0)
    assert contrast_effective_n(weighted_w, weighted_y) > contrast_effective_n(None, matched_y)


def test_a_contrast_is_limited_by_its_smaller_side():
    y = np.concatenate([np.ones(10), np.zeros(10_000)])
    # The classes combine by harmonic mean, so majority rows beyond a point add almost nothing.
    assert contrast_effective_n(None, y) == pytest.approx(19.96, abs=0.1)


def test_contrast_needs_exactly_two_classes():
    for y in (np.ones(10), np.array([0, 1, 2] * 4)):
        with pytest.raises(ValueError, match="two classes"):
            contrast_effective_n(None, y)


def test_a_class_with_no_weight_leaves_nothing_to_contrast():
    y = np.concatenate([np.ones(5), np.zeros(5)])
    assert contrast_effective_n(np.concatenate([np.ones(5), np.zeros(5)]), y) == 0.0


def test_mantel_haenszel_beats_equal_stratum_weighting():
    """Why `mh` is the default stratum weighting, shown on two strata, one informative and one not.

    `equal` gives the 1-vs-1 stratum the same say as the 200-vs-200 one, whereas `mh` weights each
    stratum by the harmonic mean of its class counts, so the informative stratum outweighs the
    other.
    """
    sizes = [(200, 200), (1, 1)]
    for scheme in ("equal", "mh"):
        weights, labels = [], []
        for npos, nneg in sizes:
            scale = 1.0 if scheme == "equal" else 2 / (1 / npos + 1 / nneg)
            weights += [scale / npos] * npos + [scale / nneg] * nneg
            labels += [1] * npos + [0] * nneg
        effective = contrast_effective_n(np.array(weights), np.array(labels))
        if scheme == "equal":
            equal_n = effective
        else:
            assert effective > equal_n
            assert effective == pytest.approx(201.0, abs=1.0)   # nearly all 402 rows
    assert equal_n == pytest.approx(4.0, abs=0.5)               # the 1-vs-1 stratum dominates
