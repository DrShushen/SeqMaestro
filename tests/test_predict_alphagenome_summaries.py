"""AlphaGenome's predictions summarised over each element, tested with stand-ins for its outputs.

The script runs in a separate environment and imports AlphaGenome's client only where it calls it,
so everything that decides a number can be tested here: the scaffold's shuffle, where the element
sits in its window, which bins of a track cover it, and which fold model predicts it. Each test
guards against a failure that would pass silently downstream. A window one base off moves every
summary a little, a slice computed at the wrong resolution summarises the wrong part of the genome,
and a model chosen from the wrong fold predicts a locus it was trained on.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "prepare"))
from predict_alphagenome_summaries import (
    WINDOW,
    assign_models,
    dinucleotide_shuffle,
    element_bins,
    native_window,
    scaffold,
    span_mismatches,
    summarise,
)

RNG = np.random.default_rng(7)


def dinucleotides(sequence: str) -> Counter:
    return Counter(sequence[i:i + 2] for i in range(len(sequence) - 1))


def random_sequence(length: int) -> str:
    return "".join(RNG.choice(list("ACGT"), length))


def test_the_shuffle_keeps_every_dinucleotide_count_and_both_ends():
    for length in (3, 10, 57, 800):
        sequence = random_sequence(length)
        shuffled = dinucleotide_shuffle(sequence, np.random.default_rng(length))
        assert len(shuffled) == length
        assert shuffled[0] == sequence[0] and shuffled[-1] == sequence[-1]
        assert dinucleotides(shuffled) == dinucleotides(sequence)


def test_the_shuffle_changes_the_sequence_and_follows_its_seed():
    sequence = random_sequence(500)
    first = dinucleotide_shuffle(sequence, np.random.default_rng(1))
    assert first != sequence
    assert first == dinucleotide_shuffle(sequence, np.random.default_rng(1))
    assert first != dinucleotide_shuffle(sequence, np.random.default_rng(2))


def test_a_sequence_with_one_letter_or_too_short_to_shuffle_comes_back_unchanged():
    assert dinucleotide_shuffle("AAAAAAAA", np.random.default_rng(0)) == "AAAAAAAA"
    assert dinucleotide_shuffle("AC", np.random.default_rng(0)) == "AC"


def test_the_scaffold_centres_the_element_in_the_window():
    element = random_sequence(2_001)
    built, offset = scaffold(element, np.random.default_rng(3))
    assert len(built) == WINDOW
    assert built[offset:offset + len(element)] == element
    assert offset == (WINDOW - len(element)) // 2


@pytest.mark.parametrize("start,end", [(1_001, 3_000), (1_001, 3_001)])
def test_the_native_window_is_centred_as_alphagenome_resize_centres_it(start, end):
    # One-based inclusive (start, end) is zero-based [start - 1, end). AlphaGenome's
    # Interval.resize rounds an odd width's centre up, then starts the window half its width before.
    first, stop = start - 1, end
    centre = first + (stop - first + 1) // 2
    window_start, offset = native_window(start, end)
    assert window_start == centre - WINDOW // 2
    assert offset == first - window_start


def test_coordinates_spanning_another_length_than_the_sequence_are_found():
    # The same 200 bp element, once one-based with an inclusive end and once zero-based with an
    # exclusive end, as a BED file gives it. The second spans 201 bases.
    elements = pd.DataFrame({"name": ["one_based", "zero_based"], "sequence": ["ACGT" * 50] * 2,
                             "start": [1_001, 1_000], "end": [1_200, 1_200]})
    columns = {"id": "name", "sequence": "sequence", "start": "start", "end": "end"}
    assert span_mismatches(elements, columns) == ["zero_based"]

def test_the_bins_covering_an_element_include_every_partly_covered_bin():
    assert element_bins(offset=6_774, length=2_835, resolution=1, n_bins=WINDOW) == (6_774, 9_609)
    assert element_bins(offset=6_774, length=2_835, resolution=128, n_bins=128) == (52, 76)
    assert element_bins(offset=128, length=128, resolution=128, n_bins=128) == (1, 2)


def test_an_element_outside_the_prediction_is_refused():
    with pytest.raises(ValueError, match="does not lie inside"):
        element_bins(offset=16_300, length=200, resolution=128, n_bins=128)


def track(values: np.ndarray, names: list[str], strands: list[str], resolution: int):
    return SimpleNamespace(values=values, resolution=resolution,
                           metadata=pd.DataFrame({"name": names, "strand": strands}))


def test_each_output_is_summarised_at_its_own_resolution_and_each_strand_kept():
    per_base = np.zeros((WINDOW, 2))
    per_base[1_000:1_100, 0] = 1.0            # inside the element
    per_base[5_000, 1] = 9.0                  # outside it
    coarse = np.arange(WINDOW // 128, dtype=float)[:, None].repeat(2, axis=1)
    output = SimpleNamespace(
        dnase=track(per_base, ["T1", "T2"], [".", "."], 1),
        cage=track(per_base, ["C1", "C1"], ["+", "-"], 1),
        chip_tf=track(coarse, ["F1", "F2"], [".", "."], 128))
    values, tracks = summarise(output, ["DNASE", "CAGE", "CHIP_TF"], offset=1_000, length=1_000)
    assert values["DNASE|T1|.|mean"] == pytest.approx(0.1)
    assert values["DNASE|T1|.|max"] == 1.0
    assert values["DNASE|T2|.|max"] == 0.0
    assert {"CAGE|C1|+|mean", "CAGE|C1|-|mean"} <= set(values)
    # Bins 7 to 15 cover bases 1,000 to 1,999 at 128 bp each.
    assert values["CHIP_TF|F1|.|mean"] == pytest.approx(np.mean(np.arange(7, 16)))
    assert values["CHIP_TF|F1|.|max"] == 15.0
    assert tracks["CAGE"] == [["C1", "+"], ["C1", "-"]]


def test_a_prediction_of_another_width_is_refused():
    output = SimpleNamespace(dnase=track(np.zeros((WINDOW // 2, 1)), ["T"], ["."], 1))
    with pytest.raises(ValueError, match="which is not"):
        summarise(output, ["DNASE"], offset=0, length=10)


def test_each_element_is_predicted_by_the_model_that_held_out_its_region():
    held = {"fold0": "FOLD_0", "fold1": "FOLD_0", "fold3": "FOLD_1", "fold4": "FOLD_1"}
    folds = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr3", "chr3"],
                          "start": [0, 1_000_000, 0, 0, 30_000],
                          "end": [200_000, 1_200_000, 200_000, 30_000, 200_000],
                          "fold": ["fold0", "fold3", "fold4", "fold0", "fold1"]})
    elements = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr5", "chr3"],
                             "window_start": [50_000, 1_050_000, 50_000, 50_000, 20_000]})
    models, met = assign_models(elements, "chrom", folds, held)
    # The last window meets two folds, both held out of FOLD_0.
    assert models == ["FOLD_0", "FOLD_1", "FOLD_1", "FOLD_0", "FOLD_0"]
    assert met == ["fold0", "fold3", "fold4", "", "fold0,fold1"]


def test_a_window_meeting_folds_no_single_model_held_out_is_refused():
    held = {"fold0": "FOLD_0", "fold3": "FOLD_1"}
    folds = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 30_000],
                          "end": [30_000, 200_000], "fold": ["fold0", "fold3"]})
    elements = pd.DataFrame({"chrom": ["chr1"], "window_start": [20_000]})
    with pytest.raises(ValueError, match="no single model held out"):
        assign_models(elements, "chrom", folds, held)
