"""Tests for the strand-convention checks."""

from __future__ import annotations

import numpy as np
import pytest

from seqmaestro.qc import (
    AS_GIVEN,
    INCONCLUSIVE,
    NO_MATCH,
    NOT_ORIENTED,
    ORIENTED,
    REVERSE_COMPLEMENT,
    classify_against_reference,
    orientation_by_group,
    orientation_evidence,
    positional_profile,
    reverse_complement,
    reverse_complement_profile,
)

ALPHABET = ("A", "C", "G", "T")


def motif_sequences(n: int, seed: int, length: int = 41) -> list[str]:
    """Random sequences with a motif off their centre, so their positional profile is asymmetric."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = list(rng.choice(list("ACGT"), size=length))
        s[length // 2 : length // 2 + 4] = list("GTAC")  # a palindrome (its own reverse complement)
        out.append("".join(s))
    return out


def random_sequences(n: int, seed: int, length: int = 41) -> list[str]:
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=length)) for _ in range(n)]


# --- reverse complement ------------------------------------------------------------------


def test_reverse_complement_basic():
    assert reverse_complement("AACGT") == "ACGTT"
    assert reverse_complement(reverse_complement("GATTACA")) == "GATTACA"


def test_reverse_complement_leaves_unknown_symbols_alone():
    # N and gaps have no complement. They pass through unchanged instead of raising or being dropped,
    # because one ambiguous base should not make the whole record unusable.
    assert reverse_complement("ANG") == "CNT"
    assert len(reverse_complement("AN-GT")) == 5


# --- profiles ----------------------------------------------------------------------------


def test_positional_profile_shape_and_normalisation():
    profile, alphabet = positional_profile(["ACGT", "ACGT", "TGCA"], ALPHABET)
    assert profile.shape == (4, 4)
    assert alphabet == ALPHABET
    np.testing.assert_allclose(profile.sum(axis=1), 1.0)
    assert profile[0, ALPHABET.index("A")] == pytest.approx(2 / 3)


def test_positional_profile_learns_the_alphabet():
    _, alphabet = positional_profile(["ACAC", "CACA"])
    assert alphabet == ("A", "C")          # only the symbols present


def test_positional_profile_rejects_ragged_and_empty():
    with pytest.raises(ValueError, match="equal length"):
        positional_profile(["ACGT", "ACG"])
    with pytest.raises(ValueError, match="No sequences"):
        positional_profile([])


def test_reverse_complement_profile_matches_actually_reverse_complementing():
    """The orientation check depends on this identity, so it is tested directly."""
    seqs = motif_sequences(50, seed=1)
    predicted = reverse_complement_profile(positional_profile(seqs, ALPHABET)[0], ALPHABET)
    actual, _ = positional_profile([reverse_complement(s) for s in seqs], ALPHABET)
    np.testing.assert_allclose(predicted, actual)


def test_reverse_complement_profile_handles_ambiguity_codes():
    """The identity holds for an alphabet with N, which is its own complement.

    Real assemblies contain N, and a full dataset can have it in a handful of rows out of hundreds
    of thousands while a subsample has none. Code checked only on a subsample could then look
    correct and fail on the full dataset.
    """
    alphabet = ("A", "C", "G", "N", "T")
    seqs = [s[:20] + "N" + s[21:] for s in motif_sequences(50, seed=13)]
    predicted = reverse_complement_profile(positional_profile(seqs, alphabet)[0], alphabet)
    actual, _ = positional_profile([reverse_complement(s) for s in seqs], alphabet)
    np.testing.assert_allclose(predicted, actual)


def test_orientation_check_survives_ambiguity_codes():
    seqs = [s[:5] + "N" + s[6:] for s in motif_sequences(200, seed=14)]
    assert orientation_evidence(seqs, ["+"] * 100 + ["-"] * 100).verdict == ORIENTED


def test_reverse_complement_profile_rejects_an_alphabet_that_cannot_express_the_flip():
    # A complements to T, which is not in the alphabet, so there is no column to map it onto.
    with pytest.raises(KeyError, match="absent from the alphabet"):
        reverse_complement_profile(np.zeros((4, 2)), ("A", "C"))


# --- the verdict -------------------------------------------------------------------------


def test_detects_data_that_is_already_oriented():
    # Both strands are in the same reading direction, so their profiles agree without a flip.
    seqs = motif_sequences(200, seed=2)
    strands = ["+"] * 100 + ["-"] * 100
    evidence = orientation_evidence(seqs, strands)
    assert evidence.verdict == ORIENTED
    assert evidence.separation > 0
    assert "already reverse-complemented" in evidence.explain()


def test_detects_data_that_is_not_oriented():
    # The reverse-strand records are still written as genomic forward-strand sequence.
    forward = motif_sequences(100, seed=3)
    reverse = [reverse_complement(s) for s in motif_sequences(100, seed=4)]
    evidence = orientation_evidence(forward + reverse, ["+"] * 100 + ["-"] * 100)
    assert evidence.verdict == NOT_ORIENTED
    assert evidence.separation < 0
    assert "need reverse complementing" in evidence.explain()


def test_structureless_data_is_inconclusive_rather_than_guessed():
    # Random sequences have a nearly palindromic profile, which fits both hypotheses, so a confident
    # verdict here would be a guess.
    seqs = random_sequences(400, seed=5)
    evidence = orientation_evidence(seqs, ["+"] * 200 + ["-"] * 200)
    assert evidence.verdict == INCONCLUSIVE
    assert "palindromic" in evidence.explain() or "within" in evidence.explain()


def test_counts_are_reported():
    seqs = motif_sequences(30, seed=6)
    evidence = orientation_evidence(seqs, ["+"] * 20 + ["-"] * 10)
    assert (evidence.n_forward, evidence.n_reverse) == (20, 10)


def test_requires_records_on_both_strands():
    with pytest.raises(ValueError, match="both strands"):
        orientation_evidence(motif_sequences(10, seed=7), ["+"] * 10)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="sequences but"):
        orientation_evidence(["ACGT", "ACGT"], ["+"])


def test_custom_strand_labels():
    seqs = motif_sequences(100, seed=8)
    evidence = orientation_evidence(
        seqs, ["fwd"] * 50 + ["rev"] * 50, forward="fwd", reverse="rev"
    )
    assert evidence.verdict == ORIENTED


# --- grouping ----------------------------------------------------------------------------


def test_grouping_rescues_an_answer_that_pooling_dilutes():
    """One class has the positional signal and the other is noise.

    When all rows are pooled, the structureless class dilutes the structured one, whereas the
    structured class checked on its own gives a verdict.
    """
    positives = motif_sequences(200, seed=9)
    negatives = random_sequences(600, seed=10)
    sequences = positives + negatives
    strands = (["+"] * 100 + ["-"] * 100) + (["+"] * 300 + ["-"] * 300)
    groups = [1] * 200 + [0] * 600

    by_group = orientation_by_group(sequences, strands, groups)

    assert by_group[1].verdict == ORIENTED           # the structured class gives a verdict
    assert by_group[0].verdict == INCONCLUSIVE       # the noise class cannot
    assert None in by_group                          # the pooled result is reported too
    assert by_group[1].separation > by_group[0].separation


def test_grouping_skips_groups_it_cannot_evaluate():
    # A group present on only one strand is omitted without stopping the check of the others.
    sequences = motif_sequences(120, seed=11)
    strands = ["+"] * 50 + ["-"] * 50 + ["+"] * 20
    groups = ["ok"] * 100 + ["one_strand_only"] * 20
    by_group = orientation_by_group(sequences, strands, groups)
    assert "ok" in by_group
    assert "one_strand_only" not in by_group


def test_grouping_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        orientation_by_group(["ACGT"], ["+"], [1, 2])


# --- against a reference -----------------------------------------------------------------


def test_classify_against_reference_three_outcomes():
    assert classify_against_reference("ACGTT", "ACGTT") == AS_GIVEN
    assert classify_against_reference("ACGTT", "AACGT") == REVERSE_COMPLEMENT
    assert classify_against_reference("ACGTT", "TTTTT") == NO_MATCH


def test_classify_against_reference_ignores_case_by_default():
    # Reference servers return soft-masked (lowercase) sequence for repeat regions. Treating that
    # as a mismatch would silently fail on repetitive loci, which are the ones most worth checking.
    assert classify_against_reference("ACGTT", "acgtt") == AS_GIVEN
    assert classify_against_reference("ACGTT", "acgtt", case_sensitive=True) == NO_MATCH


def test_classify_against_reference_on_a_palindrome():
    # A palindromic window matches both ways and is classified AS_GIVEN, so the caller needs more
    # than one locus to conclude anything.
    assert classify_against_reference("ACGT", "ACGT") == AS_GIVEN


def test_non_dna_alphabet_with_a_custom_complement():
    # The module assumes nothing about DNA: RNA works by passing its complement map.
    rna = {"A": "U", "C": "G", "G": "C", "U": "A"}
    assert reverse_complement("AACGU", rna) == "ACGUU"
    seqs = [s.replace("T", "U") for s in motif_sequences(100, seed=12)]
    evidence = orientation_evidence(seqs, ["+"] * 50 + ["-"] * 50, complement=rna)
    assert evidence.verdict == ORIENTED
