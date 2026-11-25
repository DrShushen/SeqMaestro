"""The featurisers of `seqmaestro.features` and the pools built from them.

These tests concentrate on generality, because that is where this code is most likely to be wrong
without anyone noticing. A dataset's sequences may all have one length and the alphabet ACGT, and
an implementation that assumed either would pass every test written against that data and then
break on the next dataset. The length, the alphabet and the count dtype are therefore each tested
away from any one dataset's values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from seqmaestro.features import (
    COVARIATES,
    FIXED_POOLS,
    FeatureRecord,
    KmerFeaturiser,
    PositionalFeaturiser,
    make_featuriser,
    parse_pool,
)
from seqmaestro.features.base import Inputs, count_dtype
from seqmaestro.features.positional import PAD, PositionalWindow, window_symbols
from tests.small_motifs import pool_settings

DNA = pd.Series(["ACGT", "TGCA", "GGCC", "AAAA"])
#: The labels the pools are fitted against. Of the sources, only discovery uses them, and it needs
#: two rows of each class.
LABELS = np.array([1, 0, 1, 0])

#: The pools computed from the sequence alone. Those with the covariates are tested in
#: `test_covariates.py`, on rows that have some.
SEQUENCE_POOLS = [pool for pool in FIXED_POOLS if COVARIATES not in parse_pool(pool)]


# ------------------------------------------------------------------------------- positional


def test_positional_columns_are_length_times_alphabet():
    featuriser = PositionalFeaturiser().fit(DNA)

    assert len(featuriser.records) == 4 * 4
    assert featuriser.feature_names[:4] == ["pos_0_A", "pos_0_C", "pos_0_G", "pos_0_T"]


def test_positional_encodes_the_right_cell():
    X = PositionalFeaturiser().fit_transform(DNA)

    assert X.loc[0, "pos_0_A"] == 1
    assert X.loc[0, "pos_0_C"] == 0
    assert X.loc[1, "pos_0_T"] == 1
    # Exactly one symbol per position, so each row sums to the sequence length.
    assert (X.sum(axis=1) == 4).all()


def test_positional_refuses_ragged_input():
    """Position 37 means nothing across sequences of different lengths, so the featuriser refuses
    instead of guessing."""
    with pytest.raises(ValueError, match="same length"):
        PositionalFeaturiser().fit(pd.Series(["ACGT", "ACG"]))


@pytest.mark.parametrize(("anchor", "expected"), [
    ("start", {"ACGTAC": list("ACGT"), "AC": ["A", "C", PAD, PAD]}),
    ("end", {"ACGTAC": list("GTAC"), "AC": [PAD, PAD, "A", "C"]}),
])
def test_the_window_cuts_at_the_far_end_and_pads_there(anchor, expected):
    for sequence, symbols in expected.items():
        assert window_symbols(sequence, 4, anchor, pad=True) == symbols


def test_a_window_without_padding_refuses_a_shorter_sequence():
    with pytest.raises(ValueError, match="declared without padding"):
        window_symbols("AC", 4, "start", pad=False)
    assert window_symbols("ACGTT", 4, "end", pad=False) == list("CGTT")


def test_a_padded_window_reads_ragged_sequences_with_a_pad_column_per_position():
    rows = pd.Series(["ACGTAC", "AC", "GGGG"])
    featuriser = PositionalFeaturiser(window=PositionalWindow(length=4, anchor="start", pad=True))
    X = featuriser.fit_transform(rows)
    assert featuriser.feature_names[:5] == ["pos_0_A", "pos_0_C", "pos_0_G", "pos_0_T", "pos_0_pad"]
    assert X.shape == (3, 4 * 5)
    assert X.loc[1, ["pos_2_pad", "pos_3_pad"]].tolist() == [1, 1] and X.loc[1, "pos_1_C"] == 1
    assert (X.sum(axis=1) == 4).all(), "every window position holds exactly one symbol or the pad"
    pad = next(record for record in featuriser.records if record.name == "pos_2_pad")
    assert pad.params == {"position": 2, "symbol": "pad"}
    assert "shorter than 3 bases" in pad.description
    assert featuriser.describe()["window"] == {"length": 4, "anchor": "start", "pad": True}


def test_a_window_read_from_the_end_counts_its_positions_up_to_the_last_base():
    featuriser = PositionalFeaturiser(window=PositionalWindow(length=3, anchor="end", pad=False))
    X = featuriser.fit_transform(pd.Series(["AACGT", "TTTGA"]))
    assert X.loc[0, ["pos_0_C", "pos_1_G", "pos_2_T"]].tolist() == [1, 1, 1]
    assert "pos_0_pad" not in X.columns, "a window without padding has no pad columns"
    with pytest.raises(ValueError, match="declared without padding"):
        featuriser.transform(pd.Series(["GA"]))


def test_without_a_window_nothing_changes():
    """Without a window, each record has the plain description, and the featuriser's description
    of itself names no window."""
    featuriser = PositionalFeaturiser().fit(DNA)
    assert featuriser.records[0].description == "1 when position 0 (0-based) of the sequence is A, else 0"
    assert "window" not in featuriser.describe()


def test_positional_refuses_a_length_change_after_fitting():
    featuriser = PositionalFeaturiser().fit(DNA)

    with pytest.raises(ValueError, match="Fitted on length 4"):
        featuriser.transform(pd.Series(["ACGTA"]))


# ------------------------------------------------------------------------------------ kmer


def test_kmer_vocabulary_is_learned_not_enumerated():
    """Four sequences cannot contain all 4^3 possible 3-mers, and the unseen ones are not columns."""
    featuriser = KmerFeaturiser(k_range=(3, 3), normalise="counts").fit(DNA)

    assert len(featuriser.vocabulary) < 4**3
    assert all(len(kmer) == 3 for kmer in featuriser.vocabulary)


def test_kmer_counts_overlapping_occurrences():
    X = KmerFeaturiser(k_range=(2, 2), normalise="counts").fit_transform(pd.Series(["AAAA"]))

    assert X.loc[0, "kmer_AA"] == 3, "AAAA contains AA three times when overlaps count"


def test_kmer_is_shift_invariant_where_positional_is_not():
    """This difference is why the two sources complement each other."""
    shifted = pd.Series(["ACGTAA", "AACGTA"])
    kmer = KmerFeaturiser(k_range=(4, 4), normalise="counts").fit_transform(shifted)
    positional = PositionalFeaturiser().fit_transform(shifted)

    assert kmer.loc[0, "kmer_ACGT"] == kmer.loc[1, "kmer_ACGT"] == 1
    assert not positional.loc[0].equals(positional.loc[1])


def test_kmer_range_is_inclusive_at_both_ends():
    featuriser = KmerFeaturiser(k_range=(1, 3), normalise="counts").fit(DNA)

    assert {len(kmer) for kmer in featuriser.vocabulary} == {1, 2, 3}


def test_kmer_handles_variable_length_sequences():
    """Unlike positional features, k-mer counts do not require a fixed length, and must not come to
    require one."""
    X = KmerFeaturiser(k_range=(1, 2), normalise="counts").fit_transform(pd.Series(["ACGT", "AC", "ACGTACGT"]))

    assert X.loc[0, "kmer_A"] == 1
    assert X.loc[2, "kmer_A"] == 2


def test_kmer_refuses_a_k_no_sequence_can_contain():
    with pytest.raises(ValueError, match="longest training sequence"):
        KmerFeaturiser(k_range=(1, 10), normalise="counts").fit(DNA)


def test_a_sequence_shorter_than_some_k_simply_holds_none_of_those_kmers():
    featuriser = KmerFeaturiser(k_range=(1, 3), normalise="counts")
    X = featuriser.fit_transform(pd.Series(["ACGTACGT", "AC"]))
    assert X.loc[1, "kmer_AC"] == 1 and X.loc[1, [c for c in X.columns if len(c) == 8]].sum() == 0
    assert featuriser.describe()["n_shorter_than_k"] == 1


def test_counts_per_length_are_counts_over_the_kmer_positions():
    rows = pd.Series(["AAAA", "AAAAAAAA", "C"])
    featuriser = KmerFeaturiser(k_range=(1, 2), normalise="per_length")
    X = featuriser.fit_transform(rows)
    assert X["kmer_A"].tolist() == [1.0, 1.0, 0.0]
    assert X["kmer_AA"].tolist() == [3 / 3, 7 / 7, 0.0], "one C holds no 2-mer, so no rate"
    record = next(record for record in featuriser.records if record.name == "kmer_AA")
    assert record.params == {"kmer": "AA", "k": 2, "normalise": "per_length"}
    assert featuriser.describe()["normalise"] == "per_length"


def test_counts_are_the_default_and_add_nothing_to_the_description():
    featuriser = KmerFeaturiser(k_range=(1, 2), normalise="counts").fit(DNA)
    assert "normalise" not in featuriser.describe()
    assert all("normalise" not in record.params for record in featuriser.records)
    with pytest.raises(ValueError, match="normalise must be one of"):
        KmerFeaturiser(k_range=(1, 2), normalise="per_base")


@pytest.mark.parametrize("k_range", [(0, 3), (5, 2)])
def test_kmer_rejects_a_nonsense_range(k_range):
    with pytest.raises(ValueError):
        KmerFeaturiser(k_range=k_range, normalise="counts")


# -------------------------------------------------- generality: nothing is pinned to one dataset


def test_the_alphabet_comes_from_the_data_not_from_acgt():
    """Protein sequences, an RNA alphabet and ambiguity codes all work, because the featurisers do
    not special-case any alphabet."""
    protein = pd.Series(["MKVL", "LVKM"])
    featuriser = PositionalFeaturiser().fit(protein)

    assert featuriser.alphabet == ["K", "L", "M", "V"]
    assert "pos_0_M" in featuriser.feature_names


def test_an_added_symbol_widens_the_matrix_rather_than_being_dropped():
    without_n = PositionalFeaturiser().fit(pd.Series(["ACGT"]))
    with_n = PositionalFeaturiser().fit(pd.Series(["ACGT", "NCGT"]))

    assert len(with_n.records) > len(without_n.records)
    assert "pos_0_N" in with_n.feature_names


@pytest.mark.parametrize(
    "featuriser", [PositionalFeaturiser(), KmerFeaturiser(k_range=(1, 2), normalise="counts")],
    ids=["positional", "kmer"]
)
def test_a_symbol_unseen_at_fit_is_refused_rather_than_encoded_as_absence(featuriser):
    """Encoding an unseen symbol as all zeros would make it indistinguishable from a real absence,
    and nothing would report it, so it must raise."""
    featuriser.fit(pd.Series(["ACGT", "TGCA"]))

    with pytest.raises(ValueError, match="absent when it was fitted"):
        featuriser.transform(pd.Series(["ACGN"]))


def test_a_symbol_that_cannot_be_named_is_refused_not_transliterated():
    """The right replacement depends on what the symbol means, so inventing one would be both
    arbitrary and invisible. DEFT refuses illegal column names for the same reason."""
    with pytest.raises(ValueError, match="cannot appear in a Python identifier"):
        PositionalFeaturiser().fit(pd.Series(["AC-T", "AC-A"]))


def test_every_feature_name_is_a_legal_python_identifier():
    for mode in SEQUENCE_POOLS:
        names = make_featuriser(mode, **pool_settings(mode)).fit(DNA, LABELS).feature_names
        assert all(name.isidentifier() for name in names), mode


def test_a_record_with_an_illegal_name_cannot_be_constructed():
    with pytest.raises(ValueError, match="not a legal Python identifier"):
        FeatureRecord(name="kmer-AC", description="d", kind="kmer")


# ------------------------------------------------------------- generality: the count dtype


@pytest.mark.parametrize(
    "bound,expected",
    [(1, np.uint8), (255, np.uint8), (256, np.uint16), (65_535, np.uint16), (65_536, np.uint32)],
)
def test_the_count_dtype_is_derived_from_the_bound(bound, expected):
    assert count_dtype(bound) == expected


def test_a_longer_sequence_widens_the_dtype_instead_of_overflowing():
    """The dtype is computed per transform instead of fixed at fit, because the counts for a
    validation sequence longer than any in training must not overflow and wrap around."""
    featuriser = KmerFeaturiser(k_range=(1, 2), normalise="counts").fit(pd.Series(["ACGT" * 10]))

    short = featuriser.transform(pd.Series(["ACGT" * 10]))
    long = featuriser.transform(pd.Series(["ACGT" * 200]))

    assert short["kmer_A"].dtype == np.uint8
    assert long["kmer_A"].dtype == np.uint16
    assert long.loc[0, "kmer_A"] == 200, "800 nt of ACGT repeats contains A exactly 200 times"


# ------------------------------------------------------------- composing sources into a pool


def test_a_pool_of_two_sources_is_the_two_concatenated():
    positional = make_featuriser("positional").fit(DNA)
    kmer = make_featuriser("kmer", k_range=(1, 2), kmer_normalise="counts").fit(DNA)
    both = make_featuriser("positional+kmer", k_range=(1, 2), kmer_normalise="counts").fit(DNA)

    assert len(both.records) == len(positional.records) + len(kmer.records)
    assert both.feature_names == positional.feature_names + kmer.feature_names


def test_composition_stays_flat():
    """`a + b + c` describes three parts instead of a nested pair and a third."""
    combined = (PositionalFeaturiser() + KmerFeaturiser(k_range=(1, 1), normalise="counts")
                + KmerFeaturiser(k_range=(2, 2), normalise="counts"))

    assert len(combined.describe()["parts"]) == 3


def test_composition_does_not_silently_widen_to_float64():
    X = make_featuriser("positional+kmer", k_range=(1, 2), kmer_normalise="counts").fit_transform(DNA)

    assert X.to_numpy().dtype.kind == "u", "counts and indicators are unsigned integers"


def test_an_unknown_pool_member_is_named_in_the_error():
    with pytest.raises(ValueError, match="onehot"):
        make_featuriser("onehot")


def test_a_pool_with_kmers_is_refused_until_it_is_told_how_they_are_counted():
    """How k-mers are counted is a setting of a run, so the featuriser does not fill it in."""
    for given in ({}, {"k_range": (1, 2)}, {"kmer_normalise": "counts"}):
        with pytest.raises(ValueError, match="must say how they are counted"):
            make_featuriser("positional+kmer", **given)
    assert make_featuriser("positional").fit(DNA).is_fitted, "a pool without them needs neither"


# ------------------------------------------------------------------------ the fit/transform line


@pytest.mark.parametrize("mode", SEQUENCE_POOLS)
def test_transform_before_fit_is_refused(mode):
    with pytest.raises(RuntimeError, match="not fitted"):
        make_featuriser(mode, **pool_settings(mode)).transform(DNA)


@pytest.mark.parametrize("mode", SEQUENCE_POOLS)
def test_the_vocabulary_comes_only_from_what_fit_saw(mode):
    """A featuriser that learned from anything but the training split would leak information from
    the other splits, so the columns after transform must be exactly those learned at fit."""
    featuriser = make_featuriser(mode, **pool_settings(mode)).fit(DNA, LABELS)
    transformed = featuriser.transform(DNA.iloc[:2])

    assert list(transformed.columns) == featuriser.feature_names


@pytest.mark.parametrize("mode", SEQUENCE_POOLS)
def test_the_index_survives_transform(mode):
    """Rows must stay joinable to their labels."""
    sequences = pd.Series(["ACGT", "TGCA", "GGCC", "AAAA"], index=[17, 42, 5, 9])
    X = make_featuriser(mode, **pool_settings(mode)).fit_transform(sequences, LABELS)

    assert list(X.index) == [17, 42, 5, 9]


@pytest.mark.parametrize("mode", SEQUENCE_POOLS)
def test_a_pools_featuriser_describes_itself_for_the_cells_record(mode):
    described = make_featuriser(mode, **pool_settings(mode, (1, 3))).fit(DNA, LABELS).describe()

    assert described["kind"] in {"positional", "kmer", "motif_known", "motif_discovered", "composed"}
    assert "3" in str(described) or "kmer" not in parse_pool(mode)


# --------------------------------------------------------------------------------- input guards


def test_a_frame_is_refused_rather_than_guessed_at():
    """Which columns of a frame are the sequences and the covariates is recorded in `dataset.json`
    and read by `Columns.inputs`, never guessed by a featuriser."""
    with pytest.raises(TypeError, match="not the frame"):
        PositionalFeaturiser().fit(pd.DataFrame({"window": ["ACGT"]}))


def test_a_missing_sequence_is_refused():
    with pytest.raises(ValueError, match="missing"):
        PositionalFeaturiser().fit(pd.Series(["ACGT", None]))


def test_a_non_string_sequence_is_refused():
    with pytest.raises(TypeError, match="must be strings"):
        PositionalFeaturiser().fit(pd.Series(["ACGT", 42]))


def test_no_sequences_at_all_is_refused():
    with pytest.raises(ValueError, match="No sequences"):
        PositionalFeaturiser().fit(pd.Series([], dtype=object))


def test_a_plain_list_is_accepted():
    X = PositionalFeaturiser().fit_transform(["ACGT", "TGCA"])

    assert len(X) == 2


# ------------------------------------------------------------------------------------- Inputs


def rows_with(**covariates) -> Inputs:
    """`DNA` with the given covariate columns beside it."""
    return Inputs(DNA, pd.DataFrame(covariates, index=DNA.index))


def test_bare_sequences_are_rows_with_no_covariates():
    rows = Inputs.from_sequences(DNA)
    assert rows.covariate_columns == () and len(rows) == 4
    assert rows.sequences.tolist() == DNA.tolist()


def test_the_sequence_featurisers_read_the_same_rows_with_or_without_covariates():
    """A covariate never enters a positional, k-mer or composition feature."""
    rows = rows_with(signal=[0.1, 0.2, 0.3, 0.4])
    for pool in SEQUENCE_POOLS:
        bare = make_featuriser(pool, **pool_settings(pool)).fit_transform(DNA, LABELS)
        again = make_featuriser(pool, **pool_settings(pool)).fit_transform(rows, LABELS)
        pd.testing.assert_frame_equal(again, bare)


def test_covariates_are_held_as_floats_in_the_order_given():
    rows = rows_with(peaks=[0, 1, 2, 3], open=[True, False, True, False])
    assert rows.covariate_columns == ("peaks", "open")
    assert rows.covariates.dtypes.tolist() == [np.float64, np.float64]
    assert rows.covariates["open"].tolist() == [1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("covariates, error, message", [
    ({"2nd": [1, 2, 3, 4]}, ValueError, "not legal Python identifiers"),
    ({"label": ["a", "b", "c", "d"]}, TypeError, "not numeric"),
    ({"signal": [1.0, np.nan, 3.0, 4.0]}, ValueError, "missing or infinite"),
    ({"signal": [1.0, np.inf, 3.0, 4.0]}, ValueError, "missing or infinite"),
])
def test_a_covariate_no_featuriser_can_read_is_refused(covariates, error, message):
    with pytest.raises(error, match=message):
        rows_with(**covariates)


def test_covariates_must_line_up_with_the_sequences():
    with pytest.raises(ValueError, match="share one index"):
        Inputs(DNA, pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0]}, index=[5, 6, 7, 8]))


def test_rows_are_taken_with_their_covariates():
    rows = rows_with(signal=[0.1, 0.2, 0.3, 0.4]).take([3, 1])
    assert rows.sequences.tolist() == ["AAAA", "TGCA"]
    assert rows.covariates["signal"].tolist() == [0.4, 0.2]
    assert rows.take(slice(0, 1)).sequences.tolist() == ["AAAA"]


def test_selecting_covariates_keeps_the_named_ones_and_refuses_an_absent_one():
    rows = rows_with(signal=[0.1, 0.2, 0.3, 0.4], peaks=[0, 1, 0, 1])
    assert rows.select(["peaks"]).covariate_columns == ("peaks",)
    assert rows.select([]).covariate_columns == ()
    with pytest.raises(ValueError, match="no covariate depth"):
        rows.select(["depth"])


def test_the_frame_generated_code_is_handed():
    """The frame holds the sequences under the name the code was written against and the covariates
    under their own names, indexed from 0 however the rows were indexed."""
    rows = Inputs(pd.Series(["ACGT", "TTTT"], index=[7, 9]),
                  pd.DataFrame({"signal": [1.5, 2.5]}, index=[7, 9]))
    frame = rows.frame("window")
    assert list(frame.columns) == ["window", "signal"]
    assert frame.index.tolist() == [0, 1]
    assert frame["window"].tolist() == ["ACGT", "TTTT"]
    with pytest.raises(ValueError, match="names a covariate"):
        rows.frame("signal")



def test_the_frame_of_many_covariates_read_from_a_csv_draws_no_fragmentation_warning(tmp_path):
    """A frame read from a CSV holds each column in a separate block, and adding a column to one
    with more than a hundred covariates makes pandas warn on every call. Generated code is handed
    this frame once per feature, and the handover adds the label to it."""
    names = [f"track_{index}" for index in range(150)]
    path = tmp_path / "rows.csv"
    pd.DataFrame({"sequence": ["ACGT", "TTTT"], **{name: [0.5, 1.5] for name in names}}).to_csv(
        path, index=False)
    read = pd.read_csv(path)
    rows = Inputs(read["sequence"], read[names])
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        frame = rows.frame("window").assign(outcome=[0, 1])
    assert list(frame.columns) == ["window", *names, "outcome"]
    assert frame["window"].tolist() == ["ACGT", "TTTT"]

def test_the_index_survives_transform_of_rows_with_covariates():
    rows = Inputs(pd.Series(["ACGT", "TGCA"], index=[10, 20]),
                  pd.DataFrame({"signal": [1.0, 2.0]}, index=[10, 20]))
    assert make_featuriser("positional").fit_transform(rows).index.tolist() == [10, 20]


def test_a_written_feature_named_like_an_enumerated_column_is_kept_under_another_name():
    """A language model shown a pool's columns can name a function after one of them. The two
    compute different things, so the pool keeps both, and the written one records the name it was
    proposed under."""
    from seqmaestro.proposer import GeneratedFeaturiser

    written = FeatureRecord(
        name="kmer_A", description="A among the first two bases", kind="llm_generated",
        source="def kmer_A(X):\n    return X['sequence'].str.slice(0, 2).str.count('A')")
    featuriser = KmerFeaturiser(k_range=(1, 1), normalise="counts") + GeneratedFeaturiser.from_records(
        [written], sequence_column="sequence")
    X = featuriser.fit_transform(pd.Series(["AACA", "CAAA", "CCAA"]))

    assert list(X.columns) == ["kmer_A", "kmer_C", "kmer_A_written"]
    assert X["kmer_A"].tolist() == [3, 3, 2]
    assert X["kmer_A_written"].tolist() == [2, 1, 0]
    assert featuriser.records[-1].extra == {"name_as_proposed": "kmer_A"}
