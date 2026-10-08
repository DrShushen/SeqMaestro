"""What DEFT is given, and how (`GenericCsvDatasetPlugin`, `get_naive_features`).

Three things are being pinned down here, and each replaces a silent failure with a loud one.

**The plugin does not split.** It used to, with `test_size` and `random_state` — and the
training bundle has no place for a test set, so the held-out rows were simply *discarded*.
Pointing it at a 3,500-row training file trained on 2,800 while reporting nothing amiss.

**Covariates reach the naive baseline.** `get_naive_features` used to emit positional
nucleotide indicators and nothing else. Handing DEFT covariates while the baseline could not
see them would measure `llm_split_share` against a sequence-only yardstick and flatter the
model.

**The sequence column is the user's own.** It was hard-coded to `raw_sequence`, and the
plugins renamed the user's column to match.

All offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.data.dataset_plugins import check_column_names
from deft.data.generic_csv_plugin import GenericCsvDatasetPlugin
from deft.llm.prompt_builder import serialize_features
from deft.utils.feature import get_naive_features

# --------------------------------------------------------------------------------------- data


def make_csv(tmp_path, frame, name="train.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_path(tmp_path):
    """A small dataset with a sequence, a numeric covariate and a categorical one."""
    frame = pd.DataFrame(
        {
            "raw_sequence": ["ACGT", "TGCA", "AACC", "GGTT", "CGCG", "TATA"],
            "gc_signal": [0.5, 0.5, 0.5, 0.5, 1.0, 0.0],
            "strand": ["+", "-", "+", "-", "+", "-"],
            "paused": [1, 0, 1, 0, 1, 0],
        }
    )
    return make_csv(tmp_path, frame)


def load(csv_path, **kwargs):
    kwargs.setdefault("target_column", "paused")
    return GenericCsvDatasetPlugin(csv_path=csv_path, **kwargs).load_for_deft()


# ------------------------------------------------------------------- the plugin does not split


def test_every_row_is_used_for_training(csv_path):
    """No internal split: 6 rows in, 6 rows out. The caller holds the other splits."""
    bundle = load(csv_path)

    assert len(bundle["X_train"]) == 6
    assert len(bundle["y_train"]) == 6


@pytest.mark.parametrize(
    "parameter,value", [("test_size", 0.2), ("random_state", 7), ("val_size", 0.1)]
)
def test_a_split_parameter_is_refused_rather_than_ignored(csv_path, parameter, value):
    """An old config must fail, not silently train on a subset of the file it was given."""
    with pytest.raises(TypeError) as excinfo:
        GenericCsvDatasetPlugin(csv_path=csv_path, **{parameter: value})

    message = str(excinfo.value)
    assert parameter in message, "the message must name the offending key"
    assert "csv_path" in message, "and say what to do instead"


def test_an_unknown_parameter_is_refused(csv_path):
    with pytest.raises(TypeError, match="nonsense"):
        GenericCsvDatasetPlugin(csv_path=csv_path, nonsense=1)


# ------------------------------------------------------------------------------ column selection


def test_covariates_default_to_everything_but_the_target(csv_path):
    bundle = load(csv_path)

    assert list(bundle["X_train"].columns) == ["raw_sequence", "gc_signal", "strand"]
    assert list(bundle["y_train"].columns) == ["paused"]


def test_the_target_keeps_its_own_name(csv_path):
    """Nothing downstream assumes `label`, so renaming the user's column would only obscure."""
    assert "paused" in load(csv_path)["y_train"].columns


def test_an_allowlist_narrows_the_covariates_and_keeps_the_sequence(csv_path):
    bundle = load(csv_path, covariate_columns=["gc_signal"])

    assert list(bundle["X_train"].columns) == ["raw_sequence", "gc_signal"]


def test_a_blocklist_removes_columns(csv_path):
    bundle = load(csv_path, drop_columns=["strand"])

    assert list(bundle["X_train"].columns) == ["raw_sequence", "gc_signal"]


def test_dropping_the_sequence_column_is_a_contradiction(csv_path):
    with pytest.raises(ValueError, match="sequence_column"):
        load(csv_path, drop_columns=["raw_sequence"])


def test_columns_are_given_in_file_order(csv_path):
    """Order is what the in-context prompt rows follow, so it should not depend on set order."""
    bundle = load(csv_path, covariate_columns=["strand", "gc_signal"])

    assert list(bundle["X_train"].columns) == ["raw_sequence", "gc_signal", "strand"]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"target_column": "absent"}, "absent"),
        ({"sequence_column": "absent"}, "absent"),
        ({"covariate_columns": ["absent"]}, "absent"),
        ({"drop_columns": ["absent"]}, "absent"),
    ],
)
def test_a_column_that_is_not_there_is_named_in_the_error(csv_path, kwargs, expected):
    kwargs.setdefault("target_column", "paused")
    with pytest.raises(KeyError) as excinfo:
        GenericCsvDatasetPlugin(csv_path=csv_path, **kwargs).load_for_deft()

    assert expected in str(excinfo.value)


def test_a_missing_sequence_column_says_how_to_run_without_one(csv_path):
    with pytest.raises(KeyError) as excinfo:
        load(csv_path, sequence_column="absent")

    assert "sequence_column: null" in str(excinfo.value)


# ---------------------------------------------------------------------------- sequence validity


def test_sequences_of_different_lengths_are_refused(tmp_path):
    """Positional features index into the string by offset, so this is a precondition."""
    path = make_csv(
        tmp_path,
        pd.DataFrame({"raw_sequence": ["ACGT", "ACG"], "paused": [1, 0]}),
    )

    with pytest.raises(ValueError) as excinfo:
        load(path)

    assert "length" in str(excinfo.value).lower()


def test_a_missing_sequence_value_is_refused(tmp_path):
    path = make_csv(
        tmp_path,
        pd.DataFrame({"raw_sequence": ["ACGT", None], "paused": [1, 0]}),
    )

    with pytest.raises(ValueError, match="missing"):
        load(path)


def test_an_empty_csv_is_refused(tmp_path):
    path = make_csv(tmp_path, pd.DataFrame({"raw_sequence": [], "paused": []}))

    with pytest.raises(ValueError, match="no rows"):
        load(path)


# ------------------------------------------------------------------------ the sequence column


def test_the_users_own_column_name_survives(tmp_path):
    """No renaming to a canonical `raw_sequence`; the bundle carries the real name instead."""
    path = make_csv(
        tmp_path,
        pd.DataFrame({"dna": ["ACGT", "TGCA"], "paused": [1, 0]}),
    )

    bundle = load(path, sequence_column="dna")

    assert list(bundle["X_train"].columns) == ["dna"]
    assert bundle["sequence_column"] == "dna"


def test_naive_features_follow_the_configured_column(tmp_path):
    X = pd.DataFrame({"dna": ["ACGT", "TGCA"]})

    features = get_naive_features(X, sequence_column="dna")

    assert features, "no positional features were generated"
    assert all("X['dna']" in feature.string for feature in features)


def test_a_named_but_absent_sequence_column_raises(tmp_path):
    """A typo here would otherwise remove the sequence from the baseline without a word."""
    X = pd.DataFrame({"dna": ["ACGT", "TGCA"]})

    with pytest.raises(KeyError, match="raw_sequence"):
        get_naive_features(X, sequence_column="raw_sequence")


def test_the_alphabet_comes_from_every_row_not_just_the_first(tmp_path):
    """Upstream read the alphabet off `iloc[0]`, so a nucleotide absent from the first row
    produced no features anywhere — an `N` in row two was invisible to the baseline."""
    X = pd.DataFrame({"raw_sequence": ["AAAA", "NNNN"]})

    names = {feature.name for feature in get_naive_features(X)}

    assert "pos_0_is_N" in names
    assert "pos_0_is_A" in names


def test_pre_split_position_columns_are_used_when_there_is_no_sequence_column():
    X = pd.DataFrame(
        {"seq_-1": ["A", "C"], "seq_0": ["G", "T"], "seq_1": ["C", "A"]}
    )

    features = get_naive_features(X, sequence_column=None)
    names = {feature.name for feature in features}

    assert "pos_-1_is_A" in names, f"negative positions were not read back: {sorted(names)}"
    assert "pos_0_is_G" in names
    assert len(features) == 3 * 4, "one indicator per position and nucleotide"


def test_a_position_prefix_column_with_a_non_integer_suffix_is_a_covariate():
    """`seq_source` is not position -1; treating it as one would fabricate an offset."""
    X = pd.DataFrame({"seq_0": ["A", "C"], "seq_source": ["x", "y"]})

    names = {feature.name for feature in get_naive_features(X, sequence_column=None)}

    assert "seq_source_is_x" in names
    assert "pos_0_is_A" in names


def test_covariates_alone_are_enough():
    """With no sequence and no position columns, DEFT is a tabular method. That follows from
    making the sequence optional, and is worth stating explicitly."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})

    features = get_naive_features(X, sequence_column=None)

    assert {feature.name for feature in features} == {"value_of_a", "b_is_x", "b_is_y"}


# ------------------------------------------------------------------ naive features: covariates


def _evaluate(feature, X):
    return np.asarray(feature.fn(X.copy())).astype(float).flatten()


def test_a_numeric_covariate_gets_an_identity_feature():
    """Let the threshold search find the cut point — exactly what CART does."""
    X = pd.DataFrame({"raw_sequence": ["AC", "GT", "AA"], "signal": [0.5, 2.5, 9.0]})

    feature = next(f for f in get_naive_features(X) if f.name == "value_of_signal")

    assert list(_evaluate(feature, X)) == [0.5, 2.5, 9.0]


def test_a_boolean_covariate_gets_an_identity_feature():
    X = pd.DataFrame({"raw_sequence": ["AC", "GT"], "flag": [True, False]})

    feature = next(f for f in get_naive_features(X) if f.name == "value_of_flag")

    assert list(_evaluate(feature, X)) == [1.0, 0.0]


def test_a_low_cardinality_categorical_gets_one_indicator_per_level():
    X = pd.DataFrame({"raw_sequence": ["AC", "GT", "AA"], "strand": ["+", "-", "+"]})

    names = {f.name for f in get_naive_features(X) if f.name.startswith("strand_is")}

    assert names == {"strand_is_+", "strand_is_-"}


def test_the_indicator_is_correct():
    X = pd.DataFrame({"raw_sequence": ["AC", "GT", "AA"], "strand": ["+", "-", "+"]})

    feature = next(f for f in get_naive_features(X) if f.name == "strand_is_+")

    assert list(_evaluate(feature, X)) == [1.0, 0.0, 1.0]


def test_a_high_cardinality_categorical_is_skipped_under_the_cap():
    """An identifier would otherwise contribute one indicator per row."""
    X = pd.DataFrame(
        {"raw_sequence": ["AC"] * 6, "gene_id": [f"g{i}" for i in range(6)]}
    )

    names = {f.name for f in get_naive_features(X, max_categorical_levels=3)}

    assert not any(name.startswith("gene_id_is") for name in names)


def test_the_cap_is_configurable():
    X = pd.DataFrame(
        {"raw_sequence": ["AC"] * 6, "gene_id": [f"g{i}" for i in range(6)]}
    )

    names = {f.name for f in get_naive_features(X, max_categorical_levels=10)}

    assert sum(name.startswith("gene_id_is") for name in names) == 6


def test_a_constant_covariate_is_skipped():
    """It can never split anything; generating it only costs a threshold search."""
    X = pd.DataFrame({"raw_sequence": ["AC", "GT"], "sample": ["one", "one"]})

    names = {f.name for f in get_naive_features(X)}

    assert not any(name.startswith("sample") for name in names)


def test_a_covariate_with_missing_values_is_skipped():
    """`optimize_threshold` has no missing-value handling: NaN compares false against every
    threshold, so those rows would silently go right at every split."""
    X = pd.DataFrame({"raw_sequence": ["AC", "GT", "AA"], "signal": [1.0, np.nan, 3.0]})

    names = {f.name for f in get_naive_features(X)}

    assert "value_of_signal" not in names


def test_naive_feature_generation_does_not_itself_depend_on_the_ingest_check():
    """`check_column_names` refuses an illegal column name at load time, so in a real run this
    cannot arise. `get_naive_features` is a library function that may be called directly, so it
    stays robust rather than relying on a check made elsewhere."""
    X = pd.DataFrame({"raw_sequence": ["AC", "GT"], "SIG:NET-seq_COND:+Spt5": [1.0, 2.0]})

    feature = next(
        f for f in get_naive_features(X) if f.name == "value_of_SIG:NET-seq_COND:+Spt5"
    )

    assert list(_evaluate(feature, X)) == [1.0, 2.0]


def test_two_columns_that_sanitise_alike_keep_distinct_functions():
    X = pd.DataFrame({"a:b": [1.0, 2.0], "a-b": [3.0, 4.0]})

    features = get_naive_features(X, sequence_column=None)

    assert len(features) == 2
    assert len(set(features)) == 2, "deduplication must not collapse them"
    assert {tuple(_evaluate(f, X)) for f in features} == {(1.0, 2.0), (3.0, 4.0)}


def test_the_sequence_column_is_not_also_a_covariate():
    X = pd.DataFrame({"raw_sequence": ["AC", "GT"], "signal": [1.0, 2.0]})

    names = {f.name for f in get_naive_features(X)}

    assert "value_of_raw_sequence" not in names
    assert not any(name.startswith("raw_sequence_is") for name in names)


# --------------------------------------------------------------- column names must be legal


def test_legal_column_names_pass():
    check_column_names(["raw_sequence", "seq_0", "sig_netseq_cond_wt_str_plus_rep_r1", "paused"])


@pytest.mark.parametrize(
    "column",
    [
        "seq_-50",                                    # a coordinate sign
        "dist_gene_end_-_start",                      # an arithmetic minus
        "SIG:NET-seq_COND:+Spt5_STR:+_REP:r2_SIGNAL",  # colons, a plus, a hyphen
        "ChIP-seq H3K4me3",                           # a space
        "2_fold_change",                              # a leading digit
    ],
)
def test_an_illegal_column_name_is_refused(column):
    with pytest.raises(ValueError) as excinfo:
        check_column_names(["raw_sequence", column])

    message = str(excinfo.value)
    assert column in message, "the message must name the offending column"
    assert "Rename" in message, "and say whose job it is to fix"


def test_a_python_keyword_is_allowed():
    """A column called `class` is only ever referenced as a string subscript, so it costs
    nothing. The rule is 'a legal identifier', not 'a usable variable name'."""
    check_column_names(["raw_sequence", "class", "lambda"])


def test_every_offender_is_reported_not_just_the_first():
    """A dataset with 85 bad names should be fixed in one pass, not 85 reruns."""
    with pytest.raises(ValueError) as excinfo:
        check_column_names(["seq_-1", "seq_-2", "seq_-3"])

    message = str(excinfo.value)
    assert "3 column name(s)" in message
    for column in ("seq_-1", "seq_-2", "seq_-3"):
        assert column in message


def test_a_long_list_of_offenders_is_summarised():
    with pytest.raises(ValueError, match="and 42 more"):
        check_column_names([f"bad-{i}" for i in range(50)])


def test_the_check_runs_before_the_language_model_is_reached(tmp_path):
    """Cheap to fix, expensive to discover late — so it fires on the bundle, at load."""
    from deft.cli.run_deft_tree import _load_training_bundle

    path = make_csv(
        tmp_path,
        pd.DataFrame({"raw_sequence": ["ACGT", "TGCA"], "seq_-1": ["A", "T"], "paused": [1, 0]}),
    )
    cfg = {
        "dataset": {
            "plugin": {
                "_target_": "deft.data.generic_csv_plugin.GenericCsvDatasetPlugin",
                "csv_path": path,
                "target_column": "paused",
            }
        }
    }

    with pytest.raises(ValueError, match="seq_-1"):
        _load_training_bundle(cfg)


# ---------------------------------------------------------------------- the prompt description


def test_a_column_mixing_text_and_missing_values_can_be_described():
    """Upstream sorted before dropping NaN, so this raised `TypeError: '<' not supported
    between 'str' and 'float'` — unreachable with one sequence column, immediate with
    covariates."""
    X = pd.DataFrame({"region": ["exon", "intron", None] * 40})

    description = serialize_features(X)

    assert "exon" in description and "intron" in description


def test_a_sparse_numeric_column_is_given_a_range_not_a_list_of_floats():
    """`analyze_column_types` calls any column with under 5% distinct values categorical, which
    a sparse continuous signal satisfies — and enumerating it put ~300 tokens of raw floats
    into every prompt."""
    # 60 distinct values in 2,000 rows is a 3% unique ratio, under the classifier's threshold.
    values = list(np.linspace(0, 1, 60)) + [0.0] * 1940
    X = pd.DataFrame({"signal": values})

    description = serialize_features(X)

    assert "range:" in description
    assert "distinct values" in description
    assert description.count(",") < 5, f"values were enumerated after all: {description}"


def test_a_small_categorical_is_still_enumerated():
    X = pd.DataFrame({"strand": ["+", "-"] * 50})

    description = serialize_features(X)

    assert "values: +, -" in description


def test_a_long_categorical_is_truncated_with_a_count():
    X = pd.DataFrame({"gene": [f"g{i}" for i in range(40)] * 30})

    description = serialize_features(X)

    assert "40 distinct values" in description
    assert "..." in description


# ---------------------------------------------------------------- the positional window


RAGGED = pd.DataFrame({"raw_sequence": ["ACGTAC", "TGCA", "AACCG"], "paused": [1, 0, 1]})


def test_sequences_of_different_lengths_are_read_in_a_declared_window(tmp_path):
    bundle = load(make_csv(tmp_path, RAGGED),
                  positional_window={"length": 4, "anchor": "start", "pad": False})
    assert bundle["positional_window"] == {"length": 4, "anchor": "start", "pad": False}
    assert bundle["X_train"]["raw_sequence"].str.len().tolist() == [6, 4, 5], "not cut on load"


def test_a_sequence_shorter_than_a_window_without_padding_is_refused(tmp_path):
    with pytest.raises(ValueError, match="shorter than the 5-position window"):
        load(make_csv(tmp_path, RAGGED),
             positional_window={"length": 5, "anchor": "end", "pad": False})


@pytest.mark.parametrize("window", [
    {"length": 4, "anchor": "start"},
    {"length": 4, "anchor": "middle", "pad": True},
    {"length": 0, "anchor": "start", "pad": True},
    {"length": 4, "anchor": "start", "pad": "yes"},
])
def test_a_window_must_be_stated_in_full_and_make_sense(tmp_path, window):
    with pytest.raises(ValueError, match="positional window"):
        load(make_csv(tmp_path, RAGGED), positional_window=window)


def test_a_window_needs_a_sequence_column(tmp_path):
    with pytest.raises(ValueError, match="sequence column"):
        load(make_csv(tmp_path, RAGGED), sequence_column=None,
             positional_window={"length": 4, "anchor": "start", "pad": True})


def indicators(X, window):
    features = get_naive_features(X, sequence_column="raw_sequence", positional_window=window)
    return {feature.name: _evaluate(feature, X).tolist() for feature in features}


def test_window_positions_count_from_the_anchor_and_the_padding_is_its_own_feature():
    X = pd.DataFrame({"raw_sequence": ["ACGTAC", "TG"]})
    start = indicators(X, {"length": 4, "anchor": "start", "pad": True})
    assert start["pos_0_is_A"] == [1.0, 0.0] and start["pos_0_is_T"] == [0.0, 1.0]
    assert start["pos_3_is_T"] == [1.0, 0.0], "the longer sequence is cut at the far end"
    assert start["pos_2_is_pad"] == [0.0, 1.0] and start["pos_1_is_pad"] == [0.0, 0.0]

    end = indicators(X, {"length": 4, "anchor": "end", "pad": True})
    assert end["pos_3_is_C"] == [1.0, 0.0] and end["pos_3_is_G"] == [0.0, 1.0]
    assert end["pos_0_is_G"] == [1.0, 0.0] and end["pos_0_is_pad"] == [0.0, 1.0]


def test_a_window_without_padding_has_no_pad_feature_and_says_what_it_reads():
    X = pd.DataFrame({"raw_sequence": ["ACGTAC", "TGCATG"]})
    features = get_naive_features(X, sequence_column="raw_sequence",
                                  positional_window={"length": 3, "anchor": "end", "pad": False})
    assert not any(feature.name.endswith("_is_pad") for feature in features)
    described = {feature.name: feature.description for feature in features}
    assert described["pos_0_is_T"] == ("Check if position 0 of the window of the last 3 positions "
                                       "of `raw_sequence` is T")


# ---------------------------------------------------------------- what each column means


def test_a_described_column_carries_its_description_into_the_prompt(csv_path):
    bundle = load(csv_path, column_descriptions={"gc_signal": "a measured GC signal"})
    assert bundle["column_descriptions"] == {"gc_signal": "a measured GC signal"}
    lines = serialize_features(bundle["X_train"], bundle["column_descriptions"]).splitlines()
    [signal] = [line for line in lines if line.startswith("gc_signal")]
    [strand] = [line for line in lines if line.startswith("strand")]
    assert signal.endswith(": a measured GC signal")
    assert strand == serialize_features(bundle["X_train"]).splitlines()[2], "undescribed: as was"


def test_a_description_of_a_column_deft_is_not_given_is_refused(csv_path):
    with pytest.raises(KeyError, match="not given"):
        load(csv_path, column_descriptions={"no_such_column": "anything"})


def test_a_text_column_of_different_lengths_states_its_range():
    X = pd.DataFrame({"raw_sequence": ["ACGTAC", "TG", "ACG"]})
    assert serialize_features(X) == ("raw_sequence: text (length 2 to 6 characters, varying by "
                                     "row)")


def test_a_fractional_range_is_shown_to_four_significant_figures():
    X = pd.DataFrame({"signal": np.linspace(0.123456789, 9.87654321, 40)})
    assert "(range: 0.1235 to 9.877)" in serialize_features(X)
