"""`seqmaestro.harness.features`: the shared feature record on disk.

The most important writer of the record cannot import this module: the DEFT arm runs in a separate
conda environment and writes `features.json` by hand, so several tests below build the payload as
raw JSON. If hand-written JSON stops loading, the boundary is broken, and these tests fail before a
results table is affected.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seqmaestro.features import KmerFeaturiser
from seqmaestro.features.base import ORIGIN_SEQMAESTRO, FeatureRecord
from seqmaestro.harness.features import (
    SCHEMA_VERSION,
    definition_digest,
    load_feature_values,
    load_features,
    rank_by_importance,
    save_feature_values,
    save_features,
)


def a_record(name="kmer_ACGT", **overrides):
    return FeatureRecord(
        **{"name": name, "description": f"description of {name}", "kind": "kmer",
           "params": {"kmer": "ACGT", "k": 4}, **overrides}
    )


def hand_written(tmp_path, features, **header):
    """A payload built as raw JSON, the way an arm in another environment must write it."""
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, **header, "features": features}))
    return path


# ------------------------------------------------------------------------------- round trip


def test_a_record_survives_the_round_trip_unchanged(tmp_path):
    records = [a_record(source="def f(X): ...", rationale="because", origin="deft", importance=0.4,
                        extra={"depth": 2, "threshold": 0.5})]
    save_features(tmp_path / "features.json", records, importance_kind="deft_split_score")

    loaded, header = load_features(tmp_path / "features.json")

    assert loaded == records
    assert header["importance_kind"] == "deft_split_score"


def test_the_header_carries_run_level_facts(tmp_path):
    """What is true of the whole fit, such as its seed, is recorded once in the header instead of
    on every record."""
    save_features(tmp_path / "f.json", [a_record()], importance_kind="l1_coefficient",
                  labels="real", seed=42)

    _, header = load_features(tmp_path / "f.json")

    assert header["labels"] == "real" and header["seed"] == 42
    assert header["schema_version"] == SCHEMA_VERSION


def test_a_featuriser_stamps_its_own_origin():
    records = KmerFeaturiser(k_range=(1, 2), normalise="counts").fit(["ACGT", "TGCA"]).records

    assert {record.origin for record in records} == {ORIGIN_SEQMAESTRO}
    assert {record.kind for record in records} == {"kmer"}


# --------------------------------- the boundary: files written without importing this module


def test_hand_written_json_loads(tmp_path):
    """A payload as the DEFT arm writes it, with only the three required fields."""
    path = hand_written(tmp_path, [{"name": "gc_content_40_60", "description": "GC fraction over "
                                    "positions 40-60", "kind": "llm_generated"}])

    loaded, _ = load_features(path)

    assert loaded[0].name == "gc_content_40_60"
    assert loaded[0].params == {} and loaded[0].extra == {}
    assert loaded[0].importance is None


def test_a_missing_required_field_is_named(tmp_path):
    path = hand_written(tmp_path, [{"name": "f", "kind": "llm_generated"}])

    with pytest.raises(ValueError, match=r"missing \['description'\]"):
        load_features(path)


def test_an_unknown_field_is_refused_rather_than_dropped(tmp_path):
    """Silently discarding it would lose a writer's data and hide the drift that caused it."""
    path = hand_written(tmp_path, [{**json.loads(json.dumps({"name": "f", "description": "d",
                                    "kind": "kmer"})), "depth": 3}])

    with pytest.raises(ValueError, match="unknown field"):
        load_features(path)


def test_a_drifted_schema_version_is_refused(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"schema_version": 99, "features": []}))

    with pytest.raises(ValueError, match="schema version"):
        load_features(path)


def test_a_file_that_is_not_a_feature_file_is_refused(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"metrics": {"auroc": 1.0}}))

    with pytest.raises(ValueError, match="no `features` key"):
        load_features(path)


def test_an_illegal_feature_name_is_still_refused_on_load(tmp_path):
    """The identifier rule holds whoever wrote the file."""
    path = hand_written(tmp_path, [{"name": "gc-content", "description": "d", "kind": "llm_generated"}])

    with pytest.raises(ValueError, match="not a legal Python identifier"):
        load_features(path)


# -------------------------------------------------------------------------------- ranking


def test_rank_is_by_absolute_importance():
    """An L1 coefficient is signed, and a large negative one is as influential as a large positive."""
    records = [a_record("a", importance=0.1), a_record("b", importance=-0.9), a_record("c", importance=0.5)]

    assert rank_by_importance(records) == [3, 1, 2]


def test_records_without_importance_rank_last():
    records = [a_record("a", importance=None), a_record("b", importance=0.2)]

    assert rank_by_importance(records) == [2, 1]


def test_ranking_an_empty_set_is_not_an_error():
    assert rank_by_importance([]) == []


# ----------------------------------------------------------------------- definition digest


def test_two_features_computing_the_same_thing_share_a_digest():
    """Names, formatting, comments and the docstring do not change what a function computes."""
    one = a_record("f_one", source="def f_one(X):\n    return X['a'] + 1\n")
    two = a_record("f_two", source='def f_two(X):\n    """Doc."""\n    # comment\n    return X["a"] + 1\n')

    assert definition_digest(one) == definition_digest(two)


def test_different_computations_get_different_digests():
    one = a_record("f", source="def f(X):\n    return X['a'] + 1\n")
    two = a_record("f", source="def f(X):\n    return X['a'] + 2\n")

    assert definition_digest(one) != definition_digest(two)


def test_a_fixed_vocabulary_feature_falls_back_to_its_recipe():
    """With no source, the recipe `{kind, params}` is the definition, and the name is not part of
    it."""
    same = definition_digest(a_record("kmer_ACGT")), definition_digest(a_record("renamed"))
    different = definition_digest(a_record("kmer_ACGT", params={"kmer": "TTTT", "k": 4}))

    assert same[0] == same[1]
    assert same[0] != different


def test_unparseable_source_falls_back_rather_than_raising():
    digest = definition_digest(a_record(source="def f(X: ->"))

    assert len(digest) == 16


# ------------------------------------------------------------------------- feature values


def test_feature_values_round_trip(tmp_path):
    names, values = ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]])
    save_feature_values(tmp_path / "v.npz", names, values)

    loaded_names, loaded_values = load_feature_values(tmp_path / "v.npz")

    assert loaded_names == names
    assert np.allclose(loaded_values, values)


def test_values_are_stored_as_float32(tmp_path):
    """float32 halves the file size, and its precision is far beyond what a rank correlation across
    seeds can use."""
    path = save_feature_values(tmp_path / "v.npz", ["a"], np.array([[1.0], [2.0]], dtype=np.float64))

    assert load_feature_values(path)[1].dtype == np.float32


def test_a_values_matrix_that_disagrees_with_its_names_is_refused(tmp_path):
    with pytest.raises(ValueError, match="for 3 features"):
        save_feature_values(tmp_path / "v.npz", ["a", "b", "c"], np.zeros((10, 2)))


# ---------------------------------------------------------- direction and marginal effect


def test_direction_and_marginal_effect_round_trip(tmp_path):
    path = save_features(tmp_path / "features.json",
                         [a_record(direction="higher_in_positive", marginal_effect=-0.25)],
                         importance_kind="l1_coefficient", direction_kind="l1_coefficient_sign")
    loaded, header = load_features(path)

    assert loaded[0].direction == "higher_in_positive"
    assert loaded[0].marginal_effect == pytest.approx(-0.25)
    assert header["direction_kind"] == "l1_coefficient_sign"


def test_a_file_without_the_optional_fields_loads_with_them_unset(tmp_path):
    """Only name, description and kind are required, so a `features.json` without `direction` and
    `marginal_effect` (such as one from an adapter that reports no direction) loads with both
    unset. Because both fields are optional, they need no schema version bump."""
    path = hand_written(tmp_path, [{"name": "f", "description": "d", "kind": "kmer"}])
    record, = load_features(path)[0]

    assert record.direction is None
    assert record.marginal_effect is None


def test_a_direction_outside_the_vocabulary_is_refused(tmp_path):
    """The direction word goes into a judge prompt and a report table. A hand-written adapter that
    gets it subtly wrong (writing `positive`, for example) must fail here, before the word reaches
    those files."""
    path = hand_written(tmp_path, [{"name": "f", "description": "d", "kind": "kmer",
                                    "direction": "positive"}])
    with pytest.raises(ValueError, match="not one of"):
        load_features(path)


def test_a_direction_word_of_the_other_kind_of_target_is_refused(tmp_path):
    """A file that states its kind of target holds that kind's words: a binary cell's feature
    cannot be said to raise the target's value."""
    records = [FeatureRecord(name="f", description="d", kind="kmer", direction="increases_target")]
    path = save_features(tmp_path / "features.json", records, target_kind="binary")
    with pytest.raises(ValueError, match="not one of higher_in_positive, higher_in_negative"):
        load_features(path)
    save_features(path, records, target_kind="continuous")
    assert load_features(path)[0][0].direction == "increases_target"


def test_a_provenance_outside_the_vocabulary_is_refused(tmp_path):
    """An adapter declares how each feature entered its arm's answer, and the word is shown beside
    the feature in the recurrence table and the spreadsheet, so any word but `selected` and
    `generated` is refused when the file is loaded."""
    path = hand_written(tmp_path, [{"name": "f", "description": "d", "kind": "kmer",
                                    "provenance": "proposed"}])
    with pytest.raises(ValueError, match="provenance 'proposed'"):
        load_features(path)
