"""Co-discovery: how many arms independently found each concept.

These tests cover the counting rules, because a wrong count would not show in the output. The
clustering uses the stability module's behavioural matcher, which `test_stability.py` covers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seqmaestro.pipeline.analyses import co_discovery as CO


def witnesses(arms: set[str]) -> int:
    """How many independent witnesses these arms are, where no two of them shared a proposal."""
    return CO.independent_witnesses(arms, CO.linked_by_fit(arms))


# ---------------------------------------------------------------- definitionally linked arms


def test_linked_arms_count_as_one_witness():
    """Arms that consumed one cached proposal were handed the same candidate features, so their
    agreement is definitional. They match on most features under both real and shuffled labels,
    and counting them as two witnesses would make every proposed feature look corroborated at no
    cost."""
    shared = frozenset({"ebm/llm_generated", "xgboost/llm_generated"})
    assert CO.independent_witnesses(set(shared), [shared]) == 1


def test_two_models_on_one_written_pool_are_two_witnesses():
    """Each arm's proposal is made for that arm, so two models on one written pool were handed
    different features and found what they found apart."""
    assert witnesses({"ebm/llm_generated", "xgboost/llm_generated"}) == 2


def test_a_linked_arm_plus_an_unlinked_one_is_two_witnesses():
    assert witnesses(
        {"deft_or_cart/positional", "deft_or_cart/positional+llm_generated_joint",
         "l1_logistic/positional"}) == 2


def test_unlinked_arms_each_count():
    assert witnesses(
        {"l1_logistic/positional", "ebm/positional", "deft_or_cart/positional"}) == 3


def test_a_single_arm_is_one_witness():
    assert witnesses({"deft_or_cart/positional+llm_generated_joint"}) == 1


def test_the_tree_with_and_without_written_candidates_counts_as_one_witness():
    """CART and DEFT share the tree, the naive candidate pool and, under `permutation_scope=seed`,
    the shuffled labels. On shuffled labels the written candidates win no splits, so DEFT's tree is
    close to a copy of CART's, and the two agree far more under the null than under real labels.
    Counting them as two witnesses would inflate the tree's chance level and push its corrected
    figure strongly negative for a reason unrelated to the data."""
    cart, deft = "deft_or_cart/positional", "deft_or_cart/positional+llm_generated_joint"
    assert witnesses({cart, deft}) == 1
    assert witnesses({cart, deft, "ebm/positional"}) == 2
    assert witnesses({"deft_or_cart/positional+covariates",
                      "deft_or_cart/positional+llm_generated_joint+covariates"}) == 1
    assert witnesses({cart, "deft_or_cart/positional+llm_generated_joint+covariates"}) == 2, \
        "the two trees were handed different columns besides what the second one wrote"


def test_the_arms_linked_by_their_fit_follow_from_the_arms_and_no_list_names_them():
    """The pair is found for any pool the tree is fitted on with and without the source it writes,
    so a pool added to a config needs no entry anywhere."""
    arms = ["deft_or_cart/positional+kmer", "deft_or_cart/positional+kmer+llm_generated_joint",
            "ebm/positional+kmer", "l1_logistic/llm_generated", "l1_logistic/llm_generated_iterative",
            "igsr/llm_generated_joint", "igsr/llm_generated_joint+covariates"]
    assert CO.linked_by_fit(arms) == [frozenset(arms[:2])], \
        "a model whose pool is what it writes and nothing else is linked to no other arm"


def _run_with_proposal_key(root, arm, key):
    cell = root / arm / "real" / "seed_42"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "run.json").write_text(
        json.dumps({"featuriser": {"proposer": {"proposal_key": key}}}))


def test_an_arm_is_named_the_same_way_by_every_reader():
    """Three readers build an arm name from a path, and a group is matched against that name as a
    string.

    A name built from the wrong path components matches nothing, so the group silently never
    applies: the analysis completes, and only the arms the group covers are scored wrongly.
    """
    results = Path("/runs/x/results")

    assert CO.arm_name(results / "ebm" / "positional" / "real" / "seed_42", results) \
        == "ebm/positional"


def test_arms_that_shared_a_proposal_do_not_replace_the_arms_linked_by_their_fit(tmp_path):
    """A group that shared a proposal and a group linked by its fit must both apply.

    Returning the first kind whenever any exists would drop the tree's pair on every run a
    proposer took part in. Nothing reports this: the analysis completes, and only the tree's arms
    are scored wrongly.
    """
    _run_with_proposal_key(tmp_path, "ebm/positional", "shared")
    _run_with_proposal_key(tmp_path, "xgboost/positional", "shared")
    _run_with_proposal_key(tmp_path, "deft_or_cart/positional", None)
    _run_with_proposal_key(tmp_path, "deft_or_cart/positional+llm_generated_joint", None)

    groups = CO.linked_groups(tmp_path, "real", "seed_42")

    assert frozenset({"ebm/positional", "xgboost/positional"}) in groups, (
        "the group read off the run was lost, or its arms were named differently from everywhere else")
    assert frozenset({"deft_or_cart/positional", "deft_or_cart/positional+llm_generated_joint"}) \
        in groups, "the tree's pair was dropped because a shared proposal existed"


def test_a_shared_proposal_is_found_wherever_a_pool_records_it(tmp_path):
    """A pool that holds more than the proposal records its proposer inside one of its parts, and
    the iterative source records the blind proposal inside its pool."""
    described = {
        "ebm/kmer+llm_generated": {"kind": "composed", "parts": [
            {"kind": "kmer"}, {"kind": "llm_generated", "proposer": {"proposal_key": "blind"}}]},
        "ebm/llm_generated_iterative": {
            "kind": "iterative_llm_generated", "proposer": {"proposal_key": "informed"},
            "pool": {"kind": "llm_generated", "proposer": {"proposal_key": "blind"}}},
        "ebm/positional": {"kind": "positional"},
    }
    for arm, featuriser in described.items():
        cell = tmp_path / arm / "real" / "seed_42"
        cell.mkdir(parents=True)
        (cell / "run.json").write_text(json.dumps({"featuriser": featuriser}))

    assert CO.proposal_keys(described["ebm/llm_generated_iterative"]) == {"blind", "informed"}
    assert CO.linked_groups(tmp_path, "real", "seed_42") == [
        frozenset({"ebm/kmer+llm_generated", "ebm/llm_generated_iterative"})]


def test_a_group_is_not_counted_twice_when_both_sources_name_it(tmp_path):
    _run_with_proposal_key(tmp_path, "deft_or_cart/positional", "shared")
    _run_with_proposal_key(tmp_path, "deft_or_cart/positional+llm_generated_joint", "shared")

    groups = CO.linked_groups(tmp_path, "real", "seed_42")

    assert len(groups) == len(set(groups))


# ----------------------------------------------------------------------- input pool counting


def test_arms_handed_the_same_pool_are_one_pool():
    """Three models fitted on the same positional features count as one pool, because they were
    handed the same inputs."""
    pools = {arm: frozenset({"positional"})
             for arm in ("l1_logistic/positional", "ebm/positional", "xgboost/positional")}
    assert CO.distinct_pools(set(pools), pools) == 1


def test_different_pools_are_counted_apart():
    pools = {"l1_logistic/positional": frozenset({"positional"}),
             "xgboost/positional+kmer": frozenset({"positional", "kmer"}),
             "ebm/llm_generated": frozenset({"llm_generated"})}
    assert CO.distinct_pools(set(pools), pools) == 3


def test_the_recorded_pool_beats_the_directory_name():
    """Pools are read from the run, because a directory name is only a label. Two arms whose runs
    record the same members were handed the same features, whatever their directories say."""
    pools = {"deft_or_cart/positions": frozenset({"positional"}),
             "ebm/positional": frozenset({"positional"})}
    assert CO.distinct_pools(set(pools), pools) == 1


def test_pools_are_read_from_the_run_and_a_cell_without_one_is_refused(tmp_path):
    for arm, pool in (("deft_or_cart/positional", ["positional"]),
                      ("l1_logistic/positional+kmer", ["positional", "kmer"])):
        cell = tmp_path / arm / "real" / "seed_42"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "run.json").write_text(json.dumps({"sources": pool}))

    pools = CO.recorded_pools(tmp_path, "real", "seed_42")
    assert pools["deft_or_cart/positional"] == frozenset({"positional"})
    assert pools["l1_logistic/positional+kmer"] == frozenset({"positional", "kmer"})

    cell = tmp_path / "ebm" / "nameonly" / "real" / "seed_42"
    cell.mkdir(parents=True)
    (cell / "run.json").write_text(json.dumps({"sources": None}))
    with pytest.raises(ValueError, match="records no input pool"):
        CO.recorded_pools(tmp_path, "real", "seed_42")


# ------------------------------------------------------------------------ per-arm summary


def frame():
    return pd.DataFrame([
        # concept found by two independent arms, on real labels
        {"labels": "real", "seed": "seed_42", "concept": 0, "n_arms": 2, "n_witnesses": 2,
         "n_pools": 2, "arms": "a/x | b/y", "names": "f1"},
        # concept unique to one arm
        {"labels": "real", "seed": "seed_42", "concept": 1, "n_arms": 1, "n_witnesses": 1,
         "n_pools": 1, "arms": "a/x", "names": "f2"},
        # the same arm corroborated under shuffled labels too
        {"labels": "permuted", "seed": "seed_42", "concept": 0, "n_arms": 2, "n_witnesses": 2,
         "n_pools": 2, "arms": "a/x | b/y", "names": "f3"},
        {"labels": "permuted", "seed": "seed_42", "concept": 1, "n_arms": 1, "n_witnesses": 1,
         "n_pools": 1, "arms": "b/y", "names": "f4"},
    ])


def test_per_arm_shares_and_correction():
    summary = CO.per_arm_summary(frame()).set_index("arm")

    # arm a/x: 1 of 2 concepts corroborated on real, 1 of 1 on permuted
    assert summary.loc["a/x", "corroborated_share"] == pytest.approx(0.5)
    assert summary.loc["a/x", "corroborated_share_permuted"] == pytest.approx(1.0)
    assert summary.loc["a/x", "unique_share"] == pytest.approx(0.5)
    # chance of 1.0 leaves no room on the scale
    assert pd.isna(summary.loc["a/x", "corroborated_share_corrected"])

    # arm b/y: 1 of 1 on real, 1 of 2 on permuted -> corrected (1 - 0.5) / (1 - 0.5) = 1.0
    assert summary.loc["b/y", "corroborated_share_corrected"] == pytest.approx(1.0)


def test_a_label_set_where_no_arm_used_a_feature_is_reported_and_not_divided_by():
    """On shuffled labels a sparse model, or a symbolic one that chose its constant, can use no
    feature at all, so the permuted label set has no concepts."""
    lines = CO.shares_by_label_set(frame()[lambda rows: rows.labels == "real"])

    assert "  permuted : no arm used a feature" in lines
    assert any(line.startswith("  real     :") and "pools" in line for line in lines)


def test_empirical_p_is_the_permuted_tail():
    p = CO.empirical_p(frame(), "n_witnesses")

    assert p[1] == pytest.approx(1.0)     # every permuted concept has >= 1
    assert p[2] == pytest.approx(0.5)     # one of two has >= 2


def test_only_real_arms_appear_in_the_summary():
    assert set(CO.per_arm_summary(frame())["arm"]) == {"a/x", "b/y"}


# --------------------------------------------------- agreeing about which way a concept runs


def test_two_arms_using_a_concept_the_same_way_agree():
    values = np.arange(20.0)
    direction, agreement = CO.agreed_direction(
        [("higher_in_positive", values), ("higher_in_positive", values * 3)])

    assert direction == "higher_in_positive"
    assert agreement == 1.0


def test_a_feature_and_its_negation_are_one_property_pointing_one_way():
    """The matcher clusters on absolute correlation, so a GC fraction and an AT fraction over the
    same window land in one concept, with opposite directions that describe the same finding.
    Compared without aligning their signs, they would read as total disagreement."""
    values = np.arange(20.0)
    direction, agreement = CO.agreed_direction(
        [("higher_in_positive", values), ("higher_in_negative", -values)])

    assert direction == "higher_in_positive"
    assert agreement == 1.0


def test_two_arms_using_a_concept_in_opposite_ways_do_not_agree():
    values = np.arange(20.0)
    direction, agreement = CO.agreed_direction(
        [("higher_in_positive", values), ("higher_in_negative", values)])

    assert agreement == 0.5
    assert direction in ("higher_in_positive", "higher_in_negative")


def test_a_concept_only_one_arm_gave_a_direction_for_has_nothing_to_agree_with():
    """An agreement of 1.0 would mean unanimity among one arm, and in a corroboration table it
    would suggest agreement where no second arm exists."""
    direction, agreement = CO.agreed_direction([("higher_in_positive", np.arange(20.0))])

    assert direction == "higher_in_positive"
    assert agreement is None


def test_a_member_that_cannot_be_aligned_is_dropped_rather_than_assumed_aligned():
    constant = np.ones(20)
    direction, agreement = CO.agreed_direction(
        [("higher_in_positive", np.arange(20.0)), ("higher_in_negative", constant)])

    assert direction == "higher_in_positive"
    assert agreement is None


def test_no_directions_at_all_leaves_both_blank():
    assert CO.agreed_direction([]) == (None, None)
