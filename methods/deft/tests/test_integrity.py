"""Tests for the run-integrity policy (`src/deft/provenance.py`, `src/deft/conf/integrity/`).

These run offline. The point of the policy is to catch a *dead* language model, so the way to
test it is with an engine that returns nothing -- exactly what the real engine does once a
request has been retried to exhaustion.

Without the policy, every one of these fits succeeds and returns a plausible tree built
entirely from naive features. That is the behaviour being fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.features.feature_finder import LLMFeatureFinder
from deft.provenance import (
    ORIGIN_LLM,
    ORIGIN_NAIVE,
    DegradedRunError,
    IntegrityPolicy,
    LlmCallCounters,
    RunProvenance,
)
from deft.trees.split_criterion import GiniCriterion
from deft.trees.tree import AdaptiveDecisionTree
from deft.utils.selection import TopKSelector

SEQUENCE_LENGTH = 20


class DeadLlmEngine:
    """An engine whose every request fails, the way a wrong deployment name behaves.

    `LiteLLMEngine._single_request` returns None once retries are exhausted, and `query_llm`
    passes those Nones straight through. This reproduces that contract, including the provenance
    bookkeeping the real engine does.
    """

    def __init__(self):
        self.provenance = None

    async def query_llm(self, list_prompts, n_generations_per_prompt):
        for _ in list_prompts:
            if self.provenance is not None:
                self.provenance.record_request(
                    succeeded=False, requested=int(n_generations_per_prompt), returned=0
                )
        return [None] * len(list_prompts)


def make_dataset(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    sequences = ["".join(rng.choice(list("ACGT"), size=SEQUENCE_LENGTH)) for _ in labels]
    return (
        pd.DataFrame({"raw_sequence": sequences}),
        pd.DataFrame({"label": labels}),
    )


GENERATION_PARAMS = {
    "target_name": "a test label",
    "dataset_info": "A test dataset.",
    "context_template_name_desc": "Predict {target_name}. {dataset_info}",
    "context_code": "Write a feature function.",
    "instructions_name_desc": "Return JSON.",
    "instructions_code": "Return a function.",
    "include_ic_samples": False,
    "include_history": False,
    "include_feature_description": False,
    "n_ic_samples": 1,
    "population_size": 4,
    "n_samples_per_prompt": 2,
    "use_reflection": False,
    "n_reflections": 0,
    "initialize_with_naive_features": True,
}


def build_tree(policy: IntegrityPolicy, tmp_path) -> tuple[AdaptiveDecisionTree, RunProvenance]:
    engine = DeadLlmEngine()
    finder = LLMFeatureFinder(llm_engine=engine)
    tree = AdaptiveDecisionTree(
        feature_finder=finder,
        splitting_criterion=GiniCriterion(),
        max_depth=1,
        min_samples_leaf=5,
        save_locally=False,
        folder_path=str(tmp_path),
    )
    provenance = RunProvenance(policy=policy)
    engine.provenance = provenance
    finder.provenance = provenance
    tree.provenance = provenance
    return tree, provenance


def fit(tree, **overrides):
    X, y = make_dataset()
    params = {**GENERATION_PARAMS, **overrides}
    tree.fit(X, y, **params, selector=TopKSelector())
    return X, y


# --------------------------------------------------------------------------------------------
# The behaviour being fixed
# --------------------------------------------------------------------------------------------


def test_dead_llm_fails_the_run(tmp_path):
    """Under the default strict policy, a dead language model stops the run."""
    tree, provenance = build_tree(IntegrityPolicy(), tmp_path)

    with pytest.raises(DegradedRunError) as excinfo:
        fit(tree)

    message = str(excinfo.value)
    assert "usable language-model feature" in message
    assert "naive features" in message
    # The message must tell a human what to look at first.
    assert "deployment name" in message

    # It stopped at the very first node rather than after building a whole tree.
    assert len(provenance.nodes) == 1
    assert provenance.nodes[0].llm_features_usable == 0
    assert provenance.nodes[0].naive_features_in_pool > 0


def test_without_the_policy_the_failure_is_silent(tmp_path):
    """Documents exactly what the policy prevents: upstream's behaviour, made visible.

    With checks off, the identical dead engine produces a complete, well-formed tree that
    predicts and prints. Nothing about it says the language model contributed nothing -- except
    the provenance record, which is still collected.
    """
    tree, provenance = build_tree(IntegrityPolicy(enabled=False), tmp_path)
    X, y = fit(tree)

    assert tree.root is not None
    assert tree.root.feature.name, "a split was still found, purely from naive features"
    assert len(tree.predict(X)) == len(X)

    summary = provenance.summary()
    assert summary["llm_split_share"] == 0.0
    assert summary["total_usable_llm_features"] == 0
    assert summary["nodes_with_no_usable_llm_feature"] == len(provenance.nodes)
    assert summary["request_success_rate"] == 0.0


def test_warn_only_completes_but_records_the_violation(tmp_path):
    """`integrity=warn_only`: same visibility, no failure."""
    tree, provenance = build_tree(IntegrityPolicy(on_violation="warn"), tmp_path)
    fit(tree)

    assert provenance.violations, "the breach should still be recorded"
    assert provenance.summary()["llm_split_share"] == 0.0
    assert tree.root.feature.name


def test_split_origin_is_recorded_as_naive(tmp_path):
    """The winning split is attributed, which is what `llm_split_share` is built from."""
    tree, provenance = build_tree(IntegrityPolicy(enabled=False), tmp_path)
    fit(tree)

    node = provenance.nodes[0]
    assert node.split_feature_origin == ORIGIN_NAIVE
    assert node.split_feature_name.startswith("pos_")


def test_naive_features_are_not_generated_when_disabled(tmp_path):
    """With no naive fallback, a dead LLM yields no split at all rather than a fake one."""
    tree, provenance = build_tree(IntegrityPolicy(enabled=False), tmp_path)
    fit(tree, initialize_with_naive_features=False)

    assert provenance.nodes[0].naive_features_in_pool == 0
    assert provenance.nodes[0].split_feature_origin is None


# --------------------------------------------------------------------------------------------
# Policy logic, independent of any tree
# --------------------------------------------------------------------------------------------


def test_run_check_fails_on_low_request_success_rate():
    provenance = RunProvenance(policy=IntegrityPolicy(min_request_success_rate=0.95))
    provenance.calls = LlmCallCounters(
        requests_attempted=10, requests_succeeded=8, requests_failed=2
    )
    with pytest.raises(DegradedRunError, match="API requests succeeded"):
        provenance.check_run()


def test_run_check_passes_when_rates_are_healthy():
    provenance = RunProvenance()
    provenance.calls = LlmCallCounters(requests_attempted=10, requests_succeeded=10)
    provenance.check_run()  # must not raise


def test_low_usable_feature_yield_is_reported_separately_from_api_failure():
    """A parsing problem must not be misreported as an API problem."""
    provenance = RunProvenance(policy=IntegrityPolicy(min_usable_feature_yield=0.5))
    provenance.start_node(depth=0, n_samples=100)
    provenance.record_generation(
        semantic_prompts=1, semantic_proposals=8, code_prompts=8, code_features=2
    )
    provenance.record_candidate_pool(llm_usable=2, naive=0)

    with pytest.raises(DegradedRunError) as excinfo:
        provenance.finish_node(split_feature=None)
    assert "failing to compile" in str(excinfo.value)


def test_rates_are_one_when_nothing_was_attempted():
    """An empty run must not trip a minimum through division-by-zero handling."""
    counters = LlmCallCounters()
    assert counters.request_success_rate == 1.0
    assert counters.completion_yield == 1.0
    RunProvenance().check_run()  # must not raise


def test_policy_from_config_rejects_a_bad_on_violation_value():
    with pytest.raises(ValueError, match="on_violation"):
        IntegrityPolicy.from_config({"on_violation": "explode"})


def test_policy_from_config_defaults_to_strict_when_absent():
    policy = IntegrityPolicy.from_config(None)
    assert policy.enabled is True
    assert policy.on_violation == "error"
    assert policy.min_llm_candidates_per_node == 1


def test_summary_reports_llm_share_when_the_llm_wins():
    provenance = RunProvenance(policy=IntegrityPolicy(enabled=False))
    for origin in (ORIGIN_LLM, ORIGIN_LLM, ORIGIN_NAIVE):
        provenance.start_node(depth=0, n_samples=10)
        provenance.record_candidate_pool(llm_usable=3, naive=8)
        provenance.finish_node(
            split_feature=type("F", (), {"name": "x", "origin": origin})()
        )
    summary = provenance.summary()
    assert summary["splits_by_llm_feature"] == 2
    assert summary["splits_by_naive_feature"] == 1
    assert summary["llm_split_share"] == pytest.approx(2 / 3)


def test_reflection_candidates_count_towards_the_node():
    """Recording only the initial round under-reported a node by whatever reflection added.

    At the paper's `n_reflections: 20` that is most of the node's candidates, so the headline
    `total_usable_llm_features` would have been badly low on every real run.
    """
    provenance = RunProvenance(policy=IntegrityPolicy(enabled=False))
    provenance.start_node(depth=0, n_samples=100)
    provenance.record_candidate_pool(llm_usable=4, naive=404)   # initial generation
    provenance.record_candidate_pool(llm_usable=6)              # reflection round 1
    provenance.record_candidate_pool(llm_usable=5)              # reflection round 2
    provenance.finish_node(split_feature=None)

    node = provenance.nodes[0]
    assert node.llm_features_usable == 15, "reflection rounds must add, not overwrite"
    assert node.naive_features_in_pool == 404, "naive features are generated once, not per round"


# --------------------------- a feature that runs but cannot split is not a code-generation fault


def test_a_small_node_where_features_run_but_cannot_split_does_not_trip_the_yield_check():
    """The failure that killed a healthy Pol II tree at node 29 of 30.

    On 6,803 samples with `min_samples_leaf` 2,798 a legal cut must fall between the 2,798th and
    4,005th sorted value, so a sparse feature -- a motif count that is zero for most rows -- has
    no legal cut at all. It compiled, it ran, it simply cannot split *here*. Only 5 features in
    that entire run failed to compile, the API was at a 1.00 success rate, and language-model
    features had already won 19 of 30 splits.
    """
    provenance = RunProvenance(policy=IntegrityPolicy(min_usable_feature_yield=0.25))
    provenance.start_node(depth=5, n_samples=6803)
    provenance.record_generation(
        semantic_prompts=11, semantic_proposals=110, code_prompts=110, code_features=110
    )
    provenance.record_candidate_pool(llm_usable=9, llm_ran=101, naive=404)

    provenance.finish_node(split_feature=None)  # must not raise

    assert provenance.nodes[0].llm_features_usable == 9
    assert provenance.nodes[0].llm_features_ran == 101


def test_code_that_does_not_run_still_trips_the_yield_check():
    """The check must keep catching what it was written for."""
    provenance = RunProvenance(policy=IntegrityPolicy(min_usable_feature_yield=0.25))
    provenance.start_node(depth=0, n_samples=100_000)
    provenance.record_generation(
        semantic_prompts=11, semantic_proposals=110, code_prompts=110, code_features=110
    )
    provenance.record_candidate_pool(llm_usable=4, llm_ran=4, naive=404)

    with pytest.raises(DegradedRunError) as excinfo:
        provenance.finish_node(split_feature=None)
    assert "failing to compile" in str(excinfo.value)


def test_a_node_where_nothing_scored_still_trips_the_candidate_check():
    """The other check is untouched: zero usable features means naive features decided the split."""
    provenance = RunProvenance(policy=IntegrityPolicy(min_llm_candidates_per_node=1))
    provenance.start_node(depth=5, n_samples=6803)
    provenance.record_generation(
        semantic_prompts=11, semantic_proposals=110, code_prompts=110, code_features=110
    )
    provenance.record_candidate_pool(llm_usable=0, llm_ran=110, naive=404)

    with pytest.raises(DegradedRunError):
        provenance.finish_node(split_feature=None)


def test_one_starved_node_among_healthy_ones_does_not_fail_the_run():
    """Node 31 of the Pol II tree. Healthy pipeline, node too small for a sparse feature to cut."""
    provenance = RunProvenance(policy=IntegrityPolicy())
    for _ in range(30):
        provenance.start_node(depth=3, n_samples=50_000)
        provenance.record_generation(semantic_prompts=11, semantic_proposals=110,
                                     code_prompts=110, code_features=110)
        provenance.record_candidate_pool(llm_usable=100, llm_ran=104, naive=404)
        provenance.finish_node(split_feature=None)

    provenance.start_node(depth=5, n_samples=6803)
    provenance.record_generation(semantic_prompts=11, semantic_proposals=110,
                                 code_prompts=110, code_features=110)
    provenance.record_candidate_pool(llm_usable=0, llm_ran=101, naive=404)

    provenance.finish_node(split_feature=None)   # must not raise
    provenance.check_run()                       # 1 of 31 is under the permitted share


def test_a_run_where_most_nodes_are_starved_still_fails():
    """The run-level form of the same question, which is where it actually belongs."""
    provenance = RunProvenance(policy=IntegrityPolicy(enabled=False))
    for usable in [50, 0, 0, 0, 0]:
        provenance.start_node(depth=3, n_samples=50_000)
        provenance.record_generation(semantic_prompts=11, semantic_proposals=110,
                                     code_prompts=110, code_features=110)
        provenance.record_candidate_pool(llm_usable=usable, llm_ran=104, naive=404)
        provenance.finish_node(split_feature=None)

    provenance.policy = IntegrityPolicy()        # checks on, for the run-level pass only
    with pytest.raises(DegradedRunError) as excinfo:
        provenance.check_run()
    assert "4 of 5 nodes" in str(excinfo.value)
