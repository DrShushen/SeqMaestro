"""Token accounting and run budgets (`src/deft/budget.py`, `src/deft/conf/budget/`).

The behaviour that matters most is what happens when a limit is reached: the tree must stop
*growing* and still return a complete, valid, predicting tree. Aborting would throw away work
already paid for. These tests run offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deft.budget import (
    REASON_API_CALLS,
    REASON_TOKENS,
    REASON_WALL_CLOCK,
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
)
from deft.provenance import IntegrityPolicy, RunProvenance
from deft.trees.split_criterion import GiniCriterion
from deft.trees.tree import AdaptiveDecisionTree
from deft.utils.feature import FeatureInfo


def usage(total=100, prompt=40, completion=60, reasoning=0):
    return {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        }
    }


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# --------------------------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------------------------


def test_records_every_component_of_spend():
    budget = RunBudget()
    budget.record_response(usage(total=150, prompt=50, completion=100, reasoning=30))
    budget.record_response(usage(total=150, prompt=50, completion=100, reasoning=30))

    assert budget.spend.total_tokens == 300
    assert budget.spend.prompt_tokens == 100
    assert budget.spend.completion_tokens == 200
    assert budget.spend.reasoning_tokens == 60
    assert budget.spend.api_calls == 2


def test_a_failed_request_still_counts_as_a_call():
    """Retried-to-death requests cost wall clock even when they return nothing."""
    budget = RunBudget()
    budget.record_response(None)

    assert budget.spend.api_calls == 1
    assert budget.spend.total_tokens == 0


def test_total_is_derived_when_the_provider_omits_it():
    budget = RunBudget()
    budget.record_response({"usage": {"prompt_tokens": 40, "completion_tokens": 60}})

    assert budget.spend.total_tokens == 100


def test_spend_is_attributed_to_the_node_being_expanded():
    """`mean_tokens_per_node` is what a budget gets chosen from, so per-node spend must be right."""
    budget = RunBudget()
    budget.start_node(index=0, depth=0)
    budget.record_response(usage(total=500))
    budget.finish_node()
    budget.start_node(index=1, depth=1)
    budget.record_response(usage(total=300))
    budget.finish_node()

    assert [n.total_tokens for n in budget.nodes] == [500, 300]
    assert budget.to_dict()["mean_tokens_per_node"] == 400


# --------------------------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------------------------


def test_no_limits_never_exhaust():
    budget = RunBudget()
    for _ in range(100):
        budget.record_response(usage(total=10_000))

    assert budget.exhausted is False
    assert budget.remaining_tokens is None
    budget.check()  # must not raise


def test_token_limit_stops_the_run():
    budget = RunBudget(BudgetLimits(max_total_tokens=250))
    budget.record_response(usage(total=100))
    budget.check()  # under the limit, fine
    budget.record_response(usage(total=200))

    assert budget.remaining_tokens == 0
    with pytest.raises(BudgetExceeded) as excinfo:
        budget.check()
    assert excinfo.value.reason == REASON_TOKENS
    assert "300" in str(excinfo.value) and "250" in str(excinfo.value)
    # The message must say what happens next, not just that something stopped.
    assert "shallower tree" in str(excinfo.value)


def test_api_call_limit_stops_the_run():
    """A backstop independent of token size: retries can generate many cheap calls."""
    budget = RunBudget(BudgetLimits(max_api_calls=3))
    for _ in range(3):
        budget.record_response(None)

    with pytest.raises(BudgetExceeded) as excinfo:
        budget.check()
    assert excinfo.value.reason == REASON_API_CALLS


def test_wall_clock_limit_stops_the_run():
    clock = FakeClock()
    budget = RunBudget(BudgetLimits(max_wall_seconds=60), clock=clock)
    clock.now = 59
    budget.check()  # not yet
    clock.now = 61

    with pytest.raises(BudgetExceeded) as excinfo:
        budget.check()
    assert excinfo.value.reason == REASON_WALL_CLOCK


def test_remaining_tokens_never_goes_negative():
    budget = RunBudget(BudgetLimits(max_total_tokens=100))
    budget.record_response(usage(total=250))

    assert budget.remaining_tokens == 0


def test_limits_from_config_reject_nonsense():
    with pytest.raises(ValueError, match="positive or null"):
        BudgetLimits.from_config({"max_total_tokens": 0})
    with pytest.raises(ValueError, match="positive or null"):
        BudgetLimits.from_config({"max_wall_seconds": -1})


def test_limits_from_config_treats_null_as_no_limit():
    limits = BudgetLimits.from_config({"max_total_tokens": None, "max_api_calls": 10})

    assert limits.max_total_tokens is None
    assert limits.max_api_calls == 10
    assert limits.any_set is True
    assert BudgetLimits.from_config(None).any_set is False


def test_report_names_the_limit_that_stopped_the_run():
    budget = RunBudget(BudgetLimits(max_total_tokens=100))
    budget.record_response(usage(total=150))
    with pytest.raises(BudgetExceeded):
        budget.check()

    assert budget.to_dict()["stopped_early"] is True
    assert budget.to_dict()["stopped_reason"] == REASON_TOKENS
    assert "STOPPED EARLY" in budget.format_report()


# --------------------------------------------------------------------------------------------
# Graceful termination: the behaviour that matters
# --------------------------------------------------------------------------------------------


def alternating(X):
    """A feature that always splits its node in half, so the tree can grow to any depth."""
    return np.arange(len(X)) % 2


class BudgetBurningFeatureFinder:
    """Stands in for the LLM finder: spends budget, then returns a workable split.

    It mimics the engine's contract -- check the budget before dispatching, then record what the
    request cost -- without any network or model involved.
    """

    def __init__(self, budget: RunBudget, tokens_per_node: int):
        self.budget = budget
        self.tokens_per_node = tokens_per_node
        self.logging_artifacts = []
        self.provenance = None
        self.calls = 0

    def get_optimal_feature(self, X, y, **kwargs):
        self.budget.check()
        self.budget.record_response(usage(total=self.tokens_per_node))
        self.calls += 1
        if self.provenance is not None:
            # As the real finder does, so the integrity policy sees a healthy node.
            self.provenance.record_candidate_pool(llm_usable=1, naive=0)
        return FeatureInfo(
            fn=alternating,
            name=f"alternating_{self.calls}",
            description="splits the node in half",
            string="def alternating(X): ...",
            threshold=0.5,
            score=0.5,
            origin="llm",
        )


def build_tree(max_total_tokens, tokens_per_node=100, max_depth=4):
    budget = RunBudget(BudgetLimits(max_total_tokens=max_total_tokens))
    finder = BudgetBurningFeatureFinder(budget, tokens_per_node)
    tree = AdaptiveDecisionTree(
        feature_finder=finder,
        splitting_criterion=GiniCriterion(),
        max_depth=max_depth,
        min_samples_leaf=1,
        save_locally=False,
    )
    tree.budget = budget
    return tree, budget, finder


def make_dataset(n=256):
    """Labels must not line up with `alternating`, or nodes go pure and growth stops on its
    own before any budget is reached."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=n)
    return (
        pd.DataFrame({"raw_sequence": ["".join(rng.choice(list("ACGT"), 8)) for _ in range(n)]}),
        pd.DataFrame({"label": labels}),
    )


def walk(node):
    yield node
    for child in (node.left, node.right):
        if child is not None:
            yield from walk(child)


def test_budget_stop_still_returns_a_complete_usable_tree():
    """The whole point: a stopped run yields a smaller tree, not a lost one."""
    tree, budget, finder = build_tree(max_total_tokens=250, tokens_per_node=100, max_depth=4)
    X, y = make_dataset()

    tree.fit(X, y)  # must not raise

    assert tree.budget_stopped is True
    assert tree.budget_stopped_reason == REASON_TOKENS

    # Every node is resolved: leaves carry a prediction, internal nodes carry a feature.
    for node in walk(tree.root):
        assert node.feature is not None, "a node was left unresolved"
        if node._is_leaf():
            assert node.prediction is not None

    predictions = np.asarray(tree.predict(X))
    assert len(predictions) == len(X), "the tree must still predict for every row"


def test_budget_stop_produces_a_shallower_tree_than_asked_for():
    generous, _, _ = build_tree(max_total_tokens=None, tokens_per_node=100, max_depth=4)
    X, y = make_dataset()
    generous.fit(X, y)
    full_depth_calls = generous.feature_finder.calls

    stopped, _, finder = build_tree(max_total_tokens=250, tokens_per_node=100, max_depth=4)
    stopped.fit(X, y)

    assert finder.calls < full_depth_calls, "the budget should have cut growth short"
    assert stopped.budget_stopped and not generous.budget_stopped


def test_no_further_llm_calls_after_the_budget_stops():
    """Draining the queue must not spend anything more."""
    tree, budget, finder = build_tree(max_total_tokens=250, tokens_per_node=100, max_depth=5)
    X, y = make_dataset()
    tree.fit(X, y)

    # Three calls of 100 tokens: the third is the one that trips the check on the fourth.
    assert finder.calls == 3
    assert budget.spend.total_tokens == 300


def test_a_generous_budget_does_not_stop_anything():
    tree, budget, finder = build_tree(max_total_tokens=1_000_000, max_depth=3)
    X, y = make_dataset()
    tree.fit(X, y)

    assert tree.budget_stopped is False
    assert budget.to_dict()["stopped_early"] is False


# --------------------------------------------------------------------------------------------
# Interaction with the integrity policy
# --------------------------------------------------------------------------------------------


def test_a_budget_stop_is_not_reported_as_degradation():
    """Running out of budget is not the language model failing to contribute.

    Without the exemption, the node cut short by the budget has zero usable LLM features and
    the strict integrity policy would fail the run -- blaming the wrong thing.
    """
    tree, budget, finder = build_tree(max_total_tokens=250, tokens_per_node=100, max_depth=4)
    provenance = RunProvenance(policy=IntegrityPolicy())  # strict
    tree.provenance = provenance
    finder.provenance = provenance
    X, y = make_dataset()

    tree.fit(X, y)  # must not raise DegradedRunError

    assert provenance.violations == []
    stopped = [n for n in provenance.nodes if n.budget_stopped]
    assert len(stopped) == 1
    assert provenance.summary()["nodes_stopped_by_budget"] == 1
    # The stopped node must not be counted against the language model.
    assert provenance.summary()["nodes_with_no_usable_llm_feature"] == 0
