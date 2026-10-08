"""Run-integrity tracking: did the language model actually contribute to this tree?

DEFT seeds every node's candidate pool with *naive* features -- one indicator per
(position, nucleotide), such as `pos_50_is_G` -- generated locally with no API call. So when
the language model contributes nothing, DEFT does not fail. It builds the whole tree out of
naive features. The result fits, serializes and prints exactly like a real DEFT run.

That makes several ordinary faults invisible:

    - a dead endpoint or wrong deployment name, once retries are exhausted
    - a request parameter the provider rejects
    - generated code that never compiles to a callable
    - generated features that compile but raise on the data

This module counts what happened at every stage, so the question "how much of this tree is
actually DEFT?" has an answer on every run, and enforces a policy that turns a degraded run
into a loud failure instead of a plausible-looking tree.

Configured through the `integrity` config group -- see `src/deft/conf/integrity/strict.yaml`, which
documents each setting.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Where a candidate feature came from. Stored on `FeatureInfo.origin`.
ORIGIN_LLM = "llm"
ORIGIN_NAIVE = "naive"
ORIGIN_UNKNOWN = "unknown"


class DegradedRunError(RuntimeError):
    """The language model contributed too little for the run to be a DEFT result.

    Raised during `fit`, so it surfaces at the node where the problem first appeared rather
    than after a full tree has been built and a budget spent.
    """


def _rate(numerator: int, denominator: int) -> float:
    """Rate that reads as 1.0 when nothing was attempted, so it never trips a minimum."""
    return 1.0 if denominator == 0 else numerator / denominator


@dataclass
class LlmCallCounters:
    """API-level counts, accumulated across the whole run."""

    requests_attempted: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    completions_requested: int = 0
    completions_returned: int = 0

    @property
    def request_success_rate(self) -> float:
        return _rate(self.requests_succeeded, self.requests_attempted)

    @property
    def completion_yield(self) -> float:
        """Completions returned per completion requested.

        Below 1.0 means the provider short-changed us. *Above* 1.0 also indicates a bug: the
        Azure GPT-5 adapter ignores the requested count and always returns its own `n`.
        """
        return _rate(self.completions_returned, self.completions_requested)


@dataclass
class NodeRecord:
    """What the language model produced at one node, from prompt to usable feature."""

    index: int
    depth: int
    n_samples: int

    semantic_prompts: int = 0
    semantic_proposals_parsed: int = 0
    code_prompts: int = 0
    code_features_parsed: int = 0

    # The repair round (`params_generation.repair_rounds`): prompts that sent a failed candidate
    # back with its failure, candidates they corrected, and candidates still failing when the
    # rounds ran out. `code_features_parsed` counts first attempts only, so its yield stays a
    # measure of the code as first written.
    repair_prompts: int = 0
    repaired: int = 0
    unrepaired: int = 0

    # Parsed, compiled to a callable, ran on the node's data without raising, and scored
    # finitely. This is the number that matters: anything less than 1 means the split at this
    # node was decided entirely by locally-generated naive features.
    llm_features_usable: int = 0

    # Parsed, compiled, and ran on this node's data without raising -- but *not* necessarily
    # scored, because a feature that runs perfectly still gets no score when no cut point leaves
    # `min_samples_leaf` samples on both sides. That is a property of the node's size, not of the
    # generated code, and the two must be counted apart: `usable_feature_yield` below is checked
    # against a policy whose stated subject is code that fails to compile or raises, so it has to
    # measure that and nothing else.
    llm_features_ran: int = 0

    naive_features_in_pool: int = 0

    split_feature_name: str | None = None
    split_feature_origin: str | None = None  # ORIGIN_LLM | ORIGIN_NAIVE | None if no split

    # True when the run budget ran out while this node was being expanded. Such nodes are
    # exempt from the integrity checks: running out of budget is a different thing from the
    # language model failing to contribute, and reporting it as degradation would be wrong.
    budget_stopped: bool = False

    @property
    def usable_feature_yield(self) -> float:
        """Fraction of code-generation prompts that ended as a feature we could actually run.

        Deliberately **not** the fraction that got a score. Those differ, and only this one is
        about the generated code. On a node of 6,803 samples with `min_samples_leaf` 2,798, a
        legal cut must fall between the 2,798th and 4,005th sorted value -- a window covering
        under a fifth of the range -- so a sparse feature such as a motif count that is zero for
        most rows has no legal cut at all. It ran fine; it simply cannot split *here*.

        Scoring that as a code-generation failure killed a healthy Pol II tree at node 29 of 30,
        with the API at a 1.00 success rate, 5 compile failures in the entire run, and
        language-model features having already won 19 of 30 splits.
        """
        return _rate(self.llm_features_ran, self.code_prompts)


@dataclass
class IntegrityPolicy:
    """When a run should be called degraded. See `src/deft/conf/integrity/strict.yaml`."""

    enabled: bool = True
    on_violation: str = "error"  # "error" | "warn"

    min_llm_candidates_per_node: int = 1
    # Share of nodes allowed to produce no usable language-model feature.
    # Failure means most of the tree was decided without the language model, which is what a
    # pipeline contributing nothing looks like. Set well above the geometric rate: a deep tree
    # legitimately holds a minority of nodes too small for any sparse feature to cut -- 4 of 30
    # (13%) on a measured Pol II tree -- and failing on those discards good runs. The warning
    # threshold keeps the number visible without acting on it.
    max_no_candidate_node_share: float = 0.5
    warn_no_candidate_node_share: float = 0.25
    min_request_success_rate: float = 0.95
    min_usable_feature_yield: float = 0.25
    warn_if_naive_split_share_above: float = 0.5

    @classmethod
    def from_config(cls, cfg: Any) -> IntegrityPolicy:
        """Build from a Hydra config node. A missing block means the strict defaults."""
        if cfg is None:
            return cls()
        fields = {
            "enabled",
            "on_violation",
            "min_llm_candidates_per_node",
            "max_no_candidate_node_share",
            "warn_no_candidate_node_share",
            "min_request_success_rate",
            "min_usable_feature_yield",
            "warn_if_naive_split_share_above",
        }
        values = {k: cfg[k] for k in fields if k in cfg}
        policy = cls(**values)
        if policy.on_violation not in ("error", "warn"):
            raise ValueError(
                f"integrity.on_violation must be 'error' or 'warn', got "
                f"{policy.on_violation!r}."
            )
        return policy


@dataclass
class RunProvenance:
    """One per run. Attached to the engine, the feature finder and the tree."""

    policy: IntegrityPolicy = field(default_factory=IntegrityPolicy)
    calls: LlmCallCounters = field(default_factory=LlmCallCounters)
    nodes: list[NodeRecord] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    _current: NodeRecord | None = field(default=None, repr=False)

    # ---- recording, called from the engine ---------------------------------------------

    def record_request(self, *, succeeded: bool, requested: int, returned: int) -> None:
        self.calls.requests_attempted += 1
        if succeeded:
            self.calls.requests_succeeded += 1
        else:
            self.calls.requests_failed += 1
        self.calls.completions_requested += requested
        self.calls.completions_returned += returned

    # ---- recording, called from the feature finder and the tree -------------------------

    def start_node(self, *, depth: int, n_samples: int) -> NodeRecord:
        self._current = NodeRecord(index=len(self.nodes), depth=depth, n_samples=n_samples)
        self.nodes.append(self._current)
        return self._current

    @property
    def current_node(self) -> NodeRecord | None:
        return self._current

    def record_generation(
        self, *, semantic_prompts: int, semantic_proposals: int, code_prompts: int,
        code_features: int,
    ) -> None:
        """Accumulate one `generate_features` pass. Reflection calls this repeatedly."""
        if self._current is None:
            return
        self._current.semantic_prompts += semantic_prompts
        self._current.semantic_proposals_parsed += semantic_proposals
        self._current.code_prompts += code_prompts
        self._current.code_features_parsed += code_features

    def record_repair(self, *, repair_prompts: int, repaired: int, unrepaired: int) -> None:
        """Accumulate one repair round's outcome at the current node."""
        if self._current is None:
            return
        self._current.repair_prompts += repair_prompts
        self._current.repaired += repaired
        self._current.unrepaired += unrepaired

    def record_candidate_pool(self, *, llm_usable: int, llm_ran: int | None = None,
                              naive: int = 0) -> None:
        """Accumulate this node's usable candidates.

        Called after every generation round, so a reflection loop **adds** to the count rather
        than replacing it. Recording only the initial round under-reported the node by however
        much reflection contributed, which is most of it at the paper's settings.

        Naive features are generated once, at initialisation, so the largest value seen is the
        count rather than a sum.
        """
        if self._current is None:
            return
        self._current.llm_features_usable += llm_usable
        # Defaults to `llm_usable` so a caller that has not been updated keeps the old meaning
        # rather than silently reporting a yield of zero.
        self._current.llm_features_ran += llm_usable if llm_ran is None else llm_ran
        self._current.naive_features_in_pool = max(
            self._current.naive_features_in_pool, naive
        )

    def finish_node(self, *, split_feature: Any | None, budget_stopped: bool = False) -> None:
        """Close the current node and apply the per-node policy.

        Raises `DegradedRunError` under `on_violation: error`, at the first bad node, so a
        broken run stops before spending a budget on a tree that means nothing.

        A node cut short by the run budget is recorded but not checked -- see
        `NodeRecord.budget_stopped`.
        """
        node = self._current
        self._current = None
        if node is None:
            return

        node.budget_stopped = budget_stopped

        if split_feature is not None and getattr(split_feature, "name", None):
            node.split_feature_name = split_feature.name
            node.split_feature_origin = getattr(split_feature, "origin", ORIGIN_UNKNOWN)

        if not self.policy.enabled or budget_stopped:
            return

        # Fires while every node so far has produced nothing usable -- which is what a dead
        # endpoint, a bad deployment name or code that never compiles all look like, and is the
        # case this check was written for. It stays a fast fail: a broken run stops at its first
        # node rather than spending a budget.
        #
        # It deliberately does *not* fire for an isolated node, because that has a mundane cause
        # with nothing wrong behind it. A feature only scores when some cut point leaves
        # `min_samples_leaf` samples on both sides, so a node barely above twice that size can
        # defeat every sparse feature in the pool while the pipeline is perfectly healthy. Node
        # 31 of a Pol II tree -- depth 5, 6,803 samples against a leaf minimum of 2,798 -- hit
        # exactly that, on a run whose API success rate was 1.00 and whose generated features had
        # already won 19 of 30 splits. Killing that run discarded a good tree and reported a
        # geometry problem as a credentials problem.
        #
        # The share across the whole run is checked in `check_run`, which is where "the language
        # model contributed nothing" is properly a question about a run rather than about a node.
        checked = [n for n in self.nodes if not n.budget_stopped]
        starved = [n for n in checked
                   if n.llm_features_usable < self.policy.min_llm_candidates_per_node]
        if node.llm_features_usable < self.policy.min_llm_candidates_per_node and (
            len(starved) == len(checked)
        ):
            self._violation(
                f"Every node so far ({len(checked)}) produced fewer than "
                f"{self.policy.min_llm_candidates_per_node} usable language-model feature, so "
                f"every split has been decided by locally-generated naive features and this is "
                f"not a DEFT result.\n"
                f"  Latest, node {node.index} (depth {node.depth}, {node.n_samples} samples): "
                f"{node.semantic_prompts} semantic prompt(s) -> "
                f"{node.semantic_proposals_parsed} proposal(s) parsed -> "
                f"{node.code_prompts} code prompt(s) -> "
                f"{node.code_features_parsed} feature(s) parsed -> "
                f"{node.llm_features_usable} usable.\n"
                f"  Run totals so far: {self.calls.requests_succeeded}/"
                f"{self.calls.requests_attempted} API requests succeeded.\n"
                f"  Check first: credentials and deployment name; then the run log for "
                f"repeated request failures or code that failed to compile."
            )

        if node.code_prompts and node.usable_feature_yield < self.policy.min_usable_feature_yield:
            self._violation(
                f"Node {node.index}: only {node.llm_features_ran} of "
                f"{node.code_prompts} code-generation prompts produced a feature that could "
                f"be run at all (yield {node.usable_feature_yield:.2f}, minimum "
                f"{self.policy.min_usable_feature_yield:.2f}).\n"
                f"  That points at generated code failing to compile or raising on the data, "
                f"rather than at the API. Check the run log for parse errors.\n"
                f"  For reference, {node.llm_features_usable} of those also scored; a feature "
                f"that runs but cannot split this node ({node.n_samples} samples) is not "
                f"counted against this check."
            )

    # ---- run-level check, called once after fit -----------------------------------------

    def check_run(self) -> None:
        if not self.policy.enabled:
            return

        if self.calls.request_success_rate < self.policy.min_request_success_rate:
            self._violation(
                f"Only {self.calls.requests_succeeded} of {self.calls.requests_attempted} "
                f"API requests succeeded (rate {self.calls.request_success_rate:.2f}, minimum "
                f"{self.policy.min_request_success_rate:.2f}). Failed requests are retried to "
                f"exhaustion and then dropped silently, so the candidate pool was smaller than "
                f"configured throughout."
            )

        # The run-level form of the per-node candidate check. A handful of nodes too small for any
        # sparse feature to split is normal and says nothing about the pipeline; most of the tree
        # in that state means the language model is not contributing, which is the thing this
        # whole policy exists to refuse to call a DEFT result.
        checked = [n for n in self.nodes if not n.budget_stopped]
        if checked:
            starved = sum(1 for n in checked
                          if n.llm_features_usable < self.policy.min_llm_candidates_per_node)
            share = starved / len(checked)
            if self.policy.warn_no_candidate_node_share < share <= (
                self.policy.max_no_candidate_node_share
            ):
                logger.warning(
                    "%d of %d nodes (%.0f%%) produced no usable language-model feature. Allowed, "
                    "and expected for nodes barely larger than twice `min_samples_leaf`, but "
                    "worth reading beside `llm_split_share`.",
                    starved, len(checked), 100 * share,
                )
            if share > self.policy.max_no_candidate_node_share:
                self._violation(
                    f"{starved} of {len(checked)} nodes ({share:.0%}) produced fewer than "
                    f"{self.policy.min_llm_candidates_per_node} usable language-model feature, "
                    f"above the permitted {self.policy.max_no_candidate_node_share:.0%}. Those "
                    f"splits were decided by locally-generated naive features.\n"
                    f"  A few such nodes are ordinary -- a node barely larger than twice "
                    f"`min_samples_leaf` can defeat every sparse feature in the pool. This many "
                    f"is not.\n"
                    f"  Run totals: {self.calls.requests_succeeded}/"
                    f"{self.calls.requests_attempted} API requests succeeded."
                )

        split_nodes = [n for n in self.nodes if n.split_feature_origin]
        if split_nodes:
            naive = sum(1 for n in split_nodes if n.split_feature_origin == ORIGIN_NAIVE)
            share = naive / len(split_nodes)
            if share > self.policy.warn_if_naive_split_share_above:
                # A warning, not a violation. Naive features winning a split is a legitimate
                # modelling outcome; it only becomes evidence of breakage alongside the
                # per-node candidate check above, which has already passed by this point.
                logger.warning(
                    "%d of %d splits (%.0f%%) were won by naive features. That is allowed, but "
                    "worth a look: the language model's features are being outscored "
                    "consistently.",
                    naive, len(split_nodes), 100 * share,
                )

    # ---- reporting ----------------------------------------------------------------------

    def _violation(self, message: str) -> None:
        self.violations.append(message)
        if self.policy.on_violation == "error":
            raise DegradedRunError(message)
        logger.warning("Integrity check failed (on_violation=warn): %s", message)

    def summary(self) -> dict:
        """Run-level roll-up. The headline number is `llm_split_share`."""
        split_nodes = [n for n in self.nodes if n.split_feature_origin]
        llm_splits = sum(1 for n in split_nodes if n.split_feature_origin == ORIGIN_LLM)
        naive_splits = sum(1 for n in split_nodes if n.split_feature_origin == ORIGIN_NAIVE)
        return {
            "nodes_expanded": len(self.nodes),
            "nodes_split": len(split_nodes),
            "splits_by_llm_feature": llm_splits,
            "splits_by_naive_feature": naive_splits,
            "llm_split_share": _rate(llm_splits, len(split_nodes)) if split_nodes else None,
            # Budget-stopped nodes are excluded: they never got the chance to produce a feature,
            # which is a different thing from producing none.
            "nodes_with_no_usable_llm_feature": sum(
                1 for n in self.nodes if n.llm_features_usable == 0 and not n.budget_stopped
            ),
            "nodes_stopped_by_budget": sum(1 for n in self.nodes if n.budget_stopped),
            "total_usable_llm_features": sum(n.llm_features_usable for n in self.nodes),
            "repair_prompts": sum(n.repair_prompts for n in self.nodes),
            "repaired": sum(n.repaired for n in self.nodes),
            "unrepaired": sum(n.unrepaired for n in self.nodes),
            "request_success_rate": self.calls.request_success_rate,
            "completion_yield": self.calls.completion_yield,
        }

    def to_dict(self) -> dict:
        return {
            "policy": asdict(self.policy),
            "calls": asdict(self.calls),
            "summary": self.summary(),
            "violations": self.violations,
            "nodes": [asdict(n) for n in self.nodes],
        }

    def format_report(self) -> str:
        s = self.summary()
        share = s["llm_split_share"]
        share_str = "n/a" if share is None else f"{100 * share:.0f}%"
        repairs = (
            f"\n  candidates repaired / not      : {s['repaired']} / {s['unrepaired']}  "
            f"({s['repair_prompts']} repair prompt(s))"
            if s["repair_prompts"] else ""
        )
        return (
            "Run integrity\n"
            f"  nodes expanded                  : {s['nodes_expanded']}\n"
            f"  splits won by an LLM feature    : {s['splits_by_llm_feature']} of "
            f"{s['nodes_split']}  ({share_str})\n"
            f"  splits won by a naive feature   : {s['splits_by_naive_feature']}\n"
            f"  nodes with no usable LLM feature: {s['nodes_with_no_usable_llm_feature']}\n"
            f"  API requests succeeded          : {self.calls.requests_succeeded} of "
            f"{self.calls.requests_attempted}\n"
            f"  completions returned/requested  : {self.calls.completions_returned}/"
            f"{self.calls.completions_requested}"
            + repairs
        )
