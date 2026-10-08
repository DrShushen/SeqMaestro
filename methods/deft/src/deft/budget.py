"""Token accounting and run budgets.

Upstream read `response["usage"]["total_tokens"]` in `src/deft/utils/parsing.py`, added it to a
local, and never used it again. There was no run-level counter, no cost figure and no way to
cap spend. A depth-6 tree with reflection is hundreds of LLM calls, so "how much did that
cost?" and "stop before it costs more than X" both needed answering.

Two separable things live here:

    accounting   always on. Every run reports what it spent, per node and in total.
    limits       opt-in. When a limit is reached the run stops *gracefully* -- see below.

Graceful is the important word. Aborting mid-fit would throw away a tree that is most of the
way built and has already been paid for. Instead the tree stops growing: the node being
expanded becomes a leaf with the correct majority prediction, every node still queued becomes a
leaf too, and `fit` returns a complete, valid, shallower tree. It is smaller than you asked for
and it says so, in the report and in `budget_stopped` on the tree.

Configured through the `budget` config group -- see `src/deft/conf/budget/unlimited.yaml`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Why a run stopped.
REASON_TOKENS = "max_total_tokens"
REASON_API_CALLS = "max_api_calls"
REASON_WALL_CLOCK = "max_wall_seconds"


class BudgetExceeded(RuntimeError):
    """A run budget limit was reached.

    Raised from the engine before dispatching more work, and caught by the tree, which stops
    growing rather than failing. It is not an error condition: a budget-stopped run is a
    successful run that was told when to stop.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class BudgetLimits:
    """When to stop. `None` means no limit of that kind; accounting continues regardless."""

    max_total_tokens: int | None = None
    max_api_calls: int | None = None
    max_wall_seconds: float | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> BudgetLimits:
        """Build from a Hydra config node. A missing block means no limits."""
        if cfg is None:
            return cls()
        fields = ("max_total_tokens", "max_api_calls", "max_wall_seconds")
        values = {k: cfg[k] for k in fields if k in cfg and cfg[k] is not None}
        limits = cls(**values)
        for name, value in asdict(limits).items():
            if value is not None and value <= 0:
                raise ValueError(
                    f"budget.{name} must be positive or null, got {value!r}. Use null to mean "
                    f"'no limit'."
                )
        return limits

    @property
    def any_set(self) -> bool:
        return any(v is not None for v in asdict(self).values())


@dataclass
class TokenSpend:
    """What has been consumed so far."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0


@dataclass
class NodeSpend:
    """Per-node spend, so cost per node can be read off rather than estimated."""

    index: int
    depth: int
    total_tokens: int = 0
    api_calls: int = 0


class RunBudget:
    """One per run. Attached to the engine, which records, and the tree, which stops."""

    def __init__(
        self,
        limits: BudgetLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limits = limits or BudgetLimits()
        self._clock = clock
        self._started = clock()
        self.spend = TokenSpend()
        self.nodes: list[NodeSpend] = []
        self.stopped_reason: str | None = None
        self._current: NodeSpend | None = None

    # ---- accounting ---------------------------------------------------------------------

    def start_node(self, *, index: int, depth: int) -> None:
        self._current = NodeSpend(index=index, depth=depth)
        self.nodes.append(self._current)

    def finish_node(self) -> None:
        self._current = None

    def record_response(self, response: Any) -> None:
        """Accumulate one API response. Call once per HTTP request, not per merged response."""
        self.spend.api_calls += 1
        if self._current is not None:
            self._current.api_calls += 1
        if response is None:
            return

        usage = response.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0) or (prompt + completion)
        details = usage.get("completion_tokens_details") or {}
        try:
            reasoning = int(details.get("reasoning_tokens", 0) or 0)
        except AttributeError:  # pragma: no cover - provider-shape dependent
            reasoning = 0

        self.spend.prompt_tokens += prompt
        self.spend.completion_tokens += completion
        self.spend.reasoning_tokens += reasoning
        self.spend.total_tokens += total
        if self._current is not None:
            self._current.total_tokens += total

    # ---- limits -------------------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started

    @property
    def remaining_tokens(self) -> int | None:
        if self.limits.max_total_tokens is None:
            return None
        return max(0, self.limits.max_total_tokens - self.spend.total_tokens)

    def exceeded_reason(self) -> str | None:
        """Which limit has been reached, if any. Pure query; raises nothing."""
        limits = self.limits
        if limits.max_total_tokens is not None and self.spend.total_tokens >= limits.max_total_tokens:
            return REASON_TOKENS
        if limits.max_api_calls is not None and self.spend.api_calls >= limits.max_api_calls:
            return REASON_API_CALLS
        if limits.max_wall_seconds is not None and self.elapsed_seconds >= limits.max_wall_seconds:
            return REASON_WALL_CLOCK
        return None

    @property
    def exhausted(self) -> bool:
        return self.exceeded_reason() is not None

    def check(self) -> None:
        """Raise `BudgetExceeded` if a limit has been reached. Called before dispatching work."""
        reason = self.exceeded_reason()
        if reason is None:
            return
        self.stopped_reason = reason
        raise BudgetExceeded(reason, self._describe(reason))

    def _describe(self, reason: str) -> str:
        if reason == REASON_TOKENS:
            detail = (
                f"{self.spend.total_tokens:,} tokens spent, limit "
                f"{self.limits.max_total_tokens:,}"
            )
        elif reason == REASON_API_CALLS:
            detail = f"{self.spend.api_calls:,} API calls made, limit {self.limits.max_api_calls:,}"
        else:
            detail = (
                f"{self.elapsed_seconds:.0f}s elapsed, limit {self.limits.max_wall_seconds:.0f}s"
            )
        return (
            f"Run budget reached ({reason}): {detail}. The tree stops growing here; every node "
            f"still queued becomes a leaf, so the result is a complete but shallower tree."
        )

    # ---- reporting ----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "limits": asdict(self.limits),
            "spend": asdict(self.spend),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "stopped_early": self.stopped_reason is not None,
            "stopped_reason": self.stopped_reason,
            "tokens_remaining": self.remaining_tokens,
            "mean_tokens_per_node": (
                round(self.spend.total_tokens / len(self.nodes)) if self.nodes else None
            ),
            "nodes": [asdict(n) for n in self.nodes],
        }

    def format_report(self) -> str:
        spend = self.spend
        limit = self.limits.max_total_tokens
        budget_line = (
            f"{spend.total_tokens:,}" if limit is None else f"{spend.total_tokens:,} of {limit:,}"
        )
        per_node = (
            f"{round(spend.total_tokens / len(self.nodes)):,}" if self.nodes else "n/a"
        )
        lines = [
            "Token usage",
            f"  total tokens                    : {budget_line}",
            f"    prompt                        : {spend.prompt_tokens:,}",
            f"    completion                    : {spend.completion_tokens:,}"
            + (f"  (reasoning {spend.reasoning_tokens:,})" if spend.reasoning_tokens else ""),
            f"  API calls                       : {spend.api_calls:,}",
            f"  mean tokens per node            : {per_node}",
            f"  wall clock                      : {self.elapsed_seconds:.0f}s",
        ]
        if self.stopped_reason:
            lines.append(f"  STOPPED EARLY                   : {self.stopped_reason}")
        return "\n".join(lines)
