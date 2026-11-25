"""What the proposer, the judge and the recurrence analysis share: how to call the language model
and what to tell it about the task.

Neither belongs to one consumer. The provider and its quirks are configured once, so that every
arm is budget-matched against one model, and `TaskContext` is a single type, so that a dataset is
described the same way to every consumer.

The code here does not know what a feature is, what a score means, or which arm is running.
"""

from __future__ import annotations

from seqmaestro.llm.client import (
    PROVIDERS,
    REASONING_EFFORTS,
    CallSettings,
    LiteLLMClient,
    ScriptedClient,
    describe_calls,
    missing_settings,
    provider_from_environment,
    reply_conditions,
)
from seqmaestro.llm.context import TaskContext, blinding_leaks, guidance_leaks, load_task

__all__ = [
    "PROVIDERS",
    "REASONING_EFFORTS",
    "CallSettings",
    "LiteLLMClient",
    "ScriptedClient",
    "TaskContext",
    "blinding_leaks",
    "describe_calls",
    "guidance_leaks",
    "load_task",
    "missing_settings",
    "provider_from_environment",
    "reply_conditions",
]
