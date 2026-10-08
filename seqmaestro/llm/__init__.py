"""What the proposer, the judge and the recurrence analysis share: how to call the model, and what
to tell it about the task.

Neither belongs to one consumer. The provider and its quirks are configured once, so every arm
stays budget-matched against one model, and `TaskContext` is one type, so a dataset is described
the same way to every consumer.

Nothing here knows what a feature is, what a score means, or which arm is running.
"""

from __future__ import annotations

from seqmaestro.llm.client import (
    PROVIDERS,
    LiteLLMClient,
    ScriptedClient,
    missing_settings,
    provider_from_environment,
)
from seqmaestro.llm.context import TaskContext, blinding_leaks, load_task

__all__ = [
    "PROVIDERS",
    "LiteLLMClient",
    "ScriptedClient",
    "TaskContext",
    "blinding_leaks",
    "load_task",
    "missing_settings",
    "provider_from_environment",
]
