"""Base for LLM engines: the cross-cutting concerns that are not the LLM client's job.

There is only one engine now, `litellm_engine.LiteLLMEngine`. This base exists so that three
things survive any change to it, because each was a defect once:

    credential redaction   a fitted tree reaches the engine through the feature finder, so
                           `dill.dump(tree)` used to write the API key into the tree artifact
    provenance recording   a request that fails after retries returns None, and that loss is
                           otherwise invisible -- see `src/deft/provenance.py`
    budget recording       per HTTP request, so a split request is not double-counted

Anything provider-specific belongs in LiteLLM, not here.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

REDACTED = "<redacted>"

# Config keys whose values must never reach a serialized artifact.
_SECRET_CONFIG_KEYS = frozenset({"api_key", "key", "token", "password", "secret"})


def _redact(obj):
    """Recursively replace secret-looking values in a plain container."""
    if isinstance(obj, dict):
        return {
            k: REDACTED if str(k).lower() in _SECRET_CONFIG_KEYS else _redact(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def redact_engine_config(cfg):
    """Return a credential-free copy of an engine's model config, safe to serialize.

    Falls back to dropping the config entirely if it cannot be converted, since leaking a key
    is far worse than losing a provenance detail.
    """
    if cfg is None:
        return None
    try:
        container = (
            OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg
        )
        return _redact(container)
    except Exception:
        return REDACTED


class GenericEngine:
    """Provider-agnostic base. Holds the system prompt and the run-level recorders."""

    def __init__(self, content: str, config: Any = None) -> None:
        """Initialize the instance."""
        self.content = content
        self.config = config

        # Set by `run_experiment` after instantiation, to a `deft.provenance.RunProvenance` and
        # a `deft.budget.RunBudget`. Left as None when the engine is used standalone (tests,
        # notebooks), in which case nothing is recorded and no check or limit applies.
        self.provenance = None
        self.budget = None

    def __getstate__(self):
        """Strip credentials before this engine is serialized.

        A fitted tree holds the feature finder, which holds this engine, which holds the
        resolved model config -- so `dill.dump(tree)` writes the API key into the tree artifact
        in plaintext. Tree artifacts are the primary output of a run and get committed and
        shared, so the redaction belongs here rather than at each dump site, where it would
        eventually be forgotten.

        Every attribute is redacted, not only `config`. Redacting `config` alone left a hole:
        `LiteLLMEngine.extra_params` is passed straight through to the provider, so a key set
        there -- which is a documented way to reach a parameter the engine does not model --
        was serialized in plaintext. Verified against a real dilled engine.

        A deserialized engine cannot make API calls. That is intended: a saved tree is a
        record, not a live client.
        """
        state = self.__dict__.copy()
        state["config"] = redact_engine_config(state.get("config"))
        return {key: _redact(value) for key, value in state.items()}

    # ---- recording ------------------------------------------------------------------------

    def _record_request(self, response: dict | None, n_generations_per_prompt: int) -> None:
        """Report one HTTP request's outcome to the run's recorders.

        Call once per request, never on a merged response: merging sums usage across chunks, so
        counting there as well would double-count a split request.

        A `response` of None means the request was retried to exhaustion and then dropped.
        Without this, that loss is invisible -- the caller just receives a shorter candidate
        list. See `src/deft/conf/integrity/strict.yaml`.
        """
        if self.provenance is not None:
            returned = 0
            if response is not None:
                try:
                    returned = len(response["choices"])
                except (KeyError, TypeError):
                    returned = 0
            self.provenance.record_request(
                succeeded=response is not None,
                requested=int(n_generations_per_prompt),
                returned=returned,
            )
        if self.budget is not None:
            self.budget.record_response(response)
