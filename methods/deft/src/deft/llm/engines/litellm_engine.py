"""The one LLM engine, backed by LiteLLM.

This replaces four engine classes and three request adapters that hand-rolled what LiteLLM
does: per-provider request shapes, the `max_tokens` / `max_completion_tokens` split, dropping
parameters a reasoning model rejects, and a retry loop. It also removes the last exact
dependency pin, `openai==0.28.1`, which existed only because those engines used the pre-1.0 API
surface.

What is *not* LiteLLM's job, and is therefore still here:

    chunking       a provider can cap completions per request. Ours rejects `n > 8` with a
                   400. A larger ask is split across requests and merged, so a caller always
                   receives the number it asked for.
    recording      provenance and budget, per HTTP request -- see `GenericEngine._record_request`
    failing loudly non-retryable errors are re-raised rather than returning None, because a
                   silently degraded run still produces a plausible tree

## The Azure trap

With Azure the model string is `azure/<deployment_name>`, and deployment names are opaque --
ours is `gpt-5.4-es`, which appears in no model map. LiteLLM keys behaviour on that string, so
`base_model` supplies the real family as a hint. It is consulted by the capability gate
(`get_supported_openai_params`), which is what lets `reasoning_effort` through for a deployment
LiteLLM has never heard of.

Two things worth knowing, both established against live traffic:

- Azure does **not** echo the deployment name. It echoes a dated snapshot, e.g.
  `gpt-5.4-2026-03-05`. Anything that resolves a model from the response sees that, not the
  deployment name -- which is why cost tracking there breaks for deployments newer than the
  installed price map. We track tokens only, so this does not affect us; it is recorded because
  it is the opposite of what one would assume.
- `base_model` does **not** help on the Anthropic-on-Foundry path (`azure_ai/` + Claude), where
  routing and capability both key on the deployment name itself, and Foundry makes that
  immutable after creation. Such a deployment must be *named* `claude-opus-4-8-es` or similar.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import litellm
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)

from deft.llm.engines.generic_engine import GenericEngine

logger = logging.getLogger(__name__)

# LiteLLM prints a provider-support banner on first use; it is noise in a run log.
litellm.suppress_debug_info = True

# Retrying cannot help with these: a bad key, no access to the deployment, a deployment that
# does not exist, or a request the provider rejects outright. Upstream returned None here, which
# let DEFT fall back to naive features and "succeed" against a dead endpoint.
NON_RETRYABLE = (AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError)

# One `BadRequestError` covers two opposite situations, and they need opposite handling.
#
# A malformed or misconfigured request is wrong for *every* call, so it must stop the run -- that
# is what the list above is for. A content-filter rejection is wrong for *one prompt*: the
# provider declined this particular text and the next one usually succeeds. Aborting a run for
# that is the same mistake as aborting for a single unparseable completion, and it is not
# hypothetical at scale -- a Pol II run issues on the order of 36,000 requests, so a run that
# cannot survive one flagged prompt cannot finish.
#
# Dropping is safe here only because the failure is still counted: `_record_request(None, n)`
# feeds `min_request_success_rate`, so a run that is filtered *systematically* still fails loudly
# at `check_run`. Occasional drops continue; widespread ones stop the run.
CONTENT_FILTER_MARKERS = (
    "content_filter",
    "content management policy",
    "responsibleaipolicyviolation",
    "violating our usage policy",
    "flagged as potentially violating",
)


def _is_content_filter_rejection(exc: Exception) -> bool:
    """Whether the provider declined this specific prompt rather than the request as such.

    Matched on the message because LiteLLM flattens provider error bodies into the string and
    does not surface a stable machine-readable code across providers. Deliberately broad: a
    false positive here degrades to dropping one prompt, which the success-rate check catches,
    whereas a false negative loses the whole run.
    """
    text = str(exc).lower()
    return any(marker in text for marker in CONTENT_FILTER_MARKERS)


class LiteLLMEngine(GenericEngine):
    """Provider-agnostic engine. Everything provider-specific is LiteLLM's problem."""

    def __init__(
        self,
        content: str,
        config: Any = None,
        max_tokens: int | None = 4000,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        max_n_per_request: int | None = None,
        num_retries: int = 5,
        request_timeout: int = 60,
        extra_params: dict | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            config: the `llm_models` block -- `model`, and optionally `base_model`, `api_base`,
                `api_key`, `api_version`. `model` is a LiteLLM model string, so it carries its
                own provider prefix (`azure/<deployment>`).
            max_n_per_request: the provider's ceiling on completions per request, or None if it
                has none. A larger ask is split and merged; this is a cap, not a fixed count.
            num_retries: handled by LiteLLM, which backs off on rate limits and timeouts.
            extra_params: passed through to `litellm.acompletion`, last, so it can override
                anything above.
        """
        super().__init__(content=content, config=config)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        if max_n_per_request is not None and max_n_per_request < 1:
            raise ValueError(
                f"max_n_per_request must be at least 1, got {max_n_per_request}."
            )
        self.max_n_per_request = max_n_per_request
        self.num_retries = num_retries
        self.request_timeout = request_timeout
        self.extra_params = dict(extra_params or {})

    # ---- config -----------------------------------------------------------------------

    @staticmethod
    def _cfg_get(cfg, key: str, default=None):
        if cfg is None:
            return default
        try:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            value = cfg.get(key)
        except Exception:
            value = getattr(cfg, key, default)
        return default if value is None else value

    def _model_kwargs(self) -> dict[str, Any]:
        """Model and credentials, as LiteLLM wants them."""
        if self.config is None:
            raise ValueError("Missing engine config: no `llm_models` block was provided.")
        model = self._cfg_get(self.config, "model")
        if not model:
            raise ValueError(
                "Engine config has no `model`. It must be a LiteLLM model string carrying its "
                "provider prefix, e.g. `azure/my-deployment`."
            )
        kwargs: dict[str, Any] = {"model": str(model)}
        for key in ("base_model", "api_base", "api_key", "api_version"):
            value = self._cfg_get(self.config, key)
            if value:
                kwargs[key] = str(value)
        return kwargs

    # ---- chunking ---------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(total: int, max_per_request: int | None) -> list[int]:
        """Split a completion count into per-request chunks: 20 under a cap of 8 -> [8, 8, 4]."""
        total = max(1, int(total))
        if not max_per_request or max_per_request >= total:
            return [total]
        chunks = [max_per_request] * (total // max_per_request)
        remainder = total % max_per_request
        if remainder:
            chunks.append(remainder)
        return chunks

    @staticmethod
    def _merge_responses(responses: list[dict]) -> dict:
        """Merge several responses into one, as though a single request had returned them all.

        Everything downstream expects exactly one response object per prompt: `parsing_end2end`
        walks `response["choices"]`, and `extract_generations` reads
        `response["usage"]["total_tokens"]`.

        Token counts are summed rather than taken from the first chunk, because every chunk
        re-sends the prompt and is billed for it.
        """
        merged = responses[0]
        choices = [choice for response in responses for choice in response["choices"]]
        for position, choice in enumerate(choices):
            # Renumbering is cosmetic; a provider whose choices are not writable dicts is
            # not worth failing a merge over.
            with contextlib.suppress(Exception):  # pragma: no cover - provider-shape dependent
                choice["index"] = position
        merged["choices"] = choices

        if len(responses) > 1:
            usage = merged.get("usage")
            if usage is not None:
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[field] = sum(
                        int((response.get("usage") or {}).get(field, 0) or 0)
                        for response in responses
                    )
                details = usage.get("completion_tokens_details")
                if details is not None:
                    with contextlib.suppress(Exception):  # pragma: no cover
                        details["reasoning_tokens"] = sum(
                            int(
                                (
                                    (response.get("usage") or {}).get(
                                        "completion_tokens_details"
                                    )
                                    or {}
                                ).get("reasoning_tokens", 0)
                                or 0
                            )
                            for response in responses
                        )
        return merged

    # ---- requests ---------------------------------------------------------------------

    def _build_request_kwargs(self, user_message: str, n: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": self.content},
                {"role": "user", "content": user_message},
            ],
            "n": int(n),
            "timeout": self.request_timeout,
            "num_retries": self.num_retries,
            # Reasoning models reject `temperature`, `top_p` and `stop`, and take
            # `max_completion_tokens` rather than `max_tokens`. Letting LiteLLM drop and
            # translate is precisely what removes the need for per-provider adapters.
            "drop_params": True,
            **self._model_kwargs(),
        }
        for key, value in (
            ("max_tokens", self.max_tokens),
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("reasoning_effort", self.reasoning_effort),
        ):
            if value is not None:
                kwargs[key] = value
        kwargs.update(self.extra_params)
        return kwargs

    async def _single_request(self, user_message: str, n: int) -> dict | None:
        """One request to the provider. Returns a plain dict, or None if it failed.

        The response is normalised to a plain dict at this boundary, so nothing downstream --
        parsing, merging, provenance, budget -- depends on LiteLLM's object model.
        """
        response: dict | None = None
        try:
            raw = await litellm.acompletion(**self._build_request_kwargs(user_message, n))
            response = raw.model_dump()
        except NON_RETRYABLE as exc:
            if isinstance(exc, BadRequestError) and _is_content_filter_rejection(exc):
                # One prompt declined, not a broken endpoint. Dropped and counted, so a run that
                # is filtered systematically still fails at the run-level success-rate check.
                logger.warning(
                    "[LLM] Prompt refused by the provider's content filter; dropping it "
                    "and continuing: %s",
                    exc,
                )
                response = None
            else:
                # Re-raised deliberately. Returning None here would let DEFT build a whole tree
                # out of naive features against a dead or misconfigured endpoint, and call it a
                # result.
                logger.error("[LLM] Non-retryable API error, aborting the run.")
                raise
        except Exception as exc:
            logger.warning(
                "[LLM] Request failed after %d retries, dropping it: %s: %s",
                self.num_retries,
                type(exc).__name__,
                exc,
            )
            response = None

        self._record_request(response, n)
        return response

    async def _async_generate(self, user_message: str, n_generations_per_prompt: int):
        """Return exactly `n_generations_per_prompt` completions, whatever the provider caps at.

        Chunks are issued sequentially rather than concurrently: `query_llm` already fans out
        across prompts, and stacking a second layer of concurrency multiplies rate-limit
        pressure for little gain.

        Returns None only when every chunk failed; a partial result beats none.
        """
        chunks = self._split_into_chunks(n_generations_per_prompt, self.max_n_per_request)
        if len(chunks) > 1:
            logger.info(
                "Requesting %d completions as %d requests of %s (provider cap %s).",
                n_generations_per_prompt,
                len(chunks),
                chunks,
                self.max_n_per_request,
            )

        responses: list[dict] = []
        for position, chunk in enumerate(chunks):
            # Re-check between chunks so a long split ask stops as soon as the budget runs out
            # rather than at the end. Returning what we have is honest; the next `query_llm`
            # will raise and the tree will stop growing.
            if self.budget is not None and position and self.budget.exhausted:
                logger.warning(
                    "Run budget reached mid-prompt; returning %d of the %d completions "
                    "requested.",
                    sum(len(r["choices"]) for r in responses),
                    n_generations_per_prompt,
                )
                break
            response = await self._single_request(user_message, chunk)
            if response is not None:
                responses.append(response)

        if not responses:
            return None

        if len(responses) < len(chunks):
            logger.warning(
                "Only %d of %d requests for this prompt succeeded; continuing with %d "
                "completions instead of the %d requested.",
                len(responses),
                len(chunks),
                sum(len(response["choices"]) for response in responses),
                n_generations_per_prompt,
            )

        return self._merge_responses(responses)

    async def query_llm(self, list_prompts: list, n_generations_per_prompt: int):
        """Generate for every prompt concurrently. One response object per prompt, or None.

        Checks the run budget before dispatching. Because the batch goes out concurrently,
        spend can overshoot the limit by roughly one batch of in-flight requests -- the report
        shows what was actually spent, so the overshoot is visible rather than hidden.
        """
        if self.budget is not None:
            self.budget.check()  # raises BudgetExceeded; the tree catches it and stops growing

        results = await asyncio.gather(
            *(
                self._async_generate(prompt, n_generations_per_prompt)
                for prompt in list_prompts
            )
        )
        await self._drain_litellm_logging()
        return list(results)

    @staticmethod
    async def _drain_litellm_logging() -> None:
        """Let LiteLLM's background logging worker finish before the event loop closes.

        The feature finder runs every batch on one event loop and closes it when the tree is
        fitted (`LLMFeatureFinder._ask`), and a caller outside a run may use a loop of its own
        for a single batch. Draining after every batch leaves nothing queued when either loop
        closes. Without it the worker's queued coroutines are collected un-awaited and Python
        emits a RuntimeWarning per call -- harmless, but it would bury the run log. Best-effort:
        a LiteLLM upgrade that renames this simply restores the warning.
        """
        try:
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

            await GLOBAL_LOGGING_WORKER.stop()
        except Exception:  # pragma: no cover - depends on LiteLLM internals
            pass
