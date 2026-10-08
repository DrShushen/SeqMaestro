"""A prompt the provider refuses must not end the run.

Azure's content filter declines individual prompts and reports it as `BadRequestError` -- the
same exception class LiteLLM raises for a genuinely malformed request. The engine treated the
whole class as fatal, which is right for a bad key or a missing deployment (wrong for every
call) and wrong for one flagged prompt (wrong for one call).

A tree died on exactly this, mid reflection round, after 1,848 successful requests.
At Pol II scale the same run issues on the order of 36,000, so surviving a flagged prompt is not
optional.

The drop is safe because it is still counted: a filtered request is recorded as a failure and
feeds `min_request_success_rate`, so systematic filtering still stops the run at `check_run`.
That boundary is what these tests pin down. Offline throughout.
"""

from __future__ import annotations

import asyncio

import litellm
import pytest
from litellm.exceptions import AuthenticationError, BadRequestError

from deft.llm.engines.litellm_engine import LiteLLMEngine, _is_content_filter_rejection
from deft.provenance import DegradedRunError, IntegrityPolicy, RunProvenance

MODEL_CONFIG = {
    "model": "azure/some-deployment",
    "base_model": "azure/gpt-5.4",
    "api_base": "https://example.openai.azure.com/",
    "api_key": "k",
}

FLAGGED = (
    "litellm.BadRequestError: AzureException BadRequestError - Invalid prompt: your prompt was "
    "flagged as potentially violating our usage policy. Please try again with a different "
    "prompt. LiteLLM Retried: 5 times"
)


def engine():
    return LiteLLMEngine(content="system", config=dict(MODEL_CONFIG))


def bad_request(message):
    return BadRequestError(message=message, model="azure/some-deployment", llm_provider="azure")


def run_one(monkeypatch, exception):
    """Drive `_single_request` with the network replaced by one raising call."""

    async def raising(**kwargs):
        raise exception

    monkeypatch.setattr(litellm, "acompletion", raising)
    return asyncio.run(engine()._single_request("a prompt", 1))


# ------------------------------------------------------------------ recognising the rejection


@pytest.mark.parametrize(
    "message",
    [
        FLAGGED,
        "The response was filtered due to the prompt triggering Azure OpenAI's content "
        "management policy.",
        "ResponsibleAIPolicyViolation",
        "error code: content_filter",
        "YOUR PROMPT WAS FLAGGED AS POTENTIALLY VIOLATING our usage policy",
    ],
)
def test_a_content_filter_rejection_is_recognised(message):
    assert _is_content_filter_rejection(bad_request(message))


@pytest.mark.parametrize(
    "message",
    [
        "Invalid value for 'n': must be less than or equal to 8",
        "unknown parameter: 'reasoning_effort'",
        "max_tokens is too large",
    ],
)
def test_an_ordinary_bad_request_is_not_mistaken_for_one(message):
    assert not _is_content_filter_rejection(bad_request(message))


# ------------------------------------------------------------------------ what the engine does


def test_a_flagged_prompt_is_dropped_rather_than_raised(monkeypatch):
    """The failure that killed the run: this must return, not raise."""
    assert run_one(monkeypatch, bad_request(FLAGGED)) is None


def test_a_malformed_request_still_stops_the_run(monkeypatch):
    """The behaviour the NON_RETRYABLE list exists for must survive the fix."""
    with pytest.raises(BadRequestError):
        run_one(monkeypatch, bad_request("Invalid value for 'n': must be <= 8"))


def test_a_bad_key_still_stops_the_run(monkeypatch):
    """Building a tree from naive features against a dead endpoint is the thing to prevent."""
    with pytest.raises(AuthenticationError):
        run_one(
            monkeypatch,
            AuthenticationError(
                message="invalid api key", model="azure/some-deployment", llm_provider="azure"
            ),
        )


def test_a_dropped_prompt_is_counted_as_a_failed_request(monkeypatch):
    """What makes dropping safe: systematic filtering must still fail the run-level check.

    A drop that went unrecorded would turn a filtered-out run into a quiet success built from
    naive features -- the exact outcome `NON_RETRYABLE` was introduced to prevent.
    """
    subject = engine()
    subject.provenance = RunProvenance(policy=IntegrityPolicy(min_request_success_rate=0.9))

    async def raising(**kwargs):
        raise bad_request(FLAGGED)

    monkeypatch.setattr(litellm, "acompletion", raising)
    asyncio.run(subject._single_request("a prompt", 1))

    assert subject.provenance.calls.requests_attempted == 1
    assert subject.provenance.calls.requests_succeeded == 0

    with pytest.raises(DegradedRunError):
        subject.provenance.check_run()
