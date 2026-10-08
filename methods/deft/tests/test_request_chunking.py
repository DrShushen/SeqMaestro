"""A caller must receive exactly the number of completions it asked for.

Providers cap how many completions one request may ask for -- the GPT-5 deployment rejects
`n > 8`. The adapter used to handle that by applying its own fixed `n` to every request and
discarding the caller's argument. Two things followed:

    * a configured population of 20 silently became 8, so the paper's stated population size
      was never what this backend produced;
    * a code-generation request that asked for 1 completion was billed for 8, and seven were
      thrown away -- measured as an 8x overcharge on that half of the workload.

The engine now splits a larger ask across several requests and merges them. These tests run
offline.
"""

from __future__ import annotations

import asyncio

import pytest

from deft.llm.engines.litellm_engine import LiteLLMEngine

split = LiteLLMEngine._split_into_chunks
merge = LiteLLMEngine._merge_responses

MODEL_CONFIG = {"model": "azure/some-deployment", "base_model": "azure/gpt-5.4",
                "api_base": "https://example.openai.azure.com/", "api_key": "k"}


def make_response(n_choices: int, prompt_tokens: int = 100, completion_tokens: int = 10):
    return {
        "choices": [
            {"index": i, "message": {"content": f"completion {i}"}} for i in range(n_choices)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens * n_choices,
            "total_tokens": prompt_tokens + completion_tokens * n_choices,
            "completion_tokens_details": {"reasoning_tokens": 5 * n_choices},
        },
    }


# --------------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total, cap, expected",
    [
        (20, 8, [8, 8, 4]),   # the configured population size against the real cap
        (16, 8, [8, 8]),      # exact multiple, no stray zero-sized chunk
        (8, 8, [8]),
        (4, 8, [4]),          # under the cap: one request, unchanged behaviour
        (1, 8, [1]),          # code generation: one completion, one request, not eight
        (20, None, [20]),     # provider declares no cap
        (1, None, [1]),
        (0, 8, [1]),          # degenerate input still yields one request
    ],
)
def test_split_into_chunks(total, cap, expected):
    assert split(total, cap) == expected


@pytest.mark.parametrize("total, cap", [(20, 8), (16, 8), (7, 3), (1, 1), (100, 8)])
def test_chunks_always_sum_to_the_request_and_respect_the_cap(total, cap):
    chunks = split(total, cap)
    assert sum(chunks) == total
    assert all(0 < c <= cap for c in chunks)


# --------------------------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------------------------


def test_merge_concatenates_choices_and_reindexes():
    merged = merge([make_response(8), make_response(8), make_response(4)])

    assert len(merged["choices"]) == 20
    assert [c["index"] for c in merged["choices"]] == list(range(20))


def test_merge_sums_token_usage_across_requests():
    """Each chunk re-sends the prompt and is billed for it, so prompt tokens must add up."""
    merged = merge([make_response(8), make_response(8), make_response(4)])

    usage = merged["usage"]
    assert usage["prompt_tokens"] == 300, "three requests, each billed for the prompt"
    assert usage["completion_tokens"] == 200
    assert usage["total_tokens"] == 500
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 100


def test_merge_of_a_single_response_is_a_no_op():
    original = make_response(8)
    merged = merge([original])

    assert len(merged["choices"]) == 8
    assert merged["usage"]["total_tokens"] == original["usage"]["total_tokens"]


def test_merged_response_keeps_the_shape_downstream_expects():
    """`parsing_end2end` and `extract_generations` read exactly these paths."""
    merged = merge([make_response(2), make_response(2)])

    assert merged["usage"]["total_tokens"] > 0
    assert all(c["message"]["content"] for c in merged["choices"])


# --------------------------------------------------------------------------------------------
# End to end through the engine
# --------------------------------------------------------------------------------------------


def drive(engine, requested, responder):
    """Run `_async_generate` with the network replaced, returning (response, chunks issued)."""
    issued = []

    async def fake_single_request(user_message, n):
        issued.append(n)
        return responder(n)

    engine._single_request = fake_single_request
    response = asyncio.run(engine._async_generate("a prompt", requested))
    return response, issued


def make_engine(max_n_per_request=8):
    return LiteLLMEngine(
        content="system", config=dict(MODEL_CONFIG), max_n_per_request=max_n_per_request
    )


def test_caller_receives_every_completion_it_asked_for():
    response, issued = drive(make_engine(), requested=20, responder=make_response)

    assert issued == [8, 8, 4], "should have been split across three requests"
    assert len(response["choices"]) == 20, "the caller asked for 20 and must receive 20"


def test_a_single_completion_costs_a_single_completion():
    """The 8x overcharge on code generation: one asked for, one requested, one returned."""
    response, issued = drive(make_engine(), requested=1, responder=make_response)

    assert issued == [1]
    assert len(response["choices"]) == 1


def test_partial_failure_returns_what_succeeded():
    """One chunk of three fails; the caller still gets the other two rather than nothing."""
    outcomes = iter([make_response(8), None, make_response(4)])

    response, issued = drive(make_engine(), requested=20, responder=lambda n: next(outcomes))

    assert issued == [8, 8, 4]
    assert response is not None, "a partial result beats none"
    assert len(response["choices"]) == 12


def test_total_failure_returns_none():
    response, issued = drive(make_engine(), requested=20, responder=lambda n: None)

    assert issued == [8, 8, 4]
    assert response is None, "the existing contract: None when nothing came back"


def test_no_cap_means_one_request():
    _, issued = drive(make_engine(max_n_per_request=None), requested=20,
                      responder=make_response)

    assert issued == [20]


# --------------------------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------------------------


def test_request_carries_the_count_the_engine_was_asked_for():
    kwargs = make_engine()._build_request_kwargs("a prompt", 3)

    assert kwargs["n"] == 3, "the engine must not substitute its own count"
    assert kwargs["model"] == "azure/some-deployment"
    assert kwargs["base_model"] == "azure/gpt-5.4", (
        "base_model is what lets reasoning_effort through for an opaque Azure deployment"
    )


def test_provider_quirks_are_delegated_rather_than_hand_rolled():
    """`drop_params` is what replaced the per-provider request adapters.

    A reasoning model rejects `temperature`, `top_p` and `stop`, and wants
    `max_completion_tokens` rather than `max_tokens`. We send the generic form and let LiteLLM
    drop and translate, instead of maintaining a request shape per provider.
    """
    engine = LiteLLMEngine(
        content="s", config=dict(MODEL_CONFIG), temperature=1, top_p=0.95, max_tokens=4000,
        reasoning_effort="minimal",
    )
    kwargs = engine._build_request_kwargs("a prompt", 1)

    assert kwargs["drop_params"] is True
    assert kwargs["max_tokens"] == 4000
    assert kwargs["reasoning_effort"] == "minimal"


def test_unset_parameters_are_not_sent_at_all():
    """A None must not become an explicit null the provider then rejects."""
    kwargs = LiteLLMEngine(
        content="s", config=dict(MODEL_CONFIG), temperature=None, top_p=None,
        reasoning_effort=None,
    )._build_request_kwargs("a prompt", 1)

    for absent in ("temperature", "top_p", "reasoning_effort"):
        assert absent not in kwargs


def test_extra_params_win():
    """The documented escape hatch, applied last."""
    kwargs = LiteLLMEngine(
        content="s", config=dict(MODEL_CONFIG), max_tokens=10,
        extra_params={"max_tokens": 999, "custom": "x"},
    )._build_request_kwargs("a prompt", 1)

    assert kwargs["max_tokens"] == 999
    assert kwargs["custom"] == "x"


def test_a_nonsense_cap_is_rejected_at_construction():
    with pytest.raises(ValueError, match="at least 1"):
        LiteLLMEngine(content="s", config=dict(MODEL_CONFIG), max_n_per_request=0)


def test_no_cap_declared_by_default():
    assert LiteLLMEngine(content="s", config=dict(MODEL_CONFIG)).max_n_per_request is None


def test_a_config_without_a_model_fails_clearly():
    with pytest.raises(ValueError, match="provider prefix"):
        LiteLLMEngine(content="s", config={"api_key": "k"})._build_request_kwargs("p", 1)
