"""The two parsing paths from an API response to a `FeatureInfo` (`src/deft/utils/parsing.py`).

These had no offline coverage, which let a refactor delete a helper both paths call: every
offline test passed and the break surfaced only on a live run, after tokens had been spent.
That is the wrong order. Nothing here touches the network.

`extract_generations` also carries the token accounting the budget is built on, so it is
checked here too.
"""

from __future__ import annotations

import pytest

from deft.provenance import ORIGIN_LLM
from deft.utils.parsing import (
    clean_code_string,
    clean_json_string,
    extract_generations,
    parse_code_feature,
    parse_proposed_feature,
)


def response(*contents: str, total_tokens: int = 10) -> dict:
    """The shape `LiteLLMEngine` normalises every provider response to."""
    return {
        "choices": [{"message": {"content": c}} for c in contents],
        "usage": {"total_tokens": total_tokens, "prompt_tokens": 1, "completion_tokens": 1},
    }


PROPOSAL = '{"rationale": "GC is informative", "description": "GC fraction", "name": "gc_frac"}'
CODE = "def gc_frac(X):\n    return X['raw_sequence'].str.count('G')"


# ------------------------------------------------------------------------------- proposals


def test_a_proposal_becomes_a_feature_marked_as_llm_origin():
    """`origin` is what the integrity policy counts, so it must survive parsing."""
    parsed, tokens = parse_proposed_feature([response(PROPOSAL)])

    feature = parsed[0][0]
    assert feature.name == "gc_frac"
    assert feature.description == "GC fraction"
    assert feature.rationale == "GC is informative"
    assert feature.origin == ORIGIN_LLM
    assert tokens == 10


def test_the_rationale_can_be_suppressed():
    parsed, _ = parse_proposed_feature([response(PROPOSAL)], output_rationale=False)

    assert parsed[0][0].rationale == ""


def test_several_completions_for_one_prompt_all_parse():
    parsed, _ = parse_proposed_feature([response(PROPOSAL, PROPOSAL, PROPOSAL)])

    assert len(parsed[0]) == 3


def test_an_unparseable_proposal_is_dropped_and_the_rest_survive():
    """One malformed completion must not lose the whole batch."""
    parsed, _ = parse_proposed_feature([response(PROPOSAL, "not json at all", PROPOSAL)])

    assert len(parsed[0]) == 2


def test_a_failed_request_contributes_nothing_rather_than_raising():
    parsed, tokens = parse_proposed_feature([None, response(PROPOSAL)])

    assert len(parsed) == 1, "only the successful response yields a group"
    assert tokens == 10


# ------------------------------------------------------------------------------------ code


def test_code_is_compiled_and_attached_to_the_proposed_feature():
    proposals, _ = parse_proposed_feature([response(PROPOSAL)])
    features, _ = parse_code_feature([response(CODE)], [proposals[0][0]])

    feature = features[0][0]
    assert callable(feature.fn)
    assert feature.name == "gc_frac", "the semantic name is carried over, not re-derived"
    assert feature.string.startswith("def gc_frac(X):")
    assert feature.origin == ORIGIN_LLM


def test_the_compiled_function_runs():
    import pandas as pd

    proposals, _ = parse_proposed_feature([response(PROPOSAL)])
    features, _ = parse_code_feature([response(CODE)], [proposals[0][0]])

    X = pd.DataFrame({"raw_sequence": ["GGA", "ATT"]})
    assert list(features[0][0].fn(X)) == [2, 0]


def test_code_that_does_not_compile_is_dropped_not_raised():
    """This is the silent-loss path: it must stay a drop, but the caller sees a shorter list."""
    proposals, _ = parse_proposed_feature([response(PROPOSAL)])
    features, _ = parse_code_feature([response("def broken-name(X):\n    return X")], [proposals[0][0]])

    assert features == [[]]


def test_a_completion_with_no_function_is_dropped_not_raised():
    """The failure that cost a full run. `clean_code_string` was called outside the guard that
    exists to drop unusable candidates, so one prose-only completion among thousands raised out
    of the parse loop and took the whole tree fit with it, hours in."""
    proposals, _ = parse_proposed_feature([response(PROPOSAL)])
    features, _ = parse_code_feature([response("I cannot help with that.")], [proposals[0][0]])

    assert features == [[]]


def test_one_prose_completion_does_not_lose_the_rest_of_its_batch():
    proposals, _ = parse_proposed_feature([response(PROPOSAL)])
    batch = response(CODE, "I cannot help with that.", CODE)

    features, _ = parse_code_feature([batch], [proposals[0][0]])

    assert [f.name for f in features[0]] == ["gc_frac", "gc_frac"]


def test_code_rejected_by_the_guard_is_dropped_not_raised():
    proposals, _ = parse_proposed_feature([response(PROPOSAL)])
    hostile = "def f(X):\n    return __import__('os').listdir('.')"
    features, _ = parse_code_feature([response(hostile)], [proposals[0][0]])

    assert features == [[]]


def test_a_mismatched_response_count_is_an_error_not_a_silent_misalignment():
    """Responses are zipped to proposals by index; a mismatch would attach the wrong code to
    the wrong feature."""
    proposals, _ = parse_proposed_feature([response(PROPOSAL)])

    with pytest.raises(AssertionError):
        parse_code_feature([response(CODE), response(CODE)], [proposals[0][0]])


# ------------------------------------------------------------------------------- cleaning


@pytest.mark.parametrize(
    "raw",
    [
        "```python\ndef f(X):\n    return X\n```",
        "Here is the function:\ndef f(X):\n    return X",
        "def f(X):\n    return X",
    ],
)
def test_prose_and_fences_around_the_function_are_stripped(raw):
    assert clean_code_string(raw).startswith("def f(X):")


def test_a_response_with_no_function_is_rejected():
    with pytest.raises(ValueError, match="No function definition"):
        clean_code_string("I cannot help with that.")


def test_json_is_extracted_from_surrounding_prose():
    assert clean_json_string('Sure! {"name": "x"} Hope that helps.') == '{"name": "x"}'


# ------------------------------------------------------------------------ token accounting


def test_tokens_are_summed_across_responses():
    """The budget is built on this number."""
    _, _, tokens = extract_generations(
        [response(PROPOSAL, total_tokens=7), response(PROPOSAL, total_tokens=11)]
    )

    assert tokens == 18


def test_indices_track_which_prompt_each_group_came_from():
    """`parse_code_feature` uses these to pair code with its proposal."""
    _, indices, _ = extract_generations([None, response(PROPOSAL), None, response(PROPOSAL)])

    assert indices == [1, 3]
