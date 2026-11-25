"""What the agent sends a model, by the kind of model: its connection settings, and a conversation
in the shape the Claude API takes."""

from types import SimpleNamespace

import pytest

from igsr.agent.agent import (
    ANTHROPIC,
    CONTINUING_TURN,
    OPENING_TURN,
    LiteLLMAgent,
    completion_settings,
    retry_settings,
    with_user_turns,
)

SYSTEM = {"role": "system", "content": "the task"}


def test_each_kind_of_model_gets_the_settings_it_takes():
    llm = SimpleNamespace(kind="azure", endpoint="https://x.invalid/", api_version="v1")
    assert completion_settings(llm, "k") == {"api_base": "https://x.invalid/", "api_version": "v1", "api_key": "k"}
    llm.kind = "azure_ai"
    assert completion_settings(llm, "k") == {"api_base": "https://x.invalid/", "api_key": "k"}
    for kind in ("openai", ANTHROPIC):
        assert completion_settings(SimpleNamespace(kind=kind), "k") == {"api_key": "k"}
    with pytest.raises(ValueError, match="Invalid LLM kind: other"):
        completion_settings(SimpleNamespace(kind="other"), "k")


def test_a_reasoning_effort_the_config_states_is_passed_on():
    """A Foundry deployment's name is in no model map, so LiteLLM would drop the effort unless told
    to pass it on. A config that states none sends none."""
    llm = SimpleNamespace(kind="azure_ai", endpoint="https://x.invalid/", reasoning_effort="none")
    assert completion_settings(llm, "k") == {"api_base": "https://x.invalid/", "api_key": "k",
                                             "reasoning_effort": "none",
                                             "allowed_openai_params": ["reasoning_effort"]}
    llm.reasoning_effort = None
    assert "reasoning_effort" not in completion_settings(llm, "k")


def test_a_wait_the_config_states_reaches_every_call():
    llm = SimpleNamespace(kind="openai", request_timeout=90)
    assert completion_settings(llm, "k") == {"api_key": "k", "timeout": 90}
    llm.request_timeout = None
    assert "timeout" not in completion_settings(llm, "k")


def test_retries_the_config_states_are_the_agents_after_the_first_attempt():
    """The agent's `max_retries` counts every attempt, the first included, and LiteLLM is asked
    for none, so a config's count of retries is what a failed call gets."""
    assert retry_settings(SimpleNamespace(num_retries=5)).max_retries == 6
    assert retry_settings(SimpleNamespace(num_retries=0)).max_retries == 1
    assert retry_settings(SimpleNamespace(num_retries=None)) is None
    assert retry_settings(SimpleNamespace()) is None
    assert "num_retries" not in completion_settings(SimpleNamespace(kind="openai", num_retries=5), "k")


def test_a_conversation_with_no_user_turn_is_opened_by_one():
    assert with_user_turns([SYSTEM]) == [SYSTEM, {"role": "user", "content": OPENING_TURN}]


def test_the_opening_turn_stays_where_it_was_on_every_later_call():
    """The agent keeps the model's replies and its own reminders and not the opening turn, so the
    turn is put back at the same place each time."""
    reply, reminder = {"role": "assistant", "content": "a"}, {"role": "user", "content": "format"}
    assert with_user_turns([SYSTEM, reply, reminder]) == [
        SYSTEM, {"role": "user", "content": OPENING_TURN}, reply, reminder]


def test_a_conversation_that_ends_on_the_models_turn_is_continued_by_the_user():
    first, reply = {"role": "user", "content": "go"}, {"role": "assistant", "content": "a"}
    assert with_user_turns([SYSTEM, first, reply]) == [
        SYSTEM, first, reply, {"role": "user", "content": CONTINUING_TURN}]


def test_a_conversation_already_in_shape_is_left_as_it_is():
    messages = [SYSTEM, {"role": "user", "content": "go"}]
    assert with_user_turns(messages) == messages


@pytest.mark.parametrize("model, shaped", [("anthropic/some-model", True), ("openai/some-model", False),
                                           ("azure/some-deployment", False)])
def test_only_a_claude_model_is_sent_the_added_turns(monkeypatch, model, shaped):
    sent = {}

    def completion(**arguments):
        sent.update(arguments)
        return "reply"

    monkeypatch.setattr("igsr.agent.agent.litellm.completion", completion)
    agent = LiteLLMAgent(task_description="t", model=model, completion_kwargs={"api_key": "k"})
    assert agent._create_chat_completion([SYSTEM], {"api_key": "k"}) == "reply"
    assert sent["model"] == model and sent["api_key"] == "k"
    assert sent["messages"] == ([SYSTEM, {"role": "user", "content": OPENING_TURN}] if shaped else [SYSTEM])
