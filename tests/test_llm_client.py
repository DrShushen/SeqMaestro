"""The client's providers: which one a run calls, what a call to each states, what a run records
of it, and where a key goes.

The environment names one provider for every part of a run, and the run's config says how every
call is made. The failures these tests guard against are quiet: a setting a provider refuses fails
every call of an arm, a config for one provider that reads another's key runs under the wrong
account, a method called under different settings from the rest of the run is compared unfairly,
and a key kept on the client ends up in a run's files. The tests do not call a language model.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from seqmaestro.llm import client as llm
from seqmaestro.llm import missing_settings, provider_from_environment
from seqmaestro.pipeline.vendored import deft, igsr
from seqmaestro.pipeline.vendored.handover import call_overrides

REPO = Path(__file__).resolve().parents[1]
DEFT_CONF = REPO / "methods" / "deft" / "src" / "deft" / "conf"
IGSR_CONF = REPO / "methods" / "igsr" / "src" / "igsr" / "conf"

AZURE = {llm.ENV_PROVIDER: "azure", llm.ENV_KEY: "azure-key", llm.ENV_DEPLOYMENT: "my-deployment",
         llm.ENV_ENDPOINT: "https://example.invalid/"}
FOUNDRY = {llm.ENV_PROVIDER: "azure_foundry", llm.AZURE_FOUNDRY.key: "foundry-key",
           llm.AZURE_FOUNDRY.model: "an-open-model", "SEQMAESTRO_AZURE_FOUNDRY_ENDPOINT":
           "https://example.services.ai.azure.com/"}
OPENAI = {llm.ENV_PROVIDER: "openai", llm.OPENAI.key: "openai-key", llm.OPENAI.model: "gpt-x"}
CLAUDE = {llm.ENV_PROVIDER: "claude", llm.CLAUDE.key: "claude-key", llm.CLAUDE.model: "claude-x"}
CLAUDE_ADAPTIVE = {**CLAUDE, llm.CLAUDE.model: "claude-sonnet-5-5"}

#: How a test's calls are made: as the shared config states it.
SETTINGS = llm.CallSettings(reasoning_effort="minimal", retries=5, timeout_seconds=120)


def test_the_environment_names_one_of_the_providers():
    assert provider_from_environment(AZURE) is llm.AZURE
    assert provider_from_environment(FOUNDRY) is llm.AZURE_FOUNDRY
    assert provider_from_environment(OPENAI) is llm.OPENAI
    assert provider_from_environment(CLAUDE) is llm.CLAUDE
    assert list(llm.PROVIDERS) == ["azure", "azure_foundry", "openai", "claude"]


def test_what_is_missing_is_named_and_no_provider_is_assumed():
    """A key of one provider in the environment says nothing about which provider is meant: an
    environment can hold the keys of all of them."""
    assert missing_settings({**AZURE, **FOUNDRY, **OPENAI, **CLAUDE, llm.ENV_PROVIDER: ""}) == [
        f"{llm.ENV_PROVIDER} (one of azure, azure_foundry, openai, claude)"]
    assert missing_settings({llm.ENV_PROVIDER: "gemini"}) == [
        f"{llm.ENV_PROVIDER} is 'gemini', and must be one of azure, azure_foundry, openai, claude"]
    assert missing_settings({llm.ENV_PROVIDER: "azure", llm.ENV_KEY: "k"}) == [
        llm.ENV_DEPLOYMENT, llm.ENV_ENDPOINT]
    assert missing_settings({llm.ENV_PROVIDER: "claude", llm.CLAUDE.model: ""}) == [
        llm.CLAUDE.key, llm.CLAUDE.model]
    assert missing_settings({llm.ENV_PROVIDER: "azure_foundry"}) == [
        llm.AZURE_FOUNDRY.key, llm.AZURE_FOUNDRY.model, "SEQMAESTRO_AZURE_FOUNDRY_ENDPOINT"]
    for complete in (AZURE, FOUNDRY, OPENAI, CLAUDE):
        assert missing_settings(complete) == []
    with pytest.raises(RuntimeError, match=f"missing {llm.OPENAI.key}"):
        provider_from_environment({llm.ENV_PROVIDER: "openai", llm.OPENAI.model: "gpt-x"})


def test_a_call_is_told_where_each_providers_model_is():
    assert llm.connection(llm.AZURE, AZURE) == {
        "model": "azure/my-deployment", "api_key": "azure-key",
        "api_base": "https://example.invalid/", "api_version": llm.DEFAULT_API_VERSION,
        "base_model": llm.DEFAULT_BASE_MODEL}
    stated = {**AZURE, llm.ENV_API_VERSION: "v9", llm.ENV_BASE_MODEL: "azure/other"}
    assert llm.connection(llm.AZURE, stated)["api_version"] == "v9"
    assert llm.connection(llm.AZURE, stated)["base_model"] == "azure/other"
    assert llm.connection(llm.AZURE_FOUNDRY, FOUNDRY) == {
        "model": "azure_ai/an-open-model", "api_key": "foundry-key",
        "api_base": "https://example.services.ai.azure.com/"}
    assert llm.connection(llm.OPENAI, OPENAI) == {"model": "openai/gpt-x", "api_key": "openai-key"}
    assert llm.connection(llm.CLAUDE, CLAUDE) == {"model": "anthropic/claude-x",
                                                 "api_key": "claude-key"}


@pytest.fixture
def calls(monkeypatch):
    """Record the arguments of each call the client makes, and make no call."""
    import litellm

    made = []

    def completion(**arguments):
        made.append(arguments)
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
        return SimpleNamespace(usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="a reply"))])

    monkeypatch.setattr(litellm, "completion", completion)
    return made


def configured(monkeypatch, environment):
    for name in (llm.ENV_PROVIDER, *(v for p in llm.PROVIDERS.values() for v in p.required)):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("environment, model, effort", [
    (AZURE, "azure/my-deployment", "minimal"), (OPENAI, "openai/gpt-x", "minimal"),
    (CLAUDE, "anthropic/claude-x", None), (CLAUDE_ADAPTIVE, "anthropic/claude-sonnet-5-5", "minimal"),
    (FOUNDRY, "azure_ai/an-open-model", "none")])
def test_each_provider_states_the_reasoning_effort_it_takes(monkeypatch, calls, environment, model,
                                                            effort):
    """A current Claude model thinks adaptively whatever a call says, so it is told the client's
    effort, which sets how much it thinks. An effort sent to an older Claude model, or to one that
    LiteLLM's map does not know, would turn on extended thinking, whose budget the reply's token
    limit must exceed, so it is sent none and does not think. A model on Azure AI Foundry may reason
    unless told not to, and can spend the whole reply on it, so every call to it turns reasoning off
    whatever the client's effort, and LiteLLM is told to pass that setting on."""
    configured(monkeypatch, environment)
    client = llm.LiteLLMClient(SETTINGS)
    assert client.complete("system", "user", max_completion_tokens=512).text == "a reply"

    (sent,) = calls
    assert sent["model"] == model == client.model == client.usage()["model"]
    assert sent.get("reasoning_effort") == effort
    assert ("allowed_openai_params" in sent) is (environment is FOUNDRY)
    assert sent["max_completion_tokens"] == 512 and sent["drop_params"] is True
    assert sent["num_retries"] == 5 and sent["timeout"] == 120
    assert sent["api_key"] == environment[provider_from_environment(environment).key]
    assert client.usage()["total_tokens"] == 10 and client.usage()["uncounted_calls"] == 0


def test_a_call_takes_the_runs_effort_and_waits_at_least_the_runs_wait(monkeypatch, calls):
    """A reply that grows with its request is waited for as long as it needs, and never for less
    than the run states."""
    configured(monkeypatch, OPENAI)
    client = llm.LiteLLMClient(llm.CallSettings(reasoning_effort="high", retries=0, timeout_seconds=300))
    client.complete("system", "user", max_completion_tokens=64)
    client.complete("system", "user", max_completion_tokens=64, timeout=900)
    client.complete("system", "user", max_completion_tokens=64, timeout=30)
    assert [sent["timeout"] for sent in calls] == [300, 900, 300]
    assert {sent["reasoning_effort"] for sent in calls} == {"high"}
    assert {sent["num_retries"] for sent in calls} == {0}


@pytest.mark.parametrize("environment, recorded", [
    ({**AZURE, llm.ENV_API_VERSION: "v9"},
     {"provider": "azure", "model": "azure/my-deployment", "api_version": "v9",
      "base_model": llm.DEFAULT_BASE_MODEL, "reasoning_effort": "minimal"}),
    (OPENAI, {"provider": "openai", "model": "openai/gpt-x", "reasoning_effort": "minimal"}),
    (CLAUDE, {"provider": "claude", "model": "anthropic/claude-x", "reasoning_effort": None}),
    (FOUNDRY, {"provider": "azure_foundry", "model": "azure_ai/an-open-model",
               "reasoning_effort": "none"})])
def test_a_record_of_the_calls_says_which_model_and_how_and_holds_no_secret(monkeypatch, environment,
                                                                             recorded):
    """A record names the model and what the calls stated (the effort as each provider takes it),
    so that two runs at different efforts or on different models can be told apart. The record
    goes into a run's files, so it holds neither the key nor the endpoint, which names a private
    resource."""
    configured(monkeypatch, environment)
    record = llm.LiteLLMClient(SETTINGS).describe()
    assert record == {**recorded, "retries": 5, "timeout_seconds": 120}
    assert record == llm.describe_calls(provider_from_environment(environment), SETTINGS, environment)
    for secret in ("azure-key", "foundry-key", "openai-key", "claude-key", "https://"):
        assert all(secret not in str(value) for value in record.values()), secret
    assert llm.reply_conditions(record) == recorded, "retries and the wait change no reply"


def test_the_efforts_a_run_may_state_are_those_litellm_passes_on():
    from typing import get_args

    from litellm.types.llms.openai import REASONING_EFFORT

    assert set(llm.REASONING_EFFORTS) == set(get_args(REASONING_EFFORT))


def test_a_reply_that_reports_no_token_counts_is_counted_apart(monkeypatch):
    """A deployment may answer without token counts (DeepSeek's on Azure AI Foundry does). Such
    calls add nothing to the token totals, and counting them apart keeps the totals from being
    mistaken for those of every call."""
    import litellm

    configured(monkeypatch, FOUNDRY)
    reply = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                            choices=[SimpleNamespace(message=SimpleNamespace(content="a reply"))])
    monkeypatch.setattr(litellm, "completion", lambda **arguments: reply)
    client = llm.LiteLLMClient(SETTINGS)
    assert client.complete("system", "user", max_completion_tokens=64).ok
    assert client.complete("system", "user", max_completion_tokens=64).ok
    assert client.usage() | {"model": None} == {
        "model": None, "calls": 2, "failures": 0, "uncounted_calls": 2, "prompt_tokens": 0,
        "completion_tokens": 0, "total_tokens": 0}


def test_no_key_is_kept_on_the_client_or_in_a_recorded_failure(monkeypatch):
    """A client is held by objects a run saves, and a failed call's message is recorded with the
    run, so neither may hold the key."""
    import litellm

    configured(monkeypatch, CLAUDE)
    client = llm.LiteLLMClient(SETTINGS)
    assert b"claude-key" not in pickle.dumps(client)

    def refused(**arguments):
        raise RuntimeError(f"Incorrect API key provided: {arguments['api_key']}")

    monkeypatch.setattr(litellm, "completion", refused)
    failed = client.complete("system", "user", max_completion_tokens=64)
    assert not failed.ok and "claude-key" not in failed.error
    assert failed.error == "RuntimeError: Incorrect API key provided: <key>"
    assert client.usage()["failures"] == 1


def test_a_client_cannot_be_made_for_a_provider_that_is_not_configured(monkeypatch):
    configured(monkeypatch, {llm.ENV_PROVIDER: "openai"})
    with pytest.raises(RuntimeError, match="The language model is not configured"):
        llm.LiteLLMClient(SETTINGS)


def test_each_vendored_method_has_a_config_for_every_provider_reading_its_variables():
    """DEFT and IGSR cannot import `seqmaestro`, so each reads the provider's variables in a config
    of its own, which an arm's stage selects by the provider's name. A config naming another
    variable, or another route, would call a model the rest of the run does not."""
    for name, provider in llm.PROVIDERS.items():
        models = (DEFT_CONF / "llm_models" / f"{deft.LANGUAGE_MODEL[name]['llm_models']}.yaml").read_text()
        assert f"model: {provider.route}/${{oc.env:{provider.model}}}" in models, name
        assert f"api_key: ${{oc.env:{provider.key}}}" in models, name
        assert (DEFT_CONF / "llm" / f"{deft.LANGUAGE_MODEL[name]['llm']}.yaml").exists(), name

        config = IGSR_CONF / "llm" / f"{igsr.LANGUAGE_MODEL[name]}.yaml"
        text = config.read_text() + "".join(
            (IGSR_CONF / "llm" / f"{base}.yaml").read_text() for base in ("azure_base",) if name == "azure")
        assert f"kind: {provider.route}" in text, name
        assert f"deployment: ${{oc.env:{provider.model}}}" in text, name
        assert f"api_key: ${{oc.env:{provider.key}}}" in text, name

        # Where a provider's calls state a fixed effort whatever the client's, each method's calls
        # state it too, so that the proposer, DEFT and IGSR call the model alike.
        if provider.effort not in (None, llm.CLIENT_EFFORT, llm.ADAPTIVE_EFFORT):
            engine = (DEFT_CONF / "llm" / f"{deft.LANGUAGE_MODEL[name]['llm']}.yaml").read_text()
            assert f"reasoning_effort: {provider.effort}" in engine, name
            assert "allowed_openai_params: [reasoning_effort]" in engine, name
            assert f"reasoning_effort: {provider.effort}" in text, name
    assert set(deft.LANGUAGE_MODEL) == set(igsr.LANGUAGE_MODEL) == set(llm.PROVIDERS)
    assert all(tuple(chosen) == deft.LANGUAGE_MODEL_KEYS for chosen in deft.LANGUAGE_MODEL.values())


@pytest.mark.parametrize("name", list(llm.PROVIDERS))
def test_each_vendored_method_is_called_under_the_runs_settings(name):
    """DEFT and IGSR are told the run's retries and wait, and its effort where the provider takes
    the run's, as the client's calls are. Each key overridden is one their configs for the provider
    already hold, so the override replaces a value, and a misspelt key fails to compose."""
    provider = llm.PROVIDERS[name]
    settings = llm.CallSettings(reasoning_effort="low", retries=2, timeout_seconds=90)
    takes_effort = provider.effort in (llm.CLIENT_EFFORT, llm.ADAPTIVE_EFFORT)
    environ = {provider.model: "claude-sonnet-5-5" if provider is llm.CLAUDE else "a-model"}
    engine = (DEFT_CONF / "llm" / f"{deft.LANGUAGE_MODEL[name]['llm']}.yaml").read_text()
    config = (IGSR_CONF / "llm" / f"{igsr.LANGUAGE_MODEL[name]}.yaml").read_text()
    for method, text in ((deft, engine), (igsr, config)):
        told = call_overrides(settings, provider, method.CALL_KEYS, environ)
        expected = [f"{method.CALL_KEYS['retries']}=2", f"{method.CALL_KEYS['timeout_seconds']}=90"]
        if takes_effort:
            expected.append(f"{method.CALL_KEYS['reasoning_effort']}=low")
        assert told == expected, (method.__name__, name)
        for item in told:
            key = item.split("=")[0].removeprefix("llm.")
            assert f"\n{key}:" in text, (method.__name__, name, key)


@pytest.mark.parametrize("model, client_effort, sent", [
    ("anthropic/claude-sonnet-5-5", "medium", "medium"), ("anthropic/claude-haiku-5-5", "none", "minimal"),
    ("anthropic/claude-haiku-4-5", "medium", None), ("anthropic/claude-unknown", "low", None)])
def test_claude_is_told_an_effort_only_where_its_thinking_is_adaptive(model, client_effort, sent):
    """A model that thinks adaptively cannot be told not to think, so `none` becomes the lowest
    effort. A model that thinks only when asked, or one LiteLLM's map does not know, is told none,
    and DEFT and IGSR are told what the client's calls state."""
    assert llm.CLAUDE.reasoning(client_effort, model).get("reasoning_effort") == sent
    settings = llm.CallSettings(reasoning_effort=client_effort, retries=1, timeout_seconds=60)
    environ = {llm.CLAUDE.model: model.removeprefix("anthropic/")}
    told = call_overrides(settings, llm.CLAUDE, igsr.CALL_KEYS, environ)
    assert any(item.startswith(f"{igsr.CALL_KEYS['reasoning_effort']}=") for item in told) is (sent is not None)
