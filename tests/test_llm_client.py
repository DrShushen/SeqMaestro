"""The client's providers: which one a run calls, what a call to each states, and where a key goes.

One provider serves every part of a run, named in the environment. The failures here are quiet
ones: a setting a provider refuses fails every call of an arm, a config for one provider reading
another's key runs under the wrong account, and a key kept on the client ends up in a run's files.
No test here makes a call.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from seqmaestro.llm import client as llm
from seqmaestro.llm import missing_settings, provider_from_environment
from seqmaestro.pipeline.vendored import deft, igsr

REPO = Path(__file__).resolve().parents[1]
DEFT_CONF = REPO / "methods" / "deft" / "src" / "deft" / "conf"
IGSR_CONF = REPO / "methods" / "igsr" / "src" / "igsr" / "conf"

AZURE = {llm.ENV_PROVIDER: "azure", llm.ENV_KEY: "azure-key", llm.ENV_DEPLOYMENT: "my-deployment",
         llm.ENV_ENDPOINT: "https://example.invalid/"}
OPENAI = {llm.ENV_PROVIDER: "openai", llm.OPENAI.key: "openai-key", llm.OPENAI.model: "gpt-x"}
CLAUDE = {llm.ENV_PROVIDER: "claude", llm.CLAUDE.key: "claude-key", llm.CLAUDE.model: "claude-x"}


def test_the_environment_names_one_of_the_providers():
    assert provider_from_environment(AZURE) is llm.AZURE
    assert provider_from_environment(OPENAI) is llm.OPENAI
    assert provider_from_environment(CLAUDE) is llm.CLAUDE
    assert list(llm.PROVIDERS) == ["azure", "openai", "claude"]


def test_what_is_missing_is_named_and_no_provider_is_assumed():
    """A key of one provider in the environment says nothing about which provider is meant: an
    environment can hold the keys of all three."""
    assert missing_settings({**AZURE, **OPENAI, **CLAUDE, llm.ENV_PROVIDER: ""}) == [
        f"{llm.ENV_PROVIDER} (one of azure, openai, claude)"]
    assert missing_settings({llm.ENV_PROVIDER: "gemini"}) == [
        f"{llm.ENV_PROVIDER} is 'gemini', and must be one of azure, openai, claude"]
    assert missing_settings({llm.ENV_PROVIDER: "azure", llm.ENV_KEY: "k"}) == [
        llm.ENV_DEPLOYMENT, llm.ENV_ENDPOINT]
    assert missing_settings({llm.ENV_PROVIDER: "claude", llm.CLAUDE.model: ""}) == [
        llm.CLAUDE.key, llm.CLAUDE.model]
    for complete in (AZURE, OPENAI, CLAUDE):
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
    assert llm.connection(llm.OPENAI, OPENAI) == {"model": "openai/gpt-x", "api_key": "openai-key"}
    assert llm.connection(llm.CLAUDE, CLAUDE) == {"model": "anthropic/claude-x",
                                                 "api_key": "claude-key"}


@pytest.fixture
def calls(monkeypatch):
    """The arguments of each call the client makes, in place of the calls."""
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
    (AZURE, "azure/my-deployment", True), (OPENAI, "openai/gpt-x", True),
    (CLAUDE, "anthropic/claude-x", False)])
def test_a_reasoning_effort_is_stated_to_the_providers_that_take_one(monkeypatch, calls, environment,
                                                                    model, effort):
    """Sent to Claude, an effort turns on extended thinking with a budget the reply's limit has to
    exceed, and a call with a short limit is refused. Left out, thinking is off."""
    configured(monkeypatch, environment)
    client = llm.LiteLLMClient(max_completion_tokens=512)
    assert client.complete("system", "user").text == "a reply"

    (sent,) = calls
    assert sent["model"] == model == client.model == client.usage()["model"]
    assert ("reasoning_effort" in sent) is effort
    assert sent["max_completion_tokens"] == 512 and sent["drop_params"] is True
    assert sent["api_key"] == environment[provider_from_environment(environment).key]
    assert client.usage()["total_tokens"] == 10


def test_no_key_is_kept_on_the_client_or_in_a_recorded_failure(monkeypatch):
    """A client is held by objects a run saves, and a failed call's message is recorded with the
    run, so neither may hold the key."""
    import litellm

    configured(monkeypatch, CLAUDE)
    client = llm.LiteLLMClient()
    assert b"claude-key" not in pickle.dumps(client)

    def refused(**arguments):
        raise RuntimeError(f"Incorrect API key provided: {arguments['api_key']}")

    monkeypatch.setattr(litellm, "completion", refused)
    failed = client.complete("system", "user")
    assert not failed.ok and "claude-key" not in failed.error
    assert failed.error == "RuntimeError: Incorrect API key provided: <key>"
    assert client.usage()["failures"] == 1


def test_a_client_cannot_be_made_for_a_provider_that_is_not_configured(monkeypatch):
    configured(monkeypatch, {llm.ENV_PROVIDER: "openai"})
    with pytest.raises(RuntimeError, match="The language model is not configured"):
        llm.LiteLLMClient()


def test_each_vendored_method_has_a_config_for_every_provider_reading_its_variables():
    """DEFT and IGSR cannot import this package, so each reads the provider's variables in a config
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
    assert set(deft.LANGUAGE_MODEL) == set(igsr.LANGUAGE_MODEL) == set(llm.PROVIDERS)
    assert all(tuple(chosen) == deft.LANGUAGE_MODEL_KEYS for chosen in deft.LANGUAGE_MODEL.values())
