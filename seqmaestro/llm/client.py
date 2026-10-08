"""Calls the model, with each provider's quirks configured in one place.

Every caller of a language model in this package goes through this client: the proposer, the judge
and the recurrence analysis. A run calls one model at one provider for all of them, so that arms
are compared at one model and one model does all the judging. The environment names the provider
(`SEQMAESTRO_LLM_PROVIDER`) and holds its key and its model's name, and `PROVIDERS` lists the ones
a run can use: Azure OpenAI, the OpenAI API and the Claude API.

The providers differ in what a call has to state. An Azure deployment's name appears in no model
map, so the model family behind it is declared beside it. The OpenAI reasoning models reject
`temperature` and take `max_completion_tokens` in place of `max_tokens`. A reasoning effort sent
to Claude turns on extended thinking, with a budget that has to fit under the reply's limit. If
each caller handled that on its own, a fix made in one and missed in another would stop the arms
being budget-matched without anyone noticing.

**A failure is returned, not raised.** A proposal round or a judgement that fails is a data point
about the run, and it is counted and reported. Raising would abort a run that is most of the way
through a paid budget.

**No key is kept on the client.** A call reads the key from the environment as it is made. A
client is held by objects a run saves, such as a fitted featuriser, and a key stored on it would
be written into the run's files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

#: Names the provider every language-model call of a run goes to: a key of `PROVIDERS`. Read from
#: the repo-root `.env`, with the provider's settings: `set -a && source .env && set +a`.
ENV_PROVIDER = "SEQMAESTRO_LLM_PROVIDER"

ENV_DEPLOYMENT = "SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT"
ENV_ENDPOINT = "SEQMAESTRO_AZURE_OPENAI_ENDPOINT"
ENV_KEY = "SEQMAESTRO_AZURE_OPENAI_KEY"
ENV_API_VERSION = "SEQMAESTRO_AZURE_OPENAI_API_VERSION"
ENV_BASE_MODEL = "SEQMAESTRO_AZURE_OPENAI_BASE_MODEL"

DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_BASE_MODEL = "azure/gpt-5.4"


@dataclass(frozen=True)
class Provider:
    """One way of reaching a language model, and the variables its settings are read from."""

    name: str
    #: The prefix of LiteLLM's model string, which is what routes a call.
    route: str
    #: The variable holding the API key.
    key: str
    #: The variable holding the model's name. For Azure it is the deployment's name.
    model: str
    #: Each further setting a call takes: LiteLLM's argument, the variable it is read from, and
    #: the value used where the variable is unset. A setting with no such value is required.
    further: tuple[tuple[str, str, str | None], ...] = ()
    #: Whether a call states the client's reasoning effort. The OpenAI reasoning models take one,
    #: and LiteLLM leaves it out for a model that does not. Claude is called without it, which
    #: leaves its extended thinking off: the counterpart of the lowest effort.
    states_effort: bool = True

    @property
    def required(self) -> tuple[str, ...]:
        """The variables a call cannot be made without."""
        return (self.key, self.model,
                *(variable for _, variable, default in self.further if default is None))


AZURE = Provider(
    "azure", route="azure", key=ENV_KEY, model=ENV_DEPLOYMENT,
    further=(("api_base", ENV_ENDPOINT, None),
             ("api_version", ENV_API_VERSION, DEFAULT_API_VERSION),
             # A deployment's name appears in no capability map, so the model family it is based on
             # is declared separately. That is what lets `reasoning_effort` through.
             ("base_model", ENV_BASE_MODEL, DEFAULT_BASE_MODEL)))
OPENAI = Provider("openai", route="openai", key="SEQMAESTRO_OPENAI_KEY",
                  model="SEQMAESTRO_OPENAI_MODEL")
CLAUDE = Provider("claude", route="anthropic", key="SEQMAESTRO_CLAUDE_KEY",
                  model="SEQMAESTRO_CLAUDE_MODEL", states_effort=False)

#: The providers a run can call, by the name the environment gives one.
PROVIDERS = {provider.name: provider for provider in (AZURE, OPENAI, CLAUDE)}

SOURCE_HINT = "Settings come from the repo-root .env: `set -a && source .env && set +a`."


@dataclass(frozen=True)
class Completion:
    """One reply. `text is None` means the call failed; `error` says how."""

    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def missing_settings(environ: Mapping[str, str] | None = None) -> list[str]:
    """What the environment lacks for a call to be made, in words: the provider's name where none
    is given or it is no provider, and otherwise each of the provider's variables that is unset.
    Empty when a call can be made."""
    environ = os.environ if environ is None else environ
    named = environ.get(ENV_PROVIDER)
    if not named:
        return [f"{ENV_PROVIDER} (one of {', '.join(PROVIDERS)})"]
    if named not in PROVIDERS:
        return [f"{ENV_PROVIDER} is {named!r}, and must be one of {', '.join(PROVIDERS)}"]
    return [variable for variable in PROVIDERS[named].required if not environ.get(variable)]


def provider_from_environment(environ: Mapping[str, str] | None = None) -> Provider:
    """The provider the environment names, with every setting it requires present, or
    `RuntimeError` saying what is not."""
    environ = os.environ if environ is None else environ
    missing = missing_settings(environ)
    if missing:
        raise RuntimeError(f"The language model is not configured: missing {', '.join(missing)}. "
                           f"{SOURCE_HINT}")
    return PROVIDERS[environ[ENV_PROVIDER]]


def model_string(provider: Provider, environ: Mapping[str, str] | None = None) -> str:
    """The model as LiteLLM names it, such as `openai/gpt-5.4`. It is what a run records."""
    environ = os.environ if environ is None else environ
    return f"{provider.route}/{environ[provider.model]}"


def connection(provider: Provider, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """What a call tells LiteLLM about where the model is: its model string, its key, and what the
    provider adds, such as Azure's endpoint. It holds the key, so it is made for a call and not
    kept."""
    environ = os.environ if environ is None else environ
    return {"model": model_string(provider, environ), "api_key": environ[provider.key],
            **{argument: environ.get(variable) or default
               for argument, variable, default in provider.further}}


class LiteLLMClient:
    """The run's provider through LiteLLM, one call at a time, with running token counts.

    Token accounting lives here instead of in each caller because the budget is shared. A run's
    cost adds up what each caller spent, and a figure assembled from counters that count differently
    is worse than no figure.
    """

    def __init__(
        self,
        max_completion_tokens: int = 4096,
        reasoning_effort: str = "minimal",
        num_retries: int = 5,
        request_timeout: int = 120,
    ):
        self.provider = provider_from_environment()
        self.model = model_string(self.provider)
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.num_retries = num_retries
        self.request_timeout = request_timeout
        self.calls = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, system: str, user: str, max_completion_tokens: int | None = None,
                 timeout: int | None = None) -> Completion:
        """One reply. A caller whose reply grows with its request states the tokens the reply may
        take and the seconds to wait for it, and any other call takes the client's."""
        import litellm

        # LiteLLM prints a provider banner to stderr on every call, which buries progress output
        # under hundreds of identical blocks. Failures still surface, because they come back as
        # exceptions and are recorded on the Completion.
        litellm.suppress_debug_info = True

        effort = {"reasoning_effort": self.reasoning_effort} if self.provider.states_effort else {}
        try:
            response = litellm.completion(
                **connection(self.provider),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # LiteLLM turns this into `max_tokens` for a provider that takes that.
                max_completion_tokens=max_completion_tokens or self.max_completion_tokens,
                **effort,
                num_retries=self.num_retries,
                timeout=timeout or self.request_timeout,
                # Reasoning models reject `temperature` and `top_p`. LiteLLM strips whatever the
                # model does not accept, so the list is not encoded here.
                drop_params=True,
            )
        except Exception as exc:
            self.failures += 1
            # A failure is recorded with the run, and a provider's message can quote the key.
            key = os.environ.get(self.provider.key) or "\0"
            return Completion(None, error=f"{type(exc).__name__}: {exc}".replace(key, "<key>"))

        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        return Completion(response.choices[0].message.content or "")

    def usage(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class ScriptedClient:
    """A deterministic stand-in, so everything above the call can be tested without a bill.

    Takes a callable from `(system, user)` to a reply string, or a fixed reply, which lets a test
    simulate a well-behaved model, a malformed one, or an outright failure (by returning an
    exception instance).
    """

    def __init__(self, reply):
        self.reply = reply
        self.seen: list[tuple[str, str]] = []
        #: The tokens and the seconds each call allowed its reply, None where it stated none.
        self.limits: list[tuple[int | None, int | None]] = []

    def complete(self, system: str, user: str, max_completion_tokens: int | None = None,
                 timeout: int | None = None) -> Completion:
        self.seen.append((system, user))
        self.limits.append((max_completion_tokens, timeout))
        text = self.reply(system, user) if callable(self.reply) else self.reply
        if isinstance(text, BaseException):
            return Completion(None, error=f"{type(text).__name__}: {text}")
        return Completion(text)

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.seen), "failures": 0}
