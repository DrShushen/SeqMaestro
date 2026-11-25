"""The client that calls the language model, with each provider's quirks configured in one place.

Every caller of a language model in this package goes through this client: the proposer, the judge
and the recurrence analysis. A run calls one model at one provider for all of them, so that arms
are compared at one model and one model does all the judging. The environment names the provider
(`SEQMAESTRO_LLM_PROVIDER`) and holds its key and its model's name, and `PROVIDERS` lists the ones
a run can use: Azure OpenAI, a model deployed on Azure AI Foundry, the OpenAI API and the Claude
API.

The providers differ in what a call has to state. An Azure OpenAI deployment's name does not
appear in any model map, so the model family behind it is declared beside it. The OpenAI reasoning
models reject `temperature` and take `max_completion_tokens` in place of `max_tokens`. A current
Claude model thinks adaptively whatever a call says, and the effort sets how much, whereas an older
one thinks only when a call sends an effort, with a budget that has to fit under the reply's limit.
The open models on Azure AI Foundry reason unless a call turns their reasoning off.
If each caller handled that on its own, a fix made in one and missed in another would stop the
arms being budget-matched without anyone noticing.

**How every call is made is a setting of the run.** The reasoning effort, the retries and the wait
are stated in the run's config (`language_model`, read into `CallSettings`), so a run's record
says what its answers were asked under, and two runs made at different efforts can be told apart.
Each consumer records which model it called and how (`LiteLLMClient.describe`), without the key or
the endpoint.

**A failure is returned, not raised.** A proposal round or a judgement that fails is a data point
about the run, and it is counted and reported. Raising would abort a run that is most of the way
through a paid budget.

**The client does not keep the key.** A call reads the key from the environment as it is made,
because a client is held by objects a run saves, such as a fitted featuriser, and a key stored on
it would be written into the run's files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

#: The variable that names the provider every language-model call of a run goes to, as a key of
#: `PROVIDERS`. It comes from the repo-root `.env`, with the provider's settings:
#: `set -a && source .env && set +a`.
ENV_PROVIDER = "SEQMAESTRO_LLM_PROVIDER"

# The variables each provider reads, named once here. The table of providers below refers to them
# by these names, so that no line of it pairs `key=` with a long string, which a scan for secrets
# would take for a key's value.
ENV_DEPLOYMENT = "SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT"
ENV_ENDPOINT = "SEQMAESTRO_AZURE_OPENAI_ENDPOINT"
ENV_KEY = "SEQMAESTRO_AZURE_OPENAI_KEY"
ENV_API_VERSION = "SEQMAESTRO_AZURE_OPENAI_API_VERSION"
ENV_BASE_MODEL = "SEQMAESTRO_AZURE_OPENAI_BASE_MODEL"

ENV_FOUNDRY_DEPLOYMENT = "SEQMAESTRO_AZURE_FOUNDRY_DEPLOYMENT"
ENV_FOUNDRY_ENDPOINT = "SEQMAESTRO_AZURE_FOUNDRY_ENDPOINT"
ENV_FOUNDRY_KEY = "SEQMAESTRO_AZURE_FOUNDRY_KEY"

ENV_OPENAI_KEY = "SEQMAESTRO_OPENAI_KEY"
ENV_OPENAI_MODEL = "SEQMAESTRO_OPENAI_MODEL"

ENV_CLAUDE_KEY = "SEQMAESTRO_CLAUDE_KEY"
ENV_CLAUDE_MODEL = "SEQMAESTRO_CLAUDE_MODEL"

DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_BASE_MODEL = "azure/gpt-5.4"

#: The value of `Provider.effort` that has a call state the client's own reasoning effort.
CLIENT_EFFORT = "client"

#: The value of `Provider.effort` that has a call state the client's effort only to a model whose
#: thinking is adaptive (LiteLLM's `supports_adaptive_thinking`), as Claude's current models are.
#: Such a model thinks whatever a call says, and the effort sets how much, so a model left without
#: one would think at its own default and spend the reply's tokens on it. It cannot be told not to
#: think, so the lowest effort (`minimal`, which LiteLLM sends as Claude's `low`) stands in for
#: `none`. A model that thinks only when asked, such as Claude Haiku 4.5, is sent no effort, which
#: keeps its thinking off, the counterpart of the lowest effort.
ADAPTIVE_EFFORT = "adaptive"

#: The reasoning efforts a run may state, as LiteLLM passes them on. Which of them a model accepts
#: depends on the provider, and a model that refuses an effort fails every call made with it.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


@dataclass(frozen=True)
class CallSettings:
    """How every call of a run is made, stated in the run's config as `language_model`.

    `reasoning_effort` is what a call asks of a model that reasons, where the provider takes the
    run's effort (`Provider.effort`). `retries` is how many times a call is made again after a
    failure that may be temporary, such as a rate limit or a timeout, and `timeout_seconds` is how
    long a reply is waited for. A request whose reply grows with its size, such as the recurrence
    analysis's grouping, states a longer wait, and the call waits for the longer of the two.
    """

    reasoning_effort: str
    retries: int
    timeout_seconds: int


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
    #: The reasoning effort a call states. `CLIENT_EFFORT` states the client's effort, which the
    #: OpenAI reasoning models take and LiteLLM leaves out for a model that does not.
    #: `ADAPTIVE_EFFORT` states it only to a model whose thinking is adaptive, as for Claude. None
    #: states no effort. Any other value is stated on every call, whatever the client's effort, for
    #: a provider whose models reason unless a call turns the reasoning off.
    effort: str | None = CLIENT_EFFORT

    @property
    def required(self) -> tuple[str, ...]:
        """The variables a call cannot be made without."""
        return (self.key, self.model,
                *(variable for _, variable, default in self.further if default is None))

    def reasoning(self, client_effort: str, model: str) -> dict[str, object]:
        """What a call to `model` (its model string, such as `anthropic/claude-sonnet-5-5`) states
        about the model's reasoning, as LiteLLM's arguments, for a client whose own effort is
        `client_effort`."""
        if self.effort is None:
            return {}
        if self.effort == CLIENT_EFFORT:
            return {"reasoning_effort": client_effort}
        if self.effort == ADAPTIVE_EFFORT:
            if not thinks_adaptively(model):
                return {}
            return {"reasoning_effort": "minimal" if client_effort == "none" else client_effort}
        # A fixed effort goes to deployments whose names are not in any model map. LiteLLM drops a
        # setting unless it can tell that the model takes it, so it is told to pass this one on.
        return {"reasoning_effort": self.effort, "allowed_openai_params": ["reasoning_effort"]}



def thinks_adaptively(model: str) -> bool:
    """Whether `model` thinks adaptively, by LiteLLM's map of models. A model the map does not
    know is taken not to, so it is sent no effort, as before the map knew of adaptive thinking."""
    import litellm

    try:
        return litellm.get_model_info(model).get("supports_adaptive_thinking") is True
    except Exception:
        return False


AZURE = Provider(
    "azure", route="azure", key=ENV_KEY, model=ENV_DEPLOYMENT,
    further=(("api_base", ENV_ENDPOINT, None),
             ("api_version", ENV_API_VERSION, DEFAULT_API_VERSION),
             # A deployment's name does not appear in any capability map, so the model family it
             # is based on is declared separately, which is what lets `reasoning_effort` through.
             ("base_model", ENV_BASE_MODEL, DEFAULT_BASE_MODEL)))
OPENAI = Provider("openai", route="openai", key=ENV_OPENAI_KEY, model=ENV_OPENAI_MODEL)
CLAUDE = Provider("claude", route="anthropic", key=ENV_CLAUDE_KEY, model=ENV_CLAUDE_MODEL,
                  effort=ADAPTIVE_EFFORT)
#: A model deployed on an Azure AI Foundry resource, such as the open models Kimi and DeepSeek,
#: called through the resource's model inference API (LiteLLM's `azure_ai` route). These models
#: may reason before they answer, and the reasoning counts against the reply's limit, so a model
#: that reasons can spend the whole reply on it and return nothing. Every call therefore turns the
#: reasoning off, which is the counterpart of the lowest effort the other providers are called at.
#: The effort `none` does so for Kimi K2.6 and DeepSeek V3.2 alike, whereas Kimi refuses `minimal`,
#: the lowest effort of the OpenAI models.
AZURE_FOUNDRY = Provider(
    "azure_foundry", route="azure_ai", key=ENV_FOUNDRY_KEY, model=ENV_FOUNDRY_DEPLOYMENT,
    further=(("api_base", ENV_FOUNDRY_ENDPOINT, None),), effort="none")

#: The providers a run can call, by the name the environment gives one.
PROVIDERS = {provider.name: provider for provider in (AZURE, AZURE_FOUNDRY, OPENAI, CLAUDE)}

SOURCE_HINT = "Settings come from the repo-root .env: `set -a && source .env && set +a`."


@dataclass(frozen=True)
class Completion:
    """One reply. `text is None` means the call failed, in which case `error` says how."""

    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def missing_settings(environ: Mapping[str, str] | None = None) -> list[str]:
    """What the environment lacks for a call to be made, in words: the variable naming the provider
    where it is unset or names no known provider, and otherwise each of the provider's variables
    that is unset. The list is empty when a call can be made."""
    environ = os.environ if environ is None else environ
    named = environ.get(ENV_PROVIDER)
    if not named:
        return [f"{ENV_PROVIDER} (one of {', '.join(PROVIDERS)})"]
    if named not in PROVIDERS:
        return [f"{ENV_PROVIDER} is {named!r}, and must be one of {', '.join(PROVIDERS)}"]
    return [variable for variable in PROVIDERS[named].required if not environ.get(variable)]


def provider_from_environment(environ: Mapping[str, str] | None = None) -> Provider:
    """The provider the environment names, with every setting it requires present, or a
    `RuntimeError` that says what is missing."""
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


#: What a record of the calls leaves out of their connection: the key, and the endpoint, which
#: names the resource the calls go to.
UNRECORDED = ("api_key", "api_base")

#: The entries of a record of the calls (`describe_calls`) that do not change a reply: how many
#: times a failed call is made again, and how long a reply is waited for.
NOT_IN_THE_REPLY = ("retries", "timeout_seconds")


def describe_calls(provider: Provider, settings: CallSettings,
                   environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """What a run records of how a consumer calls its model: the provider, the model string and
    what else the provider states (Azure OpenAI's API version and base model), the reasoning
    effort each call states (None where it states none), the retries and the wait. It holds
    neither the key nor the endpoint, so it can be written into a run's files.

    DEFT and IGSR read the same variables in their configs for the provider, so a record made
    from the environment says which model they called too."""
    connected = connection(provider, environ)
    return {"provider": provider.name,
            **{key: value for key, value in connected.items() if key not in UNRECORDED},
            "reasoning_effort": provider.reasoning(settings.reasoning_effort,
                                                  connected["model"]).get("reasoning_effort"),
            "retries": settings.retries, "timeout_seconds": settings.timeout_seconds}


def reply_conditions(record: Mapping[str, object]) -> dict[str, object]:
    """What in a record of the calls decides their replies: which model was called, and how it was
    asked to reason. A stored answer is kept under these (the proposal cache's key, and the digest
    an analysis keeps its answers under), so a change of model or of effort means the model is
    asked again, whereas a change in how failures are retried or how long a reply is waited for
    keeps the stored answer."""
    return {key: value for key, value in record.items() if key not in NOT_IN_THE_REPLY}


class LiteLLMClient:
    """The run's provider through LiteLLM, one call at a time, with running token counts.

    Token accounting is kept here instead of in each caller, because the budget is shared: a run's
    cost adds up what each caller spent, and a figure assembled from counters that count differently
    is worse than no figure.

    The client holds what is the same for every call of a run: the provider, the model and the
    run's `CallSettings`. The tokens a reply may take differ between callers, from a judge's one
    sentence to a proposal's code, so each call states them.
    """

    def __init__(self, settings: CallSettings):
        self.provider = provider_from_environment()
        self.model = model_string(self.provider)
        self.settings = settings
        self.calls = 0
        self.failures = 0
        #: Calls whose reply reported no token counts, which the token totals therefore leave out.
        self.uncounted = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, system: str, user: str, max_completion_tokens: int,
                 timeout: int | None = None) -> Completion:
        """One reply of up to `max_completion_tokens`. A caller whose reply grows with its request
        also states the seconds such a reply takes, and the call waits the longer of those and the
        run's `timeout_seconds`."""
        import litellm

        # LiteLLM prints a provider banner to stderr on every call, which buries progress output
        # under hundreds of identical blocks. Failures still surface, because they come back as
        # exceptions and are recorded on the Completion.
        litellm.suppress_debug_info = True

        try:
            response = litellm.completion(
                **connection(self.provider),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # LiteLLM turns this into `max_tokens` for a provider that takes that.
                max_completion_tokens=max_completion_tokens,
                **self.provider.reasoning(self.settings.reasoning_effort, self.model),
                num_retries=self.settings.retries,
                timeout=max(timeout or 0, self.settings.timeout_seconds),
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
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        if not prompt + completion:
            # A reply always takes tokens, so none counted means the provider reported none, as
            # the DeepSeek deployment on Azure AI Foundry does. Such calls are counted apart, so
            # that a total which leaves them out is not taken for the total of every call.
            self.uncounted += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion

        return Completion(response.choices[0].message.content or "")

    def describe(self) -> dict[str, object]:
        """Which model the calls go to and how they are made, for a consumer's record
        (`describe_calls`)."""
        return describe_calls(self.provider, self.settings)

    def usage(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "uncounted_calls": self.uncounted,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class ScriptedClient:
    """A deterministic stand-in, so everything above the call can be tested without a bill.

    It takes a callable from `(system, user)` to a reply string, or a fixed reply, so a test can
    simulate a well-behaved model, a malformed one, or an outright failure (by returning an
    exception instance). `described` is the record it gives of the model it stands for, which a
    test can vary to check what is stored under that record.
    """

    def __init__(self, reply, described: Mapping[str, object] | None = None):
        self.reply = reply
        self.described = dict(described) if described is not None else {"model": "scripted"}
        self.seen: list[tuple[str, str]] = []
        #: The tokens each call allowed its reply, and the seconds where it stated a wait (None
        #: where it stated none).
        self.limits: list[tuple[int, int | None]] = []

    def complete(self, system: str, user: str, max_completion_tokens: int,
                 timeout: int | None = None) -> Completion:
        self.seen.append((system, user))
        self.limits.append((max_completion_tokens, timeout))
        text = self.reply(system, user) if callable(self.reply) else self.reply
        if isinstance(text, BaseException):
            return Completion(None, error=f"{type(text).__name__}: {text}")
        return Completion(text)

    def describe(self) -> dict[str, object]:
        return dict(self.described)

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.seen), "failures": 0}
