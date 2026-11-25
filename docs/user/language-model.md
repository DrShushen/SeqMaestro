# Configuring the language model

SeqMaestro calls one language model for everything a language model does in a run:

- Writing the features of the [written sources](../glossary.md#written-source), including those of DEFT and IGSR.
- Rating features, in runs that use the [judge](../glossary.md#judge).
- Grouping written features into ideas (the [recurrence analysis](../glossary.md#recurrence-analysis)).

One model serves every part, so that differences between the arms cannot come from different models. This page says how to name that model and check that it answers. The [overview](overview.md) explains the arms and the written sources.

The [agent](../glossary.md#agent) runs on a language model of its own, which you set up in its harness ([setting up an agent](agents.md)). SeqMaestro, by contrast, calls its language model with the key you put in `.env`. The two can be the same model, set up in both places.

## Providers

SeqMaestro can reach the model through four providers, named in `.env` as follows:

- `azure`: an [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai/) deployment
- `azure_foundry`: a model deployed on [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry), such as the open models Kimi and DeepSeek
- `openai`: a model of the [OpenAI API](https://platform.openai.com/)
- `claude`: a model of the [Claude API](https://platform.claude.com/)

Every call goes through [LiteLLM](https://docs.litellm.ai/), which passes it to the provider.

## Filling in `.env`

The installer made a file named `.env` in the repository's folder, from the template `.env.example`. Open `.env` in an editor, set `SEQMAESTRO_LLM_PROVIDER` to `azure`, `azure_foundry`, `openai` or `claude`, and fill in that provider's values:

| Provider | In `.env` | Value | Where to find it |
|---|---|---|---|
| `azure` | `SEQMAESTRO_AZURE_OPENAI_KEY` | The key of your Azure OpenAI resource | The [Azure portal](https://portal.azure.com/): your Azure OpenAI resource, then **Keys and Endpoint** |
| | `SEQMAESTRO_AZURE_OPENAI_ENDPOINT` | The resource's address, such as `https://my-resource.openai.azure.com/` | The same page |
| | `SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT` | The name you gave the deployment, which can differ from the model's name | The [Azure AI Foundry portal](https://ai.azure.com/): **Deployments** |
| `azure_foundry` | `SEQMAESTRO_AZURE_FOUNDRY_KEY` | The key of your Azure AI Foundry resource | The [Azure AI Foundry portal](https://ai.azure.com/): the deployment's page, **Key** |
| | `SEQMAESTRO_AZURE_FOUNDRY_ENDPOINT` | The resource's address, such as `https://my-resource.services.ai.azure.com/` | The same page: its **Target URI**, which can be used as it is |
| | `SEQMAESTRO_AZURE_FOUNDRY_DEPLOYMENT` | The name you gave the deployment | The [Azure AI Foundry portal](https://ai.azure.com/): **Deployments** |
| `openai` | `SEQMAESTRO_OPENAI_KEY` | An API key of the OpenAI platform | [platform.openai.com](https://platform.openai.com/api-keys): **API keys** |
| | `SEQMAESTRO_OPENAI_MODEL` | The model's name, such as `gpt-5.4` | The platform's [list of models](https://developers.openai.com/api/docs/models) |
| `claude` | `SEQMAESTRO_CLAUDE_KEY` | An API key of the Claude Console | [platform.claude.com](https://platform.claude.com/settings/keys): **API keys** |
| | `SEQMAESTRO_CLAUDE_MODEL` | The model's name, such as `claude-sonnet-5-5` | The Console's [list of models](https://platform.claude.com/docs/en/models/overview) |

You can leave the blocks of the providers you do not use empty.

**Azure OpenAI has two more values, which seldom need changing:**

- `SEQMAESTRO_AZURE_OPENAI_API_VERSION`, the API version the deployment is called with, is filled in already.
- `SEQMAESTRO_AZURE_OPENAI_BASE_MODEL` names the base model behind the deployment, the model the deployment serves. While its line is commented out, SeqMaestro takes the base model to be `azure/gpt-5.4`. If your deployment serves another model, uncomment the line and give that model as `azure/<model>`.

**Azure AI Foundry needs only the three values in the table.** SeqMaestro turns the model's reasoning off, because a model that reasons before it answers can spend the whole of its reply on the reasoning.

**SeqMaestro's settings are kept apart from the agent's.** `./agent.sh` does not pass `.env` to the harness. Instead, the agent loads `.env` into each command that needs the language model (`set -a && source .env && set +a && <command>`). The name of each setting begins with `SEQMAESTRO_`, so that a harness cannot mistake one for its own, even where `.env` is loaded in the shell the harness starts from. (Claude Code uses `ANTHROPIC_API_KEY`, Codex uses `OPENAI_API_KEY`, and OpenHands has settings of its own.) So if you choose `claude` as the provider, `SEQMAESTRO_CLAUDE_KEY` is charged only for SeqMaestro's own calls, and never for the agent's work.

## Checking the connection

```bash
./install.sh --check-llm
```

This makes one short call and prints the answer, or the provider's error. The call asks for the reasoning effort that the shared config sets for every run, so a model that refuses that effort fails here, before any run starts. [Troubleshooting](troubleshooting.md#the-call-to-the-language-model-fails) lists the usual causes of a failure.

## Call settings

The settings of every call a run makes are in `experiments/configs/_shared.yaml`, under `language_model`, and a run records them with the name of the model it called:

| Setting | Effect |
|---|---|
| `reasoning_effort` | How much reasoning a call asks for. Azure OpenAI and the OpenAI API take it as it is. A current Claude model, such as Claude Haiku 5.5, thinks adaptively and cannot be told not to, so it is called at this effort, or at its lowest effort where the setting is `none`. An older Claude model, such as Claude Haiku 4.5, is called with its extended thinking off. A call to a model on Azure AI Foundry turns its reasoning off |
| `retries` | How many times a failed call is repeated, where the failure may be temporary (such as a rate limit or a timeout) |
| `timeout_seconds` | How long a call waits for a reply, in seconds |

Only `.env` sets which provider and model a run calls. The [command line](command-line.md) calls the same model with the same settings, to draft a task description and to write a report, and loads `.env` itself.

## Tested models

SeqMaestro has been run in full with `gpt-5.4` on Azure OpenAI. Other models have been tried as follows:

- On one task, the arms in which a language model writes features (in one round or over several) have been run with `gpt-4o` and `gpt-5.4-mini` on Azure OpenAI, and with Kimi K2.6 and DeepSeek V3.2 on Azure AI Foundry.
- DEFT and IGSR have been run on Azure AI Foundry in small tests.
- With the OpenAI API and the Claude API, every part has been run in small tests only, with `gpt-5.4-nano` and `claude-haiku-4-5`.

## Keeping `.env` safe

Because `.env` holds a secret key, Git ignores it, the installer makes it readable by you alone, and the agent is set up not to open it. Keep it out of anything you share.

## Rate limits

A run makes many calls to the model, from up to 20 processes at once. Before the launch, the agent shows you the requests and tokens per minute that the run would make for each number of processes allowed to call the model at once ([the launch](running.md#stages-2-3a-and-3b-the-launch)). Compare them with your account's rate limits, and tell the agent how many processes to allow.
