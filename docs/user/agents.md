# Setting up an agent

The [agent](../glossary.md#agent) that carries out a run with you is a language model that works inside a program on your machine, called its [harness](../glossary.md#agent-harness), which lets the agent run commands and edit files. `./agent.sh` starts any of four harnesses (Claude Code, Codex, OpenHands and Claude Science) in the repository's folder.

Each harness runs the agent on a language model that you set up in the harness. This model is separate from the one SeqMaestro calls during a run, which `.env` names ([configuring the language model](language-model.md)), although the two can be the same model.

## Supported harnesses

| Harness | Start it with | The agent's model | Permissions |
|---|---|---|---|
| <img src="logos/claude-code.svg" alt="" height="24" class="harness-logo"> [Claude Code](#claude-code) | `./agent.sh claude`, and the agent speaks first | Your Claude account | In its default auto mode, a second model checks each action |
| <img src="logos/codex.svg" alt="" height="24" class="harness-logo"> [Codex](#codex) | `./agent.sh codex`, and the agent speaks first | Your ChatGPT account, an OpenAI API key, or a provider in `~/.codex/config.toml` | Commands run in a sandbox limited to the repository's folder, and Codex asks before a command leaves it |
| <img src="logos/openhands.svg" alt="" height="24" class="harness-logo"> [OpenHands](#openhands) | `./agent.sh openhands`, then the message it prints, typed in the browser | **Settings > LLM** in Agent Canvas, its browser interface | The agent acts without asking, unless you turn on **Settings > Verification** |
| <img src="logos/claude-science.svg" alt="" height="24" class="harness-logo"> [Claude Science](#claude-science) | `./agent.sh claude-science`, then the message it prints, typed in the browser | Your Claude account | A permission card asks you before the agent runs code |

## Claude Code

<img src="logos/claude-code.svg" alt="" height="24" class="harness-logo"> [Claude Code](https://code.claude.com/docs/en/overview) works in your terminal. Install it:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

[Its quickstart](https://code.claude.com/docs/en/quickstart) lists the other ways to install it. The first time it starts, it asks you to sign in with a Claude subscription or a Claude Console account. The agent runs on that account's model, and `/model` chooses another. Start it from the repository's folder:

```bash
./agent.sh claude
```

The first time, Claude Code asks whether you trust the folder. It then starts with SeqMaestro's instructions (`/seqmaestro`) as its first message, so the agent checks the installation and asks about your data.

Claude Code starts in auto mode, in which a second model checks each of the agent's actions in your place and blocks those it judges risky. `Shift+Tab` switches to manual mode, in which Claude Code asks you before it runs a command or changes a file. Its documentation describes the [permission modes](https://code.claude.com/docs/en/permission-modes).

## Codex

<img src="logos/codex.svg" alt="" height="24" class="harness-logo"> [Codex](https://github.com/openai/codex), from OpenAI, works in your terminal. Install it with `npm install -g @openai/codex`, or without npm with `curl -fsSL https://chatgpt.com/codex/install.sh | sh`.

Sign in with `codex login`, which uses your ChatGPT account, or with an OpenAI API key (`printenv OPENAI_API_KEY | codex login --with-api-key`). To run the agent on a model of another provider, such as an Azure OpenAI deployment, name the provider in `~/.codex/config.toml`, the file of Codex's settings in your home folder:

```toml
model_provider = "azure"
model = "<deployment>"

[model_providers.azure]
name = "Azure"
base_url = "https://my-resource.openai.azure.com/openai"
env_key = "AZURE_OPENAI_API_KEY"
query_params = { api-version = "2025-04-01-preview" }
wire_api = "responses"
```

Put your deployment's name in place of `<deployment>` and your resource's name in place of `my-resource`, and export the key as `AZURE_OPENAI_API_KEY` in the shell you start Codex from. Codex takes a provider only from this file in your home folder. Then start Codex from the repository's folder:

```bash
./agent.sh codex
```

The first time, Codex asks whether you trust the folder. Answer yes, because Codex applies the repository's settings for it (`.codex/config.toml`) only in a folder you trust, and those settings give the run's commands the network access they need to call SeqMaestro's language model. Codex then starts with SeqMaestro's instructions (`$seqmaestro`) as its first message.

Codex runs each command in a sandbox that may write only in the repository's folder and in the two folders of the PySR environment that PySR's Julia writes to (`./agent.sh` adds these two folders). The sandbox ends every process a command started once the command returns, whereas a run has to keep going for hours. The agent therefore asks you before it launches the run outside the sandbox, and you need to approve this.

## OpenHands

<img src="logos/openhands.svg" alt="" height="24" class="harness-logo"> [OpenHands](https://docs.openhands.dev) works in your browser through [Agent Canvas](https://github.com/OpenHands/agent-canvas), which runs the agent's commands on your machine. Agent Canvas needs [Node.js](https://nodejs.org/en/download) 24 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/). Install it:

```bash
npm install -g @openhands/agent-canvas
```

Then start it from the repository's folder:

```bash
./agent.sh openhands
```

Agent Canvas serves its page at <http://localhost:8000>, and `./agent.sh` prints the steps to take there:

1. The first time, Agent Canvas asks for the agent's model under **Set up your LLM**, and **Settings > LLM** changes it later. For an Azure OpenAI deployment, choose the tab **All**, fill in **Custom Model** as `azure/<deployment>` with your deployment's name, then fill in **Base URL**, **API Key** and **API Version**. Give the API version as `2025-04-01-preview`, since a GPT-5 model on Azure needs `2025-03-01-preview` or later.
2. Choose **Open Workspace > Add Workspace**, select the repository's folder, keep **Local**, and confirm.
3. Agent Canvas cannot be handed a first message, so type it yourself: `Start SeqMaestro.`

The agent's commands run on your machine, in the folder you chose, with your permissions. The agent acts without asking you, unless you turn on confirmation under **Settings > Verification**.

## Claude Science

<img src="logos/claude-science.svg" alt="" height="24" class="harness-logo"> [Claude Science](https://claude.com/docs/claude-science/get-started), from Anthropic, works in your browser and runs the agent's code on your machine. On Linux it needs [bubblewrap](https://github.com/containers/bubblewrap) 0.8 or later and socat, installed for the whole system, which takes administrator access. On Ubuntu or Debian:

```bash
sudo apt-get install -y bubblewrap socat
curl -fsSL https://claude.ai/install-claude-science.sh | bash
```

[Its installation guide](https://claude.com/docs/claude-science/get-started) covers the other systems. The first time it starts, sign in with your Claude account (on a Pro, Max, Ultra, Team or Enterprise plan). The agent runs on that account's model. Start it from the repository's folder:

```bash
./agent.sh claude-science
```

Claude Science does not open any file in the folder by itself, so you need to tell it where SeqMaestro's instructions are. The first time, make a project for SeqMaestro and start a session in it. Then type the following, with the folder's path in place of `<the repository's folder>`:

```text
Read AGENTS.md in <the repository's folder> and follow it.
```

`./agent.sh` prints this message with the path filled in. The first time, the agent asks on a permission card to run shell commands, which it needs at every stage, so allow them for the whole conversation.

**`./agent.sh` starts Claude Science without its sandbox** (`--dangerously-no-sandbox`). The sandbox gives code read-only access to the folders you grant it, whereas every stage of a run writes into the repository's folder. The code the agent runs therefore has your access to your home folder and to the network, as it has in Claude Code, and Claude Science still asks you on a permission card before the agent runs code. Claude Science's documentation reserves the flag for a machine that is already isolated, such as a container or a disposable virtual machine.

## Other harnesses

A harness that reads [`AGENTS.md`](../../AGENTS.md) at the start of a session, or loads skills (instruction files) from `.agents/skills/`, follows SeqMaestro's instructions as they are: start it in the repository's folder and type `Start SeqMaestro.` Tell any other harness to read `AGENTS.md` and follow it.

Whichever harness you use, it must be able to:

- Run commands on your machine, with network access and the conda environments that `install.sh` built.
- Launch the run outside any sandbox it runs commands in, so that the run keeps going after the command that launched it returns.
