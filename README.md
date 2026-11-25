# SeqMaestro

SeqMaestro finds interpretable features of DNA and RNA sequences. You give it sequences with one measurement for each, either a class (such as whether a protein was found bound to the sequence) or a number (such as the strength of a signal). SeqMaestro then looks for the properties of the sequence that predict the measurement, and describes each one in terms a biologist can follow.

To find them, SeqMaestro:

- Fits many kinds of model on many kinds of feature, including features that a language model writes as code.
- Compares every fit with the same model fitted on shuffled labels, because a search for features also finds patterns in noise.
- Reduces each model to a few features and tests how far each of those features can be trusted.
- Sets part of the data aside at the start and scores the final models on it only after every choice has been made.

An AI agent goes through these steps with you, in a program such as Claude Code or Codex. It prepares your data with you, runs the computation, checks the results and writes a report for the people who supplied the data. The [user guide](https://drshushen.github.io/SeqMaestro/user/) explains the three steps below in full, and its [overview](https://drshushen.github.io/SeqMaestro/user/overview.html) describes how SeqMaestro works.

## 1. Install

You need Linux or macOS (on Windows, use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)) and [conda](https://docs.conda.io/en/latest/). If you have no conda, the installer offers to install it.

```bash
git clone https://github.com/DrShushen/SeqMaestro.git
cd SeqMaestro
./install.sh
```

The installer builds SeqMaestro's conda environments and checks them, which takes several minutes and about 7 GB of disk. [Installation](https://drshushen.github.io/SeqMaestro/user/installation.html) lists what else you need and what the installer does.

## 2. Configure the language model

SeqMaestro calls a language model to write features and to analyse them. It can reach the model through [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai/), [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry), the [OpenAI API](https://platform.openai.com/) or the [Claude API](https://platform.claude.com/).

1. Open the file `.env`, which the installer made in the repository's folder.
2. Set `SEQMAESTRO_LLM_PROVIDER` to `azure`, `azure_foundry`, `openai` or `claude`.
3. Fill in that provider's key and model. [Configuring the language model](https://drshushen.github.io/SeqMaestro/user/language-model.html) says where to find each value.
4. Check that the model answers:

   ```bash
   ./install.sh --check-llm
   ```

## 3. Start a run

Install one of the four programs the agent can work in, which [setting up an agent](https://drshushen.github.io/SeqMaestro/user/agents.html) describes:

<img src="docs/user/logos/claude-code.svg" alt="" height="24"> [Claude Code](https://code.claude.com/docs/en/overview) &emsp;
<img src="docs/user/logos/codex.svg" alt="" height="24"> [Codex](https://github.com/openai/codex) &emsp;
<img src="docs/user/logos/openhands.svg" alt="" height="24"> [OpenHands](https://docs.openhands.dev) &emsp;
<img src="docs/user/logos/claude-science.svg" alt="" height="24"> [Claude Science](https://claude.com/docs/claude-science/get-started)

Then put your data in a folder under `data/`, and start the agent from the repository's folder:

```bash
./agent.sh claude     # or codex, openhands, claude-science
```

The agent checks the installation, then asks where your data is, what one row of it holds and what you want to learn from it. [Running with an agent](https://drshushen.github.io/SeqMaestro/user/running.html) describes what the agent does at each stage and which decisions are yours. To see a whole run before you bring your own data, try [the example dataset](https://drshushen.github.io/SeqMaestro/user/example.html), which runs in minutes without a language model.

### Without an agent

The command line takes a single table of sequences through a run to its report without an agent, and asks only what it cannot decide for itself ([running from the command line](https://drshushen.github.io/SeqMaestro/user/command-line.html)):

```bash
conda activate SeqMaestro_Core
seqmaestro run my_sequences.csv
```

## Documentation

The documentation is published at [drshushen.github.io/SeqMaestro](https://drshushen.github.io/SeqMaestro/) from the markdown pages in [`docs/`](docs/). Its user guide has these pages:

- [Overview](https://drshushen.github.io/SeqMaestro/user/overview.html)
- [Installation](https://drshushen.github.io/SeqMaestro/user/installation.html)
- [Configuring the language model](https://drshushen.github.io/SeqMaestro/user/language-model.html)
- [Setting up an agent](https://drshushen.github.io/SeqMaestro/user/agents.html)
- [Running with an agent](https://drshushen.github.io/SeqMaestro/user/running.html)
- [Running from the command line](https://drshushen.github.io/SeqMaestro/user/command-line.html)
- [The example dataset](https://drshushen.github.io/SeqMaestro/user/example.html)
- [Running by hand](https://drshushen.github.io/SeqMaestro/user/by-hand.html)
- [Troubleshooting](https://drshushen.github.io/SeqMaestro/user/troubleshooting.html)

The [developer guide](https://drshushen.github.io/SeqMaestro/developer/) explains what each stage computes and why.

## Licence

SeqMaestro is licensed under the [Apache License 2.0](LICENSE). The copies of DEFT and IGSR in `methods/` are licensed under the MIT License ([DEFT's licence](methods/deft/LICENSE), [IGSR's licence](methods/igsr/LICENSE)). Claude Code, Claude Science, Codex and OpenHands, whose logos appear in this README and the user guide, are trademarks of their owners.
