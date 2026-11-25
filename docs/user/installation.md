# Installation

This page takes you from a clone of the repository to a checked installation. A single script, `install.sh`, builds every part of SeqMaestro in conda environments and checks the result.

## Requirements

- **Linux or macOS.** On Windows, use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (the Windows Subsystem for Linux). The installer is tested on Linux. It is written for macOS as well, but is untested there, so please report how it behaves.
- **conda.** Any of Miniforge, Miniconda, Anaconda, mamba or micromamba. If you have none, the installer offers to install [Miniforge](https://conda-forge.org/download/) into your home directory.
- **About 7 GB of disk** for the environments, and a network connection while they are built.
- **A language model to call**, for the arms in which a language model writes features and for the judge. This can be an Azure OpenAI deployment, a model deployed on Azure AI Foundry, or a key for the OpenAI API or the Claude API. [Configuring the language model](language-model.md) says what to fill in for each.
- **An [agent harness](../glossary.md#agent-harness)**, the program the AI agent works in: Claude Code, Codex, OpenHands or Claude Science, with an account for the language model the agent runs on. [Setting up an agent](agents.md) says how to install each. The [command line](command-line.md) needs no harness.
- **For a report as a PDF**, pandoc, XeLaTeX and the DejaVu fonts ([PDF tools](#pdf-tools)). A run does not need them, since the report is also written as Markdown.

## Running the installer

```bash
git clone https://github.com/DrShushen/SeqMaestro.git
cd SeqMaestro
./install.sh
```

The installer builds five conda environments, which takes about five minutes, or longer on a slow connection. Each part of SeqMaestro runs in its own environment, so that the packages one method needs cannot change what another computes:

| Environment | Contents |
|---|---|
| `SeqMaestro_Core` | SeqMaestro itself: the models it fits, the program that drives a run, the command line (`seqmaestro`) and the tests |
| `SeqMaestro_Method_deft` | DEFT, a decision tree whose candidate features a language model can write |
| `SeqMaestro_Method_igsr` | IGSR, a linear model over terms that a language model writes |
| `SeqMaestro_Method_pysr` | PySR (a search for one short equation) and the [Julia](https://julialang.org/) it runs on |
| `SeqMaestro_LocalUtils` | The code that draws a report's figures |

[`ENVIRONMENTS.md`](../../ENVIRONMENTS.md) describes each environment in full and explains why they are kept apart.

The installer also makes the file `.env` in the repository's folder, from the template `.env.example`, to hold the language model's settings. In your home directory it leaves only the usual caches of the tools it runs (conda, pip and matplotlib). Julia and its packages go inside the PySR environment, which leaves `~/.julia` untouched.

You do not need to activate an environment afterwards, because SeqMaestro starts each part in the right environment itself. The one exception is the command `seqmaestro`, which you run in `SeqMaestro_Core` (`conda activate SeqMaestro_Core`).

## Motif discovery

A run can find motifs in your sequences with [STREME](https://meme-suite.org/meme/doc/streme.html), from the [MEME Suite](https://meme-suite.org/meme/). The MEME Suite needs one more environment, `SeqMaestro_Method_meme`, which the installer builds only when asked:

```bash
./install.sh --with-motifs
```

The MEME Suite comes from the [bioconda](https://bioconda.github.io/) channel. It is free for educational, research and non-profit use, whereas commercial use needs a licence from UC San Diego ([its copyright notice](https://meme-suite.org/meme/doc/copyright.html)). Scanning your sequences for the published motifs of known proteins needs nothing beyond the five environments above.

## Checking the installation

The installer checks what it built before it finishes. To check an installation again without changing it:

```bash
./install.sh --check
```

The check reports, line by line:

- Whether each environment can do its part.
- Whether the language model's settings in `.env` are filled in.
- Whether the PDF tools are installed.

For anything missing, it says what to run. `./install.sh --check --with-motifs` checks the MEME Suite's environment as well, and `./install.sh --help` lists the other options, such as `--only`, which builds a single environment.

The quick run of [the example dataset](example.md) goes through every stage of a run without a language model, so it tests the whole installation before you have filled in any credentials.

## Updating

You can run `./install.sh` again at any time. It keeps the environments that exist and brings their packages up to date, so run it again after you update the repository with `git pull`.

## PDF tools

The report is written as Markdown, and rendered as a PDF where [pandoc](https://pandoc.org/), [XeLaTeX](https://tug.org/xetex/) and the [DejaVu fonts](https://dejavu-fonts.github.io/) are installed. They come from your system's package manager, outside conda:

- On Ubuntu: `sudo apt install pandoc texlive-xetex fonts-dejavu`
- On macOS, with [Homebrew](https://brew.sh/): `brew install pandoc` and `brew install --cask mactex-no-gui font-dejavu`

## Uninstalling

Remove each of the five environments in the table above with `conda env remove -n <name>`, and `SeqMaestro_Method_meme` as well if you built it. Removing `SeqMaestro_Method_pysr` removes Julia with it.

To remove SeqMaestro completely, delete the repository's folder as well, after copying anything you need out of `data/` and `experiments/runs/`.
