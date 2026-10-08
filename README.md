# SeqMaestro

SeqMaestro finds interpretable features of nucleic acid sequence, DNA or RNA. You give it sequences
with something measured for each, a class or a number, and it asks: *what property of the sequence
predicts that, stated in terms a biologist can follow?*

It asks the question several ways at once. Each way is an **arm**: one model fitted on one source
of features. The sources range from a fixed vocabulary, such as which base sits at which position
and counts of short motifs, to a language model that writes feature functions itself. Every arm is
scored by the same harness and fitted twice, once on the real labels and once on shuffled ones,
because a search for features finds structure in noise too. What survives is cut to models of a few
features, each feature is tested for whether it would be chosen again and whether the model needs
it, and a held-out part of the data is opened once at the end.

**What you get** is a report for the people who supplied the data: which properties of the sequence
carry the signal, the few-feature models worth showing, and how far each of their features can be
trusted.

A run is driven by an AI agent, [Claude Code](https://code.claude.com/docs/en/overview), which
prepares your data with you, starts the computation, checks it and writes the report. This page
takes you from a clone of the repository to a run that has started.

## What you need

- **Linux or macOS.** On Windows, use WSL. The installer has been run on Linux. It is written for
  macOS as well and has not yet been run there, so please report what it does.
- **conda.** Any of Miniforge, Miniconda, Anaconda, mamba or micromamba. If you have none, the
  installer offers to install [Miniforge](https://conda-forge.org/download/) into your home
  directory.
- **About 7 GB of disk** for the environments, and a network connection while they are built.
- **A language model to call**, for the arms in which a language model writes features and for
  the judge: an Azure OpenAI deployment, or a key of the OpenAI API or of the Claude API. Step 2
  says what to fill in for each.
- **Claude Code**, with an account for it. Step 3 gives the install command.
- For a report as a PDF, `pandoc`, `xelatex` and the DejaVu fonts. A run does not need them: the
  report is also written as Markdown.

## 1. Install

```bash
git clone https://github.com/DrShushen/SeqMaestro.git
cd SeqMaestro
./install.sh
```

The installer builds five conda environments. That took five minutes on the machine it was tested
on, and takes longer on a slow connection. Each part of SeqMaestro runs in an environment of its
own, so that the packages one method needs cannot change what another computes:

| environment | holds |
|---|---|
| `SeqMaestro_Core` | SeqMaestro itself: the models it fits, the program that drives a run, and the tests |
| `SeqMaestro_Method_deft` | DEFT, a decision tree whose candidate features a language model can write |
| `SeqMaestro_Method_igsr` | IGSR, a linear model over terms a language model writes |
| `SeqMaestro_Method_pysr` | PySR, a search for one short equation, with the Julia it runs on |
| `SeqMaestro_LocalUtils` | what draws a report's figures |

It builds the environments inside your conda and makes the file `.env` in the repository's folder.
Beyond that it leaves only the usual caches of the tools it runs (conda, pip, matplotlib) in your
home directory. Julia and its packages go inside the PySR environment, and nothing is written to
`~/.julia`. You do not have to activate anything afterwards: SeqMaestro starts each part with the
right environment by itself.

You can run `./install.sh` again at any time. It keeps an environment that exists and brings its
packages up to date, so run it again after you update the repository with `git pull`. To check an
installation without changing it:

```bash
./install.sh --check
```

It says, line by line, whether each environment can do its part, whether the credentials of step 2
are filled in, and what to run where something is missing. `./install.sh --help` lists the other
options, such as building one environment only.

## 2. Configure the language model

SeqMaestro calls one language model for everything a language model does: writing features, DEFT,
IGSR, and judging. It can be an Azure OpenAI deployment, a model of the OpenAI API, or a model of
the Claude API. The installer has made a file named `.env` in the repository's folder, from the
template `.env.example`. Open `.env` in an editor, set `SEQMAESTRO_LLM_PROVIDER` to `azure`,
`openai` or `claude`, and fill in that provider's values:

| provider | in `.env` | what it is | where to find it |
|---|---|---|---|
| `azure` | `SEQMAESTRO_AZURE_OPENAI_KEY` | the key of your Azure OpenAI resource | the Azure portal: your Azure OpenAI resource, then **Keys and Endpoint** |
| | `SEQMAESTRO_AZURE_OPENAI_ENDPOINT` | the resource's address, such as `https://my-resource.openai.azure.com/` | the same page |
| | `SEQMAESTRO_AZURE_OPENAI_DEPLOYMENT` | the name you gave the deployment, which can differ from the model's name | the Azure AI Foundry portal: **Deployments** |
| `openai` | `SEQMAESTRO_OPENAI_KEY` | an API key of the OpenAI platform | platform.openai.com: **API keys** |
| | `SEQMAESTRO_OPENAI_MODEL` | the model's name, such as `gpt-5.4` | the platform's list of models |
| `claude` | `SEQMAESTRO_CLAUDE_KEY` | an API key of the Claude Console | platform.claude.com: **API keys** |
| | `SEQMAESTRO_CLAUDE_MODEL` | the model's name, such as `claude-sonnet-5-5` | the Console's list of models |

The blocks of the providers you do not use can stay empty. For Azure, two more values seldom need
changing. `SEQMAESTRO_AZURE_OPENAI_API_VERSION`, the API version the deployment is called with, is
filled in already. `SEQMAESTRO_AZURE_OPENAI_BASE_MODEL` names the model family behind the
deployment. Its line is commented out, which stands for `azure/gpt-5.4`. If your deployment is of
another family, uncomment the line and give it as `azure/<model>`.

**The Claude key is not your Claude Code sign-in.** Claude Code, the agent of step 3, works under
your Claude account. `SEQMAESTRO_CLAUDE_KEY` pays for what Claude writes inside a run, and only
where you chose `claude` as the provider. It has a name of its own, and not the usual
`ANTHROPIC_API_KEY`, so that Claude Code does not take it for its own work.

Then check that the model answers:

```bash
./install.sh --check-llm
```

It makes one short call and prints the answer, or the provider's error.

SeqMaestro has been run in full with `gpt-5.4` on Azure OpenAI. With the OpenAI API and the Claude
API every part has been run on small tests only, with `gpt-5.4-nano` and `claude-haiku-4-5`, so
how good the features of another model are is yours to judge.

`.env` holds a secret. Git ignores it, the installer makes it readable by you alone, and the agent
is set up not to open it. Keep it out of anything you share.

A run makes many calls to the model, from up to 20 processes at once. If your account's rate
limit is low, tell the agent, which can start the run with fewer.

## 3. Run SeqMaestro with Claude Code

Install Claude Code and sign in, if you have not:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

[Its quickstart](https://code.claude.com/docs/en/quickstart) has the other ways to install it. The
first time you start it, it asks you to sign in, with a Claude subscription or a Claude Console
account.

Put your data in a folder under `data/`, which you make, in whatever form it arrived: a table with
the sequences and what was measured, a FASTA file with a table beside it, a collaborator's delivery
with its note. Git ignores `data/`, so nothing there is committed. Then, from the repository's
folder:

```bash
set -a && source .env && set +a
claude "/seqmaestro My data is in data/my_delivery. Each row is a 200 nt sequence around a binding site, with a column 'bound' that is 1 or 0. I want to know what in the sequence predicts binding."
```

The first line hands the credentials to everything the session starts. The second starts Claude
Code with SeqMaestro's instructions for it (`/seqmaestro`) and your request, written as you would
say it to a colleague. Say where the data is, what one row is, and what you want to predict. What
you leave out, the agent asks.

What happens then:

1. **Your data becomes a prediction task.** The agent looks at what you delivered, and builds the
   training, validation and held-out parts and a description of the task. Several decisions here
   are yours, and it puts each to you: what exactly is predicted, which rows are left out, what
   the models are told about the data. This stage ends with a summary that you approve.
2. **The run.** The agent estimates how long each arm will take, asks you about any that would
   take hours, and starts the computation. A run takes hours, and how many depends on the data
   and the arms. It runs detached from the session, so you can close the session while the machine
   stays on.
3. **The checks, and the held-out data.** When the run has finished, the agent checks it and
   tells you what it found. The held-out part is then opened, once, with your agreement. This
   cannot be undone, which is why it asks.
4. **The report.** The agent writes it from the tables and figures the run prepared. It is at
   `experiments/runs/<date>_<name>_full/report/REPORT.md`, and beside it as a PDF where `pandoc`
   and `xelatex` are installed.

Things to know while it works:

- **Claude Code asks before it runs a command or changes a file.** You approve each one, or allow
  a kind of command for the session. `Shift+Tab` changes how much it may do without asking, and
  its documentation describes the
  [permission modes](https://code.claude.com/docs/en/permission-modes).
- **To come back later**, run `claude -c` in the repository's folder, which continues the last
  conversation there, or start a new session and say which run to continue:
  `claude "/seqmaestro Continue the run in experiments/runs/<its folder>."`
- **You can also start `claude` with no message** and type `/seqmaestro` followed by your request
  once it is open.
- **Everything is kept on disk.** A run's folder holds its config, every fitted model, the logs and
  the report, and the agent's decisions about your data are written beside the dataset.

## To try a run before you bring data

The repository holds one example dataset: sequences of 60 nucleotides around a made-up binding
site, made of random bases with one motif planted in them. It shows what a prepared dataset looks
like. A cut-down run on it goes through every stage in about ten minutes and calls no language
model, so it is also a check of the installation that needs no credentials:

```bash
eval "$(./install.sh --interpreters)"     # CORE and the other environments' pythons
$CORE experiments/prepare/build_example_dataset.py
$CORE experiments/prepare/finalise_dataset.py --splits data/example/splits \
    --declared experiments/prepare/declarations/example__splits.yaml
$CORE experiments/runners/run_experiment.py --config experiments/configs/example_quick.yaml
```

The run goes to `experiments/runs/<date>_example_quick_full/`, which holds what a run on your data
will hold: the fitted models, the tables and the report's figures. It finds the planted motif, and
says nothing about biology. [`experiments/prepare/README.md`](experiments/prepare/README.md)
describes the example. The agent can take you through it too:
`claude "/seqmaestro Build the example dataset and run its quick config."`

## Running it by hand

Every step the agent takes is a command you can run yourself. One program drives a run from one
config file:

```bash
eval "$(./install.sh --interpreters)"     # CORE, DEFT, IGSR, PYSR and LOCALUTILS: each environment's python
set -a && source .env && set +a
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --stop-before-reveal
```

[`experiments/README.md`](experiments/README.md) describes the commands and a run's folder, and
[`process/`](process/README.md) what is done and checked at each stage, which is what the agent
follows.

## Where to read more

| | |
|---|---|
| [`process/`](process/README.md) | the process a run follows, one page per stage |
| [`docs/`](docs/README.md) | what each stage and analysis computes, and why it is computed that way |
| [`experiments/README.md`](experiments/README.md) | the commands, the configs and what a run's folder holds |
| [`ENVIRONMENTS.md`](ENVIRONMENTS.md) | what each environment holds and why they are separate |
| [`docs/repository.md`](docs/repository.md) | how the repository is laid out, for working on the code |

## If something goes wrong

- **The installer stopped.** Its last lines say which step failed. Fix what they name and run
  `./install.sh` again: it goes on from what is already built.
- **`./install.sh --check` reports a problem.** Each line marked `PROBLEM` ends with the command
  that repairs it. To build one environment afresh, remove it and build it again, for example
  `conda env remove -n SeqMaestro_Method_pysr` and then `./install.sh --only pysr`.
- **On macOS, `SeqMaestro_Core` cannot import XGBoost.** It needs the OpenMP runtime:
  `brew install libomp`, then `./install.sh --check`.
- **`./install.sh --check-llm` fails.** The message is the provider's. With Azure, the usual
  causes are an endpoint without `https://` or its final `/`, a deployment's name mistaken for the
  model's, and an API version the deployment does not support. With the OpenAI and Claude APIs,
  it is usually a model's name the key cannot use.
- **To remove SeqMaestro's environments**, run `conda env remove -n <name>` for each of the five
  in the table of step 1.
- **No PDF of the report.** Install `pandoc`, `xelatex` and the DejaVu fonts. On Ubuntu:
  `sudo apt install pandoc texlive-xetex fonts-dejavu`. On macOS: `brew install pandoc` and
  `brew install --cask mactex-no-gui font-dejavu`.
