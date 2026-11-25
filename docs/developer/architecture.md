# Architecture and layout

This page is the place to start on SeqMaestro's code: how a run moves through it, where each part of the repository is, and the two boundaries that shape the design. A user who wants to install and run SeqMaestro starts from the [user guide](../user/index.md) instead.

## The flow of a run

A run is carried out by the [agent](../glossary.md#agent) with the user, in three steps:

1. In Stage 1, the agent prepares the dataset with the user and writes the run's [config](../glossary.md#config).
2. The [driver](../glossary.md#driver) then runs every stage up to the report's inputs. Each stage runs as processes that take their inputs from the run's directory and write their outputs into it.
3. The agent checks each stage and writes the report.

The [command line](../glossary.md#command-line) can take the agent's place for a dataset made from one table ([the command line](#the-command-line), below).

```{mermaid}
flowchart TD
  subgraph s1["Stage 1: agent and user"]
    D["Delivered data"] --> P["Prepared dataset<br/>dataset.json"]
    P --> C["Run config"]
  end
  C --> B["Composition baseline"]
  B --> A["Stage 2: every arm,<br/>real and shuffled labels,<br/>every seed"]
  A --> N["Stage 2 analyses"]
  A --> R["Stage 3A: reduction"]
  R --> T["Stage 3B: reliability panel"]
  N --> V["The reveal"]
  T --> V
  V --> I["Report inputs:<br/>facts, tables, figures"]
  I --> W["The report,<br/>written by the agent"]
```

Each stage is one module of {py:mod}`seqmaestro.pipeline`, whose docstring lists them. A script under [`experiments/runners/`](../../experiments/runners/) starts the stage, given the run's directory, and the stage takes every setting from the config recorded there. The driver ({py:mod}`seqmaestro.pipeline.driver`) knows which stage waits for which and how much of the machine each may use, and starts each stage when it can. A stage started by hand behaves as it does under the driver. {py:mod}`seqmaestro.pipeline.layout` lists where a run keeps each thing it writes.

## The repository

| Path | What it holds |
|---|---|
| `seqmaestro/` | The shared library, importable as `seqmaestro`: featurisation, the evaluation harness, the proposer, the judge, the reduction, the reliability panel, the report, and the stages of a run (`seqmaestro/pipeline/`). It names no dataset |
| `methods/` | The vendored methods, DEFT and IGSR, each with its own environment and tests. Nothing here imports `seqmaestro` |
| `experiments/` | The scripts that prepare datasets, drive runs and analyse them, the shared run configs, the adapters that run a vendored method, and the example dataset ([running experiments](experiments.md)) |
| `docs/` | This documentation: the user guide, this developer guide and the API reference generated from the docstrings |
| `process/` | One instruction file per stage for the agent, and the template it writes a report against. These guide the agent and are not part of this site |
| `AGENTS.md` | The instructions that most agent harnesses load when a session starts: act as the SeqMaestro orchestrator and follow the skill |
| `.agents/skills/seqmaestro/` | The skill: the brief the agent follows from its first message to the report, which points at `process/`. `.claude/skills/seqmaestro` links to it, because Claude Code looks for skills there |
| `agent.sh` | Starts an agent harness in the repository, with the first message where the harness takes one ([setting up an agent](../user/agents.md)) |
| `.codex/config.toml` | Codex's settings for the repository: network access for the commands it runs in its sandbox |
| `install.sh` | Builds the conda environments and checks them ([environments](environments.md)) |
| `tests/` | The test suite of the library and the pipeline, run in `SeqMaestro_Core`. Each vendored method has its own suite under `methods/` |
| `data/`, `experiments/runs/` | Datasets and run records, which git ignores. Anything worth citing from them is copied somewhere durable when it is written |
| The files of one dataset | Its builder, declaration, description and configs under `experiments/`, and the tests of its builder. Git ignores them, so they stay on the machine that holds the dataset, and the repository holds only the example dataset's ([preparing a dataset](preparation.md)) |

## The agent's instructions

Every [agent harness](../glossary.md#agent-harness) gives the agent the same instructions, in three layers:

1. **`AGENTS.md`**, which most harnesses load when a session starts, tells the agent to act as the orchestrator and to follow the skill from its first step, even when the first message is only a greeting.
2. **The skill** ([`.agents/skills/seqmaestro/SKILL.md`](../../.agents/skills/seqmaestro/SKILL.md)) says how to open the conversation and how to work in the repository, and points at `process/`.
3. **[`process/`](../../process/README.md)** has one file per stage, which says what is done and checked at that stage.

Codex and OpenHands find the skill under `.agents/skills/`, whereas Claude Code looks under `.claude/skills/`, where a link leads to the same folder.

None of the three layers names a harness, so a new harness leaves them unchanged. Adding a harness takes:

- A case in `agent.sh`, which starts the harness in the repository with the first message, or prints the steps to take where the harness cannot be given one.
- A section in [setting up an agent](../user/agents.md).

The harness must be able to run commands on the user's machine with the conda environments, and to launch the run outside any sandbox it runs commands in. The driver's `--detach` then starts the run in a session of its own, so that the run keeps going after the command that launched it returns.

## The command line

The [command line](../glossary.md#command-line) ({py:mod}`seqmaestro.cli`) is a front end to the same stages, for a dataset made from one table. It works through the same files and commands as the agent:

1. It prepares the dataset with fixed defaults, writes the files of Stage 1 where the agent writes them, and finalises the dataset with {py:func}`seqmaestro.dataset.finalise`.
2. It writes the run's config as an extension of a shared config, and starts the driver through the driver's command line, as a user would type it.
3. It follows the run through the status page's gatherer ({py:mod}`seqmaestro.status.gather`), so that the terminal and the status page show the same state.
4. When the run has finished, one call to the language model writes the report's source against the agent's template ({py:mod}`seqmaestro.cli.report`). Every placeholder in the source is checked against the run's tables, figures, components and settings before [`assemble_report.py`](../../experiments/analysis/assemble_report.py) expands it.

Nothing else in the package imports the command line, and the agent's way through a run does not depend on it. A change to anything the command line relies on (the files of Stage 1, the driver's command line, the gatherer, the shared configs or the report template) is therefore checked against it in the same commit.

Every question the command asks has an option, and {py:class}`seqmaestro.cli.ui.Questions` takes each answer from the option, from the default under `--yes`, or from the user, in that order. A question about the data has no default, so a scripted run must answer it. The defaults the command takes are listed in the docstrings of {py:mod}`seqmaestro.cli.prepare` (the rows, the lengths and the split) and {py:mod}`seqmaestro.cli.launch` (the presets).

## Two boundaries

**Vendored methods and the library never import each other.** They exchange files instead, so that their environments can disagree about every package they contain. A vendored arm's cell is prepared in Core as files, handed to the method's adapter under the method's interpreter, and read back as a bundle ([the adapters](adapters.md)). PySR's search and the MEME Suite's motif discovery cross the same kind of boundary.

```{mermaid}
flowchart LR
  subgraph core["SeqMaestro_Core"]
    S["The stages,<br/>the harness,<br/>the Core models"]
  end
  subgraph files["Files in the run"]
    IN["A cell's inputs"]
    BU["The bundle<br/>handed back"]
  end
  subgraph methods["Environments of their own"]
    DE["DEFT"]
    IG["IGSR"]
    PY["PySR's search"]
    ME["STREME and Tomtom"]
  end
  S --> IN --> DE & IG & PY & ME
  DE & IG & PY & ME --> BU --> S
  S --> F["SeqMaestro_LocalUtils:<br/>the report's figures"]
```

**Results are kept apart from the code that produced them.** The code is identified by the commit it ran at, and a result by its own record: each command of a run records the command line, the commit and the time it ran at, and each environment it used is snapshotted into the run ({py:mod}`seqmaestro.provenance`).

## The module map

The packages of `seqmaestro/`, each with what it holds. The [API reference](../api/index) lists every module inside them.

| Package | What it holds |
|---|---|
| {py:mod}`arms <seqmaestro.arms>` | What an arm can be: which model can be fitted on which pool, and how |
| {py:mod}`target <seqmaestro.target>` | The kinds of target (binary and continuous), and the arithmetic that differs between them |
| {py:mod}`dataset <seqmaestro.dataset>` | The Stage 1 standard: a prepared dataset and the `dataset.json` that describes it |
| {py:mod}`grouping <seqmaestro.grouping>` | The rows a split must not separate |
| {py:mod}`qc <seqmaestro.qc>` | Checks on a dataset before it is modelled |
| {py:mod}`features <seqmaestro.features>` | Featurisation (rows in, a named numeric matrix out) and the feature record |
| {py:mod}`motifs <seqmaestro.motifs>` | Motifs as position frequency matrices: loading them, matching them to sequences and discovering them |
| {py:mod}`models <seqmaestro.models>` | The models fitted in Core: sparse linear, additive (EBM), boosted (XGBoost) and symbolic (PySR) |
| {py:mod}`harness <seqmaestro.harness>` | The evaluation harness, which fits and scores every arm with the same code, and the held-out split |
| {py:mod}`llm <seqmaestro.llm>` | What every part that calls a language model shares: the client and the task description |
| {py:mod}`proposer <seqmaestro.proposer>` | The proposer, which asks a language model for features as code, and the guard that lets the code run |
| {py:mod}`judge <seqmaestro.judge>` | The judge, which rates how meaningful a feature would be without being told which arm produced it |
| {py:mod}`reduction <seqmaestro.reduction>` | Stage 3A: each arm reduced to a model of a few features |
| {py:mod}`reliability <seqmaestro.reliability>` | Stage 3B: the reliability panel's three tiers |
| {py:mod}`report <seqmaestro.report>` | A revealed run turned into the facts, tables and figures a report is written from |
| {py:mod}`status <seqmaestro.status>` | The run's status page, built from its files while it runs ([the status page](status.md)) |
| {py:mod}`cli <seqmaestro.cli>` | The command line, `seqmaestro` ([the command line](#the-command-line)) |
| {py:mod}`pipeline <seqmaestro.pipeline>` | The stages of a run and the driver that runs them |
| {py:mod}`execution <seqmaestro.execution>`, {py:mod}`files <seqmaestro.files>`, {py:mod}`provenance <seqmaestro.provenance>` | How a process runs (threads and memory), writing a file whole, and the record of what produced a result |

## Documentation layout

| Kind of fact | Where it is written |
|---|---|
| How to install, configure and start a run | The [user guide](../user/index.md) |
| What a stage or analysis computes, and why | This developer guide |
| What each module, class and function is | The [API reference](../api/index) |
| How scripts and configs are organised and run | [Running experiments](experiments.md) and the READMEs beside the code |
| What the agent does and decides at each stage | [`process/README.md`](../../process/README.md) |
| The conda environments, and which runs what | [Environments](environments.md) |
