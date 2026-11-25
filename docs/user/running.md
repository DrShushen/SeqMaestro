# Running with an agent

The [agent](../glossary.md#agent) is an AI model that works inside a program on your machine (its [harness](../glossary.md#agent-harness)), and runs commands and edits files with the permissions you give it. Following SeqMaestro's instructions, it prepares your data with you, starts the computation, checks the results and writes the report.

This page says how to start the agent, what it does at each stage, and which decisions are yours. The [overview](overview.md) explains the stages and the terms this page uses. To take a single table through a run without an agent, use [the command line](command-line.md) instead.

## Starting a run

First set up a harness for the agent, as [setting up an agent](agents.md) describes. Then make a folder under `data/` and put your data in it, in whatever form it arrived, for example:

- A table with the sequences and what was measured.
- A FASTA file, with a table beside it.
- A collaborator's delivery, with the notes that came with it.

Git ignores `data/`, so nothing there is committed. Start the agent from the repository's folder, with the name of your harness:

<img src="logos/claude-code.svg" alt="" height="24" class="harness-logo"> [Claude Code](agents.md#claude-code)
&emsp; <img src="logos/codex.svg" alt="" height="24" class="harness-logo"> [Codex](agents.md#codex)
&emsp; <img src="logos/openhands.svg" alt="" height="24" class="harness-logo"> [OpenHands](agents.md#openhands)
&emsp; <img src="logos/claude-science.svg" alt="" height="24" class="harness-logo"> [Claude Science](agents.md#claude-science)

```bash
./agent.sh claude          # or codex, openhands, claude-science
```

The agent checks the installation, tells you what it found, and asks where your data is, what one row of it holds and what you want to learn from it. Answer as you would answer a colleague, since the agent asks about anything you leave out. It also offers to continue a run that is already under `experiments/runs/`, or to try SeqMaestro on [the example dataset](example.md) first.

Claude Code and Codex are started with a first message, so the agent speaks first. OpenHands and Claude Science run in the browser and cannot be handed a first message, so for them `./agent.sh` prints the steps to open the repository's folder there, and the message to type.

## Working with the agent

The agent does the work of each stage, and stops wherever a decision is yours (a [checkpoint](../glossary.md#checkpoint)). The computation of Stages 2 to 3B, the reveal, and the report's tables and figures is done by the [driver](../glossary.md#driver), a program that runs them from the run's config without supervision. The agent starts the driver, follows its progress and checks what it produced.

```{mermaid}
sequenceDiagram
  actor You
  participant A as Agent
  participant D as Driver
  Note over You,D: Stage 1
  A->>You: Where the data is, what one row holds
  You->>A: Data and question
  A->>You: Questions about the data
  You->>A: Answers
  A->>You: Summary of the task
  You->>A: Approval
  Note over You,D: Stages 2, 3A and 3B
  A->>D: Estimate
  D-->>A: Time and cost of each arm
  A->>You: The grid of arms, with time and cost
  You->>A: The arms to keep
  A->>You: Settings, if you want to change any
  You->>A: Changes, or none
  A->>You: The load on your account
  You->>A: How many calls at once
  A->>D: Launch
  D-->>A: Stopped before the reveal
  A->>You: What the checks found
  You->>A: Agreement to the reveal
  Note over You,D: The reveal and the report
  A->>D: Reveal
  D-->>A: Held-out scores, tables, figures
  A->>You: Report
```

After your first answer, each arrow from you is a decision the agent waits for. The sections below take the stages in turn.

### Stage 1: task construction

The agent looks at what you delivered, and builds from it the training, validation and held-out splits, the [task description](../glossary.md#task-description) that the language models are told, and the run's [config](../glossary.md#config). Several decisions here are yours, and the agent puts each one to you, saying what each choice would change:

- What is predicted: a class or a number, and for a class, which rows belong to each.
- What one row stands for, where the delivery measures the same event more than once.
- Which measurements beside the sequence become [covariates](../glossary.md#covariate).
- The stretch of sequence each row keeps, and whether shorter sequences are padded or left out.
- Which rows are filtered out.
- What the models are told about the data, and whether a fact, such as the identity of the protein under study, is withheld from the models that write features. Withholding a fact is called [blinding](../glossary.md#blinding). A model told the name of the protein could write features from what it already knows about that protein, and the run could not show whether it had.
- Whether the run adds the motif sources.
- What to make of each check that measures a property of the dataset without judging it, such as identical sequences in more than one split.
- Whether to give [guidance](../glossary.md#guidance) on the form of the features the language models write: a few sentences, shown to each of them word for word, such as "Write each feature as a ratio or a difference of two counts". By default there is none. Guidance steers the search towards the forms it names, so features of other forms are less likely to be found. The report states the guidance a run was given.

The stage ends with a summary of each decision and its effect on the rows, of what the models are told and what is withheld, and of the questions that only the people who supplied the data can answer. Nothing is fitted until you approve it. If you supply more data or clarify something, the agent rebuilds the dataset and presents it again.

### Stages 2, 3A and 3B: the launch

Before the launch, the agent estimates each arm's time and cost on your machine:

- Its time, from a few fits on your data.
- For an arm whose features a language model writes, the number and cost of its calls to the model, and the time spent waiting for answers (measured with two small calls to the model).

It then puts three decisions to you:

- **The arms.** It shows the grid of arms, with the pools of feature sources down the side and the models across the top, each with its time and cost, and you choose which to keep. An arm that would take more than two hours is marked, with the options for it: leaving the arm out, leaving it out of the reduction or the reliability panel, or running the panel on fewer random halves of the data.
- **The settings, if you want to change any**, such as the number of seeds or a model's number of trees. The agent shows each setting with its value and what changing it does.
- **The load on your language model's account.** For each number of processes allowed to call the model at once, the agent shows the requests and tokens per minute and how long the calls take. Fewer processes spread the same calls over a longer time, at the same cost.

The agent then launches the driver in the background, so that the run keeps going after the agent's session ends. The driver runs Stages 2 and 3A, and the part of Stage 3B that needs no held-out rows, then stops before the reveal.

### Checks and the reveal

When the driver has stopped, the agent checks each stage and tells you what it found: whether every arm and analysis finished without a failure, and whether the fits on shuffled labels score near chance, as a control should. It then asks to open the [held-out split](../glossary.md#held-out-split). The [reveal](../glossary.md#reveal) opens it once and cannot be undone, which is why the agent waits for your agreement. If part of the run cannot be finished, the agent also asks you before it opens the held-out split for the rest.

After the reveal, the driver finishes the reliability panel on the held-out rows and prepares the report's tables and figures.

### The report

The agent writes the report from the tables and figures the run prepared. The report is `experiments/runs/<date>_<name>_full/report/REPORT.md`, with `REPORT.pdf` beside it if the [PDF tools](installation.md#pdf-tools) are installed. [Results](overview.md#results) describes what the report holds.

## Run time

A run takes hours, and how many depends on the data and the arms. The estimate before the launch gives each arm's time for Stages 2, 3A and 3B, and the total for the run. For the arms that call a language model, it also gives the time their calls take. The cost it shows does not include any discount your provider gives for cached prompts.

The run goes on in the background, so you can close the agent's session, as long as the machine stays on.

## Monitoring a run

The run's [status page](../glossary.md#status-page), `status.html` in the run's folder, shows where the run stands. Open it in a browser, where it reloads itself every 30 seconds while the run is going:

- Each stage has a column with what the stage has decided or found so far, and the checkpoint that ends it.
- What is running is marked.
- Details stay collapsed until you open them.
- Links lead to the files behind each number and, once it is written, to the report.

```{figure} status-page.png
:alt: The status page of a run during Stage 3B
:width: 100%

This is the status page of a quick run of the example dataset, during Stage 3B.
```

- **The run's folder** is `experiments/runs/<date>_<name>_full/`, named after the date and the config. Its `logs/driver.log` records what the driver started and when, what ended and how, and what failed. The agent follows the same logs, so you can also ask it where the run stands.
- **To come back later**, start the agent again with `./agent.sh` and tell it which run to continue, such as `experiments/runs/<its folder>`.
- **If a run stops part way**, for example because the machine restarted, the agent continues it. Every stage resumes from what is on disk, and finished work is kept.

## Permissions

The agent runs commands and changes files on your machine, and each harness has its own way of asking you before the agent acts ([supported harnesses](agents.md#supported-harnesses) compares them). A run has to keep going after the session that launched it ends. Where the harness runs commands in a sandbox, the agent therefore launches the run outside it, which the harness asks you to approve.

## Files on your machine

Everything a run produces stays in its folder ([results](overview.md#results)). The agent writes its decisions about your data, with the reasons for each, beside the dataset in its `README.md`.

Like the data, the files that Stage 1 makes for your dataset stay on your machine: the script that builds the dataset, the [declaration](../glossary.md#declaration) of what was decided, the task description and the config. Git ignores them, so they are never committed, and an update of SeqMaestro leaves them alone.
