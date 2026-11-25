# Running by hand

Every step the agent takes is a command you can run yourself. This page gives the commands in the order a run needs them, with links to the pages that explain each in full. [Running with an agent](running.md) describes what happens at each stage and which decisions it involves.

## Setup

Each part of SeqMaestro runs under the Python of its own environment. The first command below sets a shell variable to each environment's Python (`$CORE`, `$DEFT`, `$IGSR`, `$PYSR` and `$LOCALUTILS`). The second exports the language model's credentials, which every stage that calls the language model needs:

```bash
eval "$(./install.sh --interpreters)"
set -a && source .env && set +a
```

The commands on the rest of this page assume both.

## Preparing a dataset

Stage 1 by hand means making what the agent makes with you:

- A script that builds the training, validation and held-out splits from your data.
- A [declaration](../glossary.md#declaration) of what was decided.
- A [task description](../glossary.md#task-description).
- A [config](../glossary.md#config).

`experiments/prepare/finalise_dataset.py` then checks the splits against the declaration and writes `dataset.json` beside them. The [example dataset](example.md) has one file of each kind, and [`experiments/prepare/README.md`](../../experiments/prepare/README.md) says where each goes. [`process/task-construction.md`](../../process/task-construction.md) lists every decision and check of the stage.

## Running Stages 2 to 3B

The [driver](../glossary.md#driver) carries out a run from one config, in three steps:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --dry-run
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --estimate
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --stop-before-reveal --detach
```

1. **The dry run** (`--dry-run`) prints the config with the shared settings it extends filled in, and every command the run would issue. It checks the credentials, the interpreters and the dataset without running anything, and prints the settings each model is fitted with, which a config may change ([`experiments/configs/README.md`](../../experiments/configs/README.md)).
2. **The estimate** (`--estimate`) prints each arm's time on your machine and flags the slow arms. For the arms that call a language model, it prints their calls, cost and waiting time (measured with two small calls to the model), and the load on your account for each number of processes allowed to call the model at once. Last, it prints the grid of every arm's time and cost.
3. **The launch** (`--stop-before-reveal --detach`) starts the run in `experiments/runs/<date>_<name>_full/` and stops before the reveal. A run takes hours, so `--detach` starts it in the background, where it keeps running after the terminal closes. The command returns once the run is under way, and prints the paths of the run's log and of its status page. If the driver refuses the config, it says why before the command returns. To allow a number of processes other than 20 to call the language model at once, add `--llm-slots N`.

**To continue an interrupted run**, give the launch command again with `--run-dir` naming the run's folder. Keep `--stop-before-reveal` until you have made the checks below, because without it the driver opens the held-out split as soon as every stage before the reveal has finished. The driver skips every part that has finished.

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir experiments/runs/<date>_<name>_full --stop-before-reveal --detach
```

## The reveal and the report

Before you open the [held-out split](../glossary.md#held-out-split), make the checks that the agent's instructions in [`process/`](../../process/README.md) list for Stages 2, 3A and 3B. Then run the driver again on the run's folder, without `--stop-before-reveal`:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir experiments/runs/<date>_<name>_full
```

This opens the held-out split, finishes the reliability panel and builds the report's tables and figures. Name the run's folder exactly as it is on disk: given a folder that does not exist, the driver starts a new run, and this call would take that new run through its reveal.

You then write the report from those tables and figures, following the template that [`process/reporting.md`](../../process/reporting.md) describes. The [command line](command-line.md#the-report) can also have the language model write it, with `seqmaestro report <run>`.

## Further reading

| Page | Contents |
|---|---|
| [`process/`](../../process/README.md) | What is done and checked at each stage, which is what the agent follows |
| [`experiments/README.md`](../../experiments/README.md) | The commands, the configs and what a run's folder holds |
| [The developer guide](../developer/index.md) | What each stage and analysis computes, and why |
