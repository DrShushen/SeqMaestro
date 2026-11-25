# Running SeqMaestro as an agent

This folder holds the instructions an agent follows when it runs the SeqMaestro process: this file
and one file for each stage. Read this file first, and each stage's file when that stage begins.

| stage | file | what it produces |
|---|---|---|
| 1 | [`task-construction.md`](task-construction.md) | a prepared dataset: splits, `dataset.json`, the task description the models are told, and the run config |
| 2 | [`suite.md`](suite.md) | every arm fitted and analysed, on training and validation data |
| 3A | [`few-feature-models.md`](few-feature-models.md) | each arm reduced to a small model |
| 3B | [`reliability-panel.md`](reliability-panel.md) | three kinds of evidence for every candidate feature, and the reveal of the held-out split |
| report | [`reporting.md`](reporting.md) | one document for the people who supplied the data |

Most of your judgement goes into Stage 1, which ends when the user signs off what you built.
Before the launch, the user chooses the arms, any settings and the parallelism from the estimate of
each arm's time and cost ([`suite.md`](suite.md)). From then on the driver (`run_experiment.py`) runs Stages 2 to 3B, the reveal and the report's inputs
from one config, while you launch it, watch it, recover what fails and check the result. **Launch
it with `--stop-before-reveal`**, make the checks that Stages 2, 3A and 3B list, and only then run
it again to open the held-out split, because the reveal cannot be undone. You write the report
yourself, from the tables and figures the driver prepared.

## These files are a floor, not a ceiling

Each file lists what its stage must do and must leave behind. **It does not list everything worth
doing.** If you notice something that matters and the file does not mention it (a distribution
that looks wrong, a label that seems to run the other way, a delivery that disagrees with its note,
a result that is too good), investigate it. Record what you did and why beside the stage's other
records, and raise anything that affects the design with the user. A checklist cannot anticipate a
dataset nobody has seen, so an agent that does only what it lists can miss what matters most.

## Rules that hold at every stage

- **The held-out split stays closed until the reveal.** After Stage 1, `test.csv` is neither read
  nor used for any number until the reveal in Stage 3B opens it once for every stage together.
  Stage 1 makes the split and counts its rows and classes, and does nothing else with it.
- **Use the shared code rather than writing new code.** `seqmaestro/` and `experiments/` already do
  most of what a stage needs, and each stage file names the parts. A second implementation of
  something the library does gives a second answer, with nothing to say which one is right. Where
  you do need new code (typically a builder for a new dataset), put it where the stage file says and
  name it for what it does.
- **Record the decision, not only the outcome.** Write down what you chose, what else you
  considered, and what the choice gives up, because a later reader has only your record.
- **Nothing about one dataset in shared code, and no silent defaults.** A column name, a split path
  or a task description defaulted in shared code is one dataset's answer applied silently to the
  next one.
- **Record who you are in each run you work on.** The report names the agent that carried out the
  analysis and the model it runs on. The code cannot find out either of these, so you state them.
  When you start a stage of a run, record yourself for it:

  ```bash
  $CORE experiments/runners/record_agent.py --run-dir <run> --agent "<the agent you are>" \
      --model "<the model you run on>" --stages <stage>
  ```

  Give the model by the exact name your harness tells you, such as `claude-opus-5-5`, and the
  stages as the table above names them (`1`, `2`, `3A`, `3B`, `report`). Stage 1 is carried out
  before the run exists, so record it at the launch, with Stage 2 (`--stages 1 2`). A later
  session that carries out a stage again records itself for that stage too.

## The interpreters the commands name

Each part of the pipeline runs in a conda environment of its own, so the commands in these files
name each environment's python by a variable: `$CORE` for `SeqMaestro_Core`, which runs every
stage, and `$DEFT`, `$IGSR`, `$PYSR` and `$LOCALUTILS` for the others. One command sets them all:

```bash
eval "$(./install.sh --interpreters)"
```

`./install.sh --check` says whether every environment is built and whether the language model's
credentials are filled in. Make it the first thing you run.

## Where things are written down

| kind of fact | where |
|---|---|
| what a stage or analysis computes, and why | [`../docs/`](../docs/developer/index.md) |
| how scripts and configs are organised and run | `../experiments/README.md` and the READMEs beside it |
| what you do and decide at each stage | here |
| the conda environments, and which runs what | [`../ENVIRONMENTS.md`](../ENVIRONMENTS.md) |
| how the repository is laid out | [`../docs/developer/architecture.md`](../docs/developer/architecture.md) |
| what a user is told about installing and starting a run | [`../README.md`](../README.md) |
