# Running experiments

The `experiments/` folder holds the scripts that prepare datasets, start [runs](../docs/glossary.md#run) and analyse them, together with the run configs and the records of runs. This page explains how they are organised and why, the rules every script follows, and how to start a run.

## Organisation

Experiment code and run records are organised along two axes: how often a thing changes, and which part of a run it serves. A result can be reproduced months later only if each kind of file is kept apart on both axes.

| Kind | Where | Changes | Identified by |
|---|---|---|---|
| Library | `seqmaestro/` | Often, safely | Git and the test suite |
| Script | `experiments/{prepare,runners,analysis}/` | Occasionally | The commit it ran at |
| Run record | `experiments/runs/<date>_<name>_full/` | Never, once the run is over | Itself, since git ignores it |

The scripts are in three folders, by the part of a run they serve:

| Folder | Reads | Writes | When it runs |
|---|---|---|---|
| `prepare/` | Raw or upstream data | `data/` | Before any run |
| `runners/` | `data/` | A run's folder | During the run |
| `analysis/` | The results of one or more runs | Tables and reports | After the run |

The three kinds of script follow different rules, and a folder that mixed them would invite applying the wrong rules to a script:

- A preparation script may run once a year, and writes under `data/`, which git does not hold.
- A [stage](../docs/glossary.md#stage) of a run is started for that run alone, and must never overwrite another run's results.
- An analysis script reads a run's [cells](../docs/glossary.md#cell) and must be safe to run again, so it rewrites its own tables and leaves what the stages wrote unchanged.

The folders of `experiments/`:

| Folder | What it holds |
|---|---|
| [`prepare/`](prepare/README.md) | The scripts that build and check datasets |
| [`runners/`](runners/) | The [driver](../docs/glossary.md#driver) (`run_experiment.py`) and the scripts of the stages it starts, apart from the analyses and the report. Also `experiment_config.py`, which checks a config, and `record_agent.py`, with which the [agent](../docs/glossary.md#agent) records itself in a run |
| [`analysis/`](analysis/README.md) | The scripts that read runs back, and those that build a run's report |
| [`adapters/`](adapters/README.md) | The scripts that run an [arm](../docs/glossary.md#arm) in a vendored method's environment |
| [`configs/`](configs/README.md) | The run [configs](../docs/glossary.md#config), each of which states what one run is |
| [`datasets/`](datasets/README.md) | The [task descriptions](../docs/glossary.md#task-description), one for each dataset |
| `runs/` | One folder per run, which the driver names after the config (`<date>_<name>_full/`), and `_archive/`, which holds superseded runs. Git ignores the whole folder |

Most scripts under `runners/` and `analysis/` are short entry points into the package: each one's docstring names the module that holds its logic, and `--help` describes it. The docstring of [`seqmaestro/pipeline/layout.py`](../seqmaestro/pipeline/layout.py) lists every part of a run's folder, and the code takes the layout from that module.

**The files of one dataset are kept out of git.** Its builder under `prepare/`, its declaration, its description under `datasets/` and its configs under `configs/` stay on the machine that holds the dataset. The repository holds the files of one example dataset instead, and [`prepare/README.md`](prepare/README.md) says which files those are and why.

**A superseded run is moved to `runs/_archive/`.** Deleting it would destroy the only copy of its numbers, whereas leaving it beside the current runs would give two answers to one question, with nothing to tell the next reader which is current. The archive's README names what superseded each run, and each archived run holds a `SUPERSEDED.md`.

**A superseded dataset is moved to `data/_archive/`**, together with the runs built on it, for the same reasons and with the same two markers. A run whose data has been deleted cannot be checked against anything, so a dataset and its runs are always archived together.

## Rules for every script

**Logic belongs in the library.** A script holds little logic of its own, because logic that needs a test belongs in `seqmaestro/`, where the scripts call it ([`seqmaestro/qc/`](../seqmaestro/qc/) and [`seqmaestro/harness/effect.py`](../seqmaestro/harness/effect.py), for example).

**No results in code.** A docstring says what something does and why it is built that way. It does not state a measured AUROC, rate, effect size, count or stability score, because those describe a dataset or a run, and the code outlives both. To decide whether a number may stay, ask whether it is a property of the code or a finding about some data. A default's computational cost, a complexity bound and the arithmetic of a parameter are properties of the code, so they stay. A finding moves to one of the places below, and the docstring keeps the statement that the finding was evidence for.

| Finding about | Where it goes |
|---|---|
| The data | The dataset's `dataset.json` and its README |
| One run | That run's report |
| How an analysis behaves in general | [`analysis/README.md`](analysis/README.md) |

The rule guards against stale numbers. A committed table measured on a superseded dataset invites the next reader to trust it, and a gain that a docstring advertises can turn into a loss when the data changes, with nothing in the code to show it.

**A script is named for what it does.** A script that encodes one dataset's quirks names that dataset (`build_example_dataset.py`), because a generic name such as `build_dataset.py` in a shared folder invites someone to point it at data it does not fit.

**A script that writes into a run is told which run.** A stage's script requires `--run-dir` and has no default for it. Naming the run attaches each result to the run it belongs to, whereas a default would let one run be written on top of another.

- The driver is the only script that chooses a folder itself. Without `--run-dir`, it starts `experiments/runs/<date>_<name>_full/`, and it refuses to start a second run for a config that already has one ([starting a run](#starting-a-run)).
- An analysis script that works on one run takes `--run-dir`, or `--report-dir` for the run's `report/`, and writes into that run under names that no stage uses ([`analysis/README.md`](analysis/README.md) lists what each script takes and writes). An analysis run by hand over a dataset or over chosen files prints its results, or writes them to a path given explicitly.
- A preparation script does not produce a run, so `check_sequence_provenance.py` takes `--run-dir` only as an option, which names where its record is written.

**Only the [reveal](../docs/glossary.md#reveal) reads the [held-out split](../docs/glossary.md#held-out-split).** Every stage that fits or chooses works on the training and validation rows, and labels its numbers `val`. [`run_reveal.py`](runners/run_reveal.py) opens the test split once, at the end, and scores every model the run saved. From then on, the stages that fit refuse to run in that run's folder, so that the run's decisions cannot be revisited in the light of its test numbers, and adding an arm means a new run. [The held-out split and the reveal](../docs/developer/heldout.md) states the rule in full, with the test that enforces it.

## Preparation

Anything that changes what the data *is* belongs in `prepare/`, where it is recorded in a manifest, whereas in the library or in a stage of a run it would happen invisibly at fit time. A script that builds a dataset follows these rules:

- **A filter that discards rows reports how many it dropped, and refuses to drop many.** A silent filter that drops a sliver of one dataset can drop a third of the next and still report a good number. The window filter in [`build_example_dataset.py`](prepare/build_example_dataset.py) does both.
- **A filter is checked against the variable it names.** Check what a column counts before deciding what a threshold on it means. For example, a depth column that counts RNA molecules instead of reads is correlated with clonal expansion, so a threshold on it removes the pseudo-replicated members of expanded clones. A sweep over that threshold cannot tell this apart from removing shallow reads, because both explanations predict the same sweep.
- **Rows that are repeated measurements of one event are counted once.** The members of a clonal family share the mutation of their ancestor, and a site called in two replicates is one site. A builder counts each event once, keeps every distinct event, and records what the collapse removed. It needs an identifier of the event from the source, because a combination of other columns that looks unique can be shared by distinct events.
- **A design is sized before it is run.** Two quantities are cheap to compute: the effect that a split's test set can separate from chance, and how much information its training set holds. A design that cannot detect the effect would still spend a full grid reporting that it did not, so both quantities are part of the dataset. [`finalise_dataset.py`](prepare/finalise_dataset.py) measures them when Stage 1 ends and records them in `dataset.json`, and [`design_power.py`](prepare/design_power.py) answers the same question for a split that is still being planned.

**Stage 1 ends the same way for every dataset**, however it was built: with a [declaration](../docs/glossary.md#declaration) of what was decided, from which `finalise_dataset.py` counts and checks the rest and writes `dataset.json` beside the splits. A run config names only the split and its description, and takes the columns from `dataset.json`. A dataset that fails a check is refused before any compute is spent. [Stage 1: the prepared dataset](../docs/developer/dataset.md) states the standard in full, with every field of a declaration.

## Run records

A dated folder under `runs/` is the frozen record of one run. Nothing in it is edited afterwards except the write-ups.

**Each call of a run records its own provenance**, under `invocations/`. A call is one process of the run, such as one cell of an arm or the reveal, and its record holds the command, the resolved arguments, the commit, and whether the working tree was dirty ([`seqmaestro/provenance.py`](../seqmaestro/provenance.py)). The dirty flag matters most, because a commit recorded against modified files does not identify the code that ran, and nobody can reproduce a result produced that way, its author included. The code makes the record itself, so that no run depends on someone remembering to write it.

Commit before a run you intend to keep, so that the recorded commit identifies the code that ran.

## Results and write-ups

**A run is written up once, at the end**, after the reveal, from everything it produced. A stage before the reveal does not write a report, because it has only validation numbers, and a report on those would be taken for the result.

A run produces hundreds of megabytes, all of which the scripts can regenerate from the recorded invocations, and a repository is a poor place to version that much. Git therefore ignores the whole of `experiments/runs/`, including the write-ups beside the results, and two rules follow:

- **A write-up must be self-contained.** One that only points at `results/` says nothing once the results are cleared, so put the numbers in it.
- **A run record is the only copy of its numbers.** A report that is not copied somewhere durable (usually the write-up repository) is lost with the machine it sits on, and a clone cannot recover it. Copy anything you intend to cite when you write it.

**Every arm writes its results in one shape**, including the arms that run in another conda environment and cannot import `seqmaestro`, so that one analysis can read them all. A new arm must write that shape too. [The feature record](../docs/developer/feature_record.md) describes what a cell records about each feature, including which way the feature points, and [the adapters](adapters/README.md) describe how a vendored method hands its results back.

**Every cell keeps its fitted model**, so that a question the run did not answer, or one about sequences it never saw, needs a load instead of a refit, and so that the reveal scores the test split with the model each cell saved. A cell of a model fitted in Core saves the model in `fitted.pkl` together with its featuriser, because a model alone cannot turn a sequence into the columns it was fitted on ([`seqmaestro/harness/fitted.py`](../seqmaestro/harness/fitted.py)). A PySR model is saved in the same way, since its equation is kept as SeqMaestro's own expression tree, which loads without PySR. DEFT and IGSR save their models in their own formats, and each adapter predicts with what its bundle saved ([the held-out split and the reveal](../docs/developer/heldout.md)).

**An arm is named `<model>/<pool>`**, and that name is its folder under `results/` and its row in every table. Both halves are descriptive (`ebm/positional`, `igsr/llm_generated_joint`), so that a reader can tell what varied between two arms without a key. [`seqmaestro/arms.py`](../seqmaestro/arms.py) declares which model can be fitted on which [pool](../docs/glossary.md#pool), and [`configs/README.md`](configs/README.md) says how a config lists the arms. A new model or feature source also needs an entry in [`seqmaestro/report/naming.py`](../seqmaestro/report/naming.py), which gives the name a reader sees and raises an error on a name it does not know.

## Starting a run

The driver runs the whole pipeline from one config, unattended and resumable: the composition [baseline](../docs/glossary.md#baseline), every arm, the analyses, the [reduction](../docs/glossary.md#reduction) of each arm to a few features, the [reliability panel](../docs/glossary.md#reliability-panel), the reveal and the report's inputs. A run is started with two calls, so that it can be checked before the reveal:

```bash
eval "$(./install.sh --interpreters)"                       # CORE and the other environments' Python interpreters
set -a && source .env && set +a                             # the language model's credentials

$CORE experiments/prepare/<script>.py                       # only when the data changes
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --stop-before-reveal
# ... check the run: process/suite.md, few-feature-models.md, reliability-panel.md
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir experiments/runs/<date>_<name>_full
```

- **The first call stops before the reveal**, because the reveal cannot be undone. The driver also runs the reveal only when every stage before it has finished without a failure, and the reveal itself refuses a run whose cells or panels are incomplete.
- **The second call continues the same run.** Every stage starts again and skips what it has finished. Then come the reveal, the rest of the reliability panel and the report's inputs, whose figures are drawn under `SeqMaestro_LocalUtils`. A run that is already revealed resumes from the reveal. The report is then written as [`process/reporting.md`](../process/reporting.md) says.

**The language model's credentials are needed whenever a stage that calls the model is among those started**: an arm whose pool holds a [written source](../docs/glossary.md#written-source), the [recurrence analysis](../docs/glossary.md#recurrence-analysis) and the [judge](../docs/glossary.md#judge). The driver checks for them before it starts anything, under `--dry-run` too, and names the stages that need them. [`.env.example`](../.env.example) lists the variables.

**The driver's options** are listed by `run_experiment.py --help`, and the docstring of [`seqmaestro/pipeline/driver.py`](../seqmaestro/pipeline/driver.py) explains how the driver orders, resumes and checks the stages. `--dry-run` and `--estimate` run nothing: the first prints the resolved config, the budget and every command, and the second estimates each arm's time on the machine and flags the slow arms, which the agent puts to the user before a launch ([`process/suite.md`](../process/suite.md)). Two refusals keep the results of different runs apart:

- Without `--run-dir`, the driver refuses to start a second run for a config that already has one, so that the results are never split across two folders. `--new-run` says that a second run is meant.
- The driver refuses to continue a run under a config that differs from the one the run recorded, since a stage cannot tell a finished cell from one fitted under other settings. `--allow-config-change` continues all the same, and keeps the earlier config in the run's folder.

**The machine's budget.** As it starts, the driver probes the CPUs and the memory it may use, draws a budget from them, and runs the run's processes side by side within it ([`seqmaestro/pipeline/schedule.py`](../seqmaestro/pipeline/schedule.py)). `--max-cpus`, `--max-memory` and `--llm-slots` lower the budget on a machine that is in use for other work, and do not change any result. Each process is told the threads and the memory it may use through environment variables ([`seqmaestro/execution.py`](../seqmaestro/execution.py)), and its invocation record states its threads. A stage started by hand without those variables uses every core and sets itself no memory limit. `logs/machine.jsonl` records the probe and the budget of each call of the driver, and `logs/processes.csv` what each process took.

**The status page.** The driver keeps the run's [status page](../docs/glossary.md#status-page), `status.html` at the root of the run's folder, current while it runs ([the status page](../docs/developer/status.md)).

**Notifications.** With `NTFY_TOPIC` set in its environment, the driver sends a notification to that topic on [ntfy.sh](https://ntfy.sh/) when a stage fails and when the run ends. The topic is kept out of the config, so that a watched run records the same settings as an unwatched one. Anyone who knows the topic's name can read what is sent to it, so the name stays out of every tracked file.

## Running a single stage

To run one stage or one analysis of a run on its own:

```bash
RUN=experiments/runs/<date>_<name>_full

$CORE experiments/runners/run_arm.py           --run-dir $RUN --arms ebm/positional --seeds 42
$CORE experiments/analysis/analyse_stability.py --run-dir $RUN
```

**A stage takes every setting from the run.** It is given the run's folder, and reads the grid, the feature sources and the settings of each model, of the reduction and of the panel from the `config.yaml` that the driver recorded there. A stage takes none of them as a flag, so a stage started by hand cannot run under settings other than the run's. The flags of a stage under `runners/` only choose which of the run's arms, [label conditions](../docs/glossary.md#label-condition) and [seeds](../docs/glossary.md#seed) the call works on. The analyses take their settings in the same way, from the config's `analyses` section.
