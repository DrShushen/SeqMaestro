# How experiments are organised

Experiment code and records are separated along two axes. Collapsing either one makes a result
impossible to reproduce a few months later.

**By how often it changes:**

| | Where | Changes | Identified by |
|---|---|---|---|
| **Library** | `seqmaestro/` | often, safely | git, and the test suite |
| **Script** | `experiments/{prepare,runners,analysis}/` | occasionally | the commit it was at when it ran |
| **Run record** | `experiments/runs/<date>_<name>_full/` | never, once the run is over | itself (**not in git**) |

**By stage**, which gives the three script directories:

| | Reads | Writes | Runs |
|---|---|---|---|
| `prepare/` | raw or upstream data | `data/` | before any experiment |
| `runners/` | `data/` | a run's `results/` | the experiment itself |
| `analysis/` | one or more runs' `results/` | reports, tables | after the run |

```
experiments/
    prepare/                      build and check datasets   } -- see prepare/README.md
    runners/                      drive a run                } in git
    analysis/                     read runs back             } -- see analysis/README.md
    adapters/                     run an arm in another environment -- see adapters/README.md
    configs/                      what each *run* is, stated once -- see configs/README.md
    datasets/                     what each dataset *is*, described once -- see datasets/README.md
    runs/                         gitignored in full
        <date>_<name>_full/       one run, frozen, as the driver names it for the config <name>
            config.yaml           what it was configured with, which every stage reads
            invocations/          what was run, one record per call    } written by the scripts
            failures/             what failed in a call, absent when every call ended clean
            environment/          what it ran with  }
            logs/                 the driver's log, one log per process, how long each took and
                                  the threads and memory it held (processes.csv), the threads
                                  each panel may use now (threads/), and the machine as each
                                  invocation found it (machine.jsonl)
            baseline/seed_<n>/    the composition baseline, one cell per seed
            results/              the arms' cells, and the Stage 2 analyses' tables
            proposals/            the features a language model proposed, one file per proposal
            judge/                the judge's scores, where the config runs it
            reduction/            Stage 3A: each arm reduced to a model of N features
            reliability/          Stage 3B: each reduced model's reliability panel
            reveal.json           written once, when the test split is opened
            report/               after the reveal: what the report is written from, and the report
        _archive/                 superseded runs -- see _archive/README.md
```

`seqmaestro/pipeline/layout.py` is where that layout is written down for the code.

**The files of one dataset are not in git either.** Its builder under `prepare/`, its declaration,
its description under `datasets/` and its configs under `configs/` stay on the machine that holds
the dataset, and the repository holds one example dataset whole.
[`prepare/README.md`](prepare/README.md) says which files those are and why.

**A superseded run moves to `runs/_archive/`** instead of being deleted or left in place. Deleting
it would destroy the only copy of its numbers. Leaving it beside the current runs would give two
live answers to one question, with no way for the next reader to tell which is current. The
archive's README names what superseded each run, and each archived run carries a `SUPERSEDED.md`.

**A superseded dataset moves to `data/_archive/`** together with the runs built on it, for the same
reasons and with the same two markers. A run whose data has been deleted cannot be checked against
anything, so a dataset and its runs are always archived together.

**`experiments/runs/` is gitignored in full.** Git holds the scripts that produce a run, and
[Results and write-ups](#results-and-write-ups) says what that means for a run's numbers.

The stage split exists because the three kinds of script follow different rules, and keeping them
in one directory invites applying the wrong rules to a script. A preparation script may run once a
year, and its output goes under `data/`, which git does not hold. A stage of a run is started for
that run alone and must never overwrite another run's results. An analysis script reads a run's
cells and must be safe to run again: it writes its tables again and changes nothing a stage wrote.

## Scripts, whichever stage

A script should hold little logic of its own. When a script grows logic that needs a test, that is
a sign the logic belongs in `seqmaestro/`, as it did for `seqmaestro.qc` and
`seqmaestro.harness.effect`.

**No results in code.** A docstring says what something does and why it is built that way. It does
not carry a measured AUROC, rate, effect size, count or stability score, because those describe a
dataset or a run and the code outlives both. To decide whether a number may stay, ask whether it is
a property of the code or a finding about some data. A default's cost, a complexity bound and the
arithmetic of a parameter are properties of the code, so they stay. A finding moves to one of the
places below, and the docstring keeps the statement the finding was evidence for.

Where findings go:

| finding about | goes to |
|---|---|
| the data | the dataset's `dataset.json` and its README |
| one run | that run's report |
| how an analysis behaves in general | `analysis/README.md` |

The rule guards against stale numbers. A committed table measured on a superseded dataset invites
the next reader to trust it, and a gain that a docstring advertises can turn into a loss when the
data changes, with nothing in the code to show it.

Name a script for what it does. A script that encodes one dataset's quirks names that dataset,
because a generic name in a shared directory invites someone to point it at data it does not fit.
For example, `build_example_dataset.py` and not `build_dataset.py`.

## The scripts that drive a run

`runners/` holds the driver (`run_experiment.py`) and one script for each stage it starts. They
drive the library to produce a run, and they live outside the dated run directories because the
same scripts serve every run.

**A stage's script requires `--run-dir` and has no default for it.** Naming the run is what
attaches a result to the run it belongs to, and a default would let one run be written on top of
another. The driver is the one script that can name the directory: without `--run-dir` it starts
`experiments/runs/<date>_<name>_full/`, and it refuses to start a second run for a config that
already has one ([Starting a run](#starting-a-run)).

**No stage reads the test split, except the reveal.** Every stage that fits or selects works on
the training and validation rows. Its log lines say `val`, and `reduction/summary.csv`, the one
summary table written before the reveal, names its columns `val_<metric>_<model>`.
`run_reveal.py` opens the test split once, at the end: it writes `reveal.json`, predicts the test
rows with every model the run saved, and scores them into each cell's files. From then on the
stages that fit refuse to run in that directory, so a run's decisions cannot be revisited in the
light of its test numbers, and adding an arm means a new run.
[`../docs/heldout.md`](../docs/heldout.md) is the rule in full; `tests/test_heldout.py` fails if a
module other than `seqmaestro.harness.heldout` names the split.

**A `prepare/` script produces no run.** `check_sequence_provenance.py` takes an optional
`--run-dir`, as the place its record is written.

**An `analysis/` script that works on one run is told which.** The three `analyse_*.py`,
`run_judge.py`, `build_report_inputs.py` and `export_features_xlsx.py` take `--run-dir`, and
`build_report_figures.py` and `assemble_report.py` take `--report-dir`, the run's `report/`. Each
writes into that run under names of its own (`results/stability.csv`, `judge/`, `report/`). The
analyses run by hand over a dataset or over chosen files (`direct_effect_test.py`,
`feature_group_dependence.py`) print, or write to a path named explicitly.

## Preparation

Anything that changes what the data *is* belongs in `prepare/`, so that it is recorded in a
manifest. In the library or in a stage of a run it would happen invisibly at fit time.

A filter that discards rows must **report how many it dropped, and refuse to drop many**. A silent
filter that drops a sliver of one dataset can drop a third of the next and still report a good
number. The window filter in `prepare/build_example_dataset.py` is an example.

**A filter must also be checked against the variable it names.** Check what a column counts before
deciding what a threshold on it means. For example, a depth column that counts RNA molecules instead
of reads is correlated with clonal expansion, so a threshold on it removes pseudo-replicated members
of expanded clones. A sweep over that threshold cannot tell this apart from removing shallow reads,
because both explanations predict the same sweep.

**Rows that are repeated measurements of one event are counted once.** The members of a clonal
family share the mutation of their ancestor, and a site called in two replicates is one site. A
builder counts each event once, keeps every distinct event, and records what the collapse removed.
It needs an identifier of the event from the source, because a combination of other columns that
looks unique can be shared by distinct events.

**Size a design before running it.** Two quantities are cheap to compute: the effect a split's test
set can separate from chance, and how much information its training set holds. A design that cannot
detect the effect will still spend a full grid reporting that it did not, so both quantities are
part of the dataset. `prepare/finalise_dataset.py` measures them when Stage 1 ends and records them
in `dataset.json`, and `prepare/design_power.py` answers the same question for a split still being
planned.

**Every dataset ends the same way.** However it was built, Stage 1 finishes with a declaration of
what was decided: the columns, the kind of target and what it means (each class of a binary
target, or what a continuous target measures), what each covariate measures, the window (and,
where the sequences differ in length, the positional window), the grouping, and every filter with
what it cost. It then runs `prepare/finalise_dataset.py`, which counts and checks the rest and
writes `dataset.json` beside the splits. A run config names only the split and its
description, and the columns come from `dataset.json`. A dataset with a failed check is refused
before any compute is spent. [`../docs/dataset.md`](../docs/dataset.md) is the standard in full.

## Run records

A dated directory is the frozen record of one run. Nothing in it is edited afterwards except the
write-ups.

**Each call of a run records its provenance** through `seqmaestro.provenance`, under
`invocations/`: the exact command, the resolved arguments after defaults, the commit, the
environment, and **whether the working tree was dirty**. The dirty flag matters most, because a commit recorded against modified files does not
identify the code that ran, and a result produced that way cannot be reproduced by anyone, including
its author.

The recording is automatic because a note someone has to remember to write gets skipped on the
first busy day.

## Results and write-ups

**A run is written up once, at the end**, after the reveal, from everything it produced. The
suite writes no report of its own: before the reveal it has only validation numbers, and a report
on those would be read as the result.

Runs produce hundreds of megabytes, all of it regenerable from the scripts plus the recorded
invocation, and a repository is a poor place to version that. So the whole of `experiments/runs/`
is ignored, including the write-ups beside `results/`.

Two rules follow.

**A write-up must be self-contained.** One that only points at `results/` says nothing once the
results are cleared. Put the numbers in it.

**A run record is the only copy of its own numbers.** Nothing under `runs/` is in git, so a report
that is not copied somewhere durable (usually the write-up repository) is lost with the machine it
sits on, and a clone cannot recover it. Copy anything you intend to cite when you write it.

Every arm writes results in the same shape, including arms that run in another conda environment
and cannot import `seqmaestro`. That lets one analysis read them all, and any new arm must do the
same.

**Every cell keeps its fitted model** in `fitted.pkl`, beside the features and scores it reported.
The featuriser is saved with it, because a model alone cannot turn a sequence into the columns it
was fitted on. `seqmaestro.harness.fitted.load_fitted` returns the pair as one object that predicts.
A question the run did not answer, or one about sequences it never saw, therefore costs a load
instead of a refit, and the reveal scores the test split with this saved model. The vendored arms
are the exception: DEFT serialises its own tree into its bundle, in a format only DEFT can read,
and IGSR saves its model and every term its search accepted into its bundle, as JSON whose terms
only IGSR runs. Each adapter predicts with what its bundle saved. A reduced or refitted IGSR model
is least squares saved as `model.pkl`, predicting from terms IGSR's side evaluates. A PySR model
searches in its own environment but keeps its equation as SeqMaestro's own tree
(`seqmaestro/models/equation.py`), so its `fitted.pkl` loads and predicts in Core, without PySR.

**Every run starts with the composition baseline**: the sparse linear model of the target's kind
(logistic for a binary target, least squares for a continuous one) on the standardised base
fractions of each window, fitted at every seed of the grid and written to `baseline/`. Its
settings are fixed, so it means the same thing for every dataset.
[`../docs/baseline.md`](../docs/baseline.md) says what it is.

**What a cell records about each feature is documented in full** in
[`../docs/feature_record.md`](../docs/feature_record.md), which is maintained alongside the
record itself. The two paragraphs below are the parts a reader of this page needs.

**Every feature records which way it points**, in two fields that answer different questions.
`direction` is the class the arm's *model* uses the feature to argue for, or which way it moves a
continuous target. It is taken from how that model works (a coefficient's sign, a shape function's
trend, an attribution, a tree's splits) and reduced to one of three words, per kind of target, that
mean the same thing for every arm. `marginal_effect` is how far the feature goes with the target
*on its own*, with no model, measured over every training row against the real target. It is
computed in sample because it describes the association the model learned from, so it says nothing
about whether that association holds elsewhere. When the effect is far enough from zero to have a
reliable sign and the two fields disagree, the model is using the feature to correct for something
else. That is a finding about the model, not a fault. `seqmaestro.harness.direction` is the
definition, and the report and the significance judge both use it. The judge is shown the sign and
never the size, so that permuted-label features, where a run judges them too, stay
indistinguishable from the real ones.

**An arm is named `<model>/<pool>`**, and that name is its directory under `results/` and its row
in every table. Both halves are descriptive (`ebm/positional`, `xgboost/llm_generated`,
`deft_or_cart/positional+llm_generated_joint`, `igsr/llm_generated_joint`), so a reader can tell
what varied between two arms without a key. A pool joins its sources with `+`
([`configs/README.md`](configs/README.md), Arms), and `seqmaestro/arms.py` declares which model can
be fitted on which pool. A new model or source also needs an entry in `seqmaestro.report.naming`,
which gives the name a reader sees and raises on a name it does not know.

## Starting a run

The whole pipeline, from one config, unattended and resumable: the baseline, every arm, every
analysis, the reduction of each arm to N features ([`../docs/reduction.md`](../docs/reduction.md)),
the reliability panel ([`../docs/reliability.md`](../docs/reliability.md)), the reveal
([`../docs/heldout.md`](../docs/heldout.md)), and the report's inputs:

```bash
eval "$(./install.sh --interpreters)"                       # CORE and the other environments' pythons
set -a && source .env && set +a                             # the language model's credentials

$CORE experiments/prepare/<script>.py                       # only when the data changes
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --stop-before-reveal
# ... check the run: process/suite.md, few-feature-models.md, reliability-panel.md
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir experiments/runs/<date>_<name>_full
```

**The credentials are needed whenever a stage that calls a language model is among those started**:
an arm on a pool with a written source, the recurrence analysis, and the judge. The driver checks
for them before it starts anything, under `--dry-run` too, and names the stages that need them.
`.env.example` lists the variables.

**The reveal cannot be undone**, so the first call stops before it. The driver runs the reveal only
when every stage before it finished clean. The reveal itself also refuses a run whose cells or
panels are incomplete, and afterwards nothing that fits can run in the directory.

**The second call starts every stage again, and each keeps what it has finished.** The cells, the
reductions and the panels are skipped where they are complete. Stability and co-discovery are
computed again, which gives the same tables. The recurrence analysis and the judge ask a language
model, so they keep the tables on file while the features they were computed from are unchanged.
Then come the reveal, the panel's tables and the report's inputs (the figures under
`SeqMaestro_LocalUtils`). A run that is already revealed resumes from the reveal and runs nothing
before it. The report is then written as [`../process/reporting.md`](../process/reporting.md)
says.

`--dry-run` prints the resolved config and every command it would issue, without running any.
`--estimate` estimates each arm's time on the machine from a few fits on the run's data, and flags
the arms that would be slow, also without running any, and `process/suite.md` puts those arms to
the user before a launch. `--run-dir <existing>` continues an interrupted run. Without it, the
driver refuses to start a second run for a config that already has one, so the results are never
split across two directories, and `--new-run` says that a second run is meant. The driver also
refuses to continue a run under a config that differs from the one the run recorded, since a stage
cannot tell a finished cell from one fitted under other settings. `--allow-config-change` continues
all the same, and keeps the earlier config beside the new one as `config_superseded_<n>.yaml`.
What a config may contain is in `configs/README.md`.

**How a run uses the machine.** As it starts, the driver probes the CPUs and the memory it may
use, draws a budget from them, and starts the run's processes side by side within it
(`seqmaestro/pipeline/schedule.py`; `run_experiment.py --help` says what waits for what).
`--max-cpus`, `--max-memory` and `--llm-slots` lower the budget on a machine in use for other
work, and change no result. The driver gives each process its threads through `SEQMAESTRO_THREADS`
and the memory it may hold through `SEQMAESTRO_MEMORY`, in bytes, by which a reliability panel
decides how many workers to run (`seqmaestro/execution.py`, and `ENVIRONMENTS.md` for the
threads). A stage started by hand without them uses every core and sets itself no limit.
`logs/machine.jsonl` records the probe and the budget of each call of the driver, and
`logs/processes.csv` what each process took.

**Notifications.** With `NTFY_TOPIC` set in its environment, the driver sends a notification to
that topic on ntfy.sh when a stage fails and when the run ends. The topic is no part of the
config, so a watched run records the same settings as an unwatched one. Anyone who knows the
topic's name can read what is sent to it, so it stays out of every tracked file.

To run one stage or one analysis of a run on its own:

```bash
RUN=experiments/runs/<date>_<name>_full

$CORE experiments/runners/run_arm.py           --run-dir $RUN --arms ebm/positional --seeds 42
$CORE experiments/analysis/analyse_stability.py --run-dir $RUN
```

A stage takes the run directory and reads the run's settings from the `config.yaml` the driver
recorded there: the grid, the sources, each model's settings, the reduction's and the panel's. No
stage takes one of them as a flag, so a stage started by hand cannot run under other settings
than the run's. The flags of a stage under `runners/` only say which of the run's arms, label
conditions and seeds the call works on. The analyses take theirs in the same way, from the
config's `analyses` section: how many features of a cell each takes, how often the judge rates
one, how the recurrence analysis groups.

The scripts under `runners/` and `analysis/` are entry points. The code they call is
`seqmaestro/pipeline/`, and for the report's figures, its assembly and the spreadsheet,
`seqmaestro/report/`.

Commit before a run you intend to keep, so the recorded commit means something.
