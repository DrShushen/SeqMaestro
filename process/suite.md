---
name: suite
stage: 2
---

# Stage 2. Run the suite

Every arm the run names (each model on each pool of sources, on real and on permuted labels, at
every seed) is fitted and scored on validation, and the analyses then examine what each one found.
The driver does all of it from one config. Your job is to launch it, watch it, recover what fails,
and check that the result is complete before the next stage starts.

Read [`README.md`](README.md) in this folder first if you have not. This stage begins only once the
user has approved the dataset and the config at the end of Stage 1.

## What this stage must leave behind

```
experiments/runs/<date>_<name>_full/
    config.yaml, invocations/, environment/, logs/
    baseline/seed_<n>/                                  the composition baseline, one cell per seed
    results/<model>/<pool>/<labels>/seed_<n>/           one cell per arm, labels and seed
    results/*.csv, judge/, proposals/                   the analyses (the judge where the config runs it),
                                                        and the proposals the arms used
    motifs/                                             where a pool holds a motif source: the run's copy of
                                                        each collection, every scan the run made, and
                                                        each discovery, with STREME's and Tomtom's output
```

Each cell holds its validation scores and its fitted model. [`../docs/developer/suite.md`](../docs/developer/suite.md)
says how a cell is fitted and scored and what each of its files holds, and
`../docs/developer/heldout.md` what it holds before the reveal. There is **no report** from this stage, and
no test numbers anywhere.

## Run it

```bash
eval "$(./install.sh --interpreters)"            # CORE and the other environments' pythons
RUN=experiments/runs/$(date +%F)_<name>_full     # write the name down: later calls give it as it is
set -a && source .env && set +a       # when a pool holds a written source, or the judge runs
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir $RUN --dry-run
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --estimate --run-dir $RUN
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir $RUN --stop-before-reveal --detach
```

**Run the launch outside any sandbox your harness runs commands in.** `--detach` starts the run in
a session of its own, so it keeps running after the command and the session that started it end.
A sandbox can still end every process a command started as soon as the command returns, so run
the launch outside it, which your harness does once the user approves.

**The launch checks the run before it returns.** A config the driver cannot load, a config that
differs from the one the run recorded, or a missing credential or interpreter is refused with the
reason, and nothing starts. Otherwise the launch watches the run for a few seconds, says that it
is under way, and names its journal (`logs/driver.log`) and its status page. A minute later,
check that it is still running (`pgrep -af run_experiment.py`).

**Once it has started, record who you are** (`README.md`, "Rules that hold at every stage"), with
Stage 1 where you carried it out (`--stages 2` where another session did):

```bash
$CORE experiments/runners/record_agent.py --run-dir $RUN --agent "<the agent you are>" \
    --model "<the model you run on>" --stages 1 2
```

**Before the launch, the user decides the arms, the settings and the parallelism, from the
estimate.** `--estimate` fits each arm's model a few times on the run's data, on one thread as the
run does, and prints each arm's time for Stage 2, the reduction and the panel, slowest first, with
the run's total. It flags each arm whose time would pass two hours (`--slow-hours` changes that),
with what can be done about it. For the arms that call a language model it then prints their calls,
the tokens, the cost and the time spent waiting for answers, measured with two small calls to the
model (well under a cent), and the load on the provider at each number of language-model slots.
Last it prints the grid: the pools down, the models across, and each arm's time and cost. Put three
things to the user, in this order:

1. **The grid.** Show it as the estimate prints it, with the totals, and ask which arms to keep.
   Give each flagged arm its options: leave the model out, leave the arm out of the panel or the
   reduction with a reason (`not_profiled`, `not_reduced`), or lower the panel's pairs. An arm the
   user drops comes out of the dataset's config (the model from `arms.<pool>.models`, or the pool
   set to `null`), and one they add goes in, which the dry run checks.
2. **The settings, only if the user wants to change any.** Ask whether they do. If so, show the
   settings in `../experiments/configs/README.md` ("The settings a user may change at the launch")
   with the values the dry run prints, and say what each change does. Write a change into the
   dataset's config.
3. **The parallelism.** Show the threads and the memory the run takes, and the load table: at each
   number of slots for processes that call a language model, the calls in flight, the requests and
   tokens a minute, and how long the arms' calls take. Fewer slots spread the same calls over a
   longer time, so the cost stays the same and the run takes longer. Ask the user to compare the
   requests and tokens a minute with their account's rate limits. Their answers become the launch's
   `--llm-slots`, `--max-cpus` and `--max-memory`.

Run the estimate again when an answer changes the config. Without the language model's
credentials the estimate counts the calls and says that their cost and waiting need the
credentials. With `--run-dir` it is kept in `logs/estimate.json`, beside `logs/processes.csv`,
which records what each process of the run then took.

**Launch with `--stop-before-reveal`.** Without the flag the driver goes on through the reduction,
the reliability panel, the reveal and the report's inputs. With it, the driver stops before the
reveal, so you can make this stage's checks below and those of Stages 3A and 3B before the
held-out split is opened for good. Running it again with `--run-dir <run>` and without the flag then goes
on through the reveal ([`reliability-panel.md`](reliability-panel.md) says what that call runs).

The dry run prints the resolved config, the machine as the driver probes it, the budget it takes
and every command, and it runs the preflight checks of the credentials, the interpreters and the
dataset. The driver runs every arm as one process per label condition and seed, side by side
within that budget, with the composition baseline first, the scan of the known motifs and the
discoveries of motifs where a pool holds them (the arms on them wait for these), and the analyses
once every arm has finished. `run_experiment.py --help` says what waits for what and how much of
the machine a run takes. On a machine in use for other work, `--max-cpus`, `--max-memory` and
`--llm-slots` lower it. `../experiments/configs/README.md` says what a config may contain and how
to run a small smoke version with `--set` before the full grid.

## Watch it, and recover

| to see | read |
|---|---|
| where the run stands, stage by stage, with what is running and each stage's checkpoint | `status.html` at the run's root, which the driver keeps current. Give the user its path once the run has started |
| what the driver started and when, what ended and how, and what failed | `logs/driver.log` |
| one process's progress, validation numbers per cell as it finishes | `logs/<stage>.log`. A stage with several processes has a log for each: `logs/<model>__<pool>__<labels>__seed_<n>.log` for a cell of an arm, `logs/reduction__<model>__<pool>.log` and `logs/reliability__<model>__<pool>.log` for an arm's reduction and its panel, and `logs/motifs__motif_known.log` and `logs/motifs__motif_discovered__<labels>__seed_<n>.log` for the scan and the discoveries before the arms |
| how long each process took, its threads, its CPU time and the most memory it held | `logs/processes.csv` |
| the machine as each invocation found it, and the budget it took | `logs/machine.jsonl` |
| cells that failed, with their tracebacks | `failures/<call>.json`, named as the call's log is. A call that ends clean removes its file, so a run with nothing broken has no file under `failures/`. A process that ended before it reached its cells (killed, or refused at its start) leaves no file, and shows as `exited <n>` in `logs/driver.log`, so read the two together. A failed PySR search quotes the end of its search process's log |

Every stage is resumable. **To continue an interrupted run, run the driver again with
`--run-dir <run>`**, the credentials exported as for the launch, and every `--set` the launch had:
finished cells are skipped. The driver refuses to resume onto a changed config. If you meant to
change it and the change does not affect the cells already finished, pass `--allow-config-change`
and note why. A DEFT cell whose tree was fitted but whose post-processing failed is repaired with
`run_arm.py --run-dir <run> --arms <model>/<pool> --labels <labels> --seeds <n> --reuse-tree`,
never refitted, because refitting draws a different tree, and on a permuted cell that quietly
chooses which draw from the null the run keeps.

## Check before moving on

1. **Every cell is there.** `experiments/runners/run_reveal.py --run-dir <run> --check` lists any
   cell the config expects that is missing or incomplete, and reveals nothing.
2. **The controls look like controls.** Permuted-label cells should score near chance on
   validation: an AUROC near 0.5 for a binary target, an R-squared near zero (or below it) for a
   continuous one. One that does not points to a leak and stops the run until it is explained.
3. **Nothing is broken.** A real-label arm at chance, a cell that used no features, a proposer arm
   whose features mostly failed the guard or still failed on the rows after the repair round
   (`token_usage.json`: `proposal_stats` and `fit_report`, which count the written features of
   the whole cell, and `rounds` for each round of an iterative arm), a judge that refused most
   items: each is a fault to find, not a result.
4. **The analyses ran**: `results/stability.csv`, `results/co_discovery.csv`, the recurrence
   tables (`results/concept_recurrence.csv` and the files beside it) where an arm's pool holds a
   written source (`llm_generated`, `llm_generated_iterative` or `llm_generated_joint`), and
   `judge/` where the config runs the judge (`grid.judge`). A run with no such arm has nothing to
   group, and its recurrence analysis says so in its log and writes no table.
5. **A PySR cell's equation.** Its `run.json` holds, under `model_settings.fitted`, the chosen
   equation in the features' names and the front it was chosen from: each equation's complexity,
   and its loss on the rows searched and on the rows set aside. The chosen one has the lowest loss
   on the rows set aside. Check that the equation makes sense for its features, that `left_out`
   (equations the export could not express) is empty or rare, and that `metrics.json`'s
   `filled_rows` (validation rows on which the equation was undefined and the constant was used)
   is small. Every cell that exists has passed the check of Core's evaluation against PySR's, since
   a fit that fails it raises an error. A PySR cell that chose the constant used no features. On
   permuted labels that is the null behaving as it should, whereas on real labels it means that no
   equation the search found beat the constant on the rows set aside, which you report.
   `search_seconds` is the search's wall-clock time, and the first search in each process also
   compiles Julia.

Validation numbers are for these checks and for the choices the later stages make. They are not
the result, and nothing is written up from them.

## Do not

- **Open `test.csv`, or run the reveal.** The reveal opens the held-out split once, for every stage
  together, after Stages 3A and 3B have done their pre-reveal work. Revealing now would close this
  run to those stages.
- **Change the grid in response to validation numbers without recording it.** A change after
  seeing results is a new config: say what changed and why, and if it affects finished cells,
  start a new run.
- **Adjust the composition baseline.** It is fixed so that it means the same thing for every dataset.

## Beyond this file

This file is a floor. A run can fail in ways nobody has seen, and a result can be wrong in ways that
look fine. Follow what looks odd, record what you found and what you did, and bring anything that
affects the design to the user.
