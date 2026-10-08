---
name: suite
stage: 2
---

# Stage 2. Run the suite

Every arm the run names (each model on each pool of sources, on real and on shuffled labels, at
every seed) is fitted and scored on validation, and the analyses read what each one
found. One driver does all of it from one config. Your job is to launch it, watch it, recover what
fails, and check the result is whole before the next stage starts.

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
```

Each cell holds its validation scores and its fitted model. [`../docs/suite.md`](../docs/suite.md)
says how a cell is fitted and scored and what each of its files holds, and
`../docs/heldout.md` what it holds before the reveal. There is **no report** from this stage, and
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
nohup $CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir $RUN --stop-before-reveal > /dev/null 2>&1 &
```

**The launch discards what the driver prints, so confirm that it started.** A launch the driver
refuses before the run starts (a config it cannot load, a config that differs from the one a run
recorded, a missing credential or interpreter) writes nothing to `logs/driver.log`. A launch that
started writes its banner there within seconds. If the log has no new banner, run the same command
with `--dry-run` in place of `nohup` and the redirection, and it prints the refusal.

**Estimate the run, and put every slow arm to the user before the launch.** `--estimate` fits each
arm's model a few times on the run's data, on one thread as the run fits, and prints each arm's
time for Stage 2, the reduction and the panel, slowest first, with the run's total. It flags each
arm whose time would pass two hours (`--slow-hours` changes that), with what can be done about it:
leave the model out, leave the arm out of the panel or the reduction with a reason
(`not_profiled`, `not_reduced`), or lower the panel's pairs. Put each flagged arm to the user with
those options, and write their answer into the config before the launch. The estimate leaves out
the time a language model takes to answer, and marks the arms that have it. With `--run-dir` it is
kept in `logs/estimate.json`, beside `logs/processes.csv`, which records what each process then
took.

**Launch with `--stop-before-reveal`.** Without the flag the driver goes on through the reduction,
the reliability panel, the reveal and the report's inputs. With it, the driver stops before the
reveal, so you can make this stage's checks below and those of Stages 3A and 3B before the test
split is opened for good. Running it again with `--run-dir <run>` and without the flag then goes
on through the reveal ([`reliability-panel.md`](reliability-panel.md) says what that call runs).

The dry run prints the resolved config, the machine as the driver probes it, the budget it takes,
and every command, and preflights credentials, interpreters and the dataset. The driver runs every
arm as one process per labels and seed, side by side within that budget, with the composition
baseline first and the analyses once every arm has finished. `run_experiment.py --help` says what
waits for what and how much of the machine a run takes. On a machine in use for other work,
`--max-cpus`, `--max-memory` and `--llm-slots` lower it. `../experiments/configs/README.md` says
what a config may contain and how to run a small smoke version with `--set` before the full grid.

## Watch it, and recover

| to see | read |
|---|---|
| what the driver started and when, what ended and how, and what failed | `logs/driver.log` |
| one process's progress, validation numbers per cell as it finishes | `logs/<stage>.log`. A stage with several processes has a log for each: `logs/<model>__<pool>__<labels>__seed_<n>.log` for a cell of an arm, `logs/reduction__<model>__<pool>.log` and `logs/reliability__<model>__<pool>.log` for an arm's reduction and its panel |
| how long each process took, its threads, its CPU time and the most memory it held | `logs/processes.csv` |
| the machine as each invocation found it, and the budget it took | `logs/machine.jsonl` |
| cells that failed, with their tracebacks | `failures/<call>.json`, named as the call's log is. A call that ends clean removes its file, so a run with nothing broken has no file under `failures/`. A process that ended before it reached its cells (killed, or refused at its start) leaves no file, and shows as `exited <n>` in `logs/driver.log`, so read the two together. A failed PySR search quotes the end of its search process's log |

Every stage is resumable. **To continue an interrupted run, run the driver again with
`--run-dir <run>`**, the credentials exported as for the launch, and every `--set` the launch had:
finished cells are skipped. The driver refuses to resume onto a changed config. If you meant to
change it and the change does not affect the cells already finished, pass `--allow-config-change`
and note why. A DEFT cell whose tree was fitted but whose post-processing
failed is repaired with `run_arm.py --run-dir <run> --arms <model>/<pool> --labels <labels> --seeds
<n> --reuse-tree`, never refitted: refitting draws a different tree, and on a permuted cell that
quietly selects the null.

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
   equation in the features' names and the front it was chosen from: each equation's complexity, its loss on the
   rows searched and on the rows set aside. The chosen one has the lowest loss on the rows set
   aside. Check that the equation makes sense for its features, that `left_out` (equations the
   export could not express) is empty or rare, and that `metrics.json`'s `filled_rows` (validation
   rows on which the equation was undefined and took the constant) is small. A cell that exists has
   passed the check of Core's evaluation against PySR's own, since a fit that fails it raises. A
   PySR cell that chose the constant used no features: on permuted labels that is the null behaving
   as it should, and on real labels it says no equation the search found beat the constant on the
   rows set aside, which you report. `search_seconds` is the search's wall clock; the first search
   in each process also compiles Julia.

Validation numbers are for these checks and for the choices the later stages make. They are not
the result, and nothing is written up from them.

## Do not

- **Open `test.csv`, or run the reveal.** The reveal opens the held-out split once, for every stage
  together, after Stages 3A and 3B have done their pre-reveal work. Revealing now would close this
  run to those stages.
- **Change the grid in response to validation numbers without recording it.** A change after
  seeing results is a new config: say what changed and why, and if it affects finished cells,
  start a new run.
- **Adjust the composition baseline.** It is fixed so it means the same thing for every dataset.

## Beyond this file

This file is a floor. A run can fail in ways nobody has seen, and a result can be wrong in ways that
look fine. Follow what looks odd, record what you found and what you did, and bring anything that
affects the design to the user.
