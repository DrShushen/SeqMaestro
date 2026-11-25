---
name: reliability-panel
stage: 3B
---

# Stage 3B. The reliability panel and the reveal

Stage 3A left each arm with a model of a few features. For every feature that model uses, and every
feature it could have used instead, this stage examines three things: whether the arm's selection
would have chosen it on another half of the data, whether it goes with the target on its own
(separating the classes, or rising and falling with a continuous target), and whether the model
gets worse without it. The last two are measured on the held-out split, which this stage opens
once, for every stage together.

Read [`README.md`](README.md) in this folder first if you have not. This stage begins when the
reduction has finished and been checked ([`few-feature-models.md`](few-feature-models.md)).

## What this stage must leave behind

```
experiments/runs/<date>_<name>_full/
    reliability/<model>/<pool>/               one panel per profiled arm: its record, its tiers,
                                              its leave-one-out refits as cells, its three tables
    reliability/summary.csv                   one row per arm
    reveal.json                               what the reveal sealed, and when
```

The docstring of [`../seqmaestro/reliability/panel.py`](../seqmaestro/reliability/panel.py) says what each file holds,
and [`../docs/developer/reliability.md`](../docs/developer/reliability.md) how each number is computed. [`../docs/developer/heldout.md`](../docs/developer/heldout.md) says what the reveal does.

## Run it

The driver runs all of it when the config has a `reliability` section, which `_shared.yaml` gives
every config. After the reduction it runs the part of the panel that comes before the reveal. Then,
unless it was launched with `--stop-before-reveal` as [`suite.md`](suite.md) says to, it runs
the reveal, the panel's Tier 3 and tables, and the report's inputs. It runs the reveal only when
every stage before it finished clean. So, once you have stopped before the reveal and made the
checks below, continue the run:

```bash
RUN=experiments/runs/<date>_<name>_full       # the directory the launch made, as it is named
set -a && source .env && set +a               # as for the launch
$CORE experiments/runners/run_reveal.py --run-dir $RUN --check        # is everything ready?
$CORE experiments/runners/run_experiment.py --config <config> --run-dir $RUN --detach
```

**Name the directory the launch made.** The driver starts a new run in a `--run-dir` that does not
exist, and this call has no `--stop-before-reveal`, so a mistyped name would take a new run through
its reveal. Give every `--set` the launch had too, or the driver refuses the call as a changed
config. As at the launch, run it outside any sandbox your harness runs commands in, and check a minute later that it is still running.

**What this call runs.** The run is not yet revealed, so the driver starts every stage again, and
each keeps what it has finished: the cells, the reductions and the panels are skipped. Stability
and co-discovery are computed again and give the same tables. The recurrence analysis and the
judge call a language model, so each keeps the tables you checked as long as the features, the
settings and the model they were computed from are unchanged. The driver still asks for the
credentials before it starts, and they must name the model the launch called: with another model,
both analyses would call it again and replace the tables you checked. Then the driver runs the
reveal and what follows it.

A revealed run resumes from the reveal and does not run the stages before it again. To recover one
step, run it on its own:

```bash
$CORE experiments/runners/run_reliability.py --run-dir $RUN          # before the reveal
$CORE experiments/runners/run_reveal.py --run-dir $RUN                # the reveal
$CORE experiments/runners/run_reliability.py --run-dir $RUN --after-reveal
```

**Tier 1 is the expensive part.** It runs a full selection on every half-sample, twice
`reliability.pairs` of them per arm (a hundred at the shared setting) and three such sweeps on the
check arm, and on an EBM arm over a large pool that takes core-hours. The driver runs one panel
process per arm, each once that arm's reduction has finished, and gives each, as it starts, every
thread that is free, so the panels run one at a time. It also lends a panel the threads that free
up while it runs, and takes them back for a process whose turn comes. When a panel is run by hand,
it uses every core unless `--workers` says otherwise. Every part resumes from what is on disk, so
running an interrupted panel again continues it. DEFT's arms are fitted in DEFT's environment and
do not call a language model. An IGSR arm's terms are evaluated in IGSR's environment, once over
the training and validation rows, and its fits are made in Core. It does not call a language model
either. A PySR arm's selections are searches: each worker runs a search process of its own on one
thread, so their time grows with the rows and with the pairs, and each of those processes holds a
Julia runtime in memory. The panel measures one worker before it starts the others, and runs as
many as fit in the memory it may use (`seqmaestro/reliability/selections.py`). The driver sets
that memory for it, and `--memory-gb` sets it when the panel is run by hand.

## Check before the reveal

The reveal cannot be undone, and after it the run refuses any step that fits or selects, so check
these first:

1. **Every profiled arm has its record.** `failures/` holds no file named `reliability*.json` (a
   call for one arm, as each of the driver's is, names the file `reliability__<model>__<pool>.json`),
   and `run_reveal.py --check` lists nothing missing among the suite's cells, the baseline, the
   reduced models and the panels.
2. **The reproduction checks.** A panel stops, with the failure in the log, if its selection does
   not return the reduction's model from the features the Stage 2 model used. `prepare.json` also
   records whether the selection returns the same model from the whole pool. A different model
   there is a fact about the arm, not a fault: mention it when the stage is reported. A PySR arm's
   searches are not reproducible, so its `prepare.json` records that the checks were not run, and
   why. Mention that when the stage is reported, and confirm instead that its column check passed.
3. **The checks on train and validation look right** on the check arm, which is
   `reliability.check_arm` of the run's `config.yaml`. With the targets permuted, `tier1_null.json`
   counts no stable feature at any threshold. With a planted signal, the planted features are
   stable: `planted.json` names them, and `tier1_planted.npz` holds every candidate's selection
   frequency (`pi_hat`), in the order of `names` in `prepare.npz`. `tier1_planted.json` holds the
   counts alone, which do not say which features were stable. If either check fails, stop and find
   out why before opening the held-out split.
4. **What the panel leaves out, and why**, is printed at its start: the black-box arms, the arms
   left out of the reduction, and those the config leaves out under `not_profiled`. Each needs a
   reason the user would accept.

Then run the reveal. If something cannot be finished, the reveal can go ahead over a partial run:
`run_reveal.py --run-dir <run> --override-incomplete "<reason>"`, run by hand, records the reason.
The driver has no such flag, and never starts the reveal after a stage before it failed. The
report's inputs are then refused for as long as the run's config still lists what is missing, so
the config has to leave it out with a reason first. Ask the user before doing this.

## Check after the reveal

1. **`reveal.json` is complete** (`completed_utc` is set) and `failures/reveal.json` does not
   exist. Otherwise, run the reveal again to finish what failed. The steps after it are complete
   too: `failures/reliability_after_reveal.json` does not exist, and `logs/driver.log` ends with
   "every stage exited clean" and a line naming the report's inputs.
2. **The global null on the held-out split** (`checks.json`, every arm): almost nothing called,
   and p-values near uniform. **The planted signal**: recovered in-sample, and ranked high on the
   held-out split even where a small split cannot call it.
3. **Read the tables for what the report will say**: which features of each reduced model are
   stable, associated and necessary; which are substitutable, and by what; which stable features the
   model left out. A small held-out split limits Tier 2 and Tier 3, and the in-sample Tier 2 shows
   whether a feature lacks a signal or the split lacks the rows to show it.

## Do not

- **Change anything after the reveal.** No arm, no reduction, no setting: the run refuses, and a
  new idea means a new run. Acting on the panel would consume the split it was measured on.
- **Open `test.csv` yourself, or compute anything from it**, outside `run_reveal.py`.
- **Profile a black-box arm, or interpret Tier 1 or Tier 3 as a per-feature significance test.** A
  selection frequency and a leave-one-out comparison describe the model. Tier 2 is the only tier
  with calibrated per-feature p-values.
- **Quote the in-sample Tier 2 as evidence of generalisation.** It is computed on the rows the
  features were chosen on.

## Beyond this file

This file is a floor. If a panel looks wrong in a way nothing above names (a cluster that should
not be one, a stable feature no model uses, a check that passes too well, two arms that disagree
about the same position), look into it, record what you found, and bring what matters to the user.
