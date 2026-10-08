---
name: few-feature-models
stage: 3A
---

# Stage 3A. Reduce each arm to a model of a few features

Each arm of the suite leans on as many features as its model found useful, often hundreds. This
stage cuts every arm down to the N features its own model keeps when it may keep only N, which gives
a model small enough to read in full. That is the model the reliability panel examines and the
report draws. The stage works on the training and validation rows only.

Read [`README.md`](README.md) in this folder first if you have not. This stage begins when the
suite has finished and been checked ([`suite.md`](suite.md)).

## What this stage must leave behind

```
experiments/runs/<date>_<name>_full/reduction/
    <model>/<pool>/reduction.json, steps.npz               one per reduced arm
    <model>/<pool>/{full,reduced,top_n}/                   three standard cells
    summary.csv                                            validation headline of the three, per arm
```

[`../docs/reduction.md`](../docs/reduction.md) says what each file holds and how it is computed.

## Run it

The driver runs it when the config has a `reduction` section, one process per arm, each as soon as
that arm's real-label cells are there. `_shared.yaml` gives every config one, which sets N. Every
arm is reduced but those whose entry in `arms` leaves them out under `not_reduced`, with a reason the
reduction prints. To run it on its own:

```bash
$CORE experiments/runners/run_reduction.py --run-dir experiments/runs/<date>_<name>_full
$CORE experiments/runners/run_reduction.py --run-dir ... --arms l1_logistic/positional   # one arm
```

It is resumable: an arm with a `reduction.json` is skipped unless `--force`. DEFT's arms are reduced
in DEFT's environment, which needs no credentials because nothing in the reduction calls a language
model. A PySR arm's fits are new searches, run in PySR's environment on the threads the
reduction's process is given: usually one for the elimination and one for the top-N control, since
its equation names few features. Run by hand, they take every core, so do not run another stage
beside it.

## Check before moving on

1. **Every arm is reduced.** `failures/` holds no file named `reduction*.json`, and
   `run_reveal.py --run-dir <run> --check` lists nothing missing. A call that reduces one arm, as
   each of the driver's does, writes `failures/reduction__<model>__<pool>.json` when that arm
   fails, and a call for every arm writes `failures/reduction.json`.
2. **The columns are the arm's own.** Every `reduction.json` records a `featurisation_check` with
   no column over tolerance. When a column is over tolerance, that arm's reduction stops without
   writing `reduction.json`, and the failure is in `failures/reduction__<model>__<pool>.json` and
   in the arm's log, `logs/reduction__<model>__<pool>.log`. The other arms are reduced all the
   same.
3. **The trajectories are plausible.** The validation headline at each step (AUROC, or R-squared for
   a continuous target) is in `reduction.json`. The full refit should score near the Stage 2 cell it
   came from. A large gap means either that the
   cell was fitted on a much wider pool than it used or that something was rebuilt wrongly, and you
   must find out which.
4. **The reduced model against the top-N control.** Where the recursion does worse than keeping the
   full model's favourites, say so when the stage is reported. It is a finding about the arm, not a
   fault.
5. **A PySR arm's equations.** Each cell's `run.json` holds its equation under
   `model_settings.fitted`. Its search is not reproducible, so the full refit's equation can
   differ from the Stage 2 cell's even
   on the same columns; compare their validation scores, not their equations. A cell whose
   `metrics.json` counts many `filled_rows` has an equation undefined on part of the validation
   split, which the report must state.

## Do not

- **Choose N, or anything else, per arm on validation.** N is fixed in the config for every arm. The
  trajectory shows where an arm breaks, and is not for picking a better N for it.
- **Reduce the permuted-label cells, or tune the models.** The suite's permuted cells are the
  null, and each reduced model is the arm's own model with fewer columns.
- **Open `test.csv`, or run the reveal.** Stage 3B's pre-reveal work comes first.

## Beyond this file

This file is a floor. If an arm's reduction looks wrong in a way nothing above names (a trajectory
that jumps, a survivor that makes no sense for its arm, two arms that should agree and do not), look
into it, record what you found, and bring what matters to the user.
