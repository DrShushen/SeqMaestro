# Stage 2. The suite: every arm fitted, on real and shuffled labels, at several seeds

> **Maintained page.** Change it in the same commit as `seqmaestro/pipeline/stage2.py`,
> `seqmaestro/harness/runner.py`, `seqmaestro/harness/cell.py` or the Core side of a vendored
> method under `seqmaestro/pipeline/vendored/`.

Stage 2 fits every arm the run's config lists and writes each fit as a **cell**. Nothing in it
reads the test split. Every later stage starts from these cells: the analyses compare them, the
reduction and the panel refit from them, and the reveal scores the models they saved.

An **arm** is one model fitted on one pool of feature sources, spelled `model/pool`
(`seqmaestro/arms.py` says which model takes which pool). A **cell** is one arm, under one condition
of the labels, at one seed:

```
results/<model>/<pool>/<labels>/seed_<n>/
```

`run_arm.py` fits the cells, one process per arm, label condition and seed when the driver runs it.
It takes the run directory and reads every setting from the `config.yaml` the driver recorded
there, so a cell cannot be fitted under a setting the run does not record.

## What makes two cells comparable

Every cell of a run is fitted and scored the same way, whatever its model. `run_model` and
`score_arm` (`seqmaestro/harness/runner.py`) hold all of it, and a model supplies only its fit.

**The rows a seed fits on.** Each seed draws `grid.train_fraction` of the training rows, in the
target's proportions: the classes of a binary target, and ten quantile bins of a continuous one, so
that every seed fits on the target's whole range. Most models here give the same fit on the same
rows, so without the draw every seed would return one result and stability would be 1 by
construction. Every arm and the baseline fit on the same rows at a seed, and a vendored method is
handed the drawn rows as a file, so the draw is made in one place.

**The features are computed from the fitting rows.** A featuriser is part of the fit: a vocabulary
learned from rows the model never saw would be a leak. It is fitted on the seed's rows, with the
labels that seed trains against, and then applied to the validation rows and to every training row.

**The two label conditions.** A `real` cell is fitted on the labels as they are. A `permuted` cell
is fitted on the training labels shuffled, and is the control its arm is read against: a number
counts as a finding only if it exceeds what the same arm returns on shuffled labels. The validation
labels a cell is scored against are never shuffled, so a permuted cell should score at chance, and
one that does not shows a leak. An arm that also *chooses* on the validation rows (the iterative
proposer, IGSR's search) chooses against the validation labels shuffled, or its control would carry
real signal. Per-row weights are a property of the design and are not drawn again under the shuffle.

**How many independent shuffles a run has** is `grid.permutation_scope`. Under `seed`, every arm at
a seed is given the same shuffled labels, so the level of the null rests on as many draws as there
are seeds, and a draw that lands above chance lands there in every arm at once. Under `cell`, each
arm draws its own, one per cell, so the level can be averaged and quoted with an interval.

**Scoring.** A cell is scored on the validation rows. For a binary target the score is the
probability of the positive class, a threshold is chosen on validation to give the best balanced
accuracy, and the block holds AUROC (the headline), AUPRC and its lift over the positive rate,
accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, the Brier score, the log
loss and a calibration error. For a continuous target it holds R-squared (the headline), Spearman's
rho, Pearson's r, RMSE and MAE. All but the lift, the precision, recall and specificity, the log
loss and the calibration error come with a percentile interval from `grid.n_bootstrap` resamples of
the validation rows, drawn with the cell's seed. The reveal later scores the test rows with the
same threshold, seed and count, so a test block is what scoring at fit time would have written
(`heldout.md`).

**The features a cell reports** are the ones its model used, each with the model's importance, the
direction the model uses it in, and its association with the real target over every training row
(`feature_record.md`). `grid.permutation_repeats` above 0 adds an importance measured the same way
for every arm, by shuffling one column at a time on the validation rows.

## A cell's files

| file | holds |
|---|---|
| `run.json` | what was fitted: `model`, `pool` and the pool's `sources`, the seed, the label condition, the settings the fit used, and `predict_with`. A Core cell records its settings as `model_settings` and `featuriser`. A vendored cell records what its method was started with: its `config_overrides`, with `feature_finder` for DEFT and the `fit` its search made for IGSR |
| `metrics.json` | the validation block, and after the reveal the test block. **Written last: a cell that has it is finished** |
| `predictions.npz` | the validation labels and scores, and after the reveal the test ones |
| `features.json`, `feature_values.npz` | the features the model used, and their values on the validation rows |
| `fitted.pkl` | a Core cell's fitted featuriser and model, which the reveal predicts with |
| `inputs/`, `bundle/` | a vendored cell's handed rows, and what its method handed back, its saved model included |
| `token_usage.json` | a proposer cell's record of its written features. `usage` is what its requests cost in tokens. `proposal_stats` counts what the language model returned and what the guard and the repair round made of it, over every proposal of the cell. `fit_report` counts the written features proposed, already held under the same name, kept, and dropped on the rows for each reason. `rounds` holds the same counts for each round of the iterative source |

`seqmaestro/harness/cell.py` names the first five and writes them, each whole: a file is written
under another name and renamed into place. A process killed in the middle of a cell therefore leaves
no `metrics.json`, and the driver fits the cell again when it resumes.

`predict_with` says where the cell's model lives: `fitted` for a Core model, `deft_bundle` or
`igsr_bundle` for a vendored one. The later stages find the code that can use the model from it
(`seqmaestro/pipeline/sides.py`).

## How each kind of arm is fitted

| kind | arms | where the fit runs |
|---|---|---|
| fixed | a Core model on a pool whose columns follow from the rows by rule: positions, k-mers, covariates | in the stage's process. A PySR model's search runs in a process of `SeqMaestro_Method_pysr` that the stage starts, and its equation is evaluated in Core |
| proposer | a Core model on a pool holding `llm_generated` or `llm_generated_iterative` | the same, after a language model has written the pool's features |
| DEFT | `deft_or_cart`: plain CART on a pool without `llm_generated_joint`, and DEFT's generative finder on one with it | under DEFT's interpreter |
| IGSR | `igsr`, on `llm_generated_joint` | under IGSR's interpreter |

**The Core models** (`seqmaestro/models/`) are the sparse linear model (`l1_logistic` or
`l1_linear`), the EBM, XGBoost and PySR. Their settings are fixed in their classes, and each cell
records them in `model_settings`. A run states only what all of them share: the class weighting of
a binary target (`grid.class_weight`).

**The proposer** (`seqmaestro/proposer/`) asks a language model for features as short Python
functions of the sequence, shows it labelled example rows, and keeps the ones that run.
`sources.llm_generated` sets how many requests it makes and how many features each asks for. Each
written function is checked before it is run (`seqmaestro/proposer/guard.py`: no imports beyond an
allowed list, and no `open`, `eval` or attribute tricks), run under a time limit, and dropped if it
fails on the rows or is undefined on too many of them. The check is a guardrail against accidents
and is no security boundary. A failed function goes back to the model `repair_rounds` times with
the error. `grid.generated_features` states that policy once for every arm that writes. A proposal
is made for one arm at one seed, from the rows and labels its cell fits on, and is kept under
`proposals/`, so a cell fitted again asks nothing twice, and two models compared on a written source
are compared on separate draws.

**The iterative source** makes the one-round source's proposal first, then `feedback_rounds` more
proposals, each of as many requests, and each shown what a fit of the cell's model made of the
features proposed so far: what it leaned on, what it could not use, and how well it scored on the
validation rows. A later proposal's feature under a name the cell already holds is left out, and a
round whose features all fail on the rows adds nothing and the cell goes on.

**DEFT and IGSR** run under their own interpreters and never import this package. For each cell the
Core side writes the fitting rows, the validation rows and every training row as files, starts the
method's adapter, and scores the bundle that comes back through the same `score_arm`
(`seqmaestro/pipeline/vendored/`, and `experiments/adapters/README.md` for the bundle). The
settings a run starts a method with are the method's defaults, overridden first by what the grid
states for it (the class weighting, the policy for written features) and then by
`models.<name>.config_overrides`. A key in both is refused.

## Beside the arms

**The composition baseline** (`baseline.md`) is fitted first, at every seed, on the rows the arms
fit on. Its cells are under `baseline/`, outside `results/`, so no analysis counts it as an arm.

**The analyses** run once every arm has finished, and take the cells as they find them:
`analyses/stability_and_co_discovery.md`, `analyses/recurrence.md`, and `analyses/judge.md` where
the config asks for the judge. The last two ask a language model. A run needs its credentials
where an arm's pool holds a written source or the judge runs, and no others do.

## What a call leaves, and resuming

A call is one process of a run. The driver starts one per cell and names it for the cell:
`model__pool__labels__seed_<n>`. A call made by hand is named for everything it fits, and leaves
no log, since the driver is what keeps one.

| | |
|---|---|
| `logs/<call>.log` | what a call the driver started printed, with each cell's validation numbers as it finished |
| `invocations/<call>.json` | the command, the commit it ran at and the time |
| `failures/<call>.json` | each cell of the call that raised, with its traceback. A call that ends clean removes the file |

A cell that raises is recorded and the others go on. A finished cell is skipped when the stage runs
again, and `--force` fits it again. A revealed run is refused, with `--force` too.

A DEFT cell needs more care, because a second fit draws another tree, and on a permuted cell a
choice between two trees would choose which draw of the null is kept. A cell whose fit ended before
its tree was saved is fitted again like any other. A cell whose tree was grown and whose bundle is
incomplete is refused, and is completed from that tree:
`run_arm.py --run-dir <run> --arms <model>/<pool> --labels <labels> --seeds <n> --reuse-tree`. Only
`--force` grows a new tree for it.
