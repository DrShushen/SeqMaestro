# Stage 2: the suite

> This page tracks {py:mod}`seqmaestro.pipeline.stage2`, {py:mod}`seqmaestro.harness.runner`, {py:mod}`seqmaestro.harness.cell` and the Core side of each vendored method ({py:mod}`seqmaestro.pipeline.vendored`). Change it in the same commit as any of them.

Stage 2 measures how well each [arm](../glossary.md#arm) of a run predicts the target and which features each arm uses. It fits every arm that the run's [config](../glossary.md#config) lists, under both [label conditions](../glossary.md#label-condition) and at every [seed](../glossary.md#seed), and scores each fit on the validation rows. One such fit is a [cell](../glossary.md#cell). Every later stage starts from the cells:

- The [Stage 2 analyses](analyses/index.md) compare the cells' features across seeds and arms.
- The [reduction](../glossary.md#reduction) and the [reliability panel](../glossary.md#reliability-panel) refit each arm from its cells.
- The [reveal](../glossary.md#reveal) scores the model each cell saved on the [held-out split](../glossary.md#held-out-split), which Stage 2 never opens.

The [driver](../glossary.md#driver) starts [`run_arm.py`](../../experiments/runners/run_arm.py) once per cell, many at a time. The script is given the run's directory and takes every setting from the config recorded there, so a cell cannot be fitted under a setting the run does not record. The docstring of {py:mod}`seqmaestro.pipeline.stage2` shows how to fit a cell by hand. Which model can be fitted on which [pool](../glossary.md#pool) is declared in {py:mod}`seqmaestro.arms`.

Most models are fitted in Core, the `SeqMaestro_Core` environment where SeqMaestro's own code runs. DEFT and IGSR are vendored methods: their code is kept under `methods/` and runs in conda environments of their own ([environments](../../ENVIRONMENTS.md)), so a cell of theirs is fitted across the boundary between environments ([DEFT and IGSR](#deft-and-igsr), below).

## Comparable cells

The cells of every model are fitted and scored by the same code, so that any two cells of a run can be compared. {py:func}`~seqmaestro.harness.runner.run_model` does everything except the model's own fit. Both ways into a cell end in {py:func}`~seqmaestro.harness.runner.score_arm`: `run_model` for a model fitted in Core, and {py:func}`~seqmaestro.harness.runner.ingest_bundle` for what a vendored method hands back. The analyses therefore treat every cell alike and cannot tell which way it came. The grid's settings that this code takes have no defaults, so a run states them once, in its config (`grid`, commented in [`_shared.yaml`](../../experiments/configs/_shared.yaml)).

These rules make two cells comparable:

- **Each seed fits on its own draw of the training rows.** A seed draws `grid.train_fraction` of the training rows (its *fitting rows*), stratified on the target (the classes of a binary target, or ten quantile bins of a continuous one), so that every seed fits on the target's whole range. Most of the models give the same fit on the same rows, so without the draw every seed would return one result, and [stability](analyses/stability_and_co_discovery.md) across seeds would be 1 by construction. Every arm and the [baseline](../glossary.md#baseline) fit on the same rows at a seed, and a vendored method is handed the drawn rows as a file, so that the draw is made in one place ({py:func}`~seqmaestro.harness.runner.subsample_training_rows`).
- **The featuriser is part of the fit.** A featuriser computes a pool's features from the rows ({py:class}`~seqmaestro.features.base.Featuriser`). It is fitted on the fitting rows, with the labels the seed fits on, and then applied to the validation rows and to every training row. A vocabulary learned from other rows would leak those rows into the fit, and a [proposer](../glossary.md#proposer) shown the real labels would hand a permuted cell features written from the real signal.
- **A permuted cell is the control for its arm.** A `real` cell is fitted on the labels as they are, and a `permuted` cell on the training labels shuffled. A number counts as a finding only if it exceeds what the same arm scores on shuffled labels. Every cell is scored against the real validation labels, so a permuted cell should score at chance, and one that scores above chance shows a leak. An arm that also chooses something on the validation rows (the [iterative source](../glossary.md#iterative-source) and IGSR's search) chooses against shuffled validation labels in a permuted cell ({py:func}`~seqmaestro.harness.runner.labels_to_choose_on`), since otherwise its control would see the real signal. Per-row weights describe how the rows were sampled, so a permuted cell keeps them.
- **The number of independent shuffles** is set by `grid.permutation_scope`. Under `seed`, every arm at a seed is given the same shuffled labels, so the control's level rests on as many draws as there are seeds, and a draw that happens to score above chance does so in every arm at once. Under `cell`, each arm draws its own shuffle at each seed, so the level can be averaged over arms and quoted with an interval ({py:func}`~seqmaestro.harness.runner.permutation_rng`).
- **Every cell is scored on the validation rows**, by {py:func}`~seqmaestro.harness.runner.score_validation`, with the metrics of the target's kind ({py:mod}`seqmaestro.target`). For a binary target, a model's score is the probability of the positive class, AUROC ranks the arms, and the metrics that need a threshold (such as accuracy) use the one that gives the best balanced accuracy on validation. For a continuous target, R-squared ranks the arms. The metrics that comparisons between arms rest on come with a percentile interval from `grid.n_bootstrap` resamples of the validation rows, drawn with the cell's seed ({py:mod}`seqmaestro.harness.metrics`). The reveal scores the held-out rows with the same threshold, seed and number of resamples, so a held-out block is what scoring at fit time would have written ([the held-out split and the reveal](heldout.md)).
- **The features a cell reports** are those its model used, each with the model's importance, the direction the model uses it in, and its association with the real target over every training row ([the feature record](feature_record.md)). A `grid.permutation_repeats` above 0 adds an importance measured in the same way for every arm, by shuffling one column at a time on the validation rows.

## A cell's files

{py:mod}`seqmaestro.harness.cell` lists the files of a cell and is the only module that writes them. A cell's directory is `results/<model>/<pool>/<labels>/seed_<n>/` in the run ({py:mod}`seqmaestro.pipeline.layout`). Other code relies on two of the files:

- **`metrics.json` is written last.** A cell that has one is finished, so a resumed run skips it and the reveal scores it. A process killed partway through a cell leaves none, and the cell is fitted again.
- **`run.json` records `predict_with`**, which names the side that wrote the cell and can run its saved model: Core, or the adapters of DEFT or IGSR. Every later stage finds the code for a cell's model through it ({py:mod}`seqmaestro.pipeline.sides`).

## Kinds of arm

How a cell is fitted depends on the kind of its arm ({py:func}`seqmaestro.arms.kind`), and {py:func}`~seqmaestro.pipeline.stage2.arm_of` chooses the class that fits each kind:

- **Fixed sources**: a Core model on a pool whose columns follow from the rows by rule (positions, k-mers, known and discovered motifs, covariates), fitted in the stage's process ({py:class}`~seqmaestro.pipeline.stage2.FixedArm`).
- **Written sources**: a Core model on a pool that includes `llm_generated` or `llm_generated_iterative`, fitted in the stage's process once a language model has written the pool's features ({py:class}`~seqmaestro.pipeline.stage2.ProposerArm`).
- **DEFT**: `deft_or_cart`, fitted under DEFT's interpreter. On a pool without `llm_generated_joint` it is plain CART, and on a pool with it, DEFT's generative finder has a language model write further candidates at each node of the tree ({py:mod}`seqmaestro.pipeline.vendored.deft`).
- **IGSR**: `igsr`, on `llm_generated_joint`, fitted under IGSR's interpreter ({py:mod}`seqmaestro.pipeline.vendored.igsr`).

Two parts of a Core arm's fit also run outside Core. A PySR model's search runs in `SeqMaestro_Method_pysr`, and its equation is then evaluated in Core. Motif discovery, by [STREME](https://meme-suite.org/meme/doc/streme.html) (with [Tomtom](https://meme-suite.org/meme/doc/tomtom.html) where the run compares the motifs found with a collection), runs in `SeqMaestro_Method_meme`.

### Core models

The models fitted in Core ({py:mod}`seqmaestro.models`) are the sparse linear model (L1-penalised, from [scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html)), the [EBM](https://interpret.ml/docs/ebm.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [PySR](https://github.com/MilesCranmer/PySR). Each model's class lists the few settings a run may change, such as XGBoost's number of trees, and fixes the rest, so that an arm's model is the same in every run that leaves those few alone. A run changes them under `models.<name>` ([run configs](../../experiments/configs/README.md#source-and-model-settings) says how), and states the class weighting of a binary target once for every model (`grid.class_weight`). Each cell records its model's settings in its `run.json` (`model_settings`), and the reduction and the reliability panel rebuild the model from them.

### The proposer

The proposer ({py:mod}`seqmaestro.proposer`) writes the features of the one-round and the iterative sources. It asks a language model for features as short Python functions of the sequence, shows it labelled example rows and any [guidance](../glossary.md#guidance) the run gives, and keeps the functions that run on the rows. `sources.llm_generated` sets how many requests a proposal makes, how many features each asks for and how many example rows each shows.

- **Each function's code is checked before it runs.** The check is a guardrail against accidents and casual misuse, and {py:mod}`seqmaestro.proposer.guard` says what it stops.
- **A function that fails on the rows goes back to the model** with its error, up to `repair_rounds` times, and is dropped if it still fails. `grid.generated_features` states this policy once for every arm whose language model writes features, and [`_shared.yaml`](../../experiments/configs/_shared.yaml) lists the ways a function can fail.
- **Each cell has a proposal of its own**, written from the rows and labels the cell fits on. It is kept under `proposals/`, keyed on everything that would change it (the language model and its reasoning effort among them). A cell fitted again therefore makes no request twice, and two models compared on a written source are compared on separate draws of it.

### The iterative source

The iterative source ({py:mod}`seqmaestro.proposer.iterative`) starts from the one-round source's proposal and then makes `feedback_rounds` more proposals. Each of them is shown what a fit of the cell's model made of the features proposed so far: which features it relied on, which it could not use, and how well it scored on the validation rows. A round in which every feature fails adds nothing, and the cell goes on.

### DEFT and IGSR

[DEFT](https://arxiv.org/abs/2604.12060) and [IGSR](https://arxiv.org/abs/2605.29184) run under their own interpreters and never import this package. For each cell, the Core side ({py:class}`~seqmaestro.pipeline.vendored.handover.VendoredArm`) does three things:

1. It draws the cell's fitting rows, as for any arm, and writes them, the validation rows and every training row as files.
2. It starts the method's *adapter*, a script that fits the method under the method's interpreter and writes the result as a *bundle*, a directory of files.
3. It scores the bundle through the same `score_arm` as every other cell.

[The adapters](../../experiments/adapters/README.md) describe the bundle and the rules every adapter follows.

Each method starts from its own configuration, and a run overrides it in two layers ({py:func}`~seqmaestro.pipeline.vendored.handover.config_overrides`):

1. What the grid states for every arm, translated into the method's settings: the class weighting of a binary target, and the policy for written features.
2. The config's `models.<name>.config_overrides`, passed to the method as they are.

A key set in both layers is refused, since two places that set one value can come to disagree. So is a key that names the language model or how it is called, because the environment and the config's `language_model` state those once for the whole run. An arm whose method writes features is started with that language model and those settings, and is passed the run's guidance, which its adapter adds to the prompts it builds from the [task description](../glossary.md#task-description). [Run configs](../../experiments/configs/README.md) describes each of these settings.

A DEFT cell also passes its seed to DEFT as `random_state` ({py:func}`~seqmaestro.pipeline.vendored.deft.cell_overrides`), so that a new seed changes both the rows and the tree's random draws, as it does for every other arm.

## Other steps of the stage

Stage 2 runs these steps besides the arms:

- **The composition baseline** ([the composition baseline](baseline.md)) is fitted first, at every seed, on the rows the arms fit on. Its cells are under `baseline/`, outside `results/`, so that no analysis counts it as an arm.
- **The motif step**, where a pool includes a motif source, scans the training and validation sequences once with the known motifs, and makes each discovery of motifs once per label condition and seed. The arms on such a pool start after the part of the step they need, so that a cell does not repeat a scan or a discovery that another cell has made ([motifs in a run](motifs.md#motifs-in-a-run)). Under `permutation_scope: cell`, a control's shuffled labels depend on its arm, so each control cell makes its own discovery as it is fitted ([discovered motifs](motifs.md#discovered-motifs)).
- **The analyses** run once every arm has finished, and use every cell that finished: [stability and co-discovery](analyses/stability_and_co_discovery.md), [recurrence](analyses/recurrence.md), and [the judge](analyses/judge.md) where the config sets `grid.judge`. The recurrence analysis groups the features of the [written sources](../glossary.md#written-source), and both it and the judge call a language model.

A run therefore needs the language model's credentials only where an arm's pool includes a written source or the judge runs.

## Calls and resuming

Each process of a run is a *call*. The driver starts one call per cell and names it after the cell, as `model__pool__labels__seed_<n>`. A call's log, its record of the command, commit and time, and its record of failures share its name. {py:mod}`seqmaestro.pipeline.layout` lists where each is kept. A call started by hand is named after everything it fits, and prints to the terminal, since the driver is what keeps a call's log.

A cell that raises an error is recorded in its call's failures file, and the call goes on with its other cells. When the stage runs again, a finished cell is skipped, and `--force` fits it again. A revealed run is refused, even with `--force`.

A DEFT cell needs more care, because a second fit grows a different tree, and on a permuted cell a choice between two trees would choose which draw of the control is kept. An unfinished DEFT cell is therefore treated by what its bundle holds:

- **No tree**, because the fit ended before its tree was saved: the cell is fitted again like any other.
- **A tree, with the rest of the bundle incomplete**: the cell is refused. `run_arm.py --run-dir <run> --arms <model>/<pool> --labels <labels> --seeds <n> --reuse-tree` completes it from that tree, and only `--force` grows a new one.
