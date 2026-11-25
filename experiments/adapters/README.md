# The adapters

An adapter is a script that runs a vendored method, [DEFT](https://arxiv.org/abs/2604.12060) or [IGSR](https://arxiv.org/abs/2605.29184), for an [arm](../../docs/glossary.md#arm) of SeqMaestro, and hands the result back. The repository keeps a copy of each vendored method's code under `methods/`, and the method runs in a conda environment of its own ([ENVIRONMENTS.md](../../ENVIRONMENTS.md)), whose packages can disagree with SeqMaestro's. The method therefore cannot import `seqmaestro`, and SeqMaestro does not import the method. The two sides exchange files instead:

1. SeqMaestro's side ([`seqmaestro/pipeline/vendored/`](../../seqmaestro/pipeline/vendored/__init__.py)) writes the rows of a [cell](../../docs/glossary.md#cell) as files, and starts the adapter under the method's interpreter.
2. The adapter fits the method on those rows and writes the result as a **bundle**, a directory of files.
3. SeqMaestro's side checks the bundle and scores it with the same code as every other cell.

[`tests/test_boundaries.py`](../../tests/test_boundaries.py) checks that `seqmaestro` imports no vendored method, and that no vendored method imports `seqmaestro`.

## Adapters in this folder

- [`deft_bundle.py`](deft_bundle.py) fits a DEFT arm's [Stage 2](../../docs/developer/suite.md) cell, with either of DEFT's feature finders, and predicts the held-out rows with the cell's saved tree at the [reveal](../../docs/glossary.md#reveal).
- [`deft_cart.py`](deft_cart.py) does what the stages after Stage 2 need from a DEFT arm: the [reduction](../../docs/glossary.md#reduction), the fits of the [reliability panel](../../docs/glossary.md#reliability-panel), the reveal's predictions of the reduced and refitted trees, and a reduced tree's structure for the report. It also times a fit for the driver's estimate before a run.
- [`igsr_bundle.py`](igsr_bundle.py) fits an IGSR arm's Stage 2 cell, predicts the held-out rows with the cell's saved model at the reveal, and evaluates the terms IGSR's search accepted for every later stage.

The DEFT adapters run in `SeqMaestro_Method_deft`, and the IGSR adapter in `SeqMaestro_Method_igsr`. Each script's docstring lists its modes and their arguments. Only a fit whose method writes features calls a language model, and so needs the provider's credentials: `deft_bundle.py fit` with DEFT's generative finder, and `igsr_bundle.py fit`.

## Adapters beside the pipeline

The bundle's format belongs to SeqMaestro, whereas a vendored method stands alone: nothing in `methods/deft/` may know about SeqMaestro's paths, splits or schemas. An adapter written there would be the dependency that this arrangement exists to prevent. The adapters therefore live on SeqMaestro's side, in this folder, and the method's interpreter runs them.

## The bundle

The docstring of [`seqmaestro/harness/bundle.py`](../../seqmaestro/harness/bundle.py) gives the bundle's layout. That module holds the definition, and its `describe_contract()` returns the contract as data, so that an adapter can be checked against the code. `read_bundle` checks each bundle as SeqMaestro's side takes it back, so a malformed bundle fails there instead of producing a plausible number.

Every bundle follows these rules:

- **A bundle holds the scores of the validation rows only.** An adapter's fit is never handed the [held-out split](../../docs/glossary.md#held-out-split), and a bundle that holds test predictions is refused. The held-out split is scored once, at the reveal ([the held-out split and the reveal](../../docs/developer/heldout.md)), when the adapter's `predict` mode loads the model that `fitted_model` names in `arm.json` and predicts the held-out rows. The saved model is therefore required, because a bundle without one could never be scored.
- **The scores are probabilities or predicted values.** For a binary target, the scores are probabilities of the positive class, in [0, 1], so the adapter applies the link function. `read_bundle` refuses a binary arm's scores outside [0, 1], because the Brier score and calibration mean nothing on an unbounded decision function. For a continuous target, the scores are the predicted values. `read_bundle` is told the dataset's kind of target, and checks the labels and scores by it.
- **The feature records use the shared schema** ([`FeatureRecord`](../../seqmaestro/features/base.py), whose docstring describes each field, and [the feature record](../../docs/developer/feature_record.md)). They are plain JSON, which an adapter writes with `json.dump`, since it cannot import `save_features`. Only `name`, `description` and `kind` are required, so an adapter with nothing to say about a field leaves it out instead of inventing a value.
- **`direction` comes from the adapter, and `marginal_effect` from SeqMaestro.** Which way a model uses a feature can only come from how that model works (a coefficient, a shape function, a split), so the adapter takes it from the fitted model and names its basis in `direction_kind`. The words are those of [`seqmaestro/harness/direction.py`](../../seqmaestro/harness/direction.py), which the adapter restates and `read_bundle` checks. How far a feature goes with the target on its own needs no model, only the feature's values and their labels, which the bundle holds in `effect_values.npz`. SeqMaestro's side computes it for every arm (`score_arm`), and an adapter that wrote its own would be a second definition of the same number.
- **The adapter measures `permutation_importance` itself.** SeqMaestro measures it for a model fitted in its own environment by predicting again with one column shuffled, which it cannot do for a model in another environment. An adapter is therefore handed `--permutation-repeats` and `--permutation-seed`. Where the count is above 0, it writes for each feature the fall in the headline score on the validation rows (AUROC, or R-squared for a continuous target) when that feature's values are shuffled, averaged over that many shuffles. A bundle without the number is accepted, and its features are then left out of any ranking by it.
- **The `features` key is required.** `"features": []` is a valid result, meaning that the method selected nothing, whereas a bundle without the key comes from a broken adapter, and the two cases must stay distinguishable.

## Rules for every adapter

- **Row order is the contract.** An adapter returns predictions in the order the rows were given. A reordered bundle scores plausibly and means nothing, and the metrics alone would never show it. The bundle therefore holds the labels the adapter saw, so that SeqMaestro's side can check the order. The `predict` mode is handed no labels, so there every row has an identifier, and the scores come back beside the identifiers for the caller to check.
- **An adapter fits on the rows it is handed.** SeqMaestro's side writes the rows a [seed](../../docs/glossary.md#seed) fits on to a file and hands over the file. Drawing them again from the seed on the method's side would implement the draw twice, and the two copies would come to disagree without anything failing.
- **An adapter chooses against the labels handed for choosing.** `--val` holds the validation rows' real labels, because the cell is scored against them. A method that chooses anything on those rows (the iteration a search keeps, the terms it prunes, the error its prompts show) is also handed the labels to choose against, in `choose_on.csv` ([`handover.py`](../../seqmaestro/pipeline/vendored/handover.py)), and uses those alone. In a `permuted` cell, which is the control of its arm ([label condition](../../docs/glossary.md#label-condition)), they are shuffled as the labels it fits on are, so the control's choices never depend on the real target. A method that chooses nothing on the validation rows, such as DEFT, is handed none.
- **An adapter states the contract's constants itself.** It declares `BUNDLE_VERSION` and `FEATURE_SCHEMA_VERSION` instead of importing them. `read_bundle` rejects a mismatch, and a test checks that the two copies agree.
- **Code shared across the boundary is loaded by path, never imported.** `deft_cart.py` loads three of SeqMaestro's files directly and records their digests, so that it runs the same code as SeqMaestro's own arms: the elimination loop ([`eliminate.py`](../../seqmaestro/reduction/eliminate.py)), the column check ([`check.py`](../../seqmaestro/reduction/check.py)) and the resumable half-sample sweep ([`selections.py`](../../seqmaestro/reliability/selections.py)). All three import only numpy and the standard library, which [`tests/test_reduction.py`](../../tests/test_reduction.py) checks, so DEFT's environment needs no other SeqMaestro code. Whatever else `deft_cart.py` needs from Stage 2, such as evaluating a tree's features, it loads from `deft_bundle.py` in the same way, so that each has one implementation.
- **An adapter is not named after the library it wraps.** A script's directory comes first on `sys.path`, so an adapter named `deft.py` would shadow the `deft` library, and loading the saved tree against `deft.trees` would fail with "`deft` is not a package".

## Task descriptions

A method whose prompts are templated on the task has to be told what the task is. Which dataset is run is SeqMaestro's decision, so a method that shipped a description of one dataset would have to be edited to run another. A wrong description also usually fails silently: the language model is told confidently that it is looking at something it is not, and the run completes.

The adapter does not hold the description either. Each dataset's [task description](../../docs/glossary.md#task-description) is one file, [`../datasets/<name>.json`](../datasets/README.md), which the proposer, the judge and the recurrence analysis also load. A dataset described in several places would come to be described differently in each, so each adapter translates the one description into the form its method takes, at run time:

- **DEFT**: `deft_bundle.py` generates a `params_generation` config from the description and puts it on [Hydra](https://hydra.cc/)'s search path with `--config-dir`. [`deft_conf/README.md`](deft_conf/README.md) gives the order in which a task's prompts are found, and the case that needs a hand-written config.
- **IGSR**: `igsr_bundle.py` turns the description into IGSR's data dictionary and each column's description.

A new adapter for a method with configurable prompts does the same: it takes `--task`, loads the shared description, translates it, and fails early with a clear message when the name is not found.

## The columns a cell is handed

SeqMaestro's side writes `columns.json` beside each cell's inputs ([`handover.py`](../../seqmaestro/pipeline/vendored/handover.py) gives its shape). It records the sequence column, with the dataset's positional window, and every other column the method is given, with its recipe and description. Such a column is either a covariate, handed as it is, or a column that SeqMaestro computed, such as a k-mer count. Only DEFT's pools have computed columns, since IGSR is handed the whole sequence and the covariates. The file serves two purposes:

- **The method is told what each column means.** `deft_bundle.py` writes the descriptions and the window into a DEFT dataset config inside the bundle, so that DEFT's prompts show what each column means, its positional candidates are taken in the window by the rule every other arm uses, and the frozen run records what DEFT was told. `igsr_bundle.py` puts each covariate's description and the sequence layout into IGSR's prompt.
- **A column the method selects comes back as a known feature.** A covariate that DEFT's tree splits on, or that an IGSR term returns unchanged, comes back under the recipe and description the file recorded, so it is the same feature as the one a SeqMaestro arm would select.

Every later stage (the reveal, the reduction, the reliability panel) builds the rows it hands the method from the same file (`input_frame`, in `handover.py`), so that each sees the columns the model was fitted on, and only those.

## Adding an adapter

The adapter needs two modes:

1. A `fit` mode that takes `--train`, `--val`, `--effect` and `--out-bundle`, with `--target-column`, `--target-kind`, `--permutation-repeats` and `--permutation-seed`, and whatever else the method needs. `--effect` holds every training row with its real label. The mode fits on `--train` (choosing against the labels handed for choosing, if the method chooses on the validation rows), predicts `--val` in row order, and saves the fitted model inside the bundle. It evaluates the features the method used on `--val` (`feature_values.npz`) and on `--effect` (`effect_values.npz`), measures each feature's permutation importance on `--val`, and writes the bundle, with `fitted_model` in `arm.json` naming the saved model.
2. A `predict` mode that takes `--bundle`, `--rows`, `--id-column` and `--out`. It loads the model, predicts every row of `--rows` in order, and writes `row_id` and `score` to `--out`. The rows have no label column.

SeqMaestro's side needs three additions:

1. A module under [`seqmaestro/pipeline/vendored/`](../../seqmaestro/pipeline/vendored/__init__.py) with a subclass of `VendoredArm` ([`handover.py`](../../seqmaestro/pipeline/vendored/handover.py)), which writes a cell's inputs and runs the adapter under the method's interpreter, after which `VendoredArm` scores the bundle with `ingest_bundle`. The module also needs a function that predicts a cell through the `predict` mode, as `predict_cell` in [`deft.py`](../../seqmaestro/pipeline/vendored/deft.py) does.
2. A side for the method in [`seqmaestro/pipeline/sides.py`](../../seqmaestro/pipeline/sides.py), found by the `predict_with` value its cells record in `run.json`.
3. An entry for its model in `MODEL_SPECS`, in [`seqmaestro/arms.py`](../../seqmaestro/arms.py) (its environment, that it is vendored, and the sources it accepts), without which a config cannot name it. The model also needs a kind there (`kind`), and a branch in `arm_of`, in [`stage2.py`](../../seqmaestro/pipeline/stage2.py), that builds its arm.

The logic that walks the method's own structures, such as a fitted tree, is tested from `SeqMaestro_Core` with stand-in objects. This works as long as the adapter imports the method inside its modes, and never at module scope. [`tests/test_deft_adapter.py`](../../tests/test_deft_adapter.py) and [`tests/test_igsr_adapter.py`](../../tests/test_igsr_adapter.py) do this.
