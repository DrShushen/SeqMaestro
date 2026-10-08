# Adapters: arms that run in another environment

A method with its own dependency set lives in its own conda environment and cannot import
`seqmaestro`. An adapter is the far side of that boundary. It runs under the method's interpreter
and hands the result back as a **bundle**, a directory of files.

Neither side imports the other, so the two environments can disagree about every package they
contain. `tests/test_boundaries.py` enforces this on the SeqMaestro side.

## Why adapters live here and not in `methods/<m>/`

The bundle format belongs to SeqMaestro. A vendored method stands alone: nothing in
`methods/deft/` may know about SeqMaestro's paths, splits or schemas, so an adapter written there
would be the dependency this arrangement exists to prevent. The adapter lives on the SeqMaestro
side and is executed by the method's interpreter.

## The bundle

```
<bundle>/
    arm.json             bundle_version (3) and fitted_model, which are required, then whatever
                         the method wants to carry (its model's name, the settings it ran with,
                         its environment, wall clock, token usage)
    predictions.npz      val_true, val_score
    features.json        {schema_version, importance_kind, direction_kind, features: [...]}
    feature_values.npz   names, values: the features evaluated on the reference split
    effect_values.npz    names, values, y: the same features over the training rows with
                         their real labels, which is where `marginal_effect` is measured
    <fitted_model>       the model the adapter fitted, in the method's own format
```

`seqmaestro.harness.bundle` holds the definition, and its `describe_contract()` returns it as data,
so an adapter can be checked against the code instead of against this page. `read_bundle` validates
a bundle on ingest, so a malformed bundle fails there instead of producing a plausible number.

**No test split.** An adapter is fitted and scored on validation only, and a bundle carrying test
predictions is refused. The test split is scored once, at the reveal
([`docs/heldout.md`](../../docs/heldout.md)). For a bundle cell, the reveal asks the adapter's
**predict mode** to load the model that `fitted_model` names and predict the test rows. The saved
model is therefore mandatory, because a bundle without one could never be scored.

**Scores are probabilities of the positive class** for a binary target, in [0, 1], and predicted
values for a continuous one. Apply the link function in the adapter. The harness refuses a binary
arm's scores outside [0, 1], because Brier score and calibration are meaningless on an unbounded
decision function. `read_bundle` is told the dataset's kind of target and reads the labels and
scores by it.

**Feature records** use the shared schema: `name`, `description`, `kind`, `params`, `source`,
`rationale`, `origin`, `importance`, `permutation_importance`, `direction`, `marginal_effect`,
`provenance`, `extra` ([`docs/feature_record.md`](../../docs/feature_record.md)). They are plain
JSON, which an adapter writes with `json.dump` because it cannot import `save_features`. Only
`name`, `description` and `kind` are required, so an adapter with nothing to say about a field
omits it instead of inventing a value.

**The adapter writes `direction`, and never `marginal_effect`.** Which way a model uses a feature
can only come from how that model works (a coefficient, a shape function, a split), so the adapter
takes it from the fitted model and names the basis in `direction_kind`. How far a feature goes
with the target on its own needs no model, only the feature's values and their labels, which the
bundle already carries in `effect_values.npz`. `score_arm` computes it on the SeqMaestro side for
every arm, and an adapter that wrote its own would be a second definition of the same number. See
`seqmaestro.harness.direction`, whose vocabulary an adapter restates and `read_bundle` checks.

**The adapter measures `permutation_importance` itself.** The harness computes it for a model it
fitted, by predicting again with one column shuffled, and it cannot do that for a model that
lives in another environment. So an adapter is handed `--permutation-repeats` and
`--permutation-seed`, and where the count is above 0 it writes, for each feature, the fall in the
headline score on the validation rows (AUROC, or R-squared for a continuous target) when that
feature's values are shuffled, averaged over that many shuffles. A bundle without the number is
accepted, and its features are then left out of any ranking by it.

**Features are mandatory.** `"features": []` is a valid result meaning the method selected nothing.
A bundle that omits the key comes from a broken adapter, and the two cases must stay
distinguishable.

## The four rules that matter

**Row order is the contract.** Return predictions in the order the rows were given. The bundle
carries the labels the adapter saw, so ingest can check the order. A reordered bundle scores
plausibly and means nothing, and the metrics alone would never reveal it. The predict mode is handed
no labels, so the order is checked another way: every row carries an identifier, and the scores come
back beside the identifiers for the caller to check.

**Train on the rows you are given, not on a seed.** The method's Core side
(`seqmaestro/pipeline/vendored/`) writes the rows a seed fits on to disk and hands over the file.
Drawing them again from a seed on the far side would implement the draw twice, and the two
implementations would eventually drift apart without anything failing.

**Choose against the labels handed for choosing.** `--val` carries the validation rows' real
labels, because the cell is scored against them. A method that chooses anything on those rows (the
iteration a search keeps, the terms it prunes, the error its prompts show) is handed the labels to
choose against as well (`choose_on.csv`, from `handover.write_inputs`), and chooses against those
alone. On a control arm they are shuffled as the fitting rows' are, so the
control's choices never read the real target and it stays a null. A method that fits on the
fitting rows alone, as DEFT does, is handed none.

**Duplicate the contract constants, do not import them.** An adapter declares
`BUNDLE_VERSION` and `FEATURE_SCHEMA_VERSION` itself. `read_bundle` rejects a mismatch, and a test
asserts the two copies agree, so the duplication is checked.

## Writing a new one

1. A `fit` mode taking `--train --val --effect --out-bundle`, with `--target-column`,
   `--target-kind`, `--permutation-repeats` and `--permutation-seed`, plus whatever the method
   needs. `--effect` holds every training row with its real label.
2. Fit on `--train`. A method that chooses on the validation rows chooses against the labels
   handed for choosing (above). Predict on `--val` in row order, and save the fitted model inside
   the bundle.
3. Extract the features the method used. Evaluate them on `--val` (`feature_values.npz`) and on
   `--effect` (`effect_values.npz`, which a bundle must hold), and measure each one's permutation
   importance on `--val`.
4. Write the bundle, with `fitted_model` in `arm.json` naming the saved model.
5. A `predict` mode taking `--bundle --rows --id-column --out`: load the model, predict every row
   of `--rows` in order, and write `row_id` and `score` to `--out`. The rows have no label column.
6. Add the method's Core side under `seqmaestro/pipeline/vendored/`: a `VendoredArm` that
   writes a cell's inputs and runs the adapter as a subprocess under the method's interpreter
   (`handover.py` then scores the bundle with `ingest_bundle`), and a function that predicts a
   cell through the predict mode, as `deft.py`'s `predict_cell` does. Then give the method a side
   in `seqmaestro/pipeline/sides.py`, found by the `predict_with` value its cells record in
   `run.json`. In `seqmaestro/arms.py`, give its model an entry in `MODEL_SPECS` (its environment,
   that it is vendored, the sources it accepts), without which a config cannot name it, and a
   kind, which `seqmaestro/pipeline/stage2.py` builds its arm from.

Logic that walks the method's own structures should be tested. It can usually be tested from
`SeqMaestro_Core` with stand-in objects, as long as the adapter does not import the method at
module scope. `tests/test_deft_adapter.py` does this, and among other things checks that a leaf
node is not counted as a selected feature.

## Dataset descriptions stay on this side, and there is only one of them

A method whose prompts are templated on the task has to be told what the task is, and which
dataset is being run is SeqMaestro's decision. A method that ships a description of one dataset
has to be edited to run another. A wrong description also usually fails silently: the model is told
confidently that it is looking at something it is not, and the run completes.

**The adapter does not hold the description either.** It lives in
[`../datasets/<name>.json`](../datasets/README.md), one file per dataset, which the proposer, the
judge and the recurrence analysis also read. The facts are the same for every consumer, and a
dataset described in several places would eventually be described differently in each. The adapter
only translates the description into the shape its method wants, at run time.

For DEFT that means generating a `params_generation` config and putting it on hydra's search path
with `--config-dir`. See [`deft_conf/README.md`](deft_conf/README.md) for the resolution order and
for the case that needs a hand-written config.

**A new adapter for a method with configurable prompts should do the same**: take `--task`, read
the shared description, translate it, and fail early with a clear message when the name is not
found. Do not add a second place that describes the data.

## What a vendored arm is handed about its columns

`seqmaestro/pipeline/vendored/handover.py`, which DEFT's arms and IGSR's share, writes
`columns.json` beside each cell's inputs: the sequence column with the dataset's positional window,
and every other column the method is given, with its recipe (`kind`, `params`) and its description.
A covariate is handed as the column it is, with the dataset's own description of what it measures
(kind `covariate`). A pool member computed on this side is handed with the recipe and description
of the feature that computed it, and its featuriser is saved beside it; DEFT's pools are the only
ones with such members, since IGSR is handed the sequence whole. Every later stage (the reveal,
the reduction, the reliability panel) builds the rows it hands the method from that file
(`handover.input_frame`), so each sees the columns the model was fitted on and no others.

`deft_bundle.py` reads the same file on the far side. Where there is anything to tell, it writes a
DEFT dataset config beside the prompt config (`dataset/handed.yaml`, inheriting `generic_csv`) with
the descriptions (`column_descriptions`) and the window (`positional_window`), so DEFT's prompts
show what each column means and its positional candidates are read in the window by the rule every
other arm uses. The config is written into the bundle, so the frozen run records what DEFT was
told. A column the tree splits on comes back under the recipe and description the manifest
recorded, so a covariate DEFT selects is the same feature as the one a SeqMaestro arm selects.

`igsr_bundle.py` reads it too: each covariate's description, and the dataset's sequence layout for
the sequence column, go into IGSR's prompt beside the columns they describe, and a term that reads
a handed column unchanged comes back under that column's recipe, as DEFT's split on a covariate
does.

## Present adapters

| | Runs | Notes |
|---|---|---|
| `deft_bundle.py` | `SeqMaestro_Method_deft` | Stage 2. `fit`: both the CART finder (`--feature-finder cart`, no credentials) and the LLM finder (credentials required), growing a classification or a regression tree by `--target-kind`; `--task` names the dataset, whose prompts come from a hand-written config in `deft_conf/params_generation/` when one exists and otherwise from its description (see above). `predict`: the saved tree, for the reveal; no credentials |
| `igsr_bundle.py` | `SeqMaestro_Method_igsr` | `fit`: IGSR's search in process, its terms written as Python functions of the whole sequence and the handed covariates (`--target-kind continuous` only; credentials required), choosing on the validation rows against `--choose-on`'s labels and scored against `--val`'s; `--task` names the dataset, whose description becomes IGSR's data dictionary and each column's description. `predict`: the saved least-squares model over the final terms, for the reveal. `evaluate`: any terms the search accepted, on rows handed over, for every later stage; the fits on them are made on the SeqMaestro side (`seqmaestro/pipeline/vendored/igsr_terms.py`), with the least-squares fit the search made, which `fit` records in `arm.json`. No credentials outside `fit` |
| `deft_cart.py` | `SeqMaestro_Method_deft` | Everything after Stage 2. `reduce`: a DEFT arm's own features, evaluated by the tree its cell saved, eliminated to N with DEFT's CART (Stage 3A). `pool`: the arm's candidate pool on the rows given, which is DEFT's naive population for the CART finder and the recorded features for the generative one. `reproduce`, `select`, `refit`: the reliability panel's fits, on that pool (Stage 3B). `predict`: a reduced or refitted tree, for the reveal. `tree`: a reduced tree's structure, read from the saved tree with the fitting rows routed down it, for a report to draw. `time`: how long the tree takes to grow on samples of the rows a cell would be handed, for the driver's estimate before a run. No credentials: no language model is called |

**Code shared across the boundary is loaded by path, never imported.** `deft_cart.py` loads three
files directly and records their digests, so it runs the same code as the SeqMaestro arms: the
elimination loop (`seqmaestro/reduction/eliminate.py`), the column check
(`seqmaestro/reduction/check.py`) and the resumable half-sample sweep
(`seqmaestro/reliability/selections.py`). All three import only numpy and the standard library,
which `tests/test_reduction.py` enforces, so DEFT's environment needs no SeqMaestro code beyond
those files. Whatever else `deft_cart.py` needs from Stage 2, such as evaluating a tree's features
and computing its node statistics, it loads from `deft_bundle.py` the same way, so each has one
implementation.

**The filename matters.** An adapter must not be named `deft.py`. A script's own directory goes
first on `sys.path`, so a module named `deft` would shadow the library it wraps, and unpickling the
tree against `deft.trees` would fail with "`deft` is not a package".
