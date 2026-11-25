# The held-out split and the reveal

> This page tracks the following code. Change it in the same commit as any of it.
>
> - {py:mod}`seqmaestro.harness.heldout` and {py:mod}`seqmaestro.pipeline.reveal`.
> - Prediction with a cell's saved model on each side (Core, DEFT and IGSR): {py:mod}`seqmaestro.pipeline.sides`, the `predict_cell` of each module of {py:mod}`seqmaestro.pipeline.vendored`, the predict modes of [`deft_bundle.py`](../../experiments/adapters/deft_bundle.py) and [`deft_cart.py`](../../experiments/adapters/deft_cart.py), and the `predict` and `evaluate` modes of [`igsr_bundle.py`](../../experiments/adapters/igsr_bundle.py).
> - The scoring of a cell on validation alone, in {py:mod}`seqmaestro.harness.runner` and {py:mod}`seqmaestro.harness.bundle`.
> - `reveal_tier2` in {py:mod}`seqmaestro.reliability.panel`.

A [run](../glossary.md#run) makes many choices: which features each [arm](../glossary.md#arm) keeps, what the analyses group together, and which few features survive the [reduction](../glossary.md#reduction). Every one of them is made on the training and validation rows. The [held-out split](../glossary.md#held-out-split), which the code calls the test split, is opened once, at the end of the run, in the [reveal](../glossary.md#reveal): every model the run saved predicts the test rows, and the predictions are scored. The stages before the reveal never see a test number, so a test number cannot have steered any of their choices.

The test split could instead be scored as each [cell](../glossary.md#cell) is fitted, with the numbers hidden from the later stages. That would keep the numbers out of sight by discipline alone, whereas here they do not exist until the reveal.

```{mermaid}
flowchart LR
  TV["Training and<br/>validation rows"] --> S["Stages 2 to 3B:<br/>fit, choose,<br/>save every model"]
  S --> R["The reveal"]
  TE["Test rows,<br/>closed until now"] --> R
  R --> N["Test scores in<br/>every saved cell"]
  N --> A["Tier 3 of the<br/>reliability panel,<br/>and the report"]
```

## Access to the test split

**Only {py:mod}`seqmaestro.harness.heldout` opens `test.csv`, and only the reveal calls it.** The reveal is {py:mod}`seqmaestro.pipeline.reveal`, which [`run_reveal.py`](../../experiments/runners/run_reveal.py) starts. Stage 1 makes the split and is exempt: {py:mod}`seqmaestro.dataset` measures every split it describes, and {py:mod}`seqmaestro.cli.prepare` makes the split for the [command line](../glossary.md#command-line).

`tests/test_heldout.py` checks the rule in the source. It fails if a module in `seqmaestro/` or `experiments/{runners,analysis,adapters}/` names the test split, or imports `SPLITS`, the names of all three splits. Code that uses test results after the reveal takes the split's names from `heldout` (`TEST_SPLIT`, `TEST_ARRAYS`) and checks `reveal_record` first, so every dependency on the test split shows up as an import of that one module.

**The rule guards against accidents in code.** `test.csv` is in `data/`, where anyone can open it. The rule keeps code from opening it by accident, and so keeps a test number from deciding something that the run was meant to decide without one.

## Cells before the reveal

The reveal scores a cell from what the cell recorded when it was fitted. Before the reveal, every cell therefore holds:

- Its validation block in `metrics.json`, which holds a binary target's threshold (chosen on validation), and its validation predictions.
- The `seed`, `n_bootstrap` and `target_kind` that its test block will be scored with, in `run.json`.
- Its fitted model. A Core cell saves it as `fitted.pkl`. A cell of DEFT or IGSR names its model's file in the `fitted_model` of its `run.json`, as {py:mod}`seqmaestro.pipeline.sides` describes. For a Stage 2 cell, the file is in the bundle the method handed back ([the adapters](../../experiments/adapters/README.md)), and for a cell fitted later, such as a reduced model, it is in the cell itself.

**The fitted model is required**, since a cell that saved none could never be scored on the test split. {py:func}`seqmaestro.harness.runner.run_model` therefore saves the model before it writes anything else, and {py:func}`seqmaestro.harness.bundle.read_bundle` refuses a bundle without one.

## The reveal

The [driver](../glossary.md#driver) runs the reveal as a stage, once every stage before it has finished cleanly. `--stop-before-reveal` holds it back, so that a run can be checked before its test split is opened. `run_reveal.py --check` reports how many items are ready and lists the rest, without sealing or opening anything, and exits with status 2 when something is not ready.

{py:mod}`seqmaestro.pipeline.reveal` lists the steps of the reveal. The design decisions behind them are these:

- **`reveal.json` is written before any test row is read** ({py:func}`seqmaestro.harness.heldout.begin_reveal`). From then on, every stage that fits or selects refuses to run in the run's directory. The record lists the items the reveal seals: every cell that the run's config accounts for, and the tier 2 of each arm that the [reliability panel](../glossary.md#reliability-panel) profiles. A resumed reveal scores those and nothing added since. Anything in the run that its config does not account for is recorded and left unscored.
- **A partial reveal needs a reason.** The reveal refuses a run whose items are not all ready, unless `--override-incomplete REASON` is given. It then records the reason and what was missing, because a reveal over a partial run qualifies every number it produces.
- **The test file's digest is checked when the file is opened**, against the digest that `dataset.json` recorded. The digest is checked only at this point, since the file stays closed until then.
- **A vendored method's cell is predicted on its side of the [environment boundary](architecture.md#two-boundaries)**, by the adapter that its `run.json` names in `predict_with`. The test rows go across numbered and without their labels, and the row numbers that come back with the scores are checked before anything is scored.
- **A model that can be undefined on a row**, such as a [PySR](https://github.com/MilesCranmer/PySR) equation that divides by a feature, gives such a row its constant prediction. The cell's `metrics.json` counts these rows under `filled_rows`: on validation when the cell is fitted, and on test at the reveal.
- **Tier 2 of the reliability panel is computed on the test split.** Tier 2 measures each feature's association with the target, and it is the only test-side work that does not score a model. On a dataset fitted with per-row weights it computes nothing, since the panel has no tier 2 for a weighted design ([the reliability panel](reliability.md)).

**The reveal refits nothing, and it can be resumed.** When it is run again after an interruption, it completes what was sealed and is not yet done. Afterwards, `run_reliability.py --after-reveal` assembles the panel's tier 3 (whether the model needs each feature) and its tables from what the reveal wrote.

## Equivalence with scoring at fit time

A cell's test block needs four things, and each is the same at the reveal as it would have been at fit time:

- **The test targets** come from the same file.
- **The model's scores** come from the same fitted objects, restored from the files the cell saved.
- **A binary target's threshold** was chosen on validation and stored in the validation block, and it survives the round trip through JSON exactly.
- **The bootstrap resamples** are drawn with the cell's own seed and number of resamples, by the same function.

The test block that {py:func}`seqmaestro.harness.heldout.score_cell` writes is therefore identical to the block that scoring at fit time would have written. `tests/test_heldout.py` checks this for each Core model of a binary target.

## Stages after the reveal

Once `reveal.json` exists, every stage that fits or selects refuses to run in that directory ({py:func}`seqmaestro.harness.heldout.refuse_if_revealed`), and so does the driver's preflight. Adding an arm to a revealed run therefore means starting a new run. `tests/test_heldout.py` checks that the module behind each stage's `run_*.py` script calls `refuse_if_revealed`.

The reliability panel has work on both sides of the reveal. It calls `refuse_if_revealed` before the reveal, whereas its `--after-reveal` mode fits and selects nothing, takes only what the reveal wrote, and refuses the opposite case: a run that has not been revealed.

## Test numbers after the reveal

After the reveal, a cell's test block is in its `metrics.json` under the key `test` (`TEST_SPLIT`), and {py:func}`seqmaestro.harness.heldout.revealed_metrics` returns it, refusing before the reveal. The spreadsheet of a run's features ([`export_features_xlsx.py`](../../experiments/analysis/export_features_xlsx.py)) reports the headline metric of the run's kind of target (AUROC or R-squared) on test once the reveal has finished, and on validation before, with the split named in every header.
