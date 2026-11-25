# Stage 1: the prepared dataset

> This page tracks {py:mod}`seqmaestro.dataset` and its command line, [`finalise_dataset.py`](../../experiments/prepare/finalise_dataset.py). Change it in the same commit as either of them when what Stage 1 hands over changes, or when a later stage uses it differently.

Stage 1 turns the data a user delivers into a prediction task: sequences with a target, split into training, validation and held-out rows. Each dataset is built by a script written for its delivery, so Stage 1 ends by writing a record of each dataset in one standard form. Every later [stage](../glossary.md#stage) relies on that record, and so never needs to know how the dataset was built. This page explains the record, the checks every dataset passes, and what the later stages take from it.

A prepared dataset is a directory with three split files (`train.csv`, `val.csv` and `test.csv`) and their record, `dataset.json`. A grouped split also has its group map there (`groups.csv`, [the group map](#the-group-map)). [`finalise_dataset.py`](../../experiments/prepare/finalise_dataset.py) writes `dataset.json` as the last step for every dataset, from the split files and the dataset's [declaration](../glossary.md#declaration). The [command line](../glossary.md#command-line) calls the function behind it, {py:func}`seqmaestro.dataset.finalise`, directly. The dataset's other files (its README, its builder, its declaration, its [task description](../glossary.md#task-description) and its [config](../glossary.md#config)) stay on the machine that holds the data, and [preparing a dataset](../../experiments/prepare/README.md) says where each one is kept.

## Two kinds of fact

`dataset.json` holds two kinds of fact, which come from different places:

- **What was decided**, which only whoever built the dataset knows: which column is the sequence and which the target, what the target means, how the sequences were taken from the delivery, how the split was grouped, and which rows each filter removed. The builder states these in the declaration. They have no defaults, since a default would apply one dataset's answer to the next without anyone noticing.
- **What can be counted**, which `finalise_dataset.py` measures from the split files: the rows and the target in each split, the alphabet and the lengths of the sequences, the digest of each file, the result of every check, and design power. A builder never writes these, so two datasets built by different scripts are described in the same terms and can be compared.

{py:func}`seqmaestro.dataset.finalise` writes the record, and {py:func}`seqmaestro.dataset.load_dataset` reads it back for every later stage.

## The target

The declaration states the target's kind (`target.kind`):

- **Binary**: a label of 0 or 1 on every row, with class 1 the positive class. `classes` gives each label a name and a meaning.
- **Continuous**: one finite number on every row, such as a change in exon inclusion or a binding strength. `target.name` and `target.meaning` say what the number is.

The kind is declared because the values cannot show it: a column of 0s and 1s can be a measured rate as well as a label. Every later stage fits, scores and reports by the declared kind. The arithmetic that differs between the kinds is in {py:mod}`seqmaestro.target`. Transforming or filtering a continuous target (a log, a clip, a threshold on a measurement's count) is a Stage 1 decision, recorded as a preparation step like any filter, so that every [arm](../glossary.md#arm) fits the same target.

## The declaration

The declaration is a YAML or JSON file, `experiments/prepare/declarations/<dataset>__<splits>.yaml`. It is kept apart from `data/`, so that it survives a rebuild of the data. {py:func}`seqmaestro.dataset.validate_declared` lists its fields, with what each holds and whether it is required, and refuses a declaration that lacks a required field.

A required field whose answer is none is still stated, as null or an empty list, because a missing key would be taken for an answer nobody gave. For example, a split that is not grouped states `grouped_by: []`. If the key could be left out, a split whose grouping was only forgotten would pass as ungrouped, and its grouping would never be checked.

The example dataset's declaration, [`example__splits.yaml`](../../experiments/prepare/declarations/example__splits.yaml), is a worked example.

## The checks

`finalise_dataset.py` runs every check on the split files and the declaration together, and records each result in `dataset.json`. {py:func}`seqmaestro.dataset.run_checks` lists the checks and when each one fails or is flagged.

Each check comes out as `passed`, `failed`, `flagged`, `acknowledged` or `not_applicable`. A check that cannot apply is kept apart from one that passed, because a grouping check on a split that declares no grouping has not shown that the grouping holds.

**A check is blocking or advisory:**

- A **blocking** check comes out `failed` when it finds the dataset broken, for example a split whose target does not vary. {py:func}`seqmaestro.dataset.load_dataset` refuses a dataset with a failed check. The declaration can accept a failure under `acknowledged`, with the reason, and the check then records `acknowledged`.
- An **advisory** check comes out `flagged` when it finds a property that only somebody who knows the data can judge. Identical sequences in two splits, for example, are a leak in one dataset and the nature of the measurement in the next. A flagged finding does not stop a [run](../glossary.md#run), and it is printed wherever the dataset is summarised.

Each advisory check puts a question to the user:

- **`no_sequence_in_two_splits`**: is a sequence that appears in two splits a leak, or expected (one site measured in two replicates, one context hit by several events)? The check counts the validation and test rows whose sequence is also a training sequence, and reports the share of each split (by class, for a binary target), because a count of sequences does not say how much of a score is memory. A test row that repeats a validation sequence is counted separately, since a run makes its choices on validation.
- **`sequence_lengths`**: do the lengths go with the target because of the biology, or because of how the rows were collected? A model can take a sequence's length from its k-mer counts and from padded positions, so a length that goes with the target can predict it on its own.
- **`target_distribution`**: are a continuous target's tails the signal, or the noise of the measurement? R-squared weighs each row by its squared distance from the mean, so a few rows in a long tail can decide a model's score.

**Stage 1 settles every finding.** `finalise_dataset.py` prints each flagged finding with its question, and the user's answer goes in the declaration under `acknowledged`. Once the dataset is finalised again, the check records `acknowledged`, with both the finding and the answer, and the report prints them.

**A failed check stops Stage 2**, because the run's config loads the dataset through `load_dataset`, which refuses it. `dataset.json` is written all the same, so that the failure can be inspected and fixed. `finalise_dataset.py` exits with an error while any failure or finding is unanswered, so that neither can be accepted by scrolling past it.

## Covariates

A [covariate](../glossary.md#covariate) is one number per row, measured alongside the sequence, such as a chromatin signal over the window, an experimental measurement or a count of peaks. The split files hold each declared covariate as a column, and the stages take the covariates together with the sequences, in one object ({py:class}`seqmaestro.features.base.Inputs`):

- An arm takes the covariates when its [pool](../glossary.md#pool) names them, as in `positional+covariates` ([run configs](../../experiments/configs/README.md)), and gives them to its model unchanged.
- The [language model](../glossary.md#language-model) of an arm on a [written source](../glossary.md#written-source) is shown the covariates, and may write features that use them.
- The covariates never enter the positional, k-mer or motif features.

**Each covariate's description is written once, in the declaration.** The language models are told it, and a report prints it beside a feature built on the covariate. In a [blinded](../glossary.md#blinding) run it must not reveal what the run withholds, because the language models that write features are told it too.

**A missing value is refused.** `covariates_valid` fails when a declared covariate is absent from a split, has a value that is missing or not a number, is infinite, or is constant on the training rows. A sparse linear fit cannot take a missing value, and a tree would sort one to a side without saying so. Whether and how to impute is therefore a Stage 1 decision, agreed with the user and recorded as a preparation step like any filter. The values `true` and `false` are taken as 1 and 0, and a categorical measurement is encoded as numbers when the dataset is built.

**A column that the declaration does not name** is allowed in the split files, since a builder may keep provenance columns there. `dataset.json` lists it under `columns.unused`, and the summary prints it, so that it cannot go unnoticed.

## Variable lengths and the positional window

A dataset may hold sequences of different lengths (`sequences.length: variable`). The k-mer, motif, composition and written features are computed over the whole sequence. A positional feature, by contrast, needs one frame of reference, since position 37 means nothing across sequences of different lengths. A variable-length dataset therefore declares a positional window when its run has an arm whose pool names `positional`, as every tree arm's pool does:

```yaml
sequences:
  length: variable
  positional_window: {length: 100, anchor: start, pad: false}
```

The window is the `length` bases counted from the `anchor` (`start` or `end`), and a longer sequence is truncated at the far end ({py:class}`seqmaestro.features.positional.PositionalWindow`). What happens to a shorter sequence is the `pad` decision, which the user confirms in Stage 1:

- `pad: true` pads a shorter sequence at the far end, and each padded position becomes a feature of its own (`pos_<i>_pad`). A model can then infer a sequence's length from those features, which the advisory check `sequence_lengths` raises in Stage 1.
- `pad: false` does not allow a sequence shorter than the window. Stage 1 filters the short ones out as a recorded step, so that every arm fits on the same rows, and `positional_window_fits` fails on any that remain.

A fixed-length dataset may declare a window too, to measure positional features over part of the sequence.

## The group map

A grouped split keeps related rows, such as the rows of one gene or of one region, on the same side of the split, because a model that has seen one row has in effect seen its relatives. Checking that a split did so needs a record of each row's group that can be joined to the split files, and the only key they are sure to hold is the sequence. The group map, `<splits>/groups.csv`, therefore has one row per sequence and group ({py:func}`seqmaestro.grouping.group_map`):

- A group is the connected component of the rows joined by any of the declared relations, because a split keyed on one relation alone leaves the others free to straddle it ({py:mod}`seqmaestro.grouping`).
- Only the declared relations count. A split grouped on the gene promises to keep each gene on one side, and nothing more. Two genes that share a window are therefore a fact about the data, which the advisory check `no_sequence_in_two_splits` reports. A sequence that belongs to two groups gets a row for each.

[`write_group_map.py`](../../experiments/prepare/write_group_map.py) writes the map from the dataset file that the splits were drawn from, which holds the grouping columns. It leaves the split files unchanged, so an existing split gains a group map without being redrawn. The check `groups_disjoint` joins the map to the split files, and so checks where each row went.

## Design power

`dataset.json` records two figures under `design_power`, because a design can fail in either of two ways:

- **`test`**: what the [held-out split](../glossary.md#held-out-split) can detect. For a binary target it is the smallest AUROC whose 95% interval excludes 0.5, from the test split's class counts (Hanley and McNeil's interval). For a continuous target it is the smallest correlation between predictions and targets whose 95% interval excludes zero, from the test split's row count (Fisher's interval).
- **`train`**: the effective size of the training split. For a binary target it is the size of the two-class contrast, which combines the sizes of the two classes harmonically, so that 500 rows of one class and 5 of the other count as about 10. For a continuous target it is the row count. A weighted design counts its rows by Kish's effective sample size, which can be far below the row count.

{py:mod}`seqmaestro.harness.power` holds the arithmetic and the assumptions of each interval, and {py:mod}`seqmaestro.target` applies it to a split of each kind of target. A score below the test figure is below what the split can separate from chance, which does not show that the score is at chance. [`design_power.py`](../../experiments/prepare/design_power.py) computes the same figures for a split that is still being planned.

## The held-out split in Stage 1

Stage 1 is the only stage that opens `test.csv` before the [reveal](../glossary.md#reveal). It makes the split, so `finalise` counts the test rows and measures what they can detect. `load_dataset`, which every later stage calls, checks the training and validation files against the digests in `dataset.json`, and leaves the test file closed. The test file's digest is checked at the reveal ([the held-out split and the reveal](heldout.md)).

## The README

`data/<dataset>/README.md` describes the dataset for a person. The [checkpoint](../glossary.md#checkpoint) at the end of Stage 1, where the user approves the dataset, is built from it and from what `finalise_dataset.py` prints. It has these sections, in this order:

1. The source of the data.
2. The target: the two classes and whether both come from one experiment, or what the number measures and how it was measured.
3. The window and its anchor.
4. Each preparation step, with the rows it removed.
5. The grouping, and the reason for it.
6. What the language models are told, and what is withheld from them.
7. The open questions for the people who supplied the data.

**Baselines and the analysis that chose the dataset stay out of the README.** Whoever writes the report reads the README before writing, so such an analysis would tell them what to expect. How far base composition alone separates the classes is measured by the composition [baseline](../glossary.md#baseline) at the start of Stage 2, in the same way for every dataset. A variable that the window leaves out, such as a length that separates the classes, is recorded only as the reason for the decision it led to.

## Uses of `dataset.json`

- **The run's config** ({py:mod}`seqmaestro.pipeline.config`) takes the columns and the target's kind from it, so that a config names only the split and the task description.
- **Every stage of a run** loads the dataset through `load_dataset`, from the split that the run's recorded config names, and takes the columns, the covariates with their descriptions, the positional window and the target's kind from it. A stage therefore does not need a column's name as a command-line option.
- **The report** takes the target with its classes or its meaning, the counts, the filters, the checks with every finding, and design power ({py:func}`seqmaestro.report.facts.stage1`).
