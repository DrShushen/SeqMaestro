# What Stage 1 hands over: the prepared dataset

> **Maintained page.** Change it in the same commit as `seqmaestro/dataset.py` or
> `experiments/prepare/finalise_dataset.py`: a field added, a check added or changed, or a change
> to what a later stage reads.

Stage 1 turns a delivery into a prediction task. However a dataset was built, it ends as three split
files and one description of them, `dataset.json`, in a shape every later stage can rely on:

```
data/<dataset>/
    README.md                  what the dataset is, for a person; see "The README" below
    raw/                       the delivery, as received
    <splits>/
        train.csv, val.csv, test.csv
        groups.csv             which group each sequence belongs to, where the split is grouped
        dataset.json           written by finalise_dataset.py, read by everything after
experiments/prepare/declarations/<dataset>__<splits>.yaml    what was decided
experiments/datasets/<task>.json                              what the language models are told
experiments/configs/<name>.yaml                               the run config
```

All of these belong to one dataset and stay on the machine that holds it: git ignores the last
three as it ignores `data/`. The repository holds one example dataset whole, with the script that
builds it ([`../experiments/prepare/README.md`](../experiments/prepare/README.md)).

## Two kinds of fact, from two places

`dataset.json` holds what was **decided** and what can be **counted**, and the two come from
different places.

**Decided**, and stated by whoever built the dataset in a declaration: which column is the sequence
and which the target, what kind of target it is and what it means, which columns are covariates and
what each measures, the window and its anchor, the positional window where lengths vary, the
orientation and how it was checked, how the split was grouped and stratified, and every filter with
what it cost. Only the builder knows these, so nothing about them is defaulted.

**Counted**, by `finalise_dataset.py` from the files themselves: rows and groups per split, with
each split's classes or the distribution of its target, the alphabet and the lengths, each
covariate's spread on the training rows, the columns in the split files that nothing declares
(`columns.unused`), file digests, the checks, and design power. No builder is trusted to write
these, so two datasets built by different scripts are described in one shape and can be compared.

## The target

A dataset's target is one of two kinds, and the declaration states which (`target.kind`):

- **binary**: a label of 0 or 1 on every row, with class 1 the positive class. `classes` says what
  each label means.
- **continuous**: one finite number on every row, such as a change in exon inclusion or a binding
  strength. `target.name` gives a short name for it and `target.meaning` says what it measures, in
  which units, and what a higher value means.

The kind is never inferred from the values, since a column of 0s and 1s can be a measured rate as
well as a label. Every later stage fits, scores and reports by the declared kind
([`seqmaestro/target.py`](../seqmaestro/target.py) holds the arithmetic of each). Transforming or
filtering a continuous target (a log, a clip, a threshold on a measurement's count) is a Stage 1
decision, recorded as a preparation step like any filter, so that every arm fits the same target.

The declaration is kept under `experiments/prepare/declarations/`, beside the builder and apart
from `data/`. A dataset rebuilt from its builder and finalised from its declaration gets an
identical description.

## The declaration

YAML or JSON. Every field below is required unless marked optional. A field whose answer is none
is stated all the same, as null or an empty list: a key left out is refused, since its absence
would be taken for an answer nobody gave.

| field | what it holds |
|---|---|
| `columns.sequence`, `columns.target` | the split files' sequence and target columns |
| `columns.weight` | the per-row fitting weight column, or null. Weights are for fitting only; every metric is unweighted |
| `columns.covariates` | optional: the covariate columns, numbers measured beside each sequence (below), in the order the models and the language models are given them. Each name must be a legal Python identifier |
| `covariates.<name>.description` | required for each declared covariate: what it measures, in the words a domain expert would use |
| `target.kind` | `binary` or `continuous` |
| `target.name`, `target.meaning` | for a continuous target only: a short noun phrase a report uses in a sentence, and what the number measures, in which units, and what a higher value means |
| `classes.1`, `classes.0` | for a binary target only: for each label, a short `name` and a one-line `meaning`. A report prints these beside every direction, so they must say which way round the label runs |
| `sequences.window` | what each sequence is, in words: its width and what it covers |
| `sequences.anchor` | the landmark the window is cut from, so a position means the same kind of site in every row |
| `sequences.orientation` | the strand or reading direction, and how it was verified |
| `sequences.length` | the fixed length every sequence has, or `variable` |
| `sequences.positional_window` | optional: `{length, anchor, pad}`, the fixed frame positional features are read in (below). Needed for a variable-length dataset whose run has a positional or tree arm |
| `split.method` | how the rows were divided, in words |
| `split.grouped_by` | what makes two rows dependent, as column names; empty for an ungrouped split. A split that declares one needs a group map beside it (below) |
| `split.stratified_on` | what the split balances, as column names; empty for a split that balances nothing |
| `split.seed` | the seed the split used, or null |
| `filters` | optional but expected: each step that could remove rows, in order, with `step`, `why`, and `rows_in` and `rows_out`, per label for a binary target and one count for a continuous one. A step that removed nothing is recorded too |
| `sources`, `builders` | optional: the delivered files and their digests; the scripts that built the dataset |
| `acknowledged` | optional: a check name and the answer to what it found. For an advisory finding the answer explains it. For a failure it gives the reason the failure is accepted |
| `notes` | optional: anything a reader of the dataset needs that has no field |

The example dataset's declaration, `declarations/example__splits.yaml`, is a complete worked
example.

## Covariates

A covariate is one number per sample, measured beside the sequence: a chromatin signal over the
window, an experimental measurement, a count of peaks. The split files carry each declared covariate
as a column, and every stage reads the covariates together with the sequences, through one object
(`Inputs` in [`seqmaestro/features/base.py`](../seqmaestro/features/base.py)). An arm whose pool
holds the covariates hands them to its model as they are, and a proposer arm's language model is
shown them and may write features that use them. They never enter the positional or k-mer features.
A run config says, arm by arm, whether an arm takes them
([`../experiments/configs/README.md`](../experiments/configs/README.md)).

**The description is written once, here.** It is what the language models are told about the
column, and what a report prints beside a feature built on it. On a blinded dataset it is written
blind, because every model consumer reads it.

**A missing value is refused, never filled.** `covariates_valid` fails when a declared covariate is
absent from a split, holds a value that is missing or not a number, is infinite, or is constant on
the training rows. Whether and how to impute is a Stage 1 decision, agreed with the user and
recorded as a preparation step like any other filter. `true` and `false` are read as 1 and 0. A
categorical measurement is encoded as numbers when the dataset is built.

A column in a split file that nothing declares is not an error, since a builder may keep provenance
columns there. It is listed under `columns.unused` and printed, so it cannot sit in the files
unnoticed.

## Variable lengths and the positional window

A dataset may hold sequences of different lengths (`sequences.length: variable`). The k-mer,
composition and language-model features read the whole sequence. Positional features need one frame
of reference, so a variable-length dataset whose run has a positional or tree arm declares one:

```yaml
sequences:
  length: variable
  positional_window: {length: 100, anchor: start, pad: false}
```

The window is `length` bases from the `anchor`: the first bases with `start`, the last with `end`.
A longer sequence is cut at the far end, which costs no row. What happens to a shorter one is the
`pad` decision, which the user confirms at Stage 1:

- `pad: true` pads a shorter sequence at the far end, and a padded position is a feature of its own,
  named in every table. A model can read a sequence's length from those positions, which the
  advisory check `sequence_lengths` puts to Stage 1.
- `pad: false` allows no sequence shorter than the window. Stage 1 filters the short ones out as a
  recorded step, so every arm reads the same rows, and `positional_window_fits` fails on any that
  remain.

A fixed-length dataset may declare a window too, to read positional features over part of it.

## The group map

A grouped split promises that rows which are relatives of each other never land on opposite sides of
it. Checking that promise needs a record of which group each row is in, and that record has to be
joinable to the split files, which carry only the sequence and the label. So `<splits>/groups.csv`
is keyed on the sequence: `sequence,group`, where `group` is the connected component over all the
declared relations together ([`seqmaestro.grouping`](../seqmaestro/grouping.py)).
`experiments/prepare/write_group_map.py` writes it from the dataset CSV the splits were cut from,
which is where the grouping columns live. The split files are only read, so an existing split gains
a group map without being redrawn.

The map uses the declared relations and nothing else. Adding the sequence as a relation would hold
a split to a promise it never made: a split grouped on the gene did group on the gene. If two genes
share a window, that is a fact about the data, and the advisory check `no_sequence_in_two_splits`
reports it. A sequence belonging to two groups therefore gets a row for each.

## The checks

Each comes out `passed`, `failed`, `flagged`, `acknowledged` or `not_applicable`. `not_applicable` is
distinct from `passed`: a grouping check with nothing declared has not shown the grouping holds.

**Blocking or advisory.** A blocking check finds the dataset broken, and `load_dataset` refuses it.
An advisory check finds a property that only somebody who knows the data can judge. Identical
sequences in two splits, for example, are a leak on one dataset and the nature of the measurement on
the next. So an advisory check reports `flagged` with what it measured, does not stop a run, and is
printed wherever the dataset is summarised. Stage 1 is where a flag is settled: `finalise_dataset.py`
prints it with the question it raises, and the answer goes in the declaration under `acknowledged`.
The check then reads `acknowledged` and carries both the answer and what it answered.

| check | fails when |
|---|---|
| `readme_present` | `data/<dataset>/README.md` is missing |
| `target_valid` | a binary target holds anything but 0 or 1, or a continuous one holds anything but a finite number (an empty field, a word, NaN, infinity) |
| `target_varies_in_every_split` | a split holds only one class of a binary target, or one value of a continuous one |
| `target_distribution` | **advisory**, on a continuous target. Reports each split's spread, quantiles, skewness and tails, and is flagged when more than 1% of the training rows lie beyond three robust standard deviations of the median (a normal distribution puts 0.3% there), or when one repeated value covers more than 10% of them. R-squared weighs a row by its squared distance from the mean, so a few rows in a long tail can decide a model's score, and whether the tails are the signal or the noise of the measurement is for Stage 1 to say |
| `alphabet_seen_in_training` | validation or test uses a symbol the training rows never contain. A featuriser fitted on training has no column for it |
| `lengths_as_declared` | a sequence is not the declared fixed length |
| `sequence_lengths` | **advisory**, on a variable-length dataset. Lengths vary: reports their quartiles by split and class for a binary target, or by split with the rank correlation of length with the target for a continuous one, and, with a positional window, the share of rows it pads (or, without padding, would refuse) and cuts short, because a length that goes with the target is something a model can use |
| `positional_window_fits` | a sequence is shorter than a positional window declared without padding |
| `covariates_valid` | a declared covariate is absent from a split, missing or not a number, infinite, or constant on the training rows |
| `no_sequence_in_two_splits` | **advisory.** A validation or test sequence is also a training sequence, so a model has seen a held-out row verbatim, or a test sequence is also a validation sequence, so the choices a run makes on validation have seen it (counted apart, as `test_in_val`). Reports the share of each held-out split affected, and for a binary target the share by class, because a count of sequences does not say how much of a score is memory. Where the dataset has covariates, it also counts the repeated rows whose covariates differ from every training copy, which are different samples |
| `groups_disjoint` | two *different* sequences of one declared group sit in different splits, or the group map is missing or does not cover the splits. A sequence sitting in two splits is left to the check above, and one belonging to two groups is set aside: the split files name no row, so neither can say which group's row went where |
| `weights_valid` | a weight is missing, negative or non-finite, or the training rows carry no weight (for a binary target, the training rows of either class) |
| `filters_account_for_rows` | the last declared filter leaves a different number of rows (per class, for a binary target) than the splits hold, so a step that changed the row count is missing from the record |

**A failed check stops Stage 2.** The run config loads the dataset through `load_dataset`, which
refuses one with a failed check that was not acknowledged. The file is still written, so the failure
can be read and fixed. An advisory finding does not stop Stage 2 whether or not it has been
acknowledged, but `finalise_dataset.py` exits non-zero while either kind is unanswered, so neither is
accepted by scrolling past.

## Design power

`design_power` holds two numbers, because a design can fail on either:

- **`test`**: for a binary target, the smallest AUROC whose 95% interval clears 0.5 on the held-out
  split, and the interval half-widths at a few AUROCs, from the test split's class counts alone
  (Hanley and McNeil's closed form). Null where the split is too small for any interval, below two
  rows of a class. For a continuous target, the smallest correlation between predictions and
  targets whose 95% interval clears zero, its square, and the half-widths at a few correlations,
  from the test split's row count alone (Fisher's z-transform). The arithmetic is in
  `seqmaestro.harness.power`.
- **`train`**: the effective size of the training split: for a binary target the two-class
  contrast, which for a weighted design is Kish's figure per class combined harmonically, and for a
  continuous target Kish's figure over all rows. Rows with zero weight are left out, as the split
  intends.

A result below the test floor has not been shown to be at chance, only to be below what the split
can separate from chance. `experiments/prepare/design_power.py` computes the same numbers for a
split being planned rather than one already built.

## Reading the held-out split

Stage 1 is the only stage that reads `test.csv` before the reveal. It makes the split, so
`finalise` counts the test rows and measures what they can detect. `load_dataset`, which every later stage
calls, re-checks the digests of the training and validation files and leaves the test file alone.
The test file's digest is checked at the reveal ([`heldout.md`](heldout.md)).

## The README

`data/<dataset>/README.md` is for a person, and the checkpoint that closes Stage 1 is built from it
and from `finalise_dataset.py`'s summary. It covers, in this order: the source; the target (the two
classes and whether both come from one experiment, or what the number measures and how); the window
and its anchor; each preparation step with what it cost; the grouping and why; what the language
models are told and what is withheld; open questions for the people who supplied the data.

It carries **no baselines and no analysis that chose the dataset**. How far base composition alone
separates the classes is measured by the composition baseline at the start of Stage 2, the same way
for every dataset. A variable the window keeps out, such as a length that separates the classes, is
recorded as the reason for the decision it drove, not as a floor to read results against.

## What reads it

| reader | reads |
|---|---|
| `seqmaestro/pipeline/config.py` | the columns and the target's kind, so a run config names only the split and the description |
| every stage of a run | the columns, the covariates and their descriptions, the positional window and the target's kind, through `load_dataset` on the split the run's recorded config names: none takes a column as a flag |
| the report | the target with its classes or its meaning, the counts, the filters, the checks and design power |
