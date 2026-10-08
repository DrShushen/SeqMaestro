# What we record about every feature

> **Maintained page.** If you change `FeatureRecord`, what fills a field, or what reads one, change
> this page in the same commit. It is the only place the whole picture is written down, and the
> docstrings each say one part of it.

Every arm in this system reports the features it used, and they all report them in **one shape**. A
*k*-mer count, a positional indicator, a decision-tree split and a function a language model wrote
are all a `FeatureRecord`. So the stability estimator, the significance judge, the spreadsheet and
the report treat every arm identically, and none of them needs to know which arm produced a row.

The type is in `seqmaestro/features/base.py`. Reading and writing it is in
`seqmaestro/harness/features.py`.

---

## 1. The two files per cell

A finished results cell (one model, one pool, one label condition, one seed) holds the five
files `score_arm` writes, plus `fitted.pkl` for an arm fitted in this process. Two of the five are
the feature record:

```
results/<model>/<pool>/<labels>/seed_<n>/
    features.json        the records, plus a run-level header
    feature_values.npz   each of those features evaluated on the reference split
```

`features.json` is what the feature *is*. `feature_values.npz` is what it *does* on a common set of
rows, stored as a `float32` matrix with one column per feature in record order, so the two files
are paired **by position**. `read_bundle` and `load_run` both refuse a cell whose value names
disagree with the records in content or in order, because a mismatch would silently compare the
wrong features.

**Why the values are stored and not recomputed.** A generated feature is code a language model
wrote. Re-executing it months later would need the code guard, the exact environment and
reproducible generation, and it fails outright for arms whose features are not executable code at
all. Storing the values reduces every downstream comparison to loading arrays and correlating them,
works identically for every arm, and means no model-written code is ever run outside the run that
produced it.

**The reference split is validation.** Matching features uses no labels, and validation is already
the split doing model selection, so evaluating there exposes nothing new. The behavioural matcher
and permutation importance are both measured on those rows, so they see the same data.

**`marginal_effect` is the exception, and it is measured on the training rows.** It asks what a
feature does in the data the model learned from. That is a different question from matching two
arms' features, so it uses a different split: every training row, with the real labels, in sample.
The reasoning and what it costs are in [`analyses/direction.md`](analyses/direction.md). A bundle
therefore carries a second matrix, `effect_values.npz`, with the labels alongside it.

---

## 2. The header

`features.json` carries a small header beside the records:

| key | what it says |
|---|---|
| `schema_version` | the record layout this file was written against. A reader that does not know it **refuses** the file instead of coercing it |
| `importance_kind` | what the `importance` numbers in this file are; see §3 |
| `direction_kind` | what the `direction` words were read off; see §3 |
| `seed`, `labels` | which cell this is |
| `target_kind` | the kind of target the cell was fitted against, `binary` or `continuous` ([`dataset.md`](dataset.md)). It decides the words `direction` is given in and what `marginal_effect` measures. |
| `model_settings`, `featuriser` | a Core cell's model and featuriser, each as its `describe()` gives it, so a file says how it was produced. Both are null in a DEFT or an IGSR cell |

`importance_kind` and `direction_kind` sit in the header, and not on each record, because every
record in one file shares one. They are free text drawn from a stated vocabulary
(`IMPORTANCE_KINDS` in `harness/features.py`, `DIRECTION_KINDS` in `harness/direction.py`), so
adding a model means adding a name there and leaves the record unchanged.

---

## 3. The fields

Thirteen, and only three are required.

### Required

| field | |
|---|---|
| **`name`** | Must be a **legal Python identifier**, enforced in `__post_init__`. A language model copies the naming conventions it is shown into the function names it writes, so a name like `upstream_-15nt_GC` produces code that will not compile downstream. The check catches such a name when the record is made, including one arriving from a vendored method. |
| **`description`** | Free text, what the feature computes. For a generated feature this is the model's own description, and it is what the judge and the concept grouper read. |
| **`kind`** | What it computes, as a category: `positional`, `kmer`, `llm_generated`, `covariate` (a measurement supplied with each sequence, as it is), `composition` (used only by the composition baseline), `unknown`. Used to filter: `analyse_recurrence.py` restricts to `llm_generated` by default, because a naive positional split is drawn from an enumeration and its recurrence is arithmetic, not evidence. |

### The recipe

| field | |
|---|---|
| **`params`** | The recipe, where there is one: `{"position": 87, "symbol": "G"}`, `{"kmer": "GGT", "k": 3}`, `{"column": "signal_mean"}` for a covariate. Empty for generated code. A position is read in the dataset's positional window where it declares one, and the featuriser's description records the window. A window that pads adds `{"position": 87, "symbol": "pad"}`, 1 where the sequence stops before that position. A k-mer counted per k-mer position in the sequence, instead of in total, says so: `{"kmer": "GGT", "k": 3, "normalise": "per_length"}`. |
| **`source`** | The function body, for a feature that is code. |
| **`rationale`** | For a feature a language model proposed, the model's stated reason it might predict the outcome. The judge is shown it. |

`params` and `source` are the **extension point**. Adding gapped *k*-mers, PWM scans or motif
features means emitting a new `kind` with its own `params`, with no change to the record and none
downstream:

```
gapped_kmer   {"pattern": "GG.TCA", "k": 5, "n_gaps": 1}
pwm_scan      {"motif_id": "MA0139.1", "db": "JASPAR", "mode": "max"}
```

They also decide feature **identity** for exact matching. `definition_digest` hashes the normalised
source (parsed, with names, formatting, comments and the docstring stripped), or falls back to
`{kind, params}` where there is no source. So two candidates that differ only in what the model
called them hash alike.

> **A trap.** The digest takes the source branch when a record has source and the params branch
> when it does not, so a code-carrying record and a params-carrying record for the *same function*
> can never match. That is why the DEFT adapter recovers the recipe for its naive positional
> candidates, and for a covariate it was handed, and drops the source: it puts both sides on the
> `{kind, params}` branch, where they agree by construction.

### Where it came from

| field | |
|---|---|
| **`origin`** | *Whose method* produced it: `seqmaestro`, `deft`, `igsr`. |
| **`provenance`** | *How it entered this arm's answer*: `selected` (taken from the pool the arm was handed) or `generated` (constructed while fitting, so it was not available to any other arm drawing on the same pool). |

**`kind` and `origin` are independent, and both are needed.** `kind` is what the feature computes,
and `origin` is whose method made it. They differ where the research question lives: an
`llm_generated` feature proposed by DEFT and one from our own proposer have the same kind and
different origins, and once features from several runs pool into one library, `origin` is the only
field that tells them apart. A kind is also a different fact from the source of the arm's pool
(`experiments/configs/README.md`, Arms): the features of all three written sources,
`llm_generated`, `llm_generated_iterative` and `llm_generated_joint`, have the kind
`llm_generated`, and the arm's pool says which source wrote them.

**`provenance` is relative to the arm.** A language-model pool handed to a model is `selected` for
that arm, because the arm only chose from it. The same feature constructed mid-fit by an arm on the
iterative source is `generated`. Two arms can report one feature under different provenance and both be right.
It is **declared by the producer, never inferred**. The alternative reading ("is this function
inside the pool's span") is undecidable in general and ill-defined for nonlinear features, so the
field records what the producer did.

The field is shown beside each feature in `feature_recurrence.csv` and in the spreadsheet of a
run's features (`export_features_xlsx.py`). `analyse_recurrence.py` chooses its arms by their pool:
every arm whose pool holds a written source, the one-round source included.

### How much the model used it

| field | |
|---|---|
| **`importance`** | The fitted model's own number. Units differ by model, named once in the header: L1 fits on columns standardised on its fitting rows and reports `l1_coefficient_per_sd`, a signed coefficient per standard deviation of the feature, in log-odds for a binary target and in the target's units for a continuous one. An IGSR cell reports `igsr_influence`, IGSR's own number: the rise in validation mean squared error when the term's weight is set to zero, against the validation labels its search chose on, which are shuffled on a control arm. After Stage 2, its refits, made with the least-squares fit its search used, report `least_squares_coefficient_per_sd`, the signed coefficient per standard deviation of the term. A PySR cell reports `mean_abs_shapley`: the feature's mean absolute Shapley value in the chosen equation's output, over up to 2,000 fitting rows, in log-odds for a binary target and in the target's units for a continuous one. |
| **`permutation_importance`** | How far the arm's headline falls when this feature's column is shuffled: AUROC for a binary target, R-squared for a continuous one. A Core cell's `run.json` names the metric and the repeats. |

**The two are not interchangeable.** An L1 coefficient, a boosting gain, a shape-function
magnitude and a tree's impurity decrease are four different quantities, and the `Model` protocol
states that they are never comparable between arms. Within an arm they are the right number. Across
arms only their *ranks* are meaningful, which is why `rank_by_importance` exists and why every
analysis takes a top-N cut and not a threshold.

`permutation_importance` is the model-agnostic answer: shuffle a column, predict again, and see how
far the score falls. It needs only a fitted model, the rows and a metric, so it is computed
identically for every arm, the vendored tree included, when the grid asks for it
(`grid.permutation_repeats` above 0). `_shared.yaml` sets 0, since nothing reads the field yet, so a
run's records hold no value for it unless its config raises the count. Positive means the model leans on the
feature, and around zero means it does not. **A negative value is left as measured** and not
clipped, because a floor at zero would turn noise into a weak positive signal for every unused
feature.

> **Its known weakness here is correlated columns.** Shuffling `pos_50_A` leaves `pos_50_C/G/T` in
> place, and between them they largely determine what was removed, so the drop understates the
> feature. One-hot and *k*-mer columns are deflated relative to standalone continuous ones. Ranks
> are computed within an arm, and most arms hold one kind of feature throughout, so the deflation
> applies evenly and the ordering survives. An arm mixing indicators with continuous features is the
> case where it does not.

### Which way it points

| field | |
|---|---|
| **`direction`** | Which class the **model** uses the feature to argue for, or which way it moves a continuous target. |
| **`marginal_effect`** | How far the feature goes with the target **on its own**. |

Full detail is in [`analyses/direction.md`](analyses/direction.md). In brief, `direction` is one of
`higher_in_positive`, `higher_in_negative` or `non_monotone` for a binary target, and
`increases_target`, `decreases_target` or `non_monotone` for a continuous one, taken from how
each model works and named in `direction_kind`. `marginal_effect` is a rank correlation over
the **training rows against the real target**: `2 × AUC − 1` (rank-biserial) for a binary target,
positive meaning higher values are more common in the positive class, and Spearman's rho for a
continuous one, positive meaning higher values go with a higher target. The harness
computes it, not the model, so it means the same thing for an arm fitted in this process and for one
that arrived from another environment. It is in sample, for the reasons
[`analyses/direction.md`](analyses/direction.md) sets out. A control arm carries the same value as
its real counterpart, so `direction` is the only field that separates the two.

**When the two disagree, that is a finding.** It says the model uses the feature to correct for
something else, and not for the feature's own association with the outcome.

### Whatever else a method knows

| field | |
|---|---|
| **`extra`** | Unschema'd, per-method. A language-model feature may carry three keys. `repaired` is `true` where the code the model first wrote failed on the training rows and this is the corrected code it returned when the failure was sent back. `renamed_from` is the name it was proposed under, where the correction measures something different and so carries a new name and description. `undefined_share` is the share of training rows on which the feature is undefined (NaN or infinite), present where it is above 0; those rows are filled with the feature's median, since a model needs a number in every row. |

For what one method knows and others do not. The DEFT adapter populates it with
`{"deft_origin": "naive"|"llm", "nodes": [...]}`: one entry per node the feature splits at, carrying
depth, path from the root, threshold, split score, rows reaching it, impurity decrease, and the class
rate (`rate_left`, `rate_right`) or a continuous target's mean (`mean_left`, `mean_right`) either
side. It has no schema because most arms have no notion of a node. The proposer arms add
the three keys above where they apply, and so does the DEFT adapter for a feature its language
model's repair round corrected (`repaired`, `renamed_from`). DEFT keeps the name its language model
proposed for a feature, which may not be a Python identifier, and the adapter then records the
feature under a legal name and keeps the proposed one as `name_as_proposed`. A proposer arm does
the same where a written feature is proposed under the name of an enumerated column of its pool,
such as `kmer_CG`: the written feature is recorded as `kmer_CG_written`, with `name_as_proposed`
holding the name it was given. The IGSR adapter records a term's `weight`, the iteration of the
search it was `first_seen` in, its `undefined_share` and the `fill` its undefined rows take, with
`repaired` and `renamed_from` where they apply. **No analysis and no part of the report uses
`extra`**, so the tree structure a DEFT run recovers is recorded in those files and not used.

---

## 4. Who fills what

| producer | `kind` | `params` | `source` | `provenance` | `origin` |
|---|---|---|---|---|---|
| `PositionalFeaturiser` | `positional` | position, symbol | none | `selected` | `seqmaestro` |
| `KmerFeaturiser` | `kmer` | kmer, k | none | `selected` | `seqmaestro` |
| `CovariateFeaturiser` | `covariate` | column | none | `selected` | `seqmaestro` |
| `CompositionFeaturiser` (the baseline) | `composition` | symbol | none | `selected` | `seqmaestro` |
| `GeneratedFeaturiser` (proposer) | `llm_generated` | none | the function | `selected` | `seqmaestro` |
| `IterativeGeneratedFeaturiser` | `llm_generated` | none | the function | **`generated`** | `seqmaestro` |
| DEFT adapter, naive candidate | recovered recipe | position, symbol; or kmer, k for a handed k-mer count; or column, for a covariate | none | `selected` | `deft` |
| DEFT adapter, its own proposal | `llm_generated` | none | the function | **`generated`** | `deft` |
| IGSR adapter, a term its search wrote | `llm_generated` | none | the function | **`generated`** | `igsr` |
| IGSR adapter, a column it was handed | the column's recipe | column | none | `selected` | `igsr` |

An arm on the iterative source reports both provenances. Its first, blind round is the proposal
its pool holds, written by a `GeneratedFeaturiser`, so those features are `selected`. Only the
rounds shown a fit are written by `IterativeGeneratedFeaturiser` and are `generated`.

`importance`, `permutation_importance` and `direction` are attached by `run_model` after the fit,
or written by the adapter for an arm that arrives as a bundle. `marginal_effect` is attached by
`score_arm`, which is its **only** writer on either path into a run directory.

---

## 5. Two ways in, one shape out

```
run_model()  ─┐
              ├─→  score_arm()  ─→  the five files
ingest_bundle()┘
```

`run_model` is the in-process path. `ingest_bundle` is for an arm that ran in another conda
environment and cannot import this package at all. It reads a **bundle** the adapter wrote by hand
and scores it through the same function. That is why the record is flat and plain-typed: a foreign
adapter has to be able to produce conforming JSON with `json.dump`.

It is also why `load_features` **validates what it reads**. An unknown field, a missing required
field, a wrong `schema_version` or a `direction` outside the vocabulary each raises on read, so
drift is caught on the first read and not discovered in a results table.

Both paths score validation only. The test split is scored once, at the reveal, from the model each
cell saved, into the same files (see [`heldout.md`](heldout.md)).

DEFT's and IGSR's arms keep their model in their bundle, because using it needs the method, which
this package must not import. Every Core cell also writes `fitted.pkl`, the featuriser and model
together, so a question the run did not already answer costs a load instead of a refit.

---

## 6. Adding a field

A new optional field needs no `SCHEMA_VERSION` bump. `permutation_importance`, `direction` and
`marginal_effect` are optional fields of this kind. To add one:

1. Add it to `FeatureRecord` with a `None` default and a comment saying what fills it.
2. Add it to `_FIELDS` in `harness/features.py`. Serialisation is derived from that tuple, so
   writing needs nothing else.
3. Read it in `_from_dict` with `.get()`, and validate it there if it has a vocabulary.
4. **No version bump if it is optional.** `_from_dict` requires only name, description and kind, so
   a `features.json` without the field (an older file, or one from an adapter that does not report
   it) still loads with the field unset. Bump the version only when a reader would get an existing
   file *wrong*.
5. Update this page.

**Bump the version** when the meaning of an existing field changes, or when a new field is required.

---

## 7. What the record cannot carry

**A term over more than one feature.** An interaction, meaning a model component defined on a
*pair*, has no place here. `name` must be a legal identifier, every record owes a column to
`feature_values.npz`, and an interaction has no model-independent value to put there. Every
`generated` feature is a model-*independent* function of the sequence, evaluable on new rows with no
model present, whereas a pairwise term is a model-*dependent* component. In practice, an Explainable
Boosting Machine fitted with `interactions > 0` returns per-term importances, and `run_model`
silently drops every term that does not name a feature. Nothing raises. Do not add such an arm
without dealing with that first.

**A result.** No field here holds a finding about a dataset. The record says what a feature is and
what this fit made of it. What that means goes in the run's report.
