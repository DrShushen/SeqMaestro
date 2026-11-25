# The feature record

> This page tracks the record ({py:class}`~seqmaestro.features.base.FeatureRecord`), the files that hold it ({py:mod}`seqmaestro.harness.features`), and every producer and reader of a field: the featurisers, the proposer, the harness ({py:mod}`seqmaestro.harness.runner`) and the adapters of the vendored methods ([`deft_bundle.py`](../../experiments/adapters/deft_bundle.py) and [`igsr_bundle.py`](../../experiments/adapters/igsr_bundle.py)). Change it in the same commit as any of them.

All [arms](../glossary.md#arm) report the features their models used as the same record, the {py:class}`~seqmaestro.features.base.FeatureRecord`. A k-mer count, a positional indicator, a decision-tree split and a function that a language model wrote all take this form. As a result, the [stability analysis](analyses/stability_and_co_discovery.md), the [judge](../glossary.md#judge), the spreadsheet of a run's features and the report treat every arm alike, without needing to know which arm produced a feature.

The class describes each of its fields. This page explains what spans several parts of the code:

- The files that hold the records, and why the values of the features are stored beside them.
- The reasoning behind the fields that several parts of the code depend on.
- Which producer fills which field.
- The two paths by which records reach a run.
- How to add a field.
- What the record cannot hold.

## Files

A finished Stage 2 [cell](../glossary.md#cell) keeps its records in two files:

- `features.json` holds the records, under a header with what every record of the file shares, such as the cell's seed and the unit of the importances.
- `feature_values.npz` holds the value of each recorded feature on the validation rows, with one column per record, in the order of the records.

{py:mod}`seqmaestro.harness.features` lists these files with the keys of the header, and {py:mod}`seqmaestro.harness.cell` lists the other files of a cell. The two files are paired by position, so a mismatch between them would compare the wrong features without any error. {py:func}`~seqmaestro.harness.bundle.read_bundle` and {py:func}`~seqmaestro.harness.stability.load_run` therefore refuse a cell whose value columns do not match its records in name and order.

**The values are stored** so that no feature has to be computed again. Running a written feature again later would need the code's checks and the same environment, and the other features are recipes that only their own featuriser or method can compute. With the values stored, a comparison of any two arms' features only loads arrays and correlates them, and code that a language model wrote never runs outside the run that produced it.

**The values are computed on the validation rows**, the split on which models are already chosen. Matching features by their values uses no labels, so evaluating the features there exposes nothing new. Permutation importance is measured on the same rows, so that it and the matching see the same data.

**`marginal_effect` is measured on the training rows instead**: on every training row, against the real labels. It describes what a feature does in the data the model learned from, which is a different question from whether two arms' features match. [Direction](analyses/direction.md) gives the reasoning and its limits. The harness computes it for every arm, so an arm fitted in another environment hands over its features' values on the training rows, with the labels, as `effect_values.npz` in its **bundle**: the directory of files in which a vendored method returns its results ([paths into a run](#paths-into-a-run)).

## Fields

{py:class}`~seqmaestro.features.base.FeatureRecord` says what each field holds and when it is filled. The sections below give the reasoning behind the fields that several parts of the code depend on.

### Kind and params

A new kind of feature enters through two fields: `kind`, the category of what a feature computes, and `params`, its recipe where the feature is not code. The motif features, for example, are the kind `pwm_scan` with their own `params` ([motifs](motifs.md)). Gapped k-mers would be another kind, with no change to the record or to anything that reads it:

```
gapped_kmer   {"pattern": "GG.TCA", "k": 5, "n_gaps": 1}
```

Analyses filter on `kind`. The [recurrence analysis](../glossary.md#recurrence-analysis), for example, groups only features of the kind `llm_generated` (its `kinds` in [`_shared.yaml`](../../experiments/configs/_shared.yaml)), because a positional feature is drawn from an enumeration, and its recurrence across seeds follows from arithmetic alone.

A kind is a different fact from the [feature source](../glossary.md#feature-source) in the arm's [pool](../glossary.md#pool). The features of all three [written sources](../glossary.md#written-source) have the kind `llm_generated`, whereas only the arm's pool says which source wrote them.

### Identity

The stability analysis pairs the features that have the same definition before it matches the rest by their values, and {py:func}`~seqmaestro.harness.features.definition_digest` decides what counts as the same definition. It hashes the normalised `source` (parsed, with the function's name, formatting, comments and docstring removed) where a record has one, and `{kind, params}` where it has none. Two written features that differ only in what the language model called them therefore have the same digest.

**A record with source and a record with params never match**, even for the same function, because the digest takes a different branch for each. The DEFT adapter therefore records the candidates that DEFT's tree derives without its language model (a positional indicator, or a column it was handed) by their recipe and without source. Such a record and SeqMaestro's record of the same feature both reach the digest through `{kind, params}`, where they agree by construction.

### Origin

`origin` names whose method produced a feature, whereas `kind` names what it computes. Both are needed to compare methods: an `llm_generated` feature proposed by DEFT and one written by SeqMaestro's [proposer](../glossary.md#proposer) have the same kind and different origins. Once features from several runs are pooled into one library, `origin` is the only field that tells them apart.

### Provenance

`provenance` says whether the arm took a feature from the pool it was handed (`selected`) or constructed it while fitting (`generated`). The class explains why provenance is relative to the arm, and why the producer declares it, so every producer in the [table below](#fields-by-producer) states one.

Readers show the provenance beside each feature without filtering on it: the recurrence analysis in `feature_recurrence.csv`, and the spreadsheet of a run's features ({py:mod}`seqmaestro.report.spreadsheet`). The recurrence analysis, for instance, chooses its arms by their pool: every arm whose pool includes a written source, the one-round source included.

### Importance

`importance` is the fitted model's own number, and its unit differs by model: a coefficient per standard deviation, a boosting gain, the size of a shape function or a tree's impurity decrease. The header of `features.json` names the unit (`importance_kind`, one of the `IMPORTANCE_KINDS` that [`features.py`](../../seqmaestro/harness/features.py) lists with the model of each), and the `importances` method of each Core model ({py:mod}`seqmaestro.models`) says what its number is.

**Importances are compared between arms only by rank** ({py:func}`~seqmaestro.harness.features.rank_by_importance`). Within an arm the model's own number is the right measure, whereas the numbers of different models measure different quantities. Every analysis therefore takes each arm's N most important features, since a threshold on importance would mean something different in each arm.

`permutation_importance` is measured in the same way for every arm: shuffle a column, predict again, and see how far the score falls. {py:mod}`seqmaestro.harness.importance` computes it for an arm fitted in Core, and the adapters compute it by the same rule. The module also explains its weakness on correlated columns, such as the four indicators of one position. It is computed only where the config sets `grid.permutation_repeats` above 0. [`_shared.yaml`](../../experiments/configs/_shared.yaml) sets 0, because no analysis uses the field, so a run's records hold no value for it unless the run's config raises the number.

### Direction

`direction` says which class the model takes a feature to point to, or which way it moves a continuous target, whereas `marginal_effect` says how far the feature goes with the target on its own. [Direction](analyses/direction.md) explains both fields and how each model reaches a direction. Two points concern the record itself:

- **The harness alone writes `marginal_effect`** ({py:func}`~seqmaestro.harness.runner.score_arm`), so it means the same for an arm fitted in Core as for an arm from another environment.
- **A feature has the same `marginal_effect` in a [permuted](../glossary.md#label-condition) cell as in a real one**, because the effect is measured against the real labels on every training row, whatever the cell was fitted on. Of the two fields, only `direction` can tell the two cells apart.

### Extra

`extra` holds what one method knows and the others do not. It has no schema, because most arms have no notion of a node or a matrix. Each producer puts its own facts there:

- **A written feature** may have `repaired`, where the code first written failed on the rows it was checked on and the language model corrected it, and `renamed_from`, where the correction measures something different under a new name ({py:mod}`seqmaestro.proposer.proposer`). It has `undefined_share` where the feature is undefined on some of its fitting rows, which then take the feature's median ({py:mod}`seqmaestro.proposer.featuriser`). A written feature recorded under a name other than the one it was proposed under, because a column of its pool had that name or the name was not a valid identifier, keeps the proposed name in `name_as_proposed`.
- **The DEFT adapter** records DEFT's own label for the candidate (`deft_origin`) and each node the feature splits at (`nodes`), with the threshold, the rows that reach the node, and the class rate or the target's mean on either side.
- **The IGSR adapter** records a term's `weight`, the iteration of the search in which it was `first_seen`, and the `fill` that its undefined rows take.
- **A discovered motif** has its matrix, STREME's E-value and its number of sites, and, where the run compared it with a collection, the closest known motif (`closest_known`, [discovered motifs](motifs.md#discovered-motifs)).

**The analyses do not use `extra`.** The report uses it only to name the known motif closest to a discovered one, in the tables of a reduced arm's features, where Tomtom's q-value shows the two are alike ({py:mod}`seqmaestro.report.tables`).

## Fields by producer

| Producer | `kind` | `params` | `source` | `provenance` | `origin` |
|---|---|---|---|---|---|
| {py:class}`~seqmaestro.features.positional.PositionalFeaturiser` | `positional` | `position`, `symbol` | None | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.features.kmer.KmerFeaturiser` | `kmer` | `kmer`, `k`, and `normalise` for a normalised count | None | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.features.motif.KnownMotifFeaturiser` | `pwm_scan` | `source`, `summary`, `strands`, `collection`, `motif_id`, `motif_name` | None | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.features.motif.DiscoveredMotifFeaturiser` | `pwm_scan` | `source`, `summary`, `strands`, `tool`, `direction`, `rank`, `consensus`, `motif_digest` | None | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.features.covariates.CovariateFeaturiser` | `covariate` | `column` | None | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.features.composition.CompositionFeaturiser` (the baseline) | `composition` | `symbol` | None | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.proposer.featuriser.GeneratedFeaturiser` (the proposer) | `llm_generated` | Empty | The function | `selected` | `seqmaestro` |
| {py:class}`~seqmaestro.proposer.iterative.IterativeGeneratedFeaturiser` | `llm_generated` | Empty | The function | **`generated`** | `seqmaestro` |
| DEFT adapter, a candidate the tree derives | `positional`, or the kind of the column it was handed | `position` and `symbol`, or the column's recipe | None | `selected` | `deft` |
| DEFT adapter, a candidate its language model writes | `llm_generated` | Empty | The function | **`generated`** | `deft` |
| IGSR adapter, a term its search writes | `llm_generated` | Empty | The function | **`generated`** | `igsr` |
| IGSR adapter, a column it was handed | The column's kind (`covariate`) | `column` | None | `selected` | `igsr` |

A motif's count also states its `p_threshold`, and a `normalise` other than `counts`, since neither changes a best match ([motifs](motifs.md)).

An arm on the [iterative source](../glossary.md#iterative-source) reports both provenances. Its first round, which is shown no fit, is the proposal its pool holds, written by a `GeneratedFeaturiser`, so those features are `selected`. Only the rounds that are shown a fit are written by `IterativeGeneratedFeaturiser`, and their features are `generated`.

The fields that describe the fit are filled after it. {py:func}`~seqmaestro.harness.runner.run_model` attaches `importance`, `permutation_importance` and `direction` to the records of an arm fitted in Core, whereas the adapter writes them for an arm that arrives as a bundle. `marginal_effect` is attached by {py:func}`~seqmaestro.harness.runner.score_arm` on both paths.

## Paths into a run

```{mermaid}
flowchart LR
  A["run_model()<br/>An arm fitted in Core"] --> S["score_arm()"]
  B["ingest_bundle()<br/>A vendored method's bundle"] --> S
  S --> F["The five files of a cell"]
```

Every record reaches a run's directory by one of two paths. Both end in {py:func}`~seqmaestro.harness.runner.score_arm`, so the files of a cell are the same whichever path produced them:

1. {py:func}`~seqmaestro.harness.runner.run_model` fits an arm in Core (the `SeqMaestro_Core` environment) and takes the records from the arm's featuriser and model.
2. {py:func}`~seqmaestro.harness.runner.ingest_bundle` takes the bundle of an arm that ran in the environment of a [vendored method](architecture.md#two-boundaries). The method's adapter writes the bundle ([the adapters](../../experiments/adapters/README.md)), and `ingest_bundle` checks it with {py:func}`~seqmaestro.harness.bundle.read_bundle` and scores it through the same function.

An adapter runs where this package cannot be imported, so it writes its records by hand, with `json.dump`. **The record is flat and plainly typed** for this reason. For the same reason {py:func}`~seqmaestro.harness.features.load_features` checks what it reads: an unknown field, a missing required field, an unknown `schema_version`, or a `direction` or `provenance` outside its vocabulary each raises an error when the file is read. A writer that has drifted from the record is caught on the first read, before its records reach a results table.

Both paths score the validation rows only. The [held-out split](../glossary.md#held-out-split) is scored once, at the [reveal](../glossary.md#reveal), from the model that each cell saved ([the held-out split and the reveal](heldout.md)). A Core cell saves its featuriser and model together in `fitted.pkl`, so a question the run did not answer needs only a load, without a refit. DEFT's and IGSR's arms keep their model in their bundle, because using it needs the method, which this package must not import.

## Adding a field

A new optional field needs no change to `SCHEMA_VERSION`. To add one:

1. Add it to {py:class}`~seqmaestro.features.base.FeatureRecord` with a `None` default, and describe it, with what fills it, in the class's list of fields.
2. Add it to `_FIELDS` in {py:mod}`seqmaestro.harness.features`. Writing a record is derived from that tuple, so it needs nothing else.
3. Read it in `_from_dict` with `.get()`, and check it there if it has a vocabulary.
4. Keep `SCHEMA_VERSION`. `_from_dict` requires only `name`, `description` and `kind`, so a `features.json` without the new field (an older file, or one from an adapter that does not report it) still loads, with the field unset.
5. Update this page where the field changes which producer fills what.

**Change `SCHEMA_VERSION`** when a reader would interpret an existing file wrongly: when the meaning of an existing field changes, or when a new field is required.

## Limits of the record

**A term over more than one feature.** An interaction, meaning a model's term defined on a pair of features, has no place in the record. Every feature the record holds, a `generated` one included, is a function of a row that can be evaluated on new rows without any model, whereas an interaction exists only inside one model. It therefore has no values of its own for `feature_values.npz`, and its name, such as `a & b`, is not a valid identifier.

This matters for the [EBM](https://interpret.ml/docs/ebm.html). An EBM fitted with `interactions` above 0 reports an importance for each pairwise term, and `run_model` keeps only the importances that name a feature of the featuriser, so it drops every pairwise term without an error. An arm with interactions therefore needs its pairwise terms handled before it is added.

**A result.** The record holds no finding about a dataset. It says what a feature is and what one fit made of it, whereas what that means for the dataset goes in the run's report.
