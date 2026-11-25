# The Stage 2 analyses

> These pages track the analyses in `seqmaestro/pipeline/analyses/`, the matching they share in `seqmaestro/harness/stability.py`, and the scripts `direct_effect_test.py` and `feature_group_dependence.py`. Change a page in the same commit as the code of its analysis. [The analysis scripts](../../../experiments/analysis/README.md) say which script to use and when, whereas these pages say how each number is computed and what it can support.

Stage 2 fits every [arm](../../glossary.md#arm) at every [seed](../../glossary.md#seed), once on the real labels and once on shuffled ones. The analyses compare these fits by the features they report, and each answers one question about those features:

- [Stability and co-discovery](stability_and_co_discovery.md): whether one arm reports the same features again at another seed (`analyse_stability.py`), and whether several arms arrive at the same property (`analyse_co_discovery.py`).
- [Recurrence](recurrence.md): whether an arm whose features a language model writes arrives at the same ideas again, although it names them differently each time, and whether different arms arrive at the same idea (`analyse_recurrence.py`).
- [The judge](judge.md): how biologically significant each feature would be if it held (`run_judge.py`).
- [Direct measurement](direct_measurement.md): what one feature does on its own, with no model involved (`direct_effect_test.py`, `feature_group_dependence.py`).
- [Direction](direction.md): which class a feature points to, or which way it moves a continuous target, both in the fitted model and in the data.

Once every arm has finished, each analysis takes the [feature records](../../glossary.md#feature-record) that the [cells](../../glossary.md#cell) wrote ([the feature record](../feature_record.md)). The analyses do not fit any model or open the [held-out split](../../glossary.md#held-out-split). The [driver](../../glossary.md#driver) runs stability, co-discovery and recurrence after Stage 2, and the judge where the [config](../../glossary.md#config) sets `grid.judge`, whereas the scripts of direct measurement are run by hand on the cells of interest.

The analyses share three ideas, which the other pages of this section rely on:

- [Comparing features by their values](#comparing-features-by-their-values), whatever each arm calls them.
- [The top N](#the-top-n): each cell contributes its N most important features, and no more.
- [The permuted cells](#the-permuted-cells): every number is compared with what the same analysis returns on fits to shuffled labels.

## Comparing features by their values

Arms do not share a vocabulary. The positional featuriser calls a feature `pos_50_A`, DEFT's tree calls the same feature `pos_50_is_A`, and the [proposer](../../glossary.md#proposer) may write it as `central_base_is_a`, which gives three names and three implementations of one property. The analyses therefore never compare names.

Instead, every cell evaluates the features it reports on the rows of the validation split, and writes their values to `feature_values.npz` beside its feature records. These rows, which are the same for every cell of a run, are the *reference rows*. Two features count as the same when their values agree on the reference rows, which happens when they compute the same thing. Matching uses no labels, and models are already chosen on the validation split, so evaluating features on it exposes nothing new (`REFERENCE_SPLIT` in [`runner.py`](../../../seqmaestro/harness/runner.py)).

Stability and co-discovery build on these values. Recurrence does not, because an idea computed over a shifted window does not correlate with itself, as [recurrence](recurrence.md) explains.

### Matching two sets of features

{py:func}`~seqmaestro.harness.stability.match_pair` matches the features of two fits one to one, in three steps:

1. Features with an identical definition are paired first.
2. The others are compared by the absolute [Spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) of their values on the reference rows ({py:class}`~seqmaestro.harness.stability.BehaviouralMatcher`).
3. A [Hungarian assignment](https://en.wikipedia.org/wiki/Hungarian_algorithm) pairs them so that the total correlation is as high as it can be, and a pair counts as a match only if its correlation is at least 0.8.

The docstrings of the two give the reason for each choice. One choice has a consequence that is easy to miss: since the correlation is absolute, a feature matches its own negation, so the signs of two matched features have to be aligned before their directions are compared ([stability and co-discovery](stability_and_co_discovery.md) says how co-discovery aligns them).

Two kinds of column need care:

- **A column with undefined values.** A written feature can be undefined on some rows: a feature that measures the span of a CpG cluster has no value for a sequence without one. A value that is not finite is replaced by its column's median before ranking ({py:func}`~seqmaestro.harness.stability.impute_non_finite`), since a NaN would otherwise spread across the whole column. The stability analysis prints how many columns it imputed.
- **A constant column.** A column that does not vary on the reference rows has no ranks, so it scores 0 against everything and can match only by its definition. The stability analysis prints, for each arm, how many such columns each seed has among its top N (`constant_features`).

### Chaining matches into clusters

{py:func}`~seqmaestro.harness.stability.consensus_labels` chains the matches among many fits into clusters. Every feature of every fit is a node of one graph, each matched pair is an edge, and a cluster is a connected component. The clusters therefore do not depend on the order of the fits.

The chaining is transitive, however, so a chain of borderline matches can merge two different properties. The symptom is a cluster that holds two features of the same fit. The stability analysis prints a line for each arm where that happens, with the number of seeds it happens in (`runs_with_chained_clusters`), so check its log before trusting a surprising cluster.

A cluster that holds features of several arms is a [concept](../../glossary.md#concept). Co-discovery groups the arms' features in this way, and so do the concept tables of [the report](../report.md).

## The top N

Models differ by orders of magnitude in how many features they use. An [explainable boosting machine](https://interpret.ml/docs/ebm.html) fits a shape function to every input and so uses all of them, whereas a shallow tree uses a handful. A comparison of a set of 400 features with one of 15 would measure the sizes of the two sets, and say nothing about their agreement. Every analysis therefore takes the same number of features from each cell: its N most important, by the model's own importance. N is `analyses.top_n` in the run's config ([`_shared.yaml`](../../../experiments/configs/_shared.yaml) sets the shared value), whereas `direct_effect_test.py`, which is run by hand, takes N from its option `--top-n`.

N is a convention, which the config states. Within a cell, the features are ranked by the size of their importance, whatever its sign, and a model that used fewer than N features contributes all of them ({py:func}`~seqmaestro.harness.features.top_features`). Each model's importance is on a scale of its own, so importances cannot be compared between arms, whereas ranks can. The top N is therefore chosen by rank.

## The permuted cells

Every arm is fitted at every seed twice: on the real labels, and on the training labels shuffled. The fits on shuffled labels are the *permuted cells*, and the two kinds of fit are the two [label conditions](../../glossary.md#label-condition) ([Stage 2: the suite](../suite.md) says how the labels are shuffled). Every cell is scored on the real labels, so a permuted cell should score at chance. For the analyses, the permuted cells show what each analysis returns when there is nothing to find.

What an analysis returns on an arm's permuted cells is the arm's *chance level*, and it differs widely between arms. An arm that selects from a fixed vocabulary has a chance level near zero. A language model, by contrast, proposes similar features whatever the labels say, so an arm on a [written source](../../glossary.md#written-source) agrees with itself substantially on shuffled labels. Such an arm can show the highest raw agreement of any arm and still rank low once its chance level is taken into account. A figure that ignores the chance level therefore flatters such arms. Stability and co-discovery instead correct each arm's agreement by its own chance level:

```
kappa = (observed - chance) / (1 - chance)
```

Here `observed` is the agreement among the arm's real cells, and `chance` is the same agreement among its permuted cells ({py:func}`~seqmaestro.harness.stability.chance_corrected_agreement`). A kappa of 0 means the arm agrees no better than it does on shuffled labels, and 1 means perfect agreement. Kappa is the column to quote wherever there is one. The other analyses use the permuted cells in ways of their own: recurrence counts each idea's features under both label conditions, and the judge can rate the permuted cells' features, to measure the scores it gives when there is nothing to find.

**The control is weaker for an arm on a written source.** A permuted cell on a written source is still told the [task description](../../glossary.md#task-description), so its language model proposes a sensible set of features whatever the labels. Those features can be associated with the target although no label reached the proposal. The control is therefore strong for an arm that selects from a fixed pool and weaker for an arm that writes its features, and a statement that relies on the control should say which kind of arm it concerns.

```{toctree}
:hidden:

stability_and_co_discovery
recurrence
judge
direct_measurement
direction
```
