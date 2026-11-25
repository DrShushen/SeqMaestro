# Stability and co-discovery

> This page tracks {py:mod}`seqmaestro.pipeline.analyses.stability` and {py:mod}`seqmaestro.pipeline.analyses.co_discovery`, which [`analyse_stability.py`](../../../experiments/analysis/analyse_stability.py) and [`analyse_co_discovery.py`](../../../experiments/analysis/analyse_co_discovery.py) run, and the matching they share in {py:mod}`seqmaestro.harness.stability`. Change it in the same commit as any of them.

Two analyses of [Stage 2](../../glossary.md#stage) measure whether a [feature](../../glossary.md#feature) found in one fit is found again:

- **Stability** measures whether one [arm](../../glossary.md#arm) reports the same features again when it is fitted at another [seed](../../glossary.md#seed), that is, on another draw of the training rows.
- **Co-discovery** measures whether different arms, fitted on the same rows at the same seed, arrive at the same property.

A feature that an arm reports at every seed, or that arms on different [feature sources](../../glossary.md#feature-source) reach independently, is less likely to be an accident of one fit.

Both analyses rest on the three ideas that [the Stage 2 analyses](index.md) explains:

- Features are [compared by their values](index.md#comparing-features-by-their-values) on the validation rows, whatever each arm calls them, and matched features are chained into clusters.
- Each [cell](../../glossary.md#cell) contributes only its [top N](index.md#the-top-n) features, the N most important by its model's own importance.
- Every figure is compared with the same figure on the [permuted cells](index.md#the-permuted-cells) (the fits on shuffled labels), which is the arm's chance level.

The [analysis scripts](../../../experiments/analysis/README.md#stability-co-discovery-and-recurrence) set out how these two analyses differ from the [recurrence analysis](../../glossary.md#recurrence-analysis), which groups features by idea.

## Stability

Stability takes one arm at all its seeds, under one [label condition](../../glossary.md#label-condition), and measures its agreement in four steps:

1. Each seed's features are reduced to the top N.
2. For each pair of seeds, the two sets are matched, and the pair's [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) is the number of matched features over the size of the union of the two sets, `matched / (|A| + |B| − matched)`. The mean over all pairs is `top_n_jaccard`.
3. The same mean over the arm's permuted cells is the arm's chance level.
4. `top_n_kappa` corrects the real-label `top_n_jaccard` by the chance level, as [the permuted cells](index.md#the-permuted-cells) describes.

**`top_n_kappa` is the stability figure to report**, because it is the only one that means the same for every arm. It is 0 when an arm reproduces its features no better than it does on shuffled labels, and 1 when every seed returned the same set. As an illustration of the arithmetic, an arm whose seeds agree at a Jaccard index of 0.40 on real labels and of 0.05 on shuffled ones has a kappa of (0.40 − 0.05) / (1 − 0.05) = 0.37. The table, `results/stability.csv`, has one row per arm and label condition, and the value is written on both of an arm's rows, so a reader who filters the table to one label condition still finds it.

The analysis treats three cases specially:

- **An arm with fewer than two seeds is skipped**, because stability is a property of a set of fits. A [run](../../glossary.md#run) in which every arm is skipped writes no table, and its report says that stability was not measured.
- **Two seeds that both selected nothing score 0**, because they have no feature to agree on ({py:func}`~seqmaestro.harness.stability.jaccard`). An arm whose permuted cells select nothing therefore has a chance level of 0, so its kappa equals its agreement on real labels.
- **Kappa is computed on the top N only.** An explainable boosting machine keeps every feature it is given, so its full sets agree almost perfectly under both label conditions. Dividing by `1 − chance` then turns a small difference into a large one, and the figure is undefined where the chance level reaches 1. Taking the same number of features from every arm removes this artefact.

### The Nogueira estimator

The table also holds `full_phi` and `top_n_phi`, the stability estimator of [Nogueira, Sechidis and Brown](https://jmlr.org/papers/v18/17-514.html), computed on the full sets and on the top N ({py:func}`~seqmaestro.harness.stability.nogueira_phi`). They are kept for diagnosis, as is `full_jaccard`, and a report quotes none of them.

Phi is the established measure of the stability of feature selection, but it needs the size of the set of features that the arm chose from. An arm that chooses from a fixed vocabulary records that size, whereas an arm whose features a [language model](../../glossary.md#language-model) writes has no fixed set to choose from, so the union of the features it found stands in for that size. Phi values computed under the two conventions cannot be compared, whereas kappa needs no such size and means the same for every arm.

## Co-discovery

Co-discovery works on every arm at once, within each label condition and seed:

1. Each arm's features are reduced to the top N.
2. The features of all arms are clustered together, in one graph. Each cluster is a [concept](../../glossary.md#concept): one property, together with the arms that found it.
3. Each concept is scored by the arms in it, and becomes one row of `results/co_discovery.csv`.

Each concept has these counts:

- `n_arms`: how many arms found it.
- `n_witnesses`: how many independent arms found it, with each group of [linked arms](#linked-arms) counted once.
- `n_pools`: how many different [pools](../../glossary.md#pool) the arms that found it were fitted on. This is the stricter count, because three models fitted on one pool are three arms but one representation of the sequence.
- `direction` and `direction_agreement`: the direction that most of the concept's members agree on, and the share of them that hold it ([direction agreement](#direction-agreement), below).

The analysis then pools the concepts of all seeds and summarises each arm in one row of `results/co_discovery_by_arm.csv` ({py:func}`~seqmaestro.pipeline.analyses.co_discovery.per_arm_summary` describes each column). An arm has at most N concepts per seed, one for each of its top N features unless two of them fall in one cluster. Its row gives the share of its concepts that were found by at least 2, 3 or 4 independent arms (the arm itself included). The arm's chance level is the share found by at least 2 arms among the concepts of its permuted cells, and `corroborated_share_corrected` corrects the observed share by it, as kappa does.

**`cross_pool_share` is the column for comparing arms**: the share of an arm's concepts that an arm on a different pool also found. Corroboration by any second arm is a weak criterion, for two reasons. It happens for some concepts even under shuffled labels. It also flatters whichever arm has the most company, since three models fitted on one pool corroborate each other by construction. As an illustration, a concept that holds `pos_50_A` from an arm on `positional` and `central_base_is_a` from an arm on the one-round [written source](../../glossary.md#written-source) was reached from two pools, and counts towards the `cross_pool_share` of both arms. A concept found only by three models on `positional` counts towards none.

### Linked arms

Some arms agree because of how the pipeline fits them, whatever the data hold. The decision tree (`deft_or_cart`) is plain CART on `positional`, and DEFT on `positional+llm_generated_joint`, where a language model proposes extra candidate features at each split. The two arms share the tree, its enumeration of naive candidate splits and, under `grid.permutation_scope: seed`, the shuffled labels of each seed. On shuffled labels the written candidates win few splits, so DEFT's tree is close to a copy of CART's, and the pair agrees more on shuffled labels than on real ones. Subtracting the chance level cannot remove that agreement, because the chance level is measured on shuffled labels, where the pair agrees most. Each such pair of arms therefore counts as one witness. The same holds for the tree on any other pool, with and without the [joint source](../../glossary.md#joint-source) ({py:func}`~seqmaestro.pipeline.analyses.co_discovery.linked_by_fit`).

Arms that took the same cached proposal would be linked as well ({py:func}`~seqmaestro.pipeline.analyses.co_discovery.linked_groups`). Each arm's proposal is made for that arm alone, so in a run of the pipeline the arms do not share one. Arms that only share a pool are left unlinked, because each selects from the common vocabulary independently, and `n_pools` shows that weaker dependence.

### Direction agreement

Finding the same concept is weaker evidence than agreeing about which way it runs. Two arms can both rely on the GC content of one window, with higher GC favouring the positive class in one arm and the negative class in the other, which corroborates nothing. Each concept therefore holds the [direction](direction.md) that most of its members agree on, and the share of them that hold it ({py:func}`~seqmaestro.pipeline.analyses.co_discovery.agreed_direction`):

- **Signs are aligned first.** Features match on the absolute correlation of their values, so a concept can hold a GC fraction from one arm and an AT fraction from another, whose directions are opposite although they measure one property. Each member is compared with the first member, and its direction is flipped where the two are negatively correlated.
- **The comparison uses each model's `direction`**, which says how the model uses the feature. Features that matched already have nearly the same `marginal_effect`, their association with the target on their own, so agreement about it would only restate the match. Each arm's model conditions on different other features, so agreement about how the models use the property is informative.
- **`direction_agreement` is blank for a concept in which only one member has a direction**, since that member has nothing to agree with, and 1.0 would look like unanimity.

**The column is informative only where the models could disagree.** A model can use a feature in a direction other than the feature's own association only if the model has other features to trade it against. On a pool of nearly independent features, such as positional indicators, every arm's direction for a feature follows the feature's association with the target, so all arms agree and a share of 1.0 is expected. The column can show more for concepts found by arms on different pools, and for pools of correlated features.

## Limits of the permuted control

Both corrections assume that a permuted cell differs from its real counterpart only in its labels. That holds for an arm that selects from a fixed vocabulary, because it selects from the same pool whatever the labels.

An arm that writes its features breaks the assumption, because its mix of features depends on the labels. DEFT's arm (`deft_or_cart` on a pool with `llm_generated_joint`) can use far fewer written features on shuffled labels than on real ones, because its written candidates win splits only where there is a signal to find. Its permuted cells are then close to the CART arm's, so the correction subtracts the agreement of a different mix of features. Once like is compared with like, the corrected figure can even change sign. Two consequences follow:

- **A mixed arm's row can be interpreted only after its features are split by provenance.** The features the arm selected from its pool and the features written during its fit (their `provenance`, [the feature record](../feature_record.md)) are corroborated at different rates and have different controls.
- **The corrected figure comes last in each row**, after the observed share and the chance level. It divides by `1 − chance`, so for an arm with a high chance level a small shortfall becomes a large negative number. As an illustration of the arithmetic, an observed share of 0.70 against a chance level of 0.80 gives (0.70 − 0.80) / (1 − 0.80) = −0.50.

**The chance levels rest on few draws.** Under `grid.permutation_scope: seed`, the shared [config](../../glossary.md#config)'s setting, every arm of a seed is fitted on the same shuffled labels. The run then has as many independent draws of shuffled labels as it has seeds, so an unusual draw moves the chance level of every arm at once ([`_shared.yaml`](../../../experiments/configs/_shared.yaml) explains the setting and its alternative, `cell`).

## Caveats

- **Both analyses rest on the top N**, which each arm ranks by its own importance, and importances cannot be compared between models. An arm whose importances rank the wrong features looks unstable and uncorroborated, whatever it found.
- **Every fit must be evaluated on the same reference rows**, since matching compares values row by row. The code refuses to compare fits whose reference rows differ in number.
- **Co-discovery's figure for each arm depends on every other arm.** Concepts are connected components over the features of all arms, so adding or improving one arm changes every arm's row.
- **Co-discovery depends on the clustering more than stability does.** Stability's figure comes from the matches between pairs of seeds, and it uses the clusters only for phi, whereas co-discovery counts clusters. A chain of borderline matches can merge two properties into one concept ([chaining matches into clusters](index.md#chaining-matches-into-clusters)), in which case the arms that found either property count as corroborating each other.
