# Direct measurement

> This page tracks [`direct_effect_test.py`](../../../experiments/analysis/direct_effect_test.py), which runs {py:mod}`seqmaestro.pipeline.analyses.direct_effect` with the statistics of {py:mod}`seqmaestro.harness.effect`, and [`feature_group_dependence.py`](../../../experiments/analysis/feature_group_dependence.py), which runs {py:mod}`seqmaestro.qc.group_dependence`. Change it in the same commit as any of them.

Two scripts measure what one [feature](../../glossary.md#feature) does in the data, with no fitted model involved. They answer the questions that a row of a report raises about a single feature, such as "is that real?" or "is that just the batch effect?":

- `direct_effect_test.py` tests whether a feature differs between the two classes, by how much, and whether a difference of that size could have arisen by chance.
- `feature_group_dependence.py` measures how much of a feature is the variable that the split groups on.

Both are run by hand, outside the [driver](../../glossary.md#driver), because each needs arguments that only the caller can choose: which features to test, and which variable to group on. Each script's `--help` lists its arguments.

**The direct effect test needs a binary target**, because it contrasts two classes, and it refuses a label column with any other number of values. For a continuous target, a feature's association on its own is its marginal effect, Spearman's rho ([direction](direction.md#the-marginal-effect)), and Tier 2 of the [reliability panel](../../glossary.md#reliability-panel) tests that association against permuted targets on the [held-out split](../../glossary.md#held-out-split) ([Stage 3B: the reliability panel](../reliability.md)). `feature_group_dependence.py` does not use the target, so it serves either kind.

## The direct effect test

### The question

A fitted model's score is a joint statement: these features, together, predict the label this well. The score cannot be attributed to any one feature. An [arm](../../glossary.md#arm) can also report a feature because it helps prediction in combination with others, although the feature's own difference between the classes cannot be told apart from noise. The direct effect test answers the narrower question, which is usually the one a collaborator asks: does this feature differ between the classes?

### The features tested

The test computes each feature from the code that its [cell](../../glossary.md#cell) recorded in `features.json`, so it tests the feature the arm reported, down to the function a [language model](../../glossary.md#language-model) wrote. It can pool the files of several cells, since each [seed](../../glossary.md#seed) of an arm reports its own set. A name that appears in several files is tested once, in its most important copy ({py:func}`~seqmaestro.pipeline.analyses.direct_effect.gather_records`), and each file contributes its [top N](index.md#the-top-n) by default.

- **Only a feature with code can be computed.** A positional indicator, a k-mer count or a tree's naive split has no code, so `--features` drops it. Such a feature can be tested as a column of the table, with `--columns`, once the table holds it.
- **The sequence column and the [covariates](../../glossary.md#covariate) are taken from the files.** Written code indexes them by name, so a wrong guess would crash at best and compute a different feature at worst. Only a cell whose featuriser is the [proposer](../../glossary.md#proposer)'s records them: a [pool](../../glossary.md#pool) of `llm_generated` alone, or a pool that includes `llm_generated_iterative`. For any other cell (`llm_generated` beside other sources, or a DEFT or an IGSR cell), `--sequence-column` names the sequence column, and a feature whose code uses a covariate cannot be computed and is dropped. Files that disagree about either are refused.

The featuriser is fitted on the rows it reports on. That would be wrong for a model, but nothing is trained here: fitting only decides which features compile and vary. A constant feature has no difference between the classes to measure, so it is dropped.

### Stratifying on a confounder

The case that matters most is a dataset with a confounder, such as the batch or the donor of each row, which is usually the variable that the split already groups on. `--group-column` names it, and each of its levels is a stratum. The test then guards against the confounder in two ways:

1. **Matching** subsamples the more abundant class inside every stratum until the two classes are equal there ({py:func}`~seqmaestro.harness.effect.match_within_strata`). Every stratum then says the same about the label, so a row's stratum cannot predict its label.
2. **Permuting within strata** builds the null distribution by shuffling the labels inside each stratum ({py:func}`~seqmaestro.harness.effect.stratified_permutation_p`), so that the null has the same structure as the design.

Both are needed, because they remove the confounder from different places. Matching removes its association with the label from the estimate, whereas permuting within strata removes it from the null distribution. With matching alone, differences between the strata can inflate the p-value. With permuting alone, the composition of the strata still moves the estimate. `--no-match` permutes within strata and keeps every row in the estimate.

Matching drops a stratum that holds only one class. Such a stratum offers no contrast between the classes, and keeping its rows would restore the association between stratum and label that matching removes. Under `--no-match` the stratum stays: its rows count towards the mean of their class, and a shuffle within the stratum leaves their labels unchanged.

### The output

The test prints one row per feature, and `--out` writes the table as CSV, with the design beside it as JSON ({py:func}`~seqmaestro.harness.effect.direct_effect` names the columns). Three choices shape the table:

- **The difference of the class means is the statistic to compare across features.** For a binary feature, it is a difference of rates.
- **The ratio of the class means is reported because an enrichment is quoted that way.** It has an interval only for a binary feature, whose class means are proportions with a variance that follows from the counts (the [delta method](https://en.wikipedia.org/wiki/Delta_method)). A feature that is not binary gets a ratio without an interval, and only where neither class mean is negative and the negative class's mean is above zero. Otherwise the ratio would swing with a denominator near zero and reverse its sign through it, giving a sign without a magnitude.
- **The p-value is `(hits + 1) / (draws + 1)`**, which counts the observed arrangement as one of the arrangements, because a permutation test cannot support a p-value of zero. It is two-sided and not corrected for multiplicity, because the caller chose the few features tested ({py:mod}`seqmaestro.pipeline.analyses.direct_effect` says when to correct).

### The spread of a matched estimate

A matched estimate is a statistic of one random subsample, so its interval and its p-value describe that subsample and say nothing about the next draw. For a common feature this hardly matters. For a feature that is non-zero on few rows, which of those rows the subsampling keeps decides much of the estimate, so where the abundant class is heavily subsampled, the estimate can move with the draw by as much as its interval.

The test therefore draws the match again (`--match-repeats` times) and reports how far the ratio and the difference moved (`ratio_draw_min`, `ratio_draw_max` and `difference_draw_sd`). That spread is imprecision which the interval does not include. Where it is wide, matching loses more than it gains, and a weighted design is the better instrument. The estimate, its interval and its p-value all come from the first draw, so that they describe one design, and the draws are not averaged ({py:func}`~seqmaestro.harness.effect.matched_effect` gives the reason).

### Relation to the marginal effect

The harness records a simpler measure of the same contrast for every feature of every cell: its marginal effect, `2 × AUC − 1` for a binary target ({py:func}`~seqmaestro.harness.effect.rank_biserial`) and Spearman's rho for a continuous one ({py:func}`~seqmaestro.harness.effect.rank_correlation`). It needs no model and means the same for every arm. It is not stratified, though, and it has no p-value, since a p-value for every feature of a whole grid would say more about the analysis than about any one feature. A question of significance about one feature is for the direct effect test. [Direction](direction.md#the-marginal-effect) explains the marginal effect.

## Feature-group dependence

### The question

A grouped split keeps every row of a group on one side of the split, so a model cannot succeed by recognising a group it saw in training. That guarantee is about rows. The features can still encode the grouping variable, and where they do, a model that seems to have learned biology may have learned which donor, gene or batch a row came from.

For each feature, `feature_group_dependence.py` measures η² ([eta squared](https://en.wikipedia.org/wiki/Effect_size#Eta-squared_%28%CE%B72%29)), the share of the feature's variance that the grouping variable explains:

```
eta_squared = between-group sum of squares / total sum of squares
```

A value near 0 means that the feature varies within groups, so it says little about which group a row is in. A value of 1 means that the feature is constant within each group, so it is a property of the group alone. A high value therefore says that the feature's information is at the level of the group. Where the differences between groups are the biology, such a feature measures biology, but a split grouped on that variable cannot be trusted to have isolated it ({py:mod}`seqmaestro.qc.group_dependence`).

The check runs before any arm is fitted, because its answer is one per pool. If a pool's features largely encode the grouping variable, no split of the rows fixes that, and every arm on that pool would produce numbers that mean something other than what the report says they mean. The check takes one pass over the data.

### What it scores

It scores the pools of positions and k-mers (`--pools`), whose features it computes as the pool's arm would. It also scores the written features of chosen `features.json` files (`--features`), every record of each file, computed from their code as the direct effect test computes them. It prints a summary of η² for each pool or file, with every feature's value where a pool or file has few features, and `--out` writes the value of every feature.

**The pools are computed with the [run](../../glossary.md#run)'s settings**, which have no defaults here:

- A pool with `kmer` needs the run's k-mer settings, `--k-range` and `--kmer-normalise`, as the [config](../../glossary.md#config) states them under `sources.kmer`. A count grows with a sequence's length, so on sequences of several lengths a count made differently from the run's could depend on a group through its lengths alone.
- A pool with `positional` needs the positional window that the dataset declares (`--positional-window`) where the sequences differ in length, because the table may not have a `dataset.json` yet.

## Limits

- **Both scripts measure a feature on its own.** A feature with a large direct effect may add nothing to a model that already holds three correlated features, and a feature with none may be essential to one, as a [suppressor](https://en.wikipedia.org/wiki/Suppressor_variable). That is the gap between `marginal_effect` and `direction` in the [feature record](../feature_record.md), and a disagreement between them is a finding.
- **They measure only the features they are given.** A property that no arm proposed, and that no column of the table holds, is invisible to them.
