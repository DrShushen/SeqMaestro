# Measuring a feature directly, with no model in the way

> **Maintained page.** Change it in the same commit as `direct_effect_test.py`,
> `feature_group_dependence.py` or `seqmaestro/harness/effect.py`.

Two scripts, both answering a question about **one feature** and not about a fitted model. They are
the ones to reach for when a collaborator points at a row in the report and asks "is that real?" or
"isn't that just the batch effect?".

| | asks |
|---|---|
| `direct_effect_test.py` | does *this* feature differ between the classes, by how much, and could a difference that size have arisen by chance? |
| `feature_group_dependence.py` | how much of *this* feature is just the thing the split groups on? |

Both are run **on demand** and are not part of the standard pipeline, because each needs an argument
nobody can guess: which features, and which grouping variable.

**The direct effect test is a contrast between two classes**, so it refuses a label column holding
any other number of values. For a continuous target, a feature's association on its own is its
`marginal_effect` (Spearman's rho, [`direction.md`](direction.md)) and, calibrated against permuted
targets, Tier 2 of the reliability panel ([`../reliability.md`](../reliability.md)).
`feature_group_dependence.py` does not look at the target and serves either kind.

---

## Part 1: the direct effect test

### Why it exists beside the models

A fitted model's score is a **joint** statement: these features, together, predict the label this
well. It cannot be attributed to any one input. An arm can also report a feature because it helps
prediction in combination with others while its own class difference is nowhere near separable from
noise. That is a different claim from "this feature differs between the classes".

So this script answers the narrower question, which is usually the one that was actually asked.

### It runs the arm's own code

Features are **compiled out of the run's own `features.json`**, not reimplemented, so what gets
tested is what the arm reported, down to the function body it wrote. Several files can be pooled,
because each of an arm's seeds reports its own set and a feature worth a direct test usually turned
up in more than one. Pooled features are deduplicated by name, keeping the most important copy.

The sequence column, and any covariates the code reads, are taken **from the files** and not
assumed. Generated code indexes those columns by name, so guessing one wrong is a crash at best and
a silently different feature at worst. If two files disagree, the script refuses to pick one, and
where no file records the sequence column it asks for `--sequence-column`. A file records them
where its cell's featuriser is the proposer's: a pool of `llm_generated` alone, or one holding
`llm_generated_iterative`. For `llm_generated` beside other sources, and for a DEFT or an IGSR
cell, `--sequence-column` is needed and the code is handed no covariates, so a feature that uses
one is dropped.

`--features` tests the records that carry code. A positional indicator, a k-mer count or a tree's
naive split has none and is dropped as uncompilable, so such a feature is tested as a column of
the table, with `--columns`. The script needs `--dataset`, `--label-column`, and `--features` or
`--columns`.

Fitting the featuriser on the same rows it reports on is right here, and would be wrong in a model,
because nothing is being trained: the only thing `fit` decides is which features compile and vary at
all. Constant features are dropped, because a feature taking one value has no class difference to
measure.

### Stratifying on a confounder

The case worth measuring is a dataset with a confounder in it: a batch, a donor, a gene, whatever
the split already groups on. `--group-column` names it. The analysis has three pieces, kept separate
because they are separate decisions:

**`match_within_strata`** equalises the classes inside every stratum by subsampling the abundant
one, so stratum identity carries no information about the label. The fixed ratio is what makes the
stratum uninformative: whatever the abundant class is overall, every stratum now says the same thing
about the label.

**`stratified_permutation_p`** shuffles the labels *within* each stratum, so the null has the same
structure as the design and the confounder cannot contribute to it either.

**`direct_effect`** runs the whole feature set against one label vector and returns the table.

**Matching and permuting do different jobs, and both are needed.** Matching removes the association
from the *estimate*, and permuting within strata removes it from the *null*. Doing only the first
leaves a p-value that a stratum imbalance can inflate. Doing only the second leaves an estimate that
stratum composition still moves. `--no-match` does only the second, keeping every row in the
estimate.

Matching drops a stratum that holds only one class. It offers no contrast, so it says nothing about
the label, and leaving it in would only put the association back. Under `--no-match` such a
stratum's rows stay in both class means, and a shuffle within the stratum leaves them as they are.

### What comes out

One row per feature: `mean_positive`, `mean_negative`, `difference`, `ratio` with an interval,
`p_permutation`.

- **The statistic is the difference of class means**, which reads as a difference of *rates* when
  the feature is binary. That is the column to compare across features.
- **Ratios are reported because that is how an enrichment is quoted**, but with an interval only
  where the delta method applies: a binary feature, where each class mean is a proportion and its
  variance follows from the count. A continuous feature has a ratio of means with no such variance,
  and one taking both signs has a ratio that swings with how close the denominator sits to zero and
  reverses through it. A feature that is not binary gets a ratio with no interval where the
  negative class's mean is above zero and the positive class's is not below it, and no ratio
  otherwise, since it would carry a sign and no magnitude.
- **The p-value is `(hits + 1) / (draws + 1)`**, not `hits / draws`. The observed arrangement is one
  of the arrangements, and the correction stops a p of exactly zero, which no permutation test can
  support.

### The matched variant prices its own draw

`matched_effect` matches, contrasts, and then **re-draws the match** to say what the draw itself was
worth.

A matched estimate is a statistic of a random subsample, and both its interval and its p-value
condition on the subsample that was drawn. Neither says anything about the next draw. For a common
feature that hardly matters. For a rare one the abundant class is sampled down hard, and the
estimate moves with the seed by an amount that can rival the stated interval.

So the estimate, its interval and its p-value all come from **one** matched sample, which makes them
an internally consistent analysis of one design. `ratio_draw_min`, `ratio_draw_max` and
`difference_draw_sd` report how far the same quantity moved over further draws. Read them as
imprecision that the interval does not include. Where they are wide, matching is costing more than
it is buying, and a weighted design is the better instrument.

Averaging the draws would give a lower-variance estimator, but it is not done, because its interval
and its p-value would then describe a quantity nothing else in the table is about.

### `rank_biserial`, the one that lives in the record

The same contrast reduced to one scale-free number, `2 × AUC − 1`. The harness records it per
feature as `marginal_effect` for a binary target, and `rank_correlation` of the same module
(Spearman's rho) for a continuous one, because each is cheap, needs no model, and means the same
thing for every arm. It is unstratified and carries no null of its own. A
per-feature p over a whole grid is a claim about the analysis more than about the feature, and the
stratified functions above are where that question belongs. See [`direction.md`](direction.md).

---

## Part 2: feature-group dependence

### The question

A grouped split promises that a model cannot succeed by memorising the grouping variable. That
promise is about **rows**. It says nothing about whether the *features* encode the grouping variable
anyway, and if they do, a model that looks like it learned biology may have learned which donor,
which V gene or which batch a row came from.

`feature_group_dependence.py` measures that directly. For each feature it computes **η²
(eta-squared)**, the share of the feature's variance explained by the grouping variable.

```
eta_squared = between-group variance / total variance
```

0 means the feature says nothing about which group a row is in. 1 means the feature *is* the group.

### What it reports

Per pool: `n`, median, p90, max, and the share of features above 0.25 and above 0.5. Whole
fixed pools can be scored (`--pools positional kmer`), and so can a run's generated features, by
pointing `--features` at one or more `features.json`. Those are compiled and evaluated the same way
`direct_effect_test.py` does it, and every record of a file is scored. The script needs
`--dataset`, `--sequence-column` and `--group-column`, and at least one of `--pools` and
`--features`.

For a small set it prints every feature's η² individually, which makes it usable as a check on a
specific arm as well as a summary.

A fixed pool is featurised as its arm featurises it. Where the sequences differ in length, a pool
holding `positional` needs the positional window the dataset declares, given as it is declared
(`--positional-window 100 start no-pad`), and is refused without one. A pool holding `kmer` needs
how the run counts k-mers, as its config states it under `sources.kmer`: `--k-range 1 6` and
`--kmer-normalise counts`, or `per_length`, which counts per k-mer position. Neither has a value
here to fall back on. Counts carry a sequence's length, so on a table of several lengths a count
can depend on a group through its lengths alone.

### Why before a grid and not after

It gives **one answer per pool, not one per split**. If a pool's features largely encode
the grouping variable, no split of the rows fixes that, and a full grid spent on that pool produces
numbers that mean something other than what the report will say they mean. Running it first costs
one pass over the data.

---

## What neither of these can support

**Neither is a model, and neither should be read as one.** A feature with a large direct effect may
add nothing to a model that already holds three correlated features, and a feature with none may be
essential to one, as a suppressor. That is the gap between `marginal_effect` and `direction` in the
feature record, and it is a finding when they disagree.

**Both depend on what an arm reported.** They test the features that were written down. A property
no arm proposed is invisible to them.

**A filter has to be measured against the variable it names.** A threshold sweep can support a
filter for the wrong reason. If a column believed to count reads in fact counts molecules, a
threshold on it removes whatever the molecule count tracks (in a clonal design, pseudo-replicated
members of expanded clones). When the two variables are correlated, both explanations predict the
same sweep and nothing in it can separate them. Check what a column *is* before deciding what
stratifying or filtering on it means.
