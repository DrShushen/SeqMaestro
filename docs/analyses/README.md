# How each post-run analysis works

> **Maintained pages.** If you change what an analysis computes, change its page here in the same
> commit. `experiments/analysis/README.md` says *which* script to reach for and when; these pages
> say *how the number is arrived at* and what it can and cannot support.

| page | covers |
|---|---|
| [`stability_and_co_discovery.md`](stability_and_co_discovery.md) | `analyse_stability.py`, `analyse_co_discovery.py`: does one arm find the same features again, and do several arms find the same thing? |
| [`recurrence.md`](recurrence.md) | `analyse_recurrence.py`: does an arm arrive at the same *ideas* again when it renames them every time, and do different arms arrive at the same idea? |
| [`judge.md`](judge.md) | `run_judge.py`: how biologically significant would each feature be, if it were true? |
| [`direct_measurement.md`](direct_measurement.md) | `direct_effect_test.py`, `feature_group_dependence.py`: what does one feature do, with no model in the way? |
| [`direction.md`](direction.md) | which class a feature argues for, in the two senses of that |

What a feature record holds, and what every one of these reads: [`../feature_record.md`](../feature_record.md).

---

## Comparing features by their values

Arms do not share a vocabulary. The fixed-vocabulary featuriser calls a feature `pos_50_A`, the
vendored tree calls the same thing `pos_50_is_A`, and a language-model proposer writes
`central_base_is_a`. That is three names and three implementations of one property.

So **nothing here compares names**. Every arm evaluates its reported features on the *same reference
rows* (the validation split) and writes that matrix to `feature_values.npz`. Features are compared
on those columns of numbers, which means on what they compute.

Stability and co-discovery build on that matrix. Recurrence does not, and
[`recurrence.md`](recurrence.md) explains why behaviour cannot answer its question.

### Matching two feature sets

`match_pair` in `seqmaestro/harness/stability.py` produces a **one-to-one** correspondence between
two sets, in two passes.

**Pass 1: by definition.** Features whose `definition_digest` is identical are paired before any
statistic runs. The digest hashes normalised source (parsed, with names, formatting, comments and
docstring stripped) or `{kind, params}` where there is no source, so it ignores naming.

This pass exists for the feature no statistic can match. A feature that is **constant** on the
reference split has no rank variation, scores 0 against everything including its own counterpart in
another run, and would otherwise become a fresh singleton in every run it appears in.

**Pass 2: by behaviour.** Everything still unpaired goes to the `BehaviouralMatcher`: the
**absolute Spearman rank correlation** of the two value columns.

- *Rank* correlation, not Pearson, because a count and a fraction of the same thing over the same
  window are the same property under a monotone map and should match perfectly.
- *Absolute*, because a feature and its negation carry identical information, and which sign a model
  happened to learn is not a difference in what was discovered. **This has a consequence that is
  easy to miss**: see the sign-alignment note in
  [`stability_and_co_discovery.md`](stability_and_co_discovery.md).

Pairing is by **Hungarian assignment** maximising total similarity. Greedy nearest-neighbour pairing
would let one strong match consume a feature that was the only plausible partner for another, and
the answer would then depend on the order features happen to appear in. After assignment a pair
counts only if its similarity reaches the threshold, **0.8**.

Two data details decide edge cases:

- Non-finite entries are imputed to the column median before ranking. A generated feature can be
  undefined on some rows (`CpG_cluster_span` is NaN for a sequence with no CpG cluster), and a NaN
  would otherwise propagate across the whole column.
- A zero-variance column becomes zeros and scores 0 against everything. The stability analysis
  prints, per arm, how many such features each seed has among its top 20 (`constant_features`).

### Chaining matches into clusters

`consensus_labels` puts every feature of every run into one graph, with features as nodes and
matched pairs as edges. A **cluster** is a connected component.

This is order-independent, unlike growing a consensus set run by run. But it is **transitive**,
which is its known weakness: a chain of borderline matches can merge two different properties. The
symptom is a cluster holding two features from the same run. The stability analysis prints a line
for an arm where that happens (`runs_with_chained_clusters`, the number of seeds it happens in,
counted on each seed's top 20), so read its log before trusting a surprising cluster.

---

## The top-N cut, and why every analysis takes one

Models disagree by orders of magnitude about how many features they "use". An Explainable Boosting
Machine fits a shape function to every input and so selects all of them, while a shallow tree names
a handful. Comparing a 400-feature set against a 15-feature one measures set size, not agreement.

So every analysis of the standard pipeline reports on each cell's **top 20 by the model's
importance** (`analyses.top_n` in the run's config, 20 in the shared one), and
`direct_effect_test.py` takes the top 20 of each file it is given (`--top-n`). That is a stated convention, not a fact about the data. Within an arm the
importance ordering is meaningful, and across arms only the rank is, so the cut is by rank and not
by a threshold.

## The permuted-label control, and why every number is read against it

Half of every grid is arms fitted on **shuffled training labels**. Every cell is scored against the
real validation and test labels, so a permuted arm should score at chance. A permuted arm that also
chooses on the validation rows, as the iterative source's rounds and IGSR's search do, chooses against
shuffled validation labels too (`labels_to_choose_on`), so nothing it chooses reads the real target.
More importantly, the control shows what each analysis produces *when there is nothing to find*.

Chance agreement differs by more than an order of magnitude between arms. A fixed vocabulary has a
chance Jaccard near zero. A language-model proposer draws from a concentrated prior and agrees with
itself substantially on shuffled labels, because it proposes similar features whatever the labels
say. So a proposer arm can have the **highest raw agreement of any arm** and rank in the bottom half
once its own baseline is subtracted. Any figure that ignores the baseline flatters generative arms.

Hence the correction that recurs across these pages:

```
kappa = (observed - chance) / (1 - chance)
```

0 means the arm does no better than it does on noise, and 1 means perfect agreement. It is the
column to quote whenever one exists.

**Where the control is weak.** A permuted-label proposer arm still reads the task
description, so it proposes a sensible repertoire regardless of the labels. Its features can carry
real marginal association with the outcome even though no label information reached the proposal.
The control is strong for arms that *select* from a fixed pool and weaker for arms that *generate*.
Say which you are relying on.
