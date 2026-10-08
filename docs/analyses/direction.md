# Which class a feature argues for

> **Maintained page.** Change it in the same commit as `seqmaestro/harness/direction.py`,
> `rank_biserial` in `harness/effect.py`, or any model's `directions()`.

An importance says **how much** a model leaned on a feature. It does not say **which class** the
feature argues for, which is the part a reader wants first. "Number of upstream start codons" is
plausible in either direction, and only one of the two is a claim anyone can check.

Every feature record carries two answers, because there are two different questions:

| field | question | needs a model? |
|---|---|---|
| `direction` | what does the **fitted model** do with this feature, given everything else it holds? | yes |
| `marginal_effect` | what does the **feature** do in the data, on its own? | no |

**They can point opposite ways, and when they do it is a finding.** A suppressor feature, one with
no marginal association that the model still relies on, is one the model uses to *cancel* something
else. Recording only one of the two answers would get half the interesting cases wrong.

---

## 1. The vocabulary

Three words for each kind of target, which mean the same thing for every arm:

| binary target | continuous target | |
|---|---|---|
| `higher_in_positive` | `increases_target` | the model's score (the probability of the positive class, or the predicted value) rises with the feature |
| `higher_in_negative` | `decreases_target` | it falls |
| `non_monotone` | `non_monotone` | the relationship reverses inside the observed range, so no single direction describes it |

The arithmetic that picks a word is the same for both kinds (below), and only the pair of words
differs, which `seqmaestro.target` holds for each kind.

`non_monotone` is a **real answer, not a missing one**. A feature that helps at one end and hurts at
the other is a different claim from one that does nothing, and reporting either as the other
misleads. `None` means the model had no basis for an answer, or the feature moves the model
nowhere. A flat curve has no direction, so it gets `None` and not a word of its own.

---

## 2. One primitive underneath

A shape function is a curve. A boosted model's dependence on a feature is a curve. A tree's use of a
feature is a set of splits. All three reduce to a sequence of **signed increments**, and one function
(`from_increments` in `seqmaestro/harness/direction.py`) turns any of them into a word:

```
net       = sum(increments)
travelled = sum(|increments|)

|net| / travelled  <  MONOTONICITY_FLOOR   ->  non_monotone
otherwise, the sign of net
```

`|net| / travelled` is 1 exactly when the curve never doubles back. The floor is **0.5**, which means
the curve must move at least three times as far one way as the other. That is a stated convention,
like `DEFAULT_TOP_N`, and not a measured quantity. At 1.0 only a perfectly monotone curve would
qualify, and no fitted shape function is one. At 0 nothing would ever be `non_monotone`.

**Adding a model means producing increments**, and the vocabulary stays as it is.

---

## 3. How each model answers

The underlying quantities are no more comparable across models than the importances are, but
**their signs are**. That is why the file names the basis in `direction_kind` and the record carries
only the word.

| model | `direction_kind` | how | cost |
|---|---|---|---|
| `l1_logistic`, `l1_linear` | `l1_coefficient_sign` | the sign of the selected coefficient | free |
| `ebm` | `shape_function_trend` | the term's curve across its bins, in log-odds for a binary target and in the target's units for a continuous one | free |
| `xgboost` | `treeshap_dependence` | TreeSHAP contributions, binned by the feature, read as a curve | one prediction pass |
| `deft_or_cart` | `tree_split_rates`, or `tree_split_means` for a continuous target | class rate, or the target's mean, either side of each split, weighted by rows reaching it | free |
| `igsr` | `least_squares_coefficient_sign` | the sign of the term's weight in the fit its search made (least squares, plain, ridge or lasso); a continuous target only | free |
| `pysr` | `shapley_dependence` | exact Shapley values of the equation's output, binned by the feature, read as a curve | one Shapley computation on up to 2,000 rows |

**The sparse linear model is exact and complete, and needs no rows.** The fit is monotone in every feature, so
one sign describes the whole of what it does with one. `non_monotone` is unreachable there.

**`ebm` is the best case, because of `interactions=0`.** The model *is* the sum of its shape
functions, so the curve is the whole of what the model does with a feature. Its `term_scores_` pad
each curve with a leading `missing` bin and a trailing `unknown` bin. Both sit outside the observed
range, so counting them as the ends would read a step into a bin no row is in. Terms over more than
one feature are skipped, because a pairwise term's shape belongs to the pair.

**`xgboost` has no signed number in the fitted object at all**, since gain is unsigned. TreeSHAP
supplies one: every row's signed contribution from every feature, in log-odds for a binary target
and in the target's units for a continuous one, computed *exactly* and not by sampling. It ships inside XGBoost, so it costs no new dependency. Grouping those
contributions into **quantile bins of the feature** turns them into a curve, read the same way an
additive model's is. The bins are quantiles so that every point is backed by rows that exist, and
the curve never asks the model about a value the data does not contain. That is where a
partial-dependence sweep goes wrong on features as correlated as these. A row's bin follows from
the share of rows whose value is below its own, so the bins hold about as many rows each and rows
that share a value share a bin: a feature that is zero on most rows gives one point for its zeros
and the rest for the rows above them. Averaging the contributions
instead would return near zero for a feature that helps at one end and hurts at the other, and zero
is also what a feature the model ignores returns.

**`pysr`'s equation can combine features in any way its operators allow**, so no single number in
it belongs to one feature. Its Shapley values divide each row's output among the features the
equation names: exactly, by enumerating every subset of them, and interventionally, against 100
fitting rows as the background. A feature's values are then binned and read as a curve, as
XGBoost's contributions are. The part of a product that belongs to neither of its features alone,
their interaction, is split between the two, half each, so each feature's curve carries half of it.

**The vendored tree** has neither coefficient nor shape function, so the direction comes from what a
split *does*: rows above the threshold carry a higher share of the positive class, or a lower one,
and for a continuous target a higher mean of it, or a lower one.
Each node contributes one signed step weighted by the rows reaching it, so a split near the root
counts for more than a conditional one deep in a branch. A feature the tree uses **one way at the
root and the other way inside a branch** is `non_monotone`, by the same rule every other arm's curve
is read by. Left is `<= threshold`, matching how the rows are routed.

> The adapter runs where `seqmaestro` cannot be imported, so it restates the rule. A test holds the
> two implementations together, including the floor and the three words, because a restatement
> that nothing checks is a second definition that will eventually disagree.

---

## 4. The marginal effect

A rank correlation of each feature with the target, by the target's kind (`association` in
`seqmaestro.target`):

- **Binary**: `rank_biserial`, `2 × AUC − 1` per feature, where AUC is that feature's column used
  alone as a score for the positive class. Range −1 to 1, **positive meaning higher in the positive
  class**, whichever way round the labels happen to be encoded. It has a direct meaning, which is
  why it is the number that goes in a report: *0.4 means that of one row drawn from each class, the
  positive one holds the higher value 70% of the time.*
- **Continuous**: `rank_correlation`, Spearman's rho per feature with average ranks for ties. Range
  −1 to 1, **positive meaning higher values of the feature go with a higher target**. A heavy-tailed
  target cannot let a few rows decide it, since only ranks enter.

The cell's `run.json` names the statistic (`marginal_effect.statistic`).

**Ranks, not means.** A count and a fraction of the same thing over the same window are the same
feature under a monotone map, and only a rank statistic scores them alike. The behavioural matcher
uses Spearman correlation for the same reason. Ranks also make the number insensitive to class
imbalance, which a difference of means is not.

**Computed in one place.** `score_arm` is the only writer, on either path into a run directory. It
needs no model, only the effect values and their labels, which every arm hands over there. So the
number means the same thing for an arm fitted in this process and for one that arrived from another
environment, and the vendored adapter supplies a matrix and not a statistic.

A column carrying any non-finite value scores NaN and is not imputed. The matrix should not contain
one, and repairing it here would hide that it did.

### Measured on the training rows, and in sample

`marginal_effect` is measured on **every training row against the real labels**, whatever this cell
was fitted on. `EFFECT_SPLIT` in `seqmaestro/harness/runner.py` is the constant.

**It is an in-sample number by design.** It describes the association present in the data the model
learned from. Whether that association survives to held-out rows is a different question, which this
field does not answer and the test-split metrics do.

**Why not validation.** A validation split can be too small to carry a sign. Split the validation
rows in half and compute the statistic on each half: at small |effect| (below about 0.05) on a small
split, the two halves can disagree about the sign about as often as they agree. A disagreement with
`direction` in that range then describes the estimator and not the feature. The training split is
several times larger.

**Why not the fitted rows, though those are what the model saw.** `train_fraction` subsamples them
per seed, and a control arm's fitted rows carry the shuffled label vector. Measuring against those
would make `marginal_effect` and `direction` agree under shuffled labels for the same reason they
agree under real ones, and the permuted control for that comparison would stop saying anything.
Real labels on every training row give one number per feature per dataset, identical in the real
and control arms of a seed, which leaves `direction` as the only thing that differs between them.

---

## 5. Where it is used

**Reports.** Both are in every cell's `features.json`, which is where a run's report reads them, and
the spreadsheet carries both plus glossary entries. Where the two agree, one sentence covers both.
Where they disagree, the disagreement is the finding.

**The judge.** It is shown the marginal direction, sign only. The full reasoning is in
[`judge.md`](judge.md). In short, an effect size would let the judge separate real features from
permuted-label ones, where a run judges those too, and a sign is a coin flip for such a feature.

**Co-discovery.** Two arms finding the same concept *pointing the same way* is stronger corroboration
than finding it at all. Signs are aligned first, because the matcher uses absolute correlation. See
[`stability_and_co_discovery.md`](stability_and_co_discovery.md).

---

## 6. What it cannot support

**Correlated columns.** Reading `pos_50_A`'s direction while `pos_50_C/G/T` are in the model has the
same problem permutation importance has: between them the others largely determine what was
measured. This is a per-feature record, and there is no per-*position* row to put a joint answer in.

**A model fitted on a confounded dataset points along the confound.** `direction` is the model's
direction, not nature's.

**One number is noisy**, even on the training rows. `marginal_effect` carries no multiplicity
correction, because a per-feature p over a whole grid is a claim about the analysis more than about
the feature. The evidence that a direction is real is that it **recurs**: same sign, same concept,
across seeds and arms.

**It is in sample, so it is not evidence of generalisation.** A feature can separate the training
rows cleanly and do nothing on test, and nothing in this field can tell the two apart.

**How often the two answers agree is not a clean instrument.** Both are estimates, and each becomes
reliable at a different rate: `marginal_effect`'s sign settles as its row count grows, and
`direction`'s settles as the model comes to rely on the feature. At small |effect| the second can be
unreliable however many rows the first has, so even on a split large enough for `marginal_effect`'s
sign to be stable, the two can agree little better than chance below |effect| = 0.05. A low
agreement rate at small effects therefore describes the pair of estimators and not the features,
and no single threshold holds across datasets. The half-split check above is how to find one for a
given run.

**Do not read agreement as a label-driven signal without its control.** The shuffled-label arms are
what make it readable: they share this field with their real counterparts and differ only in
`direction`, so a gap between the two conditions is about `direction` alone.
