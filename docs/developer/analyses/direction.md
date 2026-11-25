# Direction

> This page tracks {py:mod}`seqmaestro.harness.direction`, the statistics of the marginal effect in {py:mod}`seqmaestro.harness.effect` (`rank_biserial` and `rank_correlation`), the `directions()` method of every model, and their counterparts in the adapters of DEFT and IGSR. Change it in the same commit as any of them.

A [feature](../../glossary.md#feature)'s importance says how much a model relied on it, whereas its direction says which way it pushes the prediction: for a binary target, which class higher values of the feature favour, and for a continuous target, whether higher values raise or lower the predicted value. The direction is often what a reader of a report wants to know first. "Number of upstream start codons" is plausible in either direction, and only a stated direction makes it a claim that someone can check.

Every [feature record](../../glossary.md#feature-record) holds two directions, because there are two questions:

| Field | Question | Needs a model |
|---|---|---|
| `direction` | Which way does the fitted model use the feature, given the other features it holds? | Yes |
| `marginal_effect` | How is the feature associated with the target on its own? | No |

**The two can disagree, and a disagreement is a finding.** A [suppressor](https://en.wikipedia.org/wiki/Suppressor_variable) has little association with the target on its own, yet the model relies on it to cancel part of another feature's effect, so its direction in the model can differ from its association in the data. A record that held only one of the two would hide such a feature.

## The words

Every direction is one of five words, which mean the same for every [arm](../../glossary.md#arm) ({py:mod}`seqmaestro.harness.direction`). For a binary target, `higher_in_positive` and `higher_in_negative` say that the model's score (the probability of the positive class) rises or falls as the feature rises. For a continuous target, `increases_target` and `decreases_target` say the same of the predicted value. Both kinds of target use `non_monotone` for a relationship that reverses inside the observed range. Only the pair of words differs between the kinds ({py:mod}`seqmaestro.target`), and one rule chooses the word for both.

**`non_monotone` is an answer.** A feature that raises the score at one end of its range and lowers it at the other makes a different claim from a feature that does nothing, and reporting either as the other would mislead. A missing direction (`None`) means that the model had no basis for an answer, or that the feature does not move the model at all, since a flat curve has no direction.

## One rule for every model

A model uses a feature through a coefficient, a curve or a set of splits, and each of these reduces to a sequence of signed steps. One function turns any such sequence into a word ({py:func}`~seqmaestro.harness.direction.from_increments`). The direction is the sign of the net movement, unless the net movement is less than half of the total distance travelled, in which case the feature is `non_monotone`:

```
|sum(steps)| / sum(|steps|)  <  0.5   ->  non_monotone
otherwise                             ->  the sign of sum(steps)
```

The ratio is 1 when the curve never doubles back. At the floor of 0.5, the curve must move at least three times as far one way as the other. The floor (`MONOTONICITY_FLOOR`) is a stated convention, as N is ([the top N](index.md#the-top-n)): at 1.0 only a perfectly monotone curve would get a direction, and a fitted curve rarely is one, whereas at 0 no feature would ever be `non_monotone`.

A new model therefore needs only to produce its steps, and the words and the rule stay as they are.

## Each model's steps

Each model takes its steps from what it offers, such as the sign of a coefficient, the curve of an additive term or the splits of a tree, and each [cell](../../glossary.md#cell) names the basis once, in its `features.json` (`direction_kind`). {py:mod}`seqmaestro.harness.direction` describes the basis of each model. The quantities behind the steps are on scales of their own, as importances are, so a record holds only the word.

Two points about the bases are easy to miss:

- **XGBoost and PySR need rows to answer**, because their curves are built from per-row attributions ({py:func}`~seqmaestro.harness.direction.dependence_curve`). The harness gives them the validation rows, the same reference rows on which features are compared (`REFERENCE_SPLIT` in {py:mod}`seqmaestro.harness.runner`). The decision tree's steps, by contrast, come from the rows it was fitted on.
- **DEFT's adapter implements the rule a second time.** The adapter runs where `seqmaestro` cannot be imported, so [`deft_bundle.py`](../../../experiments/adapters/deft_bundle.py) restates the words, the floor and the rule, and `tests/test_deft_adapter.py` fails if its copy and {py:func}`~seqmaestro.harness.direction.from_increments` differ.

## The marginal effect

The marginal effect is a rank correlation of each feature with the target, measured without a model, by the statistic of the target's kind ({py:mod}`seqmaestro.target`):

- **For a binary target**, the [rank-biserial correlation](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Rank-biserial_correlation) `2 × AUC − 1`, where AUC is the area under the ROC curve of the feature's column used alone as a score for the positive class ({py:func}`~seqmaestro.harness.effect.rank_biserial`). It runs from −1 to 1, and a positive value means higher values in the positive class. It has a direct meaning, which is why a report quotes it: a value of 0.4 means that of one row drawn from each class, the positive one has the higher value 70% of the time.
- **For a continuous target**, [Spearman's rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient), with average ranks for ties ({py:func}`~seqmaestro.harness.effect.rank_correlation`). It runs from −1 to 1, and a positive value means that higher values of the feature go with a higher target. A few rows in a heavy tail cannot decide it, since only ranks enter.

The cell's `run.json` names the statistic (`marginal_effect.statistic`).

**Both statistics use ranks**, because a count and a fraction of the same thing over the same window are one feature under a monotone map, and only a rank statistic scores them alike. Features are matched by the rank correlation of their values for the same reason. Ranks also make the binary statistic insensitive to class imbalance, which a difference of means is not.

**One function writes the marginal effect for every arm** ({py:func}`~seqmaestro.harness.runner.score_arm`), from the features' values and labels that every arm hands over on its way into a [run](../../glossary.md#run)'s directory. The number therefore means the same for an arm fitted in Core as for one fitted in another environment. A feature with a value that is not finite on some training row gets no marginal effect (`None`). The matching imputes such values ([matching two sets of features](index.md#matching-two-sets-of-features)), whereas the marginal effect is left out, because repairing the values here would hide that the feature is undefined on some rows.

### The rows it is measured on

`marginal_effect` is measured on every training row against the real labels, whatever the cell was fitted on (`EFFECT_SPLIT` in {py:mod}`seqmaestro.harness.runner`). It describes the association in the data the model learned from, so it is an in-sample number by design, and it is no evidence of generalisation: a feature can separate the training rows cleanly and do nothing on the test rows. Whether an association holds on held-out rows is a question for the [held-out split](../../glossary.md#held-out-split), on which Tier 2 of the [reliability panel](../../glossary.md#reliability-panel) tests the features of each reduced model.

The other choices of rows would serve worse:

- **The validation split can be too small to settle a sign.** When the validation rows are split in half and the statistic is computed on each half, the two halves can disagree about the sign about as often as they agree, at small effects on a small split. A disagreement with `direction` in that range would be a property of the estimator. The training split is several times larger.
- **The fitting rows would weaken the control.** They are what the model saw, but `train_fraction` subsamples them for each [seed](../../glossary.md#seed), and a permuted cell's fitting rows have the shuffled labels. Measured against those, `marginal_effect` and `direction` would agree under shuffled labels for the same reason as under real ones, so the permuted cells would show nothing about that agreement. Real labels on every training row give one number per feature for the whole dataset, the same in the real and the permuted cells of a seed, which leaves `direction` as the only one of the two that differs between them.

## Uses

- **Reports.** Both fields are in every cell's `features.json`, from which a run's report takes them, and the spreadsheet of a run's features holds both, with an explanation of each. Where the two agree, one sentence covers both, and where they disagree, the disagreement is the finding.
- **The [judge](../../glossary.md#judge)** is shown the sign of the marginal effect and never its size, where `grid.judge_direction` is on, as it is in the shared [config](../../glossary.md#config). An effect size would let the judge tell real features from the permuted cells' features, where a run has the judge rate those too, whereas a sign is a coin flip for such a feature ([the judge](judge.md)).
- **Co-discovery.** Two arms that find the same [concept](../../glossary.md#concept) pointing the same way corroborate each other more strongly than by finding it at all, once their signs are aligned ([direction agreement](stability_and_co_discovery.md#direction-agreement)).

## Limits

- **Correlated columns.** Taking the direction of `pos_50_A` while `pos_50_C`, `pos_50_G` and `pos_50_T` are in the model has the problem that permutation importance has: between them, the other three largely determine what is measured. The record holds one row per feature, with no row per position in which to put a joint answer.
- **A model fitted on a confounded dataset points along the confounder.** `direction` is the model's direction, which need not be nature's.
- **One number is noisy**, even on the training rows. `marginal_effect` has no p-value, because a p-value for every feature of a whole grid would say more about the analysis than about any one feature. The evidence that a direction is real is that it recurs: the same sign, in the same concept, across seeds and arms.
- **The rate at which the two directions agree is a poor measure.** Both are estimates, and each becomes reliable at its own rate: the sign of `marginal_effect` settles as its number of rows grows, whereas the sign of `direction` settles as the model comes to rely on the feature. At small effects the second can be unreliable however many rows the first has, so even where the sign of `marginal_effect` is stable, the two can agree little better than chance. A low rate of agreement at small effects therefore describes the pair of estimators more than the features, and no single threshold of effect size holds across datasets. The check by halves above finds where the sign of `marginal_effect` settles for a given run.
- **Agreement means something only beside its control.** The permuted cells share `marginal_effect` with their real counterparts and differ only in `direction`, so a difference in agreement between the two [label conditions](../../glossary.md#label-condition) concerns `direction` alone.
