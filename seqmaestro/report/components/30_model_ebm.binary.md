<!-- component: the additive model and its figure. Dataset independent, for a binary target; its twin for a continuous
     one is 30_model_ebm.continuous.md.
     Include in Stage 3A, immediately before the shape-function figure. -->

### The additive model (EBM) and its figure

An **explainable boosting machine (EBM)** learns one curve for each feature, called its shape
function. To score a sequence, the model takes the height of each feature's curve at that sequence's
value and adds the heights together. The model has no terms that combine two features, so the curves
in the figure are the whole model.

Heights are in **log-odds**. A height of 0 means the feature favours neither class at that value. A
height of +0.7 roughly doubles the odds of the positive class, and +1.4 roughly quadruples them.
Negative heights work the same way in the other direction: -0.7 roughly halves the odds.

**One panel.**

- The horizontal axis is the feature's value.
- The curve is what the model adds to the score at that value. A curve above zero favours the
  positive class, and one below zero favours the negative class.
- The grey strip underneath shows how many training sequences have each value. A steep or extreme
  stretch of curve above an empty part of the strip rests on very few sequences, so treat it with
  caution.
- Panels are ordered by how much each curve rises and falls overall, so the first panel is the
  feature the model relies on most.

**What to look for.** A curve that rises steadily means "the more of this, the more likely positive".
A curve that is flat and then jumps means the model found a threshold, and the value at the jump is
itself a testable prediction. A curve that rises and then falls means the feature favours one class
at middle values and the other at the extremes, which is a different claim from either simple
direction.

**Limitations.** Each curve shows what the model does with a feature **given the other
{{param:n_features_minus_one}} features in the model**, so it describes the feature's part in this
model rather than its effect in general. When two features measure nearly the same thing, the model
may give most of the effect to one and leave the other nearly flat, and a model that made the
opposite choice could predict about as well. Stage 3B looks at this directly.
