<!-- component: the additive model and its figure. Dataset independent, for a continuous target;
     its twin for a binary one is 30_model_ebm.binary.md.
     Include in Stage 3A, immediately before the shape-function figure. -->

### The additive model (EBM), and what its figure shows

An **explainable boosting machine (EBM)** learns one curve for each feature, called its shape
function. To predict the target for a sequence, the model takes the height of each feature's curve
at that sequence's value and adds the heights to a constant. The model has no terms that combine two
features, so the curves in the figure are the whole model.

Heights are in **the target's own units**. A height of 0 means the feature moves the prediction
neither way at that value. A height of +0.05 adds 0.05 to the predicted value of the target, and a
height of -0.05 takes 0.05 away.

**One panel.**

- The horizontal axis is the feature's value.
- The curve is what the model adds to the prediction at that value. Above zero raises the
  prediction, and below zero lowers it.
- The grey strip underneath shows how many training sequences have each value. A steep or extreme
  stretch of curve above an empty part of the strip rests on very few sequences, so treat it with
  caution.
- Panels are ordered by how much each curve rises and falls overall, so the first panel is the
  feature the model relies on most.

**What to look for.** A curve that rises steadily means "the more of this, the higher the target". A
curve that is flat and then jumps means the model found a threshold, and the value at the jump is
itself a testable prediction. A curve that rises and then falls means the feature raises the target
at middle values and lowers it at the extremes, which is a different claim from either simple
direction.

**Limitations.** Each curve shows what the model does with a feature **given the other
{{param:n_features_minus_one}} features in the model**, so it describes the feature's part in this
model rather than its effect in general. When two features measure nearly the same thing, the model
may give most of the effect to one and leave the other nearly flat, and a model that made the
opposite choice could predict about as well. Stage 3B looks at this directly.
