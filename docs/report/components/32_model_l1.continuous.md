<!-- component: the sparse linear model and its figure, plus the two direction answers.
     Dataset independent, for a continuous target; its twin for a binary one is
     32_model_l1.binary.md. Include in Stage 3A, immediately before the weights figure. -->

### The sparse linear model (L1 linear regression, the Lasso), and what its figure shows

A **linear regression** multiplies each feature by a weight and adds up the results, with a constant,
to predict the target. The **L1 penalty** pushes weights toward zero and holds most of them at
exactly zero, so the fitted model uses only a handful of features. Selecting features is therefore
part of fitting the model.

Before fitting, every feature is rescaled to have a standard deviation of 1 across the training
sequences. This makes the {{param:n_features}} weights directly comparable, whether a feature is a
count in the hundreds or a yes/no indicator. Each weight is **in the target's units per standard
deviation of its feature**: a weight of +0.02 means that raising the feature by one standard deviation
raises the predicted value of the target by 0.02, with the other features held fixed.

**The figure** has two panels, with the same features in the same rows.

- **Left: how the model uses the feature.** The bar is the feature's weight. Its side shows whether
  the feature raises or lowers the prediction, and its length shows by how much.
- **Right: what the feature does on its own**, measured with no model involved: Spearman's rank
  correlation between the feature and the target across the training sequences. It runs from -1 to
  +1. At 0 the feature does not rise or fall with the target, and at +0.4 higher values of the
  feature go with higher values of the target, fairly consistently.

### Why two directions are shown

The two panels answer different questions, and they can disagree.

- **The model's direction** is how the model uses the feature while holding the other
  {{param:n_features_minus_one}} features fixed.
- **The feature's own direction** is what the data shows when nothing else is held fixed.

When the model uses a feature **against** its own direction, the feature is acting as a correction.
It overlaps with another feature in the model, and the model uses it to subtract out that shared
part. These rows are shaded in the figure. For them, a simple statement of "higher values mean a
higher target" (or a lower one) would be misleading, which is why the report shows both directions
side by side.

When a feature barely goes with the target on its own (close to 0 in the right panel), its direction
is uncertain, and a disagreement between the panels carries little weight.
