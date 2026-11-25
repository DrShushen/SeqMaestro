<!-- component: the sparse linear model and its figure, plus the two direction answers.
     Dataset independent, for a binary target; its twin for a continuous
     one is 32_model_l1.continuous.md. Include in Stage 3A, immediately before the weights figure. -->

### The sparse linear model (L1 logistic regression) and its figure

A **logistic regression** multiplies each feature by a weight and adds up the results to score a
sequence. The **L1 penalty** pushes weights toward zero and sets most of them to zero, so the fitted
model uses only a handful of features. Selecting features is therefore part of fitting the model.

Before fitting, every feature is rescaled to have a standard deviation of 1 across the training
sequences. This makes the {{param:n_features}} weights directly comparable, whether a feature is a
count in the hundreds or a yes/no indicator. Each weight is **in log-odds per standard deviation of
its feature**: a weight of +0.7 means that raising the feature by one standard deviation roughly
doubles the odds of the positive class, with the other features held fixed.

**The figure** has two panels, with the same features in the same rows.

- **Left: how the model uses the feature.** The bar is the feature's weight. Its side shows which
  class the feature favours in the model, and its length shows how strongly.
- **Right: what the feature does on its own**, measured with no model involved. Take one sequence
  from each class. The number is how much more often the positive one has the higher value than the
  negative one. It runs from -1 to +1: 0 means the feature does not separate the classes, and +0.4
  means the positive sequence has the higher value 70% of the time.

### Why two directions are shown

The two panels answer different questions, and they can disagree.

- **The model's direction** is how the model uses the feature while holding the other
  {{param:n_features_minus_one}} features fixed.
- **The feature's direction by itself** is what the data shows when nothing else is held fixed.

When the model uses a feature **against** the direction it has by itself, the feature is acting as a
correction: it overlaps with another feature in the model, and the model uses it to subtract out
that shared part. These rows are shaded in the figure. For them, a simple statement of "higher
values mean more likely positive" (or negative) would be misleading, which is why the report shows
both directions side by side.

When a feature barely separates the classes on its own (close to 0 in the right panel), its direction
is uncertain, and a disagreement between the panels means little.
