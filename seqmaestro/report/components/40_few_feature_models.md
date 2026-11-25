<!-- component: how the few-feature models were produced. Dataset independent.
     Include at the top of Stage 3A. -->

### How the few-feature models were produced

Each configuration in Stage 2 ended with a fitted model that can use hundreds of features. Stage 3A
reduces each of these models to at most **{{param:n_features}} features** and reports how well the
smaller model performs.

The reduction uses **recursive feature elimination**: fit the model, remove the features it relies
on least, refit on what remains, and repeat until {{param:n_features}} are left. Each step removes
a quarter of the remaining features, or every feature the model does not use at all where there are
more of those, and near the end the features are removed one at a time.

The model is refitted after every removal because, once a feature is gone, the model can shift
weight onto the others: a feature that looked unimportant next to a near-duplicate can become
important once the duplicate is removed. Keeping only the {{param:n_features}} features the first
model ranked highest would miss this.

Three points about the procedure:

- **It works only with what each configuration already found.** There is no new search, so a
  configuration that never had a good feature cannot gain one here.
- **The held-out sequences played no part.** Which features to remove, when to stop, and any
  decision threshold a model uses were all decided on the training and validation sequences.
  Every model was saved before the held-out set was opened, and each held-out number is a saved
  model's prediction, scored once.
- **No language model was involved.** Features written by a language model in Stage 2 are computed by
  the code saved with them, so this step is an ordinary statistical procedure on a fixed table of
  numbers.

### The reference line

The line across the results is the base-composition model from Stage 2
({{param:baseline_n_features}} numbers per sequence, in a sparse linear model). Every configuration
here is limited to {{param:n_features}} features. If the base-composition model also fits within that
limit (the baseline table says whether it does), the comparison is like for like: a configuration
that falls below the line used its {{param:n_features}} features less well than simply counting bases.

### What the reduction loses

Reducing a model usually lowers its performance somewhat, and the size of the loss is informative. A
configuration that loses almost nothing had its signal concentrated in a few features all along,
whereas one that loses a lot had a weak signal spread across many features, which is harder to act
on. Occasionally a configuration improves, meaning the removed features were adding noise.

### Which way does a feature point?

Each feature is labelled with the direction its higher values point in. The tables and the figures
take this direction from different models:

- **The tables** give the direction in the configuration's Stage 2 model, **before** the reduction,
  when the model still held all its features. This is recorded for every configuration.
- **The figures** show the reduced model itself, so they give the direction **after** the reduction.

The two can disagree, because how a model uses one feature depends on which other features it
holds, and that changes when the model is reduced to {{param:n_features}} features. When they
disagree, the figure describes the model reported here, and the table describes the model it was
reduced from.

A third, separate answer is **what the feature does on its own**, with no model involved. Each
table of a model's features gives it in its last column.
