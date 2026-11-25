<!-- component: the decision tree and its figure. Dataset independent, for a binary target; its twin for a continuous
     one is 31_model_tree.continuous.md.
     Include in Stage 3A, immediately before the tree figure. -->

### The decision tree and its figure

A **decision tree** classifies a sequence by asking a series of yes-or-no questions. Each box asks
whether one feature is at or below a threshold. "Yes" leads down the left branch and "no" down the
right, until the sequence reaches an end point, which gives the prediction. Every sequence follows a
single path, so the prediction for any sequence can be written as a short rule.

The study fits the tree on positions alone, and on positions together with features a language
model wrote at each branch (**written per split**). The figure's title names the tree it shows,
and both are drawn the same way.

Each threshold was chosen by the fitting procedure as the value that best separated the two classes
among the training sequences reaching that box. Whether it corresponds to a meaningful biological
cut-off is a hypothesis to test.

**The figure.**

- Each box names the feature it tests and the threshold, and each end point names the class it
  predicts.
- Below that, every box shows how many training sequences reached the box and what share of them are
  positive. A box at 50% positive has not separated anything yet.
- The box colour shows which class is in the majority, and the colour deepens as the share moves
  away from 50%. A branch that separates the classes well gets darker as it goes down.
- A wedge stands for a branch that continues below the depth drawn. It shows how many sequences reach
  it and their class balance, so you can see how much of the model is not drawn.

**What to look for.** The top box is the most useful single question the model found. The path
ending in the darkest boxes is the combination of conditions that most cleanly picks out one class.
If one feature is tested several times at different thresholds, the model is approximating a graded
effect with a series of steps.

**Limitations.** Trees can be sensitive to the exact training sequences: changing a few of them can
change an early question, and the branches below it then change too. The tree shown is therefore
best seen as one of several trees that would fit the data about equally well. Stage 3B's first tier
measures how stable the choice of features is, and the tree's results there should be considered
alongside this figure.
