<!-- component: the decision tree and its figure. Dataset independent, for a continuous target;
     its twin for a binary one is 31_model_tree.binary.md.
     Include in Stage 3A, immediately before the tree figure. -->

### The decision tree, and what its figure shows

A **decision tree** predicts the target for a sequence by asking a series of yes-or-no questions.
Each box asks whether one feature is at or below a threshold. "Yes" leads down the left branch and
"no" down the right, until the sequence reaches an end point, which gives the prediction: the average
target of the training sequences that ended there. Every sequence follows a single path, so the
prediction for any sequence comes with a short rule that produced it.

The study fits the tree on positions alone, and on positions together with features a language
model wrote at each branch (**written per split**). The figure's caption names the tree it shows,
and both are drawn the same way.

Each threshold was chosen by the fitting procedure as the cut that best divided the training
sequences reaching that box into one group with higher values of the target and one with lower
values, by how much it reduced the spread of the target within the two groups. Whether it
corresponds to a meaningful biological cut-off is a hypothesis to test.

**The figure.**

- Each box names the feature it tests and the threshold, and each end point gives the value it
  predicts.
- Below that, every box shows how many training sequences reached the box and their mean target.
- The box colour shows whether that mean lies above the mean of all the training sequences (blue)
  or below it (red), and the colour deepens the further it lies, measured in standard deviations of
  the target. A branch that separates high values from low ones gets darker as it goes down.
- A wedge stands for a branch that continues below the depth drawn. It shows how many sequences reach
  it and their mean, so you can see how much of the model is not drawn.

**What to look for.** The top box is the most useful single question the model found. The paths
ending in the darkest boxes are the combinations of conditions that pick out the sequences with the
highest and the lowest values of the target. If one feature is tested several times at different
thresholds, the model is approximating a graded effect with a series of steps.

**Limitations.** Trees can be sensitive to the exact training sequences: changing a few of them can
change an early question, and the branches below it then change too. The tree shown is therefore
best seen as one of several trees that would fit the data about equally well. Stage 3B's first tier
measures how stable the choice of features is, and the tree's results there belong beside this
figure.
