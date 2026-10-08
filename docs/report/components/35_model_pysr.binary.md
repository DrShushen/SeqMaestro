<!-- component: the symbolic model, its equation and its two figures. Dataset independent, for a
     binary target; its twin for a continuous one is 35_model_pysr.continuous.md.
     Include in Stage 3A, immediately before the equation's table and the two figures. -->

### The symbolic model (PySR), and what its figures show

**Symbolic regression** searches for a formula that predicts the target from the features. PySR
(Cranmer, 2023) builds formulas out of the features, constants, the four arithmetic operations, the
exponential and the logarithm. It evolves a population of formulas, keeping those that predict the
training sequences well, and returns the best formula it found at each size, from a single number
up to a size limit. That list is the **front**.

The model shown here is one formula from its front. The search saw most of the training sequences,
and the rest were held back to choose with: the model is the formula with the lowest error on the
held-back sequences. A longer formula can win by a margin no larger than chance, and bring features
with it that matter little, which is one reason Stage 3B asks how often each feature is chosen
again.

**The equation** is given in full above its figures, one added term to a line, with its constants
to three significant figures and each feature it uses beside what that feature measures. It gives
the **log-odds** of {{param:positive_class}}: the higher its value, the more likely the model finds
that class, and its value passed through the logistic curve is the model's probability. It is the
whole model: the sequence's value of each feature goes in, and the prediction comes out.

**The first figure** shows each feature's share of each prediction, its **Shapley value**: how much
the log-odds move when that feature takes the sequence's value and not a typical one, averaged over
every order in which the features could be given their values. Each dot is one training sequence,
with the feature's value on the horizontal axis and its Shapley value on the vertical axis. For one
sequence, the Shapley values of all the features add up to its log-odds minus the average log-odds.
A feature that enters the formula on its own traces a curve. A feature multiplied or divided by
another traces a band, because its share then depends on the other feature's value too. Panels are
ordered by the average size of the values, so the first is the feature the formula relies on most.

**The second figure** is the front: for each size of formula, its error on the sequences the search
used and on the held-back sequences, with the chosen formula marked. The error is the log-loss, the
usual measure of how far predicted probabilities are from the classes.

**What to look for.** A short formula with a recognisable form, such as a ratio of two counts or a
logarithm of one, is a precise claim that can be tested on new sequences. A band in the first figure
marks an interaction that the formula states exactly, and the formula names the features that take
part.

**Limitations.** The search is random and runs on many processors at once, so a second search on
the same sequences can return a different formula of similar accuracy. The formula shown is one good
description of the data, and Stage 3B measures how often its features are chosen again. Where the
formula divides by a feature or takes its logarithm, it can be undefined for some sequences, and
such a sequence is given the model's average probability as its prediction. The report's files
count how many held-out sequences that happened to.
