<!-- component: how performance is measured and what it is read against. Dataset independent, for a binary target; its twin for a continuous
     one is 10_evaluation.continuous.md.
     Include in Stage 2, before the first results table. -->

### How performance is measured

**AUROC** is the main measure. Pick one sequence from each class at random: AUROC is the probability
that the model gives the higher score to the one from the positive class. A value of 0.5 means the
model does no better than a coin toss, and 1.0 means it always ranks the pair correctly. AUROC does
not depend on where a decision threshold is placed, and it is not affected by one class being rarer
than the other.

**AUPRC** is reported beside it and is most useful when positives are rare. It measures how well the
model finds the positives without also flagging many negatives. A model that knows nothing scores the
share of positives among the held-out sequences, here {{param:positive_rate}}, so read every AUPRC
against that number.

Where a table gives a **95% interval**, it comes from resampling the held-out sequences, and shows
how much the number could move on a different held-out set of the same size.

### Three comparisons for every result

A performance number is easiest to interpret beside reference points, and each result in this report
is shown beside three.

**1. The same model trained on shuffled labels.** Every configuration was also fitted with the
training labels randomly shuffled, which destroys any real link between sequence and class, and then
scored in the same way. It should land near 0.5, and it shows what the configuration can reach with
no real signal to learn. A real result has to beat it clearly. In the figures it is the short
vertical tick on each bar.

**A shuffled-label score can sit well away from 0.5, above it or below it, with nothing leaked.** A
model fitted to shuffled labels still settles on a direction among its features. Where one feature
separates the classes far better than the rest, such as base composition or the count of one binding
site, that direction can fall along it, with a sign the shuffle happened to set. The score then moves
from 0.5 by as much as that one feature allows, further than the size of the held-out set alone
would suggest. When the configurations of a repeat share its shuffle, as they do by default, they
move together: all above 0.5 in one repeat, all below it in another. A leak would show as shuffled
scores above 0.5 in every repeat. Scores on both sides of 0.5 are what shuffled labels give.

**2. Base composition.** Beating chance does not mean a model has learned anything interesting. The
more useful comparison is against the simplest possible description of a sequence: the fraction of
each base over the window ({{param:baseline_n_features}} numbers per sequence), fed to a sparse linear
model. It is trained on the same sequences as every other configuration, with no tuning. Its score
is drawn as a line across the results. A configuration needs to clear this line to show that it
captures more than base composition does.

**3. The size of the held-out set.** With a fixed number of held-out sequences, there is a smallest
effect that can be told apart from chance, and every score carries some uncertainty. Stage 1 states
both. Two configurations whose scores differ by less than that uncertainty are not distinguished by
this study, whichever is listed first.

### Repeats

Each configuration in Stage 2 was run {{param:n_repeats}} times, each time with a different random seed
and a different random subsample of the training sequences. Tables give the average and the range
across these repeats. A wide range means the result depends on which sequences the model happened to
see, and deserves as much attention as the average.
