<!-- component: how performance is measured and what it is compared against. Dataset independent,
     for a continuous target; its twin for a binary one is 10_evaluation.binary.md.
     Include in Stage 2, before the first results table. -->

### How performance is measured

**R²** (R-squared) is the main measure. It is the share of the variation in the target across the
held-out sequences that a model's predictions account for: one minus the mean squared error of the
predictions, divided by the variance of the target. A model that predicts the target's average for
every sequence scores 0, and one that predicts every value exactly scores 1. R² falls below 0 when
the predictions are further from the true values than that average is, which is what a model that
learned noise from its training sequences does on new ones. Because each sequence counts by the
square of its distance from the average, R² on a target whose values mostly sit close together, with
a few far out, depends mostly on how well a model predicts those few.

**Spearman's rho** is reported beside it: the correlation between the ranks of the predictions and
the ranks of the true values, from -1 to +1, with 0 meaning no relation. It measures only whether a
model puts the sequences in the right order, so a few extreme values move it far less than they move
R². A model can order the sequences well while predicting values on the wrong scale, and its rho is
then high while its R² stays low.

Where a table gives a **95% interval**, it comes from resampling the held-out sequences, and shows
how much the number could move on a different held-out set of the same size.

### Three comparisons for every result

A performance number is easiest to interpret beside reference points, and each result in this report
is shown beside three.

**1. The same model trained on shuffled labels.** Every configuration was also fitted with the
target's training values randomly shuffled among the training sequences, which destroys any real
link between sequence and target, and then scored in the same way. It should land near 0 or below,
and it shows what the configuration can reach with no real signal to learn. A real result has to
beat it clearly. In the figures it is the short vertical tick on each bar.

**A shuffled-label score can sit away from 0, on either side of it, with nothing leaked.** A model
fitted to shuffled values still settles on a direction among its features. Where one feature carries
far more of the target than the rest, such as base composition, that direction can fall along it,
with a sign the shuffle happened to set, so the predictions follow the real target a little or run
against it. The score then moves from 0 further than the size of the held-out set alone would
suggest. When the configurations of a repeat share its shuffle, as they do by default, they move
together. A leak would show as shuffled scores above 0 in every repeat.

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
