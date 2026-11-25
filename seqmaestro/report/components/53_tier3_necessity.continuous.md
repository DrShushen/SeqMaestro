<!-- component: tier 3, necessity within the reported model. Dataset independent, for a continuous
     target; its twin for a binary one is 53_tier3_necessity.binary.md. Include in Stage 3B. -->

### Tier 3: does the model need this feature?

The first two tiers look at a feature by itself, whereas this one measures what the model loses
without it. For each of the {{param:n_features}} features in the model:

1. Remove the feature, leaving {{param:n_features_minus_one}}.
2. Refit the model on the same sequences with the same settings, so the missing feature is the only
   difference.
3. Compare the two models' predictions on each held-out sequence.
4. Test whether the smaller model is reliably worse.

Each prediction is scored by its **squared error** (the square of its distance from the sequence's
true value), which is the penalty the model was fitted to make small. For every held-out sequence,
the comparison takes the smaller model's squared error minus the full model's, and the test is run
on these differences. The table's **fall in R² without it** is how far the model's R² falls
without the feature: the average of the differences divided by the variance of the target on the
held-out sequences, so it does not depend on the target's units. A positive value means the model
got worse without the feature.

**The comparison is paired.** Both models are scored on the same sequences, and the test is run on
the per-sequence differences. This is much more sensitive than comparing two overall scores.

**Only a loss counts.** If removing a feature *improves* the model, that is no evidence the feature
was needed, so the test is one-sided.

### The test

The table's **needs it (p)** comes from a t-test of whether the average of the per-sequence
differences is greater than zero. It is corrected for the number of removals tested (Holm
correction), and a feature counts as needed when the corrected p-value is below 0.05.

A few sequences with large differences can dominate an average, which a t-test handles poorly. A
second test is therefore run as a check: a **sign-flip test**, which estimates how often the average
would be as large if each sequence's difference were equally likely to be positive or negative. It
makes no assumption about how the differences are distributed, so where the two tests disagree, the
sign-flip test is the more trustworthy.

### Near-duplicates are also removed together

Two features that hold the same information can each look unnecessary on their own, because the
other one covers for it, and yet be needed together. Testing them one at a time would wrongly mark
both as removable. So when a model contains two or more features from one near-duplicate group, the
whole group is also removed at once, and the result appears as a row of its own.

### Limitations

How much tier 3 can resolve depends on the size of the held-out set.

- **On a small held-out set, few features may come out as needed.** The features overlap, so
  another feature can often supply what one provides. The correction covers every removal tested
  ({{param:n_features}} single features plus any groups), and the loss has to be detected from the
  held-out sequences alone. A result of "none needed" then mostly reflects the size of the study.
- **On a large held-out set, almost every feature may come out as needed**, because removing even a
  feature that contributes very little makes the model measurably worse.

In both cases, tier 3 is most useful as a ranking: which removals harm the model most, and which harm
it little or not at all. The column **fall in R² without it** gives that ranking.

- A feature whose removal harms the model little, and which no other tier supports, is a candidate
  to leave out of a follow-up.
- A feature that passes tiers 1 and 2 remains worth pursuing whatever tier 3 shows.
