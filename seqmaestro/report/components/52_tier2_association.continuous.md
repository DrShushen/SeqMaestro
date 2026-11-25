<!-- component: tier 2, marginal association with multiplicity control. Dataset independent, for a
     continuous target; its twin for a binary one is 52_tier2_association.binary.md.
     Include in Stage 3B. -->

### Tier 2: does the feature hold on sequences no model was fitted on?

This tier sets the models aside and tests each feature by itself. The test measures how far the
feature rises or falls with the target on the **held-out sequences**. The measure is Spearman's rank
correlation: 0 means no relation, +1 means that ranking the sequences by the feature ranks them
exactly by the target, and -1 means it ranks them exactly in reverse.

**Negative values are just as informative.** A feature at -0.3 goes with the target as strongly as a
feature at +0.3, in the opposite direction: its higher values go with lower values of the target.
The distance from 0 is the strength, and the sign is the direction.

Because the measure depends only on how sequences rank, it applies equally to yes/no indicators,
counts and continuous ratios, so all features can be compared on one scale, and a few extreme values
of the target cannot dominate it.

No model is involved in this tier, which is what allows it to give a valid p-value for each feature.
It is the only tier that makes a claim of statistical significance.

### Where the p-values come from

The target's values on the held-out sequences are shuffled among them at random
{{param:permutations}} times, and each feature's correlation is recomputed after every shuffle. A
feature's p-value is the share of shuffles in which its correlation was at least as far from 0 as with
the real values (counting the real values as one of the shuffles). This makes no assumption about how
the feature's values or the target are distributed, which matters for features that are zero in
almost every sequence. The smallest possible p-value is {{param:smallest_p}}, about one in
{{param:permutations}}.

### Correcting for the number of features tested

A configuration that picked {{param:n_features}} features from many candidates effectively tested
all of them. At p < 0.05, about one in twenty candidates with no real effect would pass by chance,
so the p-values are corrected for the number tested. Three corrections are reported, from least to
most strict:

| correction | what it controls | when to use it |
|---|---|---|
| **BH (Benjamini-Hochberg, giving the q-value)** | the expected share of false findings among the features called, assuming the features are independent or positively correlated | the main result: a feature "holds" at q ≤ 0.05 |
| **BY (Benjamini-Yekutieli)** | the same share, with no assumption about how the features are related | a version that holds whatever the relations between features, useful when many candidates are near-duplicates |
| **Westfall-Young** | the chance of making even one false call | the strictest claim. It accounts for the actual correlations between features, so it is much less harsh than a simple Bonferroni correction |

**A q-value applies to a list, not to a single feature.** A q of 0.05 across twenty called features
means about one of the twenty is expected to be a false call. It does not say which one.

### The number of calls depends on the size of the held-out set

Whether a correlation passes the test depends on how many held-out sequences there are, so the same
features give different results in a small study and a large one.

- **In a small held-out set**, real but modest correlations can fail to pass. A tier that calls
  almost nothing may be reflecting the size of the study rather than the quality of the features.
- **In a large held-out set**, almost any real correlation passes, including ones too small to be
  useful. A tier that calls almost everything no longer distinguishes between features.

**So use the size of the correlation to judge a feature, and the p-value to confirm it is not
chance.** Both are reported for every feature. When nearly everything passes, the correlation is the
column that still ranks them.

### A direct check on the corrections

Alongside the corrections, a simpler count is reported: at a given uncorrected p-value cut-off, how
many features are called with shuffled values of the target compared with the real values. The ratio
directly estimates the share of calls that are noise, with no statistical theory involved. When it
agrees with the corrected counts, the corrections are working as intended.

### Limitations

The test is designed for features that rise or fall steadily with the target. A feature that matters
mainly at its extremes (high and low values both going with a high target) can score close to 0 here
even if a model uses it well. For such patterns, the additive model's curves in Stage 3A are the
better guide.
