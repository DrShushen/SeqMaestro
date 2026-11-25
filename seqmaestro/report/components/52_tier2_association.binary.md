<!-- component: tier 2, marginal association with multiplicity control.
     Dataset independent, for a binary target; its twin for a continuous
     one is 52_tier2_association.continuous.md. Include in Stage 3B. -->

### Tier 2: does the feature hold on sequences no model was fitted on?

This tier sets the models aside and tests each feature by itself. The feature's value is used
directly as a score, and the test measures how well that score separates the two classes on the
**held-out sequences**. The measure is AUROC, as for the models: 0.5 means no separation, and 0.8
means that for one sequence drawn from each class, the feature gives the positive one the higher value
80% of the time.

**Values below 0.5 are just as informative.** A feature at 0.2 gives the *negative* sequence the higher
value 80% of the time, so it separates the classes as well as a feature at 0.8, in the opposite
direction. The distance from 0.5 is the strength, and the side of 0.5 is the direction.

Because the measure depends only on how sequences rank, it applies equally to yes/no indicators,
counts and continuous ratios, so all features can be compared on one scale.

No model is involved in this tier, which is what allows it to give a valid p-value for each feature.
It is the only tier that makes a claim of statistical significance.

### Where the p-values come from

The class labels of the held-out sequences are shuffled at random {{param:permutations}} times, and
each feature's separation is recomputed after every shuffle. A feature's p-value is the share of
shuffles in which it separated the classes at least as well as with the real labels (counting the real
labels as one of the shuffles). This makes no assumption about how the feature's values are
distributed, which matters for features that are zero in almost every sequence. The smallest possible
p-value is {{param:smallest_p}}, about one in {{param:permutations}}.

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

Whether a separation passes the test depends on how many held-out sequences there are, so the same
features give different results in a small study and a large one.

- **In a small held-out set**, real but modest separations can fail to pass. A tier that calls almost
  nothing may be reflecting the size of the study rather than the quality of the features.
- **In a large held-out set**, almost any real separation passes, including ones too small to be
  useful. A tier that calls almost everything no longer distinguishes between features.

**So use the size of the separation to judge a feature, and the p-value to confirm it is not
chance.** Both are reported for every feature. When nearly everything passes, the separation is the
column that still ranks them.

### A direct check on the corrections

Alongside the corrections, a simpler count is reported: at a given uncorrected p-value cut-off, how
many features are called with shuffled labels compared with the real labels. The ratio directly
estimates the share of calls that are noise, with no statistical theory involved. When it agrees with
the corrected counts, the corrections are working as intended.

### Limitations

The test is designed for features whose values rise or fall steadily with the class. A feature that
separates the classes mainly at its extremes (high and low values both pointing to one class) can
score close to 0.5 here even if a model uses it well. For such patterns, the additive model's curves
in Stage 3A are the better guide.
