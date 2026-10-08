<!-- component: tier 1, selection stability. Dataset independent. Include in Stage 3B. -->

### Tier 1: would this feature be chosen again?

The training and validation sequences are split at random into two equal halves, and the **whole
procedure** is rerun on each half: fit the model on all its candidate features, keep the ones it uses,
then cut down to {{param:n_features}}. This was done {{param:half_sample_pairs}} times, giving
{{param:half_samples}} half-samples, each producing its own list of {{param:n_features}} features. The
two halves of each split share no sequences, so they cannot agree simply because the same sequence
appears in both.

**π̂ (pi-hat) is the fraction of the {{param:half_samples}} half-samples that chose the feature.** A
feature with π̂ = 1.00 was chosen every time. A feature with π̂ = 0.10 was chosen in one run in ten,
and a repeat of the study would probably not report it.

### Near-duplicate groups

Many candidate features measure almost the same thing. Features whose values track each other
closely (a rank correlation of {{param:cluster_similarity}} or more between every pair) are therefore
grouped, and a group counts as chosen in a half-sample if **any** of its members is chosen. This gives
a second frequency for each feature: that of its group.

When the two differ, read the group's frequency. A feature with π̂ = 0.5 in a group with π̂ = 1.0 is
not doubtful: the property it measures is found every time, and the half-samples are simply
splitting between two equivalent ways of measuring it.

### How many could be chance?

With a large pool of candidates, a few will be chosen often by luck alone. A standard upper bound
estimates how many:

```
expected number of chance features  ≤  q̂² / ( (2τ − 1) · p )
```

Here **τ** is the cut-off (for example 0.7, meaning chosen in 70% of half-samples), **p** is the number
of candidate features the configuration chose from, and **q̂** is the average number of features a
half-sample selected. The result is an expected count. A value of 0.2 means "at this cut-off, expect
about one chance feature in every five studies". A value of 6 means chance could explain everything
that passed.

**The bound is only useful when the candidate pool is large.** Choosing {{param:n_features}} features
from several hundred candidates gives a bound well below one. Choosing {{param:n_features}} from a few
dozen gives a bound above one at every cut-off, because the number chosen, squared, already exceeds
the pool. In those cases the bound is too loose to be informative, and the figure marks them. π̂ itself is
still a valid selection frequency.
