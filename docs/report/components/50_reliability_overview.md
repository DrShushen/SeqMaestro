<!-- component: what the reliability panel is and how the three tiers relate.
     Dataset independent. Include at the top of Stage 3B. -->

### What this section is for

Stage 3A produced models. This section asks a different question about the features inside them:
**if the study were repeated, which of these features would you expect to find again?**

A feature can fail to come back for three quite different reasons, and each matters differently for a
follow-up experiment:

- It may be one of several interchangeable ways of measuring the same property, so a repeat picks a
  different one.
- It may help inside a model while carrying little information on its own.
- It may be informative but not needed, because another feature in the model already says the same
  thing.

So three separate tests are run, called tiers, and reported separately.

| tier | the question | what a strong result means | counted as passing when |
|---|---|---|---|
| **1. Chosen again** | if the whole selection is rerun on a different half of the sequences, is this feature still picked? | the feature was picked because of the data, and not because of which sequences happened to be in the training set | it is picked in at least 70% of reruns |
| **2. Holds on held-out sequences** | leaving the model aside, does this one feature go with the target on sequences no model was fitted on? | the association is real and does not depend on the rest of the model | its corrected p-value (q) is 0.05 or less |
| **3. The model needs it** | if this feature is removed and the model refitted, does the model get worse? | the feature contributes something no other feature in the model provides | its corrected p-value is below 0.05 |

**The tiers often disagree, and the disagreements are informative.**

- **Chosen again, but does not hold on its own.** The model uses it to correct for another feature.
  It does useful work in the model, but is not a measurement to follow up by itself.
- **Holds on its own, but not chosen again.** The property is real, but this particular way of
  measuring it is one of several. Follow up the property, not this exact feature.
- **Chosen again and holds, but the model does not need it.** This is the most common pattern and the
  least worrying. Usually two features in the model carry the same information, so either one can be
  removed as long as the other stays.

**A feature that does not pass tier 3 may still be useful.** Tier 3 asks whether a model of
{{param:n_features}} overlapping features gets measurably worse without one of them. On a small
held-out set this test rarely detects anything, which reflects the size of the study more than the
feature. Tiers 1 and 2 are the ones to rely on.

### Which sequences each tier uses

Tier 1, and the grouping of near-duplicate features, use only the training and validation sequences.
Tiers 2 and 3 are measured on the held-out sequences, which were opened once, after every model and
every refit in this section had been fitted and saved. No choice in this section was made using the
held-out sequences.
