<!-- component: how to read this report. Dataset independent. Include once, near the front. -->

## How to read this report

The analysis has four stages, and the report has one section for each.

| stage | what it answers |
|---|---|
| **1. The data** | what the sequences are and what was measured on them, how they were prepared before any modelling, and what that preparation means for the results |
| **2. Representations and model classes** | which way of describing a sequence captures the signal, and which kind of model makes best use of it |
| **3A. Few-feature models** | the smallest models worth showing, drawn so that every step of their reasoning is visible |
| **3B. Reliability** | for each feature those models use, three separate tests of whether it would come up again if the study were repeated |

**Stages 3A and 3B serve different goals.** Stage 3A finds the short list of measurements that
predicts best, which is what you would use to score new sequences, whereas Stage 3B finds the
measurements that are dependable enough to design a follow-up experiment around. The two lists
overlap but need not match. A feature can be in the best model without being dependable, and a
dependable feature can be missing from the best model because a near-duplicate of it was already
there.

### Findings and interpretation are kept apart

Everything outside a marked block is either a number or a description of what was done.
Interpretation, meaning anything another reader might weigh differently, sits inside a block marked:

> **AI Agent Analysis Note**
>
> Like this.

Every section ends with one such block, even if it only says there is nothing to add.

### Terms and abbreviations

Each machine-learning term is explained where it first appears, and the **glossary** at the end
collects them all. Model and feature-source abbreviations are listed once, in a key at the start of
Stage 2.

### Which sequences each number comes from

Before any modelling, the sequences were split into three sets:

- a **training set**, which the models learned from;
- a **validation set**, used for choices made along the way, such as which repeat of a configuration to reduce;
- a **held-out test set**, which played no part in fitting any model or making any choice.

The held-out set stayed closed until every model in this report had been fitted and saved, and was
then opened once. Unless a sentence says otherwise, a performance number is measured on the
held-out set, and a number describing a model's internal workings (a weight, a curve, a split) is
measured on the sequences it was fitted on.
