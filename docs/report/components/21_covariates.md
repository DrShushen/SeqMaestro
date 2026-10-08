<!-- component: covariates, the measurements supplied with each sequence. Dataset independent.
     Include in Stage 2, straight after components/20_feature_sources.md, when the dataset supplies
     covariates. -->

### Measurements supplied with each sequence

This dataset supplies **covariates**: numbers measured for each sample beside its sequence, such as
a chromatin signal over the window or an experimental measurement. Stage 1 lists what each one
measures.

A covariate reaches a model in two ways. A configuration whose feature source reads **+ covariates**
was given each covariate as one more column, beside the features computed from the sequence. And a
written source that reads **+ covariates** was shown the covariates and what each measures, so the
features the language model wrote may combine a covariate with the sequence, for example "the
signal, counted only where the sequence holds a given motif". An enumerated source computes nothing
from a covariate: its features describe the sequence alone.

Comparing a configuration with and without **+ covariates** shows what the measurements add to what
the sequence already gives. The base-composition reference line reads the sequence only, on every
dataset.

**Limitations.** A covariate enters as it was measured, with that measurement's noise and biases, so
a model that leans on one says as much about the measurement as about the sequence. A covariate
measured in a way that depends on the outcome can carry the answer without saying anything about the
sequence, and whether any covariate could do that is a question about how it was collected, which
Stage 1 settles.
