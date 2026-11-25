<!-- component: covariates, the measurements supplied with each sequence. Dataset independent.
     Include in Stage 2, straight after components/20_feature_sources.md, when the dataset supplies
     covariates. -->

### Measurements supplied with each sequence

This dataset supplies **covariates**: numbers measured for each sample beside its sequence, such as
a chromatin signal over the window or an experimental measurement. Stage 1 lists what each one
measures.

A covariate reaches a model in two ways. A configuration whose feature sources end in
**+ covariates** was given each covariate as one more column, beside the features computed from the
sequence. Where such a configuration also has a written source, the language model was shown the
covariates and what each measures, so the features it wrote may combine a covariate with the
sequence, for example "the signal, counted only where the sequence holds a given motif". An
enumerated source computes nothing from a covariate, since its features describe the sequence alone.

Comparing a configuration with and without **+ covariates** shows what the measurements add to what
the sequence already gives. The base-composition reference line uses the sequence alone, on every
dataset.

**Limitations.** A covariate enters as it was measured, with that measurement's noise and biases, so
a model that relies on one says as much about the measurement as about the sequence. A covariate
measured in a way that depends on the outcome can give the answer away without saying anything
about the sequence, and whether any covariate could do that is a question about how it was
collected, which Stage 1 settles.
