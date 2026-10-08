<!-- component: the glossary. Dataset independent, for a continuous target; its twin for a binary
     one is 90_glossary.binary.md. Include as the last appendix, and point at it from "How to read
     this report". -->

Terms used in this report that are not standard in molecular biology, roughly in the order they
first appear.

| term | meaning in this report |
|---|---|
| **feature** | a number computed from a sequence, such as "fraction of C or T in the first 100 bases" |
| **covariate** | a number measured for each sample beside its sequence and supplied with the dataset, such as a chromatin signal. A model can be given it as a column, and a language model writing features can be shown it |
| **padded position** | where sequences differ in length and the positional window pads the shorter ones, a position past the end of a sequence. It has its own yes/no feature, which also says the sequence is shorter than that position |
| **feature source** | one way of producing the features a model chooses from: generated mechanically (enumerated), supplied with the data (covariates), or written by a language model as named pieces of code. A model may be given several sources together, as in positions + motifs |
| **configuration** | one kind of model paired with the feature sources it was given. Each table has one row per configuration |
| **training, validation and held-out sets** | the three sets the sequences were split into. Models learned from the training set, choices such as which repeat of a configuration to cut down were made on the validation set, and the held-out set was used only for the final measurements |
| **R²** | the share of the variation in the target across the held-out sequences that a model's predictions account for. Predicting the target's average for every sequence scores 0, and 1 is perfect. It falls below 0 for predictions further from the true values than that average |
| **Spearman's rho** | the correlation between the ranks of two sets of numbers, from -1 to +1. Reported beside R² for a model's predictions, and on its own for a feature's association with the target |
| **95% interval** | the range a number could plausibly take on a different held-out set of the same size, estimated by resampling the held-out sequences |
| **shuffled-label control** | the same model fitted with the target's training values randomly shuffled among the training sequences. It shows what the configuration reaches with no real signal |
| **base composition** | the fraction of each base over the window, used in a sparse linear model. The reference line every result is compared with |
| **the target's units** | the scale on which the additive and linear models add up their terms. A term of +0.05 adds 0.05 to the predicted value of the target |
| **rank correlation on its own** | how far one feature rises or falls with the target with no model involved: Spearman's rho, from -1 to +1. At 0 the feature does not track the target |
| **EBM (explainable boosting machine)** | an additive model: one curve per feature, with the curves' heights added together. It has no terms combining two features |
| **L1 linear regression (the Lasso)** | a linear model whose penalty holds most weights at exactly zero, so fitting it also selects a few features. Features are rescaled first, so each weight is per standard deviation of its feature |
| **decision tree** | a series of yes-or-no questions, each comparing one feature with a threshold. Each end point predicts the average target of the training sequences that reach it |
| **XGBoost** | a powerful model made of many decision trees, whose reasoning cannot be read off. Used only as a performance reference. It stops adding trees once they no longer improve its fit on a part of its training sequences set aside for that |
| **IGSR (influence-guided symbolic regression)** | a linear model over terms a language model wrote as short functions of the whole sequence. A term is kept only if the error on the validation sequences rises without it |
| **influence** | how much the error on the validation sequences rises when one term's weight is set to zero. IGSR keeps the terms with the largest |
| **PySR (symbolic regression)** | a search for a formula that predicts the target from the features, built from arithmetic, the exponential and the logarithm. It returns the best formula it found at each size (the front), and the one with the lowest error on sequences held back from the search is kept |
| **front** | the best formula the search found at each size, from a single number up to its size limit |
| **Shapley value** | one feature's share of one sequence's prediction: how far the prediction moves when the feature takes that sequence's value and not a typical one, averaged over every order in which the features could be given their values. A sequence's Shapley values add up to its prediction minus the average prediction |
| **recursive feature elimination** | shrinking a model by repeatedly removing the features it relies on least and refitting on the rest |
| **half-sample** | a random half of the training and validation sequences, on which the whole feature-selection procedure is rerun |
| **π̂ (selection frequency)** | the share of the {{param:half_samples}} half-samples in which a feature was chosen |
| **near-duplicate group** | features whose values agree at a rank correlation of {{param:cluster_similarity}} or more, treated as one |
| **concept** | features from different configurations whose values agree at a rank correlation of {{param:concept_similarity}} or more, so that one idea found under different names is counted once |
| **p-value (permutation)** | the share of {{param:permutations}} random shuffles of the target's values in which a feature's rank correlation with the target was at least as far from 0 as with the real values |
| **q-value (BH)** | a p-value corrected for the number of features tested. A q of 0.05 across twenty called features means about one of the twenty is expected to be a false call |
| **BY correction** | like BH but with no assumption about how the features are related. Stricter |
| **Westfall-Young correction** | controls the chance of making even one false call, taking the actual correlations between features into account |
| **Holm correction** | a correction for the number of tests that controls the chance of even one false call. Used in tier 3 |
| **squared error** | the penalty on a single prediction: the square of its distance from the true value |
| **leave-one-out refit** | removing one feature from a model, refitting, and testing on held-out sequences whether the model got worse |
| **detection floor** | the smallest effect the held-out set can distinguish from chance. Smaller effects are beyond what this study can resolve |
