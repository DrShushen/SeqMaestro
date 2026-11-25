<!-- component: the glossary. Dataset independent, for a binary target; its twin for a continuous
     one is 90_glossary.continuous.md. Include as the last appendix, and point at it
     from "How to read this report". -->

Terms used in this report that are not standard in molecular biology, roughly in the order they
first appear.

| term | meaning in this report |
|---|---|
| **feature** | a number computed from a sequence, such as "fraction of C or T in the first 100 bases" |
| **covariate** | a number measured for each sample beside its sequence and supplied with the dataset, such as a chromatin signal. A model can be given it as a column, and a language model writing features can be shown it |
| **padded position** | where sequences differ in length and the positional window pads the shorter ones, a position past the end of a sequence. It has a yes/no feature of its own, which also says the sequence is shorter than that position |
| **feature source** | one way of producing the features a model chooses from: generated mechanically (enumerated), supplied with the data (covariates), or written by a language model as named pieces of code. A model may be given several sources together, as in positions + k-mers |
| **motif** | a pattern of bases a protein binds, written as how often each base occurs at each position of its binding sites. A run's motif features say how strongly, and how often, each motif matches a sequence |
| **configuration** | one kind of model paired with the feature sources it was given. Each table of results has one row per configuration |
| **training, validation and held-out sets** | the three sets the sequences were split into. Models learned from the training set, choices such as decision thresholds were made on the validation set, and the held-out set was used only for the final measurements |
| **AUROC** | the probability that, for one sequence from each class, the model gives the positive one the higher score. 0.5 is chance and 1.0 is perfect |
| **AUPRC** | how well the model finds positives without flagging many negatives. Its chance level is the share of positives, so always compare it with that |
| **95% interval** | the range a number could plausibly take on a different held-out set of the same size, estimated by resampling the held-out sequences |
| **shuffled-label control** | the same model fitted with the training labels randomly shuffled. It shows what the configuration reaches with no real signal |
| **base composition** | the fraction of each base over the window, used in a sparse linear model. The reference line every result is compared with |
| **log-odds** | the scale on which the additive and linear models add up evidence. Adding 0.7 roughly doubles the odds of the positive class, and adding 1.4 roughly quadruples them |
| **separation on its own** | how well one feature separates the classes with no model involved, from -1 to +1. At +0.4, the positive sequence of a random pair has the higher value 70% of the time |
| **EBM (explainable boosting machine)** | an additive model: one curve per feature, with the curves' heights added together. It has no terms combining two features |
| **L1 logistic regression** | a linear model whose penalty sets most weights to zero, so fitting it also selects a few features. Features are rescaled first, so each weight is per standard deviation of its feature |
| **decision tree** | a series of yes-or-no questions, each comparing one feature with a threshold |
| **XGBoost** | a powerful model made of many decision trees, whose reasoning cannot be followed. Used only as a performance reference |
| **PySR (symbolic regression)** | a search for a formula that predicts the target from the features, built from arithmetic, the exponential and the logarithm. It returns the best formula it found at each size (the front), and the one with the lowest error on sequences held back from the search is kept |
| **front** | the best formula the search found at each size, from a single number up to its size limit |
| **Shapley value** | one feature's share of one sequence's prediction: how far the prediction moves when the feature takes that sequence's value instead of a typical one, averaged over every order in which the features could be given their values. A sequence's Shapley values add up to its prediction minus the average prediction |
| **recursive feature elimination** | shrinking a model by repeatedly removing the features it relies on least and refitting on the rest |
| **half-sample** | a random half of the training and validation sequences, on which the whole feature-selection procedure is rerun |
| **π̂ (selection frequency)** | the share of the {{param:half_samples}} half-samples in which a feature was chosen |
| **near-duplicate group** | features whose values agree at a rank correlation of {{param:cluster_similarity}} or more, treated as one |
| **concept** | features from different configurations whose values agree at a rank correlation of {{param:concept_similarity}} or more, so that one idea found under different names is counted once |
| **p-value (permutation)** | the share of {{param:permutations}} random shuffles of the labels in which a feature separated the classes at least as well as with the real labels |
| **q-value (BH)** | a p-value corrected for the number of features tested, by the Benjamini-Hochberg procedure. A q of 0.05 across twenty called features means about one of the twenty is expected to be a false call |
| **BY correction** | the Benjamini-Yekutieli correction: like BH but with no assumption about how the features are related, and stricter |
| **Westfall-Young correction** | controls the chance of making even one false call, taking the actual correlations between features into account |
| **Holm correction** | a correction for the number of tests that controls the chance of even one false call. Used in tier 3 |
| **log-loss** | a penalty on a single prediction, larger the less probability the model gave to the true class |
| **leave-one-out refit** | removing one feature from a model, refitting, and testing on held-out sequences whether the model got worse |
| **detection floor** | the smallest effect the held-out set can distinguish from chance. Smaller effects are beyond what this study can resolve |
