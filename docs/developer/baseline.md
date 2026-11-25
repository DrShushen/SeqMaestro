# The composition baseline

> This page tracks {py:mod}`seqmaestro.pipeline.baseline`, {py:mod}`seqmaestro.features.composition` and {py:mod}`seqmaestro.models.sparse`. Change it in the same commit as any of them.

Every [run](../glossary.md#run) fits one reference model before its [arms](../glossary.md#arm): a sparse linear model on the fraction of each base in the sequence. It measures how well base composition alone predicts the target on the same rows as the arms. Where two classes differ in composition, or a continuous target goes with it, the baseline already predicts the target to that extent. An arm that scores no better than the baseline has therefore not shown that it found anything beyond composition. A report draws the baseline's score as the [reference line](../glossary.md#reference-line) beside every arm.

## Features and model

{py:mod}`seqmaestro.pipeline.baseline` sets the baseline's features and model:

- **The features** are the fractions of the sequence's positions that hold each symbol (`fraction_A`, `fraction_C` and so on), for every symbol in the rows the baseline is fitted on ({py:class}`~seqmaestro.features.composition.CompositionFeaturiser`). A fraction, unlike a count, is comparable across sequences of different lengths, and a symbol such as `N` becomes a column where the data holds it.
- **The model** is the arms' sparse linear model for the target's kind: {py:class}`~seqmaestro.models.sparse.L1Logistic` for a binary target, and {py:class}`~seqmaestro.models.sparse.L1Linear` for a continuous one. As in every L1 fit, the columns are standardised on the rows the model is fitted on, so that the penalty does not depend on a column's units. Each coefficient is then per standard deviation of its fraction.
- **The penalty is weak**, because there is nothing to select among four or five columns. C is 1 for the logistic model, whereas the arms' default is 0.1. The linear model's penalty is a hundredth of the universal penalty, which the arms use at full strength by default ({py:class}`~seqmaestro.models.sparse.L1Linear` explains the universal penalty). The penalty also copes with the fractions being collinear, since they sum to one.
- **What it shares with the arms** is what the comparison needs: the rows each [seed](../glossary.md#seed) fits on ([Stage 2: the suite](suite.md)), the class weighting of a binary target or a dataset's per-row weights, and how a [cell](../glossary.md#cell) is scored (on the validation rows when it is fitted, and on the held-out split at the [reveal](../glossary.md#reveal)).
- **It is fitted on the real labels only**, since the arms' permuted cells already measure chance.

Its cells are at `<run>/baseline/seed_<n>/`, in the standard cell format. They are kept outside `results/` because the [Stage 2 analyses](analyses/index.md) treat every cell under `results/` as an arm's, and the baseline must not appear in a stability table or a co-discovery cluster as though it were an arm.

## One baseline for every dataset

The features and the model are set in one place, {py:mod}`seqmaestro.pipeline.baseline`, which [`run_baseline.py`](../../experiments/runners/run_baseline.py) runs. Neither the [config](../glossary.md#config) nor whoever prepared the dataset can change them, so every dataset gets the same baseline, and nobody can tune it towards a conclusion.

A dataset has no baseline of its own. Baselines computed while each dataset was built would each be measured differently, on different rows, so no two of them could be compared. A dataset's build manifest, its README and its config's description therefore do not report a composition, GC or dinucleotide model.

On a dataset with [covariates](../glossary.md#covariate), the baseline is still the only reference line, and it still uses the sequence alone. The covariates enter through the arms (`covariates` in a pool of the config). What they add is measured by comparing an arm that has them with the same arm without them, such as `ebm/positional+covariates` with `ebm/positional`.
