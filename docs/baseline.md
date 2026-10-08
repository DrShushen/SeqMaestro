# The composition baseline

> Tracks `seqmaestro/pipeline/baseline.py`, `seqmaestro/features/composition.py` and
> `seqmaestro/models/sparse.py`. Change one, change this page.

Every run fits one reference line before any arm: **what the proportion of each base in the window,
and nothing else, achieves on the same rows.** Two classes that differ in composition are separated
by it, and a continuous target that goes with composition is predicted by it, whatever else they
differ in, so an arm that scores no better has not shown it found anything composition does not
already give. A report draws it as one line beside every arm.

## What it fits

| | |
|---|---|
| features | `fraction_<symbol>`: the fraction of the window's positions holding each symbol, for every symbol in the training rows (`CompositionFeaturiser`) |
| model | the sparse linear model of the target's kind, on columns standardised on the fitting rows: `L1Logistic` at C = 1 for a binary target, and `L1Linear` at a hundredth of the universal penalty for a continuous one |
| rows | the same stratified subsample of the training rows the arms fit on at each seed, through the same harness |
| weighting | the grid's class weighting for a binary target, or the design's per-row weights where it has them, as for every arm |
| target | real only; the arms' permuted controls already measure chance |
| scoring | as any cell: validation when fitted, the test split at the reveal |
| cells | `<run>/baseline/seed_<n>/`, in the standard cell format |

**Why these settings.** Base fractions are few, sum to one, and sit on a narrow scale. The model is
the arms' L1 fit, on columns standardised on the fitting rows as every L1 fit is, so the penalty
does not depend on a column's units. The penalty is weak, which leaves the four or five columns
essentially unpenalised because there is nothing to select among: C is 1 for the logistic model,
against the arms' 0.1, and the linear model's is a hundredth of the universal penalty the arms use
(`seqmaestro.models.sparse.L1Linear`). The penalty also copes with the fractions
being collinear. Coefficients are per standard deviation of each fraction (`importance_kind`
`l1_coefficient_per_sd`).

**Why it is fixed.** The features and the model are set in `seqmaestro/pipeline/baseline.py`, which
`run_baseline.py` calls, and nowhere else.
Neither the config nor whoever prepared the dataset can change them, so every dataset gets the same
baseline, computed the same way, and it cannot be tuned towards a conclusion. The run supplies only
what the baseline must share with the arms for the comparison to hold: the rows, the seeds, a binary
target's class weighting and how a cell is scored.

**Why it is outside `results/`.** The Stage 2 analyses read every cell under `results/` as an arm.
The baseline is a reference line, and must not appear in a stability table or a co-discovery
cluster as though it were one.

## The only baseline

A dataset carries no baseline of its own: no build manifest, dataset README or config description
reports a composition, GC or dinucleotide model. Baselines computed while each dataset was built
would each be measured differently, on different rows, so none could be read against another. This
one is computed the same way in every run.

**On a dataset with covariates it is still the one reference line**, and it still reads the sequence
only. The covariates enter through the arms (`covariates` in a run config's arm section), and what
they add is read from an arm with them against the same arm without them, such as `ebm/positional`
against `ebm/positional+covariates`.
