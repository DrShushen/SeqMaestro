# Stage 3A: each arm reduced to a model of a few features

> Tracks `seqmaestro/reduction/`, `seqmaestro/pipeline/reduction.py`, how each side reduces an arm
> (`seqmaestro/pipeline/sides.py`), `seqmaestro/pipeline/vendored/deft_cart.py`,
> `experiments/adapters/deft_cart.py`, `seqmaestro/pipeline/vendored/igsr_terms.py` and
> `experiments/adapters/igsr_bundle.py`. Change one, change this page.

The suite says what each arm found, and an arm's model can lean on hundreds of features. Stage 3A
asks a narrower question of every arm: **which N of those features does its own model keep when it
may keep only N**, and what does that cost it on validation? N is the config's
`reduction.n_features`. The answer is a small model a person can read in full, which is what the
reliability panel then examines and the report draws.

## 1. What is reduced

**Every arm of the config**, but those its arm entry leaves out under `not_reduced`, each with a
reason the reduction prints. A black-box model's arms are reduced too, since its model ranks the
columns it used. Each arm gets one reduction, from its **real-label cell with the highest validation
headline** (AUROC for a binary target, R-squared for a continuous one), with ties going to the
smaller seed. The permuted-label cells are not reduced, because they are already the suite's null.
Validation is the only split read before the reveal. For an arm on the iterative source it is also
the split its proposals were steered by, so its best seed is partly the seed that steered best, and
the report template has a report of such an arm say so.

The stage is `experiments/runners/run_reduction.py --run-dir <run>`, with `--arms MODEL/POOL ...`
for some of the config's arms. An arm that has a `reduction.json` is skipped unless `--force`, an
arm is reduced only once every seed's real-label cell is finished, and a revealed run is refused.

**It starts from the features the arm's model used** (the records in its `features.json`), not from
the pool the arm was handed. For the language-model arms that set is the only starting point
reproducible without calling a model, and one rule for every arm keeps them comparable. A
positional arm is therefore reduced over the positions its model used, not over the whole window.

## 2. What is rebuilt, and checked

Nothing is restated; everything is taken from the Stage 2 cell.

| | from | checked by |
|---|---|---|
| the fitting rows | the cell's `seed`, `train_fraction` and weight column, through `subsample_training_rows`, real labels | the same function made them |
| the columns, ours | the featuriser the cell saved in `fitted.pkl`, on the fitting and validation rows | the validation matrix against the cell's `feature_values.npz`, column by column |
| the columns, DEFT's | the tree the cell saved, evaluating its own feature functions on DEFT's side | the same comparison, on DEFT's side |
| the columns, IGSR's | the terms the cell's search accepted, evaluated by IGSR's copy on its side (`igsr_bundle.py evaluate`) under the guard and the timeout the search used | the same comparison, on this side |
| the model, ours | the cell's `model_settings`: name, seed, class weight | the rebuilt model must describe itself the same way, or the reduction refuses; what a fit found (a PySR model's equation and front, under `fitted`) is left out of the comparison |
| the model, IGSR's | the fit the cell's search made, which its bundle records (`fit` in `arm.json`): least squares (`LeastSquares`), plain or with IGSR's ridge or lasso penalty, on the term values as IGSR fitted them; it needs nothing of IGSR, so it is fitted on this side | a bundle that records no fit is refused, and a search configured to fit other than by least squares is refused at Stage 2 |
| the model, DEFT's | DEFT's `AdaptiveDecisionTree` with its CART finder, built from the configuration hydra recorded for the cell's fit (`bundle/deft_run/.hydra/config.yaml`): the tree's settings and its splitting criterion as the cell's tree had them, with every override of the run applied; `min_samples_leaf` from the configured fraction of the fitting rows | a cell that holds no such record is refused |

**The columns are the values the arm's Stage 2 model was fitted on**, as the featuriser or the
tree computes them, in float64. Nothing is rounded on the way. An ensemble that bins its inputs,
such as an EBM, can take a different path through the elimination when a generated feature is
rounded, so a reduction on rounded values would not be a reduction of the arm's own model.

A DEFT arm's tree is handed a numeric frame of the arm's feature columns, so its candidate pool is
one identity candidate per column: plain CART over those columns, with DEFT's own threshold search.
No language model is called. The generative arm's functions are the ones its tree saved. An IGSR
arm's terms are likewise the functions its search saved, run by IGSR's copy, and no language model
is called either.

## 3. The elimination

`seqmaestro/reduction/eliminate.py`, the same loop for every arm and both environments:

- **Fit, rank by the model's own importance, drop the weakest, refit.** Importance is taken as an
  absolute value throughout, because an L1 coefficient is signed and a large negative weight is as
  influential as a large positive one. A DEFT tree's importance is each feature's share of its
  impurity reduction, computed as the Stage 2 adapter computes it. An IGSR arm's is the absolute
  coefficient per standard deviation of the term, of the fit its search made. A PySR arm's is each
  feature's mean absolute Shapley value in its equation, so every feature the equation does not
  name has zero importance.
- **A quarter of the surviving features go at each step while more than 30 remain, then one at a
  time.** The number of fits grows with the logarithm of the pool, and the part of the trajectory
  anyone reads is at full resolution.
- **Features the model does not use are dropped together**, as soon as they appear, because a zero
  importance marks an unused feature and says nothing about how weak it is. The count never drops
  below N.
- **Ties** are broken by the most recent step at which a feature carried importance, then by how
  much it carried then, then by record order, with the latest leaving first. In practice the first
  key rarely decides, because unused features leave at the step they appear. The second decides
  which survive when a model stops using several features at once.
- **A pool no larger than N is fitted once and kept whole.** A model that ends using fewer than N
  keeps N features, some with zero importance, and the record says so. The reliability panel takes
  M\*'s features to be the ones it uses, since the others hold their places by the tie-break
  alone.

Nothing in the loop reads a label other than the fitting rows'. The loop is numpy only, because
DEFT's side loads it by path and records its digest, and that environment never imports
`seqmaestro`.

**A PySR arm usually fits once or twice.** Each fit is a new search on the surviving columns, in
the PySR environment. The Stage 2 cell's features are the ones its equation named, rarely more than
N, and a pool no larger than N is fitted once and kept whole. When the equation named more than N,
the first search drops every column the new equation does not name, which usually leaves N or
fewer. The top-N control is one more search. The searches run on the threads the reduction's
process is given and are not reproducible, so each fit is saved and every number comes from what was saved.

## 4. What each arm leaves

```
reduction/<model>/<pool>/
    reduction.json      the source cell and seed; every step's features, importances, and
                        validation headline and its companion (AUROC and AUPRC, or R-squared
                        and Spearman's rho); the order features left; the survivors and
                        their importances; the top-N columns; the column check; settings
    steps.npz           validation scores at every step, and the validation targets
    full/               the model refitted on every feature the cell used: the first step
    reduced/            the model the elimination arrives at, on N features
    top_n/              the N features the full model ranked highest, fitted once
    elimination.json,   a DEFT arm alone: the elimination as DEFT's side handed it back, from
    elimination.npz     which this side writes the files above
reduction/summary.csv   one row per reduced arm: its model, pool and seed, the features it
                        started from and ended on, its steps, and the validation headline of
                        the three models (`val_<metric>_<model>`)
```

The column check compares the rebuilt columns with the Stage 2 cell's at the precision a cell
stores its values in, 32-bit floats, and allows a relative difference of 1e-5
(`seqmaestro/reduction/check.py`), which admits rounding and nothing else.

**The top-N control** is the one-shot alternative to recursion: keep the N features the first fit
ranked highest, and fit once. The reduced model beating it is the evidence that the recursion earned
its cost.

**Each of the three is a standard cell**: `run.json`, `metrics.json` and `predictions.npz`,
validation only, and its model. Our arms save the model as `fitted.pkl`, with the featuriser narrowed
to the model's columns so `load_fitted` predicts from sequence. A PySR cell's `metrics.json` also
counts the validation rows on which its equation was undefined and took its constant prediction
(`filled_rows`). DEFT's arms save it as `model.dill`,
which the adapter's predict mode loads. IGSR's arms save their least-squares model as `model.pkl`, and
the reveal predicts with it on the columns IGSR's side evaluates (`igsr_terms.predict_cell`). The reveal scores all three on the test split like any other
cell, with the arm's seed, its Stage 2 cell's bootstrap count, and the threshold chosen here on
validation ([`heldout.md`](heldout.md)).

**The full refit is not the Stage 2 model.** It is fitted on the features that model used, and the
Stage 2 model was fitted on everything it was offered. Where an arm used every feature it was given
the two coincide. Where it did not, the gap is visible beside the cell's own validation score.

## 5. What it does not do

- **It does not search the input pools again**, or choose N on validation. The trajectory shows
  where each arm's performance breaks, and N is fixed in the config.
- **It is one seed per arm.** Its intervals are sampling error on the evaluation split. The stability
  of the selection is what the reliability panel measures.
- **It tunes nothing.** Each model is the arm's own model with fewer columns.
