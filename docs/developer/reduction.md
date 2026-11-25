# Stage 3A: the reduction

> This page tracks the reduction ({py:mod}`seqmaestro.reduction`), the stage that runs it ({py:mod}`seqmaestro.pipeline.reduction`), where each arm is reduced ({py:mod}`seqmaestro.pipeline.sides`), and the reduction of the vendored arms: {py:mod}`seqmaestro.pipeline.vendored.deft_cart` with [`deft_cart.py`](../../experiments/adapters/deft_cart.py), and {py:mod}`seqmaestro.pipeline.vendored.igsr_terms` with [`igsr_bundle.py`](../../experiments/adapters/igsr_bundle.py). Change it in the same commit as any of them.

An [arm](../glossary.md#arm)'s model in Stage 2 can use hundreds of features, too many for a person to examine one by one. Stage 3A therefore answers a narrower question for every arm: which N of its features does the arm's own model keep when it may keep only N, and how much does it lose on the validation split by keeping so few? N is the [config](../glossary.md#config)'s `reduction.n_features`, which [`_shared.yaml`](../../experiments/configs/_shared.yaml) gives every config.

The answer is a reduced model that is small enough to interpret in full. The [reliability panel](../glossary.md#reliability-panel) (Stage 3B) then tests each of its features before the report draws it.

## The three models

The reduction uses recursive feature elimination: fit the arm's model, drop the features it relies on least, refit on the rest, and repeat until N are left ([the elimination](#the-elimination)). Each reduced arm keeps three models from it:

- **The full refit**: the arm's model refitted on every feature its Stage 2 fit used. It is the first step of the elimination.
- **The reduced model**: the N-feature model that the elimination arrives at. It is the model that the reliability panel profiles and the report draws.
- **The top-N control**: the arm's model fitted once on the N features that the full refit ranked highest. This is the one-shot alternative to the elimination, so where the reduced model scores better than the top-N control, refitting after each removal found a better set of features than the first ranking gives.

The full refit can score differently from the Stage 2 model, because it is fitted only on the features that model used, whereas the Stage 2 model was fitted on everything it was offered. Where an arm used every feature it was given, the two models are the same. Where it did not, comparing the full refit's validation score with the Stage 2 cell's shows the difference.

The reduction also records its trajectory: the features, their importances and the validation score at every step of the elimination.

Each of the three models is saved as a standard [cell](../glossary.md#cell), so the [reveal](../glossary.md#reveal) scores it on the held-out split like any other cell. A Core arm's featuriser is saved narrowed to the model's columns ({py:class}`seqmaestro.reduction.arm.SelectedColumns`), so that the saved model predicts from the sequence. [The held-out split and the reveal](heldout.md) says how a reduced DEFT or IGSR model predicts on its side.

{py:mod}`seqmaestro.reduction.outputs` lists the files each arm leaves under `reduction/<model>/<pool>/`. The run's `reduction/summary.csv` has one row per reduced arm, with the validation score of each of its three models.

## The source cell

Every arm of the config is reduced, except those its entry lists under `not_reduced`, each with a reason that the stage prints. A black-box model's arms are reduced too, because the model ranks the columns it used, although the reliability panel leaves them out.

Each arm is reduced once, from its source cell: the real-label cell with the best validation score on the headline metric (AUROC for a binary target, R-squared for a continuous one), with ties going to the smaller [seed](../glossary.md#seed) ({py:func}`seqmaestro.reduction.arm.best_seed`).

- Only real-label cells are reduced, because the cells fitted on permuted labels (the other [label condition](../glossary.md#label-condition)) already serve as the suite's control.
- Validation decides because it is the only split a model is scored on before the reveal. For an arm on the [iterative source](../glossary.md#iterative-source), and for an IGSR arm, validation also steered what the arm wrote, so its best seed is partly the seed whose steering worked best. The report template requires the report of such an arm to say so.
- An arm is reduced only once all of its real-label cells have finished, because a best seed chosen among some of the seeds might differ from the one the run would choose.

The [driver](../glossary.md#driver) runs the stage once per arm, as soon as the arm's cells have finished. {py:mod}`seqmaestro.pipeline.reduction` gives the command for running it by hand. A revealed run is refused.

## The starting features

The elimination starts from the features the arm's model used in Stage 2 (the [feature records](../glossary.md#feature-record) in its cell's `features.json`), whereas the arm was offered its whole [pool](../glossary.md#pool). Two reasons decide this:

- For an arm on a [written source](../glossary.md#written-source), the features its model used are the only starting set that can be rebuilt without calling a language model.
- One rule for every arm keeps the arms comparable.

A positional arm is therefore reduced over the positions its model used. The reliability panel, by contrast, starts its selections from everything the arm was offered, because its Tier 1 measures the whole selection ([candidates](reliability.md#candidates)).

## Rebuilding the arm

Everything the refit needs is rebuilt from the arm's source cell ({py:mod}`seqmaestro.reduction.arm` describes the rebuild): the fitting rows and their weights from the cell's seed, `train_fraction` and weight column, the feature values by whatever computed them in Stage 2, and the model from the description the cell recorded. Each setting is therefore stated once, in the cell, and the reduction changes only the columns of the model the cell reported. Two checks make sure that the rebuild is the arm's:

- **The rebuilt columns** must match the values the cell recorded on the validation rows when it was fitted (`feature_values.npz`), column by column, at the precision the cell stored them in and within a tolerance that admits rounding alone ({py:mod}`seqmaestro.reduction.check`). This checks the step that could otherwise go wrong silently, such as a column taken from the wrong featuriser, or a written function evaluated against a differently named column.
- **The rebuilt model** must describe itself as the cell's model did, or the reduction refuses it, because a setting the rebuild dropped would fit a different model under the arm's name ({py:func}`seqmaestro.reduction.arm.rebuild_model`). For the same reason, a DEFT cell without the record of the configuration its tree was fitted with ({py:func}`seqmaestro.pipeline.vendored.deft_cart.reduce_arm`), or an IGSR bundle that does not record the fit its search made ({py:func}`seqmaestro.pipeline.vendored.igsr_terms.igsr_fit`), is refused.

**The elimination fits on the values the arm's Stage 2 model was fitted on**, in float64, with nothing rounded on the way (the check above compares at the stored precision only). An ensemble that bins its inputs, such as an EBM, can take a different path through the elimination when a written feature is rounded, so a reduction on rounded values would not be a reduction of the arm's own model.

**The reduction does not call a language model.** {py:mod}`seqmaestro.pipeline.sides` says on which side of the environment boundary each arm's columns are computed and its fits made, and the vendored arms work as follows:

- A DEFT arm's columns are its recorded features, computed by its Stage 2 tree, which on a pool with the [joint source](../glossary.md#joint-source) holds the functions a language model wrote. The reduction's tree is given a numeric frame of those columns, so its candidates at each node are the columns themselves: it is plain CART over them, using DEFT's own threshold search.
- An IGSR arm's terms are the functions its search saved, run by IGSR's copy on IGSR's side. The least-squares fit on their values needs nothing of IGSR, so it is made on the Core side.

## The elimination

The loop is {py:mod}`seqmaestro.reduction.eliminate`, the same for every arm in both environments: fit, rank by the model's importance, drop the weakest and refit (recursive feature elimination, as in [scikit-learn's RFE](https://scikit-learn.org/stable/modules/feature_selection.html#rfe)). Its docstring states the three choices that the textbook loop leaves open: how many features go at each step, what happens to the features a model does not use, and how ties are broken. DEFT's side loads the same file by path, which is why the loop uses numpy alone.

The importance is each model's own measure, taken as an absolute value, because an L1 coefficient is signed and a large negative weight is as influential as a large positive one:

- **A Core model** ranks by the importance it reports in Stage 2 ([the feature record](feature_record.md)). For PySR, that is each feature's mean absolute Shapley value in the equation, so every feature the equation does not name has zero importance.
- **A DEFT tree** ranks by each feature's share of the tree's impurity reduction, computed as the Stage 2 adapter computes it.
- **An IGSR arm** ranks by the absolute coefficient per standard deviation of each term, in the least-squares fit its search made, whereas its Stage 2 cell reports IGSR's own influence measure.

**A set of features no larger than N is fitted once and kept whole.** When the final model of a longer elimination uses fewer than N features, the reduction still keeps N columns, some with zero importance, as the record shows. The reliability panel takes the reduced model's features to be the ones it uses, since the others hold their places by the tie-break alone.

**A PySR arm's elimination usually takes one or two searches.** Each fit is a new search on the surviving columns, in the PySR environment. The Stage 2 cell's features are the ones its equation named, which are rarely more than N, and a set no larger than N is fitted once. When the equation named more than N, the first search drops the columns that the new equation does not name, keeping at least N, which usually leaves exactly N for the second search. The top-N control is one more search. The searches run on the threads the reduction's process is given and are not reproducible, so each fit is saved and every number comes from what was saved.

## Limits

- **N is fixed in the config, and the pools are not searched again.** The trajectory shows where each arm's score starts to fall as features are removed.
- **Each arm is reduced at one seed.** The interval around a reduced model's score is a bootstrap over the rows it is scored on, so it measures the sampling error of that split alone. How far the elimination would choose other features on other training rows is what Tier 1 of the reliability panel measures.
- **The three models keep the arm's settings.** Each is the arm's own model with fewer columns, so the reduction tunes nothing.
