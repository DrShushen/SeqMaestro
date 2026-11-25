# Stage 3B: the reliability panel

> This page tracks the panel ({py:mod}`seqmaestro.reliability`), the stage that runs it ({py:mod}`seqmaestro.pipeline.reliability`), the panel's part of the reveal ({py:mod}`seqmaestro.pipeline.reveal`), where each arm's fits are made ({py:mod}`seqmaestro.pipeline.sides`), and the fits of the vendored arms: {py:class}`seqmaestro.pipeline.vendored.deft_cart.DeftFits` with the modes `pool`, `reproduce`, `select` and `refit` of [`deft_cart.py`](../../experiments/adapters/deft_cart.py), and {py:mod}`seqmaestro.pipeline.vendored.igsr_terms` with the `evaluate` mode of [`igsr_bundle.py`](../../experiments/adapters/igsr_bundle.py). Change it in the same commit as any of them.

The [reduction](../glossary.md#reduction) (Stage 3A) leaves each [arm](../glossary.md#arm) with a reduced model of N features, where N is the config's `reduction.n_features`. The panel measures how far each of those features can be trusted, and the same for each candidate: each feature that the reduced model could have used in its place ([candidates](#candidates)). It gives three tiers of independent evidence, each with a threshold that corresponds to a stated guarantee:

| Tier | Question | Rows | Guarantee |
|---|---|---|---|
| 1, stability | Would the arm's own selection have chosen the feature on another half of the data? | Training and validation | A bound on the expected number of chance-level features above a threshold |
| 2, association | Is the feature associated with the target on its own, separating the classes or moving a continuous target? | Held-out | A p-value per feature, with false-discovery and family-wise adjustments |
| 3, necessity | Does the reduced model get worse without the feature? | Held-out | A one-sided paired test per feature, Holm-adjusted across the reduced model's features |

The panel's results are evidence about the reduced model, which stays as the reduction left it, because acting on the results after the [reveal](../glossary.md#reveal) would use up the split they were measured on.

In the modules and below, D stands for the training and validation rows together, and M\* for the reduced model.

## Profiled arms

The panel profiles every reduced arm whose model can be interpreted feature by feature. It leaves out the black boxes, such as XGBoost, because a profile of a black box's N columns would describe those columns and say nothing about the model. {py:mod}`seqmaestro.arms` declares which models are black boxes.

A [config](../glossary.md#config)'s arm entry can leave out more arms under `not_profiled`, each with its reason, and an arm left out of the reduction (`not_reduced`) is left out here too. The panel prints each arm it leaves out with the reason, and the report lists them.

Each profiled arm's panel builds on the reduction:

- **The reduced model and its seed** are the reduction's: its `reduced` cell, and the seed it was reduced from.
- **The fits** are rebuilt from the same Stage 2 [cell](../glossary.md#cell) and checked as the reduction's are ([rebuilding the arm](reduction.md#rebuilding-the-arm)). They are made where the arm's model lives ({py:mod}`seqmaestro.pipeline.sides`): in Core for a Core arm and for an IGSR arm, whose candidates IGSR's side evaluates, and on DEFT's side for a DEFT arm.
- **The reduced model's features are the ones it uses.** The reduction keeps N features even when its model uses fewer, and a survivor with zero importance holds its place by the tie-break alone. The panel therefore profiles such a survivor as a candidate, alongside the features the model discarded.

## Candidates

The candidates are everything the arm's model was offered, including what it discarded, so that Tier 1 measures the whole selection that led to the reduced model:

| Arm | Candidates | Evaluated by |
|---|---|---|
| A Core model | The whole vocabulary of the featuriser the Stage 2 cell saved: every indicator, every k-mer the run learned, every written feature of every round | That featuriser |
| `deft_or_cart` as CART, on a pool without `llm_generated_joint` | DEFT's naive candidate population, built on the fitting rows, as it was offered to the tree's root | DEFT, on its side |
| `deft_or_cart` as DEFT, on a pool with `llm_generated_joint` | The features its tree split on | DEFT, on its side |
| `igsr` | Every term its search accepted (`terms.json`), including the terms the search later pruned | IGSR, on its side, with the least-squares fits on them made in Core |

**The generative DEFT row is the exception to the rule.** DEFT's generative finder wrote its candidates node by node, each conditioned on the rows that reach the node, so the candidates never existed as a set and cannot be rebuilt without calling a language model. Candidates that are already the model's output make Tier 1 optimistic by construction, and the arm's record says so.

**Discovered motifs make Tier 1 optimistic too.** STREME found them in the Stage 2 cell's fitting rows, using those rows' labels ([discovered motifs](motifs.md#discovered-motifs)), and the fitting rows are part of D, which Tier 1 resamples. Each half-sample therefore holds many of the rows a motif was found to separate, and the arm's record states this caveat ({py:class}`seqmaestro.reliability.pool.Pool`). Tiers 2 and 3 use the [held-out split](../glossary.md#held-out-split), which the motif discovery never saw, so the caveat does not apply to them. Finding the motifs again inside every half-sample would remove the optimism, but it would take one STREME run for each half-sample of each arm.

## Step 0: the prefilter and the clusters

Before any tier, Step 0 decides which candidates are profiled and which of them are near-duplicates. Both decisions are made on D without looking at a label, so the same columns can be applied unchanged to the held-out split, whereas a filter or a clustering recomputed there would be a decision made with held-out rows. {py:mod}`seqmaestro.reliability.prefilter` gives the reasoning in full.

- **The prefilter** drops a candidate that is constant, or whose commonest value covers more than `max_modal_share` of D. A near-constant feature has nothing to associate and no selection to be stable about, and it would enter the multiplicity corrections as a hypothesis that can almost never be called.
- **The clusters** group near-duplicates, such as overlapping k-mers, by complete linkage on 1 - |Spearman's rho|, so that every pair inside a cluster agrees at |rho| of at least `cluster_rho`. The worst pair is measured and recorded. On a D of more than `MAX_CORRELATION_ROWS` rows, the correlations are estimated on a seeded subsample of that many rows.
- **Position groups** gather the indicators of each position. These indicators are mutually exclusive, so the correlation threshold never groups them, yet "this position matters" is often the finding.
- **An undefined value** (NaN or infinite) is ranked as its column's median over the defined values, in the clusters and in Tier 2 ({py:func}`seqmaestro.harness.stability.impute_non_finite`). The Core featurisers fill such values before a pool is built, whereas a DEFT feature written for one node can return NaN on rows unlike those that reach the node (DEFT's own fits keep the NaN and route it as its tree does). `prepare.json` lists every candidate that is undefined on rows of D, with the share of D (`undefined_share_of_D`), because a feature undefined on most rows differs from one defined everywhere, and the imputation alone would hide the difference.
- **A cluster's representative** is its member with the highest selection frequency in Tier 1. Ties go to the kind of feature a reader must take least on trust (an indicator, then a k-mer, a motif match and a written feature), and then to the name.

## Tier 1: stability

Tier 1 measures whether the arm's selection, run again on another half of the data, would choose the feature. It follows the complementary pairs design of [Shah and Samworth (2013)](https://arxiv.org/abs/1105.5578):

1. Draw B pairs of half-samples, where B is the config's `pairs`. Each pair splits D, of n rows, into two disjoint halves of floor(n/2) rows.
2. On each half, run the arm's whole selection, which is the reduction started from every candidate that Step 0 kept: fit the arm's model on the candidates, keep the features it used, and eliminate to N with the reduction's own loop ({py:mod}`seqmaestro.reduction.eliminate`, imported and never copied, with its digest recorded). On a weighted design, each half keeps its rows' weights.
3. Count as chosen by a selection only the features its final model uses, which can be fewer than N ({py:func}`seqmaestro.reduction.eliminate.used_columns`). The survivors the model does not use are the same columns on every half-sample, because the tie-break that keeps them follows the record order, so counting them would make the first columns stable whatever the labels.
4. Compute pi-hat for each feature, the share of the 2B selections that chose it, and q-hat, the mean number of features a selection chose (at most N).

**The bound.** For a threshold tau in (1/2, 1], with p the number of candidates left after the prefilter,

    E[V_tau] <= q-hat^2 / ((2 tau - 1) p)

bounds the expected number of features with pi-hat at or above tau whose true selection probability is no better than chance. The bound is an expected count of such features, and it is informative only where p is large compared with q-hat ({py:mod}`seqmaestro.reliability.tier1` works an example).

**Clusters and positions** get the same computation, with a group counted as selected when any of its members is. A position whose indicators share the selection between them is therefore stable as a position and unstable as a feature.

**Two reproduction checks** run first, in Step 0, on the fitting rows, to show that Tier 1 resamples the procedure that produced the reduced model:

1. Started from the features the Stage 2 model used, the selection must return the reduced model exactly. This shows end to end that the candidates, the model and the rows are the reduction's, and the panel stops otherwise.
2. Started from all the candidates, the selection may return something else, which shows that the elimination is sensitive to columns its own model ignores. This is reported, and the panel goes on, since such sensitivity is part of what Tier 1 measures.

**A model whose fits are not reproducible skips both checks**, and `prepare.json` records that they were not run, with the reason. PySR's search does not repeat itself under one seed, so a rerun of the reduction need not return the reduced model. Every number the panel reports comes from a saved fit, because the Stage 2 cells, the reduction's models and the refits of Tier 3 are all saved and scored from what was saved. The inputs are still checked as for every arm: the candidates' values against the Stage 2 cell's, and the model's description against a rebuild. For such an arm, the summary's `reproduced_from_pool` is empty.

**For PySR, Tier 1 measures the search as well as the rows.** A selection frequency then includes the variation of the search itself beside that of the half-samples, which is the variation the arm's selection has. The bound still holds, since Shah and Samworth's argument needs only that the two halves of a pair are selected independently, which separate searches are. Stability selection allows a randomised procedure: it was introduced with a randomised lasso ([Meinshausen and Bühlmann, 2010](https://arxiv.org/abs/0809.2932)).

**Tier 1 is the slowest part of the panel**, since it runs a whole elimination for each of the 2B half-samples. It therefore runs last before the reveal, in parallel workers, and resumes from the selections already on disk. {py:mod}`seqmaestro.reliability.selections` describes the sweep: how many workers run, how the memory they take is measured, and how the sweep shares the machine with the rest of a run. The results do not depend on the number of workers.

## Tier 2: association

Tier 2 measures whether a feature on its own, without any model, is associated with the target on the held-out split. {py:mod}`seqmaestro.reliability.tier2` gives the computation, which is exact, and {py:mod}`seqmaestro.reliability.multitest` what each correction assumes.

1. **The statistic** is a rank association of the feature with the target, the same for an indicator, a count and a ratio. For a binary target it is |AUC - 1/2| of the feature used as a score (reported as `auc`), and for a continuous target it is |rho|, Spearman's rank correlation (reported as `rho`). It detects monotone association only. A candidate that is undefined on some rows is ranked as Step 0 describes.
2. **The null distribution** comes from permuting the target `permutations` times. Each feature's permutation p-value has the +1 correction, so the smallest possible p-value is 1/(`permutations` + 1).
3. **The adjustments** are reported side by side for each feature:
   - BH q-values, which are the ones to quote.
   - BY q-values, which hold under any dependence among the candidates.
   - Holm-adjusted p-values.
   - The Westfall-Young min-P adjusted p-value, which controls the family-wise error rate using the dependence present among the candidates.
4. **The plug-in FDR estimate** is reported beside them at a range of raw cutoffs: how many features a permuted run calls, divided by how many the real run calls.

**The statistic uses the feature's values alone.** Permuting the labels calibrates a model's importance only under the global null (where the target is independent of every feature), so a per-feature p-value computed from an importance would not be valid.

**The number of permutations decides what can be called.** Calling even the most associated of p features at BH level q needs its p-value below q/p, so the smallest possible p-value, 1/(`permutations` + 1), must be below q/p. The config's number of permutations is set with that in mind, as [`_shared.yaml`](../../experiments/configs/_shared.yaml) explains.

**Tier 2 is computed on two sets of rows.** It is computed on the held-out split at the reveal, for every arm. It is also computed in-sample on D, before the reveal, and reported separately under a label that says so. Every candidate was selected by a model fitted on those rows, or proposed by a language model shown examples from them, so the in-sample number is anticonservative by an amount the panel does not estimate. It bounds the power from above, which lets a reader tell "no association" from "not enough held-out rows".

**A weighted design is refused.** The statistic and its null count every row once, whereas a design fitted with per-row weights would need a weighted statistic, which is not implemented. For such an arm, Tier 2 is not computed: the reason is in the arm's record and the log, the reveal skips Tier 2 for it, and the tables leave its columns empty. Tiers 1 and 3 fit with the weights.

## Tier 3: necessity

Tier 3 measures whether the reduced model needs each of its features, by leaving the feature out and comparing the two models row by row on the held-out split:

1. Before the reveal, refit the reduced model without each of its features, with the same model, rows, weights and seed. Refit it also without each cluster that holds two or more of its features, because two near-duplicates can each look unnecessary alone and be necessary together.
2. Save each refit as a standard cell (`loco/<label>/`), so that the reveal predicts and scores it like any other cell. The reduced model itself is the reduction's cell, and is not refitted.
3. After the reveal, compare each refit with the reduced model row by row, in the loss that the target's kind is fitted by: d = loss(refit) - loss(M\*) for each held-out row, with log-loss for a binary target and squared error for a continuous one. Probabilities are clipped away from 0 and 1 (`EPSILON`) before the logarithm. The change in the headline metric (AUROC, or R-squared) is reported beside each comparison.
4. Test H0: E[d] <= 0 one-sided by a paired t-test, because a feature whose removal improves the model is no evidence that the model needs it. A sign-flip permutation test (`N_SIGN_FLIPS` flips) is computed beside it, because heavy-tailed differences strain a t-test on a few hundred rows ({py:mod}`seqmaestro.reliability.tier3` gives the reasoning).
5. Across the family of refits, adjust the t-test p-values by Holm and by BH, and the sign-flip p-values by Holm. A feature, or a cluster, is **necessary** when its Holm-adjusted t-test p-value is below 0.05 (`HOLM_ALPHA`), and **substitutable** otherwise.

**Where one feature or one cluster is the whole model**, the refit has no feature left and predicts, for every row, the fitting rows' mean target, weighted as the reduced model was fitted ({py:class}`seqmaestro.reduction.arm.NoFeatures`). For a binary target that is the positive rate (one half under balanced class weights), and a DEFT arm's refit is then a tree of one leaf, which predicts the same constant. The test then measures whether the reduced model beats that constant.

**For a binary target, the number of rows at the clip is reported.** A pure tree leaf predicts 0 or 1, so one wrong row can dominate the mean loss. Where the count is above zero, the sign-flip test is the one to use.

**A substitutable feature is one whose information the other features already hold.** The leave-one-out importance of strongly correlated features is low, even negative, because it is conditional on what the other features already contain. Molnar's chapter on [leaving one feature out](https://christophm.github.io/interpretable-ml-book/lofo.html) describes this in detail.

## Global-null and planted-signal checks

Two checks test the panel itself, on labels whose answer is known. {py:mod}`seqmaestro.reliability.checks` gives both in full.

- **The global null** replaces the target with a permutation of itself. Tier 2 is rerun under it on the held-out split for every arm, where the p-values should be uniform, with hardly any feature called. Tier 1 is rerun under it on the check arm, where features should not come out stable.
- **The planted signal** is a target drawn, on the check arm, from a linear predictor on `planted_count` real columns of the candidates, with alternating signs on standardised columns and the predictor scaled to a standard deviation of `planted_signal_sd`. The columns are drawn from the candidates whose commonest value covers less than `PLANTED_MAX_MODAL_SHARE` of D, so that each varies enough to have an effect, or from all of them where too few do. The planted features keep the candidates' correlations, so their near-duplicates are associated too, and the check therefore measures recovery and says nothing about the false-discovery rate. The planted labels are used for Tier 1 and the in-sample Tier 2 on D, and for Tier 2 on the held-out split at the reveal.

The check arm is the arm the config names under `reliability.check_arm`, and the panel must profile it. Tier 1's reruns are made on that arm alone, because each needs another round of selections, and what they test (the half-samples and the bound) is shared by every arm.

## Order of the parts

The panel works in three parts: before the reveal, at it and after it.

```{mermaid}
flowchart LR
  subgraph before["Before the reveal"]
    S0["Step 0:<br/>prefilter, clusters"] --> T1["Tier 1:<br/>2B selections"]
    S0 --> T2D["Tier 2<br/>in-sample"]
    S0 --> L["Tier 3 refits,<br/>saved as cells"]
  end
  subgraph at["At the reveal"]
    LS["Tier 3 refits<br/>scored"]
    T2["Tier 2 on<br/>the held-out split"]
  end
  subgraph after["After the reveal"]
    T3["Tier 3 tests"]
    C["The checks<br/>and the tables"]
  end
  L --> LS --> T3
  S0 --> T2
  T1 --> C
  T2 --> C
  T3 --> C
```

1. **Before the reveal**, [`run_reliability.py`](../../experiments/runners/run_reliability.py) works on D alone: Step 0 with the reproduction checks, Tier 2 in-sample, the refits of Tier 3 (scored on validation only, as every cell is) and Tier 1, followed by the arm's record. The [driver](../glossary.md#driver) runs it for each arm once the arm's reduction has finished. It refuses a revealed run, and `--force` computes again what is on disk.
2. **At the reveal**, [`run_reveal.py`](../../experiments/runners/run_reveal.py) scores the refits of Tier 3 and computes Tier 2 on the held-out split, under the real labels and the null, and on the check arm under the planted labels too. It refuses to start while a profiled arm has no record ([the held-out split and the reveal](heldout.md)).
3. **After the reveal**, `run_reliability.py --after-reveal` computes Tier 3's tests, the checks and the tables from what the reveal wrote. It refuses a run that has not been revealed.

The parts before the reveal and at it write their files and skip the work whose files exist, so an interrupted panel resumes. The part after the reveal is computed again on every call. Every random draw the panel makes is seeded from the configured `seed`, with an offset for each kind of draw, so that each draw can be reproduced on its own, whereas the panel's model fits keep the arm's seed.

## Files

{py:mod}`seqmaestro.reliability.panel` lists the files each profiled arm leaves under `reliability/<model>/<pool>/`, and {py:mod}`seqmaestro.reliability.tables` describes the three tables assembled after the reveal. The run's `reliability/summary.csv` has one row per arm, with the numbers a report quotes.

## Settings

Every setting of the config's `reliability` section is required, because the settings decide what the panel's numbers mean. [`_shared.yaml`](../../experiments/configs/_shared.yaml) gives every config the section and explains each value. Once the config is loaded, {py:class}`seqmaestro.pipeline.config.ReliabilitySettings` holds it. Which arms are profiled is set by the config's `arms` entries ([profiled arms](#profiled-arms)).

The options `--workers` and `--memory-gb` of `run_reliability.py` are kept out of the config, because the results do not depend on them.

## Limits

- **The panel uses the reduction's seed alone.** Tier 1 varies the rows at that seed, and the suite's stability analysis covers the seeds ([stability and co-discovery](analyses/stability_and_co_discovery.md)).
- **It does not estimate how far the choice of candidates inflates the results.** Written candidates were written by a language model shown labelled training rows. Tier 2's null distribution is exact given the candidates and does not correct for how they were chosen, which only new sequences can settle.
- **Tiers 1 and 3 do not make a per-feature significance claim** in Tier 2's sense. A selection frequency and a leave-one-out comparison are statements about the model, and neither is a calibrated test of a feature's marginal association.
- **It compares values exactly.** A written function can compute one value in two ways (0.15 and 0.15000000000000002), and the prefilter, the clusters and Tier 2 would count them as two values. The Core featuriser of written features rounds its values to a fixed number of significant digits (`SIGNIFICANT_DIGITS` in {py:mod}`seqmaestro.proposer.checks`), so such a value is one value to every model and rank. DEFT's written features are computed on DEFT's side as its tree computes them, so for those, check how many distinct values a feature takes before quoting its modal share or its association to the third decimal.

## Sources

| Method | Source | Explanation |
|---|---|---|
| Stability selection with complementary pairs (Tier 1) | Meinshausen and Bühlmann (2010), JRSSB, [doi:10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x) ([arXiv:0809.2932](https://arxiv.org/abs/0809.2932)); with complementary pairs, Shah and Samworth (2013), JRSSB, [doi:10.1111/j.1467-9868.2011.01034.x](https://doi.org/10.1111/j.1467-9868.2011.01034.x) ([arXiv:1105.5578](https://arxiv.org/abs/1105.5578)); at the level of clusters, Faletto and Bien, [arXiv:2201.00494](https://arxiv.org/abs/2201.00494) | [Samworth's slides](https://statistique.cuso.ch/fileadmin/statistique/SamworthSlides.pdf) |
| Single-feature AUC as Mann-Whitney U (Tier 2) | Hanley and McNeil (1982), Radiology, [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747) | [AUC meets the Wilcoxon-Mann-Whitney U statistic](https://www.r-bloggers.com/2017/03/auc-meets-the-wilcoxon-mann-whitney-u-statistic/) |
| Permutation p-values with the +1 correction | Phipson and Smyth (2010), SAGMB, [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585) ([arXiv:1603.05766](https://arxiv.org/abs/1603.05766)) | [statmod::permp](https://rdrr.io/cran/statmod/man/permp.html) |
| BH and BY | Benjamini and Hochberg (1995), JRSSB, [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); Benjamini and Yekutieli (2001), [Annals of Statistics](https://doi.org/10.1214/aos/1013699998) | Efron and Hastie, *Computer Age Statistical Inference*, ch. 15, [free PDF](https://hastie.su.domains/CASI/) |
| Permutation plug-in FDR (Tier 2) | Storey and Tibshirani (2003), PNAS, [doi:10.1073/pnas.1530509100](https://doi.org/10.1073/pnas.1530509100) | [What are p-values, what are q-values?](https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.2/faq/pq-values.aspx) |
| Westfall-Young min-P (Tier 2) | Westfall and Young (1993), *Resampling-Based Multiple Testing*, Wiley | Dudoit, Shaffer and Boldrick, [preprint](https://biostats.bepress.com/ucbbiostat/paper110/) |
| Label-blind prefiltering (Step 0) | Bourgon, Gentleman and Huber (2010), PNAS, [doi:10.1073/pnas.0914005107](https://doi.org/10.1073/pnas.0914005107) | [DESeq2 vignette](https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html), independent filtering |
| Leave-one-covariate-out (Tier 3) | Lei, G'Sell, Rinaldo, Tibshirani and Wasserman (2018), JASA, [doi:10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116) ([arXiv:1604.04173](https://arxiv.org/abs/1604.04173)) | Molnar, *Interpretable Machine Learning*, [Leave One Feature Out](https://christophm.github.io/interpretable-ml-book/lofo.html) |
| Holm (Tiers 2 and 3) | Holm (1979), Scand. J. Statist., [JSTOR](https://www.jstor.org/stable/4615733) | [Holm-Bonferroni, step by step](https://www.statisticshowto.com/holm-bonferroni-method/) |
