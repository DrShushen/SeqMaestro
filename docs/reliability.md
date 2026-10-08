# Stage 3B: the reliability panel

> Tracks `seqmaestro/reliability/`, `seqmaestro/pipeline/reliability.py`, the panel's part of
> `seqmaestro/pipeline/reveal.py`, which side makes an arm's fits (`seqmaestro/pipeline/sides.py`),
> DEFT's fits (`DeftFits` in `seqmaestro/pipeline/vendored/deft_cart.py`, with the modes `pool`,
> `reproduce`, `select` and `refit` of `experiments/adapters/deft_cart.py`), and IGSR's
> (`seqmaestro/pipeline/vendored/igsr_terms.py`, with the `evaluate` mode of
> `experiments/adapters/igsr_bundle.py`). Change one, change this page.

Stage 3A leaves each arm with a model of N features, M\*. The panel asks how far each of those
features, and each feature M\* could have used instead, can be trusted, and answers with three
independent pieces of evidence per candidate, each with a threshold that maps to a stated guarantee:

| tier | question | data | guarantee |
|---|---|---|---|
| 1, stability | would the arm's own selection have chosen it on another half of the data? | train + val | a bound on the expected number of chance-level features above a threshold |
| 2, association | does it go with the target on its own: separate the classes, or move a continuous target? | test | a p-value per feature, with false-discovery and family-wise adjustments |
| 3, necessity | does M\* get worse without it? | test | a one-sided paired test per feature, Holm-adjusted across M\* |

It is a report about M\* and its candidates, not a selection step: nothing in it changes M\*.

## 1. What is profiled

**Arms.** Every reduced arm whose model can be interpreted feature by feature: the L1, EBM and PySR
arms, the tree's (`deft_or_cart`), and IGSR's. A profile of a black box's N columns would describe
the profile and say nothing about the model. Which models are black boxes is declared in
`seqmaestro/arms.py`, which takes it from the model class for the models fitted in Core
(`interpretable`), and the XGBoost arms are not profiled. A config's arm entry can leave more out
under `not_profiled`, each with its reason, and an arm left out of the reduction (`not_reduced`) is
left out here too. The panel prints every arm it leaves out with its reason, and the report lists
them.

**M\* and its seed** are the reduction's: its `reduced` cell and the seed it reduced from. The
panel's models are rebuilt from the same Stage 2 cell, as the reduction rebuilds them. **M\*'s
features are the ones it uses.** The reduction keeps N features even when its model uses fewer, and
a survivor with no importance holds its place by the tie-break alone, so the panel profiles it as
a candidate and not as one of M\*'s features.

**The candidate pool** is what the arm's model was offered, not what it kept, so that Tier 1 is a
statement about the whole selection and not only its last step:

| arm | pool | evaluated by |
|---|---|---|
| a model fitted in Core | the whole vocabulary of the featuriser the source cell saved: every indicator, every *k*-mer the run learned, every generated feature of every round | that featuriser |
| `deft_or_cart`, CART: a pool without `llm_generated_joint` | DEFT's naive candidate population, built on the fitting rows as its tree's root was offered it | DEFT, on its side |
| `deft_or_cart`, DEFT: a pool with `llm_generated_joint` | the features its tree split on | DEFT, on its side |
| `igsr` | every term its search accepted, at every node of the tree search (`terms.json`), including those the pruning later dropped | IGSR, on its side; the least-squares fits on them are made on this side |

The DEFT row is a stated deviation. The generative finder wrote its candidates node by node,
conditioned on the rows reaching each node, so its pool never existed as a set and cannot be rebuilt
without calling a language model. A pool that is already the model's output makes Tier 1 optimistic
by construction, and the arm's record says so. For every arm, the features the Stage 2 model used
are checked against the values its cell recorded (`feature_values.npz`), as in the reduction.

**The data each part sees**:

| | rows |
|---|---|
| Step 0, Tier 1 | D = train + val |
| the LOCO refits | fitted on the arm's fitting rows, scored on validation only as every cell is |
| Tier 2 in-sample (supplementary) | D |
| Tier 2, Tier 3 | test, at and after the reveal |

## 2. Step 0: the prefilter and the clusters

Both are computed on D and neither looks at a label, so their column set carries onto the test split
unchanged.

- **The prefilter** drops a candidate that is constant, or whose single commonest value covers more
  than `max_modal_share` of D. With `max_modal_share` at 0.99, on a 0/1 column this is the same as
  "non-zero prevalence below 1% or above 99%", but a prevalence rule would delete every continuous
  feature that is never zero. A near-constant feature has nothing to associate or be stable about, and would
  enter the multiplicity correction as a free non-discovery.
- **The clusters** are complete linkage on 1 - |Spearman rho|, cut so every pair inside a cluster
  agrees at |rho| >= `cluster_rho`. Complete linkage makes that a guarantee about every pair, where
  single linkage would guarantee only a chain, and the worst pair is measured and recorded. The
  correlation is Spearman because the kinds of feature are on different scales, and the sign is
  discarded because an anti-correlated twin carries the same information. Above 20,000 rows the
  correlations are estimated on a seeded subsample.
- **An undefined value** (NaN or infinite) is ranked as its column's median over the defined ones,
  in the clusters and in Tier 2, as the matching of features across repeats ranks one
  (`impute_non_finite` in `seqmaestro/harness/stability.py`). Our own featurisers fill such values
  before a pool is built. A DEFT feature written for one node can return NaN on rows unlike those
  that reach the node, and DEFT's fits on its side keep the NaN and route it as its tree does.
  `prepare.json` lists every candidate undefined on rows of D, with the share of D
  (`undefined_share_of_D`), because a feature undefined on most rows is a different thing from one
  defined everywhere, and the imputation alone would hide it.
- **Position groups** are formed by construction. The indicators of one position are mutually
  exclusive, so no correlation threshold groups them, yet "this position matters" is often the
  finding.
- A cluster's **representative** is its most stable member. Ties go to an indicator before a *k*-mer
  before a generated feature, then by name.

## 3. Tier 1: stability

**The selection is the reduction itself, started from the whole pool**: fit the arm's model on the
pool, keep what it used, then eliminate to N with the reduction's own loop
(`seqmaestro.reduction.eliminate`, imported and never copied, with its digest recorded). **What it
selects is what its final model uses**, which can be fewer than N. The survivors the model does not
use are the same columns on every half-sample, since the tie-break that keeps them follows the
record order, so counting them would make the first columns stable whatever the labels. Two
reproduction checks run first, on the fitting rows. Started from the features the Stage 2 model
used, the selection must return M\* exactly, which proves end to end that the pool, the model and
the rows are the reduction's. The panel stops otherwise. Started from the whole pool it may differ,
which says the elimination is sensitive to columns its own model ignores. That is reported, and is
not an error.

**A model whose fits are not reproducible skips both checks**, and `prepare.json` records that
they were not run, and why. PySR's search does not repeat itself under one seed, on one thread as
on many, so the same rows and seed can give another equation, and a rerun of the reduction need not
return M\*. Nothing the panel
reports depends on refitting the same model: Stage 2's cells, the reduction's models and the LOCO
refits are saved, and scored from what was saved. What the checks would show, that the pool, the
model and the rows are the reduction's, still rests on the inputs, which are checked as for every
arm: the pool's values against the Stage 2 cell's (the column check) and the model's description
against a rebuild. The summary's `reproduced_from_pool` is empty for such an arm.

**For PySR, Tier 1 measures the search as well as the rows.** A selection frequency then includes
the variation of the search itself beside that of the half-samples, which is the variation the arm's
selection has. The bound below still holds: Shah and Samworth's argument needs only that the two
halves of a pair are selected independently, which separate searches are, and stability selection
was first shown with a randomised lasso (Meinshausen and Bühlmann, 2010). Each Tier 1 worker runs
its own PySR search process with one Julia thread (the model's `one_core` option), so the
workers' searches run side by side, one per core.

**Complementary pairs** (Shah and Samworth, 2013). For b = 1..B (`pairs`), D is split into two
disjoint halves of floor(n/2) rows, and the selection is run on each: 2B selections, each with its
rows' own weights on a weighted design. pi-hat is the share of selections that chose a feature, and
q-hat the mean number chosen, at most N. For a threshold tau in (1/2, 1],

    E[V_tau] <= q-hat^2 / ((2 tau - 1) p)

bounds the expected number of features at or above tau whose true selection probability is no
better than chance. It is an expected count, not a probability, and it is informative only where p
is large against q-hat. The same is computed for clusters and positions, with a group counted as
selected when any member is. A position whose indicators share the selection between them is then
stable as a position and unstable as a feature.

Tier 1 is the panel's main cost, a full selection per half-sample, so it runs last before the
reveal, in parallel (`--workers`, by default the threads the panel's process is given, or every core
when it is given none), and resumes from the selections already on disk. Under the driver, the
panel is also lent the threads that free up while it runs, and Tier 1 starts a selection only while
fewer run than the threads it may use at that moment. The workers fill those
threads, so each fit in them runs on one core: an EBM without its own jobs, a PySR search
on one Julia thread, and every fit made in Core with the numeric libraries' thread pools held to
one thread (`one_thread` in `seqmaestro/reliability/fits.py`), without which a sparse linear
model's coordinate descent on a large half of D runs many times slower. A DEFT arm's selections
are fitted on DEFT's side, as its tree fits.

**Memory decides how many of those workers run.** Each worker holds its half of D and its fits,
which on a wide pool of many rows runs to gigabytes. Given the memory the panel may hold (the
driver tells each process what it may hold, and `--memory-gb` sets it by hand), Tier 1 runs its
first selection alone in a worker and measures what that worker adds, its search process included
for PySR. It then runs as many workers as fit beside what the panel holds, each expected to hold a
quarter more than the one measured, and never more than the threads it may use. Results do not
depend on how many ran (`seqmaestro/reliability/selections.py`).

## 4. Tier 2: association

**The statistic** is a rank association of the feature alone with the target, the same for an
indicator, a count and a ratio, and it detects monotone association only. For a binary target it is
|AUC - 1/2| of the feature used as a score, Mann-Whitney with average ranks, reported as `auc`. For a
continuous target it is |rho|, Spearman's rank correlation with average ranks on both sides,
reported as `rho`. A candidate undefined on some rows is ranked as section 2 says.

**The null** permutes the target (`permutations` times). Ranking the columns once makes every null
statistic one matrix product: with the labels for a binary target, and with the target's own
doubled, centred ranks for a continuous one, whose norm a permutation leaves fixed. Doubling and
centring the ranks makes every statistic an exact integer, so a permuted statistic equal to the
observed one is counted exactly. That matters for a binary feature on a few hundred rows, much of
whose null sits exactly on the observed value.

**Per feature** the panel reports:

- the permutation p-value with the +1 correction;
- BH and BY q-values. BY holds under any dependence, and BH is the one to quote;
- Holm;
- the Westfall-Young min-P adjusted p-value, which controls the family-wise rate using the
  dependence actually present in the pool.

Beside them is the plug-in FDR estimate at a range of raw cutoffs: how many features a permuted run
calls, divided by how many the real run calls. **Model importances are never used here**, because
permuting labels calibrates a multivariate importance only under the global null.

The smallest p-value is 1/(permutations + 1), and calling the most associated of p features at BH q
needs it below q/p. The count in the config is set with that in mind.

**On the test split**, at the reveal, for every arm. It is also computed **in-sample on D**, before
the reveal, and reported separately and labelled so. Every feature in a pool was selected by a model
fitted on those rows, or proposed by a language model shown examples from them, so the in-sample
number is anticonservative by an amount nothing here estimates. It bounds power from above, which
lets a reader tell "no association" from "not enough test rows".

**A weighted design is refused.** The statistic and its null count every row once, and a design
fitted with per-row weights needs a weighted statistic, which is not implemented. For such an arm
Tier 2 is not computed: the reason is in the arm's record and the log, the reveal computes no
Tier 2 for it, and the tables leave its columns empty. Tiers 1 and 3 fit with the weights.

## 5. Tier 3: necessity

For each feature of M\*, **M\* is refitted without it**, with the same model, rows, weights and seed.
For each cluster holding two or more of M\*'s features, M\* is also refitted without all of them,
because two near-duplicates each look unnecessary alone and are necessary together. Where one
feature or one cluster is the whole of M\*, the refit has no feature left: it predicts the fitting
rows' mean target, weighted as M\* was fitted, for every row. For a binary target that is the
positive rate, one half under balanced class weights. A DEFT arm's refit is then a tree of one
leaf, which predicts the same constant. The test then asks whether M\* beats that constant. The refits are saved before the reveal as standard cells (`loco/<label>/`), so the reveal
predicts and scores them like any other. M\* itself is the reduction's cell and is not refitted.

After the reveal, each refit is compared with M\* **row by row** in the loss the target's kind is
fitted by: d = loss(refit) - loss(M\*) per test row, with log-loss for a binary target
(probabilities clipped at 1e-12) and squared error for a continuous one. H0: E[d] <= 0 is tested
one-sided by a paired t-test, with a sign-flip permutation test of 10,000 flips beside it, because heavy-tailed
differences strain a t-test on a few hundred rows, and a squared error on a heavy-tailed target is
the most heavy-tailed of them. The change in the headline (AUROC, or R-squared) is reported beside
it. Across the family of refits, the t-test p-values are adjusted by Holm and by BH, and the
sign-flip p-values by Holm. A feature is **necessary** when its Holm-adjusted t-test p is below
0.05, and **substitutable** otherwise.
For a binary target the count of rows at the clip is reported: a pure tree leaf predicts 0 or 1, so
one wrong row can carry the mean, and then the sign-flip test is the one to read.

## 6. The checks

- **Global null.** The target replaced by a permutation of itself: Tier 2 on the test split for
  every arm (expect uniform p-values and almost nothing called), and Tier 1 on the check arm
  (expect nothing to be stable).
- **Planted signal**, on the check arm (`check_arm`): a target drawn from a linear predictor on
  `planted_count` real columns of the pool, alternating signs on standardised columns, the predictor
  scaled to `planted_signal_sd`. The columns are drawn from the candidates whose commonest value
  covers less than 0.9 of D, so that each varies enough to carry a signal, or from all of them
  where too few do. A binary target's labels are drawn through the logistic link, and a
  continuous target is the predictor plus unit Gaussian noise, so that at the shared 1.5 its
  R-squared is about 0.69. The recipe is saved with its kind, so the same model draws the target
  for D (Tier 1, and Tier 2 in-sample) and for the test split (Tier 2). The planted features keep the
  pool's correlations, so their near-duplicates are associated too. This check therefore measures
  recovery, and says nothing about the false-discovery rate.

## 7. When each part runs

| | what | how |
|---|---|---|
| before the reveal | Step 0 and the reproduction checks; Tier 2 in-sample; the LOCO refits; Tier 1 (real, and null and planted on the check arm); the arm's record | `run_reliability.py --run-dir <run> --arms <arm>`, run by the driver for each arm once its reduction has finished; refuses a revealed run. `--force` computes again what is on disk |
| at the reveal | the LOCO cells scored; Tier 2 on the test split (real, null, planted on the check arm) | `run_reveal.py`, which refuses to start while a profiled arm has no record |
| after it | Tier 3, the checks, the tables | `run_reliability.py --run-dir <run> --after-reveal`; refuses a run not yet revealed |

Each part before the reveal and at it writes its files and is skipped when they exist, so a panel
resumes. The part after the reveal is computed again on every call, from what the reveal wrote.
Every random stream is the configured `seed` plus an offset of its own.

## 8. What an arm leaves

```
reliability/<model>/<pool>/
    pool.npz                      a DEFT or an IGSR arm's candidates over D, as its method evaluated them
    prepare.json, prepare.npz     the pool, the prefilter, the clusters, the half-samples, the
                                  fitting rows, the labels on D; the column and reproduction checks
    planted.json                  the check arm's planted-signal recipe
    reproduction.json             a DEFT arm alone: what DEFT's side found in the two
                                  reproduction checks
    selections[_null|_planted].npz, tier1[_null|_planted].json/.npz
    tier2_discovery[_planted].json/.npz     in-sample
    loco/<label>/                 the refits, as standard cells
    reliability.json              the arm's record: what the reveal seals
    tier2_test[_null|_planted].json/.npz    at the reveal
    tier3.json, checks.json       after it
    profile_features.csv          one row per candidate: kind, prefilter, cluster, pi-hat at three
                                  levels, the association (auc or rho) and every Tier 2
                                  adjustment, the LOCO result in the kind's loss
    profile_clusters.csv          one row per cluster
    profile_curves.csv            |S_tau| and the bound per tau; features called per q and method;
                                  the plug-in FDR per cutoff
reliability/summary.csv           one row per arm
```

## 9. Settings

`reliability` in the config. `_shared.yaml` gives every config one, and every setting is required.

| key | meaning |
|---|---|
| `pairs` | B, complementary pairs: 2B selections per arm |
| `permutations` | label permutations behind Tier 2 |
| `cluster_rho` | the least \|Spearman rho\| within a cluster |
| `max_modal_share` | the prefilter's cut |
| `planted_count`, `planted_signal_sd` | the planted-signal check |
| `check_arm` | `model/pool` of the arm carrying Tier 1's checks, which the panel must profile; `l1` names the sparse linear model of the dataset's kind |
| `seed` | the base of every random stream |

Which arms are profiled is set by the config's `arms` entries (section 1).

`--workers` and `--memory-gb` are flags of `run_reliability.py` and not settings, because results
do not depend on them.

## 10. What it does not do

- **It does not change M\***, re-select anything, or tune anything on the test split. Acting on the
  panel after the reveal would consume the split it was measured on.
- **It does not estimate the cost of a pool's adaptivity.** Generated candidates were written by a
  model shown labelled training rows. Tier 2's null is exact given the pool, and does not correct
  for how the pool was chosen. Only new sequences settle that.
- **It is one seed per arm**, the reduction's. Tier 1 varies the rows and not the seed. The seed axis
  is covered by the suite's stability analysis.
- **Tiers 1 and 3 make no per-feature significance claim** in Tier 2's sense. A selection frequency
  and a leave-one-out comparison are statements about the model, and neither is a calibrated test of
  a feature's marginal association.
- **It does not merge values that differ only in their last bits.** The featuriser does that, where
  it can. A generated function can compute one value two ways (0.15 and 0.15000000000000002), and the
  prefilter, the clusters and Tier 2 compare values exactly. Our generated featuriser keeps its
  values to 12 significant digits (`SIGNIFICANT_DIGITS` in `seqmaestro/proposer/checks.py`, and the
  `significant_digits` argument of `GeneratedFeaturiser`), so such a value is one value to every
  model and rank. DEFT's generated
  features are not covered, because they are computed on DEFT's side as its tree computes them. For
  those, check how many distinct values a feature takes before quoting its modal share or its
  association to the third decimal.

## 11. Sources

| method | source | explanation |
|---|---|---|
| complementary pairs stability selection (Tier 1) | Shah and Samworth (2013), JRSSB, [doi:10.1111/j.1467-9868.2011.01034.x](https://doi.org/10.1111/j.1467-9868.2011.01034.x) ([arXiv:1105.5578](https://arxiv.org/abs/1105.5578)); cluster level: Faletto and Bien, [arXiv:2201.00494](https://arxiv.org/abs/2201.00494) | [Samworth's slides](https://statistique.cuso.ch/fileadmin/statistique/SamworthSlides.pdf) |
| single-feature AUC as Mann-Whitney U (Tier 2) | Hanley and McNeil (1982), Radiology, [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747) | [AUC meets the Wilcoxon-Mann-Whitney U statistic](https://www.r-bloggers.com/2017/03/auc-meets-the-wilcoxon-mann-whitney-u-statistic/) |
| permutation p-values with the +1 correction | Phipson and Smyth (2010), SAGMB, [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585) ([arXiv:1603.05766](https://arxiv.org/abs/1603.05766)) | [statmod::permp](https://rdrr.io/cran/statmod/man/permp.html) |
| BH and BY | Benjamini and Hochberg (1995), JRSSB, [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); Benjamini and Yekutieli (2001), [Annals of Statistics](https://doi.org/10.1214/aos/1013699998) | Efron and Hastie, *Computer Age Statistical Inference*, ch. 15, [free PDF](https://hastie.su.domains/CASI/) |
| permutation plug-in FDR | Storey and Tibshirani (2003), PNAS, [doi:10.1073/pnas.1530509100](https://doi.org/10.1073/pnas.1530509100) | [What are p-values, what are q-values?](https://www.nonlinear.com/progenesis/qi-for-proteomics/v4.2/faq/pq-values.aspx) |
| Westfall-Young min-P | Westfall and Young (1993), *Resampling-Based Multiple Testing*, Wiley | Dudoit, Shaffer and Boldrick, [preprint](https://biostats.bepress.com/ucbbiostat/paper110/) |
| label-blind prefiltering | Bourgon, Gentleman and Huber (2010), PNAS, [doi:10.1073/pnas.0914005107](https://doi.org/10.1073/pnas.0914005107) | [DESeq2 vignette](https://bioconductor.org/packages/devel/bioc/vignettes/DESeq2/inst/doc/DESeq2.html), independent filtering |
| leave-one-covariate-out (Tier 3) | Lei, G'Sell, Rinaldo, Tibshirani and Wasserman (2018), JASA, [doi:10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116) ([arXiv:1604.04173](https://arxiv.org/abs/1604.04173)) | Molnar, *Interpretable Machine Learning*, [Leave One Feature Out](https://christophm.github.io/interpretable-ml-book/lofo.html) |
| Holm | Holm (1979), Scand. J. Statist., [JSTOR](https://www.jstor.org/stable/4615733) | [Holm-Bonferroni, step by step](https://www.statisticshowto.com/holm-bonferroni-method/) |

The LOFO chapter matches Tier 3 closely: leave-one-out importance of strongly correlated features is
low, even negative, because it is conditional on what the other features carry. That is the
"substitutable" reading here.
