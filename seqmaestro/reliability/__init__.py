"""Stage 3B: the reliability panel, which gathers three independent pieces of evidence about each
feature of an arm's reduced model and each candidate it could have used instead.

- **Tier 1, stability** (`tier1`): would the arm's selection have chosen the feature on another
  half of the data? Complementary-pairs subsampling of train + val, with a bound on the expected
  number of chance-level features above a threshold, at feature, cluster and position level.
- **Tier 2, association** (`tier2`): does the feature go with the target on its own, on the test
  split, separating the classes or moving a continuous target? A permutation-calibrated rank
  association (an AUC, or Spearman's rho), with BH, BY, Holm, Westfall-Young and a plug-in FDR.
- **Tier 3, necessity** (`tier3`): does the reduced model get worse on the test split without it?
  Leave-one-covariate-out refits, compared row by row.

The other modules:

- `prefilter`: Step 0, which drops near-constant candidates and clusters near-duplicates, on
  train + val and without labels;
- `checks`: the global-null and planted-signal reruns;
- `multitest`: the multiplicity corrections;
- `pool`: an arm's candidates;
- `selections`: Tier 1's resumable parallel sweep (numpy only, shared with the vendored tree's
  side);
- `fits`: the fits made in this process, which are our own arms' and an IGSR arm's;
- `panel`: one arm's panel, part by part, before the reveal, at it and after it;
- `files`: the names of the files the panel writes;
- `tables`: the tables and the per-arm summary assembled from those files.

The stage is `seqmaestro/pipeline/reliability.py`, which `experiments/runners/run_reliability.py`
calls. The reveal computes the panel's test-split parts, and `docs/reliability.md` describes what
is computed and why.
"""
