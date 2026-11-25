"""Stage 3B: the reliability panel, three independent pieces of evidence about each feature of an
arm's reduced model and each candidate it could have used instead.

- `tier1`: stability, by complementary pairs subsampling of the training and validation rows;
- `tier2`: association with the target on its own, on the held-out split, calibrated by permutation;
- `tier3`: necessity, by refitting the reduced model without each feature (LOCO) and comparing the
  two row by row;
- `prefilter`: Step 0, which drops near-constant candidates and clusters near-duplicates, on the
  training and validation rows and without labels;
- `checks`: the global-null and planted-signal reruns;
- `multitest`: the multiplicity corrections;
- `pool`: an arm's candidates;
- `selections`: Tier 1's resumable parallel sweep (numpy only, so that DEFT's side can share it);
- `fits`: the fits made in this process, for an arm fitted in Core and for an IGSR arm;
- `panel`: one arm's panel, part by part, before the reveal, at it and after it;
- `files`: the names of the files the panel writes;
- `tables`: the tables and the per-arm summary assembled from those files.

The stage itself is `seqmaestro/pipeline/reliability.py`, which `experiments/runners/run_reliability.py`
calls, and the reveal computes the panel's parts on the held-out split. `docs/developer/reliability.md`
describes what is computed and why.
"""
