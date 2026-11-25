"""The analyses of a run's Stage 2 cells, which the driver runs once every arm has finished.

Each reads the cells under a run's `results/` as it finds them. The first three write their tables
beside the cells, whereas the judge writes its scores under the run's `judge/`. They fit nothing,
and they do not read the held-out split.

| module | question |
|---|---|
| `stability` | does one arm select the same features when the seed changes, beyond chance? |
| `co_discovery` | do several arms, pointed at the same data, reach the same properties? |
| `recurrence` | does an arm on a written source arrive at the same ideas across seeds? |
| `judge` | how significant would each feature be, if it were true? |

A language model groups the features for `recurrence` and scores them for `judge`, so both need
credentials. The judge runs only where the run's config turns it on.

One more analysis is run by hand, on the feature files of the cells of interest and a table of the
dataset: `direct_effect`, which tests whether one feature differs between the classes on its own,
by how much, and whether a difference of that size could be chance.

`experiments/analysis/README.md` says how the first three differ, and `docs/developer/analyses/`
says what each computes.
"""
