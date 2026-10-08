# docs: what this system computes, and why each number is arrived at that way

These pages are maintained. Each page about a stage, a record or an analysis carries a banner
naming the code it tracks, and the rule is: if you change what something computes, change its page
in the same commit.

They hold the reasoning behind the numbers, which does not fit in a docstring. A docstring says what
one function does. These pages say how a figure in a report was arrived at, what it is comparable
with, and where it breaks.

| page | |
|---|---|
| [`repository.md`](repository.md) | how the repository is laid out, and where each kind of fact is written down |
| [`dataset.md`](dataset.md) | what Stage 1 hands over: the prepared dataset, its `dataset.json`, and the checks every dataset has to pass |
| [`heldout.md`](heldout.md) | how the test split stays closed until one reveal, and how the reveal scores every saved model on it |
| [`suite.md`](suite.md) | Stage 2: what a cell is, how its rows and labels are drawn, how it is scored, and how each kind of arm is fitted |
| [`baseline.md`](baseline.md) | the composition baseline every arm is read against: what it fits, and why it is fixed |
| [`reduction.md`](reduction.md) | Stage 3A: each arm cut to a model of N features by recursive elimination, and the three models it keeps |
| [`reliability.md`](reliability.md) | Stage 3B: stability, association and necessity for every candidate feature of a reduced model, with the literature behind each |
| [`report/README.md`](report/README.md) | the report: what its build decides (the headline set, the reference line, concepts), the checks behind every drawn model, and the template and components a report is written with |
| [`feature_record.md`](feature_record.md) | what we record about every feature, field by field: the single record type every arm reports in |
| [`analyses/README.md`](analyses/README.md) | how the post-run analyses work: the shared machinery, and an index of the rest |
| [`analyses/stability_and_co_discovery.md`](analyses/stability_and_co_discovery.md) | does one arm find the same features again; do several arms find the same thing |
| [`analyses/recurrence.md`](analyses/recurrence.md) | does an arm arrive at the same *ideas* again, when it renames them every time |
| [`analyses/judge.md`](analyses/judge.md) | how biologically significant would each feature be, if it were true |
| [`analyses/direct_measurement.md`](analyses/direct_measurement.md) | what one feature does, with no model in the way |
| [`analyses/direction.md`](analyses/direction.md) | which class a feature argues for, in the two senses of that |

## What belongs here, and what does not

**Here:** how a number is computed, why that way and not the obvious alternative, what it is
comparable with, and the failure modes a reader should check before quoting it.

**Not here:** what a particular run measured. A finding about a dataset goes in that run's
report or the dataset's `README.md`, the same rule the code follows. Worked examples on these
pages are labelled as illustrations of the arithmetic.

**Also not here:** which script to run and when. That is in `experiments/analysis/README.md`, which
links to these pages instead of repeating them.

## Related records kept elsewhere

| | |
|---|---|
| `experiments/README.md` | how a run is organised, and the contracts every stage follows |
| `experiments/analysis/README.md` | which analysis answers which question, and when to reach for it |
| `experiments/adapters/README.md` | the bundle contract for an arm running in another environment |
| `experiments/datasets/README.md` | how a dataset describes itself to the models that read it |
| `ENVIRONMENTS.md` | which conda environment runs what, and the constraints keeping them apart |
