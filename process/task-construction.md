---
name: task-construction
stage: 1
---

# Stage 1. Turn the delivered data into a prediction task

You have been given data by people who know the biology and may not know how a sequence model will
treat it. Your job is to turn it into a set of sequences with a target, split so the held-out score
means something, and described so that everyone downstream (the models, the report, the user)
understands it the same way. The later stages can only find what this one makes findable, so the user
signs off what you build before anything is fitted.

Read [`README.md`](README.md) in this folder first if you have not.

## What this stage must leave behind

```
data/<date>_<dataset>/
    README.md                     for a person; docs/dataset.md lists its sections
    raw/                          the delivery exactly as received, with any note that came with it
    <splits>/train.csv, val.csv, test.csv
    <splits>/groups.csv           which group each sequence is in, when the split is grouped
    <splits>/dataset.json         written by finalise_dataset.py
experiments/prepare/declarations/<date>_<dataset>__<splits>.yaml    what you decided
experiments/prepare/build_<dataset>...py, prepare_<dataset>_split.py  your builder, if you wrote one
experiments/datasets/<task>.json                                      what the models are told
experiments/configs/<name>.yaml                                       the run config
```

`docs/dataset.md` defines the declaration, `dataset.json` and every check.

**Everything listed there belongs to this dataset and stays on this machine.** Git ignores the
builder, the declaration, the description and the config as it ignores `data/`, so an update of
SeqMaestro leaves them alone and they are never committed. A test of your builder goes in
`tests/test_local_<dataset>.py`, which git ignores too. `experiments/prepare/README.md` says why.

## Use these rather than writing your own

| need | use |
|---|---|
| is the strand convention what the delivery claims? | `seqmaestro.qc.orientation`, and `experiments/prepare/check_sequence_provenance.py` for data with coordinates |
| are windows cut from the introns beside an exon in the transcript's orientation? | `seqmaestro.qc.splice_site_check`, on the dinucleotides at the two junctions |
| a split that keeps related rows together | `seqmaestro.grouping.connected_groups`, over every relation that makes two rows dependent |
| does a grouped split isolate what it promises, for each kind of feature? | `experiments/analysis/feature_group_dependence.py` |
| what could a split of this size detect? | `experiments/prepare/design_power.py` while planning; `finalise_dataset.py` records it for the split you build |
| a record of the grouping, for a grouped split | `experiments/prepare/write_group_map.py`, before finalising |
| AlphaGenome's predictions as covariates, for elements with genomic coordinates | `experiments/prepare/predict_alphagenome_summaries.py`, in `SeqMaestro_Method_alphagenome`. Prefer `--model held_out`, which predicts each element with a model whose training excluded its locus, and needs `--folds`, AlphaGenome's file of regions and folds. `native` context also shows the model the element's genomic neighbourhood, so its covariates carry information from outside the element |
| counts, checks, design power, `dataset.json` | `experiments/prepare/finalise_dataset.py --splits <dir> --declared <declaration>` |

**The example dataset is the worked example.** `experiments/prepare/build_example_dataset.py`
builds a small synthetic dataset through the steps a real builder takes, and its declaration,
description and config sit where yours go (`experiments/prepare/README.md` lists them). Build it
and finalise it once to see what this stage leaves behind. Take its shape and none of its
decisions: each decision below depends on how your data was produced.

## Decide, and write down why

Each of these goes in the declaration and, with its reasoning, in the dataset README.

1. **What the target is.** First its kind (`target.kind`): **binary**, a label of 0 or 1, or
   **continuous**, one number per row. Decide it from what was measured, never from the values. A
   continuous measurement thresholded into two classes discards its middle, so ask whether the
   delivery has the number before a threshold was applied.
   - *Binary.* Which rows are class 1 and which class 0, the rule that assigns them, and a short name
     and one-line meaning for each. Say whether both classes come from one experiment, and what the
     negative class is: observed, constructed, or matched. If the delivery leaves the direction of
     the labels ambiguous, say so and ask.
   - *Continuous.* A short name (`target.name`) and what the number measures, in which units, and
     what a higher value means (`target.meaning`), so a report can say which way a feature moves
     it. Ask how it was measured and what makes it noisy (a low count, a detection limit). A
     transformation or a filter on it (a log, a clip, a count threshold) is a preparation step,
     agreed with the user and recorded like any filter, so that every arm fits the same target.
2. **What one row is.** One independent observation. Where the delivery repeats one event (the
   members of a clonal family, the same site measured twice), collapse it, and record what the
   collapse removed.
3. **The measurements beside the sequence**, where the delivery has any: a signal over the window,
   an experimental measurement. Decide with the user which become covariates. For each, write what
   it measures in the declaration (`covariates.<name>.description`), in the words a domain expert
   would use; every model is told it, so on a blinded dataset write it blind. A covariate must be a
   number on every row, so decide with the user whether a missing value is imputed, and how, or its
   row filtered, and record either as a step. Ask too whether a measurement could carry the answer,
   for example one taken after the outcome, since a model would lean on it and say nothing about the
   sequence. Name each column as a legal Python identifier. A column you keep in the split files
   for provenance and do not declare is listed by the finaliser as unused.
4. **The window and its anchor.** Usually a fixed length, cut at a fixed distance from a landmark
   chosen for biological reasons, so that a position means the same kind of site in every row. For
   each alternative you considered, say what it would cost in rows, class by class for a binary
   target and across the target's range for a continuous one: a window that drops a third of one
   class and a twentieth of the other changes the task. Where the whole
   sequence matters, such as a whole transcript, the sequences may keep their different lengths
   (`sequences.length: variable`). Positional features then need a frame of their own: declare a
   positional window, its length and the landmark it is read from (`start` or `end`). **Whether
   shorter sequences are padded is the user's decision; put it to them.** Padding keeps every row,
   and lets a model read a sequence's length from where the padding begins. Without it, filter the
   sequences shorter than the window out, as a recorded step, so that every arm reads the same rows.
   Say what either choice costs, class by class.
5. **The orientation**, and how you verified it. Nothing downstream can detect a reverse-complemented
   dataset, and every feature built on one still looks interpretable. `seqmaestro.qc.orientation`
   compares positional profiles, so it needs one length. On sequences of different lengths, run it
   on the cut the positional window makes, over the sequences at least that long.
6. **Every filter**, in order, with the rows it removed (from each class, for a binary target). A
   filter that removes a sliver of one dataset can remove a third of the next, so never discard many
   rows silently. Check also that each filter measures the variable it names.
7. **The grouping.** What makes two rows dependent here (a gene, a domain, a clonal family, an
   identical window), and the split grouped on all of those relations together. Say what the
   stratification balances: the classes of a binary target, or bins of a continuous one. Record the
   assignment of groups to splits.
8. **What the models are told.** Write the description in `experiments/datasets/<task>.json`: what is
   predicted, from what, and what the positions mean. Decide with the user whether to withhold the
   identity of the factor under study. If so, it goes in `judge_only` and nowhere else: no column
   name and no covariate description may hold a word of it either, since the models are shown both.
   The run's config then states `dataset.blinded: true`, and the driver refuses the run where a
   withheld word shows in a column name or a covariate description.

A confound you measure because it drives one of these decisions (for example, a length that
separates the classes and is the reason for cropping) is recorded as the reason for that decision.

## Do not

- **Compute baselines or model performance.** How far base composition alone separates the classes
  is measured by the composition baseline at the start of Stage 2, the same way for every dataset.
- **Put the analysis that chose the dataset in its README.** Whoever writes the report reads the
  README, so a note of what the answer is expected to be would prime the report. That record goes in
  the session notes instead.
- **Look at the test split beyond its counts.** Once the split exists, the held-out rows are for the
  reveal.

## Before you hand over: the checkpoint

1. Run `write_group_map.py` if the split is grouped, then `finalise_dataset.py` with your
   declaration. Every blocking check must pass, or be accepted in the declaration under
   `acknowledged` with a reason you would defend to the user.
2. **Answer every advisory finding, and answer it with the user.** These are the checks that measure
   a property but cannot judge it: identical sequences appearing in more than one split; on a
   dataset whose sequences differ in length, how those lengths go with the target; and for a
   continuous target, a distribution whose tails or single commonest value hold an unusual share of
   the rows. Take the first as the example. Whether that repetition is a leak or the shape of the
   data depends on how the data was produced, and a wrong answer is expensive in either direction.
   If a sound dataset is treated as defective, it gets redrawn for nothing. If a leak is treated as
   normal, every held-out number afterwards is partly memory, and the report says so nowhere. Do not
   answer it from the code alone. Bring the user the measurement the check reports, what you found
   in the delivery and the build about where the repetition comes from, and what it would cost to
   remove it. Their answer goes in the declaration under `acknowledged`, where it travels with the
   dataset and is printed in the report.
3. Check the run config resolves and preflights:
   `experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --dry-run`,
   with the credentials exported first (`set -a && source .env && set +a`) where a pool holds a
   written source or the judge runs, since the dry run checks for them.
   On a dataset with covariates, the covariates are a source like any other: an arm is given them
   when its pool names them, as in `positional+covariates`. Running a pool with and without them
   shows what the measurements add, and doubles that pool's cost, so choose with the user the pools
   whose comparison the question needs (`experiments/configs/README.md`, Arms). A config that hands
   them to no pool gets a warning, which `--estimate` prints and a launch writes at the head of
   `logs/driver.log`. The dry run does not show it. On sequences of different lengths, set
   `sources.kmer.normalise` with the user, since counts carry a sequence's length.
4. Present the user with a summary: the finaliser's output, each decision above with what it cost,
   what the models are told and what is withheld, the design power, every advisory finding with
   your reading of it, and **the questions only the people who supplied the data can answer**, each
   with what would change depending on the answer. Put first any question whose answer would invert
   every direction in a report.
5. **Wait for the user's approval before Stage 2.** If they supply more data or clarify something,
   rebuild, finalise again and present again.

## Beyond this file

This file is a floor. Datasets arrive in shapes nobody anticipated, and the most important decision
in a given delivery may be one listed nowhere here. Follow what you find, record it, and bring it to
the user.
