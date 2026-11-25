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
    README.md                     for a person; docs/developer/dataset.md lists its sections
    raw/                          the delivery exactly as received, with any note that came with it
    <splits>/train.csv, val.csv, test.csv
    <splits>/groups.csv           which group each sequence is in, when the split is grouped
    <splits>/dataset.json         written by finalise_dataset.py
experiments/prepare/declarations/<date>_<dataset>__<splits>.yaml    what you decided
experiments/prepare/build_<dataset>...py, prepare_<dataset>_split.py  your builder, if you wrote one
experiments/datasets/<task>.json                                      what the models are told
experiments/configs/<name>.yaml                                       the run config
```

`docs/developer/dataset.md` explains the declaration, `dataset.json` and the checks, and `validate_declared` and
`run_checks` in `seqmaestro/dataset.py` list the declaration's fields and the checks.

**Everything listed above belongs to this dataset and stays on this machine.** Git ignores the
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
| what could a split of this size detect? | `experiments/prepare/design_power.py` while planning, and `finalise_dataset.py` records it for the split you build |
| a record of the grouping, for a grouped split | `experiments/prepare/write_group_map.py`, before finalising |
| AlphaGenome's predictions as covariates, for elements with genomic coordinates | `experiments/prepare/predict_alphagenome_summaries.py`, in `SeqMaestro_Method_alphagenome`. Prefer `--model held_out`, which predicts each element with a model whose training excluded its locus, and needs `--folds`, AlphaGenome's file of regions and folds. `--context native` also shows the model the element's genomic neighbourhood, so its covariates include information from outside the element |
| counts, checks, design power, `dataset.json` | `experiments/prepare/finalise_dataset.py --splits <dir> --declared <declaration>` |
| a published motif collection, for a run that scans for known motifs or compares discovered ones with it | `experiments/prepare/fetch_motif_collection.py` |

**Use the example dataset as a worked example.** `experiments/prepare/build_example_dataset.py`
builds a small synthetic dataset through the steps a real builder takes, and its declaration,
description and config sit where yours go (`experiments/prepare/README.md` lists them). Build it
and finalise it once to see what this stage leaves behind. Copy its shape but none of its
decisions, because each decision below depends on how your data was produced.

## Decide and write down why

Each of these goes in the declaration and, with its reasoning, in the dataset README.

1. **What the target is.** First its kind (`target.kind`): **binary** (a label of 0 or 1) or
   **continuous** (one number per row). Decide it from what was measured, never from the values. A
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
2. **What one row is.** Each row should be one independent observation. Where the delivery repeats
   one event (the members of a clonal family, the same site measured twice), collapse it, and
   record what the collapse removed.
3. **The measurements beside the sequence**, where the delivery has any: a signal over the window,
   an experimental measurement. Decide with the user which become covariates. For each, write what
   it measures in the declaration (`covariates.<name>.description`), in the words a domain expert
   would use. Every model is told it, so on a blinded dataset write it without naming what is
   withheld. A covariate must be a number on every row, so decide with the user whether a missing
   value is imputed (and how) or its row filtered, and record either as a step. Ask too whether a
   measurement could give the answer away, for example one taken after the outcome, since a model
   would lean on it and say nothing about the sequence. Name each column as a legal Python
   identifier. A column you keep in the split files for provenance and do not declare is listed by
   the finaliser as unused.
4. **The window and its anchor.** The window is usually of a fixed length, cut at a fixed distance
   from a landmark chosen for biological reasons, so that a position means the same kind of site in
   every row. For each alternative you considered, say how many rows it would lose, class by class
   for a binary target and across the target's range for a continuous one: a window that drops a
   third of one class and a twentieth of the other changes the task. Where the whole sequence
   matters, such as a whole transcript, the sequences may keep their different lengths
   (`sequences.length: variable`). Positional features then need a frame of their own: declare a
   positional window, its length and the landmark it is anchored to (`start` or `end`). **Whether
   shorter sequences are padded is the user's decision, so put it to them.** Padding keeps every
   row, and lets a model infer a sequence's length from where the padding begins. Without it,
   filter out the sequences shorter than the window, as a recorded step, so that every arm fits on
   the same rows. Say what each choice gives up, class by class.
5. **The orientation**, and how you verified it. The later stages cannot detect a
   reverse-complemented dataset, and every feature built on one still looks interpretable.
   `seqmaestro.qc.orientation` compares positional profiles, so it needs sequences of one length.
   On sequences of different lengths, run it on the positional window, over the sequences at least
   that long.
6. **Every filter**, in order, with the rows it removed (from each class, for a binary target). A
   filter that removes a sliver of one dataset can remove a third of the next, so never discard many
   rows silently. Check also that each filter measures the variable it names.
7. **The grouping.** What makes two rows dependent here (a gene, a domain, a clonal family, an
   identical window), and the split grouped on all of those relations together. Say what the
   stratification balances: the classes of a binary target, or bins of a continuous one. Record the
   assignment of groups to splits.
8. **What the models are told.** Write the task description in `experiments/datasets/<task>.json`:
   what is predicted, from what, and what the positions mean. Decide with the user whether to
   withhold the identity of the factor under study. If so, it goes in `judge_only` and nowhere
   else: neither a column name nor a covariate description may contain a word of it, since the
   models are shown both. The run's config then states `dataset.blinded: true`, and the driver
   refuses a run in which a withheld word appears in a column name or a covariate description.
   Whether to guide the form of the features the models write is asked at the checkpoint (below).
9. **Whether the run takes the motif sources.** Where the question is about binding by
   transcription factors or RNA-binding proteins, motifs can be feature sources
   (`experiments/configs/README.md`, Motifs): the known motifs (every motif of a published
   collection) and the discovered motifs, which STREME finds between the classes of the training
   rows, or between the rows at the two ends of a continuous target. Decide with the user whether
   to add them, which of the two, and with which collection: JASPAR's for the transcription factors
   of the dataset's taxon where the sequences are DNA, or CisBP-RNA's for the RNA-binding proteins
   of its species where they stand for RNA. Fetch it with
   `experiments/prepare/fetch_motif_collection.py`. The dataset's config then extends
   `_motifs.yaml` beside the shared config and states `strands` for each source it keeps (`both`
   for DNA, `given` for RNA), `sources.motif_known.collection`, and, where the user wants each
   discovered motif compared with the collection, `sources.motif_discovered.annotate_with`. A
   source the run leaves out is set to `null`, with its arms. Where the sequences stand for RNA,
   both sources state `p_threshold: 1.0e-3`, since the motifs of RNA-binding proteins are too short
   to count often at the 1e-4 that `_motifs.yaml` sets. Any other setting of `_motifs.yaml` can be
   changed the same way, with the user.

   Motif discovery runs the MEME Suite in `SeqMaestro_Method_meme`. Check that environment with
   `./install.sh --check --with-motifs`, and where it is missing, ask the user to build it with
   `./install.sh --with-motifs`, saying that the MEME Suite is free for non-profit use only. Say
   what each source adds. A known motif is named after its protein, so it brings what is known
   about that protein into the run, and a blinded run refuses it in a pool with a written source. A
   discovered motif is found in the training rows with their labels, and is named by its consensus.

A confound you measure because it drives one of these decisions (for example, a length that
separates the classes and is the reason for cropping) is recorded as the reason for that decision.

## Do not

- **Compute baselines or model performance.** The composition baseline measures how far base
  composition alone separates the classes, at the start of Stage 2 and in the same way for every
  dataset.
- **Put the analysis that chose the dataset in its README.** Whoever writes the report reads the
  README, so a note of what the answer is expected to be would prime the report. That record goes in
  the session notes instead.
- **Look at the held-out split beyond its counts.** Once the split exists, the held-out rows are for
  the reveal.

## Before you hand over: the checkpoint

1. Run `write_group_map.py` if the split is grouped, then `finalise_dataset.py` with your
   declaration. Every blocking check must pass, or be accepted in the declaration under
   `acknowledged` with a reason you would defend to the user.
2. **Answer every advisory finding, and answer it with the user.** These are the checks that measure
   a property but cannot judge it: identical sequences appearing in more than one split; on a
   dataset whose sequences differ in length, how those lengths relate to the target; and for a
   continuous target, a distribution whose tails or single commonest value hold an unusual share of
   the rows. Take the first as an example. Whether that repetition is a leak or the shape of the
   data depends on how the data was produced, and a wrong answer is expensive in either direction.
   If a sound dataset is treated as defective, it gets redrawn for nothing. If a leak is treated as
   normal, every held-out number afterwards is partly memorisation, and the report never says so.
   Do not answer it from the code alone. Bring the user the measurement the check reports, what you
   found in the delivery and the build about where the repetition comes from, and what removing it
   would lose. Their answer goes in the declaration under `acknowledged`, where it stays with the
   dataset and is printed in the report.
3. Check that the run config resolves and passes the preflight checks:
   `experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --dry-run`,
   with the credentials exported first (`set -a && source .env && set +a`) where a pool holds a
   written source or the judge runs, since the dry run checks for them.
   On a dataset with covariates, the covariates are a source like any other: an arm is given them
   when its pool names them, as in `positional+covariates`. Running a pool with and without them
   shows what the measurements add, but doubles that pool's cost, so choose with the user the pools
   whose comparison the question needs (`experiments/configs/README.md`, Arms). A config that gives
   them to no pool gets a warning, which `--estimate` prints and a launch writes at the head of
   `logs/driver.log`, whereas the dry run does not show it. On sequences of different lengths, set
   `sources.kmer.normalise` with the user, since counts grow with a sequence's length.
4. **Ask the user whether to guide the form of the features the language models write.** By
   default there is no guidance: the models that write features (the one-round and iterative
   sources, DEFT's generative finder and IGSR) write whatever forms the task and the rows suggest
   to them. Guidance is a few sentences from the user, which each of them is shown word for word.
   Say that it is optional, and show what is meant with examples such as these:
   - a functional form: "Write each feature as a ratio or a difference of two counts";
   - a stretch of sequence: "Measure only within 50 nt of either splice site";
   - a symmetry: "Count each motif together with its reverse complement";
   - a kind of feature: "Prefer counts of short words to measures of overall composition";
   - a known constraint: "No feature should depend on the length of the sequence".

   Say what it changes: guidance steers the search towards the forms it names, so the run is less
   likely to find what lies outside them, and the report states it. On a blinded run it must name
   nothing withheld, and the driver refuses a run whose guidance does. A task with a hand-written
   DEFT prompt config cannot take guidance on DEFT's arms. Write the answer in the config as
   `grid.generated_features.guidance`: the user's words, or `null` for none, as the shared config
   has it.
5. Present the user with a summary: the finaliser's output, each decision above with what it gave
   up, what the models are told and what is withheld, the guidance on the form of the features or
   that there is none, the design power, every advisory finding with your interpretation of it, and
   **the questions only the people who supplied the data can answer**, each with what would change
   depending on the answer. Put first any question whose answer would invert every direction in a
   report.
6. **Wait for the user's approval before Stage 2.** If they supply more data or clarify something,
   rebuild, finalise again and present again. Which arms the run fits, any change to their
   settings and how many processes may call the language model at once are decided at the launch,
   from the estimate of each arm's time and cost ([`suite.md`](suite.md)).

## Beyond this file

This file is a floor. Datasets arrive in shapes nobody anticipated, and the most important decision
in a given delivery may not be listed here at all. Follow what you find, record it, and bring it to
the user.
