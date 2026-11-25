# Terminology

This page defines, in alphabetical order, the terms used throughout this documentation. Where a config or the code names a thing differently, that name follows in brackets. A finished report is written for readers outside the project, so where it uses a different word, the report's word is given as well.

## Adapter

The script that runs a [vendored method](#vendored-method) for SeqMaestro, under the method's own interpreter. It fits or predicts from files that SeqMaestro writes, and hands the result back as files: for a Stage 2 [cell](#cell), a [bundle](#bundle). The adapters are in `experiments/adapters/`. See [the adapters](developer/adapters.md).

## Agent

The AI agent that carries out a [run](#run) with the user, working in an [agent harness](#agent-harness) such as Claude Code or Codex. After preparing the dataset with the user in Stage 1, it launches the [driver](#driver), checks each stage and writes the report. The run records the language model the agent ran on (in `agents.json`).

## Agent harness

The program an [agent](#agent) works in, such as Claude Code, Codex, OpenHands or Claude Science. The harness gives the agent its language model and runs the agent's commands on the user's machine. `./agent.sh` starts any of the four in the repository's folder ([setting up an agent](user/agents.md)). The user guide also calls it the *harness*.

## Arm

One model fitted on one [pool](#pool) of feature sources, written `model/pool`: `ebm/kmer`, for example, is an explainable boosting machine fitted on k-mer counts. Each arm is fitted at every [seed](#seed) under both [label conditions](#label-condition). A report calls an arm a *configuration*.

## Baseline

The composition baseline: a sparse linear model on the fraction of each base in the sequence, fitted at every seed on the same rows as the arms. Every arm is compared with it, and a report draws its score as the [reference line](#reference-line). See [the composition baseline](developer/baseline.md).

## Black-box model

A model that can be interpreted only through its predictions, such as XGBoost. Its arms are fitted and reduced like any others, as a reference for how much signal the data holds, but the [reliability panel](#reliability-panel) leaves them out.

## Blinding

Withholding a fact of the [task description](#task-description), such as the name of the protein under study, from every language model that writes features. The fact is kept in the description's `judge_only` field, which only the [judge](#judge) is told. A run records whether it is blinded (`dataset.blinded`). Separately, and in every run, the judge is not told which arm produced the feature it rates ([the judge](developer/analyses/judge.md)).

## Bundle

The directory of files in which an [adapter](#adapter) hands back one fit of a [vendored method](#vendored-method): its predictions on the validation rows, the [feature records](#feature-record) of the features it used with their values, and the fitted model. SeqMaestro checks the bundle and scores it with the same code as every other [cell](#cell). See [the adapters](developer/adapters.md).

## Call

One process of a [run](#run), such as the fit of one [cell](#cell), one arm's [reduction](#reduction) or the [reveal](#reveal). Its log, its record of the command, the commit and the time, and its record of failures share its name.

## Cell

One fit: one [arm](#arm) under one [label condition](#label-condition) at one [seed](#seed). Its directory, `results/<model>/<pool>/<labels>/seed_<n>/`, holds what the fit recorded and how it scored. See [Stage 2: the suite](developer/suite.md).

## Checkpoint

A point at which the [agent](#agent) stops and waits for the user's decision before going on, such as the approval of the prepared task at the end of Stage 1 and the agreement to the [reveal](#reveal). The [status page](#status-page) shows the checkpoint that ends each stage.

## Command line

The command `seqmaestro`, which takes one table of sequences through a [run](#run) without an [agent](#agent). It prepares the dataset with fixed defaults, writes the [config](#config), starts the [driver](#driver), follows the run and has the language model write the report. See [the command line](user/command-line.md).

## Concept

A group of features from different arms whose values agree closely (by rank correlation), so that they count as one property found under different names. Concepts group features by their values, whereas the [recurrence analysis](#recurrence-analysis) groups written features into ideas by asking a language model.

## Config

The YAML file that defines a [run](#run): the dataset, the seeds and label conditions, how the language model is called, the arms, and the settings of each stage. A dataset's config builds on (`extends`) one of the shared configs in `experiments/configs/`. See [run configs](developer/configs.md).

## Core

SeqMaestro's own conda environment, `SeqMaestro_Core`, in which its library, its stages and most of its models run. A Core model, or a Core arm, is fitted there, whereas a [vendored method](#vendored-method) runs in an environment of its own. See [environments](developer/environments.md).

## Covariate

A number measured for each sequence and supplied with the dataset, such as a chromatin signal. A [pool](#pool) can include the covariates as a [feature source](#feature-source) (`covariates`), whose features are the covariates as they are.

## Declaration

The YAML file that records how a dataset was prepared in Stage 1: its columns, its split, each decision taken and the user's answer to each check that needs one. Stage 1 checks the prepared dataset against it before writing the dataset's record (`dataset.json`). See [Stage 1: the prepared dataset](developer/dataset.md).

## Driver

The program that carries out a run as its [config](#config) describes, from the [baseline](#baseline) to the tables and figures the report is written from ([`experiments/runners/run_experiment.py`](../experiments/runners/run_experiment.py)). It starts each stage once the stages it depends on have finished.

## Feature

A number computed from a sequence, such as the fraction of C or T in its first 100 bases. Every arm reports the features its model used as [feature records](#feature-record).

## Feature record

What every arm records about each feature of its model: its name, what it computes, how much the model used it, which way it points, and its association with the target on its own. See [the feature record](developer/feature_record.md).

## Feature source

One way of producing the features a model chooses from. The fixed sources compute features without a language model: positions (`positional`), k-mers (`kmer`), known motifs (`motif_known`), discovered motifs (`motif_discovered`) and the [covariates](#covariate). The other sources are [written sources](#written-source).

## Fitting rows

The training rows that one [cell](#cell) is fitted on: a share of the training split (`grid.train_fraction`), drawn at the cell's [seed](#seed). Every arm and the [baseline](#baseline) fit on the same rows at a seed, and a featuriser that learns from the data, such as that of the discovered motifs, learns from these rows alone.

## Guidance

A few sentences from the user on the form the features of the [written sources](#written-source) should take, such as a functional form or a stretch of sequence to measure in. Every language model that writes features is shown the guidance word for word. A run has none unless the user asks for it (`grid.generated_features.guidance`).

## Headline metric

The metric that ranks arms and seeds: AUROC for a binary target, and R-squared for a continuous one. The [reduction](#reduction) takes each arm's best seed by it, and the report quotes it for every model.

## Headline set

The arms a report leads with: the best arm of each interpretable model class, the best interpretable arm whose pool has no written source, and the strongest black-box model as a reference. See [the report](developer/report.md).

## Held-out split

The part of the data (the `test` split) that no stage uses to fit a model or to choose anything. It stays closed until the [reveal](#reveal). See [the held-out split and the reveal](developer/heldout.md).

## Iterative source

The [written source](#written-source) whose language model writes features over several rounds, and is shown in each round how a fit of the arm's model used the features written so far (`llm_generated_iterative`).

## Joint source

The [written source](#written-source) whose features a language model writes inside a model's fit: DEFT's candidate features at each split of its tree, and IGSR's terms (`llm_generated_joint`).

## Judge

The language model that rates how meaningful each of the most important features would be if its effect held. It runs where the config sets `grid.judge`. See [the judge](developer/analyses/judge.md).

## Label condition

Whether a [cell](#cell) is fitted on the labels as they are (`real`) or on the training labels shuffled (`permuted`). The permuted cells are the control that every number is compared with. A report calls them the *shuffled-label control*.

## Language model

The model a run calls to write features, to judge them and to group them into ideas. One model, which the file `.env` names, serves every part of a run, and the config's `language_model` section states how it is called. The [agent](#agent), by contrast, runs on a model of its own, set up in its [agent harness](#agent-harness).

## Motif

A description of the sites that a protein binds, held as a position frequency matrix: how often each base occurs at each position of a site. The known motifs come from a published collection such as JASPAR, and the discovered motifs are those that STREME finds in a cell's [fitting rows](#fitting-rows). A motif feature measures how strongly, or how often, one motif matches a sequence. See [motifs](developer/motifs.md).

## Pool

The [feature sources](#feature-source) an arm's model may choose its features from, written with `+` between them, such as `positional+kmer`.

## Positional window

The stretch of each sequence in which positional features are measured when a dataset's sequences differ in length: a number of bases counted from the start or the end of each sequence (`sequences.positional_window` in the [declaration](#declaration)). A longer sequence is truncated, and a shorter one is padded or filtered out, as the user decides in Stage 1. See [Stage 1: the prepared dataset](developer/dataset.md#variable-lengths-and-the-positional-window).

## Proposal

The features that the [proposer](#proposer) writes for one [cell](#cell), from the rows and labels the cell fits on. The run keeps it under `proposals/`, keyed on everything that would change it, so a cell fitted again makes no request twice.

## Proposer

The part of SeqMaestro that asks a language model for features as short Python functions of the sequence, checks the code, and keeps the functions that run. It produces the features of the one-round and the [iterative](#iterative-source) written sources.

## Recurrence analysis

The analysis that asks a language model to group an arm's written features into ideas, and counts how often each idea is found again across seeds and across arms. See [recurrence](developer/analyses/recurrence.md).

## Reduced model

The model of a few features that the [reduction](#reduction) leaves for each arm: the arm's own model, refitted on the features that recursive feature elimination kept. The [reliability panel](#reliability-panel) tests its features, and the report draws it. See [Stage 3A: the reduction](developer/reduction.md).

## Reduction

Stage 3A, in which each arm's model is reduced to a few features by recursive feature elimination, at the arm's best seed. See [Stage 3A: the reduction](developer/reduction.md).

## Reference line

The line in a report's performance figures, and the row in its tables, that every arm is compared with: the score of the [baseline](#baseline).

## Reliability panel

Stage 3B, which gathers three tiers of evidence for each feature of a reduced model. Tier 1 measures whether the feature would be chosen again (stability), tier 2 whether it is associated with the target on the held-out split (association), and tier 3 whether the model needs it (necessity). See [Stage 3B: the reliability panel](developer/reliability.md).

## Reveal

The single opening of the [held-out split](#held-out-split), after every stage that fits or chooses has finished. It scores every saved model on the held-out rows, and afterwards the run refuses to start any stage that fits.

## Run

One execution of a [config](#config): a directory under `experiments/runs/` that holds the recorded config, every cell, every stage's output and the report.

## Seed

One draw of the training rows a cell fits on, and of anything else the fit draws at random. Every arm is fitted at every seed of the config, so that the stability of its choices can be measured.

## Side

Where a fitted model lives and can be used: in [Core](#core), or in the environment of the [vendored method](#vendored-method) that fitted it (DEFT or IGSR). A cell records its side in `predict_with` in its `run.json`, and the stages after Stage 2 ask that side for the model's columns, fits and predictions ([`sides.py`](../seqmaestro/pipeline/sides.py)).

## Stage

One of the steps from the delivered data to the report: Stage 1 (task construction, which prepares the dataset and the config), Stage 2 (fitting every arm), Stage 3A (the [reduction](#reduction)), Stage 3B (the [reliability panel](#reliability-panel), with the [reveal](#reveal)), and the report.

## Status page

The page in each run's folder (`status.html`) that shows where the run stands. It has one column per [stage](#stage), with what the stage has decided or found so far and the [checkpoint](#checkpoint) that ends it. The [driver](#driver) keeps it current while the run goes on. See [the status page](developer/status.md).

## Task description

The description of a dataset that every language model of a run is told: what is predicted and how the target was measured, and what the positions of a sequence mean (`experiments/datasets/<task>.json`). See [dataset descriptions](developer/descriptions.md).

## Vendored method

A method developed outside SeqMaestro whose code is kept in the repository, under `methods/`: DEFT and IGSR. Each has its own conda environment and its own tests. SeqMaestro and a vendored method never import each other, and exchange files through an [adapter](#adapter) instead. See [two boundaries](developer/architecture.md#two-boundaries).

## Written source

A [feature source](#feature-source) whose features a language model writes as code: the one-round source (`llm_generated`), the [iterative source](#iterative-source) and the [joint source](#joint-source).
