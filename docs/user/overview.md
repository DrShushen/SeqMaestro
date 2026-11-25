# Overview

SeqMaestro looks for properties of DNA or RNA sequence that predict something measured for each sequence. This page explains how it finds them and what you receive at the end.

A run is carried out with you by an AI [agent](../glossary.md#agent) working on your machine, in a program such as Claude Code or Codex. [Running with an agent](running.md) describes the same process from your side. Without an agent, the [command line](command-line.md) takes a single table through these stages, with fixed defaults where the agent would ask you.

## The question

You bring sequences with one measurement for each: either a class, such as whether a protein was found bound to the sequence, or a number, such as the strength of a signal. SeqMaestro answers the question *which properties of the sequence predict that measurement?*, and states each property in terms a biologist can follow.

Such a property is a [feature](../glossary.md#feature): a number computed from a sequence, such as whether a given base sits at a given position, or how often a short motif occurs. A model that predicts the measurement from a few features of this kind can be understood feature by feature. For this reason, SeqMaestro reduces each of its models to a few features and then tests those features one by one.

## Feature sources

A [feature source](../glossary.md#feature-source) is one way of producing features from a sequence. The name in the second column is how a config and the folders of a run refer to the source:

| Feature source | Name | Its features |
|---|---|---|
| Positions | `positional` | Which base sits at each position of the sequence |
| k-mers | `kmer` | How often each short substring of length k (a k-mer) occurs |
| Known motifs | `motif_known` | How strongly and how often each motif of a published collection (such as the transcription factors' motifs in [JASPAR](https://jaspar.elixir.no/)) matches the sequence |
| Discovered motifs | `motif_discovered` | How strongly and how often each motif that [STREME](https://meme-suite.org/meme/doc/streme.html) (from the MEME Suite) finds in the training sequences matches the sequence |
| Covariates | `covariates` | The measurements supplied with each sequence, such as a chromatin signal |
| Written sources | `llm_generated`, `llm_generated_iterative`, `llm_generated_joint` | Functions of the sequence that a language model writes as code |

In the [written sources](../glossary.md#written-source), a language model writes features as short Python functions of the sequence. The three written sources differ in how the model writes them:

- **In one round** (`llm_generated`), from the [task description](../glossary.md#task-description) and some labelled example rows.
- **Over several rounds** (`llm_generated_iterative`), in which the model is shown how a fit used the features it wrote so far (the [iterative source](../glossary.md#iterative-source)).
- **Inside a model's fit** (`llm_generated_joint`), as the models DEFT and IGSR do (the [joint source](../glossary.md#joint-source)).

## Models

| Model | Description |
|---|---|
| Sparse linear model | L1-penalised logistic regression for a class, or L1-penalised linear regression ([the Lasso](https://en.wikipedia.org/wiki/Lasso_%28statistics%29)) for a number |
| EBM | An [explainable boosting machine](https://interpret.ml/docs/ebm.html): a sum of one learned curve per feature |
| XGBoost | [Gradient-boosted trees](https://xgboost.readthedocs.io/en/stable/), which serve as a black-box reference |
| PySR | One equation over the features, which the evolutionary search of [PySR](https://github.com/MilesCranmer/PySR) finds |
| Decision tree | [CART](https://en.wikipedia.org/wiki/Decision_tree_learning), or [DEFT](https://arxiv.org/abs/2604.12060), which asks a language model for candidate features at each split of the tree |
| IGSR | [Influence-guided symbolic regression](https://arxiv.org/abs/2605.29184): a linear model over terms that a language model writes (for a number only) |

## Arms

SeqMaestro approaches the question in several ways at once. Each way is an [arm](../glossary.md#arm): one model fitted on one [pool](../glossary.md#pool), the set of feature sources that the model may choose its features from.

An arm is written `model/pool`, with `+` between the names of the sources in a pool. For example, `ebm/positional` is the EBM on positions, and `xgboost/positional+kmer` is XGBoost on positions and k-mers. You will see these names in the agent's messages and in the folders of a run.

The arms of a run are listed in its [config](../glossary.md#config), the YAML file under `experiments/configs/` that describes the run. Every dataset's config builds on a shared config, which pairs each model with the pools of positions, k-mers and written sources that suit it. When the agent prepares the task with you (Stage 1, below), it can add two more kinds of source:

- **The covariates**, where your data supplies measurements beside the sequences.
- **The motif sources**, where the question concerns binding by transcription factors or RNA-binding proteins.

## Controls

A search for features finds structure in noise too, so SeqMaestro compares every result with two controls.

- **Shuffled labels.** Every arm is fitted both on the real labels and on shuffled training labels. These are the two [label conditions](../glossary.md#label-condition), `real` and `permuted`. A result counts as a finding only where it exceeds what the same arm achieves on shuffled labels.
- **The composition baseline.** Every run also fits the [composition baseline](../glossary.md#baseline): a sparse linear model on the fraction of each base in the sequence, and nothing else. Where the classes differ in base composition, or the measured number varies with it, the baseline already predicts the measurement. An arm that scores no better than the baseline has therefore shown nothing beyond what composition gives. The report draws the baseline as a line beside every arm's score.

Each arm is also fitted at several [seeds](../glossary.md#seed), each of which draws a different random sample of the training rows, so that SeqMaestro can tell whether the arm chooses the same features again.

## Stages of a run

SeqMaestro goes from your data to a report in five [stages](../glossary.md#stage). Stage 1 prepares the dataset and the config, and the stages after it make up a [run](../glossary.md#run): one execution of that config, kept in a folder under `experiments/runs/`.

```{mermaid}
graph TD
  D["Your data"] --> S1["Stage 1<br/>Task construction"]
  S1 -->|"Training and<br/>validation rows"| S2["Stage 2<br/>The suite"]
  S2 --> S3A["Stage 3A<br/>The reduction"]
  S3A --> S3B["Stage 3B<br/>The reliability panel"]
  S1 -->|"Held-out rows"| RV["The reveal"]
  S3B --> RV
  RV --> R["The report"]
```

1. **Stage 1, task construction.** Your data becomes a prediction task. The sequences and their measurement (the target) are split into training, validation and held-out rows, and a [task description](../glossary.md#task-description) states what every language model of the run is told: what is predicted, from what, and what the positions mean. The agent builds the task with you, and nothing is fitted until you approve it.
2. **Stage 2, the suite.** Every arm is fitted on the training rows and scored on the validation rows, under both label conditions and at every seed. Analyses then compare the fits: whether an arm chooses the same features again across seeds, and whether different arms find the same features. A language model called the [judge](../glossary.md#judge) can also rate how meaningful each of the most important features would be if its effect held, where the config turns it on.
3. **Stage 3A, the reduction.** In the [reduction](../glossary.md#reduction), features are removed from each arm's model, the weakest first, until only a few remain (ten, in the shared config). The result is a model small enough to interpret in full.
4. **Stage 3B, the reliability panel.** The [reliability panel](../glossary.md#reliability-panel) gives three tiers of evidence for each feature of a reduced model, and for each alternative feature the model could have used instead:
   - Whether the arm would choose the feature again on another half of the data (stability).
   - Whether the feature goes with the target on its own (association).
   - Whether the model gets worse without it (necessity).

   Stability is measured before the reveal, whereas association and necessity are measured on the held-out rows once the reveal has opened them. The panel leaves out XGBoost, whose fit cannot be interpreted feature by feature.
5. **The report.** The agent writes it from the tables and figures the run prepared.

The held-out rows form the [held-out split](../glossary.md#held-out-split), which no stage uses to fit a model or to choose anything. They stay closed until [the reveal](../glossary.md#reveal), which opens them a single time, for all stages together, and scores every model the run saved. After the reveal, a stage that fits or chooses refuses to run, so a held-out score cannot have steered any decision of the run.

## Results

The report is written for the people who supplied the data, in a form that a biologist and a bioinformatician can both follow. It says which properties of the sequence predict the measurement, which few-feature models are worth showing, and how far each of their features can be trusted. It holds two kinds of result, which serve different goals:

- **A model** small enough to interpret end to end, which you could apply to a new sequence.
- **A set of ideas** that held up across independent arms and on sequences that nothing was fitted on, around which a follow-up experiment can be designed.

Every statement in the report is either a number or a description of what was done, apart from the agent's interpretation, which is kept in blocks headed *AI Agent Analysis Note*. The report also names the language model the run called and the agent that carried out the analysis.

The report is `REPORT.md`, in the `report/` folder of the run, with a PDF beside it if the [PDF tools](installation.md#pdf-tools) are installed. The run's folder also holds everything the report was written from: the config, every fitted model, the tables, the figures and the logs. The run's [status page](running.md#monitoring-a-run), `status.html`, shows what each stage found and links to these files.
