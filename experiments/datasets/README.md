# Dataset descriptions

Each dataset has one file in this folder, its [task description](../../docs/glossary.md#task-description), which describes the problem in the terms a domain expert would use: what is predicted, from what, and what the positions of a sequence mean. **Every part of SeqMaestro that tells a language model about the data takes the description from here**: the [proposer](../../docs/glossary.md#proposer), the [judge](../../docs/glossary.md#judge), the [recurrence analysis](../../docs/glossary.md#recurrence-analysis), and the vendored methods DEFT and IGSR, whose prompts are built from it.

A run names its description once, as `dataset.task` in its [config](../../docs/glossary.md#config), and every stage that calls a language model takes the name from the config recorded in the run. [`context.py`](../../seqmaestro/llm/context.py) loads a description and renders it for each part.

**One file serves every part.** The facts are the same for all of them, and a dataset described separately in each place would sooner or later be described differently in each. That failure goes unnoticed, because a language model told the wrong thing answers confidently, and the run completes. With one file, adding a dataset means writing one file instead of making three edits in three formats.

**A description stays on the machine that holds its dataset, and git ignores it**, like the dataset's other files ([preparing a dataset](../prepare/README.md)), because it can hold a fact that a run withholds from its models. The only description in the repository is [`example.json`](example.json), which describes the example dataset.

## Fields

A description is a JSON object with these fields, of which only `task` is required:

- **`task`**: what is predicted, from what, and how the target was obtained (the rule that assigned a label, or how a continuous value was measured).
- **`sequence_layout`**: what the positions mean. A feature such as "GC fraction of positions 41 to 49" cannot be interpreted without it.
- **`target_name`**: a short noun phrase for the target, for DEFT's prompt templates and IGSR's data dictionary, which name the target in a sentence. The proposer, the judge and the recurrence analysis are not shown it.
- **`notes`**: anything else. The proposer, the judge and the recurrence analysis are shown it, whereas DEFT's and IGSR's prompts leave it out. It is usually empty ([content that goes elsewhere](#content-that-goes-elsewhere)).
- **`judge_only`**: the fact that the judge is told and the models that write features are not ([blinding](#blinding)).

An unknown key is refused, because a misspelled key would be a fact missing from every prompt without anyone noticing. [`context.py`](../../seqmaestro/llm/context.py) holds the list of fields (`FIELDS`).

## Blinding

[Blinding](../../docs/glossary.md#blinding) withholds a fact, such as the name of the protein under study, from every language model that writes features, and tells it to the judge. The fact goes in `judge_only`:

- `task` and `sequence_layout` reach every part.
- `judge_only` reaches the judge alone. It does not reach any part that writes features, by any path: the proposer (in one round and over several rounds), DEFT's generative finder and IGSR are all given the description without it. Tests check this for the proposer's prompt, which both of the proposer's sources share, and for the config DEFT is given.

The parts that write features are what the experiment tests. Given a fact that lets them retrieve an answer instead of discovering it (the name of the protein whose loss defines the target, for example), their output could no longer be taken as discovery, and a finished run could not show which had happened. The judge, by contrast, stands outside the test: it rates how meaningful each feature would be if its effect held, and it cannot judge that without knowing what the phenotype is.

Three rules apply to a blinded run:

1. **Blind the models, and state the blinding.** A run whose report does not say that it was blinded, and to what, cannot later be told apart from one whose description was simply thin. The run's config therefore states whether the run is blinded (`dataset.blinded`, [run configs](../configs/README.md)), the driver refuses a run whose flag disagrees with its description, and the report takes the flag and the withheld fact from the config and from the description.
2. **Withhold the fact from everything else the models are shown.** The models see the name of every column they are handed, and the target column's name ends every example row of a proposer's prompt, so a target called `is_hepg2` would tell them the cell line. They also see each covariate's description from the dataset's declaration, and the [guidance](../../docs/glossary.md#guidance) on the form of the features. The driver refuses a blinded run in which a word that only `judge_only` holds appears in a column name, or as a whole word in a covariate's description or in the guidance. Every such word of three letters or more counts, so a common word can be refused by chance ("line" in `baseline`). The remedy is then to rename the column, or to reword `judge_only` or the guidance.
3. **Each part records what it was told.** A proposer arm's `run.json` holds the description without the withheld fact, and the judge's output holds the fact as well. The record therefore shows which side of the blind each number came from, without relying on a separate claim that the blinding was applied.

## Content that goes elsewhere

**Framing for one part.** The judge knows it is judging and the proposer knows it is proposing, and that framing belongs in their own prompts. A description describes only the data. `judge_only` fits this rule, because it is a fact about the data, withheld from one side of the experiment.

**Known confounds.** Telling the judge that GC content is confounded in a dataset would make it mark GC features down. That is a judgement about whether a feature is true, which the judge is told to set aside: it rates how meaningful a feature would be if its effect held. Telling the proposer would steer it away from a hypothesis that the experiment should let it reach. Confounds therefore belong in the report. `notes` can hold one if you decide otherwise, but that makes a different experiment, which should be recorded as one.

**Guidance on the form of the features.** What the person running the analysis asks of the features the models write, such as a functional form or a stretch of sequence to measure in, is a choice made for the run, whereas a description holds facts about the data. It goes in the run's config, as `grid.generated_features.guidance` ([run configs](../configs/README.md)). Every part that writes features is shown it, and the judge is not.

**Column descriptions.** A covariate's description is declared in Stage 1 with the dataset (`covariates.<name>.description`, [Stage 1: the prepared dataset](../../docs/developer/dataset.md)), because it is a fact about a column of the split files and travels with them. Every arm whose pool holds the covariates passes the description on: the proposer in its block of columns, and DEFT and IGSR through their adapters. On a blinded dataset the description is therefore written blind, like everything in a task description except `judge_only`. The sequence column needs no description, because `sequence_layout` gives its layout, and its lengths are taken from the data.

## Adding a dataset

Write `<name>.json` with the fields above. That is all a new dataset needs: the proposer, the judge and the recurrence analysis load it directly, [`deft_bundle.py`](../adapters/deft_bundle.py) generates DEFT's prompt config from it, and [`igsr_bundle.py`](../adapters/igsr_bundle.py) builds IGSR's data dictionary from it.

The exception is reproducing published prompts, where the wording is itself part of what is compared. A hand-written config in `../adapters/deft_conf/params_generation/` does that for DEFT, and takes precedence over the generated one ([hand-written prompt configs for DEFT](../adapters/deft_conf/README.md)). A test checks each such config against the description of the same name, so that the two cannot state different sequence layouts. The wording of a hand-written config is fixed, so DEFT's arms on its dataset cannot take guidance on the form of the features.
