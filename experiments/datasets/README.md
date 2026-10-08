# Dataset descriptions: described once, read by every consumer

One file per dataset. It says what the problem is, in the terms a domain expert would state it, and
**every consumer that has to tell a language model about the data reads it from here**: the
proposer, the judge, and the vendored methods whose prompts are templated on the task.

A run names its description once, as `dataset.task` in its config, and every stage takes the name
from the run's recorded config: Stage 2's arms (`run_arm.py`), the judge (`run_judge.py`) and the
recurrence analysis (`analyse_recurrence.py`).

**A description stays on the machine that holds its dataset, and git ignores it**, like the
dataset's other files ([`../prepare/README.md`](../prepare/README.md)): it can hold the very fact a
run withholds from its models. The one description in the repository is `example.json`, of the
example dataset.

**One file serves every consumer.** The facts are the same for all of them, and a dataset described
separately in each place would eventually be described differently in each. That failure is
silent, because a model told the wrong thing answers confidently and the run completes. With this
directory, adding a dataset means writing one file instead of making three edits in three formats.

## The fields

| | |
|---|---|
| `task` | **Required.** What is being predicted, from what, and how the target was obtained: the rule that assigned a label, or how a continuous value was measured |
| `sequence_layout` | What the positions mean. "GC fraction of positions 41-49" is uninterpretable without it |
| `target_name` | A short noun phrase for the target, for prompt templates that interpolate one. Not shown to the judge or the proposer |
| `notes` | Anything else. Shown to the proposer and the judge. DEFT's and IGSR's prompts are built from `task`, `sequence_layout` and `target_name`, and do not carry it. Usually empty (see below) |
| `judge_only` | The one fact the judge is told and the models are not (see **Blinding** below) |

An unknown key is refused, not ignored, because a misspelled key is a fact silently missing from
every prompt.

## Blinding: `judge_only`

`task` and `sequence_layout` go to every consumer. `judge_only` goes to the judge alone and
reaches no consumer that writes features by any path: the proposer, in one round and over several,
DEFT's generative finder and IGSR are all given the blind form. A test checks it for the
proposer's prompt, which both of its sources share, and for the config DEFT is given.

It exists because those consumers are what the experiment tests. Given a fact that lets them retrieve
an answer instead of discovering it (the name of the protein whose loss defines the target, for
example), their output could no longer be read as discovery, and nothing in a finished run would
show which had happened. The judge is not under test. It is the instrument that scores what they found, and it
is asked how significant a feature would be if it held, which it cannot answer without knowing what
the phenotype is.

`judge_only` does not break the rule against consumer-specific framing (below). Framing concerns
what a consumer is for. `judge_only` is a fact about the data, disclosed to one consumer and
withheld from others, and which consumers are blind is a property of the experiment.

Three rules follow for whoever sets one up:

* **Blind the models, not the report.** A run whose report does not say it was blinded, and to
  what, cannot later be told apart from one whose description was simply thin, so the report
  states the blinding. A run's config says whether it is blinded (`dataset.blinded`, see
  [`../configs/README.md`](../configs/README.md)), and the report takes the flag and the withheld
  fact from there and from this file.
* **Withhold the fact from the column names too.** The models see the name of every column they are
  handed, and the target column's name ends every example row of a proposer's prompt, so a target
  called `is_hepg2` tells them the cell line. They also see each covariate's description from the
  dataset's declaration. The driver refuses a blinded run where a word of `judge_only` that no other
  field holds shows in a column name, or whole in a covariate's description. Every such word of
  three letters or more counts, so a common word can be refused by chance ("line" in `baseline`),
  and the remedy is then to rename the column or reword `judge_only`.
* **Each consumer records what it was told.** A proposer arm's `run.json` carries the description
  without the withheld fact, and the judge's output carries the fact as well. The record therefore
  shows which side of the blind each number came from, without relying on a separate claim that
  the blinding was applied.

## What does *not* go here

**Consumer-specific framing.** The judge knows it is judging and the proposer knows it is
proposing, and that framing belongs in their own prompts. This file describes only the data.
`judge_only` fits that rule, because it is a fact about the data held back from one side of the
experiment (see **Blinding** above).

**Known confounds.** Telling the judge that GC content is confounded in a dataset would make it
mark GC features down. That is a judgement about whether a feature is true, which the judge's
question sets aside by asking how significant a feature would be if it were true. Telling the
proposer would steer it away from a hypothesis the experiment should let it reach. Confounds belong
in the report. `notes` can carry one if you decide otherwise, but that is a different experiment
and should be recorded as one.

**Column descriptions.** A covariate's description is declared at Stage 1 with the dataset
(`covariates.<name>.description`, [`../../docs/dataset.md`](../../docs/dataset.md)), because it
is a fact about a column of the split files and travels with them. Every generator whose pool
holds the covariates is told it: the proposer in its columns block, and DEFT and IGSR through
their adapters. So on a blinded dataset it is
written blind, like everything here but `judge_only`. The sequence column needs none: its
layout is `sequence_layout`, and its lengths are read from the data.

## Adding a dataset

Write `<name>.json` with the fields above. Nothing else is needed: the proposer, the judge and the
recurrence analysis load it directly, `deft_bundle.py` generates DEFT's prompt config from it, and
`igsr_bundle.py` builds IGSR's data dictionary from it.

The exception is reproducing published prompts, where the wording is itself part of what is being
compared. A hand-written config in `../adapters/deft_conf/params_generation/` does that for DEFT,
and takes precedence over the generated one. A test checks each such config against the description
of the same name, so the two cannot state different sequence layouts
([`../adapters/deft_conf/README.md`](../adapters/deft_conf/README.md)).
