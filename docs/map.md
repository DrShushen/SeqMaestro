# Documentation map

Every page of this documentation, with a line on what each covers.

## User guide

| Page | Contents |
|---|---|
| [User guide](user/index.md) | Who the guide is for, and the order to read it in |
| [Overview](user/overview.md) | The question SeqMaestro answers, its feature sources, models and arms, the controls, the stages of a run, and the results |
| [Installation](user/installation.md) | Requirements, the installer and its environments, motif discovery, checking, updating and uninstalling |
| [Configuring the language model](user/language-model.md) | The providers, filling in `.env`, checking the connection, the call settings and the models tested |
| [Setting up an agent](user/agents.md) | Installing and starting each of the four harnesses the agent can work in, and what any other harness needs |
| [Running with an agent](user/running.md) | Starting a run, what the agent does at each stage, the decisions that are yours, and monitoring a run |
| [Running from the command line](user/command-line.md) | The `seqmaestro` command: its input table, its steps and defaults, the quick and full runs, the report, and its options |
| [The example dataset](user/example.md) | The quick run of the example dataset, and the other example configs |
| [Running by hand](user/by-hand.md) | The commands the agent runs, in the order a run needs them |
| [Troubleshooting](user/troubleshooting.md) | Known problems with the installer, the language model, the agents and a run, with how to fix each |

## Developer guide

| Page | Contents |
|---|---|
| [Developer guide](developer/index.md) | Setting up, the checks before a commit, publishing the site, and the rules for maintaining these docs |
| [Architecture and layout](developer/architecture.md) | The flow of a run through the code, the repository's layout, the agent's instructions, the command line, the two boundaries and the module map |
| [Stage 1: the prepared dataset](developer/dataset.md) | The record of a prepared dataset (`dataset.json`), the declaration, the checks, covariates, the positional window, the group map and design power |
| [Stage 2: the suite](developer/suite.md) | What makes two cells comparable, a cell's files, the kinds of arm, the other steps of the stage, and resuming |
| [The composition baseline](developer/baseline.md) | The reference model every arm is compared with, its features and model, and why it is the same for every dataset |
| [Stage 3A: the reduction](developer/reduction.md) | The three models of a reduced arm, the source cell, the rebuild and its checks, the elimination, and the limits |
| [Stage 3B: the reliability panel](developer/reliability.md) | The arms profiled and their candidates, the prefilter and clusters, the three tiers, the checks on the panel, and the order of its parts |
| [The held-out split and the reveal](developer/heldout.md) | Access to the test split, the cells before the reveal, the reveal, and why its scores equal scoring at fit time |
| [The report](developer/report.md) | How a report is built, the separations the layer keeps, the decisions it makes, the checks on every drawn model, and changing the layer |
| [The status page](developer/status.md) | The page in each run's folder that shows where the run stands, its design rules, and how to change it |
| [The feature record](developer/feature_record.md) | The files that hold the records, the reasoning behind the fields, the fields by producer, the paths into a run, and adding a field |
| [Motifs](developer/motifs.md) | Matching a motif to a sequence, the best match and the count, the known motifs, the discovered motifs, and motifs in a run |
| [The Stage 2 analyses](developer/analyses/index.md) | What the analyses share: comparing features by their values, the top N, and the permuted cells |
| [Stability and co-discovery](developer/analyses/stability_and_co_discovery.md) | Whether an arm finds the same features again, and whether several arms find the same thing |
| [Recurrence](developer/analyses/recurrence.md) | Whether an arm arrives at the same ideas again when it names them differently each time |
| [The judge](developer/analyses/judge.md) | How meaningful each feature would be if it held, and what the judge's scores can show |
| [Direct measurement](developer/analyses/direct_measurement.md) | What one feature does with no model in the way, and how far it encodes a grouping variable |
| [Direction](developer/analyses/direction.md) | Which class a feature points to, or which way it moves a continuous target, in the model and in the data |
| [Run configs](developer/configs.md) | What a config states, sharing settings through `extends`, the checks on a config, arms, settings, and the settings a user may change at the launch |
| [Running experiments](developer/experiments.md) | How scripts, configs and runs are organised, the rules every script follows, and starting a run |
| [Preparing a dataset](developer/preparation.md) | The preparation scripts, the files of one dataset, and the example dataset |
| [Dataset descriptions](developer/descriptions.md) | The task description every language model is told, its fields, and blinding |
| [The adapters](developer/adapters.md) | The scripts that run a vendored method, the bundle they hand back, and the rules every adapter follows |
| [The analysis scripts](developer/analysis-scripts.md) | Which script answers which question, and when to use it |
| [Environments](developer/environments.md) | Which conda environment runs what, how each is built, and why they are kept apart |

## Reference

| Page | Contents |
|---|---|
| [API reference](api/index) | Every module, class and function of the `seqmaestro` package, from its docstrings |
| [Terminology](glossary.md) | The terms these pages use, each with one meaning |
