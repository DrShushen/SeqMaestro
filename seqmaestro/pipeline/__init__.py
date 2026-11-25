"""The code that takes a prepared dataset from its first cell to the inputs of its report.

The scripts under `experiments/runners/` and `experiments/analysis/` are the commands of a run.
Each calls the `main` of the module here that holds its stage. A stage is given the run directory
and reads every setting from the config that the driver recorded there (`<run>/config.yaml`). The
script that draws a report's figures is the exception: it runs under `SeqMaestro_LocalUtils`, has
its `main` in the script itself, and is given the report's directory.

| module | holds |
|---|---|
| `repository` | where the task descriptions, adapters, scripts, motif collections and runs are |
| `layout` | where a run keeps each thing it writes, and how a call of a run is named |
| `config` | a run's config: reading, extending, checking, and the cells it produces |
| `rows` | the training and validation rows every stage before the reveal works on |
| `sources` | the featuriser of a pool, built from the run's settings for each feature source |
| `motifs` | a run's motif collections, its store of scans and discoveries, and the step before the arms |
| `records` | what a call leaves beside its results: its failures, the environments it used |
| `driver` | the whole run: its stages, their order, and what is checked before any starts |
| `schedule` | starting a run's processes side by side within the machine's budget |
| `estimate` | how long each arm would take on this machine, estimated before a run |
| `calls` | each arm's calls to the language model, their cost and waiting, and the load on the provider |
| `baseline` | the composition baseline's cells |
| `stage2` | Stage 2: fitting the cells of the arms, for each kind of arm |
| `vendored` | the Core side of DEFT and IGSR |
| `analyses` | the analyses of the Stage 2 cells, which a run makes once every arm has finished |
| `sides` | where a fitted model lives (Core, DEFT, IGSR), and what the later stages need from it there |
| `reduction` | Stage 3A: each arm reduced to a model of N features |
| `reliability` | Stage 3B: the reliability panel, before the reveal and after it |
| `reveal` | the held-out split opened once, and every saved model scored on it |
| `report` | what a run's report is written from |
| `agents` | the record of the agents that carried out a run's stages, and of their models |
"""
