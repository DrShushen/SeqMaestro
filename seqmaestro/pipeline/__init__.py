"""What takes a prepared dataset from its first cell to its report inputs.

The scripts under `experiments/runners/` and `experiments/analysis/` are the commands of a run.
A stage's script calls the `main` of a module here, which holds the stage. A stage is given the run
directory and reads every setting from the config the driver recorded there (`<run>/config.yaml`).
The script that draws a report's figures is the exception: it runs under `SeqMaestro_LocalUtils`,
holds its `main`, and is given the report's directory.

| module | holds |
|---|---|
| `repository` | where the descriptions, the adapters, the scripts and the runs are |
| `layout` | where a run keeps each thing it writes, and how a call of a run is named |
| `config` | a run's config: reading, extending, checking, and the cells it produces |
| `rows` | the training and validation rows every stage before the reveal works on |
| `records` | what a call leaves beside its results: its failures, the environments it used |
| `driver` | the whole run: its stages, their order, and what is checked before any starts |
| `schedule` | starting a run's processes side by side within a budget of the machine |
| `estimate` | what each arm would cost on this machine, before a run |
| `baseline` | the composition baseline's cells |
| `stage2` | Stage 2: the cells of the arms, by the kind of arm |
| `vendored` | the Core side of DEFT and IGSR |
| `analyses` | the analyses of the Stage 2 cells, which a run makes once every arm has finished |
| `sides` | where a fitted model lives (Core, DEFT, IGSR), and what the later stages ask of it there |
| `reduction` | Stage 3A: each arm reduced to a model of N features |
| `reliability` | Stage 3B: the reliability panel, before the reveal and after it |
| `reveal` | the test split opened once, and every saved model scored on it |
| `report` | what a run's report is written from |
"""
