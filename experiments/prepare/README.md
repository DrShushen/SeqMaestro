# Preparation: build and check datasets

The scripts here read raw or upstream data, write `data/`, and run before any experiment. The
contracts every stage follows are in [`../README.md`](../README.md). This file says which scripts
every dataset uses, which files belong to one dataset and where they are kept, and how the example
dataset is built.

## The scripts every dataset uses

| | for | what it does |
|---|---|---|
| `finalise_dataset.py` | any dataset | **the last step for every dataset.** Counts, checks and measures a prepared split and writes its `dataset.json`, the file every later stage reads. See [`../../docs/dataset.md`](../../docs/dataset.md) |
| `write_group_map.py` | any grouped split | the record a grouped split is checked against: which group each sequence belongs to, keyed so it can be joined to the split files. Run it before finalising, where the script that made the split did not write the map |
| `design_power.py` | any dataset | what a split of a given size could detect, for planning one. A finalised dataset already records it |
| `check_sequence_provenance.py` | a dataset with genomic coordinates | does the strand convention and genome build match what the dataset claims? Composition check offline, reference check against UCSC on request |
| `predict_alphagenome_summaries.py` | any set of elements with genomic coordinates | AlphaGenome's predictions for each element, every track summarised over the element (mean and maximum) at the track's resolution, with both strands of a stranded track kept. In native context (the 16,384 bp of the genome around the element) or scaffold context (the element in shuffled flanks), by the default model or, per element, by the fold model whose training excluded its region. **Runs in `SeqMaestro_Method_alphagenome`** and calls AlphaGenome's API |
| `build_example_dataset.py` | the example dataset | builds the example from a stand-in delivery to a grouped split, as the worked example of a builder (below) |

`check_sequence_provenance.py` is the one to reach for whenever a dataset arrives with genomic
coordinates and an assumed orientation. Nothing downstream can detect a globally reverse-complemented
dataset, and every feature about it will still look interpretable.

**Nothing here runs in the grid.** These scripts are run once when the data changes, and every
dataset ends the same way: a declaration of what was decided, under `declarations/`, and
`finalise_dataset.py`, which writes `dataset.json` beside the splits. If a number from one of these
scripts is worth quoting, quote it from `dataset.json`, not from a script or from here.

**No script here computes a baseline.** How far base composition alone separates the classes is
measured once, the same way for every dataset, by the composition baseline at the start of Stage 2.
A builder may measure a variable the window keeps out (a length, a masked fraction) when that is
the reason for a design decision, and records it as a confound in its build manifest.

## The files of one dataset, and where they are kept

A dataset has files in five places beside its data:

| | where |
|---|---|
| the script that builds it from the delivery, and the script that splits it where that is a second one | `build_<dataset>.py` and `prepare_<dataset>_split.py`, in this folder |
| the declaration of what was decided | `declarations/<dataset>__<splits>.yaml` |
| the description the language models are told | `../datasets/<task>.json` |
| the run config | `../configs/<name>.yaml` |
| the tests of its builder, where it has any | `tests/test_local_<dataset>.py` |

**All of them stay on the machine that holds the dataset. Git ignores them, as it ignores `data/`.**
A builder encodes the quirks of one delivery, and a description can hold a fact that a run
withholds from its models, so neither is of use without the data and neither belongs in a
repository that other people clone. The repository therefore holds the scripts above, the shared
configs and one example, and updating it with `git pull` leaves the files of your datasets alone.

Git keeps no history of these files, so copy them with the dataset when it moves to another
machine. A dataset rebuilt from its builder and finalised from its declaration gets the same
description, which is why the declaration is kept apart from `data/`.

Facts that constrain how a result on a dataset may be read (a class balance that was imposed, a
confound the window removes, rows that are not independent) go in the dataset's README,
`data/<dataset>/README.md`, which whoever writes the report reads first.

## The example dataset

One dataset is in the repository whole, so that each of those files can be seen filled in, and so
that a run can be tried before there is real data. It is synthetic: 60 nucleotides around a made-up
binding site, labelled by whether the factor was found bound, with random bases and one motif
planted at the centre of most bound sites. Nothing in it is biology.

```bash
eval "$(./install.sh --interpreters)"     # CORE and the other environments' pythons
$CORE experiments/prepare/build_example_dataset.py
$CORE experiments/prepare/finalise_dataset.py --splits data/example/splits \
    --declared experiments/prepare/declarations/example__splits.yaml
```

The first command writes `data/example/`, and the second writes `dataset.json` beside its splits
and prints the summary a Stage 1 checkpoint presents.

| file | what it shows |
|---|---|
| `build_example_dataset.py` | the steps of a builder, one function each: reading the delivery, cutting one window around an anchor, a filter that counts what it drops from each class and refuses to drop many, a split grouped on what makes two rows dependent, the group map, and the dataset's README |
| `declarations/example__splits.yaml` | a declaration with every field answered |
| `../datasets/example.json` | a description: what is predicted, how the label was obtained, and what the positions mean |
| `../configs/example.yaml` | a run config that states only what belongs to the dataset |
| `../configs/example_quick.yaml` | a config that extends another and cuts it down: every stage in a few minutes, with no language model |
| `data/example/README.md`, which the builder writes | the sections a dataset's README has |

[`../configs/README.md`](../configs/README.md) shows how to run it, in full or in a few minutes
without a language model. `tests/test_example_dataset.py` holds these files to each other.
