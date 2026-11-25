# Preparing a dataset

The scripts in this folder build and check datasets. They read the delivered data, write under `data/`, and run whenever the data changes, before any [run](../../docs/glossary.md#run). This page lists the scripts, says which files belong to one dataset and where they are kept, and shows how the example dataset is built.

[Stage 1: the prepared dataset](../../docs/developer/dataset.md) explains what preparation hands over to the later stages, and [running experiments](../README.md) gives the rules that every script follows, including those for preparation.

## Scripts

| Script | Use | What it does |
|---|---|---|
| [`finalise_dataset.py`](finalise_dataset.py) | Every dataset, as its last step | Counts, checks and measures a prepared split, and writes its `dataset.json`, the record every later stage takes the dataset from |
| [`write_group_map.py`](write_group_map.py) | A grouped split | Writes the group map that the split is checked against: the group of each sequence. Run it before finalising, where the script that made the split did not write the map |
| [`design_power.py`](design_power.py) | Planning a split | Says what a split of a given size could detect. A finalised dataset records the same figures |
| [`check_sequence_provenance.py`](check_sequence_provenance.py) | A dataset with genomic coordinates | Checks that the strand convention and the genome build are what the dataset claims: offline from the base composition, and against UCSC on request |
| [`predict_alphagenome_summaries.py`](predict_alphagenome_summaries.py) | Elements with genomic coordinates | Summarises AlphaGenome's predicted tracks over each element, for use as [covariates](../../docs/glossary.md#covariate). It runs in `SeqMaestro_Method_alphagenome` and calls AlphaGenome's API |
| [`fetch_motif_collection.py`](fetch_motif_collection.py) | A run that scans for known motifs | Fetches a published motif collection ([JASPAR](https://jaspar.elixir.no/)'s or CisBP-RNA's) into `data/motif_collections/<name>/`, where the run's config names it ([motifs](../../docs/developer/motifs.md)) |
| [`build_example_dataset.py`](build_example_dataset.py) | The example dataset | Builds the example from a stand-in delivery to a grouped split ([below](#the-example-dataset)) |

Each script's docstring says how to run it and what it decides.

**Check the orientation of every dataset that comes with genomic coordinates.** Run `check_sequence_provenance.py` whenever a dataset arrives with genomic coordinates and an assumed orientation. A later stage cannot detect a dataset that was reverse complemented as a whole, and every feature found on it would still look interpretable.

**Every dataset ends with the same two steps**, however it was built: a [declaration](../../docs/glossary.md#declaration) of what was decided, under `declarations/`, and `finalise_dataset.py`, which writes `dataset.json` beside the splits. A number from preparation is quoted from `dataset.json`, never from a script's output or from this page.

**Preparation does not compute a baseline.** How far base composition alone separates the classes is measured once, in the same way for every dataset, by the composition [baseline](../../docs/glossary.md#baseline) at the start of Stage 2. A builder may measure a variable that the window leaves out (a length, a masked fraction) where that measurement is the reason for a design decision. It then records the measurement as the reason for that decision, in its build manifest (the record a builder writes beside the data, such as the example's `build_manifest.json`).

## Files of one dataset

Each dataset has files in these places:

```
data/<dataset>/
    README.md                        what the dataset is, for a person
    raw/                             the delivered data, as received
    <splits>/
        train.csv, val.csv, test.csv
        groups.csv                   the group of each sequence, where the split is grouped
        dataset.json                 written by finalise_dataset.py
experiments/prepare/
    build_<dataset>.py               the builder, which builds the dataset from the delivery
    prepare_<dataset>_split.py       the script that splits it, where that is a second script
    declarations/<dataset>__<splits>.yaml    the declaration of what was decided
experiments/datasets/<task>.json     the task description that the language models are told
experiments/configs/<name>.yaml      the run's config
tests/test_local_<dataset>.py        the tests of the builder, where it has any
```

**Git ignores all of these files, so they stay on the machine that holds the dataset.** A builder encodes the quirks of one delivery, and a [task description](../../docs/glossary.md#task-description) can hold a fact that a run withholds from its models, so neither is of use without the data, and neither belongs in a repository that other people clone. The repository therefore holds the scripts above, the shared configs and the example dataset, and updating it with `git pull` leaves the files of your datasets alone. Git keeps no history of these files, so copy them with the dataset when it moves to another machine.

The declaration is kept under `declarations/`, apart from `data/`, so that it survives when the data is rebuilt: a dataset rebuilt by its builder and finalised from its declaration gets the same `dataset.json`.

Facts that limit how a result on the dataset may be interpreted (a class balance that was imposed, a confound that the window removes, rows that are not independent) go in the dataset's README, which whoever writes the report reads before writing.

## The example dataset

The repository holds every file of one dataset, so that each of those files can be seen filled in, and so that a run can be tried before there is real data. The dataset is synthetic and holds no biology: 60 nucleotides around a made-up binding site, labelled by whether a factor was found bound there, with random bases and one motif planted at the centre of most bound sites.

```bash
eval "$(./install.sh --interpreters)"     # CORE and the other environments' Python interpreters
$CORE experiments/prepare/build_example_dataset.py
$CORE experiments/prepare/finalise_dataset.py --splits data/example/splits \
    --declared experiments/prepare/declarations/example__splits.yaml
```

`build_example_dataset.py` writes `data/example/`. `finalise_dataset.py` then writes `dataset.json` beside its splits, and prints the summary that the [checkpoint](../../docs/glossary.md#checkpoint) at the end of Stage 1 presents to the user.

| File | What it shows |
|---|---|
| [`build_example_dataset.py`](build_example_dataset.py) | The steps of a builder, one function each: reading the delivery, cutting one window around an anchor, a filter that counts what it drops from each class and refuses to drop many, a split grouped on what makes two rows dependent, the group map, and the dataset's README |
| [`declarations/example__splits.yaml`](declarations/example__splits.yaml) | A declaration for a binary target, in which each answer is written as a real dataset's would be |
| [`../datasets/example.json`](../datasets/example.json) | A task description: what is predicted, how the label was obtained, and what the positions mean |
| [`../configs/example.yaml`](../configs/example.yaml) | A run config that states only what belongs to the dataset |
| [`../configs/example_quick.yaml`](../configs/example_quick.yaml) | A config that extends another and makes it smaller: every stage in a few minutes, with no language model |
| [`../configs/example_motifs.yaml`](../configs/example_motifs.yaml) | A config that extends both the quick run and `_motifs.yaml` (the shared config of the motif sources), and states what `_motifs.yaml` leaves to the dataset: the motif collection and the strands |
| `data/example/README.md`, which the builder writes | The sections of a dataset's README |

[Run configs](../configs/README.md) shows how to run the example, in full or in a few minutes without a language model. [`test_example_dataset.py`](../../tests/test_example_dataset.py) checks that these files agree with each other.
