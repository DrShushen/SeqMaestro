# The example dataset

The repository holds one example dataset, so that you can see a run before you bring your own data. It is synthetic: sequences of 60 nucleotides around a made-up binding site, each labelled by whether a factor was found bound. The bases are random, apart from one motif planted at the centre of most bound sites, so the planted motif is the dataset's only signal. Its files show what Stage 1 makes from your data: a prepared dataset.

## The quick run

A shortened run of the example goes through every stage in about ten minutes on a machine with eight cores. It does not call a language model, so it also tests the installation without any credentials:

```bash
eval "$(./install.sh --interpreters)"     # CORE and the other environments' Python interpreters
$CORE experiments/prepare/build_example_dataset.py
$CORE experiments/prepare/finalise_dataset.py --splits data/example/splits \
    --declared experiments/prepare/declarations/example__splits.yaml
$CORE experiments/runners/run_experiment.py --config experiments/configs/example_quick.yaml
```

The commands do the following:

1. The first sets a shell variable to each environment's Python, `$CORE` among them.
2. The second builds the dataset into `data/example/`.
3. The third checks the dataset and writes `dataset.json`, the record from which every later stage takes the dataset, beside its splits. It also prints the summary that the agent presents at the end of Stage 1.
4. The last runs every stage, with fewer arms, two seeds and a small reliability panel.

The run goes to `experiments/runs/<date>_example_quick_full/`, which holds what a run on your data will hold: the fitted models, the tables and the report's figures. The run finds the planted motif and says nothing about biology. Since nothing rests on the example's held-out split, the quick run opens it at the end without stopping before the [reveal](../glossary.md#reveal).

The agent can take you through the example too. Start it with `./agent.sh` ([starting a run](running.md#starting-a-run)), and when it asks about your data, ask it for the example's quick run.

## Other example configs

- **`example.yaml`** runs every arm of the shared config on the example. It takes hours and calls the language model, so it needs the credentials.
- **`example_motifs.yaml`** adds the motif sources to the quick run. It needs the MEME Suite's environment ([motif discovery](installation.md#motif-discovery)), and the motif collection it names, which you fetch once per machine:

  ```bash
  $CORE experiments/prepare/fetch_motif_collection.py jaspar --release 2026 --taxon vertebrates
  ```

[`experiments/prepare/README.md`](../../experiments/prepare/README.md) describes each file of the example (its builder, the [declaration](../glossary.md#declaration) of what was decided, its task description and its configs) and how the example is built.
