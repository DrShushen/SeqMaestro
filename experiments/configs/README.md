# Run configs

> This page tracks [`config.py`](../../seqmaestro/pipeline/config.py), [`arms.py`](../../seqmaestro/arms.py) and the shared configs in `experiments/configs/`. Change it in the same commit as any of them.

A run [config](../../docs/glossary.md#config) is a YAML file that states one [run](../../docs/glossary.md#run) in full: the dataset, the [seeds](../../docs/glossary.md#seed) and [label conditions](../../docs/glossary.md#label-condition), the [arms](../../docs/glossary.md#arm), how the [language model](../../docs/glossary.md#language-model) is called, and the settings of each [stage](../../docs/glossary.md#stage). Each variant of a dataset has a config of its own, which the [driver](../../docs/glossary.md#driver) runs:

```bash
eval "$(./install.sh --interpreters)"     # CORE and the other environments' Python interpreters
$CORE experiments/runners/run_experiment.py --config experiments/configs/example.yaml
```

Every setting of a run is stated in its config. A stage has no value of its own to fall back on, because settings kept anywhere else would cause two problems:

- Settings passed as environment variables, or as flags typed again for each stage, could be found only by reading the driver. One left unset would take a default defined somewhere else.
- A finished run would keep no record of its settings, so a resumed run could not tell a finished [cell](../../docs/glossary.md#cell) from one computed under other settings.

When a run starts, the driver records its resolved config in the run's directory as `config.yaml`. The resolved config is the config with the files it extends merged in ([Sharing settings through `extends`](#sharing-settings-through-extends)), any `--set` applied, and every implied value stated (such as the dataset's columns). It therefore describes the run on its own. Every stage takes its settings from that copy. The driver refuses to continue a run under a config that differs from it, so that two configs never mix in one run. [Running experiments](../README.md#starting-a-run) says how a deliberate change is allowed.

## Shared configs and dataset configs

The shared configs are in git, so that the grid and the arms every dataset runs are versioned with the code that runs them:

- [`_shared.yaml`](_shared.yaml) holds what every dataset's run has in common: the grid, the arms, how the language model is called, and the settings of the [feature sources](../../docs/glossary.md#feature-source), the models, the analyses, the [reduction](../../docs/glossary.md#reduction) and the [reliability panel](../../docs/glossary.md#reliability-panel). Comments beside the settings say what each does and why it has its value. The file cannot run on its own, because it names no dataset.
- [`_shared_continuous.yaml`](_shared_continuous.yaml) holds what a continuous target changes ([Sharing settings through `extends`](#sharing-settings-through-extends)).
- [`_motifs.yaml`](_motifs.yaml) is the fragment that adds the motif sources to a run ([Motifs](#motifs)).

Git ignores a dataset's own configs, which stay with the dataset's other files on the machine that holds it ([preparing a dataset](../prepare/README.md)). The repository holds the configs of the example dataset in their place ([Exercising the pipeline cheaply](#exercising-the-pipeline-cheaply)):

- [`example.yaml`](example.yaml) runs the shared grid and arms on the example dataset.
- [`example_quick.yaml`](example_quick.yaml) reduces that run to a few minutes, with no language model.
- [`example_motifs.yaml`](example_motifs.yaml) adds the motif sources to the quick run.

## Sections of a config

`example.yaml` shows the sections that a dataset's config states itself (`name`, `description` and `dataset`), whereas `_shared.yaml` shows the others. The loader (`load`, in [`config.py`](../../seqmaestro/pipeline/config.py)) checks each section and keeps it in the field of the same name of `ExperimentConfig`. The field's type, such as `Grid` for `grid`, lists the section's keys. The comments of `_shared.yaml` say what each of its keys does, whereas the page of each stage in the [developer guide](../../docs/developer/index.md) explains how the stage uses its settings.

## Sharing settings through `extends`

A dataset's config extends `_shared.yaml` and states only what belongs to that dataset. A dataset with a continuous target extends `_shared_continuous.yaml` instead, which extends `_shared.yaml` in its turn and changes only what the kind of target changes: it removes the class weighting and adds [IGSR](https://arxiv.org/abs/2605.29184), with its arm and its settings. The two kinds of target therefore share every other setting. `l1` in the shared arms names the sparse linear model of the dataset's kind, so the arms need no second list.

The loader merges a config into the files it extends (`resolve`, in [`config.py`](../../seqmaestro/pipeline/config.py)):

- `extends` names a file by its path relative to the extending file, or a list of such files. Each file named is resolved first, so `extends` chains.
- The files of a list are merged in their order, and the extending file last, so a later file's setting replaces an earlier one's.
- Mappings are merged key by key, at every depth. A list replaces the inherited list whole, so that an extending file can shorten one.
- `null` removes a key, so a config can drop an arm as easily as change one.
- `extends` is consumed as the config is resolved, so the copy recorded beside a run is complete on its own.

```yaml
extends: _shared.yaml
name: example_no_kmer
dataset: { ... }
sources:
  kmer: null                       # no pool below holds the k-mers
arms:
  positional+kmer: null            # this dataset does not run the pool
  positional:
    models: [l1, ebm, deft_or_cart]  # replaces the shared list whole
```

A fragment is a file that adds a group of sources and arms, which a dataset may take or leave. [`_motifs.yaml`](_motifs.yaml) is one: a dataset's config takes the motif sources by extending it beside the shared config of its kind of target ([Motifs](#motifs)).

## Checks on a config

The loader refuses a config that cannot run as written, before any work is done, and names the file and the key at fault:

- **Every key is known.** An unknown key is refused, because a misspelt key is a setting silently missing from the run, which would complete without it. The names of models and pools, and every other value chosen from a fixed list (such as a class weighting), are checked against the library's own lists. A typo therefore cannot quietly skip an arm or name a model that does not exist.
- **Nothing is defaulted.** `grid`, `language_model` and `analyses` are required in full, and so are `reduction` and `reliability` in a run that has them, because their settings decide what a run's numbers mean. A grid that filled in its own bootstrap count, for example, would give intervals that mean something different from those of the run it is compared with, with nothing in either run to show it.
- **Facts about the data come from the dataset.** `dataset` names the split that [Stage 1](../../docs/developer/dataset.md) prepared (`splits`) and its [task description](../../docs/glossary.md#task-description) (`task`, which selects a file in [`../datasets/`](../datasets/README.md)). A default for either would apply one dataset's answer to another. For the task description, that would mean a language model is told the wrong thing and answers confidently all the same. The columns and the kind of target come from the split's `dataset.json`, so a stage takes no column as a flag. A config may restate them (as the copy a run records does, so that the run describes itself), but only to agree with the dataset. A split that was never finalised, or whose `dataset.json` records a failed check, is refused.
- **Every setting is used.** Settings for a source that no [pool](../../docs/glossary.md#pool) holds, or for a model that no arm fits, are refused, since a setting that nothing uses is a mistake the run would not show.
- **The settings fit the kind of target.** A class weighting on a continuous target, which has no classes to weight, is refused. So are a binary model on a continuous target, a binary target without a class weighting, and IGSR on a binary target.
- **Each arm can be fitted.** The pools, their models and what Stage 3 leaves out of each are checked as [Arms](#arms) describes. Each model's settings are checked as [Source and model settings](#source-and-model-settings) describes.
- **Stage 3 has what it needs.** A run without a `reduction` section has no Stage 3A. `reliability` (Stage 3B) profiles the reduced models, so it needs `reduction`. Its `check_arm`, the arm on which the panel runs its global-null and planted-signal checks ([the reliability panel](../../docs/developer/reliability.md)), must be an arm the panel profiles.
- **The version matches.** A config may state the `version` of its shape (its sections, their keys and what each means), whereas the copy a run records always states it. A stage refuses a run that recorded another version, because it could not tell which of that run's settings now mean something else.

Before it starts anything, the driver makes the checks that need more than the config ([`driver.py`](../../seqmaestro/pipeline/driver.py)):

- **Blinding.** `dataset.blinded` says whether the run withholds a fact from the language models that write features ([blinding](../../docs/glossary.md#blinding)). It must agree with the task description, which holds any withheld fact in `judge_only`. A blinded run is refused where a withheld word shows in a column name, in the description of a [covariate](../../docs/glossary.md#covariate) or in the [guidance](../../docs/glossary.md#guidance) on the form of the features. `run_arm.py` makes the same checks whenever an arm it is to fit calls a language model, so a stage started by hand is covered too ([dataset descriptions](../datasets/README.md)).
- **What DEFT and IGSR are told.** `config_overrides`, the overrides of [DEFT](https://arxiv.org/abs/2604.12060)'s or IGSR's own configuration that a config may give, are refused where they restate a setting of the grid, name the language model or say how it is called. So is guidance that cannot reach DEFT's prompts ([Source and model settings](#source-and-model-settings)).
- **What the stages need.** For the stages it is about to start, the driver checks for the language model's credentials, the interpreter of each environment a stage runs in, and any motif collection the run scans with.

## Arms

An arm is one model fitted on one pool of feature sources. The `arms` section maps each pool to the models fitted on it, so each model of an entry is one arm:

```yaml
arms:
  positional:
    models: [l1, ebm, xgboost, pysr, deft_or_cart]
  kmer+llm_generated_iterative:
    models: [l1, ebm, xgboost]
    not_profiled:
      ebm: "Each of the panel's 100 selections refits an EBM over the whole k-mer pool"
```

The loader checks every entry against what the code declares an arm can be:

- **Sources.** [`features/__init__.py`](../../seqmaestro/features/__init__.py) lists the sources a pool may hold, in a fixed order (`POOL_MEMBERS`). A pool's name joins its sources with `+` in that order. A pool written in another order is refused, so that one set of sources has one name. A pool holds at most one [written source](../../docs/glossary.md#written-source), so that each arm stands for one way of writing features.
- **Models.** [`arms.py`](../../seqmaestro/arms.py) declares each model: the kinds of target it fits, whether it takes per-row weights, and which sources it can be handed, with the reason for each combination it refuses. `l1` names the sparse linear model of the dataset's kind of target (`l1_logistic` for a binary target, `l1_linear` for a continuous one). The recorded config and the cells name the model it resolved to. A model named twice in one entry is refused.

**The order of the pools.** The driver offers the machine to the arms in the order the config lists them. A config therefore lists its cheap arms first, so that an early abort still leaves them finished. A config that extends another cannot reorder the pools it inherits: an inherited pool keeps its place, and a pool the config adds comes last.

**Covariates.** An arm is handed the dataset's covariates only where its pool holds `covariates`, as `positional+covariates` does, so running a pool with and without them shows what the measurements add. A pool that holds them on a dataset that declares none is refused. A config that hands a dataset's covariates to no pool goes ahead with a warning, which `--estimate` prints and a launch writes in the run's `logs/driver.log`.

**Written sources.** A proposal is made for one arm, label condition and seed, so a difference between two models on one written pool includes the difference between their proposals ([Stage 2](../../docs/developer/suite.md)). The report says so wherever it compares them.

**Sequences of different lengths.** On such a dataset, a pool that holds `positional` takes its positions in the positional window that the dataset declares. A config with such a pool is refused where the dataset declares no window ([Stage 1](../../docs/developer/dataset.md#variable-lengths-and-the-positional-window)). The k-mer, motif and written sources compute on the whole sequence, so they need no window. `sources.kmer.normalise: per_length` keeps a k-mer count from growing with the sequence's length.

**Stage 3 takes every arm unless its entry leaves the arm out with a reason:**

- `not_reduced` maps a model of the entry to the reason the reduction leaves it out, which leaves it out of the reliability panel too.
- `not_profiled` maps a model to the reason the panel alone leaves it out.
- The panel leaves out a black-box model (one that can be interpreted only through its predictions, such as [XGBoost](https://xgboost.readthedocs.io/en/stable/)) by itself, so `not_profiled` naming one is refused. A model named under both keys is refused too.

The reduction and the panel print every arm they leave out with its reason. The report lists those arms too.

**The composition baseline takes its settings from the grid and its code.** The [baseline](../../docs/glossary.md#baseline) runs first in every run, at the grid's seeds and with its class weighting. Its other settings are fixed in its code, so that it means the same thing for every dataset ([the composition baseline](../../docs/developer/baseline.md)). `--only baseline` runs it alone.

## Source and model settings

**`sources` states how each source is computed.** Its settings for a source serve every pool that holds the source. `SOURCE_SETTINGS`, in [`config.py`](../../seqmaestro/pipeline/config.py), lists the keys of each source. The comments of `_shared.yaml` and `_motifs.yaml` say what each one does. Each source that some pool holds must be stated in full. Since a source that no pool holds is refused, a config that extends `_shared.yaml` and drops every pool holding a source sets that source to `null`. `llm_generated` stays while a pool holds `llm_generated_iterative`, whose proposals are made with its settings.

**`models` holds the settings a run changes for its models:**

- **A Core model** (the sparse linear model, the [EBM](https://interpret.ml/docs/ebm.html), XGBoost or [PySR](https://github.com/MilesCranmer/PySR), which are fitted in SeqMaestro's own environment) takes the few settings its class lists as `settings`, in [`sparse.py`](../../seqmaestro/models/sparse.py), [`additive.py`](../../seqmaestro/models/additive.py), [`boosting.py`](../../seqmaestro/models/boosting.py) and [`symbolic.py`](../../seqmaestro/models/symbolic.py). A comment beside the list says what each setting changes, and each takes a positive number. The class fixes the model's other options, so that an arm's model is the same in every run that leaves these few alone. The loader builds the model once with the config's values, so a value the model refuses (such as an equation of fewer than seven nodes) is refused as the config loads. The dry run prints every setting of each Core model the run fits, with the config's value or the model's default.
- **DEFT and IGSR** take `config_overrides`: a list of `key=value` overrides of the method's own configuration, which the arm's stage passes verbatim to the method's adapter. `deft_or_cart` is DEFT's decision tree, which is plain CART on a pool without `llm_generated_joint` and DEFT on a pool with it. Its overrides reach the tree in both cases, so CART and DEFT share the tree's settings. CART calls no language model, so it ignores any override that configures one. [Stage 2](../../docs/developer/suite.md) says in what order a method takes its settings.

**A setting that several models share is stated once in the grid.** The driver refuses `config_overrides` that restate one, because two places that set one value will eventually disagree:

- `grid.class_weight` is an argument of each Core model, whereas DEFT's tree takes it as the override `tree.class_weight`.
- `grid.generated_features` is the policy for every feature a language model writes, in every arm that runs such code. The [proposer](../../docs/glossary.md#proposer) takes it as its settings, whereas DEFT's generative finder and IGSR take it as overrides. Its `guidance` reaches DEFT and IGSR through their adapters, which add it to the prompts they build from the task description. A task with a hand-written DEFT prompt config ([`deft_conf`](../adapters/deft_conf/README.md)) cannot take guidance on DEFT's arms that write features, so the driver refuses that combination. DEFT stops a feature its language model writes after 60 seconds, a limit fixed in DEFT's code, so `max_seconds` does not reach it.

## The language model

A config never names a language model. The file `.env` names the provider and the model (`SEQMAESTRO_LLM_PROVIDER`, with the provider's key and model, as [`.env.example`](../../.env.example) shows), once for every part of a run that calls one: the proposer, the [judge](../../docs/glossary.md#judge), the [recurrence analysis](../../docs/glossary.md#recurrence-analysis), DEFT and IGSR. Every arm is therefore compared under one model ([configuring the language model](../../docs/user/language-model.md)).

The `language_model` section states how every call is made, by every part that makes one, so that the arms are compared under one setting as well as under one model. Its keys are the fields of `CallSettings`, in [`client.py`](../../seqmaestro/llm/client.py). The comments of `_shared.yaml` give the reason for each value. Each arm's stage passes the model and these settings to DEFT and IGSR as overrides of their own settings, so `config_overrides` that set either are refused. An answer that a run keeps (a proposal, a judgement or a grouping) is stored under the model and its reasoning effort, so a change of either has the model asked again, whereas a change in the retries or the wait keeps the answers already stored.

## The settings a user may change at the launch

Before a launch, the [agent](../../docs/glossary.md#agent) offers these settings to a user who wants to change one ([`process/suite.md`](../../process/suite.md)). The dry run prints the value each takes in the run, except a setting of DEFT or IGSR that the config leaves at the method's own default. The estimate shows what a change does to the time and the cost.

| Setting | What it changes |
|---|---|
| `grid.seeds` | How many fits each arm makes under each label condition. Stability is measured across the seeds, so fewer than five leave few pairs of seeds to measure it on. The time and the cost grow with the count |
| `reduction.n_features` | How many features each reduced model keeps |
| `reliability.pairs`, `reliability.permutations` | The half-sample pairs of the panel's Tier 1 (stability) and the permutations of its Tier 2 (association). The panel's time grows with the pairs, whereas the permutations set the smallest p-value Tier 2 can give |
| `sources.llm_generated`: `proposal_rounds`, `n_per_round`, `n_examples` | How many requests a proposal makes, how many features each asks for, and how many example rows each shows. The cost grows with the first two |
| `sources.llm_generated_iterative.feedback_rounds` | The rounds after the first, each shown a fit of the arm's model. Each is one more proposal |
| `grid.generated_features.repair_rounds` | How many times a written feature that fails on the rows goes back to the model |
| `language_model.reasoning_effort` | How much a model that reasons does before it answers, where the provider takes an effort |
| `models.<model>` | A Core model's settings, which its class lists ([Source and model settings](#source-and-model-settings)) |
| `models.deft_or_cart.config_overrides` | DEFT's `params_generation.n_reflections`, `params_generation.n_samples_per_prompt` and `tree.max_depth`. Its calls grow with the product of the first two, and with the nodes the depth allows |
| `models.igsr.config_overrides` | IGSR's `experiment.total_budget` and `experiment.n_successors`: the nodes its search expands, and how many times its agent runs at each |

## Motifs

The known motifs (`motif_known`) and the discovered motifs (`motif_discovered`) are off unless a dataset's config extends [`_motifs.yaml`](_motifs.yaml) beside the shared config of its kind of target. The fragment adds the arms on both sources and the settings that suit every dataset, which a dataset's config may change like any other. It leaves out the settings that are facts about the dataset, so a run that does not state them is refused:

```yaml
extends: [_shared.yaml, _motifs.yaml]
sources:
  motif_known:
    collection: JASPAR2026_CORE_vertebrates_non-redundant   # fetched at Stage 1
    strands: both                                           # given, for RNA
  motif_discovered:
    strands: both
    annotate_with: JASPAR2026_CORE_vertebrates_non-redundant   # optional
```

- `collection` names a collection of published motifs, such as those of [JASPAR](https://jaspar.elixir.no/), that [`fetch_motif_collection.py`](../prepare/fetch_motif_collection.py) fetched into `data/motif_collections/` at Stage 1.
- `strands` is `both` for DNA, which a factor binds on either strand, and `given` for RNA. Each source states it.
- `annotate_with` is optional. It names a fetched collection with which each discovered motif is compared, so that the report can name the closest known motif.

A config that takes one source only sets the other's settings and arms to `null`. In a blinded run, a pool that holds the known motifs beside a written source is refused. A known motif is named after its factor, so a language model that writes beside it could see a factor's name among the columns a fit relied on. The discovered motifs need the [MEME Suite](https://meme-suite.org/meme/), which `./install.sh --with-motifs` builds into `SeqMaestro_Method_meme` ([motif discovery](../../docs/user/installation.md#motif-discovery)). The driver checks for it before a run with `motif_discovered` starts.

The comments of [`_motifs.yaml`](_motifs.yaml) say what each setting does, and [motifs](../../docs/developer/motifs.md) says how a motif is matched and how a discovery is made. `example_motifs.yaml` runs the motif arms on the example dataset.

## Adding a dataset

1. Finalise the prepared split with [`finalise_dataset.py`](../prepare/finalise_dataset.py), since a config cannot name a split that has no `dataset.json`.
2. Write the dataset's description, `experiments/datasets/<task>.json` ([dataset descriptions](../datasets/README.md)).
3. Write `experiments/configs/<name>.yaml`. It extends `_shared.yaml`, or `_shared_continuous.yaml` for a continuous target, and states `name` and the `dataset` section. The dataset needs no registration anywhere else.
4. Check what the config resolves to. This prints the config as a run would record it, with `extends` merged and any `--set` applied. It works out no run directory, so it suits a config still being edited:

   ```bash
   $CORE experiments/runners/experiment_config.py --config experiments/configs/<name>.yaml
   ```

5. Make the dry run before the launch. It validates the config, makes the driver's checks and prints every command the run would issue, without running any of them:

   ```bash
   $CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --dry-run
   ```

   The dry run needs what the run would need: the language model's credentials in the environment where a stage calls a language model (`set -a && source .env && set +a`), and each interpreter a stage runs under. It also works out the run's directory, so it refuses a config that already has a run, unless `--run-dir` names that run or `--new-run` says that another run is meant.
6. Estimate how long each arm would take on the machine, and decide about any arm the estimate flags as slow ([`process/suite.md`](../../process/suite.md)):

   ```bash
   $CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --estimate
   ```

## Exercising the pipeline cheaply

**On the example dataset**, `example_quick.yaml` runs every stage, from the baseline to the report's figures, in a few minutes and without a language model. It fits fewer arms, at two seeds and with a small reliability panel. Build the example first ([the example dataset](../../docs/user/example.md) gives the commands), and then run:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/example_quick.yaml
```

- The quick run opens the example's [held-out split](../../docs/glossary.md#held-out-split) at the end, since nothing rests on that split.
- It extends `example.yaml` and states only what it changes, so the two cannot drift apart. `example.yaml` runs the shared grid and arms as they are, which takes hours and needs the language model's credentials.
- `example_motifs.yaml` extends `example_quick.yaml` and `_motifs.yaml`, and runs the arms of both motif sources beside the quick run's, once the collection it names has been fetched and the MEME Suite built ([Motifs](#motifs)).

**On any config**, `--set` makes a run smaller without a second file. Each `--set` overrides one value, which the run records, so a smaller run is documented like any other:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir experiments/runs/smoke --set dataset.splits=<tiny> \
    --set grid.seeds='[42, 43]' --set grid.n_bootstrap=0 --only l1_logistic/positional
```

- `--set` parses its value as YAML, and a value of `null` removes the key. The section it names must exist already, so a misspelt section is refused instead of being created.
- A tiny split is a dataset like any other, so it needs its own `dataset.json` before a config can name it.
- Two seeds are the fewest that measure stability, which is a property of a set of fits. The analyses that measure it are undefined below two.

**`--only` and `--skip` select stages by name.** An arm's stage is named `model/pool`. A model or a pool alone selects every arm with it, so `--only positional` runs each arm on the positional pool. The dry run prints every command under the name of its stage, which shows the names a config's run has. A name that selects nothing is refused.

`--skip` leaves out stages and nothing else. The reduction and the panel are one stage each, with one process per arm, so `--skip deft_or_cart` leaves out the tree's arms and still starts their reductions and panels. These fail where the arms' cells are missing, which holds the [reveal](../../docs/glossary.md#reveal) back. To leave an arm out of a run, remove it from the config file, or drop it with `--set`.
