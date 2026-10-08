# Run configs: what one experiment *is*, in one file

One file per dataset variant, read by `../runners/run_experiment.py`. It is the whole statement of
a run: which data, which grid, which arms. Nothing else configures the stages of a run, and no
stage has a value of its own to fall back on for a setting the config states.

```bash
eval "$(./install.sh --interpreters)"     # CORE and the other environments' pythons
$CORE experiments/runners/run_experiment.py --config experiments/configs/example.yaml
```

`_shared.yaml` and `_shared_continuous.yaml` are in git: they are the grid and the arms every
dataset runs, versioned with the code that runs them. A dataset's config stays on the machine that
holds the dataset, with its other files, and git ignores it
([`../prepare/README.md`](../prepare/README.md)). The dataset configs in the repository are those
of the example dataset: `example.yaml`, and `example_quick.yaml`, which cuts it down to a run of a
few minutes (below).

The **resolved** config is copied into each run directory as `config.yaml`, with the version of its
shape, and every stage of the run takes its settings from that copy. Re-running the driver against
a directory whose recorded config differs is refused, so two configs never mix in one run.
`--allow-config-change` continues such a run all the same, for a change that does not affect the
cells already finished, and keeps the earlier config beside the new one as
`config_superseded_<n>.yaml`.

## Why a file and not flags

Settings carried loose, as environment variables or as flags retyped per stage, have three costs.
They are invisible unless you read the driver. One left unset takes a default defined somewhere
else. And nothing records what a finished run was configured with, so a resume cannot tell a
finished cell from one computed under different settings.

## The sections

| | |
|---|---|
| `version` | The version of the config's shape: its sections, their keys and what each means. A file may leave it out. The copy a run records always states it, and a stage refuses a run that recorded another version than the code reads |
| `name` | **Required.** Names the run directory and every notification |
| `description` | Why this variant exists, when a dataset has more than one. Free text |
| `dataset` | **Required**: `task`, `splits` and `blinded` (see below) |
| `grid` | **Required in full**: `seeds`, `labels`, `train_fraction` (the share of the training rows each seed fits on), `n_bootstrap`, `permutation_scope`, `class_weight` (for a binary target only, below), `permutation_repeats`, `judge` (whether the significance judge runs; `_shared.yaml` leaves it out), `judge_direction`, and `generated_features` (below) |
| `analyses` | **Required in full**: `top_n`, the features of a cell each Stage 2 analysis takes, by the model's importance; `judge` (`repeats`, `show_code`, `labels`, `min_answered`, `seed`), how the judge rates where `grid.judge` has it run; and `recurrence` (`kinds`, `items_per_group`, `draws`, `across_arms`, `seed`), how the recurrence analysis groups. `_shared.yaml` states them. Each decides what a table of the run holds, so an analysis takes them from the run's recorded config and has no flag for any |
| `arms` | **Required.** Each pool of sources, the models fitted on it, and what Stage 3 leaves out of each (see Arms, below) |
| `sources` | How each source the pools hold is computed: `kmer`, `llm_generated` and `llm_generated_iterative`. Required for every source some pool holds, and refused for one no pool holds. `llm_generated` is also required while a pool holds `llm_generated_iterative`, which writes with its settings |
| `models` | The settings of the models that take any, `deft_or_cart` and `igsr`: their methods' `config_overrides` |
| `reduction` | Stage 3A: `n_features`, required. Every arm is reduced but those its entry leaves out under `not_reduced`. `_shared.yaml` sets it, and without the section there is no reduction stage |
| `reliability` | Stage 3B, which needs `reduction`: `pairs`, `permutations`, `cluster_rho`, `max_modal_share`, `planted_count`, `planted_signal_sd`, `check_arm` and `seed`, all required. The panel profiles every reduced arm but a black-box model's and those its entry leaves out under `not_profiled`, and `check_arm` names one of them as `model/pool`. `_shared.yaml` sets it. See [`../../docs/reliability.md`](../../docs/reliability.md) |

**`dataset` names the prepared split and its description, with nothing defaulted**: `splits`, a
directory Stage 1 finalised, and `task`, which selects the description in `../datasets/`. A default
here would silently apply one dataset's answer to another. For `task`, that would mean a language
model is told the wrong thing and answers confidently.

**`dataset.blinded` says whether the run withholds a fact from the language models**, the fact its
description holds in `judge_only` (see [`../datasets/README.md`](../datasets/README.md)). It is
stated in every config, so whether a run was blinded shows in its config and its report without
opening the description. Before a run starts, the driver refuses a flag that disagrees with the
description: `true` where the description withholds nothing, or `false` where it has `judge_only`.
It also refuses a blinded run whose models would be shown a withheld word in a column name or in a
covariate's description. `run_arm.py` makes both checks too, whenever an arm it is to fit calls a
language model, so a stage started by hand is covered.

**The columns belong to the dataset, not the config.** Which column is the sequence, which the
label, which the per-row weight and which the covariates are read from the split's `dataset.json`
when the config is loaded, so they are stated once, and no stage takes a column as a flag. A
config may restate a column, but only to agree with the dataset, and a
config that disagrees with its dataset is refused. (The copy recorded beside a run restates them, so
the run describes itself.) A config naming a split that was never finalised, or whose `dataset.json`
records a failed check, is refused too. See [`../../docs/dataset.md`](../../docs/dataset.md).

**`grid` is required in full** for a related reason. A grid that fills in its own bootstrap count
is a grid whose intervals mean something different from the run it is being read beside, with
nothing in either to say so.

**`grid.generated_features` is the one policy for features a language model writes**, in every arm
that runs a language model's code:
- `repair_rounds`, how many times a round's failed candidates go back to the model with their
  failures;
- `max_undefined_share`, the largest share of rows a kept feature may be undefined on;
- `max_example_length`, how much of one sequence an example in a prompt shows;
- `max_seconds`, the longest one feature may run on the rows it is checked on, in whole seconds.
  Past it the feature has failed, and goes back to the model like any other failure. On more rows
  it is given proportionally longer.

`_shared.yaml` states it. Each arm's stage takes it from the run's config: the proposer as its
settings, and DEFT's generative finder and IGSR as config overrides. IGSR applies the time limit to
each term on each split. DEFT's limit is 60 seconds, fixed in its code, so `max_seconds` does not
reach it.

**An unknown key is refused**, as in `../datasets/`: a misspelled one is a setting silently missing
from the run, and the run completes without it. The same holds for a pool or a model name, so a typo
cannot quietly skip an arm. Model names, pools, class weights and permutation scopes are checked
against the library's vocabularies, so a config cannot name a model that does not exist, and each
model is checked against its pool and the dataset (see Arms, below).

## Sharing, via `extends`

`_shared.yaml` holds the grid and the arm settings every dataset uses. Each dataset config extends
it and states only what is its own. `_shared.yaml` is not runnable alone, because it names no
dataset.

**A dataset with a continuous target extends `_shared_continuous.yaml`**, which extends
`_shared.yaml` and changes only what the kind of target changes: no `class_weight`, and IGSR, which
fits a continuous target only, with its settings. `l1` in the shared arms is the sparse linear model
of the dataset's kind, so the arms need no second list. `extends` chains, so the two kinds share
every other setting. The dataset's `dataset.json` states its kind, and a config whose settings do
not fit it is refused: a binary model or a class weighting on a continuous target, a missing class
weighting on a binary one, or IGSR on a binary target. The tree needs nothing more: DEFT grows a
regression tree for a continuous target, by variance reduction, with prompts written for it.

Merging is recursive. A list **replaces** the inherited list instead of being appended to it, and
`null` **removes** a key, so a config can drop an arm as easily as retune it:

```yaml
extends: _shared.yaml
name: example_no_kmer
dataset: { ... }
sources:
  kmer: null                       # no pool below holds the k-mers
arms:
  positional+kmer: null            # this dataset does not run the pool
  positional:
    models: [l1, ebm, deft_or_cart]  # replaces the shared list wholesale
```

## Arms

An **arm** is one model fitted on one pool of sources. `arms` lists each pool by its name, with the
models fitted on it, so each model of an entry is one arm:

```yaml
arms:
  positional:
    models: [l1, ebm, xgboost, pysr, deft_or_cart]
  kmer+llm_generated_iterative:
    models: [l1, ebm, xgboost]
    not_profiled:
      ebm: "EBM over the k-mers: each of the panel's 100 selections refits it about 40 times"
```

An arm's cells are `results/<model>/<pool>/<labels>/seed_<n>`, and its reduction and its panel are
kept under the same two names. The driver runs each arm as one stage, with one process per labels
and seed, and offers the arms the machine in the order the config lists them. A config that
extends another cannot reorder the pools it inherits: an inherited pool keeps its place, and a
pool the config adds comes last.

**A source** is one way of producing candidate features:

| source | the features | produced |
|---|---|---|
| `positional` | one indicator per position and base, over the sequence or the dataset's positional window | from the sequence, enumerated |
| `kmer` | a count of every k-mer with a length in `sources.kmer.k_range` | from the sequence, enumerated |
| `llm_generated` | functions of the sequence a language model writes, in one round of calls | by the proposer |
| `llm_generated_iterative` | the same, written over several rounds, each shown a fit of the arm's model | by the proposer |
| `llm_generated_joint` | functions a language model writes while the model is fitted, by a procedure only that model runs: DEFT's candidates at each node of its tree, and IGSR's terms | inside the model's fit |
| `covariates` | the measurements the dataset supplies with each sequence, unchanged | by the dataset |

**A pool** joins its sources with `+` in the order of the table, so `positional+kmer` hands a model
both vocabularies, and `positional+covariates` the positions and the covariates. A pool written in
another order is refused, so that one set of sources has one name. A pool holds at most one written
source, so that each arm stands for one way of writing features.

**A model** is one of:
- `l1`, the sparse linear model for the dataset's kind of target: `l1_logistic` for a binary target
  and `l1_linear` for a continuous one, which may also be written by name. The recorded config and
  the cells name the model `l1` resolved to.
- `ebm`, `xgboost`, and `pysr`, one equation found by PySR's symbolic search. PySR's search runs in
  `SeqMaestro_Method_pysr`, on the threads its process is given (`seqmaestro/execution.py`), so the
  driver checks that interpreter before every stage that fits a PySR model: its arms, the reduction
  and the panel.
- `deft_or_cart`, DEFT's decision tree. It is plain CART on a pool without `llm_generated_joint`, and
  DEFT on a pool with it, where a language model writes further candidates at each node. Both share
  the tree, its splitting and stopping rules, and the naive candidates DEFT derives from the sequence.
- `igsr`, influence-guided symbolic regression: a linear model over terms a language model writes as
  functions of the whole sequence, each kept by its influence on the fit. It fits a continuous target
  only.

`seqmaestro/arms.py` declares which sources each model can be handed. A combination it does not
allow is refused when the config is loaded, with the reason, before anything is spent:

| model | sources it can be handed | sources its pool must hold |
|---|---|---|
| `l1`, `ebm`, `xgboost`, `pysr` | `positional`, `kmer`, `llm_generated`, `llm_generated_iterative`, `covariates` | |
| `deft_or_cart` | `positional`, `kmer`, `llm_generated_joint`, `covariates` | `positional`, from which DEFT derives its candidates |
| `igsr` | `llm_generated_joint`, `covariates` | `llm_generated_joint`, the terms its model is fitted on |

The tree takes `kmer` or `llm_generated_joint`, and not both: DEFT's prompts show every column it is
handed, and thousands of k-mer counts would swamp them. On `kmer`, CART searches a threshold for
each count at every node, so it costs several times what it costs on the positions alone.

The loader also refuses a model that does not fit the dataset's kind of target or cannot take its
per-row weights (IGSR fits without them), `covariates` on a dataset that declares none, and a model
named twice in one entry. A dataset with covariates whose config hands them to no pool gets a
warning, which the driver prints, and the run goes ahead.

**Stage 3 takes every arm unless its entry leaves the arm out, with a reason.** `not_reduced` maps a
model of the entry to why the reduction leaves it out, which leaves it out of the panel too, and
`not_profiled` to why the panel leaves it out. The panel leaves out a black-box model by itself
(`xgboost`, whose fit cannot be interpreted feature by feature), so `not_profiled` naming one is
refused. The reduction and the panel print every arm they leave out with its reason, and the report
lists them.

**`sources` says how each source is computed**, once for every pool that holds it:
- `kmer`: `k_range`, as `[min, max]`, and `normalise`. `counts` counts each k-mer in total, and
  `per_length` per k-mer position in the sequence, so that a count does not carry the sequence's
  length.
- `llm_generated`: `proposal_rounds`, the calls per proposal, each told the names proposed before it;
  `n_per_round`, the features asked for per call; and `n_examples`, the labelled rows shown. The
  iterative source writes with the same settings, so its first round makes the one-round request.
- `llm_generated_iterative`: `feedback_rounds`, the rounds after the blind first one.

A source some pool holds must be stated, and a source no pool holds is refused, since a setting that
nothing uses is a mistake the run would not show. A config that extends `_shared.yaml` and drops
every pool holding a source sets that source to `null`. `llm_generated` stays while any pool holds
`llm_generated_iterative`, whose proposals are made with its settings.

**Each arm's proposal is made for that arm.** The proposer's cache is keyed by the arm, so no two
arms share a proposal, not even two models on one pool, and the iterative source's blind first round
is a proposal for its arm too. A difference between two models
on one written pool therefore includes the difference between their proposals, and the report says
so where it compares them. A proposal is made per arm, labels and seed, and the language-model
slots of the driver's budget bound how many are made at once.

**A vocabulary beside a written source** is handed to the model with the proposal:
`kmer+llm_generated` fits the model on the k-mers and the proposal together, and the proposer is
asked as it would be without them. In the iterative source, each informed round is shown which of
all the pool's columns the fit leaned on, the k-mers included, and the language model can answer
with features that build on them.

**`covariates` in a pool hands the model each covariate as a column**, and shows the arm's proposer
what each one measures, so it may write features that use them. An arm whose pool does not hold them
is given rows without them, so neither its model nor its proposer sees them. A tree whose pool holds
them hands DEFT each covariate as a column, with the dataset's description of what it measures, and
DEFT's naive candidates split on its values as CART's would.

**On a dataset whose sequences differ in length**, every pool holding `positional` takes its
positions in the positional window the dataset declares, and the tree computes its positional
candidates in the same window by the same rule. A config with such a pool on such a dataset without
a window is refused. The written sources compute on the whole sequence at whatever length it comes,
and `kmer` counts the whole sequence, so neither needs a window. `sources.kmer.normalise:
per_length` keeps a count from carrying the sequence's length.

**`class_weight` is stated once, in the grid, and translated per model**: an argument of the models
fitted in Core, and a `tree.class_weight` config override for the tree. It is required for a binary
target and refused for a continuous one, which has no classes to weight. A model whose
`config_overrides` also sets it is refused, because two places that set one value will eventually
disagree.

**`models.<model>.config_overrides`** is passed to DEFT, or to IGSR, verbatim as repeated
`--config-override key=value`: each method's configuration, with its budget.
`deft_or_cart`'s reach the tree under both of its finders, so CART and DEFT share their tree's
settings, and CART, which calls no language model, ignores the rest. The grid's
`generated_features` is the policy of every arm that writes features, so it is translated into each
method's settings, and `config_overrides` restating one is refused. A model no arm fits may not be
given settings.

**No config names a language model.** Which provider and model a run calls is set in `.env`
(`SEQMAESTRO_LLM_PROVIDER`, with the provider's key and model: `.env.example`), once for the
proposer, the judge, the recurrence analysis, DEFT and IGSR, so that every arm is compared at one
model. Each arm's stage passes it to DEFT or IGSR as the cell starts, and `config_overrides` that
sets one (`llm_models=`, `llm=`) is refused. A proposer's cell records the model it called, in
`token_usage.json`, and a DEFT or IGSR cell records the provider's config among the overrides in
its bundle.

**The composition baseline is not an arm and has no section.** It runs first in every run, at the
grid's seeds and with its class weighting, and nothing else about it can be configured (see
[`../../docs/baseline.md`](../../docs/baseline.md)). `--only baseline` runs it alone.

## Adding a dataset

Finalise the prepared split first (`../prepare/finalise_dataset.py`), since a config cannot name one
that has no `dataset.json`. Then write `<name>.yaml` extending `_shared.yaml` with a `dataset`
section, and write the matching description in `../datasets/<task>.json`. Nothing registers it
anywhere else.

Check it before spending compute on it. This resolves, validates and prints every command the run
would issue, without running any of them:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --dry-run
```

The dry run makes the driver's checks before a launch, so it needs what the run would need: the
credentials in the environment where a stage calls a language model
(`set -a && source .env && set +a`), and each interpreter a stage runs under.

To see only what a config resolves to, with `extends` folded in and any `--set` applied, and
without the run directory being worked out at all:

```bash
$CORE experiments/runners/experiment_config.py --config experiments/configs/<name>.yaml
```

That prints the form the run would record. Use it while editing a config, because `--dry-run`
resolves a run directory and refuses if the config already has runs.

Before launching, estimate what each arm would cost on the machine, and decide about any arm it
flags as slow (`process/suite.md`):

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml --estimate
```

## Exercising the pipeline cheaply

**On the example dataset**, `example_quick.yaml` is a run of every stage that calls no language
model: fewer arms, two seeds and a small panel, from the baseline to the report's figures. It took
about eight minutes on eight cores. Build the example first
([`../prepare/README.md`](../prepare/README.md)), then:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/example_quick.yaml
```

It opens the example's test split at the end, since nothing rests on that split. It extends
`example.yaml` and states only what it cuts, so the two cannot drift apart. `example.yaml` itself
runs the shared grid and arms as they are: hours of computation, and the credentials.

**On any config**, `--set` cuts a run down without a second file. It overrides one value, can be
repeated, and is recorded with the run, so a cut-down run is documented like any other:

```bash
$CORE experiments/runners/run_experiment.py --config experiments/configs/<name>.yaml \
    --run-dir experiments/runs/smoke --set dataset.splits=<tiny> \
    --set grid.seeds='[42, 43]' --set grid.n_bootstrap=0 --only l1_logistic/positional
```

`--only` and `--skip` take a stage's name. An arm's stage is named `model/pool`, and a model or a
pool alone selects every arm with it, so `--only positional` runs each arm on the positional pool.
The other stages are `baseline`, the analyses (`analyse_stability`, `analyse_co_discovery`,
`analyse_recurrence`, `judge`), `reduction`, `reliability`, `reveal`, `reliability_after_reveal`,
`report_inputs` and `report_figures`.

`--skip` leaves out stages and nothing else. The reduction and the panel are one stage each, with
one process per arm of the config, so `--skip deft_or_cart` leaves out the tree's arms and still
starts their reductions and panels, which fail where the arms' cells are missing and hold the
reveal back. To leave an arm out of a run, take it out of the run's config, in the file or with
`--set`.

Two seeds are the minimum, because stability is a property of a *set* of runs, and the analyses that
read it are undefined below two. A tiny split is a dataset like any other, so it needs its own
`dataset.json` before a config can name it.
