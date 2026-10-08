# Hand-written prompt configs for DEFT

**Most datasets need nothing here.** DEFT's prompts are templated on `{target_name}` and
`{dataset_info}`, so the adapter fills those from `../../datasets/<name>.json`, the description the
proposer and the judge also read, and generates the config into the run directory. Every consumer
reads the one description file, so there is nothing to keep in step.

A generated config inherits the prompts DEFT writes for the target's kind: `default` for a binary
target, `regression` for a continuous one, which asks for features that separate higher values of
the target from lower ones.

This directory is for the exception: **when the prompt wording is itself part of what is being
compared.** The case it was made for is a config that reproduces DEFT's published prompts verbatim.
An arm measured against published numbers has to ask what the paper asked, and a reworded prompt
would quietly make it a different method.

A config here belongs to one dataset, so it stays on the machine that holds the dataset and git
ignores it, like the dataset's other files ([`../../prepare/README.md`](../../prepare/README.md)).
The repository holds none, so make the folder `params_generation/` when you write the first.

## How resolution works

`--task <name>` resolves in this order:

1. `params_generation/<name>.yaml` here → used as-is.
2. `../../datasets/<name>.json` → a config is generated into the run's `deft_conf/`.
3. Neither → the adapter stops, before the subprocess starts, listing what is available.

Which one was used is printed (`prompts: hand-written …` / `prompts: generated from …`) and the
generated file is written into the run, so a frozen run records which prompts produced it.

**Where DEFT is also told about its columns** (a covariate's description, or a positional window;
see [`../README.md`](../README.md)), the adapter writes `dataset/handed.yaml` into the run's
`deft_conf/`. Hydra reads one `--config-dir`, so a hand-written prompt config is then copied in
beside it, unchanged, and the frozen run holds both.

**A hand-written config is selected by its presence, with no flag.** A flag can be forgotten, and
forgetting it would swap the paper's prompts for generated ones while the run completes normally.
With presence as the trigger the safe choice happens by default, and the printed line and the
recorded copy make the choice visible.

## Writing one

Inherit from DEFT's default config, or from `regression` for a continuous target, and override only
what your task changes. A config that drops the `defaults` block silently loses the population
size, the reflection count and every naive-feature setting, which are method parameters a dataset
should not restate. DEFT refuses prompts written for another kind of target than the dataset's.

```yaml
defaults:
  - default        # DEFT's own, from the installed package
  - _self_

target_name: "..."
# ... only the prompt strings you actually need to differ
```

Two tests in `tests/test_deft_adapter.py` guard this. One fails if a config here drops
`defaults: [default, _self_]` (or `[regression, _self_]`). The other fails if a config states a sequence layout inconsistent
with its dataset description. A hand-written config restates the layout in its own prose, so the
test checks that the numbers in the description's layout (such as the window length and the site's
position) appear in the config, instead of comparing the wording.

Each file here must still have a matching `../../datasets/<name>.json`, because the proposer and
the judge read that and know nothing about this directory.

## Why any of this is on our side

DEFT's prompts are templated on the task, and a wrong description fails **silently**: the model is
told, confidently and in detail, that it is looking at something it is not, and the run completes.
Which dataset is being run is SeqMaestro's decision, and a method that ships a description of one
dataset has to be edited to run another. `methods/deft/` therefore describes no dataset. It ships
only `params_generation/default.yaml`, and everything task-specific reaches it through hydra's
stock `--config-dir` flag, so these files also work when `deft-tree` is run by hand.
