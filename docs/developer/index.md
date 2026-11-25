# Developer guide

This guide is for anyone who works on SeqMaestro's code. Its pages explain why each stage computes what it does, what a number can be compared with and where it breaks down, and how the parts fit together. What each module, class and function is (its signature and its fields) is in the [API reference](../api/index), generated from the docstrings. The [terminology](../glossary.md) defines the terms every page uses.

## Setting up

Install as a user does ([installation](../user/installation.md)): the installer builds every conda environment, and installs SeqMaestro into `SeqMaestro_Core` with its tests and this documentation's toolchain. Each part of SeqMaestro runs in an environment of its own, and [environments](environments.md) says which environment runs what and why they are kept apart. The commands below name each environment's Python by a variable, which one command sets:

```bash
eval "$(./install.sh --interpreters)"     # CORE, DEFT, IGSR, PYSR, LOCALUTILS
```

## Checks before a commit

Seven checks must pass before a commit, each run from the directory shown:

```bash
$CORE -m pytest tests/ -q && $CORE -m ruff check .                        # the repository root
(cd methods/deft && $DEFT -m pytest -q && $DEFT -m ruff check .)
(cd methods/igsr && $IGSR -m pytest -m "not slow" -q && $IGSR -m ruff check .)
rm -rf docs/_build && $CORE -m sphinx -b html docs docs/_build/html -W --keep-going
```

- The root suite runs in parallel (`-n 6`, set in `pyproject.toml`) and makes no call to a language model.
- DEFT's suite loads the repository's `.env` and makes a few live calls to the provider it names.
- IGSR's live tests are marked `slow`, and the command above leaves them out.
- Each vendored method is tested from its own directory with its own interpreter, because the vendored methods and the library never import each other.

**The documentation is built with warnings as errors**, after the build directory is cleared, because the API reference is generated at each build and a stale cache can hide a broken link. Open `docs/_build/html/index.html` to see the result. The build is a check on code changes too, since every docstring of `seqmaestro/` is part of the API reference.

**GitHub Actions publishes the site** (`.github/workflows/docs.yml`): it builds the site as above, and deploys it with GitHub Pages wherever the repository is public. The repository's Pages settings must take their source from GitHub Actions (**Settings > Pages**). The public repository's site is at <https://drshushen.github.io/SeqMaestro/>, the address `html_baseurl` gives in `docs/conf.py`, and the README links to its pages.

## The pages

[Architecture and layout](architecture.md) is the place to start: where everything is, the two boundaries that shape the design, and a map of the packages.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Architecture and layout
:link: architecture
:link-type: doc

Where each part of the repository is, the two boundaries that shape the design, and a map of the packages.
:::

:::{grid-item-card} Stage 1: the prepared dataset
:link: dataset
:link-type: doc

The record Stage 1 writes for every dataset, the checks each dataset passes, and what the later stages take from it.
:::

:::{grid-item-card} Stage 2: the suite
:link: suite
:link-type: doc

What makes two cells comparable, how each kind of arm is fitted, and how the stage resumes.
:::

:::{grid-item-card} The composition baseline
:link: baseline
:link-type: doc

The reference model every arm is compared with, and why it is the same for every dataset.
:::

:::{grid-item-card} Stage 3A: the reduction
:link: reduction
:link-type: doc

Each arm reduced to a model of a few features, and the three models the reduction keeps.
:::

:::{grid-item-card} Stage 3B: the reliability panel
:link: reliability
:link-type: doc

Three tiers of evidence for each feature of a reduced model: stability, association and necessity.
:::

:::{grid-item-card} The held-out split and the reveal
:link: heldout
:link-type: doc

How the test split stays closed until the reveal, and why the reveal's scores equal scoring at fit time.
:::

:::{grid-item-card} The report
:link: report
:link-type: doc

The layer that prepares a report's numbers, tables and figures, the rules it follows, and how to change it.
:::

:::{grid-item-card} The status page
:link: status
:link-type: doc

The page in each run's folder that shows where the run stands, its design rules, and how to change it.
:::

:::{grid-item-card} The feature record
:link: feature_record
:link-type: doc

The record every arm writes for each feature: who fills each field, how records reach a run, and its limits.
:::

:::{grid-item-card} Motifs
:link: motifs
:link-type: doc

How a motif is matched to a sequence, what a motif feature measures, and the known and discovered motifs.
:::

:::{grid-item-card} The Stage 2 analyses
:link: analyses/index
:link-type: doc

Stability, co-discovery, recurrence, the judge, direct measurement and direction, and the ideas they share.
:::

:::{grid-item-card} Run configs
:link: configs
:link-type: doc

What a config states, how configs share settings, the checks on a config, and the settings a user may change.
:::

::::

The READMEs beside the code are shown here too, each kept where it lives so that it is written once: [running experiments](experiments.md), [preparing a dataset](preparation.md), [dataset descriptions](descriptions.md), [the adapters of the vendored methods](adapters.md), [the analysis scripts](analysis-scripts.md) and [environments](environments.md). The instructions the [agent](../glossary.md#agent) follows when it carries out a run are in `process/`, which guides the agent and is not part of this site.

## Maintaining these docs

**A page explains why and how the parts fit, whereas the API reference holds what each part is.** Code changes, and a paraphrase of it does not, with nothing to flag the difference. A page therefore does not paraphrase code or reproduce signatures. It explains what the code cannot say for itself: why a number is computed the way it is, what it can be compared with, where it breaks down, and how the stages hand over to each other. Where a reader needs a procedure that a module's docstring already describes, the page summarises it in a sentence and links to the module.

- **Each fact is written once**, and any other page that needs it links to it. A README kept beside the code it describes is shown on the site by inclusion, so it is edited where it lives.
- **A list that the code holds is linked, never copied.** Where a module, a dataclass or a commented YAML file holds a list (the files a stage writes, the fields of a record, the settings of a config with their reasons), a page links to it. A page keeps a list by hand only where the code holds none in one place and a reader needs it at a glance, and it keeps that list short. The API reference shows docstrings but not `#:` comments, so an explanation meant for it goes in a docstring, and a list explained by comments is linked as a source file.
- **A file layout is listed in the docstring of the module that names the files**, such as {py:mod}`seqmaestro.pipeline.layout` for a run's directory, and a page links to it.
- **The maps are kept by hand**: the module map on the [architecture page](architecture.md), the [documentation map](../map.md) and the [terminology](../glossary.md). A map drifts visibly, since a reader notices a missing entry.
- **A page about a stage, a record or an analysis names the code it tracks** in a note at its top, and changes in the same commit as that code.
- **Results stay out of the pages.** A finding about a dataset goes in its run's report or the dataset's README, and a worked example on a page is labelled as an illustration of the arithmetic.
- **A page states what is.** It holds no history of the code and no notes on one piece of work.

Links follow three conventions, so that a page works on GitHub as well as on the site:

1. **A term** is linked to its entry in the terminology on its first use in a page, by a markdown link to the term's heading (`[arm](../glossary.md#arm)`). A new term is added to the terminology, in alphabetical order, before a page uses it.
2. **A module, class or function** of the package is linked to the API reference on the pages of this guide (`` {py:mod}`seqmaestro.reduction.eliminate` ``, or `py:class` and `py:func`). A README beside the code, which is also read on GitHub, links to the file instead (`[eliminate.py](../../seqmaestro/reduction/eliminate.py)`), as any page does for a script outside the package.
3. **A file outside `docs/`** is linked by its relative path, and the site's build turns the link into one to the file on GitHub (`docs/conf.py`).

```{toctree}
:hidden:

architecture
dataset
suite
baseline
reduction
reliability
heldout
report
status
feature_record
motifs
analyses/index
configs
experiments
preparation
descriptions
adapters
analysis-scripts
environments
```
