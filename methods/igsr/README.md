# IGSR: influence-guided symbolic regression

IGSR fits a model `y = Σ_j w_j · ψ_j(x)` by least squares, where a language model proposes the terms
`ψ_j`. Each term's **influence**, the rise in validation error when its weight is set to zero,
decides which terms are kept, and a Monte-Carlo tree search explores the sequence of proposals and
prunings. The result is a short linear combination of terms a reader can interpret.

Paper: [arXiv:2605.29184](https://arxiv.org/abs/2605.29184).

> **This is a vendored and modified copy.** Where it came from is in [`VENDOR.md`](VENDOR.md), and
> how it differs from upstream is under [What changed](#what-changed). A result from this copy is a
> result of IGSR as published only for the expression form on a numeric table.

## Install

Its own conda environment, as SeqMaestro keeps one per method:

```bash
conda create -n SeqMaestro_Method_igsr python=3.12 -y
uv pip install --python "$(conda info --base)/envs/SeqMaestro_Method_igsr/bin/python" -e "methods/igsr[dev]"
```

The import is `igsr`. `src/` is a container directory.

## Credentials

Runs read SeqMaestro's root `.env`, exported before the run (`set -a && source .env && set +a`).
`.env` names one provider for a SeqMaestro run, and `src/igsr/conf/llm/` has a config for each,
which reads that provider's variables: `llm=seqmaestro_azure` for Azure OpenAI,
`llm=seqmaestro_azure_foundry` for a model deployed on Azure AI Foundry, `llm=seqmaestro_openai`
for the OpenAI API and `llm=seqmaestro_claude` for the Claude API. Every call goes through LiteLLM.
A config may state a `reasoning_effort` for every call, as the Foundry one does to turn off the
reasoning its open models may otherwise do before they answer. Nothing in this directory holds a
key, and the configuration is printed and logged with the key redacted.

## Running it

The copy is called in process, with its configuration composed by `compose_config` and the dataset
handed over as a `DataBundle`:

```python
from igsr.compose import compose_config
from igsr.dataset import DataBundle
from igsr.logging_setup import configure_logging
from igsr.method.igsr import igsr

configure_logging(log_level="INFO")  # importing igsr configures no logging and writes no file
cfg = compose_config(["llm=seqmaestro_azure", "experiment.term_language=function"])
bundle = DataBundle(name="...", data_settings=None, dataset_train=train, dataset_validation=val,
                    dataset_test=None, target_columns=["y"], data_dictionary="what the data is",
                    column_descriptions={"seq": "the sequence", ...})
result = igsr(cfg, seed=42, data_bundle=bundle)
result["best_model"].predict(new_rows)   # the chosen terms, refitted on the training split
```

The search chooses the iteration with the lowest validation MSE. The test split is optional, and
evaluation on it is reporting only. `result["best_model"]` is a `FittedModel`, which saves to JSON
(`to_dict`, `from_dict`) and predicts new rows on its own.

## Two term languages

`experiment.term_language` sets how a term is written.

- **`expression`**, the method as published: a numpy expression over named numeric columns, such
  as `np.sin(x2) * x1`. The prompt previews the first values of every column.
- **`function`**: a Python function `def <name>(X)` of the rows as a pandas frame, returning one
  number per row, NaN where undefined. It reads the columns by name, so it can compute anything a
  short function can over a numeric table or over text columns of any length, such as a motif's
  count in a sequence. The model replies in JSON, giving each term a name, a description and a
  rationale with its code. The prompt shows each column (a text column with its range of lengths)
  with its description, and example rows with their target.

In the function form a term is run on the training rows before it is accepted, and refused if it
raises, returns other than one number per row, is undefined on more than `max_undefined_share` of
the rows, or is constant. A refused term goes back to the model with its failure,
`repair_rounds` times per proposal. Up to that share, undefined rows take the term's median on the
training rows, on every split. Terms that compute the same thing are one term. The settings are
under `experiment.function_terms` in `src/igsr/conf/experiment/igsr.yaml`, and
`src/igsr/method/function_terms.py` explains each.

## Running model-written code

In the function form the model writes code and the search runs it, and the prompt shows rows of the
dataset, so a crafted dataset is a prompt-injection path to code execution. The code is checked
before it runs (`src/igsr/code_guard.py`, ported from DEFT's vendored copy): only allowlisted
imports, no `open`, `eval`, `exec`, `getattr` or dunder attributes, and nothing at the top level
but imports and function definitions. Each call runs under a timeout on the main thread. **This is
a guardrail, not a security boundary**: only process or container isolation stops a determined
adversary.

## What changed

The search itself is as upstream has it: the proposals, the least-squares fit, the influence, the
pruning, the tree search and the history. What differs:

| | Upstream | Here |
|---|---|---|
| Which iteration is the result | the one with the lowest error on the test split | the one with the lowest error on the validation split. Its test metrics are returned for reporting only |
| The seed of the tree search | 42, whatever the run's seed | the run's seed |
| The fitted model | not kept, so a result cannot predict | saved with its terms, weights and intercept (`method/fitted_model.py`), and it predicts new rows |
| Term languages | expressions over numeric columns | that, and Python functions of the rows ([Two term languages](#two-term-languages)), with the guard on model-written code |
| The dataset | loaded by name from the config | handed in as a `DataBundle`. The copy has no loaders, and the test split may be absent |
| The XGBoost-predictor ablation | present | removed, with its machine-wide lock file |
| A round whose pruning keeps no term | raises | is recorded, and its model predicts the mean |
| Logging | creates `./logs/` and a file handler on import | configures nothing until `configure_logging` is called |
| The configuration | printed whole at the start, the API key included | printed with the key left out, and composed in process: no dataset group, no Hydra run directory, no default language model |
| Packaging | uses `tabulate` without declaring it | a src-layout package `igsr` that declares it and ships its configs as package data |

## Tests

```bash
cd methods/igsr
python -m pytest -m "not slow" -q   # offline; the language model is stubbed or scripted
python -m pytest -m slow -q         # live: one call per form, with the credentials above
python -m ruff check .
```
