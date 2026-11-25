# DEFT — Dynamic Feature Generation in Decision Trees

DEFT grows a decision tree in which the features are **generated during construction**: at each
node a language model proposes biologically-motivated sequence features tailored to the
sequences that reached that node, and a reflection loop refines them. The result is a tree whose
splits are named, described and executable.

Paper: [Interpretable DNA Sequence Classification via Dynamic Feature Generation in Decision
Trees](https://arxiv.org/abs/2604.12060) (AISTATS 2026).

> **This is a vendored and modified copy.** It is not the upstream repository, and it does not
> behave identically to it — do not read a result from this tree as a result from the paper
> without checking what changed.
>
> | | |
> |---|---|
> | Upstream | `git@github.com:DrShushen/deft.git`, branch `DrShushen/misc` |
> | Commit | `110378e9143af85fc791207d8666e09c1c602b53` ("Include tests", 2026-07-06) |
> | Copied | 2026-08-05, via `git archive HEAD`; all 183 tracked files md5-verified against upstream |
> | Since then | reduced to the method itself, and modified — see [What changed](#what-changed) |
> | Licence | MIT ([`LICENSE`](LICENSE)) |
>
> There is no nested `.git`, no remote and no submodule: SeqMaestro owns the history from the
> copy onwards.

---

## Install

Per SeqMaestro convention each method gets its own conda environment, named
`SeqMaestro_Method_<method>`:

```bash
conda create -n SeqMaestro_Method_deft python=3.12 -y
uv pip install --python "$(conda info --base)/envs/SeqMaestro_Method_deft/bin/python" \
    -e "methods/deft[dev]"
```

Six runtime dependencies, about 650 MB installed. Plain `pip install -e "methods/deft[dev]"`
from inside an activated environment works equally well; `uv` is just faster.

The environment SeqMaestro runs DEFT in also holds `scikit-learn`, which SeqMaestro's adapter
imports and DEFT does not need. `ENVIRONMENTS.md` at the repository root has the recipe for that
environment and the reason.

The distribution is `deft` and so is the import — `import deft`, `from deft.trees.tree import
AdaptiveDecisionTree`. `src/` is a container directory, not a package.

`requires-python` is `>=3.10` with no ceiling; 3.10 and 3.12 are both exercised against the full
suite. 3.12 is what the environment above builds.

## Credentials

Runs read the **SeqMaestro root `.env`** (see `.env.example` there). Nothing in this directory
holds a key.

```bash
set -a && source .env && set +a     # exports, rather than only setting
```

`.env` names one provider for a SeqMaestro run, and there is a config for each: the model and its
key under `src/deft/conf/llm_models/`, and how the engine calls a model of that family under
`src/deft/conf/llm/`.

| provider | `llm_models=` | `llm=` |
|---|---|---|
| Azure OpenAI | `seqmaestro_azure` | `gpt5` |
| a model deployed on Azure AI Foundry | `seqmaestro_azure_foundry` | `foundry` |
| the OpenAI API | `seqmaestro_openai` | `gpt5` |
| the Claude API | `seqmaestro_claude` | `claude` |

`generic_user.yaml`, `openai_gpt4o_mini.yaml` and `env.yaml` remain as alternatives. `llm/claude.yaml`
differs from `llm/gpt5.yaml` in three settings, each because the Claude API refuses or ignores what
the GPT-5 settings send: no `top_p` beside the temperature, no reasoning effort, and one completion
per request. `llm/foundry.yaml` differs in one: it turns the model's reasoning off
(`reasoning_effort: none`), since the open models on Foundry may otherwise spend the whole reply
reasoning.

All LLM calls go through **LiteLLM**, so `model` is a LiteLLM model string carrying its own
provider prefix (`azure/<deployment>`, `azure_ai/<deployment>`, `openai/<model>`,
`anthropic/<model>`). One Azure subtlety
is worth knowing: a deployment's name appears in no model map, so `base_model` supplies the real
family and is what lets `reasoning_effort` through. The config comments and
`src/deft/llm/engines/litellm_engine.py`'s docstring explain it in full.

## Run

There is no default dataset: `dataset.csv_path` is mandatory, and a run without one stops with
`MissingMandatoryValue` before spending anything.

```bash
python experiments/deft_tree.py --config-name=config \
    llm_models=seqmaestro_azure \
    llm=gpt5 \
    dataset.csv_path=<your training split>.csv \
    dataset.target_column=<your label column>
```

Outputs land in a Hydra run directory, `outputs/<date>/<time>/`:

| File | What |
|---|---|
| `tree_<dataset>.dill` | the fitted tree; dill, because generated features are `exec`'d callables |
| `intermediate_tree.pkl` | snapshot rewritten before each node is expanded; survives a crash |
| `deft_tree.log` | full run log, including the fitted tree and the integrity and token reports |
| `.hydra/` | the resolved config, for the record |

A fast end-to-end check costs about 90 seconds:

```bash
python experiments/deft_tree.py --config-name=config \
    llm_models=seqmaestro_azure llm=gpt5 \
    dataset.csv_path=<your training split>.csv dataset.target_column=<your label column> \
    max_depth=2 params_generation.n_reflections=0
```

**A DEFT run fits and serializes a tree. It does not evaluate.** There is no test-set load and
no metric computation in the run path. SeqMaestro scores the saved tree separately, so that
every arm in the comparison is measured by one implementation.

### A baseline that costs nothing

`feature_finder=cart` runs the same tree, splitting criterion, threshold search and stopping
rules with no language model at all — so a comparison against it isolates the feature generation
and nothing else. It needs no credentials, makes no API call, and the integrity policy is
skipped for it (its question has no meaning without a model).

Its candidates are exactly the naive population DEFT seeds every node with: positional
indicators decoded from the sequence column, plus the covariates. So the CART baseline and a
DEFT run's `llm_split_share` are scored against the same yardstick.

```bash
python experiments/deft_tree.py --config-name=config \
    feature_finder=cart dataset.csv_path=train.csv dataset.target_column=label
```

## Run integrity: did the language model actually contribute?

This is the single most important thing to understand about running DEFT.

DEFT seeds every node's candidate pool with **naive** features — one indicator per position and
nucleotide, such as `pos_50_is_G`, plus one per covariate column — generated locally in Python,
with no API call. So when the language model contributes nothing, DEFT does not fail. It builds
the entire tree out of naive features, and the result fits, serializes, prints and looks exactly
like a real DEFT run.

Every ordinary fault behaves this way: a dead endpoint, a deployment name that does not exist,
a request parameter the provider rejects, generated code that never compiles, generated
features that compile but raise on the data. All of them are retried, logged, and then dropped.

So every run writes **`provenance.json`** into its output directory and prints a summary:

```
Run integrity
  nodes expanded                  : 3
  splits won by an LLM feature    : 1 of 3  (33%)
  splits won by a naive feature   : 2
  nodes with no usable LLM feature: 0
  API requests succeeded          : 15 of 15
  completions returned/requested  : 24/24
```

The headline is the LLM share of splits. Zero means the tree is a naive-feature tree wearing a
DEFT label. `provenance.json` breaks the same thing down per node, following the whole chain
from prompts sent, through proposals parsed and code generated, to features that actually ran
and scored.

The last line should read 1:1. Anything else means you are not getting the number of completions
you configured. On this same run it read **216/36** before the provider's per-request cap was
handled properly — six completions generated and billed for every one asked
for, and the surplus discarded.

By default a run **fails** when the language model did not contribute — at the first offending
node, before a budget is spent:

| Preset | Behaviour |
|---|---|
| `integrity=strict` | default; report and fail |
| `integrity=warn_only` | report, log breaches, never fail |
| `integrity=off` | report only, no checks — upstream's behaviour |

`src/deft/conf/integrity/strict.yaml` documents the failure mode and every threshold, and is the
place to start if you are meeting this for the first time.

**Credentials never reach an artifact.** A fitted tree holds the feature finder, which holds the
engine, which holds the resolved model config — so a dilled tree used to contain the API key in
plaintext. `GenericEngine.__getstate__` redacts on serialization; the model name, deployment,
endpoint and API version are kept, because they record which model produced the tree. A
deserialized engine cannot make API calls, which is intended: a saved tree is a record, not a
live client.

## Running model-generated code

DEFT's method is to have a model write feature functions and then `exec` them. That is the
mechanism, not an implementation detail — and the prompts embed rows of **your** dataset as
in-context examples, so a crafted CSV is a prompt-injection path to code execution here.

Generated code is therefore checked before it runs: no imports outside an allowlist, no
`open`/`eval`/`exec`/`compile`/`__import__`/`getattr`, no dunder attribute access, and nothing at
module level but imports and function definitions. Execution gets a minimal `__builtins__` and a
guarded `__import__`, and the *validated* AST is compiled directly so what runs is what was
checked.

| Preset | Behaviour |
|---|---|
| `code_guard=strict` | default |
| `code_guard=off` | no checks — upstream's behaviour; logs a warning at startup |

**This is a guardrail, not a security boundary.** It raises the cost of an accident or of casual
misbehaviour from zero to substantial. It will not stop a determined adversary; only process or
container isolation does that, and that has deliberately not been built. Do not read it as a
promise of safety when running genuinely hostile data.

Allowed modules are `numpy`, `pandas`, `re`, `math`, `collections` and `itertools`, with `numpy`
and `pandas` also pre-bound as `np` and `pd` — a model writes `np.` without importing it more
often than not, so that raises the yield rather than lowering it. Widen the list without
restating it:

```bash
python experiments/deft_tree.py ... '+code_guard.extra_allowed_modules=[scipy]'
```

A rejected feature is logged and discarded exactly like one that fails to compile, so it shows up
as a falling `min_usable_feature_yield` — an allowlist that is too tight announces itself.
`src/deft/conf/code_guard/strict.yaml` explains the threat and every setting.

## Cost, and stopping before it gets away from you

**Token accounting is always on.** Every run writes `token_usage.json` and prints:

```
Token usage
  total tokens                    : 8,532
    prompt                        : 5,318
    completion                    : 3,214
  API calls                       : 15
  mean tokens per node            : 2,844
  wall clock                      : 31s
```

**Spend limits are opt-in**, through the `budget` config group:

| Preset | Behaviour |
|---|---|
| `budget=unlimited` | default; measure without capping |
| `budget=pilot` | a worked capped example |

Set any of `max_total_tokens`, `max_api_calls` or `max_wall_seconds`; `null` means no limit of
that kind. The last two are backstops independent of token size — the retry policy allows 100
attempts per request, so a flaky endpoint can otherwise produce a great many cheap calls, or
simply stall.

**Reaching a limit does not fail the run.** The tree stops *growing*: the node being expanded
becomes a leaf with the correct majority prediction, every node still queued becomes a leaf too,
and `fit` returns a complete, valid tree that is simply shallower than `max_depth` would allow.
It is saved and scored like any other. Aborting would throw away a tree that is most of the way
built and already paid for.

The result says so about itself — `stopped_early` and `stopped_reason` in `token_usage.json`,
`budget_stopped` on the tree, and a STOPPED EARLY line in the report. Nodes cut short this way
are exempt from the integrity checks above: running out of budget is not the same failure as the
language model not contributing.

A limit is a target rather than a hard ceiling. A batch of requests goes out concurrently, so
spend overshoots by roughly one batch — measured at 4,130 against a 4,000 limit. The report shows
what was actually spent.

Choose a number by measuring, not guessing: run once with `budget=unlimited`, read
`mean_tokens_per_node`, and multiply by the number of internal nodes you expect. Reflection
dominates — each round adds another generate-and-score pass per node.
`src/deft/conf/budget/unlimited.yaml` covers all of this in more detail.

## Tests and linting

```bash
pytest                # LLM tests skip themselves unless .env is filled in
pytest -m "not llm"   # offline only
pytest -m llm         # only the LLM-backed regression tests (costs tokens)

ruff check .          # clean; a finding means something new
ruff check --fix .    # safe autofixes, including import order
```

Four of the tests call the language model and are marked `llm`. The others run offline.

Ruff's rules are **pinned in `pyproject.toml`** rather than left to its defaults, which have
widened between releases — an unconfigured run reported a different set on 0.16 than on 0.6, so
"clean" meant nothing. Line length is 120, imports are sorted by `I`, and two rule families are
deliberately excluded: `S` (security) and `BLE` (blind except), because this code `exec`s
model-generated Python on purpose and swallows exceptions on purpose. Both are the method, they
are documented where they happen, and flagging them on every run would only teach the reader to
ignore the output.

`tests/test_class_weight.py` is LLM-free by construction, using `CartFeatureFinder`, so it
verifies the environment without spending anything.

`tests/test_integrity.py` drives a `DeadLlmEngine` whose every request returns `None`, exactly
as the real engine does once retries are exhausted. It asserts that the strict policy stops the
run at the first node with an actionable message, that `warn_only` completes while recording
the breach, and — as executable documentation of what is being fixed — that with checks off the
identical dead engine still produces a complete, well-formed, predicting tree whose LLM share
of splits is zero.

`tests/test_artifact_safety.py` plants a fake key in a model config and asserts it never
reaches a dilled tree, at any nesting depth and under any secret-looking name.

`tests/test_budget.py` asserts the behaviour that matters most about a limit: that a stopped run
returns a complete tree whose every node is resolved and which still predicts for every row,
that it is genuinely shallower than an uncapped run, that nothing more is spent while the queue
drains, and that a strict integrity policy does not fire on it.

`tests/test_request_chunking.py` asserts a caller receives exactly the number of completions it
asked for, whatever the provider's per-request cap.

`tests/test_llm_regression.py` exercises the real API path. It exists because **DEFT degrades
silently**: the feature population is seeded with naive `pos_N_is_X` features, so a dead
endpoint or a rejected request parameter yields a tree that still fits, still serializes and
still prints — just with no generated features in it. The test fits a depth-1 tree on a
synthetic dataset whose label depends only on aggregate upstream G+C content, where the best
single-position rule is right 68% of the time and an aggregate feature is right 100%. It then
asserts that the chosen root split is not a naive feature and that it clears 85% accuracy.

## Use DEFT on your own dataset

For a CSV, no code is needed — the shipped `generic_csv` dataset covers it:

```bash
python experiments/deft_tree.py --config-name=config \
    llm_models=seqmaestro_azure llm=gpt5 \
    dataset.csv_path=train.csv \
    dataset.target_column=paused \
    dataset.sequence_column=raw_sequence
```

Set `params_generation.target_name` and `params_generation.dataset_info` to describe your task.
They are interpolated into every prompt, so they matter more than their size suggests.

**The prompts are task-neutral**, and deliberately vague about the data. They used to describe
RNA polymerase II pausing on a 101-nucleotide window centred on a pause site, which is a confident
and wrong description of anything else — and unlike a missing dataset, a wrong task description
says nothing.

**This package now describes no dataset at all.** Which one is being run is the caller's decision,
and a library shipping a description of one dataset has to be edited to run another. Put your own
group config on hydra's search path instead, inheriting this one so you override only what your
task changes:

```yaml
# <your-dir>/params_generation/promoters.yaml
defaults: [default, _self_]
target_name: "promoter strength"
dataset_info: "200-nucleotide E. coli promoter regions, labelled by whether transcription initiates."
```

```bash
deft-tree --config-dir <your-dir> params_generation=promoters dataset.csv_path=...
```

The model learns the *columns* from the data, never from the prompt: `serialize_features`
describes every column's type and range, and in-context rows show their values. So these
settings only have to supply what the data cannot — what the target means, and what makes a
good feature for it.

### What DEFT is given

| | |
|---|---|
| `csv_path` | **the training split.** DEFT trains on every row of it and does not split. Passing `test_size` or `random_state` is an error, not a no-op — the old plugin split internally and then discarded the held-out rows. |
| `target_column` | the target, kept under its own name |
| `target_kind` | `binary`, the labels 0 and 1, or `continuous`, finite numbers; a column holding anything else is refused. See [A continuous target](#a-continuous-target) |
| `sequence_column` | equal-length strings, positionally aligned to a landmark, or strings of different lengths read in `positional_window`. `null` for a dataset with no sequence: DEFT then uses pre-split per-position columns (`seq_-50 … seq_50`), or covariates alone. |
| `positional_window` | how positional features read sequences of different lengths: `{length, anchor, pad}`, `length` positions from the `start` or the `end` of each sequence. A longer sequence is cut at the far end. A shorter one is padded at the far end with `pad: true`, which adds a `pos_<i>_is_pad` feature per position, and refused with `pad: false`. `pad` has no default. Generated features still read the whole sequence. |
| `column_descriptions` | what each column means, `{column: text}`, shown in every prompt after the column's type and range. A description of a column DEFT is not given is refused. |
| everything else | covariates. Numeric, boolean, categorical and text columns are all described to the model and available to generated features. |

**Every column name must be a legal Python identifier**, and a dataset whose names are not is
refused at load. This is not tidiness. DEFT has a model write Python that refers to columns by
name, and the model copies the conventions it is shown into the *function* names it chooses — a
dataset carrying `seq_-50` taught it to propose `AT_fraction_upstream_-10_to_-1`, which does not
parse, so the generation was discarded after its tokens were paid for and nothing in the result
said so. DEFT will not rename for you: in one real dataset `-` was an arithmetic minus, a
coordinate sign and a depletion marker in three different columns, and only whoever prepared the
data knows which. Python keywords are fine — a column called `class` is only ever a string
subscript.

**Every column that is not the target is given to the model.** Excluding one that should not be
learned from — an identifier, a leaked score, a redundant encoding — belongs to whatever
prepared the file. `dataset.covariate_columns` (allowlist) and `dataset.drop_columns`
(blocklist) exist for when re-preparing the file is not worth it.

Covariates are not free: on a 164-column dataset a node cost 23,746 tokens against 2,764 for
the same run given the sequence alone, and 95% of that was prompt. The naive baseline covers
covariates too — an identity feature per numeric column, indicators per level for a
low-cardinality categorical one — so `llm_split_share` still compares like with like.

With the sequence optional, DEFT works as a general tabular method, not only a DNA one.

### A continuous target

DEFT grows a regression tree for a numeric target. Three settings say so, and the run refuses any
one of them without the others:

```bash
deft-tree ... dataset.target_kind=continuous splitting_criterion=squared_error \
    params_generation=regression
```

**The criterion is variance reduction**, `offset - p_left * I(left) - p_right * I(right)` as for
Gini, with a node's impurity I its weighted variance divided by the variance of every fitting row.
That unit is fixed once, when the tree is fitted, so a candidate's score reads the same whatever
the target's scale, and the impurity a split removes is a share of the variance the tree began
with. **A leaf predicts the mean of its rows**, weighted by `sample_weight` where the rows carry
weights, and `predict` returns it, with nothing to threshold. `tree.class_weight` weights the
classes of a binary target and is refused for a continuous one.

**The prompts** (`params_generation/regression.yaml`) inherit the default ones and change what
names the task: a feature should separate the rows with a higher value of the target from the rows
with a lower one. Your own prompt config for a continuous target inherits `regression` in place of
`default`. Each prompt config states the kind it is written for (`target_kind`), and a run whose
dataset declares the other kind is refused before any prompt is sent.

A dataset with no `target_kind` is binary, as every dataset was before, and a binary tree refuses a
target that holds anything but 0 and 1.

### Generated candidates that fail, and long sequences in prompts

A generated candidate is run on the node's rows before it is scored, and dropped when it raises,
returns other than one number per row, is undefined (NaN) on too many rows, or is constant where it
is defined. The run log says which, for each one. Four `params_generation` settings govern this,
and each default keeps DEFT as it was:

| setting | default | |
|---|---|---|
| `max_undefined_share` | `0.0` | the share of a node's rows on which a candidate may return NaN and still be kept. The threshold search and the routing send NaN right of every threshold, at fit and at predict alike |
| `repair_rounds` | `0` | how many times a failed candidate goes back to the model, one prompt each, with its code, what went wrong and how to correct that kind of failure (`repair_guidance`). Code at fault is fixed without changing what the feature computes. A constant or mostly undefined feature is changed as little as possible, and comes back under a name and description for what it now computes. Provenance counts the repair prompts, the candidates corrected and those still failing |
| `repair_guidance`, `instructions_code_repair` | shipped wording | how a repair prompt asks for each kind of failure to be corrected, and for its answer |
| `max_sequence_chars_in_prompt` | `null` | the most characters of one text value an in-context row shows. A longer one is shown by its first and last characters with its length stated, since a whole transcript would fill the prompt |

### A plugin, for anything a CSV cannot express

Implement the single abstract method:

```python
from typing import Any
import pandas as pd
from deft.data.dataset_plugins import BaseDatasetPlugin


class MyDatasetPlugin(BaseDatasetPlugin):
    def __init__(self, path: str, sequence_column: str = "raw_sequence",
                 target_column: str = "label"):
        self.path = path
        self.sequence_column = sequence_column
        self.target_column = target_column

    def load_for_deft(self) -> dict[str, Any]:
        df = pd.read_parquet(self.path)          # already the training split
        return {
            "X_train": df.drop(columns=[self.target_column]),
            "y_train": df[[self.target_column]].astype(int),
            "tree_filename": "tree_my_dataset.dill",
            "prompt_context": {},                # keys interpolated into the prompts
            "sequence_column": self.sequence_column,   # None if there is no sequence
        }
```

Then add `src/deft/conf/dataset/my_dataset.yaml` with a `plugin._target_` pointing at the class, and
run with `dataset=my_dataset`. No manual registration is needed.

The contract is **narrower than upstream's**: `load_for_baseline` and `load_for_dl` are gone
along with the baselines they served.

## What changed

This tree is reduced to the DEFT method itself — the baselines, deep-learning comparators,
paper-reproduction material and dataset-specific loaders are gone, and so is the dataset that
used to ship with the code. **There is no default dataset: a run must be told what to train on.**

What matters more is where behaviour differs, because a run here can produce a different tree
from the same configuration upstream.

### Behaviour that differs from upstream

| | Upstream | Here |
|---|---|---|
| A run where the model contributed nothing | completes and prints a normal-looking tree built from naive features | **fails** by default, at the first offending node — [Run integrity](#run-integrity-did-the-language-model-actually-contribute) |
| Generated code | `exec`'d after a substring scan for `"exec"`/`"eval"` | validated against an allowlist and run in a restricted namespace — [Running model-generated code](#running-model-generated-code) |
| A saved tree | contained the API key in plaintext | credentials redacted on serialization |
| Completions | the GPT-5 path requested 8 regardless of what was configured | the cap is a ceiling; larger asks are split and merged |
| Token usage | read and discarded | accounted always, with opt-in limits that stop the tree growing |
| Same `random_state`, same run | in-context examples and both stochastic selectors drew from global RNGs, so runs differed | all seeded |
| `remove_duplicates` | deduplicated by *score collision*, silently shrinking the pool | deduplicates on what the feature computes |
| An exception during fit | logged, exit 0 | re-raised |
| The dataset | polymerase, shipped with the code | none; `dataset.csv_path` is mandatory, columns must be legal identifiers, and the plugin does not split |
| The prompts | RNA Pol II pausing, hardcoded | task-neutral, and no dataset described at all; the caller supplies one through `--config-dir` |
| A candidate feature that could not be scored | still chosen, and split on at a threshold no search picked — a feature that raised took the run down with it | no finite score means no valid split, so the node becomes a leaf |
| Provider support | four hand-rolled engine classes, `openai==0.28.1` | one LiteLLM engine, nothing pinned exactly |
| A credential-free baseline | none — the CART-equivalent finder had no config | `feature_finder=cart`, which builds no engine and needs no key |
| Working directory | Hydra changes it | `hydra.job.chdir: false` |
| Re-running into a used output directory | silently interleaves and corrupts both records | refuses unless `allow_run_dir_reuse=true` |
| A generated candidate that fails on a node's rows | dropped, most often without a word | dropped with its reason in the log; with `repair_rounds`, sent back to the model with that reason first |
| Sequences of different lengths | refused | read by position in a declared `positional_window` |
| The target | binary, read as whole numbers, so 0.4 and 0.6 both became 0 | binary or continuous (`dataset.target_kind`), each checked as it is read; a continuous target grows a regression tree by variance reduction ([A continuous target](#a-continuous-target)) |
| What a column means | the prompts show its name, type and range | also its `column_descriptions` text, where given |

### Dependencies

Six, and **nothing is pinned to an exact version** — upstream had 29 exact pins, including
torch, transformers, lightning, peft, accelerate, captum, datasets and xgboost. `openai` now
arrives transitively through LiteLLM and nothing here touches it directly. There is no
`scikit-learn`: a DEFT run fits and serializes, it does not score.

## Citation

```bibtex
@misc{huynh2026interpretablednasequenceclassification,
      title={Interpretable DNA Sequence Classification via Dynamic Feature Generation in Decision Trees},
      author={Nicolas Huynh and Krzysztof Kacprzyk and Ryan Sheridan and David Bentley and Mihaela van der Schaar},
      year={2026},
      eprint={2604.12060},
      url={https://arxiv.org/abs/2604.12060},
}
```
