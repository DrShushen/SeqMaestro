"""The proposer, which asks a language model for features written as Python functions, with the
guard that vets their code before it runs.

A `Proposer` produces feature records and a `GeneratedFeaturiser` evaluates them. Keeping the two
apart lets one design cover both ways a language model takes part in a method:

```python
dataset = load_dataset(splits)      # the columns and the kind of target, from dataset.json
client = LiteLLMClient(config.language_model)       # the run's effort, retries and wait
proposer = Proposer(client, context, settings, cache=ProposalCache(run_dir / "proposals"),
                    seed=seed, sequence_column=dataset.sequence_column,
                    target_column=dataset.target_column, target_kind=dataset.target_kind,
                    arm="ebm/llm_generated")
featuriser = GeneratedFeaturiser(proposer)          # takes the column from the proposer
```

- **The proposer arms** wrap the proposer in a `GeneratedFeaturiser`, which is an ordinary
  `Featuriser`, and hand it to `run_model` as a fixed arm hands over its featuriser
  (`seqmaestro.pipeline.stage2.ProposerArm`). `run_model` needs no change for them.
- **DEFT's generative finder** calls the language model inside its tree search, once per node,
  conditioned on that node. It uses no featuriser, because the tree is the model and its features
  come out of the fitted tree, and it needs nothing in this package.

Two properties are easy to undo by accident. The proposer is **label-aware**, so a permuted cell
gets features proposed from shuffled labels, which is what makes it a control. Proposals are also
**cached by everything that would change the reply, the arm included** (`Proposer.key_for`). A
resumed run takes the proposals it already has from the cache, and each arm gets a proposal of its
own, so a feature that two arms both find comes from two separate draws.

`guard.py` is a port of DEFT's `code_guard.py`, with its tests, and should stay easy to diff
against it.
"""

from __future__ import annotations

from seqmaestro.proposer.cache import (
    ProposalCache,
    digest_rows,
)
from seqmaestro.proposer.featuriser import GeneratedFeaturiser
from seqmaestro.proposer.iterative import IterativeGeneratedFeaturiser
from seqmaestro.proposer.prompt import (
    Example,
    parse_proposals,
    render_request,
    sample_examples,
)
from seqmaestro.proposer.proposer import (
    KIND_LLM_GENERATED,
    ProposalSettings,
    Proposer,
)

__all__ = [
    "KIND_LLM_GENERATED",
    "Example",
    "GeneratedFeaturiser",
    "IterativeGeneratedFeaturiser",
    "ProposalCache",
    "ProposalSettings",
    "Proposer",
    "digest_rows",
    "parse_proposals",
    "render_request",
    "sample_examples",
]
