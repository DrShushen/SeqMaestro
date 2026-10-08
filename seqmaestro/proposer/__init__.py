"""The shared language-model proposer, and the guard that lets its code be run.

A `Proposer` produces feature records and a `GeneratedFeaturiser` evaluates them. Keeping the two
apart lets one design cover both ways a language model enters a method:

```python
dataset = load_dataset(splits)      # the columns and the kind of target, from dataset.json
proposer = Proposer(LiteLLMClient(), context, settings, cache=ProposalCache(run_dir / "proposals"),
                    seed=seed, sequence_column=dataset.sequence_column,
                    target_column=dataset.target_column, target_kind=dataset.target_kind,
                    arm="ebm/llm_generated")
featuriser = GeneratedFeaturiser(proposer)          # takes the column from the proposer
```

- **The proposer arms** wrap the proposer in a `GeneratedFeaturiser`, which is an ordinary
  `Featuriser`, and hand it to `run_model` as a fixed arm hands over its featuriser
  (`seqmaestro.pipeline.stage2.ProposerArm`). `run_model` needs no change for them.
- **DEFT's generative finder** calls the model inside its tree search, once per node with that
  node's own conditioning. It uses no featuriser: the tree is the model, and its features come out
  of the fitted tree. Nothing in this package is needed for it, which is why proposing is kept
  separate from evaluating.

Two properties are easy to undo by accident. The proposer is **label-aware**, so a permuted-label
arm gets features proposed from shuffled labels, which is what makes it a null control. And
proposals are **cached by everything that would change the reply, the arm included**
(`Proposer.key_for`). A resumed run takes the proposals it already has from the cache, and each
arm gets a proposal of its own, so a feature that two arms both find comes from two separate draws.

`guard.py` is a port of DEFT's `code_guard.py`, with its tests, and should stay diffable against
it.
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
