"""An arm's candidates (`Pool`): every feature its selection could have chosen, over the rows of D.

**The candidates are what the arm's model was offered, not only what it kept.** This makes Tier 1
a statement about the whole selection and not only its last step. A positional arm is offered
every (position, symbol) indicator and keeps few of them, so a profile over those few would say
nothing about the step that discarded the rest. Holding every candidate also makes the
position-level summary possible, since grouping a position's indicators needs all of them.

| arm | candidates | evaluated by |
|---|---|---|
| a Core model | the whole vocabulary of the featuriser the source cell saved, every round's | that featuriser |
| `deft_or_cart`, CART | DEFT's naive candidate population, as its tree's root was offered it | DEFT, on its side |
| `deft_or_cart`, DEFT | the features its tree split on | DEFT, on its side |
| `igsr` | every term its search accepted, at every node | IGSR, on its side |

The tree is CART on a pool without `llm_generated_joint` and DEFT on one with it
(`seqmaestro.arms`). The DEFT row departs from this, and cannot be brought into line without
calling a language model. Its generative finder wrote its candidates node by node, conditioned on
the rows reaching each node, so they never existed as a set. Its profile is therefore over
candidates that are already the model's output, which makes its Tier 1 optimistic by construction,
and every record of the arm says so (`RECORDED_CAVEAT`).

No feature is re-implemented here: each column is computed, in float64, by the Stage 2 model's
featuriser, by DEFT's feature function, or by IGSR's evaluation of its terms.

Candidates computed in another environment are saved as `pool.npz` (`Pool.save`, `Pool.load`): the
names, kinds, parameters and descriptions, the values, where the candidates come from (`source`),
and the row identifiers, which are checked on reading to be every row handed over, in order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seqmaestro.features.base import Featuriser, Inputs
from seqmaestro.harness.runner import evaluate_columns
from seqmaestro.reliability.prefilter import position_group

#: Where an arm's candidates come from, as every record names it (`Pool.source`).
MODEL_INPUT = "model_input_pool"
NAIVE_CANDIDATES = "naive_candidates"
RECORDED_FEATURES = "recorded_features"
SEARCH_HISTORY = "search_history"

#: The caveat that candidates taken from the recorded features carry wherever they are reported.
RECORDED_CAVEAT = (
    "The pool is the features the tree split on: the generative finder wrote its candidates node "
    "by node and never held them as a set, so this pool is already the model's output and its "
    "Tier 1 is optimistic by construction.")


@dataclass
class Pool:
    """An arm's candidates (name, kind, parameters, description) and their values over D."""

    names: list[str]
    kinds: list[str]
    params: list[dict]
    descriptions: list[str]
    values: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("A pool names a candidate twice.")
        if self.values.shape[1] != len(self.names):
            raise ValueError(f"{self.values.shape[1]} columns for {len(self.names)} names.")

    @property
    def p(self) -> int:
        return len(self.names)

    def index_of(self, wanted: list[str]) -> np.ndarray:
        lookup = {name: i for i, name in enumerate(self.names)}
        missing = [name for name in wanted if name not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} feature(s) are not in the pool, first {missing[:5]}.")
        return np.array([lookup[name] for name in wanted], dtype=np.int64)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, names=np.array(self.names, dtype=str), kinds=np.array(self.kinds, dtype=str),
            params=np.array([json.dumps(p, sort_keys=True) for p in self.params], dtype=str),
            descriptions=np.array(self.descriptions, dtype=str), values=self.values,
            row_id=np.arange(self.values.shape[0]), source=np.array(self.source))

    @classmethod
    def load(cls, path: str | Path, n_rows: int) -> Pool:
        """A saved pool, refused unless it holds every one of the `n_rows` rows, in order."""
        with np.load(path) as saved:
            if not np.array_equal(saved["row_id"], np.arange(n_rows)):
                raise RuntimeError(f"{path}: the pool came back for other rows, or in another order.")
            return cls(names=[str(name) for name in saved["names"]],
                       kinds=[str(kind) for kind in saved["kinds"]],
                       params=[json.loads(str(params)) for params in saved["params"]],
                       descriptions=[str(text) for text in saved["descriptions"]],
                       values=np.asarray(saved["values"], dtype=float), source=str(saved["source"]))

    def position_groups(self) -> list[str]:
        return [position_group(kind, params) for kind, params in zip(self.kinds, self.params, strict=True)]


def from_featuriser(featuriser: Featuriser, inputs: Inputs) -> Pool:
    """One of our arms' candidates: its featuriser's whole vocabulary, evaluated on these rows."""
    records = list(featuriser.records)
    names = [record.name for record in records]
    return Pool(names=names, kinds=[record.kind for record in records],
                params=[dict(record.params or {}) for record in records],
                descriptions=[record.description or "" for record in records],
                values=evaluate_columns(featuriser, inputs, names), source=MODEL_INPUT)
