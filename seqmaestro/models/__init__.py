"""The models that consume a feature matrix: sparse, additive, boosted and symbolic, for each kind of
target.

```python
from seqmaestro.models import make_model
model = make_model("ebm", target_kind="binary", seed=42)
```

All of them consume a named feature matrix and know nothing about sequences. Swapping the pool
underneath them is how the comparison isolates the feature source from the model that consumes it.

**One model family per name, one class per kind of target.** `ebm` and `xgboost` name the same
family for a binary target and a continuous one, so an arm keeps its name whichever kind its
dataset has. The sparse linear model has a name per kind, `l1_logistic` and `l1_linear`, because
logistic and least-squares regression are different models, while the other two families differ
only in the loss they boost. `pysr` names one equation found by PySR's symbolic search, for either
kind: its search runs in the PySR environment and everything else in Core
(`seqmaestro.models.symbolic`).

CART is not here. The comparison uses **DEFT's** CART, so that it shares the tree, the splitting
criterion, the threshold search, the stopping rules and the naive candidate pool with the DEFT arm
and is measured against the same yardstick. It therefore runs on the other side of the environment
boundary and reports through the same files as everything else.

`seqmaestro.arms` lists every model an arm can name, these four and the two fitted in their methods'
environments (`deft_or_cart` and `igsr`), with the pools each can be fitted on. It reads what it
says about these four from their classes.
"""

from __future__ import annotations

from typing import Any

from seqmaestro.models.additive import ExplainableBoosting, ExplainableBoostingRegression
from seqmaestro.models.base import Model
from seqmaestro.models.boosting import GradientBoosting, GradientBoostingRegression
from seqmaestro.models.sparse import L1Linear, L1Logistic
from seqmaestro.models.symbolic import SymbolicClassification, SymbolicRegression
from seqmaestro.target import BINARY, CONTINUOUS, TARGET_KINDS

#: Every model, by the kind of target it fits and then by the name a results table reports it under.
MODELS = {
    BINARY: {
        "l1_logistic": L1Logistic,
        "ebm": ExplainableBoosting,
        "xgboost": GradientBoosting,
        "pysr": SymbolicClassification,
    },
    CONTINUOUS: {
        "l1_linear": L1Linear,
        "ebm": ExplainableBoostingRegression,
        "xgboost": GradientBoostingRegression,
        "pysr": SymbolicRegression,
    },
}

#: Every model name, whatever kind it fits.
MODEL_NAMES = tuple(sorted({name for models in MODELS.values() for name in models}))

__all__ = ["MODELS", "MODEL_NAMES", "L1Linear", "L1Logistic", "environment", "interpretable",
           "make_model", "models_for", "one_core", "reproducible"]


def models_for(target_kind: str) -> tuple[str, ...]:
    """The model names that fit a kind of target."""
    if target_kind not in MODELS:
        raise ValueError(f"Unknown target kind {target_kind!r}; expected one of {TARGET_KINDS}.")
    return tuple(sorted(MODELS[target_kind]))


def interpretable(name: str) -> bool:
    """Whether a model family can be interpreted feature by feature, which does not depend on the
    kind of target."""
    found = {models[name].interpretable for models in MODELS.values() if name in models}
    if len(found) != 1:
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}.")
    return found.pop()


def reproducible(name: str) -> bool:
    """Whether refitting a model family on the same rows, weights and seed gives the same model,
    which does not depend on the kind of target."""
    found = {models[name].reproducible for models in MODELS.values() if name in models}
    if len(found) != 1:
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}.")
    return found.pop()


def environment(name: str) -> str | None:
    """The conda environment besides Core that fitting a model family runs code in (PySR's search),
    or None for a family fitted in Core alone."""
    found = {getattr(models[name], "environment", None) for models in MODELS.values() if name in models}
    if len(found) != 1:
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}.")
    return found.pop()


def one_core(name: str) -> dict[str, Any]:
    """The constructor options that keep a fit of a model family on one core, which are none for a
    family that fits on one anyway."""
    found = [getattr(models[name], "one_core", {}) for models in MODELS.values() if name in models]
    if not found or any(options != found[0] for options in found):
        raise ValueError(f"Unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}.")
    return dict(found[0])


def make_model(name: str, *, target_kind: str, **kwargs: Any) -> Model:
    """Build a model by the name a results table reports it under, for a kind of target.

    A continuous target has no classes to weight, so any `class_weight` but `None` is refused for
    one, instead of being dropped where the caller would not see it.
    """
    available = models_for(target_kind)
    if name not in available:
        raise ValueError(f"No model {name!r} fits a {target_kind} target; the models that do are "
                         f"{', '.join(available)}.")
    if target_kind == CONTINUOUS and kwargs.pop("class_weight", None) is not None:
        raise ValueError("A continuous target has no classes to weight; give no class_weight.")
    return MODELS[target_kind][name](**kwargs)
