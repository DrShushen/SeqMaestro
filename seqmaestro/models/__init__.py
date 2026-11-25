"""The models that consume a feature matrix: sparse, additive, boosted and symbolic, for each kind of
target.

```python
from seqmaestro.models import make_model
model = make_model("ebm", target_kind="binary", seed=42)
```

All of them consume a named feature matrix and know nothing about sequences. Changing the pool
under a model is how the comparison separates the effect of the feature source from that of the
model.

**One model family per name, one class per kind of target.** `ebm` and `xgboost` name the same
family for a binary target and a continuous one, so an arm keeps its name whichever kind its
dataset has. The sparse linear model has a name per kind, `l1_logistic` and `l1_linear`, because
logistic and least-squares regression are different models, whereas the other two families differ
between the kinds only in the loss they boost. `pysr` names a model that is one equation found by
PySR's symbolic search, for either kind: its search runs in the PySR environment and everything
else in Core (`seqmaestro.models.symbolic`).

CART is absent from this package, because the comparison uses **DEFT's** CART, so that it shares
the tree, the splitting criterion, the threshold search, the stopping rules and the naive candidate
pool with the DEFT arm, and the two are compared on equal terms. It therefore runs in DEFT's conda
environment and reports through the same files as every other arm.

`seqmaestro.arms` lists every model an arm can name (these four, and the two fitted in their
methods' environments, `deft_or_cart` and `igsr`), with the pools each can be fitted on. It takes
what it states about these four from their classes.
"""

from __future__ import annotations

from collections.abc import Mapping
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

#: Every model name, whatever kind of target it fits.
MODEL_NAMES = tuple(sorted({name for models in MODELS.values() for name in models}))

__all__ = ["MODELS", "MODEL_NAMES", "L1Linear", "L1Logistic", "effective_settings", "environment",
           "given_settings", "interpretable", "make_model", "models_for", "one_core", "reproducible",
           "settings"]


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


def settings(name: str, target_kind: str) -> dict[str, type]:
    """The constructor options a run's config may change for a model, each with the type it takes,
    as the model's class lists them (`settings`). Each is a positive number. The model's other
    options stay fixed, so that an arm's model is the same in every run that leaves these alone.
    """
    available = models_for(target_kind)
    if name not in available:
        raise ValueError(f"No model {name!r} fits a {target_kind} target; the models that do are "
                         f"{', '.join(available)}.")
    return dict(getattr(MODELS[target_kind][name], "settings", {}))


def given_settings(description: Mapping[str, Any], target_kind: str) -> dict[str, Any]:
    """The settings a fitted model was built with, from its description (a cell's `model_settings`).

    A model records each setting it always holds, and each option it passes to its library only
    where that option was given, so the settings a description records are the ones to rebuild the
    model with.
    """
    return {key: description[key] for key in settings(description["model"], target_kind)
            if key in description}


def effective_settings(name: str, target_kind: str, given: Mapping[str, Any]) -> dict[str, Any]:
    """The value of each setting of a model built with `given`. A setting the model passes to its
    library only where given takes the library's own value otherwise (`library_defaults`). This is
    how a user sees the defaults before choosing to change one."""
    model = make_model(name, target_kind=target_kind, **given)
    described = model.describe()
    library = model.library_defaults() if hasattr(model, "library_defaults") else {}
    return {key: described[key] if key in described else library[key]
            for key in settings(name, target_kind)}


def make_model(name: str, *, target_kind: str, **kwargs: Any) -> Model:
    """Build a model by the name a results table reports it under, for a kind of target.

    A continuous target has no classes to weight, so any `class_weight` other than `None` is
    refused for it, instead of being dropped where the caller would not see it.
    """
    available = models_for(target_kind)
    if name not in available:
        raise ValueError(f"No model {name!r} fits a {target_kind} target; the models that do are "
                         f"{', '.join(available)}.")
    if target_kind == CONTINUOUS and kwargs.pop("class_weight", None) is not None:
        raise ValueError("A continuous target has no classes to weight; give no class_weight.")
    return MODELS[target_kind][name](**kwargs)
