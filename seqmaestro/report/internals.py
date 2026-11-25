"""The contents of a reduced model that a report draws, exported as plain data.

A shape function, a set of weights, a tree's splits and an equation's Shapley values exist only
inside the fitted model, so drawing any of them means loading the model. The figures are drawn from
JSON in another environment (`seqmaestro.report.figures`, under `SeqMaestro_LocalUtils`), so this
module loads the model and writes out what a figure needs. Nothing here draws.

**The model exported is the saved one, and a check confirms it.** A reduced cell saves its model
(`fitted.pkl`, or `model.pkl` for the least-squares model of an IGSR arm), the reveal scored the
test predictions from that file, and the report quotes that score. `check_reproduces` compares the
loaded model's predictions for the validation rows with the scores the cell recorded when it was
fitted, so a figure of a different model (other columns, other rows, another pickle) fails there.
DEFT's trees are loaded on DEFT's side (`experiments/adapters/deft_cart.py tree`), which checks
that each fitting row's leaf holds the prediction the tree makes for that row, and returns the
validation predictions to be compared with the cell's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from seqmaestro.target import BINARY, CONTINUOUS

#: How closely the loaded model's validation scores must match the ones its cell recorded. The same
#: model on the same rows gives the same scores, so the tolerance only absorbs differences in the
#: order of a floating-point sum.
TOLERANCE = 1e-10


class InternalsError(RuntimeError):
    """A drawn model that is not the model the report quotes."""


def check_reproduces(recorded: np.ndarray, predicted: np.ndarray, where: str) -> float:
    """Raise unless the loaded model's validation scores match the recorded ones within `TOLERANCE`.
    Returns the largest difference."""
    recorded, predicted = np.asarray(recorded, dtype=float), np.asarray(predicted, dtype=float)
    if recorded.shape != predicted.shape:
        raise InternalsError(f"{where}: {predicted.shape} scores for {recorded.shape} recorded.")
    worst = float(np.max(np.abs(recorded - predicted))) if len(recorded) else 0.0
    if worst > TOLERANCE:
        raise InternalsError(f"{where}: the loaded model's validation scores differ from the ones its "
                             f"cell recorded by up to {worst:.2e}. The figure would not be of the "
                             f"model the report quotes.")
    return worst


#: The `kind` exported for a model that used no feature. A report draws nothing for such a model.
NO_FEATURES = "no_features"


def no_features(model, predicted: np.ndarray, where: str) -> dict[str, Any] | None:
    """What is exported for a model that used no feature, or None for a model that used some.

    A reduced model can use no feature in two ways. Its arm's Stage 2 model used none, so the
    reduction had no column to keep and recorded `n_final` as 0. Or the reduction kept columns and
    the reduced model gave every one of them a weight of zero. Such a model predicts one value for
    every row, so it has no weights, curves, splits or equation to draw. The export says so and
    holds that value. A model that reports no feature and still predicts different values for
    different rows is refused.
    """
    if model.importances():
        return None
    predicted = np.asarray(predicted, dtype=float)
    if len(predicted) and float(np.ptp(predicted)) > TOLERANCE:
        raise InternalsError(f"{where}: the model reports that it used no feature, and its "
                             f"predictions differ between rows.")
    return {"kind": NO_FEATURES,
            "prediction": float(predicted[0]) if len(predicted) else None}


#: The models whose insides are one weight per feature (`l1_weights`), by the name of their class.
LINEAR = frozenset({"l1_logistic", "l1_linear", "least_squares"})

#: What a model's score is measured in, by its kind of target: the log-odds of the positive class, or
#: the target's units. A figure labels its axis with it.
SCALE = {BINARY: "log-odds", CONTINUOUS: "change in the target"}


def ebm_shapes(model, X_fit: np.ndarray, columns: list[str]) -> dict[str, Any]:
    """Every single-feature term of an additive model, strongest first: its bin boundaries, its
    score per bin in the model's scale (`SCALE`), how much it moves, and the fitted rows' values of
    its feature.

    `term_scores_` pads each curve with a leading `missing` bin and a trailing `unknown` bin. Both
    sit outside the range any row occupies, so they are dropped: if they were drawn, they would put
    a step at each end of the curve that no sequence can reach.
    """
    inner = model.estimator
    importances = dict(zip([str(n) for n in inner.term_names_],
                           [float(v) for v in inner.term_importances()], strict=True))
    terms = []
    for index, name in enumerate(inner.term_names_):
        features = inner.term_features_[index]
        if len(features) != 1:
            continue
        feature = features[0]
        scores = np.asarray(inner.term_scores_[index], dtype=float).ravel()[1:-1]
        cuts = np.asarray(inner.bins_[feature][0], dtype=float).ravel()
        low, high = (float(bound) for bound in inner.feature_bounds_[feature])
        boundaries = np.concatenate([[low], cuts, [high]])
        if len(boundaries) != len(scores) + 1:
            # A binary or near-constant column can have fewer cut points than bins. The bins are
            # then spaced evenly over the observed range, so the curve can still be drawn against
            # the feature's values.
            boundaries = np.linspace(low, high, len(scores) + 1)
        terms.append({"name": str(name), "importance": importances.get(str(name), 0.0),
                      "boundaries": boundaries.tolist(), "scores": scores.tolist(),
                      "fit_values": np.asarray(X_fit[:, columns.index(str(name))], dtype=float).tolist()})
    terms.sort(key=lambda term: -term["importance"])
    return {"kind": "ebm_shapes", "scale": SCALE[model.target_kind], "terms": terms,
            "intercept": float(np.ravel(inner.intercept_)[0])}


def pysr_equation(model) -> dict[str, Any]:
    """A symbolic model's written-out equation, the front it was chosen from, and, for each feature
    it names (strongest first), its values and Shapley values on the fitting rows the fit
    explained, in the model's scale (`SCALE`).

    The Shapley values are the ones the fit kept and ranked the features by, so the figure shows the
    numbers behind the reduction instead of computing them a second time.
    """
    names, values, shapley = model.shapley_sample()
    importances = model.importances()
    features = [{"name": name, "importance": float(importances.get(name, 0.0)),
                 "values": values[:, j].tolist(), "shapley": shapley[:, j].tolist()}
                for j, name in enumerate(names)]
    features.sort(key=lambda feature: -feature["importance"])
    fitted = model.describe()["fitted"]
    return {"kind": "pysr_equation", "scale": SCALE[model.target_kind],
            "equation": model.equation_text(), "latex_terms": model.equation_terms(latex=True),
            "features": features, "front": model.front(),
            "loss": "log-loss" if model.target_kind == BINARY else "squared error",
            "constant": fitted["constant"], "n_explained": len(values)}


def l1_weights(model, columns: list[str], marginal: dict[str, float | None]) -> dict[str, Any]:
    """A linear model's weights (`LINEAR`), each per standard deviation of its feature, in the
    model's scale (`SCALE`): log-odds for the logistic model, the target's units for the
    least-squares ones.

    A weight per standard deviation is comparable across features: a feature counted in hundreds
    and a yes/no indicator are each measured in units of their own spread. A sparse linear model is
    fitted on standardised columns, so its coefficients are such weights. The least-squares model
    of an IGSR arm is fitted on the columns as they are, and reports each coefficient multiplied by
    its column's standard deviation. Each feature's standard deviation on the fitting rows is
    exported beside its weight, so a reader can convert the weight to one per unit of the feature.
    """
    weights = model.importances()
    scale = model.column_scales()
    return {"kind": "l1_weights", "importance_kind": model.importance_kind,
            "scale": SCALE[model.target_kind],
            "features": [{"name": name, "weight": float(weights.get(name, 0.0)),
                          "sd_on_fitting_rows": float(scale[index]),
                          "marginal_effect": marginal.get(name)}
                         for index, name in enumerate(columns)],
            # One number for every model. The logistic model holds its intercept in an array, and
            # `intercept` takes its entry.
            "intercept": model.intercept()}


def read(model, columns: list[str], values: Callable[[], np.ndarray],
         marginal: Mapping[str, float | None]) -> dict[str, Any]:
    """The insides of a model loaded in this process, exported as its class requires.

    `columns` are the columns it was fitted on. `values` computes them on its fitting rows, and is
    called only for a class whose drawing shows them. `marginal` is each feature's marginal effect,
    which a linear model's weights are shown beside.
    """
    if model.name in LINEAR:
        return l1_weights(model, columns, marginal)
    if model.name == "ebm":
        return ebm_shapes(model, values(), columns)
    if model.name == "pysr":
        return pysr_equation(model)
    raise InternalsError(f"No reader for the insides of a {model.name} model.")
