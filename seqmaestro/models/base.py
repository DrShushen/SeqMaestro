"""What a model has to provide to be comparable with the others.

A model fits one kind of target, predicts a score for each row, says which features it used and how
much, says which way each of them points, describes itself for the cell's `run.json`, and reports
on its fit to a proposer. Everything else (featurisation, thresholds, metrics, feature records,
where the files go) is the job of `run_model` (`seqmaestro.harness.runner`) and is identical for
every model, so the arms form one comparison instead of separate experiments.

To add a model, implement this protocol: `run_model`, the record, the metrics and the stability
estimator need no change.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from seqmaestro.target import target_for


def fit_summary(model, X: pd.DataFrame, y) -> dict[str, Any]:
    """The part of `Model.feedback` every model shares: how it scored and which features it relied
    on.

    `unused` is as informative as `most_used` and easier to overlook: the next round should not
    spend a proposal restating a feature the model could not use.
    """
    scored = target_for(model.target_kind).feedback_metrics(np.asarray(y), np.asarray(model.predict(X)))
    importances = model.importances()
    ranked = sorted(importances.items(), key=lambda item: -abs(item[1]))
    return {
        # The headline metric of the target's kind and its companion: AUROC and AUPRC for a binary
        # target, R-squared and Spearman's rho for a continuous one.
        **{name: round(float(value), 4) for name, value in scored.items()},
        "n_features_offered": int(X.shape[1]),
        "n_features_used": len(importances),
        # The scale differs by model, so it is named.
        "importance_kind": model.importance_kind,
        "most_used": [{"feature": name, "importance": round(float(value), 5)}
                      for name, value in ranked[:15]],
        "unused": [name for name in X.columns if name not in importances][:15],
    }


def fitted_values(X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """`X` as one array of floats, after its columns are checked against `columns` (the columns a
    model was fitted on, in their order).

    A model predicts from this array instead of handing the frame to its library. scikit-learn,
    XGBoost and EBM each check a frame's columns and convert it to an array one column at a time,
    on every call, and on a pool of thousands of columns that takes far longer than the prediction.
    The check made here is the one the libraries make: the same names, in the same order. A frame
    built over one array of floats gives its values back without a copy.
    """
    given, columns = list(X.columns), list(columns)
    if given != columns:
        present, fitted = set(given), set(columns)
        missing = [name for name in columns if name not in present]
        unknown = [name for name in given if name not in fitted]
        if missing or unknown or len(given) != len(columns):
            raise ValueError(f"The model was fitted on {len(columns)} columns and is given "
                             f"{len(given)}: missing {missing[:5]}, not fitted on {unknown[:5]}.")
        position = next(i for i, (a, b) in enumerate(zip(given, columns, strict=True)) if a != b)
        raise ValueError(f"The model was fitted on these columns in another order: column "
                         f"{position} is {given[position]!r}, and was {columns[position]!r}.")
    return X.to_numpy(dtype=float)


def filled_rows(model, X: pd.DataFrame) -> int | None:
    """How many rows of `X` get the model's constant prediction because what it fitted is undefined
    there, or None for a model that is defined everywhere (one without `filled`)."""
    filled = getattr(model, "filled", None)
    return None if filled is None else int(np.count_nonzero(filled(X)))


#: The key of `metrics.json` that holds, per split, how many of its rows took the model's constant
#: prediction. Present only for a model that can be undefined on a row.
FILLED_ROWS = "filled_rows"


class Model(Protocol):
    """A model of one kind of target over a named feature matrix."""

    #: Short label in the cell's `run.json`. An arm names its model by this label too, so it also
    #: appears in the results path. A binary and a continuous model can share one, because an arm's
    #: kind of target is its dataset's.
    name: str
    #: The kind of target it fits (`seqmaestro.target`).
    target_kind: str
    #: What `importances()` returns, in the vocabulary of `seqmaestro.harness.features`. It is
    #: never comparable across models, so the harness ranks the importances before comparing them.
    importance_kind: str
    #: What `directions()` derives its words from, in the vocabulary of
    #: `seqmaestro.harness.direction`. The underlying quantities are no more comparable across
    #: models than the importances are, whereas the words they reduce to are, so the basis is named
    #: and a word is reported.
    direction_kind: str
    #: Whether the fitted model can be interpreted feature by feature (a coefficient, a shape
    #: function) and not only through its predictions. The reliability panel profiles only models
    #: that can, because a profile of a black box's features would describe those columns and say
    #: nothing about the model.
    interpretable: bool
    #: Whether refitting on the same rows, weights and seed returns the same model. The reliability
    #: panel reruns the reduction to check that its inputs are the reduction's, which only a
    #: reproducible model can pass. PySR, whose search is multithreaded, is the only model that is
    #: not reproducible.
    reproducible: bool

    # Three optional members, which only some models have:
    #
    # `environment`, a class attribute naming the conda environment a fit runs code in besides
    # Core's (PySR's search). A stage that fits such a model records that environment with the
    # run, as it records Core's.
    #
    # `one_core`, a class attribute holding the constructor options that keep a fit on one core,
    # for a model that would otherwise spread over several. Such an option changes how a fit runs
    # and never which model it fits. Code that already runs one fit per core passes it
    # (`seqmaestro.models.one_core`).
    #
    # `filled(X) -> np.ndarray`, a boolean mask of the rows of `X` whose prediction is the model's
    # constant, because what it fitted is undefined there (an equation dividing by zero). The
    # harness records how many rows of each split it scores were filled (`filled_rows`).

    def fit(self, X: pd.DataFrame, y, sample_weight: np.ndarray | None = None) -> Model:
        """Fit on a named feature matrix.

        `sample_weight` is one weight per row, and when given it **replaces** whatever the model's
        `class_weight` would have computed, without being multiplied with it. A caller supplying
        per-row weights has stated what it wants, and a design whose weights already balance the
        classes (a matched design expressed as weights instead of subsampling) would be corrected
        twice if the two were combined.
        """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """The model's score for each row: the probability of the positive class for a binary
        target, the predicted value for a continuous one.

        A binary model's score must be a probability and not a decision function, because the
        metrics refuse an unbounded margin instead of reporting a Brier score that looks valid and
        is not.
        """

    def importances(self) -> dict[str, float]:
        """Feature name to importance, **for the features the model used** and no others.

        Omitting the rest makes the result the set of features the model found. A model that
        assigns every input a non-zero weight has selected nothing, and says so by returning
        everything.
        """

    def directions(self, X: pd.DataFrame) -> dict[str, str]:
        """Feature name to which class the model uses it to argue for, or which way it moves a
        continuous target, for the features it used.

        Each is one of the words in `seqmaestro.harness.direction`, derived from whatever this model
        offers for the purpose: a coefficient's sign, a shape function's trend, or an attribution's
        dependence on the feature. A model with no basis for a direction returns fewer keys, or
        none. A feature the model uses in both directions is `non_monotone`, which is an answer and
        not a missing value.

        **The frame is given explicitly**, for the same reason as in `feedback`. Some models need
        rows to answer, and a model should never use a frame it kept from fitting. The caller
        decides on which rows a direction is measured, and the harness measures it on the reference
        split, so that every arm's answer is measured on the same data.

        **The labels are not given.** What a model does with a feature is a property of the model,
        whereas what the feature does in the data is a separate question with a separate answer
        (`marginal_effect`), and the harness records both.
        """

    def describe(self) -> dict[str, Any]: ...

    def feedback(self, X: pd.DataFrame, y) -> dict[str, Any]:
        """What this model can tell a proposer about its fit, for an arm that proposes in rounds.

        **The frame to report on is given explicitly**, so the model never uses one it kept. Which
        rows a model may be judged on is the caller's decision and belongs at the call site, where
        it can be read.

        `fit_summary` is what every model can say. A model extends it with what only it knows: a
        cover table beside the gain, how far each shape function swings, the sign of a coefficient.
        The proposer treats the result as data, so a model may add a key without any change to the
        prompt.
        """
        ...
