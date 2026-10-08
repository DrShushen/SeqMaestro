"""The model a search arrives at, fitted once more and saved, and its predictions on new rows.

The search chooses a set of terms (`igsr.method.igsr.igsr`, by validation error) but keeps no fitted
weights: each iteration's fit is made, scored and dropped. `fit_final_model` fits the chosen terms
on the training rows with the search's own least-squares method, and `FittedModel` holds what is
needed to predict without the search: each term (an expression, or a function term's record with
its code and fill), the weights and the intercept. `to_dict` and `from_dict` save and restore it as
JSON.

On new rows a function term is compiled behind the code guard and run under its timeout, as in the
search (`SavedTerms`). Where it is undefined it takes its fill, and where it fails outright its whole
column takes the fill, since the weights were fitted with the term; `term_values` says which terms
failed. `SavedTerms` evaluates any saved term records the same way, such as every term a search
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from igsr.code_guard import CodeGuardPolicy, UnsafeCodeError
from igsr.config_typing import MainConfig
from igsr.method.function_terms import (
    EXPRESSION,
    FUNCTION,
    FunctionTermSettings,
    Proposal,
    compile_term,
    values_of,
)
from igsr.method.igsr_utils import DesignMatrix, PreparedDataset, _get_intercept, design_matrix, get_optimization_method


class SavedTerms:
    """Function terms saved as records (`FunctionTerm.record`), run on new rows as the search ran
    them: compiled behind the guard the search used, under its timeout, rounded, and filled."""

    def __init__(self, records: List[Dict[str, Any]], code_guard: Optional[Dict[str, Any]],
                 timeout_seconds: int) -> None:
        self.records = list(records)
        self.policy = CodeGuardPolicy.from_config(code_guard)
        self.timeout_seconds = timeout_seconds
        self._compiled: Dict[str, Any] = {}

    @property
    def names(self) -> List[str]:
        return [record["name"] for record in self.records]

    def values(self, rows: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, str]]:
        """Every term on `rows`, one column each, and the terms that failed on them, by name."""
        failed: Dict[str, str] = {}
        columns = []
        for record in self.records:
            try:
                values = values_of(self._function(record), rows, self.timeout_seconds)
            except UnsafeCodeError:
                raise  # a saved term the guard refuses is not run at all
            except Exception as error:  # model-written code can raise anything
                failed[record["name"]] = f"{type(error).__name__}: {error}"
                values = np.full(len(rows), np.nan)
            columns.append(np.where(np.isfinite(values), values, record["fill"]))
        return (np.column_stack(columns) if columns else np.empty((len(rows), 0))), failed

    def _function(self, record: Dict[str, Any]):
        if record["name"] not in self._compiled:
            proposal = Proposal(record["name"], record["description"], record["rationale"], record["code"])
            self._compiled[record["name"]] = compile_term(proposal, self.policy)
        return self._compiled[record["name"]]


@dataclass
class FittedModel:
    """A linear model over terms: `y = intercept + sum_k weight_k * term_k(rows)` for each target."""

    term_language: str
    #: The terms by name, in the order of the weights' rows. An expression is its own name.
    terms: List[str]
    target_columns: List[str]
    #: (n_terms, n_targets).
    weights: np.ndarray
    #: (n_targets,).
    intercept: np.ndarray
    #: The function form's term records (`FunctionTerm.record`), in the order of `terms`.
    functions: Optional[List[Dict[str, Any]]] = None
    #: What the function form's code may do, and how long one term may take on one set of rows.
    code_guard: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    _saved: Optional[SavedTerms] = field(default=None, repr=False, compare=False)

    def term_values(self, rows: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, str]]:
        """Every term on `rows`, one column each, and the terms that failed on them, by name."""
        if not self.terms:
            return np.empty((len(rows), 0)), {}
        if self.term_language == EXPRESSION:
            return DesignMatrix.from_terms(self.terms, {c: rows[c].to_numpy() for c in rows.columns}).phi, {}
        if self._saved is None:
            self._saved = SavedTerms(self.functions, self.code_guard, self.timeout_seconds)
        return self._saved.values(rows)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_targets) predictions."""
        values, _ = self.term_values(rows)
        return self.intercept[None, :] + values @ self.weights

    def to_dict(self) -> Dict[str, Any]:
        return {"term_language": self.term_language, "terms": list(self.terms),
                "target_columns": list(self.target_columns), "weights": self.weights.tolist(),
                "intercept": self.intercept.tolist(), "functions": self.functions,
                "code_guard": self.code_guard, "timeout_seconds": self.timeout_seconds}

    @classmethod
    def from_dict(cls, saved: Dict[str, Any]) -> "FittedModel":
        n_targets = len(saved["target_columns"])
        return cls(term_language=saved["term_language"], terms=list(saved["terms"]),
                   target_columns=list(saved["target_columns"]),
                   weights=np.asarray(saved["weights"], dtype=float).reshape(len(saved["terms"]), n_targets),
                   intercept=np.asarray(saved["intercept"], dtype=float), functions=saved["functions"],
                   code_guard=saved["code_guard"], timeout_seconds=saved["timeout_seconds"])


def guard_record(settings: FunctionTermSettings) -> Dict[str, Any]:
    """The guard a function term was run under, as a model saves it."""
    return {"enabled": settings.guard.enabled, "allowed_modules": list(settings.guard.allowed_modules),
            "extra_allowed_modules": list(settings.guard.extra_allowed_modules)}


def fit_final_model(cfg: MainConfig, terms: List[str], prepared_dataset: PreparedDataset) -> FittedModel:
    """The chosen terms fitted on the training split with the search's own least-squares method. A
    model with no terms predicts the training mean of each target."""
    y = np.column_stack(list(prepared_dataset.y.values()))
    if terms:
        dm = design_matrix(terms, prepared_dataset.data)
        method, method_kwargs = get_optimization_method(cfg)
        reg = method(**method_kwargs).fit(dm.phi, y)
        terms = dm.term_names  # an expression that fails to evaluate is left out, as in the search
        weights = np.asarray(reg.coef_, dtype=float).reshape(y.shape[1], len(terms)).T
        intercept = _get_intercept(reg)
        intercept = np.zeros(y.shape[1]) if intercept is None else intercept
    else:
        weights, intercept = np.zeros((0, y.shape[1])), y.mean(axis=0)
    targets = list(prepared_dataset.y)
    if cfg.experiment.term_language == EXPRESSION:
        return FittedModel(EXPRESSION, list(terms), targets, weights, intercept)
    registry = prepared_dataset.registry
    return FittedModel(FUNCTION, list(terms), targets, weights, intercept,
                       functions=registry.records(list(terms)), code_guard=guard_record(registry.settings),
                       timeout_seconds=registry.settings.timeout_seconds)
