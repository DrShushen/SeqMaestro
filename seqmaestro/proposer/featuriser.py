"""Proposed code in, a named numeric matrix out.

A `GeneratedFeaturiser` is an ordinary `Featuriser`, so `run_model` drives it the way it drives the
*k*-mer and positional featurisers, with the same subsample, threshold selection, metrics and
output files. A proposer arm therefore differs from a fixed one in the featuriser it builds
(`seqmaestro.pipeline.stage2.ProposerArm`), and `run_model` holds no logic for it.

**The proposer runs at `fit` and only at `fit`.** `run_model` fits on the seed's subsample, so the
model is shown only those rows and their labels. `transform` on validation and test evaluates what
was already proposed and makes no call. This is the ordinary split discipline. It also means the
proposer arms vary between seeds through the same subsample perturbation as every other arm, and
not through sampling temperature, which is what makes their stability comparable with the other
arms in a row.

**At `fit`, a feature that cannot be used is dropped** (`seqmaestro.proposer.checks`): one that
raises, does not finish within `max_seconds`, returns other than a number per row, is constant, or
is undefined (NaN or infinite) on more than `max_undefined_share` of the rows. Up to that share it
is kept, its undefined rows are filled with its fit-time median, and the share is recorded in the
record's `extra["undefined_share"]`. Where a proposer wrote the features, it has already sent the
failures back to the model in its repair round, so this is the second guard.

**A feature that fails at `transform` cannot be dropped**, because the model is already fitted on
it. It is filled with its fit-time median and the failure is counted, as is every undefined value
filled. Dropping a failing feature at `fit` is fine, because those rows are the training rows, but
dropping one at `transform` would be selecting features on test. At `transform` a feature is given
`max_seconds` scaled up by the rows' count against the rows it was fitted on (`scaled_limit`), so
one that finished at `fit` fails on time only where some rows cost it far more than those did.

**Generated code reads the covariates the featuriser was fitted with.** It is handed the rows as one
frame, the sequences under the column the prompt named and each covariate under its own name
(`Inputs.frame`). The covariates present at `fit` are recorded, and `transform` selects those by
name, so rows lacking one are refused instead of evaluated against a different frame.

**Values are rounded to `SIGNIFICANT_DIGITS` significant digits** as they are computed, at `fit`
and at `transform` alike. Generated code can compute one value two ways that differ in the last
bits. After rounding, the two are one value to every model, rank and count downstream, and a
feature that is constant apart from that noise is dropped as constant. `significant_digits=None`
keeps the raw values.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from seqmaestro.features.base import PROVENANCE_SELECTED, FeatureRecord, Featuriser, Inputs
from seqmaestro.proposer.checks import (
    FAILURES,
    SIGNIFICANT_DIGITS,
    TimeLimitUnavailable,
    check,
    run,
    scaled_limit,
)
from seqmaestro.proposer.guard import CodeGuardPolicy, UnsafeCodeError, compile_feature_function


class NoFeatureSurvived(RuntimeError):
    """Nothing of a proposal can be used on the rows: every feature failed to compile, raised, ran
    out of time, gave the wrong shape, was undefined too often or was constant. The `fit_report` of
    the featuriser that raised it counts each."""


class GeneratedFeaturiser(Featuriser):
    """Features a language model proposed, compiled and evaluated behind the guard.

    Takes a proposer: anything with a `propose(inputs, y)` method returning records whose
    `source` is the code, and a `sequence_column` attribute naming the column that code indexes.
    `from_records` builds one from existing records, for a re-scoring or a test, with no model and
    no network call. There `sequence_column` must be supplied, because records carry code without
    the frame it was written against.
    """

    def __init__(
        self,
        proposer=None,
        records: list[FeatureRecord] | None = None,
        sequence_column: str | None = None,
        policy: CodeGuardPolicy | None = None,
        provenance: str = PROVENANCE_SELECTED,
        significant_digits: int | None = SIGNIFICANT_DIGITS,
        max_undefined_share: float | None = None,
        max_seconds: float | None = None,
    ):
        super().__init__()
        if (proposer is None) == (records is None):
            raise ValueError("Give exactly one of `proposer` or `records`.")
        # Taken from the proposer when there is one, so it is not passed twice. The generated code
        # indexes this column by name because the prompt told it to, so this class reads the name
        # from the same place the prompt did.
        if sequence_column is None:
            sequence_column = getattr(proposer, "sequence_column", None)
        if sequence_column is None:
            raise ValueError(
                "`sequence_column` is required: the generated code indexes a column by that name "
                "and this class has no dataset to guess it from. It is taken from the proposer "
                "when there is one and the proposer declares it."
            )
        self.proposer = proposer
        self.given_records = records
        self.sequence_column = sequence_column
        self.policy = policy or CodeGuardPolicy()
        # The provenance these features have for the arm that reports them. A one-round proposal
        # is `selected`: the arm chose from it, and any other arm handed the same proposal could
        # have chosen the same. The iterative source builds one of these per round and stamps its
        # rounds `generated`.
        self.provenance = provenance
        if significant_digits is not None and (isinstance(significant_digits, bool)
                                               or not isinstance(significant_digits, int)
                                               or significant_digits < 1):
            raise ValueError(f"significant_digits must be a positive integer or None, "
                             f"not {significant_digits!r}.")
        self.significant_digits = significant_digits
        # Taken from the proposer's settings when there is a proposer, so a run states it once.
        # Records alone (a re-scoring, an analysis) drop only a feature undefined on every row.
        if max_undefined_share is None:
            max_undefined_share = getattr(getattr(proposer, "settings", None),
                                          "max_undefined_share", 1.0)
        if not 0 <= max_undefined_share <= 1:
            raise ValueError(f"max_undefined_share must be between 0 and 1, not {max_undefined_share!r}.")
        self.max_undefined_share = float(max_undefined_share)
        # Taken from the proposer's settings in the same way. Records alone run with no time limit
        # unless one is given, since the proposer that would have stated it is absent.
        if max_seconds is None:
            max_seconds = getattr(getattr(proposer, "settings", None), "max_seconds", None)
        if max_seconds is not None and not max_seconds > 0:
            raise ValueError(f"max_seconds must be above 0, or None for no limit, not {max_seconds!r}.")
        self.max_seconds = None if max_seconds is None else float(max_seconds)
        #: How many rows the features were checked on at `fit`, which scales the limit at transform.
        self._fitted_rows: int | None = None
        #: The covariates the generated code may read: those the rows carried at `fit`.
        self.covariate_columns: tuple[str, ...] = ()

        self._functions: list = []
        self._fill_values: np.ndarray = np.zeros(0)
        self.fit_report: dict[str, Any] = {}
        self.transform_failures: dict[str, int] = {}
        self.transform_undefined: dict[str, int] = {}

    @classmethod
    def from_records(cls, records: list[FeatureRecord], **kwargs) -> GeneratedFeaturiser:
        return cls(records=records, **kwargs)

    # ------------------------------------------------------------------ saving and restoring

    def __getstate__(self) -> dict[str, Any]:
        """Everything except the compiled functions and the proposer.

        A compiled function has no importable name, so pickle cannot reach it. Each kept record
        carries the source it was compiled from, so `__setstate__` compiles the same code again and
        restores the same columns. Recompiling is also the only route that needs no data.
        Rebuilding through `fit` would re-run the validation and could drop a feature that happens
        to be constant on whatever rows were at hand, leaving a model fitted on columns the
        featuriser no longer produces.

        The proposer is dropped because it holds a language-model client and is used only while
        fitting. A saved arm should not be able to issue API calls.
        """
        state = self.__dict__.copy()
        state["_functions"] = None
        state["proposer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self._functions is None:
            self._functions = [
                compile_feature_function(record.source, policy=self.policy)
                for record in self.records
            ]

    # ------------------------------------------------------------------ Featuriser contract

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        proposed = (
            self.given_records if self.proposer is None else self.proposer.propose(inputs, y)
        )
        self.covariate_columns = inputs.covariate_columns
        frame = self._frame(inputs)

        kept: list[FeatureRecord] = []
        functions, fills = [], []
        dropped = {"uncompilable": 0, **dict.fromkeys(FAILURES, 0)}

        for record in proposed:
            if not record.source:
                dropped["uncompilable"] += 1
                continue
            try:
                function = compile_feature_function(record.source, policy=self.policy)
            except (UnsafeCodeError, ValueError):
                # The proposer already compiled these, so reaching here means the records came from
                # elsewhere: a cache written under a wider policy, or another method's file.
                dropped["uncompilable"] += 1
                continue

            # A constant feature cannot help any model, and the stability estimator can match it to
            # nothing, which is why the estimator reports constants as a diagnostic.
            checked = check(function, frame, max_undefined_share=self.max_undefined_share,
                            significant_digits=self.significant_digits,
                            max_seconds=self.max_seconds)
            if not checked.usable:
                dropped[checked.failure] += 1
                continue

            extra = dict(record.extra)
            if checked.undefined_share:
                extra["undefined_share"] = round(checked.undefined_share, 4)
            kept.append(replace(record, provenance=self.provenance, extra=extra))
            functions.append(function)
            fills.append(float(np.median(checked.values[np.isfinite(checked.values)])))

        # Stated before a proposal with nothing left is refused, so that a caller that goes on
        # without it can still count what it lost.
        self.fit_report = {"n_proposed": len(proposed), "n_kept": len(kept), "dropped": dropped}
        if not kept:
            raise NoFeatureSurvived(
                f"No proposed feature survived fitting: {len(proposed)} proposed, dropped "
                f"{dropped}. An arm with no features cannot be scored; check the proposal stats "
                f"and the guard policy before re-running."
            )

        self._functions = functions
        self._fill_values = np.asarray(fills, dtype=float)
        self._fitted_rows = len(frame)
        self.transform_failures, self.transform_undefined = {}, {}
        return kept

    def _transform(self, inputs: Inputs) -> np.ndarray:
        frame = self._frame(inputs)
        columns = np.empty((len(frame), len(self._functions)), dtype=float)
        limit = scaled_limit(self.max_seconds, self._fitted_rows, len(frame))

        for position, (record, function) in enumerate(zip(self.records, self._functions, strict=True)):
            fill = self._fill_values[position]
            try:
                values = run(function, frame, self.significant_digits, limit)
            except TimeLimitUnavailable:
                raise
            except Exception:
                # Cannot be dropped, because the model is already fitted on this column. Filled
                # with the fit-time median, a training statistic, and counted.
                self.transform_failures[record.name] = self.transform_failures.get(record.name, 0) + 1
                values = np.full(len(frame), fill)

            non_finite = ~np.isfinite(values)
            if non_finite.any():
                self.transform_undefined[record.name] = (
                    self.transform_undefined.get(record.name, 0) + int(non_finite.sum()))
                values = np.where(non_finite, fill, values)
            columns[:, position] = values

        return columns

    def describe(self) -> dict[str, Any]:
        described = {
            "kind": "llm_generated",
            "sequence_column": self.sequence_column,
            "covariate_columns": list(self.covariate_columns),
            "guard": self.policy.describe(),
            "significant_digits": self.significant_digits,
            "max_undefined_share": self.max_undefined_share,
            "max_seconds": self.max_seconds,
            "n_features": len(self._records) if self.is_fitted else None,
            "fit_report": self.fit_report,
            # Normally empty. A name here is a feature that failed in a `transform` after it was
            # fitted, so its values on those rows are imputed, which belongs in the record and not
            # only in a log.
            "transform_failures": dict(self.transform_failures),
            # Undefined values filled at transform, by feature. Expected for a feature kept with
            # some undefined rows at fit.
            "transform_undefined": dict(self.transform_undefined),
        }
        if self.proposer is not None:
            described["proposer"] = self.proposer.describe()
        return described

    # ---------------------------------------------------------------------------- internals

    def _frame(self, inputs: Inputs) -> pd.DataFrame:
        """The rows as the generated code expects them: one DataFrame argument named `X`, as the
        guard's acceptance tests specify and the prompt demonstrates, holding the sequences under
        the column the code indexes and the covariates it was fitted with."""
        return inputs.select(self.covariate_columns).frame(self.sequence_column)
