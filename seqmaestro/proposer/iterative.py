"""The iterative source: propose, fit, show the language model the fit, propose again.

The one-round source (`llm_generated`) has a language model make one proposal, from the task
description and a sample of labelled rows. This source (`llm_generated_iterative`) has it make
several, one per round, and from round two the language model is shown what a fit of the arm's
model made of the features proposed so far: which features it relied on, which it could not use,
and how well it scored. Comparing the two sources isolates the effect of that loop. A round here is
one proposal, which the proposer may make over several requests (`ProposalSettings.n_rounds`).

**Round one is blind, whatever the pool.** There is no fit to report on yet, so round one makes the
same request as the one-round source, which makes the two comparable. It is a proposal made for this
arm (`Proposer.arm`). The pool of an arm on this source holds it (`stage2.ProposerArm` builds the
pool from that proposal, with this proposer), and the loop then takes the pool's proposal as its
round one: it is neither requested nor checked on the rows a second time. At `rounds=1` this
featuriser reduces to its pool plus a single one-round proposal.

**Provenance.** The pool's features are `selected`: they were fixed before fitting, and any arm on
the same pool could have chosen the same features. What this class proposes is `generated`,
because those features exist only as a result of a fit that only this arm made.

**Deduplication is required.** A language model asked again repeats earlier ideas, under names the
arm already holds. `Featuriser.fit` refuses duplicate names, so without deduplication the arm would
not run at all. With it, `rounds=1` over a pool that holds the same proposal from another proposer
reproduces the one-round source, which is the check the design rests on.

**What became of the written features is counted for every round** (`round_report`) and in total
(`fit_report`): how many were proposed, how many the arm already held, how many were kept, and how
many were dropped for each reason. The columns of a fixed source in the pool are not counted.

**The fitted model decides what the language model is shown.** Feedback comes from
`Model.feedback`, so an arm using XGBoost sends gain and cover, and one using an additive model
sends how far each shape function swings. This class never needs to know the difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from seqmaestro.features.base import PROVENANCE_GENERATED, FeatureRecord, Featuriser, Inputs, as_inputs
from seqmaestro.proposer.checks import SIGNIFICANT_DIGITS
from seqmaestro.proposer.featuriser import GeneratedFeaturiser, NoFeatureSurvived
from seqmaestro.proposer.guard import CodeGuardPolicy


class IterativeGeneratedFeaturiser(Featuriser):
    """A pool, plus `rounds` of language-model proposals, each after the first informed by a fit.

    `model_factory` returns a fresh, unfitted model per round, of the class the arm reports under,
    so the features are shaped by the model that will use them.

    `validation` holds the rows (`Inputs`, or the sequences alone) and the labels each round's fit is
    reported on. It is passed in so that which rows a proposer may see is decided at the call site,
    where it can be read, and so this class never has to know what a split is.
    """

    def __init__(
        self,
        proposer,
        model_factory,
        validation: tuple[Inputs | pd.Series, np.ndarray],
        *,
        rounds: int,
        pool: Featuriser | None = None,
        sequence_column: str | None = None,
        policy: CodeGuardPolicy | None = None,
        significant_digits: int | None = SIGNIFICANT_DIGITS,
    ):
        super().__init__()
        if rounds < 1:
            raise ValueError(f"rounds must be at least 1; got {rounds}.")
        self.proposer = proposer
        self.model_factory = model_factory
        self.validation = validation
        self.pool = pool
        self.rounds = rounds
        self.sequence_column = sequence_column or getattr(proposer, "sequence_column", None)
        self.policy = policy or CodeGuardPolicy()
        #: Handed to every round's featuriser: see `GeneratedFeaturiser`.
        self.significant_digits = significant_digits
        #: Handed to every round's featuriser, from the proposer's settings, as `GeneratedFeaturiser`
        #: takes it.
        self.max_undefined_share = float(getattr(getattr(proposer, "settings", None),
                                                 "max_undefined_share", 1.0))
        #: Handed to every round's featuriser in the same way. None, where the proposer states no
        #: limit, sets none.
        self.max_seconds = getattr(getattr(proposer, "settings", None), "max_seconds", None)
        self._parts: list[Featuriser] = []
        #: The covariates the rows held at `fit`, which every round's generated code may use.
        self.covariate_columns: tuple[str, ...] = ()
        #: One entry per round: the features proposed, those the arm already held under the same
        #: name, those kept, and those dropped for each reason.
        self.round_report: list[dict[str, Any]] = []
        #: The rounds' counts added up, under the keys `GeneratedFeaturiser` reports, so a stage
        #: can summarise this arm without knowing which of the two it is holding. They count the
        #: written features alone: `n_proposed` is the sum of `n_already_present`, `n_kept` and
        #: every count of `dropped`.
        self.fit_report: dict[str, Any] = {}
        self.transform_failures: dict[str, int] = {}
        self.transform_undefined: dict[str, int] = {}

    # ------------------------------------------------------------------ saving and restoring

    def __getstate__(self) -> dict[str, Any]:
        """The fitted parts, without what was used to build them.

        `_transform` uses `_parts` and nothing else. The proposer, the model factory and the
        validation rows are used only while fitting, so none of them is saved. The factory is
        usually a closure, which cannot be pickled, and the validation rows are data that belongs to
        the run and not to the arm.
        """
        state = self.__dict__.copy()
        for fit_time_only in ("proposer", "model_factory", "validation"):
            state[fit_time_only] = None
        return state

    # ------------------------------------------------------------------ Featuriser contract

    def _fit(self, inputs: Inputs, y: np.ndarray | None = None,
             sample_weight: np.ndarray | None = None) -> list[FeatureRecord]:
        self._parts, self.round_report = [], []
        self.covariate_columns = inputs.covariate_columns
        if self.pool is not None:
            self.pool.fit(inputs, y, sample_weight)
            self._parts.append(self.pool)

        blind = self._blind_proposal()
        seen = {record.name for part in self._parts for record in part.records}
        feedback = None

        for index in range(self.rounds):
            if index == 0 and blind is not None:
                # The pool made this round's request and checked its features on these rows.
                counts = {"proposed": blind.fit_report["n_proposed"], "already_present": 0,
                          "kept": blind.fit_report["n_kept"],
                          "dropped": dict(blind.fit_report["dropped"])}
            else:
                proposed = self.proposer.propose(inputs, y, feedback=feedback)
                fresh = [record for record in proposed if record.name not in seen]
                kept, dropped = self._compile(fresh, inputs, y)
                if kept is not None:
                    self._parts.append(kept)
                    seen |= {record.name for record in kept.records}
                counts = {"proposed": len(proposed), "already_present": len(proposed) - len(fresh),
                          "kept": 0 if kept is None else len(kept.records), "dropped": dropped}
            self.round_report.append({"round": index + 1, "had_feedback": feedback is not None,
                                      **counts})
            # No fit after the last round: there is no later round for its feedback to inform.
            if index + 1 < self.rounds:
                feedback = self._feedback(inputs, y, sample_weight)

        dropped_total: dict[str, int] = {}
        for entry in self.round_report:
            for reason, count in entry["dropped"].items():
                dropped_total[reason] = dropped_total.get(reason, 0) + count
        self.fit_report = {
            "n_proposed": sum(entry["proposed"] for entry in self.round_report),
            "n_already_present": sum(entry["already_present"] for entry in self.round_report),
            "n_kept": sum(entry["kept"] for entry in self.round_report),
            "dropped": dropped_total,
            "rounds": len(self.round_report),
        }
        if not self._parts:
            raise RuntimeError(
                f"No features survived {self.rounds} round(s): nothing was proposed that compiled "
                f"and varied, and no pool was given. An arm with no features cannot be scored."
            )
        return [record for part in self._parts for record in part.records]

    def _transform(self, inputs: Inputs) -> np.ndarray:
        self.transform_failures, self.transform_undefined = {}, {}
        matrix = self._matrix(inputs).to_numpy(dtype=float)
        # A pool that holds more than the proposal is composed, and the proposal is one of its
        # parts.
        for part in self._parts:
            for leaf in getattr(part, "parts", [part]):
                self.transform_failures.update(getattr(leaf, "transform_failures", {}) or {})
                self.transform_undefined.update(getattr(leaf, "transform_undefined", {}) or {})
        return matrix

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "iterative_llm_generated",
            "rounds": self.rounds,
            "significant_digits": self.significant_digits,
            "sequence_column": self.sequence_column,
            "covariate_columns": list(self.covariate_columns),
            "pool": None if self.pool is None else self.pool.describe(),
            # Per round: what was proposed, what the arm already held, what was kept and what was
            # dropped. Look here when an arm's feature count is not what the round count would
            # suggest.
            "rounds_report": self.round_report,
            "proposer": self.proposer.describe(),
        }

    # ---------------------------------------------------------------------------- internals

    def _blind_proposal(self) -> GeneratedFeaturiser | None:
        """The part of the pool that holds this proposer's blind proposal, or None where the pool
        holds none. The pool's fit made the request round one would make and checked its features
        on the same rows, so the loop takes them as its round one."""
        for part in getattr(self.pool, "parts", [self.pool]):
            if isinstance(part, GeneratedFeaturiser) and part.proposer is self.proposer:
                return part
        return None

    def _compile(self, records: list[FeatureRecord], inputs: Inputs, y
                 ) -> tuple[Featuriser | None, dict[str, int]]:
        """One round's fresh proposals as a fitted featuriser, or None if none of them survived,
        and how many were dropped for each reason."""
        if not records:
            return None, {}
        part = GeneratedFeaturiser.from_records(
            records, sequence_column=self.sequence_column, policy=self.policy,
            provenance=PROVENANCE_GENERATED, significant_digits=self.significant_digits,
            max_undefined_share=self.max_undefined_share, max_seconds=self.max_seconds,
        )
        try:
            part.fit(inputs, y)
        except NoFeatureSurvived:
            # A round can come to nothing without the arm being broken, so the round is reported
            # and the arm goes on. Any other failure of the fit is the arm's, such as a time limit
            # that cannot be set.
            return None, part.fit_report["dropped"]
        return part, part.fit_report["dropped"]

    def _matrix(self, inputs: Inputs) -> pd.DataFrame:
        blocks = [part.transform(inputs).reset_index(drop=True) for part in self._parts]
        return pd.concat(blocks, axis=1)

    def _feedback(self, inputs: Inputs, y, sample_weight) -> dict[str, Any] | None:
        """Fit this round's model, with the weights the arm's model is fitted with, and ask it what
        it made of the features so far."""
        if not self._parts:
            return None
        model = self.model_factory()
        model.fit(self._matrix(inputs), np.asarray(y), sample_weight=sample_weight)
        validation_rows, validation_y = self.validation
        return model.feedback(self._matrix(as_inputs(validation_rows)), np.asarray(validation_y))
