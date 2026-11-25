"""Ask a language model for candidate features and keep the ones that are well-formed and safe.

The proposer arms test whether a language model proposes features that a fixed vocabulary cannot
express. This module turns a task and some labelled rows into `FeatureRecord`s holding executable
code. It runs that code only to check it, on the rows it proposed from. Computing the columns a
model is fitted and scored on is left to `GeneratedFeaturiser`, which evaluates the records on
whatever rows it is given.

**Every rejection is counted.** In the results table, a guard that silently drops most of what the
language model writes looks the same as a language model that had nothing to say. One that writes
illegal identifiers can lose most of its generations this way, with nothing in any output file to
show it. Each rejection is therefore counted in `stats`, which the cached proposal keeps in its
header and the stage writes into the cell's `token_usage.json`
(`seqmaestro.pipeline.stage2.ProposerArm`).

**A candidate that fails on the rows is sent back to the language model.** After each round, the
round's candidates are run on the rows the proposer was given (`seqmaestro.proposer.checks`). Those
that raise, run past the time limit, return the wrong shape, are undefined on too many rows or are
constant go back to the language model together in one call, each with its code and its failure,
and the corrected code is kept where it now works, marked `extra["repaired"]`. A code fault, or
code too slow for these sequences, is fixed without changing what the feature measures. A feature
constant or mostly undefined on these rows can only be corrected by measuring something a little
different, and then comes back under a name and description for what it now measures, with the
name it was proposed under in `extra["renamed_from"]`. `repair_rounds` is how many such calls a
round may make, and `stats` counts what came of them. A case the language model did not foresee,
such as a short sequence, then takes one more call instead of losing the feature.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from seqmaestro.features.base import IDENTIFIER, ORIGIN_SEQMAESTRO, FeatureRecord, Inputs, as_inputs
from seqmaestro.llm import TaskContext, reply_conditions
from seqmaestro.llm.columns import describe_columns, render_columns
from seqmaestro.proposer.cache import (
    ProposalCache,
    digest_feedback,
    digest_rows,
    proposal_key,
)
from seqmaestro.proposer.checks import SIGNIFICANT_DIGITS, Check, check
from seqmaestro.proposer.guard import CodeGuardPolicy, UnsafeCodeError, compile_feature_function
from seqmaestro.proposer.prompt import (
    SYSTEM_PROMPT,
    parse_proposals,
    render_repair_request,
    render_request,
    sample_examples,
    wording_digest,
)

#: The `kind` of any feature a language model wrote, whichever method's language model wrote it.
#: `origin` says which method that was. Keeping the two apart lets the share of generated features be counted
#: the same way for DEFT and for SeqMaestro (see `experiments/adapters/deft_bundle.py`).
KIND_LLM_GENERATED = "llm_generated"


@dataclass(frozen=True)
class ProposalSettings:
    """The proposal budget, set mainly by the number of rounds and the features asked for in each.

    A **round is one request for features** (one API call), plus up to `repair_rounds` further
    calls that send back the round's candidates that failed on the rows. The request asks for
    `n_per_round` features and lists what earlier rounds already produced, and the language model
    is told to propose different ones. DEFT's reflection is a different mechanism: it critiques and
    revises the same features, so rounds add new features, whereas reflection refines the ones it
    has. The number of calls differs too: rounds make `n_rounds` calls in total, plus at most
    `repair_rounds` each, whereas DEFT writes `n_samples_per_prompt` candidates in each of
    `1 + 2 x n_reflections` prompts at every node of its tree, and makes a call for each candidate's
    code (`seqmaestro.pipeline.calls` counts them), which is orders of magnitude more tokens.

    **Rounds make the set less redundant without making it larger.** A language model asked for the
    whole budget in one call returns the whole budget, so splitting the request adds no features,
    but it changes which features come back. Four ways of writing GC fraction over slightly
    different windows are one hypothesis, and a round shown what the earlier ones proposed is
    pushed away from them. This matters because near-duplicates chain clusters together in the
    stability estimator and crowd out distinct hypotheses in the judge's top slice.

    The extra tokens do not matter, because these arms cost orders of magnitude less than the
    per-node generation they are compared against.

    `n_per_round` x `n_rounds` should stay well above `DEFAULT_TOP_N` (`seqmaestro.harness.features`),
    or an arm's top slice would be its whole set instead of a selection.
    """

    n_rounds: int = 3
    n_per_round: int = 10
    n_examples: int = 20
    max_completion_tokens: int = 8000
    #: The largest number of characters of one sequence that an example row shows. A longer one is
    #: shown by its ends with its length stated (`prompt.shown_sequence`), because a whole
    #: transcript would fill the prompt.
    max_example_length: int = 300
    #: How many times a round's failed candidates are sent back to the language model, one call
    #: each time. 0 sends nothing back.
    repair_rounds: int = 1
    #: The largest share of rows on which a candidate may be undefined (NaN or infinite) and still
    #: be kept. Above it the candidate has failed (`seqmaestro.proposer.checks`).
    max_undefined_share: float = 0.2
    #: The longest a candidate may run on the rows it is checked on, in seconds. Past it the
    #: candidate has failed, and on more rows it is given proportionally longer
    #: (`seqmaestro.proposer.checks`).
    max_seconds: float = 60.0
    #: What the person running the analysis asks of the form of the features, shown word for word
    #: in every request, or None for no guidance. As a setting, it enters the cache key, so a
    #: proposal made under other guidance is never reused.
    guidance: str | None = None

    def __post_init__(self) -> None:
        if not self.max_seconds > 0:
            raise ValueError(f"max_seconds must be above 0, not {self.max_seconds!r}.")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProposalStats:
    """Where the language model's output went. The stats are written beside the features, never
    inferred from the count."""

    rounds: int = 0
    returned: int = 0
    accepted: int = 0
    rejected_unsafe: int = 0
    rejected_syntax: int = 0
    rejected_name: int = 0
    duplicates: int = 0
    #: Calls that sent failed candidates back, candidates they corrected, and candidates still
    #: failing when the repair rounds ran out. The last stay as proposed, for the featuriser to drop.
    repair_calls: int = 0
    repaired: int = 0
    unrepaired: int = 0
    cache_hit: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def __add__(self, other: ProposalStats) -> ProposalStats:
        """Two proposals counted together. `cache_hit` is true only where both came from the cache."""
        together = {name: getattr(self, name) + getattr(other, name)
                    for name in self.__dataclass_fields__ if name != "cache_hit"}
        return ProposalStats(**together, cache_hit=self.cache_hit and other.cache_hit)


class Proposer:
    """Turns (task, rows, labels) into feature records, over `n_rounds` requests and their repair
    calls. A proposal is cached under everything that would change it, the arm included (`key_for`).

    **It is label-aware.** Shown only sequences, it would propose the same features for a permuted
    cell as for a real one, and the permuted cells would have no features fitted to noise to serve
    as a control.

    **It is shown the covariates the rows hold**, each with the description `covariates` gives it
    (the dataset's, `Dataset.covariates`), so the code the language model writes may use them. Rows
    holding a covariate it has no description for are refused.
    """

    def __init__(
        self,
        client,
        context: TaskContext,
        settings: ProposalSettings | None = None,
        policy: CodeGuardPolicy | None = None,
        cache: ProposalCache | None = None,
        seed: int = 0,
        # Keyword-only and required. The prompt tells the language model which column its code
        # will be handed, which one holds the target and what kind of target it is. A wrong name
        # here does not raise an error: it makes a wrong prompt, and every feature the language
        # model writes indexes a column that is not there.
        *,
        sequence_column: str,
        target_column: str,
        target_kind: str,
        covariates: Mapping[str, str] | None = None,
        arm: str | None = None,
    ):
        self.client = client
        self.context = context
        self.settings = settings or ProposalSettings()
        self.policy = policy or CodeGuardPolicy()
        self.cache = cache or ProposalCache(None)
        self.seed = seed
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.target_kind = target_kind
        self.covariates = dict(covariates or {})
        #: The arm the proposals are made for, `model/pool`. It enters the cache key, so two arms
        #: never share a proposal, and each arm's written features are a separate draw. None for a
        #: proposer used outside an arm.
        self.arm = arm
        self.last_key: str | None = None
        #: Where the output of the last proposal went.
        self.stats = ProposalStats()
        #: The same for every proposal this proposer has made, counted together, which for an arm
        #: that proposes in rounds is the cell's whole count. None until the first proposal.
        self.total: ProposalStats | None = None

    # ------------------------------------------------------------------------------- public

    def propose(self, inputs, y=None, feedback: dict | None = None) -> list[FeatureRecord]:
        """Propose features from these rows (`Inputs`, or the sequences alone). Returns records with
        `source` set to the code.

        `feedback` is what a fitted model reported back, for an arm that proposes in rounds. It
        enters both the prompt and the cache key. `None` means a blind first round, and is what
        every arm on the one-round source passes.
        """
        inputs = as_inputs(inputs)
        key = self.key_for(inputs, y, feedback)
        # Kept so the run can record which entry an arm drew on. Two arms reporting one key would
        # not be independent evidence of anything. The co-discovery analysis finds such arms by this
        # key (`seqmaestro.pipeline.analyses.co_discovery.linked_groups`), so a new arm needs no
        # hand-maintained list to be counted correctly. A proposer given its arm never shares a key
        # with another arm.
        self.last_key = key

        # Held while the entry is looked up and made, so a process running beside this one and
        # asking for the same proposal waits for this answer instead of paying for another.
        with self.cache.held(key):
            cached = self.cache.get(key)
            if cached is not None:
                records, header = cached
                self.stats = ProposalStats(**{**header.get("proposal_stats", {}),
                                              "cache_hit": True})
            else:
                records = self._propose_uncached(inputs, y, feedback)
                self.cache.put(
                    key, records,
                    proposal_stats=self.stats.as_dict(),
                    # The blind context, matching the key: an entry records the conditions a reply
                    # was produced under, and a fact the language model was never shown is not one
                    # of them.
                    context=self.context.as_dict_blind(),
                    settings=self.settings.as_dict(),
                )
        self.total = self.stats if self.total is None else self.total + self.stats
        return records

    def key_for(self, inputs, y=None, feedback: dict | None = None) -> str:
        """The cache key: a digest of everything that would change the answer.

        **The key includes the arm and the feedback.** The arm is included so that two arms never
        share a proposal: two models compared on a written source are compared on separate
        proposals. The iterative source's blind first round makes the one-round source's request,
        under its arm's key. The feedback is included so that each informed round of an iterative
        arm makes a new request.

        **The context enters in its blind form**, because a proposer is never shown `judge_only`,
        so its replies cannot depend on it. Keying on the disclosing form would discard every
        cached proposal whenever the judge's half of a description was edited, and pay again for
        replies that could not have changed.

        **The columns block enters where the prompt has one**, so a covariate described differently
        makes a different request. Rows with no columns block are keyed without one.

        **The language model enters with the effort it is asked to reason at** (`reply_conditions`),
        and without the retries and the wait, which do not change a reply.
        """
        inputs = as_inputs(inputs)
        parts = {
            "context": self.context.as_dict_blind(),
            "settings": self.settings.as_dict(),
            "guard": self.policy.describe(),
            # The prompt wording, as distinct from the task it asks about. Editing the shared prompt
            # changes every reply, and without this the cache would keep serving the old ones.
            "wording": wording_digest(),
            # How a function's values are rounded before they are checked, which decides whether a
            # function counts as constant and so goes back to the language model for repair.
            "significant_digits": SIGNIFICANT_DIGITS,
            "language_model": reply_conditions(self.client.describe()),
            "seed": self.seed,
            "rows": digest_rows(inputs, y),
            "feedback": digest_feedback(feedback),
        }
        if self.arm is not None:
            parts["arm"] = self.arm
        block = render_columns(self.columns(inputs))
        if block is not None:
            parts["columns"] = block
        return proposal_key(**parts)

    def columns(self, inputs) -> dict[str, Any]:
        """What the language model is told about the columns of these rows (`describe_columns`)."""
        return describe_columns(inputs, self.sequence_column, self.covariates)

    def describe(self) -> dict[str, Any]:
        """For the cell's record (`run.json`), so a result names the conditions that produced it.

        The context is the blind one, because **each consumer records what it was told**, so the
        records of a blinded run show afterwards who was told what. A proposer arm's record shows
        the description without the withheld fact, whereas the judge's output includes it, because
        the judge was told it.
        """
        described = {
            "kind": "llm_proposer",
            # Which language model was called, and how.
            "language_model": self.client.describe(),
            "seed": self.seed,
            "settings": self.settings.as_dict(),
            "guard": self.policy.describe(),
            "context": self.context.as_dict_blind(),
            # The cache entry of the last proposal. Two arms reporting one key would have used one
            # proposal, so their agreement would corroborate nothing.
            "proposal_key": self.last_key,
        }
        if self.covariates:
            # What the language model was told each covariate measures.
            described["covariates"] = dict(self.covariates)
        return described

    def usage(self) -> dict:
        return self.client.usage()

    # ---------------------------------------------------------------------------- internals

    def _propose_uncached(self, inputs: Inputs, y,
                          feedback: dict | None = None) -> list[FeatureRecord]:
        self.stats = ProposalStats()
        examples = sample_examples(inputs, y, self.settings.n_examples, self.seed, self.target_kind)
        columns = self.columns(inputs)
        records: dict[str, FeatureRecord] = {}

        for round_index in range(self.settings.n_rounds):
            user = render_request(
                self.context,
                n_features=self.settings.n_per_round,
                examples=examples,
                already_proposed=list(records),
                sequence_column=self.sequence_column,
                target_column=self.target_column,
                target_kind=self.target_kind,
                feedback=feedback,
                columns=columns,
                max_example_length=self.settings.max_example_length,
                guidance=self.settings.guidance,
            )
            completion = self.client.complete(
                SYSTEM_PROMPT, user, max_completion_tokens=self.settings.max_completion_tokens
            )
            self.stats.rounds += 1

            if not completion.ok:
                self.stats.errors.append(f"round {round_index}: {completion.error}")
                continue

            proposals, error = parse_proposals(completion.text)
            if error:
                self.stats.errors.append(f"round {round_index}: {error}")
            self.stats.returned += len(proposals)

            fresh = []
            for proposal in proposals:
                record = self._accept(proposal, taken=records)
                if record is not None:
                    records[record.name] = record
                    fresh.append(record.name)
            self._repair(fresh, records, inputs, columns, round_index)

        self.stats.accepted = len(records)
        return list(records.values())

    def _check(self, code: str, frame) -> Check:
        """One candidate's code on the proposer's rows, as `GeneratedFeaturiser` will check it."""
        function = compile_feature_function(code, policy=self.policy)
        return check(function, frame, max_undefined_share=self.settings.max_undefined_share,
                     significant_digits=SIGNIFICANT_DIGITS, max_seconds=self.settings.max_seconds)

    def _repair(self, names: list[str], records: dict[str, FeatureRecord], inputs: Inputs,
                columns: dict[str, Any], round_index: int) -> None:
        """Run a round's candidates on the rows, and send those that fail back to the language
        model.

        Each call sends every candidate still failing, with the code last tried for it and how that
        code failed. The language model is asked to fix a code fault without changing what the
        feature measures, and to change a feature that cannot work on these rows as little as it
        can, naming and describing it for what it then measures (`prompt.render_repair_request`). A
        correction that works takes the place of the original, under its new name where it has
        one. A candidate still failing when the calls run out stays as proposed, so the featuriser
        drops it with its reason counted.
        """
        frame = inputs.frame(self.sequence_column)
        # Each failing candidate, under the name it was proposed with: the code last tried for it,
        # and how that code failed.
        failing: dict[str, tuple[str, Check]] = {}
        for name in names:
            checked = self._check(records[name].source, frame)
            if not checked.usable:
                failing[name] = (records[name].source, checked)
        for _ in range(self.settings.repair_rounds):
            if not failing:
                break
            user = render_repair_request(
                self.context, [(name, code, checked) for name, (code, checked) in failing.items()],
                sequence_column=self.sequence_column, columns=columns)
            completion = self.client.complete(
                SYSTEM_PROMPT, user, max_completion_tokens=self.settings.max_completion_tokens)
            self.stats.repair_calls += 1
            if not completion.ok:
                self.stats.errors.append(f"round {round_index} repair: {completion.error}")
                continue
            proposals, error = parse_proposals(completion.text)
            if error:
                self.stats.errors.append(f"round {round_index} repair: {error}")
            for proposal in proposals:
                # Matched on the name it says it replaces, or, where it names none, on its own name.
                target = proposal.replaces or proposal.name
                if target not in failing:
                    continue
                code, before = failing[target]
                try:
                    checked = self._check(proposal.code, frame)
                except (UnsafeCodeError, ValueError) as refused:
                    problem = f"was refused: {refused}"
                else:
                    if not checked.usable:
                        failing[target] = (proposal.code, checked)
                        continue
                    problem = self._naming_problem(proposal, target, records)
                if problem:
                    # The correction cannot be kept, so the code tried before it stands, and the
                    # next call is told both.
                    failing[target] = (code, Check(before.failure,
                                                   f"{before.detail}, and its correction {problem}"))
                    continue
                self._replace(records, target, proposal)
                del failing[target]
                self.stats.repaired += 1
        self.stats.unrepaired += len(failing)

    @staticmethod
    def _naming_problem(proposal, target: str, records: dict[str, FeatureRecord]) -> str | None:
        """Why a working correction's name cannot be kept, or None. A new name must be a legal
        identifier that no other feature has, and must come with a description of what the feature
        now measures."""
        if proposal.name == target:
            return None
        if not IDENTIFIER.match(proposal.name):
            return f"was named {proposal.name!r}, which is not a legal Python identifier"
        if proposal.name in records:
            return f"was named {proposal.name!r}, which another feature already has"
        if not proposal.description:
            return (f"was given a new name, {proposal.name!r}, with no description of what it "
                    f"now measures")
        return None

    @staticmethod
    def _replace(records: dict[str, FeatureRecord], target: str, proposal) -> None:
        """Put a working correction in place of the original. A renamed correction records the
        original's name in `extra["renamed_from"]`, and its description is the one the correction
        came with."""
        original = records[target]
        renamed = proposal.name != target
        corrected = replace(
            original, name=proposal.name, source=proposal.code,
            description=proposal.description or original.description,
            rationale=proposal.rationale or original.rationale,
            extra={**original.extra, "repaired": True, **({"renamed_from": target} if renamed else {})})
        # In place and in order, so a renamed feature keeps its original's position.
        entries = list(records.items())
        records.clear()
        records.update((corrected.name, corrected) if name == target else (name, record)
                       for name, record in entries)

    def _accept(self, proposal, taken: dict) -> FeatureRecord | None:
        """Validate one proposal by compiling it. Evaluation needs rows and happens elsewhere."""
        if not IDENTIFIER.match(proposal.name):
            # A language model can write a name such as `upstream_-15nt_GC`, and every downstream
            # use of that name then fails to parse. It is counted, so the loss is visible.
            self.stats.rejected_name += 1
            return None
        if proposal.name in taken:
            self.stats.duplicates += 1
            return None

        try:
            compile_feature_function(proposal.code, policy=self.policy)
        except UnsafeCodeError:
            self.stats.rejected_unsafe += 1
            return None
        except ValueError:
            # A bad identifier inside the code or an unbalanced bracket is a language model writing
            # badly, not an attack. It is counted apart from `rejected_unsafe`, so a run's
            # rejection counts show which of the two happened.
            self.stats.rejected_syntax += 1
            return None

        return FeatureRecord(
            name=proposal.name,
            description=proposal.description,
            kind=KIND_LLM_GENERATED,
            params={},
            source=proposal.code,
            rationale=proposal.rationale,
            origin=ORIGIN_SEQMAESTRO,
        )
