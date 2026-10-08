import asyncio
import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import copy
from typing import Any

import numpy as np
import pandas as pd

from deft.features.helpers import GUIDANCE_FOR, optimize_features, screen_features
from deft.llm.prompt_builder import (
    construct_prompt_code_generation,
    construct_prompt_code_repair,
    construct_prompt_name_description,
    construct_prompt_reflection,
)
from deft.provenance import ORIGIN_LLM, ORIGIN_NAIVE
from deft.utils.feature import get_naive_features, list_to_str
from deft.utils.parsing import parse_repaired_feature, parsing_end2end

logger = logging.getLogger(__name__)


def _best_scoring_feature(sorted_features):
    """The best candidate, or None when not one of them scored.

    `optimize_features` leaves a candidate it could not score at `score = -inf`, `threshold = 0`
    -- that is its failure sentinel, set when the feature raised, or when the 90-second
    threshold-search timeout fired, or when no cut point satisfied `min_samples_leaf`. The
    candidate stays in the pool, so the best-scoring one can be a feature that never scored at
    all.

    Returning it would be wrong twice over. The `threshold = 0` is a sentinel no search ever
    chose, and the tree evaluates the function again to split on it -- so a feature that raises
    takes the whole run down at that point, after its tokens have been spent. Verified.

    No finite score means no valid split at this node, which is the same conclusion CART reaches
    when no cut point improves the criterion: the node becomes a leaf. The caller already
    handles None that way, and provenance records the node with `split_feature=None` so the
    integrity policy sees it.
    """
    for feature in sorted_features:
        score = getattr(feature, "score", None)
        if score is not None and np.isfinite(score):
            return feature

    if sorted_features:
        logger.error(
            "None of the %d candidate features at this node could be scored; every one is at "
            "the failure sentinel (score=-inf). The node becomes a leaf rather than splitting "
            "on a threshold no search chose.",
            len(sorted_features),
        )
    return None


class FeatureFinder(ABC):
    # Whether `run_experiment` has to build an LLM engine for this finder. False lets a
    # credential-free finder run without one, rather than requiring a key it never uses.
    requires_llm_engine = True

    def __init__(self):
        """Initialize the instance."""
        self.logging_artifacts = None

        # Set by `run_experiment` after instantiation, to a `deft.provenance.RunProvenance`.
        # None means nothing is recorded and no integrity check runs.
        self.provenance = None

    def close(self) -> None:  # noqa: B027 - a hook, empty where the finder holds nothing
        """Release what the finder holds for its run. A finder with no language model holds
        nothing."""

    def naive_features(
        self,
        X,
        sequence_column="raw_sequence",
        naive_position_prefix="seq_",
        naive_max_categorical_levels=20,
        positional_window=None,
        **_,
    ):
        """The local candidate population: positional indicators, plus the covariates.

        Shared, so the defaults exist once. It is DEFT's yardstick *and* the whole of CART's
        candidate set -- which is what makes `llm_split_share` and a CART baseline comparable:
        they are scored against the same population. Extra keyword arguments are ignored, so a
        caller can forward the whole of `params_generation`.
        """
        return get_naive_features(
            X,
            sequence_column=sequence_column,
            position_prefix=naive_position_prefix,
            max_categorical_levels=naive_max_categorical_levels,
            positional_window=positional_window,
        )

    def _record_candidate_pool(self, list_features):
        """Record how much of this node's scored candidate pool came from the language model.

        `llm_usable` is the number that the integrity policy checks: a candidate counts only
        if it was parsed, compiled to a callable, survived execution on this node's data, and
        received a finite score. A node where that is zero was split entirely by naive
        features. See `src/deft/conf/integrity/strict.yaml`.

        On the base class because a CART run records the same thing -- its whole pool is naive,
        and a provenance record that said nothing about it would be a gap for no reason.
        """
        if self.provenance is None:
            return
        llm_usable = 0
        llm_ran = 0
        naive = 0
        for feature in list_features:
            origin = getattr(feature, "origin", None)
            if origin == ORIGIN_NAIVE:
                naive += 1
            elif origin == ORIGIN_LLM:
                # Everything reaching here survived `filter_features`, so it compiled and ran on
                # this node's rows without raising. Scoring is the separate, later question of
                # whether any cut point left `min_samples_leaf` on both sides -- which a small
                # node can make impossible for a feature that is perfectly well written.
                llm_ran += 1
                score = feature.score
                if score is not None and np.isfinite(score):
                    llm_usable += 1
        self.provenance.record_candidate_pool(
            llm_usable=llm_usable, llm_ran=llm_ran, naive=naive
        )

    @abstractmethod
    def generate_features(self, X, y):
        """Generate features."""

    @abstractmethod
    def get_optimal_feature(
        self,
        X,
        y,
        min_samples_leaf,
        splitting_criterion,
        class_weight: Mapping[Any, float] | None = None,
        **kwargs,
    ):
        """Get optimal feature."""


class CartFeatureFinder(FeatureFinder):
    """Plain CART: axis-aligned splits on the data as given, and nothing generated.

    No language model, so no API call and no credentials -- which makes it the natural baseline
    for a DEFT run. It is the *same* tree, splitting criterion, threshold search and stopping
    rules, so a comparison against it isolates the feature generation and nothing else.

    Its candidates are `get_naive_features`, which is deliberately the identical population DEFT
    seeds every node with. Two things follow. A sequence column is decoded into one indicator
    per position and nucleotide, so this works on sequence data rather than only on tabular
    data -- upstream's version made one identity feature per column, which on a DNA string is a
    candidate that cannot be cast to a number and therefore never splits. And because the
    population is the same one, `llm_split_share` from a DEFT run and this baseline are measuring
    against the same yardstick.

    Select it with `feature_finder=cart`. Upstream called it `VanillaFeatureFinder`; the name
    said what it was not.
    """

    requires_llm_engine = False

    def __init__(self):
        """Initialize the instance."""
        super().__init__()

    def generate_features(self, X: pd.DataFrame, **kwargs):
        """Every naive candidate, and nothing else. Pass `sequence_column=None` for pure
        tabular data; `run_experiment` passes the dataset plugin's value through `fit`."""
        return self.naive_features(X, **kwargs)

    def get_optimal_feature(
        self,
        X,
        y,
        min_samples_leaf,
        splitting_criterion=None,
        class_weight: Mapping[Any, float] | None = None,
        **kwargs,
    ):
        """Get optimal feature."""
        list_features = self.generate_features(X, **kwargs)
        if not list_features:
            return None

        list_features = optimize_features(
            splitting_criterion=splitting_criterion,
            list_features=list_features,
            X=X,
            y=y,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )
        self._record_candidate_pool(list_features)

        list_features = sorted(list_features, key=lambda x: x.score, reverse=True)
        return _best_scoring_feature(list_features)


class LLMFeatureFinder(FeatureFinder):
    def __init__(
        self,
        llm_engine,
        parsing_module_name="deft.utils.parsing",
        type_engine="old",
        random_state=42,
    ):
        """Initialize the instance."""
        super().__init__()
        self.llm_engine = llm_engine
        self.type_engine = type_engine
        # The event loop every request of this finder runs on, made at the first (`_ask`).
        self._runner = None

        # Seeds the in-context examples shown in every prompt. Previously unseeded, so two runs
        # with the same `random_state` still saw different examples and were not comparable.
        # The generator is shared across calls, so examples still vary between prompts while
        # the whole sequence is reproducible.
        self.rng = np.random.default_rng(random_state)

        # Dynamically import the chosen module
        parsing_module = importlib.import_module(parsing_module_name)

        # Assign the functions from the module to instance methods
        # Now, self.parse_code_feature will point to the correct version
        self.parse_code_feature = parsing_module.parse_code_feature
        self.parse_proposed_feature = parsing_module.parse_proposed_feature
        self.parse_repaired_feature = getattr(
            parsing_module, "parse_repaired_feature", parse_repaired_feature
        )

    def _ask(self, prompts, n_generations_per_prompt):
        """The engine's replies to `prompts`, on the one event loop this finder keeps.

        LiteLLM keeps an HTTP client for each event loop it is called from, and closes none of
        them. With a fresh loop for every batch of prompts, as `asyncio.run` makes, each batch
        left its client behind with a connection open for every request it had made, until the
        process reached its limit on open files and requests failed. On one loop, every batch
        reuses one client and its connections.
        """
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner.run(self.llm_engine.query_llm(prompts, n_generations_per_prompt))

    def close(self) -> None:
        """Close the event loop the requests ran on. A later request opens a new one."""
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def __getstate__(self):
        """Everything except the event loop, which belongs to the process that fitted the tree
        and cannot be saved with it."""
        state = self.__dict__.copy()
        state["_runner"] = None
        return state

    def _record_generation(
        self, *, semantic_prompts, semantic_proposals, code_prompts, code_features
    ):
        """Record one `generate_features` pass. Reflection iterations accumulate."""
        if self.provenance is None:
            return
        self.provenance.record_generation(
            semantic_prompts=semantic_prompts,
            semantic_proposals=semantic_proposals,
            code_prompts=code_prompts,
            code_features=code_features,
        )

    def get_optimal_feature(
        self,
        X,
        y,
        min_samples_leaf,
        target_name,
        history,
        dataset_info,
        context_template_name_desc,
        context_code,
        instructions_name_desc,
        instructions_code,
        include_ic_samples,
        include_history,
        include_feature_description,
        use_description_in_history=True,
        splitting_criterion=None,
        n_ic_samples=2,
        population_size=10,
        n_samples_per_prompt=10,
        n_outcomes_per_sample_code=1,
        use_reflection=False,
        list_instructions_reflection=None,
        n_reflections=1,
        remove_duplicates=False,
        initialize_with_naive_features=False,
        include_code_in_reflection=False,
        include_rationale_in_reflection=True,
        selector=None,
        n_prompts=1,
        n_parents_per_prompt=1,
        sort_reflection_by_score=True,
        # Direction of that sort. False keeps the long-standing behaviour, best-scoring parent
        # last. See `serialize_list_features`.
        sort_reflection_best_first=False,
        reflection_banner_length=42,
        output_rationale=True,
        class_weight: Mapping[Any, float] | None = None,
        # What the dataset says each column means (`dataset.column_descriptions`), shown in
        # every prompt beside the column's type and range.
        column_descriptions: Mapping[str, str] | None = None,
        # The most characters of one text value an in-context row shows; None shows it whole.
        max_sequence_chars_in_prompt: int | None = None,
        # The share of a node's rows a candidate may be undefined (NaN) on and still be kept.
        max_undefined_share: float = 0.0,
        # How many times candidates that failed on the node's rows go back to the model, with
        # the prompt's wording (`_repair`). 0 sends nothing back.
        repair_rounds: int = 0,
        instructions_code_repair: str | None = None,
        repair_guidance: Mapping[str, str] | None = None,
        **kwargs,
    ):
        # Set the logging artifacts to none
        """Get optimal feature."""
        self.logging_artifacts = []
        generation = {
            "column_descriptions": column_descriptions,
            "max_undefined_share": max_undefined_share,
            "repair_rounds": repair_rounds,
            "instructions_code_repair": instructions_code_repair,
            "repair_guidance": repair_guidance,
        }

        prompt_name_description_generation = [
            construct_prompt_name_description(
                X=X,
                target_name=target_name,
                dataset_info=dataset_info,
                history=history,
                context_template=context_template_name_desc,
                instructions=instructions_name_desc,
                include_ic_samples=include_ic_samples,
                include_history=include_history,
                include_feature_description=include_feature_description,
                n_ic_samples=n_ic_samples,
                use_description_in_history=use_description_in_history,
                rng=self.rng,
                column_descriptions=column_descriptions,
                max_sequence_chars_in_prompt=max_sequence_chars_in_prompt,
            )
        ]

        list_features = self.generate_features(
            X=X,
            prompts_first_step=prompt_name_description_generation,
            context_code=context_code,
            instructions_code=instructions_code,
            n_outcomes_per_sample_name_desc=n_samples_per_prompt,
            n_outcomes_per_sample_code=n_outcomes_per_sample_code,
            output_rationale=output_rationale,
            **generation,
        )

        if initialize_with_naive_features:
            logger.info("Incorportaing the naive features in the initial population")
            # `sequence_column` and the two `naive_*` settings reach the shared helper through
            # `kwargs`, exactly as they do for `CartFeatureFinder`, so their defaults are
            # declared once.
            list_features.extend(self.naive_features(X, **kwargs))

        list_features = optimize_features(
            splitting_criterion=splitting_criterion,
            list_features=list_features,
            X=X,
            y=y,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )

        self._record_candidate_pool(list_features)

        list_original_features = copy(list_features)

        # Do Top-K selection
        list_features = sorted(list_features, key=lambda x: x.score, reverse=True)
        list_features = list_features[:population_size]



        # Add this list of features to the logging artifacts
        artifact = {"Type": "initial", "Features": copy(list_features)}
        self.logging_artifacts.append(artifact)

        logger.info(
            f"List features after initialization and Top-K selection: {list_to_str(list_features)}"
        )

        if use_reflection:
            if not list_instructions_reflection:
                logger.warning(
                    "use_reflection=True but no reflection instructions were provided; skipping reflection."
                )
                use_reflection = False
            elif selector is None:
                logger.warning(
                    "use_reflection=True but selector is None; skipping reflection."
                )
                use_reflection = False

        if use_reflection:
            logger.info("Using reflection" + reflection_banner_length * "-")
            for k in range(n_reflections):
                logger.info(f"Reflection Iteration number {k + 1}")
                prompts_reflection = []

                for instructions_reflection in list_instructions_reflection:
                    for _ in range(n_prompts):
                        prompts_reflection.append(
                            construct_prompt_reflection(
                                X=X,
                                list_features=list_features,
                                target_name=target_name,
                                dataset_info=dataset_info,
                                history=history,
                                context_template=context_template_name_desc,
                                instructions=instructions_reflection,
                                include_ic_samples=include_ic_samples,
                                include_history=include_history,
                                include_feature_description=include_feature_description,
                                sort_by_score=sort_reflection_by_score,
                                sort_best_first=sort_reflection_best_first,
                                use_description_in_history=use_description_in_history,
                                n_ic_samples=n_ic_samples,
                                include_code=include_code_in_reflection,
                                include_rationale=include_rationale_in_reflection,
                                selector=selector,
                                n_parents_per_prompt=n_parents_per_prompt,
                                rng=self.rng,
                                column_descriptions=column_descriptions,
                                max_sequence_chars_in_prompt=max_sequence_chars_in_prompt,
                            )
                        )

                list_features_reflection = self.generate_features(
                    X=X,
                    prompts_first_step=prompts_reflection,
                    context_code=context_code,
                    instructions_code=instructions_code,
                    n_outcomes_per_sample_name_desc=n_samples_per_prompt,
                    n_outcomes_per_sample_code=n_outcomes_per_sample_code,
                    output_rationale=output_rationale,
                    **generation,
                )

                logger.info(
                    f"List features generated by reflection: {list_to_str(list_features_reflection)}"
                )

                # optimize the features

                list_features_reflection = optimize_features(
                    splitting_criterion=splitting_criterion,
                    list_features=list_features_reflection,
                    X=X,
                    y=y,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                )

                # Reflection candidates count towards this node's pool too. Recording only the
                # initial round under-reports the node by however much reflection contributed.
                self._record_candidate_pool(list_features_reflection)



                artifact = {
                    "Type": f"Reflection {k} generation",
                    "Features": copy(list_features_reflection),
                }
                self.logging_artifacts.append(artifact)

                # do top k selection
                list_initial_and_reflection = list_features + list_features_reflection

                if remove_duplicates:
                    # `dict.fromkeys` rather than `set`: it preserves insertion order, so the
                    # stable sort below breaks ties by generation order rather than by hash
                    # bucket. Identity is what a feature computes -- see `FeatureInfo._identity`.
                    before = len(list_initial_and_reflection)
                    list_initial_and_reflection = list(
                        dict.fromkeys(list_initial_and_reflection)
                    )
                    removed = before - len(list_initial_and_reflection)
                    if removed:
                        logger.info(
                            "Removed %d duplicate candidate(s) of %d.", removed, before
                        )



                list_initial_and_reflection = sorted(
                    list_initial_and_reflection, key=lambda x: x.score, reverse=True
                )

                # Top-K selection
                list_features = (
                    list_initial_and_reflection[:population_size]
                    if population_size < len(list_initial_and_reflection)
                    else list_initial_and_reflection
                )

                artifact = {
                    "Type": f"Reflection {k} selection",
                    "Features": copy(list_features),
                }
                self.logging_artifacts.append(artifact)

        # Sort the features

        if len(list_features) == 0:
            logger.error(
                "Candidate pool is empty after selection; falling back to the initial "
                "population"
            )
            list_features = list_original_features

        if len(list_features) == 0:
            # Nothing to split on at all: the language model produced no usable feature and
            # naive-feature initialization is off. Upstream asserted here, which surfaced as a
            # bare AssertionError and pre-empted the integrity check's far more useful message.
            # Returning None lets the caller turn this node into a leaf, and lets the check in
            # `src/deft/provenance.py` report why the pool was empty.
            logger.error(
                "No candidate features at this node. The language model produced none, and "
                "`initialize_with_naive_features` is false, so there is nothing to split on."
            )
            return None

        sorted_features = sorted(list_features, key=lambda x: x.score, reverse=True)

        return _best_scoring_feature(sorted_features)

    def generate_features(
        self,
        X,
        prompts_first_step,
        context_code,
        instructions_code,
        n_outcomes_per_sample_name_desc=1,
        n_outcomes_per_sample_code=1,
        output_rationale=True,
        column_descriptions=None,
        max_undefined_share=0.0,
        repair_rounds=0,
        instructions_code_repair=None,
        repair_guidance=None,
        **kwargs,
    ):
        # The prompts first step is the prompt which ask for the semantic representations.
        """Generate features: propose them, write their code, keep what runs on these rows, and
        send what fails back to the model where `repair_rounds` allows (`_repair`)."""
        if not prompts_first_step:
            logger.warning("No prompts provided for feature generation. Returning empty list.")
            self._record_generation(
                semantic_prompts=0, semantic_proposals=0, code_prompts=0, code_features=0
            )
            return []
        logger.info(prompts_first_step[-1])

        # Generate the features
        responses_name_desc = self._ask(prompts_first_step, n_outcomes_per_sample_name_desc)

        # Parse the features
        # Token accounting is the budget's job (`src/deft/budget.py`), recorded per HTTP request
        # in the engine. The count returned here is redundant and deliberately dropped.
        list_features_name_desc, _ = self.parse_proposed_feature(
            responses_name_desc, output_rationale=output_rationale
        )

        # logger.info(f"List features: {list_features_name_desc}")

        # Flatten the list
        list_features_name_desc = [
            item for sublist in list_features_name_desc for item in sublist
        ]

        list_prompts_code = []
        # Loop over the features / Generate prompt for code generation / Generate the responses / Parse the responses
        for proposed_feature in list_features_name_desc:
            prompt_code_generation = construct_prompt_code_generation(
                X=X,
                context=context_code,
                proposed_feature=proposed_feature,
                instructions=instructions_code,
                column_descriptions=column_descriptions,
            )
            list_prompts_code.append(prompt_code_generation)
        if not list_prompts_code:
            logger.warning("No code-generation prompts could be constructed. Returning empty list.")
            self._record_generation(
                semantic_prompts=len(prompts_first_step),
                semantic_proposals=len(list_features_name_desc),
                code_prompts=0,
                code_features=0,
            )
            return []
        logger.info(list_prompts_code[-1])
        responses_code = self._ask(list_prompts_code, n_outcomes_per_sample_code)
        list_features, _ = self.parse_code_feature(
            responses_code, list_features_name_desc
        )

        # Flatten the list and return the

        list_features = [item for sublist in list_features for item in sublist]

        n_features_before_filter = len(list_features)
        # Filter the features
        list_features, failed = screen_features(list_features, X, max_undefined_share)
        n_features_after_filter = len(list_features)

        logger.info(
            f"Number of features before filtering: {n_features_before_filter} and after "
            f"filtering: {n_features_after_filter}"
        )
        self._record_generation(
            semantic_prompts=len(prompts_first_step),
            semantic_proposals=len(list_features_name_desc),
            code_prompts=len(list_prompts_code),
            code_features=n_features_after_filter,
        )
        if repair_rounds and failed:
            list_features += self._repair(
                X, failed, rounds=repair_rounds, context=context_code,
                instructions=instructions_code_repair, guidance=repair_guidance,
                column_descriptions=column_descriptions, max_undefined_share=max_undefined_share,
            )
        return list_features

    def _repair(self, X, failed, *, rounds, context, instructions, guidance, column_descriptions,
                max_undefined_share):
        """Send candidates that failed on this node's rows back to the model, and return those
        it corrects.

        Each round sends every candidate still failing, one prompt each
        (`construct_prompt_code_repair`): the feature as proposed, the code last tried for it,
        what went wrong, and how to correct that kind of failure. A code fault is fixed without
        changing what the feature computes. A feature constant or mostly undefined on these rows
        can only be corrected by computing something a little different, and then comes back
        under a name and description for it (`parse_repaired_feature`). A correction is kept
        where it now passes `screen_features`; one that fails again goes to the next round as it
        was written, with its own failure. Language-model candidates only: naive ones never
        reach here.
        """
        if not instructions or not guidance:
            raise ValueError(
                "`params_generation.repair_rounds` is above 0, so "
                "`params_generation.instructions_code_repair` and `repair_guidance` must be set."
            )
        repaired, prompts_sent = [], 0
        for _ in range(rounds):
            if not failed:
                break
            prompts = [
                construct_prompt_code_repair(
                    X, context, feature, detail, guidance[GUIDANCE_FOR[failure]], instructions,
                    column_descriptions,
                )
                for feature, failure, detail in failed
            ]
            responses = self._ask(prompts, 1)
            prompts_sent += len(prompts)
            still = []
            for (feature, failure, detail), response in zip(failed, responses, strict=True):
                if response is None:
                    # The request failed, which says nothing about the code: it stays as it was.
                    still.append((feature, failure, detail))
                    continue
                try:
                    corrected = self.parse_repaired_feature(parsing_end2end(response)[0], feature)
                except Exception as error:
                    # Unusable, so the code tried before it stands, and the next round is told.
                    still.append((feature, failure,
                                  f"{detail}, and its correction could not be used: {error}"))
                    continue
                kept, again = screen_features([corrected], X, max_undefined_share)
                repaired += kept
                still += again
            failed = still
        logger.info("Repair: %d candidate(s) corrected, %d still failing, %d prompt(s).",
                    len(repaired), len(failed), prompts_sent)
        if self.provenance is not None:
            self.provenance.record_repair(
                repair_prompts=prompts_sent, repaired=len(repaired), unrepaired=len(failed)
            )
        return repaired


