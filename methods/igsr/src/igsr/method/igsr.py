import json
import os
import re
import time
import traceback
from textwrap import dedent
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from igsr.agent.agent import LiteLLMAgent, completion_settings, retry_settings
from igsr.compute_profiler import ComputeProfiler
from igsr.config_typing import MainConfig
from igsr.dataset import DataBundle
from igsr.formula_tracker import FormulaTrackerMultiMetric
from igsr.logging_setup import get_logger, safe_cfg_yaml
from igsr.method.fitted_model import fit_final_model
from igsr.method.function_prompts import (
    ProposalRound,
    answer_format,
    postprocessor_generate_function_terms,
    prompt_generate_function_terms,
)
from igsr.method.function_terms import FUNCTION, TERM_LANGUAGES, FunctionTermSettings
from igsr.method.igsr_utils import (
    IterState,
    OptimizationMethod,
    PreparedDataset,
    _get_intercept,
    add_to_history,
    check_early_stop,
    check_token_budget_and_maybe_break,
    check_wallclock_budget_and_maybe_break,
    compute_raw_feature_importance,
    design_matrix,
    evaluate,
    get_iter_or_node_id,
    get_optimization_method,
    get_total_iters_for_print,
    ols_influence,
    postprocessor_generate_terms,
    postprocessor_pruning,
    prepare_dataset,
    prepare_function_dataset,
    pretty_equations,
)
from igsr.retry_decor import retryable
from igsr.tree_search.problem import Problem
from igsr.tree_search.session import SearchSession

# ================================================================================
# Prompt building
# ================================================================================


def _format_history(history: List[Dict[str, Any]]) -> str:
    """Pretty-print history for prompt injection."""
    if not history:
        return "- (no prior rounds) -"
    return "\n".join(
        f"Round {h['round']}:  KEEP={h['keep']}  |  DROP={h['drop']}  |  "
        f"MSE before pruning={h['MSE_before (total)']:.6f} (per-output={[float(m) for m in list(h['MSE_before (per-output)'])]}) |  "
        f"MSE after pruning={h['MSE_after (total)']:.6f} (per-output={[float(m) for m in list(h['MSE_after (per-output)'])]})"
        for h in history
    )


def _format_feature_importance(
    feature_importances: List[Tuple[str, float]],
    total_features: int,
) -> str:
    """Format feature importance as a markdown table for prompt injection."""
    n_shown = len(feature_importances)
    header = f"Top {n_shown} of {total_features} raw input features by importance (standardized Ridge |coefficient|, averaged across targets):"
    rows = "\n".join(
        f"| {rank} | {name} | {imp:.4f} |"
        for rank, (name, imp) in enumerate(feature_importances, 1)
    )
    return (
        f"{header}\n"
        f"| Rank | Feature | Importance |\n"
        f"|------|---------|------------|\n"
        f"{rows}\n"
        f"Higher importance = more predictive (linear effects). Nonlinear interactions may also matter."
    )


def _parse_seq_feature_name(feature_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (side, idx) for sequence one-hot features like seq_neg20__A or seq_20__T."""
    neg_match = re.fullmatch(r"seq_neg(\d+)__.+", feature_name)
    if neg_match:
        return "neg", int(neg_match.group(1))
    pos_match = re.fullmatch(r"seq_(\d+)__.+", feature_name)
    if pos_match:
        return "pos", int(pos_match.group(1))
    return None, None


def _iter_preview_feature_lines(
    data: Dict[str, np.ndarray],
    preview_N_elements: int,
    collapse_seq_feats: bool,
) -> Iterable[str]:
    """
    Yield feature preview lines, collapsing middle sequence positions to a single "...".
    """
    if not collapse_seq_feats:
        for feat_name, feat_val in data.items():
            yield f"{feat_name}[:{preview_N_elements}]:\n{feat_val[:preview_N_elements]}"
        return

    seq_indices: Dict[str, set[int]] = {"neg": set(), "pos": set()}
    for feat_name in data.keys():
        side, idx = _parse_seq_feature_name(feat_name)
        if side is not None and idx is not None:
            seq_indices[side].add(idx)

    keep_indices: Dict[str, set[int]] = {}
    for side, indices in seq_indices.items():
        if len(indices) <= 2:
            keep_indices[side] = set(indices)
        else:
            keep_indices[side] = {min(indices), max(indices)}

    inserted_gap_for_side: Dict[str, bool] = {"neg": False, "pos": False}
    for feat_name, feat_val in data.items():
        side, idx = _parse_seq_feature_name(feat_name)
        if side is not None and idx is not None:
            if idx not in keep_indices[side]:
                if not inserted_gap_for_side[side]:
                    yield "..."
                    inserted_gap_for_side[side] = True
                continue
        yield f"{feat_name}[:{preview_N_elements}]:\n{feat_val[:preview_N_elements]}"


def prompt_generate_terms(
    cfg: MainConfig,
    data_bundle: DataBundle,
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    history: List[Dict[str, Any]],
    current_terms: List[str],
    terms_sentinel: str,
    current_equation: Optional[str] = None,
    feature_importance_text: Optional[str] = None,
) -> str:
    preview_N_elements = getattr(cfg.experiment, "preview_N_elements", 50)
    collapse_seq_feats = getattr(cfg.experiment, "collapse_seq_feats", False)
    snippet_parts = ["Input data:\n"]

    for line in _iter_preview_feature_lines(data, preview_N_elements, collapse_seq_feats):
        snippet_parts.append(line)
    snippet_parts.append("\nTarget variables:\n")
    for y_name, y_val in y.items():
        snippet_parts.append(f"{y_name}[:{preview_N_elements}]:\n{y_val[:preview_N_elements]}")
    snippet = "\n".join(snippet_parts)

    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."

    focus_on_seq_feats = getattr(cfg.experiment, "focus_on_seq_feats", False)
    focus_on_seq_feats_text = """

========================================================
**NOTE FOR THIS SPECIFIC EXPERIMENT**
See if you can find "complex DNA sequence features" to further improve the predictions.
However, balance this with predictive accuracy - if simpler signal features increase accuracy, do not lose them.
========================================================
"""

    current_equation = current_equation if current_equation is not None else "N/A"
    prompt = f"""You are an automated assistant for proposing linear terms for the equations in a symbolic regression pipeline.

Your proposed terms will be:
1. Concatenated with the current candidate terms.
2. Sent to a LLM term pruner agent that will use various computed signals to decide which terms to keep and which to drop.

# Instructions:
Given below information, propose some candidate terms. The terms can be any valid numpy expressions.
Make use of the dataset and problem description to propose relevant terms.
Make sure to use the learnings from the history of previous rounds.

Return these between triple backticks and one term on each line.
The first backticks must be prepended with {terms_sentinel}

Example output:

{terms_sentinel}
```
x1
x2**2
np.sin(x3)
```

NOTES:
* Propose around {cfg.experiment.terms_per_round} terms, generally not too many unless this is the first round.
* If this *is* the first round, propose around {cfg.experiment.first_round_n_candidates} terms.
* You may propose no terms (nothing between the TERMS backticks) if you think it is appropriate.

DATASET AND PROBLEM DESCRIPTION
------------------------------
{data_bundle.data_dictionary}
{"" if feature_importance_text is None else chr(10) + "RAW FEATURE IMPORTANCE" + chr(10) + "------------------------------" + chr(10) + feature_importance_text + chr(10)}
CURRENT TERMS:
------------------------------
{current_terms}

# Current equation:
{current_equation}

HISTORY OF KEEP/DROP DECISIONS
------------------------------
{hist_txt}

=========
The input data and target variable(s) preview:
{snippet}
"""
    if focus_on_seq_feats:
        prompt += focus_on_seq_feats_text
    if cfg.experiment.print_prompts:
        print(f"prompt_generate_terms:\n==============\n{prompt}\n==============\n")
    return prompt


def prompt_generate_terms_simple(
    cfg: MainConfig,
    data_bundle: DataBundle,
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    history: List[Dict[str, Any]],
    current_terms: List[str],
    terms_sentinel: str,
    current_equation: Optional[str] = None,
    feature_importance_text: Optional[str] = None,
) -> str:
    """
    A simplified version of the generate-terms prompt.
    Keeps history but removes verbose previews and extra notes.
    """
    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."
    current_equation = current_equation if current_equation is not None else "N/A"
    prompt = f"""Propose candidate numpy expressions (terms) for linear symbolic regression.

Return your proposal between triple backticks, one term per line, and prepend the block with {terms_sentinel}.

Guidelines:
- Propose around {cfg.experiment.terms_per_round} terms (if first round: ~{cfg.experiment.first_round_n_candidates}).
- Use the dataset/problem description and the history to stay relevant.
- It's OK to return no terms if appropriate.

Example:
{terms_sentinel}
```
x1
x2**2
np.sin(x3)
```

DATASET AND PROBLEM DESCRIPTION
------------------------------
{data_bundle.data_dictionary}
{"" if feature_importance_text is None else chr(10) + "RAW FEATURE IMPORTANCE" + chr(10) + "------------------------------" + chr(10) + feature_importance_text + chr(10)}
CURRENT TERMS
------------------------------
{current_terms}

Current equation:
{current_equation}

HISTORY
------------------------------
{hist_txt}
"""
    if cfg.experiment.print_prompts:
        print(f"prompt_generate_terms_simple:\n==============\n{prompt}\n==============\n")
    return prompt


def prompt_pruning(
    cfg: MainConfig,
    data_bundle: DataBundle,
    dfs: Dict[str, pd.DataFrame],
    mse: list[float],
    mse_total: float,
    history: List[Dict[str, Any]],
    current_terms: List[str],
    decision_sentinel: str,
    current_equation: str,
    feature_importance_text: Optional[str] = None,
) -> str:
    tbls = ""
    for y_name, df in dfs.items():
        tbls += f"{y_name}:\n" + df.to_markdown(index=False) + "\n\n"

    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."

    if cfg.experiment.influence_feedback:
        if getattr(cfg.experiment, "refit_aware", False):
            influence_definition = (
                "**Influence definition (refit-aware)**\n\n"
                "Δ_k^{refit} is the increase in validation mean-squared-error (MSE) when the k-th term is "
                "removed and the model is refit:\n\n"
                "    Δ_k^{refit} = MSE_val( w^{(-k)} ) - MSE_val( w ).\n\n"
                "Here w^{(-k)} denotes the least-squares (or ridge) solution after dropping term k.\n"
            )
        else:
            influence_definition = (
                "**Influence definition (no refit)**\n\n"
                "Δ_k is the increase in mean-squared-error (MSE) if the k-th weight is deleted while all other "
                "weights stay fixed. For OLS this is\n\n"
                "    Δ_k = (w_k² / n) · ∑ φ_k(x_i)²  (always ≥ 0).\n"
            )
        # Add note that it isn't strictly "always ≥ 0" as values for validation rather than train are used.
        influence_definition += "\n\n* Note that the influence values are computed on the validation (rather than training) set, and thus may not always be ≥ 0."

        input_you_receive = f"""* A table (or dictionary/json representation) where each row has:

| field     | meaning                                                                   |
|-----------|---------------------------------------------------------------------------|
| term      | Name of the symbolic basis function φ_k(x) (e.g. "x1", "sin(x2)", ...).   |
| weight    | Fitted scalar coefficient w_k obtained by ordinary least squares (OLS).   |
| influence | Influence score for term k (see definition below).                        |

* The validation set MSE.

{influence_definition}

Hence:

* If influence is large ⇒ the term is important (its removal hurts the loss a lot).  
* If influence ≈ 0 ⇒ the term is useless (its removal makes no noticeable difference).
"""
    else:
        input_you_receive = """* A table (or dictionary/json representation) where each row has:

| field     | meaning                                                                   |
|-----------|---------------------------------------------------------------------------|
| term      | Name of the symbolic basis function φ_k(x) (e.g. "x1", "sin(x2)", ...).   |
| weight    | Fitted scalar coefficient w_k obtained by ordinary least squares (OLS).   |

* The validation set MSE.
"""

    if cfg.experiment.influence_feedback:
        example_input = """y_1:
| term   | weight | influence |
| ------ | ------ | --------- |
| x1     | 3.00   | 12.21     |
| x2     | -1.96  | 5.14      |
| x1**2  | 0.53   | 0.91      |
| sin    | 0.93   | 0.52      |
| cos    | -0.05  | 0.0009    |

y_2:
| term   | weight | influence |
| ------ | ------ | --------- |
| x1     | 3.00   | 12.21     |
| x2     | -1.96  | 5.14      |

MSE (per-output): [0.217, 0.145]
MSE overall: 0.181
"""
    else:
        example_input = """y_1:
| term   | weight |
| ------ | ------ |
| x1     | 3.00   |
| x2     | -1.96  |
| x1**2  | 0.53   |
| sin    | 0.93   |
| cos    | -0.05  |

y_2:
| term   | weight |
| ------ | ------ |
| x1     | 3.00   |
| x2     | -1.96  |

MSE (per-output): [0.217, 0.145]
MSE overall: 0.181
"""

    extra_notes = ""
    if cfg.experiment.influence_feedback:
        extra_notes += """\n* (!) You should also consider BOTH the weights and the influence of the terms."""
    if cfg.experiment.keep_n_terms is not None:
        extra_notes += (
            f"""\n* (!) You MUST keep {cfg.experiment.keep_n_terms} terms at most, to keep the model interpretable."""
        )

    prompt = f"""You are an equation-pruning assistant for symbolic regression.

========================================================
INPUT YOU RECEIVE
========================================================

{input_you_receive}

========================================================
YOUR TASK
========================================================
1. **Inspect every row**.
2. **Decide “keep” or “drop”** for each term using the rule:

* Use the heuristic: "Δₖ ≈ 0 → drop", "large Δₖ → keep" and your own judgement.

3. **Return** a python dictionary after "{decision_sentinel}" with exactly the two keys

{decision_sentinel}
```
{{
  "keep":  ["term_a", "term_b", ...],
  "drop":  ["term_c", "term_d", ...]
}}
````

Place each term name in either **keep** or **drop** — never both, never neither.

**IMPORTANT:**
* Make use of the dataset and problem description to make the best decision.
* Make sure to use the learnings from the history of previous rounds.
* (!) You must consider the generalization beyond the validation set and make decisions accordingly.{extra_notes}

────────────────────────────────────────────────────────
CONVENTIONS & NOTES
────────────────────────────────────────────────────────

# Important notes
* Treat terms independently; no need to refit or update weights.
* Note that everything was evaluated on the validation set to avoid overfitting.
* If there are multiple outputs (targets), you will see multiple tables, one for each target.
* Use all the information available, but keep in mind that you must only return one keep/drop decision even if there are multiple outputs.
* Keep only the most important terms for each output.

# Output format
* Feel free to comment briefly (≤ 30 chars) about each decision, but keep the python dictionary in the right format.
* The dictionary MUST be provided between triple backticks, otherwise it cannot be parsed.
* It must be prepended with "{decision_sentinel}", otherwise it cannot be parsed.

────────────────────────────────────────────────────────
EXAMPLE
────────────────────────────────────────────────────────
# INPUT TABLE(S):

{example_input}

# Output:

{decision_sentinel}
```
{{
  "keep": ["x1", "x2", "x1**2", "sin"],
  "drop": ["cos"]
}}
```

That's it — perform the keep/drop decision based on the information provided.


========================================================
DATASET AND PROBLEM DESCRIPTION
========================================================
{data_bundle.data_dictionary}
{"" if feature_importance_text is None else chr(10) + "========================================================" + chr(10) + "RAW FEATURE IMPORTANCE" + chr(10) + "========================================================" + chr(10) + feature_importance_text + chr(10)}

========================================================
CURRENT TERMS
========================================================
{current_terms}

# Current equation:
{current_equation}

========================================================
HISTORY OF KEEP/DROP DECISIONS
========================================================
{hist_txt}


========================================================
INPUT YOU RECEIVE
========================================================

INPUT TABLE(S):

{tbls}

MSE (per-output): {[float(m) for m in list(mse)]}
MSE overall: {mse_total:.6f}
"""
    if cfg.experiment.print_prompts:
        print(f"prompt_pruning:\n==============\n{prompt}\n==============\n")
    return prompt


def prompt_pruning_simple(
    cfg: MainConfig,
    data_bundle: DataBundle,
    dfs: Dict[str, pd.DataFrame],
    mse: list[float],
    mse_total: float,
    history: List[Dict[str, Any]],
    current_terms: List[str],
    decision_sentinel: str,
    current_equation: str,
    feature_importance_text: Optional[str] = None,
) -> str:
    """
    A simplified version of the pruning prompt.
    Keeps history, presents compact instructions and the essential tables/metrics.
    """
    tbls = ""
    for y_name, df in dfs.items():
        tbls += f"{y_name}:\n" + df.to_markdown(index=False) + "\n\n"
    if cfg.experiment.history_enabled:
        hist_txt = _format_history(history)
    else:
        hist_txt = "History is disabled."
    extra_notes = ""
    if cfg.experiment.influence_feedback:
        extra_notes += "Consider both weights and influence if provided. "
    if cfg.experiment.keep_n_terms is not None:
        extra_notes += f"You MUST keep at most {cfg.experiment.keep_n_terms} terms. "
    fi_section = ""
    if feature_importance_text is not None:
        fi_section = f"""
RAW FEATURE IMPORTANCE
------------------------------
{feature_importance_text}
"""
    prompt = f"""Decide which terms to KEEP or DROP for the linear symbolic model.

What you have:
- Table(s) per target with columns: term, weight{", influence" if cfg.experiment.influence_feedback else ""}.
- Validation MSE (per-output and overall).
- Dataset/problem context and the history of previous rounds.

Instructions:
- Treat each term independently; use weights{"+influence" if cfg.experiment.influence_feedback else ""}, MSE, history and context.
- Favor generalization beyond validation.
- {extra_notes.strip()}

Return a python dict after "{decision_sentinel}" exactly in this format:
{decision_sentinel}
```
{{
  "keep": ["term_a", "term_b"],
  "drop": ["term_c", "term_d"]
}}
```

CURRENT TERMS
------------------------------
{current_terms}

Current equation:
{current_equation}
{fi_section}
HISTORY
------------------------------
{hist_txt}

INPUT TABLE(S)
------------------------------
{tbls}

MSE (per-output): {[float(m) for m in list(mse)]}
MSE overall: {mse_total:.6f}
"""
    if cfg.experiment.print_prompts:
        print(f"prompt_pruning_simple:\n==============\n{prompt}\n==============\n")
    return prompt


# ================================================================================
# Main components
# ================================================================================


def _append_new_terms_to_json(
    cfg: MainConfig,
    seed: int,
    iter_or_node_id: str,
    new_terms: List[str],
    exp_logger: structlog.stdlib.BoundLogger,
) -> None:
    """
    Append the new_terms for a given iter_or_node_id into a per-seed JSON file
    located next to the configured log_file. If log_file is not set, write in CWD.
    This function is a no-op if cfg.experiment.save_new_terms is False.
    """
    if not getattr(cfg.experiment, "save_new_terms", False):
        return
    try:
        base_log = getattr(cfg.logging, "log_file", None)
        dir_path = os.path.dirname(base_log) if base_log else "."
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"new_terms.seed_{seed}.json")

        # Load existing
        data: Dict[str, Any]
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                # If the file is corrupted or empty, start fresh
                data = {"seed": seed, "entries": []}
        else:
            data = {"seed": seed, "entries": []}

        # Ensure structure
        if not isinstance(data, dict):
            data = {"seed": seed, "entries": []}
        if "entries" not in data or not isinstance(data["entries"], list):
            data["entries"] = []
        if "seed" not in data:
            data["seed"] = seed

        # Append new entry
        data["entries"].append(
            {
                "id": str(iter_or_node_id),
                "new_terms": list(new_terms),
            }
        )

        # Write back
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        exp_logger.info(
            "new_terms_saved",
            seed=seed,
            iter_or_node_id=iter_or_node_id,
            file_path=file_path,
            num_terms=len(new_terms),
        )
    except Exception as e:
        exp_logger.error("new_terms_save_error", seed=seed, iter_or_node_id=iter_or_node_id, error=e)


def train_and_evaluate(
    cfg: MainConfig,
    terms: List[str],
    data: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    data_eval: Dict[str, np.ndarray],
    y_eval: Dict[str, np.ndarray],
) -> Tuple[
    OptimizationMethod,
    np.ndarray,  # per-output MSE
    float,  # total MSE
    Dict[str, float],  # All metrics per-output
]:
    Φ_final = design_matrix(terms, data).phi
    reg, reg_kwargs = get_optimization_method(cfg)
    reg = reg(**reg_kwargs).fit(Φ_final, np.column_stack(list(y.values())))
    mse_eval, mse_eval_total, all_metrics_eval = evaluate(terms, data_eval, y_eval, reg)
    return reg, mse_eval, mse_eval_total, all_metrics_eval


def propose_and_prune_once(
    current_terms: List[str],
    history: List[Dict[str, Any]],
    prepared_dataset: PreparedDataset,
    # ---
    cfg: MainConfig,
    data_bundle: DataBundle,
    exp_logger: structlog.stdlib.BoundLogger,
    seed: int,
    # --
    azure_model_key: str,
    iter_num: Optional[int] = None,
    node_id: Optional[str] = None,
    # ---
    compute_profiler: ComputeProfiler = None,
    feature_importance_text: Optional[str] = None,
) -> IterState:
    iter_or_node_id = get_iter_or_node_id(cfg, iter_num, node_id)
    total_iters_str = get_total_iters_for_print(cfg)
    stage_name = "propose"

    TERMS_SENTINEL = "TERMS:"
    agent_generate_terms_sentinels = [TERMS_SENTINEL]
    function_form = cfg.experiment.term_language == FUNCTION
    if function_form:
        task_generate_terms = prompt_generate_function_terms(
            data_dictionary=data_bundle.data_dictionary,
            context=prepared_dataset.function_context,
            settings=prepared_dataset.registry.settings,
            registry=prepared_dataset.registry,
            current_terms=current_terms,
            current_equation=history[-1]["equation_after"] if history else None,
            history_text=_format_history(history) if cfg.experiment.history_enabled else "History is disabled.",
            terms_per_round=cfg.experiment.terms_per_round,
            first_round_n_candidates=cfg.experiment.first_round_n_candidates,
            terms_sentinel=TERMS_SENTINEL,
        )
        if cfg.experiment.print_prompts:
            print(f"prompt_generate_function_terms:\n==============\n{task_generate_terms}\n==============\n")
        correct_format_generate_terms = f"Reminder of correct output format:\n\n{answer_format(TERMS_SENTINEL)}\n"
    else:
        task_generate_terms = (
            prompt_generate_terms_simple if cfg.experiment.simplified_prompts else prompt_generate_terms
        )(
            cfg,
            data_bundle,
            data=prepared_dataset.data,
            y=prepared_dataset.y,
            history=history,
            current_terms=current_terms,
            terms_sentinel=TERMS_SENTINEL,
            current_equation=history[-1]["equation_after"] if history else None,
            feature_importance_text=feature_importance_text,
        )
        correct_format_generate_terms = dedent(f"""
            Reminder of correct output format:

            {TERMS_SENTINEL}
            ```
            term1
            ...
            termN
            ```
            """)

    def on_retry(exc: Exception, tries_left: int):
        exp_logger.info("retry_attempt", seed=seed, error=exc, tries_left=tries_left)

    @retryable(attempts=cfg.experiment.retry_attempts, delay=0, on_retry=on_retry)
    def _():
        completion_kwargs = completion_settings(cfg.llm, azure_model_key)
        if function_form:
            # A fresh round per attempt: the terms it accepts join the search's registry at once.
            postprocessor = postprocessor_generate_function_terms
            postprocessor_kwargs = {
                "registry": prepared_dataset.registry,
                "training": prepared_dataset.data,
                "progress": ProposalRound(),
                "terms_sentinel": TERMS_SENTINEL,
                "first_seen": iter_or_node_id,
                "correct_format": correct_format_generate_terms,
            }
        else:
            postprocessor = postprocessor_generate_terms
            postprocessor_kwargs = {
                "iter_num": iter_num,
                "node_id": node_id,
                "cfg": cfg,
                "data_bundle": data_bundle,
                "seed": seed,
                "terms_sentinel": TERMS_SENTINEL,
                "stage_name": stage_name,
                "current_terms": current_terms,
                # ---
                "prepared_dataset": prepared_dataset,  # To check for invalid values.
                # ---
                "correct_format_generate_terms": correct_format_generate_terms,
            }
        agent_generate_terms = LiteLLMAgent(
            task_description=task_generate_terms,
            system_prompt="",  # Only use the task description.
            logger=exp_logger,
            max_steps=cfg.experiment.max_agent_steps,
            model=f"{cfg.llm.kind}/{cfg.llm.deployment}",
            completion_kwargs=completion_kwargs,
            retry_config=retry_settings(cfg.llm),
            task_complete_sentinels=agent_generate_terms_sentinels,
            postprocessor=postprocessor,
            postprocessor_kwargs=postprocessor_kwargs,
            reminder=correct_format_generate_terms,
        )

        try:
            time_start = time.time()

            result = agent_generate_terms.run()

            time_end = time.time()
            compute_profiler.accumulate("wall_clock_llm_calls", time_end - time_start)
            for usage_entry in agent_generate_terms.usage_history:
                if "completion_tokens" in usage_entry:
                    compute_profiler.accumulate("output_tokens", usage_entry["completion_tokens"])
                if "prompt_tokens" in usage_entry:
                    compute_profiler.accumulate("input_tokens", usage_entry["prompt_tokens"])
            compute_profiler.accumulate("number_llm_calls", agent_generate_terms.num_llm_calls)

            new_terms = result.postprocessor_output.parsed_output
            exp_logger.info(
                f"{stage_name}_agent_run_finished",
                seed=seed,
                stage_name=stage_name,
                iter_num=iter_num,
                node_id=node_id,
                terms=new_terms,
                conversation_history=agent_generate_terms.conversation_history,
                usage_history=agent_generate_terms.usage_history,
            )
            print(
                f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - "
                f"Stage {stage_name} - New terms:\n{new_terms}"
            )
            return new_terms
        except Exception as e:
            exp_logger.error(
                f"{stage_name}_agent_run_error",
                seed=seed,
                stage_name=stage_name,
                iter_num=iter_or_node_id,
                node_id=node_id,
                error=e,
            )
            traceback.print_exc()
            print(f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - " f"Stage {stage_name} - Error: {e}")
            exp_logger.info(
                f"{stage_name}_agent_run_error",
                seed=seed,
                stage_name=stage_name,
                iter_num=iter_num,
                node_id=node_id,
                error=e,
            )
            raise

    new_terms = _()

    # Optionally save new terms for this iteration/node.
    _append_new_terms_to_json(cfg, seed, iter_or_node_id, new_terms, exp_logger)

    # Concatenate terms, unique & order-stable:
    candidate_terms = list(dict.fromkeys(current_terms + new_terms))
    print(f"Candidate terms:\n{candidate_terms}")
    # OLS + influence:
    dm = design_matrix(candidate_terms, prepared_dataset.data)
    dm_validation = design_matrix(candidate_terms, prepared_dataset.data_validation)
    influence_compute_results = ols_influence(
        cfg,
        dm.phi,
        np.column_stack(list(prepared_dataset.y.values())),
        dm_validation.phi,
        np.column_stack(list(prepared_dataset.y_validation.values())),
    )
    mse_before_validation, mse_before_validation_total, all_metrics_before_validation = evaluate(
        dm.term_names, prepared_dataset.data_validation, prepared_dataset.y_validation, influence_compute_results.reg
    )
    if prepared_dataset.data_test is not None:
        _, _, all_metrics_before_test = evaluate(
            dm.term_names, prepared_dataset.data_test, prepared_dataset.y_test, influence_compute_results.reg
        )
    else:
        all_metrics_before_test = None
    if prepared_dataset.data_ood_test is not None:
        _, _, all_metrics_before_ood_test = evaluate(
            dm.term_names, prepared_dataset.data_ood_test, prepared_dataset.y_ood_test, influence_compute_results.reg
        )
    else:
        all_metrics_before_ood_test = None
    print(f"New candidate terms: {new_terms}")
    print(f"MSE before pruning = {mse_before_validation_total:.6f} -- per-output: {list(mse_before_validation)}")
    equation_before = pretty_equations(
        w=influence_compute_results.weights, terms=dm.term_names, y=prepared_dataset.y,
        intercept=influence_compute_results.intercept,
    )
    print(equation_before)

    # Construct the dataframes for preview.
    # In the function form a term's name is the model's own, so each row also says what it measures.
    dfs = dict()
    for i, y_name in enumerate(prepared_dataset.y.keys()):
        columns = {"term": dm.term_names}
        if function_form:
            columns["description"] = [prepared_dataset.registry[name].description for name in dm.term_names]
        columns["weight"] = influence_compute_results.weights[:, i]
        if cfg.experiment.influence_feedback:
            columns["influence"] = influence_compute_results.delta_validation[:, i]  # Validation!
        dfs[y_name] = pd.DataFrame(columns)
    print(f"dfs:\n{dfs}")

    stage_name = "prune"
    PRUNING_SENTINEL = "DECISION:"
    agent_pruning_sentinels = [PRUNING_SENTINEL]

    # Algorithmic pruning option: keep top-K by influence, drop the rest.
    if cfg.experiment.algorithmic_pruning is True:
        delta_val = influence_compute_results.delta_validation
        if delta_val.ndim != 2:
            raise ValueError(f"delta_val.ndim {delta_val.ndim} must be 2 in algorithmic pruning")
        # Take worst across dimensions - other alternatives like mean are possible.
        aggregated_influence = np.max(delta_val, axis=1)

        n_keep_cfg = cfg.experiment.keep_n_terms
        n_keep = max(0, min(int(n_keep_cfg), len(dm.term_names)))

        sorted_idx = np.argsort(aggregated_influence)[::-1]
        selected_idx = set(sorted_idx[:n_keep].tolist())
        keep = [t for i, t in enumerate(dm.term_names) if i in selected_idx]
        drop = [t for i, t in enumerate(dm.term_names) if i not in selected_idx]

        exp_logger.info(
            f"{stage_name}_algorithmic_decision",
            seed=seed,
            stage_name=stage_name,
            iter_num=iter_num,
            node_id=node_id,
            n_keep=n_keep,
            keep=keep,
            drop=drop,
        )
        print(
            f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - "
            f"Stage {stage_name} (algorithmic) - Keep: {keep}, Drop: {drop}"
        )
    else:
        task_prompt_pruning = (prompt_pruning_simple if cfg.experiment.simplified_prompts else prompt_pruning)(
            cfg,
            data_bundle,
            dfs=dfs,
            mse=list(influence_compute_results.mse_validation),
            mse_total=influence_compute_results.mse_validation_total,
            history=history,
            current_terms=dm.term_names,
            decision_sentinel=PRUNING_SENTINEL,
            current_equation=equation_before,
            feature_importance_text=feature_importance_text,
        )

        correct_format_pruning = dedent(f"""
            Reminder of correct output format:

            {PRUNING_SENTINEL}
            ```
            {{
                "keep": [term1, term2, ...],
                "drop": [term1, term2, ...],
            }}
            ```
            """)
        completion_kwargs = completion_settings(cfg.llm, azure_model_key)
        agent_pruning = LiteLLMAgent(
            task_description=task_prompt_pruning,
            system_prompt="",  # Only use the task description.
            logger=exp_logger,
            max_steps=cfg.experiment.max_agent_steps,
            model=f"{cfg.llm.kind}/{cfg.llm.deployment}",
            completion_kwargs=completion_kwargs,
            retry_config=retry_settings(cfg.llm),
            task_complete_sentinels=agent_pruning_sentinels,
            postprocessor=postprocessor_pruning,
            postprocessor_kwargs={
                "iter_num": iter_num,
                "node_id": node_id,
                "cfg": cfg,
                "data_bundle": data_bundle,
                "seed": seed,
                "pruning_sentinel": PRUNING_SENTINEL,
                "stage_name": stage_name,
                "current_terms": dm.term_names,
                # ---
                # "data": prepared_dataset.data,  # To check for invalid values.
                # ---
                "correct_format_pruning": correct_format_pruning,
            },
            reminder=correct_format_pruning,
        )

        @retryable(attempts=cfg.experiment.retry_attempts, delay=0, on_retry=on_retry)
        def _():
            try:
                time_start = time.time()

                result = agent_pruning.run()

                time_end = time.time()
                compute_profiler.accumulate("wall_clock_llm_calls", time_end - time_start)
                for usage_entry in agent_pruning.usage_history:
                    if "completion_tokens" in usage_entry:
                        compute_profiler.accumulate("output_tokens", usage_entry["completion_tokens"])
                    if "prompt_tokens" in usage_entry:
                        compute_profiler.accumulate("input_tokens", usage_entry["prompt_tokens"])
                compute_profiler.accumulate("number_llm_calls", agent_pruning.num_llm_calls)

                keep, drop = result.postprocessor_output.parsed_output
                exp_logger.info(
                    f"{stage_name}_agent_run_finished",
                    seed=seed,
                    stage_name=stage_name,
                    iter_num=iter_num,
                    keep=keep,
                    drop=drop,
                    conversation_history=agent_pruning.conversation_history,
                    usage_history=agent_pruning.usage_history,
                )
                exp_logger.info(
                    f"{stage_name}_agent_run_finished",
                    seed=seed,
                    stage_name=stage_name,
                    iter_num=iter_num,
                    keep=keep,
                    drop=drop,
                )
                print(
                    f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - "
                    f"Stage {stage_name} - Keep: {keep}, Drop: {drop}"
                )
                return keep, drop
            except Exception as e:
                exp_logger.error(
                    f"{stage_name}_agent_run_error",
                    seed=seed,
                    stage_name=stage_name,
                    iter_num=iter_num,
                    error=e,
                )
                traceback.print_exc()
                print(
                    f"Seed {seed} - Iteration {iter_or_node_id}/{total_iters_str} - " f"Stage {stage_name} - Error: {e}"
                )
                raise

        keep, drop = _()

    # Concatenate terms, unique & order-stable:
    print(f"\nkeep: {keep}")
    print(f"drop: {drop}")
    surviving_terms = [t for t in candidate_terms if t in keep]
    print(f"Surviving terms:\n{surviving_terms}")


    if len(surviving_terms) == 0:
        mse_after_validation_total = float("inf")
        all_metrics_after_validation = {
            "mse": float("inf"),
            "rmse": float("inf"),
            "r2": float("-inf"),
            "nrmse": float("inf"),
            "nmse": float("inf"),
            "accuracy_tol": 0,
            "accuracy_tol_max": 0,
        }
        all_metrics_after_test = None if prepared_dataset.data_test is None else {
            "mse": float("inf"),
            "rmse": float("inf"),
            "r2": float("-inf"),
            "nrmse": float("inf"),
            "nmse": float("inf"),
            "accuracy_tol": 0,
            "accuracy_tol_max": 0,
        }
        all_metrics_after_ood_test = None
        mse_after_validation = [float("inf")] * len(prepared_dataset.y.keys())
        equation_after = "N/A"
        print("No surviving terms.")
    else:
        # Refit on the surviving terms:
        (
            reg_pruned,
            mse_after_validation,
            mse_after_validation_total,
            all_metrics_after_validation,
        ) = train_and_evaluate(
            cfg,
            surviving_terms,
            prepared_dataset.data,
            prepared_dataset.y,
            prepared_dataset.data_validation,
            prepared_dataset.y_validation,
        )
        print(
            f"MSE AFTER PRUNING = {mse_after_validation_total:.6f} -- "
            f"per-output: {[float(m) for m in list(mse_after_validation)]}\n"
        )
        # Evaluate on the test set, where there is one (reporting only):
        if prepared_dataset.data_test is not None:
            _, _, all_metrics_after_test = evaluate(
                surviving_terms,
                prepared_dataset.data_test,
                prepared_dataset.y_test,
                reg_pruned,
            )
        else:
            all_metrics_after_test = None
        if prepared_dataset.data_ood_test is not None:
            _, _, all_metrics_after_ood_test = evaluate(
                surviving_terms,
                prepared_dataset.data_ood_test,
                prepared_dataset.y_ood_test,
                reg_pruned,
            )
        else:
            all_metrics_after_ood_test = None
        equation_after = pretty_equations(
            w=reg_pruned.coef_.T, terms=surviving_terms, y=prepared_dataset.y,
            intercept=_get_intercept(reg_pruned),
        )
    print(equation_after)

    iter_result = IterState(
        iter_num=iter_num,
        node_id=node_id,
        terms_before=candidate_terms,
        terms_after=surviving_terms,
        keep=keep,
        drop=drop,
        # ---
        mse_before=influence_compute_results.mse_validation,
        mse_before_total=influence_compute_results.mse_validation_total,
        mse_after=mse_after_validation,
        mse_after_total=mse_after_validation_total,
        # ---
        metrics_before=all_metrics_before_validation,
        metrics_after=all_metrics_after_validation,
        metrics_test_before=all_metrics_before_test,
        metrics_test_after=all_metrics_after_test,
        # ---
        metrics_ood_test_before=all_metrics_before_ood_test,
        metrics_ood_test_after=all_metrics_after_ood_test,
        # ---
        history=[],  # Set below.
        equation_before=equation_before,
        equation_after=equation_after,
    )
    history = add_to_history(
        history=history,
        iter_id=get_iter_or_node_id(cfg, iter_num, node_id),
        iter_result=iter_result,  # Doesn't store iter_result itself, just takes some info from it.
        exp_logger=exp_logger,
        seed=seed,
    )
    iter_result.history = history  # Set the history after adding to it.
    return iter_result


# For ablation only.
def drop_columns_randomly(
    df: pd.DataFrame,
    n_keep: int,
    target_columns: List[str],
    random_seed: int,
    always_keep_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Randomly drop columns from a dataframe, keeping n_keep non-target columns plus all target columns.
    Target columns are never dropped. If always_keep_columns is provided, those columns are guaranteed
    to be among the kept columns (counting toward n_keep).
    """
    np.random.seed(random_seed)

    all_columns = list(df.columns)

    missing_targets = set(target_columns) - set(all_columns)
    if missing_targets:
        raise ValueError(f"Target columns not found in dataframe: {missing_targets}")

    if n_keep < 0:
        raise ValueError(f"n_keep ({n_keep}) cannot be negative")

    droppable_columns = [col for col in all_columns if col not in target_columns]

    if n_keep > len(droppable_columns):
        raise ValueError(
            f"n_keep ({n_keep}) cannot be greater than available non-target columns ({len(droppable_columns)})"
        )

    # Handle always_keep_columns
    if always_keep_columns:
        # Filter to only those that actually exist as droppable columns
        forced_keep = [col for col in always_keep_columns if col in droppable_columns]
        missing_ak = [col for col in always_keep_columns if col not in droppable_columns and col not in target_columns]
        if missing_ak:
            print(f"WARNING: keep_n_columns_always_keep columns not found in data (ignoring): {missing_ak}")
        if len(forced_keep) > n_keep:
            print(
                f"WARNING: always_keep_columns ({len(forced_keep)}) > n_keep ({n_keep}), "
                f"keeping all forced columns ({len(forced_keep)} total)."
            )
            n_keep = len(forced_keep)
    else:
        forced_keep = []

    if n_keep >= len(droppable_columns):
        columns_to_keep = all_columns
    else:
        # Random pool excludes forced-keep columns
        random_pool = [col for col in droppable_columns if col not in forced_keep]
        n_random = n_keep - len(forced_keep)
        randomly_selected = np.random.choice(random_pool, size=n_random, replace=False).tolist() if n_random > 0 else []
        columns_to_keep = target_columns + forced_keep + randomly_selected

    return df[columns_to_keep]


def igsr(cfg: MainConfig, seed: int, data_bundle: DataBundle) -> Dict[str, Any]:
    """
    Run the search with a specific seed on a dataset.

    Args:
        cfg (MainConfig): The configuration.
        seed (int): The random seed for this run; it also seeds the tree search.
        data_bundle (DataBundle): The dataset. Its test split may be None.

    Returns:
        Dict[str, Any]: A dictionary containing:
            {
                "seed": int,
                "best_iter_id": str,  # the iteration (or node) with the lowest validation MSE
                "best_formula": str,  # its equation
                "best_metric": float,  # its validation MSE
                "best_terms": List[str],
                "best_test_metrics": Optional[Dict[str, float]],  # reporting only; None without a test split
                "best_model": FittedModel,  # its terms refitted on the training split
                "term_records": Optional[List[Dict]],  # function form: every term accepted during the search
                "term_failures": Optional[Dict],  # function form: terms that failed on a split after acceptance
                "proposal_stats": Optional[Dict[str, int]],  # function form: what came of the proposals
                "history_validation": list, "history_test": Optional[list], "history_ood_test": Optional[list],
                "compute_profiler": Dict[str, float],
            }
    """
    # Profiling:
    compute_profiler = ComputeProfiler(
        {"wall_clock_total", "wall_clock_llm_calls", "input_tokens", "output_tokens", "number_llm_calls"}
    )
    start_overall = time.time()

    # The configuration with the API key redacted, since the resolved key would otherwise be printed.
    print(safe_cfg_yaml(cfg))

    # Create a seed-specific logger
    log_file = cfg.logging.log_file.replace(".jsonl", f".seed_{seed}.jsonl") if cfg.logging.log_file else None
    exp_logger = get_logger(f"{cfg.logging.logger_name}.seed_{seed}", log_file_path=log_file, mode=cfg.logging.mode)
    exp_logger.info("run_start", seed=seed, config=safe_cfg_yaml(cfg))

    # Azure / LLM config:
    azure_model_key = cfg.llm.api_key

    term_language = cfg.experiment.term_language
    if term_language not in TERM_LANGUAGES:
        raise ValueError(f"experiment.term_language must be one of {', '.join(TERM_LANGUAGES)}, not {term_language!r}.")

    # Drop columns ablation ------------------------------------------------------------
    if cfg.experiment.keep_n_columns is not None:
        actually_keep_n_columns = min(
            cfg.experiment.keep_n_columns, data_bundle.dataset_train.shape[1] - len(data_bundle.target_columns)
        )
        always_keep = getattr(cfg.experiment, "keep_n_columns_always_keep", None)
        print(f"Actually keeping {actually_keep_n_columns} columns.")
        if always_keep:
            print(f"Always keeping columns: {always_keep}")
        data_bundle.dataset_train = drop_columns_randomly(
            data_bundle.dataset_train, actually_keep_n_columns, data_bundle.target_columns, seed, always_keep
        )
        data_bundle.dataset_validation = drop_columns_randomly(
            data_bundle.dataset_validation, actually_keep_n_columns, data_bundle.target_columns, seed, always_keep
        )
        if data_bundle.dataset_test is not None:
            data_bundle.dataset_test = drop_columns_randomly(
                data_bundle.dataset_test, actually_keep_n_columns, data_bundle.target_columns, seed, always_keep
            )
        remaining_columns = [f"{c}" for c in list(data_bundle.dataset_train.columns) if c not in data_bundle.target_columns]
        print(f"Remaining columns: {remaining_columns}")
        data_bundle.data_dictionary += f"""
# Important note:
In this run some data columns were dropped and will not be available.
ONLY THE FOLLOWING {len(remaining_columns)} COLUMNS (and the target columns) ARE AVAILABLE:
{remaining_columns}
"""
    # -----------------------------------------------------------------------------------

    # Prepare the dataset, in the form the term language reads:
    if term_language == FUNCTION:
        if getattr(cfg.experiment, "show_feature_importance", False):
            raise ValueError("show_feature_importance ranks numeric input columns, which the function form's "
                             "terms need not be; it is available in the expression form only.")
        settings = FunctionTermSettings.from_config(cfg.experiment.function_terms)
        prepared_dataset = prepare_function_dataset(data_bundle, settings, seed)
    else:
        prepared_dataset = prepare_dataset(data_bundle, exp_logger)
    has_test = prepared_dataset.data_test is not None

    # Compute raw feature importance (if enabled):
    if getattr(cfg.experiment, "show_feature_importance", False):
        max_fi = getattr(cfg.experiment, "feature_importance_max_features", 20)
        feature_importances = compute_raw_feature_importance(prepared_dataset, max_features=max_fi)
        feature_importance_text = _format_feature_importance(
            feature_importances, total_features=len(prepared_dataset.data)
        )
        print(f"Feature importance (top {len(feature_importances)}):\n{feature_importance_text}")
    else:
        feature_importance_text = None

    # Initializations
    def new_tracker() -> FormulaTrackerMultiMetric:
        return FormulaTrackerMultiMetric(
            metric_directions={
                "mse": "min",
                "rmse": "min",
                "r2": "max",
                "nrmse": "min",
                "nmse": "min",
                "accuracy_tol": "max",
                "accuracy_tol_max": "max",
            }
        )

    formula_tracker_val_set = new_tracker()
    formula_tracker_test_set = new_tracker() if has_test else None
    formula_tracker_ood_test_set = new_tracker() if data_bundle.dataset_ood_test is not None else None

    def track(iter_id: str, state: IterState) -> None:
        """Record an iteration's metrics after pruning, on every split there is."""
        formula_tracker_val_set.update(iter_id, state.metrics_after, metadata=state.to_dict())
        if formula_tracker_test_set is not None:
            formula_tracker_test_set.update(iter_id, state.metrics_test_after, metadata=state.to_dict())
        if formula_tracker_ood_test_set is not None:
            formula_tracker_ood_test_set.update(iter_id, state.metrics_ood_test_after, metadata=state.to_dict())

    try:
        # ================================================================================
        # Non-tree flow
        # ================================================================================

        if not cfg.experiment.tree:
            patience = cfg.experiment.early_stop_patience

            iter_state = IterState.get_empty_state(iter_num=0)
            for iter_num in range(cfg.experiment.n_iters):
                print(f"Iteration {iter_num} of {cfg.experiment.n_iters}")

                iter_state = propose_and_prune_once(
                    current_terms=iter_state.terms_after,
                    history=iter_state.history,
                    prepared_dataset=prepared_dataset,
                    # ---
                    cfg=cfg,
                    data_bundle=data_bundle,
                    exp_logger=exp_logger,
                    seed=seed,
                    iter_num=iter_num,
                    node_id=None,
                    # --
                    azure_model_key=azure_model_key,
                    # ---
                    compute_profiler=compute_profiler,
                    feature_importance_text=feature_importance_text,
                )

                track(get_iter_or_node_id(cfg, iter_num, iter_state.node_id), iter_state)

                exp_logger.info(
                    "igsr_iter_finished",
                    seed=seed,
                    iter_num=iter_state.iter_num,
                    node_id=iter_state.node_id,
                    iter_result=iter_state.to_dict(),
                )

                best_result = formula_tracker_val_set.best_result(metric="mse")["metric_value"]

                if cfg.experiment.early_stop_enabled:
                    patience = check_early_stop(
                        metric_name="mse",
                        direction="min",
                        best_result=best_result,
                        iter_result=iter_state,
                        patience_left=patience,
                        patience=cfg.experiment.early_stop_patience,
                    )
                    if patience <= 0:
                        exp_logger.info("igsr_early_stop", seed=seed, iter_num=iter_num)
                        break

                if check_token_budget_and_maybe_break(
                    cfg=cfg, round=iter_num, compute_profiler=compute_profiler, seed=seed, logger=exp_logger
                ):
                    break
                if check_wallclock_budget_and_maybe_break(
                    cfg=cfg, round=iter_num, start_time=start_overall, seed=seed, logger=exp_logger
                ):
                    break

        # ================================================================================
        # Tree flow
        # ================================================================================

        if cfg.experiment.tree:

            class SRProblem(Problem[IterState, float]):
                def successors(self, s: IterState) -> Iterable[IterState]:
                    print(f"====== Generating successors of {s.node_id} ======")
                    for i in range(cfg.experiment.n_successors):
                        output = propose_and_prune_once(
                            current_terms=s.terms_after,
                            history=s.history,
                            prepared_dataset=prepared_dataset,
                            # ---
                            cfg=cfg,
                            data_bundle=data_bundle,
                            exp_logger=exp_logger,
                            seed=seed,
                            iter_num=None,
                            node_id=f"{s.node_id}_{i}",
                            # --
                            azure_model_key=azure_model_key,
                            # --
                            compute_profiler=compute_profiler,
                            feature_importance_text=feature_importance_text,
                        )
                        track(get_iter_or_node_id(cfg, None, output.node_id), output)
                        yield output

                # lower = better
                def evaluate(self, s: IterState) -> float:
                    return s.mse_after_total

                def is_terminal(self, s: IterState) -> bool:
                    return False

                # pretty labels for the visualiser
                def label(self, s: IterState) -> str:
                    return (
                        f"{s.node_id}" f"\nmse={s.mse_before:.3g}"
                        # f"\nr2={s.r2:.8f}"
                        # f"\nnrmse={s.nrmse:.8f}"
                    )

                def key(self, s: IterState) -> Hashable:
                    return str(s.node_id)

            problem = SRProblem()
            sess = SearchSession(
                problem,
                visualiser=None,
                early_stopping_enabled=cfg.experiment.early_stop_enabled,
                early_stopping_patience=cfg.experiment.early_stop_patience,
            )
            start = IterState.get_empty_state(node_id="node_0")
            best_val, best_state = sess.mcts(
                start,
                depth_limit=cfg.experiment.depth_limit,
                total_budget=cfg.experiment.total_budget,
                stop_on_terminal=True,
                seed=seed,
                # As successor is expensive. =========================================
                rollout_is_just_node_reward=cfg.experiment.rollout_is_just_node_reward,
                rollout_depth=cfg.experiment.rollout_depth,
                cache_successors=True,
                always_exit_on_terminal=True,
                # ====================================================================
                c=cfg.experiment.c,  # with default sqrt(2) value explores too much, lower it.
                # --- For token/wallclock early stopping:
                extras_dict={
                    "cfg": cfg,
                    "compute_profiler": compute_profiler,
                    "start_time": start_overall,
                    "seed": seed,
                    "logger": exp_logger,
                },
                # ---
            )

            exp_logger.info(
                "igsr_iter_finished",
                seed=seed,
                iter_num=best_state.iter_num,
                node_id=best_state.node_id,
                iter_result=best_state.to_dict(),
            )

    except Exception as e:
        len_history = len(formula_tracker_val_set.get_history(metric=None))
        if len_history >= cfg.experiment.allow_error_history_length:
            exp_logger.error("error_allow_result", len_history=len_history, seed=seed, error=e)
        else:
            exp_logger.error("error_raise_result", len_history=len_history, seed=seed, error=e)
            raise e

    elapsed_overall = time.time() - start_overall
    compute_profiler.accumulate("wall_clock_total", elapsed_overall)
    print(f">>> Compute profiler: {compute_profiler.as_dict()}")

    # The result is the iteration with the lowest validation MSE; the test split chooses nothing.
    best = formula_tracker_val_set.best_result(metric="mse")
    chosen = best["metadata"]
    registry = prepared_dataset.registry
    return {
        "seed": seed,
        "best_iter_id": best["iter_id"],
        "best_formula": chosen["equation_after"],
        "best_metric": best["metric_value"],
        "best_terms": list(chosen["terms_after"]),
        "best_test_metrics": chosen["metrics_test_after"],
        "best_model": fit_final_model(cfg, list(chosen["terms_after"]), prepared_dataset),
        "term_records": registry.records() if registry is not None else None,
        "term_failures": registry.failures if registry is not None else None,
        "proposal_stats": registry.stats.as_dict() if registry is not None else None,
        "history_validation": formula_tracker_val_set.get_history(metric=None),
        "history_test": formula_tracker_test_set.get_history(metric=None) if formula_tracker_test_set else None,
        "history_ood_test": (
            formula_tracker_ood_test_set.get_history(metric=None) if formula_tracker_ood_test_set is not None else None
        ),
        "compute_profiler": compute_profiler.as_dict(),
    }
