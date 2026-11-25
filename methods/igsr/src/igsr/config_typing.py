"""
This file contains the config classes for the discovery experiments.
It's used for better IDE experience and can be used to validate the config schema if needed.

This will validate the config schema, but it isn't necessary. It doesn't support some data types like Literal.
```
cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
```
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Union


@dataclass
class LoggingConfig:
    logger_name: str
    mode: str
    log_file: str


@dataclass
class LLMConfig:
    kind: str
    model_type: str
    model_version: str
    deployment: str
    api_key: str
    # Azure-only (the OpenAI and Claude configs omit these):
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    # The reasoning effort every call states, where the config gives one (`completion_settings`).
    reasoning_effort: Optional[str] = None
    # Retries after a failed call (`retry_settings`) and the seconds a reply is waited for
    # (`completion_settings`), where the config gives them.
    num_retries: Optional[int] = None
    request_timeout: Optional[int] = None


@dataclass
class FunctionTermsConfig:
    """The function form's settings (`igsr.method.function_terms.FunctionTermSettings`)."""

    max_undefined_share: float
    repair_rounds: int
    n_examples: int
    max_example_length: Optional[int]
    timeout_seconds: int
    code_guard: Optional[dict] = None


@dataclass
class ExperimentConfig:
    kind: str
    n_seeds: int
    max_workers_seeds: Union[int, Literal["auto"]]
    max_retry_seed: int
    use_timeoutable_parallelism: bool
    process_timeout_seconds: Optional[int]
    stop_loss_threshold: Optional[float]
    use_ops_subset: bool
    n_iters: int
    show_formula_examples: bool
    include_grad_info: bool
    top_k_history: int
    max_agent_steps: int
    history_mode: Literal["top_k", "recent_n", "none"]
    recent_n_history: Optional[int]
    loss: Literal["mse", "bce"]
    dataset_preview_mode: Literal["to_dict", "to_string", "to_records", "none"]
    dataset_preview_head_n: int
    use_metrics_classification: List[str]
    use_metrics_regression: List[str]
    use_metrics_sets: List[Literal["train", "val", "test"]]
    best_result_metric: str
    best_result_set: Literal["train", "val", "test"]
    # PySR parameters:
    pysr_maxsize: int
    pysr_niterations: int
    pysr_batching: bool
    pysr_batch_size: int
    # IGSR parameters:
    tree: bool
    total_budget: int
    depth_limit: int
    n_successors: int
    c: float
    # Prompting:
    terms_per_round: int
    first_round_n_candidates: int
    keep_n_terms: Optional[int]
    # Feedback
    influence_feedback: bool
    history_enabled: bool
    # Early stopping:
    early_stop_enabled: bool
    early_stop_patience: int
    # Optimization:
    optimization_method: Literal["linear", "ridge", "lasso"]
    # Influence computation options
    refit_aware: bool
    refit_aware_efficient: bool
    # Zero shot:
    zero_shot_mode: Literal["zero_shot", "zero_shot_optimization"]
    zero_shot_n_terms: int
    # SINDy parameters:
    sindy_degree: int = 2  # Polynomial degree for the SINDy library (2 in D3).
    sindy_alpha: float = 0.5  # STLSQ regularization parameter alpha (0.5 in D3).
    sindy_threshold: float = 0.02  # Default sparsity threshold (0.02 in D3; COVID handled separately).
    sindy_extended_library: bool = False  # If True, add trig/exp/log/sqrt/abs functions to the SINDy library.
    # CAAFE parameters:
    caafe_iterations: int = 10
    caafe_temperature: float = 0.5
    caafe_max_tokens: int = 500
    caafe_n_splits: int = 10
    caafe_n_repeats: int = 2
    caafe_eval_regressor: str = "ridge"
    caafe_sandbox_enabled: bool = True
    caafe_verbose: bool = True
    caafe_max_features: Optional[int] = None
    # Term language (igsr): numpy expressions, or Python functions of the rows.
    term_language: Literal["expression", "function"] = "expression"
    function_terms: Optional[FunctionTermsConfig] = None
    # Raw feature importance in prompts (igsr):
    show_feature_importance: bool = False
    feature_importance_max_features: Optional[int] = 20
    # LIES parameters:
    lies_n_trials: int = 3
    lies_log_transform: bool = True
    lies_allow_no_log_transform_fallback: bool = True
    lies_hidden_layers: Optional[int] = None
    lies_weak_admm_epochs: int = 20
    lies_weak_admm_rho: float = 0.9
    lies_weak_admm_l1_lambda: float = 0.05
    lies_weak_admm_alpha: float = 5e-4
    lies_strong_admm_epochs: int = 30
    lies_strong_admm_rho: float = 0.005
    lies_learning_rate: float = 0.015
    lies_batch_size: int = 128
    lies_lr_decay: float = 0.95
    lies_gradient_prune_threshold: float = 0.01
    lies_rounding_threshold: float = 0.1
    lies_device: Optional[int] = None
    # SyMANTIC parameters:
    symantic_operators: Optional[List[str]] = None
    symantic_n_expansion: Optional[int] = None
    symantic_n_term: int = 3
    symantic_sis_features: int = 20
    symantic_metrics_rmse: float = 0.06
    symantic_metrics_r2: float = 0.995
    symantic_max_expansion_depth: Optional[int] = 2
    symantic_initial_screening: Optional[str] = None
    symantic_initial_screening_quantile: float = 0.75
    symantic_device: str = "cpu"
    symantic_verbose: bool = False


@dataclass
class DatasetConfig:
    kind: str
    name: str
    task: Literal["regression", "classification"]
    target_columns: List[str]
    # Cancer dataset parameters:
    num_patients: int
    env_name: str
    random_additional_features: bool
    # Multi-output flag:
    multiple_output_targets: bool
    min_x: float
    max_x: float
    steps: int
    requires_grad: bool
    # Seed:
    seed: Optional[int]
    # Toy within-term dataset (optional params):
    equation_name: Optional[str] = None
    j: Optional[int] = None
    n_samples: Optional[int] = None
    n_irrelevant: Optional[int] = None
    extra_text: Optional[str] = None
    # LLM-SRBench related parameters:
    llmsrbench_datamodule: Optional[str] = None
    llmsrbench_problem_id: Optional[str] = None


@dataclass
class MainConfig:
    logging: LoggingConfig
    llm: LLMConfig
    experiment: ExperimentConfig
    dataset: DatasetConfig
