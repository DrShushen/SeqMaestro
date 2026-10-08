import json
import logging
from dataclasses import replace
from pathlib import Path

import dill
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate

from deft.budget import BudgetLimits, RunBudget
from deft.code_guard import CodeGuardPolicy, set_active_policy
from deft.data.dataset_plugins import check_column_names, instantiate_dataset_plugin
from deft.provenance import DegradedRunError, IntegrityPolicy, RunProvenance
from deft.targets import BINARY, checked_kind


def _format_generation_prompts(cfg, prompt_context: dict) -> None:
    if not prompt_context:
        return

    cfg.params_generation.dataset_info = cfg.params_generation.dataset_info.format(
        **prompt_context
    )

    reflection_templates = cfg.params_generation.get("list_instructions_reflection", [])
    cfg.params_generation.list_instructions_reflection = [
        prompt_template.format(**prompt_context)
        for prompt_template in reflection_templates
    ]


# What a previous run leaves behind. `.hydra/` and the log file are deliberately excluded:
# Hydra writes both before `run_experiment` is ever reached, so they are present on a first run
# too and would make every run look like a reuse.
_RUN_ARTIFACT_PATTERNS = ("*.dill", "*.pkl", "provenance.json", "token_usage.json")


def check_run_directory(log_dir, allow_reuse: bool = False) -> None:
    """Refuse to start in a directory that already holds a run's artifacts.

    `intermediate_tree.pkl` is rewritten at the start of every node, and the tree, provenance
    and token report all land in the same place. Two runs sharing a directory therefore
    interleave and silently corrupt each other's record -- and Hydra's timestamped directories
    resolve only to the second, so a scripted multi-seed sweep really can collide.

    Cheaper to refuse than to debug a sweep that quietly overwrote half its own results.
    """
    log_dir = Path(log_dir)
    existing = sorted(
        {
            path.name
            for pattern in _RUN_ARTIFACT_PATTERNS
            for path in log_dir.glob(pattern)
            if path.is_file()
        }
    )
    if not existing:
        return

    if allow_reuse:
        logging.warning(
            "Run directory %s already contains %s; overwriting because "
            "allow_run_dir_reuse=true.",
            log_dir,
            ", ".join(existing),
        )
        return

    raise FileExistsError(
        f"Run directory {log_dir} already contains artifacts from a previous run: "
        f"{', '.join(existing)}.\n"
        f"Two runs sharing a directory interleave and corrupt each other's record. Give this "
        f"run its own directory with `hydra.run.dir=<path>`, or pass "
        f"`allow_run_dir_reuse=true` to overwrite deliberately."
    )


#: The key in `params_generation` naming the kind of target its prompts are written for. It
#: configures no prompt, so it is taken out before the prompts' settings reach the tree.
PROMPTS_TARGET_KIND = "target_kind"


def check_prompts_fit_target(cfg, target_kind: str) -> None:
    """Refuse prompts written for another kind of target than the dataset's.

    A model asked to classify a continuous target, or to predict the value of a binary one, is
    told the wrong task, and the run would complete with nothing to show it. A prompt config that
    names no kind is binary, as every prompt DEFT shipped before regression was.
    """
    written_for = cfg.params_generation.get(PROMPTS_TARGET_KIND, BINARY)
    if written_for != target_kind:
        raise ValueError(
            f"The prompts (`params_generation`) are written for a {written_for} target, and the "
            f"dataset's target is {target_kind}. Select prompts written for it: "
            f"`params_generation=regression` for a continuous target, or a config of your own "
            f"that inherits it."
        )


def _load_training_bundle(cfg) -> dict:
    dataset_cfg = cfg.get("dataset")
    if dataset_cfg is None:
        raise ValueError("Missing required config block: `dataset`.")

    plugin = instantiate_dataset_plugin(dataset_cfg)
    bundle = plugin.load_for_deft()

    required_keys = ("X_train", "y_train", "tree_filename", "prompt_context")
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise KeyError(
            "Dataset plugin output is missing keys: "
            f"{missing}. Received keys: {list(bundle.keys())}"
        )

    # Applied here rather than inside any one plugin, so a custom plugin cannot skip it.
    # Before the language model is reached, so a rename costs nothing but a rerun.
    check_column_names(bundle["X_train"].columns)

    return bundle


def run_experiment(cfg) -> dict:
    logging.info("Running unified DEFT experiment.")

    bundle = _load_training_bundle(cfg)
    _format_generation_prompts(cfg, bundle["prompt_context"])

    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    tree_filename = bundle["tree_filename"]
    # A plugin that names no kind hands a binary target, as every plugin did before regression; the
    # tree refuses one that holds anything but 0 and 1.
    target_kind = checked_kind(bundle.get("target_kind", BINARY))

    min_samples_leaf_fraction = cfg.get("min_samples_leaf_fraction")
    if min_samples_leaf_fraction is None:
        raise ValueError("Missing required config key: `min_samples_leaf_fraction`.")
    if not (0 < float(min_samples_leaf_fraction) <= 1):
        raise ValueError(
            "`min_samples_leaf_fraction` must be in the interval (0, 1]. "
            f"Received: {min_samples_leaf_fraction}"
        )

    # What generated code is allowed to do. Run-level, established before any feature is
    # compiled. See `src/deft/conf/code_guard/strict.yaml`.
    set_active_policy(CodeGuardPolicy.from_config(cfg.get("code_guard")))

    min_samples_leaf = max(1, int(float(min_samples_leaf_fraction) * len(y_train)))
    logging.info(
        "Min samples leaf: %d (fraction=%.4f, n_samples=%d)",
        min_samples_leaf,
        float(min_samples_leaf_fraction),
        len(y_train),
    )

    feature_finder_cfg = cfg.get("feature_finder")
    if feature_finder_cfg is None:
        raise ValueError("Missing required config block: `feature_finder`.")

    # Only build an engine for a finder that uses one. `feature_finder=cart` is a
    # credential-free baseline, and requiring an API key it never calls would defeat the point.
    # The class is read without instantiating it, so the decision costs nothing.
    if getattr(get_class(feature_finder_cfg["_target_"]), "requires_llm_engine", True):
        # Only a finder that prompts reads the prompts, so only its run needs them to fit.
        check_prompts_fit_target(cfg, target_kind)
        llm_cfg = cfg.get("llm")
        if llm_cfg is None:
            raise ValueError("Missing required config block: `llm`.")
        llm_engine = instantiate(llm_cfg)
        feature_finder = instantiate(feature_finder_cfg, llm_engine=llm_engine)
    else:
        llm_engine = None
        feature_finder = instantiate(feature_finder_cfg)
        logging.info(
            "Feature finder %s needs no language model; none was created.",
            feature_finder_cfg["_target_"],
        )

    log_dir = HydraConfig.get()["runtime"]["output_dir"]
    logging.info(f"Logging dir: {log_dir}")
    # Before anything is spent on the language model.
    check_run_directory(log_dir, allow_reuse=bool(cfg.get("allow_run_dir_reuse", False)))

    splitting_criterion_cfg = cfg.get("splitting_criterion")
    if splitting_criterion_cfg is None:
        raise ValueError("Missing required config block: `splitting_criterion`.")
    splitting_criterion = instantiate(splitting_criterion_cfg)

    tree_cfg = cfg.get("tree")
    if tree_cfg is None:
        raise ValueError("Missing required config block: `tree`.")

    tree = instantiate(
        tree_cfg,
        feature_finder=feature_finder,
        splitting_criterion=splitting_criterion,
        min_samples_leaf=min_samples_leaf,
        folder_path=log_dir,
        target_kind=target_kind,
    )

    selector = instantiate(cfg.selector)

    # One recorder for the whole run, attached to each of the three places that needs to reach
    # it. It answers "how much of this tree is actually DEFT?" and enforces the policy in the
    # `integrity` config group. See `src/deft/provenance.py` and `src/deft/conf/integrity/strict.yaml`.
    integrity_policy = IntegrityPolicy.from_config(cfg.get("integrity"))
    if llm_engine is None and integrity_policy.enabled:
        # The policy's whole question is "did the language model contribute?". With a finder
        # that has no language model the answer is no by construction, so the checks would fail
        # every node. Skipping is honest; leaving them on would only be theatre.
        logging.info(
            "Integrity checks do not apply to a finder with no language model; skipping them. "
            "Provenance is still recorded."
        )
        integrity_policy = replace(integrity_policy, enabled=False)

    run_provenance = RunProvenance(policy=integrity_policy)
    if llm_engine is not None:
        llm_engine.provenance = run_provenance
    feature_finder.provenance = run_provenance
    tree.provenance = run_provenance

    # Token accounting is always on; limits are opt-in via the `budget` config group. When one
    # is reached the tree stops growing rather than failing. See `src/deft/conf/budget/unlimited.yaml`.
    run_budget = RunBudget(limits=BudgetLimits.from_config(cfg.get("budget")))
    if llm_engine is not None:
        llm_engine.budget = run_budget
    tree.budget = run_budget
    if run_budget.limits.any_set:
        logging.info("Run budget: %s", run_budget.limits)

    tree_path = Path(log_dir) / tree_filename
    provenance_path = Path(log_dir) / "provenance.json"
    token_usage_path = Path(log_dir) / "token_usage.json"

    def write_provenance() -> None:
        provenance_path.write_text(json.dumps(run_provenance.to_dict(), indent=2) + "\n")
        token_usage_path.write_text(json.dumps(run_budget.to_dict(), indent=2) + "\n")

    # The sequence column is a property of the data, not of the prompts, so it comes from the
    # plugin's bundle rather than from `params_generation`. `get_naive_features` needs it to
    # emit positional features against the user's own column name.
    sequence_column = bundle.get("sequence_column", "raw_sequence")
    # The same holds for what the columns mean and how a sequence is read by position.
    for key in ("sequence_column", "column_descriptions", "positional_window"):
        if key in cfg.params_generation:
            raise ValueError(
                f"`params_generation.{key}` is not a valid setting: it is a property of the "
                f"data, declared by the dataset as `dataset.{key}`. Move it there."
            )

    # Per-row fitting weights, when the dataset supplies them. Set after instantiation because the
    # config can name a column but cannot carry data, which is how `provenance` and `budget` arrive
    # too. When absent the tree falls back to `class_weight` and nothing changes.
    if bundle.get("sample_weight") is not None:
        tree.sample_weight = bundle["sample_weight"]
        logging.info("Fitting with per-row sample weights.")

    try:
        tree.fit(
            X_train,
            y_train,
            **{key: value for key, value in cfg.params_generation.items()
               if key != PROMPTS_TARGET_KIND},
            selector=selector,
            sequence_column=sequence_column,
            column_descriptions=bundle.get("column_descriptions") or {},
            positional_window=bundle.get("positional_window"),
        )
    except DegradedRunError:
        write_provenance()
        logging.error(
            "Run stopped: the language model did not contribute enough for this to be a DEFT "
            "result. Per-node detail is in %s. To see the numbers without failing, rerun with "
            "integrity=warn_only; src/deft/conf/integrity/strict.yaml explains the policy.",
            provenance_path,
        )
        raise
    except Exception:
        # Deliberately re-raised. Upstream logged and exited 0 here, which made a failed run
        # indistinguishable from a completed one unless you read the returned status.
        write_provenance()
        logging.exception("An error occurred during tree fitting:")
        raise
    finally:
        # The event loop the finder's requests ran on, with the connections it held.
        feature_finder.close()

    with open(tree_path, "wb") as f:
        dill.dump(tree, f)

    write_provenance()
    # Logged rather than printed, so the fitted tree is in `deft_tree.log` and survives a
    # detached run. Hydra's default logging already echoes INFO to the console, so printing as
    # well would show everything twice.
    logging.info("\n%s", tree.format_tree())
    logging.info("\n%s", run_provenance.format_report() + "\n\n" + run_budget.format_report())

    # Run-level checks come after the tree is written, so that a violation still leaves an
    # artifact to inspect.
    try:
        run_provenance.check_run()
    except DegradedRunError:
        write_provenance()
        logging.error(
            "Run completed but failed its integrity checks; see %s. The tree was still saved "
            "to %s for inspection, but should not be treated as a DEFT result.",
            provenance_path,
            tree_path,
        )
        raise

    result = {
        "status": "fit_complete",
        "tree_path": str(tree_path),
        "provenance_path": str(provenance_path),
        "token_usage_path": str(token_usage_path),
        "integrity": run_provenance.summary(),
        "budget_stopped": tree.budget_stopped,
        "budget_stopped_reason": tree.budget_stopped_reason,
        "total_tokens": run_budget.spend.total_tokens,
    }
    logging.info(f"Run result: {result}")
    return result
