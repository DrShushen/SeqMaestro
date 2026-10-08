#!/usr/bin/env python3
"""Run DEFT and hand the result back as an arm bundle. **Runs in `SeqMaestro_Method_deft`.**

This is the far side of the environment boundary. It depends on `deft`, through DEFT's console
script and the trees DEFT saves, and never imports `seqmaestro`. The two sides exchange files, so
their dependency sets can disagree about every package. This file and `deft_cart.py` are the only
places that know both DEFT's object model and SeqMaestro's handover format. They live here and not
in `methods/deft/` because the handover format belongs to SeqMaestro, and the vendored library must
stand alone without knowing about SeqMaestro.

    $DEFT experiments/adapters/deft_bundle.py fit \\
        --train fit.csv --val val.csv --effect effect.csv --out-bundle <dir> \\
        --target-column <column> --target-kind <binary|continuous> --feature-finder default \\
        --task <dataset>
    $DEFT experiments/adapters/deft_bundle.py predict \\
        --bundle <dir> --rows rows.csv --id-column <column> --out scores.npz

`fit` trains and converts in one call, because splitting the two would cross the boundary a
second time and leave the caller to find hydra's output directory. It is never given the test
split. The bundle names the tree it saved in `fitted_model`.

`predict` serves the reveal. It loads that tree and predicts the rows it is handed, which carry an
identifier column and no labels. The scores go back with the identifiers, so the other side can
check the row order before it scores anything, and those rows' labels never cross the boundary.

**`--task` names the dataset, and its prompt config is held on this side of the boundary.** That is
a hand-written config in `deft_conf/params_generation/` when one exists, and otherwise a config
generated from the dataset's description in `experiments/datasets/`. DEFT's prompts are templated
on the task, and a wrong description fails silently: the model is told in detail that it is looking
at something it is not, and the run completes. Which dataset is being run is SeqMaestro's decision,
so SeqMaestro holds the description, and DEFT ships only a task-neutral default. See
`deft_conf/README.md`.

**Row order is the contract.** Predictions are returned in the order the rows were given, and the
labels come back alongside them so the other side can check the order. A reordered bundle would
produce a plausible but meaningless score, and nothing downstream could detect it.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The bundle contract this adapter writes. `seqmaestro.harness.bundle` holds the definition. The
# constants are copied here because importing them would create the cross-environment dependency
# that exchanging files avoids, and `read_bundle` rejects a bundle whose versions do not match.
BUNDLE_VERSION = 3
FEATURE_SCHEMA_VERSION = 1

#: Where `fit` has hydra run DEFT, inside the bundle, and so where the fitted tree ends up.
WORK_DIR = "deft_run"

#: What a feature's importance number means, per finder. Keyed on DEFT's config-group names: `cart`,
#: and `default` for the language-model finder. A key that is not one of those names never matches,
#: and the finder silently falls through to `deft_split_score` in `fit`.
IMPORTANCE_KIND = {"cart": "impurity_decrease", "default": "impurity_decrease"}

#: Hand-written prompt configs, for when the prompt wording is itself part of what is being
#: compared. They go on hydra's search path through the stock `--config-dir` flag, so they also work
#: when `deft-tree` is run by hand.
TASK_CONFIG_DIR = Path(__file__).resolve().parent / "deft_conf"
TASK_GROUP = "params_generation"
#: The config group for what DEFT is told about the columns it is given, generated per cell from
#: `columns.json` (`dataset_config`) under this name when there is anything to tell.
DATASET_GROUP, HANDED_DATASET = "dataset", "handed"

#: The dataset descriptions, one per dataset, which the proposer and the judge also read.
DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"

#: The kinds of target, in the words `seqmaestro.target` and DEFT's `deft.targets` both use, and
#: the prompt config each kind's generated task config inherits from DEFT.
BINARY, CONTINUOUS = "binary", "continuous"
PROMPTS_FOR = {BINARY: "default", CONTINUOUS: "regression"}
#: The splitting criterion a continuous target is grown with. A binary target keeps DEFT's
#: configured criterion, Gini unless the cell's overrides name another.
CONTINUOUS_CRITERION = "squared_error"


def generate_task_config(description: dict, name: str, into: Path, target_kind: str) -> Path:
    """Write DEFT's prompt config from the dataset description, and return the config directory.

    DEFT's templates interpolate only `{target_name}` and `{dataset_info}`, so both can be filled
    from the description every other consumer reads, and a new dataset needs no DEFT-specific file.
    The config inherits the prompts DEFT writes for the target's kind (`PROMPTS_FOR`).

    The config is written into the run directory and never committed. There is then nothing to keep
    in step with the description, and the frozen run records which prompts produced it.
    """
    layout = description.get("sequence_layout")
    dataset_info = description["task"] + (f" {layout}" if layout else "")

    group = into / TASK_GROUP
    group.mkdir(parents=True, exist_ok=True)
    (group / f"{name}.yaml").write_text(
        "# Generated from the dataset description; edit that, not this.\n"
        f"defaults:\n  - {PROMPTS_FOR[target_kind]}\n  - _self_\n\n"
        f"target_name: {json.dumps(description.get('target_name') or name)}\n"
        f"dataset_info: {json.dumps(dataset_info)}\n"
    )
    return into


def resolve_task(task: str | None, config_dir: Path, dataset_dir: Path, generate_into: Path,
                 target_kind: str):
    """Turn `--task <name>` into hydra overrides, and return them with a note of their source.

    A hand-written config takes precedence when one exists. It is used where the prompt wording is
    part of the comparison, as when reproducing published numbers. Its presence alone selects it,
    with no flag, because a forgotten flag would silently swap the paper's prompts for generated
    ones and the run would still complete normally. Which config was used is printed and recorded.

    The name is resolved here because hydra would report a bad name as a composition error from
    inside the subprocess, after the run has started, and for an LLM arm after it has begun spending.
    """
    if task is None:
        return [], "deft default (task-neutral)"

    handwritten = config_dir / TASK_GROUP / f"{task}.yaml"
    if handwritten.exists():
        return [f"--config-dir={config_dir}", f"{TASK_GROUP}={task}"], f"hand-written {handwritten}"

    description_path = dataset_dir / f"{task}.json"
    if not description_path.exists():
        available = sorted(
            {p.stem for p in dataset_dir.glob("*.json")}
            | {p.stem for p in (config_dir / TASK_GROUP).glob("*.yaml")}
        )
        raise SystemExit(
            f"No dataset description {task!r} in {dataset_dir} and no hand-written config in "
            f"{config_dir / TASK_GROUP}. Available: {', '.join(available) or '(none)'}. "
            f"See {dataset_dir / 'README.md'}."
        )

    description = json.loads(description_path.read_text())
    generated = generate_task_config(description, task, generate_into, target_kind)
    return [f"--config-dir={generated}", f"{TASK_GROUP}={task}"], f"generated from {description_path}"


def train(args, work_dir: Path) -> Path:
    """Run DEFT through its console script and return the path to the fitted tree.

    DEFT runs as a subprocess because its `run_experiment` cannot simply be imported and called. It
    reads `HydraConfig.get()`, which exists only inside hydra's runtime, and composing a config by
    hand would mean reproducing that setup and repairing it on every hydra upgrade. An in-process
    call would only have made the output easy to find, and `hydra.run.dir` pins it anyway.
    """
    if getattr(args, "reuse_tree", False):
        existing = sorted(work_dir.rglob("*.dill"))
        if len(existing) != 1:
            raise RuntimeError(
                f"--reuse-tree expects exactly one .dill already under {work_dir}, "
                f"found {[str(t) for t in existing]}."
            )
        print(f"reusing the fitted tree at {existing[0]}", flush=True)
        return existing[0]

    generated = work_dir.parent / "deft_conf"
    task_overrides, task_source = resolve_task(
        args.task, Path(args.task_config_dir), Path(args.dataset_dir), generate_into=generated,
        target_kind=args.target_kind,
    )
    print(f"prompts: {task_source}", flush=True)
    told = dataset_config(Path(args.train).parent)
    if told:
        task_overrides = [*on_one_search_path(task_overrides, generated),
                          f"{DATASET_GROUP}={HANDED_DATASET}"]
        write_dataset_config(told, generated)
        print(f"columns: {', '.join(sorted(told))} from {COLUMNS_MANIFEST}", flush=True)

    overrides = [
        "--config-name=config",
        *task_overrides,
        f"hydra.run.dir={work_dir}",
        f"dataset.csv_path={Path(args.train).resolve()}",
        f"dataset.target_column={args.target_column}",
        f"dataset.target_kind={args.target_kind}",
        *criterion_override(args.target_kind, args.config_override),
        f"dataset.weight_column={args.weight_column}" if args.weight_column
        else "dataset.weight_column=null",
        f"feature_finder={args.feature_finder}",
        # With no sequence column the inputs are a written feature matrix, and the naive generator
        # emits one identity feature per numeric column instead of decoding a sequence.
        f"dataset.sequence_column={args.sequence_column}" if args.sequence_column
        else "dataset.sequence_column=null",
        *args.config_override,
    ]
    # Looked up beside this interpreter, because the caller runs this script by absolute path
    # without activating the environment, so `deft-tree` is not on PATH.
    console_script = Path(sys.executable).parent / "deft-tree"
    if not console_script.exists():
        raise RuntimeError(
            f"{console_script} not found; this adapter must be run by the interpreter of the "
            f"environment DEFT is installed in. See ENVIRONMENTS.md."
        )

    print(f"$ {console_script.name} {' '.join(overrides)}", flush=True)
    completed = subprocess.run([str(console_script), *overrides], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"deft-tree exited {completed.returncode}; see the output above.")

    trees = sorted(work_dir.rglob("*.dill"))
    if len(trees) != 1:
        raise RuntimeError(
            f"expected exactly one .dill under {work_dir}, found {[str(t) for t in trees]}."
        )
    return trees[0]


def criterion_override(target_kind: str, overrides: list[str]) -> list[str]:
    """The splitting criterion to name for this kind of target: the squared error for a continuous
    target, unless the cell's overrides name a criterion, which DEFT then checks against the kind.
    A binary target keeps DEFT's configured one."""
    named = any(item.partition("=")[0].strip() == "splitting_criterion" for item in overrides)
    if target_kind == CONTINUOUS and not named:
        return [f"splitting_criterion={CONTINUOUS_CRITERION}"]
    return []


#: The method that produced these features. `kind` says what a feature computes and `origin` says
#: which method produced it (see `seqmaestro.features.base.FeatureRecord`). DEFT's internal
#: `llm`/`naive` label belongs in `kind` and must not be written here, because `origin` would then
#: mean the method on the SeqMaestro arms and the way a feature was proposed on this one. The two
#: meanings collide on SeqMaestro's own proposer, whose features are `seqmaestro` by origin and
#: language-model generated by kind.
ORIGIN = "deft"

#: DEFT's per-candidate origin, mapped onto the shared `kind` vocabulary. `llm_generated` is the
#: same kind whichever method's language model wrote the feature, so language-model features can be
#: counted over `kind` for any method, without reading DEFT's private labels.
KIND_BY_DEFT_ORIGIN = {
    "llm": "llm_generated",
    # A naive candidate DEFT derived from the sequence is positional, and `naive_recipe` sets that
    # kind. Any other naive candidate is a column SeqMaestro supplied, whose kind is recorded in the
    # pool manifest, so this fallback says `unknown` instead of guessing.
    "naive": "unknown",
    "unknown": "unknown",
}

#: How a feature entered the arm's answer. DEFT's naive candidates are the pool it was seeded with,
#: so the tree selected them. Anything its language model wrote during the fit, it generated.
PROVENANCE_BY_DEFT_ORIGIN = {"llm": "generated", "naive": "selected", "unknown": None}

#: DEFT's name for one naive positional candidate, e.g. `pos_50_is_A`. SeqMaestro names the same
#: feature `pos_50_A`. The names need not agree, because `definition_digest` does not read them, but
#: the recipes must. `definition_digest` hashes the normalised `source` when a record has one and
#: `{kind, params}` when it does not, so a record carrying DEFT's code and one carrying SeqMaestro's
#: params would take different branches and never match, even for the same function. `naive_recipe`
#: therefore emits `{kind, params}` and no source, which puts both records on the same branch.
DEFT_POSITIONAL_NAME = re.compile(r"^pos_(\d+)_is_(.+)$")

#: What DEFT calls a candidate built from a column it was handed: `_covariate_features` names it
#: `value_of_<column>`. The prefix is stripped to recover the column, so a supplied feature can be
#: looked up in the pool manifest and reported under its recipe instead of an unrecognised name.
DEFT_COVARIATE_PREFIX = "value_of_"

#: Written beside the inputs by the Core side (`COLUMNS_MANIFEST` in
#: `seqmaestro/pipeline/vendored/handover.py`): the sequence column with its positional window, and
#: each other column DEFT is given, with its recipe and description,
#: `{"sequence": {"column", "window"}, "columns": {name: {kind, params, description}}}`. The name
#: and shape are stated on both sides, since the two sides exchange files and import nothing of
#: each other.
COLUMNS_MANIFEST = "columns.json"


def read_pool(inputs_dir: Path) -> dict:
    """Each column DEFT was given besides the sequence, `{name: {kind, params, description}}`, or
    empty when it was given the sequence alone."""
    return json.loads((Path(inputs_dir) / COLUMNS_MANIFEST).read_text())["columns"]


def read_window(inputs_dir: Path) -> dict | None:
    """The positional window DEFT read the sequence in, or None where it read one length."""
    return json.loads((Path(inputs_dir) / COLUMNS_MANIFEST).read_text())["sequence"].get("window")


def dataset_config(inputs_dir: Path) -> dict:
    """What DEFT is told about the columns it is given (`dataset.column_descriptions` and
    `dataset.positional_window`, settings of DEFT's config), or empty where there is nothing to
    tell."""
    told = {}
    descriptions = {name: entry["description"] for name, entry in read_pool(inputs_dir).items()
                    if entry.get("description")}
    if descriptions:
        told["column_descriptions"] = descriptions
    window = read_window(inputs_dir)
    if window is not None:
        told["positional_window"] = window
    return told


def write_dataset_config(told: dict, into: Path) -> None:
    """DEFT's dataset config for this cell: the generic CSV plugin, told `told`. Written into the
    run, like the prompts, so the frozen run records what DEFT was told about its columns."""
    group = into / DATASET_GROUP
    group.mkdir(parents=True, exist_ok=True)
    # JSON for each value, which YAML reads as it is, so no description needs YAML quoting.
    (group / f"{HANDED_DATASET}.yaml").write_text(
        f"# Generated from the cell's {COLUMNS_MANIFEST}: what DEFT is told about its columns.\n"
        "defaults:\n  - generic_csv\n  - _self_\n\n"
        + "".join(f"{key}: {json.dumps(value)}\n" for key, value in told.items()))


def on_one_search_path(task_overrides: list[str], generated: Path) -> list[str]:
    """`task_overrides` with its config directory moved to `generated`, since hydra reads one
    `--config-dir` and the dataset config is generated there. A hand-written prompt config is
    copied in beside it, unchanged, so the frozen run holds both."""
    flag = f"--config-dir={generated}"
    if not task_overrides:
        return [flag]
    config_dir = Path(task_overrides[0].removeprefix("--config-dir="))
    if config_dir.resolve() != generated.resolve():
        selection = task_overrides[1].split("=", 1)[1]
        (generated / TASK_GROUP).mkdir(parents=True, exist_ok=True)
        shutil.copy(config_dir / TASK_GROUP / f"{selection}.yaml",
                    generated / TASK_GROUP / f"{selection}.yaml")
    return [flag, *task_overrides[1:]]


def naive_recipe(name: str, pool: dict | None = None) -> dict | None:
    """`{kind, params, source}` for a naive candidate, or None when it cannot be recognised.

    There are two sources. **Positional** candidates DEFT derives from the sequence itself, and
    their recipe is read from DEFT's naming convention. In a positional window the position is the
    window's, as it is in SeqMaestro's positional features. **Handed** columns, a covariate or a
    column we computed, come back as `value_of_<column>`, and their recipe and description are the
    ones the manifest recorded when the column was written. The manifest records what was built, so
    nothing here parses a *k*-mer's or a generated feature's name.
    """
    matched = DEFT_POSITIONAL_NAME.match(name)
    if matched:
        return {"kind": "positional", "source": None,
                "params": {"position": int(matched.group(1)), "symbol": matched.group(2)}}
    if pool and name.startswith(DEFT_COVARIATE_PREFIX):
        recorded = pool.get(name[len(DEFT_COVARIATE_PREFIX):])
        if recorded:
            return {"kind": recorded["kind"], "params": recorded["params"], "source": None,
                    "description": recorded.get("description")}
    return None


#: What `node_statistics` records on either side of a split, by the tree's kind of target: the share
#: of the positive class, or the target's mean. The keys are `<statistic>_left` and
#: `<statistic>_right`, and `assign_directions` builds a direction from them.
SIDE_STATISTIC = {BINARY: "rate", CONTINUOUS: "mean"}


def target_kind_of(tree) -> str:
    """The kind of target a fitted tree was grown for, which DEFT's tree records."""
    return tree.target_kind


def node_statistics(tree, X: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
    """Coverage and impurity decrease at every internal node, keyed by node path.

    DEFT stores neither: `Node` holds the split and its children, and `store_node_artifacts` keeps
    the feature finder's logging, with no node counts. Both are recomputed here by routing the rows
    the tree was fitted on back down it.

    **The criterion and the weighting are read from the tree.** `tree.splitting_criterion` is the
    configured criterion, and `tree.class_weight_` is the weighting it was actually given: a per-row
    `Series` when the split carried weights (these take precedence over `balanced`), and a
    class-to-weight mapping otherwise. Rebuilding either from the config would be a second source
    of truth, and on a weighted split it would make every node's statistics wrong.

    **The offset cancels.** Calling the criterion with an empty right-hand side yields
    `offset - I(node)`, which recovers the node's impurity. Subtracting that from the split's
    value gives the impurity decrease for whichever criterion the tree was configured with, without
    knowing the offset or reimplementing the impurity.
    """
    criterion = tree.splitting_criterion
    weights = getattr(tree, "class_weight_", None)
    kind = target_kind_of(tree)
    statistic = SIDE_STATISTIC[kind]
    if kind == BINARY:
        # Which label counts as positive, taken from the data rather than assumed to be 1: the rates
        # below and the harness's marginal effect have to agree about which class they point at.
        positive = np.max(np.unique(np.asarray(y)))

        def side(labels) -> float | None:
            return _positive_rate(labels, positive)
    else:
        side = _mean
    stats: dict[str, dict] = {}

    def value(left, right) -> float:
        return float(criterion(left, right, weights))

    def walk(node, depth: int, path: str, rows: pd.Index) -> None:
        if node is None or getattr(node, "feature", None) is None:
            return
        if node.left is None and node.right is None:
            return
        feature = node.feature
        if feature.fn is None:
            raise RuntimeError(
                f"node {path or '<root>'!r} splits on {feature.name!r}, which carries no callable, "
                f"so the rows reaching it cannot be followed. Every node below it would be counted "
                f"against the wrong rows."
            )
        node_y = y.loc[rows]
        threshold = float(feature.threshold) if feature.threshold is not None else 0.0
        values = np.asarray(feature.fn(X.loc[rows].copy()), dtype=float).ravel()
        # `<=` goes left, matching DEFT's split search. NaN compares False and so goes right,
        # which is also what the search does.
        left_rows, right_rows = rows[values <= threshold], rows[~(values <= threshold)]

        split_value = value(y.loc[left_rows], y.loc[right_rows])
        undivided = value(node_y, node_y.iloc[:0])
        stats[path] = {
            "depth": depth,
            "node_path": path,
            "threshold": threshold,
            "n_samples": len(rows),
            "impurity_decrease": split_value - undivided,
            "deft_split_score": float(feature.score) if feature.score is not None else None,
            "recomputed_split_score": split_value,
            # The share of the positive class, or the target's mean, on each side, which says which
            # way the split runs and is what `assign_directions` builds a direction from. Both are
            # unweighted: a class weighting exists to stop the abundant class dominating the
            # criterion, and applying any weighting here would describe the fitting weights instead
            # of the rows.
            f"{statistic}_left": side(y.loc[left_rows]),
            f"{statistic}_right": side(y.loc[right_rows]),
        }
        walk(node.left, depth + 1, path + "L", left_rows)
        walk(node.right, depth + 1, path + "R", right_rows)

    walk(tree.root, 0, "", X.index)
    return stats


#: `seqmaestro.harness.direction`'s vocabulary and its monotonicity threshold, restated because this
#: file runs where `seqmaestro` cannot be imported (the feature record is written by hand for the
#: same reason). `tests/test_deft_adapter.py` loads this file in `SeqMaestro_Core` and checks it
#: against the shared implementation, so any drift between the two fails a test.
HIGHER_IN_POSITIVE = "higher_in_positive"
HIGHER_IN_NEGATIVE = "higher_in_negative"
INCREASES_TARGET = "increases_target"
DECREASES_TARGET = "decreases_target"
NON_MONOTONE = "non_monotone"
MONOTONICITY_FLOOR = 0.5
#: Each kind's two words, the rising one first, as `from_increments` takes them.
WORDS = {BINARY: (HIGHER_IN_POSITIVE, HIGHER_IN_NEGATIVE),
         CONTINUOUS: (INCREASES_TARGET, DECREASES_TARGET)}

#: What `direction` on these records was read off, for the file header. In `DIRECTION_KINDS`.
DIRECTION_KIND = {BINARY: "tree_split_rates", CONTINUOUS: "tree_split_means"}


def direction_from_increments(increments, words: tuple[str, str]) -> str | None:
    """A direction from signed steps: a verbatim copy of `seqmaestro.harness.direction.from_increments`."""
    increments = np.asarray(increments, dtype=float).ravel()
    increments = increments[np.isfinite(increments)]
    if increments.size == 0:
        return None

    net = float(increments.sum())
    travelled = float(np.abs(increments).sum())
    if travelled == 0:
        return None
    if abs(net) / travelled < MONOTONICITY_FLOOR:
        return NON_MONOTONE
    return words[0] if net > 0 else words[1]


def _positive_rate(labels, positive) -> float | None:
    """Share of `labels` equal to `positive`, or None where no rows reached that side."""
    labels = np.asarray(labels)
    return float((labels == positive).mean()) if labels.size else None


def _mean(values) -> float | None:
    """The mean of `values`, or None where no rows reached that side."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else None


def assign_directions(records: list, target_kind: str) -> None:
    """Set each record's direction from the splits the tree makes on it.

    A tree has no coefficient or shape function, so the direction comes from what each split does:
    the rows above the threshold carry either a higher or a lower share of the positive class, or
    a higher or lower mean of a continuous target (`SIDE_STATISTIC`). Each node contributes one
    signed step, weighted by the rows reaching it, so a split near the root counts for more than a
    conditional one deep in a branch.

    **A feature can point different ways at different nodes**, and a single sign would hide that.
    A feature the tree uses one way at the root and the other way inside a branch comes out
    `non_monotone`, by the same rule that gives every other arm's curve its direction.

    Left is `<= threshold` and right is above it, as `node_statistics` routes the rows, so a
    positive step means higher values of the feature go with the positive class, or with a higher
    target.
    """
    left, right = (f"{SIDE_STATISTIC[target_kind]}_{side}" for side in ("left", "right"))
    for record in records:
        steps = [
            node["n_samples"] * (node[right] - node[left])
            for node in record["extra"]["nodes"]
            if node.get(left) is not None and node.get(right) is not None
        ]
        record["direction"] = (direction_from_increments(steps, WORDS[target_kind]) if steps
                               else None)


def check_node_statistics(stats: dict[str, dict], tolerance: float = 1e-6) -> None:
    """Refuse statistics whose recomputed split score disagrees with the one DEFT recorded.

    The two are the same quantity by definition, so a mismatch means the rows were routed wrongly,
    for example by an inverted threshold, the wrong weighting or a misaligned index. Every
    importance built on these statistics would then be wrong and still look plausible, so this
    raises instead of warning.
    """
    for path, node in sorted(stats.items()):
        recorded = node["deft_split_score"]
        if recorded is None or not np.isfinite(recorded):
            continue
        if abs(node["recomputed_split_score"] - recorded) > tolerance:
            raise RuntimeError(
                f"node {path or '<root>'!r}: DEFT recorded a split score of {recorded!r} and "
                f"routing the training rows gives {node['recomputed_split_score']!r}. The rows are "
                f"not reaching the nodes DEFT sent them to, so no importance built on this is "
                f"trustworthy. Check the threshold comparison, `tree.class_weight_`, and that the "
                f"training frame is the one the tree was fitted on, in its original row order."
            )


def collect_features(tree, pool: dict | None = None, stats: dict[str, dict] | None = None) -> list:
    """Every distinct feature the fitted tree splits on, with each node it splits at.

    A tree names a feature at each internal node, so the features it splits on are its selection.
    One feature can split at several nodes, and every occurrence is kept because importance sums
    over them. Keeping only the best-scoring occurrence would lose the others and bias toward the
    smallest, deepest node.

    `stats` comes from `node_statistics` and carries the node counts the tree does not store.
    Without it the records still describe the selection, and `importance` is left `None`, because
    any number put there would not mean what the field says.
    """
    found: dict[str, dict] = {}

    def walk(node, depth: int, path: str):
        if node is None or getattr(node, "feature", None) is None:
            return
        # Only a node with children is a split. A leaf can still carry a stale `feature` from the
        # search that decided not to split there, and collecting it would invent a selected
        # feature the tree never uses, unnamed and of unknown origin.
        if node.left is None and node.right is None:
            return
        feature = node.feature
        name = feature.name or f"unnamed_depth{depth}"
        measured = (stats or {}).get(path) or {}
        occurrence = {
            "depth": depth,
            "node_path": path,
            "threshold": _maybe_float(feature.threshold),
            "deft_split_score": _maybe_float(feature.score),
        }
        # Present only when the rows were routed, so a record never implies a coverage it does not
        # have. `n_samples` is what makes one node's decrease comparable with another's, and the two
        # sides' statistics are what `assign_directions` reads.
        for key in ("n_samples", "impurity_decrease", *(f"{statistic}_{side}"
                    for statistic in SIDE_STATISTIC.values() for side in ("left", "right"))):
            if key in measured:
                occurrence[key] = measured[key]

        if name not in found:
            # Where the recipe can be recovered it replaces DEFT's rendering, so a feature that
            # SeqMaestro could also have supplied hashes the same on both sides. Everything that
            # describes the node, as opposed to the feature, goes in `extra["nodes"]`, one entry per
            # occurrence. The threshold is such a property: every naive split carries the same 0.5,
            # so putting it in `params` would make them all hash alike.
            recipe = naive_recipe(name, pool) if feature.origin == "naive" else None
            # A handed column is described as the dataset describes it, not by DEFT's template.
            described = (recipe or {}).get("description") or feature.description or ""
            repair = ({"repaired": True} if getattr(feature, "repaired", False) else {}) | (
                {"renamed_from": feature.renamed_from}
                if getattr(feature, "renamed_from", None) else {}) | (
                {"name_as_proposed": feature.name_as_proposed}
                if getattr(feature, "name_as_proposed", None) else {})
            found[name] = {
                "name": name,
                "description": described,
                "kind": recipe["kind"] if recipe else KIND_BY_DEFT_ORIGIN.get(feature.origin, "unknown"),
                "params": recipe["params"] if recipe else {},
                "source": recipe["source"] if recipe else feature.string,
                "rationale": feature.rationale,
                "origin": ORIGIN,
                "importance": None,
                # Set by `assign_directions` once every node this feature splits at is known,
                # because the tree may use it in opposite directions at different nodes.
                "direction": None,
                "provenance": PROVENANCE_BY_DEFT_ORIGIN.get(feature.origin),
                # `deft_origin` is kept verbatim as well as mapped, because `extra` holds what one
                # method knows and others do not, and DEFT's label is such a fact.
                "extra": {"deft_origin": feature.origin, **repair, "nodes": []},
            }
        found[name]["extra"]["nodes"].append(occurrence)
        walk(node.left, depth + 1, path + "L")
        walk(node.right, depth + 1, path + "R")

    walk(tree.root, 0, "")
    records = list(found.values())
    assign_importances(records)
    assign_directions(records, target_kind_of(tree))
    return records


def assign_importances(records: list) -> None:
    """Set each record's importance to its share of the tree's total impurity reduction.

    This is the standard tree importance that `scikit-learn` returns as `feature_importances_`: each
    node's impurity decrease, weighted by the rows reaching it, summed over every node the feature
    splits at, and normalised so the tree's features sum to 1.

    **Summed**, because a feature splitting at three nodes did three pieces of work. The maximum
    would rate it level with a feature used once, and the mean would rank it below one, penalising
    reuse. The `n_samples` weight stops many small deep nodes from adding up to a large total, and
    it also ranks a conditional split (one reached by few rows, beneath another feature's cut) below
    a marginal one.

    **Normalised**, because the tree's size and total impurity reduction vary from seed to seed, so
    raw sums are not comparable between runs of the same arm. Every record's importance stays `None`
    when the total is not positive, as for a tree that never split on anything.
    """
    total_rows = max((node["n_samples"] for record in records
                      for node in record["extra"]["nodes"] if "n_samples" in node), default=0)
    if not total_rows:
        return

    raw = {}
    for record in records:
        nodes = [node for node in record["extra"]["nodes"] if "n_samples" in node]
        if len(nodes) != len(record["extra"]["nodes"]):
            return  # partial statistics describe no tree, so no record gets an importance
        raw[record["name"]] = sum(
            (node["n_samples"] / total_rows) * node["impurity_decrease"] for node in nodes
        )

    total = sum(raw.values())
    if total <= 0:
        return
    for record in records:
        record["importance"] = raw[record["name"]] / total


def tree_predictions(tree, X: pd.DataFrame, overrides: dict | None = None) -> np.ndarray:
    """The tree's prediction for every row, by routing rather than by calling `predict`.

    Routing is what lets a feature be permuted. An arm fitted in Core holds a feature matrix, so
    permuting a feature means shuffling a column of the input. A DEFT tree is handed the raw frame
    and computes its features from it, so there is no column to shuffle. Substituting shuffled
    values wherever a node tests the feature is the equivalent intervention for this model.

    `overrides` maps a feature name to values that replace the computed ones. With no overrides this
    reproduces `tree.predict(..., return_proba=True)`, which `check_tree_predictions` asserts.
    """
    positions = np.arange(len(X))
    out = np.full(len(X), np.nan)
    cache: dict[tuple[str, object], np.ndarray] = {}

    def column(feature) -> np.ndarray:
        name = feature.name
        if overrides is not None and name in overrides:
            return overrides[name]
        # Keyed on the callable as well as the name, because DEFT can put two different functions
        # on two nodes under one name (for example a signed quantity at one node and its absolute
        # value at another). A cache keyed by name alone would route the second node through the
        # first node's values, and `check_tree_predictions` would catch that only when the
        # substitution happened to change a prediction.
        key = (name, feature.fn)
        if key not in cache:
            cache[key] = np.asarray(feature.fn(X.copy()), dtype=float).ravel()
        return cache[key]

    def walk(node, rows: np.ndarray) -> None:
        if node is None or len(rows) == 0:
            return
        if node.left is None and node.right is None:
            out[rows] = node.prediction
            return
        feature = node.feature
        threshold = float(feature.threshold) if feature.threshold is not None else 0.0
        goes_left = column(feature)[rows] <= threshold
        walk(node.left, rows[goes_left])
        walk(node.right, rows[~goes_left])

    walk(tree.root, positions)
    return out


def check_tree_predictions(tree, X: pd.DataFrame, tolerance: float = 1e-9) -> None:
    """Refuse routing that does not reproduce the tree's own predictions.

    This is the counterpart of `check_node_statistics`: a permutation importance built on a route
    the tree would not take is wrong in a way no later check would reveal.
    """
    mine = tree_predictions(tree, X)
    theirs = np.asarray(tree.predict(X, return_proba=True), dtype=float).ravel()
    if not np.allclose(mine, theirs, atol=tolerance, equal_nan=True):
        differing = int(np.sum(~np.isclose(mine, theirs, atol=tolerance, equal_nan=True)))
        raise RuntimeError(
            f"routing the rows reproduces a different prediction from the tree's own on "
            f"{differing} of {len(X)} rows, so no feature can be permuted through it. Check the "
            f"threshold comparison and that every node's feature evaluates on this frame."
        )


def permutation_importances(tree, X: pd.DataFrame, y, names: list[str], *,
                            repeats: int, rng: np.random.Generator) -> dict[str, float]:
    """How far the headline falls when each feature's values are shuffled, as the harness
    measures it: AUROC for a binary target, R-squared for a continuous one.

    The metric and the form of the output match the harness's, so a DEFT feature's number is
    comparable with a Core arm's. See `seqmaestro.harness.importance` for what it does and does not
    measure.
    """
    from sklearn.metrics import r2_score, roc_auc_score

    score = roc_auc_score if target_kind_of(tree) == BINARY else r2_score
    y = np.asarray(y).ravel()
    if repeats < 1 or len(np.unique(y)) < 2 or not names:
        return {}

    check_tree_predictions(tree, X)
    baseline = score(y, tree_predictions(tree, X))

    by_name = {}
    def walk(node):
        if node is None or getattr(node, "feature", None) is None:
            return
        if node.left is not None or node.right is not None:
            by_name.setdefault(node.feature.name, node.feature)
        walk(node.left)
        walk(node.right)
    walk(tree.root)

    out: dict[str, float] = {}
    for name in names:
        feature = by_name.get(name)
        if feature is None or feature.fn is None:
            continue
        try:
            values = np.asarray(feature.fn(X.copy()), dtype=float).ravel()
        except Exception as exc:  # one unevaluable feature must not stop the rest
            print(f"  feature {name!r} could not be permuted: {exc}", file=sys.stderr)
            continue
        drops = []
        for _ in range(repeats):
            shuffled = tree_predictions(tree, X, overrides={name: rng.permutation(values)})
            drops.append(baseline - score(y, shuffled))
        out[name] = float(np.mean(drops))
    return out


def evaluate_features(records: list, tree, X: pd.DataFrame) -> np.ndarray:
    """The selected features evaluated on the rows of `X`, one column each: the reference split for
    stability matching, the training rows for the marginal effect, and whatever rows a later stage
    hands `deft_cart.py`.

    A feature that raises or returns the wrong length yields a column of NaN and the run continues,
    because generated code is expected to be imperfect and one bad feature should not cost the rest.
    """
    by_name = {}

    def walk(node):
        if node is None or getattr(node, "feature", None) is None:
            return
        by_name.setdefault(node.feature.name, node.feature)
        walk(node.left)
        walk(node.right)

    walk(tree.root)

    columns = []
    for record in records:
        feature = by_name.get(record["name"])
        column = np.full(len(X), np.nan)
        if feature is not None and feature.fn is not None:
            try:
                values = np.asarray(feature.fn(X.copy()), dtype=float).ravel()
                if values.size == len(X):
                    column = values
            except Exception as exc:
                print(f"  feature {record['name']!r} failed to evaluate: {exc}", file=sys.stderr)
        columns.append(column)
    return np.column_stack(columns) if columns else np.zeros((len(X), 0))


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: A feature name SeqMaestro accepts: a Python identifier, as `seqmaestro.features.base` requires.
#: Restated because this side cannot import it.
LEGAL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def legal_names(tree) -> None:
    """Give every feature the tree holds a name SeqMaestro accepts, in place.

    DEFT keeps the name a language model proposed for a feature, which can hold a character no
    Python identifier may (`Half-window_CpG_imbalance`), even where the function it wrote has a
    legal name. SeqMaestro refuses such a name, because a model shown it copies it into code that
    will not compile. Each run of other characters becomes an underscore, a name that would start
    with a digit gets a leading underscore, and a name another feature already holds gets a
    numbered suffix. One proposed name maps to one new name, so two functions DEFT put under one
    name keep sharing it, as `evaluate_features` expects. The proposed name is kept on the feature
    as `name_as_proposed`. Every use of a saved tree goes through `load_tree`, which calls this, so
    a feature carries the same name in every record made from the tree.
    """
    features = []

    def walk(node):
        if node is None:
            return
        if getattr(node, "feature", None) is not None and isinstance(node.feature.name, str):
            features.append(node.feature)
        walk(getattr(node, "left", None))
        walk(getattr(node, "right", None))

    walk(tree.root)
    taken = {feature.name for feature in features if LEGAL_NAME.match(feature.name)}
    renamed: dict[str, str] = {}
    for feature in features:
        proposed = feature.name
        if LEGAL_NAME.match(proposed):
            continue
        if proposed not in renamed:
            base = re.sub(r"[^A-Za-z0-9_]+", "_", proposed).strip("_") or "feature"
            base = base if LEGAL_NAME.match(base) else f"_{base}"
            name, suffix = base, 2
            while name in taken:
                name, suffix = f"{base}_{suffix}", suffix + 1
            taken.add(name)
            renamed[proposed] = name
        feature.name = renamed[proposed]
        feature.name_as_proposed = proposed


def load_tree(path: Path):
    """The fitted tree saved at `path`, refused if it is not something that predicts, with every
    feature under a name SeqMaestro accepts (`legal_names`)."""
    import dill

    with open(path, "rb") as handle:
        tree = dill.load(handle)
    if not hasattr(tree, "predict"):
        raise RuntimeError(
            f"{path} holds a {type(tree).__name__}, which has no `predict`. The adapter needs "
            f"the fitted tree; if the saved artifact changed shape, this is where to follow it."
        )
    legal_names(tree)
    return tree


def predict_rows(tree, rows: pd.DataFrame, id_column: str) -> tuple[np.ndarray, np.ndarray]:
    """The identifiers and the tree's score for each row, in the order the rows came: the
    probability of the positive class, or the predicted value of a continuous target.

    Every column but the identifier is handed to the tree, as `fit` hands it every column of the
    validation rows but the label, so the frame has the shape the tree was fitted on.
    """
    if id_column not in rows.columns:
        raise ValueError(f"the rows have no identifier column {id_column!r}; "
                         f"columns are {list(rows.columns)}.")
    features = rows.drop(columns=[id_column])
    scores = np.asarray(tree.predict(features, return_proba=True), dtype=float).ravel()
    return rows[id_column].to_numpy(), scores


def predict(args) -> int:
    """Load the tree a bundle names, predict the rows given, and write identifiers and scores."""
    arm = json.loads((args.bundle / "arm.json").read_text())
    if arm.get("bundle_version") != BUNDLE_VERSION or "fitted_model" not in arm:
        raise SystemExit(f"{args.bundle}/arm.json is not a version {BUNDLE_VERSION} bundle naming "
                         f"its fitted model, so there is no tree here to predict with.")
    tree = load_tree(args.bundle / arm["fitted_model"])
    ids, scores = predict_rows(tree, pd.read_csv(args.rows), args.id_column)
    np.savez_compressed(args.out, row_id=ids, score=scores)
    print(f"predicted {len(scores)} rows with {arm['fitted_model']}", flush=True)
    return 0


def main() -> int:
    top = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = top.add_subparsers(dest="mode", required=True)

    predicting = modes.add_parser("predict", help="predict rows with the tree a bundle saved")
    predicting.add_argument("--bundle", type=Path, required=True,
                            help="a bundle written by `fit`; its arm.json names the tree")
    predicting.add_argument("--rows", type=Path, required=True,
                            help="CSV of the rows to predict: an identifier column and the columns "
                                 "the tree was fitted on, and no label")
    predicting.add_argument("--id-column", required=True,
                            help="the identifier column, echoed back beside the scores")
    predicting.add_argument("--out", type=Path, required=True, help="where to write the scores")

    parser = modes.add_parser("fit", help="fit DEFT and write a bundle")
    parser.add_argument("--train", type=Path, required=True, help="the rows to fit on")
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--feature-finder", required=True,
                        help="cart, or DEFT's finder that has a language model write candidates. "
                             "The arm's pool decides it, so the caller states it.")
    # Required, with no default, because which column is the label, and what it holds, are facts
    # about the data.
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--target-kind", required=True, choices=(BINARY, CONTINUOUS),
                        help="what the target holds; DEFT grows a classification or a regression "
                             "tree for it")
    parser.add_argument("--permutation-repeats", type=int, required=True,
                        help="Shuffles per feature for the permutation importance, which is "
                             "measured the same way for every arm; 0 skips it. The run's "
                             "`grid.permutation_repeats`.")
    parser.add_argument("--permutation-seed", type=int, required=True,
                        help="Seeds the shuffles, so the number is reproducible.")
    parser.add_argument("--weight-column", default=None,
                        help="Column of per-row fitting weights in the training CSV. Kept out "
                             "of the features and passed to the tree.")
    parser.add_argument(
        "--sequence-column",
        default="",
        help="Empty means the inputs are already a numeric feature matrix.",
    )
    # The dataset, whose description SeqMaestro holds (see `deft_conf/README.md`). Omitting it gives
    # DEFT's task-neutral prompts. That suits CART, which never prompts, but for an LLM arm it is
    # almost certainly a mistake, so the Core side (`seqmaestro/pipeline/vendored/deft.py`) passes
    # it for every cell.
    parser.add_argument("--task", default=None,
                        help="dataset name; read from --dataset-dir, or a hand-written config if one exists")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR,
                        help=f"dataset descriptions, shared with every other consumer; default {DATASET_DIR}")
    parser.add_argument("--task-config-dir", type=Path, default=TASK_CONFIG_DIR,
                        help=f"hand-written prompt configs, which win when present; default {TASK_CONFIG_DIR}")
    parser.add_argument("--config-override", action="append", default=[],
                        help="Raw hydra override, repeatable, e.g. tree.max_depth=3")
    parser.add_argument("--effect", type=Path, default=None,
                        help="Every training row with its real label. Where `marginal_effect` is "
                             "measured, which is neither --train (this seed's subsample, shuffled "
                             "labels on a control arm) nor --val (too small to carry a sign).")
    # For a cell whose fit succeeded and whose post-processing failed. Refitting would draw a
    # different tree, and on a permuted cell that would quietly filter the null: the tree kept would
    # be one that happened to survive post-processing, and not the one the first fit drew.
    parser.add_argument("--reuse-tree", action="store_true",
                        help="Use the .dill already under the bundle instead of fitting again.")
    args = top.parse_args()
    return predict(args) if args.mode == "predict" else fit(args)


def fit(args) -> int:
    """Fit DEFT on the training rows and write the bundle, predicting the validation rows only."""
    started = time.time()
    out = args.out_bundle
    work_dir = (out / WORK_DIR).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"training DEFT (feature_finder={args.feature_finder}) on {args.train.name}", flush=True)
    tree_path = train(args, work_dir)
    tree = load_tree(tree_path)
    if target_kind_of(tree) != args.target_kind:
        raise RuntimeError(f"{tree_path} was grown for a {target_kind_of(tree)} target, and this "
                           f"arm's target is {args.target_kind}.")

    val = pd.read_csv(args.val)
    feature_columns = [c for c in val.columns if c != args.target_column]

    p_val = np.asarray(tree.predict(val[feature_columns], return_proba=True), dtype=float).ravel()

    # The rows the tree was fitted on, in their original order: `node_statistics` routes them back
    # down it to recover the coverage and impurity DEFT does not store. The weight column is written
    # into the training CSV but kept out of the features, as it was at fit time.
    fitted_on = pd.read_csv(args.train)
    fit_features = [c for c in fitted_on.columns
                    if c not in {args.target_column, args.weight_column}]
    stats = node_statistics(tree, fitted_on[fit_features], fitted_on[args.target_column])
    check_node_statistics(stats)
    records = collect_features(tree, read_pool(Path(args.train).parent), stats)
    # The permutation importance every arm reports, on the validation rows the behavioural matcher
    # also uses, so a DEFT feature's figure is comparable with a Core arm's.
    permuted = permutation_importances(
        tree, val[feature_columns], val[args.target_column].to_numpy(),
        [record["name"] for record in records],
        repeats=args.permutation_repeats,
        rng=np.random.default_rng(args.permutation_seed),
    )
    for record in records:
        record["permutation_importance"] = permuted.get(record["name"])
    values = evaluate_features(records, tree, val[feature_columns])

    # The same features over the training rows, handed back with their labels so the SeqMaestro side
    # can measure `marginal_effect` from them. The values are computed here because only this
    # environment can call a DEFT feature. The statistic is measured on the other side so that every
    # arm's `marginal_effect` comes from one definition.
    if args.effect is None:
        raise SystemExit("--effect is required: the bundle must carry the rows marginal_effect is "
                         "measured on. See seqmaestro.harness.runner.EFFECT_SPLIT.")
    effect = pd.read_csv(args.effect)
    effect_features = [c for c in effect.columns if c != args.target_column]
    effect_values = evaluate_features(records, tree, effect[effect_features])

    np.savez_compressed(
        out / "predictions.npz",
        val_true=val[args.target_column].to_numpy(),
        val_score=p_val,
    )
    (out / "features.json").write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "importance_kind": IMPORTANCE_KIND.get(args.feature_finder, "deft_split_score"),
                "direction_kind": DIRECTION_KIND[args.target_kind],
                "features": records,
            },
            indent=2, default=str,
        ) + "\n"
    )
    np.savez_compressed(
        out / "feature_values.npz",
        names=np.array([r["name"] for r in records], dtype=object),
        values=values.astype(np.float32),
    )
    # The labels travel with the values, so the two cannot be handed back unaligned.
    np.savez_compressed(
        out / "effect_values.npz",
        names=np.array([r["name"] for r in records], dtype=object),
        values=effect_values.astype(np.float32),
        y=effect[args.target_column].to_numpy(),
    )
    (out / "arm.json").write_text(
        json.dumps(
            {
                "bundle_version": BUNDLE_VERSION,
                # What `predict` loads at the reveal, relative to the bundle so it moves with it.
                "fitted_model": str(tree_path.relative_to(out.resolve())),
                # The model as `seqmaestro.arms` names it, and the finder says which of DEFT's two
                # grew this tree.
                "model": "deft_or_cart",
                "feature_finder": args.feature_finder,
                "environment": "SeqMaestro_Method_deft",
                "python": sys.version.split()[0],
                "n_fit_rows": len(pd.read_csv(args.train, usecols=[args.target_column])),
                "n_features_used": len(records),
                "wall_clock_seconds": round(time.time() - started, 1),
                "config_overrides": list(args.config_override),
            },
            indent=2, default=str,
        ) + "\n"
    )
    print(f"bundle written to {out}: {len(records)} features, {len(p_val)} validation rows, "
          f"{len(effect)} rows for the marginal effect", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
