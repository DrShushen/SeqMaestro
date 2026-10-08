#!/usr/bin/env python3
"""Everything asked of DEFT after Stage 2. **Runs in `SeqMaestro_Method_deft`.**

    $DEFT experiments/adapters/deft_cart.py reduce --cell <stage 2 cell> --fit fit.csv \\
        --val val.csv --id-column row_id --target-column <column> --n-target 10 --out <arm dir>
    $DEFT experiments/adapters/deft_cart.py predict --cell <stage 2 cell> --model <reduced cell> \\
        --rows rows.csv --id-column row_id --out scores.npz
    $DEFT experiments/adapters/deft_cart.py tree --cell <stage 2 cell> --model <reduced cell> \\
        --fit fit.csv --val val.csv --id-column row_id --target-column <column> --out tree.json
    $DEFT experiments/adapters/deft_cart.py pool --cell <stage 2 cell> --fit fit.csv \\
        --rows rows.csv --id-column row_id --sequence-column <column> --out pool.npz
    $DEFT experiments/adapters/deft_cart.py {reproduce,select,refit} --cell <stage 2 cell> \\
        --pool pool.npz --prepared prepare.npz ...
    $DEFT experiments/adapters/deft_cart.py time --inputs <handed inputs> --target-column <column> \\
        --target-kind binary --sequence-column <column> --out timings.json

A DEFT arm's features are functions its tree carries: positional candidates DEFT enumerates and,
on a pool with `llm_generated_joint`, functions a language model wrote. Only this environment can
evaluate them. Nothing is recompiled on the other side and no language model is called. The
functions are the ones the tree saved, or the naive population DEFT builds from the rows the way
its finder does.

**Stage 3A.** `reduce` eliminates the arm down to N features with DEFT's tree, built from the
configuration the cell's fit recorded (`recorded_configuration`). The tree therefore has the
splitting criterion, the stopping rules and every other setting the arm's tree was fitted with.
Given a numeric frame, the tree's candidate pool is one identity candidate per column, which makes
it plain CART over those columns. The Stage 2 tree evaluates the arm's recorded features on the
rows. `reduce` writes the elimination trajectory, the validation predictions and three trees
(`full`, `reduced` and `top_n`), and the other side scores them.

**Stage 3B.** `pool` evaluates the arm's candidate pool on the rows it is handed. For the CART
finder the pool is DEFT's whole naive population, built on the fitting rows, as it was offered to
the tree's root. For the generative finder it is the features the tree split on, because that
finder's candidates never existed as a set. The other side runs Step 0 on that matrix and writes
its decisions to `prepare.npz`. Then `reproduce` runs the selection on the fitting rows, for the
other side to check that it returns the reduced model, `select` runs Tier 1's half-sample
selections, and `refit` fits and saves the leave-one-out trees. All three fit on the pool matrix
this side computed, with the cell's tree settings and the rows' weights.

**The reveal.** `predict` evaluates the features on the rows given, narrows them to the columns a
saved tree was fitted on, and returns the tree's scores (probabilities, or a continuous target's
predicted values) beside the row identifiers. `pool` gives Tier 2 the pool on the test rows.

**Before a run.** `time` grows CART on samples of the rows a cell would be handed, each twice the
last, for the driver's time estimate (`seqmaestro/pipeline/estimate.py`).

**The report.** `tree` writes a reduced tree's structure for a figure. It reads the saved tree
without refitting it, routes the fitting rows down it for each node's counts, and checks that
routing against the tree's own predictions. It also writes the tree's validation predictions, which
the other side checks against the cell's.

The elimination loop, the column check and Tier 1's sweep are `seqmaestro` modules that import
only numpy and the standard library (`SHARED`). This side loads them by path and records their
digests, so it runs the same code as the other side without importing `seqmaestro`. Labels cross
the boundary only for rows that are fitted on: the fitting rows and, for Tier 1, the train and
validation rows. Test rows come without labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ADAPTERS = Path(__file__).resolve().parent
REPO = ADAPTERS.parents[1]
DEFT_CONF = REPO / "methods" / "deft" / "src" / "deft" / "conf"

#: The numpy-only modules this side shares with `seqmaestro`, loaded by file and never imported as
#: a package, so this environment needs no SeqMaestro code beyond these files.
SHARED = {
    "eliminate": REPO / "seqmaestro" / "reduction" / "eliminate.py",
    "check": REPO / "seqmaestro" / "reduction" / "check.py",
    "selections": REPO / "seqmaestro" / "reliability" / "selections.py",
}

#: What a pool is, by the feature finder a cell records. The CART finder's pool is DEFT's naive
#: population, and the generative finder's is the features its tree split on. The labels are
#: `seqmaestro.reliability.pool`'s, restated because that module runs on the other side, and
#: `tests/test_reliability.py` checks the two agree.
POOLS = {"cart": "naive_candidates", "default": "recorded_features"}

#: What a reduced tree is saved as, inside its cell.
MODEL_FILE = "model.dill"

#: How `_covariate_features` names the identity candidate for a numeric column.
VALUE_PREFIX = "value_of_"

#: Where hydra keeps the configuration of a run, inside the directory the run was made in.
HYDRA_RECORD = Path(".hydra") / "config.yaml"


def load_by_path(name: str, path: Path):
    """A module loaded from its file, and the file's digest.

    The module is registered in `sys.modules` under `name`, as an import would be, so what it
    defines can be found by name. A process pool relies on that to hand a worker the function to
    run.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def stage2_adapter():
    """`deft_bundle.py`, the adapter that fitted the Stage 2 cells, for its functions that walk a
    fitted tree. It is loaded and not copied, because a feature evaluated by a second
    implementation might not be the feature the cell recorded."""
    return load_by_path("deft_bundle", ADAPTERS / "deft_bundle.py")[0]


class ArmFeatures:
    """A Stage 2 DEFT cell's recorded features, evaluated by the tree that saved them."""

    def __init__(self, cell: Path, bundle) -> None:
        self.cell = Path(cell)
        self.bundle = bundle
        self.records = json.loads((self.cell / "features.json").read_text())["features"]
        arm = json.loads((self.cell / "bundle" / "arm.json").read_text())
        self.tree = bundle.load_tree(self.cell / "bundle" / arm["fitted_model"])

    @property
    def names(self) -> list[str]:
        return [record["name"] for record in self.records]

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        """Every recorded feature on these rows, as the tree's functions compute it."""
        return self.bundle.evaluate_features(self.records, self.tree, frame)


def cell_target_kind(cell: Path) -> str:
    """The kind of target a Stage 2 cell was fitted to, from its `run.json`."""
    return json.loads((Path(cell) / "run.json").read_text())["target_kind"]


def recorded_configuration(cell: Path, bundle):
    """The configuration DEFT fitted a Stage 2 cell with: DEFT's settings with every override the
    run gave it, as hydra composed them and recorded them inside the cell when the fit started.

    Every later tree of the arm (the reduction's, the panel's) is built from this record by
    `cart_tree`. It therefore has the settings the cell's tree had, whichever of them a run
    changed, and no list of settings is kept here that a new one could be missing from.
    """
    from omegaconf import OmegaConf

    path = Path(cell) / "bundle" / bundle.WORK_DIR / HYDRA_RECORD
    if not path.exists():
        raise SystemExit(f"{path} does not exist, so the cell holds no record of the "
                         f"configuration its tree was fitted with.")
    return OmegaConf.load(path)


def composed_configuration(overrides: list[str], target_kind: str, bundle):
    """DEFT's configuration under a run's overrides, composed by hydra as a fit composes it. For a
    tree no cell has fitted yet, which only the time estimate grows."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(DEFT_CONF), version_base=None):
        return compose(config_name="config",
                       overrides=[*bundle.criterion_override(target_kind, overrides), *overrides])


def cart_tree(configuration, n_rows: int, sample_weight: np.ndarray | None, target_kind: str):
    """An unfitted tree as `configuration` sets it, with the CART finder, for a fit on `n_rows`.

    Built as DEFT's run builds its tree (`deft.cli.run_deft_tree`): the tree and its splitting
    criterion are instantiated from the configuration, and a leaf must hold the configured
    fraction of the rows being fitted. Nothing is saved to disk by the tree.
    """
    from deft.features.feature_finder import CartFeatureFinder
    from hydra.utils import instantiate

    return instantiate(
        configuration.tree,
        feature_finder=CartFeatureFinder(),
        splitting_criterion=instantiate(configuration.splitting_criterion),
        min_samples_leaf=min_samples_leaf(configuration, n_rows),
        store_node_artifacts=False,
        save_locally=False,
        folder_path=None,
        sample_weight=sample_weight,
        target_kind=target_kind,
    )


def min_samples_leaf(configuration, n_rows: int) -> int:
    """The fewest rows a leaf may hold in a fit on `n_rows`: the configured fraction of them."""
    return max(1, int(float(configuration.min_samples_leaf_fraction) * n_rows))


def tree_settings(configuration) -> dict:
    """What `configuration` says of the tree, as plain values, for the record a stage leaves: the
    tree's settings, the fraction of the fitting rows a leaf must hold, and the splitting
    criterion with its settings."""
    from omegaconf import OmegaConf

    tree = OmegaConf.to_container(configuration.tree, resolve=True)
    criterion = OmegaConf.to_container(configuration.splitting_criterion, resolve=True)
    return {
        **{key: value for key, value in tree.items()
           if key not in {"_target_", "min_samples_leaf", "store_node_artifacts", "save_locally",
                          "folder_path"}},
        "min_samples_leaf_fraction": float(configuration.min_samples_leaf_fraction),
        "criterion": criterion,
    }


class UnsplittableNodes(logging.Filter):
    """Count DEFT's "no candidate could be scored" errors instead of printing thousands of them.

    Each such error is ordinary CART behaviour and signals no fault: a node too small for any cut
    to satisfy `min_samples_leaf` becomes a leaf. The messages are counted instead of silenced, and
    the count is recorded.
    """

    def __init__(self):
        super().__init__()
        self.count = 0

    def filter(self, record):
        if "could be scored" in record.getMessage():
            self.count += 1
            return False
        return True


class FittedTree:
    """A fitted DEFT tree in the shape the elimination loop expects."""

    def __init__(self, tree, columns: list[str], importances: dict[str, float]):
        self.tree = tree
        self.columns = columns
        self._importances = importances

    def importances(self) -> np.ndarray:
        return np.array([abs(float(self._importances.get(name, 0.0))) for name in self.columns])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.tree.predict(pd.DataFrame(X, columns=self.columns), return_proba=True), dtype=float
        ).ravel()


def split_features(tree) -> dict[str, str]:
    """Node path to the feature it splits on, in `node_statistics`' path convention."""
    found = {}

    def walk(node, path):
        if node is None or getattr(node, "feature", None) is None:
            return
        if node.left is None and node.right is None:
            return
        found[path] = node.feature.name
        walk(node.left, path + "L")
        walk(node.right, path + "R")

    walk(tree.root, "")
    return found


def impurity_importances(tree, X: pd.DataFrame, y: pd.Series, node_statistics) -> dict[str, float]:
    """Each feature's share of the tree's total impurity reduction.

    Computed as `assign_importances` in `deft_bundle.py` computes the `importance` of every DEFT
    feature record: each node's impurity decrease weighted by the rows reaching it, summed over the
    nodes a feature splits at, and normalised across the tree.
    """
    stats = node_statistics(tree, X, y)
    names = split_features(tree)
    if set(stats) != set(names):
        raise RuntimeError(f"Node statistics cover {sorted(stats)[:3]}... and splits {sorted(names)[:3]}...")
    if not stats:
        return {}

    total_rows = max(node["n_samples"] for node in stats.values())
    raw: dict[str, float] = {}
    for path, node in stats.items():
        raw[names[path]] = raw.get(names[path], 0.0) + (node["n_samples"] / total_rows) * node["impurity_decrease"]
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {name: value / total for name, value in raw.items()}


def make_fit_step(configuration, node_statistics, sample_weight: np.ndarray | None,
                  target_kind: str):
    """One CART fit on the given columns, with the tree the cell's configuration sets and the
    fitting weights, for a `target_kind` target."""
    def fit_step(X: np.ndarray, y: np.ndarray, columns: list[str]) -> FittedTree:
        frame = pd.DataFrame(np.asarray(X), columns=columns)
        labels = pd.Series(np.asarray(y).ravel(), index=frame.index)
        tree = cart_tree(configuration, len(labels), sample_weight, target_kind)
        # `sequence_column=None`: the frame holds the arm's feature columns, not raw sequence, so
        # the naive pool is one identity candidate per column and nothing is decoded.
        tree.fit(frame, labels, sequence_column=None)

        importances = impurity_importances(tree, frame, labels, node_statistics)
        stripped = {
            (name[len(VALUE_PREFIX):] if name.startswith(VALUE_PREFIX) else name): value
            for name, value in importances.items()
        }
        unknown = set(stripped) - set(columns)
        if unknown:
            raise RuntimeError(f"Tree split on feature(s) not in the pool: {sorted(unknown)[:5]}.")
        return FittedTree(tree, columns, stripped)

    return fit_step


def features_of(rows: pd.DataFrame, *drop: str | None) -> pd.DataFrame:
    """The rows as the Stage 2 tree saw its own: every column but the identifier, label and weight."""
    return rows.drop(columns=[column for column in drop if column])


def reduce(args) -> int:
    import dill

    started = time.time()
    eliminate, eliminate_digest = load_by_path("reduction_eliminate", SHARED["eliminate"])
    check, check_digest = load_by_path("reduction_check", SHARED["check"])
    bundle = stage2_adapter()
    arm = ArmFeatures(args.cell, bundle)
    names = arm.names

    fit = pd.read_csv(args.fit)
    val = pd.read_csv(args.val)
    X_fit = arm.evaluate(features_of(fit, args.id_column, args.target_column, args.weight_column))
    X_val = arm.evaluate(features_of(val, args.id_column))
    checked = check.compare(X_val, names, args.cell)
    y_fit = fit[args.target_column].to_numpy()
    weights = fit[args.weight_column].to_numpy(dtype=float) if args.weight_column else None

    kind = cell_target_kind(args.cell)
    configuration = recorded_configuration(args.cell, bundle)
    unsplittable = UnsplittableNodes()
    logging.getLogger("deft.features.feature_finder").addFilter(unsplittable)
    fit_step = make_fit_step(configuration, bundle.node_statistics, weights, kind)
    result = eliminate.run(
        fit_step, X_fit, y_fit, X_val, names, args.n_target,
        on_step=lambda s: print(f"  n={s['n_features']:5d} used={s['n_used_by_model']:5d}"
                                f"  {time.time() - started:7.1f}s", flush=True),
    )
    top = eliminate.top_n(result["steps"][0]["importance"], args.n_target)
    top_names = [names[i] for i in top]
    top_fitted = fit_step(X_fit[:, top], y_fit, top_names)

    out = Path(args.out)
    fitted = {"full": result["first_fitted"], "reduced": result["fitted"], "top_n": top_fitted}
    for role, model in fitted.items():
        (out / role).mkdir(parents=True, exist_ok=True)
        with open(out / role / MODEL_FILE, "wb") as handle:
            dill.dump(model.tree, handle)
    np.savez_compressed(
        out / "elimination.npz",
        row_id=val[args.id_column].to_numpy(),
        val_by_step=np.vstack(result["val_scores_by_step"]),
        full=np.asarray(result["val_scores_by_step"][0], dtype=float),
        reduced=np.asarray(result["val_scores"], dtype=float),
        top_n=top_fitted.predict(X_val[:, top]),
    )
    (out / "elimination.json").write_text(json.dumps({
        "steps": result["steps"],
        "removed_at": [int(v) for v in result["removed_at"]],
        "final_columns": result["final_columns"],
        "final_importance": [float(v) for v in result["final_importance"]],
        "top_n_columns": top_names,
        "featurisation_check": checked,
        "n_fit_rows": len(fit),
        "settings": {
            "tree": tree_settings(configuration),
            "min_samples_leaf": min_samples_leaf(configuration, len(fit)),
            "unsplittable_nodes": unsplittable.count,
            "shared_modules": {"eliminate.py": eliminate_digest, "check.py": check_digest},
            "environment": "SeqMaestro_Method_deft",
            "python": sys.version.split()[0],
            "wall_clock_seconds": round(time.time() - started, 1),
        },
    }, indent=2, default=str) + "\n")
    print(f"reduced {len(names)} features to {len(result['final_columns'])} in "
          f"{len(result['steps'])} steps", flush=True)
    return 0


def predict(args) -> int:
    bundle = stage2_adapter()
    arm = ArmFeatures(args.cell, bundle)
    model = bundle.load_tree(Path(args.model) / MODEL_FILE)
    rows = pd.read_csv(args.rows)
    X = reduced_columns(arm, args.model, features_of(rows, args.id_column))
    scores = np.asarray(model.predict(X, return_proba=True), dtype=float).ravel()
    np.savez_compressed(args.out, row_id=rows[args.id_column].to_numpy(), score=scores)
    print(f"predicted {len(scores)} rows with {args.model}", flush=True)
    return 0


# ------------------------------------------------------------------------------ the report

#: Agreement required between a fitting row's leaf, as the export routes it, and the tree's own
#: prediction for that row.
ROUTING_TOLERANCE = 1e-12


def reduced_columns(arm: ArmFeatures, model_dir: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """The columns a reduced tree was fitted on, for these rows, as `predict` builds them."""
    columns = json.loads((Path(model_dir) / "run.json").read_text())["columns"]
    values = arm.evaluate(rows)
    index = [arm.names.index(column) for column in columns]
    return pd.DataFrame(values[:, index], columns=columns)


def reached(y: pd.Series, kind: str, positive) -> dict:
    """What a node's fitting rows hold: the positive class's count and share for a binary target,
    the target's mean and standard deviation for a continuous one."""
    if kind == "binary":
        return {"n_positive": int((y == positive).sum()),
                "positive_rate": float((y == positive).mean()) if len(y) else None}
    return {"mean": float(y.mean()) if len(y) else None,
            "sd": float(y.std(ddof=0)) if len(y) else None}


def tree(args) -> int:
    """A reduced tree's structure, for a report to draw: every node, the feature and threshold it
    splits on, and the fitting rows that reach it.

    The tree is the saved one and is never refitted. Rows are routed as `node_statistics` routes
    them (`<=` goes left, DEFT's convention), and the routing is checked: every fitting row's leaf
    must carry the tree's own prediction for that row, so a drawn node's counts are those of the
    leaves the model uses. The validation rows' predictions are written too, for the other side to
    check against the ones the cell recorded.
    """
    bundle = stage2_adapter()
    arm = ArmFeatures(args.cell, bundle)
    model = bundle.load_tree(Path(args.model) / MODEL_FILE)
    fit = pd.read_csv(args.fit)
    X = reduced_columns(arm, args.model, features_of(fit, args.id_column, args.target_column,
                                                     args.weight_column))
    y = pd.Series(fit[args.target_column].to_numpy(), index=X.index)
    stats = bundle.node_statistics(model, X, y)
    kind = bundle.target_kind_of(model)
    positive = int(np.max(np.unique(y.to_numpy()))) if kind == "binary" else None
    nodes: list[dict] = []
    routed = pd.Series(np.nan, index=X.index)

    def walk(node, path: str, depth: int, rows: pd.Index) -> None:
        leaf = node.left is None and node.right is None
        entry = {"path": path, "depth": depth, "n_samples": len(rows),
                 **reached(y.loc[rows], kind, positive),
                 "prediction": None if node.prediction is None else float(node.prediction),
                 "is_leaf": leaf, "feature": None, "threshold": None, "impurity_decrease": None,
                 "left": None, "right": None}
        nodes.append(entry)
        if leaf or getattr(node, "feature", None) is None:
            entry["is_leaf"] = True
            routed.loc[rows] = entry["prediction"]
            return
        feature = node.feature
        threshold = float(feature.threshold) if feature.threshold is not None else 0.0
        goes_left = np.asarray(feature.fn(X.loc[rows].copy()), dtype=float).ravel() <= threshold
        name = feature.name
        entry.update(feature=name[len(VALUE_PREFIX):] if name.startswith(VALUE_PREFIX) else name,
                     threshold=threshold, left=path + "L", right=path + "R",
                     impurity_decrease=stats.get(path, {}).get("impurity_decrease"))
        walk(node.left, path + "L", depth + 1, rows[goes_left])
        walk(node.right, path + "R", depth + 1, rows[~goes_left])

    walk(model.root, "", 0, X.index)
    predicted = np.asarray(model.predict(X, return_proba=True), dtype=float).ravel()
    worst = float(np.nanmax(np.abs(routed.to_numpy() - predicted)))
    if routed.isna().any() or worst > ROUTING_TOLERANCE:
        raise SystemExit(f"{args.model}: routing the fitting rows down the saved tree does not "
                         f"reproduce its predictions (largest difference {worst:.2e}); the drawn "
                         f"counts would not be the model's.")

    val = pd.read_csv(args.val)
    val_scores = np.asarray(model.predict(reduced_columns(arm, args.model,
                                                          features_of(val, args.id_column)),
                                          return_proba=True), dtype=float).ravel()
    Path(args.out).write_text(json.dumps({
        "kind": "tree", "target_kind": kind, "nodes": nodes, "columns": list(X.columns),
        **({"positive_label": positive} if kind == "binary" else {}),
        "n_fit_rows": len(X), "leaves": sum(node["is_leaf"] for node in nodes),
        "routing_max_difference": worst,
        "val_row_id": val[args.id_column].tolist(), "val_score": val_scores.tolist(),
    }, indent=1) + "\n")
    print(f"exported {len(nodes)} nodes of {args.model}", flush=True)
    return 0


# ------------------------------------------------------------------------------ Stage 3B

#: The label set the fitting rows are fitted under: the real labels, by the other side's name.
REAL_LABELS = "real"


def naive_population(fit: pd.DataFrame, sequence_column: str, window: dict | None = None) -> list:
    """DEFT's naive candidates for these rows, built as its CART finder builds them at a node:
    positions read in the cell's positional `window` where it had one, and every handed column."""
    from deft.features.feature_finder import CartFeatureFinder

    return CartFeatureFinder().naive_features(fit, sequence_column=sequence_column,
                                              positional_window=window)


def pool(args) -> int:
    """The arm's candidate pool on the rows given: names, what each is, and its values."""
    bundle = stage2_adapter()
    run = json.loads((Path(args.cell) / "run.json").read_text())
    source = POOLS[run["feature_finder"]]
    rows = pd.read_csv(args.rows)
    frame = features_of(rows, args.id_column)
    if source == POOLS["cart"]:
        handed = Path(args.cell) / "inputs"
        features = naive_population(features_of(pd.read_csv(args.fit), args.id_column),
                                    args.sequence_column, bundle.read_window(handed))
        manifest = bundle.read_pool(handed)
        recipes = [bundle.naive_recipe(feature.name, manifest) for feature in features]
        unknown = [feature.name for feature, recipe in zip(features, recipes, strict=True) if recipe is None]
        if unknown:
            raise SystemExit(f"naive candidate(s) with no recognisable recipe: {unknown[:5]}")
        names = [feature.name for feature in features]
        kinds = [recipe["kind"] for recipe in recipes]
        params = [recipe["params"] for recipe in recipes]
        descriptions = [recipe.get("description") or feature.description or ""
                        for feature, recipe in zip(features, recipes, strict=True)]
        values = np.column_stack(
            [np.asarray(feature.fn(frame.copy()), dtype=float).ravel() for feature in features])
    else:
        arm = ArmFeatures(args.cell, bundle)
        names = arm.names
        kinds = [record["kind"] for record in arm.records]
        params = [record.get("params") or {} for record in arm.records]
        descriptions = [record.get("description") or "" for record in arm.records]
        values = arm.evaluate(frame)
    np.savez_compressed(args.out, names=np.array(names, dtype=str), kinds=np.array(kinds, dtype=str),
                        params=np.array([json.dumps(p, sort_keys=True) for p in params], dtype=str),
                        descriptions=np.array(descriptions, dtype=str), values=values,
                        row_id=rows[args.id_column].to_numpy(), source=np.array(source))
    print(f"evaluated {len(names)} candidates ({source}) on {len(rows)} rows", flush=True)
    return 0


class PanelInputs:
    """The pool this side evaluated, and what the other side decided on it (`prepare.npz`)."""

    def __init__(self, cell: Path, pool_path: Path, prepared_path: Path) -> None:
        with np.load(pool_path) as saved:
            self.names = [str(name) for name in saved["names"]]
            self.values = saved["values"]
        with np.load(prepared_path) as saved:
            if [str(name) for name in saved["names"]] != self.names:
                raise SystemExit(f"{prepared_path} was prepared over another pool than {pool_path}.")
            self.kept, self.halves = saved["kept"], saved["halves"]
            self.fit_index, self.val_index = saved["fit_index"], saved["val_index"]
            self.labels = {key.removeprefix("labels_"): saved[key] for key in saved.files
                           if key.startswith("labels_")}
            self.weights = saved["weights"] if "weights" in saved.files else None
        self.bundle = stage2_adapter()
        self.target_kind = cell_target_kind(cell)
        self.configuration = recorded_configuration(cell, self.bundle)
        self.settings = tree_settings(self.configuration)
        self.unsplittable = UnsplittableNodes()
        logging.getLogger("deft.features.feature_finder").addFilter(self.unsplittable)

    def fit_step(self, weights: np.ndarray | None):
        return make_fit_step(self.configuration, self.bundle.node_statistics, weights,
                             self.target_kind)

    def fitting(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """The pool on the fitting rows, their real labels and their weights."""
        fit = self.fit_index
        return (self.values[fit], self.labels[REAL_LABELS][fit],
                None if self.weights is None else self.weights[fit])


def reproduce(args) -> int:
    """What the selection keeps on the fitting rows, from the whole pool and from the recorded set."""
    eliminate, eliminate_digest = load_by_path("reduction_eliminate", SHARED["eliminate"])
    selections, selections_digest = load_by_path("reliability_selections", SHARED["selections"])
    arm = PanelInputs(args.cell, args.pool, args.prepared)
    recorded = [record["name"] for record in
                json.loads((Path(args.cell) / "features.json").read_text())["features"]]
    X, y, weights = arm.fitting()
    found = {}
    for start, names in (("from_pool", arm.names), ("from_recorded", recorded)):
        index = [arm.names.index(name) for name in names]
        select = selections.selector(eliminate.run, names, args.n_target)
        found[start] = select(arm.fit_step(weights), np.ascontiguousarray(X[:, index]), y)
    Path(args.out).write_text(json.dumps({
        **found, "tree": arm.settings, "unsplittable_nodes": arm.unsplittable.count,
        "shared_modules": {"eliminate.py": eliminate_digest, "selections.py": selections_digest},
    }, indent=2) + "\n")
    return 0


def select(args) -> int:
    """Tier 1: the selection on every half-sample of train and validation, under one label set."""
    eliminate, _ = load_by_path("reduction_eliminate", SHARED["eliminate"])
    selections, _ = load_by_path("reliability_selections", SHARED["selections"])
    arm = PanelInputs(args.cell, args.pool, args.prepared)
    kept = np.flatnonzero(arm.kept)
    names = [arm.names[i] for i in kept]
    _, timing = selections.run(
        selections.selector(eliminate.run, names, args.n_target),
        np.ascontiguousarray(arm.values[:, kept]), arm.labels[args.label_set], names, arm.halves,
        make_fit_step=arm.fit_step, weights=arm.weights, out_path=args.out, workers=args.workers,
        force=args.force, memory=args.memory, threads_file=args.threads_file)
    print(f"{timing['n_half_samples']} selections, {timing['core_seconds'] / 3600:.2f} core-hours",
          flush=True)
    return 0


def refit(args) -> int:
    """The leave-one-out trees, fitted on the fitting rows and saved in their cells, and their
    validation scores."""
    import dill

    arm = PanelInputs(args.cell, args.pool, args.prepared)
    plan = json.loads(Path(args.plan).read_text())
    X, y, weights = arm.fitting()
    fit_step = arm.fit_step(weights)
    X_val = arm.values[arm.val_index]
    scores = []
    for entry in plan:
        index = [arm.names.index(name) for name in entry["columns"]]
        fitted = fit_step(X[:, index], y, entry["columns"])
        cell = Path(entry["cell"])
        cell.mkdir(parents=True, exist_ok=True)
        with open(cell / MODEL_FILE, "wb") as handle:
            dill.dump(fitted.tree, handle)
        scores.append(fitted.predict(X_val[:, index]))
    settings = {**arm.settings, "min_samples_leaf": min_samples_leaf(arm.configuration, len(y)),
                "unsplittable_nodes": arm.unsplittable.count}
    np.savez_compressed(args.out, labels=np.array([entry["label"] for entry in plan], dtype=str),
                        scores=np.vstack(scores), settings=np.array(json.dumps(settings)))
    print(f"refitted {len(plan)} trees", flush=True)
    return 0


def time_fits(args) -> int:
    """How long the tree takes to grow on samples of the rows a cell would be handed: the first
    `--first-rows` of them, then twice as many, until one fit takes `--enough` seconds, uses every
    row, or the next would run past `--budget`. The rows are a random draw, so their first rows are
    one too. Each fit is grown as a Stage 2 cell's is, from the sequence and the handed columns."""
    from deft.utils.feature import get_naive_features

    handed = stage2_adapter()
    rows = pd.read_csv(args.inputs / "train.csv")
    labels = rows.pop(args.target_column)
    weights = rows.pop(args.weight_column).to_numpy(dtype=float) if args.weight_column else None
    window = handed.read_window(args.inputs)
    configuration = composed_configuration(args.config_override, args.target_kind, handed)
    # Every candidate is logged at every node, thousands of lines a node for a vocabulary.
    logging.getLogger().setLevel(logging.WARNING)
    candidates = len(get_naive_features(rows, sequence_column=args.sequence_column,
                                        positional_window=window))

    fits, spent, n = [], 0.0, min(args.first_rows, len(rows))
    while True:
        tree = cart_tree(configuration, n, None if weights is None else weights[:n],
                         args.target_kind)
        started = time.perf_counter()
        tree.fit(rows.head(n), labels.head(n), sequence_column=args.sequence_column,
                 positional_window=window)
        seconds = time.perf_counter() - started
        spent += seconds
        fits.append({"rows": n, "seconds": seconds, "used": len(set(split_features(tree).values()))})
        if seconds >= args.enough or n == len(rows) or spent + 2 * seconds > args.budget:
            break
        n = min(2 * n, len(rows))
    args.out.write_text(json.dumps({"candidates": candidates, "fits": fits}, indent=2) + "\n")
    return 0


def main() -> int:
    top = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = top.add_subparsers(dest="mode", required=True)

    reducing = modes.add_parser("reduce", help="eliminate a DEFT arm to N features with CART")
    reducing.add_argument("--cell", type=Path, required=True, help="the Stage 2 cell reduced")
    reducing.add_argument("--fit", type=Path, required=True,
                          help="the rows the cell was fitted on: identifier, sequence, label")
    reducing.add_argument("--val", type=Path, required=True,
                          help="the validation rows: identifier and sequence, no label")
    reducing.add_argument("--id-column", required=True)
    reducing.add_argument("--target-column", required=True)
    reducing.add_argument("--weight-column", default=None,
                          help="per-row fitting weights in --fit, when the design has them")
    reducing.add_argument("--n-target", type=int, required=True)
    reducing.add_argument("--out", type=Path, required=True, help="the reduced arm's directory")

    predicting = modes.add_parser("predict", help="predict rows with a reduced tree")
    predicting.add_argument("--cell", type=Path, required=True,
                            help="the Stage 2 cell whose features the tree was fitted on")
    predicting.add_argument("--model", type=Path, required=True,
                            help="the reduced cell: its run.json names the columns")
    predicting.add_argument("--rows", type=Path, required=True,
                            help="identifier and sequence, no label")
    predicting.add_argument("--id-column", required=True)
    predicting.add_argument("--out", type=Path, required=True)

    exporting = modes.add_parser("tree", help="a reduced tree's structure, for a report to draw")
    exporting.add_argument("--cell", type=Path, required=True,
                           help="the Stage 2 cell whose features the tree was fitted on")
    exporting.add_argument("--model", type=Path, required=True,
                           help="the reduced cell: its run.json names the columns")
    exporting.add_argument("--fit", type=Path, required=True,
                           help="the rows the tree was fitted on: identifier, sequence, label")
    exporting.add_argument("--val", type=Path, required=True,
                           help="the validation rows: identifier and sequence, no label")
    exporting.add_argument("--id-column", required=True)
    exporting.add_argument("--target-column", required=True)
    exporting.add_argument("--weight-column", default=None,
                           help="per-row fitting weights in --fit, when the design has them")
    exporting.add_argument("--out", type=Path, required=True)

    pooling = modes.add_parser("pool", help="evaluate the arm's candidate pool on rows")
    pooling.add_argument("--cell", type=Path, required=True, help="the Stage 2 cell")
    pooling.add_argument("--fit", type=Path, required=True,
                         help="the rows the cell was fitted on, which the naive population is "
                              "built from: identifier and sequence, no label")
    pooling.add_argument("--rows", type=Path, required=True, help="identifier and sequence, no label")
    pooling.add_argument("--id-column", required=True)
    pooling.add_argument("--sequence-column", required=True)
    pooling.add_argument("--out", type=Path, required=True)

    for name, purpose in (("reproduce", "the selection on the fitting rows, from two starting sets"),
                          ("select", "Tier 1's half-sample selections under one label set"),
                          ("refit", "the leave-one-out trees")):
        mode = modes.add_parser(name, help=purpose)
        mode.add_argument("--cell", type=Path, required=True, help="the Stage 2 cell")
        mode.add_argument("--pool", type=Path, required=True, help="what `pool` wrote for D")
        mode.add_argument("--prepared", type=Path, required=True,
                          help="the other side's Step 0: kept columns, half-samples, rows, labels")
        mode.add_argument("--out", type=Path, required=True)
        if name in ("reproduce", "select"):
            mode.add_argument("--n-target", type=int, required=True)
        if name == "select":
            mode.add_argument("--label-set", required=True)
            mode.add_argument("--workers", type=int, required=True)
            mode.add_argument("--memory", type=int, default=None,
                              help="the bytes the panel may hold, workers included; fewer workers "
                                   "run where fewer fit")
            mode.add_argument("--threads-file", type=Path, default=None,
                              help="a file holding how many selections may run at once, looked at "
                                   "as each starts, which the driver changes as threads free up "
                                   "and are taken back (default: --workers)")
            mode.add_argument("--force", action="store_true")
        if name == "refit":
            mode.add_argument("--plan", type=Path, required=True,
                              help="JSON list of {label, columns, cell}: each tree to fit and where")

    timing = modes.add_parser("time", help="how long the tree takes on samples of a cell's rows")
    timing.add_argument("--inputs", type=Path, required=True,
                        help="what a cell would be handed (`write_inputs` of "
                             "`seqmaestro/pipeline/vendored/deft.py`)")
    timing.add_argument("--target-column", required=True)
    timing.add_argument("--target-kind", required=True, choices=["binary", "continuous"])
    timing.add_argument("--sequence-column", required=True)
    timing.add_argument("--weight-column", default=None)
    timing.add_argument("--config-override", action="append", default=[])
    timing.add_argument("--first-rows", type=int, required=True)
    timing.add_argument("--enough", type=float, required=True,
                        help="a fit this long, in seconds, is timed well enough")
    timing.add_argument("--budget", type=float, required=True,
                        help="the seconds of fitting the samples may take together")
    timing.add_argument("--out", type=Path, required=True)

    args = top.parse_args()
    return {"reduce": reduce, "predict": predict, "tree": tree, "pool": pool,
            "reproduce": reproduce, "select": select, "refit": refit,
            "time": time_fits}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
