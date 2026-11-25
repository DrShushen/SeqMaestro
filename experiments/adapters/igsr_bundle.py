#!/usr/bin/env python3
"""Run IGSR and hand the result back as an arm bundle. **Runs in `SeqMaestro_Method_igsr`.**

    $IGSR experiments/adapters/igsr_bundle.py fit \\
        --train fit.csv --val val.csv --choose-on choose_on.csv --effect effect.csv \\
        --columns columns.json --target-column <column> --target-kind continuous --task <dataset> \\
        --seed <seed> --out-bundle <dir> [--config-override llm=seqmaestro_openai ...]
    $IGSR experiments/adapters/igsr_bundle.py predict \\
        --bundle <dir> --rows rows.csv --id-column <column> --out scores.npz
    $IGSR experiments/adapters/igsr_bundle.py evaluate --bundle <dir> --columns columns.json \\
        [--names names.json] --rows rows.csv --id-column <column> --out values.npz

This is the far side of the environment boundary. It imports `igsr` inside its three modes, so that
what it builds from IGSR's result can be tested from SeqMaestro's environment, and it never imports
`seqmaestro`. The bundle format belongs to SeqMaestro, so the adapter lives here and the vendored
copy stays standalone.

**`fit` runs IGSR's search in process, in its function form**: each term is a Python function the
language model writes over the rows, which hold the whole sequence (at whatever length it comes)
and the covariates the cell was handed (`columns.json`). The search fits on the fitting rows and
chooses on the validation rows (the iteration with the lowest error, and the terms it prunes), and
its prompts show the language model the error and each term's influence there. On those rows it is
given the labels `--choose-on` holds, and `--val` keeps the real labels the model is scored
against. The two differ on a control arm, whose labels to choose on are shuffled as its fitting
rows' are, so its search never sees the real target. IGSR then refits the chosen terms by least
squares on the fitting rows, and that model is saved as `model.json` and predicts the validation
rows. The held-out split is never handed over. IGSR is told about the data from the dataset's
description in `experiments/datasets/<task>.json`, which the proposer and the judge also read: the
task and the target in its data dictionary, and the sequence layout and each covariate's
description beside the column they describe.

**A term becomes a feature record** (`records_for`): its name, description and rationale as the
model wrote them, its code as `source`, kind `llm_generated`, origin `igsr`, provenance `generated`.
A term that only reads a handed column unchanged (`return X["column"]`) is that column, and comes
back with the recipe and description `columns.json` recorded for it, provenance `selected`, as the
DEFT adapter returns a covariate its tree splits on. Its `importance` is IGSR's own number, the
term's influence: the rise in validation mean squared error when its weight is set to zero, against
the labels the search chose on. Its `direction` is the sign of its weight. Its
`permutation_importance`, the measure every arm reports, is the fall in validation R-squared when
the term's values are shuffled, against the real labels.

**Every term the search accepted** is written to `terms.json`, with the round it first appeared in.
The reliability panel takes them as the arm's candidate pool. **The fit the search made** is
recorded in `arm.json` (`fit`: ordinary least squares, or ridge or lasso at IGSR's alpha, with or
without an intercept), so the later stages refit the arm as IGSR fitted it.

`predict` serves the reveal: it loads `model.json` and predicts the rows it is handed, which have
an identifier column and no target, and writes the identifiers back beside the scores.

`evaluate` serves every later stage: it runs the named terms of a cell's search (all of them when
none are named) on the rows it is handed, and writes their values, each term's recipe as its record
would state it, and the identifiers beside the values. The model-written code runs only here, under
the guard and timeout the search used, and the least-squares fits on the values are made on the
other side.

IGSR fits unweighted least squares, so a dataset with per-row weights is refused, and it fits a
continuous target only (a binary one would need a logistic fit and influence through its link).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The bundle contract this adapter writes. `seqmaestro.harness.bundle` holds the definition. The
# constants are copied, since importing them would create the cross-environment dependency that
# exchanging files avoids, and `read_bundle` rejects a bundle whose versions do not match.
BUNDLE_VERSION = 3
FEATURE_SCHEMA_VERSION = 1

#: The only kind of target IGSR fits.
CONTINUOUS = "continuous"
#: The file names inside the bundle: the saved model, and every term the search accepted.
MODEL_FILE, TERMS_FILE = "model.json", "terms.json"
#: IGSR's per-seed log of the search, prompts and replies included.
LOG_FILE = "igsr.jsonl"
#: How the arm writes its terms: the function form, over the rows as a frame.
TERM_LANGUAGE = "function"

#: What a record says about itself, restated from SeqMaestro's vocabulary and checked against it by
#: a test.
ORIGIN = "igsr"
KIND_GENERATED = "llm_generated"
PROVENANCE_GENERATED, PROVENANCE_SELECTED = "generated", "selected"
IMPORTANCE_KIND = "igsr_influence"
DIRECTION_KIND = "least_squares_coefficient_sign"
#: IGSR's fits, by the scikit-learn estimator it calls, with the penalty each adds, as
#: `seqmaestro.models.sparse.LeastSquares` takes it.
PENALTIES = {"LinearRegression": None, "Lasso": "l1", "Ridge": "l2"}
INCREASES_TARGET, DECREASES_TARGET = "increases_target", "decreases_target"

#: The dataset descriptions, one per dataset, which the proposer and the judge also read.
DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


# ================================================================================
# What IGSR is told about the data
# ================================================================================


def read_description(task: str, dataset_dir: Path) -> dict:
    """The dataset's description, refused early when there is none: a wrong or missing description
    fails silently, since the model is told something else and answers anyway."""
    path = dataset_dir / f"{task}.json"
    if not path.exists():
        available = sorted(p.stem for p in dataset_dir.glob("*.json"))
        raise SystemExit(f"No dataset description {task!r} in {dataset_dir}. Available: "
                         f"{', '.join(available) or '(none)'}.")
    return json.loads(path.read_text())


#: How IGSR's prompts introduce the guidance the person running the analysis gave, in the words the
#: proposer's requests use (`seqmaestro.proposer.prompt.GUIDANCE_INTRODUCTION`), which this side
#: cannot import.
GUIDANCE_INTRODUCTION = ("The person running this analysis gives this guidance on the form of the "
                         "features:")


def data_dictionary(description: dict, target_column: str, guidance: str | None) -> str:
    """IGSR's data dictionary: the task, what the target column holds, and the guidance on the form
    of the terms where the person running the analysis gave any. IGSR states the dictionary in
    every prompt that asks for terms, so the guidance reaches each of them."""
    target = description.get("target_name")
    stated = f"\n\nThe target column `{target_column}` is {target}." if target else ""
    guided = f"\n\n{GUIDANCE_INTRODUCTION} {guidance}" if guidance is not None else ""
    return description["task"] + stated + guided


def column_descriptions(manifest: dict, description: dict) -> dict[str, str]:
    """What each input column is, for the function form's prompt: the sequence column by the
    dataset's sequence layout, and every handed column as `columns.json` describes it."""
    described = {name: entry["description"] for name, entry in manifest["columns"].items()
                 if entry.get("description")}
    layout = description.get("sequence_layout")
    if layout:
        described[manifest["sequence"]["column"]] = layout
    return described


def search_validation(val: pd.DataFrame, choose_on: pd.DataFrame, inputs: list[str],
                      target_column: str) -> pd.DataFrame:
    """The validation split IGSR's search is given: the validation rows' inputs, with the labels to
    choose on as their target in place of the real ones. Those must be the target's column alone,
    one label per validation row, since they are matched to the rows by position."""
    if list(choose_on.columns) != [target_column] or len(choose_on) != len(val):
        raise SystemExit(f"--choose-on holds {list(choose_on.columns)} over {len(choose_on)} rows; "
                         f"it must hold {target_column!r} alone, one label for each of the "
                         f"{len(val)} validation rows, in their order.")
    return val[inputs].assign(**{target_column: choose_on[target_column].to_numpy(dtype=float)})


# ================================================================================
# What a term becomes
# ================================================================================


def _without_docstring(body: list) -> list:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def bare_column(code: str) -> str | None:
    """The column a term reads unchanged, or None: a term whose code is one function returning
    `X["column"]`, or `X["column"].astype(float)`, and nothing else."""
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None
    body = _without_docstring(module.body[0].body)
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    value = body[0].value
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "astype" and len(value.args) == 1 and not value.keywords
            and isinstance(value.args[0], ast.Name) and value.args[0].id == "float"):
        value = value.func.value
    if (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
            and value.value.id == "X" and isinstance(value.slice, ast.Constant)
            and isinstance(value.slice.value, str)):
        return value.slice.value
    return None


def fit_record(estimator: type, settings: dict) -> dict:
    """The fit IGSR's search makes, as the later stages refit it: its penalty, its alpha and whether
    it has an intercept, from the estimator IGSR's configuration chose and the settings it passes.
    A fit that is not least squares (IGSR's `no_optimization`, which fixes every weight at 1) is
    refused."""
    name = estimator.__name__
    if name not in PENALTIES:
        raise SystemExit(f"IGSR is configured to fit with {name}, which is not a least-squares fit, "
                         f"so the later stages could not refit it. Choose experiment."
                         f"optimization_method linear, ridge or lasso.")
    params = estimator(**settings).get_params()
    penalty = PENALTIES[name]
    return {"penalty": penalty, "alpha": None if penalty is None else float(params["alpha"]),
            "fit_intercept": bool(params["fit_intercept"])}


def direction_of(weight: float) -> str | None:
    """The sign of a least-squares weight, which says everything the linear model does with its
    term. None for a weight of zero, which does not move the prediction."""
    if weight > 0:
        return INCREASES_TARGET
    if weight < 0:
        return DECREASES_TARGET
    return None


def recipe_of(term: dict, handed: dict[str, dict]) -> dict:
    """What a term is, as its record says: a bare read of a handed column is that column, with the
    recipe and description `columns.json` recorded for it, and any other term is the code the model
    wrote."""
    column = bare_column(term["code"])
    recipe = handed.get(column) if column is not None else None
    if recipe:
        return {"kind": recipe["kind"], "params": recipe["params"], "source": None,
                "description": recipe.get("description") or term["description"],
                "provenance": PROVENANCE_SELECTED}
    return {"kind": KIND_GENERATED, "params": {}, "source": term["code"],
            "description": term["description"], "provenance": PROVENANCE_GENERATED}


def records_for(model: dict, influence: dict[str, float], permutation: dict[str, float],
                handed: dict[str, dict]) -> list[dict]:
    """One feature record per term of the saved model (`FittedModel.to_dict`), in its order.

    `influence` is each term's rise in validation MSE with its weight set to zero, against the labels
    the search chose on, `permutation` its fall in validation R-squared when shuffled, against the
    real labels, and `handed` the columns `columns.json` recorded, whose recipes a bare read of one
    recovers.
    """
    records = []
    for index, term in enumerate(model["functions"]):
        weight = float(model["weights"][index][0])
        recipe = recipe_of(term, handed)
        extra = {"weight": weight, "first_seen": term["first_seen"],
                 "undefined_share": term["undefined_share"], "fill": term["fill"],
                 **({"repaired": True} if term["repaired"] else {}),
                 **({"renamed_from": term["renamed_from"]} if term["renamed_from"] else {})}
        records.append({
            "name": term["name"],
            "description": recipe["description"],
            "kind": recipe["kind"],
            "params": recipe["params"],
            "source": recipe["source"],
            "rationale": term["rationale"],
            "origin": ORIGIN,
            "importance": influence.get(term["name"]),
            "permutation_importance": permutation.get(term["name"]),
            "direction": direction_of(weight),
            "provenance": recipe["provenance"],
            "extra": extra,
        })
    return records


def permutation_importances(values: np.ndarray, weights: np.ndarray, intercept: float,
                            y: np.ndarray, names: list[str], *, repeats: int,
                            rng: np.random.Generator) -> dict[str, float]:
    """How far validation R-squared falls when each term's values are shuffled, as the harness
    measures it for every arm: the mean over `repeats` shuffles. The model is linear, so a shuffled
    prediction is the intercept plus the terms' values times their weights."""
    from sklearn.metrics import r2_score

    y = np.asarray(y, dtype=float).ravel()
    if repeats < 1 or not names or np.ptp(y) == 0:
        return {}
    weights = np.asarray(weights, dtype=float).ravel()
    baseline = r2_score(y, intercept + values @ weights)
    out = {}
    for column, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            shuffled = values.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            drops.append(baseline - r2_score(y, intercept + shuffled @ weights))
        out[name] = float(np.mean(drops))
    return out


# ================================================================================
# The three modes
# ================================================================================


def predict(args) -> int:
    """Load the model a bundle saved, predict the rows given, and write identifiers and scores."""
    from igsr.method.fitted_model import FittedModel

    arm = json.loads((args.bundle / "arm.json").read_text())
    if arm.get("bundle_version") != BUNDLE_VERSION or "fitted_model" not in arm:
        raise SystemExit(f"{args.bundle}/arm.json is not a version {BUNDLE_VERSION} bundle naming "
                         f"its fitted model, so there is no model here to predict with.")
    model = FittedModel.from_dict(json.loads((args.bundle / arm["fitted_model"]).read_text()))
    rows = pd.read_csv(args.rows)
    if args.id_column not in rows.columns:
        raise SystemExit(f"the rows have no identifier column {args.id_column!r}; columns are "
                         f"{list(rows.columns)}.")
    values, failed = model.term_values(rows.drop(columns=[args.id_column]))
    for name, error in failed.items():
        print(f"  term {name!r} failed on these rows and takes its fill: {error}", file=sys.stderr)
    scores = (model.intercept[None, :] + values @ model.weights)[:, 0]
    np.savez_compressed(args.out, row_id=rows[args.id_column].to_numpy(), score=scores)
    print(f"predicted {len(scores)} rows with {arm['fitted_model']}", flush=True)
    return 0


def evaluate(args) -> int:
    """Named terms of a cell's search run on the rows given: their values, each term's recipe, and
    the identifiers, in the rows' order."""
    from igsr.method.fitted_model import SavedTerms

    model = json.loads((args.bundle / MODEL_FILE).read_text())
    accepted = {term["name"]: term for term in json.loads((args.bundle / TERMS_FILE).read_text())}
    names = json.loads(args.names.read_text()) if args.names else list(accepted)
    missing = [name for name in names if name not in accepted]
    if missing:
        raise SystemExit(f"{args.bundle}: no accepted term named {missing[:5]}.")
    handed = json.loads(args.columns.read_text())["columns"]
    terms = [accepted[name] for name in names]
    recipes = [recipe_of(term, handed) for term in terms]
    rows = pd.read_csv(args.rows)
    if args.id_column not in rows.columns:
        raise SystemExit(f"the rows have no identifier column {args.id_column!r}.")
    values, failed = SavedTerms(terms, model["code_guard"], model["timeout_seconds"]).values(
        rows.drop(columns=[args.id_column]))
    for name, error in failed.items():
        print(f"  term {name!r} failed on these rows and takes its fill: {error}", file=sys.stderr)
    np.savez_compressed(
        args.out, row_id=rows[args.id_column].to_numpy(), names=np.array(names, dtype=str),
        kinds=np.array([recipe["kind"] for recipe in recipes], dtype=str),
        params=np.array([json.dumps(recipe["params"], sort_keys=True) for recipe in recipes],
                        dtype=str),
        descriptions=np.array([recipe["description"] for recipe in recipes], dtype=str),
        values=values, failed=np.array(json.dumps(failed)))
    print(f"evaluated {len(names)} terms on {len(rows)} rows", flush=True)
    return 0


def fit(args) -> int:
    """Run IGSR on the fitting rows, choosing on the validation rows against the labels to choose
    on, and write the bundle, scored against the real labels."""
    from igsr.compose import compose_config
    from igsr.dataset import DataBundle
    from igsr.logging_setup import configure_logging
    from igsr.method.igsr import igsr
    from igsr.method.igsr_utils import get_optimization_method, ols_influence
    from omegaconf import open_dict

    started = time.time()
    out = args.out_bundle
    out.mkdir(parents=True, exist_ok=True)
    if args.target_kind != CONTINUOUS:
        raise SystemExit(f"IGSR fits a continuous target by least squares; this arm's target is "
                         f"{args.target_kind}.")
    if args.weight_column is not None:
        raise SystemExit("IGSR fits unweighted least squares, so a dataset with per-row weights "
                         "cannot be given to it.")
    fixed = [item for item in args.config_override if item.split("=")[0].strip() in
             ("experiment.term_language", "logging.log_file")]
    if fixed:
        raise SystemExit(f"{', '.join(fixed)}: the arm writes its terms as functions of the rows "
                         f"and logs into its bundle, so neither is an override.")

    manifest = json.loads(args.columns.read_text())
    sequence_column = manifest["sequence"]["column"]
    description = read_description(args.task, args.dataset_dir)
    frames = {name: pd.read_csv(path) for name, path in
              (("train", args.train), ("val", args.val), ("effect", args.effect))}
    inputs = [sequence_column, *manifest["columns"]]
    for name, frame in frames.items():
        missing = [column for column in [*inputs, args.target_column]
                   if column not in frame.columns]
        if missing:
            raise SystemExit(f"{name}: no column(s) {missing}, so the inputs and columns.json "
                             f"disagree.")
    searched = search_validation(frames["val"], pd.read_csv(args.choose_on), inputs,
                                 args.target_column)

    cfg = compose_config([f"experiment.term_language={TERM_LANGUAGE}", *args.config_override])
    fit_used = fit_record(*get_optimization_method(cfg))
    with open_dict(cfg):
        cfg.logging.log_file = str((out / LOG_FILE).resolve())
    configure_logging(log_level="INFO")
    bundle = DataBundle(
        name=args.task, data_settings=None,
        dataset_train=frames["train"][[*inputs, args.target_column]],
        dataset_validation=searched,
        dataset_test=None, target_columns=[args.target_column],
        data_dictionary=data_dictionary(description, args.target_column, args.guidance),
        column_descriptions=column_descriptions(manifest, description),
    )
    print(f"IGSR ({TERM_LANGUAGE} terms, seed {args.seed}) on {len(frames['train'])} fitting rows, "
          f"choosing on {len(frames['val'])} validation rows", flush=True)
    result = igsr(cfg, args.seed, bundle)
    model = result["best_model"]
    saved = model.to_dict()
    names = list(model.terms)

    train_values, _ = model.term_values(frames["train"][inputs])
    val_values, val_failed = model.term_values(frames["val"][inputs])
    effect_values, effect_failed = model.term_values(frames["effect"][inputs])
    y_train = frames["train"][args.target_column].to_numpy(dtype=float)
    # The real labels, which the model is scored against and `read_bundle` checks against the split.
    y_val = frames["val"][args.target_column].to_numpy(dtype=float)
    y_chosen_on = searched[args.target_column].to_numpy(dtype=float)
    val_score = model.predict(frames["val"][inputs])[:, 0]

    influence: dict[str, float] = {}
    if names:
        # IGSR's influence of each final term on the validation rows, from the same fit it saved,
        # against the labels its search chose on, as the search measured it.
        found = ols_influence(cfg, train_values, y_train, val_values, y_chosen_on)
        influence = {name: float(found.delta_validation[index, 0])
                     for index, name in enumerate(names)}
    permutation = permutation_importances(val_values, model.weights, float(model.intercept[0]),
                                          y_val, names, repeats=args.permutation_repeats,
                                          rng=np.random.default_rng(args.permutation_seed))
    records = records_for(saved, influence, permutation, manifest["columns"])

    np.savez_compressed(out / "predictions.npz", val_true=y_val, val_score=val_score)
    (out / "features.json").write_text(json.dumps(
        {"schema_version": FEATURE_SCHEMA_VERSION, "importance_kind": IMPORTANCE_KIND,
         "direction_kind": DIRECTION_KIND, "features": records}, indent=2, default=str) + "\n")
    np.savez_compressed(out / "feature_values.npz", names=np.array(names, dtype=object),
                        values=val_values.astype(np.float32))
    # The labels travel with the values, so the two cannot be handed back unaligned.
    np.savez_compressed(out / "effect_values.npz", names=np.array(names, dtype=object),
                        values=effect_values.astype(np.float32),
                        y=frames["effect"][args.target_column].to_numpy())
    (out / MODEL_FILE).write_text(json.dumps(saved, indent=2) + "\n")
    (out / TERMS_FILE).write_text(json.dumps(result["term_records"], indent=2) + "\n")
    (out / "arm.json").write_text(json.dumps({
        "bundle_version": BUNDLE_VERSION,
        # What `predict` loads at the reveal, relative to the bundle so it moves with it.
        "fitted_model": MODEL_FILE,
        "model": ORIGIN,
        "term_language": TERM_LANGUAGE,
        "environment": "SeqMaestro_Method_igsr",
        "python": sys.version.split()[0],
        "seed": args.seed,
        "n_fit_rows": len(frames["train"]),
        "n_features_used": len(records),
        "intercept": float(model.intercept[0]),
        # How the later stages refit the arm: as the search fitted it.
        "fit": fit_used,
        "wall_clock_seconds": round(time.time() - started, 1),
        "config_overrides": list(args.config_override),
        # The guidance on the form of the terms that the prompts were given, or None.
        "guidance": args.guidance,
        "search": {
            "best_iter_id": result["best_iter_id"],
            # Against the labels the search chose on, which differ from the real ones on a control.
            "best_validation_mse": result["best_metric"],
            "terms_accepted": len(result["term_records"]),
            "proposals": result["proposal_stats"],
            "failed_after_acceptance": {name: sorted(splits)
                                        for name, splits in result["term_failures"].items()},
            "failed_on_reference_rows": sorted({*val_failed, *effect_failed}),
            "compute": result["compute_profiler"],
        },
    }, indent=2, default=str) + "\n")
    print(f"bundle written to {out}: {len(records)} terms, {len(y_val)} validation rows, "
          f"{len(frames['effect'])} rows for the marginal effect", flush=True)
    return 0


def main() -> int:
    top = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = top.add_subparsers(dest="mode", required=True)

    predicting = modes.add_parser("predict", help="predict rows with the model a bundle saved")
    predicting.add_argument("--bundle", type=Path, required=True)
    predicting.add_argument("--rows", type=Path, required=True,
                            help="CSV of the rows: an identifier column and the columns the model "
                                 "reads, and no target")
    predicting.add_argument("--id-column", required=True, help="echoed back beside the scores")
    predicting.add_argument("--out", type=Path, required=True)

    evaluating = modes.add_parser("evaluate", help="run a cell's accepted terms on rows")
    evaluating.add_argument("--bundle", type=Path, required=True)
    evaluating.add_argument("--columns", type=Path, required=True,
                            help="the cell's columns.json, whose handed columns a bare read "
                                 "recovers")
    evaluating.add_argument("--names", type=Path, default=None,
                            help="JSON list of the terms to evaluate; every accepted term if "
                                 "absent")
    evaluating.add_argument("--rows", type=Path, required=True,
                            help="CSV of the rows: an identifier column and the columns the terms "
                                 "read, and no target")
    evaluating.add_argument("--id-column", required=True, help="echoed back beside the values")
    evaluating.add_argument("--out", type=Path, required=True)

    parser = modes.add_parser("fit", help="run IGSR and write a bundle")
    parser.add_argument("--train", type=Path, required=True, help="the fitting rows")
    parser.add_argument("--val", type=Path, required=True,
                        help="the validation rows, with the real labels the model is scored against")
    parser.add_argument("--choose-on", type=Path, required=True,
                        help="the labels the search chooses against on the validation rows: the "
                             "target's column alone, one label per row of --val in its order "
                             "(shuffled on a control arm)")
    parser.add_argument("--effect", type=Path, required=True,
                        help="every training row with its real target, where marginal_effect is "
                             "measured")
    parser.add_argument("--columns", type=Path, required=True,
                        help="the cell's columns.json: the sequence column and each handed column")
    parser.add_argument("--out-bundle", type=Path, required=True)
    # Required, with no default: which column is the target, and what it holds, are facts about
    # the data.
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--target-kind", required=True, help="must be continuous")
    parser.add_argument("--task", required=True, help="the dataset's description, in --dataset-dir")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    # Absent where the run gives no guidance, which is the default.
    parser.add_argument("--guidance", default=None,
                        help="what the person running the analysis asks of the form of the terms, "
                             "added to IGSR's data dictionary")
    parser.add_argument("--seed", type=int, required=True,
                        help="seeds IGSR's tree search and its examples")
    parser.add_argument("--weight-column", default=None,
                        help="refused: IGSR fits unweighted least squares")
    parser.add_argument("--permutation-repeats", type=int, required=True,
                        help="shuffles per term for the cross-arm permutation importance; 0 skips "
                             "it")
    parser.add_argument("--permutation-seed", type=int, required=True)
    parser.add_argument("--config-override", action="append", default=[],
                        help="IGSR's own override, repeatable, e.g. experiment.total_budget=10")
    args = top.parse_args()
    return {"predict": predict, "evaluate": evaluate, "fit": fit}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
