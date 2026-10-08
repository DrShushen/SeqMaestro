"""Reading and writing the shared feature record, and the values behind it.

`docs/feature_record.md` is the maintained account of what a record holds and why. Keep the two
in step.

Every arm emits the same files, so the stability estimator and the judge read one format and never
need to know which arm produced a row:

    features.json        the records, under a header that describes the fit
    feature_values.npz   each feature evaluated on the reference split
    effect_values.npz    each feature over the training rows, with the labels (bundles only)

**Values are evaluated once, at run time, and stored.** Re-executing a generated feature function
later would need the code guard, the exact environment and reproducible generation, and is
impossible for arms whose features are not executable code. Storing the values reduces stability
matching to loading arrays and correlating them, works identically for every arm, and lets the
matching run without executing any model-written code. The reference split is **validation**.
Matching uses no labels, and validation is already the split used for model selection, so
evaluating features on it exposes nothing new.

**The writer may not be able to import this module.** DEFT's and IGSR's arms run in their methods'
conda environments, which have no `seqmaestro`, and their adapters write conforming JSON by hand.
That is why the record is flat and plain-typed, and why `load_features` validates what it reads: a
writer that has drifted from the schema is caught on the first read instead of in a results table.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from seqmaestro import files
from seqmaestro.features.base import PROVENANCES, FeatureRecord
from seqmaestro.harness.direction import DIRECTIONS, NON_MONOTONE
from seqmaestro.target import target_for

SCHEMA_VERSION = 1

# Free text, because the unit is whatever the fitted model reports. Never compare across arms: use
# `rank_by_importance`. Recorded once per file, since every record in the file shares it.
IMPORTANCE_KINDS = (
    # l1_logistic and l1_linear, on standardised columns: signed, per standard deviation.
    "l1_coefficient_per_sd",
    "xgb_gain",             # xgboost
    "mean_abs_shape",       # ebm: shape function magnitude
    "impurity_decrease",    # deft_or_cart
    "deft_split_score",     # deft_or_cart, under a finder other than DEFT's two
    "mean_abs_shapley",     # pysr: mean absolute Shapley value of the equation's output
    "igsr_influence",       # igsr: what removing the term from its search's fit costs
    # A least-squares model on an IGSR arm's terms, fitted when the arm is reduced or refitted.
    "least_squares_coefficient_per_sd",
)


def save_features(
    path: str | Path,
    records: list[FeatureRecord],
    importance_kind: str | None = None,
    **header,
) -> Path:
    """Write `features.json`. Extra keyword arguments join the header verbatim."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "importance_kind": importance_kind,
        **header,
        "features": [_as_dict(record) for record in records],
    }
    return files.write_json(path, payload)


def load_features(path: str | Path) -> tuple[list[FeatureRecord], dict]:
    """Read `features.json`, validating it. Returns the records and the header."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"{path}: not a feature file (no `features` key).")

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema version {version!r}, expected {SCHEMA_VERSION}. Written by something "
            f"that has drifted from the current record. Reconcile the two; do not coerce."
        )

    header = {key: value for key, value in payload.items() if key != "features"}
    # A file that states its kind of target holds that kind's direction words. One that states
    # none, as a file written by hand may, is held to the whole vocabulary.
    stated = header.get("target_kind")
    words = (*target_for(stated).directions, NON_MONOTONE) if stated else DIRECTIONS
    return [_from_dict(entry, path, words) for entry in payload["features"]], header


#: What a record can be ranked by. `importance` is the one each model reports, which the `Model`
#: protocol warns is never comparable across models. `permutation_importance` is computed
#: identically for every arm, so use it when comparing arms with each other. The default is the
#: model's.
RANK_BY = ("importance", "permutation_importance")


def rank_by_importance(records: list[FeatureRecord], by: str = "importance") -> list[int]:
    """Within-run ranks, 1 = most important. The only form comparable between arms.

    Ranks by absolute importance: an L1 coefficient is signed, and a large negative one is as
    influential as a large positive one. Records with no importance rank last, in the order given.

    `by` selects which importance; see `RANK_BY`.
    """
    if by not in RANK_BY:
        raise ValueError(f"Cannot rank by {by!r}; expected one of {', '.join(RANK_BY)}.")
    scored = [
        (abs(value) if (value := getattr(record, by)) is not None else None, position)
        for position, record in enumerate(records)
    ]
    order = sorted(
        range(len(records)),
        key=lambda i: (scored[i][0] is None, -(scored[i][0] or 0.0), i),
    )

    ranks = [0] * len(records)
    for rank, position in enumerate(order, start=1):
        ranks[position] = rank
    return ranks


#: How many features make up the comparable view of a discovery. This is a stated convention.
#: Models disagree by orders of magnitude about how many features they "use": an EBM fits a shape
#: function to every input and so selects all of them, while a shallow tree names a handful. With
#: set sizes that different, the interpretability and stability axes would measure set size instead
#: of agreement, and a reviewer cannot assess a thousand features in any case. For the arms that do
#: select, report the full set alongside, since it is a result in its own right.
DEFAULT_TOP_N = 20


def top_features(records: list[FeatureRecord], n: int = DEFAULT_TOP_N,
                 by: str = "importance") -> list[FeatureRecord]:
    """The `n` most important features, most important first.

    By *absolute* importance, since a signed coefficient's sign gives its direction and only its
    magnitude gives its strength. A model that used fewer than `n` features contributes all of
    them, and a short list is never padded.

    `by` selects which importance; see `RANK_BY`.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")

    ranks = rank_by_importance(records, by=by)
    return [record for _, record in sorted(zip(ranks, records, strict=True), key=lambda pair: pair[0])][:n]


def definition_digest(record: FeatureRecord) -> str:
    """A stable key for "computes the same thing", for exact matching before behavioural matching.

    For generated code this is the function body with names, formatting, comments and docstring
    normalised away, so two candidates that differ only in what the model chose to call them hash
    alike. For a fixed-vocabulary feature there is no code, so the recipe is the definition.
    """
    normalised = _normalise_source(record.source) if record.source else None
    if normalised is None:
        normalised = json.dumps({"kind": record.kind, "params": record.params}, sort_keys=True)
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def save_feature_values(path: str | Path, names: list[str], values: np.ndarray) -> Path:
    """Write `feature_values.npz`, the reference-split evaluation behind the records."""
    path = Path(path)
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"{values.shape} values for {len(names)} features; expected (n_rows, {len(names)}).")

    path.parent.mkdir(parents=True, exist_ok=True)
    return files.write_arrays(path, names=np.asarray(names, dtype=object),
                              values=values.astype(np.float32))


def load_feature_values(path: str | Path) -> tuple[list[str], np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as loaded:
        return list(loaded["names"]), loaded["values"]


def load_effect_values(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as loaded:
        return list(loaded["names"]), loaded["values"], loaded["y"]


# ------------------------------------------------------------------------------------ internals

#: Every field of a record. `_from_dict` requires only name, description and kind, and the rest are
#: optional. A writer (an adapter in another environment, a hand-written file) therefore states
#: only what it knows, and a field added here does not have to be added to every writer.
_FIELDS = (
    "name", "description", "kind", "params", "source", "rationale", "origin", "importance",
    "permutation_importance", "direction", "marginal_effect", "provenance", "extra"
)


def _as_dict(record: FeatureRecord) -> dict:
    return {field: getattr(record, field) for field in _FIELDS}


def _from_dict(entry: dict, path, directions: tuple[str, ...]) -> FeatureRecord:
    """One record of a feature file, checked. `directions` are the words its direction may be."""
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: a feature entry is {type(entry).__name__}, expected an object.")

    unknown = set(entry) - set(_FIELDS)
    if unknown:
        raise ValueError(
            f"{path}: unknown field(s) {sorted(unknown)} on feature {entry.get('name')!r}. The "
            f"record is kept narrow: method-specific detail belongs in that method's own "
            f"artifact, not here."
        )
    missing = {"name", "description", "kind"} - set(entry)
    if missing:
        raise ValueError(f"{path}: feature {entry.get('name')!r} is missing {sorted(missing)}.")

    # Checked on read, because a foreign adapter writes this field by hand. An unknown word, or
    # the word of the other kind of target, would otherwise travel all the way to a judge prompt
    # and a report table before anything noticed.
    if entry.get("direction") is not None and entry["direction"] not in directions:
        raise ValueError(
            f"{path}: feature {entry.get('name')!r} has direction {entry['direction']!r}, which is "
            f"not one of {', '.join(directions)}."
        )
    if entry.get("provenance") is not None and entry["provenance"] not in PROVENANCES:
        raise ValueError(
            f"{path}: feature {entry.get('name')!r} has provenance {entry['provenance']!r}, which "
            f"is not one of {', '.join(PROVENANCES)}."
        )

    return FeatureRecord(
        name=entry["name"],
        description=entry["description"],
        kind=entry["kind"],
        params=entry.get("params") or {},
        source=entry.get("source"),
        rationale=entry.get("rationale"),
        origin=entry.get("origin"),
        importance=entry.get("importance"),
        permutation_importance=entry.get("permutation_importance"),
        direction=entry.get("direction"),
        marginal_effect=entry.get("marginal_effect"),
        provenance=entry.get("provenance"),
        extra=entry.get("extra") or {},
    )


def _normalise_source(source: str) -> str | None:
    """The function body with everything that does not change what it computes stripped out.

    This is the normalisation DEFT uses for candidate identity. Keep the two identical, so that a
    feature hashes the same on both sides of the file boundary.
    """
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None

    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None

    body = functions[-1].body
    # Parsing drops comments, but a docstring is a real statement, so it is dropped explicitly.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    return ast.dump(ast.Module(body=body, type_ignores=[])) if body else None
