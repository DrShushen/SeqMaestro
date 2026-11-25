"""What an arm running in another environment hands back.

A vendored method cannot run in this interpreter, because its dependencies are pinned in a separate
conda environment. Its adapter therefore writes a bundle, a directory of files, using only the
standard library and numpy, and this side reads the bundle, checks it and scores it with the same
code as an arm fitted in Core, so that neither side imports the other:

    <bundle>/
        arm.json             bundle_version and fitted_model, which are required, then whatever
                             the method records (its model's name, the settings it ran with, its
                             environment, its wall clock and token usage)
        predictions.npz      val_true, val_score
        features.json        the features the method used, in the shared record schema
                             (schema_version, importance_kind, direction_kind, features)
        feature_values.npz   names, values: those features evaluated on the reference split
        effect_values.npz    names, values, y: those features over the training rows with the
                             real labels, on which marginal_effect is measured
        <fitted_model>       the model the adapter fitted, in the method's own format

`read_bundle` refuses a bundle whose labels disagree with the split's on any row (`_agree`), one
that holds test predictions, one without the model `fitted_model` names, and one without the
`features` key, whereas `"features": []` is a real result (the method selected nothing).
`describe_contract()` returns the contract as data, and `experiments/adapters/README.md` explains
each rule for whoever writes an adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.features import (
    SCHEMA_VERSION,
    load_effect_values,
    load_feature_values,
    load_features,
)
from seqmaestro.harness.heldout import TEST_ARRAYS
from seqmaestro.target import target_for

#: Bumped when the bundle's layout changes in a way an existing adapter would get wrong, so that an
#: adapter writing an older version fails loudly instead of being misread.
#:
#: A bundle of an older version is refused, never adapted. One without `effect_values.npz` has no
#: training rows to measure `marginal_effect` on, and measuring it on validation for those arms
#: alone would give one column two meanings within one run. One without `fitted_model` can never be
#: scored at the reveal.
BUNDLE_VERSION = 3

PREDICTION_KEYS = ("val_true", "val_score")
#: Arrays a bundle must not hold, because the reveal adds them to the cell.
WITHHELD_KEYS = TEST_ARRAYS
REQUIRED_FILES = ("arm.json", "predictions.npz", "features.json", "feature_values.npz",
                  "effect_values.npz")


@dataclass(frozen=True)
class Bundle:
    """A validated bundle from an arm fitted in another environment."""

    arm: dict
    y_val: np.ndarray
    p_val: np.ndarray
    #: The model the adapter fitted, which its predict mode loads at the reveal.
    fitted_model: Path
    records: list[FeatureRecord]
    header: dict
    feature_names: list[str]
    feature_values: np.ndarray
    effect_values: np.ndarray
    y_effect: np.ndarray

    @property
    def importance_kind(self) -> str | None:
        return self.header.get("importance_kind")

    @property
    def direction_kind(self) -> str | None:
        """What the adapter took its `direction` field from, or `None` when the adapter reports no
        direction. That is allowed, because the marginal effect is computed on this side and does
        not depend on it."""
        return self.header.get("direction_kind")


def read_bundle(path: str | Path, target_kind: str, y_val: np.ndarray | None = None) -> Bundle:
    """Read and validate the bundle of an arm fitted to a target of kind `target_kind`.

    Pass the validation labels to check row alignment. They are optional only so that a bundle can
    be inspected on its own. **`ingest_bundle` always passes them**, because without them an adapter
    that reorders rows goes undetected.
    """
    path = Path(path)
    target = target_for(target_kind)
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path}: incomplete bundle, missing {missing}.")

    arm = json.loads((path / "arm.json").read_text())
    version = arm.get("bundle_version")
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"{path}: bundle_version {version!r}, expected {BUNDLE_VERSION}. The adapter that "
            f"wrote this predates the current contract. Update the adapter, not the check."
        )
    fitted_model = arm.get("fitted_model")
    if not isinstance(fitted_model, str) or not (path / fitted_model).is_file():
        raise ValueError(
            f"{path}: arm.json's `fitted_model` is {fitted_model!r}, which names no file in the "
            f"bundle. The test split is scored at the reveal from that model, so a bundle without "
            f"it can never be scored."
        )

    with np.load(path / "predictions.npz") as loaded:
        absent = [key for key in PREDICTION_KEYS if key not in loaded]
        if absent:
            raise ValueError(f"{path}/predictions.npz: missing {absent}.")
        withheld = [key for key in WITHHELD_KEYS if key in loaded]
        if withheld:
            raise ValueError(
                f"{path}/predictions.npz carries {withheld}. An adapter's fit is never given the "
                f"test split: a model is scored on it once, at the reveal. See "
                f"seqmaestro.harness.heldout."
            )
        arrays = {key: np.asarray(loaded[key]).ravel() for key in PREDICTION_KEYS}

    true, score = arrays["val_true"], arrays["val_score"]
    if true.size != score.size:
        raise ValueError(f"{path}: val_true has {true.size} rows, val_score has {score.size}.")
    if target.probabilities and score.size and (np.nanmin(score) < 0 or np.nanmax(score) > 1):
        raise ValueError(
            f"{path}: val_score is not in [0, 1]. Scores must be probabilities of the positive "
            f"class. Apply a link function in the adapter, not here."
        )

    _check_alignment(path, "val", target.values(true), y_val)

    records, header = load_features(path / "features.json")
    names, values = load_feature_values(path / "feature_values.npz")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"{path}: duplicate feature name(s) {duplicates}. `Featuriser.fit` refuses these for an "
            f"in-process arm and a bundle must not be the way round it: two records under one name "
            f"can differ in provenance (one taken from the pool, one constructed mid-fit), and "
            f"every name-keyed matcher downstream would conflate them."
        )
    if names != [record.name for record in records]:
        raise ValueError(
            f"{path}: feature_values.npz names do not match features.json, in content or order. "
            f"The two files are read positionally, so they cannot disagree."
        )
    if values.shape[0] != arrays["val_true"].size:
        raise ValueError(
            f"{path}: feature values have {values.shape[0]} rows but the reference split has "
            f"{arrays['val_true'].size}."
        )

    effect_names, effect_values, y_effect = load_effect_values(path / "effect_values.npz")
    if effect_names != names:
        raise ValueError(
            f"{path}: effect_values.npz names do not match features.json, in content or order. "
            f"`marginal_effect` is read positionally off this matrix, so a mismatch would give "
            f"every feature another feature's effect."
        )
    if effect_values.shape[0] != y_effect.size:
        raise ValueError(
            f"{path}: effect values have {effect_values.shape[0]} rows against {y_effect.size} "
            f"labels."
        )

    return Bundle(
        arm=arm,
        y_val=target.values(true),
        p_val=arrays["val_score"].astype(float),
        fitted_model=path / fitted_model,
        records=records,
        header=header,
        feature_names=names,
        feature_values=values,
        effect_values=effect_values,
        y_effect=y_effect,
    )


def _check_alignment(path: Path, split: str, reported: np.ndarray, expected: np.ndarray | None):
    if expected is None:
        return
    expected = np.asarray(expected).ravel()
    if reported.size != expected.size:
        raise ValueError(
            f"{path}: {split} has {reported.size} rows, the split has {expected.size}. The adapter "
            f"was given a different set of rows than the one being scored."
        )
    differing = int(np.count_nonzero(~_agree(reported, expected.astype(reported.dtype))))
    if differing:
        raise ValueError(
            f"{path}: {split} labels disagree with the split in {differing} of {expected.size} "
            f"rows. Almost always row order: the adapter must return predictions in the order "
            f"the rows were given, and a reordered bundle scores plausibly and means nothing."
        )


def _agree(reported: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Which rows hold the same value. A continuous target written to an adapter's CSV and read back
    can come back one unit in the last place away, because pandas' default parser does not always
    return the nearest float. Floating-point values therefore agree within a relative tolerance of
    1e-12, whereas a row out of order differs by far more."""
    if np.issubdtype(reported.dtype, np.floating):
        return np.isclose(reported, expected, rtol=1e-12, atol=0.0)
    return reported == expected


def describe_contract() -> dict:
    """The contract as data, so that an adapter can check itself against it in code."""
    return {
        "bundle_version": BUNDLE_VERSION,
        "feature_schema_version": SCHEMA_VERSION,
        "required_files": list(REQUIRED_FILES),
        "prediction_keys": list(PREDICTION_KEYS),
        "withheld_keys": list(WITHHELD_KEYS),
        "required_arm_keys": ["bundle_version", "fitted_model"],
    }
