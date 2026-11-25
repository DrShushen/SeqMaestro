"""The handover contract for arms that run in another environment.

Such an arm's adapter hands back its results as a bundle (a directory of files), which SeqMaestro
reads and checks. These tests write bundles the way an adapter does: by hand, with json and numpy,
and never through `save_features`, because an adapter has no access to SeqMaestro's writers. A test
that used the writers would prove only that the readers can read the writers' output, which says
nothing about the contract.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from seqmaestro.harness.bundle import (
    BUNDLE_VERSION,
    WITHHELD_KEYS,
    describe_contract,
    read_bundle,
)
from seqmaestro.harness.runner import ingest_bundle

#: A column name that matches no real dataset, for the reason given in `tests/test_runner.py`.
TARGET = "outcome"


#: Where the stand-in adapter below saves its model, inside the bundle.
MODEL = "run/model.bin"


def write_bundle(path, y_val, p_val, features=None, values=None,
                 effect_values=None, y_effect=None, **arm):
    """Write a bundle with only what an adapter has: json, numpy and the documented layout."""
    path.mkdir(parents=True, exist_ok=True)
    features = [{"name": "gc_content", "description": "GC fraction", "kind": "deft_split",
                 "params": {"threshold": 0.5}, "source": "def f(X): return X.gc",
                 "rationale": "why", "origin": "llm", "importance": 0.3, "extra": {}}
                ] if features is None else features
    if values is None:
        values = np.linspace(0, 1, len(y_val) * len(features)).reshape(len(y_val), len(features))

    (path / MODEL).parent.mkdir(parents=True, exist_ok=True)
    (path / MODEL).write_bytes(b"a fitted model, in the adapter's own format")
    np.savez_compressed(path / "predictions.npz", val_true=y_val, val_score=p_val)
    (path / "features.json").write_text(json.dumps(
        {"schema_version": 1, "importance_kind": "deft_split_score", "features": features}))
    np.savez_compressed(path / "feature_values.npz",
                        names=np.array([f["name"] for f in features], dtype=object),
                        values=np.asarray(values, dtype=np.float32))
    # The training rows that `marginal_effect` is measured on. They are a different split from the
    # values above, with a different number of rows, so code that conflated the two would fail.
    if y_effect is None:
        y_effect = np.resize(np.asarray(y_val), 2 * len(y_val))
    if effect_values is None:
        effect_values = np.linspace(0, 1, len(y_effect) * len(features)).reshape(
            len(y_effect), len(features))
    np.savez_compressed(path / "effect_values.npz",
                        names=np.array([f["name"] for f in features], dtype=object),
                        values=np.asarray(effect_values, dtype=np.float32),
                        y=np.asarray(y_effect))
    (path / "arm.json").write_text(json.dumps(
        {"bundle_version": BUNDLE_VERSION, "fitted_model": MODEL, "model": "deft_or_cart", **arm}))
    return path


class Case(NamedTuple):
    """Named so a test can take the fields it needs by name."""

    y_val: np.ndarray
    p_val: np.ndarray


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    return Case(rng.integers(0, 2, 40), rng.random(40))


def test_a_well_formed_bundle_reads_back(tmp_path, case):
    bundle = read_bundle(write_bundle(tmp_path / "b", *case), "binary", y_val=case.y_val)
    assert bundle.arm["model"] == "deft_or_cart"
    assert bundle.importance_kind == "deft_split_score"
    assert [r.name for r in bundle.records] == ["gc_content"]
    assert bundle.feature_values.shape == (40, 1)
    assert bundle.fitted_model == tmp_path / "b" / MODEL


def test_reordered_rows_are_refused(tmp_path, case):
    """Reordered rows would yield a plausible metric computed against the wrong labels, which is
    the main failure the contract prevents."""
    shuffled = case.y_val[::-1].copy()
    write_bundle(tmp_path / "b", shuffled, case.p_val)
    with pytest.raises(ValueError, match="labels disagree"):
        read_bundle(tmp_path / "b", "binary", y_val=case.y_val)


def test_a_continuous_target_a_unit_in_the_last_place_away_is_the_same_row(tmp_path):
    """An adapter reads the split from CSV, and pandas' parser can return a continuous value one unit
    in the last place away from the value SeqMaestro holds. Such a value belongs to the same row,
    whereas a row out of order still differs and is refused."""
    y = np.random.default_rng(0).normal(5.0, 1.0, 40)
    nudged = y.copy()
    nudged[::7] = np.nextafter(nudged[::7], np.inf)
    write_bundle(tmp_path / "b", nudged, y)
    read_bundle(tmp_path / "b", "continuous", y_val=y)
    write_bundle(tmp_path / "c", y[::-1].copy(), y)
    with pytest.raises(ValueError, match="labels disagree"):
        read_bundle(tmp_path / "c", "continuous", y_val=y)


def test_wrong_number_of_rows_is_refused(tmp_path, case):
    write_bundle(tmp_path / "b", case.y_val[:20], case.p_val[:20])
    with pytest.raises(ValueError, match="different set of rows"):
        read_bundle(tmp_path / "b", "binary", y_val=case.y_val)


def test_a_bundle_carrying_test_predictions_is_refused(tmp_path, case):
    """An adapter is never given the held-out split, so a bundle holding predictions on it means
    that the adapter saw it. Every number the reveal later produced for that cell would then be
    computed on rows the adapter had already seen."""
    path = write_bundle(tmp_path / "b", *case)
    np.savez_compressed(path / "predictions.npz", val_true=case.y_val, val_score=case.p_val,
                        **dict.fromkeys(WITHHELD_KEYS, case.p_val))
    with pytest.raises(ValueError, match="never given"):
        read_bundle(path, "binary", y_val=case.y_val)


@pytest.mark.parametrize("fitted_model", [None, "run/elsewhere.bin"])
def test_a_bundle_without_its_fitted_model_is_refused(tmp_path, case, fitted_model):
    """The reveal scores the held-out split with the bundle's fitted model, so a cell without one
    could never be scored."""
    path = write_bundle(tmp_path / "b", *case)
    arm = json.loads((path / "arm.json").read_text())
    arm["fitted_model"] = fitted_model
    (path / "arm.json").write_text(json.dumps(arm))
    with pytest.raises(ValueError, match="fitted_model"):
        read_bundle(path, "binary", y_val=case.y_val)


def test_missing_files_are_named(tmp_path, case):
    write_bundle(tmp_path / "b", *case)
    (tmp_path / "b" / "features.json").unlink()
    with pytest.raises(FileNotFoundError, match=re.escape("features.json")):
        read_bundle(tmp_path / "b", "binary")


def test_a_stale_adapter_fails_loudly(tmp_path, case):
    path = write_bundle(tmp_path / "b", *case)
    (path / "arm.json").write_text(json.dumps({"bundle_version": BUNDLE_VERSION - 1}))
    with pytest.raises(ValueError, match="bundle_version"):
        read_bundle(path, "binary")


def test_scores_outside_the_unit_interval_are_refused(tmp_path, case):
    write_bundle(tmp_path / "b", case.y_val, case.p_val * 3)
    with pytest.raises(ValueError, match=re.escape("not in [0, 1]")):
        read_bundle(tmp_path / "b", "binary")


def test_feature_values_must_line_up_with_the_records(tmp_path, case):
    path = tmp_path / "b"
    write_bundle(path, *case)
    np.savez_compressed(path / "feature_values.npz",
                        names=np.array(["something_else"], dtype=object),
                        values=np.zeros((40, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="do not match"):
        read_bundle(path, "binary")


def test_feature_values_must_cover_the_reference_split(tmp_path, case):
    path = tmp_path / "b"
    write_bundle(path, *case, values=np.zeros((7, 1)))
    with pytest.raises(ValueError, match="reference split"):
        read_bundle(path, "binary")


def test_selecting_nothing_is_allowed_and_distinct_from_failing(tmp_path, case):
    """An arm that found no features has still produced a result, so an empty list is accepted,
    whereas a bundle without the `features` key comes from a broken adapter and is refused."""
    path = write_bundle(tmp_path / "b", *case, features=[], values=np.zeros((40, 0)),
                        effect_values=np.zeros((80, 0)))
    assert read_bundle(path, "binary", y_val=case.y_val).records == []

    (path / "features.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="features"):
        read_bundle(path, "binary")


def test_ingest_produces_the_same_run_directory_as_an_in_process_arm(tmp_path, case):
    """An ingested bundle yields the same cell files as an arm fitted in this process, so the code
    that uses a cell cannot tell which path produced it."""
    write_bundle(tmp_path / "b", *case)
    val = pd.DataFrame({TARGET: case.y_val})

    out = tmp_path / "cell"
    run = ingest_bundle(tmp_path / "b", val, target_column=TARGET, target_kind="binary",
                        out_dir=out, seed=42, n_bootstrap=0,
                        labels="real", arm=("deft_or_cart", "positional"))

    written = {p.name for p in out.iterdir()}
    assert written == {"run.json", "metrics.json", "predictions.npz",
                       "features.json", "feature_values.npz"}
    assert run["seed"] == 42 and run["n_bootstrap"] == 0
    assert run["labels"] == "real"
    assert run["n_features_used"] == 1
    assert run["importance_kind"] == "deft_split_score"
    assert run["fitted_model"] == MODEL, "the reveal finds the model through the cell's record"
    assert set(run["metrics"]) == {"threshold_chosen_on", "val"}
    assert 0.0 <= run["metrics"]["val"]["accuracy"] <= 1.0


def test_ingest_checks_alignment_against_the_split_it_is_given(tmp_path, case):
    write_bundle(tmp_path / "b", *case)
    wrong = pd.DataFrame({TARGET: 1 - case.y_val})
    with pytest.raises(ValueError, match="labels disagree"):
        ingest_bundle(tmp_path / "b", wrong, target_column=TARGET, target_kind="binary", seed=0,
                      n_bootstrap=0)


def test_the_contract_is_available_as_data():
    contract = describe_contract()
    assert contract["bundle_version"] == BUNDLE_VERSION
    assert "predictions.npz" in contract["required_files"]
    assert "fitted_model" in contract["required_arm_keys"]
    assert not set(contract["withheld_keys"]) & set(contract["prediction_keys"])
