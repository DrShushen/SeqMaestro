"""A cell's files and the module that alone writes them (`seqmaestro.harness.cell`), the whole-file
writes it relies on (`seqmaestro.files`), and how a cell states where its model lives
(`seqmaestro.pipeline.sides`).

A cell that has `metrics.json` is finished, so a resumed run skips it and the reveal scores it. The
marker must therefore be the last file written, and a file must never exist half written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seqmaestro import files
from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness import cell as cells
from seqmaestro.pipeline import sides


def test_a_file_is_replaced_whole_and_nothing_is_left_beside_it(tmp_path):
    path = tmp_path / "record.json"
    files.write_json(path, {"n": 1})
    files.write_json(path, {"n": 2})
    assert json.loads(path.read_text()) == {"n": 2}
    files.write_arrays(tmp_path / "arrays.npz", scores=np.array([0.25, 0.75]))
    with np.load(tmp_path / "arrays.npz") as saved:
        assert saved["scores"].tolist() == [0.25, 0.75]
    assert sorted(found.name for found in tmp_path.iterdir()) == ["arrays.npz", "record.json"]


def test_a_value_json_has_no_form_for_is_refused_unless_the_caller_says_how_to_write_it(tmp_path):
    with pytest.raises(TypeError):
        files.write_json(tmp_path / "record.json", {"path": tmp_path})
    assert not list(tmp_path.iterdir()), "a refused write leaves no file under the name"
    files.write_json(tmp_path / "record.json", {"path": tmp_path}, default=str)
    assert json.loads((tmp_path / "record.json").read_text()) == {"path": str(tmp_path)}


def a_cell(directory, **given) -> dict:
    written = {"run": {"seed": 3, "labels": cells.REAL}, "metrics": {"val": {"auroc": 0.75}},
               "y_val": [0, 1, 1], "scores": [0.2, 0.9, 0.6], **given}
    cells.write(directory, **written)
    return written


def test_a_cell_holds_its_record_its_arrays_and_its_marker(tmp_path):
    cell = tmp_path / "cell"
    assert not cells.is_finished(cell)
    a_cell(cell)
    assert cells.is_finished(cell)
    assert cells.read_run(cell) == {"seed": 3, "labels": "real"}
    assert cells.read_metrics(cell) == {"val": {"auroc": 0.75}}
    assert cells.validation_scores(cell).tolist() == [0.2, 0.9, 0.6]
    assert cells.read_predictions(cell)["val_true"].tolist() == [0, 1, 1]
    assert {found.name for found in cell.iterdir()} == {
        cells.RUN_FILE, cells.METRICS_FILE, cells.PREDICTIONS_FILE}


def test_a_stage_2_cell_also_holds_its_features_and_what_its_arm_reports(tmp_path):
    cell = tmp_path / "cell"
    records = [FeatureRecord(name="gc", description="GC share", kind="composition", importance=0.5)]
    a_cell(cell, features=cells.Features(records, np.array([[0.1], [0.4], [0.7]]),
                                         {"importance_kind": "l1_coefficient_per_sd"}),
           reports={"token_usage.json": {"usage": {"total_tokens": 12}}})
    header = json.loads((cell / cells.FEATURES_FILE).read_text())
    assert header["importance_kind"] == "l1_coefficient_per_sd"
    assert [feature["name"] for feature in header["features"]] == ["gc"]
    with np.load(cell / cells.FEATURE_VALUES_FILE, allow_pickle=True) as saved:
        assert saved["values"].shape == (3, 1)
    assert json.loads((cell / "token_usage.json").read_text())["usage"]["total_tokens"] == 12


def test_the_marker_is_the_last_file_written(tmp_path, monkeypatch):
    """A cell whose writing stops partway has no marker, whichever file it stopped at."""
    written = []
    write_bytes = files.write_bytes

    def recording(path, data):
        written.append(path.name)
        return write_bytes(path, data)

    monkeypatch.setattr(files, "write_bytes", recording)
    records = [FeatureRecord(name="gc", description="GC share", kind="composition", importance=0.5)]
    a_cell(tmp_path / "cell", features=cells.Features(records, np.zeros((3, 1))),
           reports={"token_usage.json": {}})
    assert written[-1] == cells.METRICS_FILE
    assert set(written) == {cells.RUN_FILE, cells.PREDICTIONS_FILE, cells.FEATURES_FILE,
                            cells.FEATURE_VALUES_FILE, "token_usage.json", cells.METRICS_FILE}


def test_a_cell_written_again_is_unfinished_until_its_new_marker_is_there(tmp_path, monkeypatch):
    cell = tmp_path / "cell"
    a_cell(cell)

    def cut_short(path, arrays):
        raise OSError("the disk is full")

    monkeypatch.setattr(cells, "write_predictions", cut_short)
    with pytest.raises(OSError, match="the disk is full"):
        a_cell(cell, metrics={"val": {"auroc": 0.5}})
    assert not cells.is_finished(cell), "the files of two fits must not sit under one marker"


@pytest.mark.parametrize(("recorded", "side"), [
    ("fitted", sides.CORE), ("deft_bundle", sides.DEFT), ("deft_cart", sides.DEFT),
    ("igsr_bundle", sides.IGSR), ("igsr_terms", sides.IGSR)])
def test_a_cell_names_the_side_its_model_lives_on(recorded, side):
    assert sides.side_of({"predict_with": recorded}) is side
    assert recorded in (side.stage2, side.later)


def test_a_cell_that_names_no_side_is_refused_with_the_names_there_are(tmp_path):
    with pytest.raises(ValueError, match=r"no side predicts.*deft_bundle.*fitted.*igsr_terms"):
        sides.side_of({"predict_with": "another_method"}, tmp_path)
    with pytest.raises(ValueError, match="predict_with: None"):
        sides.side_of({})


def test_a_core_cell_is_saved_only_once_its_model_file_is_there(tmp_path):
    run = {"predict_with": "fitted"}
    assert not sides.CORE.is_saved(tmp_path, run)
    (tmp_path / "fitted.pkl").write_bytes(b"")
    assert sides.CORE.is_saved(tmp_path, run)


@pytest.mark.parametrize(("side", "stage2", "later"), [
    (sides.DEFT, {"predict_with": "deft_bundle", "fitted_model": "deft_run/tree.dill"},
     {"predict_with": "deft_cart", "fitted_model": "model.dill"}),
    (sides.IGSR, {"predict_with": "igsr_bundle", "fitted_model": "model.json"},
     {"predict_with": "igsr_terms", "fitted_model": "model.pkl"}),
])
def test_a_vendored_cell_is_saved_only_once_the_model_its_record_names_is_there(tmp_path, side,
                                                                              stage2, later):
    """A Stage 2 cell keeps its model in the bundle its method handed back, whereas a cell fitted
    later keeps it in the cell itself. The reveal checks this before it opens the held-out split."""
    assert not side.is_saved(tmp_path, stage2) and not side.is_saved(tmp_path, later)
    (tmp_path / stage2["fitted_model"]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / stage2["fitted_model"]).write_bytes(b"")
    assert not side.is_saved(tmp_path, stage2), "a Stage 2 cell's model is in its bundle"
    in_bundle = tmp_path / "bundle" / stage2["fitted_model"]
    in_bundle.parent.mkdir(parents=True)
    in_bundle.write_bytes(b"")
    assert side.is_saved(tmp_path, stage2)
    (tmp_path / later["fitted_model"]).write_bytes(b"")
    assert side.is_saved(tmp_path, later)
    assert not side.is_saved(tmp_path, {"predict_with": later["predict_with"]}), "no model named"
