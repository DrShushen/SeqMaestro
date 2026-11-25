"""What a call (one process of a run) leaves when one of its cells fails and is later repaired.

A failed cell is recorded in the call's failures file while the remaining cells still run. Once
the cell is repaired and the call is run again, the file is removed, so a run whose cells are all
complete has no record saying otherwise.
"""

from __future__ import annotations

import json

from seqmaestro.pipeline.layout import cell_dir
from seqmaestro.pipeline.records import Failures
from seqmaestro.pipeline.stage2 import run_cells


def test_a_call_that_ends_clean_removes_the_failures_an_earlier_attempt_left(tmp_path):
    run_dir = tmp_path / "run"
    cell = cell_dir(run_dir, "deft_or_cart", "positional", "real", 42)
    fitted = []

    def broken() -> dict:
        raise RuntimeError("the post-processing failed")

    def whole() -> dict:
        fitted.append("other")
        return {"n_features_used": 3, "metrics": {"val": {"r2": 0.1, "spearman_rho": 0.3, "rmse": 1.0}}}

    def attempt() -> Failures:
        failures = Failures(run_dir, "deft_or_cart__positional__real__seed_42")
        run_cells([("broken", cell, broken), ("other", tmp_path / "other", whole)],
                  target_kind="continuous", force=False, failures=failures)
        return failures

    first = attempt()
    assert first.close() == 1 and fitted == ["other"], "one failed cell does not stop the next"
    assert first.path == run_dir / "failures" / "deft_or_cart__positional__real__seed_42.json"
    recorded = json.loads(first.path.read_text())
    assert [entry["item"] for entry in recorded] == ["broken"]
    assert "the post-processing failed" in recorded[0]["error"]

    # The repair writes the cell in full, as `run_arm.py --reuse-tree` does for a DEFT cell.
    cell.mkdir(parents=True)
    (cell / "metrics.json").write_text(json.dumps({"val": {}}))
    assert attempt().close() == 0
    assert not first.path.exists()
