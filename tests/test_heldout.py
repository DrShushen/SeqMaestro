"""The test split stays closed until one reveal, which scores a cell exactly as inline scoring would.

There are three kinds of check:

- Static: no module names the test split but `seqmaestro.harness.heldout` and
  `seqmaestro/dataset.py`, which finalises and checks a dataset, and every stage that fits refuses
  a revealed run.
- The seal: `reveal.json` is written before the test file is read, the file is checked against the
  digest Stage 1 recorded, and a run is revealed only once.
- The reveal: a cell scored at the reveal holds exactly the numbers that scoring with the model and
  featuriser in memory gives, and `run_reveal.py` refuses a run whose cells are not all present
  unless it is given a reason.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import make_featuriser
from seqmaestro.harness.fitted import load_fitted
from seqmaestro.harness.heldout import (
    REVEAL_FILE,
    TEST_ARRAYS,
    TEST_FILE,
    TEST_SPLIT,
    SealError,
    begin_reveal,
    is_scored,
    open_test,
    refuse_if_revealed,
    reveal_record,
    revealed_metrics,
    score_cell,
)
from seqmaestro.harness.metrics import choose_threshold, classification_metrics
from seqmaestro.models import MODELS, make_model
from seqmaestro.models.base import FILLED_ROWS
from tests.planted import ANALYSES, SEQUENCE, TARGET, planted_dataset, run_cell

REPO = Path(__file__).resolve().parents[1]
RUNNERS = REPO / "experiments" / "runners"

# --------------------------------------------------------------------------------------------
# Static: one reader
# --------------------------------------------------------------------------------------------

#: Where code that must not read the test split lives.
GUARDED = ("seqmaestro", "experiments/runners", "experiments/analysis", "experiments/adapters")
#: The reader itself, and the Stage 1 standard, which counts the split it describes.
EXEMPT = {"seqmaestro/harness/heldout.py", "seqmaestro/dataset.py"}
#: A string naming the test split: the split, its file, or a key or array built on its name.
NAMES_THE_TEST_SPLIT = re.compile(r"^test$|^test[._]|test\.csv")


def references(path: Path) -> list[str]:
    """The string constants in a module that name the test split. Docstrings are prose, and are
    allowed to talk about it."""
    tree = ast.parse(path.read_text())
    prose = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                prose.add(id(first.value))
    return [f"line {node.lineno}: {node.value!r}" for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in prose and NAMES_THE_TEST_SPLIT.search(node.value)]


def guarded_modules() -> list[Path]:
    modules = sorted(path for root in GUARDED for path in (REPO / root).rglob("*.py"))
    return [path for path in modules if str(path.relative_to(REPO)) not in EXEMPT]


def test_only_the_heldout_module_names_the_test_split():
    """Code that reads test results after the reveal takes the names from `heldout`, so every
    dependency on the test split is an import of that module."""
    modules = guarded_modules()
    assert len(modules) > 40, "too few modules found; the scan is looking in the wrong place"
    offenders = {str(path.relative_to(REPO)): found for path in modules
                 if (found := references(path))}
    assert not offenders, f"only seqmaestro/harness/heldout.py may name the test split: {offenders}"


def test_the_scan_catches_a_module_that_reads_the_test_split(tmp_path):
    """A test of the scan itself: a loop over the three split names is caught, and prose about
    them is not."""
    module = tmp_path / "reads.py"
    module.write_text('import pandas as pd\n'
                      'frames = [pd.read_csv(f"{s}.csv") for s in ("train", "val", "test")]\n')
    assert references(module)
    module.write_text('"""Talks about the test split, which is allowed."""\n\nX = "test split"\n')
    assert not references(module)


def test_nothing_outside_stage_1_takes_the_list_of_splits():
    """`SPLITS` holds the test split's name, so a loop over it would read the test file without
    naming it."""
    for module in guarded_modules():
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "seqmaestro.dataset":
                assert "SPLITS" not in {alias.name for alias in node.names}, module


#: The `run_*.py` scripts that are not a stage that fits or selects: the reveal itself, and the
#: driver, whose preflight refuses a revealed run before it starts any stage.
NOT_PRE_REVEAL = {"run_reveal.py", "run_experiment.py"}


def test_every_stage_before_the_reveal_refuses_a_revealed_run():
    """Every stage but the reveal fits or selects something, so the module behind each of their
    scripts must refuse."""
    scripts = sorted(path for path in RUNNERS.glob("run_*.py") if path.name not in NOT_PRE_REVEAL)
    assert {"run_baseline.py", "run_arm.py", "run_reduction.py", "run_reliability.py"} == \
           {path.name for path in scripts}, "the scan found other scripts than the stages'"
    for script in scripts:
        imported = [node.module for node in ast.walk(ast.parse(script.read_text()))
                    if isinstance(node, ast.ImportFrom) and node.module.startswith("seqmaestro.pipeline.")]
        assert len(imported) == 1, f"{script.name} is to call one module's `main`"
        module = REPO / (imported[0].replace(".", "/") + ".py")
        calls = {node.func.id for node in ast.walk(ast.parse(module.read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "refuse_if_revealed" in calls, module.name


# --------------------------------------------------------------------------------------------
# The seal
# --------------------------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    return planted_dataset(tmp_path)


def test_the_test_split_cannot_be_opened_before_the_reveal(dataset, tmp_path):
    with pytest.raises(SealError, match="not been revealed"):
        open_test(tmp_path, dataset)


def test_the_record_is_written_first_and_says_what_was_sealed(dataset, tmp_path):
    record = begin_reveal(tmp_path, dataset, ["results/b", "baseline/a"])

    assert json.loads((tmp_path / REVEAL_FILE).read_text()) == record
    assert record["sealed"] == ["baseline/a", "results/b"]
    assert record["dataset"]["test_sha256"] == dataset.split_counts[TEST_SPLIT]["sha256"]
    assert record["override"] is None and record["completed_utc"] is None
    assert len(open_test(tmp_path, dataset)) == dataset.split_counts[TEST_SPLIT]["rows"]


def test_a_run_is_revealed_once(dataset, tmp_path):
    begin_reveal(tmp_path, dataset, [])
    with pytest.raises(SealError, match="already been revealed"):
        begin_reveal(tmp_path, dataset, [])


def test_stages_that_fit_refuse_a_revealed_run(dataset, tmp_path):
    refuse_if_revealed(tmp_path, "a stage")
    begin_reveal(tmp_path, dataset, [])
    with pytest.raises(SealError, match="a stage cannot run"):
        refuse_if_revealed(tmp_path, "a stage")


def test_a_test_file_changed_since_stage_1_is_refused(dataset, tmp_path):
    begin_reveal(tmp_path, dataset, [])
    path = dataset.directory / TEST_FILE
    path.write_text(path.read_text() + "ACGTACGTACGTACGTACGT,1\n")
    with pytest.raises(SealError, match="changed since"):
        open_test(tmp_path, dataset)


def test_a_partial_reveal_needs_a_reason_and_records_it(dataset, tmp_path):
    with pytest.raises(SealError, match="stated reason"):
        begin_reveal(tmp_path, dataset, ["a"], incomplete=["b: did not finish"])
    assert reveal_record(tmp_path) is None, "refused before anything was written"

    record = begin_reveal(tmp_path, dataset, ["a"], override="a smoke run",
                          incomplete=["b: did not finish"])
    assert record["override"] == {"reason": "a smoke run", "incomplete": ["b: did not finish"]}


# --------------------------------------------------------------------------------------------
# Scoring a cell
# --------------------------------------------------------------------------------------------


def a_cell(dataset, out: Path, model_name: str = "l1_logistic", seed: int = 5,
           n_bootstrap: int = 25):
    """One cell fitted through the harness. Returns the model and featuriser it held in memory."""
    train, val = pd.read_csv(dataset.train_path), pd.read_csv(dataset.val_path)
    model, featuriser = make_model(model_name, target_kind="binary", seed=seed), make_featuriser("positional")
    run_cell(model, featuriser, train, val, columns=dataset.columns,
              seed=seed, n_bootstrap=n_bootstrap, permutation_repeats=0, out_dir=out,
              arm=(model_name, "positional"))
    return model, featuriser


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_a_revealed_cell_holds_exactly_what_inline_scoring_wrote(dataset, tmp_path, model_name):
    """Scoring with the model and featuriser held in memory is compared with what the reveal writes
    from the saved model. The two must be identical, not merely close, because a number that
    changed when the scoring path changed could not be reproduced."""
    cell = tmp_path / "cell"
    model, featuriser = a_cell(dataset, cell, model_name)
    val = pd.read_csv(dataset.val_path)
    test = pd.read_csv(dataset.directory / TEST_FILE)
    threshold = choose_threshold(val[TARGET], model.predict(featuriser.transform(val[SEQUENCE])))
    inline_scores = np.asarray(model.predict(featuriser.transform(test[SEQUENCE])), dtype=float)
    inline = classification_metrics(test[TARGET].to_numpy(), inline_scores, threshold,
                                     n_bootstrap=25, seed=5)
    before = json.loads((cell / "metrics.json").read_text())

    begin_reveal(tmp_path, dataset, ["cell"])
    rows = open_test(tmp_path, dataset)
    scores, filled = load_fitted(cell).predict_counting(rows[SEQUENCE])
    block = score_cell(cell, rows[TARGET].to_numpy(), scores, target_kind="binary", filled=filled)

    assert json.dumps(block) == json.dumps(inline)
    written = json.loads((cell / "metrics.json").read_text())
    # A model that can be undefined on a row also records how many rows of each split it filled.
    assert [key for key in written if key != FILLED_ROWS] == ["threshold_chosen_on", "val", TEST_SPLIT]
    if hasattr(model, "filled"):
        assert set(written[FILLED_ROWS]) == {"val", TEST_SPLIT}
    else:
        assert FILLED_ROWS not in written and filled is None
    assert written["val"] == before["val"] and written["val"]["threshold"] == threshold
    with np.load(cell / "predictions.npz") as arrays:
        assert np.array_equal(arrays[TEST_ARRAYS[1]], inline_scores)
        assert np.array_equal(arrays[TEST_ARRAYS[0]], test[TARGET].to_numpy())
        assert arrays["val_score"].size == len(val)


def test_a_cell_is_scored_once(dataset, tmp_path):
    a_cell(dataset, tmp_path / "cell")
    begin_reveal(tmp_path, dataset, ["cell"])
    rows = open_test(tmp_path, dataset)
    scores = load_fitted(tmp_path / "cell").predict(rows[SEQUENCE])

    assert not is_scored(tmp_path / "cell")
    score_cell(tmp_path / "cell", rows[TARGET], scores, target_kind="binary")
    assert is_scored(tmp_path / "cell")
    with pytest.raises(SealError, match="scored once"):
        score_cell(tmp_path / "cell", rows[TARGET], scores, target_kind="binary")


def test_a_cell_that_did_not_record_its_scoring_is_refused(dataset, tmp_path):
    """Without the bootstrap count the test block's intervals could not match the validation
    block's, and nothing would say they had been drawn differently."""
    a_cell(dataset, tmp_path / "cell")
    run = json.loads((tmp_path / "cell" / "run.json").read_text())
    del run["n_bootstrap"]
    (tmp_path / "cell" / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="n_bootstrap"):
        score_cell(tmp_path / "cell", [0, 1], [0.2, 0.8], target_kind="binary")


def test_test_numbers_are_refused_to_a_reader_before_the_reveal(dataset, tmp_path):
    a_cell(dataset, tmp_path / "cell")
    with pytest.raises(SealError, match="not been revealed"):
        revealed_metrics(tmp_path / "cell", tmp_path)

    begin_reveal(tmp_path, dataset, ["cell"])
    rows = open_test(tmp_path, dataset)
    block = score_cell(tmp_path / "cell", rows[TARGET],
                       load_fitted(tmp_path / "cell").predict(rows[SEQUENCE]), target_kind="binary")
    assert revealed_metrics(tmp_path / "cell", tmp_path) == block


# --------------------------------------------------------------------------------------------
# The reveal stage, end to end, on our own arms and the baseline
# --------------------------------------------------------------------------------------------


def a_run(tmp_path: Path, dataset, runners) -> Path:
    """A run directory holding its resolved config, as the driver leaves one."""
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump({
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": 10,
                 "permutation_scope": "seed", "class_weight": "balanced",
                 "permutation_repeats": 0, "judge": False, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60}},
        "analyses": ANALYSES,
        "arms": {"positional": {"models": ["l1_logistic"]}},
    }))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(runners.config.load(source).to_yaml())
    return run_dir


def run_baseline(runners, run_dir, dataset) -> int:
    return runners.baseline.main(["--run-dir", str(run_dir)])


def run_arm(runners, run_dir, dataset) -> int:
    return runners.arm.main(["--run-dir", str(run_dir)])


def test_a_finished_run_is_revealed_and_scored_as_inline_scoring_would_have(runners, dataset,
                                                                             tmp_path):
    run_dir = a_run(tmp_path, dataset, runners)
    assert run_baseline(runners, run_dir, dataset) == 0
    assert run_arm(runners, run_dir, dataset) == 0

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0

    record = reveal_record(run_dir)
    assert record["completed_utc"] and record["scored"] == {"baseline": 2, "results": 4}
    test = pd.read_csv(dataset.directory / TEST_FILE)
    for relative in record["sealed"]:
        cell = run_dir / relative
        metrics = json.loads((cell / "metrics.json").read_text())
        run = json.loads((cell / "run.json").read_text())
        expected = classification_metrics(
            test[TARGET].to_numpy(), load_fitted(cell).predict(test[SEQUENCE]),
            metrics["val"]["threshold"], n_bootstrap=run["n_bootstrap"], seed=run["seed"])
        assert json.dumps(metrics[TEST_SPLIT]) == json.dumps(expected), relative


def test_after_the_reveal_the_arms_refuse_and_the_reveal_only_resumes(runners, dataset, tmp_path):
    run_dir = a_run(tmp_path, dataset, runners)
    run_baseline(runners, run_dir, dataset)
    run_arm(runners, run_dir, dataset)
    runners.reveal.main(["--run-dir", str(run_dir)])
    written = {path: path.read_text() for path in run_dir.rglob("metrics.json")}

    with pytest.raises(SealError):
        run_arm(runners, run_dir, dataset)
    with pytest.raises(SealError):
        run_baseline(runners, run_dir, dataset)
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert {path: path.read_text() for path in run_dir.rglob("metrics.json")} == written


def test_checking_a_run_reveals_nothing(runners, dataset, tmp_path):
    run_dir = a_run(tmp_path, dataset, runners)
    run_baseline(runners, run_dir, dataset)
    assert runners.reveal.main(["--run-dir", str(run_dir), "--check"]) == 2

    run_arm(runners, run_dir, dataset)
    assert runners.reveal.main(["--run-dir", str(run_dir), "--check"]) == 0
    assert reveal_record(run_dir) is None
    assert not any(is_scored(cell.parent) for cell in run_dir.rglob("metrics.json"))


def test_the_reveal_refuses_a_run_that_is_not_finished(runners, dataset, tmp_path, capsys):
    run_dir = a_run(tmp_path, dataset, runners)
    run_baseline(runners, run_dir, dataset)

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 2
    assert "4 of 6 items are not ready" in capsys.readouterr().err
    assert reveal_record(run_dir) is None, "refused before anything was sealed"

    assert runners.reveal.main(["--run-dir", str(run_dir),
                                "--override-incomplete", "only the baseline is wanted"]) == 0
    record = reveal_record(run_dir)
    assert record["override"]["reason"] == "only the baseline is wanted"
    assert len(record["override"]["incomplete"]) == 4
    assert record["sealed"] == ["baseline/seed_1", "baseline/seed_2"]


def test_a_cell_the_config_does_not_account_for_is_recorded_and_left_closed(runners, dataset,
                                                                            tmp_path):
    run_dir = a_run(tmp_path, dataset, runners)
    run_baseline(runners, run_dir, dataset)
    run_arm(runners, run_dir, dataset)
    # A cell of an arm the run's config no longer names, as a changed config leaves one behind.
    recorded = run_dir / "config.yaml"
    narrower = recorded.read_text()
    document = yaml.safe_load(narrower)
    document["arms"]["positional"]["models"].append("ebm")
    recorded.write_text(yaml.safe_dump(document, sort_keys=False))
    assert runners.arm.main(["--run-dir", str(run_dir), "--arms", "ebm/positional",
                             "--labels", "real", "--seeds", "1"]) == 0
    recorded.write_text(narrower)

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    record = reveal_record(run_dir)
    assert record["unsealed"] == ["results/ebm/positional/real/seed_1"]
    assert not is_scored(run_dir / record["unsealed"][0])
