"""Sequences of different lengths through the stages: the positional window, the k-mer rate, the
config's refusals, one run on the planted dataset's variable-length variant, fitted, reduced,
profiled and revealed, and the group-dependence analysis, which featurises as the arms do."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features import PositionalWindow
from seqmaestro.features.positional import PAD, window_symbols
from seqmaestro.harness.features import load_features
from seqmaestro.harness.fitted import load_fitted
from seqmaestro.harness.heldout import TEST_SPLIT, reveal_record
from seqmaestro.qc import group_dependence
from tests.planted import ANALYSES, LENGTH, planted_dataset

RUNNERS = Path(__file__).resolve().parents[1] / "experiments" / "runners"


#: The settings of the sources a test's pools hold.
KMER = {"k_range": [1, 3], "normalise": "counts"}
PROPOSER = {"proposal_rounds": 3, "n_per_round": 10, "n_examples": 20}


def a_config(tmp_path: Path, dataset, arms: dict, sources: dict | None = None, **sections) -> Path:
    held = {member for pool in arms for member in pool.split("+")}
    if sources is None:
        sources = {**({"kmer": KMER} if "kmer" in held else {}),
                   **({"llm_generated": PROPOSER} if "llm_generated" in held else {})}
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
        "sources": sources,
        "analyses": ANALYSES,
        "arms": arms,
        **sections,
    }, sort_keys=False))
    return source


# --------------------------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------------------------


def test_a_positional_arm_needs_a_window_when_lengths_vary(runners, tmp_path):
    dataset = planted_dataset(tmp_path, variable=True, window=False)
    for arm in ({"positional": {"models": ["l1_logistic"]}},
                {"positional+kmer": {"models": ["l1_logistic"]}},
                {"positional+llm_generated": {"models": ["ebm"]}}):
        with pytest.raises(runners.config.ConfigError, match="no positional window"):
            runners.config.load(a_config(tmp_path, dataset, arm))
    config = runners.config.load(a_config(tmp_path, dataset, {"llm_generated": {"models": ["ebm"]},
                                                              "kmer": {"models": ["l1"]}}))
    assert config.arm_cells() == [("ebm", "llm_generated"), ("l1_logistic", "kmer")], \
        "a proposer reads whole sequences, and k-mers are counted at any length"


def test_the_tree_arms_read_sequences_of_different_lengths_in_the_window(runners, tmp_path):
    windowed = planted_dataset(tmp_path / "windowed", variable=True)
    config = runners.config.load(a_config(tmp_path, windowed,
                                          {"positional": {"models": ["deft_or_cart"]}}))
    assert config.arm_cells() == [("deft_or_cart", "positional")]
    unwindowed = planted_dataset(tmp_path / "unwindowed", variable=True, window=False)
    with pytest.raises(runners.config.ConfigError, match="no positional window"):
        runners.config.load(a_config(tmp_path, unwindowed,
                                     {"positional": {"models": ["deft_or_cart"]}}))


def test_the_kmer_rate_is_a_source_setting_passed_to_the_arms_that_count_kmers(runners, tmp_path):
    dataset = planted_dataset(tmp_path, variable=True)
    arms = {"positional+kmer": {"models": ["l1_logistic"]}, "positional": {"models": ["l1_logistic"]}}
    config = runners.config.load(a_config(tmp_path, dataset, arms, sources={
        "kmer": {"k_range": [1, 3], "normalise": "per_length"}}))
    counted = runners.arm.pool_featuriser(config, dataset, "positional+kmer").describe()
    assert "per_length" in {part.get("normalise") for part in counted["parts"]}
    with pytest.raises(runners.config.ConfigError, match="normalise"):
        runners.config.load(a_config(tmp_path, dataset, arms, sources={
            "kmer": {"k_range": [1, 3], "normalise": "per_base"}}))


# --------------------------------------------------------------------------------------------
# One run, end to end
# --------------------------------------------------------------------------------------------

RELIABILITY = {"pairs": 3, "permutations": 200, "cluster_rho": 0.8, "max_modal_share": 0.99,
               "planted_count": 3, "planted_signal_sd": 1.5,
               "check_arm": "l1_logistic/positional", "seed": 0}


def test_a_variable_length_run_is_fitted_reduced_profiled_and_revealed(runners, tmp_path):
    dataset = planted_dataset(tmp_path, variable=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = runners.config.load(a_config(
        tmp_path, dataset,
        {"positional": {"models": ["l1_logistic"]}, "positional+kmer": {"models": ["l1_logistic"]}},
        sources={"kmer": {"k_range": [1, 3], "normalise": "per_length"}},
        reduction={"n_features": 3}, reliability=RELIABILITY))
    (run_dir / "config.yaml").write_text(config.to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main(flags) == 0

    positional = run_dir / "results" / "l1_logistic" / "positional" / "real" / "seed_1"
    arm = load_fitted(positional)
    assert arm.featuriser.describe()["window"] == {"length": LENGTH, "anchor": "start", "pad": True}
    assert f"pos_{LENGTH - 1}_pad" in arm.featuriser.feature_names
    kmer = load_features(run_dir / "results" / "l1_logistic" / "positional+kmer" / "real" /
                         "seed_1" / "features.json")[0]
    assert all(record.params.get("normalise") == "per_length"
               for record in kmer if record.kind == "kmer")

    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "2"]) == 0
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert reveal_record(run_dir)["completed_utc"] is not None
    reduced = run_dir / "reduction" / "l1_logistic" / "positional" / "reduced"
    assert TEST_SPLIT in json.loads((reduced / "metrics.json").read_text())
    test = pd.read_csv(dataset.directory / f"{TEST_SPLIT}.csv")
    assert test[dataset.sequence_column].str.len().nunique() > 1, "the test rows are ragged too"


# --------------------------------------------------------------------------------------------
# The group-dependence analysis
# --------------------------------------------------------------------------------------------


def test_the_group_dependence_analysis_keeps_each_feature_file_apart(tmp_path):
    """The feature files of two arms at one seed sit in directories of one name, `seed_<n>`, so a
    file is listed under its path."""
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import save_features

    table = tmp_path / "table.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"window": ["".join(rng.choice(list("ACGT"), size=20)) for _ in range(60)],
                  "gene": np.repeat(np.arange(20), 3)}).to_csv(table, index=False)
    given = []
    for arm, symbol in (("l1_logistic/llm_generated", "G"), ("ebm/llm_generated", "A")):
        cell = tmp_path / "results" / arm / "real" / "seed_1"
        code = f"def count_{symbol}(X):\n    return X['window'].str.count('{symbol}')"
        save_features(cell / "features.json", [FeatureRecord(
            name=f"count_{symbol}", description="a count", kind="llm_generated", source=code)])
        given.append(str(cell / "features.json"))
    out = tmp_path / "eta.csv"

    assert group_dependence.main(["--dataset", str(table), "--sequence-column", "window",
                                  "--group-column", "gene", "--features", *given,
                                  "--out", str(out)]) == 0

    scored = pd.read_csv(out)
    assert dict(zip(scored.pool, scored.feature, strict=True)) == {given[0]: "count_G",
                                                                   given[1]: "count_A"}


def test_the_group_dependence_analysis_featurises_as_the_arms_do(tmp_path, monkeypatch, capsys):
    """On a table of different lengths it reads positions in the window it is given, counts k-mers
    per length where asked, and refuses a positional pool without a window and a k-mer pool
    without how its k-mers are counted."""
    rng = np.random.default_rng(0)
    table = tmp_path / "table.csv"
    pd.DataFrame({
        "window": ["".join(rng.choice(list("ACGT"), size=length))
                   for length in rng.integers(12, 40, 60)],
        "gene": np.repeat(np.arange(20), 3),
    }).to_csv(table, index=False)
    made = []
    build = group_dependence.make_featuriser
    monkeypatch.setattr(group_dependence, "make_featuriser",
                        lambda mode, **settings: made.append((mode, settings)) or build(mode, **settings))
    given = ["--dataset", str(table), "--sequence-column", "window", "--group-column", "gene"]

    with pytest.raises(SystemExit, match="--positional-window"):
        group_dependence.main([*given, "--pools", "positional"])
    with pytest.raises(SystemExit, match="PAD is pad or no-pad"):
        group_dependence.main([*given, "--pools", "positional",
                               "--positional-window", "20", "start", "yes"])
    with pytest.raises(SystemExit):
        group_dependence.main([*given, "--pools", "kmer", "--k-range", "1", "2"])
    assert "--k-range and --kmer-normalise must say how they are counted" in capsys.readouterr().err

    out = tmp_path / "eta.csv"
    assert group_dependence.main([
        *given, "--pools", "positional", "kmer", "--k-range", "1", "2",
        "--positional-window", "20", "start", "pad", "--kmer-normalise", "per_length",
        "--out", str(out)]) == 0
    assert made == [
        ("positional", {"k_range": (1, 2), "window": PositionalWindow(20, "start", True),
                        "kmer_normalise": "per_length"}),
        ("kmer", {"k_range": (1, 2), "window": PositionalWindow(20, "start", True),
                  "kmer_normalise": "per_length"}),
    ]
    scored = pd.read_csv(out)
    positional = scored[scored.pool == "positional"].feature
    assert len(positional) == 20 * 5, "four bases and the pad at each of the 20 positions"


# --------------------------------------------------------------------------------------------
# DEFT reads a window by the same rule
# --------------------------------------------------------------------------------------------


def deft_window():
    """DEFT's copy of the rule, loaded from its file: it imports nothing, so it needs no DEFT
    environment."""
    path = RUNNERS.parents[1] / "methods" / "deft" / "src" / "deft" / "utils" / "window.py"
    spec = importlib.util.spec_from_file_location("_deft_window", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_deft_reads_a_positional_window_by_the_same_rule():
    deft = deft_window()
    assert inspect.getsource(deft.window_symbols) == inspect.getsource(window_symbols), "verbatim"
    assert deft.PAD == PAD
    for sequence in ("ACGTAC", "TG", "ACGT", ""):
        for length in (1, 4, 6):
            for anchor in ("start", "end"):
                for pad in (True, False):
                    try:
                        expected = window_symbols(sequence, length, anchor, pad)
                    except ValueError:
                        with pytest.raises(ValueError):
                            deft.window_symbols(sequence, length, anchor, pad)
                        continue
                    assert deft.window_symbols(sequence, length, anchor, pad) == expected

