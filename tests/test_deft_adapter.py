"""Tree-walking in the DEFT adapter, tested without DEFT.

The adapter runs in DEFT's environment. The code tested here walks a tree structurally and never
imports `deft`, so stand-in nodes can exercise it in this environment. The adapter must get right
which nodes count as a selected feature, because a count that is too high looks like a richer
method and not like a bug.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seqmaestro.dataset import Columns
from seqmaestro.target import BINARY

ADAPTER = Path(__file__).resolve().parents[1] / "experiments" / "adapters" / "deft_bundle.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("deft_bundle_under_test", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_adapter()


def feature(name, score=1.0, origin="naive", fn=None):
    return SimpleNamespace(
        name=name, description=f"{name} description", string=f"def {name}(X): ...",
        threshold=0.5, score=score, rationale="because", origin=origin, fn=fn,
    )


def node(feat=None, left=None, right=None):
    return SimpleNamespace(feature=feat, left=left, right=right)


def a_tree(root, target_kind="binary", **attributes):
    """A stand-in for a fitted DEFT tree, which records the kind of target it was grown for."""
    return SimpleNamespace(root=root, target_kind=target_kind, **attributes)


def adapter_feature_positional():
    """One naive candidate as DEFT's own sequence enumeration names and renders it."""
    return feature("pos_50_is_A", origin="naive")


def test_leaves_are_not_selected_features():
    """A leaf can carry a feature left over from a search that declined to split there.

    Counting it would invent a feature the tree never uses, unnamed and with an unknown origin, so
    a depth-1 tree would report two features for one split.
    """
    leaf = node(feature(None, origin="unknown"))
    tree = a_tree(root=node(feature("real_split"), left=leaf, right=node()))

    records = adapter.collect_features(tree)
    assert [r["name"] for r in records] == ["real_split"]


def test_a_proposed_name_that_is_not_an_identifier_is_made_legal_and_kept_as_proposed():
    """DEFT keeps the name a language model proposed, which can hold a hyphen. SeqMaestro refuses a
    record under such a name, so the cell would fail after its tree had been grown."""
    tree = a_tree(root=node(feature("Half-window_CpG_imbalance", origin="llm"),
                            left=node(feature("Half_window_CpG_imbalance"), left=node(), right=node()),
                            right=node(feature("3prime-GC", origin="llm"), left=node(), right=node())))
    adapter.legal_names(tree)
    records = {r["name"]: r for r in adapter.collect_features(tree)}
    assert set(records) == {"Half_window_CpG_imbalance", "Half_window_CpG_imbalance_2", "_3prime_GC"}
    assert all(adapter.LEGAL_NAME.match(name) for name in records)
    assert records["Half_window_CpG_imbalance_2"]["extra"]["name_as_proposed"] == "Half-window_CpG_imbalance"
    assert "name_as_proposed" not in records["Half_window_CpG_imbalance"]["extra"]


def test_two_features_under_one_illegal_name_keep_sharing_one_legal_name():
    first, second = feature("a-b", origin="llm"), feature("a-b", origin="llm")
    tree = a_tree(root=node(first, left=node(second, left=node(), right=node()), right=node()))
    adapter.legal_names(tree)
    assert first.name == second.name == "a_b"


def test_one_record_per_split():
    tree = a_tree(root=node(
        feature("a"),
        left=node(feature("b"), left=node(), right=node()),
        right=node(feature("c"), left=node(), right=node()),
    ))
    assert sorted(r["name"] for r in adapter.collect_features(tree)) == ["a", "b", "c"]


def test_a_feature_winning_twice_is_one_record_carrying_both_nodes():
    """The schema has one entry per feature, but a tree may split on the same feature repeatedly.

    Every occurrence is kept, because importance sums over them. Keeping only the best-scoring one
    would lose the others and bias the record toward the smallest, deepest node.
    """
    tree = a_tree(root=node(
        feature("repeated", score=0.2),
        left=node(feature("repeated", score=0.9), left=node(), right=node()),
        right=node(),
    ))
    records = adapter.collect_features(tree)
    assert len(records) == 1
    nodes = records[0]["extra"]["nodes"]
    assert [n["node_path"] for n in nodes] == ["", "L"]
    assert [n["deft_split_score"] for n in nodes] == [pytest.approx(0.2), pytest.approx(0.9)]


def test_importance_is_absent_until_the_rows_have_been_routed():
    """`importance` is a share of impurity reduction, which needs coverage the tree does not store.

    The split score must not fill the gap. It is a node's local purity, valid for choosing among
    candidates at one node and meaningless across nodes of different sizes.
    """
    tree = a_tree(root=node(feature("f", score=0.9), left=node(), right=node()))
    assert adapter.collect_features(tree)[0]["importance"] is None


def test_records_carry_what_the_schema_needs():
    tree = a_tree(root=node(feature("f", origin="llm"), left=node(), right=node()))
    record = adapter.collect_features(tree)[0]
    assert set(record) == {"name", "description", "kind", "params", "source", "rationale",
                           "origin", "importance", "direction", "provenance", "extra"}
    assert record["extra"]["nodes"] == [
        {"depth": 0, "node_path": "", "threshold": pytest.approx(0.5),
         "deft_split_score": pytest.approx(1.0)},
    ]


def test_origin_is_the_method_and_kind_is_how_the_feature_was_proposed():
    """`origin` says whose method produced a feature and `kind` says how it was proposed, and this
    adapter must keep the two apart.

    Writing DEFT's internal `llm`/`naive` into `origin` would make the field mean the method on the
    SeqMaestro arms and the way of proposing here. Our own proposer shows why that fails, because
    its features are `seqmaestro` by method and `llm` by way of proposing.
    """
    tree = a_tree(root=node(
        feature("generated", origin="llm"),
        left=node(feature("enumerated", origin="naive"), left=node(), right=node()),
        right=node(),
    ))
    by_name = {r["name"]: r for r in adapter.collect_features(tree)}

    assert {r["origin"] for r in by_name.values()} == {"deft"}, "origin is the method, always"
    assert by_name["generated"]["kind"] == "llm_generated"
    assert by_name["generated"]["provenance"] == "generated"
    # Not a recognised positional name, so it is a column we supplied: its kind travels in the pool
    # manifest and is not guessed here.
    assert by_name["enumerated"]["kind"] == "unknown"
    assert by_name["enumerated"]["provenance"] == "selected"

    # DEFT's own label is kept in `extra`, which holds what only one method knows.
    assert by_name["generated"]["extra"]["deft_origin"] == "llm"


def test_an_unrecognised_deft_origin_is_marked_unknown_rather_than_guessed():
    tree = a_tree(root=node(feature("f", origin="something_new"), left=node(), right=node()))
    record = adapter.collect_features(tree)[0]
    assert record["kind"] == "unknown"
    assert record["extra"]["deft_origin"] == "something_new"


# --------------------------------------------------------------------------------------------
# Which dataset DEFT is told it is looking at: SeqMaestro supplies this, not the vendored method
# --------------------------------------------------------------------------------------------


def resolve(task, tmp_path, config_dir=None, dataset_dir=None, target_kind="binary"):
    return adapter.resolve_task(
        task,
        config_dir if config_dir is not None else adapter.TASK_CONFIG_DIR,
        dataset_dir if dataset_dir is not None else adapter.DATASET_DIR,
        generate_into=tmp_path,
        target_kind=target_kind,
    )


def test_a_dataset_with_no_hand_written_config_gets_one_generated(tmp_path):
    """Adding a dataset takes one description file, from which DEFT's task config is generated."""
    import yaml

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "promoters.json").write_text(json.dumps({
        "target_name": "promoter strength",
        "task": "Predicting whether transcription initiates.",
        "sequence_layout": "200 nucleotides, 0-based, position 0 is the 5' end.",
    }))

    overrides, source = resolve("promoters", tmp_path / "out", dataset_dir=datasets)
    assert overrides == [f"--config-dir={tmp_path / 'out'}", "params_generation=promoters"]
    assert "generated" in source

    written = yaml.safe_load((tmp_path / "out" / "params_generation" / "promoters.yaml").read_text())
    assert written["defaults"] == ["default", "_self_"], "must inherit DEFT's method parameters"
    assert written["target_name"] == "promoter strength"
    # Both facts reach the prompt, because the task cannot be interpreted without the layout.
    assert "transcription initiates" in written["dataset_info"]
    assert "200 nucleotides" in written["dataset_info"]


def described(tmp_path, name="promoters"):
    """A folder of dataset descriptions holding one, named `name`."""
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / f"{name}.json").write_text(json.dumps({
        "task": "Predicting whether transcription initiates.",
        "sequence_layout": "200 nucleotides, 0-based, position 0 is the 5' end."}))
    return datasets


def test_a_hand_written_config_wins_over_a_generated_one(tmp_path):
    """A hand-written config exists to reproduce published prompts. It is used whenever it exists,
    with no extra flag, because a forgotten flag would swap the paper's wording for generated
    wording and the run would still complete normally."""
    written = tmp_path / "conf"
    (written / adapter.TASK_GROUP).mkdir(parents=True)
    (written / adapter.TASK_GROUP / "promoters.yaml").write_text("defaults: [default, _self_]\n")

    overrides, source = resolve("promoters", tmp_path / "out", config_dir=written,
                                dataset_dir=described(tmp_path))

    assert overrides == [f"--config-dir={written}", "params_generation=promoters"]
    assert "hand-written" in source
    assert not (tmp_path / "out").exists(), "nothing generated"


def test_an_unknown_task_fails_before_anything_is_spent(tmp_path):
    """Left to hydra, an unknown task surfaces as a composition error inside a subprocess, after
    an LLM arm has already started paying for API calls."""
    with pytest.raises(SystemExit) as excinfo:
        resolve("no_such_dataset", tmp_path / "out", config_dir=tmp_path / "conf",
                dataset_dir=described(tmp_path))

    message = str(excinfo.value)
    assert "no_such_dataset" in message
    assert "promoters" in message, "it should say what is available"


def test_no_task_means_defts_own_task_neutral_prompts(tmp_path):
    """The CART arm never prompts, so running with no task is valid, and DEFT then uses its own
    task-neutral prompts."""
    assert resolve(None, tmp_path) == ([], "deft default (task-neutral)")


def test_a_hand_written_config_inherits_rather_than_replaces():
    """A config that dropped `defaults: [default, _self_]`, or `[regression, _self_]` for a
    continuous target, would silently lose the population size, the reflection count and every
    naive-feature setting. These are method parameters, which a dataset's config should not
    restate."""
    import yaml

    for path in sorted((adapter.TASK_CONFIG_DIR / "params_generation").glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text())
        assert loaded.get("defaults") in [[prompts, "_self_"]
                                          for prompts in adapter.PROMPTS_FOR.values()], path.name


def test_a_hand_written_config_agrees_with_the_dataset_description():
    """A hand-written config restates the sequence layout in its own prose, so it and the dataset
    description can drift apart silently, because both files stay valid.

    The test compares the numbers in the layout (how long the window is and where the site sits),
    because a generated feature's positions are interpreted against them.
    """
    import yaml

    for path in sorted((adapter.TASK_CONFIG_DIR / "params_generation").glob("*.yaml")):
        description_path = adapter.DATASET_DIR / f"{path.stem}.json"
        assert description_path.exists(), (
            f"{path.name} has no dataset description in {adapter.DATASET_DIR}; the proposer and "
            f"the judge would have nothing to read for this dataset."
        )
        layout = json.loads(description_path.read_text()).get("sequence_layout", "")
        prompts = " ".join(str(v) for v in yaml.safe_load(path.read_text()).values())

        for number in re.findall(r"\b\d{2,}\b", layout):
            assert number in prompts, (
                f"{path.name} does not mention {number!r}, which {description_path.name} states "
                f"in its sequence layout. One of the two has drifted."
            )


def test_a_withheld_fact_does_not_reach_the_vendored_method(tmp_path):
    """DEFT's generative finder proposes features, so, like the proposer, it must not see a fact
    withheld from the proposing side.

    The config is generated from the same description the judge reads, and that description may
    carry a fact only the judge is told. This test pins the behaviour instead of relying on the
    template's field list, because an edit that widened the list would leak silently: the run
    would still complete, and only a reader who knew the dataset was blinded could tell the result
    was worthless.
    """
    description = {
        "task": "Predicting a thing.",
        "sequence_layout": "100 nt, 0-based.",
        "target_name": "thing-ness",
        "judge_only": "The factor is Widgetin-4.",
    }
    adapter.generate_task_config(description, "thing", tmp_path, "binary")
    written = (tmp_path / adapter.TASK_GROUP / "thing.yaml").read_text()

    assert "Predicting a thing." in written
    assert "Widgetin" not in written and "judge_only" not in written


def test_the_vendored_method_describes_no_dataset():
    """DEFT ships only task-neutral prompts, one config per kind of target, because a method that
    carried a description of one dataset would have to be edited to run another. They are the
    configs the generated ones inherit."""
    shipped = Path(__file__).resolve().parents[1] / "methods" / "deft" / "src" / "deft" / "conf"
    names = {p.stem for p in (shipped / "params_generation").glob("*.yaml")}
    assert names == set(adapter.PROMPTS_FOR.values()), \
        f"a dataset description is back inside the method: {names}"


def test_an_empty_tree_yields_no_features():
    assert adapter.collect_features(a_tree(root=node())) == []


def test_a_feature_that_raises_becomes_nan_not_a_crash():
    """Generated code is expected to be imperfect, so one feature that raises must not cost the
    others."""
    def explodes(X):
        raise RuntimeError("bad generated code")

    tree = a_tree(root=node(
        feature("good", fn=lambda X: np.arange(len(X))),
        left=node(feature("bad", fn=explodes), left=node(), right=node()),
        right=node(),
    ))
    records = adapter.collect_features(tree)
    values = adapter.evaluate_features(records, tree, pd.DataFrame({"x": [1, 2, 3]}))

    assert values.shape == (3, len(records))
    by_name = {r["name"]: values[:, i] for i, r in enumerate(records)}
    assert np.array_equal(by_name["good"], [0, 1, 2])
    assert np.isnan(by_name["bad"]).all()


def test_a_feature_returning_the_wrong_length_becomes_nan():
    tree = a_tree(root=node(
        feature("wrong_length", fn=lambda X: np.array([1.0])), left=node(), right=node()
    ))
    records = adapter.collect_features(tree)
    values = adapter.evaluate_features(records, tree, pd.DataFrame({"x": [1, 2, 3]}))
    assert np.isnan(values[:, 0]).all()


def test_the_adapter_declares_the_contract_version_the_harness_expects():
    """The adapter restates these versions to avoid importing across the environment boundary, so
    the copies are checked here."""
    from seqmaestro.harness.bundle import BUNDLE_VERSION
    from seqmaestro.harness.features import SCHEMA_VERSION

    assert adapter.BUNDLE_VERSION == BUNDLE_VERSION
    assert adapter.FEATURE_SCHEMA_VERSION == SCHEMA_VERSION


# --------------------------------------------------------------------------------------------
# Reporting the recipe, so a feature that both sides can produce hashes the same on both
# --------------------------------------------------------------------------------------------


def test_a_naive_positional_split_is_reported_as_the_recipe_not_as_code():
    """DEFT renders a positional split as a function, and SeqMaestro renders it as
    `{position, symbol}`.

    `definition_digest` hashes normalised source when a record has source, and `{kind, params}`
    when it does not. With the source kept, the two renderings of one feature would take different
    branches and never agree. Emitting the recipe and dropping the source puts both on the same
    branch.
    """
    tree = a_tree(root=node(adapter_feature_positional(), left=node(), right=node()))
    record = adapter.collect_features(tree)[0]
    assert record["kind"] == "positional"
    assert record["params"] == {"position": 50, "symbol": "A"}
    assert record["source"] is None
    assert record["provenance"] == "selected"


def test_deft_and_our_own_positional_features_hash_alike():
    """Comparing DEFT's positional features with ours depends on this. Without matching digests,
    the two arms could only be said to have found features that behave alike, and not the same
    feature."""
    from seqmaestro.features import PositionalFeaturiser
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import definition_digest

    tree = a_tree(root=node(adapter_feature_positional(), left=node(), right=node()))
    theirs = FeatureRecord(**adapter.collect_features(tree)[0])

    ours = PositionalFeaturiser().fit(pd.Series(["ACGT" * 26])).records
    ours = {r.params["position"]: r for r in ours if r.params["symbol"] == "A"}[50]

    assert definition_digest(theirs) == definition_digest(ours)
    assert theirs.name != ours.name, "and it holds despite the names differing, as they always did"


def test_a_threshold_is_a_property_of_the_node_not_of_the_feature():
    """The threshold is recorded per node and kept out of `params`. Every naive split has the same
    0.5 threshold, so in `params` it would describe no feature, and records holding only it would
    all hash alike under one kind."""
    tree = a_tree(root=node(adapter_feature_positional(), left=node(), right=node()))
    record = adapter.collect_features(tree)[0]
    assert "threshold" not in record["params"]
    assert record["extra"]["nodes"][0]["threshold"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------
# Which pools a DEFT arm can be given. Others are refused up front, before anything is spent
# --------------------------------------------------------------------------------------------


def deft_runner():
    from seqmaestro.pipeline.vendored import deft as run_deft
    return run_deft


def test_a_pool_without_positional_is_refused():
    """DEFT always gets the raw sequence and enumerates positional features from it, so a pool that
    omits `positional` would contain it anyway, and the arm's name would be wrong."""
    with pytest.raises(ValueError, match="not compatible"):
        deft_runner().supplied_members("llm_generated")


def test_the_pools_deft_can_be_given_and_what_is_computed_for_each():
    """`positional` DEFT derives itself, the covariates are handed over as the data they are, the
    k-mers are computed on this side as columns, and the features DEFT's generative finder writes
    choose that finder. The k-mers are not handed to the generative finder, whose prompts show every
    column it is given (`seqmaestro.arms`)."""
    runner = deft_runner()
    assert runner.DEFT_POOLS == ("positional", "positional+kmer", "positional+covariates",
                                 "positional+kmer+covariates", "positional+llm_generated_joint",
                                 "positional+llm_generated_joint+covariates")
    assert [runner.supplied_members(pool) for pool in runner.DEFT_POOLS] == [
        (), ("kmer",), (), ("kmer",), (), ()]
    assert [runner.finder_for(pool) for pool in runner.DEFT_POOLS] == [
        "cart", "cart", "cart", "cart", "default", "default"]


# --------------------------------------------------------------------------------------------
# Coverage and impurity at every node, which the tree does not store
#
# `feature.score` is a node's local purity, valid for choosing among candidates at that node and
# meaningless across nodes of different sizes. Ranking features by it puts a tree's root last and
# its deepest, smallest splits first. These tests cover the coverage-weighted arithmetic used
# instead. On a real run, `check_node_statistics` checks that the rows reach the nodes DEFT sent
# them to.


class GiniLike:
    """The shape of DEFT's splitting criterion: `offset - weighted impurity of the two children`.

    It is a stub because these tests must run without the vendored method, and they cover the
    arithmetic built on the criterion, not the criterion itself.
    """

    offset = 1.0

    @staticmethod
    def impurity(y):
        y = np.asarray(y)
        if y.size == 0:
            return 0.0
        shares = np.array([np.mean(y == value) for value in np.unique(y)])
        return float(1.0 - (shares**2).sum())

    def __call__(self, left, right, weights=None):
        n_left, n_right = len(left), len(right)
        total = n_left + n_right
        if total == 0:
            return -np.inf
        return self.offset - (n_left / total) * self.impurity(left) \
            - (n_right / total) * self.impurity(right)


def splitting_feature(column, threshold, score):
    return SimpleNamespace(
        name=column, description="", string="", rationale="", origin="naive",
        threshold=threshold, score=score, fn=lambda frame, c=column: frame[c].to_numpy(),
    )


class VarianceLike:
    """The shape of DEFT's squared-error criterion: `offset` less the children's variances in
    units of the variance of the rows the tree was fitted on."""

    offset = 1.0

    def __init__(self, y):
        self.root_variance = float(np.var(np.asarray(y, dtype=float)))

    def __call__(self, left, right, weights=None):
        n_left, n_right = len(left), len(right)
        total = n_left + n_right
        if total == 0:
            return -np.inf
        variance = [float(np.var(np.asarray(side, dtype=float))) if len(side) else 0.0
                    for side in (left, right)]
        return self.offset - ((n_left / total) * variance[0]
                              + (n_right / total) * variance[1]) / self.root_variance


def fitted(root, criterion=None, weights=None, target_kind="binary"):
    return a_tree(root=root, splitting_criterion=criterion or GiniLike(), class_weight_=weights,
                  target_kind=target_kind)


def scored(criterion, X, y, column, threshold):
    """The score DEFT would record at a node, computed the way DEFT computes it."""
    goes_left = X[column].to_numpy() <= threshold
    return criterion(y[goes_left], y[~goes_left])


def test_node_statistics_recover_coverage_and_a_true_decrease():
    criterion = GiniLike()
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]})
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])
    tree = fitted(node(splitting_feature("a", 0.5, scored(criterion, X, y, "a", 0.5)),
                       left=node(), right=node()), criterion)

    stats = adapter.node_statistics(tree, X, y)
    adapter.check_node_statistics(stats)

    # 1 - ((2/8)^2 + (6/8)^2) = 0.375 at the root; the left child holds 2/2 and the right is pure.
    assert stats[""]["n_samples"] == 8
    assert stats[""]["impurity_decrease"] == pytest.approx(0.375 - 0.5 * 0.5 - 0.5 * 0.0)


def test_a_deep_split_is_weighted_by_the_rows_that_reach_it():
    """A split that separates four rows perfectly has a better local score than one that halves the
    impurity of a hundred, so ranking by local score would prefer it. Weighted by coverage, the
    broad split ranks higher.
    """
    criterion = GiniLike()
    X = pd.DataFrame({"broad": [0.0] * 50 + [1.0] * 50, "narrow": [0.0] * 96 + [1.0] * 4})
    y = pd.Series([0] * 25 + [1] * 25 + [0] * 25 + [1] * 25)
    y.iloc[96:] = [0, 0, 1, 1]

    deep = node(splitting_feature("narrow", 0.5, 0.0), left=node(), right=node())
    tree = fitted(node(splitting_feature("broad", 0.5, 0.0), left=deep, right=node()), criterion)
    stats = adapter.node_statistics(tree, X, y)

    assert stats[""]["n_samples"] == 100
    assert stats["L"]["n_samples"] < stats[""]["n_samples"]

    records = adapter.collect_features(tree, None, stats)
    by_name = {record["name"]: record["importance"] for record in records}
    assert by_name["broad"] > by_name["narrow"], (
        "a split reaching fewer rows outranked one reaching every row")


def test_importance_sums_over_every_node_a_feature_splits_at():
    criterion = GiniLike()
    X = pd.DataFrame({"a": [0.0, 0.0, 1.0, 1.0], "b": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0, 1, 1, 0])
    twice = node(splitting_feature("b", 0.5, 0.0), left=node(), right=node())
    tree = fitted(node(splitting_feature("a", 0.5, 0.0),
                       left=twice,
                       right=node(splitting_feature("b", 0.5, 0.0), left=node(), right=node())),
                  criterion)

    records = adapter.collect_features(tree, None, adapter.node_statistics(tree, X, y))
    by_name = {record["name"]: record for record in records}

    assert len(by_name["b"]["extra"]["nodes"]) == 2, "both occurrences must be kept"
    assert sum(record["importance"] for record in records) == pytest.approx(1.0)
    # `a` separates nothing here and `b` separates everything, at two nodes.
    assert by_name["b"]["importance"] > by_name["a"]["importance"]


def test_importances_are_left_alone_when_nothing_was_split():
    tree = fitted(node())
    stats = adapter.node_statistics(tree, pd.DataFrame({"a": [0.0]}), pd.Series([0]))
    assert stats == {}
    assert adapter.collect_features(tree, None, stats) == []


def test_routing_that_disagrees_with_what_deft_recorded_is_refused():
    """The node statistics can be trusted only because of this check.

    An inverted threshold, the wrong weighting or a misaligned index all show up as a recomputed
    score that differs from the one DEFT recorded, and every importance built on it would be wrong
    while looking plausible.
    """
    criterion = GiniLike()
    X = pd.DataFrame({"a": [0.0, 0.0, 1.0, 1.0]})
    y = pd.Series([0, 0, 1, 1])
    tree = fitted(node(splitting_feature("a", 0.5, 0.123), left=node(), right=node()), criterion)

    with pytest.raises(RuntimeError, match="not reaching the nodes"):
        adapter.check_node_statistics(adapter.node_statistics(tree, X, y))


def test_a_split_with_no_callable_cannot_be_routed():
    feature = splitting_feature("a", 0.5, 0.0)
    feature.fn = None
    tree = fitted(node(feature, left=node(), right=node()))

    with pytest.raises(RuntimeError, match="no callable"):
        adapter.node_statistics(tree, pd.DataFrame({"a": [0.0]}), pd.Series([0]))


def test_routing_reproduces_the_trees_own_predictions():
    """Permuting a DEFT feature means routing rows, because the tree is handed the raw frame and
    computes its features from it, so there is no column to shuffle. If the routing predicted
    something the tree would not, every permutation importance built on it would be wrong."""
    X = pd.DataFrame({"a": [0.0, 0.0, 1.0, 1.0]})
    leaf_low, leaf_high = node(), node()
    leaf_low.prediction, leaf_high.prediction = 0.2, 0.8
    tree = fitted(node(splitting_feature("a", 0.5, 0.0), left=leaf_low, right=leaf_high))
    tree.predict = lambda frame, return_proba=False: np.where(
        frame["a"].to_numpy() <= 0.5, 0.2, 0.8)

    assert adapter.tree_predictions(tree, X) == pytest.approx([0.2, 0.2, 0.8, 0.8])
    adapter.check_tree_predictions(tree, X)


def test_two_nodes_may_share_a_feature_name_and_not_a_function():
    """DEFT attaches a feature to each node, so one name can carry two different callables, such
    as a signed quantity at one node and its sign-flipped twin at another.

    Caching a routed column by name alone would send the second node through the first node's
    values, and `check_tree_predictions` catches that only when the substitution happens to change
    a prediction.
    """
    X = pd.DataFrame({"a": [1.0, -1.0]})
    signed = splitting_feature("a", 0.0, 0.0)
    flipped = splitting_feature("a", 0.0, 0.0)
    flipped.fn = lambda frame: -frame["a"].to_numpy()
    signed.name = flipped.name = "same_name"

    low, high, elsewhere = node(), node(), node()
    low.prediction, high.prediction, elsewhere.prediction = 0.1, 0.9, 0.5
    tree = fitted(node(signed, left=elsewhere, right=node(flipped, left=low, right=high)))
    # Row 0 goes right at the root on `a`, then left at the inner node on `-a`.
    tree.predict = lambda frame, return_proba=False: np.array([0.1, 0.5])

    assert adapter.tree_predictions(tree, X) == pytest.approx([0.1, 0.5])
    adapter.check_tree_predictions(tree, X)


def test_routing_that_disagrees_with_the_tree_is_refused():
    X = pd.DataFrame({"a": [0.0, 0.0, 1.0, 1.0]})
    leaf_low, leaf_high = node(), node()
    leaf_low.prediction, leaf_high.prediction = 0.2, 0.8
    tree = fitted(node(splitting_feature("a", 0.5, 0.0), left=leaf_low, right=leaf_high))
    tree.predict = lambda frame, return_proba=False: np.full(len(frame), 0.5)

    with pytest.raises(RuntimeError, match="different prediction"):
        adapter.check_tree_predictions(tree, X)


def test_permuting_a_feature_costs_the_tree_its_separation():
    X = pd.DataFrame({"a": [0.0] * 10 + [1.0] * 10})
    y = np.array([0] * 10 + [1] * 10)
    leaf_low, leaf_high = node(), node()
    leaf_low.prediction, leaf_high.prediction = 0.1, 0.9
    tree = fitted(node(splitting_feature("a", 0.5, 0.0), left=leaf_low, right=leaf_high))
    tree.predict = lambda frame, return_proba=False: np.where(
        frame["a"].to_numpy() <= 0.5, 0.1, 0.9)

    scores = adapter.permutation_importances(tree, X, y, ["a"], repeats=5,
                                             rng=np.random.default_rng(0))

    # The tree separates the classes perfectly, so shuffling its only feature must cost most of it.
    assert scores["a"] > 0.3


# ---------------------------------------------------------------- which way a split runs


def test_the_adapters_direction_rule_is_the_shared_one():
    """The adapter cannot import `seqmaestro` in the environment where it runs, so it restates the
    direction rule, and an unchecked restatement can drift from the original. Both implementations
    are importable in this test environment, which is the only place they can be compared.
    """
    from seqmaestro.harness.direction import (
        BINARY_WORDS,
        CONTINUOUS_WORDS,
        DIRECTION_KINDS,
        MONOTONICITY_FLOOR,
        NON_MONOTONE,
        from_increments,
    )

    assert adapter.MONOTONICITY_FLOOR == MONOTONICITY_FLOOR
    # The direction words must match as well as the rule that picks between them: `load_features`
    # refuses a word outside this vocabulary, so a typo in the adapter would fail the whole arm at
    # ingest.
    assert adapter.WORDS == {"binary": BINARY_WORDS, "continuous": CONTINUOUS_WORDS}
    assert adapter.NON_MONOTONE == NON_MONOTONE
    assert set(adapter.DIRECTION_KIND.values()) <= set(DIRECTION_KINDS)
    cases = [[], [0.0], [1.0], [-1.0], [3.0, -3.0], [1.0, -0.9], [3.0, -0.5],
             [2.0, np.nan], [1.0, 1.0, -0.5], [-4.0, 1.0], [0.0, 0.0]]
    for words in (BINARY_WORDS, CONTINUOUS_WORDS):
        for increments in cases:
            assert adapter.direction_from_increments(increments, words) == from_increments(
                increments, words), f"the two implementations disagree on {increments}"


def test_node_statistics_record_the_class_rate_on_each_side_of_a_split():
    criterion = GiniLike()
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]})
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])
    tree = fitted(node(splitting_feature("a", 0.5, scored(criterion, X, y, "a", 0.5)),
                       left=node(), right=node()), criterion)

    stats = adapter.node_statistics(tree, X, y)

    # Left is `<= threshold`, matching how the rows are routed, so a higher rate on the right is
    # a feature running with the positive class.
    assert stats[""]["rate_left"] == pytest.approx(0.5)
    assert stats[""]["rate_right"] == pytest.approx(1.0)


def test_a_feature_takes_its_direction_from_the_splits_the_tree_makes_on_it():
    criterion = GiniLike()
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]})
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])
    tree = fitted(node(splitting_feature("a", 0.5, scored(criterion, X, y, "a", 0.5)),
                       left=node(), right=node()), criterion)

    record, = adapter.collect_features(tree, None, adapter.node_statistics(tree, X, y))

    assert record["direction"] == "higher_in_positive"
    assert record["extra"]["nodes"][0]["rate_right"] == pytest.approx(1.0)


def test_a_regression_trees_nodes_record_the_targets_mean_on_each_side():
    X = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]})
    y = pd.Series([1.0, 2.0, 1.0, 2.0, 5.0, 6.0, 5.0, 6.0])
    criterion = VarianceLike(y)
    tree = fitted(node(splitting_feature("a", 0.5, scored(criterion, X, y, "a", 0.5)),
                       left=node(), right=node()), criterion, target_kind="continuous")

    stats = adapter.node_statistics(tree, X, y)
    adapter.check_node_statistics(stats)
    record, = adapter.collect_features(tree, None, stats)

    assert stats[""]["mean_left"] == pytest.approx(1.5)
    assert stats[""]["mean_right"] == pytest.approx(5.5)
    assert "rate_left" not in stats[""]
    # The split leaves the variance within the sides, 0.25, of the root's 4.25.
    assert stats[""]["impurity_decrease"] == pytest.approx(1.0 - 0.25 / 4.25)
    assert record["direction"] == "increases_target"
    assert record["importance"] == pytest.approx(1.0)


def test_a_regression_trees_direction_is_in_the_continuous_words():
    record = {"extra": {"nodes": [{"n_samples": 50, "mean_left": 3.0, "mean_right": 1.0}]}}
    adapter.assign_directions([record], "continuous")
    assert record["direction"] == "decreases_target"


def test_a_regression_trees_permutation_importance_is_lost_r_squared():
    X = pd.DataFrame({"a": [0.0] * 10 + [1.0] * 10})
    y = np.array([1.0] * 10 + [5.0] * 10)
    leaf_low, leaf_high = node(), node()
    leaf_low.prediction, leaf_high.prediction = 1.0, 5.0
    tree = fitted(node(splitting_feature("a", 0.5, 0.0), left=leaf_low, right=leaf_high),
                  target_kind="continuous")
    tree.predict = lambda frame, return_proba=False: np.where(
        frame["a"].to_numpy() <= 0.5, 1.0, 5.0)

    scores = adapter.permutation_importances(tree, X, y, ["a"], repeats=5,
                                             rng=np.random.default_rng(0))

    # The tree explains every row, R-squared 1, and a shuffle leaves about half the rows wrong.
    assert scores["a"] > 0.5


def test_a_continuous_targets_prompts_inherit_defts_regression_prompts(tmp_path):
    description = {"task": "Predicting a number.", "target_name": "the change"}
    adapter.generate_task_config(description, "thing", tmp_path, "continuous")
    written = (tmp_path / adapter.TASK_GROUP / "thing.yaml").read_text()
    assert "defaults:\n  - regression\n" in written


def test_a_continuous_target_is_grown_by_the_squared_error_unless_the_cell_names_one():
    assert adapter.criterion_override("continuous", []) == ["splitting_criterion=squared_error"]
    assert adapter.criterion_override("continuous", ["splitting_criterion=gini"]) == []
    assert adapter.criterion_override("binary", []) == []


def test_a_feature_the_tree_uses_both_ways_has_no_single_direction():
    """A tree may split one feature one way at the root and the other way inside a branch. A net
    sign would report that as a weak direction, but it is a different claim, so it gets its own
    word, `non_monotone`."""
    record = {"extra": {"nodes": [
        {"n_samples": 100, "rate_left": 0.2, "rate_right": 0.8},
        {"n_samples": 100, "rate_left": 0.8, "rate_right": 0.2},
    ]}}
    adapter.assign_directions([record], "binary")

    assert record["direction"] == "non_monotone"


def test_a_split_is_weighted_by_the_rows_reaching_it():
    """A conditional split deep in a branch should not outvote the root, so each split is weighted
    by the rows reaching it, as the importance is."""
    record = {"extra": {"nodes": [
        {"n_samples": 1000, "rate_left": 0.2, "rate_right": 0.8},
        {"n_samples": 10, "rate_left": 0.9, "rate_right": 0.1},
    ]}}
    adapter.assign_directions([record], "binary")

    assert record["direction"] == "higher_in_positive"


def test_a_feature_whose_rows_were_never_routed_has_no_direction():
    """Without `stats` the records still describe the selection, and a direction invented from no
    rows would be a claim the run cannot support."""
    tree = a_tree(root=node(feature("f"), left=node(), right=node()))

    assert adapter.collect_features(tree)[0]["direction"] is None


# --------------------------------------------------------------------------------------------
# The predict mode, which is how the reveal scores a DEFT cell
# --------------------------------------------------------------------------------------------


class ReadsTheFrame:
    """A stand-in tree: predicts each sequence's share of G, and remembers the frame it was handed,
    whose columns matter as much as its values."""

    def __init__(self):
        self.seen = None

    def predict(self, X, return_proba=False):
        self.seen = X
        return X["sequence"].str.count("G").to_numpy() / X["sequence"].str.len().to_numpy()


def test_predict_returns_the_identifiers_and_hands_the_tree_the_rest():
    tree = ReadsTheFrame()
    rows = pd.DataFrame({"row_id": [2, 0, 1], "sequence": ["GGAA", "GAAA", "AAAA"]})

    ids, scores = adapter.predict_rows(tree, rows, "row_id")

    assert ids.tolist() == [2, 0, 1], "in the order the rows came"
    assert scores.tolist() == [0.5, 0.25, 0.0]
    assert list(tree.seen.columns) == ["sequence"], "the identifier is not a feature"


def test_predict_refuses_rows_without_their_identifiers():
    with pytest.raises(ValueError, match="identifier column"):
        adapter.predict_rows(ReadsTheFrame(), pd.DataFrame({"sequence": ["ACGT"]}), "row_id")


def test_the_deft_inputs_hold_no_test_split(tmp_path):
    """The inputs hold only the training, validation and effect rows. The test split reaches DEFT
    only as unlabelled rows at the reveal."""
    runner = deft_runner()
    frame = pd.DataFrame({"sequence": ["ACGT", "AGGT", "ACCT", "TTGA"], "label": [1, 0, 1, 0]})
    paths = runner.write_inputs("positional", frame, frame["label"].to_numpy(), frame, frame,
                                tmp_path / "inputs",
                                columns=Columns("sequence", "label", weight=None, covariates=(), target_kind=BINARY))

    assert sorted(paths) == ["effect", "train", "val"]
    assert sorted(path.name for path in (tmp_path / "inputs").iterdir()) == \
           ["columns.json", "effect.csv", "train.csv", "val.csv"]


def stub_adapter(tmp_path, reverse: bool) -> Path:
    """A stand-in for DEFT's adapter. It reads the rows and writes scores beside the identifiers it
    was given, or beside those identifiers reversed, which is the misalignment they exist to catch."""
    script = tmp_path / "stub_adapter.py"
    script.write_text(
        "import argparse, numpy as np, pandas as pd\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('mode')\n"
        "for flag in ('--bundle', '--rows', '--id-column', '--out'):\n"
        "    parser.add_argument(flag)\n"
        "args = parser.parse_args()\n"
        "rows = pd.read_csv(args.rows)\n"
        "ids = rows[args.id_column].to_numpy()\n"
        f"ids = ids[::-1] if {reverse} else ids\n"
        "scores = rows['sequence'].str.count('G').to_numpy() / 4.0\n"
        "np.savez_compressed(args.out, row_id=ids, score=scores)\n"
    )
    return script


def handed_the_sequence_alone(cell: Path) -> None:
    """The record of what a cell was handed, for a cell handed the sequence and no other column."""
    (cell / "inputs").mkdir(parents=True)
    (cell / "inputs" / "columns.json").write_text(json.dumps(
        {"sequence": {"column": "sequence", "window": None}, "columns": {}}))


def test_a_cell_is_predicted_through_the_adapter_by_row_identifier(tmp_path, monkeypatch):
    import sys

    runner = deft_runner()
    monkeypatch.setattr(runner, "DEFT_PYTHON", Path(sys.executable))
    monkeypatch.setattr(runner, "ADAPTER", stub_adapter(tmp_path, reverse=False))
    sequences = pd.Series(["GGGG", "GAAA", "AAAA"], index=[10, 11, 12])
    handed_the_sequence_alone(tmp_path)

    scores = runner.predict_cell(tmp_path, sequences, "sequence")
    assert scores.tolist() == [1.0, 0.25, 0.0]

    monkeypatch.setattr(runner, "ADAPTER", stub_adapter(tmp_path, reverse=True))
    with pytest.raises(RuntimeError, match="different order"):
        runner.predict_cell(tmp_path, sequences, "sequence")



# --------------------------------------------------------------------------------------------
# What DEFT is handed about its columns, and what comes back
# --------------------------------------------------------------------------------------------

COVARIATE_ROWS = pd.DataFrame({"sequence": ["ACGTAC", "AGGT", "ACCTAG", "TTGA"],
                               "signal": [0.5, 1.5, 2.5, 3.5], "noise": [0.1, 0.2, 0.3, 0.4],
                               "label": [1, 0, 1, 0]})
ALL_COLUMNS = Columns("sequence", "label", weight=None, covariates=("signal", "noise"), target_kind=BINARY)
DESCRIBED = {"signal": "a measured signal", "noise": "a measurement unrelated to the class"}


def write_covariate_inputs(tmp_path, window=None):
    from seqmaestro.features import PositionalWindow

    runner = deft_runner()
    runner.write_inputs("positional+covariates", COVARIATE_ROWS,
                        COVARIATE_ROWS["label"].to_numpy(), COVARIATE_ROWS, COVARIATE_ROWS,
                        tmp_path / "inputs", columns=ALL_COLUMNS, descriptions=DESCRIBED,
                        window=None if window is None else PositionalWindow(**window))
    return runner, tmp_path / "inputs"


def test_a_covariate_is_handed_over_as_it_is_and_comes_back_as_itself(tmp_path):
    _, inputs = write_covariate_inputs(tmp_path)
    train = pd.read_csv(inputs / "train.csv")
    assert list(train.columns) == ["sequence", "signal", "noise", "label"]
    assert train["signal"].tolist() == COVARIATE_ROWS["signal"].tolist()

    recipe = adapter.naive_recipe("value_of_signal", adapter.read_pool(inputs))
    assert recipe == {"kind": "covariate", "params": {"column": "signal"}, "source": None,
                      "description": "a measured signal"}, "as SeqMaestro's covariate records it"
    tree = a_tree(root=node(feature("value_of_signal"), left=node(), right=node()))
    [record] = adapter.collect_features(tree, adapter.read_pool(inputs))
    assert (record["kind"], record["description"]) == ("covariate", "a measured signal")


def test_deft_is_told_what_each_column_means_and_how_to_read_positions(tmp_path):
    import yaml

    _, inputs = write_covariate_inputs(tmp_path, {"length": 4, "anchor": "end", "pad": True})
    told = adapter.dataset_config(inputs)
    assert told == {"column_descriptions": DESCRIBED,
                    "positional_window": {"length": 4, "anchor": "end", "pad": True}}
    adapter.write_dataset_config(told, tmp_path / "conf")
    written = yaml.safe_load((tmp_path / "conf" / "dataset" / "handed.yaml").read_text())
    assert written == {"defaults": ["generic_csv", "_self_"], **told}


def test_a_sequence_alone_tells_deft_nothing_new(tmp_path):
    runner = deft_runner()
    runner.write_inputs("positional", COVARIATE_ROWS, COVARIATE_ROWS["label"].to_numpy(),
                        COVARIATE_ROWS, COVARIATE_ROWS, tmp_path / "inputs",
                        columns=ALL_COLUMNS.without_covariates())
    assert adapter.dataset_config(tmp_path / "inputs") == {}
    assert list(pd.read_csv(tmp_path / "inputs" / "train.csv").columns) == ["sequence", "label"]


def test_every_later_frame_holds_the_covariates_the_cell_was_handed(tmp_path):
    runner, _ = write_covariate_inputs(tmp_path)
    frame = runner.input_frame(tmp_path, ALL_COLUMNS.inputs(COVARIATE_ROWS), "sequence")
    assert list(frame.columns) == ["sequence", "signal", "noise"]
    handed_the_sequence_alone(tmp_path / "elsewhere")
    alone = runner.input_frame(tmp_path / "elsewhere", ALL_COLUMNS.inputs(COVARIATE_ROWS),
                               "sequence")
    assert list(alone.columns) == ["sequence"], "a cell handed no covariate is given none"


def test_the_kmers_are_handed_over_as_counted_and_come_back_as_kmers(tmp_path):
    """The k-mers are counted on this side, with the run's k-mer settings, and DEFT's naive
    generator splits on each count as it would on a covariate. A split on one comes back with the
    recipe the featuriser recorded for it."""
    runner = deft_runner()
    runner.write_inputs("positional+kmer", COVARIATE_ROWS, COVARIATE_ROWS["label"].to_numpy(),
                        COVARIATE_ROWS, COVARIATE_ROWS, tmp_path / "inputs",
                        columns=ALL_COLUMNS.without_covariates(), k_range=(1, 2),
                        kmer_normalise="per_length")
    train = pd.read_csv(tmp_path / "inputs" / "train.csv")
    assert list(train.columns)[:3] == ["sequence", "label", "kmer_A"]
    assert len(train.columns) == 2 + 4 + 11, "the sequence, the label, and every 1- and 2-mer seen"
    assert train["kmer_A"].tolist() == pytest.approx([2 / 6, 1 / 4, 2 / 6, 1 / 4])

    recipe = adapter.naive_recipe("value_of_kmer_AG", adapter.read_pool(tmp_path / "inputs"))
    assert (recipe["kind"], recipe["params"]) == ("kmer", {"kmer": "AG", "k": 2, "normalise": "per_length"})
    frame = runner.input_frame(tmp_path, ALL_COLUMNS.without_covariates().inputs(COVARIATE_ROWS),
                               "sequence")
    assert list(frame.columns) == list(train.columns.drop("label")), \
        "every later stage hands the tree the columns it was fitted on"


def test_a_handed_covariate_needs_its_description(tmp_path):
    runner = deft_runner()
    with pytest.raises(ValueError, match="no description"):
        runner.write_inputs("positional+covariates", COVARIATE_ROWS,
                            COVARIATE_ROWS["label"].to_numpy(), COVARIATE_ROWS, COVARIATE_ROWS,
                            tmp_path / "inputs", columns=ALL_COLUMNS,
                            descriptions={"signal": "a measured signal"})


def test_a_hand_written_prompt_config_joins_the_generated_one_on_one_search_path(tmp_path):
    written = tmp_path / "hand" / "params_generation"
    written.mkdir(parents=True)
    (written / "mytask.yaml").write_text("target_name: x\n")
    generated = tmp_path / "generated"
    overrides = adapter.on_one_search_path(
        [f"--config-dir={tmp_path / 'hand'}", "params_generation=mytask"], generated)
    assert overrides == [f"--config-dir={generated}", "params_generation=mytask"]
    assert (generated / "params_generation" / "mytask.yaml").read_text() == "target_name: x\n"
    assert adapter.on_one_search_path([], generated) == [f"--config-dir={generated}"]


def test_a_repaired_deft_feature_says_so_in_its_record():
    repaired = feature("g_count", origin="llm")
    repaired.repaired, repaired.renamed_from = True, "g_per_t"
    tree = a_tree(root=node(repaired, left=node(), right=node()))
    [record] = adapter.collect_features(tree)
    assert record["extra"]["repaired"] is True and record["extra"]["renamed_from"] == "g_per_t"
    plain = a_tree(root=node(feature("f", origin="llm"), left=node(), right=node()))
    assert "repaired" not in adapter.collect_features(plain)[0]["extra"]


def test_a_deft_cell_is_fitted_again_only_where_no_grown_tree_is_at_stake(tmp_path):
    """DEFT refuses a directory an earlier fit wrote in. The Core side clears it where a second
    fit chooses nothing: after a fit that saved no tree, and under `--force`. A bundle with a tree
    and no finished cell is completed from the tree, since a second fit would draw another."""
    from seqmaestro.pipeline.vendored.deft import make_room

    bundle = tmp_path / "bundle"
    make_room(bundle, force=False)
    assert not bundle.exists(), "nothing was there, and nothing is made"

    (bundle / "deft_run").mkdir(parents=True)
    (bundle / "deft_run" / "intermediate_tree.pkl").write_bytes(b"")
    make_room(bundle, force=False)
    assert not bundle.exists(), "a fit that saved no tree left nothing to choose between"

    (bundle / "deft_run").mkdir(parents=True)
    (bundle / "deft_run" / "tree_generic.dill").write_bytes(b"")
    with pytest.raises(RuntimeError, match="--reuse-tree"):
        make_room(bundle, force=False)
    assert (bundle / "deft_run" / "tree_generic.dill").exists(), "the grown tree is left alone"
    make_room(bundle, force=True)
    assert not bundle.exists()
