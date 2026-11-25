"""The report layer: names, placeholders, the headline rule, concepts, model internals, and a run
built end to end into report inputs.

The figures need matplotlib, which the Core environment does not include, so what they draw is
tested here through `figure_inputs.json`. A static check confirms that nothing outside
`seqmaestro/report/figures/` imports matplotlib.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.stability import BehaviouralMatcher
from seqmaestro.models import L1Linear, L1Logistic
from seqmaestro.models.sparse import LeastSquares
from seqmaestro.report import (
    concepts,
    facts,
    files,
    internals,
    markdown,
    naming,
    placeholders,
    spreadsheet,
    tables,
)
from tests.planted import ANALYSES, LANGUAGE_MODEL, planted_dataset

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "experiments" / "analysis"
FIGURES = REPO / "seqmaestro" / "report" / "figures"


# --------------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------------


def test_an_arm_is_named_for_a_reader():
    configuration = naming.parse("ebm/llm_generated_iterative")
    assert configuration.label == "EBM · written, iterative"
    assert configuration.is_interpretable and configuration.writes_features
    assert naming.parse("l1_logistic/positional+kmer").label == "L1 Logistic Regression · positions + k-mers"
    assert naming.slug("l1_logistic/positional+kmer") == "l1_logistic__positional+kmer"


def test_a_pool_is_named_from_its_members_so_covariates_need_no_entry_of_their_own():
    assert naming.parse("ebm/positional+covariates").label == "EBM · positions + covariates"
    with_them = naming.parse("xgboost/llm_generated_iterative+covariates")
    assert with_them.label == "XGBoost · written, iterative + covariates"
    assert with_them.writes_features
    assert naming.parse("deft_or_cart/positional+llm_generated_joint+covariates").source_short == \
        "positions + written per split + covariates"
    assert naming.parse("deft_or_cart/positional+covariates").source_short == "positions + covariates"
    assert naming.parse("l1_logistic/positional+kmer+covariates").source_long == (
        "single-base position indicators, k-mer counts and measurements supplied with "
        "each sequence, used as they are")
    assert naming.parse("xgboost/motif_known").source_short == "known motifs"
    assert not naming.parse("l1_logistic/covariates").writes_features



def test_a_written_source_handed_a_vocabulary_beside_it_is_named_for_both():
    beside = naming.parse("l1_logistic/kmer+llm_generated_iterative")
    assert beside.label == "L1 Logistic Regression · k-mers + written, iterative"
    assert beside.label != naming.parse("l1_logistic/llm_generated_iterative").label
    assert beside.writes_features
    assert naming.parse("l1_logistic/positional+kmer+llm_generated").source_short == \
        "positions + k-mers + written"
    assert naming.parse("igsr/llm_generated_joint+covariates").source_short == \
        "written terms + covariates"


def test_the_key_mentions_the_covariates_only_where_an_arm_was_given_them():
    def sources(arms):
        return [row["short"] for row in naming.key_table(arms, "binary") if row["kind"] == "feature source"]

    assert "covariates" not in sources(["ebm/positional", "l1_logistic/llm_generated"])
    assert "covariates" in sources(["ebm/positional", "ebm/positional+covariates"])


def test_the_key_names_only_the_models_and_sources_the_run_used():
    rows = naming.key_table(["l1_linear/positional", "igsr/llm_generated_joint"], "continuous")
    assert [row["short"] for row in rows if row["kind"] == "model"] == \
        ["L1 Linear Regression", "IGSR"]
    assert [row["short"] for row in rows if row["kind"] == "feature source"] == \
        ["positions", "written terms"]


def test_the_key_describes_the_joint_source_once_for_each_model_that_wrote_it():
    def sources(arms):
        return [row["short"] for row in naming.key_table(arms, "continuous")
                if row["kind"] == "feature source"]

    both = sources(["deft_or_cart/positional+llm_generated_joint", "igsr/llm_generated_joint"])
    assert "written per split" in both and "written terms" in both
    assert "written terms" not in sources(["deft_or_cart/positional+llm_generated_joint"])


def test_whether_a_model_is_interpreted_comes_from_the_registry():
    assert not naming.parse("xgboost/positional").is_interpretable
    assert naming.parse("ebm/positional").is_interpretable
    assert naming.parse("deft_or_cart/positional+llm_generated_joint").is_interpretable, \
        "DEFT's tree is interpreted node by node"


@pytest.mark.parametrize(("arm", "error"), [
    ("svm/positional", KeyError), ("ebm/one_hot", KeyError), ("ebm__positional", ValueError),
    ("ebm", ValueError),
])
def test_an_unknown_name_raises_rather_than_reaching_a_reader(arm, error):
    with pytest.raises(error):
        naming.parse(arm)


# --------------------------------------------------------------------------------------------
# Placeholders
# --------------------------------------------------------------------------------------------


@pytest.fixture
def sources(tmp_path):
    components, tables = tmp_path / "components", tmp_path / "tables"
    components.mkdir()
    tables.mkdir()
    (components / "intro.md").write_text("<!-- maintenance note -->\n### Intro\n\nOn {{param:n}} rows.\n")
    (tables / "summary.md").write_text("| a |\n|---|\n| 1 |\n")
    (tables / "spare.md").write_text("| b |\n|---|\n| 2 |\n")
    return placeholders.Sources(
        components=components, tables=tables, parameters={"n": "ten"}, target_kind="binary",
        figures={"grid": {"path": "figures/grid.png", "caption": "The grid", "width": "85%"},
                 "other": {"path": "figures/other.png", "caption": "Other", "width": "100%"}})


def test_a_component_with_twins_is_the_one_for_the_runs_kind(sources):
    (sources.components / "metric.binary.md").write_text("AUROC is the measure.\n")
    (sources.components / "metric.continuous.md").write_text("R-squared is the measure.\n")
    assert placeholders.expand("{{component:metric}}", sources) == "AUROC is the measure."
    sources.target_kind = "continuous"
    assert placeholders.expand("{{component:metric}}", sources) == "R-squared is the measure."
    assert "On ten rows." in placeholders.expand("{{component:intro}}", sources), \
        "a component without twins serves both kinds"


def test_every_component_with_twins_has_one_for_each_kind_and_no_plain_file():
    components = files.COMPONENTS_DIR
    names = {path.name.split(".")[0] for path in components.glob("*.md")}
    for name in sorted(names):
        twins = sorted(path.name for path in components.glob(f"{name}.*.md"))
        if twins:
            assert twins == [f"{name}.binary.md", f"{name}.continuous.md"], twins
            assert not (components / f"{name}.md").exists(), name


def test_placeholders_are_filled_and_a_components_own_parameters_too(sources):
    text = placeholders.expand("{{component:intro}}\n\n{{table:summary}}\n\n{{figure:grid}}\n", sources)
    assert "maintenance note" not in text and "On ten rows." in text
    assert "| 1 |" in text and "![The grid](figures/grid.png){width=85%}" in text
    assert placeholders.unused(sources) == {"table": ["spare"], "figure": ["other"]}


@pytest.mark.parametrize("source", ["{{param:m}}", "{{table:absent}}", "{{figure:absent}}",
                                    "{{component:absent}}", "{{tabel:summary}}", "{{ table:summary }}"])
def test_an_unresolved_placeholder_raises(sources, source):
    with pytest.raises(placeholders.PlaceholderError):
        placeholders.expand(f"Before. {source} After.", sources)


# --------------------------------------------------------------------------------------------
# The headline rule
# --------------------------------------------------------------------------------------------


def test_the_headline_set_is_one_per_interpretable_class_the_best_enumerated_and_the_reference():
    scores = {"ebm/positional": 0.70, "ebm/llm_generated_iterative": 0.80,
              "l1_logistic/positional": 0.75, "l1_logistic/llm_generated": 0.72,
              "deft_or_cart/positional": 0.60, "xgboost/positional": 0.90,
              "xgboost/llm_generated": 0.85, "ebm/llm_generated": None}
    assert facts.headline_set(scores, "binary") == {
        "best_ebm": "ebm/llm_generated_iterative", "best_l1_logistic": "l1_logistic/positional",
        "best_deft_or_cart": "deft_or_cart/positional", "best_enumerated": "l1_logistic/positional",
        "black_box_reference": "xgboost/positional"}


def test_a_continuous_headline_set_takes_the_least_squares_model_and_scores_below_zero():
    """An R-squared below zero is still a score to rank by. For a continuous target, the sparse
    linear slot is the least-squares model."""
    scores = {"ebm/positional": -0.05, "l1_linear/positional": 0.03,
              "l1_linear/llm_generated": -0.2, "xgboost/positional": -0.1}
    assert facts.headline_set(scores, "continuous") == {
        "best_ebm": "ebm/positional", "best_l1_linear": "l1_linear/positional",
        "best_enumerated": "l1_linear/positional", "black_box_reference": "xgboost/positional"}


def test_a_tie_in_the_headline_rule_goes_to_the_arm_named_first():
    assert facts.headline_set({"l1_logistic/positional": 0.8, "l1_logistic/llm_generated": 0.8},
                              "binary")["best_l1_logistic"] == "l1_logistic/llm_generated"


def test_a_class_nobody_reduced_has_no_headline():
    assert facts.headline_set({"xgboost/positional": 0.9}, "binary") == \
        {"black_box_reference": "xgboost/positional"}


# --------------------------------------------------------------------------------------------
# Concepts
# --------------------------------------------------------------------------------------------


#: Tier 2's association measure for a binary target, and its value when there is no association.
AUC = ("auc", 0.5)


def arm_columns(arm, columns: dict[str, np.ndarray], effects: dict[str, float],
                evidence: dict[str, dict] | None = None):
    records = [FeatureRecord(name=name, description=f"what {name} computes",
                             kind="llm_generated" if arm.endswith("llm_generated") else "positional")
               for name in columns]
    features = [{"name": name, "rank": rank, "kind": record.kind, "description": record.description,
                 "marginal_effect": effects[name]}
                for rank, (name, record) in enumerate(zip(columns, records, strict=True), start=1)]
    return concepts.ArmColumns(arm=arm, records=records, values=np.column_stack(list(columns.values())),
                               features=features, evidence=evidence or {})


def test_features_that_compute_one_thing_are_one_concept_whatever_they_are_called():
    rng = np.random.default_rng(0)
    signal, other, noise = rng.normal(size=200), rng.normal(size=200), rng.normal(size=200)
    found = concepts.build([
        arm_columns("ebm/llm_generated", {"gc_upstream": signal, "a_run": other},
                    {"gc_upstream": 0.4, "a_run": 0.1}),
        arm_columns("l1_logistic/llm_generated", {"upstream_gc_share": 3 * signal + 1},
                    {"upstream_gc_share": 0.3}),
        arm_columns("l1_logistic/positional", {"pos_4_G": np.exp(signal), "pos_9_T": noise},
                    {"pos_4_G": 0.2, "pos_9_T": -0.2}),
    ], BehaviouralMatcher(), AUC)

    shared = found[0]
    assert shared.arms == ["ebm/llm_generated", "l1_logistic/llm_generated", "l1_logistic/positional"]
    assert shared.member_names == ["gc_upstream", "pos_4_G", "upstream_gc_share"]
    assert shared.label == "gc_upstream", "named by the member ranked highest"
    assert shared.direction == "positive" and shared.written_and_enumerated
    assert [len(concept.arms) for concept in found] == [3, 1, 1]


def test_a_concept_whose_members_disagree_on_a_sign_carries_no_direction():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=200)
    found = concepts.build([
        arm_columns("ebm/llm_generated", {"gc": signal}, {"gc": 0.4}),
        arm_columns("l1_logistic/llm_generated", {"at": -signal}, {"at": -0.4}),
        arm_columns("l1_logistic/positional", {"pos": signal}, {"pos": 0.3}),
    ], BehaviouralMatcher(), AUC)
    assert len(found) == 1 and found[0].direction is None


def test_a_concepts_strongest_association_is_the_furthest_from_the_value_meaning_none():
    """A rank correlation means no association at 0, whereas an AUROC means none at a half, so
    -0.4 is the stronger of the correlations -0.4 and 0.3, and 0.1 the stronger of the AUROCs 0.1
    and 0.7."""
    rng = np.random.default_rng(1)
    signal = rng.normal(size=200)
    arms = [arm_columns("ebm/llm_generated", {"gc": signal}, {"gc": 0.4}, {"gc": {"rho": -0.4}}),
            arm_columns("l1_linear/positional", {"pos": signal}, {"pos": 0.3}, {"pos": {"rho": 0.3}})]
    assert concepts.build(arms, BehaviouralMatcher(), ("rho", 0.0))[0].strongest_association == -0.4
    arms = [arm_columns("ebm/llm_generated", {"gc": signal}, {"gc": 0.4}, {"gc": {"auc": 0.1}}),
            arm_columns("l1_logistic/positional", {"pos": signal}, {"pos": 0.3}, {"pos": {"auc": 0.7}})]
    assert concepts.build(arms, BehaviouralMatcher(), AUC)[0].strongest_association == 0.1


# --------------------------------------------------------------------------------------------
# Printing numbers and what is inside a model
# --------------------------------------------------------------------------------------------


def test_a_table_cell_holds_one_line_and_every_row_its_columns():
    assert markdown.render(["a", "b"], [[1, None]]) == "| a | b |\n|---|---|\n| 1 |  |"
    with pytest.raises(ValueError, match="one line"):
        markdown.render(["a"], [["two\nlines"]])
    with pytest.raises(ValueError, match="1 cells for 2 headers"):
        markdown.render(["a", "b"], [[1]])


def test_a_table_gives_its_columns_the_widths_it_is_asked_for():
    assert markdown.render(["a", "b"], [[1, 2]], widths=[2, 5]) == "| a | b |\n|--|-----|\n| 1 | 2 |"
    for widths in ([2], [2, 0]):
        with pytest.raises(ValueError, match="positive widths"):
            markdown.render(["a", "b"], [[1, 2]], widths=widths)


def test_a_description_reaches_its_table_whole_on_one_line_with_its_pipes_escaped():
    # A specification of the kind DEFT writes, which says what it counts only after the stretch it
    # measures and runs longer than any length a cell could be truncated to.
    description = ("For each sequence, take the 50 nt immediately upstream of the exon's 3' splice "
                   "site, at positions 150 to 199 of the window.\nCount the bases that are C or "
                   "T | U there, and divide by 50. " + "The value is a fraction. " * 10)
    cell = tables.what_it_computes({"description": description})
    assert "\n" not in cell and cell.endswith("The value is a fraction.")
    assert "Count the bases that are C or T \\| U there" in cell
    row = markdown.render(["feature", "what it computes"], [["x", cell]]).splitlines()[-1]
    assert len(re.split(r"(?<!\\)\|", row)) == 4, "two cells between three unescaped pipes"


AZURE_CALLS = {"provider": "azure", "model": "azure/a-deployment", "api_version": "v1",
               "base_model": "azure/gpt-5.4", "reasoning_effort": "minimal", "retries": 5,
               "timeout_seconds": 120}


def a_cell(record: dict) -> SimpleNamespace:
    return SimpleNamespace(run=record)


def test_the_facts_name_each_model_the_run_called_and_what_each_part_called_it_for():
    """Every record of the calls is gathered: a proposer's inside its featuriser (twice in an
    iterative cell), a vendored cell's at the top of its record, the judge's and the grouping's.
    One model under one setting makes one entry, whereas a second setting makes a second entry,
    which the report has to explain."""
    proposer = {"featuriser": {"kind": "composed", "parts": [
        {"kind": "positional"}, {"kind": "generated", "proposer": {"language_model": AZURE_CALLS}}]}}
    iterative = {"featuriser": {"pool": {"proposer": {"language_model": AZURE_CALLS}},
                                "proposer": {"language_model": AZURE_CALLS}}}
    effortful = {**AZURE_CALLS, "reasoning_effort": "high"}
    run = SimpleNamespace(
        suite={"l1_logistic/positional+llm_generated": {"real": {1: a_cell(proposer)}},
               "ebm/llm_generated_iterative": {"real": {1: a_cell(iterative)},
                                               "permuted": {1: a_cell(iterative)}},
               "deft_or_cart/positional+llm_generated_joint": {
                   "real": {1: a_cell({"language_model": effortful})}},
               "deft_or_cart/positional": {"real": {1: a_cell({"model": "deft_or_cart"})}}},
        analyses={"judge": {"language_model": AZURE_CALLS}, "concept_recurrence": None,
                  "stability": pd.DataFrame()})
    found = facts.language_models(run)
    assert found == [
        {**AZURE_CALLS, "used_for": ["llm_generated", "llm_generated_iterative", "judge"],
         "n_records": 4},
        {**effortful, "used_for": ["llm_generated_joint/deft_or_cart"], "n_records": 1}]
    assert facts.language_models(SimpleNamespace(suite={}, analyses={})) == []


def test_the_table_of_language_models_names_each_model_as_a_reader_knows_it():
    table = tables.language_models([
        {**AZURE_CALLS, "used_for": ["llm_generated", "judge"], "n_records": 2},
        {"provider": "claude", "model": "anthropic/claude-x", "reasoning_effort": None,
         "retries": 5, "timeout_seconds": 120, "used_for": ["recurrence"], "n_records": 1}])
    rows = table.splitlines()[2:]
    assert rows[0] == ("| gpt-5.4 (the deployment a-deployment, on Azure OpenAI) | minimal | Wrote "
                       "features in one round; rated how meaningful the features look |")
    assert rows[1] == ("| claude-x (through the Claude API) | none stated | Grouped the written "
                       "features into ideas, to count those found again |")
    assert "No part of this run called a language model" in tables.language_models([])
    with pytest.raises(KeyError, match=r"naming\.PROVIDERS"):
        naming.language_model({"provider": "another", "model": "x/y"})


def test_the_table_of_agents_gives_one_row_to_each_agent_and_model_with_its_stages():
    table = tables.agents([
        {"agent": "Claude Code", "model": "claude-opus-5-5", "stages": ["1", "2"]},
        {"agent": "Claude Code", "model": "claude-sonnet-5-5", "stages": ["3A"]},
        {"agent": "Claude Code", "model": "claude-opus-5-5", "stages": ["3B", "report"]}])
    assert table.splitlines()[2:] == [
        "| Claude Code | claude-opus-5-5 | Carried out Stages 1, 2 and 3B, and wrote this report |",
        "| Claude Code | claude-sonnet-5-5 | Carried out Stage 3A |"]
    assert "No agent recorded itself in this run" in tables.agents([])


def test_numbers_are_printed_at_the_precision_they_are_worth():
    assert [markdown.q_value(v) for v in (None, 0.0004, 0.0123, 0.9999)] == ["—", "<0.001", ".012", "1.00"]
    assert markdown.signed(-0.0002) == "+0.000" and markdown.signed(-0.0126) == "-0.013"
    assert markdown.word(10) == "ten" and markdown.word(1200) == "1,200"


def test_a_model_that_does_not_reproduce_its_cell_is_refused():
    assert internals.check_reproduces([0.1, 0.9], [0.1, 0.9], "cell") == 0.0
    with pytest.raises(internals.InternalsError, match="would not be of the model"):
        internals.check_reproduces([0.1, 0.9], [0.1, 0.8], "cell")


def test_a_model_that_used_no_feature_is_exported_as_such_and_draws_nothing():
    """A reduction can end with a model that uses no feature. It predicts one value for every row,
    so its export holds that value and the report builds neither a figure nor an equation for it."""
    from seqmaestro.reduction.arm import NoFeatures

    y = np.array([0, 1, 1, 1])
    empty = NoFeatures("binary").fit(pd.DataFrame(index=range(4)), y)
    exported = internals.no_features(empty, empty.predict(pd.DataFrame(index=range(4))), "cell")
    assert exported == {"kind": internals.NO_FEATURES, "prediction": 0.75}
    assert tables.equation({}, {"best_pysr": exported}) == {}

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=200)})
    labels = (frame["a"] + rng.normal(scale=0.3, size=200) > 0).astype(int).to_numpy()
    used = L1Logistic(seed=0, class_weight=None).fit(frame, labels)
    assert internals.no_features(used, used.predict(frame), "cell") is None
    with pytest.raises(internals.InternalsError, match="used no feature"):
        internals.no_features(empty, [0.2, 0.9], "cell")


def test_an_l1_weight_is_per_standard_deviation_whatever_the_units_of_its_feature():
    """Rescaling a column leaves its exported weight unchanged and scales the standard deviation
    exported beside it, because the weight is per standard deviation of its feature."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)
    found = []
    for factor in (1.0, 100.0):
        frame = pd.DataFrame(X * np.array([factor, 1.0, 1.0]), columns=["a", "b", "c"])
        model = L1Logistic(C=1.0, seed=0).fit(frame, y)
        found.append({row["name"]: row for row in internals.l1_weights(
            model, ["a", "b", "c"], {"a": 0.4, "b": None, "c": None})["features"]})
    assert found[0]["a"]["weight"] == pytest.approx(found[1]["a"]["weight"], rel=1e-6)
    assert found[0]["a"]["weight"] > 0 and found[1]["a"]["marginal_effect"] == 0.4
    assert found[1]["a"]["sd_on_fitting_rows"] == pytest.approx(100 * found[0]["a"]["sd_on_fitting_rows"])


def test_a_least_squares_models_weights_are_in_the_targets_units():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = 2.0 * frame["a"].to_numpy() + rng.normal(size=300)
    exported = internals.l1_weights(L1Linear(seed=0).fit(frame, y), ["a", "b", "c"],
                                    {"a": 0.6, "b": None, "c": None})
    weights = {row["name"]: row["weight"] for row in exported["features"]}
    assert exported["scale"] == internals.SCALE["continuous"]
    assert weights["a"] == pytest.approx(2.0, abs=0.2), "per standard deviation of a unit-sd feature"
    assert isinstance(exported["intercept"], float)


def test_an_igsr_models_weights_are_read_as_the_lassos_are():
    """IGSR's least-squares model is drawn with the Lasso's weight chart: per standard deviation, in
    the target's units, with its own importance kind."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(0, 5, size=(200, 2)), columns=["a", "b"])
    y = 0.4 * frame["a"].to_numpy() + rng.normal(size=200)
    exported = internals.l1_weights(LeastSquares().fit(frame, y), ["a", "b"], {"a": 0.5, "b": None})
    weights = {row["name"]: row["weight"] for row in exported["features"]}
    assert exported["importance_kind"] == "least_squares_coefficient_per_sd"
    assert weights["a"] == pytest.approx(0.4 * frame["a"].std(ddof=0), rel=0.1)


@pytest.mark.parametrize(("weights", "decimals"), [
    ([0.4412, -0.1], 2), ([-1.9, 1.2], 2), ([0.0048, -0.0015], 4), ([0.05, 0.03], 3), ([0.0], 2),
])
def test_a_weight_chart_keeps_the_decimals_its_weights_need(weights, decimals):
    from seqmaestro.report.figure_inputs import weight_decimals

    assert weight_decimals(weights) == decimals


# --------------------------------------------------------------------------------------------
# Where matplotlib may be imported
# --------------------------------------------------------------------------------------------


def imported(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_only_the_figures_import_matplotlib():
    """Core produces results and has no plotting stack, so the figures are drawn under
    `SeqMaestro_LocalUtils`."""
    for path in (REPO / "seqmaestro").rglob("*.py"):
        if FIGURES in path.parents:
            continue
        assert not any(name.split(".")[0] in {"matplotlib", "PIL"} for name in imported(path)), path


def test_the_figures_read_nothing_but_what_they_are_handed():
    for path in FIGURES.glob("*.py"):
        ours = {name for name in imported(path) if name.startswith("seqmaestro")}
        assert ours <= {"seqmaestro.report.figures"}, (path, ours)


def test_the_figures_are_drawn_without_a_display():
    """`style.apply` selects Agg, which needs no display, and every drawer calls it before it makes a
    figure. Where a display is available, matplotlib would otherwise choose an interactive backend and
    open a window for each figure. Core has no matplotlib, so the source is checked."""

    def calls(node: ast.AST, names: set[str]) -> list[int]:
        return [call.lineno for call in ast.walk(node)
                if isinstance(call, ast.Call) and ast.unparse(call.func) in names]

    style = ast.parse((FIGURES / "style.py").read_text())
    apply = next(node for node in style.body if isinstance(node, ast.FunctionDef) and node.name == "apply")
    assert "mpl.use('Agg')" in {ast.unparse(call) for call in ast.walk(apply) if isinstance(call, ast.Call)}
    drawers = 0
    for path in FIGURES.glob("*.py"):
        for function in ast.walk(ast.parse(path.read_text())):
            if isinstance(function, ast.FunctionDef) and (made := calls(function, {"plt.figure", "plt.subplots"})):
                drawers += 1
                applied = calls(function, {"style.apply"})
                assert applied and min(applied) < min(made), (
                    f"{path.name}: {function.name} makes a figure before it calls style.apply()")
    assert drawers, "no drawer was found, so the check above checked nothing"


# --------------------------------------------------------------------------------------------
# A run, end to end: cells, reduced, profiled, revealed, then its report inputs
# --------------------------------------------------------------------------------------------


def planted_config(dataset, linear: str, continuous: bool, others, judge: bool = False,
                   n_bootstrap: int = 10) -> dict:
    """The config of a planted run, with the sparse linear model and `others` on positions."""
    return {
        "name": "planted",
        "dataset": {"task": "planted", "splits": str(dataset.directory), "blinded": False},
        "grid": {"seeds": [1, 2], "labels": ["real", "permuted"], "train_fraction": 0.8,
                 "n_bootstrap": n_bootstrap,
                 "permutation_scope": "seed",
                 **({} if continuous else {"class_weight": "balanced"}),
                 "permutation_repeats": 0, "judge": judge, "judge_direction": False,
                 "generated_features": {"repair_rounds": 1, "max_undefined_share": 0.2,
                                        "max_example_length": 300, "max_seconds": 60, "guidance": None}},
        "language_model": LANGUAGE_MODEL,
        "analyses": ANALYSES,
        "arms": {"positional": {"models": [linear, *others]}},
        "reduction": {"n_features": 3},
        "reliability": {"pairs": 3, "permutations": 200, "cluster_rho": 0.8,
                        "max_modal_share": 0.99, "planted_count": 3, "planted_signal_sd": 1.5,
                        "check_arm": f"{linear}/positional", "seed": 0},
    }


def a_revealed_run(tmp_path, runners, continuous: bool = False, others=("xgboost",),
                   n_bootstrap: int = 10):
    """A planted run fitted, reduced and profiled, ready for the reveal: a binary target, or a
    continuous one, which weights no class and fits the least-squares model. `others` are the
    models fitted beside the sparse linear one."""
    dataset = planted_dataset(tmp_path, continuous=continuous)
    linear = "l1_linear" if continuous else "l1_logistic"
    source = tmp_path / "planted.yaml"
    source.write_text(yaml.safe_dump(planted_config(dataset, linear, continuous, others,
                                                    n_bootstrap=n_bootstrap)))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(runners.config.load(source).to_yaml())
    flags = ["--run-dir", str(run_dir)]
    assert runners.baseline.main(flags) == 0
    assert runners.arm.main(flags) == 0
    assert runners.reduction.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--workers", "2"]) == 0
    return dataset, run_dir


def test_the_judge_is_among_the_analyses_a_run_holds_only_where_its_config_runs_it(runners, tmp_path):
    dataset = planted_dataset(tmp_path)
    source = tmp_path / "planted.yaml"
    for judge in (False, True):
        source.write_text(yaml.safe_dump(planted_config(dataset, "l1_logistic", False, (), judge)))
        listed = runners.report.inventory(runners.config.load(source)).analyses
        assert ("judge" in listed) is judge
        assert set(listed) | {"judge"} == set(runners.report.STAGE2_ANALYSES)


def test_a_revealed_run_becomes_its_report_inputs(runners, tmp_path, capsys):
    dataset, run_dir = a_revealed_run(tmp_path, runners)
    assert runners.report.main(["--run-dir", str(run_dir)]) == 2
    assert "has not been revealed" in capsys.readouterr().err
    arms_before = spreadsheet.sheets_of(run_dir, top_n=5, min_witnesses=1)["Arms"]
    assert "val AUROC real" in arms_before.columns, "validation, until the run is revealed"
    assert not [column for column in arms_before.columns if column.startswith("test")]

    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    assert runners.report.main(["--run-dir", str(run_dir)]) == 0

    sheets = spreadsheet.sheets_of(run_dir, top_n=5, min_witnesses=1)
    assert list(sheets) == ["Read me", "Features", "Arms", "Agreed concepts"]
    assert {"test AUROC real", "test AUROC permuted"} <= set(sheets["Arms"].columns)
    listed = sheets["Features"]
    assert set(listed["model"]) == {naming.parse(arm).model_short
                                    for arm in ("l1_logistic/positional", "xgboost/positional")}
    assert set(listed["seed"]) == {1, 2} and listed["rank"].max() <= 5
    assert listed.groupby(["model", "seed"])["importance"].is_monotonic_decreasing.all()

    report = run_dir / "report"
    found = json.loads((report / "facts.json").read_text())
    assert found["classes"]["positive"] == dataset.classes[1].name == "motif"
    assert found["stage1"]["feature_guidance"] is None, "the run's config gives no guidance"
    assert found["stage1"]["blinding"] == {"blinded": False, "withheld": None}
    assert set(found["stage2"]["arms"]) == {"l1_logistic/positional", "xgboost/positional"}
    assert found["headline_set"] == {"best_l1_logistic": "l1_logistic/positional",
                                     "best_enumerated": "l1_logistic/positional",
                                     "black_box_reference": "xgboost/positional"}
    assert list(found["stage3b"]["arms"]) == ["l1_logistic/positional"]
    assert found["stage3b"]["not_profiled"] == {
        "xgboost/positional": "its model is read through its predictions, not its features"}
    # The config leaves the judge out, so it is not among the analyses the run should hold, and the
    # summary has no column for it.
    assert found["stage2"]["analyses_absent"] == sorted(set(runners.report.STAGE2_ANALYSES) - {"judge"})
    assert found["stage2"]["judged"] is False
    assert "how meaningful" not in (report / "tables" / "stage2_reproducibility.md").read_text()

    reduced = found["stage3a"]["arms"]["l1_logistic/positional"]
    metrics = json.loads((run_dir / "reduction/l1_logistic/positional/reduced/metrics.json").read_text())
    assert reduced["reduced_auroc"] == metrics[found["evaluated_on"]]["auroc"], \
        "a number in the facts is the one the reveal wrote"
    assert [f["rank"] for f in reduced["features"]] == [1, 2, 3]
    assert all((run_dir / path).exists() for path in reduced["files"])
    assert found["baseline"]["n_features"] == 4 and found["baseline"]["within_budget"] is False
    assert found["parameters"]["n_features"] == "three"
    assert found["parameters"]["positive_class"] == "motif"

    written = {path.stem for path in (report / "tables").glob("*.md")}
    assert {"key", "baseline", "stage2_all_arms", "stage2_headline", "stage3a_all_arms",
            "features_l1_logistic__positional", "features_xgboost__positional",
            "evidence_l1_logistic__positional", "stage3b_summary", "concepts",
            "language_models", "agents"} <= written
    assert found["language_models"] == [], "no arm of the run writes features, and no judge ran"
    assert "No part of this run called a language model" in (
        report / "tables" / "language_models.md").read_text()
    assert "evidence_xgboost__positional" not in written
    assert "motif" in (report / "tables" / "features_l1_logistic__positional.md").read_text()

    weights = json.loads((report / "internals" / "l1_logistic__positional.json").read_text())
    assert weights["kind"] == "l1_weights" and len(weights["features"]) == 3
    assert weights["arm_cell"] == "reduction/l1_logistic/positional/reduced"
    figures = json.loads((report / "figure_inputs.json").read_text())
    assert set(figures) == {"stage2_grid", "stage3a_grid", "stage3a_cost", "stage3a_l1",
                            "stage3b_evidence", "stage3b_evidence_all", "stage3b_thresholds",
                            "stage3b_concepts"}
    assert figures["stage2_grid"]["data"]["baselines"][0]["value"] == found["baseline"]["auroc"]



def test_the_report_names_the_agent_that_recorded_itself_for_it_after_its_inputs_were_built(
        runners, tmp_path, capsys):
    """An agent records itself when it starts the report, after the driver has built the report's
    inputs. Assembling the report therefore renders the table of agents again, and warns while no
    agent is recorded for the report."""
    import importlib.util

    from seqmaestro.pipeline import agents

    _, run_dir = a_revealed_run(tmp_path, runners, others=())
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    agents.record(run_dir, "Claude Code", "claude-opus-5-5", ["1", "2"])
    assert runners.report.main(["--run-dir", str(run_dir)]) == 0
    report = run_dir / "report"
    assert "Carried out Stages 1 and 2 |" in (report / "tables" / "agents.md").read_text()

    spec = importlib.util.spec_from_file_location(
        "assemble_report_under_test", Path(__file__).resolve().parents[1] / "experiments" / "analysis"
        / "assemble_report.py")
    assemble = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assemble)
    (report / "figures.json").write_text(json.dumps({"figures": {}}))
    (report / "REPORT.src.md").write_text("{{table:language_models}}\n\n{{table:agents}}\n")
    assert assemble.main(["--report-dir", str(report)]) == 0
    assert "no agent recorded itself for the report" in capsys.readouterr().err

    agents.record(run_dir, "Claude Code", "claude-opus-5-5", ["report"])
    assert assemble.main(["--report-dir", str(report)]) == 0
    assert "no agent recorded itself" not in capsys.readouterr().err
    written = (report / "REPORT.md").read_text()
    assert "| Claude Code | claude-opus-5-5 | Carried out Stages 1 and 2, and wrote this report |" in written
    assert "No part of this run called a language model" in written


def test_a_run_scored_without_bootstrap_draws_reports_no_interval(runners, tmp_path):
    """A quick run sets `grid.n_bootstrap: 0`, which scores each cell without intervals. Its report
    then states each reduced model's score with no interval beside it and draws no whiskers."""
    _, run_dir = a_revealed_run(tmp_path, runners, others=(), n_bootstrap=0)
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    assert runners.report.main(["--run-dir", str(run_dir)]) == 0

    report = run_dir / "report"
    reduced = json.loads((report / "facts.json").read_text())["stage3a"]["arms"][
        "l1_logistic/positional"]
    assert reduced["reduced_auroc"] is not None
    assert reduced["reduced_auroc_low"] is None and reduced["reduced_auroc_high"] is None
    grid = json.loads((report / "figure_inputs.json").read_text())["stage3a_grid"]["data"]
    assert grid["subtitle"] == "" and grid["rows"][0]["low"] is None


def test_the_spreadsheet_takes_test_numbers_from_a_finished_reveal_alone(runners, tmp_path):
    """While a reveal is under way, some cells hold a test block and others none, so the sheet
    stays on validation until the reveal's record says the reveal finished. A cell the reveal did
    not score (such as one whose arm the config does not name) then does not count towards any test
    mean."""
    _, run_dir = a_revealed_run(tmp_path, runners, others=())
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    record = run_dir / "reveal.json"
    finished = json.loads(record.read_text())

    record.write_text(json.dumps({**finished, "completed_utc": None}))
    arms = spreadsheet.sheets_of(run_dir, top_n=5, min_witnesses=1)["Arms"]
    assert "val AUROC real" in arms.columns

    record.write_text(json.dumps(finished))
    metrics = run_dir / "results" / "l1_logistic" / "positional" / "real" / "seed_1" / "metrics.json"
    scored = json.loads(metrics.read_text())
    metrics.write_text(json.dumps({key: value for key, value in scored.items() if key != "test"}))
    arms = spreadsheet.sheets_of(run_dir, top_n=5, min_witnesses=1)["Arms"]
    other_seed = json.loads((metrics.parent.parent / "seed_2" / "metrics.json").read_text())
    assert arms["test AUROC real"].tolist() == [round(other_seed["test"]["auroc"], 4)]


def test_a_blinded_runs_facts_name_what_the_models_were_not_told(runners, tmp_path, monkeypatch):
    """The withheld fact comes from the task description's `judge_only` field, which the driver has
    checked agrees with the config's `dataset.blinded` flag."""
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "planted.json").write_text(json.dumps(
        {"task": "Predicting a class.", "judge_only": "The planted motif is GGCGCC."}))
    monkeypatch.setattr(runners.report, "DESCRIPTIONS", datasets)
    config = SimpleNamespace(dataset=SimpleNamespace(task="planted", blinded=True))
    assert runners.report.blinding(config) == {"blinded": True,
                                               "withheld": "The planted motif is GGCGCC."}

def test_a_revealed_continuous_run_becomes_its_report_inputs(runners, tmp_path):
    """The facts, tables and figure inputs of a continuous run use its metric, its target's name and
    the wording of a continuous direction. Its sparse linear model is the least-squares one."""
    _, run_dir = a_revealed_run(tmp_path, runners, continuous=True)
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    assert runners.report.main(["--run-dir", str(run_dir)]) == 0

    report = run_dir / "report"
    found = json.loads((report / "facts.json").read_text())
    assert found["target_kind"] == "continuous" and "classes" not in found
    assert found["target"]["name"] == "planted response"
    assert found["metrics"]["headline"] == "r2" and found["metrics"]["association"] == "rho"
    assert found["directions"]["printed"]["increases_target"] == "raise the target"
    assert found["headline_set"]["best_l1_linear"] == "l1_linear/positional"

    reduced = found["stage3a"]["arms"]["l1_linear/positional"]
    metrics = json.loads((run_dir / "reduction/l1_linear/positional/reduced/metrics.json").read_text())
    assert reduced["reduced_r2"] == metrics[found["evaluated_on"]]["r2"]
    assert "reduced_balanced_accuracy" not in reduced, "no threshold, so no thresholded measure"
    assert found["parameters"]["target_name"] == "planted response"
    assert "positive_rate" not in found["parameters"]
    assert all("rho" in row and "delta_squared_error" in row
               for row in found["stage3b"]["arms"]["l1_linear/positional"]["reported"])

    tables = report / "tables"
    assert "held-out R²" in (tables / "stage2_all_arms.md").read_text()
    features = (tables / "features_l1_linear__positional.md").read_text()
    assert "rank correlation on its own" in features
    assert "the target" in features, "a continuous direction is printed as which way it moves"
    assert "fall in R² without it" in (tables / "evidence_l1_linear__positional.md").read_text()

    weights = json.loads((report / "internals" / "l1_linear__positional.json").read_text())
    assert weights["scale"] == internals.SCALE["continuous"]
    figures = json.loads((report / "figure_inputs.json").read_text())
    assert figures["stage2_grid"]["data"]["chance"] == 0.0
    assert figures["stage2_grid"]["data"]["metric_label"].startswith("R²")
    assert figures["stage3a_l1"]["data"]["note"] == "Bars to the right raise the target. Bars to " \
                                                    "the left lower it."


def test_a_long_product_in_an_equation_breaks_between_its_factors_and_never_inside_a_bracket():
    term = r"+ 2 \cdot \text{first\_long\_name} \cdot \left(a \cdot b\right) \cdot \frac{c \cdot d}{e}"
    lines = tables._wrapped(term, 21)
    assert len(lines) == 3 and lines[0] == r"+ 2 \cdot \text{first\_long\_name}"
    assert lines[1:] == [r"\quad \cdot \left(a \cdot b\right)", r"\quad \cdot \frac{c \cdot d}{e}"]
    assert tables._wrapped(term, 500) == [term], "a product that fits stays on one line"


def test_a_left_side_too_long_to_share_a_line_stands_on_its_own():
    """A target whose name leaves the terms less than half the line would squeeze them into a column
    that a feature's name overruns. Such a left side takes a row of its own, with the indented terms
    across the whole width below it, whereas a short left side keeps the terms beside it."""
    terms = [r"\text{first\_feature\_with\_a\_long\_name}", r"- \text{second\_feature}"]
    assert tables._aligned_lines(r"\text{predicted y}", terms) == [
        r"\text{predicted y} ={} & \text{first\_feature\_with\_a\_long\_name}",
        r"& - \text{second\_feature}"]
    left = r"\text{predicted " + "a target with a long name " * 3 + "}"
    lines = tables._aligned_lines(left, terms)
    assert lines == [f"& {left} ={{}}", r"& \quad \text{first\_feature\_with\_a\_long\_name}",
                     r"& \quad - \text{second\_feature}"]
    assert all(len(tables._visible(line)) <= tables.EQUATION_WIDTH for line in lines)


def test_a_factor_too_long_for_a_line_breaks_before_a_sign_inside_its_brackets():
    """The break comes at the shallowest bracket that holds a sign, which is closed and opened again
    invisibly, because LaTeX pairs each `\\left` with a `\\right` on the same line."""
    term = r"+ \left(\text{first\_name} + \log\left(\text{second\_name} + \text{first\_name}\right)\right)"
    lines = tables._wrapped(term, 45)
    assert lines == [r"+ \left(\text{first\_name} \right.",
                     r"\quad \left. {} + \log\left(\text{second\_name} + \text{first\_name}\right)\right)"]
    assert all(line.count("\\left") == line.count("\\right") for line in lines)
    assert tables._wrapped(term, 500) == [term], "a factor that fits stays whole"


def test_a_fraction_too_wide_for_a_line_is_written_as_a_division_and_broken():
    """A fraction is typeset as one box as wide as the longer of its two parts, so one wider than a
    line would run off the page. Written as a division on the line, it breaks wherever a product or
    a bracket can, and a part still too wide after one break is broken again."""
    term = (r"- 2 \cdot \text{first\_name} \cdot \left(\frac{\text{first\_name}}"
            r"{\left(\exp\left(\text{second\_name}\right) \cdot \exp\left(\text{first\_name}\right)"
            r" + \exp\left(\text{second\_name}\right)\right) \cdot \text{second\_name}}"
            r" - \text{first\_name}\right)")
    width = 30
    lines = tables._wrapped(term, width)

    assert not any("\\frac" in line for line in lines)
    assert all(line.count("\\left") == line.count("\\right") for line in lines)
    assert all(len(tables._visible(line)) <= width
               or not [place for place in tables._breaks(line) if place[0] > 0]
               for line in lines), "a line stays wide only where it has nowhere left to break"

    def bare(latex):
        return re.sub(r"\\quad|\\left\.|\\right\.|\{\}|\s", "", latex)
    on_the_line = tables._inline_wide_fractions(term, width)
    assert " / \\left(" in on_the_line, "a denominator that is a product is bracketed"
    assert bare("".join(lines)) == bare(on_the_line), "breaking a line changes nothing it says"
    assert tables._wrapped(term, 500) == [term], "a fraction that fits stays a fraction"


def test_a_wide_fraction_over_one_factor_takes_no_bracket():
    term = r"+ \frac{\text{first\_name} \cdot \text{second\_name}}{\text{second\_name}}"
    assert tables._inline_wide_fractions(term, 15) == (
        r"+ \text{first\_name} \cdot \text{second\_name} / \text{second\_name}")
    assert tables._inline_wide_fractions(term, 30) == term


def test_a_symbolic_model_is_drawn_as_its_equation_its_shapley_values_and_its_front(runners,
                                                                                   tmp_path):
    _, run_dir = a_revealed_run(tmp_path, runners, continuous=True, others=("pysr",))
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    assert runners.reliability.main(["--run-dir", str(run_dir), "--after-reveal"]) == 0
    assert runners.report.main(["--run-dir", str(run_dir)]) == 0

    report = run_dir / "report"
    found = json.loads((report / "facts.json").read_text())
    assert found["headline_set"]["best_pysr"] == "pysr/positional"
    drawn = json.loads((report / "internals" / "pysr__positional.json").read_text())
    assert drawn["kind"] == "pysr_equation" and drawn["arm_cell"] == "reduction/pysr/positional/reduced"
    assert [feature["importance"] for feature in drawn["features"]] == sorted(
        (feature["importance"] for feature in drawn["features"]), reverse=True)
    assert sum(entry["chosen"] for entry in drawn["front"]) == 1

    table = (report / "tables" / "stage3a_pysr_equation.md").read_text()
    assert "\\begin{aligned}" in table and "predicted planted response" in table
    assert not re.search(r"\bx\d+\b", table), "the search's placeholder names never reach a reader"
    assert all(feature["name"] in table for feature in drawn["features"])
    figures = json.loads((report / "figure_inputs.json").read_text())
    assert figures["stage3a_pysr"]["draw"] == "equation_shapley"
    assert figures["stage3a_pysr_front"]["draw"] == "equation_front"
    assert "pysr/positional" in found["stage3b"]["arms"]
    # An equation's reduction can keep fewer columns than the budget, and its figure says how
    # many it kept.
    own = found["stage3a"]["arms"]["pysr/positional"]["n_final"]
    assert (f"equation reduced to {markdown.word(own)} features"
            in figures["stage3a_pysr"]["caption"])


def test_a_run_whose_reveal_did_not_finish_is_refused(runners, tmp_path, capsys):
    _, run_dir = a_revealed_run(tmp_path, runners)
    assert runners.reveal.main(["--run-dir", str(run_dir)]) == 0
    record = json.loads((run_dir / "reveal.json").read_text())
    (run_dir / "reveal.json").write_text(json.dumps({**record, "completed_utc": None}))
    assert runners.report.main(["--run-dir", str(run_dir)]) == 2
    assert "did not finish" in capsys.readouterr().err
