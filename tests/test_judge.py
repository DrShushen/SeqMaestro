"""The judge's protocol: selection, blinding, ordering, repeats, parsing, aggregation.

None of these tests needs a credential or a network call, because the scoring backend sits behind
an interface and a scripted engine stands in for it. The LLM's answers cannot be tested, but the
code around them can, and that code is where a silent bias would live.
"""

from __future__ import annotations

import pytest

from seqmaestro.features.base import FeatureRecord
from seqmaestro.judge import (
    MAX_SCORE,
    MIN_SCORE,
    PanelSource,
    ScriptedEngine,
    TaskContext,
    build_panel,
    direction_shown,
    panel_key,
    parse_reply,
    render,
    score_panel,
    summarise,
)

ARMS = ("deft", "l1_logistic")

CONTEXT = TaskContext(
    task="Predicting where a DNA-binding factor binds in human cells from DNA sequence.",
    sequence_layout="101 nt, hg38, 5' to 3' in the direction of transcription; the binding site "
                    "is at index 50, so 0-49 is upstream and 51-100 downstream.",
)


def records(prefix, n, importances=None):
    return [
        FeatureRecord(
            name=f"{prefix}_feature_{i}",
            description=f"description of {prefix} feature {i}",
            kind="test",
            rationale=f"rationale {i}",
            source=f"def {prefix}_feature_{i}(X): return X.a",
            importance=(importances[i] if importances else float(n - i)),
        )
        for i in range(n)
    ]


def sources(n=5):
    """Three panel sources whose feature names carry no trace of their arm, so the blinding tests
    check the code and not the fixture. Real feature names can reveal their arm: see
    `test_blinding_hides_provenance_but_cannot_hide_a_feature_s_own_name`."""
    return [
        PanelSource(arm="deft", labels="real", records=records("alpha", n), target_kind="binary", seed=42),
        PanelSource(arm="l1_logistic", labels="real", records=records("beta", n), target_kind="binary",
                    seed=42),
        PanelSource(arm="deft", labels="permuted", records=records("gamma", n), target_kind="binary",
                    seed=42),
    ]


# --- selection ----------------------------------------------------------------------------


def test_top_n_bounds_the_panel_per_arm():
    panel = build_panel(sources(n=20), top_n=5)
    assert len(panel) == 15  # 3 sources x 5
    for arm, labels in (("deft", "real"), ("l1_logistic", "real"), ("deft", "permuted")):
        assert sum(1 for e in panel if e.arm == arm and e.labels == labels) == 5


def test_selection_takes_the_most_important():
    """The panel holds each source's most important features, because those are what an arm
    claims to have found. The tail of a large fixed vocabulary is not."""
    panel = build_panel(
        [PanelSource("a", "real", records("x", 6, importances=[0.1, 9.0, 0.2, 8.0, 0.3, 7.0]), "binary")],
        top_n=3,
    )
    assert {e.record.name for e in panel} == {"x_feature_1", "x_feature_3", "x_feature_5"}


def test_top_n_none_judges_everything():
    assert len(build_panel(sources(n=4), top_n=None)) == 12


# --- blinding -----------------------------------------------------------------------------


def test_no_prompt_mentions_the_arm_or_the_label_condition():
    """No prompt names the arm or the label condition. The renderer reads only the feature
    record's fields, so the blinding holds by construction."""
    for entry in build_panel(sources(), top_n=5):
        prompt = render(entry, CONTEXT).user + render(entry, CONTEXT).system
        lowered = prompt.lower()
        for leak in (*ARMS, "permuted", "shuffled", "control", "real labels"):
            assert leak not in lowered, f"{leak!r} leaked into a prompt"


def test_identifiers_do_not_encode_the_arm():
    """Item ids are assigned after the panel is shuffled. Assigned before it, each arm's features
    would hold one contiguous block of ids, which would reveal the arm."""
    panel = build_panel(sources(n=8), top_n=8, seed=3)
    positions = [i for i, e in enumerate(panel) if e.arm == "deft" and e.labels == "real"]
    assert max(positions) - min(positions) > len(positions), "one arm sits in a contiguous run"


def test_the_key_maps_back_to_provenance():
    panel = build_panel(sources(), top_n=2)
    key = panel_key(panel)
    assert len(key) == 6
    entry = panel[0]
    assert key[entry.item_id]["arm"] == entry.arm
    assert key[entry.item_id]["labels"] == entry.labels


# --- ordering -----------------------------------------------------------------------------


def test_ordering_is_seeded_and_reproducible():
    a = [e.record.name for e in build_panel(sources(), top_n=5, seed=7)]
    b = [e.record.name for e in build_panel(sources(), top_n=5, seed=7)]
    c = [e.record.name for e in build_panel(sources(), top_n=5, seed=8)]
    assert a == b
    assert a != c


# --- what the judge is shown ---------------------------------------------------------------


def test_code_is_hidden_by_default_and_shown_on_request():
    entry = build_panel(sources(), top_n=1)[0]
    assert "def " not in render(entry, CONTEXT).user
    assert "def " in render(entry, CONTEXT, include_code=True).user


def test_the_description_and_rationale_are_always_shown():
    entry = build_panel([PanelSource("a", "real", records("x", 1), "binary")], top_n=1)[0]
    prompt = render(entry, CONTEXT).user
    assert "description of x feature 0" in prompt
    assert "rationale 0" in prompt
    assert "if it were true" in prompt.lower()


def test_the_task_context_reaches_the_prompt():
    """Without the task context, "GC fraction of positions 41-49" names an interval with no
    organism or assay, and the judge can only score how the feature is written."""
    prompt = render(build_panel(sources(), top_n=1)[0], CONTEXT).user
    assert "a DNA-binding factor binds" in prompt
    assert "index 50" in prompt


def test_context_notes_are_optional_and_included_when_set():
    entry = build_panel(sources(), top_n=1)[0]
    bare = TaskContext(task="Predicting something.")
    assert "Sequence layout" not in render(entry, bare).user
    assert "Notes" not in render(entry, bare).user

    annotated = TaskContext(task="Predicting something.", notes="Sequences are RNA, not DNA.")
    assert "Sequences are RNA, not DNA." in render(entry, annotated).user


def test_the_context_is_recorded_with_the_scores():
    """A score is only comparable to another asked with the same wording, so the summary records
    the task context."""
    items = score_panel(build_panel(sources(), top_n=1), ScriptedEngine('{"score": 3}'), CONTEXT, repeats=1)
    assert summarise(items, CONTEXT)["task_context"]["task"] == CONTEXT.task


def test_blinding_hides_provenance_but_cannot_hide_a_feature_s_own_name():
    """A known limit of the blinding, asserted so it is not mistaken for a guarantee.

    Provenance fields never enter a prompt. A feature's name is part of what is judged, so it must
    be shown, and real names follow each arm's naming style (`kmer_GGTCA` against
    `gc_fraction_upstream_site_proximal_9nt`). A judge could infer the arm from the name, and only
    paraphrasing every feature would prevent that.
    """
    telling = [FeatureRecord(name="kmer_GGTCA", description="count of GGTCA", kind="kmer")]
    entry = build_panel([PanelSource("l1_logistic", "real", telling, "binary")], top_n=1)[0]
    prompt = render(entry, CONTEXT).user

    assert "l1_logistic" not in prompt        # provenance is hidden
    assert "kmer_GGTCA" in prompt             # the feature's name cannot be


def test_the_scale_is_anchored_in_the_prompt():
    """Every point of the scale has its meaning written in the prompt. An unanchored 1-5 scale
    drifts between calls, so its scores cannot be compared across arms."""
    prompt = render(build_panel(sources(), top_n=1)[0], CONTEXT).user
    for value in range(MIN_SCORE, MAX_SCORE + 1):
        assert f"{value} =" in prompt


# --- parsing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"score": 4, "reason": "interesting"}', 4),
        ('```json\n{"score": 2, "reason": "meh"}\n```', 2),
        ('Sure! Here is my answer:\n{"score": 5, "reason": "big"}', 5),
        (" 3\n", 3),
    ],
)
def test_scores_are_recovered_from_realistic_replies(reply, expected):
    judgement = parse_reply("F0001", reply)
    assert judgement.score == expected and judgement.error is None


@pytest.mark.parametrize("reply", ["", "   ", "I cannot answer that.", '{"score": 9}', "9",
                                   "I would say 3 out of 5.", "On a 1-5 scale, 4."])
def test_an_unusable_reply_is_a_refusal_not_a_score(reply):
    """An unusable reply gets no score and an error. Imputing a middling score would pull every
    arm towards the mean and hide how often parsing failed, and a number picked out of a sentence
    can be the scale's and not the rating."""
    judgement = parse_reply("F0001", reply)
    assert judgement.score is None
    assert judgement.error


# --- scoring and aggregation ---------------------------------------------------------------


def test_every_feature_is_judged_the_requested_number_of_times():
    panel = build_panel(sources(), top_n=3)
    engine = ScriptedEngine('{"score": 3, "reason": "fine"}')
    items = score_panel(panel, engine, CONTEXT, repeats=3)
    assert len(engine.seen) == len(panel) * 3
    assert all(len(item.scores) == 3 for item in items)


def test_repeats_is_configurable():
    panel = build_panel(sources(), top_n=2)
    engine = ScriptedEngine('{"score": 3}')
    score_panel(panel, engine, CONTEXT, repeats=1)
    assert len(engine.seen) == len(panel)


def test_repeats_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        score_panel(build_panel(sources(), top_n=1), ScriptedEngine('{"score": 3}'), CONTEXT, repeats=0)


def test_spread_across_repeats_is_measured():
    """The spread of scores across repeats is recorded, because it describes the judge itself,
    and the judge is the instrument this axis characterises."""
    scores = iter(["3", "5", "4"] * 100)
    engine = ScriptedEngine(lambda r: f'{{"score": {next(scores)}}}')
    items = score_panel(build_panel(sources(), top_n=1), engine, CONTEXT, repeats=3)
    assert items[0].mean == pytest.approx(4.0)
    assert items[0].spread == pytest.approx(1.0)


def test_refusals_are_counted_not_imputed():
    engine = ScriptedEngine(lambda r: "I cannot answer.")
    items = score_panel(build_panel(sources(), top_n=2), engine, CONTEXT, repeats=3)
    assert all(item.mean is None for item in items)
    assert all(item.n_failed == 3 for item in items)
    assert summarise(items)["n_unanswered"] == len(items) * 3


def test_the_summary_reports_the_gap_above_the_noise_floor():
    """Each arm's mean is reported with its gap above the permuted-label control, because the
    mean cannot be interpreted without it."""
    def reply(rendered):
        # The permuted source's features are the "gamma" ones. They score 2 and the rest score 4.
        return '{"score": 2}' if "gamma" in rendered.user else '{"score": 4}'

    items = score_panel(build_panel(sources(), top_n=5), ScriptedEngine(reply), CONTEXT, repeats=3)
    report = summarise(items)

    assert report["control"]["mean"] == pytest.approx(2.0)
    assert report["arms"]["deft/real"]["mean"] == pytest.approx(4.0)
    assert report["arms"]["deft/real"]["control_gap"] == pytest.approx(2.0)
    assert report["arms"]["deft/permuted"]["control_gap"] == pytest.approx(0.0)


def test_a_judge_that_cannot_discriminate_shows_a_zero_gap():
    """A judge that gives every feature the same score shows a zero gap for every arm. This is
    the failure the axis exists to detect: a judge that scores how a feature reads and ignores
    whether real labels produced it."""
    items = score_panel(build_panel(sources(), top_n=5), ScriptedEngine('{"score": 4}'), CONTEXT, repeats=2)
    report = summarise(items)
    assert all(arm["control_gap"] == pytest.approx(0.0) for arm in report["arms"].values())


def test_the_summary_survives_having_no_controls():
    items = score_panel(
        build_panel([PanelSource("a", "real", records("x", 2), "binary")], top_n=2),
        ScriptedEngine('{"score": 3}'), CONTEXT, repeats=1,
    )
    report = summarise(items)
    assert report["control"]["mean"] is None
    assert "control_gap" not in report["arms"]["a/real"]


def test_summary_is_json_serialisable():
    import json

    items = score_panel(build_panel(sources(), top_n=2), ScriptedEngine('{"score": 3}'), CONTEXT, repeats=2)
    assert json.loads(json.dumps(summarise(items), default=str))["n_items"] == len(items)


def test_records_without_importance_still_form_a_panel():
    """A model that reports no importances still has its features put on the panel."""
    plain = [FeatureRecord(name=f"f{i}", description="d", kind="test") for i in range(4)]
    panel = build_panel([PanelSource("a", "real", plain, "binary")], top_n=3)
    assert len(panel) == 3


# ------------------------------------------------- a judge that answered nothing has failed


def _verdict(items, report, min_answered=0.9, expect_control=True):
    from seqmaestro.pipeline.analyses.judge import why_unusable

    return why_unusable(items, report, min_answered, expect_control)


def _report(control_features=12, calls=100, failures=0):
    return {"control": {"n_features": control_features},
            "usage": {"calls": calls, "failures": failures}}


def scored_items(n_scored, n_unscored, repeats=3):
    panel = build_panel(sources(), top_n=None)
    engine = ScriptedEngine('{"score": 3}')
    items = score_panel(panel[: n_scored + n_unscored], engine, CONTEXT, repeats=repeats)
    for item in items[n_scored:]:
        item.scores = []          # `mean` is derived from this, and becomes None with it
    return items


def test_a_judge_that_answered_nothing_is_reported_as_failed():
    """Every call failed, as in an outage. The run is reported as failed, so the stage cannot
    exit 0 as if it had passed."""
    items = scored_items(n_scored=0, n_unscored=8)

    reason = _verdict(items, _report(control_features=0, calls=0, failures=24))

    assert reason is not None
    assert "0 of 8" in reason


def test_a_mostly_unanswered_panel_is_reported_as_failed():
    items = scored_items(n_scored=4, n_unscored=6)

    assert _verdict(items, _report()) is not None


def test_a_handful_of_dropped_calls_is_tolerated():
    """A run passes when at least `min_answered` of the panel was scored, so one unanswered
    feature in twenty is tolerated."""
    items = scored_items(n_scored=19, n_unscored=1)

    assert _verdict(items, _report()) is None


def test_an_empty_control_pool_fails_even_when_everything_was_scored():
    """An empty control pool makes the run unusable even when every feature was scored, because
    the control is the noise floor that every arm's mean is read against."""
    items = scored_items(n_scored=10, n_unscored=0)

    reason = _verdict(items, _report(control_features=0))

    assert reason is not None
    assert "control pool" in reason


def test_an_empty_panel_fails():
    assert _verdict([], _report()) is not None


def test_an_absent_control_is_fine_when_none_was_asked_for():
    """Judging permuted features is off by default, so an empty control pool passes when no
    control was asked for.

    The proposer builds its features from the task description and not from the labels, so it
    proposes nearly the same set under permuted labels. The "control" would then be the judge
    scoring one panel twice.
    """
    items = scored_items(n_scored=10, n_unscored=0)

    assert _verdict(items, _report(control_features=0), expect_control=False) is None


def test_an_unanswered_panel_still_fails_without_a_control():
    """Relaxing the control check must not relax the check that the judge answered at all."""
    items = scored_items(n_scored=0, n_unscored=8)

    reason = _verdict(items, _report(control_features=0, calls=0, failures=24),
                      expect_control=False)

    assert reason is not None
    assert "0 of 8" in reason


# --------------------------------------------------------------- showing the judge a direction


def a_panel_entry(marginal_effect=None, target_kind="binary"):
    source = PanelSource(arm="l1_logistic", labels="real", target_kind=target_kind,
                         records=[FeatureRecord(name="f", description="d", kind="test",
                                                marginal_effect=marginal_effect)])
    return build_panel([source], top_n=None)[0]


def test_a_continuous_target_is_shown_which_way_the_feature_moves_it():
    entry = a_panel_entry(marginal_effect=-0.3, target_kind="continuous")
    assert "higher values of this feature go with a lower target" in \
           render(entry, CONTEXT, include_direction=True).user


def test_the_direction_is_shown_only_when_it_is_asked_for():
    """The direction line is off by default. Turning it on changes what the question means, so
    scores from a run with it on cannot be compared with scores from a run without it."""
    entry = a_panel_entry(marginal_effect=0.4)

    assert "higher values" not in render(entry, CONTEXT).user
    assert "higher values in the positive class" in \
           render(entry, CONTEXT, include_direction=True).user


def test_the_judge_is_shown_the_sign_and_never_the_size():
    """An effect size would let the judge separate the real features from the permuted-label
    controls, which every arm's mean is read against. So a strong feature and a barely-there one
    must produce the same sentence."""
    strong = render(a_panel_entry(0.97), CONTEXT, include_direction=True).user
    weak = render(a_panel_entry(0.01), CONTEXT, include_direction=True).user

    assert strong == weak
    for shown in (strong, weak):
        assert "0.97" not in shown and "0.01" not in shown


def test_the_two_directions_read_the_same_way_round_as_the_effect():
    assert "positive class" in render(a_panel_entry(0.4), CONTEXT, include_direction=True).user
    assert "negative class" in render(a_panel_entry(-0.4), CONTEXT, include_direction=True).user


def test_a_feature_with_no_recorded_direction_gets_no_line():
    for value in (None, 0.0):
        assert "higher values" not in \
               render(a_panel_entry(value), CONTEXT, include_direction=True).user


def test_the_flag_reaches_the_prompt_through_the_scoring_loop():
    """`score_panel` passes `include_direction` on to `render`. A flag it accepted and dropped
    would be recorded as on in the output while missing from every prompt."""
    panel = [a_panel_entry(0.4)]
    engine = ScriptedEngine('{"score": 3, "reason": "r"}')
    score_panel(panel, engine, CONTEXT, repeats=1, include_direction=True)

    assert "higher values in the positive class" in engine.seen[0].user


def test_what_the_judge_would_be_shown_can_be_asked_without_rendering():
    """`direction_shown` lets a caller count how much of a panel carries a direction. Without
    that count, `grid.judge_direction` over records with no marginal effect would be on and would
    silently show nothing."""
    assert direction_shown(FeatureRecord("f", "d", "test", marginal_effect=0.4), "binary") is not None
    assert direction_shown(FeatureRecord("f", "d", "test", marginal_effect=None), "binary") is None


def test_the_count_and_the_prompt_cannot_disagree():
    """Whatever `direction_shown` returns is the line `render` appends, so a count of shown
    directions always matches the prompts."""
    for value in (0.4, -0.4, 0.0, None):
        entry = a_panel_entry(marginal_effect=value)
        shown = direction_shown(entry.record, "binary")
        rendered = render(entry, CONTEXT, include_direction=True).user
        assert (shown in rendered) if shown else ("higher values" not in rendered)


def test_the_judge_reads_each_cell_once_and_not_the_bundle_inside_it(tmp_path):
    """A vendored arm's cell keeps the bundle its method handed back, which has a `features.json`
    of its own. Only the cell's is a source."""
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import save_features
    from seqmaestro.pipeline.analyses.judge import collect_sources

    results = tmp_path / "results"
    cell = results / "deft_or_cart" / "positional+llm_generated_joint" / "real" / "seed_3"
    records = [FeatureRecord(name="gc_run", description="a run of G or C", kind="llm_generated")]
    save_features(cell / "features.json", records, target_kind="binary")
    save_features(cell / "bundle" / "features.json", records)

    sources = collect_sources(results)
    assert [(source.arm, source.labels, source.seed, source.target_kind) for source in sources] == [
        ("deft_or_cart/positional+llm_generated_joint", "real", 3, "binary")]
