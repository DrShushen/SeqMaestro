"""Recurrence in both senses (the same feature and the same idea), and the nulls each is read against."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

from seqmaestro.llm import ScriptedClient, TaskContext
from seqmaestro.pipeline.analyses import recurrence as mod


def features(real_per_seed=2, permuted_per_seed=2, seeds=("seed_42", "seed_43")):
    out = []
    for seed in seeds:
        for i in range(real_per_seed):
            out.append({"name": f"r_{seed}_{i}", "description": "", "labels": "real",
                        "seed": seed, "provenance": "generated"})
        for i in range(permuted_per_seed):
            out.append({"name": f"p_{seed}_{i}", "description": "", "labels": "permuted",
                        "seed": seed, "provenance": "generated"})
    return out


def test_a_grouping_is_parsed_from_plain_json():
    got = mod.parse_groups('{"groups": [{"name": "a", "items": [1, 2]}]}', 3)
    assert got == {"a": [1, 2]}


def test_a_code_fence_is_tolerated():
    got = mod.parse_groups('```json\n{"groups": [{"name": "a", "items": [1]}]}\n```', 2)
    assert got == {"a": [1]}


def test_item_numbers_outside_the_listing_are_dropped():
    """A hallucinated index must not silently point at someone else's feature."""
    assert mod.parse_groups('{"groups": [{"name": "a", "items": [1, 99, 0]}]}', 3) == {"a": [1]}


def test_an_empty_group_is_not_kept():
    assert mod.parse_groups('{"groups": [{"name": "a", "items": []}]}', 3) == {}


def test_a_number_a_group_names_twice_is_one_member():
    """A group's counts are taken over its members, so a repeated number would count one feature
    twice."""
    assert mod.parse_groups('{"groups": [{"name": "a", "items": [2, 1, 2]}]}', 3) == {"a": [2, 1]}


def test_a_feature_is_counted_as_placed_once_however_many_groups_name_it():
    """`n_grouped` is set beside `n_features` to say how much of the listing a grouping covers, so
    a feature in two groups must not make up for one in none."""
    placed, note = mod.placement({"a": [1, 2], "b": [2, 3]}, n_items=5)
    assert placed == 3
    assert note == ("note: 2 feature(s) were not placed in any group, and 1 feature(s) were placed "
                    "in more than one")
    assert mod.placement({"a": [1, 2], "b": [3]}, n_items=3) == (3, None)


# ------------------------------------------------------------------- which arms are grouped


def test_every_arm_on_a_written_source_is_grouped_and_no_other(tmp_path):
    """An arm renames an idea from seed to seed wherever a language model writes its features for
    each cell, which the one-round source does as the iterative and the joint source do. An arm on
    a pool of enumerated sources alone selects from one vocabulary and is left to stability."""
    arms = ("l1_logistic/llm_generated", "ebm/kmer+llm_generated_iterative",
            "deft_or_cart/positional+llm_generated_joint", "igsr/llm_generated_joint+covariates",
            "l1_logistic/positional", "deft_or_cart/positional", "xgboost/positional+kmer")
    for arm in arms:
        (tmp_path / arm / "real" / "seed_1").mkdir(parents=True)
    (tmp_path / "stability.csv").write_text("a table beside the arms\n")

    assert mod.written_arms(tmp_path) == [
        "deft_or_cart/positional+llm_generated_joint", "ebm/kmer+llm_generated_iterative",
        "igsr/llm_generated_joint+covariates", "l1_logistic/llm_generated"]
    assert mod.written_arms(tmp_path / "absent") == []


# --------------------------------------------- the room and the time a reply is given


def test_a_reply_is_given_room_and_time_for_its_listing():
    """A reply names every listed item, so one limit for every call would cut short the reply of
    the pass across arms, whose listing is several arms long."""
    one_arm, every_arm = mod.reply_tokens(200, 6), mod.reply_tokens(2000, 6)
    assert every_arm > one_arm > mod.reply_tokens(0, 6) == mod.TOKENS_TO_SPARE
    assert every_arm >= mod.TOKENS_PER_ITEM * 2000 + mod.TOKENS_PER_GROUP * 334
    assert mod.reply_tokens(200, 2) > one_arm, "smaller groups are more groups to name"
    assert mod.reply_seconds(every_arm) > mod.reply_seconds(one_arm) > mod.SECONDS_TO_SPARE


def test_a_grouping_call_states_the_limits_of_its_listing():
    items = features(real_per_seed=3, permuted_per_seed=3)
    client = ScriptedClient(json.dumps({"groups": [{"name": "all", "items": list(range(1, 13))}]}))
    settings = SimpleNamespace(seed=0, items_per_group=6, dry_run=False)

    mod.group_features(items, settings, TaskContext(task="A planted class."), client, "an arm")

    tokens = mod.reply_tokens(len(items), 6)
    assert client.limits == [(tokens, mod.reply_seconds(tokens))]


def test_seed_and_feature_counts_are_reported_per_condition():
    items = features()
    frame = mod.summarise(items, {"c": list(range(1, len(items) + 1))})
    row = frame.iloc[0]
    assert row.real_seeds == 2 and row.permuted_seeds == 2
    assert row.real_features == 4 and row.permuted_features == 4


def test_a_concept_of_only_real_features_is_flagged_as_enriched():
    items = features()
    real = [i + 1 for i, f in enumerate(items) if f["labels"] == "real"]
    frame = mod.summarise(items, {"all_real": real, "rest": [i + 1 for i in range(len(items))
                                                             if i + 1 not in real]})
    assert frame.set_index("concept").loc["all_real", "p_enriched"] < 0.05


def test_enrichment_is_not_significant_when_the_pool_is_mostly_real():
    """When real features outnumber permuted ones ten to one, any group looks real-heavy, so
    enrichment is judged against the pool's own mix."""
    items = features(real_per_seed=10, permuted_per_seed=1)
    real = [i + 1 for i, f in enumerate(items) if f["labels"] == "real"][:6]
    frame = mod.summarise(items, {"c": real})
    assert frame.set_index("concept").loc["c", "p_enriched"] > 0.2


def test_the_seed_null_is_unsurprised_by_a_large_group():
    """A concept holding most of the features reaches every seed through its size alone, so the
    seed null must not find that surprising."""
    items = features(real_per_seed=10, permuted_per_seed=1)
    groups = {"big": [i + 1 for i in range(len(items) - 2)]}
    p = mod.recurrence_null(items, groups, draws=300, rng=random.Random(0))
    assert p["big"] > 0.5


def test_the_seed_null_notices_a_group_that_is_one_seed_only():
    items = features(real_per_seed=4, permuted_per_seed=4, seeds=("a", "b", "c", "d"))
    one_seed = [i + 1 for i, f in enumerate(items) if f["seed"] == "a" and f["labels"] == "real"]
    p = mod.recurrence_null(items, groups={"narrow": one_seed}, draws=300, rng=random.Random(0))
    assert p["narrow"] > 0.5          # reaching *one* seed is easy, so this is unremarkable


# --------------------------------------------------- the two views of recurrence, side by side


def test_exact_recurrence_counts_a_feature_across_seeds():
    items = [{"name": "f", "arm": "a", "description": "", "importance": 0.5,
              "labels": "real", "seed": s} for s in ("seed_1", "seed_2", "seed_3")]
    frame = mod.exact_recurrence(items)
    assert frame.iloc[0].seeds == 3 and frame.iloc[0].of_seeds == 3


def test_exact_recurrence_ignores_permuted_features():
    """The by-name view measures reproducibility on real labels. The permuted-label null belongs
    to the concept view."""
    items = [{"name": "f", "arm": "a", "description": "", "importance": 0.5,
              "labels": lab, "seed": "seed_1"} for lab in ("real", "permuted")]
    frame = mod.exact_recurrence(items)
    assert len(frame) == 1 and frame.iloc[0].seeds == 1


def test_exact_recurrence_keeps_the_largest_importance():
    items = [{"name": "f", "arm": "a", "description": "", "importance": i,
              "labels": "real", "seed": f"seed_{i}"} for i in (0.1, 0.9)]
    assert mod.exact_recurrence(items).iloc[0].importance == 0.9


def test_renaming_one_idea_scores_zero_by_name():
    """A generative arm renames its ideas between runs, which is why both views are reported."""
    items = [{"name": f"gc_fraction_{w}nt", "arm": "a", "description": "GC fraction",
              "importance": 0.5, "labels": "real", "seed": f"seed_{i}"}
             for i, w in enumerate((11, 21, 31))]
    frame = mod.exact_recurrence(items)
    assert (frame.seeds == 1).all()
    # ... while grouping them as one concept recovers the three seeds.
    grouped = mod.summarise(items, {"local GC": [1, 2, 3]})
    assert grouped.iloc[0].real_seeds == 3


def test_an_empty_feature_set_gives_an_empty_frame():
    assert mod.exact_recurrence([]).empty


def test_collect_takes_the_top_n_by_importance(tmp_path):
    """`collect` takes the same top-n cut that the stability and co-discovery analyses report
    on."""
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import save_features

    cell = tmp_path / "arm" / "real" / "seed_42"
    cell.mkdir(parents=True)
    save_features(cell / "features.json",
                  [FeatureRecord(name=f"f{i}", description="", kind="llm_generated",
                                 importance=i / 10) for i in range(10)])
    got = mod.collect(tmp_path, "arm", {"llm_generated"}, top_n=3)
    assert [f["name"] for f in got] == ["f9", "f8", "f7"]


def test_collect_ranks_a_signed_importance_by_its_size(tmp_path):
    """A sparse linear arm reports signed weights, and a strong negative weight is among the
    features the model leaned on most. The cut keeps it, and records its size."""
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import save_features

    cell = tmp_path / "arm" / "real" / "seed_42"
    cell.mkdir(parents=True)
    weights = {"strong_negative": -0.9, "weak_positive": 0.1, "strong_positive": 0.8}
    save_features(cell / "features.json",
                  [FeatureRecord(name=name, description="", kind="llm_generated", importance=weight)
                   for name, weight in weights.items()])
    got = mod.collect(tmp_path, "arm", {"llm_generated"}, top_n=2)
    assert [(f["name"], f["importance"]) for f in got] == [("strong_negative", 0.9),
                                                           ("strong_positive", 0.8)]


# --- Recurrence across arms -------------------------------------------------------------------
#
# The per-arm pass groups each arm on its own, so its concept labels are scoped to that arm, and
# two labels from different arms that read alike prove nothing. The cross-arm pass adds
# `arm_coverage`: which arms are inside one concept, counted from its members and not its label.



def test_collect_skips_a_cell_whose_fit_failed(tmp_path):
    """A failed cell leaves its directory with no `features.json`.

    The stability and co-discovery analyses skip such a cell and the digest guards against it.
    `collect` skips it too, so a single failed cell cannot cost a whole run its recurrence tables.
    """
    from seqmaestro.features.base import FeatureRecord
    from seqmaestro.harness.features import save_features

    good = tmp_path / "arm" / "real" / "seed_42"
    good.mkdir(parents=True)
    save_features(good / "features.json",
                  [FeatureRecord(name="f", description="", kind="llm_generated", importance=1.0)])
    (tmp_path / "arm" / "permuted" / "seed_42").mkdir(parents=True)

    got = mod.collect(tmp_path, "arm", {"llm_generated"}, top_n=None)

    assert [f["name"] for f in got] == ["f"]

def armed(pairs, labels="real", seed="seed_42"):
    """One feature per (arm, name) pair, all in one condition unless stated."""
    return [{"name": name, "description": "", "labels": labels, "seed": seed,
             "provenance": "generated", "arm": arm} for arm, name in pairs]


def test_a_concept_reached_by_two_arms_counts_both():
    feats = armed([("a/pos", "gc_fraction"), ("b/llm", "gc_content")])
    row = mod.arm_coverage(feats, {"gc": [1, 2]}).iloc[0]
    assert row.real_arms == 2
    assert row.arms == "a/pos | b/llm"


def test_one_arm_contributing_twice_is_still_one_arm():
    feats = armed([("a/pos", "gc_fraction"), ("a/pos", "gc_content")])
    row = mod.arm_coverage(feats, {"gc": [1, 2]}).iloc[0]
    assert row.real_arms == 1
    assert row.arms == "a/pos"


def test_permuted_members_are_counted_apart_from_real_ones():
    feats = armed([("a/pos", "gc_real")]) + armed([("b/llm", "gc_perm")], labels="permuted")
    row = mod.arm_coverage(feats, {"gc": [1, 2]}).iloc[0]
    assert (row.real_arms, row.permuted_arms) == (1, 1)
    # only the arms that reached it on real labels are named, since that is the finding
    assert row.arms == "a/pos"


def test_every_concept_gets_a_row_even_when_no_arm_reached_it_on_real_labels():
    feats = armed([("a/pos", "x")], labels="permuted")
    frame = mod.arm_coverage(feats, {"only_permuted": [1]})
    assert len(frame) == 1
    assert frame.iloc[0].real_arms == 0
    assert frame.iloc[0].arms == ""


def test_arm_coverage_joins_onto_the_concept_summary_without_loss():
    feats = armed([("a/pos", "gc_fraction"), ("b/llm", "gc_content")])
    feats += armed([("a/pos", "at_run")], seed="seed_43")
    groups = {"gc": [1, 2], "at": [3]}
    summary = mod.summarise(feats, groups)
    merged = summary.merge(mod.arm_coverage(feats, groups), on="concept")
    assert len(merged) == len(summary) == 2
    assert set(merged.columns) >= {"concept", "real_seeds", "real_arms", "arms"}
