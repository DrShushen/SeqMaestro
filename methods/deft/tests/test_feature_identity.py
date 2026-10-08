"""When are two candidate features the same feature? (`FeatureInfo._identity`)

`remove_duplicates: true` deduplicates the candidate pool. Upstream's `__eq__`/`__hash__`
compared **`score` alone**, so it deduplicated by *score collision*: two genuinely different
features that happened to tie collapsed into one. Ties are routine — many single-position naive
features score identically on the same node — so the pool quietly shrank.

Identity is now the normalised body of the generated function. All offline.
"""

from __future__ import annotations

from copy import copy

import pytest

from deft.utils.feature import FeatureInfo

GC_BODY = 'def gc(X):\n    return X["raw_sequence"].str.count("G")'


def feature(name="f", string=None, score=0.5, description="d"):
    return FeatureInfo(name=name, string=string, score=score, description=description)


# --------------------------------------------------------------------------------------------
# The bug
# --------------------------------------------------------------------------------------------


def test_different_features_that_tie_on_score_are_not_duplicates():
    """The defect. Two unrelated features scoring identically must both survive."""
    a = feature("pos_10_is_A", 'def a(X):\n    return X["raw_sequence"].str[10] == "A"', score=0.5)
    b = feature("pos_77_is_G", 'def b(X):\n    return X["raw_sequence"].str[77] == "G"', score=0.5)

    assert a != b
    assert len(dict.fromkeys([a, b])) == 2


def test_a_pool_of_tied_naive_features_survives_deduplication():
    """The realistic case: many single-position features tie on the same node."""
    pool = [
        feature(f"pos_{i}_is_G", f'def f(X):\n    return X["raw_sequence"].str[{i}] == "G"',
                score=0.42)
        for i in range(20)
    ]

    assert len(dict.fromkeys(pool)) == 20, "score ties must not collapse the candidate pool"


# --------------------------------------------------------------------------------------------
# What *is* a duplicate
# --------------------------------------------------------------------------------------------


def test_the_same_computation_under_a_different_name_is_a_duplicate():
    """Upstream's own notes describe ten variants of `pos_51_is_G` from one generation round."""
    a = feature("upstream_gc", GC_BODY, score=0.5)
    b = feature("gc_content", GC_BODY.replace("def gc(", "def gc_content("), score=0.9)

    assert a == b
    assert len(dict.fromkeys([a, b])) == 1


@pytest.mark.parametrize(
    "label, variant",
    [
        ("reindented", 'def gc(X):\n        return X["raw_sequence"].str.count("G")'),
        ("commented", 'def gc(X):\n    # count G\n    return X["raw_sequence"].str.count("G")'),
        ("renamed and commented",
         'def compute_gc(X):\n    """GC count."""\n    return X["raw_sequence"].str.count("G")'),
    ],
)
def test_formatting_and_comments_do_not_make_a_new_feature(label, variant):
    assert feature("a", GC_BODY) == feature("b", variant)


def test_a_different_computation_is_a_different_feature():
    other = GC_BODY.replace('"G"', '"C"')

    assert feature("a", GC_BODY) != feature("b", other)


# --------------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------------


def test_deduplication_preserves_generation_order():
    """`dict.fromkeys`, not `set`.

    The sort that follows deduplication is stable, so with `set` the tie-break order was the
    hash-bucket order — arbitrary, and a source of run-to-run variation that has nothing to do
    with the method.
    """
    pool = [
        feature("first", 'def a(X):\n    return 1'),
        feature("second", 'def b(X):\n    return 2'),
        feature("third", 'def c(X):\n    return 3'),
    ]

    assert [f.name for f in dict.fromkeys(pool)] == ["first", "second", "third"]
    assert [f.name for f in dict.fromkeys(pool + pool)] == ["first", "second", "third"]


# --------------------------------------------------------------------------------------------
# Candidates without code, and malformed ones
# --------------------------------------------------------------------------------------------


def test_candidates_without_code_fall_back_to_name_and_description():
    """`parse_proposed_feature` produces these, before code generation runs."""
    a = FeatureInfo(name="gc", description="fraction of G or C", score=0.1)
    b = FeatureInfo(name="gc", description="fraction of G or C", score=0.9)
    c = FeatureInfo(name="at", description="fraction of A or T", score=0.1)

    assert a == b and a != c


def test_unparseable_source_falls_back_to_the_raw_text():
    """A hyphenated function name will not parse; identity must still work, not raise."""
    broken = "def upstream_-10bp_GC(X):\n    return 1"

    assert feature("a", broken) == feature("b", broken)
    assert feature("a", broken) != feature("b", broken.replace("return 1", "return 2"))


def test_an_empty_feature_is_hashable():
    """Leaves carry an empty `FeatureInfo`; nothing may blow up on it."""
    assert hash(FeatureInfo()) == hash(FeatureInfo())


def test_comparison_with_a_non_feature_is_not_an_error():
    assert FeatureInfo(name="a") != "a"


# --------------------------------------------------------------------------------------------
# The cache must not go stale
# --------------------------------------------------------------------------------------------


def test_identity_follows_the_source_when_it_changes():
    """`parse_code_feature` copies a semantic candidate and then assigns `.string` to it, so a
    cached identity computed before that assignment would be wrong."""
    semantic = FeatureInfo(name="gc", description="count G")
    first = semantic._identity()

    with_code = copy(semantic)
    with_code.string = GC_BODY

    assert with_code._identity() != first
    assert with_code == feature("other_name", GC_BODY)


def test_scoring_a_feature_does_not_change_its_identity():
    """`optimize_features` mutates `score` and `threshold` in place after identity may be cached."""
    f = feature("gc", GC_BODY, score=None)
    before = f._identity()
    f.score, f.threshold = 0.87, 0.5

    assert f._identity() == before
