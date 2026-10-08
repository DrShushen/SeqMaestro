"""`seqmaestro.harness.stability`.

The estimator is checked against the properties that define it (1 for identical selections, about
0 for independent ones) and not against numbers this implementation happens to produce.
The matching is checked against the cases it exists for: the same property found twice under
different names, and the ways two features can look different and compute the same thing.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from seqmaestro.features.base import FeatureRecord
from seqmaestro.harness.features import save_feature_values, save_features
from seqmaestro.harness.stability import (
    BehaviouralMatcher,
    FeatureRun,
    consensus_labels,
    load_run,
    match_pair,
    nogueira_phi,
    selection_matrix,
    stability,
)

RNG = np.random.default_rng(0)


def a_run(label, columns: dict[str, np.ndarray], **record_kwargs) -> FeatureRun:
    records = [
        FeatureRecord(name=name, description=f"{name} description", kind="llm_generated", **record_kwargs)
        for name in columns
    ]
    return FeatureRun(label, records, np.column_stack(list(columns.values())))


def noise(n=200):
    return RNG.normal(size=n)


# ------------------------------------------------------- matching on behaviour, not on names


def test_the_same_property_under_two_names_is_matched():
    """One property found under two names is matched, which is the main case the module exists
    for."""
    shared = noise()
    one = a_run("seed_1", {"gc_content_40_60": shared, "other": noise()})
    two = a_run("seed_2", {"GC_fraction_upstream": shared, "different": noise()})

    matched = match_pair(one, two, BehaviouralMatcher())

    assert (0, 0) in matched


def test_two_runs_that_found_nothing_do_not_count_as_agreeing():
    """A shuffled-label fit that selects nothing is a control behaving as it should. If two such
    fits counted as agreeing, the arm's chance level would be 1 and its chance-corrected agreement
    would be undefined, or far below zero where only some of them are empty."""
    from seqmaestro.harness.stability import (
        BehaviouralMatcher,
        chance_corrected_agreement,
        jaccard,
        stability,
    )

    def empty(label):
        return FeatureRun(label, [], np.zeros((200, 0)))

    assert jaccard(empty("a"), empty("b"), BehaviouralMatcher()) == 0.0
    chance = stability([empty("a"), empty("b"), empty("c")])["jaccard"]["mean_pairwise"]
    assert chance == 0.0
    assert chance_corrected_agreement(0.6, chance) == 0.6


def test_a_monotone_transform_of_a_feature_is_the_same_feature():
    """A GC count and a GC fraction over the same window are one property. Rank correlation sees
    that, and Pearson correlation would not."""
    base = np.abs(noise()) + 0.1
    one = a_run("seed_1", {"count": base})
    two = a_run("seed_2", {"fraction": base**3})

    assert match_pair(one, two, BehaviouralMatcher()) == [(0, 0)]


def test_a_negated_feature_is_the_same_feature():
    """The sign a model happened to learn does not change what was discovered."""
    base = noise()
    one, two = a_run("seed_1", {"gc": base}), a_run("seed_2", {"at": -base})

    assert match_pair(one, two, BehaviouralMatcher()) == [(0, 0)]


def test_unrelated_features_are_not_matched():
    one, two = a_run("seed_1", {"a": noise()}), a_run("seed_2", {"b": noise()})

    assert match_pair(one, two, BehaviouralMatcher()) == []


def test_matching_is_one_to_one():
    """Two near-identical features in one run cannot both claim the same partner."""
    shared = noise()
    one = a_run("seed_1", {"a": shared, "a_again": shared + 1e-9 * noise()})
    two = a_run("seed_2", {"b": shared})

    matched = match_pair(one, two, BehaviouralMatcher())

    assert len(matched) == 1


def test_a_constant_feature_matches_nothing_on_behaviour_alone():
    """A constant feature has no rank variation, so no correlation is defined and behaviour cannot
    pair it."""
    one = a_run("seed_1", {"always_zero": np.zeros(200)})
    two = a_run("seed_2", {"also_always_zero": np.zeros(200)})

    assert match_pair(one, two, BehaviouralMatcher()) == []


def test_a_constant_feature_still_matches_when_its_definition_is_known():
    """A rare k-mer that never varies on the reference split is invisible to correlation, but two
    runs naming the same k-mer have the same feature. Counted as distinct in every run, such
    features could make the number of "distinct" features across seeds exceed the vocabulary."""
    one = a_run("seed_1", {"kmer_ACGTAC": np.zeros(200)}, params={"kmer": "ACGTAC", "k": 6})
    two = a_run("seed_2", {"kmer_ACGTAC": np.zeros(200)}, params={"kmer": "ACGTAC", "k": 6})

    assert match_pair(one, two, BehaviouralMatcher()) == [(0, 0)]


def test_definitionless_records_are_not_matched_by_their_empty_definition():
    """A record with no code and no parameters has nothing to identify it. Hashing that fallback
    would make every such record collide with every other."""
    one, two = a_run("seed_1", {"a": noise()}), a_run("seed_2", {"b": noise()})

    assert match_pair(one, two, BehaviouralMatcher()) == []


def test_the_threshold_is_honoured():
    base = noise()
    partly = 0.5 * base + 0.5 * noise()
    one, two = a_run("seed_1", {"a": base}), a_run("seed_2", {"b": partly})

    assert match_pair(one, two, BehaviouralMatcher(threshold=0.99)) == []
    assert match_pair(one, two, BehaviouralMatcher(threshold=0.3)) == [(0, 0)]


@pytest.mark.parametrize("threshold", [0, 1.5])
def test_a_nonsense_threshold_is_refused(threshold):
    with pytest.raises(ValueError, match="threshold"):
        BehaviouralMatcher(threshold=threshold)


def test_runs_evaluated_on_different_rows_are_refused():
    """Matching is meaningless unless every run saw the same reference rows in the same order."""
    one, two = a_run("seed_1", {"a": noise(200)}), a_run("seed_2", {"b": noise(150)})

    with pytest.raises(ValueError, match="reference rows"):
        match_pair(one, two, BehaviouralMatcher())


# ------------------------------------------------------------------- pairing by definition first


def test_identical_definitions_pair_before_the_matcher_runs():
    """`exact_first` pairs features that no statistic can match.

    The two runs here share a definition, but their columns are unrelated. This mimics a feature
    that is constant on the reference split: it has no rank variation, so behavioural similarity
    scores it 0 against everything, including its own counterpart. Two features with the same
    definition are the same feature whether or not a statistic can tell.
    """
    one = a_run("seed_1", {"f": noise()}, params={"kmer": "ACGT"})
    two = a_run("seed_2", {"f_renamed": noise()}, params={"kmer": "ACGT"})

    assert match_pair(one, two, BehaviouralMatcher()) == [(0, 0)]
    assert match_pair(one, two, BehaviouralMatcher(), exact_first=False) == [], (
        "without the definition pass nothing could have paired these"
    )


def test_two_routes_to_one_property_pair_on_behaviour():
    """Two different definitions that compute one property pair on behaviour. Stability exists to
    measure this case, and no comparison of definitions can see it."""
    values = noise()
    one = a_run("seed_1", {"a": values}, params={"kmer": "ACGT"})
    two = a_run("seed_2", {"b": values}, params={"kmer": "TTTT"})

    assert match_pair(one, two, BehaviouralMatcher()) == [(0, 0)]


# ------------------------------------------------------------------------ the Nogueira estimator


def test_phi_is_one_when_every_run_selects_the_same_set():
    assert nogueira_phi(np.array([[1, 1, 0, 0]] * 4)) == pytest.approx(1.0)


def test_phi_is_near_zero_for_independent_selection():
    """Phi is corrected for chance, so agreement by luck scores about zero. This is its defining
    property."""
    rng = np.random.default_rng(1)
    universe, per_run = 200, 20
    selection = np.zeros((40, universe), dtype=int)
    for row in selection:
        row[rng.choice(universe, per_run, replace=False)] = 1

    assert abs(nogueira_phi(selection)) < 0.1


def test_phi_falls_as_runs_disagree_more():
    same = np.array([[1, 1, 1, 0, 0, 0]] * 3)
    partly = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 1]])

    assert nogueira_phi(same) > nogueira_phi(partly)


def test_a_larger_universe_convention_changes_phi():
    """The universe size changes phi, so the convention is recorded with the number, and numbers
    under two conventions are never compared."""
    selection = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])

    assert nogueira_phi(selection, universe_size=4) != nogueira_phi(selection, universe_size=5000)


def test_a_universe_smaller_than_what_was_observed_is_refused():
    with pytest.raises(ValueError, match="smaller than"):
        nogueira_phi(np.array([[1, 1], [1, 0]]), universe_size=1)


def test_one_run_is_not_a_stability_measurement():
    with pytest.raises(ValueError, match="at least two runs"):
        nogueira_phi(np.array([[1, 0, 1]]))


def test_selecting_everything_is_stable_but_vacuously():
    assert nogueira_phi(np.ones((3, 5), dtype=int)) == 1.0


# ------------------------------------------------------------------------------- consensus


def test_consensus_gives_matched_features_one_label_across_runs():
    shared, other = noise(), noise()
    runs = [
        a_run("seed_1", {"a": shared, "x": other}),
        a_run("seed_2", {"b": shared, "y": noise()}),
        a_run("seed_3", {"c": shared, "z": other}),
    ]

    labels = consensus_labels(runs, BehaviouralMatcher())
    matrix = selection_matrix(labels)

    assert labels[0][0] == labels[1][0] == labels[2][0], "the shared feature is one cluster"
    assert matrix.sum(axis=0).max() == 3
    assert matrix.shape[0] == 3


# --------------------------------------------------------------------------- the whole thing


def test_identical_runs_are_perfectly_stable():
    """The anchor at one end of the axis: a deterministic arm should land here."""
    columns = {"a": noise(), "b": noise(), "c": noise()}
    runs = [a_run(f"seed_{i}", columns) for i in range(3)]

    result = stability(runs)

    assert result["jaccard"]["mean_pairwise"] == 1.0
    assert result["nogueira"]["phi"] == pytest.approx(1.0)
    assert result["consensus"]["n_clusters"] == 3
    assert result["consensus"]["n_found_by_every_run"] == 3


def test_runs_sharing_nothing_are_maximally_unstable():
    runs = [a_run(f"seed_{i}", {f"f{i}_{j}": noise() for j in range(3)}) for i in range(3)]

    result = stability(runs)

    assert result["jaccard"]["mean_pairwise"] == 0.0
    assert result["consensus"]["n_clusters"] == 9
    assert result["consensus"]["n_found_by_one_run"] == 9


def test_partial_overlap_lands_in_between():
    shared = noise()
    runs = [a_run(f"seed_{i}", {"shared": shared, f"own_{i}": noise()}) for i in range(3)]

    result = stability(runs)

    assert 0 < result["jaccard"]["mean_pairwise"] < 1
    assert result["consensus"]["n_found_by_every_run"] == 1


def test_the_result_records_how_it_was_measured():
    """A stability number is meaningless without the matcher and the universe convention."""
    runs = [a_run(f"seed_{i}", {"a": noise()}) for i in range(2)]

    result = stability(runs, matcher=BehaviouralMatcher(threshold=0.9), universe_size=5460)

    assert result["matcher"]["threshold"] == 0.9
    assert result["matcher"]["statistic"] == "abs_spearman"
    assert result["nogueira"]["universe_size"] == 5460
    assert result["nogueira"]["convention"] == "supplied"


def test_diagnostics_flag_a_constant_feature():
    runs = [a_run(f"seed_{i}", {"a": noise(), "dead": np.zeros(200)}) for i in range(2)]

    assert stability(runs)["diagnostics"]["constant_features"] == {"seed_0": 1, "seed_1": 1}


def test_fewer_than_two_runs_is_refused():
    with pytest.raises(ValueError, match="at least two runs"):
        stability([a_run("only", {"a": noise()})])


# ------------------------------------------------------------------------- loading from disk


def test_a_run_loads_from_the_files_any_arm_writes(tmp_path):
    """The modularity contract: stability reads `features.json` and `feature_values.npz`, and
    knows nothing else about the arm that produced them."""
    records = [FeatureRecord(name="kmer_ACGT", description="d", kind="kmer", origin="deft")]
    save_features(tmp_path / "features.json", records, importance_kind="deft_split_score")
    save_feature_values(tmp_path / "feature_values.npz", ["kmer_ACGT"], np.array([[1.0], [2.0]]))

    run = load_run(tmp_path, label="seed_42")

    assert run.label == "seed_42"
    assert run.names == ["kmer_ACGT"]
    assert run.values.shape == (2, 1)
    assert run.header["importance_kind"] == "deft_split_score"


def test_values_that_do_not_line_up_with_the_records_are_refused(tmp_path):
    """They are paired by position, so a mismatch would silently compare the wrong features."""
    save_features(tmp_path / "features.json",
                  [FeatureRecord(name="a", description="d", kind="kmer"),
                   FeatureRecord(name="b", description="d", kind="kmer")])
    save_feature_values(tmp_path / "feature_values.npz", ["b", "a"], np.zeros((5, 2)))

    with pytest.raises(ValueError, match="do not match the records"):
        load_run(tmp_path)


# ------------------------------------------------- features that are undefined on some rows


def test_a_feature_undefined_on_some_rows_does_not_break_matching():
    """A generated feature can be undefined on some rows without being wrong: a CpG-cluster span
    is NaN for a sequence with no cluster.

    Unhandled, `rankdata` spreads the NaN across the column, and the NaN passes the zero-norm guard
    because `nan != 0` is True. `linear_sum_assignment` then refuses the matrix, and the stability
    analysis fails for every arm, not only the arm holding the feature.
    """
    shared = noise()
    partly_undefined = noise()
    partly_undefined[:80] = np.nan

    one = a_run("seed_1", {"shared": shared, "cluster_span": partly_undefined})
    two = a_run("seed_2", {"shared_again": shared, "other": noise()})

    matched = match_pair(one, two, BehaviouralMatcher())

    assert (0, 0) in matched, "the well-defined feature must still match its counterpart"


def test_a_column_that_is_entirely_undefined_matches_nothing():
    """An all-NaN column has nothing to impute from, so it scores 0 against everything and never
    matches by default."""
    one = a_run("seed_1", {"all_nan": np.full(200, np.nan)})
    two = a_run("seed_2", {"all_nan_too": np.full(200, np.nan)})

    assert match_pair(one, two, BehaviouralMatcher(), exact_first=False) == []


def test_undefined_columns_are_counted_for_reporting():
    from seqmaestro.harness.stability import non_finite_columns

    values = np.column_stack([noise(), np.full(200, np.nan), noise()])
    values[3, 0] = np.inf

    assert non_finite_columns(values) == 2


# ------------------------------------------- chance-corrected agreement, the reported figure


def test_kappa_is_zero_when_real_agreement_matches_its_null():
    """An arm that reproduces its features no better than on shuffled labels has found nothing."""
    from seqmaestro.harness.stability import chance_corrected_agreement

    assert chance_corrected_agreement(0.19, 0.19) == pytest.approx(0.0)


def test_kappa_is_one_for_identical_seeds():
    from seqmaestro.harness.stability import chance_corrected_agreement

    assert chance_corrected_agreement(1.0, 0.19) == pytest.approx(1.0)


def test_kappa_corrects_a_high_baseline_more_than_a_low_one():
    """Chance correction can reverse the ordering of two arms, which is why raw agreement is not
    reported.

    A generative arm can have the highest raw agreement of any arm and also the highest chance
    level, because it draws from a concentrated prior and proposes similar features whatever the
    labels say. A fixed-vocabulary arm agrees less, against a chance level near zero. After
    correction, the fixed-vocabulary arm ranks higher.
    """
    from seqmaestro.harness.stability import chance_corrected_agreement

    proposer = chance_corrected_agreement(0.290, 0.189)
    fixed_vocabulary = chance_corrected_agreement(0.237, 0.009)

    assert proposer < fixed_vocabulary
    assert proposer == pytest.approx(0.1245, abs=1e-3)
    assert fixed_vocabulary == pytest.approx(0.2300, abs=1e-3)


def test_kappa_is_negative_when_shuffled_labels_agree_more():
    """Kappa is not bounded below and must not be clipped, because clipping would hide a real
    signal."""
    from seqmaestro.harness.stability import chance_corrected_agreement

    assert chance_corrected_agreement(0.10, 0.20) < 0


def test_kappa_is_undefined_when_the_null_leaves_no_room():
    from seqmaestro.harness.stability import chance_corrected_agreement

    assert np.isnan(chance_corrected_agreement(1.0, 1.0))
    assert np.isnan(chance_corrected_agreement(float("nan"), 0.1))


def test_a_run_of_one_seed_writes_no_stability_table(tmp_path, monkeypatch):
    """Stability needs a set of runs. With one seed per arm there is none, and the analysis says so
    and writes nothing, so the report reads it as not measured."""
    from seqmaestro.pipeline.analyses import stability as analysis

    cell = tmp_path / "results" / "l1_logistic" / "positional" / "real" / "seed_42"
    cell.mkdir(parents=True)
    (cell / "run.json").write_text("{}")
    (cell / "features.json").write_text("{}")
    (tmp_path / "results" / "stability.csv").write_text("left by an earlier attempt\n")
    # The cut is the run's setting, which the analysis takes from the run's recorded config.
    stated = SimpleNamespace(analyses=SimpleNamespace(top_n=20))
    monkeypatch.setattr(analysis, "load_recorded", lambda run_dir: stated)
    assert analysis.main(["--run-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "results" / "stability.csv").exists()
