"""Motif discovery (`seqmaestro.motifs.discovery`) and the discovered-motif featuriser: which rows
are compared, how STREME's and Tomtom's output is parsed, the store and its lock, the featuriser's
records, the caveat the reliability panel gives, and how the report names the closest known motif.

The tests not marked `crossenv` run the stand-in for STREME and Tomtom (`tests/small_motifs.py`),
whereas those marked `crossenv` run the MEME Suite in its environment and are skipped where it is
not built.
"""

from __future__ import annotations

import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from seqmaestro.features import MOTIF_DISCOVERED, DiscoveredMotifFeaturiser, make_featuriser
from seqmaestro.motifs import Collection, Motif, discovery, read_meme
from seqmaestro.motifs.discovery import (
    DOWN,
    MEME_ENVIRONMENT,
    UP,
    DiscoverySettings,
    compared,
    discover,
)
from seqmaestro.pipeline.config import DiscoveredMotifSettings
from seqmaestro.pipeline.motifs import discovered_motifs as run_discovered_motifs
from seqmaestro.provenance import missing_python
from seqmaestro.reliability.pool import DISCOVERED_CAVEAT, RECORDED_CAVEAT, RECORDED_FEATURES, Pool
from seqmaestro.report.tables import what_it_computes
from seqmaestro.target import BINARY, CONTINUOUS
from tests import small_motifs
from tests.planted import planted_frame

#: The rows of the planted dataset, whose positives contain GGCGCC.
PLANTED = planted_frame(128, np.random.default_rng(3))
SEQUENCES = PLANTED["window"].tolist()
LABELS = PLANTED["outcome"].to_numpy()
#: Discovery as `_motifs.yaml` states it.
SETTINGS = DiscoverySettings(directions="both", continuous_fraction=0.333, min_width=6,
                             max_width=15, max_motifs=10, strands="both")
#: The two motifs written for the planted dataset: its site and one that matches nothing in it.
COLLECTION = Collection("small", (Motif("PL1", "PLANTED", small_motifs.expecting("GGCGCC"), sites=50),
                                  Motif("TB1", "TATA", small_motifs.expecting("TATAAA"), sites=50)),
                        digest="small")


def discovered(**changed):
    """The discovered-motif source's settings for the planted rows, with `changed` in place."""
    settings = replace(small_motifs.discovered_motifs(), discovery=SETTINGS,
                       settings=replace(small_motifs.SETTINGS, p_threshold=1e-3),
                       annotate_with=COLLECTION)
    return replace(settings, **changed)


# ---------------------------------------------------------------------------- what is compared


def test_a_binary_target_is_compared_class_against_class_both_ways():
    y = np.array([1, 0, 0, 1, 1, 0])

    pairs = compared(y, BINARY, SETTINGS)

    assert list(pairs) == [UP, DOWN]
    np.testing.assert_array_equal(pairs[UP][0], [0, 3, 4])
    np.testing.assert_array_equal(pairs[UP][1], [1, 2, 5])
    np.testing.assert_array_equal(pairs[DOWN][0], pairs[UP][1])
    np.testing.assert_array_equal(pairs[DOWN][1], pairs[UP][0])
    assert list(compared(y, BINARY, replace(SETTINGS, directions="up"))) == [UP]


def test_a_continuous_target_compares_its_two_ends_and_leaves_the_middle_out():
    y = np.array([5.0, 1.0, 3.0, 9.0, 7.0, 2.0])

    pairs = compared(y, CONTINUOUS, replace(SETTINGS, continuous_fraction=0.34))

    # The top third holds 9 and 7, the bottom third 1 and 2.
    np.testing.assert_array_equal(pairs[UP][0], [3, 4])
    np.testing.assert_array_equal(pairs[UP][1], [1, 5])
    np.testing.assert_array_equal(pairs[DOWN][0], [1, 5])


def test_too_few_rows_at_either_end_are_refused():
    with pytest.raises(ValueError, match="at least two rows"):
        compared(np.array([1, 0, 0]), BINARY, SETTINGS)
    with pytest.raises(ValueError, match="No discovery"):
        compared(np.array([1, 0, 0, 1]), "ordinal", SETTINGS)


@pytest.mark.parametrize("changed, message", [
    ({"directions": "down"}, "directions"),
    ({"continuous_fraction": 0.0}, "continuous_fraction"),
    ({"continuous_fraction": 0.6}, "continuous_fraction"),
    ({"min_width": 2}, "widths"),
    ({"min_width": 9, "max_width": 8}, "widths"),
    ({"max_width": 31}, "widths"),
    ({"max_motifs": 0}, "max_motifs"),
    ({"strands": "forward"}, "strands"),
])
def test_a_setting_that_means_nothing_is_refused(changed, message):
    with pytest.raises(ValueError, match=message):
        replace(SETTINGS, **changed)


def test_the_configs_settings_are_the_ones_discovery_and_the_scan_take():
    settings = DiscoveredMotifSettings(strands="given", summaries=("count",), p_threshold=1e-3,
                                       normalise="per_length", min_match_share=0.02,
                                       directions="up", continuous_fraction=0.25, min_width=5,
                                       max_width=12, max_motifs=4)

    assert settings.discovery() == DiscoverySettings(directions="up", continuous_fraction=0.25,
                                                     min_width=5, max_width=12, max_motifs=4,
                                                     strands="given")
    assert (settings.scan().strands, settings.scan().p_threshold) == ("given", 1e-3)


# ------------------------------------------------------------------- what the tools leave behind

#: STREME's text output for two motifs, as STREME writes it.
STREME_TEXT = """\
********************************************************************************
STREME - Sensitive, Thorough, Rapid, Enriched Motif Elicitation
********************************************************************************
MEME version 5.5.9 (Release date: Fri Nov 21 15:33:51 2025 -0800)

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.255 C 0.245 G 0.245 T 0.255

MOTIF 1-SNGGCGCC STREME-1
letter-probability matrix: alength= 4 w= 8 nsites= 74 E= 3.8e-003
 0.156529 0.436955 0.374632 0.031884
 0.312336 0.281148 0.156503 0.250013
 0.000723 0.000697 0.997858 0.000723
 0.000723 0.000697 0.997858 0.000723
 0.000723 0.997858 0.000697 0.000723
 0.000723 0.000697 0.997858 0.000723
 0.000723 0.997858 0.000697 0.000723
 0.000723 0.997858 0.000697 0.000723

MOTIF 2-AAAAGA STREME-2
letter-probability matrix: alength= 4 w= 6 nsites= 1 E= 3.0e+000
 0.885128 0.037818 0.037818 0.039236
 0.885128 0.037818 0.037818 0.039236
 0.885128 0.037818 0.037818 0.039236
 0.885128 0.037818 0.037818 0.039236
 0.039236 0.037818 0.883710 0.039236
 0.885128 0.037818 0.037818 0.039236
"""

#: Tomtom's table, as it writes it: a header, a row per match, and comment lines at the end.
TOMTOM_TABLE = """\
Query_ID\tTarget_ID\tOptimal_offset\tp-value\tE-value\tq-value\tOverlap\tQuery_consensus\tTarget_consensus\tOrientation
1-SNGGCGCC\tTB1\t-9\t0.999966\t1.99993\t0.999975\t6\tCAGGCGCC\tTATAAA\t+
1-SNGGCGCC\tPL1\t-2\t0.000976205\t0.00195241\t0.00195241\t6\tCAGGCGCC\tGGCGCC\t+
2-AAAAGA\tTB1\t0\t0.660353\t1.32071\t1\t6\tAAAAGA\tTATAAA\t+

# Tomtom (Motif Comparison Tool): Version 5.5.9
# The format of this file is described at https://meme-suite.org/meme/doc/tomtom-output-format.html.
"""


def test_streme_s_motifs_are_read_with_their_e_values():
    motifs = read_meme(STREME_TEXT)

    assert [(motif.identifier, motif.width, motif.sites) for motif in motifs] == [
        ("1-SNGGCGCC", 8, 74.0), ("2-AAAAGA", 6, 1.0)]
    assert discovery._evalues(STREME_TEXT) == {"1-SNGGCGCC": 3.8e-3, "2-AAAAGA": 3.0}


def test_a_motif_streme_could_not_test_on_held_out_rows_has_no_e_value():
    """With fewer than five sequences of either set to hold out from its search, STREME holds none
    out and writes a score on the training sequences (`S=`) in place of the E-value."""
    untested = STREME_TEXT.replace("nsites= 74 E= 3.8e-003", "nsites= 74 S= 5.6e-022")

    assert discovery._evalues(untested) == {"1-SNGGCGCC": None, "2-AAAAGA": 3.0}


def test_the_closest_known_motif_is_the_match_with_the_lowest_q_value():
    closest = discovery._closest(TOMTOM_TABLE, COLLECTION)

    assert closest == {
        "1-SNGGCGCC": {"collection": "small", "motif_id": "PL1", "motif_name": "PLANTED",
                       "q_value": 0.00195241, "p_value": 0.000976205},
        "2-AAAAGA": {"collection": "small", "motif_id": "TB1", "motif_name": "TATA",
                     "q_value": 1.0, "p_value": 0.660353}}
    assert discovery._closest("", COLLECTION) == {}, "Tomtom found nothing under its threshold"


# ---------------------------------------------------------------------- the store and its lock


def test_a_discovery_is_kept_under_the_root_and_found_there_again(tmp_path, monkeypatch):
    found = discover(SEQUENCES, LABELS, BINARY, SETTINGS, COLLECTION, tmp_path)

    [folder] = [path for path in (tmp_path / discovery.DISCOVERIES_DIR).iterdir() if path.is_dir()]
    assert (folder / discovery.FOUND_FILE).exists()
    assert folder.name == discovery.discovery_key(SEQUENCES, LABELS, BINARY, SETTINGS, COLLECTION)
    first = found[0]
    assert (first.motif.identifier, first.direction, first.rank, first.consensus) == (
        "up_1_GGCGCC", UP, 1, "GGCGCC")
    assert first.closest["motif_id"] == "PL1"

    monkeypatch.setattr(discovery, "_streme", lambda *args: pytest.fail("discovered again"))
    again = discover(SEQUENCES, LABELS, BINARY, SETTINGS, COLLECTION, tmp_path)
    assert [entry.as_dict() for entry in again] == [entry.as_dict() for entry in found]


def test_other_labels_make_another_discovery(tmp_path):
    shuffled = np.random.default_rng(0).permutation(LABELS)

    discover(SEQUENCES, LABELS, BINARY, SETTINGS, None, tmp_path)
    discover(SEQUENCES, shuffled, BINARY, SETTINGS, None, tmp_path)

    assert len([path for path in (tmp_path / discovery.DISCOVERIES_DIR).iterdir()
                if path.is_dir()]) == 2


def test_without_a_root_nothing_is_kept_and_the_motifs_are_the_same(tmp_path):
    kept = discover(SEQUENCES, LABELS, BINARY, SETTINGS, None, tmp_path)

    missing = tmp_path / "gone"
    alone = discover(SEQUENCES, LABELS, BINARY, SETTINGS, None, missing)

    assert not missing.exists()
    assert [entry.as_dict() for entry in alone] == [entry.as_dict() for entry in kept]


def discover_in_a_process(root: str, log: str) -> list[str]:
    """Ask for the discovery from one of two processes, with the stand-in for STREME slowed down and
    counting its calls in `log`. A spawned process starts a fresh interpreter, so it installs the
    stand-ins itself."""

    def counted(*args):
        with open(log, "a") as handle:
            handle.write("called\n")
        time.sleep(0.3)
        return small_motifs.stand_in_streme(*args)

    discovery._streme = counted
    discovery._tomtom = small_motifs.stand_in_tomtom
    discovery.versions = small_motifs.stand_in_versions
    return [entry.motif.identifier
            for entry in discover(SEQUENCES, LABELS, BINARY, SETTINGS, None, Path(root))]


def test_two_processes_asking_for_one_discovery_make_it_once(tmp_path):
    log = tmp_path / "calls.log"
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as processes:
        found = list(processes.map(discover_in_a_process, [str(tmp_path)] * 2, [str(log)] * 2))

    assert found[0] == found[1]
    assert log.read_text().count("called") == 2, "one search per direction, by one process"


# -------------------------------------------------------------------------------- the featuriser


def test_the_featuriser_names_each_motif_for_how_it_was_found():
    featuriser = DiscoveredMotifFeaturiser(discovered()).fit(pd.Series(SEQUENCES), LABELS)

    names = featuriser.feature_names
    assert names[:2] == ["streme_up_1_GGCGCC_best", "streme_up_1_GGCGCC_count"]
    assert any(name.startswith("streme_down_1_") for name in names)
    record = featuriser.records[0]
    assert record.kind == "pwm_scan"
    assert record.params == {"source": MOTIF_DISCOVERED, "summary": "best_match", "strands": "both",
                             "tool": "streme", "direction": UP, "rank": 1, "consensus": "GGCGCC",
                             "motif_digest": featuriser.scanner.motifs[0].digest()[:16]}
    assert "enriched in class 1 against class 0" in record.description
    assert record.extra["closest_known"]["motif_id"] == "PL1"
    assert "PL1" not in record.description and "PLANTED" not in record.description
    assert np.array(record.extra["frequencies"]).shape == (6, 4)

    described = featuriser.describe()
    assert described["kind"] == MOTIF_DISCOVERED and described["found"] == {UP: 10, DOWN: 10}
    assert described["tool_versions"] == small_motifs.STAND_IN_VERSIONS
    assert described["annotate_with"] == "small"
    assert described["discovery_key"] == discovery.discovery_key(SEQUENCES, LABELS, BINARY, SETTINGS,
                                                                 COLLECTION)


def test_the_planted_motif_separates_the_classes():
    X = DiscoveredMotifFeaturiser(discovered()).fit_transform(pd.Series(SEQUENCES), LABELS)

    count = X["streme_up_1_GGCGCC_count"].to_numpy()
    assert count[LABELS == 1].min() >= 1 and count[LABELS == 0].mean() < 0.2


def test_a_motif_without_an_e_value_is_described_without_one(monkeypatch):
    def untested(*args):
        return [(motif, None) for motif, _ in small_motifs.stand_in_streme(*args)]

    monkeypatch.setattr(discovery, "_streme", untested)
    featuriser = DiscoveredMotifFeaturiser(discovered()).fit(pd.Series(SEQUENCES), LABELS)

    record = featuriser.records[0]
    assert "(consensus GGCGCC)" in record.description and "E =" not in record.description
    assert record.extra["evalue"] is None


def test_discovery_needs_the_labels_and_at_least_one_motif(monkeypatch):
    with pytest.raises(ValueError, match="needs the labels"):
        DiscoveredMotifFeaturiser(discovered()).fit(pd.Series(SEQUENCES))

    monkeypatch.setattr(discovery, "_streme", lambda *args: [])
    with pytest.raises(ValueError, match="found no motif"):
        DiscoveredMotifFeaturiser(discovered()).fit(pd.Series(SEQUENCES), LABELS)


def test_a_fitted_featuriser_scans_new_rows_with_what_it_found(monkeypatch):
    featuriser = DiscoveredMotifFeaturiser(discovered()).fit(pd.Series(SEQUENCES), LABELS)
    loaded = pickle.loads(pickle.dumps(featuriser))
    monkeypatch.setattr(discovery, "_streme", lambda *args: pytest.fail("discovered again"))

    new = pd.Series(["AAGGCGCCAAAAAAAAAAAA", "TTTTTTTTTTTTTTTTTTTT"])
    X = loaded.transform(new)

    assert list(X.columns) == featuriser.feature_names
    assert X.loc[0, "streme_up_1_GGCGCC_count"] >= 1 and X.loc[1, "streme_up_1_GGCGCC_count"] == 0


def test_the_featuriser_is_built_for_the_pools_that_hold_it():
    with pytest.raises(ValueError, match="must say how discovery is asked"):
        make_featuriser("positional+motif_discovered")
    with pytest.raises(ValueError, match="does not hold"):
        make_featuriser("positional", motif_discovered=discovered())
    with pytest.raises(ValueError, match="what up and down are"):
        DiscoveredMotifFeaturiser(discovered(described={UP: "the bound ones"}))


def test_a_runs_directions_are_described_by_its_classes_or_its_target():
    settings = DiscoveredMotifSettings(strands="both", summaries=("count",), p_threshold=1e-4,
                                       normalise="counts", min_match_share=0.01, directions="both",
                                       continuous_fraction=0.333, min_width=6, max_width=15,
                                       max_motifs=10)
    config = SimpleNamespace(sources=SimpleNamespace(motif_discovered=settings))
    binary = SimpleNamespace(target_kind=BINARY, classes={0: SimpleNamespace(name="unbound"),
                                                          1: SimpleNamespace(name="bound")})
    continuous = SimpleNamespace(target_kind=CONTINUOUS, record={"target": {"name": "expression"}})

    assert run_discovered_motifs(config, binary, None).described == {
        UP: "bound sequences against unbound ones", DOWN: "unbound sequences against bound ones"}
    assert run_discovered_motifs(config, continuous, None).described == {
        UP: "the sequences in the highest 33% of expression against the lowest",
        DOWN: "the sequences in the lowest 33% of expression against the highest"}


# ------------------------------------------------------------------- the panel and the report


def pool(params: list[dict], source: str = "featuriser") -> Pool:
    return Pool(names=[f"f{index}" for index in range(len(params))], kinds=["pwm_scan"] * len(params),
                params=params, descriptions=[""] * len(params),
                values=np.zeros((3, len(params))), source=source)


def test_the_panel_says_tier_1_is_optimistic_for_discovered_motifs():
    discovered_motif = {"source": MOTIF_DISCOVERED, "summary": "count"}
    known_motif = {"source": "motif_known", "summary": "count"}

    assert pool([known_motif]).caveat() is None
    assert pool([known_motif, discovered_motif]).caveat() == DISCOVERED_CAVEAT
    assert pool([discovered_motif], RECORDED_FEATURES).caveat() == (
        f"{RECORDED_CAVEAT} {DISCOVERED_CAVEAT}")


def test_the_report_names_a_known_motif_close_to_a_discovered_one():
    closest = {"collection": "JASPAR", "motif_id": "MA0139.2", "motif_name": "CTCF",
               "q_value": 0.00123, "p_value": 1e-5}
    feature = {"description": "the strength of the best match to a found motif",
               "closest_known": closest}

    assert what_it_computes(feature) == ("the strength of the best match to a found motif. Close "
                                         "to the known motif CTCF (MA0139.2) of JASPAR, q = 0.0012.")
    far = {**feature, "closest_known": {**closest, "q_value": 0.3}}
    assert what_it_computes(far) == feature["description"], "the closest motif is not close"
    assert what_it_computes({**feature, "closest_known": None}) == feature["description"]


# ------------------------------------------------------------------------- the MEME Suite itself

needs_meme = pytest.mark.skipif(missing_python(MEME_ENVIRONMENT) is not None,
                                reason="the MEME Suite's environment is not built here")


@pytest.mark.crossenv
@needs_meme
def test_streme_finds_the_planted_site_and_tomtom_names_it(tmp_path):
    found = discover(SEQUENCES, LABELS, BINARY, SETTINGS, COLLECTION, tmp_path)

    first = found[0]
    assert (first.direction, first.rank) == (UP, 1) and "GGCGCC" in first.consensus
    assert first.evalue < 0.05
    assert first.closest["motif_id"] == "PL1" and first.closest["q_value"] < 0.05
    assert discovery.versions()["streme"].startswith("5.")
    assert (tmp_path / discovery.DISCOVERIES_DIR).is_dir()


@pytest.mark.crossenv
@needs_meme
def test_on_shuffled_labels_streme_reports_nothing_of_significance():
    shuffled = np.random.default_rng(0).permutation(LABELS)

    found = discover(SEQUENCES, shuffled, BINARY, SETTINGS, None, None)

    assert found, "STREME reports the motifs that made it stop, significant or not"
    assert min(entry.evalue for entry in found) > 0.05


@pytest.mark.crossenv
@needs_meme
def test_streme_finds_the_site_at_the_top_of_a_continuous_target():
    """A third of 240 rows is 80, enough for STREME to hold eight out of each set and test the
    motifs it finds on them."""
    frame = planted_frame(240, np.random.default_rng(3), continuous=True)

    found = discover(frame["window"].tolist(), frame["outcome"].to_numpy(), CONTINUOUS, SETTINGS,
                     None, None)

    first = found[0]
    assert (first.direction, first.rank) == (UP, 1) and "GGCGCC" in first.consensus
    assert first.evalue == min(entry.evalue for entry in found)


@pytest.mark.crossenv
@needs_meme
def test_with_too_few_rows_to_hold_out_streme_gives_no_e_value():
    """A third of 128 rows is 42, and a tenth of 42 is fewer than the five sequences STREME needs
    to hold any out."""
    frame = planted_frame(128, np.random.default_rng(3), continuous=True)

    found = discover(frame["window"].tolist(), frame["outcome"].to_numpy(), CONTINUOUS, SETTINGS,
                     None, None)

    assert "GGCGCC" in found[0].consensus
    assert {entry.evalue for entry in found} == {None}


@pytest.mark.crossenv
@needs_meme
def test_on_the_given_strand_streme_reads_rna(tmp_path):
    rna = [sequence.replace("T", "U") for sequence in SEQUENCES]

    found = discover(rna, LABELS, BINARY, replace(SETTINGS, strands="given"), COLLECTION, None)

    assert "GGCGCC" in found[0].consensus and found[0].closest["motif_id"] == "PL1"
