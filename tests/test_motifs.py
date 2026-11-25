"""`seqmaestro.motifs` (the motif file format, p-values, the scan and its store) and the known-motif
featuriser.

The scan is checked against a direct computation: every window scored one position at a time on
both strands, and every p-value against an enumeration of every sequence a short motif can match.
The settings and sequences range well beyond any one dataset's: widths from one to longer than some
sequences, both alphabets and both cases, invalid symbols, empty sequences, and lengths spread over
two orders of magnitude. Where the MEME Suite's environment is built, a `crossenv` test also checks
the scan against FIMO, the MEME Suite's scanner.
"""

from __future__ import annotations

import itertools
import pickle
import subprocess

import numpy as np
import pandas as pd
import pytest

from seqmaestro.features.motif import KIND_MOTIF, MOTIF_KNOWN, KnownMotifFeaturiser, name_part
from seqmaestro.motifs import Collection, Motif, ScanSettings, ScanStore, read_meme, write_meme
from seqmaestro.motifs import scan as scanning
from seqmaestro.motifs.collection import DEFAULT_SITES, MotifFileError, read_collection
from seqmaestro.motifs.discovery import MEME_ENVIRONMENT, tool
from seqmaestro.motifs.scan import (
    Scanner,
    integer_matrix,
    score_distribution,
    upper_tail,
)
from seqmaestro.motifs.store import SCANS_DIR, scan_with_store, sequence_digest
from seqmaestro.provenance import missing_python

BOTH = ScanSettings(strands="both", p_threshold=1e-2, summaries=("best_match", "count"),
                    normalise="counts")
#: A threshold a motif of three positions can reach: its best match has p = 1/64.
LOOSE = ScanSettings(strands="both", p_threshold=0.05, summaries=("best_match", "count"),
                     normalise="counts")

MEME_DNA = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF MA0001.1 FAKE1
letter-probability matrix: alength= 4 w= 3 nsites= 10 E= 0
 0.900000  0.033333  0.033333  0.033334
 0.033333  0.033333  0.900000  0.033334
 0.000000  0.000000  0.000000  1.000000
URL http://example.org/MA0001.1

MOTIF MA0002.1
letter-probability matrix: alength= 4 w= 2
 0.25 0.25 0.25 0.25
 0.10 0.20 0.30 0.40
"""

MEME_RNA = """MEME version 4

ALPHABET= ACGU

MOTIF RNCMPT0001 RBP1

letter-probability matrix: alength= 4 w= 2 nsites= 20 E= 0
  0.1 0.1 0.1 0.7
  0.7 0.1 0.1 0.1

"""


def random_motif(rng: np.random.Generator, width: int, identifier: str = "M") -> Motif:
    frequencies = rng.dirichlet(np.full(4, 0.4), size=width)
    return Motif(identifier, None, frequencies, sites=float(rng.integers(5, 500)))


def random_sequence(rng: np.random.Generator, length: int, letters: str = "ACGT") -> str:
    return "".join(rng.choice(list(letters), length))


# ------------------------------------------------------------------------------ the format


def test_meme_reads_identifiers_names_sites_and_rows():
    first, second = read_meme(MEME_DNA)

    assert (first.identifier, first.name, first.sites, first.width) == ("MA0001.1", "FAKE1", 10.0, 3)
    assert first.consensus == "AGT"
    assert np.allclose(first.frequencies.sum(axis=1), 1)
    # A motif with no name and no nsites is read as the MEME Suite reads it.
    assert (second.identifier, second.name, second.sites) == ("MA0002.1", None, DEFAULT_SITES)


def test_an_rna_file_reads_u_as_t():
    (motif,) = read_meme(MEME_RNA)

    assert motif.consensus == "TA"
    assert motif.name == "RBP1"


def test_writing_and_reading_back_keeps_every_motif():
    motifs = read_meme(MEME_DNA)

    again = read_meme(write_meme(motifs))

    assert [(m.identifier, m.name, m.sites) for m in again] == [(m.identifier, m.name, m.sites)
                                                                for m in motifs]
    assert all(np.allclose(a.frequencies, b.frequencies, atol=1e-6) for a, b in zip(again, motifs, strict=True))
    assert "ALPHABET= ACGU" in write_meme(motifs, alphabet="ACGU")


@pytest.mark.parametrize("text, message", [
    ("ALPHABET= ACDEFGHIKLMNPQRSTVWY\nMOTIF X\nletter-probability matrix: w= 1\n1 0 0 0\n", "alphabet"),
    ("MOTIF X\nletter-probability matrix: w= 1\n1 0 0 0\n", "before the file declares"),
    ("ALPHABET= ACGT\nMOTIF X\nletter-probability matrix: w= 2\n1 0 0 0\n", "states w= 2"),
    ("ALPHABET= ACGT\nMOTIF X\nletter-probability matrix: w= 1\n1 0 0\n", "row of 3 values"),
    ("ALPHABET= ACGT\nMOTIF X\nletter-probability matrix: w= 1\n0.5 0 0 0\n", "sums to"),
    ("ALPHABET= ACGT\n", "holds no motif"),
])
def test_a_damaged_file_is_refused(text, message):
    with pytest.raises(MotifFileError, match=message):
        read_meme(text)


def test_a_collection_refuses_a_repeated_identifier():
    motif = read_meme(MEME_DNA)[0]

    with pytest.raises(ValueError, match="more than once"):
        Collection("twice", (motif, motif), digest="x")


def test_a_collection_records_the_digest_of_its_file(tmp_path):
    path = tmp_path / "motifs.meme"
    path.write_text(MEME_DNA)

    collection = read_collection(path, "fake")

    assert len(collection.motifs) == 2
    path.write_text(MEME_DNA.replace("FAKE1", "FAKE9"))
    assert read_collection(path, "fake").digest != collection.digest


# --------------------------------------------------------------------------- the p-values


@pytest.mark.parametrize("width", [1, 3, 5])
def test_the_score_distribution_matches_an_enumeration(width):
    """Every sequence of the motif's width is scored, and the distribution and its upper tail must
    be those of all these scores, each sequence equally likely."""
    rng = np.random.default_rng(width)
    matrix = integer_matrix(random_motif(rng, width))
    scores = np.array([matrix[np.arange(width), list(letters)].sum()
                       for letters in itertools.product(range(4), repeat=width)])

    low, distribution = score_distribution(matrix)

    assert low == scores.min()
    assert low + len(distribution) - 1 == scores.max()
    tail = upper_tail(distribution)
    for score in np.unique(scores):
        assert tail[score - low] == pytest.approx((scores >= score).mean(), rel=1e-12)


# ------------------------------------------------------------------------------------ the scan


def direct_scan(motif: Motif, sequence: str, settings: ScanSettings) -> dict[str, float]:
    """Scan one sequence a window at a time: the best score's p-value (as `-log10`) and the count of
    passing windows (divided by the windows scanned under `per_length`), both from an enumeration of
    the motif's scores."""
    matrix = integer_matrix(motif)
    width = motif.width
    codes = [{"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}.get(letter.upper()) for letter in sequence]
    all_scores = np.array([matrix[np.arange(width), list(letters)].sum()
                           for letters in itertools.product(range(4), repeat=width)])
    passing_score = min((score for score in np.unique(all_scores)
                         if (all_scores >= score).mean() <= settings.p_threshold), default=None)
    best, count, scanned = None, 0, 0
    for start in range(len(codes) - width + 1):
        window = codes[start: start + width]
        if None in window:
            continue
        scanned += 1
        score = sum(matrix[j, code] for j, code in enumerate(window))
        if settings.strands == "both":
            reverse = [3 - code for code in reversed(window)]
            score = max(score, sum(matrix[j, code] for j, code in enumerate(reverse)))
        best = score if best is None else max(best, score)
        count += passing_score is not None and score >= passing_score
    strength = 0.0 if best is None else -np.log10((all_scores >= best).mean())
    if settings.normalise == "per_length":
        count = count / scanned if scanned else 0.0
    return {"best_match": strength, "count": float(count)}


@pytest.mark.parametrize("strands", ["both", "given"])
@pytest.mark.parametrize("normalise", ["counts", "per_length"])
def test_the_scan_matches_a_direct_computation(strands, normalise):
    rng = np.random.default_rng(7)
    motifs = [random_motif(rng, width, f"M{width}") for width in (1, 2, 4, 5, 6)]
    sequences = ["", "A", "ACGTN", "acgtacgt", "ACGUUGCA", "NNNNNNNN", "AC-GTTAC"]
    sequences += [random_sequence(rng, int(length), "ACGTN" if length % 3 == 0 else "ACGT")
                  for length in rng.integers(0, 60, size=25)]
    settings = ScanSettings(strands=strands, p_threshold=0.01, summaries=("best_match", "count"),
                            normalise=normalise)

    found = Scanner(motifs, settings).scan(sequences)

    for row, sequence in enumerate(sequences):
        for index, motif in enumerate(motifs):
            expected = direct_scan(motif, sequence, settings)
            where = (sequence, motif.identifier)
            assert found[row, 2 * index] == pytest.approx(expected["best_match"], rel=1e-9), where
            assert found[row, 2 * index + 1] == pytest.approx(expected["count"]), where


def test_batches_of_any_size_give_the_same_scan(monkeypatch):
    """Sequences are scanned in batches of similar length, and the batching must not change a value:
    batches of one row and one batch of all must agree."""
    rng = np.random.default_rng(3)
    motifs = [random_motif(rng, width, f"M{width}") for width in (3, 8)]
    sequences = [random_sequence(rng, int(length)) for length in rng.integers(1, 400, size=40)]
    whole = Scanner(motifs, BOTH).scan(sequences)

    monkeypatch.setattr(scanning, "BATCH_LETTERS", 1)

    assert np.array_equal(Scanner(motifs, BOTH).scan(sequences), whole)


def test_the_reverse_strand_is_scanned_only_for_both():
    """A motif for GGGA matches TCCC only on the reverse strand."""
    motif = Motif("GGGA", None, np.eye(4)[[2, 2, 2, 0]], sites=100)
    given = ScanSettings(strands="given", p_threshold=1e-2, summaries=("count",), normalise="counts")
    both = ScanSettings(strands="both", p_threshold=1e-2, summaries=("count",), normalise="counts")

    assert Scanner([motif], given).scan(["TTCCCTT", "GGGA"]).ravel().tolist() == [0.0, 1.0]
    assert Scanner([motif], both).scan(["TTCCCTT", "GGGA"]).ravel().tolist() == [1.0, 1.0]


def test_u_and_t_and_either_case_scan_alike():
    motif = Motif("TTAG", None, np.eye(4)[[3, 3, 0, 2]], sites=100)
    scanner = Scanner([motif], BOTH)

    values = scanner.scan(["CCTTAGCC", "ccuuagcc", "CCUUAGCC"])

    assert (values == values[0]).all()
    assert values[0, 1] == 1


def test_a_motif_too_short_for_the_threshold_never_counts():
    """A two-position motif's best match has p = 1/16, so a window never reaches p <= 1e-4."""
    motif = Motif("short", None, np.eye(4)[[0, 1]], sites=100)
    settings = ScanSettings(strands="both", p_threshold=1e-4, summaries=("count",), normalise="counts")
    scanner = Scanner([motif], settings)

    assert scanner.unreachable() == [motif]
    assert scanner.scan(["ACACACAC"]).sum() == 0


def test_the_settings_are_refused_when_they_mean_nothing():
    with pytest.raises(ValueError, match="strands"):
        ScanSettings(strands="forward", p_threshold=1e-4, summaries=("count",), normalise="counts")
    with pytest.raises(ValueError, match="threshold"):
        ScanSettings(strands="both", p_threshold=1.0, summaries=("count",), normalise="counts")
    with pytest.raises(ValueError, match="summaries"):
        ScanSettings(strands="both", p_threshold=1e-4, summaries=(), normalise="counts")
    with pytest.raises(ValueError, match="normalise"):
        ScanSettings(strands="both", p_threshold=1e-4, summaries=("count",), normalise="per_base")


def test_the_digest_follows_every_setting_and_motif():
    rng = np.random.default_rng(1)
    motifs = [random_motif(rng, 4)]
    digest = Scanner(motifs, BOTH).digest()

    assert Scanner(motifs, BOTH).digest() == digest
    for changed in [{"strands": "given"}, {"p_threshold": 1e-3}, {"summaries": ("count",)},
                    {"normalise": "per_length"}]:
        settings = ScanSettings(**{**BOTH.as_dict(), "summaries": BOTH.summaries, **changed})
        assert Scanner(motifs, settings).digest() != digest
    assert Scanner([random_motif(rng, 4)], BOTH).digest() != digest


# ------------------------------------------------------------------------------ against FIMO

#: How far a window's score may differ from FIMO's, in bits, and its p-value, in log10. FIMO uses
#: the same log-odds and pseudocount, but before computing p-values it scales the scores to a grid
#: of a thousand steps over the motif's range, which is coarser than ours. A p-value can thus differ
#: by a few percent, and a window at the threshold can fall on either side of it.
FIMO_SCORE_TOLERANCE = 0.15
FIMO_P_TOLERANCE = 0.1


def fimo_windows(motifs: list[Motif], sequences: list[str], folder) -> pd.DataFrame:
    """Every window FIMO scores, on each strand, with its score and p-value, under a uniform
    background and FIMO's default pseudocount."""
    (folder / "motifs.meme").write_text(write_meme(motifs))
    (folder / "sequences.fa").write_text("".join(f">s{index}\n{sequence}\n"
                                                 for index, sequence in enumerate(sequences)))
    completed = subprocess.run(
        [str(tool("fimo")), "--text", "--thresh", "1", "--bfile", "--uniform--", "--verbosity", "1",
         str(folder / "motifs.meme"), str(folder / "sequences.fa")],
        capture_output=True, text=True, check=True)
    lines = [line.split("\t") for line in completed.stdout.splitlines()
             if line and not line.startswith("#")]
    frame = pd.DataFrame(lines[1:], columns=lines[0])
    return frame.rename(columns={"p-value": "p_value"}).astype(
        {"start": int, "score": float, "p_value": float})


@pytest.mark.crossenv
@pytest.mark.skipif(missing_python(MEME_ENVIRONMENT) is not None,
                    reason="the MEME Suite's environment is not built here")
def test_the_scan_agrees_with_fimo(tmp_path):
    rng = np.random.default_rng(0)
    motifs = [random_motif(rng, width, f"M{width}") for width in (6, 8, 10, 12, 15)]
    sequences = [random_sequence(rng, 80) for _ in range(30)]
    settings = ScanSettings(strands="both", p_threshold=1e-3, summaries=("best_match", "count"),
                            normalise="counts")

    windows = fimo_windows(motifs, sequences, tmp_path)
    found = Scanner(motifs, settings).scan(sequences)

    complement = str.maketrans("ACGT", "TGCA")
    for index, motif in enumerate(motifs):
        matrix, bits = integer_matrix(motif), scanning.log_odds(motif)
        low, distribution = score_distribution(matrix)
        tail = upper_tail(distribution)
        positions = np.arange(motif.width)
        for row, sequence in enumerate(sequences):
            theirs = windows[(windows["motif_id"] == motif.identifier)
                             & (windows["sequence_name"] == f"s{row}")]
            assert len(theirs) == 2 * (len(sequence) - motif.width + 1)
            for window in theirs.itertuples():
                letters = sequence[window.start - 1: window.start - 1 + motif.width]
                if window.strand == "-":
                    letters = letters.translate(complement)[::-1]
                codes = scanning.encode(letters)
                p_value = tail[matrix[positions, codes].sum() - low]
                assert bits[positions, codes].sum() == pytest.approx(window.score,
                                                                     abs=FIMO_SCORE_TOLERANCE)
                assert abs(np.log10(p_value) - np.log10(window.p_value)) < FIMO_P_TOLERANCE
            best = theirs.groupby("start")["p_value"].min()
            assert found[row, 2 * index] == pytest.approx(-np.log10(best.min()),
                                                          abs=FIMO_P_TOLERANCE)
            assert abs(found[row, 2 * index + 1] - (best <= settings.p_threshold).sum()) <= 1


# ------------------------------------------------------------------------------------ the store


def test_the_store_keeps_a_scan_and_gives_it_back(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    scanner = Scanner([random_motif(rng, 5)], BOTH)
    sequences = [random_sequence(rng, 30) for _ in range(6)]
    store = ScanStore(tmp_path)
    first = scan_with_store(scanner, sequences, store)

    # A second store over the same directory stands for another process.
    monkeypatch.setattr(Scanner, "scan", lambda self, sequences: pytest.fail("scanned again"))
    again = scan_with_store(scanner, list(reversed(sequences)), ScanStore(tmp_path))

    assert np.array_equal(again, first[::-1])
    assert len(list((tmp_path / SCANS_DIR / scanner.digest()).glob("chunk-*"))) == 1


def test_the_store_scans_only_what_it_lacks(tmp_path):
    rng = np.random.default_rng(6)
    scanner = Scanner([random_motif(rng, 5)], BOTH)
    store = ScanStore(tmp_path)
    old = [random_sequence(rng, 30) for _ in range(3)]
    new = [random_sequence(rng, 30) for _ in range(2)]
    scan_with_store(scanner, old, store)

    mixed = scan_with_store(scanner, [new[0], old[1], new[1], new[0]], store)

    assert np.array_equal(mixed, scanner.scan([new[0], old[1], new[1], new[0]]))
    chunks = sorted((tmp_path / SCANS_DIR / scanner.digest()).glob("chunk-*"))
    assert len(chunks) == 2
    assert sorted(len(np.load(chunk / "digests.npy")) for chunk in chunks) == [2, 3]


def test_a_store_without_its_directory_is_not_used(tmp_path):
    rng = np.random.default_rng(8)
    scanner = Scanner([random_motif(rng, 5)], BOTH)
    gone = tmp_path / "moved"

    values = scan_with_store(scanner, ["ACGTACGTAC"], ScanStore(gone))

    assert np.array_equal(values, scanner.scan(["ACGTACGTAC"]))
    assert not gone.exists()


def test_a_sequence_digest_is_its_letters():
    assert sequence_digest("ACGT") == sequence_digest("ACGT")
    assert sequence_digest("ACGT") != sequence_digest("acgt")
    assert len(sequence_digest("ACGT")) == 32


# ------------------------------------------------------------------------------ the featuriser


def collection_of(*motifs: Motif) -> Collection:
    return Collection("fake_collection", tuple(motifs), digest="abc")


def test_known_motif_features_are_named_described_and_recorded():
    motifs = read_meme(MEME_DNA)
    featuriser = KnownMotifFeaturiser(collection_of(*motifs), LOOSE, min_match_share=0)

    featuriser.fit(pd.Series(["AGTAGTCCAGT", "CCCCCCCCCCC"]))

    names = featuriser.feature_names
    assert names[:2] == ["motif_MA0001_1_FAKE1_best", "motif_MA0001_1_FAKE1_count"]
    assert names[2] == "motif_MA0002_1_best"
    record = featuriser.records[1]
    assert record.kind == KIND_MOTIF
    assert record.params == {"source": MOTIF_KNOWN, "summary": "count", "strands": "both",
                             "p_threshold": 0.05, "collection": "fake_collection",
                             "motif_id": "MA0001.1", "motif_name": "FAKE1"}
    assert "MA0001.1 (FAKE1)" in record.description
    assert "on either strand" in record.description
    assert featuriser.records[0].params["summary"] == "best_match"
    assert "p_threshold" not in featuriser.records[0].params


def test_a_count_never_seen_in_the_fitting_rows_is_dropped():
    """The motif AGT matches none of the fitting rows, so its count gets no column, whereas its best
    match keeps one."""
    motif = read_meme(MEME_DNA)[0]
    featuriser = KnownMotifFeaturiser(collection_of(motif), LOOSE, min_match_share=0)

    featuriser.fit(pd.Series(["CCCCCCCCCC", "GGGGGGGGGG"]))

    assert featuriser.feature_names == ["motif_MA0001_1_FAKE1_best"]
    assert featuriser.describe()["count_columns_dropped"] == 1
    assert featuriser.transform(pd.Series(["AGTAGT"])).shape == (1, 1)


def test_a_count_is_kept_only_where_enough_fitting_rows_hold_a_match():
    """GGGA matches one sequence in a hundred. A share of 0.02 drops its count, whereas a share of 0
    keeps it, since any count that matches at all is then kept."""
    motif = Motif("GGGA", None, np.eye(4)[[2, 2, 2, 0]] * 0.97 + 0.0075, sites=100)
    rows = pd.Series(["GGGATTTT"] + ["TTTTTTTT"] * 99)

    strict = KnownMotifFeaturiser(collection_of(motif), LOOSE, min_match_share=0.02).fit(rows)
    loose = KnownMotifFeaturiser(collection_of(motif), LOOSE, min_match_share=0).fit(rows)

    assert strict.feature_names == ["motif_GGGA_best"]
    assert loose.feature_names == ["motif_GGGA_best", "motif_GGGA_count"]
    with pytest.raises(ValueError, match="min_match_share"):
        KnownMotifFeaturiser(collection_of(motif), LOOSE, min_match_share=1)


def test_a_motif_of_several_proteins_is_named_after_the_first():
    motif = Motif("M7", "PCBP1/PCBP2", np.eye(4)[[1, 1, 3]] * 0.97 + 0.0075, sites=20)

    featuriser = KnownMotifFeaturiser(collection_of(motif), LOOSE, min_match_share=0).fit(
        pd.Series(["CCTCCT"]))

    assert featuriser.feature_names[0] == "motif_M7_PCBP1_best"
    assert "(PCBP1/PCBP2)" in featuriser.records[0].description


def test_the_featuriser_transforms_new_rows_after_a_pickle(tmp_path):
    rng = np.random.default_rng(9)
    motifs = [random_motif(rng, width, f"M{width}") for width in (4, 6)]
    training = pd.Series([random_sequence(rng, 50) for _ in range(20)])
    unseen = pd.Series([random_sequence(rng, 70) for _ in range(5)])
    featuriser = KnownMotifFeaturiser(collection_of(*motifs), BOTH, min_match_share=0,
                                      store=ScanStore(tmp_path))
    featuriser.fit(training)

    loaded = pickle.loads(pickle.dumps(featuriser))

    assert np.array_equal(loaded.transform(unseen).to_numpy(), featuriser.transform(unseen).to_numpy())
    assert loaded.store.root == tmp_path


def test_the_featuriser_gives_the_same_values_with_and_without_a_store(tmp_path):
    rng = np.random.default_rng(10)
    motifs = [random_motif(rng, width, f"M{width}") for width in (3, 5, 7)]
    rows = pd.Series([random_sequence(rng, int(length)) for length in rng.integers(5, 80, size=30)])

    collection = collection_of(*motifs)
    stored = KnownMotifFeaturiser(collection, BOTH, min_match_share=0,
                                  store=ScanStore(tmp_path)).fit_transform(rows)
    plain = KnownMotifFeaturiser(collection, BOTH, min_match_share=0).fit_transform(rows)

    pd.testing.assert_frame_equal(stored, plain)


def test_a_name_part_holds_only_letters_digits_and_single_underscores():
    assert name_part("MA0139.2") == "MA0139_2"
    assert name_part("Ahr::Arnt") == "Ahr_Arnt"
    assert name_part("ZNF(var.2)") == "ZNF_var_2"
