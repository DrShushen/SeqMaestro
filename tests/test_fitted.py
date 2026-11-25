"""Saving a fitted arm, and loading it back to predict on sequences the run never saw.

A cell saves its fitted arm, so a question the run did not answer, including one about new
sequences, needs no refit. The restored arm must predict identically, because an arm that reloads
into something slightly different is worse than one that does not reload at all.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from seqmaestro.dataset import Columns
from seqmaestro.features import make_featuriser
from seqmaestro.harness.fitted import (
    FILENAME,
    FittedArm,
    load_featuriser,
    load_fitted,
    save_featuriser,
    save_fitted,
)
from seqmaestro.llm import ScriptedClient, TaskContext
from seqmaestro.models import MODELS, make_model
from seqmaestro.proposer import GeneratedFeaturiser, ProposalSettings, Proposer
from seqmaestro.target import BINARY
from tests.planted import run_cell

SEQUENCE, TARGET = "window", "outcome"
COLUMNS = Columns(SEQUENCE, TARGET, weight=None, covariates=(), target_kind=BINARY)

GC_CODE = (
    "def gc_fraction(X):\n"
    "    s = X['window'].astype(str)\n"
    "    return (s.str.count('G') + s.str.count('C')) / s.str.len()"
)
A_CODE = "def a_count(X):\n    return X['window'].str.count('A')"


def sequences(n=60, length=24, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(["".join(rng.choice(list("ACGT"), length)) for _ in range(n)])


def labelled(n=60, seed=0):
    seqs = sequences(n, seed=seed)
    # A planted signal, so the models have something to fit rather than noise.
    y = np.array([1 if s.count("G") > s.count("A") else 0 for s in seqs], dtype=int)
    return pd.DataFrame({SEQUENCE: seqs, TARGET: y})


def proposer_of(*features):
    reply = json.dumps([
        {"name": name, "description": "d", "rationale": "r", "code": code}
        for name, code in features
    ])
    return Proposer(
        ScriptedClient(reply), TaskContext(task="Predicting a thing."),
        settings=ProposalSettings(n_rounds=1, n_per_round=len(features), n_examples=4),
        sequence_column=SEQUENCE, target_column=TARGET, target_kind="binary",
    )


# --------------------------------------------------------------------------------------------
# The featurisers, each of which has to survive a round trip
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pool", ["positional", "kmer", "positional+kmer"])
def test_a_fixed_vocabulary_arm_reloads_and_predicts_identically(pool, tmp_path):
    frame = labelled()
    featuriser = make_featuriser(pool, k_range=(1, 3), kmer_normalise="counts")
    X = featuriser.fit_transform(frame[SEQUENCE], frame[TARGET].to_numpy())
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(X, frame[TARGET].to_numpy())

    save_fitted(tmp_path, featuriser, model, sequence_column=SEQUENCE)
    unseen = sequences(n=15, seed=99)
    restored = load_fitted(tmp_path)

    assert np.allclose(restored.predict(unseen),
                       model.predict(featuriser.transform(unseen)))


def test_a_generated_arm_reloads_and_predicts_identically(tmp_path):
    """A generated featuriser holds compiled functions, which pickle cannot serialise, so it
    recompiles them from the source each record holds."""
    frame = labelled()
    featuriser = GeneratedFeaturiser(proposer_of(("gc_fraction", GC_CODE), ("a_count", A_CODE)))
    X = featuriser.fit_transform(frame[SEQUENCE], frame[TARGET].to_numpy())
    model = make_model("ebm", target_kind="binary", seed=0).fit(X, frame[TARGET].to_numpy())

    save_fitted(tmp_path, featuriser, model, sequence_column=SEQUENCE)
    unseen = sequences(n=15, seed=99)
    restored = load_fitted(tmp_path)

    assert [r.name for r in restored.featuriser.records] == [r.name for r in featuriser.records]
    assert np.allclose(restored.predict(unseen),
                       model.predict(featuriser.transform(unseen)))


def test_a_restored_arm_carries_no_language_model_client(tmp_path):
    """A saved arm must not be able to issue API calls, and the proposer is needed only at fit
    time."""
    frame = labelled()
    featuriser = GeneratedFeaturiser(proposer_of(("gc_fraction", GC_CODE)))
    featuriser.fit(frame[SEQUENCE], frame[TARGET].to_numpy())
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(
        featuriser.transform(frame[SEQUENCE]), frame[TARGET].to_numpy())

    save_fitted(tmp_path, featuriser, model)
    assert load_fitted(tmp_path).featuriser.proposer is None


def test_the_restored_columns_are_the_ones_the_model_was_fitted_on(tmp_path):
    """Rebuilding through `fit` would re-run the validation that drops constant features, so a
    feature that happened to be constant on the rows at hand would vanish and leave the model short
    of a column. Recompiling from source keeps every column."""
    frame = labelled()
    featuriser = GeneratedFeaturiser(proposer_of(("gc_fraction", GC_CODE), ("a_count", A_CODE)))
    featuriser.fit(frame[SEQUENCE], frame[TARGET].to_numpy())
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(
        featuriser.transform(frame[SEQUENCE]), frame[TARGET].to_numpy())
    save_fitted(tmp_path, featuriser, model)

    # One sequence, so every feature is constant on it. A rebuild that re-validated would drop
    # both and the model would be handed an empty frame.
    restored = load_fitted(tmp_path)
    assert list(restored.transform(pd.Series(["ACGTACGTACGTACGTACGTACGT"])).columns) == \
           [r.name for r in featuriser.records]


def test_an_iterative_arm_reloads_and_predicts_identically(tmp_path):
    """The iterative source's featuriser holds a model factory and the rows it scored its fits on.
    Transforming needs neither, and neither pickles reliably, so only the fitted parts are saved."""
    from seqmaestro.proposer import IterativeGeneratedFeaturiser

    frame = labelled()
    y = frame[TARGET].to_numpy()
    featuriser = IterativeGeneratedFeaturiser(
        proposer_of(("gc_fraction", GC_CODE), ("a_count", A_CODE)),
        lambda: make_model("l1_logistic", target_kind="binary", seed=0),
        validation=(frame[SEQUENCE], y),
        rounds=1,
        sequence_column=SEQUENCE,
    )
    X = featuriser.fit_transform(frame[SEQUENCE], y)
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(X, y)

    save_fitted(tmp_path, featuriser, model, sequence_column=SEQUENCE)
    unseen = sequences(n=15, seed=99)
    restored = load_fitted(tmp_path)

    assert restored.featuriser.model_factory is None
    assert restored.featuriser.validation is None
    assert np.allclose(restored.predict(unseen),
                       model.predict(featuriser.transform(unseen)))


# --------------------------------------------------------------------------------------------
# Through the harness, which is how a real cell gets one
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODELS["binary"]))
def test_every_model_run_through_the_harness_leaves_a_loadable_arm(model_name, tmp_path):
    frame = labelled(n=70)
    train, val = frame.iloc[:50], frame.iloc[50:]
    out = tmp_path / model_name
    run_cell(
        make_model(model_name, target_kind="binary", seed=0), make_featuriser("positional"),
        train, val, columns=COLUMNS,
        seed=0, out_dir=out, n_bootstrap=0, permutation_repeats=0,
        arm=(model_name, "positional"),
    )

    restored = load_fitted(out)
    probabilities = restored.predict(sequences(n=12, seed=7))
    assert probabilities.shape == (12,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_the_saved_arm_agrees_with_the_predictions_the_cell_recorded(tmp_path):
    """The restored arm must reproduce the cell's validation predictions exactly, which shows that
    the file holds the model that produced the numbers instead of something similar. The reveal
    predicts the held-out split with this file alone."""
    frame = labelled(n=70)
    train, val = frame.iloc[:50], frame.iloc[50:]
    out = tmp_path / "ebm"
    run_cell(
        make_model("ebm", target_kind="binary", seed=0), make_featuriser("positional"),
        train, val, columns=COLUMNS,
        seed=0, out_dir=out, n_bootstrap=0, permutation_repeats=0,
        arm=("ebm", "positional"),
    )

    recorded = np.load(out / "predictions.npz")["val_score"]
    assert np.array_equal(load_fitted(out).predict(val[SEQUENCE]), recorded)


def test_a_featuriser_saved_on_its_own_reloads_and_transforms_identically(tmp_path):
    """A featuriser saved alone supplies the columns handed to an arm in another environment, and
    the reveal computes them again from it."""
    frame = labelled()
    featuriser = make_featuriser("kmer", k_range=(1, 2), kmer_normalise="counts")
    featuriser.fit(frame[SEQUENCE])
    save_featuriser(tmp_path / "supplied.pkl", featuriser)

    unseen = sequences(n=15, seed=99)
    assert load_featuriser(tmp_path / "supplied.pkl").transform(unseen).equals(
        featuriser.transform(unseen))


# --------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------


def test_an_unfitted_featuriser_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unfitted"):
        save_fitted(tmp_path, make_featuriser("positional"), make_model("l1_logistic", target_kind="binary"))


def test_a_missing_arm_says_which_cells_have_none(tmp_path):
    with pytest.raises(FileNotFoundError, match="A DEFT or an IGSR cell has none"):
        load_fitted(tmp_path)


def test_a_version_this_reader_does_not_know_is_refused(tmp_path):
    import pickle

    frame = labelled()
    featuriser = make_featuriser("positional")
    X = featuriser.fit_transform(frame[SEQUENCE])
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(X, frame[TARGET].to_numpy())
    save_fitted(tmp_path, featuriser, model)

    payload = pickle.loads((tmp_path / FILENAME).read_bytes())
    payload["version"] = 99
    (tmp_path / FILENAME).write_bytes(pickle.dumps(payload))

    with pytest.raises(ValueError, match="version 99"):
        load_fitted(tmp_path)


def test_the_arm_describes_both_halves():
    frame = labelled()
    featuriser = make_featuriser("positional")
    X = featuriser.fit_transform(frame[SEQUENCE])
    model = make_model("l1_logistic", target_kind="binary", seed=0).fit(X, frame[TARGET].to_numpy())

    described = FittedArm(featuriser, model, SEQUENCE).describe()
    assert described["featuriser"]["kind"] == "positional"
    assert described["model_settings"]["model"] == "l1_logistic"
    assert described["sequence_column"] == SEQUENCE
