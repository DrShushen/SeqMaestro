"""One fit of one model on one pool of features, written as a cell.

Everything except the model itself lives here, so that every arm gets the same subsample,
featurisation discipline, threshold selection, metrics and files. A model implements the `Model`
protocol in `seqmaestro.models.base`, and this module does the rest.

The cell is the interface to everything downstream, and to the arms that cannot import this
package at all. `seqmaestro.harness.cell` names its files and writes them. A cell fitted in this
process also keeps `fitted.pkl`, the fitted featuriser and model. A cell that arrived as a bundle
keeps the model its method saved.

**No setting of the grid has a default here.** The seed, the share of training rows a seed fits on,
the bootstrap count and how a control's labels are drawn are what make two cells comparable, so
every caller states them, and a run states them once, in its config.

**Nothing here reads the test split.** A cell is fitted and scored on validation. The test split
is opened once, by the reveal (see `seqmaestro.harness.heldout`), which predicts with the saved
model and adds a `test` block to `metrics.json` and the test arrays to `predictions.npz`. The
reveal scores with a binary target's threshold chosen here and the seed and bootstrap count
recorded in `run.json`, so its numbers are the ones scoring inline would have produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from seqmaestro import __version__
from seqmaestro.arms import Arm, arm_name
from seqmaestro.features import parse_pool
from seqmaestro.features.base import Featuriser, as_inputs
from seqmaestro.harness import cell as cells
from seqmaestro.harness.bundle import read_bundle
from seqmaestro.harness.fitted import PREDICT_WITH as FITTED
from seqmaestro.harness.fitted import save_fitted
from seqmaestro.harness.importance import describe as describe_permutation
from seqmaestro.harness.importance import permutation_importance
from seqmaestro.models.base import FILLED_ROWS, Model, filled_rows
from seqmaestro.provenance import git_sha
from seqmaestro.target import target_for

if TYPE_CHECKING:
    from seqmaestro.dataset import Columns

REPO = Path(__file__).resolve().parents[2]

#: Where features are evaluated for stability matching. Matching uses no labels, and validation is
#: already the split used for model selection, so evaluating features there exposes nothing new.
REFERENCE_SPLIT = cells.VALIDATION

#: Where `marginal_effect` is measured. It answers a different question from `REFERENCE_SPLIT`, so
#: it uses a different split. It asks what a feature does in the data the model learned from, so it
#: is measured there: on **every** training row, against the **real** labels.
#:
#: Validation is too small to carry the sign of a small effect reliably, so a disagreement with
#: `direction` there would describe the estimator instead of the feature. The training split is
#: usually several times larger. `docs/analyses/direction.md` explains the reasoning.
#:
#: The *fitted* rows are not used either, although they are what the model saw. `train_fraction`
#: subsamples them per seed, and a control arm's fitted rows carry the shuffled labels. Measuring
#: against those would make `marginal_effect` agree with `direction` under shuffled labels for the
#: same reason it does under real ones, and the permuted control for that comparison would stop
#: working. Real labels on every training row give one number per feature per dataset, and keep
#: the null.
#:
#: **This is in-sample.** It measures the association the model was fitted on, which has not been
#: shown to generalise. `docs/analyses/direction.md` describes what follows from that.
EFFECT_SPLIT = "train"

#: Rows per block when evaluating features across a whole split. Only the used columns are kept, so
#: a large split never needs the whole vocabulary matrix in memory at once.
EVALUATION_CHUNK = 20_000


def evaluate_columns(featuriser: Featuriser, inputs, names: list[str],
                     chunk: int = EVALUATION_CHUNK) -> np.ndarray:
    """`names` evaluated over every row of `inputs`, in blocks. Takes what `Featuriser.transform`
    takes: `Inputs`, or the sequences alone.

    The featuriser produces its whole vocabulary, and the caller wants a handful of columns from it.
    On a large split the whole vocabulary matrix takes many times the memory of the kept columns,
    so the split is transformed in blocks instead of in one `transform`.
    """
    inputs = as_inputs(inputs)
    if not names:
        return np.zeros((len(inputs), 0))
    blocks = [
        featuriser.transform(inputs.take(slice(start, start + chunk)))[names].to_numpy(dtype=float)
        for start in range(0, len(inputs), chunk)
    ]
    return np.vstack(blocks) if blocks else np.zeros((0, len(names)))


def run_model(
    model: Model,
    featuriser: Featuriser,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    # No default, here or anywhere else in this package. Which column holds the sequence, the
    # label, the weight and the covariates are facts about a dataset, recorded in its
    # `dataset.json`, and a default naming one dataset's columns would be applied silently to the
    # next dataset's frames.
    columns: Columns,
    seed: int,
    train_fraction: float | None,
    permute_labels: bool,
    permutation_scope: str,
    n_bootstrap: int,
    permutation_repeats: int,
    out_dir: str | Path | None = None,
    reports: Callable[[], Mapping[str, Any]] | None = None,
    arm: Arm | None = None,
    **manifest: Any,
) -> dict:
    """Fit, score on validation, extract features, and write the cell. Returns the manifest.

    `arm` is the model's name and the pool's name the cell reports under, which `run.json` records
    as `model` and `pool`. It is given apart from `model`, the object that is fitted, because the
    composition baseline reports under a name of its own.

    There is no test frame to pass. The test split is scored once, at the reveal, from the model
    this function saves, which is why the model is saved whenever `out_dir` is given.

    `columns` says which column of `train` and `val` is which (`seqmaestro.dataset.Columns`). The
    featuriser is handed `columns.inputs` of each frame, so of a frame's columns it reads the
    sequences and the covariates `columns` names, and never the label or the weight. Its `fit` is
    handed the labels this seed trains against and the fitting weights as arguments.

    `train_fraction` draws a stratified subsample of the training rows for this seed. This matters:
    most of the models here are deterministic given their data, so with the full training set
    every seed would produce identical output and stability would be 1 by construction, measuring
    nothing. The perturbation has to be shared by every arm for the stability axis to compare
    anything. Pass None to fit on all of the training rows and give up that axis.

    `permute_labels` shuffles the training labels. This is the control arm, and the source of the
    judge's negative examples. The validation labels it is scored against are never touched, so a
    permuted cell should score at chance, and if it does not, something leaks. An arm that also
    chooses on the validation rows chooses against `labels_to_choose_on`.

    `permutation_scope` decides how many independent nulls a run has. `"seed"` gives every arm in a
    seed the same shuffled labels, so a run with 9 arms and 3 seeds has 3 draws, and one lucky draw
    shows up as an offset in all 9 arms. `"cell"` derives the draw from the arm as well, giving 27
    draws, which is what lets the null be quoted with an interval. See `permutation_rng`.

    **Give the same seed to the model and to this call.** They are separate arguments because this
    function does not construct models. Passing the seed to only one of them is an easy mistake
    that goes unnoticed: an L1 fit is deterministic given its data, so a model seed on its own
    changes nothing. Both seeds are written to `run.json` (the cell's under `seed`, the model's
    inside `model_settings`), so a mismatch is at least visible afterwards.

    `reports` gathers further JSON files of the cell, by file name, once the model is fitted and
    every row is transformed: what a proposal cost, for an arm whose features a language model
    wrote. They are written with the cell, before its marker.
    """
    target_column, weight_column = columns.target, columns.weight
    target = target_for(columns.target_kind)
    fit_rows, y_fit = subsample_training_rows(
        train, seed=seed, train_fraction=train_fraction,
        columns=columns, permute_labels=permute_labels,
        permutation_key=permutation_key(permutation_scope, arm),
    )

    # Per-row fitting weights, if the split carries them. They are taken from `fit_rows`, not from
    # `train`, so the seed's subsample selects rows and their weights together. Reading them from
    # the unsubsampled frame would misalign them silently, and every arm would fit noise.
    #
    # **Weights are not recomputed under `permute_labels`.** A weight is a property of the sampling
    # design, not of the label a row happens to carry, so the control arm gets the same weights with
    # shuffled labels. Recomputing them from permuted labels would give the control a different
    # design from the arm it controls for.
    w_fit = None
    if weight_column is not None:
        if weight_column not in fit_rows.columns:
            raise ValueError(
                f"The weight column {weight_column!r} is not in the split; columns are "
                f"{sorted(fit_rows.columns)}. A missing weight column would otherwise fit unweighted "
                f"and report nothing about it."
            )
        w_fit = fit_rows[weight_column].to_numpy(dtype=float)

    # The featuriser is fitted on the rows the model trains on, not on all of `train`, because it is
    # part of the pipeline and a vocabulary learned from rows the model never saw is a leak. It gets
    # the labels this seed trains against, which are permuted for a control arm. Most featurisers
    # ignore labels, but a language-model proposer shown real labels while the model fits permuted
    # ones would give the control arm real features, and there would be no null.
    X_fit = featuriser.fit_transform(columns.inputs(fit_rows), y_fit, sample_weight=w_fit)
    X_val = featuriser.transform(columns.inputs(val))

    model.fit(X_fit, y_fit, sample_weight=w_fit)
    p_val = np.asarray(model.predict(X_val), dtype=float)

    importances = model.importances()
    used = [record for record in featuriser.records if record.name in importances]
    # One importance computed the same way for every arm, beside each model's. It is measured
    # on the same rows the behavioural matcher uses, so the two see identical data. It costs one
    # prediction pass per feature per repeat, which on a vocabulary-sized pool is the expensive part
    # of a cheap arm, and `permutation_repeats=0` turns it off.
    permuted_importances = permutation_importance(
        model, X_val, val[target_column].to_numpy(),
        columns=[record.name for record in used],
        repeats=permutation_repeats,
        rng=permutation_rng(seed, f"permutation-importance/{model.name}"),
    )
    # Which class the model uses each feature to argue for, or which way it moves a continuous
    # target, taken from the model's machinery on the same rows as every other cross-arm measure.
    # It is free for L1 and EBM, and costs one prediction pass for XGBoost and one Shapley
    # computation for PySR, so unlike the permutation importance it has no flag to turn it off. The
    # marginal counterpart is computed in `score_arm`, its only writer.
    directions = model.directions(X_val)
    records = [
        replace(record, importance=importances[record.name],
                permutation_importance=permuted_importances.get(record.name),
                direction=directions.get(record.name))
        for record in used
    ]
    names = [record.name for record in records]

    # Saved before the results are written, so a cell that cannot be saved fails outright instead
    # of leaving a record that looks complete but could never be revealed, since the reveal predicts
    # with this and nothing else. The featuriser is saved with the model because a model alone
    # cannot turn a sequence into the columns it was fitted on. DEFT's and IGSR's arms do not come
    # through here. Each saves its model into the bundle it hands back.
    if out_dir is not None:
        save_fitted(out_dir, featuriser, model, sequence_column=columns.sequence)

    return score_arm(
        y_val=val[target_column].to_numpy(),
        p_val=p_val,
        records=records,
        reference_values=X_val[names].to_numpy() if names else np.zeros((len(X_val), 0)),
        # Every training row and the real labels, whatever this cell was fitted on. See
        # `EFFECT_SPLIT`. The featuriser is already fitted, so this transforms rows it did not
        # learn from, which is what a held-out transform does everywhere else here.
        effect_values=evaluate_columns(featuriser, columns.inputs(train), names),
        y_effect=train[target_column].to_numpy(),
        importance_kind=model.importance_kind,
        direction_kind=model.direction_kind,
        target_kind=target.kind,
        out_dir=out_dir,
        seed=seed,
        n_bootstrap=n_bootstrap,
        filled=filled_rows(model, X_val),
        reports=None if reports is None else reports(),
        run={
            "labels": cells.PERMUTED if permute_labels else cells.REAL,
            "permutation_scope": permutation_scope,
            "train_fraction": train_fraction,
            "weight_column": weight_column,
            "n_fit_rows": len(fit_rows),
            "n_features_available": len(featuriser.records),
            "permutation_importance": describe_permutation(permutation_repeats, "val",
                                                           target.headline),
            "featuriser": featuriser.describe(),
            "model_settings": model.describe(),
            # How the reveal predicts the cell: with the featuriser and model saved above.
            "predict_with": FITTED,
            **manifest,
            **reported_under(arm),
        },
    )


def reported_under(arm: Arm | None) -> dict[str, Any]:
    """What a cell's `run.json` says of the arm it reports under: the model's name, the pool's
    name, and the pool's sources (`pool_sources`). A cell fitted outside any arm records only that
    it has no sources to name."""
    if arm is None:
        return {"sources": None}
    model, pool = arm
    return {"model": model, "pool": pool, "sources": pool_sources(pool)}


def pool_sources(pool: str | None) -> list[str] | None:
    """The sources a pool holds, as a list instead of the pool's name.

    Recorded so that "were these two arms given the same features?" can be answered from what a
    cell records instead of from the name of its directory. A name is only a label, and a label can
    change meaning between runs, so an analysis comparing arms by name could compare two different
    inputs.

    Returns `None` for a name that is not a pool's, which is allowed: the composition baseline
    reports under one, and a reader can fall back to the name.
    """
    if not pool:
        return None
    try:
        return list(parse_pool(pool))
    except ValueError:
        return None


PERMUTATION_SCOPES = ("seed", "cell")


def permutation_key(scope: str, arm: Arm | None) -> str | None:
    """The `permutation_rng` key implied by a scope, or None for one draw per seed.

    The key is the cell's arm, so a scope of `cell` cannot be asked for without naming one. A
    missing arm falling back to a shared draw would leave every arm on one draw while the run
    claimed otherwise, and the run would look fine.
    """
    if scope not in PERMUTATION_SCOPES:
        raise ValueError(f"Unknown permutation_scope {scope!r}; expected one of {PERMUTATION_SCOPES}.")
    if scope == "seed":
        return None
    if arm is None or not all(arm):
        raise ValueError(
            "permutation_scope='cell' needs the cell's arm, a model's name and a pool's, to tell "
            "the cells apart; without it every arm would share one draw while claiming not to."
        )
    return arm_name(*arm)


def permutation_rng(seed: int, key: str | None = None) -> np.random.Generator:
    """The generator that shuffles a control arm's labels.

    With `key` None, every arm sharing a seed draws the **same** permutation. That compares all arms
    under one identical null, but it has a cost that is easy to miss: the null's *level* is then
    estimated from as many draws as there are seeds, not as many as there are cells. A run with 9
    arms and 3 seeds has 3 draws, not 27, so a permutation that happens to land a little above
    chance lands there in all 9 arms at once. The permuted scores then separate by seed instead of
    by arm, which shows as a systematic offset and can look like a leak.

    Passing a `key` that identifies the cell makes each draw independent, so the null level can be
    averaged down and quoted with an interval. The generator is seeded through `sha256` because
    `hash()` is salted per process, so the same run would not reproduce.
    """
    if key is None:
        return np.random.default_rng(seed)
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def subsample_training_rows(
    train: pd.DataFrame,
    *,
    seed: int,
    columns: Columns,
    train_fraction: float | None,
    permute_labels: bool,
    permutation_key: str | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """The rows this seed trains on, and the labels to train against.

    This is a shared function because it defines the experimental protocol. Most of the models here
    are deterministic given their data, so without a common perturbation every seed would produce
    identical output and the stability axis would compare nothing. An arm running in another
    environment has to train on exactly these rows, so it is handed the frame this returns.
    Reproducing the subsample from a seed on the other side would implement the same rule twice,
    and the two copies would eventually drift apart.

    The subsample is stratified on what the target's kind keeps in proportion
    (`seqmaestro.target`): the classes of a binary target, and the quantile bins of a continuous
    one, so that every seed fits on the target's whole range.

    `permutation_key` is the run's scope of the shuffle for this arm (`permutation_key`), and has
    no default: a caller that shuffled without it would draw once per seed under a run that asks
    for a draw per cell. A call that shuffles nothing passes None.
    """
    target_column = columns.target
    fit_rows = train
    if train_fraction is not None:
        strata = target_for(columns.target_kind).strata(train[target_column].to_numpy())
        fit_rows, _ = train_test_split(
            train, train_size=train_fraction, stratify=strata, random_state=seed
        )

    y_fit = fit_rows[target_column].to_numpy()
    if permute_labels:
        y_fit = permutation_rng(seed, permutation_key).permutation(y_fit)
    return fit_rows, y_fit


def labels_to_choose_on(val: pd.DataFrame, *, seed: int, columns: Columns, permute_labels: bool,
                        permutation_key: str | None) -> np.ndarray:
    """The labels an arm chooses against on the validation rows: the real ones, or on a control arm
    the real ones shuffled.

    Every cell is scored against the real validation labels. An arm that also chooses on the
    validation rows (a round's features, the iteration a search keeps, the terms it prunes) would
    otherwise choose against the real target on a control arm too, and the control would carry
    real signal. A control arm therefore chooses against the validation labels shuffled, drawn as
    its training labels are (`permutation_rng` under the same seed and key), so the draw follows the
    run's `permutation_scope`.

    Nothing here has a default: a forgotten `permute_labels` would hand a control the real labels,
    and nothing would fail.
    """
    y_val = val[columns.target].to_numpy()
    if not permute_labels:
        return y_val
    return permutation_rng(seed, permutation_key).permutation(y_val)


def score_arm(
    *,
    y_val: np.ndarray,
    p_val: np.ndarray,
    records: list,
    reference_values: np.ndarray,
    effect_values: np.ndarray,
    y_effect: np.ndarray,
    importance_kind: str | None,
    run: dict,
    target_kind: str,
    seed: int,
    n_bootstrap: int,
    direction_kind: str | None = None,
    out_dir: str | Path | None = None,
    filled: int | None = None,
    reports: Mapping[str, Any] | None = None,
) -> dict:
    """Everything after a prediction: metrics, a threshold where the target's kind has one, and the
    cell's files.

    Both paths into a Stage 2 cell end here: `run_model` for an in-process arm, and
    `ingest_bundle` for an arm that ran in another interpreter. With one writer the output
    structure is identical by construction, where two writers would have to be kept in step by
    hand. The analyses downstream read every arm the same way and cannot tell which path produced
    it.

    **This is the only writer of `marginal_effect`**, on either path: the association of the kind of
    target (`seqmaestro.target`), rank-biserial for a binary target and Spearman's rho for a
    continuous one. It needs no model, only the effect values and their labels, which every arm
    hands over here. Computing it in one place makes it mean the same thing for an arm fitted in
    this process and one that arrived from another environment. An adapter that computed its own
    would be a second definition.

    `effect_values` are the used features over the **training** rows, and `y_effect` their real
    labels. `reference_values` are the same features over the **validation** rows, and are what
    gets written for the behavioural matchers. The two splits do two jobs (see `EFFECT_SPLIT`).

    **Validation only.** A binary target's threshold is chosen here and written into the validation
    block, and `seed`, `n_bootstrap` and `target_kind` into `run.json`: those are everything the
    reveal needs to score the test split exactly as this would have.

    `filled` is how many validation rows took the model's constant prediction, for a model that
    can be undefined on a row (`seqmaestro.models.base.filled_rows`). It is written beside the
    validation block, and the reveal adds the test split's count. `reports` are further JSON files
    of the cell, by file name (`cell.write`).
    """
    target = target_for(target_kind)
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val, dtype=float)
    metrics = score_validation(y_val, p_val, target_kind=target_kind, seed=seed,
                               n_bootstrap=n_bootstrap)
    if filled is not None:
        metrics[FILLED_ROWS] = {REFERENCE_SPLIT: int(filled)}

    associations = target.association(effect_values, np.asarray(y_effect))
    records = [
        replace(record, marginal_effect=float(value) if np.isfinite(value) else None)
        for record, value in zip(records, associations, strict=True)
    ]

    run = {
        "seqmaestro_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
        "seed": seed,
        # With `seed`, the kind and a binary target's validation threshold, what the reveal scores
        # the test split with.
        "n_bootstrap": int(n_bootstrap),
        "target_kind": target_kind,
        "reference_split": REFERENCE_SPLIT,
        "importance_kind": importance_kind,
        "direction_kind": direction_kind,
        **run,
        "n_features_used": len(records),
        # Named explicitly, like the permutation importance, so that a change of statistic or split
        # shows up in the files it changed.
        "marginal_effect": {"statistic": target.association_kind, "split": EFFECT_SPLIT,
                            "labels": cells.REAL, "n_rows": int(np.asarray(y_effect).size)},
    }

    if out_dir is not None:
        header = {key: run.get(key) for key in ("importance_kind", "direction_kind", "target_kind",
                                                "seed", "labels", "model_settings", "featuriser")}
        cells.write(out_dir, run=run, metrics=metrics, y_val=y_val, scores=p_val,
                    features=cells.Features(records, reference_values, header), reports=reports)
        run["out_dir"] = str(out_dir)

    run["metrics"] = metrics
    return run


def score_validation(y_val, p_val, *, target_kind: str, seed: int, n_bootstrap: int) -> dict:
    """A cell's `metrics.json` before the reveal: the validation block, scored by the target's kind,
    and for a binary target the threshold chosen on validation that the block is scored at.

    Every cell (an arm's, the baseline's, a reduced model's) is scored through this, so all cells
    share one threshold rule and one interval method, and the reveal can score each cell's test
    split with the threshold stored here.
    """
    target = target_for(target_kind)
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val, dtype=float)
    threshold = target.threshold(y_val, p_val)
    block = target.metrics(y_val, p_val, threshold=threshold, n_bootstrap=n_bootstrap, seed=seed)
    if threshold is None:
        return {"val": block}
    return {"threshold_chosen_on": REFERENCE_SPLIT, "val": block}


def ingest_bundle(
    bundle_dir: str | Path,
    val: pd.DataFrame,
    *,
    target_column: str,
    target_kind: str,
    seed: int,
    n_bootstrap: int,
    out_dir: str | Path | None = None,
    arm: Arm | None = None,
    **manifest: Any,
) -> dict:
    """Score an arm that ran in another environment, into a cell. `arm` is the model's name and
    the pool's name the cell reports under (`reported_under`).

    The validation frame is passed so that the bundle's labels can be checked against it. Without
    that check, a foreign adapter that returns rows in a different order would produce a number that
    looks believable and means nothing.
    """
    bundle = read_bundle(bundle_dir, target_kind, y_val=val[target_column].to_numpy())

    return score_arm(
        y_val=bundle.y_val,
        p_val=bundle.p_val,
        records=bundle.records,
        reference_values=bundle.feature_values,
        effect_values=bundle.effect_values,
        y_effect=bundle.y_effect,
        importance_kind=bundle.importance_kind,
        direction_kind=bundle.direction_kind,
        target_kind=target_kind,
        out_dir=out_dir,
        seed=seed,
        n_bootstrap=n_bootstrap,
        run={**{k: v for k, v in bundle.arm.items() if k != "bundle_version"}, **manifest,
             **reported_under(arm)},
    )

