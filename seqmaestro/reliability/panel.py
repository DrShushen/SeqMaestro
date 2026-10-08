"""One arm's reliability panel: what it computes before the reveal, at it, and after it.

    reliability/<model>/<pool>/
        pool.npz                    a DEFT or an IGSR arm's candidates over D, as its method
                                    evaluated them
        prepare.json, prepare.npz   Step 0: the candidates, the prefilter, the clusters, the
                                    half-samples, the fitting rows and the label sets on D; the
                                    column and reproduction checks
        reproduction.json           a DEFT arm alone: what its side found in the reproduction checks
        planted.json                the check arm's planted-signal recipe
        tier2_discovery[_planted]   Tier 2 in-sample, on D (supplementary, see below)
        loco/<label>/               M* without one feature, or one cluster: standard cells
        selections[_<labels>].npz   Tier 1's selections, resumable
        tier1[_<labels>]            pi-hat and the bound, at feature, cluster and position level
        reliability.json            the arm's record, written last: what the reveal seals
      at the reveal
        tier2_test[_null|_planted]  Tier 2 on the test split
      after it
        tier3.json, checks.json, profile_features.csv, profile_clusters.csv, profile_curves.csv

(`tier1` and `tier2` stand for a `.json` summary and a `.npz` of per-feature arrays.)

**Before the reveal the panel uses only D = train + val.** Step 0 never looks at a label, and
Tier 1 resamples D. The LOCO refits are fitted on the arm's fitting rows and, like every cell,
scored on validation only, so that the reveal can score them. The test split reaches the panel only
as the rows the reveal hands to `reveal_tier2`, and after the reveal the panel reads revealed
predictions.

**Each part before the reveal and at it writes its files and is skipped when they exist**, so an
interrupted panel resumes. The part after the reveal is computed again on every call. Tier 1, the
panel's main cost, runs last before the reveal, so every cheaper part has already had the chance to
fail.

**The in-sample Tier 2** is the same statistic and null computed on D. Every feature in it was
selected by a model fitted on those rows, or proposed by a language model shown examples from them,
so it is anticonservative by an amount nothing here estimates. It gives an upper bound on power, so
a reader can tell "no association" from "not enough rows". The test split's number is the one that
counts.

**Seeds.** Every random draw the panel makes is seeded from the configured seed, and a model's fit
keeps its arm's seed. The constants below are the offsets added for each kind of draw, and the
correlation subsample of Step 0 takes the seed itself, so each draw can be reproduced alone.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from seqmaestro.features.base import Inputs
from seqmaestro.harness import cell as cells
from seqmaestro.harness.heldout import TEST_ARRAYS, TEST_SPLIT, SealError, reveal_record
from seqmaestro.models.base import FILLED_ROWS
from seqmaestro.reduction import check, eliminate, outputs
from seqmaestro.reduction.arm import Source
from seqmaestro.reliability import checks, prefilter, tables, tier1, tier2, tier3
from seqmaestro.reliability.files import (
    CHECKS_FILE,
    DISCOVERY,
    NULL,
    PLANTED,
    PLANTED_FILE,
    PREPARE_ARRAYS,
    PREPARE_FILE,
    REAL,
    RECORD_FILE,
    TIER3_FILE,
    load_arrays,
    load_tier2,
    loco_cell,
    read_json,
    save_tier2,
    selections_file,
    tier1_name,
    tier2_name,
    write_json,
)

if TYPE_CHECKING:
    from seqmaestro.dataset import Columns
from seqmaestro.reliability.pool import RECORDED_CAVEAT, RECORDED_FEATURES, Pool
from seqmaestro.target import target_for

RECORD_VERSION = 1

#: Offsets from the configured seed, one per kind of draw.
HALF_SAMPLES_SEED = 101
PERMUTATIONS_SEED = 202
NULL_ON_D_SEED = 303
SIGN_FLIPS_SEED = 404
PLANTED_RECIPE_SEED = 707
PLANTED_ON_D_SEED = 808
LABELS_AT_REVEAL_SEED = 909

#: Why the reproduction check is not run for an arm whose model is not reproducible, as recorded.
#: Every number the panel reports comes from a fit it saved, and what the check would show (that
#: the panel's candidates, model and rows are the reduction's) still rests on the checks of the
#: inputs: the candidates' values against the cell's, and the model's description against a rebuild.
NOT_REPRODUCIBLE = ("the arm's model is not reproducible (its search does not repeat itself under "
                    "one seed), so the reduction rerun on its own rows need not return M*; the "
                    "pool's values and the model's description are checked instead")


@dataclass(frozen=True)
class Settings:
    """The config's `reliability` section, as far as one arm's panel reads it."""

    pairs: int
    permutations: int
    cluster_rho: float
    max_modal_share: float
    planted_count: int
    planted_signal_sd: float
    seed: int


@dataclass
class Prepared:
    """What Step 0 decided, as every later part of the panel and DEFT's side read it back."""

    names: list[str]
    kept: np.ndarray
    cluster: np.ndarray
    halves: np.ndarray
    fit_index: np.ndarray
    val_index: np.ndarray
    #: Label sets on D by name: the real labels, and on the check arm the null and planted ones.
    labels: dict[str, np.ndarray]
    weights: np.ndarray | None

    @property
    def kept_index(self) -> np.ndarray:
        return np.flatnonzero(self.kept)

    @property
    def kept_names(self) -> list[str]:
        return [self.names[i] for i in self.kept_index]

    @classmethod
    def load(cls, arm_dir: str | Path) -> Prepared:
        with np.load(Path(arm_dir) / PREPARE_ARRAYS) as saved:
            return cls(names=[str(name) for name in saved["names"]], kept=saved["kept"],
                       cluster=saved["cluster"], halves=saved["halves"],
                       fit_index=saved["fit_index"], val_index=saved["val_index"],
                       labels={key.removeprefix("labels_"): saved[key] for key in saved.files
                               if key.startswith("labels_")},
                       weights=saved["weights"] if "weights" in saved.files else None)


class Fits(Protocol):
    """The fits, made wherever an arm's model lives: in this process for our own arms and for an
    IGSR arm, whose terms IGSR evaluates on its side, and on DEFT's side of the environment boundary
    for its tree. The panel computes every statistic from what these return, with the same code for
    every arm."""

    #: Whether refitting on the same rows, weights and seed returns the same model. The reproduction
    #: check is run only when it does (`Panel.reproduction`).
    reproducible: bool

    def pool(self, arm_dir: Path) -> Pool:
        """The arm's candidates over D, in D's row order."""

    def reproduce(self, arm_dir: Path, pool: Pool, prepared: Prepared, recorded: list[str],
                  n_target: int) -> dict[str, list[str]]:
        """What the selection keeps on the fitting rows, started from every candidate (`from_pool`)
        and from the features the Stage 2 model used (`from_recorded`)."""

    def select(self, arm_dir: Path, pool: Pool, prepared: Prepared, label_sets: list[str],
               n_target: int, workers: int, force: bool, memory: int | None = None,
               threads_file: Path | None = None) -> dict[str, np.ndarray]:
        """Tier 1's (2B, p) selections over the kept candidates, per label set, each left on disk
        as `selections_file(label_set)` so a sweep resumes. At most `workers` at once, or as many
        as `threads_file` holds as each starts, and fewer where `memory`, the bytes the panel may
        hold, fits fewer (`selections.run`)."""

    def refit(self, arm_dir: Path, pool: Pool, prepared: Prepared,
              plan: list[dict]) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
        """Each planned model fitted on the fitting rows with their weights, and saved in its cell:
        per label, its validation scores and what its `run.json` says about its model. For a model
        that can be undefined on a row, that also holds `FILLED_ROWS`, how many validation rows
        took its constant prediction, which goes beside the cell's metrics instead."""

    def evaluate(self, names: list[str], inputs: Inputs) -> np.ndarray:
        """Candidates on other rows (the test split's, at the reveal), by the same functions."""


# ------------------------------------------------------------------------------ reading back

def is_ready(arm_dir: str | Path) -> bool:
    """Whether an arm's panel finished everything the reveal needs from it."""
    return (Path(arm_dir) / RECORD_FILE).exists()


def read_record(arm_dir: str | Path) -> dict:
    """An arm's record, refused when it is from another version of this layout."""
    record = read_json(Path(arm_dir) / RECORD_FILE)
    if record.get("reliability_version") != RECORD_VERSION:
        raise ValueError(f"{arm_dir}: reliability_version {record.get('reliability_version')!r}, "
                         f"expected {RECORD_VERSION}.")
    return record


def is_revealed(arm_dir: str | Path) -> bool:
    """Whether the reveal has computed every Tier 2 result the arm's record asks of it."""
    arm_dir = Path(arm_dir)
    return all((arm_dir / f"{tier2_name(TEST_SPLIT, label_set)}.json").exists()
               for label_set in read_record(arm_dir)["tier2"]["label_sets_at_reveal"])


def procedure() -> dict:
    """What `select` is, in a form a record can carry."""
    source = Path(eliminate.__file__)
    return {"eliminate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "keep_fraction": eliminate.KEEP_FRACTION, "fine_below": eliminate.FINE_BELOW}


# ------------------------------------------------------------------------------ the panel

class Panel:
    """One arm's panel: where it lives, what it profiles, and where its fits are made.

    `discovery` is D (train then val, re-indexed) with the dataset's columns, and its first
    `n_train` rows are the training split. M*, its size and the arm's seed are the reduction's:
    `reduction` is its record and `reduced_cell` the cell holding M* itself.
    """

    def __init__(self, arm_dir: str | Path, *, run_dir: str | Path, reduction: dict,
                 reduced_cell: str | Path, source: Source, fits: Fits, settings: Settings,
                 discovery: pd.DataFrame, n_train: int, columns: Columns,
                 is_check_arm: bool) -> None:
        self.arm_dir, self.run_dir = Path(arm_dir), Path(run_dir)
        self.reduction, self.reduced_cell = reduction, Path(reduced_cell)
        self.source, self.fits, self.settings = source, fits, settings
        self.discovery, self.n_train = discovery, n_train
        self.columns = columns
        self.is_check_arm = is_check_arm
        # The features M* uses. A survivor with no importance holds its place by the tie-break, and
        # the selection does not count it as chosen (`eliminate.used_columns`).
        self.reported = eliminate.used_columns(reduction["final_columns"],
                                               reduction["final_importance"])
        self.n_target = int(reduction["n_target"])
        self._pool: Pool | None = None

    @property
    def label_sets(self) -> list[str]:
        """What Tier 1 runs under: the real labels, and on the check arm the two checks as well."""
        return [REAL, NULL, PLANTED] if self.is_check_arm else [REAL]

    def tier2_refusal(self) -> str | None:
        try:
            tier2.refuse_weighted(self.columns.weight)
        except tier2.WeightedDesignError as error:
            return str(error)
        return None

    def pool(self) -> Pool:
        if self._pool is None:
            self._pool = self.fits.pool(self.arm_dir)
            if self._pool.values.shape[0] != len(self.discovery):
                raise RuntimeError(f"{self.arm_dir}: the pool has {self._pool.values.shape[0]} rows "
                                   f"and D has {len(self.discovery)}.")
        return self._pool

    # ------------------------------------------------------------------------ before the reveal

    def before_reveal(self, *, workers: int, memory: int | None = None,
                      threads_file: Path | None = None, force: bool = False, log=print) -> dict:
        """Everything the reveal needs from this arm, then its record. Returns the record. An arm
        with a record is finished and is left as it is, unless `force`. Tier 1 runs at most
        `workers` selections at once, or as many as `threads_file` holds as each starts, and fewer
        where `memory`, the bytes the panel may hold, fits fewer."""
        if not force and is_ready(self.arm_dir):
            log("    already profiled")
            return read_record(self.arm_dir)
        if force or not (self.arm_dir / PREPARE_FILE).exists():
            self.prepare(log=log)
        prepared = Prepared.load(self.arm_dir)

        refused = self.tier2_refusal()
        if refused:
            log(f"    {refused}")
        else:
            for label_set in [REAL] + ([PLANTED] if self.is_check_arm else []):
                name = tier2_name(DISCOVERY, label_set)
                if force or not (self.arm_dir / f"{name}.json").exists():
                    self.tier2(name, self.pool().values[:, prepared.kept_index],
                               prepared.labels[label_set], prepared.kept_names, log=log)

        plan = self.loco_plan(prepared)
        if force or not all(cells.is_finished(self.arm_dir / loco_cell(entry["label"]))
                            for entry in plan):
            self.refit(prepared, plan, log=log)

        missing = [label_set for label_set in self.label_sets
                   if force or not (self.arm_dir / f"{tier1_name(label_set)}.json").exists()]
        if missing:
            self.tier1(prepared, missing, workers=workers, memory=memory,
                       threads_file=threads_file, force=force, log=log)

        record = {
            "reliability_version": RECORD_VERSION,
            **outputs.provenance(self.source, self.run_dir),
            "reduced_model": str(self.reduced_cell.relative_to(self.run_dir)),
            "reported": self.reported,
            "target_kind": self.columns.target_kind,
            "n_target": self.n_target,
            "settings": asdict(self.settings),
            "is_check_arm": self.is_check_arm,
            "label_sets": self.label_sets,
            # The arm's candidates, as Step 0 found them. Named apart from `pool`, which a
            # record uses for the pool of sources its arm was handed.
            "candidates": read_json(self.arm_dir / PREPARE_FILE)["candidates"],
            "tier2": {"refused": refused,
                      "label_sets_at_reveal": [] if refused
                      else [REAL, NULL] + ([PLANTED] if self.is_check_arm else [])},
            "loco": [{**entry, "cell": loco_cell(entry["label"])} for entry in plan],
            "tier1_core_seconds": {
                label_set: round(float(load_arrays(self.arm_dir / selections_file(label_set))
                                       ["seconds"].sum()), 1)
                for label_set in self.label_sets},
        }
        write_json(self.arm_dir / RECORD_FILE, record)
        return record

    def prepare(self, log=print) -> dict:
        """Step 0, the half-samples, the label sets, and the proof that `select` is the procedure
        that produced M*."""
        started = time.time()
        pool = self.pool()
        settings, seed = self.settings, self.settings.seed
        rows, _, _ = self.source.fitting_rows(self.discovery.iloc[:self.n_train], self.columns)
        fit_index = np.asarray(rows.index, dtype=np.int64)
        val_index = np.arange(self.n_train, len(self.discovery))
        y = target_for(self.columns.target_kind).values(self.discovery[self.columns.target])
        weights = (self.discovery[self.columns.weight].to_numpy(dtype=float)
                   if self.columns.weight else None)

        recorded = self.source.names
        featurisation_check = check.compare(
            pool.values[np.ix_(val_index, pool.index_of(recorded))], recorded, self.source.cell)

        filtered = prefilter.prefilter(pool.values, pool.names, settings.max_modal_share)
        kept_index = np.flatnonzero(filtered["kept"])
        kept_names = [pool.names[i] for i in kept_index]
        undefined = prefilter.undefined(pool.values, pool.names)
        if undefined:
            log(f"    {len(undefined)} candidate(s) undefined on some rows of D, ranked at the "
                f"median of their defined values: " + ", ".join(
                    f"{name} ({share:.0%})" for name, share in sorted(undefined.items())))
        abs_rho = prefilter.spearman_abs(pool.values[:, kept_index], seed=seed)
        clusters_kept = prefilter.cluster(abs_rho, settings.cluster_rho)
        worst = prefilter.worst_within_cluster(abs_rho, clusters_kept)
        cluster = np.full(pool.p, -1, dtype=np.int64)
        cluster[kept_index] = clusters_kept
        log(f"    pool {pool.p} -> {len(kept_index)} after the prefilter, "
            f"{len(np.unique(clusters_kept))} clusters (worst within |rho| {worst:.3f})")

        halves = tier1.half_samples(len(self.discovery), settings.pairs, seed + HALF_SAMPLES_SEED)
        labels = {REAL: y}
        if self.is_check_arm:
            X_kept = pool.values[:, kept_index]
            recipe = checks.planted_recipe(X_kept, kept_names, filtered["modal_share"][kept_index],
                                           seed=seed + PLANTED_RECIPE_SEED,
                                           n_planted=settings.planted_count,
                                           signal_sd=settings.planted_signal_sd,
                                           target_kind=self.columns.target_kind)
            write_json(self.arm_dir / PLANTED_FILE, recipe)
            labels[NULL] = checks.permuted(y, seed=seed + NULL_ON_D_SEED)
            labels[PLANTED] = checks.draw_planted(X_kept, recipe, seed=seed + PLANTED_ON_D_SEED)

        groups = pool.position_groups()
        self.arm_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.arm_dir / PREPARE_ARRAYS,
            names=np.array(pool.names, dtype=str), kinds=np.array(pool.kinds, dtype=str),
            descriptions=np.array(pool.descriptions, dtype=str),
            position_group=np.array(groups, dtype=str), kept=filtered["kept"],
            modal_share=filtered["modal_share"], nonzero_rate=filtered["nonzero_rate"],
            cluster=cluster, halves=halves, fit_index=fit_index, val_index=val_index,
            **{f"labels_{name}": values for name, values in labels.items()},
            **({"weights": weights} if weights is not None else {}))

        prepared = Prepared.load(self.arm_dir)
        if self.fits.reproducible:
            reproduction = self.reproduction(prepared)
            log(f"    reproduction of M*: {reproduction['from_recorded']['n_shared']}/"
                f"{len(self.reported)} from the recorded features, "
                f"{reproduction['from_pool']['n_shared']}/{len(self.reported)} from the whole pool")
        else:
            reproduction = {"run": False, "reason": NOT_REPRODUCIBLE}
            log(f"    reproduction of M*: not run, since {NOT_REPRODUCIBLE}")

        payload = {
            "candidates": {
                "source": pool.source,
                "caveat": RECORDED_CAVEAT if pool.source == RECORDED_FEATURES else None,
                "p": pool.p, "p_after_prefilter": len(kept_index),
                "prefilter": f"constant, or one value covering more than "
                             f"{settings.max_modal_share} of D",
                "n_dropped": pool.p - len(kept_index),
                "dropped_examples": sorted(filtered["dropped"])[:10],
                "undefined_share_of_D": undefined,
                "n_clusters": len(np.unique(clusters_kept)),
                "worst_within_cluster_abs_rho": worst,
                "cluster_rho": settings.cluster_rho,
                "n_position_groups": len({group for group in groups if group}),
            },
            "n_discovery_rows": len(self.discovery), "n_fit_rows": len(fit_index),
            "half_samples": tier1.check_half_samples(halves, len(self.discovery)),
            "reported_filtered_out": [name for name in self.reported if name not in set(kept_names)],
            "featurisation_check": featurisation_check,
            "procedure": procedure(),
            "reproduction": reproduction,
            "seconds": round(time.time() - started, 1),
        }
        write_json(self.arm_dir / PREPARE_FILE, payload)
        return payload

    def reproduction(self, prepared: Prepared) -> dict:
        """Whether the selection, run on the rows M* was fitted on, returns M*.

        It runs from two starting points, which test different things. **From the recorded
        features** is the reduction itself: the same starting features, rows, weights and loop.
        It must return M* exactly. If it does not, the candidates' values, the model or the loop
        differ from what the reduction used, and the panel stops. **From every candidate**
        (`from_pool`) is the composed procedure Tier 1 resamples: fit on the candidates, keep what
        the model used, eliminate. The two starting points differ only by columns the model gives
        no importance to, so a different answer here is not a fault. It means the elimination is
        sensitive to columns its model ignores, which is the kind of fact Tier 1 measures.
        """
        started = time.time()
        found = self.fits.reproduce(self.arm_dir, self.pool(), prepared, self.source.names,
                                    self.n_target)
        reported = set(self.reported)
        outcome = {start: {"selected": sorted(chosen), "n_shared": len(set(chosen) & reported),
                           "identical": set(chosen) == reported}
                   for start, chosen in found.items()}
        if not outcome["from_recorded"]["identical"]:
            raise RuntimeError(
                f"{self.arm_dir}: started from the features its Stage 2 model used, the selection "
                f"keeps {outcome['from_recorded']['selected']}, not the reduction's M* "
                f"{sorted(reported)}. The panel's pool, model or rows are not the ones the reduction "
                f"used, so it would profile a different procedure.")
        return {"run": True, **outcome, "seconds": round(time.time() - started, 1)}

    def tier2(self, name: str, X: np.ndarray, y: np.ndarray, names: list[str], log=print) -> dict:
        """Tier 2 on these rows and labels, saved under `name`."""
        started = time.time()
        result = tier2.profile(X, y, names, target_kind=self.columns.target_kind,
                               seed=self.settings.seed + PERMUTATIONS_SEED,
                               n_permutations=self.settings.permutations)
        summary = save_tier2(self.arm_dir, name, result, seconds=round(time.time() - started, 1))
        log(f"    {name}: {summary['called'][0.05]['bh']} of {len(names)} called at BH q <= 0.05 "
            f"({summary['seconds']}s)")
        return summary

    def loco_plan(self, prepared: Prepared) -> list[dict]:
        """M* without each of its features, and without each cluster holding several of them. A
        reported feature the prefilter removed belongs to no cluster, and is dropped alone."""
        cluster_of = {}
        for name in self.reported:
            label = int(prepared.cluster[prepared.names.index(name)])
            cluster_of[name] = label if label >= 0 else f"unclustered_{name}"
        return tier3.refits(self.reported, cluster_of)

    def refit(self, prepared: Prepared, plan: list[dict], log=print) -> None:
        """The LOCO refits, each saved as a standard cell the reveal scores."""
        started = time.time()
        fitted = self.fits.refit(self.arm_dir, self.pool(), prepared, plan)
        y_val = prepared.labels[REAL][prepared.val_index]
        for entry in plan:
            scores, about_model = fitted[entry["label"]]
            filled = about_model.pop(FILLED_ROWS, None)
            run = outputs.refit_run(
                self.source, self.run_dir, entry["columns"], len(prepared.fit_index),
                target_kind=self.columns.target_kind, stage="reliability", role="loco", loco={key: entry[key] for key in
                                                        ("label", "kind", "dropped")},
                **about_model)
            outputs.write_cell(self.arm_dir / loco_cell(entry["label"]), run, y_val, scores,
                               filled=filled)
        log(f"    {len(plan)} LOCO refits ({time.time() - started:.1f}s)")

    def tier1(self, prepared: Prepared, label_sets: list[str], *, workers: int, force: bool,
              memory: int | None = None, threads_file: Path | None = None, log=print) -> None:
        """2B selections per label set, then pi-hat and the bound at all three levels."""
        selected = self.fits.select(self.arm_dir, self.pool(), prepared, label_sets, self.n_target,
                                    workers, force, memory, threads_file)
        groups = load_arrays(self.arm_dir / PREPARE_ARRAYS)["position_group"]
        for label_set in label_sets:
            payload, arrays = _tier1_statistics(selected[label_set], prepared, groups)
            np.savez_compressed(self.arm_dir / f"{tier1_name(label_set)}.npz", **arrays)
            write_json(self.arm_dir / f"{tier1_name(label_set)}.json",
                       {"label_set": label_set, **payload})
            log(f"    tier 1 [{label_set}]: q-hat {payload['feature_level']['q_hat']:.2f}, "
                f"{payload['feature_level']['curve'][5]['n_selected']} features at pi >= 0.8")

    # ------------------------------------------------------------------------ at the reveal

    def reveal_tier2(self, inputs: Inputs, y: np.ndarray, log=print) -> None:
        """Tier 2 on the test split, handed its rows by the reveal: the real labels, the null on
        every arm, the planted signal on the check arm. Each result is skipped when already there."""
        record = read_record(self.arm_dir)
        prepared = Prepared.load(self.arm_dir)
        X = self.fits.evaluate(prepared.kept_names, inputs)
        y = target_for(self.columns.target_kind).values(y)
        seed = self.settings.seed + LABELS_AT_REVEAL_SEED
        for label_set in record["tier2"]["label_sets_at_reveal"]:
            name = tier2_name(TEST_SPLIT, label_set)
            if (self.arm_dir / f"{name}.json").exists():
                continue
            if label_set == REAL:
                labels = y
            elif label_set == NULL:
                labels = checks.permuted(y, seed=seed)
            else:
                labels = checks.draw_planted(X, read_json(self.arm_dir / PLANTED_FILE), seed=seed)
            self.tier2(name, X, labels, prepared.kept_names, log=log)

    # ------------------------------------------------------------------------ after the reveal

    def after_reveal(self, log=print) -> dict:
        """Tier 3, the checks and the tables, from what the reveal wrote. Returns the arm's summary."""
        if reveal_record(self.run_dir) is None:
            raise SealError(f"{self.run_dir} has not been revealed; Tier 3 and the tables need it.")
        record = read_record(self.arm_dir)
        if not is_revealed(self.arm_dir):
            raise SealError(f"{self.arm_dir}: the reveal has not computed Tier 2 on the test split.")

        y, p_model = self._revealed(self.run_dir / record["reduced_model"])
        predictions = {}
        for entry in record["loco"]:
            y_cell, predictions[entry["label"]] = self._revealed(self.arm_dir / entry["cell"])
            if not np.array_equal(y_cell, y):
                raise RuntimeError(f"{entry['cell']} was revealed on other labels than M*.")
        result = tier3.summarise(y, p_model, predictions, record["loco"],
                                 target_kind=self.columns.target_kind,
                                 seed=self.settings.seed + SIGN_FLIPS_SEED)
        write_json(self.arm_dir / TIER3_FILE, {"split": TEST_SPLIT, **result})
        necessary = sum(row["verdict"] == "necessary" for row in result["rows"]
                        if row["kind"] == "feature")
        log(f"    tier 3: {necessary}/{len(self.reported)} features necessary at Holm "
            f"{result['holm_alpha']}")

        write_json(self.arm_dir / CHECKS_FILE, self._checks(record))
        return tables.write(self.arm_dir, record)

    @staticmethod
    def _revealed(cell: Path) -> tuple[np.ndarray, np.ndarray]:
        saved = cells.read_predictions(cell)
        if not set(TEST_ARRAYS) <= set(saved):
            raise SealError(f"{cell} was not scored by the reveal.")
        return saved[TEST_ARRAYS[0]], saved[TEST_ARRAYS[1]]

    def _checks(self, record: dict) -> dict:
        result: dict[str, Any] = {"is_check_arm": self.is_check_arm}
        if not record["tier2"]["refused"]:
            null = load_tier2(self.arm_dir, tier2_name(TEST_SPLIT, NULL))
            result["global_null_tier2"] = checks.null_tier2(null["p_value"], null["called"])
        if not self.is_check_arm:
            return result
        recipe = read_json(self.arm_dir / PLANTED_FILE)
        prepared = Prepared.load(self.arm_dir)
        result["planted_features"] = recipe["names"]
        result["planted_signal_sd"] = recipe["signal_sd"]
        for label_set, summarise in ((NULL, "global_null_tier1"), (PLANTED, "planted_tier1")):
            summary = read_json(self.arm_dir / f"{tier1_name(label_set)}.json")
            pi_hat = load_arrays(self.arm_dir / f"{tier1_name(label_set)}.npz")["pi_hat"]
            q_hat = summary["feature_level"]["q_hat"]
            result[summarise] = (checks.null_tier1(q_hat, pi_hat) if label_set == NULL else
                                 checks.planted_tier1(q_hat, len(prepared.kept_names), pi_hat,
                                                      prepared.names, recipe))
        if not record["tier2"]["refused"]:
            for split, key in ((TEST_SPLIT, "planted_tier2"), (DISCOVERY, "planted_tier2_discovery")):
                result[key] = {"split": split, **checks.planted_tier2(
                    load_tier2(self.arm_dir, tier2_name(split, PLANTED)), recipe)}
        return result


def _tier1_statistics(selected: np.ndarray, prepared: Prepared,
                      groups: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    """pi-hat and the bound for features, clusters and positions, and the arrays behind them."""
    kept_index = prepared.kept_index
    cluster = prepared.cluster[kept_index]
    kept_groups = np.array([str(group) for group in groups[kept_index]])

    feature_level = tier1.summarise(selected, p=len(kept_index))
    cluster_selected, cluster_labels = tier1.collapse(selected, cluster)
    cluster_level = tier1.summarise(cluster_selected, p=cluster_selected.shape[1])
    has_group = kept_groups != ""
    if has_group.any():
        group_selected, group_labels = tier1.collapse(selected[:, has_group], kept_groups[has_group])
        group_level = tier1.summarise(group_selected, p=group_selected.shape[1])
    else:
        group_labels, group_level = np.array([], dtype=str), None

    pi_hat = np.zeros(len(prepared.names))
    pi_hat[kept_index] = feature_level["pi_hat"]
    per_cluster = dict(zip(cluster_labels.tolist(), cluster_level["pi_hat"].tolist(), strict=True))
    cluster_pi_hat = np.zeros(len(prepared.names))
    cluster_pi_hat[kept_index] = [per_cluster[int(label)] for label in cluster]
    arrays = {
        "pi_hat": pi_hat, "cluster_pi_hat": cluster_pi_hat,
        "cluster_labels": cluster_labels, "cluster_pi": cluster_level["pi_hat"],
        "cluster_size": np.array([(cluster == label).sum() for label in cluster_labels]),
        "group_labels": np.array([str(group) for group in group_labels], dtype=str),
        "group_pi": group_level["pi_hat"] if group_level else np.zeros(0),
        "selected_counts": selected.sum(axis=0),
    }

    def without_pi(summary):
        return None if summary is None else {k: v for k, v in summary.items() if k != "pi_hat"}

    return ({"feature_level": without_pi(feature_level), "cluster_level": without_pi(cluster_level),
             "position_group_level": without_pi(group_level), "tau_grid": list(tier1.TAU_GRID)},
            arrays)
