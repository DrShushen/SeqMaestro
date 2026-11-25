"""A DEFT arm's Stage 2 cells: which finder a pool runs, what DEFT is handed, and the fit and the
prediction made on DEFT's side through `experiments/adapters/deft_bundle.py`.

The pool decides which of DEFT's feature finders the tree runs. On a pool without
`llm_generated_joint`, such as `positional`, it runs the CART finder: DEFT's tree, splitting
criterion and stopping rules with no language model, on the naive candidates DEFT seeds every node
with. On a pool with it, such as `positional+llm_generated_joint`, it runs DEFT's generative finder
instead, where a language model also writes candidates at each node. The two differ only in the
finder, which is what makes comparing them informative.

This side never imports `deft`. It prepares the inputs, starts the adapter under DEFT's
interpreter, and takes the bundle that comes back, so the two environments share files and nothing
else. The adapter's `fit` is never given the held-out split. At the reveal, `predict_cell` hands
the adapter the held-out rows without their labels and has the saved tree predict them.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from seqmaestro.arms import DEFT_ENVIRONMENT, DEFT_OR_CART, source_problems
from seqmaestro.dataset import Columns
from seqmaestro.features import (
    COVARIATES,
    FIXED_POOLS,
    LLM_JOINT,
    PositionalWindow,
    parse_pool,
    pool_name,
)
from seqmaestro.features.base import Featuriser, Inputs
from seqmaestro.llm import provider_from_environment
from seqmaestro.pipeline.repository import ADAPTERS, REPO
from seqmaestro.pipeline.sources import pool_featuriser
from seqmaestro.pipeline.vendored.handover import (
    BUNDLE,
    VendoredArm,
    call_overrides,
    config_overrides,
    input_frame,
    predict_numbered,
)
from seqmaestro.pipeline.vendored.handover import write_inputs as write_handover
from seqmaestro.provenance import environment_python

if TYPE_CHECKING:
    from seqmaestro.dataset import Dataset
    from seqmaestro.pipeline.config import ExperimentConfig
    from seqmaestro.pipeline.rows import Rows

ADAPTER = ADAPTERS / "deft_bundle.py"
#: Where a task's hand-written prompt config is kept, which the adapter takes in place of the one
#: it would generate from the task description (`experiments/adapters/deft_conf/README.md`).
HAND_WRITTEN_PROMPTS = ADAPTERS / "deft_conf" / "params_generation"
#: DEFT's interpreter on this machine, in the environment `seqmaestro.arms` names (see
#: ENVIRONMENTS.md for the override). Everything that shells out to DEFT uses it.
DEFT_PYTHON = environment_python(DEFT_ENVIRONMENT)

#: What a DEFT cell's `run.json` says about how the reveal predicts it: `predict_cell`, below.
PREDICT_WITH = "deft_bundle"

#: The pool member DEFT derives for itself. DEFT always receives the raw sequence, whatever its
#: pool, because that is what makes a `generated` feature possible: its finder writes functions over
#: the columns it is shown, so without the sequence it could only combine our columns. Any other
#: pool member is handed over as a numeric matrix, which DEFT's naive generator turns into one
#: identity feature per column. From the sequence, DEFT enumerates positional candidates itself
#: (`get_naive_features`), the same hypothesis space `PositionalFeaturiser` produces, so positional
#: columns must not also be supplied, or the pool would hold each of them twice.
DEFT_DERIVES = "positional"

#: Pool members handed to DEFT as they are in the data. The covariates are columns of the split, so
#: nothing is computed for them on this side: they are written unchanged, and `columns.json` tells
#: DEFT what each measures.
HANDED_AS_DATA = (COVARIATES,)

#: The pool member DEFT's generative finder writes at each node of the tree. A pool holding it
#: runs that finder, whereas a pool without it runs the CART finder.
WRITTEN_BY_DEFT = LLM_JOINT


def finder_for(pool: str) -> str:
    """The DEFT feature finder a pool runs, by DEFT's name for it: its config group calls the
    language-model finder `default`."""
    return "default" if WRITTEN_BY_DEFT in parse_pool(pool) else "cart"


#: The pools a DEFT arm can be given: each fixed pool that holds the positions, for each finder,
#: where `seqmaestro.arms` finds no problem with it (`source_problems`).
#:
#: A pool with `kmer` or `motif_known` hands DEFT one continuous column per k-mer or motif summary,
#: and DEFT searches a threshold for every candidate at every node, so the fit's time grows with
#: the number of columns. The driver's estimate says by how much on the dataset in hand.
DEFT_POOLS = tuple(
    pool for pool in (pool_name((*parse_pool(fixed), *written))
                      for written in ((), (WRITTEN_BY_DEFT,)) for fixed in FIXED_POOLS)
    if DEFT_DERIVES in parse_pool(pool) and not source_problems(DEFT_OR_CART, pool)
)


def supplied_members(pool: str) -> tuple[str, ...]:
    """The pool members we compute and write as columns: everything DEFT neither derives for
    itself, nor writes during its fit, nor is handed as data.

    A DEFT pool must contain `positional`, because the raw sequence is always passed and DEFT
    enumerates positional candidates from it whether we ask or not. A pool that omitted it would
    hold a member its name leaves out, so such a pool is refused instead of quietly widened.
    """
    members = parse_pool(pool)
    if DEFT_DERIVES not in members:
        raise ValueError(
            f"DEFT is not compatible with the pool {pool!r}. It is always given the raw sequence, "
            f"so it enumerates {DEFT_DERIVES!r} candidates from it whatever the pool says; a pool "
            f"without {DEFT_DERIVES!r} would contain it anyway and the arm's name would be wrong. "
            f"Use a pool that includes it, or another model."
        )
    return tuple(member for member in members
                 if member not in (DEFT_DERIVES, WRITTEN_BY_DEFT) and member not in HANDED_AS_DATA)


def supplied_featuriser(config: ExperimentConfig, dataset: Dataset, pool: str,
                        run_dir: str | Path | None) -> Featuriser | None:
    """The featuriser of the columns this side computes for a DEFT cell on `pool`
    (`supplied_members`), built as the run computes its sources for every other model, or None
    where the pool holds none."""
    supplied = supplied_members(pool)
    if not supplied:
        return None
    return pool_featuriser(config, dataset, pool_name(supplied), run_dir=run_dir)


def write_inputs(pool, fit_rows, y_fit, train, val, into: Path, *, columns: Columns,
                 descriptions: Mapping[str, str] | None = None,
                 window: PositionalWindow | None = None,
                 supplied: Featuriser | None = None) -> dict:
    """Write what DEFT is handed for a cell on `pool` (`handover.write_inputs`). The raw sequence is
    always written, the covariates are written as they are where the pool holds them, and every
    other pool member that DEFT does not derive itself is written as computed columns
    (`supplied_members`), which DEFT's naive generator turns into one identity feature each.
    `supplied` computes those columns (`supplied_featuriser`), and a pool that holds such a member
    needs it.

    DEFT grows its tree on the fitting rows alone and chooses nothing on the validation rows, so it
    is handed no labels to choose against there."""
    if supplied_members(pool) and supplied is None:
        raise ValueError(f"The pool {pool!r} holds {', '.join(supplied_members(pool))}, which this "
                         f"side computes for DEFT, and no featuriser was given for them.")
    return write_handover(fit_rows, y_fit, train, val, into, columns=columns, choose_on=None,
                          descriptions=descriptions, window=window, supplied=supplied)


def cell_overrides(given: list[str], seed: int) -> list[str]:
    """The caller's config overrides, plus this cell's seed as DEFT's `random_state`.

    **Without it, changing the seed would change only the rows DEFT is handed.** Every other arm
    reseeds both its data and its model, whereas DEFT's config pins `random_state` to a constant,
    and its feature finder uses that value in the draw that chooses which rows are shown to the
    model at each node. Without this override the DEFT arms' stability would be measured under a
    narrower perturbation than the other arms', and comparing them would confound the method with
    the perturbation.

    A single override reaches the finder, because its `random_state` is
    `${oc.select:random_state,42}`, which follows the top-level key. An explicit `random_state=`
    from the caller wins, so a caller can still pin it to a constant.
    """
    if any(override.split("=", 1)[0].strip() == "random_state" for override in given):
        return list(given)
    return [*given, f"random_state={seed}"]


def run_adapter(paths, bundle_dir: Path, feature_finder: str, task: str,
                overrides: list[str], *, sequence_column: str, target_column: str, target_kind: str,
                weight_column: str | None, permutation_repeats: int, permutation_seed: int,
                guidance: str | None, reuse_tree: bool = False):
    """Cross the boundary: the adapter runs under DEFT's interpreter, not this one. DEFT grows a
    classification or a regression tree, as `target_kind` says. `guidance`, where there is any, is
    what the person running the analysis asks of the form of the features (`guidance`, below)."""
    command = [
        str(DEFT_PYTHON), str(ADAPTER), "fit",
        "--train", str(paths["train"]), "--val", str(paths["val"]),
        # Where `marginal_effect` is measured: every training row, with the real labels. It is not
        # `--train`, which is this seed's subsample and holds shuffled labels on a permuted cell.
        "--effect", str(paths["effect"]),
        "--out-bundle", str(bundle_dir),
        "--feature-finder", feature_finder,
        "--target-column", target_column,
        "--target-kind", target_kind,
        "--task", task,
    ]
    # Passed whatever the pool, because without the sequence DEFT's finder can only combine columns
    # we supplied.
    command += ["--sequence-column", sequence_column]
    # Passed explicitly, not left to the adapter's default, so every arm's cross-arm importance is
    # measured the same number of times.
    command += ["--permutation-repeats", str(permutation_repeats),
                "--permutation-seed", str(permutation_seed)]
    if weight_column is not None:
        command += ["--weight-column", weight_column]
    if guidance is not None:
        command += ["--guidance", guidance]
    if reuse_tree:
        command += ["--reuse-tree"]
    for override in overrides:
        command += ["--config-override", override]

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"adapter exited {completed.returncode}")


def predict_cell(cell: Path, inputs: Inputs, sequence_column: str) -> np.ndarray:
    """The scores that a Stage 2 cell's saved tree gives these rows, predicted on DEFT's side:
    probabilities, or a continuous target's values. The reveal calls this for a DEFT cell, and the
    rows cross without their labels."""
    return predict_numbered(
        [str(DEFT_PYTHON), str(ADAPTER), "predict", "--bundle", str(cell / BUNDLE)],
        input_frame(cell, inputs, sequence_column), cell)


def guidance(config: ExperimentConfig, pool: str) -> str | None:
    """The guidance on the form of the features DEFT's language model writes on `pool`: the grid's,
    on a pool with `llm_generated_joint`, and None on one without, where the CART finder writes
    nothing. The adapter shows it in the prompts it generates from the task description."""
    return (config.grid.generated_features.guidance if WRITTEN_BY_DEFT in parse_pool(pool)
            else None)


def check_guidance(config: ExperimentConfig, pool: str) -> None:
    """Refuse guidance that could not reach the prompts of DEFT's language model on `pool`, which
    is the case for a task with a hand-written prompt config, whose wording is fixed so that
    published prompts are reproduced. The adapter refuses such guidance too, and this check refuses
    it before anything starts."""
    from seqmaestro.pipeline.config import ConfigError

    written = HAND_WRITTEN_PROMPTS / f"{config.dataset.task}.yaml"
    if guidance(config, pool) is not None and written.exists():
        raise ConfigError(
            f"{config.source}: `grid.generated_features.guidance` cannot reach DEFT's prompts on "
            f"{pool}, which come from the hand-written config {written}. Leave the guidance out, "
            f"or leave out DEFT's arms on {pool}.")


def stated(config: ExperimentConfig, pool: str) -> dict[str, object]:
    """What a run's grid states for DEFT's tree on `pool`, as overrides of DEFT's config.

    The tree takes a binary target's class weighting through DEFT's config. Its generative finder,
    on a pool with `llm_generated_joint`, also takes the grid's policy for features its language
    model writes, which the proposer takes as its settings too, so that one policy holds for every
    arm that writes features. The policy's time limit is the exception: DEFT stops a written
    feature after 60 seconds, a number fixed in its code that a setting cannot change.
    """
    grid = config.grid
    found: dict[str, object] = ({} if grid.class_weight is None
                                else {"tree.class_weight": grid.class_weight})
    if WRITTEN_BY_DEFT in parse_pool(pool):
        generated = grid.generated_features
        found |= {
            "params_generation.repair_rounds": generated.repair_rounds,
            "params_generation.max_undefined_share": generated.max_undefined_share,
            "params_generation.max_sequence_chars_in_prompt": generated.max_example_length,
        }
    return found


#: For each provider a run can call (`seqmaestro.llm.PROVIDERS`), DEFT's configs for it: the model
#: and its credentials (`llm_models`), and how DEFT's engine calls a model of that family (`llm`).
LANGUAGE_MODEL = {
    "azure": {"llm_models": "seqmaestro_azure", "llm": "gpt5"},
    "azure_foundry": {"llm_models": "seqmaestro_azure_foundry", "llm": "foundry"},
    "openai": {"llm_models": "seqmaestro_openai", "llm": "gpt5"},
    "claude": {"llm_models": "seqmaestro_claude", "llm": "claude"},
}
LANGUAGE_MODEL_KEYS = ("llm_models", "llm")
#: DEFT's own configuration, whose values hold where a run states no override.
DEFT_CONF = REPO / "methods" / "deft" / "src" / "deft" / "conf"
#: DEFT's engine settings (`llm`) that the run's `language_model` states, by its field.
CALL_KEYS = {"reasoning_effort": "llm.reasoning_effort", "retries": "llm.num_retries",
             "timeout_seconds": "llm.request_timeout"}


def call_settings(config: ExperimentConfig, provider: str | None) -> dict[str, int]:
    """What decides how many calls DEFT's generative finder makes, from DEFT's configuration with
    the run's overrides: the reflections (`params_generation.n_reflections`), the candidates asked
    for per prompt (`params_generation.n_samples_per_prompt`), the tree's depth (`tree.max_depth`),
    and how many completions one request may ask for, as DEFT's engine config for the provider
    states it (`max_n_per_request`). Where the provider is not known, one completion per request is
    counted, which makes the most requests any provider's engine makes."""
    stated = dict(item.split("=", 1)
                  for item in config.models.get(DEFT_OR_CART, {}).get("config_overrides", []))
    generation = yaml.safe_load((DEFT_CONF / "params_generation" / "default.yaml").read_text())
    tree = yaml.safe_load((DEFT_CONF / "config.yaml").read_text())
    candidates = int(stated.get("params_generation.n_samples_per_prompt",
                                generation["n_samples_per_prompt"]))
    per_request = 1
    if provider is not None:
        engine = yaml.safe_load(
            (DEFT_CONF / "llm" / f"{LANGUAGE_MODEL[provider]['llm']}.yaml").read_text())
        per_request = int(engine.get("max_n_per_request") or candidates)
    return {"reflections": int(stated.get("params_generation.n_reflections",
                                          generation["n_reflections"])),
            "candidates": candidates, "per_request": per_request,
            "max_depth": int(stated.get("tree.max_depth", tree["max_depth"]))}


def run_overrides(config: ExperimentConfig, pool: str) -> list[str]:
    """The overrides the run's config makes to DEFT's config for an arm on `pool`: what the grid
    states, then the run's `models.deft_or_cart.config_overrides`. They reach both of DEFT's
    finders, and the CART finder, which builds no language model, ignores the settings for one.
    The function does not read the environment, so the driver can refuse a conflicting config
    before anything starts."""
    return config_overrides(config, DEFT_OR_CART, stated(config, pool),
                            (*LANGUAGE_MODEL_KEYS, *CALL_KEYS.values()))


def overrides(config: ExperimentConfig, pool: str) -> list[str]:
    """Every override DEFT's config takes for an arm on `pool`: on a pool where DEFT writes
    features, the language model of the provider the environment names and how the run calls it
    (`call_overrides`), and then `run_overrides`. A CART arm is not told of a language model and
    needs no provider."""
    if WRITTEN_BY_DEFT not in parse_pool(pool):
        return run_overrides(config, pool)
    provider = provider_from_environment()
    return [*(f"{key}={value}" for key, value in LANGUAGE_MODEL[provider.name].items()),
            *call_overrides(config.language_model, provider, CALL_KEYS),
            *run_overrides(config, pool)]


def make_room(bundle: Path, *, force: bool) -> None:
    """Clear what an earlier fit left in a cell's bundle, where a new fit may take its place.

    DEFT refuses to start in a directory that holds an earlier fit's files, so that a cell is not
    fitted twice unnoticed: a second fit draws another tree, and on a permuted cell a choice
    between two draws would choose the null. Two cases involve no such choice. A fit that ended
    before it saved its tree left nothing to choose, so its remains are cleared and the cell is
    fitted. A fit asked for with `--force` is a stated decision to replace the cell. A bundle
    that holds a tree, found without `--force`, belongs to a cell whose fit succeeded and whose
    later steps did not. Such a cell is refused here, because it should be completed from its tree
    instead (`--reuse-tree`).
    """
    if not bundle.exists():
        return
    grown = sorted(bundle.rglob("*.dill"))
    if grown and not force:
        raise RuntimeError(
            f"{bundle} holds the tree an earlier fit grew ({grown[0].name}), and the cell is not "
            f"finished. Complete it from that tree with --reuse-tree. Fitting the cell again "
            f"would draw another tree, which only --force does.")
    shutil.rmtree(bundle)


class Arm(VendoredArm):
    """A DEFT arm of a run: its tree on one pool, with the finder the pool calls for.

    `reuse_tree` is for recovery: the adapter takes the tree already in a cell's bundle and writes
    the bundle's other files from it, without growing a new one. Fitting a cell again after its fit
    succeeded would draw a different tree, and on a permuted cell that would choose which draw of
    the null is kept (`--reuse-tree` in `experiments/adapters/deft_bundle.py`). `force` says that a
    cell may be fitted again nonetheless (`make_room`).
    """

    predict_with = PREDICT_WITH

    def __init__(self, config: ExperimentConfig, rows: Rows, model: str, pool: str, *,
                 run_dir: str | Path, reuse_tree: bool = False, force: bool = False) -> None:
        super().__init__(config, rows, model, pool)
        self.run_dir = Path(run_dir)
        self.finder = finder_for(pool)
        self.overrides = overrides(config, pool)
        self.guidance = guidance(config, pool)
        self.reuse_tree = reuse_tree
        self.force = force

    def write(self, fit_rows: pd.DataFrame, y_fit: np.ndarray, y_choose: np.ndarray,
              into: Path) -> dict:
        # `y_choose` stays on this side: DEFT chooses nothing on the validation rows.
        return write_inputs(self.pool, fit_rows, y_fit, self.rows.train, self.rows.val, into,
                            columns=self.dataset.columns_for(self.pool),
                            descriptions=self.dataset.covariates_for(self.pool),
                            window=self.dataset.positional_window,
                            supplied=supplied_featuriser(self.config, self.dataset, self.pool,
                                                         self.run_dir))

    def run(self, paths: dict, bundle: Path, seed: int) -> None:
        columns = self.dataset.columns
        if not self.reuse_tree:
            make_room(bundle, force=self.force)
        run_adapter(paths, bundle, self.finder, self.config.dataset.task,
                    cell_overrides(self.overrides, seed), sequence_column=columns.sequence,
                    target_column=columns.target, target_kind=self.dataset.target_kind,
                    weight_column=columns.weight,
                    permutation_repeats=self.config.grid.permutation_repeats,
                    permutation_seed=seed, guidance=self.guidance, reuse_tree=self.reuse_tree)
