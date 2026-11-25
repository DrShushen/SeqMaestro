"""Which discoveries several arms corroborate, and which belong to one arm alone.

    python experiments/analysis/analyse_co_discovery.py --run-dir experiments/runs/<date>_<name>

Within each label condition and seed, the top-N features of every arm are clustered into concepts
(connected components of the graph of matched features), and each concept is scored by the arms
that found it: `n_arms` as counted, `n_witnesses` with definitionally linked arms counted once
(`linked_by_fit`, `linked_groups`), `n_pools`, and the direction its members agree on, with signs
aligned first. The same clustering of the permuted cells gives each concept an empirical p-value
and each arm its chance level. Arms are compared by the share of their concepts that an arm on a
different pool also found, and the chance-corrected share is reported last, for the reasons
`per_arm_summary` gives.

The command writes two tables under the run's `results/`: `co_discovery.csv`, with one row per
concept, and `co_discovery_by_arm.csv`, with one row per arm over every seed.
`docs/developer/analyses/stability_and_co_discovery.md` explains each count, and why the raw count
of arms misleads.
"""

from __future__ import annotations

import argparse
import collections
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from seqmaestro.arms import split_arm
from seqmaestro.features import LLM_JOINT, parse_pool
from seqmaestro.harness import cell as cells
from seqmaestro.harness.direction import OPPOSITE
from seqmaestro.harness.stability import (
    BehaviouralMatcher,
    chance_corrected_agreement,
    consensus_labels,
    load_run,
)
from seqmaestro.pipeline.config import ConfigError, load_recorded
from seqmaestro.pipeline.layout import RESULTS_DIR
from seqmaestro.provenance import record_call

#: The name of this call (`seqmaestro.pipeline.layout.call_name`), and the two tables it writes
#: under the run's `results/`: one row per concept, and one per arm.
CALL = "analyse_co_discovery"
TABLE = "co_discovery.csv"
BY_ARM_TABLE = "co_discovery_by_arm.csv"


def linked_by_fit(arms: Iterable[str]) -> list[frozenset[str]]:
    """The groups of arms that are one model on pools differing only in the source the model
    writes during its fit: DEFT's tree on `positional`, which is plain CART, and on
    `positional+llm_generated_joint`.

    Their agreement is a property of the pipeline and not of the data, so each group counts as one
    witness. The two share the vendored tree, the naive candidate enumeration and, under
    `permutation_scope=seed`, the same shuffled label vector. They differ only in whether a
    language model proposes extra candidates. On shuffled labels those candidates win no split, so
    the generative tree is close to a copy of CART's, and the pair agrees *more* under the null than
    under real labels. The chance correction cannot remove that, because it subtracts a chance level
    measured when there is nothing to find, which is where this pair agrees most.

    Arms that only share a pool are not grouped: they select independently from a common
    vocabulary, which is a weaker form of dependence that is left visible in the results instead of
    being corrected away.
    """
    grouped: dict[tuple[str, tuple[str, ...]], set[str]] = collections.defaultdict(set)
    for arm in arms:
        model, pool = split_arm(arm)
        # The pool's other sources, in the order every pool lists them. The tuple is empty for a
        # model whose pool is only the source it writes.
        others = tuple(member for member in parse_pool(pool) if member != LLM_JOINT)
        grouped[model, others].add(arm)
    return [frozenset(group) for group in grouped.values() if len(group) > 1]


def arm_name(cell: Path, results: Path) -> str:
    """The arm a results cell belongs to: everything between the results root and `<labels>/<seed>`.

    Three callers need this name and must agree on it, so it is defined once. A group of arms taken
    from the run's records is compared with the arm names built here, so a name assembled with one
    component more or fewer would match nothing, and the group would silently never apply.
    """
    return "/".join(cell.parts[len(results.parts):-2])


def proposal_keys(described) -> set[str]:
    """Every proposal a cell's recorded featuriser names, by its cache key. A proposer states the
    key of its proposal, and a featuriser states its proposer where it has one: at its top for a
    pool that is the proposal alone, inside one of its parts for a pool that holds more, and for
    the iterative source both in its pool and beside it."""
    if isinstance(described, dict):
        own = {described["proposal_key"]} if described.get("proposal_key") else set()
        return own.union(*(proposal_keys(value) for value in described.values()))
    if isinstance(described, list):
        return set().union(*(proposal_keys(value) for value in described))
    return set()


def linked_groups(results: Path, labels: str, seed: str) -> list[frozenset[str]]:
    """The groups of arms of this (labels, seed) that do not count as independent witnesses: arms
    that consumed one proposal, as the run records them, and arms linked by their fit
    (`linked_by_fit`).

    A proposer records which cache entry it used, so the run itself states which arms shared a
    proposal. Each arm's proposal is made for that arm alone (`Proposer.arm`), so a run of the
    pipeline has no such group.

    **The two kinds are combined.** Returning only the groups that share a proposal whenever any
    exist would drop the tree's pair on every such run, and the omission would go unflagged,
    because the analysis would complete with only the affected arm's number wrong.
    """
    by_key: dict[str, set[str]] = {}
    present = []
    for run_file in sorted(results.rglob(f"*/{labels}/{seed}/{cells.RUN_FILE}")):
        arm = arm_name(run_file.parent, results)
        present.append(arm)
        for key in proposal_keys(cells.read_run(run_file.parent).get("featuriser")):
            by_key.setdefault(key, set()).add(arm)
    groups = [frozenset(arms) for arms in by_key.values() if len(arms) > 1]
    groups += [group for group in linked_by_fit(present) if group not in groups]
    return groups


def independent_witnesses(found_by: set[str], groups: list[frozenset[str]]) -> int:
    """Distinct arms, with each group of linked arms (`linked_groups`) counted once."""
    remaining = set(found_by)
    count = 0
    for group in groups:
        if overlap := (remaining & group):
            count += 1
            remaining -= overlap
    return count + len(remaining)


def recorded_pools(results: Path, labels: str, seed: str) -> dict[str, frozenset[str]]:
    """Each arm's input pool at this (labels, seed), taken from what its cells recorded instead of
    from its directory name.

    A directory name does not identify the pool, since a name is only a label, whereas a cell's
    `run.json` records the pool's members as `sources`, which is what the arm was handed.

    A cell whose `run.json` states no `sources` is refused, because Stage 2 records them for every
    cell of an arm.
    """
    pools: dict[str, frozenset[str]] = {}
    for run_file in sorted(results.rglob(f"*/{labels}/{seed}/{cells.RUN_FILE}")):
        arm = arm_name(run_file.parent, results)
        members = cells.read_run(run_file.parent).get("sources")
        if not members:
            raise ValueError(f"{run_file} records no input pool, so its arm cannot be counted.")
        pools[arm] = frozenset(members)
    return pools


def distinct_pools(found_by: set[str], pools: dict[str, frozenset[str]]) -> int:
    """How many different input pools found a concept: the stricter, secondary count.

    It is reported beside `n_witnesses` because arms are not equally independent of one another.
    Three models over the same positional features are three arms but one representation, and they
    agree substantially even on shuffled labels, as the permuted rows of the output show. A concept
    found by an arm on positions, an arm on k-mers and an arm on a written source has been reached
    in three different ways, whereas one found by three models over one vocabulary has not.
    """
    return len({pools[arm] for arm in found_by})


def arms_present(results: Path, labels: str, seed: str) -> list[tuple[str, Path]]:
    """Every arm holding features for this (labels, seed), as (name, directory)."""
    found = []
    for features_file in sorted(results.rglob(f"*/{labels}/{seed}/{cells.FEATURES_FILE}")):
        cell = features_file.parent
        found.append((arm_name(cell, results), cell))
    return found


def agreed_direction(members: list[tuple[str, np.ndarray]]) -> tuple[str | None, float | None]:
    """The direction a concept's members agree on, and the share of them holding it.

    **Signs are aligned before they are compared**, because skipping that would invert the answer.
    Features are matched on *absolute* rank correlation, because a feature and its negation hold
    the same information, so a cluster may legitimately hold a GC fraction from one arm and an AT
    fraction from another, whose directions are opposite while the finding is the same. Each member
    is therefore compared with the first one and flipped where it runs the other way.

    The comparison uses the **model's** direction and not the marginal effect. Two features that
    matched already have near-identical marginal effects by construction, so agreeing about those
    would only restate the match, whereas agreeing about how each arm *uses* the property adds
    information, because every arm conditions on a different set of other features.

    The share is `None` for a concept in which only one member has a usable direction (a member
    whose values do not vary on the reference rows cannot be aligned, so it is left out), because
    there is nothing for it to agree with, and 1.0 would look like unanimity.
    """
    usable = []
    reference = members[0][1] if members else None
    for word, values in members:
        if reference is None or np.std(values) == 0 or np.std(reference) == 0:
            continue
        correlation = spearmanr(reference, values).statistic
        if not np.isfinite(correlation):
            continue
        usable.append(OPPOSITE[word] if correlation < 0 else word)

    if not usable:
        return None, None
    winner, held = collections.Counter(usable).most_common(1)[0]
    return winner, (round(held / len(usable), 3) if len(usable) > 1 else None)


def concepts(results: Path, labels: str, seed: str, top_n: int) -> list[dict]:
    """Cluster every arm's top-N into concepts, and score each by how many arms found it."""
    present = arms_present(results, labels, seed)
    if len(present) < 2:
        return []

    linked = linked_groups(results, labels, seed)
    pools = recorded_pools(results, labels, seed)
    matcher = BehaviouralMatcher()
    runs, owners = [], []
    for name, cell in present:
        runs.append(load_run(cell, label=name).top(top_n))
        owners.append(name)

    found_by: dict[int, set[str]] = collections.defaultdict(set)
    names: dict[int, list[str]] = collections.defaultdict(list)
    # The values travel with the direction because comparing directions needs them: see
    # `agreed_direction`, where the sign of the match decides whether two direction words make the
    # same claim.
    pointing: dict[int, list[tuple[str, np.ndarray]]] = collections.defaultdict(list)
    for owner, run, run_labels in zip(owners, runs, consensus_labels(runs, matcher), strict=True):
        for index, (record, cluster) in enumerate(zip(run.records, run_labels, strict=True)):
            found_by[cluster].add(owner)
            names[cluster].append(record.name)
            if record.direction is not None:
                pointing[cluster].append((record.direction, run.values[:, index]))

    rows = []
    for cluster, arms in sorted(found_by.items()):
        direction, agreement = agreed_direction(pointing[cluster])
        rows.append({
            "labels": labels, "seed": seed, "concept": cluster,
            "n_arms": len(arms), "n_witnesses": independent_witnesses(arms, linked),
            "n_pools": distinct_pools(arms, pools),
            "direction": direction, "direction_agreement": agreement,
            "arms": " | ".join(sorted(arms)),
            "names": " | ".join(sorted(set(names[cluster]))),
        })
    return rows


def per_arm_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """For each arm: how often another arm found the same thing, at several strengths.

    **The columns are meant to be judged together.** Corroboration by *any* second arm is a weak
    criterion, because it fires on some concepts even under shuffled labels, and it flatters
    whichever arm has the most company: three models over one vocabulary corroborate each other by
    construction. The count is therefore reported at rising strengths beside the chance level,
    instead of being folded into a single figure.

    | | |
    |---|---|
    | `mean_witnesses` | mean number of independent arms per concept, without a threshold |
    | `corroborated_share`, `_3`, `_4` | share of concepts found by at least 2, 3 or 4 arms |
    | `cross_pool_share` | share an arm from a **different input pool** also found |
    | `unique_share` | the complement of `corroborated_share`: this arm's alone |
    | `corroborated_share_permuted` | the same criterion on shuffled labels |
    | `corroborated_share_corrected` | `(observed - chance) / (1 - chance)`, to be considered last |

    Use `cross_pool_share` to compare arms with each other, because it does not reward an arm for
    sharing a vocabulary with two others. Its pools come from the members each cell records, so an
    arm that derives a pool for itself counts as that pool whatever its directory is called.

    **The corrected column is kept but reported last.** It divides by `1 - chance`, so for an arm
    with a high chance level a small shortfall becomes a large negative. It also assumes that the
    shuffled-label cells differ from the real ones only in the labels, which is untrue for an arm
    whose feature *mix* depends on the labels: a generative arm that finds nothing generates
    nothing.
    """
    rows: dict[tuple[str, str], dict] = {}
    for record in frame.itertuples():
        for arm in record.arms.split(" | "):
            entry = rows.setdefault((arm, record.labels),
                                    {"n": 0, "witnesses": 0, 2: 0, 3: 0, 4: 0, "cross": 0})
            entry["n"] += 1
            entry["witnesses"] += record.n_witnesses
            for level in (2, 3, 4):
                entry[level] += int(record.n_witnesses >= level)
            entry["cross"] += int(record.n_pools >= 2)

    def share(entry: dict, key) -> float:
        return entry[key] / entry["n"]

    out = []
    for (arm, labels), entry in sorted(rows.items()):
        if labels != "real" or not entry["n"]:
            continue
        chance_entry = rows.get((arm, "permuted"))
        chance = share(chance_entry, 2) if chance_entry and chance_entry["n"] else None
        observed = share(entry, 2)
        out.append({
            "arm": arm,
            "n_concepts_real": entry["n"],
            "mean_witnesses": round(entry["witnesses"] / entry["n"], 3),
            "corroborated_share": round(observed, 4),
            "corroborated_share_3": round(share(entry, 3), 4),
            "corroborated_share_4": round(share(entry, 4), 4),
            "cross_pool_share": round(share(entry, "cross"), 4),
            "unique_share": round(1 - observed, 4),
            "corroborated_share_permuted": None if chance is None else round(chance, 4),
            "corroborated_share_corrected":
                None if chance is None else round(chance_corrected_agreement(observed, chance), 4),
        })
    return pd.DataFrame(out)


def empirical_p(frame: pd.DataFrame, column: str = "n_witnesses") -> dict[int, float]:
    """P(corroboration >= m) under the permuted-label null, pooled over seeds."""
    null = frame[frame.labels == cells.PERMUTED][column]
    if null.empty:
        return {}
    return {m: float((null >= m).mean()) for m in range(1, int(frame[column].max()) + 1)}


def shares_by_label_set(frame: pd.DataFrame) -> list[str]:
    """The share of each label condition's concepts that two or more input pools found, then the
    share that two or more arms found.

    A label condition can have no concepts at all, since on shuffled labels each arm may end up
    using no feature (a sparse model can, and so can a symbolic one that chose its constant). The
    line then says so.
    """
    lines = []
    for column, what in (("n_pools", "found by >=2 pools"), ("n_witnesses", "corroborated by >=2 arms")):
        for labels in cells.LABELS:
            subset = frame[frame.labels == labels]
            if subset.empty:
                lines.append(f"  {labels:9s}: no arm used a feature")
                continue
            count = int((subset[column] >= 2).sum())
            lines.append(f"  {labels:9s}: {count} of {len(subset)} concepts {what} "
                         f"({count / len(subset):.1%})")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run whose results/ to analyse, with the cut its config states "
                             "(`analyses.top_n`)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = args.run_dir / RESULTS_DIR
    try:
        args.top_n = load_recorded(args.run_dir).analyses.top_n
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    record_call(args.run_dir, CALL, vars(args))

    seeds = sorted({p.parent.name
                    for p in results.rglob(f"*/{cells.REAL}/seed_*/{cells.FEATURES_FILE}")})
    rows = []
    for labels in cells.LABELS:
        for seed in seeds:
            rows.extend(concepts(results, labels, seed, args.top_n))
    if not rows:
        print("No arms with features found; nothing to compare.")
        return 0

    frame = pd.DataFrame(rows)
    frame.to_csv(results / TABLE, index=False)

    print(f"{len(seeds)} seed(s); concepts are clustered within a (labels, seed).")
    print("\nHow many independent arms found each concept "
          "(linked arms counted once; see `linked_groups`):")
    print(frame.groupby(["labels", "n_witnesses"]).size().unstack(fill_value=0).to_string())

    print("\nHow many distinct input pools found each concept. This is the stricter count, since arms "
          "handed\nthe same features agree substantially even on shuffled labels:")
    print(frame.groupby(["labels", "n_pools"]).size().unstack(fill_value=0).to_string())
    for line in shares_by_label_set(frame):
        print(line)

    for column, title in (("n_witnesses", "independent arms"),
                          ("n_pools", "distinct input pools")):
        p_values = empirical_p(frame, column)
        if p_values:
            print(f"\nEmpirical P({title} >= m) under the permuted-label null:")
            print("   " + "   ".join(f"m>={m}: p={p:.3f}" for m, p in p_values.items()))

    summary = per_arm_summary(frame)
    summary.to_csv(results / BY_ARM_TABLE, index=False)
    print("\nPer arm: share of its reported features that another arm also found:")
    print(summary.to_string(index=False))

    print(f"\nWritten to {results / TABLE} and {results / BY_ARM_TABLE}")
    return 0

