"""Recurrence across seeds, in the two senses an arm can have it: the same feature and the same idea.

    python experiments/analysis/analyse_recurrence.py --run-dir experiments/runs/<date>_<name>

**Credentials required** for a real run: `set -a && source .env && set +a`.

A language model groups the written features of every arm on a written source (`written_arms`, or
the arms `--arms` names) by what they compute, in one pass per arm and one over every arm at once.
The command writes `results/concept_recurrence.{csv,json}`, `results/feature_recurrence.csv` and
`results/cross_arm_recurrence.csv`. `docs/developer/analyses/recurrence.md` explains what each table
counts, why the other analyses cannot answer this question, and the safeguards that make the counts
meaningful.

Each call is given the tokens and the time that a reply to its list needs (`reply_tokens`,
`reply_seconds`), since a reply names every listed feature by its number. The tables on file are
kept while what they were computed from is unchanged (`inputs.py`), and `--force` groups again. If
a grouping call fails, the command still writes the tables from the groupings that succeeded, with
no digest in their record so that the next call groups again, and it exits with status 1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom

from seqmaestro import files
from seqmaestro.arms import arm_name, calls_language_model
from seqmaestro.harness import cell as cells
from seqmaestro.harness.features import load_features, top_features
from seqmaestro.llm import LiteLLMClient, load_task, reply_conditions
from seqmaestro.pipeline.analyses import inputs
from seqmaestro.pipeline.config import ConfigError, load_recorded
from seqmaestro.pipeline.layout import RESULTS_DIR
from seqmaestro.pipeline.repository import DESCRIPTIONS
from seqmaestro.provenance import record_call

#: The name of this call (`seqmaestro.pipeline.layout.call_name`), and what it writes under the
#: run's `results/`: a table with a row per concept of each arm, one with a row per feature by
#: name, one with a row per concept of every arm together, and the record of the groupings.
CALL = "analyse_recurrence"
BY_CONCEPT_TABLE = "concept_recurrence.csv"
BY_NAME_TABLE = "feature_recurrence.csv"
ACROSS_ARMS_TABLE = "cross_arm_recurrence.csv"
RECORD = "concept_recurrence.json"

#: The arguments that say where and whether to run. The others are the settings an answer
#: depends on.
NOT_SETTINGS = ("run_dir", "dry_run", "force")

#: What a grouping reply takes, in tokens. It names every listed item by its number, at up to
#: `TOKENS_PER_ITEM` for a number and its separator, and every group at up to `TOKENS_PER_GROUP`
#: for its name and the JSON around it. `TOKENS_TO_SPARE` covers what a model spends before it
#: answers, and `reply_tokens` adds them up for a listing.
TOKENS_PER_ITEM = 4
TOKENS_PER_GROUP = 24
TOKENS_TO_SPARE = 4096
#: How long a reply is waited for: `SECONDS_TO_SPARE` for the request to be taken in, and the
#: time its tokens take at `TOKENS_PER_SECOND`. The rate is set low, so that a reply from a slow
#: endpoint is still waited for.
TOKENS_PER_SECOND = 40
SECONDS_TO_SPARE = 120


def written_arms(results: Path) -> list[str]:
    """The arms under `results` whose pool holds a source a language model writes.

    These are the arms that can give one idea another name at each seed, whichever written source
    the pool holds. On the iterative and the joint source a cell's features are written during its
    fit, whereas on the one-round source they are written before it, and each cell still has a
    proposal of its own. The arms are found from the run's directories and the pools' names, so an
    arm a config adds is included without a list kept here.
    """
    return sorted(arm_name(pool.parent.name, pool.name) for pool in results.glob("*/*")
                  if pool.is_dir() and calls_language_model(pool.name))


#: Grouping is at the level of the *mechanism being measured*, and the rules below are domain-free.
#: What counts as "the same idea" in a given dataset comes from that dataset's description, which
#: the proposer and the judge are also told. Naming one dataset's motifs here would make the
#: analysis correct for that dataset and quietly wrong on the next one.
#:
#: The granularity is set in the prompt because it has to be identical for both label conditions,
#: and because a model left to choose returns close to the identity: nearly as many groups as
#: features, with one motif split into its two strand readings.
SYSTEM = (
    "You group candidate features proposed by an automated discovery system, by the underlying "
    "property each one measures. You are given the dataset they were proposed for, then a numbered "
    "list of feature names with descriptions. Two features belong in the same group when a domain "
    "expert would call them the same idea.\n\n"
    "Group at the level of the mechanism, not the arithmetic:\n"
    "- The same quantity measured over different window widths, offsets, or regions is ONE group.\n"
    "- Presence, count, density, fraction and difference-of-counts of the same thing are ONE group.\n"
    "- The same pattern read under a symmetry of the representation is ONE group, when the dataset "
    "description implies the two readings describe one underlying process.\n"
    "- Genuinely different mechanisms stay apart. A feature indexing a single arbitrary coordinate, "
    "with no stated meaning beyond its position, is its own group.\n\n"
    "Aim for roughly one group per {items_per_group} items. Returning almost as many groups as "
    "items means you have grouped by arithmetic rather than by idea.\n\n"
    "Give each group a short lower_snake_case name describing the idea. Every item must appear in "
    "exactly one group. Reply with JSON only, of the form "
    '{{"groups": [{{"name": "...", "items": [1, 4, 9]}}]}} and nothing else.'
)


def reply_tokens(n_items: int, items_per_group: int) -> int:
    """The tokens a grouping reply over `n_items` may take.

    The pass across arms lists every arm's features in one request, so its reply is several times
    as long as any arm's. A limit that suited one arm would truncate that reply, and a truncated
    reply cannot be parsed, which fails the analysis. The limit therefore follows the listing, with
    room for every item's number and for the groups the request asks for (one for each
    `items_per_group` items).
    """
    groups = math.ceil(n_items / items_per_group)
    return TOKENS_TO_SPARE + TOKENS_PER_ITEM * n_items + TOKENS_PER_GROUP * groups


def reply_seconds(tokens: int) -> int:
    """How long to wait for a reply of up to `tokens`, so that a long reply is not given up on
    while it is still being written."""
    return SECONDS_TO_SPARE + math.ceil(tokens / TOKENS_PER_SECOND)


def build_system(context, items_per_group: int) -> str:
    """The grouping rules, followed by the task description as the models are told it."""
    return SYSTEM.format(items_per_group=items_per_group) + f"\n\nThe dataset:\n{context.render()}"


def collect(results: Path, arm: str, kinds: set[str], top_n: int | None) -> list[dict]:
    """Each cell's reported features, tagged with the seed and condition they came from.

    The features are restricted to the `top_n` of the kinds asked for that the model relied on
    most, by the size of their importance whatever its sign (`top_features`). Without this
    restriction, an arm that keeps everything it generated would be compared with one that selects
    a handful.
    """
    out = []
    for cell in sorted(results.glob(f"{arm}/*/seed_*")):
        # A cell whose fit failed leaves its directory behind with no `features.json`. It is
        # skipped, as every other analysis does, because raising would turn one failed cell into a
        # failed stage and lose the run's whole recurrence table.
        if not (cell / cells.FEATURES_FILE).exists():
            print(f"  {cell.relative_to(results)}: no {cells.FEATURES_FILE}, skipped", file=sys.stderr)
            continue
        records, _ = load_features(cell / cells.FEATURES_FILE)
        if kinds:
            records = [r for r in records if r.kind in kinds]
        for record in top_features(records, top_n or max(len(records), 1)):
            out.append({"name": record.name, "arm": arm,
                        "provenance": record.provenance,
                        "description": " ".join((getattr(record, "description", "") or "").split())[:300],
                        # Its size: a sparse linear arm's importance is a signed weight.
                        "importance": abs(record.importance or 0.0),
                        "labels": cell.parent.name, "seed": cell.name})
    return out


def exact_recurrence(features: list[dict]) -> pd.DataFrame:
    """The features that recur character for character across seeds, on real labels only.

    This view must be judged together with the concept view, because an arm drawing from a fixed
    vocabulary scores well here by construction, whereas one that writes its features scores near
    zero however consistent its ideas are.
    """
    real = [f for f in features if f["labels"] == "real"]
    seeds = len({f["seed"] for f in real})
    rows = {}
    for f in real:
        # Provenance is kept per row. A written feature recurs by name only if it was named the
        # same way again, whether its proposal was fixed before the fit (`selected`) or it was
        # written during the fit (`generated`). A feature of an enumerated kind, which a config's
        # `kinds` can include, is `selected` too and recurs by name by construction, so its rows
        # must not be averaged with the written ones.
        row = rows.setdefault(f["name"], {"feature": f["name"], "arm": f["arm"],
                                          "provenance": f.get("provenance"), "seeds": set(),
                                          "importance": 0.0, "description": f["description"]})
        row["seeds"].add(f["seed"])
        row["importance"] = max(row["importance"], f["importance"])
    frame = pd.DataFrame([{**r, "seeds": len(r["seeds"]), "of_seeds": seeds} for r in rows.values()])
    if frame.empty:
        return frame
    return frame.sort_values(["seeds", "importance"], ascending=False).reset_index(drop=True)


def parse_groups(text: str, n_items: int) -> dict[str, list[int]]:
    """The model's reply as `{group name: [item numbers]}`, tolerant of a code fence. A number
    outside the listing is dropped, and one a group names twice is kept once."""
    body = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    payload = json.loads(body)
    groups = {}
    for group in payload.get("groups", []):
        items = list(dict.fromkeys(int(i) for i in group.get("items", []) if 1 <= int(i) <= n_items))
        if items:
            groups[str(group.get("name") or f"group_{len(groups)}")] = items
    return groups


def placement(groups: dict[str, list[int]], n_items: int) -> tuple[int, str | None]:
    """How many of the `n_items` listed features the grouping placed, and a note for the log where
    it left some out or placed some in more than one group. A feature in two groups counts once
    here, and counts towards each of the two concepts in the tables."""
    memberships = Counter(item for items in groups.values() for item in items)
    found = []
    if len(memberships) < n_items:
        found.append(f"{n_items - len(memberships)} feature(s) were not placed in any group")
    several = sum(1 for count in memberships.values() if count > 1)
    if several:
        found.append(f"{several} feature(s) were placed in more than one")
    return len(memberships), ("note: " + ", and ".join(found)) if found else None


def recurrence_null(features: list[dict], groups: dict[str, list[int]], draws: int,
                    rng: random.Random) -> dict[str, float]:
    """P(a concept reaches this many real seeds by chance), holding the group sizes fixed.

    **Without this the real column cannot be interpreted.** An arm that keeps several times as many
    features on real labels as on shuffled ones will show more real seeds per concept for that
    reason alone, whatever the concepts are. The null shuffles which features have which
    (condition, seed) tag, preserving how many of each tag there are and how large each group is,
    and counts how often chance alone reaches the observed count.
    """
    tags = [(f["labels"], f["seed"]) for f in features]
    observed = {}
    for name, items in groups.items():
        observed[name] = len({tags[i - 1][1] for i in items if tags[i - 1][0] == "real"})
    at_least = dict.fromkeys(groups, 0)
    for _ in range(draws):
        shuffled = tags[:]
        rng.shuffle(shuffled)
        for name, items in groups.items():
            reached = len({shuffled[i - 1][1] for i in items if shuffled[i - 1][0] == "real"})
            if reached >= observed[name]:
                at_least[name] += 1
    return {name: round(count / draws, 4) for name, count in at_least.items()}


def summarise(features: list[dict], groups: dict[str, list[int]]) -> pd.DataFrame:
    """One row per concept: which seeds contain it, under each label condition.

    - `real_seeds`, `permuted_seeds`: how many seeds of each label condition hold a member of it.
    - `real_features`, `permuted_features`: how many features of each label condition fell into it.
    - `p_enriched`: a hypergeometric test of whether it holds more real features than drawing its
      members blind would give, given how many features each label condition contributed overall.
    - `example`: one member's name, so that the row can be understood at a glance.

    Each pass adds `p_chance`, how often chance alone reaches the concept's number of real seeds at
    its size (`recurrence_null`). A row of one arm's pass also names its `arm`, whereas a row of the
    pass across arms gives how many arms reached the concept under each label condition
    (`real_arms`, `permuted_arms`) and the arms that reached it on real labels (`arms`), from
    `arm_coverage`.
    """
    seeds = defaultdict(set)
    for condition in ("real", "permuted"):
        for feature in features:
            if feature["labels"] == condition:
                seeds[condition].add(feature["seed"])
    # Two different questions, of which the second is more informative. Seed coverage measures
    # whether a concept turns up again across reseeding, whereas **enrichment tests whether the
    # labels produced it**: given how many features each condition contributed overall, is this
    # concept more real-heavy than drawing its members blind would give? An arm keeping several
    # times as many features on real labels reaches every real seed in any large concept, so
    # coverage alone says little.
    population = len(features)
    successes = sum(1 for f in features if f["labels"] == "real")
    rows = []
    for name, items in groups.items():
        members = [features[i - 1] for i in items]
        counts = {c: len({m["seed"] for m in members if m["labels"] == c})
                  for c in ("real", "permuted")}
        n_real = sum(1 for m in members if m["labels"] == "real")
        rows.append({
            "concept": name,
            "real_seeds": counts["real"], "permuted_seeds": counts["permuted"],
            "real_features": n_real,
            "permuted_features": len(members) - n_real,
            "p_enriched": round(float(hypergeom.sf(n_real - 1, population, successes, len(members))), 4),
            "example": members[0]["name"],
        })
    frame = pd.DataFrame(rows).sort_values(["real_seeds", "real_features"], ascending=False)
    frame.attrs["n_seeds"] = {c: len(v) for c, v in seeds.items()}
    return frame


class GroupingFailed(RuntimeError):
    """A grouping call that gave nothing to count: the call failed, or its reply held no groups."""


def group_features(features: list[dict], args, context, client, label: str):
    """One grouping call over `features`. Returns the shuffled listing, the groups and the counts,
    or None under `--dry-run`, which makes no call. A call that gives no groups raises
    `GroupingFailed`.

    Both passes share it, so that a concept means the same thing however it was scoped: the model
    sees a numbered list of names and descriptions and nothing else.
    """
    # The features are shuffled and numbered, so the model cannot infer the seed, the condition or
    # the arm from the order they are listed in.
    order = list(range(len(features)))
    random.Random(args.seed).shuffle(order)
    shuffled = [features[i] for i in order]
    listing = "\n".join(
        f"{i}. {f['name']}" + (f" - {f['description']}" if f["description"] else "")
        for i, f in enumerate(shuffled, start=1))
    system = build_system(context, args.items_per_group)

    by_condition = {c: sum(1 for f in features if f["labels"] == c) for c in ("real", "permuted")}
    print(f"  {label}: {len(features)} features (" +
          ", ".join(f"{v} {k}" for k, v in by_condition.items()) + ")")
    if not by_condition.get("permuted"):
        print(f"    WARNING: no permuted features for {label}, so the real counts have no null.",
              file=sys.stderr)

    if args.dry_run:
        print("\n--- system ---\n" + system + "\n\n--- user ---\n" + listing)
        return None

    tokens = reply_tokens(len(shuffled), args.items_per_group)
    completion = client.complete(system, listing, max_completion_tokens=tokens,
                                 timeout=reply_seconds(tokens))
    if not completion.ok:
        raise GroupingFailed(f"the grouping call for {label} failed: {completion.error}")
    try:
        groups = parse_groups(completion.text, len(shuffled))
    except (ValueError, TypeError) as error:
        raise GroupingFailed(f"the grouping for {label} could not be parsed: {error}") from error
    if not groups:
        raise GroupingFailed(f"the model returned no groups for {label}.")
    return shuffled, groups, by_condition


def arm_coverage(features: list[dict], groups: dict[str, list[int]]) -> pd.DataFrame:
    """Which arms reached each concept, under each condition.

    This is meaningful only when one grouping pass covered several arms. A concept defined in one
    arm's pass cannot be compared with one defined in another's, however alike the two labels look,
    because they come from separate clusterings whose members were never compared.
    """
    rows = []
    for name, items in groups.items():
        members = [features[i - 1] for i in items]
        real = sorted({m["arm"] for m in members if m["labels"] == "real"})
        permuted = sorted({m["arm"] for m in members if m["labels"] == "permuted"})
        rows.append({"concept": name, "real_arms": len(real), "permuted_arms": len(permuted),
                     "arms": " | ".join(real)})
    return pd.DataFrame(rows)


def analyse_arm(results: Path, arm: str, args, context, client) -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    """One arm: group its features by meaning, then report both views of recurrence."""
    features = collect(results, arm, set(args.kinds), args.top_n)
    if len(features) < 2:
        print(f"  {arm}: {len(features)} feature(s) of kind {args.kinds}; skipped.")
        return None

    got = group_features(features, args, context, client, arm)
    if got is None:
        return None
    shuffled, groups, by_condition = got

    concepts = summarise(shuffled, groups)
    n_seeds = concepts.attrs["n_seeds"]
    concepts["p_chance"] = concepts.concept.map(
        recurrence_null(shuffled, groups, args.draws, random.Random(args.seed)))
    concepts.insert(0, "arm", arm)
    exact = exact_recurrence(features)

    grouped, note = placement(groups, len(features))
    record = {
        "arm": arm, "kinds": list(args.kinds), "top_n": args.top_n,
        "n_features": len(features), "n_grouped": grouped, "n_concepts": len(groups),
        "seeds": n_seeds, "features_by_condition": by_condition,
        "n_features_recurring_by_name": int((exact.seeds > 1).sum()) if not exact.empty else 0,
        "n_concepts_recurring": int((concepts.real_seeds > 1).sum()),
        "concepts": concepts.to_dict(orient="records"),
        # Written so that the grouping can be checked on the terms the model saw.
        "key": [{"item": i, "name": f["name"], "labels": f["labels"], "seed": f["seed"]}
                for i, f in enumerate(shuffled, start=1)],
    }
    if note:
        print(f"    {note}")
    return record, concepts, exact


def analyse_across_arms(results: Path, arms: list[str], args, context,
                        client) -> tuple[dict, pd.DataFrame] | None:
    """Every arm's features in **one** grouping pass, so concepts are comparable between arms.

    Whereas the per-arm pass answers "did this arm think of it again?", this pass answers "did more
    than one arm think of it?", which the per-arm pass cannot, because it groups each arm
    separately: two of its labels that look alike are the grouping model choosing the same words
    twice, which does not mean the members match.

    The question also differs from that of `analyse_co_discovery.py`, which pairs arms by how their
    feature *values* correlate on a shared reference split. That needs every arm scored on the same
    rows, and it cannot see two arms computing one idea over different windows, whereas this pass
    sees the idea and never looks at a value.
    """
    features = [f for arm in arms for f in collect(results, arm, set(args.kinds), args.top_n)]
    if len({f["arm"] for f in features}) < 2:
        print("  across arms: fewer than two arms have features; skipped.")
        return None

    got = group_features(features, args, context, client, "across arms")
    if got is None:
        return None
    shuffled, groups, by_condition = got

    concepts = summarise(shuffled, groups)
    n_seeds = concepts.attrs["n_seeds"]
    concepts["p_chance"] = concepts.concept.map(
        recurrence_null(shuffled, groups, args.draws, random.Random(args.seed)))
    concepts = concepts.merge(arm_coverage(shuffled, groups), on="concept")
    concepts = concepts.sort_values(["real_arms", "real_seeds"], ascending=False)

    grouped, note = placement(groups, len(features))
    record = {
        "arms": list(arms), "kinds": list(args.kinds), "top_n": args.top_n,
        "n_features": len(features), "n_grouped": grouped, "n_concepts": len(groups),
        "seeds": n_seeds, "features_by_condition": by_condition,
        "n_concepts_multi_arm": int((concepts.real_arms > 1).sum()),
        "n_concepts_all_arms": int((concepts.real_arms == len(arms)).sum()),
        "concepts": concepts.to_dict(orient="records"),
        "key": [{"item": i, "name": f["name"], "arm": f["arm"], "labels": f["labels"],
                 "seed": f["seed"]} for i, f in enumerate(shuffled, start=1)],
    }
    if note:
        print(f"    {note}")
    return record, concepts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Arms to group, each as model/pool. Without it, every arm of the "
                             "run whose pool holds a written source, which are the ones an "
                             "exact-name count misreads. How they are grouped is in the run's "
                             "config (`analyses.top_n` and `analyses.recurrence`).")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt, make no calls.")
    parser.add_argument("--force", action="store_true",
                        help="Group again although the tables on file were computed from the same "
                             "features, settings and description.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    results = args.run_dir / RESULTS_DIR
    args.arms = args.arms or written_arms(results)
    if not args.arms:
        print("no arm of this run is fitted on a pool that holds a written source, so there is "
              "nothing to group")
        return 0
    # The description is the one the run's config names, so the grouper is told about the dataset
    # the features were proposed for. How many features are taken, and how they are grouped, are
    # the run's settings too.
    try:
        config = load_recorded(args.run_dir)
        context = load_task(DESCRIPTIONS, config.dataset.task)
    except (ConfigError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    stated = config.analyses.recurrence
    args.top_n, args.kinds = config.analyses.top_n, list(stated.kinds)
    args.items_per_group, args.draws, args.seed = stated.items_per_group, stated.draws, stated.seed
    # Not built under --dry-run: the client demands credentials at construction, and the flag exists
    # to read the prompt on a machine that has none.
    client = None if args.dry_run else LiteLLMClient(config.language_model)
    # A grouping is a language model's answer, so it is asked for once: a call that would be
    # computed from what the tables on file were computed from keeps them (`inputs`). The model
    # and the effort it reasons at are part of what is compared. A dry run does not compare
    # anything with the file.
    settings = {key: value for key, value in vars(args).items() if key not in NOT_SETTINGS}
    if client is not None:
        settings["language_model"] = reply_conditions(client.describe())
    computed_from = inputs.digest(results, settings, context)
    if not (args.dry_run or args.force) and inputs.on_file(results / RECORD) == computed_from:
        print(f"{RECORD} was computed from these features, these settings, this description and "
              f"this model, so the tables on file are kept. --force groups again.")
        return 0

    print(f"grouping {len(args.arms)} arm(s), top {args.top_n or 'all'} features per cell:")
    records, concept_frames, exact_frames, failed = [], [], [], []
    for arm in args.arms:
        try:
            got = analyse_arm(results, arm, args, context, client)
        except GroupingFailed as error:
            print(f"    {error}", file=sys.stderr)
            failed.append(arm)
            continue
        if got is None:
            continue
        record, concepts, exact = got
        records.append(record)
        concept_frames.append(concepts)
        exact_frames.append(exact)
    across = None
    if stated.across_arms:
        try:
            across = analyse_across_arms(results, args.arms, args, context, client)
        except GroupingFailed as error:
            print(f"    {error}", file=sys.stderr)
            failed.append("across arms")
    if args.dry_run:
        return 0
    if not records:
        print("No arm produced a grouping.", file=sys.stderr)
        return 1

    results.mkdir(parents=True, exist_ok=True)
    concepts = pd.concat(concept_frames, ignore_index=True)
    exact = pd.concat([f for f in exact_frames if not f.empty], ignore_index=True)
    concepts.to_csv(results / BY_CONCEPT_TABLE, index=False)
    exact.to_csv(results / BY_NAME_TABLE, index=False)
    # Which model grouped the features, and how.
    payload = {"arms": records, "language_model": client.describe(), "usage": client.usage()}
    if across is not None:
        across_record, across_concepts = across
        across_concepts.to_csv(results / ACROSS_ARMS_TABLE, index=False)
        payload["across_arms"] = across_record
    # Tables that lack a grouping state no digest, so the next call groups again.
    if not failed:
        payload[inputs.KEY] = computed_from
    files.write_json(results / RECORD, payload)
    record_call(args.run_dir, CALL, vars(args))

    if across is not None:
        across_record, across_concepts = across
        print(f"\nacross arms: {across_record['n_features']} features from "
              f"{len(across_record['arms'])} arms -> {across_record['n_concepts']} concepts; "
              f"{across_record['n_concepts_multi_arm']} reached by more than one arm, "
              f"{across_record['n_concepts_all_arms']} by all of them")
        print(across_concepts.head(15).to_string(index=False))

    for record in records:
        arm = record["arm"]
        print(f"\n{arm}: {record['n_features']} features -> {record['n_concepts']} concepts; "
              f"{record['n_features_recurring_by_name']} recur by name, "
              f"{record['n_concepts_recurring']} concepts recur")
        rows = concepts[concepts.arm == arm]
        print(rows.drop(columns="arm").to_string(index=False))
    print("\nfeatures kept by condition is the first thing to read: an arm keeping many more on real")
    print("labels has already said something, and the per-concept columns then ask whether anything")
    print("further is there. p_enriched is the chance of a concept being this real-heavy given those")
    print("totals; p_chance is the chance of it reaching this many real seeds given its size.")
    print("\nThe by-name and by-concept counts differ most for arms on a written source, since those")
    print("give one idea another name at each seed. Read the two together.")
    print("\nConcepts from the per-arm tables are NOT comparable between arms: each arm is grouped")
    print("on its own, so two labels reading alike is the grouper choosing the same words twice.")
    print(f"{ACROSS_ARMS_TABLE} is the one pass that can say two arms reached the same idea.")
    if failed:
        print(f"\nNo grouping for {', '.join(failed)}: the tables above lack it. Run the analysis "
              f"again once the cause is found.", file=sys.stderr)
        return 1
    return 0

