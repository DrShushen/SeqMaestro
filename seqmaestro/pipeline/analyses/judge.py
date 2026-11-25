"""Score a run's discovered features on biological significance.

The command reads the `features.json` of every cell under the run's `results/`, puts the top N
features of each real-label cell on one panel that hides which arm each came from, asks the judge
`How biologically significant or interesting would this feature be, if it were true?`, and writes
the scores and a summary under the run's `judge/`.

    python experiments/analysis/run_judge.py --run-dir experiments/runs/<date>_<name>
    python experiments/analysis/run_judge.py --run-dir ... --dry-run      # prompts only, no calls

**Credentials required** for a real run: `set -a && source .env && set +a`.

The judge is told the task description the run's recorded config names
(`experiments/datasets/<task>.json`), the file every other consumer of the dataset reads.

**How the judge rates is the run's setting**, taken from its recorded config: the number of
features taken from each cell (`analyses.top_n`), the ratings per feature, the label conditions
rated and whether the code is shown (`analyses.judge`), and whether the judge is told which way a
feature runs (`grid.judge_direction`). This command has no flag for any of them, so a judge started
by hand rates as the driver's does. Only real-label features are rated in the shared config.

**The scores on file are kept** when the judge is started again while the features, the settings
and the description they were computed from are unchanged (`inputs.py`), and `--force` rates again.

`docs/developer/analyses/judge.md` describes the protocol, and why the permuted cells' features are
not rated by default.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from seqmaestro import files
from seqmaestro.arms import arm_name
from seqmaestro.harness import cell as cells
from seqmaestro.harness.features import load_features
from seqmaestro.judge import (
    LiteLLMEngine,
    PanelSource,
    build_panel,
    direction_shown,
    panel_key,
    render,
    score_panel,
    summarise,
)
from seqmaestro.judge.files import PANEL_KEY, SCORES, SUMMARY
from seqmaestro.llm import load_task, reply_conditions
from seqmaestro.pipeline.analyses import inputs
from seqmaestro.pipeline.config import ConfigError, load_recorded
from seqmaestro.pipeline.layout import JUDGE_DIR, RESULTS_DIR
from seqmaestro.pipeline.repository import DESCRIPTIONS
from seqmaestro.provenance import record_call
from seqmaestro.target import recorded_kind

#: The name of this call (`seqmaestro.pipeline.layout.call_name`). What it writes under the run's
#: `judge/` is named in `seqmaestro.judge.files`: the summary a report reads, every score, and
#: which panel entry is which feature.
CALL = "judge"

#: The arguments that say where and whether to run, and the share of answers needed for the scores
#: to be usable. The others are the settings a score depends on.
NOT_SETTINGS = ("run_dir", "dry_run", "force", "out", "min_answered")


def collect_sources(results: Path) -> list[PanelSource]:
    """Every arm's features, one source per cell of the run's `results/`:
    `<model>/<pool>/<labels>/seed_<n>/`.

    Only a cell's own `features.json` is read. A vendored arm's cell also holds the bundle its
    method handed back, with a file of the same name whose header states no kind of target, and
    reading that as a cell would judge the arm's features twice.
    """
    sources = []
    for features_file in sorted(results.glob(f"*/*/*/seed_*/{cells.FEATURES_FILE}")):
        cell = features_file.parent
        model, pool, labels = cell.relative_to(results).parts[:3]
        records, header = load_features(features_file)
        if records:
            sources.append(PanelSource(arm=arm_name(model, pool), labels=labels, records=records,
                                       seed=int(cell.name.removeprefix("seed_")),
                                       target_kind=recorded_kind(header)))
    return sources


def why_unusable(items, report: dict, min_answered: float, expect_control: bool = True) -> str | None:
    """Why the judge's scores cannot be reported, or None if they can.

    A judge that answered nothing must not report success. When every call fails (in an outage,
    for example), `judge_summary.json` is still written, with `control.mean = None` and zero
    tokens. Exiting 0 would then make the driver record the stage as passing, and missing scores
    would look the same as a judge that was not asked to run.

    It is separate from `main`, so that it can be tested without a run directory or a network.
    """
    if not items:
        return "the panel was empty, so nothing was judged."

    scored = [item for item in items if item.mean is not None]
    answered = len(scored) / len(items)
    if answered < min_answered:
        usage = report.get("usage") or {}
        attempted = (usage.get("calls") or 0) + (usage.get("failures") or 0)
        return (
            f"{len(scored)} of {len(items)} panel features received a score "
            f"({answered:.1%}, minimum {min_answered:.0%}); "
            f"{usage.get('failures') or 0} of {attempted} calls failed. "
            f"Check connectivity and credentials first."
        )

    # An empty control is a failure only when a control was asked for. Judging permuted features is
    # off by default (see the module docstring), and then an absent control is expected.
    if expect_control and not (report.get("control") or {}).get("n_features"):
        return ("the control pool is empty, so no arm's mean is interpretable. Features from "
                "permuted-label cells are the noise floor every judge score is compared with.")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run whose features are rated, as its config states "
                             "(`analyses.top_n`, `analyses.judge`, `grid.judge_direction`)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the panel and print a prompt; make no API calls.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write; defaults to <run-dir>/judge/")
    parser.add_argument("--force", action="store_true",
                        help="Rate again although the scores on file were given to the same "
                             "features under the same settings and description.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # The description is the one the run's config names, so the judge is told about the dataset
    # the features were proposed for. A judge told about another dataset would not fail, but would
    # score with confidence against data it was never shown. Whether the judge is told which way
    # each feature runs is the run's setting too (`grid.judge_direction`), since it changes what a
    # score means, so a judge started by hand rates as the driver's does.
    try:
        config = load_recorded(args.run_dir)
        context = load_task(DESCRIPTIONS, config.dataset.task)
    except (ConfigError, ValueError) as error:
        parser.error(str(error))
    stated = config.analyses.judge
    args.include_direction = config.grid.judge_direction
    args.top_n, args.repeats, args.seed = config.analyses.top_n, stated.repeats, stated.seed
    args.include_code, args.min_answered = stated.show_code, stated.min_answered
    args.label_conditions = list(stated.labels)

    results_root = args.run_dir / RESULTS_DIR
    if not results_root.exists():
        print(f"No results under {results_root}.", file=sys.stderr)
        return 1

    sources = collect_sources(results_root)
    if not sources:
        print(f"No features.json found under {results_root}.", file=sys.stderr)
        return 1

    wanted = set(args.label_conditions)
    dropped = [source for source in sources if source.labels not in wanted]
    sources = [source for source in sources if source.labels in wanted]
    if dropped:
        print(f"judging {sorted(wanted)} only; {len(dropped)} arm-cell(s) left unjudged")
    if not sources:
        print(f"No features.json under {results_root} for label conditions {sorted(wanted)}.",
              file=sys.stderr)
        return 1

    panel = build_panel(sources, top_n=args.top_n, seed=args.seed)
    n_calls = len(panel) * args.repeats
    print(f"{len(sources)} arm-cells, {len(panel)} features on the panel, "
          f"{args.repeats} repeats -> {n_calls} calls")
    if "permuted" in wanted and not any(source.labels == "permuted" for source in sources):
        print("  WARNING: permuted-label features were asked for and none were found. Without them "
              "there is no noise floor, and no arm's mean is interpretable.", file=sys.stderr)

    # The features with a direction are counted whatever the setting, because the count is a
    # property of the panel. The setting alone cannot say what the judge saw, since a feature with
    # no marginal effect goes out without a direction, so the count is recorded beside the setting,
    # and together they describe what the judge was shown.
    with_direction = sum(direction_shown(entry.record, entry.target_kind) is not None
                         for entry in panel)
    if args.include_direction:
        print(f"  {with_direction}/{len(panel)} panel features carry a direction to show")
        if not with_direction:
            print("  WARNING: the run's grid has judge_direction on and no feature on the panel "
                  "has a marginal effect, so every prompt will go out without one.",
                  file=sys.stderr)

    if args.dry_run:
        shown = render(panel[0], context, args.include_code, args.include_direction)
        print("\n--- system ---\n" + shown.system)
        print("\n--- user (first item) ---\n" + shown.user)
        print(f"\n(dry run: {n_calls} calls not made)")
        return 0

    # A rating is a language model's answer, so it is asked for once: a call that would be
    # computed from what the scores on file were computed from keeps them (`inputs`). The model
    # and the effort it reasons at are part of what is compared.
    engine = LiteLLMEngine(config.language_model)
    out = args.out or (args.run_dir / JUDGE_DIR)
    settings = {key: value for key, value in vars(args).items() if key not in NOT_SETTINGS}
    settings["language_model"] = reply_conditions(engine.describe())
    computed_from = inputs.digest(results_root, settings, context)
    if not args.force and inputs.on_file(out / SUMMARY) == computed_from:
        print(f"{SUMMARY} was computed from these features, these settings, this description and "
              f"this model, so the scores on file are kept. --force rates again.")
        return 0

    def progress(index, total, item):
        mean = f"{item.mean:.1f}" if item.mean is not None else "--"
        print(f"[{index}/{total}] {item.item_id} {item.feature[:44]:<44} {mean}", flush=True)

    items = score_panel(panel, engine, context, repeats=args.repeats,
                        include_code=args.include_code,
                        include_direction=args.include_direction, progress=progress)
    report = summarise(items, context)
    report["settings"] = {
        "top_n": args.top_n, "repeats": args.repeats,
        "include_code": args.include_code, "include_direction": args.include_direction,
        "features_with_direction": with_direction, "seed": args.seed,
    }
    # Which model was asked, and how.
    report["language_model"] = engine.describe()
    report["usage"] = engine.usage()

    # A judge that answered nothing must not report success (see `why_unusable`). Its outputs are
    # written all the same, so the evidence is kept, and they state no digest, so the next call
    # rates again.
    unusable = why_unusable(items, report, args.min_answered,
                            expect_control="permuted" in wanted)
    if not unusable:
        report[inputs.KEY] = computed_from
    out.mkdir(parents=True, exist_ok=True)
    files.write_json(out / SUMMARY, report, default=str)
    files.write_json(out / SCORES, [asdict(item) for item in items], default=str)
    # What each id the judge was shown stands for, with the seed and the importance, which the
    # scores file leaves out.
    files.write_json(out / PANEL_KEY, panel_key(panel), default=str)
    record_call(args.run_dir, CALL, vars(args))

    control = (report.get("control") or {}).get("mean")
    print(f"\ncontrol (permuted) mean: {control}" if control is not None
          else "\nno permuted control judged; arm means describe their own features only")
    for name, arm in report["arms"].items():
        gap = f"  gap {arm['control_gap']}" if "control_gap" in arm else ""
        print(f"  {name:<44} mean {arm['mean']}{gap}  "
              f"spread {arm['mean_spread_across_repeats']}")
    print(f"\nWritten to {out}   ({report['usage'].get('total_tokens', 0):,} tokens)")

    if unusable:
        print(f"\nFAILED: {unusable}\n  The scores written to {out} are partial and must not be "
              f"reported as the judge's verdict.", file=sys.stderr)
        return 1
    return 0

