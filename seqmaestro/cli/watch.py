"""Where a run stands, in the terminal: a summary of what its status page shows.

`seqmaestro status` prints it once, and `seqmaestro run` keeps it on screen while the run it started
goes on. Both read the run through the status page's own gatherer (`seqmaestro.status.gather`), so
the terminal and the page cannot disagree.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from seqmaestro.cli import launch, ui
from seqmaestro.pipeline.config import load_recorded
from seqmaestro.pipeline.layout import CONFIG_FILE, FAILURES_DIR, LOGS_DIR, REDUCED_MODEL, report_dir
from seqmaestro.pipeline.repository import REPO, RUNS
from seqmaestro.pipeline.schedule import JOBS_FILE
from seqmaestro.report.files import REPORT_FILE
from seqmaestro.report.naming import METRICS
from seqmaestro.status import PAGE_FILE, gather

#: Seconds between two looks at a run that is followed.
REFRESH_SECONDS = 3.0
#: Seconds a run that was just started is given to write its first snapshot of its processes.
START_SECONDS = 60.0
#: The processes the summary lists by name; the rest are counted.
RUNNING_SHOWN = 6

#: How each state is marked: its symbol and its colour.
MARKS = {"done": ("✓", "ok"), "running": ("●", "accent"), "waiting": ("○", "hint"),
         "failed": ("✗", "bad"), "stopped": ("■", "warn"), "held": ("■", "warn"),
         "skipped": ("-", "hint")}

#: The colour of the run's state, by its kind (`seqmaestro.status.gather.overall`).
STATE_STYLES = {"running": "accent", "done": "ok", "failed": "bad", "waiting": "warn"}


def runs() -> list[Path]:
    """The runs under `experiments/runs/`, the one that changed last first."""
    found = [path.parent for path in RUNS.glob(f"*/{CONFIG_FILE}")]

    def changed(run_dir: Path) -> float:
        jobs = run_dir / LOGS_DIR / JOBS_FILE
        return (jobs if jobs.exists() else run_dir / CONFIG_FILE).stat().st_mtime

    return sorted(found, key=changed, reverse=True)


def find(run: str | None) -> Path:
    """The run a user names, as a path or by its folder's name under `experiments/runs/`, or the one
    that changed last where none is named."""
    if run is None:
        found = runs()
        if not found:
            raise FileNotFoundError(f"There is no run under {RUNS.relative_to(REPO)} yet.")
        return found[0]
    for candidate in (Path(run), RUNS / run):
        if (candidate / CONFIG_FILE).exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{run} is no run: there is no {CONFIG_FILE} there or under "
                            f"{RUNS.relative_to(REPO)}.")


def look(run_dir: Path) -> dict[str, Any]:
    """Everything the run's status page would show now."""
    return gather.gather(run_dir, load_recorded(run_dir))


def _mark(state: str) -> Text:
    symbol, style = MARKS.get(state, ("?", "hint"))
    return Text(symbol, style=style)


def _stage_rows(seen: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Each stage's name, state and progress, as the summary lists them."""
    stages = {stage["key"]: stage for stage in seen["stages"]}
    one, two, three_a, three_b = stages["1"], stages["2"], stages["3A"], stages["3B"]
    rows_total = sum(split.get("rows", 0) for split in one["splits"].values())
    kind = one.get("target_kind") or "?"
    analyses = two["analyses"]
    rows = [("1  Task construction", one["state"],
             f"{rows_total:,} rows, {kind} target" if rows_total else ""),
            ("2  The suite", two["state"],
             f"{two['cells_done']} of {two['cells_expected']} cells; "
             f"{sum(item['state'] == 'done' for item in analyses)} of {len(analyses)} analyses")]
    if three_a["state"] != "skipped":
        reduced = [arm for arm in three_a["arms"] if arm["state"] == "done"]
        progress = (f"{len(reduced)} of {len(three_a['arms'])} arms reduced to "
                    f"{three_a['n_features']} features")
        best = three_a["best"][0] if three_a["best"] else None
        if best is not None:
            metric = METRICS.get(three_a["headline"], three_a["headline"])
            progress += f"; best {best['label']}, validation {metric} {best[f'val_{REDUCED_MODEL}']:.3f}"
        rows.append(("3A Few-feature models", three_a["state"], progress))
    if three_b["state"] != "skipped":
        arms = three_b["arms"]
        if three_b["revealed_utc"]:
            through = sum(all(part["done"] for part in arm["parts"]) for arm in arms)
            progress = f"revealed; {through} of {len(arms)} arms through Tier 3"
        else:
            sealed = sum(all(part["done"] for part in arm["parts"] if not part["after"])
                         for arm in arms)
            progress = f"{sealed} of {len(arms)} arms profiled; the held-out rows unopened"
        rows.append(("3B Reliability panel", three_b["state"], progress))
    report = seen["report"]
    rows.append(("   The report", "done" if report["written"] else "waiting",
                 "written" if report["written"] else
                 "tables and figures built; the text is to be written" if report["inputs"] else ""))
    return rows


def summary(run_dir: Path, seen: dict[str, Any]) -> Group:
    """The run in a few lines: its state, each stage, what is running, and what failed."""
    overall = seen["overall"]
    state = overall["state"]
    title = Text.assemble((overall["name"], "name"), "  ",
                          (state["word"], STATE_STYLES.get(state["kind"], "name")))
    if overall.get("elapsed"):
        title.append(f"  ({overall['elapsed']})", style="hint")
    stages = Table.grid(padding=(0, 2))
    stages.add_column(no_wrap=True)
    stages.add_column(no_wrap=True)
    stages.add_column()
    for name, stage_state, progress in _stage_rows(seen):
        stages.add_row(_mark(stage_state), name, escape(progress))
    parts: list[Any] = [title, Text(""), stages]
    running = overall["running"]
    if running:
        listed = Table.grid(padding=(0, 2))
        listed.add_column(no_wrap=True)
        listed.add_column()
        listed.add_column(style="hint", no_wrap=True)
        for job in running[:RUNNING_SHOWN]:
            listed.add_row(_mark("running"), escape(job["label"]), job["for"] or "")
        if len(running) > RUNNING_SHOWN:
            listed.add_row("", f"and {len(running) - RUNNING_SHOWN} more", "")
        parts += [Text(""), Text("Running now", style="name"), listed]
    if overall["failures"]:
        parts += [Text(""), Text.assemble(("Failed: ", "bad"),
                                          ", ".join(item["name"] for item in overall["failures"]))]
    parts += [Text(""), Text(f"Status page: {ui.shown(run_dir / PAGE_FILE)}", style="hint")]
    return Group(*parts)


def next_step(run_dir: Path, seen: dict[str, Any]) -> str | None:
    """What to do next with a run that is not running, in a sentence; None while it runs."""
    overall, report = seen["overall"], seen["report"]
    if overall["live"]:
        return None
    config = launch.CONFIGS / f"{overall['name']}.yaml"
    resume = (f"[name]{escape(' '.join(launch.driver(config, run_dir, '--detach')))}[/]"
              if config.exists() else "the driver, given its config and its folder")
    if overall["failures"] or overall["state"]["kind"] == "failed":
        return (f"What failed, with its traceback, is in {ui.shown(run_dir / FAILURES_DIR)}. Once "
                f"that is fixed, this continues the run: {resume}")
    if report["written"]:
        return f"Its report is {ui.shown(report_dir(run_dir) / REPORT_FILE)}."
    if report["inputs"]:
        return f"[name]seqmaestro report {escape(run_dir.name)}[/] writes its report."
    return f"It stopped before its end, and this continues it: {resume}"


def show(run_dir: Path) -> dict[str, Any]:
    """Print where the run stands, and return what was seen."""
    seen = look(run_dir)
    ui.panel(summary(run_dir, seen), ui.shown(run_dir))
    return seen


def _framed(run_dir: Path, seen: dict[str, Any]):
    return ui.boxed(summary(run_dir, seen), ui.shown(run_dir),
                    subtitle="Ctrl+C stops watching; the run goes on")


def follow(run_dir: Path) -> dict[str, Any] | None:
    """Keep the summary on screen until the run's driver stops, and return what was seen last; None
    where the user stopped watching with Ctrl+C, which leaves the run going."""
    waited = 0.0
    while not gather.jobs_of(run_dir)["alive"] and waited < START_SECONDS:
        time.sleep(1)
        waited += 1
    seen = look(run_dir)
    try:
        with Live(_framed(run_dir, seen), console=ui.console, refresh_per_second=2) as live:
            while seen["overall"]["live"]:
                time.sleep(REFRESH_SECONDS)
                seen = look(run_dir)
                live.update(_framed(run_dir, seen))
    except KeyboardInterrupt:
        return None
    return seen
