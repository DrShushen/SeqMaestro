#!/usr/bin/env python
"""Expand a report source into the finished `REPORT.md`. Runs in either environment (Core or
LocalUtils).

    python experiments/analysis/assemble_report.py --report-dir <run>/report

Reads `REPORT.src.md` from the report directory, fills its placeholders, and writes `REPORT.md`
beside it. Components come from `seqmaestro/report/components/`, tables from `tables/`, figures from
`figures.json` and parameters from `facts.json` (see `seqmaestro/report/placeholders.py`). An
unresolved placeholder stops it. Otherwise it lists the tables and figures the source did not
include, so the writer can check that nothing was left out by accident.

The table of the agents that carried out the run is rendered again first, from the run's record
(`seqmaestro.pipeline.agents`), since the agent that writes the report records itself after the
report's inputs were built. It warns about a run that records no agent for the report, since the
report would then not name its writer.

`render_pdf.sh` turns the result into a PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from seqmaestro.pipeline import agents
from seqmaestro.report import tables
from seqmaestro.report.files import (
    COMPONENTS_DIR,
    FACTS_FILE,
    FIGURES_FILE,
    REPORT_FILE,
    SOURCE_FILE,
    TABLES_DIR,
)
from seqmaestro.report.placeholders import PlaceholderError, Sources, expand, unused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.report_dir
    source = out / SOURCE_FILE
    if not source.exists():
        print(f"No report source at {source}. Write it against process/report-template.md first.",
              file=sys.stderr)
        return 2

    facts = json.loads((out / FACTS_FILE).read_text())
    # The report directory is the run's `report/`.
    recorded = agents.recorded(out.parent)
    (out / TABLES_DIR / f"{tables.AGENTS_TABLE}.md").write_text(tables.agents(recorded) + "\n")
    if not any("report" in entry["stages"] for entry in recorded):
        print(f"WARNING: no agent recorded itself for the report in {out.parent}, so the report "
              f"names no writer. Run experiments/runners/record_agent.py --stages report first.",
              file=sys.stderr)
    sources = Sources(components=COMPONENTS_DIR, tables=out / TABLES_DIR,
                      figures=json.loads((out / FIGURES_FILE).read_text())["figures"],
                      parameters=facts["parameters"], target_kind=facts["target_kind"])
    try:
        text = expand(source.read_text(), sources)
    except PlaceholderError as error:
        print(f"{source}: {error}", file=sys.stderr)
        return 1
    (out / REPORT_FILE).write_text(text)
    print(f"wrote {out / REPORT_FILE}  ({len(text.splitlines())} lines, {len(sources.used)} inclusions)")
    for kind, names in unused(sources).items():
        if names:
            print(f"{kind}s not included anywhere: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
