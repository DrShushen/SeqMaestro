#!/usr/bin/env python
"""Expand a report source into the finished `REPORT.md`. Runs in either environment.

    python experiments/analysis/assemble_report.py --report-dir <run>/report

Reads `REPORT.src.md` from the report directory, fills its placeholders, and writes `REPORT.md`
beside it. Components come from `docs/report/components/`, tables from `tables/`, figures from
`figures.json` and parameters from `facts.json` (see `seqmaestro/report/placeholders.py`). An
unresolved placeholder stops it. It then lists the tables and figures the source did not include,
so the writer can check that nothing was left out by accident.

`render_pdf.sh` turns the result into a PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from seqmaestro.report.files import FACTS_FILE, FIGURES_FILE, REPORT_FILE, SOURCE_FILE, TABLES_DIR
from seqmaestro.report.placeholders import PlaceholderError, Sources, expand, unused

REPO = Path(__file__).resolve().parents[2]
COMPONENTS = REPO / "docs" / "report" / "components"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.report_dir
    source = out / SOURCE_FILE
    if not source.exists():
        print(f"No report source at {source}. Write it against docs/report/TEMPLATE.md first.",
              file=sys.stderr)
        return 2

    facts = json.loads((out / FACTS_FILE).read_text())
    sources = Sources(components=COMPONENTS, tables=out / TABLES_DIR,
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
