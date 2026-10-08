#!/usr/bin/env python
"""Draw a report's figures. **Runs in `SeqMaestro_LocalUtils`, not `SeqMaestro_Core`.**

    $LOCALUTILS experiments/analysis/build_report_figures.py --report-dir <run>/report

It needs matplotlib, which no result depends on, so matplotlib stays out of the environment that
produces results (ENVIRONMENTS.md). This script therefore reads one file, the `figure_inputs.json`
that `build_report_inputs.py` wrote under Core, and never reads the run. Every number, label and
caption a figure carries was decided there.

Writes `figures/<name>.png` and `figures.json`. `figures.json` records each figure's path, caption,
width, measured aspect and whether it needs a rotated page, and `assemble_report.py` and
`render_pdf.sh` read it. A figure wider than `ROTATE_ABOVE` cannot be read at a portrait column's
width, so it gets a rotated page of its own instead of being shrunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seqmaestro.provenance import record_call
from seqmaestro.report.files import FIGURE_INPUTS_FILE, FIGURES_CALL, FIGURES_DIR, FIGURES_FILE

#: Above this width-to-height ratio a figure cannot be read at a portrait column width. A rotated
#: A4 page with the render's margins holds about 1.4, so the cut sits a little below where a figure
#: would start to overflow it.
ROTATE_ABOVE = 1.6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="the run's report directory, holding figure_inputs.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Here rather than at the top, so the parser can be read where the plotting stack is absent.
    from PIL import Image

    from seqmaestro.report.figures import draw

    out = args.report_dir
    record_call(out.parent, FIGURES_CALL, vars(args))
    inputs = json.loads((out / FIGURE_INPUTS_FILE).read_text())

    registry = {}
    for name, entry in inputs.items():
        target = draw(entry, out / FIGURES_DIR / f"{name}.png")
        with Image.open(target) as image:
            aspect = image.size[0] / image.size[1]
        registry[name] = {"path": f"{FIGURES_DIR}/{target.name}", "caption": entry["caption"],
                          "width": entry["width"], "aspect": round(aspect, 3),
                          "rotate": aspect > ROTATE_ABOVE}
        print(f"figure -> {target}  (aspect {aspect:.2f}{', rotated page' if aspect > ROTATE_ABOVE else ''})",
              flush=True)
    for stale in (out / FIGURES_DIR).glob("*.png"):
        if stale.stem not in registry:
            stale.unlink()
    (out / FIGURES_FILE).write_text(json.dumps({"figures": registry}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
