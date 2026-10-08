#!/usr/bin/env python
"""Build everything a run's report is written from, under `SeqMaestro_Core`, after the reveal.

    experiments/analysis/build_report_inputs.py --run-dir experiments/runs/<date>_<name>

`--help` describes what is built, and `seqmaestro/pipeline/report.py` holds it.
"""

from seqmaestro.pipeline.report import main

if __name__ == "__main__":
    raise SystemExit(main())
