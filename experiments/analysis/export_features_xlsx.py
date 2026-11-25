#!/usr/bin/env python
"""Write a finished run's reported features to a spreadsheet, for people who want to sort and filter
them.

    $LOCALUTILS experiments/analysis/export_features_xlsx.py --run-dir experiments/runs/<date>_<name>

Runs under `SeqMaestro_LocalUtils`. `--help` describes the sheets, and
`seqmaestro/report/spreadsheet.py` holds them.
"""

from seqmaestro.report.spreadsheet import main

if __name__ == "__main__":
    raise SystemExit(main())
