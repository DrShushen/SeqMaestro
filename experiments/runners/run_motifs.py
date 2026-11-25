#!/usr/bin/env python
"""Make a run's motif scans and discoveries once, before the arms that use them.

    experiments/runners/run_motifs.py --run-dir experiments/runs/<date>_<name>

`--help` describes the step, and `seqmaestro/pipeline/motifs.py` holds it.
"""

from seqmaestro.pipeline.motifs import main

if __name__ == "__main__":
    raise SystemExit(main())
