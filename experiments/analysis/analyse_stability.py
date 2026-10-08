#!/usr/bin/env python
"""Stability of every arm of a run across seeds, against its permuted-label null.

    experiments/analysis/analyse_stability.py --run-dir experiments/runs/<date>_<name>

`--help` describes the analysis, and `seqmaestro/pipeline/analyses/stability.py` holds it.
"""

from seqmaestro.pipeline.analyses.stability import main

if __name__ == "__main__":
    raise SystemExit(main())
