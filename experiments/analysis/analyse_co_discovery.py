#!/usr/bin/env python
"""Which discoveries of a run are corroborated by several arms, and which belong to one arm alone.

    experiments/analysis/analyse_co_discovery.py --run-dir experiments/runs/<date>_<name>

`--help` describes the analysis, and `seqmaestro/pipeline/analyses/co_discovery.py` holds it.
"""

from seqmaestro.pipeline.analyses.co_discovery import main

if __name__ == "__main__":
    raise SystemExit(main())
