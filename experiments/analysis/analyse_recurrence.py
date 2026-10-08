#!/usr/bin/env python
"""Recurrence across a run's seeds, in the two senses an arm can have it: the same feature, and the
same idea. **Credentials required**: `set -a && source .env && set +a`.

    experiments/analysis/analyse_recurrence.py --run-dir experiments/runs/<date>_<name>

`--help` describes the analysis, and `seqmaestro/pipeline/analyses/recurrence.py` holds it.
"""

from seqmaestro.pipeline.analyses.recurrence import main

if __name__ == "__main__":
    raise SystemExit(main())
