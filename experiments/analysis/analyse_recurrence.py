#!/usr/bin/env python
"""Recurrence across a run's seeds and arms, in two senses: the same feature found again, and the
same idea arrived at again. **Credentials required**: `set -a && source .env && set +a`.

    experiments/analysis/analyse_recurrence.py --run-dir experiments/runs/<date>_<name>

`--help` describes the analysis, and `seqmaestro/pipeline/analyses/recurrence.py` holds it.
"""

from seqmaestro.pipeline.analyses.recurrence import main

if __name__ == "__main__":
    raise SystemExit(main())
