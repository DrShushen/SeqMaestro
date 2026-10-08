#!/usr/bin/env python
"""How much of each feature is the thing a split groups on, checked before a run.

    experiments/analysis/feature_group_dependence.py --dataset data/<dataset>/<file>.csv \
        --sequence-column <sequence> --group-column <group> --pools positional

`--help` describes the check, and `seqmaestro/qc/group_dependence.py` holds it.
"""

from seqmaestro.qc.group_dependence import main

if __name__ == "__main__":
    raise SystemExit(main())
