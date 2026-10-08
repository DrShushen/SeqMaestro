#!/usr/bin/env python
"""Test discovered features against the label directly, matched and permuted within a confounder.

    experiments/analysis/direct_effect_test.py --dataset data/<dataset>/<file>.csv \
        --label-column <target> --features <cell>/features.json --group-column <confounder>

`--help` describes the test, and `seqmaestro/pipeline/analyses/direct_effect.py` holds it.
"""

from seqmaestro.pipeline.analyses.direct_effect import main

if __name__ == "__main__":
    raise SystemExit(main())
