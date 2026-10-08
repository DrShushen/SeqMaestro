#!/usr/bin/env python
"""Score a run's discovered features on biological significance. **Credentials required**:
`set -a && source .env && set +a`.

    experiments/analysis/run_judge.py --run-dir experiments/runs/<date>_<name>
    experiments/analysis/run_judge.py --run-dir ... --dry-run      # prompts only, no calls

`--help` describes the judge's run, and `seqmaestro/pipeline/analyses/judge.py` holds it.
"""

from seqmaestro.pipeline.analyses.judge import main

if __name__ == "__main__":
    raise SystemExit(main())
