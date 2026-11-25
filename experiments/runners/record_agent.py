#!/usr/bin/env python
"""Record in a run the agent that carries out its stages and the model the agent runs on.

    experiments/runners/record_agent.py --run-dir experiments/runs/<date>_<name> \
        --agent "Claude Code" --model claude-opus-5-5 --stages 1 2

`--help` describes it, and `seqmaestro/pipeline/agents.py` holds it.
"""

from seqmaestro.pipeline.agents import main

if __name__ == "__main__":
    raise SystemExit(main())
