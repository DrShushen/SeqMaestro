"""The judge: a language model that rates how significant each discovered feature would be, if it held.

```python
panel = build_panel(sources, top_n=20, seed=0)          # select, hide the arms, shuffle
items = score_panel(panel, engine, context, repeats=3)  # independent calls per feature
report = summarise(items, context)                      # per-arm means, and any noise floor
```

The judge is an instrument to be characterised, and its scores are not ground truth
(`docs/developer/analyses/judge.md`). The scoring backend is behind the `ScoringEngine` interface,
so that the protocol can be tested without a credential, and another model can be swapped in without
changing it.
"""

from __future__ import annotations

from seqmaestro.judge.engine import (
    LiteLLMEngine,
    ScriptedEngine,
    parse_reply,
)
from seqmaestro.judge.panel import (
    PanelSource,
    build_panel,
    panel_key,
)
from seqmaestro.judge.rubric import (
    MAX_SCORE,
    MIN_SCORE,
    TaskContext,
    direction_shown,
    render,
)
from seqmaestro.judge.scoring import DEFAULT_REPEATS, score_panel, summarise

__all__ = [
    "DEFAULT_REPEATS",
    "MAX_SCORE",
    "MIN_SCORE",
    "LiteLLMEngine",
    "PanelSource",
    "ScriptedEngine",
    "TaskContext",
    "build_panel",
    "direction_shown",
    "panel_key",
    "parse_reply",
    "render",
    "score_panel",
    "summarise",
]
