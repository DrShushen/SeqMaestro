"""The blinded significance judge: the third evaluation axis, beside accuracy and stability.

Accuracy and stability measure whether a feature predicts and whether it comes back. The judge asks
one question of every discovered feature:

> **How biologically significant or interesting would this feature be, if it were true?** (1-5)

```python
panel = build_panel(sources, top_n=20, seed=0)          # select, blind, shuffle
items = score_panel(panel, engine, context, repeats=3)  # independent calls per feature
report = summarise(items, context)                      # per-arm means, and any noise floor
```

The judge is an **instrument to be characterised**, and its scores are not ground truth. When
features from permuted-label runs are judged, they sit in the same panel, indistinguishable from
the rest, and an arm's gap above their mean says how far it rises above noise. Judging them is off
by default, for the reasons given in the docstring of `seqmaestro.pipeline.analyses.judge`.

The scoring backend sits behind `ScoringEngine`, so everything above can be tested without a
credential, and a different model can be swapped in without touching the protocol.
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
