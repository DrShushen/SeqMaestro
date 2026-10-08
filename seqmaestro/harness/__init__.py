"""The evaluation harness, which scores every arm with the same code whatever its model.

For a binary target:

```python
from seqmaestro.harness.metrics import classification_metrics, choose_threshold

t = choose_threshold(y_val, p_val)                  # picked on validation
result = classification_metrics(y_val, p_val, threshold=t, n_bootstrap=1000)
```

`result` is a plain dict, which a cell's `metrics.json` holds as its validation block. The same `t`
is applied to the test split once, at the reveal. A continuous target has no threshold and is
scored by `regression_metrics`, and `seqmaestro.target` holds which of the two a kind of target
takes. `seqmaestro.harness.heldout` is the only module that reads the test split once Stage 1 has
made it.

The package imports none of its modules itself. `seqmaestro.target` is built on the arithmetic
modules here (`metrics`, `effect`, `power`, `direction`), and `runner`, `bundle`, `heldout` and
`importance` import `seqmaestro.target`, so a package that imported every module on first use would
import each side from the other.
"""
