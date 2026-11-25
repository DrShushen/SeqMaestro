"""The evaluation harness, which scores the arms of every kind of model with the same code.

For a binary target:

```python
from seqmaestro.harness.metrics import classification_metrics, choose_threshold

t = choose_threshold(y_val, p_val)                  # picked on validation
result = classification_metrics(y_val, p_val, threshold=t, n_bootstrap=1000)
```

`result` is a plain dict, which a cell's `metrics.json` holds as its validation block. The same `t`
is applied to the test split once, at the reveal. A continuous target has no threshold and is
scored by `regression_metrics`. `seqmaestro.target` assigns each kind of target one of the two.
`seqmaestro.harness.heldout` is the only module that reads the test split once Stage 1 has made it.

The package's `__init__` imports none of its modules, because importing them all would create a cycle:
`seqmaestro.target` is built on the arithmetic modules here (`metrics`, `effect`, `power`,
`direction`), whereas `runner`, `bundle`, `heldout` and `importance` import `seqmaestro.target`.
"""
