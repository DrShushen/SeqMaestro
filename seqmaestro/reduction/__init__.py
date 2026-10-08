"""Stage 3A: each arm reduced to a model of a few features, on validation only.

The Stage 2 suite says what each arm found. This stage says which N of those features the arm's
model keeps when it may keep only N. It runs for every arm the config reduces, from the real-label
cell of its best validation seed:

- `eliminate`: the recursive elimination, numpy only, shared with the vendored tree's side and with
  the reliability panel's selection step;
- `check`: the comparison of rebuilt columns against the values a cell recorded, also numpy only;
- `arm`: one of our own arms reduced in this process, with its seed, rows, columns and model all
  rebuilt from its Stage 2 cell;
- `outputs`: what a reduced arm leaves behind, as three standard cells the reveal scores.

The stage is `seqmaestro/pipeline/reduction.py`, which `experiments/runners/run_reduction.py` calls,
and `experiments/adapters/deft_cart.py` is the vendored tree's side. `docs/reduction.md` describes
what is computed and why.
"""
