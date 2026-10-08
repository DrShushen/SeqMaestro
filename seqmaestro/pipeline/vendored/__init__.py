"""The Core side of the methods that run in another environment.

A vendored method runs under its own interpreter, so this side prepares files, starts the method's
adapter (`experiments/adapters/`) and takes back what it wrote. Nothing here imports a vendored
method.

| module | holds |
|---|---|
| `handover` | what a vendored arm is handed for a cell, how a cell of one is fitted, and prediction on numbered rows |
| `deft` | a DEFT arm's Stage 2 cells, through `deft_bundle.py`, and the settings a run starts DEFT with |
| `deft_cart` | what the later stages ask of DEFT, through `deft_cart.py` |
| `igsr` | an IGSR arm's Stage 2 cells, through `igsr_bundle.py`, and the settings a run starts IGSR with |
| `igsr_terms` | what the later stages ask of IGSR: its terms evaluated, and the fits on them |

`seqmaestro.pipeline.sides` puts these behind one interface for the stages after Stage 2.
"""
