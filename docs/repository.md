# How the repository is laid out

For someone working on the code, or an agent finding its way. A user who wants to install
SeqMaestro and run it starts from the [`README.md`](../README.md) at the root.

| | |
|---|---|
| `seqmaestro/` | the shared library: featurisation, the evaluation harness, the proposer, the judge, and the stages of a run (`seqmaestro/pipeline/`). Knows about no dataset |
| `methods/` | vendored third-party methods, each standing alone in its own environment. Nothing here imports `seqmaestro` |
| `experiments/` | the commands that prepare datasets, drive runs and read them back, with the shared run configs, the adapters that run a vendored method, and one example dataset (see [`experiments/README.md`](../experiments/README.md)) |
| `docs/` | what each stage and analysis computes, and why; the report's template and explanations |
| `process/` | one instruction file per stage, for an agent running the process |
| `.claude/skills/seqmaestro/` | the brief an agent is given when a user starts a run with Claude Code (`/seqmaestro`). It points at `process/` |
| `install.sh` | builds the conda environments and checks them. [`ENVIRONMENTS.md`](../ENVIRONMENTS.md) is the record of what each one holds |
| `tests/` | the suite of the library and the pipeline, run in `SeqMaestro_Core`. Each vendored method has a suite of its own under `methods/` |
| `data/`, `experiments/runs/` | datasets and run records. **Not in git**, so anything worth citing is copied somewhere durable when it is written |
| the files of one dataset | its builder, declaration, description and configs under `experiments/`, and the tests of its builder. **Not in git**: they stay on the machine that holds the dataset, and the example dataset's are the ones the repository holds ([`experiments/prepare/README.md`](../experiments/prepare/README.md)) |

Two boundaries shape most of the design. **A vendored method and the library never import each
other.** They exchange files, so their environments can disagree about every package they contain.
**A result is kept separate from the code that produced it**: the code is identified by the commit
it ran at, and the result by its own recorded provenance.

## Where each kind of fact is written down

| kind of fact | where |
|---|---|
| how to install, configure and start a run | [`../README.md`](../README.md) |
| what a stage or analysis computes, and why | this folder, from [`README.md`](README.md) |
| how scripts and configs are organised and run | [`../experiments/README.md`](../experiments/README.md) and the READMEs beside it |
| what an agent does and decides at each stage | [`../process/README.md`](../process/README.md) |
| the conda environments, and which runs what | [`../ENVIRONMENTS.md`](../ENVIRONMENTS.md) |
