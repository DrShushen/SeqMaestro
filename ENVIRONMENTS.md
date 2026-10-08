# Environments

**This file records the environments as they are now.** How each vendored method differs from its
upstream is in [`methods/deft/README.md`](methods/deft/README.md) and
[`methods/igsr/README.md`](methods/igsr/README.md), each under "What changed". This file carries no
counts, timings or resolved versions, which go stale without anyone noticing, so it needs editing
only when an environment changes. Run the commands below to see the current state.

**The process boundary is the dependency boundary.** SeqMaestro and a vendored method never import
each other: they exchange **files**. A featuriser writes a *k*-mer matrix, and DEFT reads that CSV
in its own interpreter. The harness scores a serialized tree after the run. Because the interface
is on disk, two environments can disagree about every package they contain without breaking
anything. `tests/test_boundaries.py` fails if anything under `seqmaestro/` imports a vendored
method, or a vendored method imports `seqmaestro`. PySR is not vendored: its search runs in a
process of its own environment, and the rows and the equations it finds cross as files and
messages, in the same way.

---

## The environments

All six are Python 3.12. Four of them are editable installs pointing at this working tree, so
only a `pyproject.toml` change needs a reinstall. `SeqMaestro_Method_pysr` and
`SeqMaestro_Method_alphagenome` install nothing from this tree: each runs one file, loaded by its
path. PySR's interpreter also runs the module `seqmaestro.provenance` from the repository root, to
write the environment's snapshot for a run that fits a PySR model.

| Environment | Contains | Runs |
|---|---|---|
| `SeqMaestro_Core` | `seqmaestro[models,llm,driver,dev]` | featurisation, every model that is not vendored, the harness, the proposer, the judge, the recurrence analysis, every stage of a run but the report's figures, the root `tests/`, and the `experiments/` scripts |
| `SeqMaestro_Method_deft` | `deft[dev]`, plus `scikit-learn` for the adapter | DEFT's suite, and the tree's arms (`deft_or_cart`: CART, and DEFT on a pool with `llm_generated_joint`) |
| `SeqMaestro_Method_igsr` | `igsr[dev]` | IGSR's own suite, and the IGSR arm |
| `SeqMaestro_Method_pysr` | `pysr`, with Julia, its packages and its project inside the environment | the search of a PySR model, in a process the model starts from Core (`seqmaestro/models/symbolic_search.py`) |
| `SeqMaestro_Method_alphagenome` | `alphagenome`, AlphaGenome's API client | `experiments/prepare/predict_alphagenome_summaries.py`: AlphaGenome's predictions for a dataset's elements, summarised into candidate covariates at Stage 1. Calls AlphaGenome's API |
| `SeqMaestro_LocalUtils` | `seqmaestro[localutils]` | reporting after a run and one-off analysis: a report's figures, and spreadsheets for collaborators. No number depends on it. A run taken through the reveal needs it for the last stage, which draws the report's figures, and the driver refuses to start such a run without it |

**The `driver` extra holds `psutil` and `pyyaml`**, which driving a run needs
(`seqmaestro/pipeline/`): the driver probes the machine and measures the processes it starts with
`psutil` (`seqmaestro/execution.py`), and reads a run's config with `pyyaml`. Core installs the
extra. `seqmaestro.execution` imports `psutil` only inside the functions that use it, because the
vendored environments run `seqmaestro.provenance`, which imports that module, without it.

Nothing conflicts between them. They are separate for **failure isolation and attribution**:
DEFT's numbers should come from a minimal declared environment, and a package upgraded for DEFT or
for IGSR must not be able to move a number of a model fitted in Core.

**Why `scikit-learn` is in the DEFT environment but not in DEFT's dependencies.** The adapter needs
it and DEFT does not: `experiments/adapters/deft_bundle.py` runs there and needs AUROC, or R² for a
continuous target, for the cross-arm permutation importance every arm reports. `methods/deft/pyproject.toml` does not list it,
so DEFT's declaration still states only what DEFT needs. Installing it moves nothing DEFT depends
on: `numpy` and `pandas` keep their versions, and the only packages added are `scipy`, `joblib`,
`narwhals`, `threadpoolctl` and `scikit-learn` itself, so no DEFT number can shift because of it.
It is recorded here because a package installed in an environment without being declared anywhere
is the failure this file exists to prevent, and that failure has occurred in this environment
before.

**Why `SeqMaestro_LocalUtils` is separate from Core.** Nothing conflicts, and `openpyxl` would
install into Core without incident. LocalUtils is separate because **Core is the environment that
produces results**, and a package no result depends on should not be able to appear in a result's
environment snapshot. Reporting tooling accumulates (a spreadsheet writer, a plotting stack, perhaps
a document converter later), and each addition to Core would widen what a finished run has to be
reproduced against.

As a result, an `experiments/analysis/` script may need this interpreter instead of Core, and says
so in its docstring, as `export_features_xlsx.py` and `build_report_figures.py` do. A report is
built in both environments. Its numbers, its tables and the JSON its figures are drawn from are
built under Core, which reads the run. The figures themselves are drawn under LocalUtils, from that
JSON alone.

**Why `SeqMaestro_Method_pysr` keeps Julia inside the environment.** PySR is a Python front end to
a Julia package. Importing it starts Julia inside the Python process (through `juliacall`), and
`juliapkg` decides where Julia, its packages and its project live. Left alone, it takes a Julia
from the path or installs one through juliaup under the home directory, and keeps the packages
under `~/.julia`. It puts the project in whichever conda environment is active in the calling shell
(its `CONDA_PREFIX`), or under `~/.julia` when none is. An interpreter started by its path does not
make its own environment the active one, so the project would land in the caller's environment.
Four variables therefore point everything into the environment itself:

| variable | points at |
|---|---|
| `JULIA_DEPOT_PATH` | `<prefix>/share/julia`: the packages, registries and compiled code |
| `PYTHON_JULIAPKG_PROJECT` | `<prefix>/julia_env`: the Julia project, and the Julia binary juliapkg installed |
| `PYTHON_JULIAPKG_EXE` | `<prefix>/julia_env/pyjuliapkg/install/bin/julia`: that binary |
| `PYTHON_JULIAPKG_OFFLINE` | `yes`: never resolve again at run time |

The recipe below sets the first three while it builds, since resolving needs the network. Anything
that imports PySR in this environment must set all four first. The search process sets them itself
from its own prefix, and refuses to start when one is already set to anything else, so no way of
starting it writes to the home directory. Removing the environment then removes all of Julia with
it.

**Why `SeqMaestro_Method_alphagenome` is separate from Core.** The client brings a plotting and
single-cell stack (`matplotlib`, `seaborn`, `anndata`, `zarr`) and a gRPC stack (`grpcio`,
`protobuf`), and no result depends on any of them. Core is the environment that produces results,
so they stay out of it. The environment is used at Stage 1 only, once per dataset. Its predictions
are written into `data/`, and runs take them from there as covariates. The client is pinned
because it holds AlphaGenome's record of which folds each fold model held out, and the held-out
predictions choose each element's model from that record. The regions of each fold come from the
file `--folds` names, whose digest the predictions record. The model itself runs on AlphaGenome's
servers and cannot be pinned, so each prediction records the client's version and the time it was
made.

**Rendering a document to PDF needs no conda environment.** `pandoc` and `xelatex` are system
binaries, and the font `experiments/analysis/render_pdf.sh` names is a system font. A report with
figures also needs a `python3` on the path, with the standard library alone, which the script uses
to read the record of the figures beside the report:

```bash
command -v pandoc xelatex python3  # /usr/bin/pandoc, /usr/bin/xelatex
fc-list | grep 'DejaVu Serif'      # the font the render script asks for
```

This is the only part of the project that depends on something outside conda, and it fails on a
machine without TeX Live. A Python wrapper would not help, because `pypandoc` calls the same
binary. If a report will not build, check `pandoc`, `xelatex` and the font first.

**Naming.** A method's environment is `SeqMaestro_Method_<method_name>`. `SeqMaestro_Core` and
`SeqMaestro_LocalUtils` are the two exceptions, because neither is a method.

**`./install.sh`, at the repository root, builds them**, and is the recipe: it runs the commands
below against whichever conda it finds, keeps an environment that exists, and checks the result
(`./install.sh --check` makes the checks alone).

```bash
./install.sh                        # the five environments a run needs
./install.sh --with-alphagenome     # and SeqMaestro_Method_alphagenome
./install.sh --only pysr            # one of them
```

What it runs for each, with `$ENVS` standing for the directory this conda keeps its environments
in:

```bash
conda create -y -n SeqMaestro_Core --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_Core/bin/python -m pip install -e ".[models,llm,driver,dev]"     # from the repo root

conda create -y -n SeqMaestro_Method_deft --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_Method_deft/bin/python -m pip install -e "methods/deft[dev]" scikit-learn

conda create -y -n SeqMaestro_Method_igsr --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_Method_igsr/bin/python -m pip install -e "methods/igsr[dev]"

conda create -y -n SeqMaestro_Method_pysr --override-channels -c conda-forge python=3.12 pip
PREFIX=$ENVS/SeqMaestro_Method_pysr
$PREFIX/bin/python -m pip install "pysr==2.5.1"
mkdir -p $PREFIX/share/julia $PREFIX/julia_env
# `env -i` hides juliaup, any julia on PATH and the caller's CONDA_PREFIX from juliapkg. Julia is
# pinned in the environment's own juliapkg file, then downloaded into the project, and the packages
# PySR declares are installed into the environment's depot.
env -i HOME=$HOME PATH=$PREFIX/bin:/usr/bin:/bin \
    JULIA_DEPOT_PATH=$PREFIX/share/julia PYTHON_JULIAPKG_PROJECT=$PREFIX/julia_env \
    $PREFIX/bin/python -c 'import juliapkg; juliapkg.require_julia("=1.11.9"); juliapkg.resolve()'
# One import compiles SymbolicRegression.jl into the depot, so the first search does not.
env -i HOME=$HOME PATH=$PREFIX/bin:/usr/bin:/bin \
    JULIA_DEPOT_PATH=$PREFIX/share/julia PYTHON_JULIAPKG_PROJECT=$PREFIX/julia_env \
    PYTHON_JULIAPKG_EXE=$PREFIX/julia_env/pyjuliapkg/install/bin/julia \
    $PREFIX/bin/python -c 'import pysr'

conda create -y -n SeqMaestro_LocalUtils --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_LocalUtils/bin/python -m pip install -e ".[localutils]"     # from the repo root

conda create -y -n SeqMaestro_Method_alphagenome --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_Method_alphagenome/bin/python -m pip install "alphagenome==0.9.0"
```

Four things about how it runs them:

- **Python comes from conda-forge, named as the only channel**, so an environment is built from the
  same packages whichever conda a machine has, and no channel's terms have to be accepted first.
  Everything else comes from PyPI.
- **pip is run by each environment's interpreter**, with packages under the home directory and the
  calling shell's `PYTHONPATH` kept out, so that what is installed is what is declared.
- **On macOS `SeqMaestro_Core` is created with `llvm-openmp`**, because XGBoost's wheel for macOS
  loads the OpenMP runtime from outside itself.
- **The calling shell's proxy and certificate variables are passed through `env -i`**, so the Julia
  download works behind a proxy.

Credentials come from the repo-root `.env`: `set -a && source .env && set +a`. It names the
language model's provider (Azure OpenAI, the OpenAI API or the Claude API) with its key and model,
and `Core`, `Method_deft` and `Method_igsr` call it, each through LiteLLM. `Method_alphagenome`
needs AlphaGenome's key, which its script reads from the variable `--key-variable` names. `LocalUtils` and `Method_pysr` make no
API calls.

Check afterwards that nothing reached the home directory: `~/.julia` and `~/.juliaup` are as they
were, and there is no `~/.pyjuliapkg.json`.

## Running things

Each side is tested and linted from its own directory, with its own interpreter and its own
config. Mixing them produces a result that belongs to neither.

```bash
# Each environment's python: CORE, DEFT, IGSR, PYSR, LOCALUTILS, and AG where it is built.
eval "$(./install.sh --interpreters)"

$CORE -m pytest -q                    # from the repo root; parallel, `-n 0` to disable
$CORE -m ruff check .                 # excludes methods/ by config

cd methods/deft                       # from the repo root
$DEFT -m pytest -m "not llm" -q       # offline only
$DEFT -m pytest -q                    # adds the live-API tests
$DEFT -m ruff check .

cd ../igsr
$IGSR -m pytest -m "not slow" -q      # offline only: the language model is stubbed or scripted
$IGSR -m pytest -m slow -q            # the live-API tests alone
$IGSR -m ruff check .
```

`ruff` is on neither environment's `PATH`, so always run it as `python -m ruff`.

The PySR environment has no suite of its own. The code it runs lives in `seqmaestro/` and is tested
from Core's suite, whose `crossenv` tests start it under `$PYSR`. The AlphaGenome environment has
no suite either. Its script imports the client only inside the functions that call it, so Core's
suite can load the script and test everything else (`tests/test_predict_alphagenome_summaries.py`).

**How the pipeline finds the other interpreters.** A stage that runs code in another environment
(every DEFT and IGSR step, a PySR model's search, and a report's figures under LocalUtils) calls
`seqmaestro.provenance.environment_python(<environment>)`, which looks for
`<conda root>/envs/<environment>/bin/python` under the conda installation of the interpreter running
it. On a machine laid out differently, point it elsewhere with one variable per environment:

```bash
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_DEFT=/path/to/python
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_IGSR=/path/to/python
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_PYSR=/path/to/python
export SEQMAESTRO_PYTHON_SEQMAESTRO_LOCALUTILS=/path/to/python
```

The driver's preflight check names a missing interpreter, and the variable that sets it, before any
stage starts. `./install.sh` creates every environment in one conda, so they sit side by side, and
`./install.sh --check` asks a Core interpreter whether it finds each of the others.

**How many threads a process's fits use.** `SEQMAESTRO_THREADS=<n>` gives a process `n` threads:
XGBoost's and EBM's jobs and PySR's search threads follow it (`seqmaestro/execution.py`). The
numeric libraries' thread pools read other variables as they load (`OMP_NUM_THREADS` and the BLAS
libraries'), so a process started on `n` threads is given all of them, which
`seqmaestro.execution.thread_environment` lists. A process started without the variable runs as the
libraries do by default, on every core. The number changes how fast a fit runs and not what it
fits, for every model but PySR, whose search does not repeat itself at any thread count. Each
invocation record states the number. A reliability panel the driver starts is also handed
`SEQMAESTRO_THREADS_FILE`, a file holding the threads it may use now, which the driver changes as
other processes start and end, and Tier 1's selections follow it.

**A root `pytest` must never reach `methods/`**, or a method's suite runs under the wrong
interpreter and reports mostly green. `norecursedirs` in the root `[tool.pytest.ini_options]`
prevents it. `testpaths` alone does not, because when it matches nothing pytest falls back to a
recursive search. `tests/test_boundaries.py` tests both facts and explains the reasoning.

---

## Recording an environment

An environment is recorded in four layers, each with one job. Treating one layer as another is the
usual source of confusion.

| Layer | Where | Answers |
|---|---|---|
| Declaration | `pyproject.toml`, per package | what a package *needs* |
| Recipe | **this file** | which environments exist and how to build them |
| Snapshot | `experiments/runs/<date>_<name>/environment/<env>/` | what was *actually installed* when something ran |
| Attribution | `environment.json` beside each result | which environment produced *this number* |

Snapshots are taken by script, so they cannot drift:

```bash
RUN=experiments/runs/<date>_<name>
$CORE -m seqmaestro.provenance SeqMaestro_Core        $RUN/environment/core
$DEFT -m seqmaestro.provenance SeqMaestro_Method_deft $RUN/environment/deft
$IGSR -m seqmaestro.provenance SeqMaestro_Method_igsr $RUN/environment/igsr
$PYSR -m seqmaestro.provenance SeqMaestro_Method_pysr $RUN/environment/pysr
```

The stages that fit models (the baseline, the arms, the reduction and the panel) write Core's
snapshot themselves. A run with a DEFT or an IGSR arm also gets that method's, written when its
cells are done, and a run that fits a PySR model gets PySR's.
Each is written by the same call: `write_snapshot` named another environment runs the module under
that environment's own interpreter.

`SeqMaestro_LocalUtils` is **not** snapshotted beside a run. It contributes to no number, so
recording it would attach an irrelevant dependency list to the record of a result.

`seqmaestro.provenance` writes `pip-freeze.txt`, `conda-environment.yml`, `python-version.txt` and
an `environment.json` carrying the interpreter, platform, repo git SHA and a sha256 of the freeze,
so two runs can be compared without diffing package lists by eye. For an environment with a Julia
project it also copies the project's `Project.toml` and `Manifest.toml` into `julia/`, and
`environment.json` names the Julia version that resolved them, since pip knows nothing of Julia's
packages. The module imports only the
standard library, so it runs in any environment without being installed there. Run it with the
interpreter of the environment you are recording.

**Take a snapshot only beside results.** A snapshot with nothing to attribute records nothing, and
is one more file to keep consistent with the rest.

**Declaration and snapshot will differ, and only the snapshot shows it.** A package can stay
installed in an environment after it has been dropped from the declared dependencies, as
`scikit-learn` once did in DEFT's environment. Only a snapshot catches that kind of drift.

---

## Constraints that decide the shape

Each constraint below determines how the environments are built. Resolved versions are not
recorded here, because snapshots record them and a version list in a document goes stale within a
week.

- **Python 3.12 is the floor**, because `xgboost` ≥ 3.3.0 requires it.
  `litellm` caps the ceiling at `<3.15`. All six environments sit at 3.12. IGSR's upstream was
  tested on 3.10; its core dependencies resolve on 3.12 and its suite passes there (checked
  2026-09-26).
- **Use `interpret-core`, not `interpret`.** The meta-package pulls dash, plotly, notebook and
  shap through extras, none of which are needed to fit an EBM and read its shape functions.
- **`hydra-core` pins `antlr4-python3-runtime==4.9.*`** exactly. The pin is rigid, but only DEFT
  and IGSR need hydra.
- **`litellm` has the largest dependency tree** of anything here, with wide direct dependencies
  including `openai`, `tokenizers`, `aiohttp` and `pydantic<3`. It is the main reason a method that
  needs it stays in its own environment.
- **`xgboost` pulls `nvidia-nccl-cu13` on Linux**, a large wheel for multi-GPU that nothing here
  uses. `xgboost-cpu` avoids it if size ever matters.
- **`statsmodels` is in `[dev]`, for the tests only.** The reliability panel's BH, BY and Holm
  corrections are implemented in `seqmaestro/reliability/multitest.py` and checked against it.
  Installing it into Core moved no existing package (checked 2026-09-23), so no number moved.
- **PySR's Julia side is resolved by `juliapkg`, not by pip or conda.** PySR 2.5.1 declares
  SymbolicRegression.jl and a Julia version range in its own `juliapkg.json`, and juliapkg resolves
  them into a Julia project at first import. The recipe pins Julia 1.11.9, the line PySR's own
  container uses, and PySR precompiles only its 32-bit code, which is the precision its models
  search in. A later PySR is a new environment: its saved models refuse to load under another
  version, although SeqMaestro keeps no PySR object after a fit.
- **`matplotlib` is in `[localutils]`, for a report's figures.** Only `seqmaestro/report/figures/`
  imports it, and only `SeqMaestro_LocalUtils` runs that; `tests/test_report.py` checks nothing else
  in `seqmaestro/` does. Installing it into LocalUtils moved no existing package (checked
  2026-09-24).

### When to add another environment

> An environment is added only when a version constraint makes coexistence **impossible**, or when
> it keeps something *out* of an environment that produces results. Either reason is recorded above
> with its evidence and the date it was checked.

An anticipated conflict does not justify a new environment. Checked 2026-08-06: no two packages this
project needs have an unresolvable conflict. `SeqMaestro_Method_deft`, `SeqMaestro_Method_igsr`,
`SeqMaestro_Method_pysr`, `SeqMaestro_Method_alphagenome` and `SeqMaestro_LocalUtils` exist for the
second reason. For the AlphaGenome environment, checked 2026-09-28: its client installs
`matplotlib`, `seaborn`, `anndata`, `zarr`, `grpcio` and `protobuf`, packages that only the client
needs. For the PySR environment, checked 2026-09-27: installed into Core, PySR would bring a second
language runtime and its package manager into the environment that produces results, and a Core
process that had started Julia could no longer fork Tier 1's workers safely, since a fork copies
only the calling thread and Julia runs several.
