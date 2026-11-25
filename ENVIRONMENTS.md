# Environments

Each part of SeqMaestro runs in a conda environment of its own. This file records which environments exist, what each one runs, why they are kept apart and how they are built. The commands it shows are the recipe that [`install.sh`](install.sh) runs, so the two are changed together.

Versions, counts and timings are left out of this file, because they would go stale without anyone noticing. Instead, each run records the versions installed in the environments it used ([recording an environment](#recording-an-environment)), so this file needs editing only when an environment changes. How the code of each vendored method differs from its upstream is in [`methods/deft/README.md`](methods/deft/README.md#what-changed) and [`methods/igsr/README.md`](methods/igsr/README.md#what-changed).

## The environments

| Environment | Contains | Runs |
|---|---|---|
| `SeqMaestro_Core` | `seqmaestro[models,llm,driver,cli,dev,docs]` | Featurisation, the models fitted in Core, the harness, the [proposer](docs/glossary.md#proposer), the [judge](docs/glossary.md#judge) and the analyses: every [stage](docs/glossary.md#stage) of a [run](docs/glossary.md#run) but the report's figures. Also the command line (`seqmaestro`), the root `tests/`, most scripts under `experiments/`, and the documentation's build |
| `SeqMaestro_Method_deft` | `deft[dev]`, and `scikit-learn` for the adapter | DEFT's suite, and the [arms](docs/glossary.md#arm) of DEFT's tree (`deft_or_cart`: CART, and DEFT on a pool with `llm_generated_joint`) |
| `SeqMaestro_Method_igsr` | `igsr[dev]` | IGSR's suite, and the IGSR arm |
| `SeqMaestro_Method_pysr` | `pysr`, with Julia, its packages and its project inside the environment | The search of a PySR model, in a process that the model starts from Core ([`symbolic_search.py`](seqmaestro/models/symbolic_search.py)) |
| `SeqMaestro_Method_meme` | The MEME Suite (`meme` from bioconda), with Perl and the modules it runs on | STREME and Tomtom, which motif discovery runs ([`discovery.py`](seqmaestro/motifs/discovery.py)). Built only on request (`--with-motifs`), and needed only by a run whose pool holds `motif_discovered` |
| `SeqMaestro_Method_alphagenome` | `alphagenome`, AlphaGenome's API client | [`predict_alphagenome_summaries.py`](experiments/prepare/predict_alphagenome_summaries.py), which summarises AlphaGenome's predictions for a dataset's elements into candidate covariates at Stage 1, by calls to AlphaGenome's API. Built only on request (`--with-alphagenome`) |
| `SeqMaestro_LocalUtils` | `seqmaestro[localutils]` | A report's figures, and spreadsheets for collaborators. The numbers of a run do not depend on it, but a run taken through the reveal needs it for its last stage, which draws the report's figures, so the [driver](docs/glossary.md#driver) refuses to start such a run without it |

- **All seven are built on Python 3.12** ([package constraints](#package-constraints) gives the reason).
- **Four are editable installs of this working tree**: Core, LocalUtils, DEFT's and IGSR's. A change to the code therefore needs no reinstall, and only a change to a `pyproject.toml` does.
- **The other three install nothing from this tree.** PySR's and AlphaGenome's environments each run one file of it, loaded by its path, and the MEME Suite's programs are run by their paths.
- **DEFT's, IGSR's, PySR's and the MEME Suite's environments run `seqmaestro.provenance` from the repository root**, without SeqMaestro installed, to write their snapshots for a run that uses them ([recording an environment](#recording-an-environment)).
- **[`pyproject.toml`](pyproject.toml) lists the packages of each extra of `seqmaestro`**, with the reason for each in its comments. Only the command line imports the packages of the `cli` extra, and the package's `seqmaestro` script, which pip puts in Core's `bin/`, starts it.

**Naming.** A method's environment is named `SeqMaestro_Method_<method_name>`. `SeqMaestro_Core` and `SeqMaestro_LocalUtils` are the two exceptions, because neither is a method.

## Building the environments

`./install.sh`, at the repository root, builds the environments and is their recipe. It runs the commands below against whichever conda it finds, keeps an environment that exists, and checks the result (`./install.sh --check` makes the checks alone):

```bash
./install.sh                        # the five environments a run needs
./install.sh --with-motifs          # and SeqMaestro_Method_meme, for discovered motifs
./install.sh --with-alphagenome     # and SeqMaestro_Method_alphagenome
./install.sh --only pysr            # one of them
```

What it runs for each environment, with `$ENVS` standing for the folder in which this conda keeps its environments:

```bash
conda create -y -n SeqMaestro_Core --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_Core/bin/python -m pip install -e ".[models,llm,driver,cli,dev,docs]"     # from the repo root

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

conda create -y -n SeqMaestro_Method_meme --override-channels -c conda-forge -c bioconda \
    python=3.12 meme=5.5.9

conda create -y -n SeqMaestro_Method_alphagenome --override-channels -c conda-forge python=3.12 pip
$ENVS/SeqMaestro_Method_alphagenome/bin/python -m pip install "alphagenome==0.9.0"
```

The script runs these commands as follows:

- **Python comes from conda-forge, named as the only channel**, so that an environment is built from the same packages whichever conda a machine has, and no channel's terms have to be accepted first. Everything else comes from PyPI, apart from the MEME Suite, whose environment alone names [bioconda](https://bioconda.github.io/) as a second channel.
- **Each environment's interpreter runs pip**, with the packages under the home directory and the calling shell's `PYTHONPATH` kept out, so that what is installed is what is declared.
- **On macOS, `SeqMaestro_Core` is created with `llvm-openmp`**, because XGBoost's wheel for macOS loads the OpenMP runtime from outside itself.
- **The calling shell's proxy and certificate variables are passed through `env -i`**, so that the Julia download works behind a proxy.

**The language model's credentials** come from the `.env` at the repository root, loaded with `set -a && source .env && set +a`. It names the language model's provider (Azure OpenAI, Azure AI Foundry, the OpenAI API or the Claude API) with its key and model ([the language model](docs/user/language-model.md)). The code in Core, DEFT's and IGSR's environments calls the model, each through [LiteLLM](https://docs.litellm.ai/). AlphaGenome's environment needs AlphaGenome's key, which its script reads from the variable that `--key-variable` names. The other environments make no API calls.

## Keeping the environments apart

**SeqMaestro and a vendored method never import each other.** They run in separate processes and exchange files, so that their environments can disagree about every package they contain ([two boundaries](docs/developer/architecture.md#two-boundaries)). [`tests/test_boundaries.py`](tests/test_boundaries.py) fails if anything under `seqmaestro/` imports a vendored method, or if a vendored method imports `seqmaestro`. PySR's search is kept apart in the same way: it runs as a process in its own environment, and the rows and the equations it finds pass between the processes as files and messages.

**The packages of the environments do not conflict** (checked 2026-08-06). The environments are kept apart for failure isolation and attribution:

- Core is the environment that produces results, so a package that the results do not depend on is kept out of it, and therefore out of the snapshot of every result.
- DEFT's numbers should come from a minimal declared environment, and a package upgraded for DEFT or for IGSR must not be able to change a number of a model fitted in Core.

> An environment is added only when a version constraint makes coexistence **impossible**, or when it keeps something *out* of an environment that produces results. The reason is recorded below, with its evidence and the date it was checked.

An anticipated conflict does not justify a new environment. Every environment besides Core exists for the second reason.

### `SeqMaestro_Method_deft`

**`scikit-learn` is installed in DEFT's environment, although DEFT does not declare it.** The adapter needs it and DEFT does not: [`deft_bundle.py`](experiments/adapters/deft_bundle.py) runs there and needs AUROC (or R² for a continuous target) for the permutation importance that every arm reports. [`methods/deft/pyproject.toml`](methods/deft/pyproject.toml) leaves it out, so that DEFT's declaration states only what DEFT needs. Installing it changes no package DEFT depends on: `numpy` and `pandas` keep their versions, and the only packages added are `scipy`, `joblib`, `narwhals`, `threadpoolctl` and `scikit-learn` itself, so a DEFT number cannot shift because of it. It is recorded here because a package installed in an environment without being declared anywhere is the failure this file exists to prevent.

### `SeqMaestro_LocalUtils`

Its packages would install into Core without incident (`openpyxl`, for example). LocalUtils is kept apart because Core is the environment that produces results, and a package that the results do not depend on should not appear in their environment snapshot. Reporting tools accumulate, and each one added to Core would widen what a finished run has to be reproduced against.

- **An analysis script that needs LocalUtils says so in its docstring**, as [`export_features_xlsx.py`](experiments/analysis/export_features_xlsx.py) and [`build_report_figures.py`](experiments/analysis/build_report_figures.py) do.
- **A report is built in both environments.** Its numbers, its tables and the JSON its figures are drawn from are built under Core, which reads the run, whereas the figures themselves are drawn under LocalUtils, from that JSON alone.
- **Only `seqmaestro/report/figures/` imports `matplotlib`**, and only LocalUtils runs that code. [`tests/test_report.py`](tests/test_report.py) checks that nothing else in `seqmaestro/` imports it. Installing `matplotlib` into LocalUtils moved no existing package (checked 2026-09-24).

### `SeqMaestro_Method_pysr`

**PySR is kept out of Core** (checked 2026-09-27), because it would bring a second language runtime and its package manager into the environment that produces results. A Core process that had started Julia could also no longer fork the workers of Tier 1 of the [reliability panel](docs/glossary.md#reliability-panel) safely, since a fork copies only the calling thread and Julia runs several.

**Julia stays inside the environment.** [PySR](https://github.com/MilesCranmer/PySR) is a Python front end to a [Julia](https://julialang.org/) package. Importing it starts Julia inside the Python process (through `juliacall`), and [juliapkg](https://github.com/JuliaPy/pyjuliapkg) decides where Julia, its packages and its project live:

- By default, juliapkg takes a Julia from the path, or installs one through juliaup under the home directory, and keeps the packages under `~/.julia`.
- It puts the project in whichever conda environment is active in the calling shell (its `CONDA_PREFIX`), or under `~/.julia` when none is. An interpreter started by its path does not make its own environment the active one, so the project would end up in the caller's environment.

Four variables therefore point everything into the environment itself:

| Variable | Points at |
|---|---|
| `JULIA_DEPOT_PATH` | `<prefix>/share/julia`: the packages, registries and compiled code |
| `PYTHON_JULIAPKG_PROJECT` | `<prefix>/julia_env`: the Julia project, and the Julia binary juliapkg installed |
| `PYTHON_JULIAPKG_EXE` | `<prefix>/julia_env/pyjuliapkg/install/bin/julia`: that binary |
| `PYTHON_JULIAPKG_OFFLINE` | `yes`: never resolve again at run time |

The recipe sets the first three while it builds, since resolving needs the network. Anything that imports PySR in this environment must set all four first. The search process sets them itself from its environment's prefix (`julia_variables` in [`symbolic_search.py`](seqmaestro/models/symbolic_search.py)), and refuses to start when one is already set to anything else, so that however it is started, it does not write to the home directory. Removing the environment then removes all of Julia with it. After a build, check that nothing reached the home directory: `~/.julia` and `~/.juliaup` are as they were, and there is no `~/.pyjuliapkg.json`.

**PySR's Julia packages are resolved by juliapkg**, and pip and conda do not see them. PySR 2.5.1 declares SymbolicRegression.jl and a Julia version range in its own `juliapkg.json`, and juliapkg resolves them into a Julia project at first import. The recipe pins Julia 1.11.9, the line that PySR's own container uses. PySR precompiles only its 32-bit code, which is the precision its models search in. A later PySR needs a new environment. PySR's own saved models refuse to load under another version, but the models a finished run saved are unaffected, because SeqMaestro keeps no PySR object after a fit.

### `SeqMaestro_Method_meme`

The [MEME Suite](https://meme-suite.org/meme/) is written in C and Perl, and published on bioconda alone. Its package brings Perl with some sixty of its modules (checked 2026-10-17), ICU, libxml2 and libxslt, which only its programs need. Results depend on them only through the output of [STREME](https://meme-suite.org/meme/doc/streme.html) and [Tomtom](https://meme-suite.org/meme/doc/tomtom.html), which passes to Core as files, so they are kept out of Core, the environment that produces results.

- It is the only environment built from two channels, conda-forge and bioconda, since bioconda is the only channel that publishes the MEME Suite. Neither channel asks for terms to be accepted.
- The MEME Suite is free for educational, research and non-profit use, whereas commercial use needs a licence from UC San Diego ([its copyright page](https://meme-suite.org/meme/doc/copyright.html)). The installer fetches it, and SeqMaestro does not ship any of it.
- Scanning sequences for motifs, which every later stage needs, is SeqMaestro's numpy code in Core. Only the step that discovers motifs needs this environment, and a run's reveal, reduction and panel do not.

### `SeqMaestro_Method_alphagenome`

The client of [AlphaGenome](https://github.com/google-deepmind/alphagenome) brings a plotting and single-cell stack (`matplotlib`, `seaborn`, `anndata`, `zarr`) and a gRPC stack (`grpcio`, `protobuf`), which only the client needs (checked 2026-09-28). Results do not depend on any of them, so they are kept out of Core, the environment that produces results.

- The environment is used at Stage 1 only, once per dataset. Its predictions are written into `data/`, and runs take them from there as covariates.
- The client is pinned because it holds AlphaGenome's record of which folds each fold model held out, and the held-out predictions choose each element's model from that record. The regions of each fold come from the file that `--folds` names, whose digest the predictions record.
- The model itself runs on AlphaGenome's servers and cannot be pinned, so each prediction records the client's version and the time it was made.

## Package constraints

Each constraint below decides how the environments are built. The comments of [`pyproject.toml`](pyproject.toml) give the reasons for the packages that SeqMaestro declares, such as the choice of `interpret-core` over `interpret`.

- **Python 3.12 is the floor**, because `xgboost` 3.3.0 and later require it, and `litellm` caps the ceiling below 3.15. All seven environments are at 3.12. IGSR's upstream was tested on 3.10, but its core dependencies resolve on 3.12 and its suite passes there (checked 2026-09-26).
- **`hydra-core` pins `antlr4-python3-runtime==4.9.*`.** The pin is rigid, but only DEFT and IGSR need hydra.
- **`litellm` has the largest dependency tree of any package here**, with wide direct dependencies that include `openai`, `tokenizers`, `aiohttp` and `pydantic<3`. It is the main reason why a method that needs it stays in its own environment.
- **`xgboost` pulls `nvidia-nccl-cu13` on Linux**, a large wheel for training on several GPUs, which nothing here uses. `xgboost-cpu` avoids it if size ever matters.
- **`statsmodels` is in the `dev` extra, for the tests only.** The reliability panel implements its BH, BY and Holm corrections itself ([`multitest.py`](seqmaestro/reliability/multitest.py)), and the tests check them against `statsmodels`. Installing it into Core moved no existing package (checked 2026-09-23), so the numbers did not change.

## Programs outside conda

Rendering a report as a PDF needs [pandoc](https://pandoc.org/) and `xelatex` (from [TeX Live](https://www.tug.org/texlive/)), which are system programs, and the DejaVu fonts that [`render_pdf.sh`](experiments/analysis/render_pdf.sh) names, which are system fonts. A report with figures also needs a `python3` on the path: the script uses it to read the record of the figures beside the report, and needs nothing beyond Python's standard library. To check for the programs and the font:

```bash
command -v pandoc xelatex python3  # /usr/bin/pandoc, /usr/bin/xelatex
fc-list | grep 'DejaVu Serif'      # the font the render script asks for
```

This is the only part of SeqMaestro that depends on something outside conda, and it fails on a machine without TeX Live. A Python wrapper would not help, because `pypandoc` calls the same program. `./install.sh --check` says whether `pandoc` and `xelatex` are installed. If a report will not build, check them and the fonts first.

## Tests and linting

Each package is tested and linted from its own folder, with its own interpreter and its own config, because mixing them produces a result that belongs to neither. The seven checks that must pass before a commit are in the [developer guide](docs/developer/index.md#checks-before-a-commit). The commands below also show how to run each method's offline tests and its live tests apart:

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

- The commands run each environment's interpreter by its path, without activating the environment, so `ruff` is run as `python -m ruff`.
- The PySR environment has no test suite of its own. The code it runs is in `seqmaestro/`, and Core's suite tests it, with `crossenv` tests that start it under `$PYSR`.
- The AlphaGenome environment has no test suite either. Its script imports the client only inside the functions that call it, so Core's suite can load the script and test everything else ([`tests/test_predict_alphagenome_summaries.py`](tests/test_predict_alphagenome_summaries.py)).
- **A `pytest` run from the repository root must never reach `methods/`**, or a method's suite runs under the wrong interpreter and reports mostly green. `norecursedirs` in the root `pyproject.toml` prevents it, whereas `testpaths` alone does not, because pytest falls back to a recursive search when `testpaths` matches nothing. [`tests/test_boundaries.py`](tests/test_boundaries.py) tests both facts.

## Finding the other interpreters

A stage that runs code in another environment (every DEFT and IGSR step, a PySR model's search, motif discovery and a report's figures) asks `environment_python` in [`seqmaestro/provenance.py`](seqmaestro/provenance.py) for that environment's interpreter. It looks for `<conda root>/envs/<environment>/bin/python` under the conda installation of the interpreter running it. On a machine laid out differently, point it elsewhere with one variable per environment:

```bash
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_DEFT=/path/to/python
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_IGSR=/path/to/python
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_PYSR=/path/to/python
export SEQMAESTRO_PYTHON_SEQMAESTRO_METHOD_MEME=/path/to/python   # STREME and Tomtom sit beside it
export SEQMAESTRO_PYTHON_SEQMAESTRO_LOCALUTILS=/path/to/python
```

The driver's preflight check names a missing interpreter, and the variable that sets it, before any stage starts. `./install.sh` creates every environment in one conda, so that they sit side by side, and `./install.sh --check` asks a Core interpreter whether it finds each of the others.

## Recording an environment

The record of an environment has four layers, from what each package declares in its `pyproject.toml` to the snapshot of what was installed when a run used the environment, and [`seqmaestro/provenance.py`](seqmaestro/provenance.py) describes each layer. This file is the second layer, the recipe. Only a snapshot shows what was actually installed, which can differ from the declaration, because a package stays installed in an environment after its declaration drops it, until the environment is rebuilt.

Snapshots are taken by script, so they cannot drift:

```bash
RUN=experiments/runs/<date>_<name>
$CORE -m seqmaestro.provenance SeqMaestro_Core        $RUN/environment/core
$DEFT -m seqmaestro.provenance SeqMaestro_Method_deft $RUN/environment/deft
$IGSR -m seqmaestro.provenance SeqMaestro_Method_igsr $RUN/environment/igsr
$PYSR -m seqmaestro.provenance SeqMaestro_Method_pysr $RUN/environment/pysr
$MEME -m seqmaestro.provenance SeqMaestro_Method_meme $RUN/environment/meme
```

- **The stages that fit models write the snapshots themselves** (the baseline, the arms, the reduction and the panel, as well as the step that discovers motifs). Each writes the snapshot of Core and of every other environment it ran code in, once per run ([`records.py`](seqmaestro/pipeline/records.py)). Given another environment's name, `write_snapshot` runs `seqmaestro.provenance` under that environment's interpreter.
- **The files of a snapshot** are listed in the docstring of `write_snapshot`. Its manifest, `environment.json`, holds a digest of the package list, so that two runs can be compared without diffing their package lists by eye.
- **The module runs in any environment without being installed there**, because it imports only the standard library and `seqmaestro.execution`, which imports `psutil` only inside the functions that use it. Run it with the interpreter of the environment you are recording.
- **`SeqMaestro_LocalUtils` is not snapshotted beside a run.** It does not contribute to any number, so recording it would attach an irrelevant list of packages to the record of a result.
- **A snapshot is taken only beside results.** A snapshot with nothing to attribute records nothing, and is one more file to keep consistent with the rest.
