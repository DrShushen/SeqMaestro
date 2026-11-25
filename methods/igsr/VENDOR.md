# Where this copy came from

| | |
|---|---|
| Upstream | `git@github.com:DrShushen/IGSR.git`, branch `main` |
| Commit | `44af1f48ce0881987cc996657b5e5c60b8661e4f` ("Initial commit", 2026-07-02) |
| Copied | 2026-09-26, from `git archive 44af1f4` of the paths below; the working tree was clean at that commit |
| Changes since | summarised in [`README.md`](README.md#what-changed) |
| Licence | MIT ([`LICENSE`](LICENSE)) |

There is no nested `.git`, no remote and no submodule. SeqMaestro keeps the history of the copy
from here on.

## What was copied

- From `src/igsr/`: the method (`method/igsr.py`, `method/igsr_utils.py`), the agent (`agent/`),
  the tree search (`tree_search/problem.py`, `tree_search/session.py`), the dataset bundle
  (`dataset/__init__.py`, `dataset/data_bundle.py`), and `formula_tracker.py`, `parser.py`,
  `util.py`, `logging_setup.py`, `retry_decor.py`, `compute_profiler.py`, `config_typing.py` and
  `const.py`.
- From `experiments/igsr/conf/`: `config.yaml`, `experiment/experiment_base.yaml`,
  `experiment/igsr.yaml` and `llm/azure_base.yaml`, now under `src/igsr/conf/`, where the package
  ships them.
- From `tests/`: `conftest.py`, `test_drop_columns.py`, `igsr/method/test_igsr.py`,
  `igsr/agent/test_agent_retry.py`, `igsr/tree_search/test_tree_search.py` and
  `igsr/util/test_compute_profiler.py`.

## What was left out

The baselines under `src/igsr/method/` (zero-shot, in-context, LLM-SR, ICSR, LaSR with its Julia
library, PySR, SINDy, gplearn, XGBoost, D3 and the neural baselines), the dataset loaders
(`dataset/util.py`, `dataset/real/`, `dataset/synthetic/`), `analysis/`, `cpu_mutex.py`,
`gpu_util.py`, the tree search's visualiser, the experiment scripts with their other configs, the
data, and the tests of those parts.

## Comparing with upstream

```bash
git -C <IGSR checkout> archive 44af1f4 src/igsr experiments/igsr/conf tests | tar -x -C <empty dir>
diff -ru <empty dir>/src/igsr methods/igsr/src/igsr
```

A copied file that no commit of this repository has touched is byte-identical to its upstream
original.
