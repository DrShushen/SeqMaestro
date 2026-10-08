"""The copy's configuration, composed in process.

Upstream composes its configuration through a Hydra app (`experiments/igsr/run.py`), which also
loads the dataset by name and runs the seeds in a process pool. The copy is called in process with
a dataset (`igsr.method.igsr.igsr(cfg, seed, data_bundle)`), so a caller composes the configuration
here, with Hydra's own override syntax, and no Hydra run directory is made.
"""

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONF_DIR = Path(__file__).resolve().parent / "conf"


def compose_config(overrides: Sequence[str] = ()) -> DictConfig:
    """`conf/config.yaml` with `overrides`, e.g. `["llm=seqmaestro_azure", "experiment.n_iters=5"]`.
    The language model has no default, so `llm=<name>` is required."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))
