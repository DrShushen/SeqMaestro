from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class BaseDatasetPlugin(ABC):
    """
    Pluggable dataset interface used by DEFT.

    To support a new dataset, add one class implementing this interface and reference it from
    `src/deft/conf/dataset/<name>.yaml` under `plugin._target_`. No manual registration is needed.
    """

    @abstractmethod
    def load_for_deft(self) -> dict[str, Any]:
        """Return the DEFT training bundle.

        The plugin is handed exactly the rows DEFT should train on. Splitting is the caller's
        job -- see `src/deft/data/generic_csv_plugin.py` for why.

        Required keys:
            X_train          DataFrame of features. May hold a sequence column of equal-length
                             strings, covariate columns, or both.
            y_train          DataFrame with a single target column: the labels 0 and 1
                             for a binary target, numbers for a continuous one
            tree_filename    filename for the fitted tree artifact, e.g. "tree_foo.dill"
            prompt_context   dict interpolated into the generation prompts; {} if unused

        Optional keys:
            sequence_column  name of the column holding the sequence, or None for a dataset
                             with no sequence at all. Defaults to "raw_sequence" if absent.
                             Reaches `get_naive_features`, which needs it to emit positional
                             nucleotide features.
            target_kind      `binary` or `continuous` (`deft.targets`). Binary if absent, the
                             only kind a plugin handed before regression; a tree grown for it
                             refuses a target holding anything but 0 and 1.
        """


def check_column_names(columns: Any) -> None:
    """Every feature column must be named as a legal Python identifier.

    DEFT's method is to have a language model write Python that references columns by name, so
    a name Python cannot express is a column DEFT cannot use. The failure is worse than that,
    though, because it is indirect and silent: the model is shown the column names, adopts their
    conventions, and writes them into the *function* names it chooses. A dataset carrying
    `seq_-50` taught it to propose `AT_fraction_upstream_-10_to_-1`, which does not parse -- so
    the generation was discarded after its tokens had been paid for, and nothing in the result
    said a feature had been lost.

    This is checked at load time, and refused, rather than being worked around. A generic
    sanitiser would have to guess what an illegal character meant, and it cannot: in one real
    dataset `-` is a minus sign in `dist_gene_end_-_start`, a coordinate sign in `seq_-50`, and
    a depletion marker in `COND:-Spt5`. Three meanings, three different right answers, all of
    them domain knowledge that belongs to whoever prepared the data.

    Python keywords are allowed: a column named `class` is only ever referenced as a string
    subscript, so it costs nothing.
    """
    illegal = [c for c in columns if not (isinstance(c, str) and c.isidentifier())]
    if not illegal:
        return

    shown = ", ".join(repr(c) for c in illegal[:8])
    more = f" (and {len(illegal) - 8} more)" if len(illegal) > 8 else ""
    raise ValueError(
        f"{len(illegal)} column name(s) are not legal Python identifiers: {shown}{more}.\n"
        f"DEFT generates Python that refers to columns by name, and the language model copies "
        f"the naming conventions it is shown into the function names it writes -- so an "
        f"illegal name silently costs whole generations.\n"
        f"Rename these columns in your data before passing it to DEFT. Doing it here is not "
        f"possible without guessing what each character meant."
    )


def instantiate_dataset_plugin(dataset_cfg: Any) -> BaseDatasetPlugin:
    """Instantiate dataset plugin."""
    plugin_cfg = None
    if isinstance(dataset_cfg, dict):
        plugin_cfg = dataset_cfg.get("plugin")
    elif hasattr(dataset_cfg, "get"):
        try:
            plugin_cfg = dataset_cfg.get("plugin")
        except Exception:
            plugin_cfg = None
    if plugin_cfg is None:
        plugin_cfg = getattr(dataset_cfg, "plugin", None)

    if plugin_cfg is None:
        raise ValueError(
            "Dataset config is missing required `plugin` block. "
            "Define `dataset.plugin._target_` in src/deft/conf/dataset/<name>.yaml."
        )

    plugin = instantiate(plugin_cfg)
    if not isinstance(plugin, BaseDatasetPlugin):
        raise TypeError(
            "Dataset plugin config must instantiate a `BaseDatasetPlugin`, "
            f"got: {type(plugin)}"
        )
    return plugin
