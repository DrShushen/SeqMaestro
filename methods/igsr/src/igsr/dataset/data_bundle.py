from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DataBundle:
    """A dataset's splits and what the prompts say about it. The test split may be None: the search
    never chooses anything by it, and evaluation on it is reporting only. `column_descriptions`
    describes input columns by name, for the function form's prompt."""

    name: str
    data_settings: Any
    dataset_train: pd.DataFrame
    dataset_validation: pd.DataFrame
    dataset_test: Optional[pd.DataFrame]
    target_columns: List[str]
    equation: Optional[str] = None
    operations_set: Optional[List[str]] = None
    data_dictionary: Optional[str] = None
    dataset_ood_test: Optional[pd.DataFrame] = None
    column_descriptions: Optional[Dict[str, str]] = None
