"""Pipeline-aware loading and validation of processed labeled datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_pipeline_data_path, get_pipeline_features


DATE_COLUMNS = ["decision_date", "execution_date", "holding_end_date"]


def load_data(config: dict, pipeline: str | None = None) -> pd.DataFrame:
    """Load one pipeline's labeled CSV and validate required columns and values."""
    selected = pipeline or config["default_pipeline"]
    path = get_pipeline_data_path(config, selected)
    if not path.is_file():
        raise FileNotFoundError(f"{selected} labeled dataset not found: {path}")

    data = pd.read_csv(path, parse_dates=DATE_COLUMNS)
    features = get_pipeline_features(config, selected)
    target = config["data"]["target"]

    required = set(DATE_COLUMNS + features + [target])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{selected} dataset is missing required columns: {missing}")

    original_rows = len(data)
    data = data.dropna(subset=list(required)).copy()
    data = data.sort_values("decision_date").reset_index(drop=True)

    if data.empty:
        raise ValueError(f"{selected} dataset has no complete observations.")
    if data["decision_date"].duplicated().any():
        raise ValueError(f"{selected} decision_date must be unique.")
    if not (data["execution_date"] > data["decision_date"]).all():
        raise ValueError(f"{selected} execution_date must follow decision_date.")
    if not (data["holding_end_date"] > data["execution_date"]).all():
        raise ValueError(f"{selected} holding_end_date must follow execution_date.")

    values = data[features + [target]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{selected} features/target contain non-finite values.")

    data.attrs["pipeline"] = selected
    data.attrs["data_path"] = str(path)
    data.attrs["rows_before_complete_case_filter"] = original_rows
    data.attrs["rows_dropped_for_missing_values"] = original_rows - len(data)
    return data
