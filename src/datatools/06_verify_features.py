"""Audit final Daily/Weekly feature values and labels against raw inputs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FEATURE_SCRIPT = Path(__file__).with_name("03_feature_engineering.py")


def load_feature_module():
    spec = importlib.util.spec_from_file_location("feature_engineering_module", FEATURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load feature functions from {FEATURE_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_features(
    labeled: pd.DataFrame, expected: pd.DataFrame, feature_columns: list[str], label: str
) -> None:
    aligned = expected.reindex(labeled.index)[feature_columns]
    actual = labeled[feature_columns].to_numpy(dtype=float)
    reference = aligned.to_numpy(dtype=float)
    if actual.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError(f"{label}: recomputed features contain missing/non-finite values.")
    if not np.allclose(actual, reference, rtol=1e-10, atol=1e-12):
        difference = np.abs(actual - reference)
        idx, col = np.unravel_index(np.nanargmax(difference), difference.shape)
        raise AssertionError(
            f"{label}: feature mismatch at {labeled.index[idx]} / "
            f"{feature_columns[col]}; max error={difference[idx, col]:.12g}"
        )


def verify_labels(
    labeled: pd.DataFrame, spy: pd.DataFrame, horizon: int, label: str
) -> None:
    calendar = pd.DatetimeIndex(spy.index)
    positions = calendar.get_indexer(labeled.index)
    if (positions < 0).any():
        raise AssertionError(f"{label}: decision dates absent from SPY calendar.")
    expected_execution = calendar[positions + 1]
    expected_exit = calendar[positions + horizon + 1]
    if not np.array_equal(
        labeled["execution_date"].to_numpy(dtype="datetime64[ns]"),
        expected_execution.to_numpy(dtype="datetime64[ns]"),
    ):
        raise AssertionError(f"{label}: execution-date alignment failed.")
    if not np.array_equal(
        labeled["holding_end_date"].to_numpy(dtype="datetime64[ns]"),
        expected_exit.to_numpy(dtype="datetime64[ns]"),
    ):
        raise AssertionError(f"{label}: holding-end alignment failed.")

    adjusted_open = spy["Open"] * (spy["Adj Close"] / spy["Close"])
    expected_return = (
        adjusted_open.iloc[positions + horizon + 1].to_numpy()
        / adjusted_open.iloc[positions + 1].to_numpy() - 1
    )
    if not np.allclose(
        labeled["target_return"].to_numpy(dtype=float),
        expected_return,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(f"{label}: target-return calculation failed.")


def main() -> None:
    daily_path = PROCESSED / "spy_labeled_daily.csv"
    weekly_path = PROCESSED / "spy_labeled_weekly.csv"
    if not daily_path.is_file() or not weekly_path.is_file():
        raise FileNotFoundError("Run steps 03 and 04 before feature verification.")

    spy = pd.read_csv(
        RAW / "spy_raw.csv", index_col="Date", parse_dates=True
    ).sort_index()
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()

    daily = pd.read_csv(
        daily_path, index_col="decision_date", parse_dates=True
    ).sort_index()
    weekly = pd.read_csv(
        weekly_path, index_col="decision_date", parse_dates=True
    ).sort_index()

    module = load_feature_module()
    prices = {
        name: module.read_price(name) for name in ("spy", "vix", "qqq", "tlt")
    }
    expected_daily = module.build_daily_features(
        prices["spy"], prices["vix"], prices["qqq"], prices["tlt"]
    )
    expected_weekly = module.build_weekly_features(
        prices["spy"], prices["vix"], prices["qqq"], prices["tlt"]
    )

    compare_features(daily, expected_daily, module.DAILY_FEATURES, "Daily")
    compare_features(weekly, expected_weekly, module.WEEKLY_FEATURES, "Weekly")
    verify_labels(daily, spy, horizon=1, label="Daily")
    verify_labels(weekly, spy, horizon=5, label="Weekly")

    if len(module.DAILY_FEATURES) != 21 or len(module.WEEKLY_FEATURES) != 35:
        raise AssertionError("Candidate feature schema counts are not 21 Daily / 35 Weekly.")

    print(
        f"Feature audit passed: Daily {len(daily)} rows/{len(module.DAILY_FEATURES)} "
        f"features; Weekly {len(weekly)} rows/{len(module.WEEKLY_FEATURES)} features."
    )
    print("Feature values, decision dates, execution dates and targets verified.")


if __name__ == "__main__":
    main()
