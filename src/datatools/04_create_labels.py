"""Create final Daily and Weekly labeled datasets from separate feature sets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DAILY_TEMP = PROCESSED / "_tmp_spy_features_daily.csv"
WEEKLY_TEMP = PROCESSED / "_tmp_spy_features_weekly.csv"
RESEARCH_END = pd.Timestamp("2025-12-31")


def load_feature_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing intermediate feature file: {path}. Run 03_feature_engineering.py first."
        )
    frame = pd.read_csv(path, index_col="decision_date", parse_dates=True).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name} contains duplicate decision dates.")
    return frame


def create_labeled_dataset(
    features: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    adjusted_open: pd.Series,
    horizon: int,
    weekly: bool,
) -> pd.DataFrame:
    if not features.index.isin(calendar).all():
        raise ValueError("Feature dates are not all present in the SPY trading calendar.")

    # Rolling indicators and lagged FRED variables have a warm-up period.
    # Remove incomplete feature rows before assigning labels; never leave NaNs
    # in either final candidate dataset.
    features = features.dropna(subset=list(features.columns)).copy()
    if features.empty:
        raise ValueError("No complete feature rows remain after warm-up filtering.")
    feature_values = features.to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature matrix contains non-finite values.")

    anchor = calendar.searchsorted(pd.Timestamp("2010-01-01"))
    if weekly:
        expected_dates = calendar[anchor::5]
        if not features.index.equals(expected_dates[:len(features)]):
            # Initial feature warm-up may remove early rows; all remaining dates
            # must still lie on the stable every-fifth-session schedule.
            if not features.index.isin(expected_dates).all():
                raise ValueError("Weekly feature dates do not follow the fixed 5-session schedule.")
    else:
        expected_dates = calendar[anchor:]
        if not features.index.isin(expected_dates).all():
            raise ValueError("Daily feature dates fall outside the research period.")

    positions = calendar.get_indexer(features.index)
    execution_positions = positions + 1
    exit_positions = positions + horizon + 1
    valid = exit_positions < len(calendar)

    features = features.loc[valid].copy()
    positions = positions[valid]
    execution_positions = execution_positions[valid]
    exit_positions = exit_positions[valid]

    execution_dates = calendar[execution_positions]
    holding_end_dates = calendar[exit_positions]
    start_prices = adjusted_open.reindex(execution_dates).to_numpy(dtype=float)
    end_prices = adjusted_open.reindex(holding_end_dates).to_numpy(dtype=float)
    target = end_prices / start_prices - 1.0

    result = features.copy()
    result["execution_date"] = execution_dates
    result["holding_end_date"] = holding_end_dates
    result["target_return"] = target
    result = result.loc[result["holding_end_date"] <= RESEARCH_END]
    result = result.dropna(
        subset=["execution_date", "holding_end_date", "target_return"]
    )

    feature_columns = list(features.columns)
    result = result[
        ["execution_date", "holding_end_date", *feature_columns, "target_return"]
    ]
    values = result[feature_columns + ["target_return"]].to_numpy(dtype=float)
    if result.empty or not np.isfinite(values).all():
        raise ValueError("Labeled dataset is empty or contains non-finite feature/target values.")
    if not (result["execution_date"] > result.index).all():
        raise ValueError("Execution date must follow decision date.")
    if not (result["holding_end_date"] > result["execution_date"]).all():
        raise ValueError("Holding end must follow execution date.")

    if weekly and len(result) > 1:
        if not np.array_equal(
            result["holding_end_date"].iloc[:-1].to_numpy(dtype="datetime64[ns]"),
            result["execution_date"].iloc[1:].to_numpy(dtype="datetime64[ns]"),
        ):
            raise ValueError("Weekly holding intervals contain gaps or overlaps.")

    return result


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    spy_path = RAW / "spy_raw.csv"
    if not spy_path.is_file():
        raise FileNotFoundError(spy_path)

    spy = pd.read_csv(spy_path, index_col="Date", parse_dates=True).sort_index()
    spy.index = pd.DatetimeIndex(spy.index).tz_localize(None).normalize()
    if spy.index.has_duplicates:
        raise ValueError("SPY raw data contains duplicate dates.")

    required = {"Open", "Close", "Adj Close"}
    if required - set(spy.columns):
        raise ValueError(f"SPY data missing columns: {sorted(required - set(spy.columns))}")

    calendar = pd.DatetimeIndex(spy.index)
    adjusted_open = pd.Series(
        (spy["Open"] * (spy["Adj Close"] / spy["Close"])).to_numpy(),
        index=calendar,
    )

    daily_features = load_feature_file(DAILY_TEMP)
    weekly_features = load_feature_file(WEEKLY_TEMP)

    daily = create_labeled_dataset(
        daily_features, calendar, adjusted_open, horizon=1, weekly=False
    )
    weekly = create_labeled_dataset(
        weekly_features, calendar, adjusted_open, horizon=5, weekly=True
    )

    daily_path = PROCESSED / "spy_labeled_daily.csv"
    weekly_path = PROCESSED / "spy_labeled_weekly.csv"
    daily.to_csv(daily_path, index_label="decision_date", date_format="%Y-%m-%d")
    weekly.to_csv(weekly_path, index_label="decision_date", date_format="%Y-%m-%d")

    # Delete only the two known intermediate files, and only after both final
    # datasets have been written successfully.
    DAILY_TEMP.unlink(missing_ok=True)
    WEEKLY_TEMP.unlink(missing_ok=True)

    print(
        f"Labeled datasets created: Daily={len(daily)} rows/{len(daily_features.columns)} "
        f"features; Weekly={len(weekly)} rows/{len(weekly_features.columns)} features."
    )
    print("Intermediate feature files removed.")


if __name__ == "__main__":
    main()
