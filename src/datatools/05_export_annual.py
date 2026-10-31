"""Export the established Daily annual files (2010-2025) and audit outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
ANNUAL = PROCESSED / "annual"

# Preserve the historical annual-file schema used by the existing project.
ANNUAL_DAILY_FEATURES = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "ma_gap_5_20",
    "vix",
    "rsi_14",
    "relative_volume_20d",
    "intraday_return",
]


def main() -> None:
    daily_path = PROCESSED / "spy_labeled_daily.csv"
    if not daily_path.is_file():
        raise FileNotFoundError(
            f"Missing {daily_path}. Run steps 03 and 04 before step 05."
        )

    daily = pd.read_csv(daily_path, parse_dates=["decision_date"])
    required = {
        "decision_date", "execution_date", "holding_end_date",
        *ANNUAL_DAILY_FEATURES, "target_return",
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Daily dataset missing annual export columns: {sorted(missing)}")

    if daily["decision_date"].duplicated().any():
        raise ValueError("Daily dataset contains duplicate decision dates.")
    if daily.isna().any().any():
        raise ValueError("Daily dataset contains missing values.")
    numeric = daily[ANNUAL_DAILY_FEATURES + ["target_return"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Daily dataset contains non-finite numeric values.")

    daily["year"] = daily["decision_date"].dt.year
    columns = [
        "year", "decision_date", "execution_date", "holding_end_date",
        *ANNUAL_DAILY_FEATURES, "target_return",
    ]
    ANNUAL.mkdir(parents=True, exist_ok=True)

    for year in range(2010, 2026):
        annual = daily.loc[daily["year"].eq(year), columns].copy()
        annual.to_csv(
            ANNUAL / f"{year}.csv", index=False, date_format="%Y-%m-%d"
        )

    annual_files = sorted(ANNUAL.glob("*.csv"))
    if len(annual_files) != 16:
        raise RuntimeError(f"Expected 16 annual CSV files, found {len(annual_files)}.")
    exported_rows = sum(len(pd.read_csv(path)) for path in annual_files)
    if exported_rows != len(daily):
        raise RuntimeError(
            f"Annual export row mismatch: {exported_rows} vs {len(daily)}."
        )

    # Remove only known legacy/intermediate artifacts from earlier data builds.
    stale_names = [
        "spy_features.csv",
        "spy_features_daily.csv",
        "spy_features_weekly.csv",
        "_tmp_spy_features_daily.csv",
        "_tmp_spy_features_weekly.csv",
        "spy_labeled.csv",
        "spy_labeled_weekly_merged.csv",
    ]
    for name in stale_names:
        (PROCESSED / name).unlink(missing_ok=True)

    expected_files = {"spy_labeled_daily.csv", "spy_labeled_weekly.csv"}
    unexpected = sorted(
        p.name for p in PROCESSED.iterdir()
        if p.is_file() and p.name not in expected_files
    )
    if unexpected:
        print(f"Review unexpected processed files (not auto-deleted): {unexpected}")

    print(
        f"Annual export complete: {len(annual_files)} files, "
        f"{exported_rows} Daily rows, {len(columns)} columns per file."
    )


if __name__ == "__main__":
    main()
