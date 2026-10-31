"""Concise quality checks for all raw market and FRED input files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PRICE_FILES = ("spy", "vix", "qqq", "tlt")
FRED_FILES = ("DGS2", "DGS10", "DFII10")
PRICE_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def check_price_file(name: str) -> pd.DataFrame:
    path = RAW / f"{name}_raw.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    missing = PRICE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name} contains duplicate dates.")

    numeric = frame[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains missing or non-finite OHLCV values.")
    if (numeric[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError(f"{path.name} contains non-positive prices.")
    if (numeric["Volume"] < 0).any():
        raise ValueError(f"{path.name} contains negative volume.")

    invalid_high = numeric["High"] < numeric[["Open", "Close", "Low"]].max(axis=1)
    invalid_low = numeric["Low"] > numeric[["Open", "Close", "High"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError(f"{path.name} contains inconsistent OHLC values.")

    return frame


def check_fred_file(series: str) -> pd.DataFrame:
    path = RAW / f"{series}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["date"])
    if series not in frame.columns:
        raise ValueError(f"{path.name} must contain date and {series} columns.")
    frame[series] = pd.to_numeric(frame[series], errors="coerce")
    frame = frame.dropna(subset=["date", series])
    if frame["date"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate observation dates.")
    if len(frame) < 1000:
        raise ValueError(f"{path.name} has too few valid observations.")
    return frame


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    prices = {name: check_price_file(name) for name in PRICE_FILES}
    fred = {series: check_fred_file(series) for series in FRED_FILES}

    spy_dates = prices["spy"].index
    print(
        f"Raw data checks passed: {len(prices)} price files, "
        f"{len(fred)} FRED files; SPY rows={len(spy_dates)}."
    )


if __name__ == "__main__":
    main()
