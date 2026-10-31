"""Download market prices and FRED macro series into data/raw/."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
START, END = "2009-01-01", "2026-01-01"  # yfinance end is exclusive
TICKERS = [("SPY", "spy"), ("^VIX", "vix"), ("QQQ", "qqq"), ("TLT", "tlt")]
FRED_SERIES = ("DGS2", "DGS10", "DFII10")


def download_prices() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for ticker, name in TICKERS:
        frame = yf.download(
            ticker, start=START, end=END, auto_adjust=False,
            actions=True, progress=False,
        )
        if frame.empty:
            raise RuntimeError(f"No price data returned for {ticker}.")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{ticker} is missing columns: {sorted(missing)}")

        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame.index.name = "Date"
        if frame.index.has_duplicates:
            raise ValueError(f"{ticker} returned duplicate dates.")
        frame.to_csv(RAW / f"{name}_raw.csv")

    print("Market price downloads completed.")


def download_fred() -> None:
    api_key = os.getenv("FRED_KEY")
    if not api_key:
        raise EnvironmentError(
            'FRED_KEY is not set. Set it in this terminal, e.g. '
            '$env:FRED_KEY="YOUR_KEY" in PowerShell.'
        )

    session = requests.Session()
    for series in FRED_SERIES:
        params = {
            "series_id": series,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": START,
            "observation_end": "2025-12-31",
        }
        for attempt in range(4):
            try:
                response = session.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params=params,
                    timeout=40,
                )
                response.raise_for_status()
                observations = response.json()["observations"]
                frame = pd.DataFrame({
                    "date": [row["date"] for row in observations],
                    series: [row["value"] for row in observations],
                })
                frame["date"] = pd.to_datetime(frame["date"])
                frame[series] = pd.to_numeric(frame[series], errors="coerce")
                if frame[series].notna().sum() < 1000:
                    raise ValueError(f"{series}: insufficient valid observations.")
                frame.to_csv(
                    RAW / f"{series}.csv", index=False, date_format="%Y-%m-%d"
                )
                break
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"Failed to download {series}: {exc}") from exc
                time.sleep(2 ** attempt)

    print("FRED downloads completed.")


def main() -> None:
    download_prices()
    download_fred()
    print(f"Raw data saved to: {RAW}")


if __name__ == "__main__":
    main()
