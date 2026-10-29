from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

START = "2009-09-01"
END = "2026-01-01"

for ticker, name in [("SPY", "spy"), ("^VIX", "vix")]:
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=False,
        actions=True,
        progress=False
    )

    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df.to_csv(RAW / f"{name}_raw.csv")

    print(f"{ticker}: {len(df)} rows")
    print(f"From {df.index.min()} to {df.index.max()}")
    print(df.tail(3))
