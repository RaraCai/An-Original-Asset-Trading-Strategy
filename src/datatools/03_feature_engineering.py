from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

# 1. Load data
spy = pd.read_csv(
    RAW / "spy_raw.csv",
    index_col="Date",
    parse_dates=True
).sort_index()

vix = pd.read_csv(
    RAW / "vix_raw.csv",
    index_col="Date",
    parse_dates=True
).sort_index()

# 2. Adjust SPY prices
factor = spy["Adj Close"] / spy["Close"]

close = spy["Adj Close"]
open_price = spy["Open"] * factor
volume = spy["Volume"]

# 3. Create features
df = pd.DataFrame(index=spy.index)

# Returns
df["return_1d"] = close.pct_change(fill_method=None)
df["momentum_5d"] = close.pct_change(5, fill_method=None)
df["momentum_20d"] = close.pct_change(20, fill_method=None)

# Volatility
daily_return = close.pct_change(fill_method=None)

df["volatility_5d"] = daily_return.rolling(5).std()
df["volatility_20d"] = daily_return.rolling(20).std()
df["volatility_60d"] = daily_return.rolling(60).std()

# Moving averages
ma_5 = close.rolling(5).mean()
ma_20 = close.rolling(20).mean()

df["ma_gap_5_20"] = ma_5 / ma_20 - 1

# VIX
df["vix"] = vix["Close"].reindex(df.index)

# RSI (Wilder method)
change = close.diff()
gain = change.clip(lower=0)
loss = -change.clip(upper=0)

avg_gain = gain.iloc[1:15].mean()
avg_loss = loss.iloc[1:15].mean()

rsi_gain = pd.Series(np.nan, index=close.index)
rsi_loss = pd.Series(np.nan, index=close.index)

rsi_gain.iloc[14] = avg_gain
rsi_loss.iloc[14] = avg_loss

for i in range(15, len(close)):
    rsi_gain.iloc[i] = (
        rsi_gain.iloc[i - 1] * 13 + gain.iloc[i]
    ) / 14

    rsi_loss.iloc[i] = (
        rsi_loss.iloc[i - 1] * 13 + loss.iloc[i]
    ) / 14

rs = rsi_gain / rsi_loss.replace(0, np.nan)
df["rsi_14"] = 100 - 100 / (1 + rs)

df.loc[(rsi_loss == 0) & (rsi_gain > 0), "rsi_14"] = 100
df.loc[(rsi_loss == 0) & (rsi_gain == 0), "rsi_14"] = 50

# Relative volume
avg_volume = volume.rolling(20).mean()
df["relative_volume_20d"] = volume / avg_volume

# Intraday return
df["intraday_return"] = close / open_price - 1

# 4. Keep research period
df = df.loc["2010-01-01":"2025-12-31"]

# 5. Check features
print("\nFEATURE ENGINEERING RESULTS")
print("=" * 50)

print("Rows:", len(df))
print("Features:", len(df.columns))
print("Start:", df.index.min())
print("End:", df.index.max())

print("\nFeature names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# 6. Save features
output = PROCESSED / "spy_features.csv"
df.to_csv(output, index_label="decision_date")

print(f"\nSaved to: {output}")
print("FEATURE ENGINEERING COMPLETED")
