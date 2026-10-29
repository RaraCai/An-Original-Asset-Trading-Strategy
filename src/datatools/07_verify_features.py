from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

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

features = pd.read_csv(
    PROCESSED / "spy_features.csv",
    index_col="decision_date",
    parse_dates=True
).sort_index()

# 2. Prepare prices
factor = spy["Adj Close"] / spy["Close"]
close = spy["Adj Close"]
open_price = spy["Open"] * factor
volume = spy["Volume"]

daily_return = close.pct_change(fill_method=None)

# 3. Calculate expected features
expected = pd.DataFrame(index=spy.index)

# Returns
expected["return_1d"] = close.pct_change(
    1, fill_method=None
)

expected["momentum_5d"] = close.pct_change(
    5, fill_method=None
)

expected["momentum_20d"] = close.pct_change(
    20, fill_method=None
)

# Volatility
for days in [5, 20, 60]:
    expected[f"volatility_{days}d"] = (
        daily_return.rolling(days).std()
    )

# Moving averages
ma5 = close.rolling(5).mean()
ma20 = close.rolling(20).mean()

expected["ma_gap_5_20"] = ma5 / ma20 - 1

# VIX
expected["vix"] = vix["Close"].reindex(spy.index)

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

expected["rsi_14"] = 100 - 100 / (1 + rs)

expected.loc[
    (rsi_loss == 0) & (rsi_gain > 0),
    "rsi_14"
] = 100

expected.loc[
    (rsi_loss == 0) & (rsi_gain == 0),
    "rsi_14"
] = 50

# Relative volume
expected["relative_volume_20d"] = (
    volume / volume.rolling(20).mean()
)

# Intraday return
expected["intraday_return"] = (
    close / open_price - 1
)

# 4. Align dates
expected = expected.reindex(features.index)

# 5. Compare features
print("\nFEATURE FORMULA VERIFICATION")
print("=" * 60)

failed = []

for column in features.columns:
    actual = features[column].to_numpy()
    reference = expected[column].to_numpy()

    valid = np.isfinite(actual) & np.isfinite(reference)

    if not valid.all():
        print(f"{column}: FAILED - missing values")
        failed.append(column)
        continue

    difference = np.abs(actual - reference)
    max_error = difference.max()

    passed = np.allclose(
        actual,
        reference,
        rtol=1e-6,
        atol=1e-8
    )

    status = "PASSED" if passed else "FAILED"

    print(
        f"{column}: {status} "
        f"(max error: {max_error:.10f})"
    )

    if not passed:
        failed.append(column)

# 6. Final result
print("\n" + "=" * 60)

if failed:
    print("FAILED FEATURES:", failed)
    print("FEATURE VERIFICATION NEEDS REVIEW")
    raise SystemExit(1)
else:
    print("ALL FEATURE FORMULAS VERIFIED")
