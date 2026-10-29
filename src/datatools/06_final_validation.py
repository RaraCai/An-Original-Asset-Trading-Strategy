from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ANNUAL = ROOT / "data" / "processed" / "annual"

FEATURES = [
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
    "intraday_return"
]

COLUMNS = [
    "year",
    "decision_date",
    "execution_date",
    "holding_end_date",
    *FEATURES,
    "target_return"
]

# 1. Load annual files
frames = []

for year in range(2010, 2026):
    path = ANNUAL / f"{year}.csv"
    assert path.exists(), f"Missing file: {path}"

    df = pd.read_csv(path)

    assert df.columns.tolist() == COLUMNS, (
        f"Invalid columns in {year}.csv"
    )
    assert (df["year"] == year).all()

    frames.append(df)
    print(f"{year}.csv: {len(df)} rows - OK")

data = pd.concat(frames, ignore_index=True)

# 2. Check dates
for col in [
    "decision_date",
    "execution_date",
    "holding_end_date"
]:
    data[col] = pd.to_datetime(data[col])

assert data["decision_date"].is_unique
assert data["decision_date"].is_monotonic_increasing

assert (
    data["decision_date"] < data["execution_date"]
).all()

assert (
    data["execution_date"] < data["holding_end_date"]
).all()

assert (
    data["decision_date"].dt.year == data["year"]
).all()

# 3. Check missing and invalid values
assert data.isna().sum().sum() == 0

numeric = data[FEATURES + ["target_return"]]

assert np.isfinite(numeric.to_numpy()).all()
assert data["rsi_14"].between(0, 100).all()
assert (data["vix"] > 0).all()
assert (data["relative_volume_20d"] >= 0).all()

# 4. Load original SPY data
spy = pd.read_csv(
    RAW / "spy_raw.csv",
    index_col="Date",
    parse_dates=True
).sort_index()

factor = spy["Adj Close"] / spy["Close"]
adjusted_open = spy["Open"] * factor

trading_dates = spy.index
date_positions = pd.Series(
    np.arange(len(trading_dates)),
    index=trading_dates
)

# 5. Check trading date alignment
positions = date_positions.reindex(
    data["decision_date"]
).to_numpy()

assert np.isfinite(positions).all()

positions = positions.astype(int)

assert (positions + 2 < len(trading_dates)).all()

expected_execution = trading_dates[positions + 1]
expected_end = trading_dates[positions + 2]

assert np.array_equal(
    data["execution_date"].to_numpy(dtype="datetime64[ns]"),
    expected_execution.to_numpy(dtype="datetime64[ns]")
)

assert np.array_equal(
    data["holding_end_date"].to_numpy(dtype="datetime64[ns]"),
    expected_end.to_numpy(dtype="datetime64[ns]")
)

# 6. Check target returns
expected_returns = (
    adjusted_open.iloc[positions + 2].to_numpy()
    / adjusted_open.iloc[positions + 1].to_numpy()
    - 1
)

actual_returns = data["target_return"].to_numpy()

assert np.allclose(
    actual_returns,
    expected_returns,
    rtol=1e-10,
    atol=1e-12
)

# 7. Check feature values
features_file = pd.read_csv(
    ROOT / "data" / "processed" / "spy_features.csv",
    parse_dates=["decision_date"]
).set_index("decision_date")

expected_features = features_file.reindex(
    data["decision_date"]
)[FEATURES]

assert np.allclose(
    data[FEATURES].to_numpy(),
    expected_features.to_numpy(),
    rtol=1e-10,
    atol=1e-12
)

# 8. Check split sizes
train = data[data["year"].between(2010, 2017)]
val = data[data["year"].between(2018, 2021)]
test = data[data["year"].between(2022, 2025)]

assert len(train) + len(val) + len(test) == len(data)

# 9. Print summary
print("\n" + "=" * 50)
print("FINAL VALIDATION SUMMARY")
print("=" * 50)

print("Total rows:", len(data))
print("Total columns:", len(data.columns))
print("Missing values:", data.isna().sum().sum())
print("Duplicate dates:", data["decision_date"].duplicated().sum())

print("\nData split:")
print("Training (2010-2017):", len(train))
print("Validation (2018-2021):", len(val))
print("Testing (2022-2025):", len(test))

print("\nDate alignment: PASSED")
print("Target returns: PASSED")
print("Feature consistency: PASSED")
print("Schema validation: PASSED")

print("\nFINAL DATA VALIDATION PASSED")
