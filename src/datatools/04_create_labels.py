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

features = pd.read_csv(
    PROCESSED / "spy_features.csv",
    index_col="decision_date",
    parse_dates=True
).sort_index()

# 2. Adjust open prices
factor = spy["Adj Close"] / spy["Close"]
adjusted_open = spy["Open"] * factor

# 3. Create trading dates
dates = spy.index.to_series()

execution_date = dates.shift(-1)
holding_end_date = dates.shift(-2)

# 4. Create target return
target_return = (
    adjusted_open.shift(-2) /
    adjusted_open.shift(-1) - 1
)

# 5. Combine features and labels
df = features.copy()

df["execution_date"] = execution_date.reindex(df.index)
df["holding_end_date"] = holding_end_date.reindex(df.index)
df["target_return"] = target_return.reindex(df.index)

# 6. Remove incomplete labels
before = len(df)

df = df.dropna(
    subset=[
        "execution_date",
        "holding_end_date",
        "target_return"
    ]
)

removed = before - len(df)

# 7. Set column order
feature_columns = features.columns.tolist()

df = df[
    ["execution_date", "holding_end_date"]
    + feature_columns
    + ["target_return"]
]

# 8. Check results
print("\nLABEL & DATE ALIGNMENT RESULTS")
print("=" * 50)

print("Rows:", len(df))
print("Columns:", len(df.columns) + 1)
print("Removed rows:", removed)

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLast 5 rows:")
print(df.tail().to_string())

# 9. Save data
output = PROCESSED / "spy_labeled.csv"

df.to_csv(
    output,
    index_label="decision_date",
    date_format="%Y-%m-%d"
)

print(f"\nSaved to: {output}")
print("LABEL CREATION COMPLETED")
