from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
ANNUAL = PROCESSED / "annual"

ANNUAL.mkdir(parents=True, exist_ok=True)

# 1. Load labeled data
df = pd.read_csv(
    PROCESSED / "spy_labeled.csv",
    parse_dates=[
        "decision_date",
        "execution_date",
        "holding_end_date"
    ]
)

# 2. Add year
df["year"] = df["decision_date"].dt.year

# 3. Set column order
columns = [
    "year",
    "decision_date",
    "execution_date",
    "holding_end_date",
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
    "target_return"
]

df = df[columns]

# 4. Export annual files
print("\nANNUAL CSV EXPORT")
print("=" * 50)

total_rows = 0

for year in range(2010, 2026):
    annual_df = df[df["year"] == year].copy()

    output = ANNUAL / f"{year}.csv"

    annual_df.to_csv(
        output,
        index=False,
        date_format="%Y-%m-%d"
    )

    print(f"{year}.csv: {len(annual_df)} rows")

    total_rows += len(annual_df)

# 5. Final checks
files = list(ANNUAL.glob("*.csv"))

print("\nEXPORT SUMMARY")
print("=" * 50)

print("Files created:", len(files))
print("Total rows:", total_rows)
print("Columns:", len(columns))
print("Missing values:", df.isna().sum().sum())

assert len(files) == 16
assert total_rows == len(df)
assert len(df.columns) == 16
assert df.isna().sum().sum() == 0

print("\nANNUAL EXPORT COMPLETED")
