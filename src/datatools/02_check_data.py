from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def check_data(name):
    print(f"\n{'=' * 50}")
    print(f"CHECKING {name.upper()}")
    print("=" * 50)

    path = RAW / f"{name}_raw.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # 1. Basic information
    print("\n[1] Basic Information")
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("End:", df.index.max())
    print("Columns:", df.columns.tolist())

    # 2. Check duplicate dates
    print("\n[2] Duplicate Dates")
    print("Duplicates:", df.index.duplicated().sum())

    # 3. Check missing values
    print("\n[3] Missing Values")
    print(df.isna().sum())

    # 4. Check invalid prices
    print("\n[4] Price Validation")
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            invalid = (
                (df[col] <= 0) |
                (~np.isfinite(df[col]))
            )
            print(f"{col} invalid values:", invalid.sum())

    # 5. Check OHLC consistency
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        bad_high = df["High"] < df[["Open", "Close", "Low"]].max(axis=1)
        bad_low = df["Low"] > df[["Open", "Close", "High"]].min(axis=1)

        print("Invalid High:", bad_high.sum())
        print("Invalid Low:", bad_low.sum())

    # 6. Check trading volume
    if "Volume" in df.columns:
        print("Negative Volume:", (df["Volume"] < 0).sum())
        print("Zero Volume:", (df["Volume"] == 0).sum())

    return df


spy = check_data("spy")
vix = check_data("vix")

# 7. Check date alignment
print("\n" + "=" * 50)
print("DATE ALIGNMENT")
print("=" * 50)

spy_dates = set(spy.index)
vix_dates = set(vix.index)

print("SPY dates:", len(spy_dates))
print("VIX dates:", len(vix_dates))
print("Common dates:", len(spy_dates & vix_dates))
print("SPY only:", len(spy_dates - vix_dates))
print("VIX only:", len(vix_dates - spy_dates))

print("\nDATA QUALITY CHECK COMPLETED")
