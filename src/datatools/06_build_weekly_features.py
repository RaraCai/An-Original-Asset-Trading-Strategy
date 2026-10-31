"""Build one consolidated Weekly feature dataset from the user's 25-feature set
plus non-duplicate features from the teammate's Weekly feature engineering.

Run from repository root:
    python src/datatools/06_build_weekly_features.py

Output:
    data/processed/spy_labeled_weekly_merged.csv

The dataset contains 35 candidate features. Select the feature list in
src/models/config.yaml. Daily feature construction and the original Daily,
W0/W1/W2 datasets are not modified.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CORE_FEATURES = [
    "weekly_mean_return_5d",
    "weekly_positive_days_5d",
    "weekly_max_return_5d",
    "weekly_min_return_5d",
    "weekly_return_std_5d",
    "weekly_downside_std_5d",
    "weekly_max_drawdown_5d",
    "weekly_mean_intraday_return_5d",
    "weekly_mean_overnight_return_5d",
    "weekly_compounded_intraday_5d",
    "weekly_compounded_overnight_5d",
    "weekly_intraday_std_5d",
    "weekly_mean_relative_volume_5d",
    "weekly_volume_trend_5d",
    "weekly_qqq_relative_return_5d",
    "weekly_tlt_return_5d",
]
GAP_FEATURES = [
    "weekly_gap_reversal_mean_5d",
    "weekly_gap_reversal_std_5d",
    "weekly_gap_reversal_positive_days_5d",
    "weekly_gap_reversal_abs_mean_5d",
]
MACRO_FEATURES = [
    "weekly_yield_2y_change_5d",
    "weekly_yield_10y_change_5d",
    "weekly_real_yield_change_5d",
    "weekly_yield_curve_change_5d",
    "weekly_yield_curve_mean_5d",
]
TEAM_ADDITIONS = [
    "weekly_log_price_slope_5d",
    "weekly_trend_fit_r2_5d",
    "weekly_upside_std_5d",
    "weekly_range_mean_5d",
    "weekly_volatility_change_5d",
    "weekly_overnight_std_5d",
    "weekly_vix_change_5d",
    "weekly_vix_std_5d",
    "weekly_qqq_return_5d",
    "weekly_tlt_relative_return_5d",
]
ALL_FEATURES = CORE_FEATURES + GAP_FEATURES + MACRO_FEATURES + TEAM_ADDITIONS


def read_price(ticker: str) -> pd.DataFrame:
    path = RAW / f"{ticker}_raw.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw price file: {path}")
    frame = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate dates in {path.name}")
    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return frame


def rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def rolling_r2(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    fitted = np.polyval(np.polyfit(x, values, 1), x)
    total = float(np.sum((values - np.mean(values)) ** 2))
    return 0.0 if total <= 0 else float(1.0 - np.sum((values - fitted) ** 2) / total)


def read_fred(code: str) -> pd.Series:
    path = RAW / f"{code}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FRED file: {path}")
    raw = pd.read_csv(path, parse_dates=["date"])
    if code not in raw.columns:
        raise ValueError(f"{path.name} must contain columns date,{code}")
    raw[code] = pd.to_numeric(raw[code], errors="coerce")
    raw = raw[["date", code]].dropna().drop_duplicates("date").sort_values("date")
    return raw.set_index("date")[code]


def build_macro_features(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    # Strictly backward as-of alignment; exact same-day FRED matches are excluded.
    fred = pd.concat([read_fred(c) for c in ("DGS2", "DGS10", "DFII10")], axis=1).sort_index()
    fred["yield_curve"] = fred["DGS10"] - fred["DGS2"]
    left = pd.DataFrame({"decision_date": calendar})
    right = fred.reset_index()
    right = right.rename(columns={right.columns[0]: "observation_date"})
    aligned = pd.merge_asof(
        left.sort_values("decision_date"), right.sort_values("observation_date"),
        left_on="decision_date", right_on="observation_date",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7),
    ).set_index("decision_date")
    out = pd.DataFrame(index=calendar)
    for col in ("DGS2", "DGS10", "DFII10", "yield_curve"):
        out[col] = aligned[col].to_numpy()
    # Five SPY-session-aligned observations: current decision date and prior four sessions.
    out["weekly_yield_2y_change_5d"] = out["DGS2"] - out["DGS2"].shift(4)
    out["weekly_yield_10y_change_5d"] = out["DGS10"] - out["DGS10"].shift(4)
    out["weekly_real_yield_change_5d"] = out["DFII10"] - out["DFII10"].shift(4)
    out["weekly_yield_curve_change_5d"] = out["yield_curve"] - out["yield_curve"].shift(4)
    out["weekly_yield_curve_mean_5d"] = out["yield_curve"].rolling(5, min_periods=5).mean()
    return out[MACRO_FEATURES]


def main() -> None:
    spy, qqq, tlt, vix = (read_price(x) for x in ("spy", "qqq", "tlt", "vix"))
    calendar = spy.index
    adj_factor = spy["Adj Close"] / spy["Close"]
    adj_open = spy["Open"] * adj_factor
    adj_high = spy["High"] * adj_factor
    adj_low = spy["Low"] * adj_factor
    adj_close = spy["Adj Close"].astype(float)
    returns = adj_close.pct_change(fill_method=None)
    intraday = adj_close / adj_open - 1.0
    overnight = adj_open / adj_close.shift(1) - 1.0
    gap = overnight * intraday
    relative_volume = spy["Volume"] / spy["Volume"].rolling(20, min_periods=20).mean()
    vix_close = vix["Close"].reindex(calendar).astype(float)

    f = pd.DataFrame(index=calendar)
    # User's original 16 Weekly core features.
    f["weekly_mean_return_5d"] = returns.rolling(5, min_periods=5).mean()
    f["weekly_positive_days_5d"] = returns.gt(0).rolling(5, min_periods=5).sum()
    f["weekly_max_return_5d"] = returns.rolling(5, min_periods=5).max()
    f["weekly_min_return_5d"] = returns.rolling(5, min_periods=5).min()
    f["weekly_return_std_5d"] = returns.rolling(5, min_periods=5).std()
    downside = returns.clip(upper=0.0)
    f["weekly_downside_std_5d"] = np.sqrt(downside.pow(2).rolling(5, min_periods=5).mean())

    def max_drawdown(prices: np.ndarray) -> float:
        peak = np.maximum.accumulate(prices)
        return float(np.min(prices / peak - 1.0))

    # Six closes: prior close plus the five daily closes in the feature window.
    f["weekly_max_drawdown_5d"] = adj_close.rolling(6, min_periods=6).apply(max_drawdown, raw=True)
    f["weekly_mean_intraday_return_5d"] = intraday.rolling(5, min_periods=5).mean()
    f["weekly_mean_overnight_return_5d"] = overnight.rolling(5, min_periods=5).mean()
    f["weekly_compounded_intraday_5d"] = (1 + intraday).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    f["weekly_compounded_overnight_5d"] = (1 + overnight).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    f["weekly_intraday_std_5d"] = intraday.rolling(5, min_periods=5).std()
    f["weekly_mean_relative_volume_5d"] = relative_volume.rolling(5, min_periods=5).mean()
    f["weekly_volume_trend_5d"] = relative_volume.rolling(5, min_periods=5).apply(rolling_slope, raw=True)
    spy_5d = adj_close.pct_change(5, fill_method=None)
    qqq_adj = qqq["Adj Close"].astype(float)
    tlt_adj = tlt["Adj Close"].astype(float)
    qqq_5d = qqq_adj.pct_change(5, fill_method=None).reindex(calendar)
    tlt_5d = tlt_adj.pct_change(5, fill_method=None).reindex(calendar)
    f["weekly_qqq_relative_return_5d"] = qqq_5d - spy_5d
    f["weekly_tlt_return_5d"] = tlt_5d

    # User's four-feature Gap Reversal family. Keep the user's original sign convention
    # (positive = overnight and intraday returns have the same sign) for comparability
    # with the already-completed W4-B experiment. Teammate's mean is sign-reversed.
    f["weekly_gap_reversal_mean_5d"] = gap.rolling(5, min_periods=5).mean()
    f["weekly_gap_reversal_std_5d"] = gap.rolling(5, min_periods=5).std()
    f["weekly_gap_reversal_positive_days_5d"] = gap.gt(0).rolling(5, min_periods=5).sum()
    f["weekly_gap_reversal_abs_mean_5d"] = gap.abs().rolling(5, min_periods=5).mean()

    # User's five weekly-aligned FRED features; do not copy teammate's single-day
    # macro values or FRED-observation-count windows into this 5-session schema.
    macro = build_macro_features(calendar)
    f = pd.concat([f, macro], axis=1)

    # Teammate's genuinely additional, non-duplicate features.
    log_price = np.log(adj_close)
    f["weekly_log_price_slope_5d"] = log_price.rolling(5, min_periods=5).apply(rolling_slope, raw=True)
    f["weekly_trend_fit_r2_5d"] = log_price.rolling(5, min_periods=5).apply(rolling_r2, raw=True)
    f["weekly_upside_std_5d"] = np.sqrt(returns.clip(lower=0.0).pow(2).rolling(5, min_periods=5).mean())
    f["weekly_range_mean_5d"] = ((adj_high - adj_low) / adj_close).rolling(5, min_periods=5).mean()
    recent_vol = returns.rolling(2, min_periods=2).std()
    earlier_vol = returns.shift(2).rolling(3, min_periods=3).std()
    f["weekly_volatility_change_5d"] = recent_vol / earlier_vol.replace(0, np.nan) - 1.0
    f["weekly_overnight_std_5d"] = overnight.rolling(5, min_periods=5).std()
    f["weekly_vix_change_5d"] = vix_close.pct_change(5, fill_method=None)
    f["weekly_vix_std_5d"] = vix_close.pct_change(fill_method=None).rolling(5, min_periods=5).std()
    f["weekly_qqq_return_5d"] = qqq_5d
    f["weekly_tlt_relative_return_5d"] = tlt_5d - spy_5d

    if list(f.columns) != ALL_FEATURES:
        raise AssertionError(f"Feature schema/order mismatch: {len(f.columns)} vs {len(ALL_FEATURES)}")

    # Stable every-fifth-session decision schedule; same next-open to +5-session-open label.
    anchor = calendar.searchsorted(pd.Timestamp("2010-01-01"))
    decision_dates = calendar[anchor::5]
    cal_series = pd.Series(calendar, index=calendar)
    adj_open_series = pd.Series(adj_open.to_numpy(), index=calendar)
    out = f.loc[decision_dates].copy()
    out["execution_date"] = cal_series.shift(-1).reindex(decision_dates)
    out["holding_end_date"] = cal_series.shift(-6).reindex(decision_dates)
    out["target_return"] = (adj_open_series.shift(-6) / adj_open_series.shift(-1) - 1.0).reindex(decision_dates)
    out = out.loc[out["holding_end_date"] <= pd.Timestamp("2025-12-31")]
    required = ALL_FEATURES + ["execution_date", "holding_end_date", "target_return"]
    out = out.dropna(subset=required)
    out = out[["execution_date", "holding_end_date"] + ALL_FEATURES + ["target_return"]]

    if out.empty or out.index.has_duplicates:
        raise ValueError("Weekly merged dataset is empty or has duplicate decision dates")
    if not (out["execution_date"] > out.index).all():
        raise ValueError("Execution date must follow decision date")
    if not (out["holding_end_date"].iloc[:-1].to_numpy() == out["execution_date"].iloc[1:].to_numpy()).all():
        raise ValueError("Weekly positions do not join without gaps or overlaps")
    if not np.isfinite(out[ALL_FEATURES + ["target_return"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in features or target")

    path = OUT / "spy_labeled_weekly_merged.csv"
    out.to_csv(path, index_label="decision_date", date_format="%Y-%m-%d")
    print(f"Saved: {path}")
    print(f"Rows: {len(out)} | Candidate features: {len(ALL_FEATURES)} | Total columns: {len(out.columns)}")
    print(f"Date range: {out.index.min().date()} to {out.index.max().date()}")
    print("Feature groups: core=16, gap=4, macro=5, teammate additions=10")
    print("No original Daily/W0/W1/W2/W3/W4 dataset was overwritten.")


if __name__ == "__main__":
    main()
