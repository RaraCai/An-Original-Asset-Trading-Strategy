"""Build independent Daily (21) and Weekly (35) candidate feature sets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

DAILY_FEATURES = [
    "return_1d", "momentum_5d", "momentum_20d",
    "volatility_5d", "volatility_20d", "volatility_60d",
    "ma_gap_5_20", "vix", "rsi_14", "relative_volume_20d",
    "intraday_return", "overnight_return", "intraday_momentum_5d",
    "gap_reversal", "qqq_spy_relative_5d", "tlt_return_5d",
    "yield_2y_change_5d", "yield_10y_change_5d",
    "yield_10y_change_20d", "yield_curve_10y_2y",
    "real_yield_10y_change_5d",
]

WEEKLY_FEATURES = [
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
    "weekly_gap_reversal_mean_5d",
    "weekly_gap_reversal_std_5d",
    "weekly_gap_reversal_positive_days_5d",
    "weekly_gap_reversal_abs_mean_5d",
    "weekly_yield_2y_change_5d",
    "weekly_yield_10y_change_5d",
    "weekly_real_yield_change_5d",
    "weekly_yield_curve_change_5d",
    "weekly_yield_curve_mean_5d",
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

DAILY_TEMP = PROCESSED / "_tmp_spy_features_daily.csv"
WEEKLY_TEMP = PROCESSED / "_tmp_spy_features_weekly.csv"


def read_price(name: str) -> pd.DataFrame:
    path = RAW / f"{name}_raw.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate dates.")
    return frame


def read_fred(series: str) -> pd.Series:
    path = RAW / f"{series}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["date"])
    if series not in frame.columns:
        raise ValueError(f"{path.name} must contain date and {series}.")
    frame[series] = pd.to_numeric(frame[series], errors="coerce")
    frame = (
        frame[["date", series]].dropna()
        .drop_duplicates("date").sort_values("date")
    )
    return frame.set_index("date")[series]


def build_daily_features(
    spy: pd.DataFrame, vix: pd.DataFrame, qqq: pd.DataFrame, tlt: pd.DataFrame
) -> pd.DataFrame:
    close = spy["Adj Close"].astype(float)
    adj_factor = spy["Adj Close"] / spy["Close"]
    open_adj = spy["Open"] * adj_factor
    volume = spy["Volume"].astype(float)
    daily_return = close.pct_change(fill_method=None)

    df = pd.DataFrame(index=spy.index)
    df["return_1d"] = daily_return
    df["momentum_5d"] = close.pct_change(5, fill_method=None)
    df["momentum_20d"] = close.pct_change(20, fill_method=None)
    for window in (5, 20, 60):
        df[f"volatility_{window}d"] = daily_return.rolling(
            window, min_periods=window
        ).std()
    df["ma_gap_5_20"] = (
        close.rolling(5, min_periods=5).mean()
        / close.rolling(20, min_periods=20).mean() - 1
    )
    df["vix"] = vix["Close"].reindex(df.index)

    # Preserve the original Wilder RSI implementation and initialization.
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = pd.Series(np.nan, index=close.index)
    avg_loss = pd.Series(np.nan, index=close.index)
    if len(close) > 14:
        avg_gain.iloc[14] = gain.iloc[1:15].mean()
        avg_loss.iloc[14] = loss.iloc[1:15].mean()
    for i in range(15, len(close)):
        avg_gain.iloc[i] = (13 * avg_gain.iloc[i - 1] + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (13 * avg_loss.iloc[i - 1] + loss.iloc[i]) / 14
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_14"] = 50

    df["relative_volume_20d"] = volume / volume.rolling(20, min_periods=20).mean()
    df["intraday_return"] = close / open_adj - 1
    overnight = open_adj / close.shift(1) - 1
    df["overnight_return"] = overnight
    df["intraday_momentum_5d"] = (
        (1 + df["intraday_return"]).rolling(5, min_periods=5)
        .apply(np.prod, raw=True) - 1
    )
    df["gap_reversal"] = overnight * df["intraday_return"]

    qqq_5d = qqq["Adj Close"].astype(float).pct_change(
        5, fill_method=None
    ).reindex(df.index)
    tlt_5d = tlt["Adj Close"].astype(float).pct_change(
        5, fill_method=None
    ).reindex(df.index)
    df["qqq_spy_relative_5d"] = qqq_5d - df["momentum_5d"]
    df["tlt_return_5d"] = tlt_5d

    # FRED changes are calculated on the observation calendar, then merged
    # strictly backward. Same-day observations are excluded conservatively.
    fred = pd.concat(
        [read_fred(code) for code in ("DGS2", "DGS10", "DFII10")], axis=1
    ).sort_index().dropna()
    fred["yield_2y_change_5d"] = fred["DGS2"].diff(5)
    fred["yield_10y_change_5d"] = fred["DGS10"].diff(5)
    fred["yield_10y_change_20d"] = fred["DGS10"].diff(20)
    fred["yield_curve_10y_2y"] = fred["DGS10"] - fred["DGS2"]
    fred["real_yield_10y_change_5d"] = fred["DFII10"].diff(5)
    fred_features = [
        "yield_2y_change_5d", "yield_10y_change_5d",
        "yield_10y_change_20d", "yield_curve_10y_2y",
        "real_yield_10y_change_5d",
    ]
    left = pd.DataFrame({"decision_date": df.index})
    right = fred[fred_features].reset_index().rename(
        columns={fred.index.name or "index": "observation_date"}
    )
    # Ensure the right-side date column has a stable name regardless of index name.
    right = right.rename(columns={right.columns[0]: "observation_date"})
    joined = pd.merge_asof(
        left.sort_values("decision_date"),
        right.sort_values("observation_date"),
        left_on="decision_date", right_on="observation_date",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7),
    ).set_index("decision_date")
    for feature in fred_features:
        df[feature] = joined[feature].reindex(df.index)

    result = df.loc["2010-01-01":"2025-12-31", DAILY_FEATURES]
    if result.empty:
        raise ValueError("Daily feature generation returned no observations.")
    return result


def rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def rolling_r2(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    fitted = np.polyval(np.polyfit(x, values, 1), x)
    total = float(np.sum((values - np.mean(values)) ** 2))
    return 0.0 if total <= 0 else float(
        1 - np.sum((values - fitted) ** 2) / total
    )


def build_weekly_features(
    spy: pd.DataFrame, vix: pd.DataFrame, qqq: pd.DataFrame, tlt: pd.DataFrame
) -> pd.DataFrame:
    """Build the original 35 Weekly candidates on the 5-session decision grid."""
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
    relative_volume = spy["Volume"] / spy["Volume"].rolling(
        20, min_periods=20
    ).mean()
    vix_close = vix["Close"].reindex(calendar).astype(float)

    features = pd.DataFrame(index=calendar)
    features["weekly_mean_return_5d"] = returns.rolling(5, min_periods=5).mean()
    features["weekly_positive_days_5d"] = returns.gt(0).rolling(5, min_periods=5).sum()
    features["weekly_max_return_5d"] = returns.rolling(5, min_periods=5).max()
    features["weekly_min_return_5d"] = returns.rolling(5, min_periods=5).min()
    features["weekly_return_std_5d"] = returns.rolling(5, min_periods=5).std()
    downside = returns.clip(upper=0.0)
    features["weekly_downside_std_5d"] = np.sqrt(
        downside.pow(2).rolling(5, min_periods=5).mean()
    )

    def max_drawdown(prices: np.ndarray) -> float:
        peak = np.maximum.accumulate(prices)
        return float(np.min(prices / peak - 1.0))

    features["weekly_max_drawdown_5d"] = adj_close.rolling(
        6, min_periods=6
    ).apply(max_drawdown, raw=True)
    features["weekly_mean_intraday_return_5d"] = intraday.rolling(5, min_periods=5).mean()
    features["weekly_mean_overnight_return_5d"] = overnight.rolling(5, min_periods=5).mean()
    features["weekly_compounded_intraday_5d"] = (
        (1 + intraday).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    )
    features["weekly_compounded_overnight_5d"] = (
        (1 + overnight).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    )
    features["weekly_intraday_std_5d"] = intraday.rolling(5, min_periods=5).std()
    features["weekly_mean_relative_volume_5d"] = relative_volume.rolling(5, min_periods=5).mean()
    features["weekly_volume_trend_5d"] = relative_volume.rolling(
        5, min_periods=5
    ).apply(rolling_slope, raw=True)

    spy_5d = adj_close.pct_change(5, fill_method=None)
    qqq_5d = qqq["Adj Close"].astype(float).pct_change(
        5, fill_method=None
    ).reindex(calendar)
    tlt_5d = tlt["Adj Close"].astype(float).pct_change(
        5, fill_method=None
    ).reindex(calendar)
    features["weekly_qqq_relative_return_5d"] = qqq_5d - spy_5d
    features["weekly_tlt_return_5d"] = tlt_5d

    features["weekly_gap_reversal_mean_5d"] = gap.rolling(5, min_periods=5).mean()
    features["weekly_gap_reversal_std_5d"] = gap.rolling(5, min_periods=5).std()
    features["weekly_gap_reversal_positive_days_5d"] = gap.gt(0).rolling(5, min_periods=5).sum()
    features["weekly_gap_reversal_abs_mean_5d"] = gap.abs().rolling(5, min_periods=5).mean()

    # Match the original 06 implementation: FRED is backward-merged to each
    # SPY session, then five SPY-session observations are used for Weekly features.
    fred = pd.concat(
        [read_fred(code) for code in ("DGS2", "DGS10", "DFII10")], axis=1
    ).sort_index()
    fred["yield_curve"] = fred["DGS10"] - fred["DGS2"]
    left = pd.DataFrame({"decision_date": calendar})
    right = fred.reset_index().rename(columns={fred.index.name or "index": "observation_date"})
    right = right.rename(columns={right.columns[0]: "observation_date"})
    aligned = pd.merge_asof(
        left.sort_values("decision_date"), right.sort_values("observation_date"),
        left_on="decision_date", right_on="observation_date",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7),
    ).set_index("decision_date")
    macro = pd.DataFrame(index=calendar)
    for col in ("DGS2", "DGS10", "DFII10", "yield_curve"):
        macro[col] = aligned[col].to_numpy()
    features["weekly_yield_2y_change_5d"] = macro["DGS2"] - macro["DGS2"].shift(4)
    features["weekly_yield_10y_change_5d"] = macro["DGS10"] - macro["DGS10"].shift(4)
    features["weekly_real_yield_change_5d"] = macro["DFII10"] - macro["DFII10"].shift(4)
    features["weekly_yield_curve_change_5d"] = macro["yield_curve"] - macro["yield_curve"].shift(4)
    features["weekly_yield_curve_mean_5d"] = macro["yield_curve"].rolling(5, min_periods=5).mean()

    log_price = np.log(adj_close)
    features["weekly_log_price_slope_5d"] = log_price.rolling(
        5, min_periods=5
    ).apply(rolling_slope, raw=True)
    features["weekly_trend_fit_r2_5d"] = log_price.rolling(
        5, min_periods=5
    ).apply(rolling_r2, raw=True)
    features["weekly_upside_std_5d"] = np.sqrt(
        returns.clip(lower=0.0).pow(2).rolling(5, min_periods=5).mean()
    )
    features["weekly_range_mean_5d"] = (
        (adj_high - adj_low) / adj_close
    ).rolling(5, min_periods=5).mean()
    recent_vol = returns.rolling(2, min_periods=2).std()
    earlier_vol = returns.shift(2).rolling(3, min_periods=3).std()
    features["weekly_volatility_change_5d"] = (
        recent_vol / earlier_vol.replace(0, np.nan) - 1.0
    )
    features["weekly_overnight_std_5d"] = overnight.rolling(5, min_periods=5).std()
    features["weekly_vix_change_5d"] = vix_close.pct_change(5, fill_method=None)
    features["weekly_vix_std_5d"] = vix_close.pct_change(
        fill_method=None
    ).rolling(5, min_periods=5).std()
    features["weekly_qqq_return_5d"] = qqq_5d
    features["weekly_tlt_relative_return_5d"] = tlt_5d - spy_5d

    if list(features.columns) != WEEKLY_FEATURES:
        raise AssertionError(
            f"Weekly feature schema mismatch: expected {len(WEEKLY_FEATURES)}, "
            f"got {len(features.columns)}."
        )

    anchor = calendar.searchsorted(pd.Timestamp("2010-01-01"))
    decision_dates = calendar[anchor::5]
    result = features.loc[decision_dates, WEEKLY_FEATURES]
    if result.empty:
        raise ValueError("Weekly feature generation returned no observations.")
    return result


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    spy, vix, qqq, tlt = (
        read_price(name) for name in ("spy", "vix", "qqq", "tlt")
    )

    daily = build_daily_features(spy, vix, qqq, tlt)
    weekly = build_weekly_features(spy, vix, qqq, tlt)

    daily.to_csv(DAILY_TEMP, index_label="decision_date", date_format="%Y-%m-%d")
    weekly.to_csv(WEEKLY_TEMP, index_label="decision_date", date_format="%Y-%m-%d")

    print(
        f"Feature engineering complete: Daily={len(DAILY_FEATURES)} features/"
        f"{len(daily)} rows; Weekly={len(WEEKLY_FEATURES)} features/"
        f"{len(weekly)} decision rows."
    )
    print("Intermediate feature files are temporary and will be removed by step 04.")


if __name__ == "__main__":
    main()
