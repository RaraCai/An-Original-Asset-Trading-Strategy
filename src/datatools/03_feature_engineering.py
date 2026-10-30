"""Create 21 candidate features using only information available at decision time.
FRED observations are conservatively lagged by at least one SPY trading session.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)


def price_file(name):
    x = pd.read_csv(RAW / f'{name}_raw.csv', index_col='Date', parse_dates=True).sort_index()
    if not x.index.is_unique:
        raise ValueError(f'{name} has duplicate dates')
    return x


def adjusted_close(x):
    return x['Adj Close']

spy, vix, qqq, tlt = (price_file(x) for x in ('spy','vix','qqq','tlt'))
close = adjusted_close(spy)
factor = spy['Adj Close'] / spy['Close']
open_adj = spy['Open'] * factor
volume = spy['Volume']
r1 = close.pct_change(fill_method=None)

df = pd.DataFrame(index=spy.index)
df['return_1d'] = r1
df['momentum_5d'] = close.pct_change(5, fill_method=None)
df['momentum_20d'] = close.pct_change(20, fill_method=None)
for n in (5,20,60):
    df[f'volatility_{n}d'] = r1.rolling(n, min_periods=n).std()
df['ma_gap_5_20'] = close.rolling(5).mean() / close.rolling(20).mean() - 1
df['vix'] = vix['Close'].reindex(df.index)

# Wilder RSI with a 14-day simple-average seed.
change = close.diff()
gain = change.clip(lower=0)
loss = -change.clip(upper=0)
avg_gain = pd.Series(np.nan, index=close.index)
avg_loss = pd.Series(np.nan, index=close.index)
if len(close) > 14:
    avg_gain.iloc[14] = gain.iloc[1:15].mean()
    avg_loss.iloc[14] = loss.iloc[1:15].mean()
for i in range(15, len(close)):
    avg_gain.iloc[i] = (13 * avg_gain.iloc[i-1] + gain.iloc[i]) / 14
    avg_loss.iloc[i] = (13 * avg_loss.iloc[i-1] + loss.iloc[i]) / 14
rs = avg_gain / avg_loss.replace(0, np.nan)
df['rsi_14'] = 100 - 100 / (1 + rs)
df.loc[(avg_loss == 0) & (avg_gain > 0), 'rsi_14'] = 100
df.loc[(avg_loss == 0) & (avg_gain == 0), 'rsi_14'] = 50

df['relative_volume_20d'] = volume / volume.rolling(20).mean()
df['intraday_return'] = close / open_adj - 1
prev_adj_close = close.shift(1)
df['overnight_return'] = open_adj / prev_adj_close - 1
# Compound intraday returns over the last five trading sessions.
df['intraday_momentum_5d'] = (1 + df['intraday_return']).rolling(5).apply(np.prod, raw=True) - 1
df['gap_reversal'] = df['overnight_return'] * df['intraday_return']

qqq_r5 = adjusted_close(qqq).pct_change(5, fill_method=None)
tlt_r5 = adjusted_close(tlt).pct_change(5, fill_method=None)
df['qqq_spy_relative_5d'] = qqq_r5.reindex(df.index) - df['momentum_5d']
df['tlt_return_5d'] = tlt_r5.reindex(df.index)

# Calculate changes on the FRED observation calendar (do not count weekends).
fred = []
for series in ('DGS2','DGS10','DFII10'):
    raw = pd.read_csv(RAW / f'{series}.csv', parse_dates=['date'])
    raw[series] = pd.to_numeric(raw[series], errors='coerce')
    raw = raw[['date',series]].dropna().drop_duplicates('date').sort_values('date')
    fred.append(raw.set_index('date'))
macro = pd.concat(fred, axis=1).sort_index()
# Keep only days where all three series were observed, to define consistent 5/20 steps.
macro = macro.dropna(subset=['DGS2','DGS10','DFII10'])
macro['yield_2y_change_5d'] = macro['DGS2'].diff(5)
macro['yield_10y_change_5d'] = macro['DGS10'].diff(5)
macro['yield_10y_change_20d'] = macro['DGS10'].diff(20)
macro['yield_curve_10y_2y'] = macro['DGS10'] - macro['DGS2']
macro['real_yield_10y_change_5d'] = macro['DFII10'].diff(5)
macro = macro.dropna()
macro = macro.reset_index().rename(columns={'date':'observation_date'})

# Strictly backward merge: FRED observation must be earlier than SPY decision date.
# This deliberately avoids same-day observations whose publication time is uncertain.
left = pd.DataFrame({'decision_date':df.index})
joined = pd.merge_asof(left.sort_values('decision_date'),
                       macro.sort_values('observation_date'),
                       left_on='decision_date', right_on='observation_date',
                       direction='backward', allow_exact_matches=False,
                       tolerance=pd.Timedelta(days=7))
macro_features = ['yield_2y_change_5d','yield_10y_change_5d',
                  'yield_10y_change_20d','yield_curve_10y_2y',
                  'real_yield_10y_change_5d']
joined = joined.set_index('decision_date').reindex(df.index)
for feature in macro_features:
    df[feature] = joined[feature]

FEATURES = [
    'return_1d','momentum_5d','momentum_20d','volatility_5d',
    'volatility_20d','volatility_60d','ma_gap_5_20','vix','rsi_14',
    'relative_volume_20d','intraday_return','overnight_return',
    'intraday_momentum_5d','gap_reversal','qqq_spy_relative_5d',
    'tlt_return_5d',*macro_features]
assert len(FEATURES) == 21 and len(set(FEATURES)) == 21
result = df.loc['2010-01-01':'2025-12-31',FEATURES]
if result.empty:
    raise ValueError('No observations in research period')
print('FEATURES:',len(FEATURES),'ROWS:',len(result))
print('DATE RANGE:',result.index.min(),result.index.max())
print('MISSING:\n',result.isna().sum().to_string())
path = OUT / 'spy_features.csv'
result.to_csv(path, index_label='decision_date', date_format='%Y-%m-%d')
print('Saved:',path)
