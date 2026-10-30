"""Create separate daily 1D and every-5-trading-days 5D datasets.
Signal is at decision close; trade at next session open, exit H sessions later.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
spy = pd.read_csv(RAW / 'spy_raw.csv', index_col='Date', parse_dates=True).sort_index()
features = pd.read_csv(OUT / 'spy_features.csv', index_col='decision_date',
                       parse_dates=True).sort_index()
if not spy.index.is_unique or not features.index.is_unique:
    raise ValueError('Duplicate trading dates')
if not features.index.isin(spy.index).all():
    raise ValueError('Some feature dates do not exist in SPY calendar')

adjusted_open = spy['Open'] * (spy['Adj Close'] / spy['Close'])
calendar = spy.index

# Stable schedule anchored at first SPY trading session on or after 2010-01-01.
# Use the complete calendar, not a filtered feature frame, to avoid schedule drift.
anchor = calendar.searchsorted(pd.Timestamp('2010-01-01'))

for name, horizon in [('daily',1), ('weekly',5)]:
    eligible = pd.Series(False, index=calendar)
    if name == 'daily':
        eligible.iloc[anchor:] = True
    else:
        eligible.iloc[anchor::5] = True

    df = features.copy()
    df['execution_date'] = pd.Series(calendar, index=calendar).shift(-1).reindex(df.index)
    df['holding_end_date'] = pd.Series(calendar, index=calendar).shift(-(horizon+1)).reindex(df.index)
    df['target_return'] = (adjusted_open.shift(-(horizon+1)) /
                           adjusted_open.shift(-1) - 1).reindex(df.index)
    df = df.loc[eligible.reindex(df.index).fillna(False)]
    df = df.dropna(subset=['execution_date','holding_end_date','target_return'])
    # No target beyond research period; the final incomplete position is excluded.
    df = df.loc[df['holding_end_date'] <= pd.Timestamp('2025-12-31')]
    df = df[['execution_date','holding_end_date'] + features.columns.tolist() + ['target_return']]
    if df.empty:
        raise RuntimeError(f'{name}: empty labeled dataset')
    assert (df['execution_date'] > df.index).all()
    assert (df['holding_end_date'] > df['execution_date']).all()
    if name == 'weekly':
        # Each 5D trade ends exactly when the next 5D trade begins.
        assert (df['execution_date'].iloc[1:].values == df['holding_end_date'].iloc[:-1].values).all(), 'Weekly schedule has a gap; inspect data and trading calendar'
    dest = OUT / f'spy_labeled_{name}.csv'
    df.to_csv(dest, index_label='decision_date', date_format='%Y-%m-%d')
    print(f'{name}: {len(df)} rows, {len(features.columns)} features, {df.index.min().date()} to {df.index.max().date()}')
    print('Missing feature values:', int(df[features.columns].isna().sum().sum()))
    print('Saved:', dest)
print('LABEL CREATION COMPLETE')
