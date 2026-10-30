"""Download SPY/VIX/QQQ/TLT and FRED DGS2/DGS10/DFII10.
Run from project root: python src/datatools/01_download_data.py
Set FRED_KEY in the environment; never hardcode credentials.
"""
from pathlib import Path
import os
import time
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
RAW.mkdir(parents=True, exist_ok=True)
START, END = '2009-01-01', '2026-01-01'  # end is exclusive for yfinance

for ticker, name in [('SPY','spy'), ('^VIX','vix'), ('QQQ','qqq'), ('TLT','tlt')]:
    frame = yf.download(ticker, start=START, end=END, auto_adjust=False,
                        actions=True, progress=False)
    if frame.empty:
        raise RuntimeError(f'No price data for {ticker}')
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = {'Open','Close','Adj Close'}
    if not required.issubset(frame.columns):
        raise ValueError(f'{ticker}: missing {required - set(frame.columns)}')
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = 'Date'
    frame.to_csv(RAW / f'{name}_raw.csv')
    print(f'{ticker}: {len(frame)} rows, {frame.index.min().date()} to {frame.index.max().date()}')

key = os.getenv('FRED_KEY')
if not key:
    raise EnvironmentError('Set FRED_KEY in the current terminal before running (PowerShell: $env:FRED_KEY="YOUR_KEY").')

session = requests.Session()
for series in ('DGS2','DGS10','DFII10'):
    params = {'series_id':series,'api_key':key,'file_type':'json',
              'observation_start':START,'observation_end':'2025-12-31'}
    for attempt in range(4):
        try:
            response = session.get('https://api.stlouisfed.org/fred/series/observations',
                                   params=params, timeout=40)
            response.raise_for_status()
            observations = response.json()['observations']
            frame = pd.DataFrame({'date':[r['date'] for r in observations],
                                  series:[r['value'] for r in observations]})
            frame[series] = pd.to_numeric(frame[series], errors='coerce')
            frame['date'] = pd.to_datetime(frame['date'])
            if frame[series].notna().sum() < 1000:
                raise ValueError(f'{series}: insufficient observations')
            frame.to_csv(RAW / f'{series}.csv', index=False, date_format='%Y-%m-%d')
            print(f'{series}: {frame[series].notna().sum()} valid observations')
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt == 3:
                raise RuntimeError(f'Failed to download {series}: {exc}') from exc
            time.sleep(2 ** attempt)
print('DOWNLOAD COMPLETE')
