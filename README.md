# SPY Daily and Weekly Trading Strategy

[📺see short video to summarize this project on **Youtube**](https://youtu.be/JjpH00E46NY?si=mfvpbsYqV3RUmG1k)  

This repository builds and evaluates daily and weekly, long-only SPY timing strategies. It is an academic research/backtesting project, not a live-trading system or investment advice.

> **Start here:** if the supplied processed data are present, install the dependencies and run the validation and independent-test commands below. Each run prints the folder where it saved its results.

## What the project does

The project predicts SPY's next open-to-open return, converts predictions to portfolio weights, and compares the resulting strategies with SPY buy-and-hold and a rolling historical-mean forecast. It has two time scales:

| Pipeline | Signal schedule | Target / holding period | Frozen model |
|---|---|---|---|
| Daily | Each trading day, after that day's close | Adjusted open-to-open return over the next trading day | Lasso, `alpha=0.000225`, 14 selected features |
| Weekly | Every fifth SPY trading session, after that day's close | Adjusted open-to-open return over the following five trading days | Lasso, `alpha=0.0003`, 14 selected features |

Orders are modeled at the next trading day's open. Research configurations start with 21 daily and 35 weekly candidate features; the frozen final configurations use smaller feature subsets. Research grids compare OLS, Ridge, Lasso, Random Forest, and XGBoost; the frozen final model for each pipeline is Lasso. Inputs include SPY market data, VIX, QQQ, TLT, and Treasury-yield series.

The supplied processed datasets currently contain 4,022 Daily rows and 804 Weekly rows. Run `python src\datatools\06_verify_features.py` to recompute and check the feature values and target-date alignment.

Validation uses 2018–2021. The final-model test configuration uses 2022–2025. Models use rolling five-year training windows and monthly retraining; only labels whose holding periods have ended before each training cutoff are eligible. Model selection and strategy evaluation are distinct: `src/models/test.py` evaluates forecasts, while the strategy scripts translate forecasts into positions and account for trading costs.

**Test-period caveat:** the strategy-search setup records that 2022–2025 had already been inspected during earlier work. Treat strategy results for that period as retrospective evidence, not as a pristine, untouched holdout.

## Strategy rules and evaluation

The strategy backtest compares four forecast-to-weight rules: the original thresholded mapping, a base-exposure mapping, a volatility-scaled mapping, and a downside-adjusted mapping. Weights are bounded between 0 and 1 (long-only, no leverage). Candidate mapping parameters are selected on validation Sharpe ratio, with annualized return and maximum drawdown as tie-breakers.

The backtest also reports:

- **Benchmarks:** SPY buy-and-hold and the historical-mean forecast passed through the original mapping.
- **Costs:** the configured daily assumption is 0 bps; the weekly base case is 5 bps. The strategy test can also report weekly sensitivity at 0, 1, 5, and 10 bps.
- **Metrics:** annualized return and volatility, Sharpe ratio, maximum drawdown, turnover, and transaction costs, plus annual breakdowns and equity/drawdown plots.

These are historical simulations with simplified cost assumptions. They do not model market impact, taxes, financing, or real-world execution constraints.

## Repository map

```text
configs/
  daily_research.yaml, weekly_research.yaml   # candidate model and feature pools
  daily_final.yaml, weekly_final.yaml         # frozen final-model settings
  daily_strategy.yaml, weekly_strategy.yaml   # frozen strategy mappings
data/
  raw/                                         # SPY, VIX, QQQ, TLT, and FRED input CSVs
  processed/                                   # labeled daily and weekly datasets
src/
  datatools/                                   # download, check, feature, and label scripts
  models/                                      # config, training, validation, and test code
  experiments/                                 # feature analysis, ablation, refinement, backtests
outputs/                                       # timestamped run artifacts
```

Run commands from the repository root. Generated outputs are written under `outputs/`; each run prints its timestamped output directory.

## Setup

Python 3.11 or newer is recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
```

The current `data/raw/` and `data/processed/` CSVs can be used as supplied. To download and rebuild them, the FRED API key must be available as the `FRED_KEY` environment variable; do not commit the key. For example, in PowerShell:

```powershell
$env:FRED_KEY = "<your FRED API key>"
python src\datatools\01_download_data.py
```

Then run the remaining data steps:

```powershell
python src\datatools\02_check_data.py
python src\datatools\03_feature_engineering.py
python src\datatools\04_create_labels.py
python src\datatools\05_export_annual.py
python src\datatools\06_verify_features.py
```

`05_export_annual.py` is an additional daily-data export; the model pipelines use the labeled daily and weekly CSVs. Market prices are retrieved from [Yahoo Finance](https://finance.yahoo.com/) through `yfinance`. Treasury yields come from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html): [DGS2](https://fred.stlouisfed.org/series/DGS2), [DGS10](https://fred.stlouisfed.org/series/DGS10), and [DFII10](https://fred.stlouisfed.org/series/DFII10).

## Reproduce model validation and test

From the repository root, run validation for the frozen daily and weekly configurations:

```bash
python -m src.models.train --config configs/daily_final.yaml
python -m src.models.train --config configs/weekly_final.yaml
```

Each validation run creates a timestamped `outputs/daily_.../` or `outputs/weekly_.../` directory containing `all_models_validation.csv`, `validation_predictions.csv`, `monthly_fit_log.csv`, `validation_predictions.png`, and a snapshot of the configuration.

Run the configured 2022–2025 final-model evaluation:

```bash
python -m src.models.test --config configs/daily_final.yaml
python -m src.models.test --config configs/weekly_final.yaml
```

Each test run creates a timestamped `outputs/daily_independent_test_.../` or `outputs/weekly_independent_test_.../` directory. Main artifacts include `test_summary.csv`, `test_metrics.csv`, `test_annual_metrics.csv`, `test_predictions.csv`, `test_monthly_fit_log.csv`, and `test_setup.yaml`. The corresponding `historical_mean_*.csv` files contain the forecast-baseline results. The test script reports prediction metrics; run the strategy backtest below for trading-performance metrics and plots.

## Reproduce the strategy backtest

Strategy search requires the Daily and Weekly validation-refinement output directories. Those directories must contain the validation predictions for the frozen Lasso alpha and `monthly_historical_mean_predictions.csv` (the Daily file is named `alpha_refinement_predictions.csv`; the Weekly file is `weekly_alpha_predictions.csv`). Strategy test also requires the model-test directories produced above.

After obtaining those output paths from each script's `Results saved:` message, run strategy selection:

```powershell
python src\experiments\strategy\strategy_search.py `
  --daily-validation-dir "<daily-refinement-output>" `
  --weekly-validation-dir "<weekly-refinement-output>"
```

This writes `validation_strategy_search.csv` under a timestamped `outputs/strategy_search_.../` directory and saves the selected mappings to `configs/daily_strategy.yaml` and `configs/weekly_strategy.yaml`.

Then run the independent-period strategy evaluation (replace the four directory placeholders with the paths printed by the corresponding model-test and refinement runs):

```powershell
python src\experiments\strategy\strategy_test.py `
  --daily-test-dir "<daily-independent-test-output>" `
  --weekly-test-dir "<weekly-independent-test-output>" `
  --daily-validation-dir "<daily-refinement-output>" `
  --weekly-validation-dir "<weekly-refinement-output>"
```

The strategy-test output directory contains `individual_test_metrics.csv`, `individual_test_returns.csv`, `individual_test_annual_metrics.csv`, `benchmark_comparison.csv`, pipeline equity/drawdown plots, and Daily-versus-Weekly metric charts. Weekly transaction-cost sensitivity is also saved separately. Strategy search rewrites the two strategy YAML files, so retain or back up any settings you want to keep before rerunning it.

## Reproducibility notes

- Configurations use relative project paths and the fixed random seed `5205`.
- Feature and label integrity can be checked with `python src\datatools\06_verify_features.py`.
- Missing or incomplete rows are handled by the data and model loaders; training cutoffs exclude labels that have not matured.
- Raw downloads can change as providers revise data. Record the input data and configuration snapshots when comparing runs.
- The project is research code. It does not place orders or guarantee that simulated results can be realized.
