# SPY Trading Strategy — Data Preparation and Model Research

## 1. Project Overview

This project prepares daily SPY and VIX data and compares four regression models for predicting the next open-to-open SPY return.

- Data period: 2010–2025, with data from 2009 used for feature warm-up
- Signal time: after the market close on decision date `t`
- Execution time: the open of the next trading day
- Target: adjusted open-to-open return from the next open to the following open

## 2. Source and Data Layout

```text
data/
  raw/                 # Downloaded SPY and VIX CSV files
  processed/           # Engineered features, labels, and annual CSVs
src/
  datatools/           # Data download, checks, feature/label creation, validation
  models/
    config.yaml        # Experiment, model, split, and output configuration
    models.py          # Candidate estimators and final model fitting
    val.py             # Expanding-window folds, parameter search, selection, predictions
    eval.py            # Prediction metrics, summaries, and plots (no model fitting)
    train.py           # Experiment orchestration and held-out test
outputs/
  models/              # Metrics, predictions, plots, configuration, fitted models
```

The CSV data remains in the project-root `data/` directory. Run commands below from the project root.

## 3. Environment Setup

Recommended Python version: 3.11.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 4. Run Instructions

To rebuild and validate the processed data, run:

```bash
python src/datatools/01_download_data.py
python src/datatools/02_check_data.py
python src/datatools/03_feature_engineering.py
python src/datatools/04_create_labels.py
python src/datatools/05_export_annual.py
python src/datatools/06_final_validation.py
python src/datatools/07_verify_features.py
```

To train and evaluate the candidate models using the existing processed dataset:

```bash
python src/models/train.py
```

The workflow reads `src/models/config.yaml` and the processed dataset path configured there, then writes results to `outputs/models/`.

## 5. Features

The dataset contains 11 features. All are calculated using information from the decision date or earlier.

| Feature | Description |
|---|---|
| `return_1d` | 1-day adjusted close return |
| `momentum_5d` | 5-day cumulative return |
| `momentum_20d` | 20-day cumulative return |
| `volatility_5d` | 5-day rolling return volatility |
| `volatility_20d` | 20-day rolling return volatility |
| `volatility_60d` | 60-day rolling return volatility |
| `ma_gap_5_20` | 5-day / 20-day moving average ratio minus 1 |
| `vix` | VIX closing value |
| `rsi_14` | 14-day Wilder RSI |
| `relative_volume_20d` | Volume divided by 20-day average volume |
| `intraday_return` | Adjusted close-to-open return |

`target_return` is the response variable and must never be included as a feature.

## 6. Candidate Models and Training

The script compares four regression models:

- **Random Forest:** a nonlinear tree ensemble. Maximum tree depth and minimum leaf size are tuned to control overfitting. A fixed random seed makes its fit reproducible.
- **OLS:** unregularized linear regression; it provides a simple linear reference candidate.
- **Lasso:** linear regression with L1 regularization, which can shrink coefficients to zero.
- **Ridge:** linear regression with L2 regularization, which shrinks coefficients without generally eliminating them.

Linear models use a `StandardScaler` inside a scikit-learn pipeline, so scaling is fit only on the training rows for each fold. Random Forest does not require feature scaling. Missing feature/target rows are removed; extreme values are retained.

The YAML config records model parameter grids, data splits, random seed, and output paths. `val.py` owns both parameter search and the selected models' expanding-window validation predictions; `train.py` does not implement a separate validation loop. For each validation year, each candidate setting is fit on the eligible expanding training window and predicts that year. The setting with the best average yearly validation RMSE is selected. `eval.py` accepts the resulting actual values, predictions, and metadata to calculate RMSE, MAE, and R² and generate summaries and plots; it never receives or fits a model.

`train.py` coordinates configuration loading, validation, final model fitting, held-out prediction, evaluation, and artifact saving. Final model objects are constructed in `models.py` and fitted on eligible 2010–2021 rows before testing.

## 7. Data Splits and Leakage Prevention

| Dataset | Period | Use |
|---|---|---|
| Initial training history | 2010–2017 | Initial expanding-window training data |
| Forward validation | 2018–2021 | One validation fold per year; training expands each year |
| Final test | 2022–2025 | One final held-out evaluation after model selection |

At each validation/test boundary, training rows are restricted to decision dates before the period and to labels whose `holding_end_date` is before the first decision date of that period. This ensures each training target was observable at the model's training cutoff. The final test period is not used to select parameters.

## 8. Model Outputs

Running the model script creates:

- `src/models/config.yaml`: feature list, year splits, selection metric, parameter grids, random seed, and output locations
- `outputs/models/config_used.yaml`: copy of the experiment configuration used for the run
- `outputs/models/validation_search_results.csv`: parameter-by-fold scores and training/validation boundary details
- `outputs/models/selected_params.json`: selected hyperparameters per candidate model
- `outputs/models/validation_summary.csv`: selected parameters and average forward-validation metrics per model
- `outputs/models/validation_metrics.csv`: selected model's metrics for each validation year
- `outputs/models/validation_predictions.csv`: validation-period actual and predicted returns
- `outputs/models/test_metrics.csv`: final held-out RMSE, MAE, and R² per model
- `outputs/models/test_predictions.csv`: test-period actual and predicted returns
- `outputs/models/validation_predictions.png` and `outputs/models/test_predictions.png`: actual versus predicted return plots
- `outputs/models/fitted_models/*.joblib`: final models refit on eligible 2010–2021 data

These are forecasting-model results only. The script does not convert forecasts to positions or simulate a trading strategy.

## 9. Data Validation

The data-preparation scripts check:

- Missing values and duplicate dates
- Invalid prices and OHLC consistency
- Trading-date alignment
- Target return calculations
- Feature consistency
- Annual CSV structure

The current processed annual datasets contain 4,022 observations in total. The final two decision dates in 2025 are excluded because future opening prices are unavailable within the downloaded period.

## 10. Important Notes

- Fit all transformations using training data only.
- Do not use future test performance to select model parameters.
- Use only labels that are already observable at each model training date.
- Adjusted OHLC data are used for research returns; actual historical execution prices may differ.
- This project currently sets trading costs to zero; strategy simulation is outside the candidate-model workflow.
