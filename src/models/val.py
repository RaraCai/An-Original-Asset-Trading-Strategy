from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from models import build_model


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'mae': float(mean_absolute_error(actual, predicted)),
        'r2': float(r2_score(actual, predicted)),
    }


def eligible_training_rows(data: pd.DataFrame, cutoff: pd.Timestamp, training_start_year: int) -> pd.DataFrame:
    """Only use labels whose holding period finished before cutoff."""
    return data.loc[
        (data['decision_date'] < cutoff)
        & (data['holding_end_date'] < cutoff)
        & (data['decision_date'].dt.year >= training_start_year)
    ]


def _validation_folds(data: pd.DataFrame, config: dict[str, Any]):
    years = range(config['validation']['start_year'], config['validation']['end_year'] + 1)
    folds = []
    for year in years:
        validation_rows = data.loc[data['decision_date'].dt.year == year]
        if validation_rows.empty:
            raise ValueError(f'No validation rows found for {year}.')
        training_rows = eligible_training_rows(
            data, validation_rows['decision_date'].min(), config['data']['training_start_year']
        )
        if training_rows.empty:
            raise ValueError(f'No eligible training rows before validation year {year}.')
        folds.append((year, training_rows, validation_rows))
    return folds


def run_validation(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate EVERY configuration; select/save no best parameters or fitted models.

    Plot data comes from the FIRST ParameterGrid combination for each model,
    purely as an illustrative fixed representative, NOT a selected winner.
    """
    features = config['features']
    target = config['data']['target']
    seed = config['experiment']['random_seed']
    folds = _validation_folds(data, config)
    search_records = []
    plot_records = []

    for model_name, model_config in config['models'].items():
        for param_index, params in enumerate(ParameterGrid(model_config.get('param_grid', {}))):
            for year, training_rows, validation_rows in folds:
                model = build_model(model_name, model_config, seed, params)
                model.fit(training_rows[features], training_rows[target])
                predicted = model.predict(validation_rows[features])
                search_records.append({
                    'model': model_name,
                    'validation_year': year,
                    'params': json.dumps(params, sort_keys=True),
                    **calculate_metrics(validation_rows[target].to_numpy(), predicted),
                })
                if param_index == 0:
                    for date, actual, estimate in zip(
                        validation_rows['decision_date'], validation_rows[target], predicted
                    ):
                        plot_records.append({
                            'model': model_name,
                            'decision_date': date,
                            'actual_return': actual,
                            'predicted_return': estimate,
                        })

    return pd.DataFrame(search_records), pd.DataFrame(plot_records)


def aggregate_results(search_results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Equal-weight mean of year-level RMSE, MAE and R2 for each candidate."""
    aggregated = (
        search_results.groupby(['model', 'params'], as_index=False)
        .agg(mean_rmse=('rmse', 'mean'), mean_mae=('mae', 'mean'), mean_r2=('r2', 'mean'))
    )
    metric = config['validation']['selection_metric']
    if metric not in {'rmse', 'mae', 'r2'}:
        raise ValueError('validation.selection_metric must be rmse, mae, or r2.')
    ascending = metric != 'r2'
    return aggregated.sort_values(
        [f'mean_{metric}', 'model', 'params'],
        ascending=[ascending, True, True], kind='stable'
    ).reset_index(drop=True)
