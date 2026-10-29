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
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def eligible_training_rows(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    training_start_year: int,
) -> pd.DataFrame:
    """Keep only training labels observable before the prediction cutoff."""
    return data.loc[
        (data["decision_date"] < cutoff)
        & (data["holding_end_date"] < cutoff)
        & (data["decision_date"].dt.year >= training_start_year)
    ]


def _validation_folds(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    validation_config = config["validation"]
    years = range(validation_config["start_year"], validation_config["end_year"] + 1)
    folds = []
    for year in years:
        validation_rows = data.loc[data["decision_date"].dt.year == year]
        if validation_rows.empty:
            raise ValueError(f"No validation rows found for {year}.")

        cutoff = validation_rows["decision_date"].min()
        training_rows = eligible_training_rows(
            data,
            cutoff,
            config["data"]["training_start_year"],
        )
        if training_rows.empty:
            raise ValueError(f"No eligible training rows before validation year {year}.")
        folds.append((year, training_rows, validation_rows))
    return folds


def run_validation(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Tune candidates and generate selected-parameter predictions in forward folds."""
    features = config["features"]
    target = config["data"]["target"]
    seed = config["experiment"]["random_seed"]
    folds = _validation_folds(data, config)
    search_records: list[dict[str, Any]] = []

    for model_name, model_config in config["models"].items():
        parameter_grid = list(ParameterGrid(model_config.get("param_grid", {})))
        for params in parameter_grid:
            for year, training_rows, validation_rows in folds:
                model = build_model(model_name, model_config, seed, params)
                model.fit(training_rows[features], training_rows[target])
                predicted = model.predict(validation_rows[features])
                search_records.append(
                    {
                        "model": model_name,
                        "validation_year": year,
                        "params": json.dumps(params, sort_keys=True),
                        **calculate_metrics(
                            validation_rows[target].to_numpy(), predicted
                        ),
                        "training_rows": len(training_rows),
                        "validation_rows": len(validation_rows),
                        "training_last_label_observed_date": training_rows[
                            "holding_end_date"
                        ].max(),
                        "validation_first_decision_date": validation_rows[
                            "decision_date"
                        ].min(),
                    }
                )

    search_results = pd.DataFrame(search_records)
    selection_metric = config["validation"]["selection_metric"]
    if selection_metric not in {"rmse", "mae", "r2"}:
        raise ValueError(
            "validation.selection_metric must be one of: rmse, mae, r2."
        )
    ascending = selection_metric != "r2"
    parameter_scores = (
        search_results.groupby(["model", "params"], as_index=False)[selection_metric]
        .mean()
        .sort_values(
            ["model", selection_metric, "params"],
            ascending=[True, ascending, True],
        )
    )
    best_params = {
        model_name: json.loads(rows.iloc[0]["params"])
        for model_name, rows in parameter_scores.groupby("model", sort=True)
    }

    prediction_records: list[dict[str, Any]] = []
    for year, training_rows, validation_rows in folds:
        for model_name, model_config in config["models"].items():
            params = best_params[model_name]
            model = build_model(model_name, model_config, seed, params)
            model.fit(training_rows[features], training_rows[target])
            predicted = model.predict(validation_rows[features])
            for row, prediction in zip(validation_rows.itertuples(index=False), predicted):
                prediction_records.append(
                    {
                        "model": model_name,
                        "validation_year": year,
                        "decision_date": row.decision_date,
                        "execution_date": row.execution_date,
                        "holding_end_date": row.holding_end_date,
                        "actual_return": getattr(row, target),
                        "predicted_return": prediction,
                        "selected_params": json.dumps(params, sort_keys=True),
                    }
                )

    predictions = pd.DataFrame(prediction_records).sort_values(
        ["model", "decision_date"]
    )
    return best_params, predictions, search_results
