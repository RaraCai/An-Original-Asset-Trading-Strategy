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
        "pooled_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "pooled_mae": float(mean_absolute_error(actual, predicted)),
        "pooled_r2": float(r2_score(actual, predicted)),
    }


def eligible_training_rows(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    training_start_year: int,
    window_years: int,
) -> pd.DataFrame:
    """Return only observations inside the rolling window whose labels matured before cutoff."""
    window_start = cutoff - pd.DateOffset(years=window_years)
    mask = (
        (data["decision_date"] >= window_start)
        & (data["decision_date"] < cutoff)
        & (data["holding_end_date"] < cutoff)
        & (data["decision_date"].dt.year >= training_start_year)
    )
    return data.loc[mask].copy()


def run_validation(
    data: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly rolling-origin validation for every model/parameter combination.

    Each month: fit on the trailing rolling window, excluding labels that have
    not matured before the first decision date in that month; predict all rows
    in that validation month. No model or hyperparameter is selected here.
    """
    features = config["features"]
    target = config["data"]["target"]
    seed = config["experiment"]["random_seed"]
    training_cfg = config["training"]
    if training_cfg.get("window") != "rolling":
        raise ValueError("This runner currently requires training.window: rolling.")
    if training_cfg.get("retraining") != "monthly":
        raise ValueError("This runner currently requires training.retraining: monthly.")

    start_year = config["validation"]["start_year"]
    end_year = config["validation"]["end_year"]
    validation = data.loc[data["decision_date"].dt.year.between(start_year, end_year)].copy()
    if validation.empty:
        raise ValueError("No validation observations in the configured validation years.")

    validation["_month"] = validation["decision_date"].dt.to_period("M")
    monthly_groups = list(validation.groupby("_month", sort=True))
    prediction_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []

    for model_name, model_config in config["models"].items():
        param_grid = list(ParameterGrid(model_config.get("param_grid", {})))
        for param_index, params in enumerate(param_grid):
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            for month_period, validation_rows in monthly_groups:
                validation_rows = validation_rows.sort_values("decision_date")
                cutoff = pd.Timestamp(validation_rows["decision_date"].min())
                training_rows = eligible_training_rows(
                    data,
                    cutoff,
                    config["data"]["training_start_year"],
                    int(training_cfg["window_years"]),
                )
                if training_rows.empty:
                    raise ValueError(
                        f"No eligible rolling training rows for {month_period}, {model_name}."
                    )

                model = build_model(model_name, model_config, seed, params)
                model.fit(training_rows[features], training_rows[target])
                predicted = np.asarray(model.predict(validation_rows[features]), dtype=float)
                if len(predicted) != len(validation_rows) or not np.isfinite(predicted).all():
                    raise ValueError(f"Invalid predictions for {model_name}, {params}, {month_period}.")

                for row, estimate in zip(validation_rows.itertuples(index=False), predicted):
                    prediction_records.append({
                        "model": model_name,
                        "params": params_json,
                        "n_features": len(features),
                        "decision_date": row.decision_date,
                        "execution_date": row.execution_date,
                        "holding_end_date": row.holding_end_date,
                        "actual_return": float(getattr(row, target)),
                        "predicted_return": float(estimate),
                    })

                fit_records.append({
                    "model": model_name,
                    "params": params_json,
                    "validation_month": str(month_period),
                    "retraining_cutoff": cutoff,
                    "training_start": training_rows["decision_date"].min(),
                    "training_end": training_rows["decision_date"].max(),
                    "n_train": len(training_rows),
                    "n_validation": len(validation_rows),
                    "n_features": len(features),
                })

    predictions = pd.DataFrame(prediction_records)
    fit_log = pd.DataFrame(fit_records)
    return predictions, fit_log, validation.drop(columns="_month")


def aggregate_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled metrics across every validation prediction for each candidate."""
    records = []
    for (model_name, params), frame in predictions.groupby(["model", "params"], sort=False):
        metrics = calculate_metrics(
            frame["actual_return"].to_numpy(dtype=float),
            frame["predicted_return"].to_numpy(dtype=float),
        )
        records.append({
            "model": model_name,
            "params": params,
            "n_features": int(frame["n_features"].iloc[0]) if "n_features" in frame else None,
            "n_valid": len(frame),
            **metrics,
        })
    result = pd.DataFrame(records)
    metric = "pooled_" + "rmse"
    return result.sort_values([metric, "model", "params"], kind="stable").reset_index(drop=True)
