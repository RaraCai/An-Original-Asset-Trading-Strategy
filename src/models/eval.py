from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics from observed and predicted values only."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    group_columns: Sequence[str] = ("model",),
    actual_column: str = "actual_return",
    predicted_column: str = "predicted_return",
) -> pd.DataFrame:
    """Persist prediction data, grouped metrics, and actual-vs-predicted plots."""
    required = {actual_column, predicted_column, "model", "decision_date"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Cannot evaluate an empty prediction dataset.")

    output_dir.mkdir(parents=True, exist_ok=True)
    data = predictions.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data.to_csv(
        output_dir / f"{split_name}_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    groups = list(group_columns)
    grouped = (
        data.groupby(groups, sort=True, dropna=False)
        if groups
        else [((), data)]
    )
    metric_records = []
    for key, frame in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        metrics = calculate_metrics(
            frame[actual_column].to_numpy(dtype=float),
            frame[predicted_column].to_numpy(dtype=float),
        )
        metric_records.append(
            {
                **dict(zip(groups, key_values)),
                "observations": len(frame),
                **metrics,
            }
        )
    metrics_frame = pd.DataFrame(metric_records)
    metrics_frame.to_csv(output_dir / f"{split_name}_metrics.csv", index=False)
    plot_predictions(
        data,
        output_dir / f"{split_name}_predictions.png",
        split_name.replace("_", " ").title(),
        actual_column,
        predicted_column,
    )
    return metrics_frame


def summarize_validation_metrics(metrics_by_year: pd.DataFrame) -> pd.DataFrame:
    """Summarize year-level metrics without pooling the yearly observations."""
    required = {"model", "validation_year", "rmse", "mae", "r2"}
    missing = required - set(metrics_by_year.columns)
    if missing:
        raise ValueError(f"Validation metrics are missing columns: {sorted(missing)}")
    return (
        metrics_by_year.groupby("model", as_index=False)
        .agg(
            validation_years=("validation_year", "nunique"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
        )
        .sort_values("model")
    )


def plot_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
    actual_column: str = "actual_return",
    predicted_column: str = "predicted_return",
) -> None:
    """Plot observed and predicted values; this function never fits a model."""
    model_names = sorted(predictions["model"].unique())
    figure, axes = plt.subplots(
        len(model_names),
        1,
        figsize=(14, 3.2 * len(model_names)),
        sharex=True,
        squeeze=False,
    )
    for axis, model_name in zip(axes[:, 0], model_names):
        frame = predictions.loc[predictions["model"] == model_name].sort_values(
            "decision_date"
        )
        axis.plot(
            frame["decision_date"],
            frame[actual_column],
            label="Actual",
            color="#6888F5",
            linewidth=0.8,
        )
        axis.plot(
            frame["decision_date"],
            frame[predicted_column],
            label="Predicted",
            color='#D77071',
            linewidth=0.8,
        )
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_ylabel("Return")
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)

    axes[-1, 0].set_xlabel("Decision date")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
