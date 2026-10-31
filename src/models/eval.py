"""Shared regression metrics and validation/test prediction visualizations."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same non-zero length.")
    actual_std = float(np.std(actual, ddof=1)) if len(actual) > 1 else 0.0
    prediction_std = float(np.std(predicted, ddof=1)) if len(predicted) > 1 else 0.0
    return {
        "pooled_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "pooled_mae": float(mean_absolute_error(actual, predicted)),
        "pooled_r2": float(r2_score(actual, predicted)),
        "actual_mean": float(np.mean(actual)),
        "actual_std": actual_std,
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": prediction_std,
        "prediction_std_ratio": prediction_std / actual_std if actual_std > 0 else np.nan,
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
    }


def plot_best_model_predictions(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot validation actual/predicted returns for each model's best pooled-RMSE candidate."""
    best = summary.sort_values("pooled_rmse").groupby("model", sort=False).head(1)
    selected = predictions.merge(best[["model", "params"]], on=["model", "params"], how="inner")
    model_names = list(best["model"])
    if not model_names:
        raise ValueError("No model candidates available to plot.")

    fig, axes = plt.subplots(
        len(model_names), 1,
        figsize=(14, max(3.5, 2.8 * len(model_names))),
        sharex=True, squeeze=False,
    )
    for ax, model_name in zip(axes[:, 0], model_names):
        frame = selected.loc[selected["model"].eq(model_name)].sort_values("decision_date")
        ax.plot(frame["decision_date"], frame["actual_return"], label="Actual", linewidth=0.8)
        ax.plot(frame["decision_date"], frame["predicted_return"], label="Predicted", linewidth=0.8)
        params = best.loc[best["model"].eq(model_name), "params"].iloc[0]
        ax.set_title(f"{model_name} | {params}")
        ax.set_ylabel("Return")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Decision date")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    """Write pooled and annual metric tables for a single frozen model's predictions."""
    required = {"decision_date", "actual_return", "predicted_return"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    metrics = calculate_metrics(
        frame["actual_return"].to_numpy(dtype=float),
        frame["predicted_return"].to_numpy(dtype=float),
    )
    pooled = pd.DataFrame([{
        "period": prefix,
        "n_valid": len(frame),
        **metrics,
    }])
    frame["year"] = frame["decision_date"].dt.year
    annual_rows = []
    for year, group in frame.groupby("year", sort=True):
        annual_rows.append({
            "year": int(year),
            "n_valid": len(group),
            **calculate_metrics(
                group["actual_return"].to_numpy(dtype=float),
                group["predicted_return"].to_numpy(dtype=float),
            ),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(output_dir / f"{prefix}_annual_metrics.csv", index=False)
    return pooled
