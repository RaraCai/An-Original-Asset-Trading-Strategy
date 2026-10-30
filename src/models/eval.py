from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "pooled_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "pooled_mae": float(mean_absolute_error(actual, predicted)),
        "pooled_r2": float(r2_score(actual, predicted)),
    }


def plot_best_model_predictions(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot the pooled-RMSE-best parameter configuration for each model."""
    best_keys = summary.sort_values("pooled_rmse").groupby("model", sort=False).head(1)
    best = predictions.merge(best_keys[["model", "params"]], on=["model", "params"], how="inner")
    models = list(best_keys["model"])
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 3.0 * len(models)), sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], models):
        frame = best.loc[best["model"] == name].sort_values("decision_date")
        ax.plot(frame["decision_date"], frame["actual_return"], label="Actual", linewidth=0.75)
        ax.plot(frame["decision_date"], frame["predicted_return"], label="Predicted", linewidth=0.75)
        params = best_keys.loc[best_keys["model"] == name, "params"].iloc[0]
        ax.set_title(f"{name} | best params: {params}")
        ax.set_ylabel("Return")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Decision date")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
