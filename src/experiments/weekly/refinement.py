"""Weekly Lasso alpha refinement for the historical R6-D 14-feature set.

Run:
  python -m src.experiments.weekly.refinement \
    --config configs/weekly_research.yaml \
    --ablation-dir outputs/weekly_ablation_round6_...

The R6-D feature list and alpha grid are taken from the previous active
Weekly configuration. After the search, the script writes configs/weekly_final.yaml
using the lowest pooled-RMSE alpha. Review the generated file and results before
running the independent test.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import get_output_root, get_pipeline_features, load_config
from src.models.data import load_data
from src.models.models import build_model
from src.models.val import calculate_metrics, eligible_training_rows


R6_D_FEATURES = [
    "weekly_mean_return_5d",
    "weekly_positive_days_5d",
    "weekly_max_return_5d",
    "weekly_min_return_5d",
    "weekly_log_price_slope_5d",
    "weekly_trend_fit_r2_5d",
    "weekly_mean_relative_volume_5d",
    "weekly_volume_trend_5d",
    "weekly_gap_reversal_mean_5d",
    "weekly_gap_reversal_abs_mean_5d",
    "weekly_gap_reversal_std_5d",
    "weekly_yield_curve_change_5d",
    "weekly_vix_change_5d",
    "weekly_vix_std_5d",
]

# Exact R6-D Lasso refinement grid from the prior Weekly config.
ALPHA_GRID = [
    0.00025, 0.00030, 0.00035, 0.00040, 0.00045, 0.00050,
    0.00055, 0.00060, 0.00065, 0.00070, 0.00075,
]


def prepare_monthly_windows(data: pd.DataFrame, config: dict):
    start_year = int(config["validation"]["start_year"])
    end_year = int(config["validation"]["end_year"])
    target = config["data"]["target"]
    validation = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date")
    if validation.empty:
        raise ValueError("No Weekly validation observations.")

    windows = {}
    historical_mean_rows = []
    for month_period, month_rows in validation.groupby(
        validation["decision_date"].dt.to_period("M"), sort=True
    ):
        month = str(month_period)
        month_rows = month_rows.sort_values("decision_date").copy()
        cutoff = pd.Timestamp(month_rows["decision_date"].min())
        training = eligible_training_rows(
            data,
            cutoff,
            int(config["data"]["training_start_year"]),
            int(config["training"]["window_years"]),
        )
        if training.empty:
            raise ValueError(f"Empty training window for {month}.")
        if not (training["holding_end_date"] < cutoff).all():
            raise AssertionError(f"Unmatured labels found in {month}.")
        windows[month] = (training, month_rows)
        mean_prediction = float(training[target].mean())
        for row in month_rows.itertuples(index=False):
            historical_mean_rows.append({
                "decision_date": row.decision_date,
                "validation_month": month,
                "actual_return": float(getattr(row, target)),
                "predicted_return": mean_prediction,
            })
    return windows, pd.DataFrame(historical_mean_rows)


def write_weekly_final_config(
    config: dict,
    alpha: float,
    metrics: dict,
    final_path: Path,
) -> None:
    final_config = {
        "default_pipeline": "weekly",
        "data": {
            "daily_path": config["data"]["daily_path"],
            "weekly_path": config["data"]["weekly_path"],
            "target": config["data"]["target"],
            "training_start_year": int(config["data"]["training_start_year"]),
        },
        "pipelines": {
            "weekly": {
                "data_key": "weekly_path",
                "features": R6_D_FEATURES,
                "signal_interval_trading_days": 5,
                "prediction_horizon_trading_days": 5,
                "holding_period_trading_days": 5,
                "execution": "next_trading_day_open",
            }
        },
        "validation": dict(config["validation"]),
        "test": dict(config["test"]),
        "training": dict(config["training"]),
        "models": {
            "lasso": {
                "max_iter": int(config["models"]["lasso"].get("max_iter", 200000)),
                "param_grid": {"regressor__alpha": [float(alpha)]},
            }
        },
        "experiment": {
            "name": "weekly_final_model",
            "random_seed": int(config["experiment"]["random_seed"]),
            "output_dir": config["experiment"].get("output_dir", "outputs"),
        },
        "final_model_record": {
            "name": "lasso",
            "params": {"regressor__alpha": float(alpha)},
            "validation_metrics": {
                key: float(metrics[key])
                for key in ("pooled_rmse", "pooled_mae", "pooled_r2")
            },
            "validation_period": [
                int(config["validation"]["start_year"]),
                int(config["validation"]["end_year"]),
            ],
            "independent_test_period": [
                int(config["test"]["start_year"]),
                int(config["test"]["end_year"]),
            ],
            "feature_set": "R6-D",
            "n_features": len(R6_D_FEATURES),
        },
    }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(
        yaml.safe_dump(final_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly R6-D Lasso alpha refinement.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument(
        "--final-config", type=Path, default=Path("configs/weekly_final.yaml")
    )
    parser.add_argument(
        "--overwrite-final-config", action="store_true",
        help="Allow replacing an existing weekly_final.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config["default_pipeline"] != "weekly":
        raise ValueError("Weekly refinement requires weekly_research.yaml.")
    if config["training"].get("window") != "rolling" or config["training"].get("retraining") != "monthly":
        raise ValueError("Expected rolling-window training and monthly retraining.")

    candidate_features = get_pipeline_features(config, "weekly")
    if len(candidate_features) != 35 or not set(R6_D_FEATURES).issubset(candidate_features):
        raise ValueError("Weekly research config must contain the Full-35 pool including R6-D.")

    ablation_dir = args.ablation_dir
    if not ablation_dir.is_absolute():
        ablation_dir = Path.cwd() / ablation_dir
    ablation_summary_path = ablation_dir / "ablation_summary.csv"
    if not ablation_summary_path.is_file():
        raise FileNotFoundError(ablation_summary_path)

    ablation_summary = pd.read_csv(ablation_summary_path)
    selected_row = ablation_summary.loc[
        ablation_summary["experiment"].eq("R6_D_B_plus_gap_std")
        & ablation_summary["model"].eq("lasso")
    ]
    if len(selected_row) != 1:
        raise ValueError("Ablation output must contain exactly one R6-D Lasso result.")
    selected_features = R6_D_FEATURES
    if int(selected_row.iloc[0]["n_features"]) != len(selected_features):
        raise ValueError("R6-D ablation row does not report 14 features.")

    data = load_data(config, "weekly")
    missing = sorted(set(selected_features) - set(data.columns))
    if missing:
        raise ValueError(f"Weekly dataset is missing R6-D features: {missing}")

    windows, historical_mean = prepare_monthly_windows(data, config)
    target = config["data"]["target"]
    lasso_config = config["models"]["lasso"]
    mean_metrics = calculate_metrics(
        historical_mean["actual_return"].to_numpy(dtype=float),
        historical_mean["predicted_return"].to_numpy(dtype=float),
    )
    mean_actual_std = float(historical_mean["actual_return"].std(ddof=1))
    mean_prediction_std = float(historical_mean["predicted_return"].std(ddof=1))
    mean_metrics["prediction_std_ratio"] = (
        mean_prediction_std / mean_actual_std if mean_actual_std > 0 else np.nan
    )
    mean_metrics["oos_r2_vs_monthly_historical_mean"] = 0.0

    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"weekly_alpha_refinement_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)

    result_rows = [{
        "model": "monthly_historical_mean",
        "feature_set": "R6-D",
        "alpha": np.nan,
        "n_features": 0,
        "n_valid": len(historical_mean),
        **mean_metrics,
        "mean_nonzero_coefficients": np.nan,
    }]
    prediction_frames = []
    fit_rows = []
    total_fits = len(ALPHA_GRID) * len(windows)
    completed = 0

    print("=" * 76, flush=True)
    print(
        f"WEEKLY LASSO ALPHA REFINEMENT | R6-D features={len(selected_features)} "
        f"| alphas={len(ALPHA_GRID)} | monthly fits={total_fits}",
        flush=True,
    )
    print(f"Alpha grid: {ALPHA_GRID}", flush=True)
    print("-" * 76, flush=True)

    for alpha_index, alpha in enumerate(ALPHA_GRID, start=1):
        params = {"regressor__alpha": float(alpha)}
        monthly_predictions = []
        print(
            f"[Alpha {alpha_index:02d}/{len(ALPHA_GRID)}] alpha={alpha:g}",
            flush=True,
        )

        for month_index, (month, (training, month_rows)) in enumerate(
            windows.items(), start=1
        ):
            estimator = build_model(
                "lasso",
                lasso_config,
                int(config["experiment"]["random_seed"]),
                params,
            )
            estimator.fit(training[selected_features], training[target])
            predicted = np.asarray(
                estimator.predict(month_rows[selected_features]), dtype=float
            )
            if not np.isfinite(predicted).all():
                raise ValueError(f"Non-finite predictions for alpha={alpha}, month={month}.")

            monthly_predictions.append(pd.DataFrame({
                "decision_date": month_rows["decision_date"].to_numpy(),
                "validation_month": month,
                "actual_return": month_rows[target].to_numpy(dtype=float),
                "predicted_return": predicted,
            }))
            regressor = estimator.named_steps["regressor"]
            fit_rows.append({
                "model": "lasso",
                "feature_set": "R6-D",
                "alpha": float(alpha),
                "validation_month": month,
                "retraining_cutoff": month_rows["decision_date"].min(),
                "training_start": training["decision_date"].min(),
                "training_end": training["decision_date"].max(),
                "n_train": len(training),
                "n_validation": len(month_rows),
                "n_features": len(selected_features),
                "n_nonzero_coefficients": int(
                    np.sum(np.abs(regressor.coef_) > 1e-10)
                ),
            })
            completed += 1
            if month_index == 1 or month_index % 12 == 0 or month_index == len(windows):
                print(
                    f"    month {month_index:02d}/{len(windows)} ({month}) "
                    f"| fits={completed}/{total_fits}",
                    flush=True,
                )

        prediction = pd.concat(monthly_predictions, ignore_index=True)
        metrics = calculate_metrics(
            prediction["actual_return"].to_numpy(dtype=float),
            prediction["predicted_return"].to_numpy(dtype=float),
        )
        aligned = prediction[["decision_date", "actual_return", "predicted_return"]].merge(
            historical_mean[["decision_date", "predicted_return"]].rename(
                columns={"predicted_return": "historical_mean_prediction"}
            ),
            on="decision_date",
            how="left",
            validate="one_to_one",
        )
        if aligned["historical_mean_prediction"].isna().any():
            raise ValueError("Historical Mean predictions do not align with model dates.")
        model_sse = float(np.sum(
            (aligned["actual_return"] - aligned["predicted_return"]) ** 2
        ))
        mean_sse = float(np.sum(
            (aligned["actual_return"] - aligned["historical_mean_prediction"]) ** 2
        ))
        actual_std = float(prediction["actual_return"].std(ddof=1))
        prediction_std = float(prediction["predicted_return"].std(ddof=1))
        fit_subset = [
            row for row in fit_rows
            if row["alpha"] == float(alpha)
        ]
        metrics["prediction_std_ratio"] = (
            prediction_std / actual_std if actual_std > 0 else np.nan
        )
        metrics["oos_r2_vs_monthly_historical_mean"] = (
            1.0 - model_sse / mean_sse if mean_sse > 0 else np.nan
        )
        result_rows.append({
            "model": "lasso",
            "feature_set": "R6-D",
            "alpha": float(alpha),
            "n_features": len(selected_features),
            "n_valid": len(prediction),
            **metrics,
            "mean_nonzero_coefficients": float(
                np.mean([row["n_nonzero_coefficients"] for row in fit_subset])
            ),
        })
        prediction.insert(0, "alpha", float(alpha))
        prediction.insert(0, "feature_set", "R6-D")
        prediction.insert(0, "model", "lasso")
        prediction_frames.append(prediction)

        print(
            f"    complete | RMSE={metrics['pooled_rmse']:.9f} "
            f"| MAE={metrics['pooled_mae']:.9f} "
            f"| R2={metrics['pooled_r2']:.6f} "
            f"| nonzero_mean={result_rows[-1]['mean_nonzero_coefficients']:.2f}",
            flush=True,
        )

    summary = pd.DataFrame(result_rows)
    lasso_summary = summary.loc[summary["model"].eq("lasso")].sort_values(
        ["pooled_rmse", "pooled_mae"], kind="stable"
    )
    best = lasso_summary.iloc[0]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["year"] = pd.to_datetime(predictions["decision_date"]).dt.year

    annual_rows = []
    for (alpha, year), frame in predictions.groupby(["alpha", "year"], sort=True):
        annual_rows.append({
            "model": "lasso",
            "alpha": float(alpha),
            "year": int(year),
            "n_valid": len(frame),
            **calculate_metrics(
                frame["actual_return"].to_numpy(dtype=float),
                frame["predicted_return"].to_numpy(dtype=float),
            ),
        })

    summary.to_csv(output / "weekly_alpha_summary.csv", index=False)
    lasso_summary.to_csv(output / "lasso_alpha_ranking.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(
        output / "weekly_alpha_annual_metrics.csv", index=False
    )
    predictions.to_csv(
        output / "weekly_alpha_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    historical_mean.to_csv(
        output / "monthly_historical_mean_predictions.csv",
        index=False, date_format="%Y-%m-%d",
    )
    pd.DataFrame(fit_rows).to_csv(
        output / "weekly_alpha_fit_log.csv", index=False, date_format="%Y-%m-%d"
    )

    final_path = args.final_config
    if not final_path.is_absolute():
        final_path = Path.cwd() / final_path
    if final_path.exists() and not args.overwrite_final_config:
        raise FileExistsError(
            f"{final_path} already exists. Use --overwrite-final-config only if you intend to replace it."
        )
    best_metrics = {
        key: float(best[key])
        for key in ("pooled_rmse", "pooled_mae", "pooled_r2")
    }
    write_weekly_final_config(config, float(best["alpha"]), best_metrics, final_path)

    setup = {
        "pipeline": "weekly",
        "stage": "R6-D Lasso Alpha Refinement",
        "ablation_dir": str(ablation_dir.resolve()),
        "feature_set": "R6-D",
        "features": selected_features,
        "alpha_grid": ALPHA_GRID,
        "best_alpha_by_pooled_rmse": float(best["alpha"]),
        "best_validation_metrics": best_metrics,
        "validation_period": [
            int(config["validation"]["start_year"]),
            int(config["validation"]["end_year"]),
        ],
        "training": config["training"],
        "selection_metric": "pooled_rmse",
        "final_config_written": str(final_path.resolve()),
        "note": (
            "Validation-only alpha selection. The 2022-2025 independent test "
            "period was not used. Review this config and result before testing."
        ),
    }
    (output / "weekly_alpha_setup.yaml").write_text(
        yaml.safe_dump(setup, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("-" * 76, flush=True)
    print(
        f"Weekly Alpha Refinement complete | fits={completed}/{total_fits}",
        flush=True,
    )
    print(
        f"Best alpha={float(best['alpha']):g} | "
        f"RMSE={float(best['pooled_rmse']):.9f} | "
        f"MAE={float(best['pooled_mae']):.9f} | "
        f"R2={float(best['pooled_r2']):.6f}",
        flush=True,
    )
    print(f"Created final config: {final_path}", flush=True)
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    main()
