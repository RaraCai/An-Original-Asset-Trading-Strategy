"""Daily Lasso Alpha Refinement using the G5 14-feature set.

Run from repository root:
  python -m src.experiments.daily.refinement \
    --config configs/daily_research.yaml \
    --ablation-dir outputs/daily_ablation_topk_YYYYMMDD_HHMMSS

The script reads the exact G5 feature list from all_ablation_results.csv.
It searches a pre-specified alpha grid, refitting monthly with a rolling
5-year window and matured labels only. Validation period is read from config.
The independent test period is never loaded for model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import get_output_root, load_config
from src.models.dataloader  import load_data
from src.models.models import build_model
from src.models.val import calculate_metrics, eligible_training_rows


ALPHA_GRID = [
    0.00015,
    0.000175,
    0.00020,
    0.0002125,
    0.000225,
    0.0002375,
    0.00025,
    0.000275,
    0.00030,
]
EXPECTED_FEATURE_COUNT = 14
G5_NAME = "G5_F5_no_rsi_momentum20_vol5"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def load_g5_features(ablation_dir: Path) -> list[str]:
    results_path = ablation_dir / "all_ablation_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing Daily Ablation results: {results_path}")

    results = pd.read_csv(results_path)
    required = {"experiment", "features", "n_features", "pooled_rmse"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Ablation results missing columns: {sorted(missing)}")

    row = results.loc[results["experiment"].eq(G5_NAME)]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one {G5_NAME} row in {results_path}.")
    features = json.loads(row.iloc[0]["features"])
    n_features = int(row.iloc[0]["n_features"])
    if n_features != EXPECTED_FEATURE_COUNT or len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"G5 should contain {EXPECTED_FEATURE_COUNT} features; "
            f"found n_features={n_features}, feature_list={len(features)}."
        )
    if len(features) != len(set(features)):
        raise ValueError("G5 feature list contains duplicates.")
    return features


def prepare_monthly_windows(
    data: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    target = config["data"]["target"]
    start_year = int(config["validation"]["start_year"])
    end_year = int(config["validation"]["end_year"])
    validation = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date").reset_index(drop=True)

    if validation.empty:
        raise ValueError("No Daily validation observations in configured years.")

    month_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    mean_records = []
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
            raise ValueError(f"No eligible training rows for {month}.")
        if not (training["holding_end_date"] < cutoff).all():
            raise AssertionError(f"Unmatured labels found in training window {month}.")
        month_cache[month] = (training, month_rows)
        historical_mean = float(training[target].mean())
        for row in month_rows.itertuples(index=False):
            mean_records.append({
                "decision_date": row.decision_date,
                "validation_month": month,
                "actual_return": float(getattr(row, target)),
                "predicted_return": historical_mean,
            })
    return month_cache, pd.DataFrame(mean_records)


def calculate_oos_r2(
    predictions: pd.DataFrame, historical_mean: pd.DataFrame
) -> float:
    aligned = predictions[
        ["decision_date", "actual_return", "predicted_return"]
    ].merge(
        historical_mean[["decision_date", "predicted_return"]].rename(
            columns={"predicted_return": "historical_mean_prediction"}
        ),
        on="decision_date",
        how="left",
        validate="one_to_one",
    )
    if aligned["historical_mean_prediction"].isna().any():
        raise ValueError("Historical Mean dates do not align with model predictions.")
    model_sse = float(np.sum(
        (aligned["actual_return"] - aligned["predicted_return"]) ** 2
    ))
    mean_sse = float(np.sum(
        (aligned["actual_return"] - aligned["historical_mean_prediction"]) ** 2
    ))
    return float(1.0 - model_sse / mean_sse) if mean_sse > 0 else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily G5 Lasso Alpha Refinement.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    if config["default_pipeline"] != "daily":
        raise ValueError("Daily Alpha Refinement requires a Daily research config.")
    if config["training"].get("window") != "rolling":
        raise ValueError("Expected rolling training window.")
    if config["training"].get("retraining") != "monthly":
        raise ValueError("Expected monthly retraining.")

    ablation_dir = resolve_path(args.ablation_dir)
    if not ablation_dir.is_dir() or not ablation_dir.name.startswith("daily_ablation_topk_"):
        raise ValueError(f"Invalid Daily Ablation output directory: {ablation_dir}")

    features = load_g5_features(ablation_dir)
    available_features = set(config["pipelines"]["daily"]["features"])
    if not set(features).issubset(available_features):
        raise ValueError("G5 contains features outside daily_research.yaml.")
    data = load_data(config, "daily")
    missing_data_features = set(features) - set(data.columns)
    if missing_data_features:
        raise ValueError(f"Daily dataset is missing G5 features: {sorted(missing_data_features)}")

    month_cache, historical_mean = prepare_monthly_windows(data, config)
    target = config["data"]["target"]
    lasso_config = config["models"].get("lasso")
    if lasso_config is None:
        raise ValueError("Daily research config does not define Lasso.")

    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (
        "daily_alpha_refinement_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    mean_metrics = calculate_metrics(
        historical_mean["actual_return"].to_numpy(dtype=float),
        historical_mean["predicted_return"].to_numpy(dtype=float),
    )
    result_rows = [{
        "model": "historical_mean",
        "feature_set": G5_NAME,
        "alpha": np.nan,
        "n_features": 0,
        "n_valid": len(historical_mean),
        **mean_metrics,
        "oos_r2_vs_monthly_historical_mean": 0.0,
    }]
    prediction_frames = []
    fit_log_rows = []
    total_fits = len(ALPHA_GRID) * len(month_cache)
    completed_fits = 0

    print("=" * 78, flush=True)
    print("DAILY LASSO ALPHA REFINEMENT STARTED", flush=True)
    print(
        f"Feature set={G5_NAME} | features={len(features)} | "
        f"validation observations={sum(len(v[1]) for v in month_cache.values())} "
        f"| months={len(month_cache)}",
        flush=True,
    )
    print(f"Alpha candidates={len(ALPHA_GRID)} | monthly fits={total_fits}", flush=True)
    print(f"Alpha grid={ALPHA_GRID}", flush=True)
    print("-" * 78, flush=True)

    for candidate_index, alpha in enumerate(ALPHA_GRID, start=1):
        candidate_started = time.perf_counter()
        monthly_predictions = []
        params = {"regressor__alpha": float(alpha)}
        print(
            f"[Alpha {candidate_index:02d}/{len(ALPHA_GRID)}] "
            f"alpha={alpha:g} | starting {len(month_cache)} monthly fits",
            flush=True,
        )

        for month_index, (month, (training, month_rows)) in enumerate(
            month_cache.items(), start=1
        ):
            estimator = build_model(
                "lasso",
                lasso_config,
                int(config["experiment"]["random_seed"]),
                params,
            )
            estimator.fit(training[features], training[target])
            predicted = np.asarray(
                estimator.predict(month_rows[features]), dtype=float
            )
            if len(predicted) != len(month_rows) or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid predictions for alpha={alpha}, month={month}.")

            monthly_predictions.append(pd.DataFrame({
                "decision_date": month_rows["decision_date"].to_numpy(),
                "validation_month": month,
                "actual_return": month_rows[target].to_numpy(dtype=float),
                "predicted_return": predicted,
            }))
            fit_log_rows.append({
                "model": "lasso",
                "feature_set": G5_NAME,
                "alpha": float(alpha),
                "validation_month": month,
                "retraining_cutoff": month_rows["decision_date"].min(),
                "training_start": training["decision_date"].min(),
                "training_end": training["decision_date"].max(),
                "n_train": len(training),
                "n_validation": len(month_rows),
                "n_features": len(features),
                "n_nonzero_coefficients": int(
                    np.sum(np.abs(estimator.named_steps["regressor"].coef_) > 1e-10)
                ),
            })
            completed_fits += 1

            if month_index == 1 or month_index % 12 == 0 or month_index == len(month_cache):
                elapsed = time.perf_counter() - started
                rate = completed_fits / elapsed if elapsed > 0 else 0.0
                remaining = (total_fits - completed_fits) / rate if rate > 0 else 0.0
                print(
                    f"    month {month_index:02d}/{len(month_cache)} ({month}) "
                    f"| train={len(training)} | fits={completed_fits}/{total_fits} "
                    f"| ETA={remaining:.0f}s",
                    flush=True,
                )

        prediction = pd.concat(monthly_predictions, ignore_index=True)
        metrics = calculate_metrics(
            prediction["actual_return"].to_numpy(dtype=float),
            prediction["predicted_return"].to_numpy(dtype=float),
        )
        oos_r2 = calculate_oos_r2(prediction, historical_mean)
        result_rows.append({
            "model": "lasso",
            "feature_set": G5_NAME,
            "alpha": float(alpha),
            "n_features": len(features),
            "n_valid": len(prediction),
            **metrics,
            "oos_r2_vs_monthly_historical_mean": oos_r2,
        })
        prediction.insert(0, "alpha", float(alpha))
        prediction.insert(0, "feature_set", G5_NAME)
        prediction.insert(0, "model", "lasso")
        prediction_frames.append(prediction)

        candidate_elapsed = time.perf_counter() - candidate_started
        print(
            f"[Alpha {candidate_index:02d}/{len(ALPHA_GRID)}] complete "
            f"| RMSE={metrics['pooled_rmse']:.9f} "
            f"| MAE={metrics['pooled_mae']:.9f} "
            f"| R2={metrics['pooled_r2']:.6f} "
            f"| OOS_R2={oos_r2:.6f} "
            f"| time={candidate_elapsed:.1f}s",
            flush=True,
        )

    summary = pd.DataFrame(result_rows)
    model_summary = summary.loc[summary["model"].eq("lasso")].sort_values(
        ["pooled_rmse", "alpha"], kind="stable"
    )
    best = model_summary.iloc[0]

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
    annual = pd.DataFrame(annual_rows)

    summary.to_csv(output / "alpha_refinement_summary.csv", index=False)
    model_summary.to_csv(output / "lasso_alpha_ranking.csv", index=False)
    annual.to_csv(output / "alpha_refinement_annual_metrics.csv", index=False)
    predictions.to_csv(
        output / "alpha_refinement_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    historical_mean.to_csv(
        output / "monthly_historical_mean_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    pd.DataFrame(fit_log_rows).to_csv(
        output / "alpha_refinement_fit_log.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    setup = {
        "stage": "Daily Lasso Alpha Refinement",
        "pipeline": "daily",
        "feature_set_source": str(ablation_dir),
        "feature_set_name": G5_NAME,
        "features": features,
        "n_features": len(features),
        "model": "lasso",
        "alpha_grid": ALPHA_GRID,
        "validation_period": [
            int(config["validation"]["start_year"]),
            int(config["validation"]["end_year"]),
        ],
        "training_window_years": int(config["training"]["window_years"]),
        "retraining": config["training"]["retraining"],
        "label_availability_rule": config["training"]["label_availability_rule"],
        "selection_metric": "pooled_rmse",
        "diagnostic_metrics": [
            "pooled_mae",
            "pooled_r2",
            "oos_r2_vs_monthly_historical_mean",
            "prediction_std_ratio",
            "annual_metrics",
            "monthly_nonzero_coefficients",
        ],
        "best_validation_candidate": {
            "alpha": float(best["alpha"]),
            "pooled_rmse": float(best["pooled_rmse"]),
            "pooled_mae": float(best["pooled_mae"]),
            "pooled_r2": float(best["pooled_r2"]),
            "oos_r2_vs_monthly_historical_mean": float(
                best["oos_r2_vs_monthly_historical_mean"]
            ),
        },
        "note": (
            "Validation-only refinement. The 2022-2025 independent test period "
            "is not used for alpha selection."
        ),
    }
    (output / "alpha_refinement_setup.yaml").write_text(
        yaml.safe_dump(setup, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    print("-" * 78, flush=True)
    print(
        f"Daily Alpha Refinement complete | candidates={len(ALPHA_GRID)} "
        f"| fits={completed_fits}/{total_fits} | elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(
        f"Best alpha={float(best['alpha']):g} | "
        f"RMSE={float(best['pooled_rmse']):.9f} | "
        f"MAE={float(best['pooled_mae']):.9f} | "
        f"R2={float(best['pooled_r2']):.6f}",
        flush=True,
    )
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    main()
