"""Monthly rolling-origin independent test for a frozen final model configuration."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT, get_pipeline_features, get_output_root, load_config
from src.models.data import load_data
from src.models.eval import evaluate_predictions
from src.models.models import build_model
from src.models.val import eligible_training_rows


def resolve_final_model(config: dict[str, Any], model_arg: str | None, params_arg: str | None):
    record = config.get("final_model_record", {})
    model_name = model_arg or record.get("name")
    if not model_name:
        raise ValueError(
            "No final model is declared. Supply --model and --params, or add final_model_record."
        )
    if model_name not in config["models"]:
        raise ValueError(f"Final model {model_name!r} is not configured.")

    if params_arg:
        params = json.loads(params_arg)
    else:
        params = record.get("params")
    if not isinstance(params, dict):
        raise ValueError("Final model parameters must be a JSON/YAML mapping.")
    return model_name, params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen final model with monthly retraining on the independent test period."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pipeline", choices=["daily", "weekly"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--params", default=None, help="Optional JSON object overriding final_model_record.params.")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = args.pipeline or config["default_pipeline"]
    if pipeline not in config["pipelines"]:
        raise ValueError(f"Pipeline {pipeline!r} is not configured.")
    model_name, params = resolve_final_model(config, args.model, args.params)
    data = load_data(config, pipeline)
    features = get_pipeline_features(config, pipeline)
    target = config["data"]["target"]

    start_year = int(config["test"]["start_year"])
    end_year = int(config["test"]["end_year"])
    test_rows = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date")
    if test_rows.empty:
        raise ValueError("No observations in the configured independent test period.")

    monthly_groups = list(
        test_rows.groupby(test_rows["decision_date"].dt.to_period("M"), sort=True)
    )
    prediction_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    historical_mean_rows: list[dict[str, Any]] = []

    for month_period, month_rows in monthly_groups:
        month = str(month_period)
        month_rows = month_rows.sort_values("decision_date")
        cutoff = pd.Timestamp(month_rows["decision_date"].min())
        training = eligible_training_rows(
            data,
            cutoff,
            int(config["data"]["training_start_year"]),
            int(config["training"]["window_years"]),
        )
        if training.empty:
            raise ValueError(f"No eligible training rows for test month {month}.")
        if not (training["holding_end_date"] < cutoff).all():
            raise AssertionError(f"Unmatured labels entered test training for {month}.")

        estimator = build_model(
            model_name,
            config["models"][model_name],
            int(config["experiment"]["random_seed"]),
            params,
        )
        estimator.fit(training[features], training[target])
        predictions = np.asarray(estimator.predict(month_rows[features]), dtype=float)
        if len(predictions) != len(month_rows) or not np.isfinite(predictions).all():
            raise ValueError(f"Invalid test predictions for {month}.")

        mean_prediction = float(training[target].mean())
        for row, estimate in zip(month_rows.itertuples(index=False), predictions):
            prediction_rows.append({
                "model": model_name,
                "decision_date": row.decision_date,
                "execution_date": row.execution_date,
                "holding_end_date": row.holding_end_date,
                "actual_return": float(getattr(row, target)),
                "predicted_return": float(estimate),
                "historical_mean_prediction": mean_prediction,
                "test_month": month,
            })
            historical_mean_rows.append({
                "decision_date": row.decision_date,
                "actual_return": float(getattr(row, target)),
                "predicted_return": mean_prediction,
                "test_month": month,
            })

        fit_rows.append({
            "model": model_name,
            "test_month": month,
            "retraining_cutoff": cutoff,
            "training_start": training["decision_date"].min(),
            "training_end": training["decision_date"].max(),
            "n_train": len(training),
            "n_test": len(month_rows),
            "n_features": len(features),
            "holding_end_max_in_training": training["holding_end_date"].max(),
        })

    predictions_frame = pd.DataFrame(prediction_rows)
    historical_frame = pd.DataFrame(historical_mean_rows)

    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / (
        f"{pipeline}_independent_test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir.mkdir(exist_ok=False)

    metrics = evaluate_predictions(predictions_frame, output_dir, "test")
    historical_metrics = evaluate_predictions(
        historical_frame, output_dir, "historical_mean_test"
    )
    predictions_frame["year"] = pd.to_datetime(
        predictions_frame["decision_date"]
    ).dt.year
    model_sse = float(np.sum(
        (predictions_frame["actual_return"] - predictions_frame["predicted_return"]) ** 2
    ))
    mean_sse = float(np.sum(
        (predictions_frame["actual_return"] - predictions_frame["historical_mean_prediction"]) ** 2
    ))
    oos_r2 = 1.0 - model_sse / mean_sse if mean_sse > 0 else np.nan

    summary = metrics.copy()
    summary["oos_r2_vs_monthly_historical_mean"] = oos_r2
    summary.to_csv(output_dir / "test_summary.csv", index=False)
    historical_metrics.to_csv(output_dir / "historical_mean_summary.csv", index=False)
    predictions_frame.to_csv(
        output_dir / "test_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    pd.DataFrame(fit_rows).to_csv(
        output_dir / "test_monthly_fit_log.csv", index=False, date_format="%Y-%m-%d"
    )

    config_path = Path(config["_meta"]["config_path"])
    (output_dir / "test_setup.yaml").write_text(
        yaml.safe_dump({
            "pipeline": pipeline,
            "config_path": str(config_path),
            "model": model_name,
            "params": params,
            "features": features,
            "test_period": [start_year, end_year],
            "training": config["training"],
            "retraining_policy": "monthly rolling-window; only labels matured before monthly cutoff",
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(
        f"{pipeline.title()} independent test complete | model={model_name} "
        f"| features={len(features)} | observations={len(predictions_frame)} "
        f"| monthly_fits={len(fit_rows)}"
    )
    print(summary.to_string(index=False))
    print(f"Results saved: {output_dir}")


if __name__ == "__main__":
    main()
