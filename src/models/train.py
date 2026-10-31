from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eval import plot_best_model_predictions
from val import run_validation, aggregate_results

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def get_pipeline_features(config: dict[str, Any], pipeline: str) -> list[str]:
    """Return pipeline-specific features, falling back to the shared feature list."""
    return list(config["pipelines"][pipeline].get("features", config["features"]))


def load_data(config: dict[str, Any], pipeline: str) -> pd.DataFrame:
    if pipeline not in config["pipelines"]:
        raise ValueError(f"Unknown pipeline {pipeline!r}; choose from {list(config['pipelines'])}.")
    data_key = config["pipelines"][pipeline]["data_key"]
    data_path = ROOT / config["data"][data_key]
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed labeled data not found: {data_path}")

    date_columns = ["decision_date", "execution_date", "holding_end_date"]
    data = pd.read_csv(data_path, parse_dates=date_columns)
    features = get_pipeline_features(config, pipeline)
    required = set(date_columns + features + [config["data"]["target"]])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    data = data.dropna(subset=list(required)).sort_values("decision_date").reset_index(drop=True)
    if data["decision_date"].duplicated().any():
        raise ValueError("decision_date must be unique within a pipeline.")
    values = data[features + [config["data"]["target"]]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features and target must be finite.")
    return data


def candidate_table(frame: pd.DataFrame, param_names: list[str]) -> pd.DataFrame:
    expanded = pd.DataFrame([json.loads(value) for value in frame["params"]], index=frame.index)
    for name in param_names:
        if name not in expanded:
            expanded[name] = None
    metric_columns = ["n_features", "n_valid", "pooled_rmse", "pooled_mae", "pooled_r2"]
    return pd.concat([
        expanded[param_names].reset_index(drop=True),
        frame[metric_columns].reset_index(drop=True),
    ], axis=1)


def create_run_dir(config: dict[str, Any], pipeline: str) -> Path:
    base = ROOT / config["experiment"]["output_dir"]
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base / f"{pipeline}_{stamp}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured monthly rolling validation pipeline.")
    parser.add_argument("--pipeline", choices=["daily", "weekly"], default="daily")
    args = parser.parse_args()

    config = load_config()
    data = load_data(config, args.pipeline)
    predictions, fit_log, _ = run_validation(data, config, pipeline=args.pipeline)
    summary = aggregate_results(predictions)
    run_dir = create_run_dir(config, args.pipeline)

    summary.to_csv(run_dir / "all_models_validation.csv", index=False)
    for model_name, model_config in config["models"].items():
        subset = summary.loc[summary["model"] == model_name].copy()
        param_names = list(model_config.get("param_grid", {}))
        candidate_table(subset, param_names).to_csv(
            run_dir / f"{model_name}_validation.csv", index=False
        )

    predictions.to_csv(run_dir / "validation_predictions.csv", index=False, date_format="%Y-%m-%d")
    fit_log.to_csv(run_dir / "monthly_fit_log.csv", index=False, date_format="%Y-%m-%d")
    plot_best_model_predictions(
        predictions,
        summary,
        run_dir / "validation_predictions.png",
        f"{args.pipeline.title()} Pipeline: Best Configuration per Model",
    )

    with (run_dir / "parameters_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {name: model_config.get("param_grid", {}) for name, model_config in config["models"].items()},
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Pipeline: {args.pipeline}")
    print(f"Validation observations per candidate: {summary['n_valid'].nunique()} unique count(s)")
    print(f"Candidate configurations evaluated: {len(summary)}")
    print(summary.to_string(index=False))
    print(f"Files saved to: {run_dir}")


if __name__ == "__main__":
    main()
