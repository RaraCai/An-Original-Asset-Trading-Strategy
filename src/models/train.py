from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from eval import evaluate_predictions, summarize_validation_metrics
from models import build_models
from val import eligible_training_rows, run_validation

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return config


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    data_path = ROOT / config["data"]["path"]
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed labeled data not found: {data_path}. "
            "Run the data-preparation scripts first."
        )

    date_columns = ["decision_date", "execution_date", "holding_end_date"]
    data = pd.read_csv(data_path, parse_dates=date_columns)
    features = config["features"]
    target = config["data"]["target"]
    required = {*date_columns, *features, target}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.dropna(subset=list(required)).copy()
    data = data.sort_values("decision_date").reset_index(drop=True)
    if data["decision_date"].duplicated().any():
        raise ValueError("decision_date must be unique.")
    numeric = data[features + [target]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Features and target must contain only finite values.")
    return data


def predict_test(
    models: dict[str, Any],
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    test_config = config["test"]
    test_rows = data.loc[
        data["decision_date"].dt.year.between(
            test_config["start_year"], test_config["end_year"]
        )
    ].copy()
    if test_rows.empty:
        raise ValueError("No rows found in the configured held-out test period.")

    records: list[dict[str, Any]] = []
    for model_name, model in models.items():
        predicted = model.predict(test_rows[config["features"]])
        for row, prediction in zip(test_rows.itertuples(index=False), predicted):
            records.append(
                {
                    "model": model_name,
                    "decision_date": row.decision_date,
                    "execution_date": row.execution_date,
                    "holding_end_date": row.holding_end_date,
                    "actual_return": getattr(row, config["data"]["target"]),
                    "predicted_return": prediction,
                }
            )
    return pd.DataFrame(records).sort_values(["model", "decision_date"])


def train_final_models(
    data: pd.DataFrame,
    best_params: dict[str, dict[str, Any]],
    config: dict[str, Any],
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    training_rows = eligible_training_rows(
        data,
        cutoff,
        config["data"]["training_start_year"],
    )
    if training_rows.empty:
        raise ValueError("No eligible rows are available for final model training.")

    models = build_models(config, best_params)
    for model in models.values():
        model.fit(training_rows[config["features"]], training_rows[config["data"]["target"]])
    return models


def save_experiment(
    models: dict[str, Any],
    best_params: dict[str, dict[str, Any]],
    search_results: pd.DataFrame,
    validation_summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    output_dir = ROOT / config["experiment"]["output_dir"]
    model_dir = output_dir / config["experiment"]["model_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    search_results.to_csv(output_dir / "validation_search_results.csv", index=False)
    validation_summary.to_csv(output_dir / "validation_summary.csv", index=False)
    with (output_dir / "selected_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    for model_name, model in models.items():
        joblib.dump(model, model_dir / f"{model_name}.joblib")


def main() -> None:
    config = load_config()
    data = load_data(config)
    best_params, validation_predictions, search_results = run_validation(data, config)
    validation_metrics = evaluate_predictions(
        validation_predictions,
        ROOT / config["experiment"]["output_dir"],
        "validation",
        group_columns=("model", "validation_year"),
    )
    validation_summary = summarize_validation_metrics(validation_metrics)
    validation_summary["selected_params"] = validation_summary["model"].map(
        lambda model_name: json.dumps(best_params[model_name], sort_keys=True)
    )

    test_cutoff_year = config["test"]["start_year"]
    cutoff = data.loc[
        data["decision_date"].dt.year == test_cutoff_year, "decision_date"
    ].min()
    if pd.isna(cutoff):
        raise ValueError(f"No test data found for {test_cutoff_year}.")
    models = train_final_models(data, best_params, config, cutoff)
    test_predictions = predict_test(models, data, config)
    test_metrics = evaluate_predictions(
        test_predictions,
        ROOT / config["experiment"]["output_dir"],
        "test",
    )
    save_experiment(
        models,
        best_params,
        search_results,
        validation_summary,
        config,
    )

    print("Selected validation parameters:")
    print(validation_summary.to_string(index=False))
    print("\nHeld-out test metrics:")
    print(test_metrics.to_string(index=False))
    print(f"\nOutputs saved to: {ROOT / config['experiment']['output_dir']}")


if __name__ == "__main__":
    main()
