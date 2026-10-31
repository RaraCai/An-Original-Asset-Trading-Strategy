"""Configuration loading and validation shared by the Daily and Weekly pipelines.

This module deliberately contains no model fitting or data-processing logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate one explicit experiment configuration."""
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    validate_config(config, config_path)
    # Metadata is kept outside YAML keys to avoid polluting model/config sections.
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(PROJECT_ROOT),
    }
    return config


def validate_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Validate required sections and pipeline-specific feature/data declarations."""
    required = {
        "default_pipeline", "data", "pipelines", "validation",
        "test", "training", "models", "experiment",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    pipeline = config["default_pipeline"]
    if pipeline not in config["pipelines"]:
        raise ValueError(
            f"default_pipeline={pipeline!r} is not defined in pipelines."
        )

    for name, settings in config["pipelines"].items():
        if not isinstance(settings, dict):
            raise ValueError(f"pipelines.{name} must be a mapping.")
        data_key = settings.get("data_key")
        if data_key not in config["data"]:
            raise ValueError(f"pipelines.{name}.data_key references unknown key {data_key!r}.")
        features = settings.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"pipelines.{name}.features must be a non-empty list.")
        if len(features) != len(set(features)):
            raise ValueError(f"pipelines.{name}.features contains duplicates.")

    training = config["training"]
    if training.get("window") != "rolling":
        raise ValueError("Only rolling training windows are supported.")
    if training.get("retraining") != "monthly":
        raise ValueError("Only monthly retraining is supported.")
    if int(training.get("window_years", 0)) < 1:
        raise ValueError("training.window_years must be a positive integer.")

    for section in ("validation", "test"):
        if "start_year" not in config[section] or "end_year" not in config[section]:
            raise ValueError(f"{section} must define start_year and end_year.")
        if int(config[section]["start_year"]) > int(config[section]["end_year"]):
            raise ValueError(f"{section} start_year must not exceed end_year.")

    if not config["models"]:
        raise ValueError("At least one model must be configured.")
    for model_name, model_settings in config["models"].items():
        if not isinstance(model_settings, dict):
            raise ValueError(f"models.{model_name} must be a mapping.")
        if "param_grid" not in model_settings:
            raise ValueError(f"models.{model_name}.param_grid is required.")


def get_pipeline_config(config: dict[str, Any], pipeline: str | None = None) -> dict[str, Any]:
    selected = pipeline or config["default_pipeline"]
    if selected not in config["pipelines"]:
        raise ValueError(f"Pipeline {selected!r} is not defined in this config.")
    return config["pipelines"][selected]


def get_pipeline_features(config: dict[str, Any], pipeline: str | None = None) -> list[str]:
    return list(get_pipeline_config(config, pipeline)["features"])


def get_pipeline_data_path(config: dict[str, Any], pipeline: str | None = None) -> Path:
    settings = get_pipeline_config(config, pipeline)
    key = settings["data_key"]
    if key not in config["data"]:
        raise ValueError(f"Unknown data path key: {key}")
    return resolve_path(config["data"][key])


def get_output_root(config: dict[str, Any]) -> Path:
    return resolve_path(config["experiment"].get("output_dir", "outputs"))
