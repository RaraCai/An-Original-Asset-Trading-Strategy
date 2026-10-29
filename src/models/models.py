from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(
    model_name: str,
    model_config: dict[str, Any],
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> Any:
    """Construct one configured estimator, applying tunable parameters."""
    params = params or {}

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=model_config["n_estimators"],
            random_state=random_seed,
            n_jobs=model_config.get("n_jobs", -1),
            **params,
        )

    estimators = {
        "ols": LinearRegression,
        "lasso": lambda: Lasso(max_iter=model_config.get("max_iter", 10000)),
        "ridge": Ridge,
    }
    if model_name not in estimators:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", estimators[model_name]()),
        ]
    ).set_params(**params)


def build_models(
    config: dict[str, Any],
    best_params: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Construct all configured candidate estimators without fitting them."""
    models = {}
    for model_name, model_config in config["models"].items():
        models[model_name] = build_model(
            model_name,
            model_config,
            config["experiment"]["random_seed"],
            best_params[model_name],
        )
    return models
