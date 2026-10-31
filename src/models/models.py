"""Central model factory shared by validation, experiments and testing."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def build_model(
    model_name: str,
    model_config: dict[str, Any],
    random_seed: int,
    params: dict[str, Any] | None = None,
) -> Any:
    """Construct one estimator; scale linear models inside the fitted pipeline."""
    params = dict(params or {})

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(model_config["n_estimators"]),
            random_state=random_seed,
            n_jobs=int(model_config.get("n_jobs", -1)),
            **params,
        )

    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=int(model_config["n_estimators"]),
            objective=model_config.get("objective", "reg:squarederror"),
            learning_rate=float(model_config.get("learning_rate", 0.03)),
            subsample=float(model_config.get("subsample", 0.8)),
            colsample_bytree=float(model_config.get("colsample_bytree", 0.8)),
            n_jobs=int(model_config.get("n_jobs", 4)),
            random_state=random_seed,
            tree_method=model_config.get("tree_method", "hist"),
            **params,
        )

    constructors = {
        "ols": LinearRegression,
        "ridge": Ridge,
        "lasso": lambda: Lasso(max_iter=int(model_config.get("max_iter", 10000))),
    }
    if model_name not in constructors:
        raise ValueError(f"Unknown model: {model_name}")

    estimator = constructors[model_name]()
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", estimator),
    ]).set_params(**params)


def build_models(config: dict, best_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build all configured estimators without fitting them."""
    return {
        name: build_model(
            name,
            model_config,
            int(config["experiment"]["random_seed"]),
            best_params[name],
        )
        for name, model_config in config["models"].items()
    }
