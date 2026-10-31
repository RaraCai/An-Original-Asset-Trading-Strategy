"""Monthly rolling-origin validation with visible, low-noise progress reporting."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from src.config import get_pipeline_features
from src.models.models import build_model


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "pooled_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "pooled_mae": float(mean_absolute_error(actual, predicted)),
        "pooled_r2": float(r2_score(actual, predicted)),
    }


def eligible_training_rows(
    data: pd.DataFrame,
    cutoff: pd.Timestamp,
    training_start_year: int,
    window_years: int,
) -> pd.DataFrame:
    """Use rolling-window observations whose labels matured strictly before cutoff."""
    cutoff = pd.Timestamp(cutoff)
    window_start = cutoff - pd.DateOffset(years=int(window_years))
    mask = (
        (data["decision_date"] >= window_start)
        & (data["decision_date"] < cutoff)
        & (data["holding_end_date"] < cutoff)
        & (data["decision_date"].dt.year >= int(training_start_year))
    )
    return data.loc[mask].copy()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def run_validation(
    data: pd.DataFrame,
    config: dict[str, Any],
    pipeline: str | None = None,
    progress: bool = True,
    progress_every_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every configured model/grid candidate monthly and predict that month.

    Progress is printed before each monthly fit using carriage-return updates,
    so long searches show activity without flooding the terminal with hundreds
    of lines. Candidate completion is printed on its own line.
    """
    selected_pipeline = pipeline or config["default_pipeline"]
    features = get_pipeline_features(config, selected_pipeline)
    target = config["data"]["target"]
    seed = int(config["experiment"]["random_seed"])
    training_cfg = config["training"]

    if training_cfg.get("window") != "rolling" or training_cfg.get("retraining") != "monthly":
        raise ValueError("Validation requires rolling-window training and monthly retraining.")
    if progress_every_months < 1:
        raise ValueError("progress_every_months must be at least 1.")

    start_year = int(config["validation"]["start_year"])
    end_year = int(config["validation"]["end_year"])
    validation = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date")
    if validation.empty:
        raise ValueError("No observations in the configured validation period.")

    monthly_groups = list(
        validation.groupby(validation["decision_date"].dt.to_period("M"), sort=True)
    )
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for model_name, model_config in config["models"].items():
        grid = list(ParameterGrid(model_config.get("param_grid", {}))) or [{}]
        for params in grid:
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            candidates.append((model_name, model_config, {"params": params, "params_json": params_json}))

    total_candidates = len(candidates)
    total_monthly_fits = total_candidates * len(monthly_groups)
    started = time.perf_counter()
    completed_fits = 0

    if progress:
        print("=" * 78, flush=True)
        print(f"{selected_pipeline.upper()} INITIAL SEARCH / VALIDATION STARTED", flush=True)
        print(
            f"Validation: {start_year}-{end_year} | observations={len(validation)} "
            f"| months={len(monthly_groups)}",
            flush=True,
        )
        print(
            f"Features={len(features)} | models={len(config['models'])} "
            f"| candidates={total_candidates} | monthly fits={total_monthly_fits}",
            flush=True,
        )
        print(
            f"Training: rolling {training_cfg['window_years']} years | "
            f"retraining={training_cfg['retraining']} | "
            f"mature-label rule={training_cfg.get('label_availability_rule')}",
            flush=True,
        )
        print("-" * 78, flush=True)

    prediction_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []

    for candidate_index, (model_name, model_config, candidate) in enumerate(candidates, start=1):
        params = candidate["params"]
        params_json = candidate["params_json"]
        candidate_started = time.perf_counter()

        if progress:
            print(
                f"[Candidate {candidate_index:02d}/{total_candidates}] "
                f"{model_name} | params={params} | starting {len(monthly_groups)} monthly fits",
                flush=True,
            )

        for month_index, (month_period, validation_rows) in enumerate(monthly_groups, start=1):
            validation_rows = validation_rows.sort_values("decision_date")
            cutoff = pd.Timestamp(validation_rows["decision_date"].min())
            training_rows = eligible_training_rows(
                data,
                cutoff,
                int(config["data"]["training_start_year"]),
                int(training_cfg["window_years"]),
            )
            if training_rows.empty:
                raise ValueError(
                    f"No eligible training rows for {month_period}, {model_name}."
                )
            if not (training_rows["holding_end_date"] < cutoff).all():
                raise AssertionError(f"Unmatured labels entered training for {month_period}.")

            if progress and (
                month_index == 1
                or month_index % progress_every_months == 0
                or month_index == len(monthly_groups)
            ):
                elapsed = time.perf_counter() - started
                rate = completed_fits / elapsed if elapsed > 0 and completed_fits else 0
                remaining = (total_monthly_fits - completed_fits) / rate if rate else 0
                print(
                    f"  [{candidate_index:02d}/{total_candidates}] "
                    f"{month_index:02d}/{len(monthly_groups)} {month_period} "
                    f"| fitting | train={len(training_rows)} "
                    f"| overall fits={completed_fits}/{total_monthly_fits} "
                    f"| elapsed={_format_duration(elapsed)} "
                    f"| ETA={_format_duration(remaining)}",
                    flush=True,
                )

            estimator = build_model(model_name, model_config, seed, params)
            estimator.fit(training_rows[features], training_rows[target])
            predicted = np.asarray(
                estimator.predict(validation_rows[features]), dtype=float
            )
            if len(predicted) != len(validation_rows) or not np.isfinite(predicted).all():
                raise ValueError(
                    f"Invalid predictions for {model_name}, {params}, {month_period}."
                )

            for row, estimate in zip(validation_rows.itertuples(index=False), predicted):
                prediction_records.append({
                    "model": model_name,
                    "params": params_json,
                    "n_features": len(features),
                    "decision_date": row.decision_date,
                    "execution_date": row.execution_date,
                    "holding_end_date": row.holding_end_date,
                    "actual_return": float(getattr(row, target)),
                    "predicted_return": float(estimate),
                })

            fit_records.append({
                "model": model_name,
                "params": params_json,
                "validation_month": str(month_period),
                "retraining_cutoff": cutoff,
                "training_start": training_rows["decision_date"].min(),
                "training_end": training_rows["decision_date"].max(),
                "n_train": len(training_rows),
                "n_validation": len(validation_rows),
                "n_features": len(features),
            })
            completed_fits += 1

        if progress:
            candidate_elapsed = time.perf_counter() - candidate_started
            print(
                f"  [Candidate {candidate_index:02d}/{total_candidates}] "
                f"completed | {len(monthly_groups)} monthly fits | "
                f"time={_format_duration(candidate_elapsed)} | "
                f"overall={completed_fits}/{total_monthly_fits}",
                flush=True,
            )

    predictions = pd.DataFrame(prediction_records)
    fit_log = pd.DataFrame(fit_records)
    if progress:
        total_elapsed = time.perf_counter() - started
        print("-" * 78, flush=True)
        print(
            f"VALIDATION FITTING COMPLETE | fits={completed_fits}/{total_monthly_fits} "
            f"| elapsed={_format_duration(total_elapsed)}",
            flush=True,
        )
    return predictions, fit_log, validation.reset_index(drop=True)


def aggregate_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pooled metrics for each model and hyperparameter candidate."""
    records = []
    for (model_name, params), frame in predictions.groupby(["model", "params"], sort=False):
        records.append({
            "model": model_name,
            "params": params,
            "n_features": int(frame["n_features"].iloc[0]),
            "n_valid": len(frame),
            **calculate_metrics(
                frame["actual_return"].to_numpy(dtype=float),
                frame["predicted_return"].to_numpy(dtype=float),
            ),
        })
    if not records:
        raise ValueError("No validation predictions were generated.")
    return pd.DataFrame(records).sort_values(
        ["pooled_rmse", "model", "params"], kind="stable"
    ).reset_index(drop=True)
