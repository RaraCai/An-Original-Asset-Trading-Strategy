"""Daily Lasso feature ablation (Stage 2C-4), migrated to the shared framework.

Run from the repository root:
  python -m src.experiments.daily.ablation \
    --config configs/daily_research.yaml \
    --run-dir outputs/daily_YYYYMMDD_HHMMSS \
    --analysis-dir outputs/daily_features_YYYYMMDD_HHMMSS

Preserved study design:
- Full21 baseline and F5_17 reference.
- Top-K (K=4..20) by monthly Lasso selection frequency.
- Top-K (K=4..20) by pooled out-of-sample permutation importance.
- Five hypothesis-driven removal groups G1-G5.
- Lasso parameters selected from the supplied Daily Initial Search run.
- Monthly refitting with rolling 5-year training windows and matured labels only.

The script is Daily-only and uses validation data for feature development.
It does not read or fit the independent test period.
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
from sklearn.model_selection import ParameterGrid

from src.config import (
    get_output_root,
    get_pipeline_features,
    load_config,
)
from src.models.dataloader  import load_data
from src.models.models import build_model
from src.models.val import eligible_training_rows, calculate_metrics


TOP_K_VALUES = list(range(4, 21))
F5_REMOVED = [
    "intraday_momentum_5d",
    "qqq_spy_relative_5d",
    "relative_volume_20d",
    "ma_gap_5_20",
]
EXTRA_REMOVALS = {
    "G1_F5_no_rsi": F5_REMOVED + ["rsi_14"],
    "G2_F5_no_momentum20": F5_REMOVED + ["momentum_20d"],
    "G3_F5_no_volatility5": F5_REMOVED + ["volatility_5d"],
    "G4_F5_no_rsi_momentum20": F5_REMOVED + ["rsi_14", "momentum_20d"],
    "G5_F5_no_rsi_momentum20_vol5": (
        F5_REMOVED + ["rsi_14", "momentum_20d", "volatility_5d"]
    ),
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def load_lasso_parameters(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read the best Lasso params from Initial Search and verify against its grid."""
    summary_path = run_dir / "all_models_validation.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing Initial Search summary: {summary_path}")

    summary = pd.read_csv(summary_path)
    required = {"model", "params", "pooled_rmse", "n_features"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Initial Search summary missing columns: {sorted(missing)}")

    lasso_rows = summary.loc[summary["model"].eq("lasso")].sort_values(
        "pooled_rmse", kind="stable"
    )
    if lasso_rows.empty:
        raise ValueError("Initial Search run contains no Lasso results.")
    if sorted(pd.to_numeric(lasso_rows["n_features"]).unique()) != [21]:
        raise ValueError("Initial Search Lasso results must use all 21 Daily features.")

    params = json.loads(lasso_rows.iloc[0]["params"])
    configured_grid = config["models"]["lasso"].get("param_grid", {})
    if params not in list(ParameterGrid(configured_grid)):
        raise ValueError(
            "Best Lasso parameters from Initial Search are not present in "
            "the current daily_research.yaml grid."
        )

    saved_grid_path = run_dir / "parameters_used.yaml"
    if saved_grid_path.is_file():
        saved_grids = yaml.safe_load(saved_grid_path.read_text(encoding="utf-8")) or {}
        if "lasso" not in saved_grids:
            raise ValueError("Initial Search parameters_used.yaml has no Lasso grid.")
        if params not in list(ParameterGrid(saved_grids["lasso"])):
            raise ValueError("Best Lasso params do not match the saved Initial Search grid.")
    return params


def load_feature_rankings(
    analysis_dir: Path, features: list[str]
) -> dict[str, list[str]]:
    """Read the migrated analysis output and produce two deterministic rankings."""
    path = analysis_dir / "lasso_feature_importance.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Lasso feature analysis file: {path}")

    frame = pd.read_csv(path)
    required = {"feature", "selection_frequency"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature analysis file missing columns: {sorted(missing)}")
    if frame["feature"].duplicated().any() or set(frame["feature"]) != set(features):
        raise ValueError("Feature ranking must contain exactly the 21 Daily candidate features.")
    if frame["selection_frequency"].isna().any():
        raise ValueError("Lasso selection_frequency contains missing values.")

    # New analysis uses pooled_permutation_mse_increase. Support the previous
    # column name as a migration fallback, but always record the resolved field.
    if "pooled_permutation_mse_increase" in frame.columns:
        permutation_column = "pooled_permutation_mse_increase"
    elif "permutation_mse_increase" in frame.columns:
        permutation_column = "permutation_mse_increase"
    else:
        raise ValueError(
            "Feature analysis file must contain pooled_permutation_mse_increase "
            "(or legacy permutation_mse_increase)."
        )
    if frame[permutation_column].isna().any():
        raise ValueError(f"{permutation_column} contains missing values.")

    frequency_rank = frame.sort_values(
        ["selection_frequency", "feature"],
        ascending=[False, True],
        kind="stable",
    )["feature"].tolist()
    permutation_rank = frame.sort_values(
        [permutation_column, "feature"],
        ascending=[False, True],
        kind="stable",
    )["feature"].tolist()

    return {
        "frequency": frequency_rank,
        "permutation": permutation_rank,
        "permutation_column": permutation_column,
    }


def build_experiments(
    features: list[str], rankings: dict[str, list[str]]
) -> list[dict[str, Any]]:
    if len(features) != 21 or len(set(features)) != 21:
        raise ValueError("Daily Stage 2C-4 requires exactly 21 unique features.")
    if not set(F5_REMOVED).issubset(features):
        raise ValueError("F5 removal definition contains unknown Daily features.")

    experiments: list[dict[str, Any]] = [
        {
            "experiment": "Full21",
            "category": "baseline",
            "features": features.copy(),
            "rationale": "All 21 initial Daily candidate features.",
        },
        {
            "experiment": "F5_17",
            "category": "reference",
            "features": [feature for feature in features if feature not in set(F5_REMOVED)],
            "rationale": "Reference 17-feature set from the previous Daily study.",
        },
    ]

    for method in ("frequency", "permutation"):
        ranked = rankings[method]
        if set(ranked) != set(features) or len(ranked) != len(features):
            raise ValueError(f"{method} ranking does not match the 21-feature pool.")
        for k in TOP_K_VALUES:
            selected = [feature for feature in features if feature in set(ranked[:k])]
            experiments.append({
                "experiment": f"top{k:02d}_{method}",
                "category": f"topk_{method}",
                "features": selected,
                "rationale": f"Top-{k} features ranked by Lasso {method}.",
            })

    for name, removed in EXTRA_REMOVALS.items():
        if not set(removed).issubset(features):
            raise ValueError(f"Unknown feature in removal group {name}.")
        experiments.append({
            "experiment": name,
            "category": "economic_group",
            "features": [feature for feature in features if feature not in set(removed)],
            "rationale": f"Remove predefined feature group: {removed}.",
        })

    seen: dict[tuple[str, ...], str] = {}
    for item in experiments:
        selected = item["features"]
        if not selected:
            raise ValueError(f"Empty feature set: {item['experiment']}.")
        key = tuple(selected)
        # Different experiment labels may intentionally lead to the same feature
        # set; prediction caching below prevents redundant model fitting.
        seen.setdefault(key, item["experiment"])
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Lasso Stage 2C-4 feature ablation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Daily Initial Search output directory.")
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Daily Feature Analysis output directory.")
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    if config["default_pipeline"] != "daily":
        raise ValueError("This ablation script is Daily-only.")
    if config["training"].get("window") != "rolling" or config["training"].get("retraining") != "monthly":
        raise ValueError("Expected rolling-window training and monthly retraining.")

    features = get_pipeline_features(config, "daily")
    if len(features) != 21:
        raise ValueError(f"Expected 21 Daily candidate features, got {len(features)}.")

    run_dir = resolve_path(args.run_dir)
    analysis_dir = resolve_path(args.analysis_dir)
    if not run_dir.is_dir() or not run_dir.name.startswith("daily_"):
        raise ValueError(f"Invalid Daily Initial Search directory: {run_dir}")
    if not analysis_dir.is_dir() or not analysis_dir.name.startswith("daily_features_"):
        raise ValueError(f"Invalid Daily Feature Analysis directory: {analysis_dir}")

    params = load_lasso_parameters(run_dir, config)
    rankings = load_feature_rankings(analysis_dir, features)
    experiments = build_experiments(features, rankings)

    data = load_data(config, "daily")
    target = config["data"]["target"]
    validation = data.loc[
        data["decision_date"].dt.year.between(
            int(config["validation"]["start_year"]),
            int(config["validation"]["end_year"]),
        )
    ].copy().sort_values("decision_date").reset_index(drop=True)
    if validation.empty:
        raise ValueError("No Daily validation observations.")

    monthly_groups = list(
        validation.groupby(validation["decision_date"].dt.to_period("M"), sort=True)
    )
    # Cache model predictions by feature tuple: the two Top-K rankings can
    # occasionally generate identical feature sets.
    prediction_cache: dict[tuple[str, ...], pd.DataFrame] = {}
    fit_log_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    unique_feature_sets = len({tuple(item["features"]) for item in experiments})

    print("=" * 78, flush=True)
    print("DAILY FEATURE ABLATION - STAGE 2C-4", flush=True)
    print(
        f"Features={len(features)} | experiments={len(experiments)} "
        f"| unique feature sets={unique_feature_sets} | models=1 (Lasso)",
        flush=True,
    )
    print(
        f"Lasso params={params} | validation observations={len(validation)} "
        f"| validation months={len(monthly_groups)}",
        flush=True,
    )
    print("-" * 78, flush=True)

    for experiment_index, item in enumerate(experiments, start=1):
        selected = item["features"]
        key = tuple(selected)

        if key not in prediction_cache:
            print(
                f"[{experiment_index:02d}/{len(experiments)}] "
                f"{item['experiment']:<38} | features={len(selected)} | fitting",
                flush=True,
            )
            parts: list[pd.DataFrame] = []
            fit_records: list[dict[str, Any]] = []

            for month_index, (month_period, month_rows) in enumerate(monthly_groups, start=1):
                month_rows = month_rows.sort_values("decision_date")
                cutoff = pd.Timestamp(month_rows["decision_date"].min())
                training = eligible_training_rows(
                    data,
                    cutoff,
                    int(config["data"]["training_start_year"]),
                    int(config["training"]["window_years"]),
                )
                if training.empty:
                    raise ValueError(f"Empty training data for {month_period}.")
                if not (training["holding_end_date"] < cutoff).all():
                    raise AssertionError(f"Unmatured labels found in {month_period}.")

                estimator = build_model(
                    "lasso",
                    config["models"]["lasso"],
                    int(config["experiment"]["random_seed"]),
                    params,
                )
                estimator.fit(training[selected], training[target])
                predicted = np.asarray(
                    estimator.predict(month_rows[selected]), dtype=float
                )
                if len(predicted) != len(month_rows) or not np.isfinite(predicted).all():
                    raise ValueError(
                        f"Invalid predictions for {item['experiment']} in {month_period}."
                    )

                parts.append(pd.DataFrame({
                    "decision_date": month_rows["decision_date"].to_numpy(),
                    "actual_return": month_rows[target].to_numpy(dtype=float),
                    "predicted_return": predicted,
                    "validation_month": str(month_period),
                }))
                fit_records.append({
                    "experiment": item["experiment"],
                    "validation_month": str(month_period),
                    "retraining_cutoff": cutoff,
                    "training_start": training["decision_date"].min(),
                    "training_end": training["decision_date"].max(),
                    "n_train": len(training),
                    "n_validation": len(month_rows),
                    "n_features": len(selected),
                })

                if month_index == 1 or month_index % 12 == 0 or month_index == len(monthly_groups):
                    print(
                        f"    month {month_index:02d}/{len(monthly_groups)} "
                        f"({month_period}) | train={len(training)}",
                        flush=True,
                    )

            prediction_cache[key] = pd.concat(parts, ignore_index=True)
            fit_log_cache[key] = fit_records
        else:
            print(
                f"[{experiment_index:02d}/{len(experiments)}] "
                f"{item['experiment']:<38} | cached identical feature set",
                flush=True,
            )

        result = prediction_cache[key]
        metrics = calculate_metrics(
            result["actual_return"].to_numpy(dtype=float),
            result["predicted_return"].to_numpy(dtype=float),
        )
        result_rows.append({
            "experiment": item["experiment"],
            "category": item["category"],
            "n_features": len(selected),
            "n_valid": len(result),
            **metrics,
            "params": json.dumps(params, sort_keys=True),
            "features": json.dumps(selected),
            "removed_features": json.dumps(
                [feature for feature in features if feature not in selected]
            ),
            "rationale": item["rationale"],
            "cached_feature_set": key in fit_log_cache and (
                fit_log_cache[key][0]["experiment"] != item["experiment"]
            ),
        })

        prediction_copy = result.copy()
        prediction_copy.insert(0, "experiment", item["experiment"])
        prediction_frames.append(prediction_copy)

        print(
            f"    completed | RMSE={metrics['pooled_rmse']:.9f} "
            f"| MAE={metrics['pooled_mae']:.9f} "
            f"| R2={metrics['pooled_r2']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(result_rows)
    full_rmse = float(
        summary.loc[summary["experiment"].eq("Full21"), "pooled_rmse"].iloc[0]
    )
    f5_rmse = float(
        summary.loc[summary["experiment"].eq("F5_17"), "pooled_rmse"].iloc[0]
    )
    summary["rmse_change_vs_full_pct"] = (summary["pooled_rmse"] / full_rmse - 1) * 100
    summary["rmse_change_vs_f5_pct"] = (summary["pooled_rmse"] / f5_rmse - 1) * 100
    summary = summary.sort_values(
        ["pooled_rmse", "experiment"], kind="stable"
    ).reset_index(drop=True)

    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / (
        "daily_ablation_topk_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    summary.to_csv(output / "all_ablation_results.csv", index=False)
    summary.loc[summary["category"].str.startswith("topk")].to_csv(
        output / "topk_results.csv", index=False
    )
    summary.loc[
        summary["category"].isin(["baseline", "reference", "economic_group"])
    ].to_csv(output / "economic_group_results.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / "ablation_predictions.csv", index=False, date_format="%Y-%m-%d"
    )

    # One fit log per unique feature set; each log row documents the first
    # experiment name that triggered the fit. Cached aliases remain in summary.
    unique_fit_rows = []
    for records in fit_log_cache.values():
        unique_fit_rows.extend(records)
    pd.DataFrame(unique_fit_rows).to_csv(
        output / "ablation_fit_log.csv", index=False, date_format="%Y-%m-%d"
    )

    setup = {
        "stage": "Daily Stage 2C-4",
        "pipeline": "daily",
        "model": "lasso",
        "params": params,
        "initial_features": features,
        "top_k_values": TOP_K_VALUES,
        "rankings": {
            "frequency": rankings["frequency"],
            "permutation": rankings["permutation"],
        },
        "permutation_ranking_column": rankings["permutation_column"],
        "f5_removed_features": F5_REMOVED,
        "economic_group_removals": EXTRA_REMOVALS,
        "experiments": [
            {
                "experiment": item["experiment"],
                "category": item["category"],
                "features": item["features"],
                "rationale": item["rationale"],
            }
            for item in experiments
        ],
        "initial_search_run": str(run_dir),
        "feature_analysis_run": str(analysis_dir),
        "validation_period": [
            int(config["validation"]["start_year"]),
            int(config["validation"]["end_year"]),
        ],
        "training_window_years": int(config["training"]["window_years"]),
        "retraining": config["training"]["retraining"],
        "label_availability_rule": config["training"]["label_availability_rule"],
        "note": (
            "Validation-only feature development. The independent test period "
            "must not be used for feature selection."
        ),
    }
    (output / "ablation_setup.yaml").write_text(
        yaml.safe_dump(setup, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    print("-" * 78, flush=True)
    print(
        f"Daily ablation complete | experiments={len(experiments)} "
        f"| unique feature sets fitted={len(prediction_cache)} "
        f"| monthly fits={sum(len(rows) for rows in fit_log_cache.values())} "
        f"| elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    main()
