"""Weekly Round 6 ablation: exact R6-A to R6-H experiment definitions.

Run:
  python -m src.experiments.weekly.ablation \
    --config configs/weekly_research.yaml \
    --run-dir outputs/weekly_<full35_initial_search>

This preserves the previous ablation_weekly.py design:
- Full-35 W5 candidate pool.
- R6-A through R6-H only.
- Ridge and Lasso only.
- Fixed best Ridge/Lasso parameters from the Full-35 Initial Search.
- Monthly rolling 5-year training and matured labels only.
- Same 2018-2021 validation rows for every experiment.
- Monthly Historical Mean benchmark.
No new feature groups, Top-K experiments, models, or automatic feature selection.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_output_root, get_pipeline_features, load_config
from src.models.dataloader  import load_data
from src.models.models import build_model
from src.models.val import eligible_training_rows


BASELINE_25 = [
    "weekly_mean_return_5d", "weekly_positive_days_5d",
    "weekly_max_return_5d", "weekly_min_return_5d",
    "weekly_return_std_5d", "weekly_downside_std_5d",
    "weekly_max_drawdown_5d", "weekly_mean_intraday_return_5d",
    "weekly_mean_overnight_return_5d", "weekly_compounded_intraday_5d",
    "weekly_compounded_overnight_5d", "weekly_intraday_std_5d",
    "weekly_mean_relative_volume_5d", "weekly_volume_trend_5d",
    "weekly_qqq_relative_return_5d", "weekly_tlt_return_5d",
    "weekly_gap_reversal_mean_5d", "weekly_gap_reversal_std_5d",
    "weekly_gap_reversal_positive_days_5d", "weekly_gap_reversal_abs_mean_5d",
    "weekly_yield_2y_change_5d", "weekly_yield_10y_change_5d",
    "weekly_real_yield_change_5d", "weekly_yield_curve_change_5d",
    "weekly_yield_curve_mean_5d",
]

FULL_35 = BASELINE_25 + [
    "weekly_log_price_slope_5d", "weekly_trend_fit_r2_5d",
    "weekly_upside_std_5d", "weekly_range_mean_5d",
    "weekly_volatility_change_5d", "weekly_overnight_std_5d",
    "weekly_vix_change_5d", "weekly_vix_std_5d",
    "weekly_qqq_return_5d", "weekly_tlt_relative_return_5d",
]

MODELS = ["ridge", "lasso"]

DROP_INTRADAY_OVERNIGHT = [
    "weekly_mean_intraday_return_5d", "weekly_mean_overnight_return_5d",
    "weekly_compounded_intraday_5d", "weekly_compounded_overnight_5d",
    "weekly_intraday_std_5d", "weekly_overnight_std_5d",
]
RISK_FEATURES_TO_REMOVE_FOR_F3 = [
    "weekly_return_std_5d", "weekly_downside_std_5d",
    "weekly_max_drawdown_5d", "weekly_upside_std_5d",
    "weekly_range_mean_5d", "weekly_volatility_change_5d",
]
CROSS_ASSET_FEATURES = [
    "weekly_qqq_relative_return_5d", "weekly_tlt_return_5d",
    "weekly_qqq_return_5d", "weekly_tlt_relative_return_5d",
]
GAP_MEAN = "weekly_gap_reversal_mean_5d"
GAP_STD = "weekly_gap_reversal_std_5d"
GAP_POS = "weekly_gap_reversal_positive_days_5d"
GAP_ABS = "weekly_gap_reversal_abs_mean_5d"
YIELD_CURVE_CHANGE = "weekly_yield_curve_change_5d"
MACRO_FEATURES = [
    "weekly_yield_2y_change_5d", "weekly_yield_10y_change_5d",
    "weekly_real_yield_change_5d", "weekly_yield_curve_change_5d",
    "weekly_yield_curve_mean_5d",
]

G3_BASE = [feature for feature in FULL_35 if feature not in DROP_INTRADAY_OVERNIGHT]
F3 = [feature for feature in G3_BASE if feature not in RISK_FEATURES_TO_REMOVE_FOR_F3]
S4_BASE = [feature for feature in F3 if feature not in CROSS_ASSET_FEATURES]


def build_experiments() -> dict[str, list[str]]:
    r6_a = [feature for feature in S4_BASE
            if feature not in MACRO_FEATURES or feature == YIELD_CURVE_CHANGE]
    r6_b = [
        feature for feature in S4_BASE
        if feature not in [GAP_STD, GAP_POS]
        and (feature not in MACRO_FEATURES or feature == YIELD_CURVE_CHANGE)
    ]
    r6_c = r6_b + [GAP_POS]
    r6_d = r6_b + [GAP_STD]
    r6_e = [feature for feature in r6_b if feature != YIELD_CURVE_CHANGE]
    r6_f = [feature for feature in r6_b if feature != GAP_ABS]
    r6_g = [feature for feature in r6_b if feature not in [GAP_ABS, YIELD_CURVE_CHANGE]]
    r6_h = [feature for feature in r6_a if feature != YIELD_CURVE_CHANGE]

    experiments = {
        "R6_A_repeat_R5_S3_full_gap_plus_yield_curve": r6_a,
        "R6_B_repeat_R5_S4_mean_abs_plus_yield_curve": r6_b,
        "R6_C_B_plus_gap_positive_days": r6_c,
        "R6_D_B_plus_gap_std": r6_d,
        "R6_E_B_without_yield_curve_change": r6_e,
        "R6_F_gap_mean_only_plus_yield_curve": r6_f,
        "R6_G_gap_mean_only_without_yield_curve": r6_g,
        "R6_H_full_gap_without_yield_curve": r6_h,
    }
    expected_counts = {
        "R6_A_repeat_R5_S3_full_gap_plus_yield_curve": 15,
        "R6_B_repeat_R5_S4_mean_abs_plus_yield_curve": 13,
        "R6_C_B_plus_gap_positive_days": 14,
        "R6_D_B_plus_gap_std": 14,
        "R6_E_B_without_yield_curve_change": 12,
        "R6_F_gap_mean_only_plus_yield_curve": 12,
        "R6_G_gap_mean_only_without_yield_curve": 11,
        "R6_H_full_gap_without_yield_curve": 14,
    }
    for name, selected in experiments.items():
        if len(selected) != len(set(selected)):
            raise AssertionError(f"Duplicate features in {name}.")
        unknown = set(selected) - set(FULL_35)
        if unknown:
            raise AssertionError(f"Unknown features in {name}: {sorted(unknown)}")
        if len(selected) != expected_counts[name]:
            raise AssertionError(
                f"{name}: expected {expected_counts[name]} features, got {len(selected)}."
            )
    return experiments


def load_best_params(run_dir: Path) -> dict[str, dict]:
    summary_path = run_dir / "all_models_validation.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    required = {"model", "params", "pooled_rmse"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Initial Search summary missing columns: {sorted(missing)}")

    best = {}
    for model_name in MODELS:
        rows = summary.loc[summary["model"].eq(model_name)].sort_values(
            "pooled_rmse", kind="stable"
        )
        if rows.empty:
            raise ValueError(f"No {model_name} result in {summary_path}.")
        best[model_name] = json.loads(rows.iloc[0]["params"])
    return best


def score(y_true: np.ndarray, y_pred: np.ndarray, y_hm: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    hm_mse = mean_squared_error(y_true, y_hm)
    return {
        "pooled_rmse": float(np.sqrt(mse)),
        "pooled_mae": float(mean_absolute_error(y_true, y_pred)),
        "pooled_r2": float(r2_score(y_true, y_pred)),
        "oos_r2_vs_historical_mean": (
            float(1.0 - mse / hm_mse) if hm_mse > 0 else np.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Round 6 ablation (R6-A to R6-H).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config["default_pipeline"] != "weekly":
        raise ValueError("Round 6 ablation requires configs/weekly_research.yaml.")
    configured = get_pipeline_features(config, "weekly")
    if set(configured) != set(FULL_35) or len(configured) != 35:
        raise ValueError("Weekly research config must contain the exact Full-35 W5 pool.")

    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    if not run_dir.is_dir() or not run_dir.name.startswith("weekly_"):
        raise ValueError("--run-dir must be the Full-35 Weekly Initial Search output.")
    best_params = load_best_params(run_dir)

    data = load_data(config, "weekly").copy()
    absent = sorted(set(FULL_35) - set(data.columns))
    if absent:
        raise ValueError(f"Weekly dataset missing Full-35 features: {absent}")

    start_year = int(config["validation"]["start_year"])
    end_year = int(config["validation"]["end_year"])
    target = config["data"]["target"]
    train_start = int(config["data"]["training_start_year"])
    window_years = int(config["training"]["window_years"])

    validation = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date")
    if validation.empty:
        raise ValueError("No Weekly validation observations.")
    monthly = list(
        validation.groupby(validation["decision_date"].dt.to_period("M"), sort=True)
    )

    experiments = build_experiments()
    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"weekly_ablation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)

    setup = {
        "round": 6,
        "stage": "Final Feature Combination Selection",
        "source_run_dir": str(run_dir.resolve()),
        "validation_years": [start_year, end_year],
        "rolling_window_years": window_years,
        "retraining": config["training"]["retraining"],
        "models": MODELS,
        "excluded_models": ["random_forest", "ols", "xgboost"],
        "fixed_best_params_from_full35_run": best_params,
        "feature_sets": experiments,
        "parameter_policy": (
            "Fixed Ridge/Lasso best parameters selected in Full-35 initial search; "
            "no hyperparameter retuning."
        ),
        "experiment_logic": {
            "R6_A": "Repeat Round 5 S3: full Gap Reversal family + Yield Curve Change.",
            "R6_B": "Repeat Round 5 S4: Gap Mean + Abs Mean + Yield Curve Change.",
            "R6_C": "R6-B + Gap Positive Days.",
            "R6_D": "R6-B + Gap Std.",
            "R6_E": "R6-B without Yield Curve Change.",
            "R6_F": "Gap Mean only + Yield Curve Change.",
            "R6_G": "Gap Mean only without Yield Curve Change.",
            "R6_H": "Full Gap Reversal family without Yield Curve Change.",
        },
    }
    (output / "ablation_setup.json").write_text(
        json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows_summary, rows_monthly, rows_predictions = [], [], []
    total_jobs = len(experiments) * len(MODELS)
    job = 0

    print("=" * 78, flush=True)
    print(
        f"WEEKLY ABLATION ROUND 6 | feature_sets={len(experiments)} "
        f"| models={MODELS} | evaluations={total_jobs} "
        f"| validation_rows={len(validation)} | months={len(monthly)}",
        flush=True,
    )
    print(f"Fixed parameters from Full-35 search: {best_params}", flush=True)
    print("-" * 78, flush=True)

    for exp_name, selected_features in experiments.items():
        for model_name in MODELS:
            job += 1
            y_all, pred_all, hm_all = [], [], []
            print(
                f"[{job:02d}/{total_jobs}] {exp_name} | {model_name} "
                f"| features={len(selected_features)}",
                flush=True,
            )

            for month_index, (month, valid) in enumerate(monthly, start=1):
                valid = valid.sort_values("decision_date")
                cutoff = pd.Timestamp(valid["decision_date"].min())
                training = eligible_training_rows(
                    data, cutoff, train_start, window_years
                )
                if training.empty:
                    raise ValueError(f"Empty training data at {month}.")
                if not (training["holding_end_date"] < cutoff).all():
                    raise AssertionError(f"Unmatured labels found in {month}.")

                estimator = build_model(
                    model_name,
                    config["models"][model_name],
                    int(config["experiment"]["random_seed"]),
                    best_params[model_name],
                )
                estimator.fit(training[selected_features], training[target])
                predicted = np.asarray(
                    estimator.predict(valid[selected_features]), dtype=float
                )
                actual = valid[target].to_numpy(dtype=float)
                historical_mean = np.full(len(valid), float(training[target].mean()))

                y_all.extend(actual.tolist())
                pred_all.extend(predicted.tolist())
                hm_all.extend(historical_mean.tolist())

                month_metrics = score(actual, predicted, historical_mean)
                rows_monthly.append({
                    "experiment": exp_name,
                    "model": model_name,
                    "month": str(month),
                    "n_features": len(selected_features),
                    "n_valid": len(valid),
                    **month_metrics,
                })
                for i, (_, row) in enumerate(valid.iterrows()):
                    rows_predictions.append({
                        "experiment": exp_name,
                        "model": model_name,
                        "decision_date": row["decision_date"],
                        "y_true": float(actual[i]),
                        "y_pred": float(predicted[i]),
                        "historical_mean_pred": float(historical_mean[i]),
                    })

                if month_index == 1 or month_index % 12 == 0 or month_index == len(monthly):
                    print(
                        f"    month {month_index:02d}/{len(monthly)} ({month}) "
                        f"| train={len(training)}",
                        flush=True,
                    )

            aggregate = score(
                np.asarray(y_all), np.asarray(pred_all), np.asarray(hm_all)
            )
            rows_summary.append({
                "experiment": exp_name,
                "model": model_name,
                "n_features": len(selected_features),
                "n_valid": len(y_all),
                "fixed_params": json.dumps(best_params[model_name], sort_keys=True),
                **aggregate,
            })
            print(
                f"    complete | RMSE={aggregate['pooled_rmse']:.6f} "
                f"| MAE={aggregate['pooled_mae']:.6f} "
                f"| R2={aggregate['pooled_r2']:.6f}",
                flush=True,
            )

    summary_df = pd.DataFrame(rows_summary).sort_values(
        ["pooled_rmse", "pooled_mae"], ascending=[True, True], kind="stable"
    ).reset_index(drop=True)
    monthly_df = pd.DataFrame(rows_monthly).sort_values(
        ["experiment", "model", "month"], kind="stable"
    ).reset_index(drop=True)
    predictions_df = pd.DataFrame(rows_predictions).sort_values(
        ["experiment", "model", "decision_date"], kind="stable"
    ).reset_index(drop=True)

    summary_df.to_csv(output / "ablation_summary.csv", index=False)
    monthly_df.to_csv(output / "ablation_by_month.csv", index=False)
    predictions_df.to_csv(output / "ablation_predictions.csv", index=False)

    print("-" * 78, flush=True)
    print(
        f"Weekly Round 6 complete | feature sets={len(experiments)} "
        f"| models={len(MODELS)} | evaluations={len(summary_df)}",
        flush=True,
    )
    print(summary_df.to_string(index=False), flush=True)
    print(f"Results saved: {output}", flush=True)


if __name__ == "__main__":
    main()
