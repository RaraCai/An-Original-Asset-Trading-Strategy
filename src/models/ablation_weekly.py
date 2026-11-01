"""Weekly final feature-combination selection: Round 6 (Ridge + Lasso only).

Run from repository root:
  python src/models/ablation_weekly.py --run-dir outputs/weekly_<full-35-initial-run>

Rounds 1-5 are retained below as historical experiment records/comments.
This file executes Round 6 only: eight carefully controlled combinations.

All experiments use identical 2018-2021 validation observations, rolling 5Y
training window, monthly retraining, and fixed best parameters from the Full-35
initial-search run. No hyperparameter search is performed in this round.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import load_config, load_data
from val import eligible_training_rows
from models import build_model

ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Canonical 25-feature and Full-35 candidate pools (unchanged from prior rounds)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# ROUND 1 historical record: Group Ablation (NOT executed in Round 6).
# BASELINE_25, BASELINE_35 and G1-G7 were tested.
# G1 Return & Trend; G2 Volatility & Risk; G3 Intraday & Overnight;
# G4 Gap Reversal; G5 Volume; G6 Cross-Asset; G7 VIX & Macro.
# -----------------------------------------------------------------------------
# ROUND1_GROUPS = {...}

# -----------------------------------------------------------------------------
# ROUND 2 historical record: Targeted Ablation (NOT executed in Round 6).
# T2-T12 tested selective restoration of raw Intraday/Overnight features,
# Gap Reversal representations, and alternative downside-risk summaries.
# -----------------------------------------------------------------------------
# DROP_INTRADAY_OVERNIGHT = [...]
# GAP_MEAN_ONLY / GAP_MEAN_PLUS_ABS / GAP_MEAN_PLUS_STD / GAP_MEAN_PLUS_POS
# RISK_MIN_RETURN_ONLY / RISK_DOWNSIDE_STD_ONLY / RISK_MAX_DRAWDOWN_ONLY

# -----------------------------------------------------------------------------
# ROUND 3 historical record: F3/F4 (NOT executed in Round 6).
# F3 = G3 with Min Return as sole downside-risk summary (23 features).
# F4 = F3 + weekly_downside_std_5d (24 features).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ROUND 4 historical record: S0-S5 (NOT executed in Round 6).
# S0=F3; S1=F4; S2=F3 with Gap Mean only;
# S3=F3 with Gap Mean + Absolute Mean;
# S4=F3 without Cross-Asset; S5=F3 without Macro.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ROUND 5 historical record: R5-S0 to R5-S5 (NOT executed in Round 6).
# R5-S0 = S4 baseline (19 features; full Gap family + five Macro features).
# R5-S1 = S4 with Gap Mean + Absolute Mean only (17 features).
# R5-S2 = S4 with Gap Mean only (16 features).
# R5-S3 = S4 with full Gap family + Yield Curve Change only (15 features).
# R5-S4 = S4 with Gap Mean + Absolute Mean + Yield Curve Change (13 features).
# R5-S5 = S4 with full Gap family and no Macro features (14 features).
# Round 5 best: R5-S3 + Lasso, RMSE 0.024440.
# -----------------------------------------------------------------------------

MODELS = ["ridge", "lasso"]

# Shared definitions for Round 6.
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

G3_BASE = [f for f in FULL_35 if f not in DROP_INTRADAY_OVERNIGHT]
F3 = [f for f in G3_BASE if f not in RISK_FEATURES_TO_REMOVE_FOR_F3]
S4_BASE = [f for f in F3 if f not in CROSS_ASSET_FEATURES]


def build_experiments() -> dict[str, list[str]]:
    """Build Round 6 candidates from the evidence accumulated in Rounds 1-5."""
    # A: repeat Round 5 S3; best observed Lasso candidate (15 features).
    r6_a = [f for f in S4_BASE if f not in MACRO_FEATURES or f == YIELD_CURVE_CHANGE]

    # B: repeat Round 5 S4; compact Gap Mean + Abs Mean + Yield Curve Change (13).
    r6_b = [
        f for f in S4_BASE
        if f not in [GAP_STD, GAP_POS]
        and (f not in MACRO_FEATURES or f == YIELD_CURVE_CHANGE)
    ]

    # C: B + Positive Days. Is directional persistence incrementally useful?
    r6_c = r6_b + [GAP_POS]

    # D: B + Gap Reversal Std. Is within-window dispersion incrementally useful?
    r6_d = r6_b + [GAP_STD]

    # E: B without Yield Curve Change. Does the macro feature add value to the
    # compact Gap Mean + Abs Mean representation?
    r6_e = [f for f in r6_b if f != YIELD_CURVE_CHANGE]

    # F: Gap Mean only + Yield Curve Change. Is Absolute Mean necessary?
    r6_f = [f for f in r6_b if f != GAP_ABS]

    # G: Gap Mean only without Yield Curve Change. Is the simplest Gap signal
    # sufficient when the selected macro feature is also removed?
    r6_g = [f for f in r6_b if f not in [GAP_ABS, YIELD_CURVE_CHANGE]]

    # H: Full Gap Reversal family without Yield Curve Change. Compare with A to
    # isolate the selected macro feature when all Gap summaries are present.
    r6_h = [f for f in r6_a if f != YIELD_CURVE_CHANGE]

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
    allowed = set(FULL_35)
    for name, features in experiments.items():
        if len(features) != len(set(features)):
            raise AssertionError(f"Duplicate feature in {name}")
        unknown = set(features) - allowed
        if unknown:
            raise AssertionError(f"Unknown features in {name}: {sorted(unknown)}")
        if len(features) != expected_counts[name]:
            raise AssertionError(
                f"{name}: expected {expected_counts[name]} features, got {len(features)}"
            )
    return experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Weekly Round 6 final feature-combination selection")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Full-35 initial-search output directory with all_models_validation.csv",
    )
    return parser.parse_args()


def load_best_params(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "all_models_validation.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model summary: {path}")
    summary = pd.read_csv(path)
    required = {"model", "params", "pooled_rmse"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Initial-search summary missing columns: {sorted(missing)}")
    best: dict[str, dict] = {}
    for model_name in MODELS:
        rows = summary.loc[summary["model"].eq(model_name)].sort_values("pooled_rmse")
        if rows.empty:
            raise ValueError(f"No rows for model {model_name} in {path}")
        best[model_name] = json.loads(rows.iloc[0]["params"])
    return best


def score(y_true: np.ndarray, y_pred: np.ndarray, y_hm: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    hm_mse = mean_squared_error(y_true, y_hm)
    return {
        "pooled_rmse": float(np.sqrt(mse)),
        "pooled_mae": float(mean_absolute_error(y_true, y_pred)),
        "pooled_r2": float(r2_score(y_true, y_pred)),
        "oos_r2_vs_historical_mean": float(1.0 - mse / hm_mse) if hm_mse > 0 else np.nan,
    }


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.name.startswith("weekly_"):
        raise ValueError("--run-dir must be an existing weekly_... output directory")

    cfg = load_config()
    data = load_data(cfg, "weekly").copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    if "holding_end_date" in data:
        data["holding_end_date"] = pd.to_datetime(data["holding_end_date"])

    configured = list(cfg["pipelines"]["weekly"].get("features", cfg["features"]))
    if set(configured) != set(FULL_35):
        raise ValueError(
            "Config must have the Full-35 W5 feature list active. "
            f"Config has {len(configured)} features; expected 35."
        )
    absent = sorted(set(FULL_35) - set(data.columns))
    if absent:
        raise ValueError(f"Weekly dataset is missing Full-35 columns: {absent}")

    start_year = int(cfg["validation"]["start_year"])
    end_year = int(cfg["validation"]["end_year"])
    target = cfg["data"]["target"]
    train_start = int(cfg["data"]["training_start_year"])
    window_years = int(cfg["training"]["window_years"])
    validation = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date")
    if validation.empty:
        raise ValueError("No validation observations in configured validation years")
    monthly = list(validation.groupby(validation["decision_date"].dt.to_period("M"), sort=True))

    best_params = load_best_params(run_dir)
    experiments = build_experiments()
    output = ROOT / cfg["experiment"]["output_dir"] / f"weekly_ablation_round6_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)

    setup = {
        "round": 6,
        "stage": "Final Feature Combination Selection",
        "source_run_dir": str(run_dir),
        "validation_years": [start_year, end_year],
        "rolling_window_years": window_years,
        "retraining": cfg["training"]["retraining"],
        "models": MODELS,
        "excluded_models": ["random_forest", "ols", "xgboost"],
        "fixed_best_params_from_full35_run": best_params,
        "feature_sets": experiments,
        "parameter_policy": "Fixed Ridge/Lasso best parameters selected in Full-35 initial search; no retuning.",
        "historical_records": "Rounds 1-5 retained in source comments/history; Round 6 executes R6-A to R6-H only.",
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
    (output / "ablation_setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")

    rows_summary, rows_monthly, rows_predictions = [], [], []
    for exp_name, features in experiments.items():
        for model_name in MODELS:
            y_all: list[float] = []
            pred_all: list[float] = []
            hm_all: list[float] = []
            for month, valid in monthly:
                valid = valid.sort_values("decision_date")
                cutoff = valid["decision_date"].min()
                train = eligible_training_rows(data, cutoff, train_start, window_years)
                if train.empty:
                    raise ValueError(f"Empty training data at {month}")
                estimator = build_model(
                    model_name, cfg["models"][model_name],
                    int(cfg["experiment"]["random_seed"]), best_params[model_name],
                )
                estimator.fit(train[features], train[target])
                pred = np.asarray(estimator.predict(valid[features]), dtype=float)
                y = valid[target].to_numpy(dtype=float)
                hm = np.full(len(valid), float(train[target].mean()))
                y_all.extend(y.tolist())
                pred_all.extend(pred.tolist())
                hm_all.extend(hm.tolist())

                month_metrics = score(y, pred, hm)
                rows_monthly.append({
                    "experiment": exp_name, "model": model_name, "month": str(month),
                    "n_features": len(features), "n_valid": len(valid), **month_metrics,
                })
                for i, (_, row) in enumerate(valid.iterrows()):
                    rows_predictions.append({
                        "experiment": exp_name, "model": model_name,
                        "decision_date": row["decision_date"],
                        "y_true": float(y[i]), "y_pred": float(pred[i]),
                        "historical_mean_pred": float(hm[i]),
                    })

            aggregate = score(np.asarray(y_all), np.asarray(pred_all), np.asarray(hm_all))
            rows_summary.append({
                "experiment": exp_name, "model": model_name,
                "n_features": len(features), "n_valid": len(y_all),
                "fixed_params": json.dumps(best_params[model_name], sort_keys=True),
                **aggregate,
            })
            print(
                f"{exp_name:52s} {model_name:6s} n_features={len(features):2d} "
                f"RMSE={aggregate['pooled_rmse']:.6f} "
                f"MAE={aggregate['pooled_mae']:.6f} R2={aggregate['pooled_r2']:.6f}"
            )

    # Match Daily output convention: best-to-worst, no ranking columns.
    summary_df = pd.DataFrame(rows_summary).sort_values(
        ["pooled_rmse", "pooled_mae"], ascending=[True, True]
    ).reset_index(drop=True)
    monthly_df = pd.DataFrame(rows_monthly).sort_values(
        ["experiment", "model", "month"]
    ).reset_index(drop=True)
    predictions_df = pd.DataFrame(rows_predictions).sort_values(
        ["experiment", "model", "decision_date"]
    ).reset_index(drop=True)

    summary_df.to_csv(output / "ablation_summary.csv", index=False)
    monthly_df.to_csv(output / "ablation_by_month.csv", index=False)
    predictions_df.to_csv(output / "ablation_predictions.csv", index=False)

    print(f"\nSaved Weekly ablation Round 6 results to: {output}")
    print(
        f"Feature sets: {len(experiments)} | Models: {', '.join(MODELS)} | "
        f"Evaluations: {len(experiments) * len(MODELS)}"
    )
    print("Round 6: R6-A to R6-H; Ridge and Lasso only; summary sorted best-to-worst by pooled RMSE, then MAE; no ranking columns.")


if __name__ == "__main__":
    main()
