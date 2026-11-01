"""Stage 2C-4: Daily Lasso Top-K and hypothesis-driven feature ablation.

Run from repository root:
 python src/models/ablation_daily.py --run-dir outputs/daily_initial_TIMESTAMP --analysis-dir outputs/daily_features_TIMESTAMP

Only validation data are used; no automatic best-model selection or test fitting.
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
from sklearn.model_selection import ParameterGrid

from models import build_model
from train import ROOT, load_config, load_data
from val import eligible_training_rows

# -----------------------------------------------------------------------------
# HISTORICAL EXPERIMENT CONFIGURATIONS (kept for project source-code audit)
# -----------------------------------------------------------------------------
# Stage 2C-1: Group ablation; models = ["lasso", "ridge", "random_forest"].
# FEATURE_GROUPS = {
#     "price_trend": ["return_1d", "momentum_5d", "momentum_20d", "ma_gap_5_20", "rsi_14"],
#     "volatility_vix": ["volatility_5d", "volatility_20d", "volatility_60d", "vix"],
#     "intraday_overnight": ["intraday_return", "overnight_return", "intraday_momentum_5d", "gap_reversal"],
#     "cross_asset_volume": ["qqq_spy_relative_5d", "tlt_return_5d", "relative_volume_20d"],
#     "macro_rates": ["yield_2y_change_5d", "yield_10y_change_5d", "yield_10y_change_20d", "yield_curve_10y_2y", "real_yield_10y_change_5d"],
# }
# Stage 2C-2: Single-feature and correlated-group removals.
# ABLATIONS_STAGE_2C2 = {
#     "full": [], "no_qqq_relative": ["qqq_spy_relative_5d"],
#     "no_relative_volume": ["relative_volume_20d"], "no_tlt": ["tlt_return_5d"],
#     "no_momentum_5d": ["momentum_5d"], "no_momentum_20d": ["momentum_20d"],
#     "no_rsi_14": ["rsi_14"], "no_ma_gap": ["ma_gap_5_20"],
#     "no_volatility_20d": ["volatility_20d"], "no_volatility_5d": ["volatility_5d"],
#     "no_gap_reversal": ["gap_reversal"], "no_intraday_momentum": ["intraday_momentum_5d"],
#     "no_overnight_return": ["overnight_return"],
#     "no_momentum20_rsi": ["momentum_20d", "rsi_14"],
#     "no_momentum20_ma_gap": ["momentum_20d", "ma_gap_5_20"],
#     "no_volatility5_20": ["volatility_5d", "volatility_20d"],
#     "no_qqq_volume": ["qqq_spy_relative_5d", "relative_volume_20d"],
#     "no_cross_asset_volume": ["qqq_spy_relative_5d", "relative_volume_20d", "tlt_return_5d"],
# }
# Stage 2C-3: Combination removals from full 21 features.
# ABLATIONS_STAGE_2C3 = {
#     "F0_full": [],
#     "F1_no_intraday_momentum": ["intraday_momentum_5d"],
#     "F2_no_intraday_momentum_qqq": ["intraday_momentum_5d", "qqq_spy_relative_5d"],
#     "F3_no_intraday_momentum_volume": ["intraday_momentum_5d", "relative_volume_20d"],
#     "F4_no_intraday_momentum_qqq_volume": ["intraday_momentum_5d", "qqq_spy_relative_5d", "relative_volume_20d"],
#     "F5_F4_no_ma_gap": ["intraday_momentum_5d", "qqq_spy_relative_5d", "relative_volume_20d", "ma_gap_5_20"],
# }

# -----------------------------------------------------------------------------
# ACTIVE STAGE 2C-4: 17 K values x 2 ranking methods, plus controls and groups.
# -----------------------------------------------------------------------------
TOP_K_VALUES = list(range(4, 21))
F5_REMOVED = ["intraday_momentum_5d", "qqq_spy_relative_5d", "relative_volume_20d", "ma_gap_5_20"]
EXTRA_REMOVALS = {
    "G1_F5_no_rsi": F5_REMOVED + ["rsi_14"],
    "G2_F5_no_momentum20": F5_REMOVED + ["momentum_20d"],
    "G3_F5_no_volatility5": F5_REMOVED + ["volatility_5d"],
    "G4_F5_no_rsi_momentum20": F5_REMOVED + ["rsi_14", "momentum_20d"],
    "G5_F5_no_rsi_momentum20_vol5": F5_REMOVED + ["rsi_14", "momentum_20d", "volatility_5d"],
}


def resolve_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def get_lasso_params(run_dir: Path) -> dict:
    table = pd.read_csv(run_dir / "all_models_validation.csv")
    required = {"model", "params", "pooled_rmse"}
    if not required.issubset(table.columns):
        raise ValueError(f"Initial ranking missing {required - set(table.columns)}")
    rows = table.loc[table["model"] == "lasso"].sort_values("pooled_rmse")
    if rows.empty:
        raise ValueError("No Lasso in initial validation ranking")
    params = json.loads(rows.iloc[0]["params"])
    saved = run_dir / "parameters_used.yaml"
    if saved.exists():
        grids = yaml.safe_load(saved.read_text(encoding="utf-8"))
        if "lasso" not in grids or params not in list(ParameterGrid(grids["lasso"])):
            raise ValueError("Selected Lasso parameters do not match original saved grid")
    return params


def load_rankings(analysis_dir: Path, features: list[str]) -> dict[str, list[str]]:
    path = analysis_dir / "lasso_feature_importance.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature ranking input: {path}")
    frame = pd.read_csv(path)
    required = {"feature", "selection_frequency", "permutation_mse_increase"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Feature importance file missing {required - set(frame.columns)}")
    if frame["feature"].duplicated().any() or set(frame["feature"]) != set(features):
        raise ValueError("Feature importance rows must match the 21 config features exactly")
    if frame[list(required - {"feature"})].isna().any().any():
        raise ValueError("Feature ranking contains missing scores")
    # Stable tie-breaker: descending importance, then alphabetical feature name.
    frequency = frame.sort_values(["selection_frequency", "feature"], ascending=[False, True], kind="stable")
    permutation = frame.sort_values(["permutation_mse_increase", "feature"], ascending=[False, True], kind="stable")
    return {"frequency": frequency["feature"].tolist(), "permutation": permutation["feature"].tolist()}


def build_experiments(features: list[str], rankings: dict[str, list[str]]) -> list[dict]:
    experiments = [
        {"experiment": "Full21", "category": "baseline", "features": features.copy()},
        {"experiment": "F5_17", "category": "reference", "features": [f for f in features if f not in F5_REMOVED]},
    ]
    for method, ranked in rankings.items():
        for k in TOP_K_VALUES:
            experiments.append({"experiment": f"top{k:02d}_{method}", "category": f"topk_{method}",
                                "features": [f for f in features if f in set(ranked[:k])]})
    for name, removed in EXTRA_REMOVALS.items():
        if not set(removed).issubset(features):
            raise ValueError(f"Unknown features in {name}")
        experiments.append({"experiment": name, "category": "economic_group",
                            "features": [f for f in features if f not in set(removed)]})
    for item in experiments:
        if not item["features"]:
            raise ValueError(f"Empty feature set for {item['experiment']}")
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Lasso Top-K ablation (Stage 2C-4)")
    parser.add_argument("--run-dir", type=Path, required=True, help="Original daily_initial directory")
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Original daily_features directory")
    args = parser.parse_args()
    config = load_config()
    if config["training"].get("window") != "rolling" or config["training"].get("retraining") != "monthly":
        raise ValueError("Expected rolling window and monthly retraining")
    features = list(config["features"])
    if len(features) != 21 or len(set(features)) != 21:
        raise ValueError("This study expects the original 21 unique features")
    run_dir, analysis_dir = resolve_path(args.run_dir), resolve_path(args.analysis_dir)
    params = get_lasso_params(run_dir)
    rankings = load_rankings(analysis_dir, features)
    experiments = build_experiments(features, rankings)
    data = load_data(config, "daily")
    validation = data.loc[data["decision_date"].dt.year.between(
        config["validation"]["start_year"], config["validation"]["end_year"])].copy()
    if validation.empty:
        raise ValueError("No validation observations")
    monthly = list(validation.groupby(validation["decision_date"].dt.to_period("M"), sort=True))
    # Cache identical feature sets to avoid refitting duplicates from two ranking methods.
    prediction_cache: dict[tuple[str, ...], pd.DataFrame] = {}
    rows_out, predictions_out = [], []
    target = config["data"]["target"]
    for idx, item in enumerate(experiments, start=1):
        chosen = item["features"]
        key = tuple(chosen)
        if key not in prediction_cache:
            parts = []
            for month, month_rows in monthly:
                month_rows = month_rows.sort_values("decision_date")
                cutoff = pd.Timestamp(month_rows["decision_date"].min())
                training = eligible_training_rows(data, cutoff, config["data"]["training_start_year"],
                                                  int(config["training"]["window_years"]))
                if training.empty:
                    raise ValueError(f"Empty training set in {month}")
                model = build_model("lasso", config["models"]["lasso"], config["experiment"]["random_seed"], params)
                model.fit(training[chosen], training[target])
                pred = np.asarray(model.predict(month_rows[chosen]), dtype=float)
                if not np.isfinite(pred).all():
                    raise ValueError(f"Nonfinite predictions for {item['experiment']} in {month}")
                parts.append(pd.DataFrame({"decision_date": month_rows["decision_date"].to_numpy(),
                                           "actual_return": month_rows[target].to_numpy(dtype=float),
                                           "predicted_return": pred}))
            prediction_cache[key] = pd.concat(parts, ignore_index=True)
        result = prediction_cache[key]
        y = result["actual_return"].to_numpy()
        p = result["predicted_return"].to_numpy()
        rows_out.append({"experiment": item["experiment"], "category": item["category"],
                         "n_features": len(chosen), "n_valid": len(result),
                         "pooled_rmse": float(np.sqrt(mean_squared_error(y, p))),
                         "pooled_mae": float(mean_absolute_error(y, p)),
                         "pooled_r2": float(r2_score(y, p)),
                         "params": json.dumps(params, sort_keys=True),
                         "features": json.dumps(chosen),
                         "removed_features": json.dumps([f for f in features if f not in chosen])})
        pred_frame = result.copy()
        pred_frame.insert(0, "experiment", item["experiment"])
        predictions_out.append(pred_frame)
        print(f"[{idx:02d}/{len(experiments)}] {item['experiment']:<38} RMSE={rows_out[-1]['pooled_rmse']:.9f}", flush=True)
    summary = pd.DataFrame(rows_out)
    full_rmse = float(summary.loc[summary["experiment"] == "Full21", "pooled_rmse"].iloc[0])
    f5_rmse = float(summary.loc[summary["experiment"] == "F5_17", "pooled_rmse"].iloc[0])
    summary["rmse_change_vs_full_pct"] = (summary["pooled_rmse"] / full_rmse - 1) * 100
    summary["rmse_change_vs_f5_pct"] = (summary["pooled_rmse"] / f5_rmse - 1) * 100
    summary = summary.sort_values(["pooled_rmse", "experiment"], kind="stable").reset_index(drop=True)
    output = ROOT / config["experiment"]["output_dir"] / ("daily_ablation_topk_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "all_ablation_results.csv", index=False)
    summary.loc[summary["category"].str.startswith("topk")].to_csv(output / "topk_results.csv", index=False)
    summary.loc[summary["category"].isin(["baseline", "reference", "economic_group"])].to_csv(output / "economic_group_results.csv", index=False)
    pd.concat(predictions_out, ignore_index=True).to_csv(output / "ablation_predictions.csv", index=False, date_format="%Y-%m-%d")
    with (output / "ablation_setup.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"stage": "2C-4", "model": "lasso", "params": params,
                        "top_k_values": TOP_K_VALUES, "rankings": rankings,
                        "economic_group_removals": EXTRA_REMOVALS,
                        "experiment_features": {x["experiment"]: x["features"] for x in experiments},
                        "initial_run": str(run_dir), "analysis_run": str(analysis_dir),
                        "validation_years": [config["validation"]["start_year"], config["validation"]["end_year"]],
                        "window_years": config["training"]["window_years"], "retraining": "monthly"},
                       f, sort_keys=False, allow_unicode=True)
    print(f"\nCompleted {len(experiments)} comparisons ({len(prediction_cache)} unique feature sets).\nSaved: {output}")


if __name__ == "__main__":
    main()
