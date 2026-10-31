"""Feature diagnostics for a completed rolling validation run.

Run from repository root, with config.yaml set to the SAME feature set used
by the supplied run directory:
  python src/models/analyze_features.py --pipeline weekly --run-dir outputs/weekly_...

Outputs include pre-validation correlations, monthly linear coefficients and
stability statistics, plus pooled out-of-sample permutation importance.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import load_config, load_data
from val import eligible_training_rows
from models import build_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ["lasso", "ridge", "random_forest"]
CORRELATION_THRESHOLD = 0.80
DEFAULT_PERMUTATION_REPEATS = 50


def plot_bar(frame: pd.DataFrame, column: str, path: Path, title: str) -> None:
    ordered = frame.sort_values(column, ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(5, len(ordered) * 0.30)))
    ax.barh(ordered["feature"], ordered[column])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(column)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def coefficient_diagnostics(rows: list[dict], features: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "feature", "selection_frequency", "positive_coefficient_frequency",
            "negative_coefficient_frequency", "coefficient_sign_consistency",
            "mean_coefficient", "median_coefficient", "mean_abs_coefficient",
            "median_abs_coefficient", "coefficient_std", "n_months"
        ])
    coef = pd.DataFrame(rows)
    summary = []
    for feature in features:
        values = coef.loc[coef["feature"] == feature, "coefficient"].astype(float)
        nonzero = values[values.abs() > 1e-10]
        positive = float((values > 1e-10).mean())
        negative = float((values < -1e-10).mean())
        nonzero_sign_consistency = 0.0
        if len(nonzero):
            nonzero_sign_consistency = float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
        summary.append({
            "feature": feature,
            "selection_frequency": float((values.abs() > 1e-10).mean()),
            "positive_coefficient_frequency": positive,
            "negative_coefficient_frequency": negative,
            "coefficient_sign_consistency": nonzero_sign_consistency,
            "mean_coefficient": float(values.mean()),
            "median_coefficient": float(values.median()),
            "mean_abs_coefficient": float(values.abs().mean()),
            "median_abs_coefficient": float(values.abs().median()),
            "coefficient_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_months": int(values.notna().sum()),
        })
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed validation output directory")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--permutation-repeats", type=int,
                        default=DEFAULT_PERMUTATION_REPEATS)
    args = parser.parse_args()
    if args.permutation_repeats < 1:
        raise ValueError("permutation-repeats must be >= 1")

    cfg = load_config()
    data = load_data(cfg, args.pipeline)
    features = list(cfg["pipelines"][args.pipeline].get("features", cfg["features"]))
    target = cfg["data"]["target"]
    start_year = int(cfg["validation"]["start_year"])
    end_year = int(cfg["validation"]["end_year"])

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not run_dir.name.startswith(args.pipeline + "_"):
        raise ValueError("run-dir does not match selected pipeline")
    summary_path = run_dir / "all_models_validation.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    run_summary = pd.read_csv(summary_path)

    # Verify the selected configuration matches the completed run's feature count.
    if "n_features" in run_summary.columns:
        observed_counts = sorted(pd.to_numeric(run_summary["n_features"], errors="coerce").dropna().unique())
        if observed_counts and observed_counts != [len(features)]:
            raise ValueError(
                f"Feature mismatch: current config has {len(features)} features, "
                f"but run summary reports {observed_counts}. Restore the matching config.yaml first."
            )

    missing = [c for c in features + [target] if c not in data.columns]
    if missing:
        raise ValueError(f"Selected features/target missing from loaded data: {missing}")

    output_root = ROOT / cfg["experiment"]["output_dir"]
    output = output_root / f"{args.pipeline}_features_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)

    # Correlations use pre-validation observations only.
    pre_validation = data.loc[data["decision_date"].dt.year < start_year].copy()
    if pre_validation.empty:
        raise ValueError("No pre-validation observations for correlation analysis")
    pearson = pre_validation[features].corr(method="pearson")
    spearman = pre_validation[features].corr(method="spearman")
    pearson.to_csv(output / "pearson_correlation.csv")
    spearman.to_csv(output / "spearman_correlation.csv")

    pairs = []
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            p = pearson.loc[left, right]
            s = spearman.loc[left, right]
            if pd.notna(p) and pd.notna(s) and (abs(p) >= CORRELATION_THRESHOLD or abs(s) >= CORRELATION_THRESHOLD):
                pairs.append({
                    "feature_1": left, "feature_2": right,
                    "pearson": float(p), "spearman": float(s),
                    "max_abs_correlation": float(max(abs(p), abs(s)))
                })
    pair_cols = ["feature_1", "feature_2", "pearson", "spearman", "max_abs_correlation"]
    pd.DataFrame(pairs, columns=pair_cols).sort_values(
        "max_abs_correlation", ascending=False
    ).to_csv(output / "high_correlation_pairs.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(12, len(features) * 0.42), max(10, len(features) * 0.36)))
    image = ax.imshow(pearson.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), features, fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    validation = data.loc[data["decision_date"].dt.year.between(start_year, end_year)].copy()
    if validation.empty:
        raise ValueError("No observations in configured validation period")
    validation = validation.sort_values("decision_date").reset_index(drop=True)
    validation["validation_month"] = validation["decision_date"].dt.to_period("M").astype(str)
    monthly_groups = list(validation.groupby("validation_month", sort=True))

    model_summaries = []
    for model_name in args.models:
        if model_name not in cfg["models"]:
            raise ValueError(f"Model not in config: {model_name}")
        ranked = run_summary.loc[run_summary["model"] == model_name].sort_values("pooled_rmse")
        if ranked.empty:
            raise ValueError(f"Model not present in run summary: {model_name}")
        best_params = json.loads(ranked.iloc[0]["params"])

        coefficient_rows: list[dict] = []
        month_importance_rows: list[dict] = []
        # Save each month-specific fitted estimator's validation predictions.
        month_models: list[tuple[str, np.ndarray, object]] = []
        prediction_rows = []

        for month, valid in monthly_groups:
            valid = valid.sort_values("decision_date")
            cutoff = valid["decision_date"].min()
            train = eligible_training_rows(
                data, cutoff,
                int(cfg["data"]["training_start_year"]),
                int(cfg["training"]["window_years"]),
            )
            if train.empty:
                raise ValueError(f"Empty training data at {month}")
            estimator = build_model(
                model_name, cfg["models"][model_name],
                int(cfg["experiment"]["random_seed"]), best_params
            )
            estimator.fit(train[features], train[target])
            x_valid = valid[features].copy()
            y_valid = valid[target].to_numpy(dtype=float)
            y_pred = estimator.predict(x_valid)
            positions = validation.index[validation["validation_month"] == month].to_numpy()
            month_models.append((month, positions, estimator))
            for row_idx, actual, predicted in zip(positions, y_valid, y_pred):
                prediction_rows.append({
                    "model": model_name, "month": month,
                    "decision_date": validation.loc[row_idx, "decision_date"],
                    "actual": float(actual), "prediction": float(predicted),
                })

            # Coefficients are recorded month by month for linear models.
            if hasattr(estimator, "named_steps") and "regressor" in estimator.named_steps:
                regressor = estimator.named_steps["regressor"]
                if hasattr(regressor, "coef_"):
                    for feature, coefficient in zip(features, np.asarray(regressor.coef_).ravel()):
                        coefficient_rows.append({
                            "month": month, "feature": feature,
                            "coefficient": float(coefficient),
                            "selected": int(abs(float(coefficient)) > 1e-10),
                        })

            # Keep monthly importance as a diagnostic, but do not treat it as
            # the primary ranking because Weekly has only a few observations/month.
            month_rng = np.random.RandomState(int(cfg["experiment"]["random_seed"]) + len(month_importance_rows))
            baseline_mse = float(np.mean((y_valid - y_pred) ** 2))
            for feature in features:
                deltas = []
                original = x_valid[feature].to_numpy(copy=True)
                for _ in range(args.permutation_repeats):
                    permuted = x_valid.copy()
                    permuted[feature] = month_rng.permutation(original)
                    perm_pred = estimator.predict(permuted)
                    deltas.append(float(np.mean((y_valid - perm_pred) ** 2) - baseline_mse))
                month_importance_rows.append({
                    "month": month, "feature": feature,
                    "importance_mse": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "n_valid": len(valid),
                })

        # Pooled out-of-sample permutation: shuffle one feature across all 202
        # validation observations, then score each observation with its own
        # month-specific model. This avoids basing the primary ranking on 4-5 rows/month.
        base_predictions = pd.DataFrame(prediction_rows).sort_values("decision_date")
        base_mse = float(np.mean((base_predictions["actual"] - base_predictions["prediction"]) ** 2))
        rng = np.random.RandomState(int(cfg["experiment"]["random_seed"]) + 1000)
        pooled_rows = []
        validation_x = validation[features].reset_index(drop=True)
        validation_y = validation[target].to_numpy(dtype=float)
        for feature in features:
            repeat_deltas = []
            original = validation_x[feature].to_numpy(copy=True)
            for _ in range(args.permutation_repeats):
                permuted_values = rng.permutation(original)
                permuted_x = validation_x.copy()
                permuted_x[feature] = permuted_values
                permuted_predictions = np.empty(len(validation), dtype=float)
                for _, positions, estimator in month_models:
                    permuted_predictions[positions] = estimator.predict(permuted_x.iloc[positions][features])
                permuted_mse = float(np.mean((validation_y - permuted_predictions) ** 2))
                repeat_deltas.append(permuted_mse - base_mse)
            pooled_rows.append({
                "feature": feature,
                "pooled_permutation_mse_increase": float(np.mean(repeat_deltas)),
                "pooled_permutation_std": float(np.std(repeat_deltas, ddof=1)) if len(repeat_deltas) > 1 else 0.0,
                "permutation_positive_repeat_fraction": float(np.mean(np.asarray(repeat_deltas) > 0)),
            })

        pooled_imp = pd.DataFrame(pooled_rows).sort_values(
            "pooled_permutation_mse_increase", ascending=False
        )
        pooled_imp.to_csv(output / f"{model_name}_pooled_permutation_importance.csv", index=False)
        plot_bar(
            pooled_imp.rename(columns={"pooled_permutation_mse_increase": "importance"}),
            "importance", output / f"{model_name}_pooled_importance.png",
            f"{model_name}: pooled out-of-sample permutation importance"
        )

        monthly_imp = pd.DataFrame(month_importance_rows)
        monthly_summary = monthly_imp.groupby("feature", as_index=False).agg(
            monthly_mean_importance=("importance_mse", "mean"),
            monthly_median_importance=("importance_mse", "median"),
            monthly_positive_fraction=("importance_mse", lambda x: float((x > 0).mean())),
            monthly_importance_std=("importance_mse", "std"),
            months=("month", "nunique"),
        )
        monthly_imp.to_csv(output / f"{model_name}_monthly_permutation_importance.csv", index=False)

        model_result = pooled_imp.merge(monthly_summary, on="feature", how="left")
        if coefficient_rows:
            coef = pd.DataFrame(coefficient_rows)
            coef.to_csv(output / f"{model_name}_monthly_coefficients.csv", index=False)
            coef_summary = coefficient_diagnostics(coefficient_rows, features)
            coef_summary.to_csv(output / f"{model_name}_coefficient_stability.csv", index=False)
            model_result = model_result.merge(coef_summary, on="feature", how="left")

        model_result.to_csv(output / f"{model_name}_feature_importance.csv", index=False)
        model_result["model"] = model_name
        model_summaries.append(model_result)
        pd.DataFrame(prediction_rows).to_csv(output / f"{model_name}_oos_predictions.csv", index=False)
        print(f"Completed {model_name}: {len(monthly_groups)} monthly refits; best params={best_params}")

    combined = pd.concat(model_summaries, ignore_index=True)
    combined.to_csv(output / "feature_analysis_combined_long.csv", index=False)

    # One row per feature, with model-specific pooled importance and stability fields.
    wide_parts = []
    for model_name in args.models:
        cols = [c for c in [
            "feature", "pooled_permutation_mse_increase", "permutation_positive_repeat_fraction",
            "monthly_positive_fraction", "selection_frequency", "coefficient_sign_consistency",
            "mean_abs_coefficient", "coefficient_std"
        ] if c in combined.columns]
        part = combined.loc[combined["model"] == model_name, cols].copy()
        rename = {c: f"{model_name}_{c}" for c in cols if c != "feature"}
        wide_parts.append(part.rename(columns=rename).set_index("feature"))
    final_summary = pd.concat(wide_parts, axis=1).reset_index()
    final_summary.to_csv(output / "feature_selection_summary.csv", index=False)

    setup = {
        "pipeline": args.pipeline,
        "run_dir": str(run_dir),
        "features": features,
        "n_features": len(features),
        "models": args.models,
        "permutation_repeats": args.permutation_repeats,
        "permutation_primary_method": "pooled OOS permutation across validation observations, scored with month-specific fitted models",
        "correlation_sample": f"decision_date year < {start_year}",
        "validation_period": [start_year, end_year],
    }
    with open(output / "analysis_setup.json", "w", encoding="utf-8") as handle:
        json.dump(setup, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved analysis to: {output}")
    print(f"Analyzed {len(features)} features across {len(monthly_groups)} validation months.")


if __name__ == "__main__":
    main()
