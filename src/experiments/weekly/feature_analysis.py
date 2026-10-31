"""Weekly feature analysis, using the same file role and method as Daily.

Run:
  python -m src.experiments.weekly.feature_analysis \
    --config configs/weekly_research.yaml \
    --run-dir outputs/weekly_YYYYMMDD_HHMMSS

This is the migrated version of the existing analyze_features.py method.
It does not define feature groups or select a final feature set.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_output_root, get_pipeline_features, load_config
from src.models.data import load_data
from src.models.models import build_model
from src.models.val import eligible_training_rows


DEFAULT_MODELS = ["lasso", "ridge"]
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
    summary = []
    coef = pd.DataFrame(rows)
    for feature in features:
        values = coef.loc[coef["feature"].eq(feature), "coefficient"].astype(float)
        nonzero = values[values.abs() > 1e-10]
        summary.append({
            "feature": feature,
            "selection_frequency": float((values.abs() > 1e-10).mean()),
            "positive_coefficient_frequency": float((values > 1e-10).mean()),
            "negative_coefficient_frequency": float((values < -1e-10).mean()),
            "coefficient_sign_consistency": (
                float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
                if len(nonzero) else 0.0
            ),
            "mean_coefficient": float(values.mean()),
            "median_coefficient": float(values.median()),
            "mean_abs_coefficient": float(values.abs().mean()),
            "median_abs_coefficient": float(values.abs().median()),
            "coefficient_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_months": int(values.notna().sum()),
        })
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly 35-feature diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--permutation-repeats", type=int, default=DEFAULT_PERMUTATION_REPEATS)
    args = parser.parse_args()

    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be >= 1.")

    config = load_config(args.config)
    if config["default_pipeline"] != "weekly":
        raise ValueError("Use configs/weekly_research.yaml.")
    features = get_pipeline_features(config, "weekly")
    if len(features) != 35:
        raise ValueError(f"Expected the full 35-feature Weekly pool, got {len(features)}.")

    run_dir = args.run_dir if args.run_dir.is_absolute() else Path.cwd() / args.run_dir
    if not run_dir.is_dir() or not run_dir.name.startswith("weekly_"):
        raise ValueError(f"Invalid Weekly Initial Search directory: {run_dir}")
    run_summary_path = run_dir / "all_models_validation.csv"
    if not run_summary_path.is_file():
        raise FileNotFoundError(run_summary_path)
    run_summary = pd.read_csv(run_summary_path)
    if set(pd.to_numeric(run_summary["n_features"]).unique()) != {35}:
        raise ValueError("Initial Search run must use all 35 Weekly candidate features.")

    data = load_data(config, "weekly")
    target = config["data"]["target"]
    start_year = int(config["validation"]["start_year"])
    end_year = int(config["validation"]["end_year"])

    for model_name in args.models:
        if model_name not in config["models"]:
            raise ValueError(f"Model {model_name!r} is not configured.")

    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"weekly_features_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)

    # Correlation diagnostics use pre-validation rows only, as in the prior script.
    pre_validation = data.loc[data["decision_date"].dt.year < start_year]
    if pre_validation.empty:
        raise ValueError("No pre-validation observations for correlation analysis.")
    pearson = pre_validation[features].corr(method="pearson")
    spearman = pre_validation[features].corr(method="spearman")
    pearson.to_csv(output / "pearson_correlation.csv")
    spearman.to_csv(output / "spearman_correlation.csv")

    pairs = []
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            p, s = pearson.loc[left, right], spearman.loc[left, right]
            if pd.notna(p) and pd.notna(s) and (
                abs(p) >= CORRELATION_THRESHOLD or abs(s) >= CORRELATION_THRESHOLD
            ):
                pairs.append({
                    "feature_1": left,
                    "feature_2": right,
                    "pearson": float(p),
                    "spearman": float(s),
                    "max_abs_correlation": float(max(abs(p), abs(s))),
                })
    pd.DataFrame(pairs, columns=[
        "feature_1", "feature_2", "pearson", "spearman", "max_abs_correlation"
    ]).sort_values("max_abs_correlation", ascending=False).to_csv(
        output / "high_correlation_pairs.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(pearson.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=6)
    ax.set_yticks(range(len(features)), features, fontsize=6)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    validation = data.loc[
        data["decision_date"].dt.year.between(start_year, end_year)
    ].copy().sort_values("decision_date").reset_index(drop=True)
    validation["validation_month"] = validation["decision_date"].dt.to_period("M").astype(str)
    monthly_groups = list(validation.groupby("validation_month", sort=True))
    if not monthly_groups:
        raise ValueError("No Weekly validation observations.")

    model_summaries = []
    print(
        f"Weekly feature analysis | features={len(features)} | validation rows={len(validation)} "
        f"| months={len(monthly_groups)} | models={args.models}",
        flush=True,
    )

    for model_index, model_name in enumerate(args.models, start=1):
        ranked = run_summary.loc[run_summary["model"].eq(model_name)].sort_values(
            "pooled_rmse", kind="stable"
        )
        if ranked.empty:
            raise ValueError(f"Model {model_name} not present in Initial Search.")
        best_params = json.loads(ranked.iloc[0]["params"])
        coefficient_rows = []
        month_importance_rows = []
        month_models = []
        prediction_rows = []

        print(
            f"[{model_index}/{len(args.models)}] {model_name} | params={best_params}",
            flush=True,
        )

        for month_index, (month, valid) in enumerate(monthly_groups, start=1):
            valid = valid.sort_values("decision_date")
            cutoff = pd.Timestamp(valid["decision_date"].min())
            training = eligible_training_rows(
                data, cutoff,
                int(config["data"]["training_start_year"]),
                int(config["training"]["window_years"]),
            )
            if training.empty:
                raise ValueError(f"Empty training data at {month}.")
            if not (training["holding_end_date"] < cutoff).all():
                raise AssertionError(f"Unmatured labels found in {month}.")

            estimator = build_model(
                model_name, config["models"][model_name],
                int(config["experiment"]["random_seed"]), best_params,
            )
            estimator.fit(training[features], training[target])
            x_valid = valid[features].copy()
            y_valid = valid[target].to_numpy(dtype=float)
            y_pred = np.asarray(estimator.predict(x_valid), dtype=float)
            if not np.isfinite(y_pred).all():
                raise ValueError(f"Non-finite predictions in {month}.")

            positions = validation.index[
                validation["validation_month"].eq(month)
            ].to_numpy()
            month_models.append((month, positions, estimator))
            for row_idx, actual, predicted in zip(positions, y_valid, y_pred):
                prediction_rows.append({
                    "model": model_name,
                    "month": month,
                    "decision_date": validation.loc[row_idx, "decision_date"],
                    "actual": float(actual),
                    "prediction": float(predicted),
                })

            if hasattr(estimator, "named_steps") and "regressor" in estimator.named_steps:
                regressor = estimator.named_steps["regressor"]
                if hasattr(regressor, "coef_"):
                    for feature, coefficient in zip(features, np.asarray(regressor.coef_).ravel()):
                        coefficient_rows.append({
                            "month": month,
                            "feature": feature,
                            "coefficient": float(coefficient),
                            "selected": int(abs(float(coefficient)) > 1e-10),
                        })

            # Retain monthly permutation as a diagnostic only.
            month_rng = np.random.RandomState(
                int(config["experiment"]["random_seed"]) + len(month_importance_rows)
            )
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
                    "month": month,
                    "feature": feature,
                    "importance_mse": float(np.mean(deltas)),
                    "importance_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "n_valid": len(valid),
                })

            if month_index == 1 or month_index % 6 == 0 or month_index == len(monthly_groups):
                print(f"  fitted {month_index}/{len(monthly_groups)} months ({month})", flush=True)

        base_predictions = pd.DataFrame(prediction_rows).sort_values("decision_date")
        base_mse = float(np.mean(
            (base_predictions["actual"] - base_predictions["prediction"]) ** 2
        ))
        validation_x = validation[features].reset_index(drop=True)
        validation_y = validation[target].to_numpy(dtype=float)
        rng = np.random.RandomState(int(config["experiment"]["random_seed"]) + 1000)
        pooled_rows = []

        for feature_index, feature in enumerate(features, start=1):
            repeat_deltas = []
            original = validation_x[feature].to_numpy(copy=True)
            for _ in range(args.permutation_repeats):
                permuted_x = validation_x.copy()
                permuted_x[feature] = rng.permutation(original)
                permuted_predictions = np.empty(len(validation), dtype=float)
                for _, positions, estimator in month_models:
                    permuted_predictions[positions] = estimator.predict(
                        permuted_x.iloc[positions][features]
                    )
                permuted_mse = float(np.mean((validation_y - permuted_predictions) ** 2))
                repeat_deltas.append(permuted_mse - base_mse)
            pooled_rows.append({
                "feature": feature,
                "pooled_permutation_mse_increase": float(np.mean(repeat_deltas)),
                "pooled_permutation_std": (
                    float(np.std(repeat_deltas, ddof=1)) if len(repeat_deltas) > 1 else 0.0
                ),
                "permutation_positive_repeat_fraction": float(
                    np.mean(np.asarray(repeat_deltas) > 0)
                ),
            })
            print(
                f"  pooled permutation {feature_index}/{len(features)}: {feature}",
                flush=True,
            )

        pooled = pd.DataFrame(pooled_rows).sort_values(
            "pooled_permutation_mse_increase", ascending=False
        )
        pooled.to_csv(output / f"{model_name}_pooled_permutation_importance.csv", index=False)
        plot_bar(
            pooled.rename(columns={"pooled_permutation_mse_increase": "importance"}),
            "importance", output / f"{model_name}_pooled_importance.png",
            f"{model_name}: pooled out-of-sample permutation importance",
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

        model_result = pooled.merge(monthly_summary, on="feature", how="left")
        if coefficient_rows:
            pd.DataFrame(coefficient_rows).to_csv(
                output / f"{model_name}_monthly_coefficients.csv", index=False
            )
            coefficient_summary = coefficient_diagnostics(coefficient_rows, features)
            coefficient_summary.to_csv(
                output / f"{model_name}_coefficient_stability.csv", index=False
            )
            model_result = model_result.merge(coefficient_summary, on="feature", how="left")

        model_result.to_csv(output / f"{model_name}_feature_importance.csv", index=False)
        model_result["model"] = model_name
        model_summaries.append(model_result)
        pd.DataFrame(prediction_rows).to_csv(
            output / f"{model_name}_oos_predictions.csv", index=False
        )
        print(f"Completed {model_name}: {len(monthly_groups)} monthly refits.", flush=True)

    combined = pd.concat(model_summaries, ignore_index=True)
    combined.to_csv(output / "feature_analysis_combined_long.csv", index=False)

    wide_parts = []
    for model_name in args.models:
        cols = [
            column for column in [
                "feature", "pooled_permutation_mse_increase",
                "permutation_positive_repeat_fraction", "monthly_positive_fraction",
                "selection_frequency", "coefficient_sign_consistency",
                "mean_abs_coefficient", "coefficient_std",
            ] if column in combined.columns
        ]
        part = combined.loc[combined["model"].eq(model_name), cols].copy()
        rename = {column: f"{model_name}_{column}" for column in cols if column != "feature"}
        wide_parts.append(part.rename(columns=rename).set_index("feature"))
    pd.concat(wide_parts, axis=1).reset_index().to_csv(
        output / "feature_selection_summary.csv", index=False
    )

    setup = {
        "pipeline": "weekly",
        "run_dir": str(run_dir),
        "features": features,
        "n_features": len(features),
        "models": args.models,
        "permutation_repeats": args.permutation_repeats,
        "correlation_sample": f"decision_date year < {start_year}",
        "validation_period": [start_year, end_year],
        "primary_importance": "pooled OOS permutation using month-specific fitted models",
    }
    (output / "analysis_setup.json").write_text(
        json.dumps(setup, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Weekly feature analysis complete: {output}", flush=True)


if __name__ == "__main__":
    main()
