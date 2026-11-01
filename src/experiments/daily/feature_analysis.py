"""Daily feature diagnostics for a completed rolling-origin validation run.

Run from the repository root:
  python -m src.experiments.daily.feature_analysis \
    --config configs/daily_research.yaml \
    --run-dir outputs/daily_YYYYMMDD_HHMMSS

Method preserved from the previous Daily analysis:
- Feature-feature Pearson/Spearman correlations use pre-validation observations.
- Best validation parameters are selected separately for each requested model.
- Each validation month is refitted using the same rolling-window/mature-label rule.
- Monthly coefficients and monthly permutation importance are recorded.
- Primary permutation importance is pooled across all validation observations,
  while each observation is scored by its own month-specific fitted model.

This script is diagnostic only. It does not select or remove features and does
not use the independent test period.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import get_pipeline_features, get_output_root, load_config
from src.models.dataloader  import load_data
from src.models.models import build_model
from src.models.val import eligible_training_rows


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS = ["lasso", "ridge"]
CORRELATION_THRESHOLD = 0.80
DEFAULT_PERMUTATION_REPEATS = 50


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


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


def coefficient_diagnostics(rows: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    columns = [
        "feature", "selection_frequency", "positive_coefficient_frequency",
        "negative_coefficient_frequency", "coefficient_sign_consistency",
        "mean_coefficient", "median_coefficient", "mean_abs_coefficient",
        "median_abs_coefficient", "coefficient_std", "n_months",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    coefficients = pd.DataFrame(rows)
    summary = []
    for feature in features:
        values = coefficients.loc[
            coefficients["feature"].eq(feature), "coefficient"
        ].astype(float)
        nonzero = values[values.abs() > 1e-10]
        positive = float((values > 1e-10).mean())
        negative = float((values < -1e-10).mean())
        sign_consistency = (
            float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
            if len(nonzero) else 0.0
        )
        summary.append({
            "feature": feature,
            "selection_frequency": float((values.abs() > 1e-10).mean()),
            "positive_coefficient_frequency": positive,
            "negative_coefficient_frequency": negative,
            "coefficient_sign_consistency": sign_consistency,
            "mean_coefficient": float(values.mean()),
            "median_coefficient": float(values.median()),
            "mean_abs_coefficient": float(values.abs().mean()),
            "median_abs_coefficient": float(values.abs().median()),
            "coefficient_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_months": int(values.notna().sum()),
        })
    return pd.DataFrame(summary)


def load_best_parameters(run_dir: Path, model_name: str) -> dict[str, Any]:
    summary_path = run_dir / "all_models_validation.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {"model", "params", "pooled_rmse", "n_features"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Validation summary is missing columns: {sorted(missing)}")

    candidates = summary.loc[summary["model"].eq(model_name)].sort_values(
        "pooled_rmse", kind="stable"
    )
    if candidates.empty:
        raise ValueError(f"No validation result found for model {model_name!r}.")
    return json.loads(candidates.iloc[0]["params"])


def verify_run_matches_config(
    run_dir: Path, pipeline: str, feature_count: int
) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not run_dir.name.startswith(f"{pipeline}_"):
        raise ValueError(f"Run directory {run_dir.name!r} does not match {pipeline!r}.")
    summary = pd.read_csv(run_dir / "all_models_validation.csv")
    counts = sorted(
        pd.to_numeric(summary["n_features"], errors="coerce").dropna().unique()
    )
    if counts != [feature_count]:
        raise ValueError(
            f"Feature mismatch: config has {feature_count}, run summary has {counts}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily feature diagnostics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--permutation-repeats", type=int, default=DEFAULT_PERMUTATION_REPEATS)
    args = parser.parse_args()

    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be at least 1.")

    config = load_config(args.config)
    pipeline = config["default_pipeline"]
    if pipeline != "daily":
        raise ValueError("This script is Daily-only; use the Weekly analysis script for Weekly.")
    features = get_pipeline_features(config, "daily")
    if len(features) != 21:
        raise ValueError(f"Daily research analysis expects 21 candidate features, got {len(features)}.")

    run_dir = resolve_path(args.run_dir)
    verify_run_matches_config(run_dir, "daily", len(features))
    data = load_data(config, "daily")
    target = config["data"]["target"]

    for model_name in args.models:
        if model_name not in config["models"]:
            raise ValueError(f"Model {model_name!r} is not configured.")
    validation_start = int(config["validation"]["start_year"])
    validation_end = int(config["validation"]["end_year"])

    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"daily_features_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)

    # Correlations are calculated only from observations before validation.
    pre_validation = data.loc[data["decision_date"].dt.year < validation_start]
    if pre_validation.empty:
        raise ValueError("No pre-validation observations available for correlation analysis.")
    pearson = pre_validation[features].corr(method="pearson")
    spearman = pre_validation[features].corr(method="spearman")
    pearson.to_csv(output / "pearson_correlation.csv")
    spearman.to_csv(output / "spearman_correlation.csv")

    pairs = []
    for index, feature_1 in enumerate(features):
        for feature_2 in features[index + 1:]:
            p = pearson.loc[feature_1, feature_2]
            s = spearman.loc[feature_1, feature_2]
            if pd.notna(p) and pd.notna(s) and (
                abs(p) >= CORRELATION_THRESHOLD or abs(s) >= CORRELATION_THRESHOLD
            ):
                pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "pearson": float(p),
                    "spearman": float(s),
                    "max_abs_correlation": float(max(abs(p), abs(s))),
                })
    pair_columns = [
        "feature_1", "feature_2", "pearson", "spearman", "max_abs_correlation"
    ]
    pd.DataFrame(pairs, columns=pair_columns).sort_values(
        "max_abs_correlation", ascending=False
    ).to_csv(output / "high_correlation_pairs.csv", index=False)

    fig, ax = plt.subplots(
        figsize=(max(12, len(features) * 0.42), max(10, len(features) * 0.36))
    )
    image = ax.imshow(pearson.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), features, fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    validation = data.loc[
        data["decision_date"].dt.year.between(validation_start, validation_end)
    ].copy().sort_values("decision_date").reset_index(drop=True)
    if validation.empty:
        raise ValueError("No observations in the configured validation period.")
    validation["validation_month"] = (
        validation["decision_date"].dt.to_period("M").astype(str)
    )
    monthly_groups = list(validation.groupby("validation_month", sort=True))

    print("=" * 76, flush=True)
    print("DAILY FEATURE ANALYSIS STARTED", flush=True)
    print(
        f"Features={len(features)} | validation observations={len(validation)} "
        f"| months={len(monthly_groups)} | permutation repeats={args.permutation_repeats}",
        flush=True,
    )
    print(f"Models={', '.join(args.models)}", flush=True)
    print("-" * 76, flush=True)

    model_summaries = []
    for model_index, model_name in enumerate(args.models, start=1):
        best_params = load_best_parameters(run_dir, model_name)
        print(
            f"[Model {model_index}/{len(args.models)}] {model_name} "
            f"| best params={best_params} | fitting {len(monthly_groups)} months",
            flush=True,
        )

        coefficient_rows: list[dict[str, Any]] = []
        monthly_importance_rows: list[dict[str, Any]] = []
        month_models: list[tuple[str, np.ndarray, Any]] = []
        prediction_rows: list[dict[str, Any]] = []

        for month_index, (month, valid) in enumerate(monthly_groups, start=1):
            valid = valid.sort_values("decision_date")
            cutoff = pd.Timestamp(valid["decision_date"].min())
            training = eligible_training_rows(
                data,
                cutoff,
                int(config["data"]["training_start_year"]),
                int(config["training"]["window_years"]),
            )
            if training.empty:
                raise ValueError(f"Empty training data in {month}.")
            if not (training["holding_end_date"] < cutoff).all():
                raise AssertionError(f"Unmatured training labels found in {month}.")

            estimator = build_model(
                model_name,
                config["models"][model_name],
                int(config["experiment"]["random_seed"]),
                best_params,
            )
            estimator.fit(training[features], training[target])
            x_valid = valid[features].copy()
            y_valid = valid[target].to_numpy(dtype=float)
            y_pred = np.asarray(estimator.predict(x_valid), dtype=float)
            if not np.isfinite(y_pred).all():
                raise ValueError(f"Non-finite validation predictions in {month}.")

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
                    for feature, coefficient in zip(
                        features, np.asarray(regressor.coef_).ravel()
                    ):
                        coefficient_rows.append({
                            "month": month,
                            "feature": feature,
                            "coefficient": float(coefficient),
                            "selected": int(abs(float(coefficient)) > 1e-10),
                        })

            # Monthly permutation is retained as a secondary diagnostic.
            month_rng = np.random.RandomState(
                int(config["experiment"]["random_seed"]) + len(monthly_importance_rows)
            )
            baseline_mse = float(np.mean((y_valid - y_pred) ** 2))
            for feature in features:
                deltas = []
                original = x_valid[feature].to_numpy(copy=True)
                for _ in range(args.permutation_repeats):
                    permuted = x_valid.copy()
                    permuted[feature] = month_rng.permutation(original)
                    perm_pred = estimator.predict(permuted)
                    deltas.append(
                        float(np.mean((y_valid - perm_pred) ** 2) - baseline_mse)
                    )
                monthly_importance_rows.append({
                    "month": month,
                    "feature": feature,
                    "importance_mse": float(np.mean(deltas)),
                    "importance_std": (
                        float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                    ),
                    "n_valid": len(valid),
                })

            if month_index == 1 or month_index % 6 == 0 or month_index == len(monthly_groups):
                print(
                    f"  {model_name}: fitted month {month_index}/{len(monthly_groups)} "
                    f"({month})",
                    flush=True,
                )

        # Pooled OOS permutation importance: all validation observations are
        # shuffled, then scored by their corresponding month-specific estimator.
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
            for repeat in range(args.permutation_repeats):
                permuted_x = validation_x.copy()
                permuted_x[feature] = rng.permutation(original)
                permuted_predictions = np.empty(len(validation), dtype=float)
                for _, positions, estimator in month_models:
                    permuted_predictions[positions] = estimator.predict(
                        permuted_x.iloc[positions][features]
                    )
                permuted_mse = float(np.mean(
                    (validation_y - permuted_predictions) ** 2
                ))
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
                f"  {model_name}: permutation {feature_index}/{len(features)} "
                f"({feature})",
                flush=True,
            )

        pooled_importance = pd.DataFrame(pooled_rows).sort_values(
            "pooled_permutation_mse_increase", ascending=False
        )
        pooled_importance.to_csv(
            output / f"{model_name}_pooled_permutation_importance.csv", index=False
        )
        plot_bar(
            pooled_importance.rename(
                columns={"pooled_permutation_mse_increase": "importance"}
            ),
            "importance",
            output / f"{model_name}_pooled_importance.png",
            f"{model_name}: pooled out-of-sample permutation importance",
        )

        monthly_importance = pd.DataFrame(monthly_importance_rows)
        monthly_summary = monthly_importance.groupby(
            "feature", as_index=False
        ).agg(
            monthly_mean_importance=("importance_mse", "mean"),
            monthly_median_importance=("importance_mse", "median"),
            monthly_positive_fraction=("importance_mse", lambda values: float((values > 0).mean())),
            monthly_importance_std=("importance_mse", "std"),
            months=("month", "nunique"),
        )
        monthly_importance.to_csv(
            output / f"{model_name}_monthly_permutation_importance.csv", index=False
        )

        model_result = pooled_importance.merge(
            monthly_summary, on="feature", how="left"
        )
        if coefficient_rows:
            coefficients = pd.DataFrame(coefficient_rows)
            coefficients.to_csv(
                output / f"{model_name}_monthly_coefficients.csv", index=False
            )
            coefficient_summary = coefficient_diagnostics(coefficient_rows, features)
            coefficient_summary.to_csv(
                output / f"{model_name}_coefficient_stability.csv", index=False
            )
            model_result = model_result.merge(
                coefficient_summary, on="feature", how="left"
            )

        model_result.to_csv(output / f"{model_name}_feature_importance.csv", index=False)
        model_result["model"] = model_name
        model_summaries.append(model_result)
        pd.DataFrame(prediction_rows).to_csv(
            output / f"{model_name}_oos_predictions.csv", index=False
        )
        print(
            f"[Model {model_index}/{len(args.models)}] {model_name} complete.",
            flush=True,
        )

    combined = pd.concat(model_summaries, ignore_index=True)
    combined.to_csv(output / "feature_analysis_combined_long.csv", index=False)

    wide_parts = []
    for model_name in args.models:
        columns = [
            column for column in [
                "feature",
                "pooled_permutation_mse_increase",
                "permutation_positive_repeat_fraction",
                "monthly_positive_fraction",
                "selection_frequency",
                "coefficient_sign_consistency",
                "mean_abs_coefficient",
                "coefficient_std",
            ]
            if column in combined.columns
        ]
        part = combined.loc[combined["model"].eq(model_name), columns].copy()
        rename = {
            column: f"{model_name}_{column}"
            for column in columns if column != "feature"
        }
        wide_parts.append(part.rename(columns=rename).set_index("feature"))
    pd.concat(wide_parts, axis=1).reset_index().to_csv(
        output / "feature_selection_summary.csv", index=False
    )

    setup = {
        "pipeline": "daily",
        "config_path": config["_meta"]["config_path"],
        "run_dir": str(run_dir),
        "features": features,
        "n_features": len(features),
        "models": args.models,
        "permutation_repeats": args.permutation_repeats,
        "permutation_primary_method": (
            "pooled OOS permutation across validation observations, "
            "scored with month-specific fitted models"
        ),
        "correlation_sample": f"decision_date year < {validation_start}",
        "validation_period": [validation_start, validation_end],
        "training_window_years": int(config["training"]["window_years"]),
        "retraining": config["training"]["retraining"],
        "label_availability_rule": config["training"]["label_availability_rule"],
        "correlation_threshold": CORRELATION_THRESHOLD,
        "note": (
            "Exploratory feature diagnostics. Validation importance is not an "
            "independent test estimate and should not be interpreted causally."
        ),
    }
    (output / "analysis_setup.json").write_text(
        json.dumps(setup, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("-" * 76, flush=True)
    print(
        f"Daily feature analysis complete | features={len(features)} "
        f"| models={len(args.models)} | validation months={len(monthly_groups)} "
        f"| permutation repeats={args.permutation_repeats}",
        flush=True,
    )
    print(f"Saved analysis: {output}", flush=True)


if __name__ == "__main__":
    main()
