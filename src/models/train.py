"""Training entry point for configured Daily or Weekly rolling validation."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.config import get_pipeline_features, get_output_root, load_config
from src.models.data import load_data
from src.models.eval import plot_best_model_predictions
from src.models.val import aggregate_results, run_validation


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m" if hours else (
        f"{minutes:d}m {seconds:02d}s" if minutes else f"{seconds:d}s"
    )


def create_run_dir(config: dict, pipeline: str) -> Path:
    output_root = get_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{pipeline}_{stamp}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def save_run_metadata(config: dict, pipeline: str, run_dir: Path) -> None:
    config_path = Path(config["_meta"]["config_path"])
    shutil.copy2(config_path, run_dir / "config_snapshot.yaml")
    metadata = {
        "pipeline": pipeline,
        "config_path": str(config_path),
        "experiment_name": config["experiment"].get("name"),
        "features": get_pipeline_features(config, pipeline),
        "n_features": len(get_pipeline_features(config, pipeline)),
        "validation_period": [
            int(config["validation"]["start_year"]),
            int(config["validation"]["end_year"]),
        ],
        "training": config["training"],
    }
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly rolling-origin validation for Daily or Weekly."
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML path.")
    parser.add_argument("--pipeline", choices=["daily", "weekly"], default=None)
    parser.add_argument(
        "--progress-every-months", type=int, default=1,
        help="Show monthly progress every N validation months (default: 1).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Disable progress messages (not recommended for long searches).",
    )
    args = parser.parse_args()

    overall_started = time.perf_counter()
    print("=" * 78, flush=True)
    print("MODEL TRAINING COMMAND STARTED", flush=True)
    print(f"Config: {args.config}", flush=True)

    config = load_config(args.config)
    pipeline = args.pipeline or config["default_pipeline"]
    if pipeline not in config["pipelines"]:
        raise ValueError(f"Pipeline {pipeline!r} is not configured in {args.config}.")

    features = get_pipeline_features(config, pipeline)
    print(
        f"Pipeline={pipeline} | features={len(features)} | models={len(config['models'])}",
        flush=True,
    )
    print("Loading processed dataset...", flush=True)
    data = load_data(config, pipeline)
    print(
        f"Dataset loaded: {len(data)} complete rows | "
        f"dropped incomplete rows={data.attrs.get('rows_dropped_for_missing_values', 0)}",
        flush=True,
    )
    print("Starting monthly rolling validation...", flush=True)

    predictions, fit_log, _ = run_validation(
        data,
        config,
        pipeline,
        progress=not args.quiet,
        progress_every_months=args.progress_every_months,
    )
    print("Aggregating validation metrics...", flush=True)
    summary = aggregate_results(predictions)
    run_dir = create_run_dir(config, pipeline)

    summary.to_csv(run_dir / "all_models_validation.csv", index=False)
    for model_name, model_config in config["models"].items():
        subset = summary.loc[summary["model"].eq(model_name)].copy()
        parameter_names = list(model_config.get("param_grid", {}))
        expanded = pd.DataFrame(
            [json.loads(value) for value in subset["params"]],
            index=subset.index,
        )
        for name in parameter_names:
            if name not in expanded:
                expanded[name] = None
        metric_columns = [
            "n_features", "n_valid", "pooled_rmse", "pooled_mae", "pooled_r2"
        ]
        pd.concat([
            expanded[parameter_names].reset_index(drop=True),
            subset[metric_columns].reset_index(drop=True),
        ], axis=1).to_csv(run_dir / f"{model_name}_validation.csv", index=False)

    print("Saving predictions, fit log, plots and configuration snapshot...", flush=True)
    predictions.to_csv(
        run_dir / "validation_predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    fit_log.to_csv(
        run_dir / "monthly_fit_log.csv", index=False, date_format="%Y-%m-%d"
    )
    plot_best_model_predictions(
        predictions,
        summary,
        run_dir / "validation_predictions.png",
        f"{pipeline.title()} Pipeline: Best Configuration per Model",
    )

    with (run_dir / "parameters_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                name: model_config.get("param_grid", {})
                for name, model_config in config["models"].items()
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    save_run_metadata(config, pipeline, run_dir)

    elapsed = time.perf_counter() - overall_started
    print("=" * 78, flush=True)
    print(
        f"{pipeline.title()} validation complete | features={len(features)} "
        f"| observations={predictions['decision_date'].nunique()} "
        f"| candidates={len(summary)} | monthly fits={len(fit_log)} "
        f"| elapsed={_duration(elapsed)}",
        flush=True,
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Results saved: {run_dir}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
