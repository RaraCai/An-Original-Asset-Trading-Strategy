from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eval import plot_predictions
from val import run_validation, aggregate_results

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name('config.yaml')


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f'Expected YAML mapping: {path}')
    return config


def load_data(config: dict[str, Any]) -> pd.DataFrame:
    data_path = ROOT / config['data']['path']
    if not data_path.is_file():
        raise FileNotFoundError(f'Processed labeled data not found: {data_path}')
    dates = ['decision_date', 'execution_date', 'holding_end_date']
    data = pd.read_csv(data_path, parse_dates=dates)
    features = config['features']
    target = config['data']['target']
    required = set(dates + features + [target])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'Dataset missing columns: {missing}')
    data = data.dropna(subset=list(required)).sort_values('decision_date').reset_index(drop=True)
    if data['decision_date'].duplicated().any():
        raise ValueError('decision_date must be unique.')
    if not np.isfinite(data[features + [target]].to_numpy(dtype=float)).all():
        raise ValueError('Features and target must be finite.')
    return data


def candidate_table(frame: pd.DataFrame, param_names: list[str]) -> pd.DataFrame:
    """One column per hyperparameter, followed by three mean validation metrics."""
    expanded = pd.DataFrame([json.loads(p) for p in frame['params']], index=frame.index)
    for name in param_names:
        if name not in expanded:
            expanded[name] = None
    return pd.concat([
        frame[['model']].reset_index(drop=True),
        expanded[param_names].reset_index(drop=True),
        frame[['mean_rmse', 'mean_mae', 'mean_r2']].reset_index(drop=True),
    ], axis=1)


def create_run_dir(config: dict[str, Any]) -> Path:
    base = ROOT / config['experiment']['output_dir']
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = base / f'validation_{stamp}'
    run_dir.mkdir(exist_ok=False)
    return run_dir


def main() -> None:
    config = load_config()
    data = load_data(config)
    search_results, plot_data = run_validation(data, config)
    summary = aggregate_results(search_results, config)
    run_dir = create_run_dir(config)

    summary[['model', 'params', 'mean_rmse', 'mean_mae', 'mean_r2']].to_csv(
        run_dir / 'all_models_validation.csv', index=False
    )
    for name, model_config in config['models'].items():
        subset = summary.loc[summary['model'] == name]
        cols = list(model_config.get('param_grid', {}))
        candidate_table(subset, cols).to_csv(run_dir / f'{name}_validation.csv', index=False)

    # Plot uses first ParameterGrid candidate for each model, NOT best candidates.
    plot_predictions(plot_data, run_dir / 'validation_predictions.png',
                     'Validation: First Grid Candidate per Model')
    with (run_dir / 'parameters_used.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(
            {name: model_config.get('param_grid', {})
             for name, model_config in config['models'].items()},
            f, sort_keys=False, allow_unicode=True
        )
    print(f'Validation complete. {len(summary)} candidate combinations evaluated.')
    print('Ranking metric:', config['validation']['selection_metric'])
    print(summary.to_string(index=False))
    print(f'Files saved to: {run_dir}')


if __name__ == '__main__':
    main()
