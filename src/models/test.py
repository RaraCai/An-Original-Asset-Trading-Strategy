from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from eval import evaluate_predictions
from models import build_model
from train import load_config, load_data
from val import eligible_training_rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit and test ONE manually chosen model and hyperparameter set.')
    parser.add_argument('--run-dir', required=True, type=Path,
                        help='Validation run directory containing parameters_used.yaml')
    parser.add_argument('--model', required=True, choices=['random_forest', 'ols', 'lasso', 'ridge', 'xgboost'])
    parser.add_argument('--params', required=True,
                        help='JSON object with exact model parameters, e.g. {"regressor__alpha": 1.0}')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = load_config()
    with (run_dir / 'parameters_used.yaml').open(encoding='utf-8') as f:
        validated_grids = yaml.safe_load(f)
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise ValueError('--params must be a JSON object.')
    if args.model not in config['models'] or args.model not in validated_grids:
        raise ValueError(f'Model {args.model} not present in validation configuration.')
    if config['models'][args.model].get('param_grid', {}) != validated_grids[args.model]:
        raise ValueError('Current config grid differs from the recorded validation grid.')
    from sklearn.model_selection import ParameterGrid
    allowed = list(ParameterGrid(validated_grids[args.model]))
    if params not in allowed:
        raise ValueError(f'Parameters are not one of the validated combinations: {params}')

    data = load_data(config)
    test_start = config['test']['start_year']
    cutoff = pd.Timestamp(year=test_start, month=1, day=1)
    training = eligible_training_rows(data, cutoff, config['data']['training_start_year'])
    test_rows = data.loc[data['decision_date'].dt.year.between(
        test_start, config['test']['end_year'])].copy()
    if training.empty or test_rows.empty:
        raise ValueError('Missing eligible training or test rows.')

    model = build_model(args.model, config['models'][args.model],
                        config['experiment']['random_seed'], params)
    model.fit(training[config['features']], training[config['data']['target']])
    predictions = test_rows[['decision_date', 'execution_date', 'holding_end_date']].copy()
    predictions.insert(0, 'model', args.model)
    predictions['actual_return'] = test_rows[config['data']['target']].to_numpy()
    predictions['predicted_return'] = model.predict(test_rows[config['features']])

    output_dir = run_dir / f'test_{args.model}'
    if output_dir.exists():
        raise FileExistsError(f'Test output already exists; refusing to overwrite: {output_dir}')
    output_dir.mkdir()
    metrics = evaluate_predictions(predictions, output_dir, 'test')
    joblib.dump(model, output_dir / f'{args.model}.joblib')
    with (output_dir / 'chosen_model_and_params.json').open('w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'params': params}, f, indent=2)
    print(metrics.to_string(index=False))
    print(f'Test results saved to: {output_dir}')


if __name__ == '__main__':
    main()
