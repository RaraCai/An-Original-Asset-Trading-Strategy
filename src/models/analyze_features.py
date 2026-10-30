"""Feature diagnostics for a previously completed rolling validation run.

Run from project root:
  python src/models/analyze_features.py --pipeline daily --run-dir outputs/daily_initial_...
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from train import load_config, load_data
from val import eligible_training_rows
from models import build_model

ROOT = Path(__file__).resolve().parents[2]
# EDIT THIS LIST TO CHANGE THE MODELS ANALYZED. Any model present in the run summary works.
DEFAULT_MODELS = ['lasso', 'ridge', 'random_forest']
CORRELATION_THRESHOLD = 0.80
PERMUTATION_REPEATS = 3


def plots_bar(frame, column, path, title, signed=False):
    frame = frame.sort_values(column)
    fig, ax = plt.subplots(figsize=(10, max(5, len(frame) * 0.32)))
    ax.barh(frame['feature'], frame[column], color='#557da8')
    if signed:
        ax.axvline(0, color='black', lw=.8)
    ax.set_title(title)
    ax.set_xlabel(column)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', choices=['daily', 'weekly'], default='daily')
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed initial validation run directory')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--permutation-repeats', type=int, default=PERMUTATION_REPEATS)
    args = parser.parse_args()
    if args.permutation_repeats < 1:
        raise ValueError('permutation-repeats must be >= 1')
    cfg = load_config()
    data = load_data(cfg, args.pipeline)
    features = list(cfg['features'])
    target = cfg['data']['target']
    start_year = int(cfg['validation']['start_year'])
    end_year = int(cfg['validation']['end_year'])
    train_only = data.loc[data['decision_date'].dt.year < start_year]
    if train_only.empty:
        raise ValueError('No pre-validation training data for correlation analysis')

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not run_dir.name.startswith(args.pipeline + '_'):
        raise ValueError('run-dir does not match the selected pipeline')
    summary_path = run_dir / 'all_models_validation.csv'
    summary = pd.read_csv(summary_path)
    output = ROOT / cfg['experiment']['output_dir'] / f'{args.pipeline}_features_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)

    pearson = train_only[features].corr('pearson')
    spearman = train_only[features].corr('spearman')
    pearson.to_csv(output / 'pearson_correlation.csv')
    spearman.to_csv(output / 'spearman_correlation.csv')
    pairs = []
    for i, left in enumerate(features):
        for right in features[i+1:]:
            p = pearson.loc[left, right]
            s = spearman.loc[left, right]
            if pd.notna(p) and (abs(p) >= CORRELATION_THRESHOLD or abs(s) >= CORRELATION_THRESHOLD):
                pairs.append({'feature_1': left, 'feature_2': right, 'pearson': p, 'spearman': s, 'max_abs_correlation': max(abs(p), abs(s))})
    pair_cols = ['feature_1', 'feature_2', 'pearson', 'spearman', 'max_abs_correlation']
    pd.DataFrame(pairs, columns=pair_cols).sort_values('max_abs_correlation', ascending=False).to_csv(output / 'high_correlation_pairs.csv', index=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pearson.values, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    fig.tight_layout()
    fig.savefig(output / 'correlation_heatmap.png', dpi=180)
    plt.close(fig)

    validation = data.loc[data['decision_date'].dt.year.between(start_year, end_year)].copy()
    monthly = list(validation.groupby(validation['decision_date'].dt.to_period('M'), sort=True))
    all_summaries = []
    for model_name in args.models:
        if model_name not in cfg['models']:
            raise ValueError(f'Model not in config: {model_name}')
        ranked = summary.loc[summary['model'] == model_name].sort_values('pooled_rmse')
        if ranked.empty:
            raise ValueError(f'Model not in run summary: {model_name}')
        params = json.loads(ranked.iloc[0]['params'])
        coefficient_rows = []
        importance_rows = []
        for month, valid in monthly:
            valid = valid.sort_values('decision_date')
            cutoff = valid['decision_date'].min()
            train = eligible_training_rows(data, cutoff, int(cfg['data']['training_start_year']), int(cfg['training']['window_years']))
            if train.empty:
                raise ValueError(f'Empty training data at {month}')
            estimator = build_model(model_name, cfg['models'][model_name], int(cfg['experiment']['random_seed']), params)
            estimator.fit(train[features], train[target])
            if hasattr(estimator, 'named_steps'):
                coef = estimator.named_steps['regressor'].coef_
                for f, c in zip(features, coef):
                    coefficient_rows.append({'month': str(month), 'feature': f, 'coefficient': float(c), 'selected': int(abs(c) > 1e-10)})
            # Permutation importance measured on future validation observations, not training fit.
            perm = permutation_importance(estimator, valid[features], valid[target], scoring='neg_mean_squared_error', n_repeats=args.permutation_repeats, random_state=int(cfg['experiment']['random_seed']), n_jobs=1)
            for j, f in enumerate(features):
                importance_rows.append({'month': str(month), 'feature': f, 'importance_mse': float(perm.importances_mean[j]), 'n_valid': len(valid)})
        imp = pd.DataFrame(importance_rows)
        # Weighted average of monthly MSE increases; monthly validation sample sizes differ.
        imp['weighted'] = imp['importance_mse'] * imp['n_valid']
        grouped = imp.groupby('feature', as_index=False).agg(weighted_sum=('weighted', 'sum'), n=('n_valid', 'sum'), positive_months=('importance_mse', lambda x: int((x > 0).sum())), months=('month', 'nunique'))
        grouped['permutation_mse_increase'] = grouped['weighted_sum'] / grouped['n']
        grouped['positive_month_fraction'] = grouped['positive_months'] / grouped['months']
        result = grouped[['feature', 'permutation_mse_increase', 'positive_month_fraction']].copy()
        if coefficient_rows:
            coefs = pd.DataFrame(coefficient_rows)
            csum = coefs.groupby('feature', as_index=False).agg(selection_frequency=('selected', 'mean'), mean_coefficient=('coefficient', 'mean'), mean_abs_coefficient=('coefficient', lambda x: x.abs().mean()), coefficient_std=('coefficient', 'std'))
            result = result.merge(csum, on='feature')
            coefs.to_csv(output / f'{model_name}_monthly_coefficients.csv', index=False)
        result = result.sort_values('permutation_mse_increase', ascending=False)
        result.to_csv(output / f'{model_name}_feature_importance.csv', index=False)
        plots_bar(result, 'permutation_mse_increase', output / f'{model_name}_importance.png', f'{model_name} validation permutation importance', signed=True)
        all_summaries.append(result[['feature', 'permutation_mse_increase']].rename(columns={'permutation_mse_increase': f'{model_name}_mse_increase'}).set_index('feature'))
        print(f'Completed {model_name} using {len(monthly)} monthly refits; params={params}')
    pd.concat(all_summaries, axis=1).reset_index().to_csv(output / 'feature_selection_summary.csv', index=False)
    print(f'\nSaved analysis to {output}')

if __name__ == '__main__':
    main()
