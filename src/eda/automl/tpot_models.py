"""
TPOT AutoML for automated machine learning pipeline optimization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tpot import TPOTRegressor
from pathlib import Path
from typing import Tuple
from ..config import DATA_OUT
from ..utils_split import temporal_split


def run_tpot_optimization(
    df: pd.DataFrame,
    target_col: str = 'suzb_r',
    feature_cols: list = None,
    generations: int = 10,
    population_size: int = 50,
    cv_splits: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    verbosity: int = 2
) -> Tuple[TPOTRegressor, dict]:
    """
    Run TPOT AutoML to find best pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target variable
    feature_cols : list, optional
        Feature columns (if None, uses all _r columns except target)
    generations : int
        Number of generations for genetic programming
    population_size : int
        Population size for each generation
    cv_splits : int
        Number of cross-validation splits
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    random_state : int
        Random seed
    verbosity : int
        Verbosity level (0=silent, 2=progress bar)
    
    Returns
    -------
    tpot_model : TPOTRegressor
        Fitted TPOT model
    results : dict
        Results dictionary with metrics and pipeline info
    """
    print("\n" + "=" * 70)
    print("TPOT AutoML Optimization")
    print("=" * 70)
    
    # Prepare data
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.endswith('_r') and col != target_col]
    
    # Remove rows with missing values
    data = df[[target_col] + feature_cols].dropna()
    
    # Temporal split
    train_df, val_df, test_df = temporal_split(data, train_ratio=0.7, val_ratio=0.15)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {len(feature_cols)}")
    
    # Configure TPOT
    print(f"\nTPOT Configuration:")
    print(f"  Generations: {generations}")
    print(f"  Population: {population_size}")
    print(f"  CV Splits: {cv_splits}")
    print(f"  Scoring: neg_mean_squared_error")
    
    tpot = TPOTRegressor(
        generations=generations,
        population_size=population_size,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_squared_error',
        random_state=random_state,
        verbosity=verbosity,
        n_jobs=n_jobs,
        early_stop=5,  # Stop if no improvement after 5 generations
        config_dict='TPOT light'  # Faster, less complex pipelines
    )
    
    print("\n[TPOT] Starting optimization...")
    print("This may take 30-60 minutes for quick mode (10 generations)")
    print("Progress will be displayed below:\n")
    
    # Fit TPOT
    tpot.fit(X_train, y_train)
    
    # Evaluate on all sets
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    y_train_pred = tpot.predict(X_train)
    y_val_pred = tpot.predict(X_val)
    y_test_pred = tpot.predict(X_test)
    
    results = {
        'feature_cols': feature_cols,
        'best_pipeline': str(tpot.fitted_pipeline_),
        'train_metrics': {
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred),
        },
        'val_metrics': {
            'MSE': mean_squared_error(y_val, y_val_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'MAE': mean_absolute_error(y_val, y_val_pred),
            'R2': r2_score(y_val, y_val_pred),
        },
        'test_metrics': {
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred),
        },
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("TPOT Optimization Complete!")
    print("=" * 70)
    print("\nBest Pipeline:")
    print(results['best_pipeline'])
    print("\nPerformance Metrics:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Set:")
        for metric, value in results[f'{split}_metrics'].items():
            print(f"  {metric}: {value:.6f}")
    
    # Export best pipeline
    pipeline_path = DATA_OUT / "tpot_best_pipeline.py"
    tpot.export(str(pipeline_path))
    print(f"\nBest pipeline exported to: {pipeline_path}")
    
    # Save results
    results_path = DATA_OUT / "tpot_results.csv"
    results_df = pd.DataFrame({
        'metric': list(results['train_metrics'].keys()),
        'train': list(results['train_metrics'].values()),
        'val': list(results['val_metrics'].values()),
        'test': list(results['test_metrics'].values()),
    })
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    return tpot, results

