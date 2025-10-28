"""
Comprehensive model comparison framework.
Tests multiple models and generates comparison reports.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from ..config import DATA_OUT
from ..utils_split import temporal_split
from ..metrics import calc_metrics, calc_information_coefficient

# Try to import optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class ModelComparator:
    """
    Compare multiple regression models for time series prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def add_models(self):
        """Initialize model candidates."""
        print("\n[MODEL COMPARISON] Initializing models...")
        
        # Linear models
        self.models['Ridge'] = Ridge(alpha=100, random_state=self.random_state)
        self.models['Lasso'] = Lasso(alpha=0.001, random_state=self.random_state)
        self.models['ElasticNet'] = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=self.random_state)
        
        # Tree-based models
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=self.random_state
        )
        
        # XGBoost
        if HAS_XGBOOST:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # LightGBM
        if HAS_LIGHTGBM:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        print(f"  Initialized {len(self.models)} models")
        for name in self.models.keys():
            print(f"    - {name}")
    
    def fit_all(self, X_train, y_train, X_val, y_val):
        """Fit all models."""
        print("\n[MODEL COMPARISON] Training all models...")
        
        for name, model in self.models.items():
            print(f"\n  Training {name}...")
            try:
                model.fit(X_train, y_train)
                
                # Evaluate
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                self.predictions[name] = {
                    'train': y_train_pred,
                    'val': y_val_pred
                }
                
                self.results[name] = {
                    'train': {
                        'MSE': mean_squared_error(y_train, y_train_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'MAE': mean_absolute_error(y_train, y_train_pred),
                        'R2': r2_score(y_train, y_train_pred),
                        'IC': calc_information_coefficient(y_train, y_train_pred),
                    },
                    'val': {
                        'MSE': mean_squared_error(y_val, y_val_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                        'MAE': mean_absolute_error(y_val, y_val_pred),
                        'R2': r2_score(y_val, y_val_pred),
                        'IC': calc_information_coefficient(y_val, y_val_pred),
                    }
                }
                
                print(f"    Val R2: {self.results[name]['val']['R2']:.4f}, IC: {self.results[name]['val']['IC']:.4f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                self.results[name] = None
    
    def create_ensemble(self, X_train, y_train, X_val, y_val, top_k: int = 3):
        """Create voting ensemble from top k models."""
        print(f"\n[MODEL COMPARISON] Creating ensemble from top {top_k} models...")
        
        # Rank models by validation R2
        valid_models = {name: res for name, res in self.results.items() if res is not None}
        ranked = sorted(valid_models.items(), key=lambda x: x[1]['val']['R2'], reverse=True)
        
        top_models = [(name, self.models[name]) for name, _ in ranked[:top_k]]
        
        print(f"  Selected models:")
        for name, _ in top_models:
            print(f"    - {name} (R2: {self.results[name]['val']['R2']:.4f})")
        
        # Create voting regressor
        ensemble = VotingRegressor(estimators=top_models)
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = ensemble.predict(X_train)
        y_val_pred = ensemble.predict(X_val)
        
        self.models['VotingEnsemble'] = ensemble
        self.predictions['VotingEnsemble'] = {
            'train': y_train_pred,
            'val': y_val_pred
        }
        
        self.results['VotingEnsemble'] = {
            'train': {
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'R2': r2_score(y_train, y_train_pred),
                'IC': calc_information_coefficient(y_train, y_train_pred),
            },
            'val': {
                'MSE': mean_squared_error(y_val, y_val_pred),
                'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'MAE': mean_absolute_error(y_val, y_val_pred),
                'R2': r2_score(y_val, y_val_pred),
                'IC': calc_information_coefficient(y_val, y_val_pred),
            }
        }
        
        print(f"  Ensemble Val R2: {self.results['VotingEnsemble']['val']['R2']:.4f}")
    
    def save_comparison(self, output_path: Path = None):
        """Save comparison results to CSV."""
        if output_path is None:
            output_path = DATA_OUT / "model_comparison.csv"
        
        # Convert results to DataFrame
        rows = []
        for model_name, metrics in self.results.items():
            if metrics is None:
                continue
            for split in ['train', 'val']:
                for metric_name, value in metrics[split].items():
                    rows.append({
                        'model': model_name,
                        'split': split,
                        'metric': metric_name,
                        'value': value
                    })
        
        df = pd.DataFrame(rows)
        df_pivot = df.pivot_table(index=['model', 'metric'], columns='split', values='value')
        df_pivot.to_csv(output_path)
        
        print(f"\n[MODEL COMPARISON] Results saved to: {output_path}")
        
        return df_pivot
    
    def plot_comparison(self, metric: str = 'R2', output_dir: Path = None):
        """Generate comparison plots."""
        if output_dir is None:
            output_dir = DATA_OUT / "plots/models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metric values
        model_names = []
        train_vals = []
        val_vals = []
        
        for name, metrics in self.results.items():
            if metrics is None:
                continue
            model_names.append(name)
            train_vals.append(metrics['train'][metric])
            val_vals.append(metrics['val'][metric])
        
        # Create bar plot
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, val_vals, width, label='Validation', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_{metric}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Plot saved to: {output_dir / f'model_comparison_{metric}.png'}")


def compare_all_models(
    df: pd.DataFrame,
    target_col: str = 'suzb_r',
    feature_cols: list = None,
    create_ensemble: bool = True
) -> Tuple[ModelComparator, pd.DataFrame]:
    """
    Run comprehensive model comparison.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target variable
    feature_cols : list, optional
        Feature columns
    create_ensemble : bool
        Whether to create voting ensemble
    
    Returns
    -------
    comparator : ModelComparator
        Fitted comparator object
    results_df : pd.DataFrame
        Results table
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    
    # Prepare data
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.endswith('_r') and col != target_col]
    
    data = df[[target_col] + feature_cols].dropna()
    train_df, val_df, test_df = temporal_split(data, train_ratio=0.7, val_ratio=0.15)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    print(f"\nData: {len(X_train)} train, {len(X_val)} val samples")
    print(f"Features: {len(feature_cols)}")
    
    # Initialize and run comparison
    comparator = ModelComparator()
    comparator.add_models()
    comparator.fit_all(X_train, y_train, X_val, y_val)
    
    if create_ensemble:
        comparator.create_ensemble(X_train, y_train, X_val, y_val, top_k=3)
    
    # Save results
    results_df = comparator.save_comparison()
    
    # Plot comparisons
    for metric in ['R2', 'RMSE', 'IC']:
        comparator.plot_comparison(metric=metric)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)
    
    # Print summary
    print("\nTop 3 Models (by Validation R2):")
    valid_models = {name: res for name, res in comparator.results.items() if res is not None}
    ranked = sorted(valid_models.items(), key=lambda x: x[1]['val']['R2'], reverse=True)
    for i, (name, metrics) in enumerate(ranked[:3], 1):
        print(f"  {i}. {name}: R2={metrics['val']['R2']:.4f}, IC={metrics['val']['IC']:.4f}")
    
    return comparator, results_df

