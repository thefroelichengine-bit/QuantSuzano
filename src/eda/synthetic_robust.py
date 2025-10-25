"""Enhanced synthetic index with anti-overfitting measures and validation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict

from .config import ROLL_Z, Z_THRESHOLD, DATA_OUT
from .utils_split import temporal_split
from .metrics import calc_metrics, calculate_all_metrics


def add_noise(
    y: pd.Series,
    alpha: float = 0.10,
    multiplicative: bool = False,
    seed: int = 42,
) -> pd.Series:
    """
    Add controlled noise to target variable for robustness.
    
    Parameters
    ----------
    y : pd.Series
        Target series
    alpha : float
        Noise level (stddev multiplier for additive, scale for multiplicative)
    multiplicative : bool
        If True, use multiplicative noise; else additive
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.Series
        Noisy version of y
    """
    rng = np.random.default_rng(seed)
    
    if multiplicative:
        # y * N(1, alpha)
        noise_factor = rng.normal(1.0, alpha, size=len(y))
        return y * noise_factor
    else:
        # y + N(0, alpha * std(y))
        noise = rng.normal(0.0, y.std() * alpha, size=len(y))
        return y + noise


def fit_synthetic_robust(
    df: pd.DataFrame,
    target_col: str = "suzb_r",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    noise_alpha: float = 0.10,
    noise_mult: bool = False,
    noise_seed: int = 42,
    ridge_alphas: Optional[np.ndarray] = None,
    cv_splits: int = 5,
) -> Tuple[RidgeCV, pd.DataFrame, Dict[str, float]]:
    """
    Fit robust synthetic index with train/val/test split and noise injection.
    
    Process:
    1. Split data temporally (train/val/test)
    2. Select return features
    3. Add noise to training targets only
    4. Standardize features using train statistics
    5. Fit RidgeCV with TimeSeriesSplit on train set
    6. Evaluate on train, val, test separately
    7. Generate synthetic index and z-scores
    8. Calculate comprehensive metrics
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix indexed by date
    target_col : str
        Target column name
    train_ratio : float
        Training set proportion
    val_ratio : float
        Validation set proportion
    noise_alpha : float
        Noise injection level
    noise_mult : bool
        Use multiplicative noise if True
    noise_seed : int
        Random seed
    ridge_alphas : np.ndarray, optional
        Ridge regularization alphas to search
    cv_splits : int
        Number of CV splits for TimeSeriesSplit
    
    Returns
    -------
    model : RidgeCV
        Fitted Ridge regression model
    output_df : pd.DataFrame
        DataFrame with predictions, errors, z-scores
    metrics : dict
        Comprehensive metrics for train/val/test
    """
    print("\n" + "=" * 70)
    print("[ROBUST SYNTHETIC INDEX]")
    print("=" * 70 + "\n")
    
    # Select feature columns
    feature_cols = [
        col for col in df.columns
        if col.endswith("_r") and col != target_col
        and not col.startswith(target_col.replace("_r", ""))
    ]
    
    print(f"[FEATURES] Selected {len(feature_cols)} features:")
    print(f"  {feature_cols}\n")
    
    # Prepare full dataset
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    data = pd.concat([X, y], axis=1).dropna()
    
    print(f"[DATA] Total observations after dropna: {len(data)}")
    print(f"  Date range: {data.index.min()} to {data.index.max()}\n")
    
    # Temporal split
    df_train, df_val, df_test = temporal_split(data, train_ratio, val_ratio)
    
    # Extract X and y for each set
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    
    X_val = df_val[feature_cols]
    y_val = df_val[target_col]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    # Add noise only to training targets
    print(f"\n[NOISE] Injecting noise (alpha={noise_alpha}, mult={noise_mult})")
    y_train_noisy = add_noise(y_train, alpha=noise_alpha, multiplicative=noise_mult, seed=noise_seed)
    print(f"  Original y_train std: {y_train.std():.6f}")
    print(f"  Noisy y_train std: {y_train_noisy.std():.6f}\n")
    
    # Standardize features using train statistics only
    print("[STANDARDIZATION] Fitting scaler on train data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit RidgeCV with TimeSeriesSplit
    if ridge_alphas is None:
        ridge_alphas = np.logspace(-4, 3, 50)
    
    print(f"\n[RIDGECV] Fitting with {len(ridge_alphas)} alphas and {cv_splits} CV splits")
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    model = RidgeCV(alphas=ridge_alphas, cv=tscv, scoring='neg_mean_squared_error')
    model.fit(X_train_scaled, y_train_noisy)
    
    print(f"  Best alpha: {model.alpha_:.4f}")
    print(f"  CV score (neg MSE): {model.best_score_:.6f}\n")
    
    # Generate predictions for all sets
    print("[PREDICTIONS] Generating predictions...")
    pred_train = pd.Series(model.predict(X_train_scaled), index=X_train.index, name="synthetic_index")
    pred_val = pd.Series(model.predict(X_val_scaled), index=X_val.index, name="synthetic_index")
    pred_test = pd.Series(model.predict(X_test_scaled), index=X_test.index, name="synthetic_index")
    
    # Combine all predictions
    predictions = pd.concat([pred_train, pred_val, pred_test]).sort_index()
    
    # Calculate errors (residuals)
    error_train = y_train - pred_train
    error_val = y_val - pred_val
    error_test = y_test - pred_test
    errors = pd.concat([error_train, error_val, error_test]).sort_index().rename("error")
    
    # Calculate z-scores based on rolling statistics of residuals
    print(f"\n[Z-SCORES] Calculating rolling z-scores (window={ROLL_Z})")
    rolling_mean = errors.rolling(window=ROLL_Z, min_periods=ROLL_Z // 2).mean()
    rolling_std = errors.rolling(window=ROLL_Z, min_periods=ROLL_Z // 2).std()
    zscores = ((errors - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).rename("zscore")
    
    # Generate signals
    signals = pd.Series(0, index=zscores.index, name="signal")
    signals[zscores > Z_THRESHOLD] = -1  # Short (overvalued)
    signals[zscores < -Z_THRESHOLD] = 1  # Long (undervalued)
    
    print(f"  Valid z-scores: {zscores.notna().sum()}")
    print(f"  Long signals: {(signals == 1).sum()}")
    print(f"  Short signals: {(signals == -1).sum()}\n")
    
    # Calculate comprehensive metrics for each set
    print("[METRICS] Calculating performance metrics...")
    metrics = {}
    
    # Train metrics (using original y_train, not noisy)
    metrics.update(calculate_all_metrics(y_train, pred_train, prefix="train_").to_dict())
    
    # Validation metrics
    metrics.update(calculate_all_metrics(y_val, pred_val, prefix="val_").to_dict())
    
    # Test metrics
    metrics.update(calculate_all_metrics(y_test, pred_test, prefix="test_").to_dict())
    
    # Display key metrics
    print("\n" + "=" * 70)
    print("[PERFORMANCE SUMMARY]")
    print("=" * 70)
    print(f"{'Metric':<20} {'Train':>12} {'Val':>12} {'Test':>12}")
    print("-" * 70)
    for metric_name in ['R2', 'MAE', 'RMSE', 'IC', 'HitRatio']:
        train_val = metrics.get(f'train_{metric_name}', np.nan)
        val_val = metrics.get(f'val_{metric_name}', np.nan)
        test_val = metrics.get(f'test_{metric_name}', np.nan)
        print(f"{metric_name:<20} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f}")
    print("=" * 70 + "\n")
    
    # Check for overfitting
    train_r2 = metrics.get('train_R2', 0)
    test_r2 = metrics.get('test_R2', 0)
    if train_r2 - test_r2 > 0.15:
        print("[WARNING] Possible overfitting detected (train R2 >> test R2)")
    elif test_r2 > train_r2:
        print("[OK] Good generalization (test R2 >= train R2)")
    else:
        print("[OK] Acceptable generalization")
    
    # Create output DataFrame
    output_df = data.copy()
    output_df = output_df.join([predictions, errors, zscores, signals], how='left')
    
    # Add split labels for analysis
    output_df['split'] = 'test'
    output_df.loc[df_train.index, 'split'] = 'train'
    output_df.loc[df_val.index, 'split'] = 'val'
    
    return model, output_df, metrics


# Backward compatibility function
def fit_synthetic(df: pd.DataFrame, target_col: str = "suzb_r", **kwargs):
    """
    Backward-compatible wrapper for fit_synthetic_robust.
    
    Returns simplified output matching old signature.
    """
    model, output_df, metrics = fit_synthetic_robust(df, target_col=target_col, **kwargs)
    
    synthetic = output_df["synthetic_index"]
    zscore = output_df["zscore"]
    signals = output_df["signal"]
    
    # Create a simple model-like object with summary method
    class SimpleModel:
        def __init__(self, ridge_model, metrics_dict):
            self.model = ridge_model
            self.params = pd.Series(ridge_model.coef_, index=output_df.columns[:len(ridge_model.coef_)])
            self.metrics = metrics_dict
            self.alpha_ = ridge_model.alpha_
            self.rsquared = metrics_dict.get('train_R2', 0)
            self.rsquared_adj = self.rsquared  # Approximation
            
        def summary(self):
            # Create a simple summary object
            class Summary:
                def __init__(self, model):
                    self.model = model
                    
                def as_text(self):
                    text = "Ridge Regression Results\n"
                    text += "=" * 60 + "\n"
                    text += f"Alpha (regularization): {self.model.alpha_:.4f}\n"
                    text += f"R-squared (train): {self.model.rsquared:.4f}\n"
                    text += "=" * 60 + "\n"
                    return text
            
            return Summary(self)
    
    simple_model = SimpleModel(model, metrics)
    return simple_model, synthetic, zscore, signals


if __name__ == "__main__":
    print("\n=== Testing Robust Synthetic Index ===\n")
    
    # This would need actual data to test
    print("[OK] Module loaded successfully")
    print("[INFO] Use fit_synthetic_robust() or fit_synthetic() for backward compatibility")

