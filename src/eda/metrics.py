"""Comprehensive metrics for model evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calc_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    prefix : str
        Prefix for metric names (e.g., 'train_', 'test_')
    
    Returns
    -------
    dict
        Dictionary with MAE, RMSE, MAPE, R2, IC, Hit Ratio
    """
    # Ensure alignment
    aligned = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    y_t = aligned["true"]
    y_p = aligned["pred"]
    
    # Basic error metrics
    error = y_t - y_p
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())
    mape = (error.abs() / (y_t.abs() + 1e-9)).mean()
    
    # R-squared
    ss_res = (error ** 2).sum()
    ss_tot = ((y_t - y_t.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-9))
    
    # Information Coefficient (correlation)
    ic = y_t.corr(y_p)
    
    # Directional accuracy (hit ratio)
    hit_ratio = (np.sign(y_t) == np.sign(y_p)).mean()
    
    return {
        f"{prefix}MAE": mae,
        f"{prefix}RMSE": rmse,
        f"{prefix}MAPE": mape,
        f"{prefix}R2": r2,
        f"{prefix}IC": ic,
        f"{prefix}HitRatio": hit_ratio,
        f"{prefix}N": len(y_t),
    }


def rolling_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 60,
    metrics: list = None,
) -> pd.DataFrame:
    """
    Calculate rolling metrics over time.
    
    Parameters
    ----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    window : int
        Rolling window size
    metrics : list
        List of metrics to calculate ['mae', 'rmse', 'r2', 'ic']
    
    Returns
    -------
    pd.DataFrame
        Rolling metrics indexed by date
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2', 'ic']
    
    # Align data
    aligned = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    error = aligned["true"] - aligned["pred"]
    
    results = pd.DataFrame(index=aligned.index)
    
    if 'mae' in metrics:
        results['rolling_mae'] = error.abs().rolling(window).mean()
    
    if 'rmse' in metrics:
        results['rolling_rmse'] = np.sqrt((error ** 2).rolling(window).mean())
    
    if 'r2' in metrics:
        # Rolling R2 is more complex
        def rolling_r2(series):
            y_t = aligned["true"].loc[series.index]
            y_p = aligned["pred"].loc[series.index]
            ss_res = ((y_t - y_p) ** 2).sum()
            ss_tot = ((y_t - y_t.mean()) ** 2).sum()
            return 1 - (ss_res / (ss_tot + 1e-9))
        
        results['rolling_r2'] = error.rolling(window).apply(rolling_r2, raw=False)
    
    if 'ic' in metrics:
        def rolling_corr(series):
            idx = series.index
            return aligned["true"].loc[idx].corr(aligned["pred"].loc[idx])
        
        results['rolling_ic'] = error.rolling(window).apply(rolling_corr, raw=False)
    
    return results.dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of periods per year for annualization
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0
    
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, target: float = 0.0) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of periods per year
    target : float
        Target return (default 0)
    
    Returns
    -------
    float
        Annualized Sortino ratio
    """
    downside_returns = returns[returns < target]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    return (returns.mean() - target) / downside_std * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative returns series
    
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    cumulative = (1 + cumulative_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def calculate_all_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    prefix: str = "",
) -> pd.Series:
    """
    Calculate all available metrics and return as Series.
    
    Parameters
    ----------
    y_true : pd.Series
        True values
    y_pred : pd.Series
        Predicted values
    prefix : str
        Prefix for metric names
    
    Returns
    -------
    pd.Series
        All metrics as a Series
    """
    metrics = calc_metrics(y_true, y_pred, prefix=prefix)
    
    # Add strategy metrics if applicable
    error = y_true - y_pred
    
    # Add residual statistics
    metrics[f"{prefix}Error_Mean"] = error.mean()
    metrics[f"{prefix}Error_Std"] = error.std()
    metrics[f"{prefix}Error_Skew"] = error.skew()
    metrics[f"{prefix}Error_Kurt"] = error.kurtosis()
    
    return pd.Series(metrics)

