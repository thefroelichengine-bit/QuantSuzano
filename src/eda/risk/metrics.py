"""Comprehensive risk and performance metrics."""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Annual risk-free rate (default: 0)
    periods_per_year : int
        Periods per year for annualization
    
    Returns
    -------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return np.nan
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Periods per year
    target_return : float
        Target/minimum acceptable return
    
    Returns
    -------
    float
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Downside deviation
    downside_returns = returns[returns < target_return] - target_return
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    if downside_std == 0:
        return np.nan
    
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    return sortino


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio (alpha / tracking error).
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int
        Periods per year
    
    Returns
    -------
    float
        Information ratio
    """
    # Align series
    combined = pd.DataFrame({
        'portfolio': returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    active_returns = combined['portfolio'] - combined['benchmark']
    
    if active_returns.std() == 0:
        return np.nan
    
    ir = active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year)
    return ir


def calculate_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized tracking error.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int
        Periods per year
    
    Returns
    -------
    float
        Tracking error
    """
    combined = pd.DataFrame({
        'portfolio': returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    active_returns = combined['portfolio'] - combined['benchmark']
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    return tracking_error


def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate beta (market sensitivity).
    
    Parameters
    ----------
    returns : pd.Series
        Asset returns
    benchmark_returns : pd.Series
        Market/benchmark returns
    
    Returns
    -------
    float
        Beta coefficient
    """
    combined = pd.DataFrame({
        'asset': returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    covariance = combined['asset'].cov(combined['benchmark'])
    benchmark_variance = combined['benchmark'].var()
    
    if benchmark_variance == 0:
        return np.nan
    
    beta = covariance / benchmark_variance
    return beta


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate alpha (excess return vs CAPM).
    
    Parameters
    ----------
    returns : pd.Series
        Asset returns
    benchmark_returns : pd.Series
        Market returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Periods per year
    
    Returns
    -------
    float
        Annualized alpha
    """
    combined = pd.DataFrame({
        'asset': returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    rf = risk_free_rate / periods_per_year
    
    beta = calculate_beta(combined['asset'], combined['benchmark'])
    
    # Annualized returns
    asset_return = combined['asset'].mean() * periods_per_year
    benchmark_return = combined['benchmark'].mean() * periods_per_year
    
    # Alpha = actual return - expected return (CAPM)
    expected_return = rf + beta * (benchmark_return - rf)
    alpha = asset_return - expected_return
    
    return alpha


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio (probability weighted gains/losses).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    threshold : float
        Threshold return
    
    Returns
    -------
    float
        Omega ratio
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    
    omega = gains / losses
    return omega


def calculate_downside_deviation(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized downside deviation.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    target : float
        Target/minimum acceptable return
    periods_per_year : int
        Periods per year
    
    Returns
    -------
    float
        Downside deviation
    """
    downside_returns = returns[returns < target] - target
    downside_dev = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(periods_per_year)
    return downside_dev


def calculate_risk_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate comprehensive risk and performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Asset returns
    benchmark_returns : pd.Series, optional
        Benchmark returns for relative metrics
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Periods per year
    
    Returns
    -------
    dict
        Dictionary of all metrics
    """
    from .drawdowns import calculate_max_drawdown, calculate_calmar_ratio
    
    # Basic stats
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else total_return
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    
    metrics = {
        # Returns
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        
        # Risk-adjusted ratios
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        'omega_ratio': calculate_omega_ratio(returns, 0),
        
        # Downside risk
        'downside_deviation': calculate_downside_deviation(returns, 0, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        
        # Distribution stats
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'var_95': -returns.quantile(0.05),
        'var_99': -returns.quantile(0.01),
    }
    
    # Benchmark-relative metrics
    if benchmark_returns is not None:
        metrics.update({
            'beta': calculate_beta(returns, benchmark_returns),
            'alpha': calculate_alpha(returns, benchmark_returns, risk_free_rate, periods_per_year),
            'information_ratio': calculate_information_ratio(returns, benchmark_returns, periods_per_year),
            'tracking_error': calculate_tracking_error(returns, benchmark_returns, periods_per_year),
        })
    
    return metrics


