"""Value at Risk (VaR) and Conditional VaR calculations."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence : float
        Confidence level (default: 0.95 for 95% VaR)
    method : str
        Method: 'historical', 'parametric', 'cornish_fisher'
    
    Returns
    -------
    float
        VaR (positive number represents loss)
    """
    returns_clean = returns.dropna()
    
    if method == 'historical':
        # Historical VaR (non-parametric)
        var = -np.percentile(returns_clean, (1 - confidence) * 100)
    
    elif method == 'parametric':
        # Parametric VaR (assumes normal distribution)
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        var = -(mu + sigma * stats.norm.ppf(1 - confidence))
    
    elif method == 'cornish_fisher':
        # Cornish-Fisher VaR (accounts for skewness and kurtosis)
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        skew = returns_clean.skew()
        kurt = returns_clean.kurtosis()
        
        z = stats.norm.ppf(1 - confidence)
        z_cf = (z +
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * (kurt - 3) / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        var = -(mu + sigma * z_cf)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return var


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that loss exceeds VaR.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence : float
        Confidence level
    method : str
        Method: 'historical' or 'parametric'
    
    Returns
    -------
    float
        CVaR (positive number represents loss)
    """
    returns_clean = returns.dropna()
    var = calculate_var(returns_clean, confidence, method)
    
    if method == 'historical':
        # Average of losses beyond VaR
        losses_beyond_var = returns_clean[returns_clean <= -var]
        cvar = -losses_beyond_var.mean() if len(losses_beyond_var) > 0 else var
    
    elif method == 'parametric':
        # Parametric CVaR (normal distribution)
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        z = stats.norm.ppf(1 - confidence)
        cvar = -(mu + sigma * stats.norm.pdf(z) / (1 - confidence))
    
    else:
        cvar = var  # Fallback
    
    return cvar


def calculate_rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = 'historical'
) -> pd.Series:
    """
    Calculate rolling VaR.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size
    confidence : float
        Confidence level
    method : str
        VaR method
    
    Returns
    -------
    pd.Series
        Rolling VaR
    """
    def calc_var_window(x):
        if len(x) < 10:  # Need minimum data
            return np.nan
        return calculate_var(pd.Series(x), confidence, method)
    
    rolling_var = returns.rolling(window=window).apply(calc_var_window, raw=True)
    return rolling_var


def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95
) -> dict:
    """
    Backtest VaR model using violation rate and independence tests.
    
    Parameters
    ----------
    returns : pd.Series
        Actual returns
    var_series : pd.Series
        VaR estimates (positive numbers)
    confidence : float
        Confidence level used for VaR
    
    Returns
    -------
    dict
        Backtest results including Kupiec and Christoffersen tests
    """
    # Align series
    combined = pd.DataFrame({
        'returns': returns,
        'var': var_series
    }).dropna()
    
    # Violations: when loss exceeds VaR
    violations = (combined['returns'] < -combined['var']).astype(int)
    
    n_violations = violations.sum()
    n_obs = len(violations)
    violation_rate = n_violations / n_obs
    expected_rate = 1 - confidence
    
    # Kupiec test (unconditional coverage)
    if n_violations > 0 and n_violations < n_obs:
        likelihood_ratio = -2 * (
            np.log((1 - expected_rate)**(n_obs - n_violations) * expected_rate**n_violations) -
            np.log((1 - violation_rate)**(n_obs - n_violations) * violation_rate**n_violations)
        )
        kupiec_pvalue = 1 - stats.chi2.cdf(likelihood_ratio, df=1)
    else:
        likelihood_ratio = np.nan
        kupiec_pvalue = np.nan
    
    # Christoffersen test (independence) - check for clustering
    # Count transitions: 00, 01, 10, 11
    violations_array = violations.values
    n_00 = np.sum((violations_array[:-1] == 0) & (violations_array[1:] == 0))
    n_01 = np.sum((violations_array[:-1] == 0) & (violations_array[1:] == 1))
    n_10 = np.sum((violations_array[:-1] == 1) & (violations_array[1:] == 0))
    n_11 = np.sum((violations_array[:-1] == 1) & (violations_array[1:] == 1))
    
    # Transition probabilities
    if (n_00 + n_01) > 0 and (n_10 + n_11) > 0:
        p_01 = n_01 / (n_00 + n_01)
        p_11 = n_11 / (n_10 + n_11)
        p = (n_01 + n_11) / (n_obs - 1)
        
        if p_01 > 0 and p_01 < 1 and p_11 > 0 and p_11 < 1 and p > 0 and p < 1:
            lr_ind = -2 * (
                (n_00 + n_10) * np.log(1 - p) + (n_01 + n_11) * np.log(p) -
                n_00 * np.log(1 - p_01) - n_01 * np.log(p_01) -
                n_10 * np.log(1 - p_11) - n_11 * np.log(p_11)
            )
            christoffersen_pvalue = 1 - stats.chi2.cdf(lr_ind, df=1)
        else:
            lr_ind = np.nan
            christoffersen_pvalue = np.nan
    else:
        lr_ind = np.nan
        christoffersen_pvalue = np.nan
    
    results = {
        'n_violations': n_violations,
        'n_observations': n_obs,
        'violation_rate': violation_rate,
        'expected_rate': expected_rate,
        'kupiec_lr': likelihood_ratio,
        'kupiec_pvalue': kupiec_pvalue,
        'kupiec_reject': kupiec_pvalue < 0.05 if not np.isnan(kupiec_pvalue) else None,
        'christoffersen_lr': lr_ind,
        'christoffersen_pvalue': christoffersen_pvalue,
        'christoffersen_reject': christoffersen_pvalue < 0.05 if not np.isnan(christoffersen_pvalue) else None,
    }
    
    return results


def monte_carlo_var(
    returns: pd.Series,
    n_simulations: int = 10000,
    horizon: int = 1,
    confidence: float = 0.95,
    method: str = 'bootstrap'
) -> Tuple[float, np.ndarray]:
    """
    Calculate VaR using Monte Carlo simulation.
    
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    n_simulations : int
        Number of Monte Carlo simulations
    horizon : int
        Forecast horizon in days
    confidence : float
        Confidence level
    method : str
        'bootstrap' or 'parametric'
    
    Returns
    -------
    tuple
        (VaR, simulated_returns)
    """
    returns_clean = returns.dropna().values
    
    if method == 'bootstrap':
        # Bootstrap resampling
        simulated = np.random.choice(returns_clean, size=(n_simulations, horizon), replace=True)
    
    elif method == 'parametric':
        # Parametric (normal) simulation
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        simulated = np.random.normal(mu, sigma, size=(n_simulations, horizon))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Cumulative returns for horizon
    cumulative_returns = np.sum(simulated, axis=1)
    
    # VaR
    var = -np.percentile(cumulative_returns, (1 - confidence) * 100)
    
    return var, cumulative_returns


