"""Volatility analysis and modeling."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from arch import arch_model
from scipy import stats


def calculate_historical_volatility(
    returns: pd.Series,
    window: int = 21,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate rolling historical volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size (default: 21 days)
    annualization_factor : int
        Days per year for annualization (default: 252)
    
    Returns
    -------
    pd.Series
        Annualized rolling volatility
    """
    vol = returns.rolling(window=window).std() * np.sqrt(annualization_factor)
    return vol


def calculate_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate Parkinson volatility (uses high/low prices).
    
    More efficient than close-to-close volatility.
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    window : int
        Rolling window
    annualization_factor : int
        Annualization factor
    
    Returns
    -------
    pd.Series
        Parkinson volatility
    """
    hl_ratio = np.log(high / low)
    parkinson = hl_ratio.rolling(window=window).apply(
        lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2)))
    )
    return parkinson * np.sqrt(annualization_factor)


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal'
) -> Tuple:
    """
    Fit GARCH(p,q) model to returns.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    p : int
        GARCH lag order
    q : int
        ARCH lag order
    dist : str
        Error distribution ('normal', 't', 'skewt')
    
    Returns
    -------
    tuple
        (model_result, conditional_volatility, forecast)
    """
    # Remove NaN
    returns_clean = returns.dropna() * 100  # Convert to percentage
    
    # Fit GARCH model
    model = arch_model(
        returns_clean,
        vol='Garch',
        p=p,
        q=q,
        dist=dist
    )
    
    result = model.fit(disp='off')
    
    # Conditional volatility
    conditional_vol = result.conditional_volatility / 100  # Convert back
    
    # Forecast
    forecast = result.forecast(horizon=5)
    forecast_vol = np.sqrt(forecast.variance.values[-1, :]) / 100
    
    return result, conditional_vol, forecast_vol


def detect_volatility_regimes(
    volatility: pd.Series,
    n_regimes: int = 3
) -> pd.Series:
    """
    Detect volatility regimes using quantile-based classification.
    
    Parameters
    ----------
    volatility : pd.Series
        Volatility series
    n_regimes : int
        Number of regimes (default: 3 for low/medium/high)
    
    Returns
    -------
    pd.Series
        Regime labels (0 = low, 1 = medium, 2 = high)
    """
    vol_clean = volatility.dropna()
    
    # Define quantile-based regimes
    quantiles = np.linspace(0, 1, n_regimes + 1)
    thresholds = vol_clean.quantile(quantiles)
    
    # Classify
    regimes = pd.cut(
        volatility,
        bins=thresholds,
        labels=range(n_regimes),
        include_lowest=True
    )
    
    return regimes.astype(float)


def calculate_volatility_cone(
    returns: pd.Series,
    windows: list = [5, 10, 21, 63, 126, 252],
    percentiles: list = [10, 25, 50, 75, 90],
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Calculate volatility cone showing realized volatility at different horizons.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    windows : list
        Window sizes to calculate volatility
    percentiles : list
        Percentiles to show
    annualization_factor : int
        Annualization factor
    
    Returns
    -------
    pd.DataFrame
        Volatility cone data
    """
    cone_data = []
    
    for window in windows:
        vols = returns.rolling(window=window).std() * np.sqrt(annualization_factor)
        vols_clean = vols.dropna()
        
        cone_row = {'window': window}
        for pct in percentiles:
            cone_row[f'p{pct}'] = np.percentile(vols_clean, pct)
        
        cone_data.append(cone_row)
    
    return pd.DataFrame(cone_data)


def calculate_realized_volatility_stats(
    returns: pd.Series,
    window: int = 21,
    annualization_factor: int = 252
) -> dict:
    """
    Calculate comprehensive volatility statistics.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window
    annualization_factor : int
        Annualization factor
    
    Returns
    -------
    dict
        Volatility statistics
    """
    vol = calculate_historical_volatility(returns, window, annualization_factor)
    vol_clean = vol.dropna()
    
    stats_dict = {
        'current': vol.iloc[-1] if len(vol) > 0 else np.nan,
        'mean': vol_clean.mean(),
        'median': vol_clean.median(),
        'std': vol_clean.std(),
        'min': vol_clean.min(),
        'max': vol_clean.max(),
        'p10': vol_clean.quantile(0.10),
        'p90': vol_clean.quantile(0.90),
    }
    
    return stats_dict


