"""Classical time series forecasting models (ARIMA, SARIMA, etc.)."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Check if series is stationary using ADF test.
    
    Parameters
    ----------
    series : pd.Series
        Time series
    significance : float
        Significance level for test
    
    Returns
    -------
    dict
        Test results
    """
    result = adfuller(series.dropna())
    
    return {
        'adf_statistic': result[0],
        'pvalue': result[1],
        'is_stationary': result[1] < significance,
        'critical_values': result[4],
    }


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    exog: Optional[pd.DataFrame] = None
) -> Tuple:
    """
    Fit ARIMA or SARIMAX model.
    
    Parameters
    ----------
    series : pd.Series
        Time series to model
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple, optional
        Seasonal ARIMA order (P, D, Q, s)
    exog : pd.DataFrame, optional
        Exogenous variables
    
    Returns
    -------
    tuple
        (model_result, fitted_values, residuals)
    """
    series_clean = series.dropna()
    
    if seasonal_order is not None:
        # SARIMAX
        model = SARIMAX(
            series_clean,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog
        )
    else:
        # ARIMA
        model = ARIMA(
            series_clean,
            order=order,
            exog=exog
        )
    
    result = model.fit()
    
    fitted = result.fittedvalues
    residuals = result.resid
    
    return result, fitted, residuals


def auto_arima_select(
    series: pd.Series,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    seasonal: bool = False,
    m: int = 12,
    information_criterion: str = 'aic'
) -> dict:
    """
    Automatic ARIMA model selection using grid search.
    
    Parameters
    ----------
    series : pd.Series
        Time series
    max_p : int
        Maximum AR order
    max_q : int
        Maximum MA order
    max_d : int
        Maximum differencing order
    seasonal : bool
        Include seasonal components
    m : int
        Seasonal period
    information_criterion : str
        'aic' or 'bic'
    
    Returns
    -------
    dict
        Best model parameters and fit results
    """
    series_clean = series.dropna()
    
    best_score = np.inf
    best_order = None
    best_seasonal_order = None
    best_result = None
    
    # Grid search over orders
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    if seasonal:
                        # Try seasonal models
                        for P in range(2):
                            for D in range(2):
                                for Q in range(2):
                                    try:
                                        model = SARIMAX(
                                            series_clean,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, m)
                                        )
                                        result = model.fit(disp=False)
                                        
                                        score = result.aic if information_criterion == 'aic' else result.bic
                                        
                                        if score < best_score:
                                            best_score = score
                                            best_order = (p, d, q)
                                            best_seasonal_order = (P, D, Q, m)
                                            best_result = result
                                    except:
                                        continue
                    else:
                        # Non-seasonal
                        model = ARIMA(series_clean, order=(p, d, q))
                        result = model.fit()
                        
                        score = result.aic if information_criterion == 'aic' else result.bic
                        
                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                            best_seasonal_order = None
                            best_result = result
                
                except:
                    continue
    
    if best_result is None:
        raise ValueError("No valid ARIMA model found")
    
    return {
        'order': best_order,
        'seasonal_order': best_seasonal_order,
        'aic': best_result.aic,
        'bic': best_result.bic,
        'result': best_result,
    }


def forecast_arima(
    result,
    horizon: int = 30,
    exog_future: Optional[pd.DataFrame] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Generate forecasts from fitted ARIMA model.
    
    Parameters
    ----------
    result
        Fitted ARIMA/SARIMAX result
    horizon : int
        Forecast horizon
    exog_future : pd.DataFrame, optional
        Future exogenous variables
    alpha : float
        Significance level for confidence intervals
    
    Returns
    -------
    pd.DataFrame
        Forecast with confidence intervals
    """
    forecast_result = result.get_forecast(steps=horizon, exog=exog_future)
    
    forecast_df = pd.DataFrame({
        'forecast': forecast_result.predicted_mean,
        'lower': forecast_result.conf_int(alpha=alpha).iloc[:, 0],
        'upper': forecast_result.conf_int(alpha=alpha).iloc[:, 1],
    })
    
    return forecast_df


def rolling_forecast(
    series: pd.Series,
    order: Tuple[int, int, int],
    window: int = 252,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Perform rolling window forecast (walk-forward analysis).
    
    Parameters
    ----------
    series : pd.Series
        Time series
    order : tuple
        ARIMA order
    window : int
        Training window size
    horizon : int
        Forecast horizon
    
    Returns
    -------
    pd.DataFrame
        Out-of-sample forecasts
    """
    series_clean = series.dropna()
    n = len(series_clean)
    
    forecasts = []
    actuals = []
    dates = []
    
    for i in range(window, n - horizon + 1):
        # Training data
        train = series_clean.iloc[:i]
        
        # Fit model
        try:
            model = ARIMA(train, order=order)
            result = model.fit()
            
            # Forecast
            fc = result.forecast(steps=horizon)
            
            # Store
            forecasts.append(fc.iloc[horizon-1])
            actuals.append(series_clean.iloc[i + horizon - 1])
            dates.append(series_clean.index[i + horizon - 1])
        
        except:
            continue
    
    df = pd.DataFrame({
        'date': dates,
        'forecast': forecasts,
        'actual': actuals,
    })
    df = df.set_index('date')
    df['error'] = df['actual'] - df['forecast']
    df['abs_error'] = df['error'].abs()
    df['squared_error'] = df['error'] ** 2
    
    return df


def calculate_forecast_metrics(forecast_df: pd.DataFrame) -> dict:
    """
    Calculate forecast accuracy metrics.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with 'forecast' and 'actual' columns
    
    Returns
    -------
    dict
        Accuracy metrics
    """
    errors = forecast_df['actual'] - forecast_df['forecast']
    
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())
    mape = (errors.abs() / forecast_df['actual'].abs()).replace([np.inf, -np.inf], np.nan).mean() * 100
    
    # Directional accuracy
    if len(errors) > 1:
        actual_direction = (forecast_df['actual'].diff() > 0).astype(int)
        forecast_direction = (forecast_df['forecast'].diff() > 0).astype(int)
        directional_accuracy = (actual_direction == forecast_direction).mean()
    else:
        directional_accuracy = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
    }


