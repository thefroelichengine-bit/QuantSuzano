"""Utility functions for scrapers."""

import time
from functools import wraps
from typing import Callable
import requests
from datetime import datetime


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Parameters
        ----------
        calls_per_minute : int
            Maximum calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


def rate_limited(calls_per_minute: int = 60):
    """
    Decorator to rate limit function calls.
    
    Parameters
    ----------
    calls_per_minute : int
        Maximum calls per minute
    """
    limiter = RateLimiter(calls_per_minute)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requests_get_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 30,
    **kwargs
) -> requests.Response:
    """
    Make HTTP GET request with retry logic.
    
    Parameters
    ----------
    url : str
        URL to fetch
    max_retries : int
        Maximum retry attempts
    timeout : int
        Request timeout in seconds
    **kwargs
        Additional arguments for requests.get
    
    Returns
    -------
    requests.Response
        Response object
    
    Raises
    ------
    requests.RequestException
        If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        
        except requests.RequestException as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff
                print(f"[HTTP] Retry {attempt + 1}/{max_retries} after {delay}s...")
                time.sleep(delay)
    
    raise last_exception


def validate_date_range(start_date: str, end_date: str) -> tuple:
    """
    Validate and parse date range.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns
    -------
    tuple
        (start_datetime, end_datetime)
    
    Raises
    ------
    ValueError
        If dates are invalid
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format (use YYYY-MM-DD): {e}")
    
    if start > end:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")
    
    if end > datetime.now():
        raise ValueError(f"End date {end_date} is in the future")
    
    return start, end


def normalize_column_names(df, column_mapping: dict = None):
    """
    Normalize DataFrame column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize
    column_mapping : dict, optional
        Custom column name mapping
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized column names
    """
    df = df.copy()
    
    # Apply custom mapping if provided
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Normalize: lowercase, replace spaces with underscores
    df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    
    return df


def handle_missing_dates(
    df,
    freq: str = "D",
    method: str = "ffill",
):
    """
    Handle missing dates in time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    freq : str
        Frequency string ('D', 'B', 'M', etc.)
    method : str
        Fill method ('ffill', 'bfill', 'interpolate', None)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with complete date range
    """
    df = df.copy()
    
    # Reindex to complete date range
    df = df.asfreq(freq)
    
    # Fill missing values
    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "interpolate":
        df = df.interpolate(method="time")
    
    return df

