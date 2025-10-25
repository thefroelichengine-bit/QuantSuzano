"""Base scraper class with retry logic, caching, and error handling."""

import time
import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..config import ROOT


class BaseScraper(ABC):
    """
    Abstract base class for all data scrapers.
    
    Provides:
    - Retry logic with exponential backoff
    - Response caching
    - Rate limiting
    - Error handling
    - Logging
    """
    
    def __init__(
        self,
        name: str,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_seconds: float = 0.5,
    ):
        """
        Initialize scraper.
        
        Parameters
        ----------
        name : str
            Scraper name
        cache_dir : Path, optional
            Cache directory
        cache_ttl_hours : int
            Cache time-to-live in hours
        retry_attempts : int
            Number of retry attempts
        retry_delay : float
            Initial retry delay in seconds
        rate_limit_seconds : float
            Minimum seconds between requests
        """
        self.name = name
        self.cache_dir = cache_dir or ROOT / "data" / "cache" / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit = rate_limit_seconds
        
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
    
    def _wait_for_rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_string = f"{self.name}_" + "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if valid.
        
        Returns None if cache miss or expired.
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > self.cache_ttl:
            print(f"[{self.name}] Cache expired (age: {cache_age})")
            return None
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"[{self.name}] Cache hit (age: {cache_age})")
            return data
        except Exception as e:
            print(f"[{self.name}] Cache read error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[{self.name}] Data cached")
        except Exception as e:
            print(f"[{self.name}] Cache write error: {e}")
    
    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Execute function with retry logic and exponential backoff.
        
        Parameters
        ----------
        func : callable
            Function to execute
        *args, **kwargs
            Arguments to pass to function
        
        Returns
        -------
        Any
            Function result
        
        Raises
        ------
        Exception
            If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                self._wait_for_rate_limit()
                result = func(*args, **kwargs)
                self.request_count += 1
                return result
            
            except Exception as e:
                last_exception = e
                self.error_count += 1
                
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[{self.name}] Attempt {attempt + 1} failed: {e}")
                    print(f"[{self.name}] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"[{self.name}] All {self.retry_attempts} attempts failed")
        
        raise last_exception
    
    @abstractmethod
    def _fetch_data(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Fetch data from source (must be implemented by subclass).
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        **kwargs
            Additional parameters
        
        Returns
        -------
        pd.DataFrame
            Fetched data indexed by date
        """
        pass
    
    def fetch(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data with caching and retry logic.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        use_cache : bool
            Whether to use cache
        **kwargs
            Additional parameters for scraper
        
        Returns
        -------
        pd.DataFrame
            Fetched data
        """
        print(f"\n[{self.name}] Fetching data: {start_date} to {end_date}")
        
        # Check cache
        cache_key = self._get_cache_key(start=start_date, end=end_date, **kwargs)
        
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch with retry
        try:
            data = self._retry_with_backoff(
                self._fetch_data,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
            
            # Validate
            if data is None or len(data) == 0:
                raise ValueError("No data returned")
            
            # Cache
            if use_cache:
                self._save_to_cache(cache_key, data)
            
            print(f"[{self.name}] Successfully fetched {len(data)} rows")
            return data
        
        except Exception as e:
            print(f"[{self.name}] Fetch failed: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get scraper statistics."""
        return {
            "name": self.name,
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
        }
    
    def clear_cache(self):
        """Clear all cached data for this scraper."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{self.name}] Cache cleared")

