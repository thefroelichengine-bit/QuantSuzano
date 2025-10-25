"""
Robust scraper for Yahoo Finance data.

Fetches equity prices with automatic retry and fallback.
"""

import pandas as pd
import yfinance as yf
from .base import BaseScraper


class YFinanceRobustScraper(BaseScraper):
    """Scraper for Yahoo Finance equity data."""
    
    def __init__(self, **kwargs):
        """Initialize Yahoo Finance scraper."""
        super().__init__(
            name="yfinance",
            cache_ttl_hours=1,  # Short TTL for market data
            retry_attempts=5,
            retry_delay=2.0,
            rate_limit_seconds=1.0,  # Respect rate limits
            **kwargs
        )
    
    def _fetch_data(
        self,
        start_date: str,
        end_date: str,
        ticker: str = "SUZB3.SA",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch equity price data from Yahoo Finance.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        ticker : str
            Stock ticker symbol
        
        Returns
        -------
        pd.DataFrame
            OHLCV data with date index
        """
        print(f"[YFINANCE] Fetching {ticker}")
        
        # Use yfinance download (more robust than Ticker)
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
        )
        
        if df is None or len(df) == 0:
            raise ValueError(f"No data returned for {ticker}")
        
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Select only Close price (or specify others if needed)
        if "close" not in df.columns:
            raise ValueError("Close price not found in data")
        
        # Keep all OHLCV data
        result = df[["open", "high", "low", "close", "volume"]].copy()
        
        # Ensure index is datetime
        result.index = pd.to_datetime(result.index)
        
        # Sort by date
        result = result.sort_index()
        
        return result
    
    def fetch_equity(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        column: str = "close",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch single equity and return specific column.
        
        Parameters
        ----------
        ticker : str
            Stock ticker
        start_date : str
            Start date
        end_date : str
            End date
        column : str
            Column to return ('close', 'open', 'high', 'low', 'volume')
        
        Returns
        -------
        pd.DataFrame
            Single column DataFrame
        """
        df = self.fetch(start_date, end_date, ticker=ticker, **kwargs)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
        
        # Extract ticker symbol base (remove exchange suffix)
        ticker_base = ticker.split(".")[0].lower()
        
        return df[[column]].rename(columns={column: ticker_base})
    
    def fetch_multiple(
        self,
        tickers: list,
        start_date: str,
        end_date: str,
        column: str = "close",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch multiple equities.
        
        Parameters
        ----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date
        end_date : str
            End date
        column : str
            Column to return
        
        Returns
        -------
        pd.DataFrame
            Merged data for all tickers
        """
        dfs = []
        
        for ticker in tickers:
            try:
                df = self.fetch_equity(ticker, start_date, end_date, column, **kwargs)
                dfs.append(df)
            except Exception as e:
                print(f"[YFINANCE] Failed to fetch {ticker}: {e}")
        
        if not dfs:
            raise ValueError("Failed to fetch any tickers")
        
        # Merge all tickers
        result = pd.concat(dfs, axis=1)
        return result

