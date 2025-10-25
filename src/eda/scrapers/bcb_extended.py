"""
Robust scraper for Brazilian Central Bank (BCB) data.

Fetches:
- PTAX (USD/BRL exchange rate) - Series 1
- SELIC (interest rate) - Series 432
"""

import pandas as pd
import requests
from datetime import datetime
from .base import BaseScraper
from .utils import requests_get_with_retry


class BCBExtendedScraper(BaseScraper):
    """Scraper for BCB SGS API (Sistema Gerenciador de Séries Temporais)."""
    
    BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados"
    
    SERIES_MAP = {
        "ptax": 1,      # USD/BRL exchange rate
        "selic": 432,   # SELIC interest rate
    }
    
    def __init__(self, **kwargs):
        """Initialize BCB scraper."""
        super().__init__(name="bcb_extended", cache_ttl_hours=6, **kwargs)
    
    def _fetch_data(
        self,
        start_date: str,
        end_date: str,
        series: str = "ptax",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from BCB API.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        series : str
            Series name ('ptax' or 'selic')
        
        Returns
        -------
        pd.DataFrame
            Data with date index and value column
        """
        if series not in self.SERIES_MAP:
            raise ValueError(f"Unknown series: {series}. Available: {list(self.SERIES_MAP.keys())}")
        
        series_code = self.SERIES_MAP[series]
        
        # Convert dates to DD/MM/YYYY format for BCB API
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        start_bcb = start_dt.strftime("%d/%m/%Y")
        end_bcb = end_dt.strftime("%d/%m/%Y")
        
        url = self.BASE_URL.format(series=series_code)
        params = {
            "formato": "json",
            "dataInicial": start_bcb,
            "dataFinal": end_bcb,
        }
        
        print(f"[BCB] Fetching series {series} ({series_code})")
        
        response = requests_get_with_retry(url, params=params)
        data = response.json()
        
        if not data:
            raise ValueError(f"No data returned for series {series}")
        
        # Parse JSON to DataFrame
        df = pd.DataFrame(data)
        
        # Convert date format DD/MM/YYYY to datetime
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        
        # Set index and rename column
        df = df.set_index("data")
        df = df.rename(columns={"valor": series})
        
        # Sort by date
        df = df.sort_index()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def fetch_ptax(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch PTAX."""
        return self.fetch(start_date, end_date, series="ptax", **kwargs)
    
    def fetch_selic(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch SELIC."""
        return self.fetch(start_date, end_date, series="selic", **kwargs)
    
    def fetch_all(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Fetch all BCB series and merge.
        
        Returns
        -------
        pd.DataFrame
            Merged data with all series
        """
        dfs = []
        
        for series_name in self.SERIES_MAP.keys():
            try:
                df = self.fetch(start_date, end_date, series=series_name, **kwargs)
                dfs.append(df)
            except Exception as e:
                print(f"[BCB] Failed to fetch {series_name}: {e}")
        
        if not dfs:
            raise ValueError("Failed to fetch any BCB series")
        
        # Merge all series
        result = pd.concat(dfs, axis=1)
        return result

