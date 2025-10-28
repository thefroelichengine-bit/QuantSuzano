"""
Scraper for Brazilian macro data from IBGE.

IBGE API provides:
- IPCA (inflation)
- GDP
- Industrial production
- Retail sales
"""

import pandas as pd
import requests
from datetime import datetime
from .base import BaseScraper
from .utils import requests_get_with_retry


class IBGEMacroScraper(BaseScraper):
    """Scraper for IBGE macroeconomic data."""
    
    BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"
    
    # Series codes
    SERIES_MAP = {
        "ipca": 1737,  # IPCA (inflation index)
        "gdp": 1620,   # GDP
        "industrial_production": 3653,  # Industrial production
        "retail_sales": 3416,  # Retail sales
    }
    
    def __init__(self, **kwargs):
        """Initialize IBGE scraper."""
        super().__init__(
            name="ibge_macro",
            cache_ttl_hours=24,
            **kwargs
        )
    
    def _fetch_data(
        self,
        start_date: str,
        end_date: str,
        series: str = "ipca",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch macro data from IBGE.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        series : str
            Series name ('ipca', 'gdp', etc.)
        
        Returns
        -------
        pd.DataFrame
            Macro data with date index
        """
        if series not in self.SERIES_MAP:
            raise ValueError(f"Unknown series: {series}. Available: {list(self.SERIES_MAP.keys())}")
        
        series_code = self.SERIES_MAP[series]
        
        print(f"[IBGE] Fetching {series} (code: {series_code})")
        
        # IBGE API endpoint
        url = f"{self.BASE_URL}/{series_code}/periodos/all/variaveis/63"
        
        response = requests_get_with_retry(url)
        data = response.json()
        
        if not data:
            raise ValueError(f"No data returned for series {series}")
        
        # Parse IBGE JSON structure
        df = self._parse_ibge_response(data, series)
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        return df
    
    def _parse_ibge_response(self, data: list, series_name: str) -> pd.DataFrame:
        """
        Parse IBGE API response.
        
        IBGE has complex nested JSON structure.
        
        Parameters
        ----------
        data : list
            Raw API response
        series_name : str
            Series name for column
        
        Returns
        -------
        pd.DataFrame
            Parsed data
        """
        records = []
        
        for item in data:
            if 'resultados' in item:
                for result in item['resultados']:
                    if 'series' in result:
                        for series_item in result['series']:
                            if 'serie' in series_item:
                                serie_data = series_item['serie']
                                
                                for period, value in serie_data.items():
                                    try:
                                        # Parse period (format: YYYYMM)
                                        if len(period) == 6:
                                            year = int(period[:4])
                                            month = int(period[4:6])
                                            date = pd.Timestamp(year=year, month=month, day=1)
                                        else:
                                            continue
                                        
                                        # Parse value
                                        val = float(value) if value != '-' else None
                                        
                                        records.append({
                                            'date': date,
                                            series_name: val
                                        })
                                    except:
                                        continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df.set_index('date').sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def fetch_ipca(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch IPCA (inflation)."""
        return self.fetch(start_date, end_date, series="ipca", **kwargs)
    
    def fetch_gdp(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Convenience method to fetch GDP."""
        return self.fetch(start_date, end_date, series="gdp", **kwargs)
    
    def fetch_all_macro(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Fetch all macro series and merge.
        
        Returns
        -------
        pd.DataFrame
            Merged macro data
        """
        dfs = []
        
        for series_name in self.SERIES_MAP.keys():
            try:
                df = self.fetch(start_date, end_date, series=series_name, **kwargs)
                dfs.append(df)
            except Exception as e:
                print(f"[IBGE] Failed to fetch {series_name}: {e}")
        
        if not dfs:
            raise ValueError("Failed to fetch any IBGE series")
        
        # Merge all series
        result = pd.concat(dfs, axis=1)
        return result


