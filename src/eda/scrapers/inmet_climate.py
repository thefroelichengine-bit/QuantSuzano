"""
Scraper for INMET (Brazilian National Institute of Meteorology) climate data.

INMET provides free historical weather data via API.
Documentation: https://portal.inmet.gov.br/
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from .base import BaseScraper
from .utils import requests_get_with_retry


class INMETClimateScraper(BaseScraper):
    """
    Scraper for INMET climate data.
    
    Note: INMET API requires station codes. We'll use major pulp production regions.
    """
    
    BASE_URL = "https://apitempo.inmet.gov.br/estacao"
    
    # Station codes for major pulp-producing regions in Brazil
    STATIONS = {
        "tres_lagoas_ms": "A742",  # Três Lagoas, MS (Suzano's largest mill)
        "imperatriz_ma": "A201",    # Imperatriz, MA
        "suzano_sp": "A701",        # Suzano, SP (headquarters region)
    }
    
    def __init__(self, **kwargs):
        """Initialize INMET scraper."""
        super().__init__(
            name="inmet",
            cache_ttl_hours=24,
            retry_attempts=3,
            rate_limit_seconds=1.0,
            **kwargs
        )
    
    def _fetch_data(
        self,
        start_date: str,
        end_date: str,
        station: str = "tres_lagoas_ms",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch climate data from INMET API.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        station : str
            Station name (key from STATIONS dict)
        
        Returns
        -------
        pd.DataFrame
            Climate data with date index
        """
        if station not in self.STATIONS:
            raise ValueError(f"Unknown station: {station}. Available: {list(self.STATIONS.keys())}")
        
        station_code = self.STATIONS[station]
        
        print(f"[INMET] Fetching station {station} ({station_code})")
        
        # INMET API format: YYYY-MM-DD
        url = f"{self.BASE_URL}/{station_code}/{start_date}/{end_date}"
        
        response = requests_get_with_retry(url)
        data = response.json()
        
        if not data:
            raise ValueError(f"No data returned for station {station}")
        
        # Parse JSON to DataFrame
        df = pd.DataFrame(data)
        
        # Expected columns vary, but typically include:
        # DT_MEDICAO (datetime), TEMP_MAX, TEMP_MIN, TEMP_MED, CHUVA (precipitation), etc.
        
        # Rename and standardize columns
        column_map = {
            "DT_MEDICAO": "date",
            "TEMP_MAX": "temp_max",
            "TEMP_MIN": "temp_min",
            "TEMP_MED": "temp_mean",
            "CHUVA": "precipitation",
            "UMID_MED": "humidity",
            "VEN_VEL": "wind_speed",
        }
        
        df = df.rename(columns=column_map)
        
        # Convert date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        else:
            raise ValueError("Date column not found in INMET response")
        
        # Select relevant columns
        available_cols = [col for col in column_map.values() if col in df.columns and col != "date"]
        df = df[available_cols]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Sort by date
        df = df.sort_index()
        
        # Drop rows with all NaN
        df = df.dropna(how="all")
        
        return df
    
    def fetch_aggregated(
        self,
        start_date: str,
        end_date: str,
        stations: list = None,
        agg_func: str = "mean",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch and aggregate data from multiple stations.
        
        Parameters
        ----------
        start_date : str
            Start date
        end_date : str
            End date
        stations : list, optional
            List of station names (default: all stations)
        agg_func : str
            Aggregation function ('mean', 'max', 'min')
        
        Returns
        -------
        pd.DataFrame
            Aggregated climate data
        """
        if stations is None:
            stations = list(self.STATIONS.keys())
        
        dfs = []
        
        for station in stations:
            try:
                df = self.fetch(start_date, end_date, station=station, **kwargs)
                dfs.append(df)
            except Exception as e:
                print(f"[INMET] Failed to fetch {station}: {e}")
        
        if not dfs:
            raise ValueError("Failed to fetch any stations")
        
        # Concatenate and aggregate
        combined = pd.concat(dfs, axis=0)
        
        if agg_func == "mean":
            result = combined.groupby(combined.index).mean()
        elif agg_func == "max":
            result = combined.groupby(combined.index).max()
        elif agg_func == "min":
            result = combined.groupby(combined.index).min()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
        
        return result

