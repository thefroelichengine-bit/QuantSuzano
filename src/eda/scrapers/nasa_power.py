"""
Scraper for NASA POWER API (Prediction Of Worldwide Energy Resources).

Provides global climate data including temperature, precipitation, and solar radiation.
API Documentation: https://power.larc.nasa.gov/docs/services/api/
"""

import pandas as pd
import requests
from .base import BaseScraper
from .utils import requests_get_with_retry


class NASAPowerScraper(BaseScraper):
    """
    Scraper for NASA POWER climate data.
    
    Provides gridded climate data for any location worldwide.
    """
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Coordinates for major Suzano mill locations
    LOCATIONS = {
        "tres_lagoas": {"lat": -20.75, "lon": -51.68},  # Três Lagoas, MS
        "imperatriz": {"lat": -5.53, "lon": -47.48},    # Imperatriz, MA
        "suzano": {"lat": -23.55, "lon": -46.31},       # Suzano, SP
        "aracruz": {"lat": -19.82, "lon": -40.27},      # Aracruz, ES
    }
    
    # Available parameters (NASA POWER codes)
    PARAMETERS = [
        "T2M",          # Temperature at 2 meters (°C)
        "T2M_MAX",      # Maximum temperature (°C)
        "T2M_MIN",      # Minimum temperature (°C)
        "PRECTOTCORR",  # Precipitation (mm/day)
        "RH2M",         # Relative humidity at 2m (%)
        "WS2M",         # Wind speed at 2m (m/s)
        "ALLSKY_SFC_SW_DWN",  # Solar radiation (MJ/m²/day)
    ]
    
    def __init__(self, **kwargs):
        """Initialize NASA POWER scraper."""
        super().__init__(
            name="nasa_power",
            cache_ttl_hours=168,  # 1 week (data rarely changes)
            retry_attempts=3,
            rate_limit_seconds=0.5,
            **kwargs
        )
    
    def _fetch_data(
        self,
        start_date: str,
        end_date: str,
        location: str = "tres_lagoas",
        parameters: list = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch climate data from NASA POWER API.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        location : str
            Location name (key from LOCATIONS dict)
        parameters : list, optional
            List of parameter codes (default: all)
        
        Returns
        -------
        pd.DataFrame
            Climate data with date index
        """
        if location not in self.LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(self.LOCATIONS.keys())}")
        
        coords = self.LOCATIONS[location]
        
        if parameters is None:
            parameters = self.PARAMETERS
        
        # Format dates for API (YYYYMMDD)
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        
        # Build request parameters
        params = {
            "parameters": ",".join(parameters),
            "community": "AG",  # Agriculture community
            "longitude": coords["lon"],
            "latitude": coords["lat"],
            "start": start_fmt,
            "end": end_fmt,
            "format": "JSON",
        }
        
        print(f"[NASA] Fetching location {location} ({coords['lat']}, {coords['lon']})")
        
        response = requests_get_with_retry(self.BASE_URL, params=params, timeout=60)
        data = response.json()
        
        # Parse response
        if "properties" not in data or "parameter" not in data["properties"]:
            raise ValueError("Unexpected NASA POWER API response format")
        
        param_data = data["properties"]["parameter"]
        
        # Convert to DataFrame
        df = pd.DataFrame(param_data)
        
        # Transpose so dates are rows
        df = df.T
        
        # Convert index to datetime (handle potential format issues)
        try:
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
        except:
            # Try without format if format fails
            df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Rename columns to more readable names
        column_map = {
            "T2M": "temp_mean",
            "T2M_MAX": "temp_max",
            "T2M_MIN": "temp_min",
            "PRECTOTCORR": "precipitation",
            "RH2M": "humidity",
            "WS2M": "wind_speed",
            "ALLSKY_SFC_SW_DWN": "solar_radiation",
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Sort by date
        df = df.sort_index()
        
        # Drop rows with all NaN
        df = df.dropna(how="all")
        
        return df
    
    def fetch_aggregated(
        self,
        start_date: str,
        end_date: str,
        locations: list = None,
        agg_func: str = "mean",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch and aggregate data from multiple locations.
        
        Parameters
        ----------
        start_date : str
            Start date
        end_date : str
            End date
        locations : list, optional
            List of location names (default: all)
        agg_func : str
            Aggregation function ('mean', 'max', 'min')
        
        Returns
        -------
        pd.DataFrame
            Aggregated climate data
        """
        if locations is None:
            locations = list(self.LOCATIONS.keys())
        
        dfs = []
        
        for location in locations:
            try:
                df = self.fetch(start_date, end_date, location=location, **kwargs)
                dfs.append(df)
            except Exception as e:
                print(f"[NASA] Failed to fetch {location}: {e}")
        
        if not dfs:
            raise ValueError("Failed to fetch any locations")
        
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

