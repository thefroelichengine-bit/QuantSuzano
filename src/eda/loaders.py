"""Data loaders with real API integrations and CSV fallbacks."""

import warnings
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from .config import DATA_RAW, DEFAULT_START_DATE, EQUITY_TICKER, SGS_PTAX, SGS_SELIC

warnings.filterwarnings("ignore", category=FutureWarning)


def _read_csv_fallback(name: str, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    """Read CSV fallback with standardized format."""
    csv_path = DATA_RAW / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV fallback not found: {csv_path}. "
            f"Please create the file or ensure API connection is working."
        )
    
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Standardize column name
    if value_col in df.columns:
        df = df.rename(columns={value_col: name})
    
    return df


def load_equity(
    ticker: str = EQUITY_TICKER,
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load equity data from Yahoo Finance (yfinance).
    
    Falls back to CSV if API fails.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    start : str
        Start date (YYYY-MM-DD)
    end : str, optional
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'suzb' column (close price) indexed by date
    """
    print(f"[LOAD] Loading equity data for {ticker}...")
    
    try:
        # Fetch from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Extract close price and rename
        df = df[["Close"]].rename(columns={"Close": "suzb"})
        df.index.name = "date"
        
        # Strip timezone to avoid merge issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"[OK] Loaded {len(df)} rows from Yahoo Finance")
        return df
    
    except Exception as e:
        print(f"[WARN] Yahoo Finance failed: {e}")
        print("[FALLBACK] Attempting CSV...")
        return _read_csv_fallback("suzb", value_col="close")


def load_ptax_sgs(start: str = DEFAULT_START_DATE, end: Optional[str] = None) -> pd.DataFrame:
    """
    Load PTAX (USD/BRL exchange rate) from BCB SGS API.
    
    Falls back to CSV if API fails.
    
    Parameters
    ----------
    start : str
        Start date (DD/MM/YYYY)
    end : str, optional
        End date (DD/MM/YYYY), defaults to today
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'ptax' column indexed by date
    """
    print("[LOAD] Loading PTAX from BCB SGS API...")
    
    try:
        # Convert date format for BCB API (DD/MM/YYYY)
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        start_bcb = start_dt.strftime("%d/%m/%Y")
        
        if end:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            end_bcb = end_dt.strftime("%d/%m/%Y")
        else:
            end_bcb = datetime.now().strftime("%d/%m/%Y")
        
        # BCB SGS API endpoint
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{SGS_PTAX}/dados"
            f"?formato=json&dataInicial={start_bcb}&dataFinal={end_bcb}"
        )
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            raise ValueError("No data returned from BCB API")
        
        # Parse JSON response
        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.set_index("data").sort_index()
        df = df.rename(columns={"valor": "ptax"})
        df.index.name = "date"
        
        print(f"[OK] Loaded {len(df)} rows from BCB API")
        return df[["ptax"]]
    
    except Exception as e:
        print(f"[WARN] BCB API failed: {e}")
        print("[FALLBACK] Attempting CSV...")
        return _read_csv_fallback("ptax")


def load_selic_sgs(start: str = DEFAULT_START_DATE, end: Optional[str] = None) -> pd.DataFrame:
    """
    Load SELIC (interest rate) from BCB SGS API.
    
    Falls back to CSV if API fails.
    
    Parameters
    ----------
    start : str
        Start date (DD/MM/YYYY)
    end : str, optional
        End date (DD/MM/YYYY), defaults to today
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'selic' column indexed by date
    """
    print("[LOAD] Loading SELIC from BCB SGS API...")
    
    try:
        # Convert date format for BCB API
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        start_bcb = start_dt.strftime("%d/%m/%Y")
        
        if end:
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            end_bcb = end_dt.strftime("%d/%m/%Y")
        else:
            end_bcb = datetime.now().strftime("%d/%m/%Y")
        
        # BCB SGS API endpoint
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{SGS_SELIC}/dados"
            f"?formato=json&dataInicial={start_bcb}&dataFinal={end_bcb}"
        )
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            raise ValueError("No data returned from BCB API")
        
        # Parse JSON response
        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.set_index("data").sort_index()
        df = df.rename(columns={"valor": "selic"})
        df.index.name = "date"
        
        print(f"[OK] Loaded {len(df)} rows from BCB API")
        return df[["selic"]]
    
    except Exception as e:
        print(f"[WARN] BCB API failed: {e}")
        print("[FALLBACK] Attempting CSV...")
        return _read_csv_fallback("selic")


def load_pulp_usd() -> pd.DataFrame:
    """
    Load pulp prices in USD from CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'pulp_usd' column indexed by date
    """
    print("[LOAD] Loading pulp USD prices from CSV...")
    df = _read_csv_fallback("pulp_usd", value_col="price")
    print(f"[OK] Loaded {len(df)} rows")
    return df


def load_climate() -> pd.DataFrame:
    """
    Load climate data (precipitation, NDVI) from CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'precip_mm' and 'ndvi' columns indexed by date
    """
    print("[LOAD] Loading climate data from CSV...")
    
    csv_path = DATA_RAW / "climate.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Climate CSV not found: {csv_path}. "
            f"Please create the file with columns: date, precip_mm, ndvi"
        )
    
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    
    # Ensure required columns exist
    required_cols = ["precip_mm", "ndvi"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in climate.csv: {missing}")
    
    print(f"[OK] Loaded {len(df)} rows")
    return df[required_cols]


def load_oil(
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load oil prices from Yahoo Finance (WTI Crude Oil).
    
    Falls back to CSV if API fails. Oil prices serve as a proxy for
    gasoline prices and energy costs.
    
    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : str, optional
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'oil_usd' column (WTI crude price) indexed by date
    """
    print("[LOAD] Loading oil prices (WTI Crude)...")
    
    try:
        # WTI Crude Oil futures ticker on Yahoo Finance
        ticker = "CL=F"
        oil = yf.Ticker(ticker)
        df = oil.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Extract close price and rename
        df = df[["Close"]].rename(columns={"Close": "oil_usd"})
        df.index.name = "date"
        
        # Strip timezone to avoid merge issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"[OK] Loaded {len(df)} rows from Yahoo Finance")
        return df
    
    except Exception as e:
        print(f"[WARN] Yahoo Finance failed: {e}")
        print("[FALLBACK] Attempting CSV fallback...")
        try:
            df = _read_csv_fallback("oil_usd", value_col="Close")
            print(f"[OK] Loaded {len(df)} rows from CSV")
            return df
        except FileNotFoundError as csv_err:
            print(f"[ERROR] Failed to load oil: {csv_err}")
            return pd.DataFrame()


def load_credit() -> pd.DataFrame:
    """
    Load credit index from CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'credit' column indexed by date
    """
    print("[LOAD] Loading credit index from CSV...")
    df = _read_csv_fallback("credit", value_col="index")
    print(f"[OK] Loaded {len(df)} rows")
    return df


if __name__ == "__main__":
    # Test loaders
    print("\n=== Testing Data Loaders ===\n")
    
    try:
        equity = load_equity()
        print(f"Equity: {equity.shape}\n{equity.head()}\n")
    except Exception as e:
        print(f"Equity load failed: {e}\n")
    
    try:
        ptax = load_ptax_sgs()
        print(f"PTAX: {ptax.shape}\n{ptax.head()}\n")
    except Exception as e:
        print(f"PTAX load failed: {e}\n")
    
    try:
        selic = load_selic_sgs()
        print(f"SELIC: {selic.shape}\n{selic.head()}\n")
    except Exception as e:
        print(f"SELIC load failed: {e}\n")

