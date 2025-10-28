"""
Loaders for benchmark indices (IMAT, IAGRO, IBOV).
Extends the data loading capabilities with sector indices.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from .config import DATA_RAW


def load_imat(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Load IMAT (Índice de Materiais Básicos) - B3 Materials Index.
    
    Ticker: IMAT11.SA
    
    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        IMAT price data with 'imat' column
    """
    print("[IMAT] Fetching materials index from Yahoo Finance...")
    
    try:
        # Correct B3 ticker for IMAT
        ticker = yf.Ticker("IMAT11.SA")
        df = ticker.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            raise ValueError("No data returned from yfinance")
        
        # Keep only Close price and rename
        df = df[['Close']].rename(columns={'Close': 'imat'})
        df.index.name = 'date'
        
        # Strip timezone to avoid merge issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"[OK] IMAT: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        return df
    
    except Exception as e:
        print(f"[WARN] Failed to fetch IMAT: {e}")
        print("[FALLBACK] Attempting CSV fallback...")
        return _read_csv_fallback("imat.csv", "imat")


def load_iagro(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Load IAGRO (Índice do Agronegócio) - B3 Agribusiness Index.
    
    Ticker: IAGR11.SA (or fallback to CSV)
    
    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        IAGRO price data with 'iagro' column
    """
    print("[IAGRO] Fetching agribusiness index...")
    
    # Try multiple ticker formats (correct B3 format first)
    tickers_to_try = ["IAGR11.SA", "IAGRO11.SA", "^IAGRO"]
    
    for ticker_symbol in tickers_to_try:
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            
            if not df.empty:
                df = df[['Close']].rename(columns={'Close': 'iagro'})
                df.index.name = 'date'
                
                # Strip timezone to avoid merge issues
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                print(f"[OK] IAGRO ({ticker_symbol}): {len(df)} rows")
                return df
        
        except Exception:
            continue
    
    # If all tickers fail, use fallback
    print(f"[WARN] IAGRO not available on Yahoo Finance")
    print("[FALLBACK] Attempting CSV fallback...")
    return _read_csv_fallback("iagro.csv", "iagro")


def load_ibov(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Load IBOVESPA index as additional benchmark.
    
    Ticker: ^BVSP
    
    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        IBOV price data with 'ibov' column
    """
    print("[IBOV] Fetching Ibovespa index...")
    
    try:
        ticker = yf.Ticker("^BVSP")
        df = ticker.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            raise ValueError("No data returned")
        
        df = df[['Close']].rename(columns={'Close': 'ibov'})
        df.index.name = 'date'
        
        # Strip timezone to avoid merge issues
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"[OK] IBOV: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        return df
    
    except Exception as e:
        print(f"[WARN] Failed to fetch IBOV: {e}")
        print("[FALLBACK] Attempting CSV fallback...")
        return _read_csv_fallback("ibov.csv", "ibov")


def _read_csv_fallback(filename: str, column_name: str) -> pd.DataFrame:
    """
    Read data from CSV fallback file.
    
    Parameters
    ----------
    filename : str
        CSV filename in DATA_RAW
    column_name : str
        Name for the price column
    
    Returns
    -------
    pd.DataFrame
        Data from CSV or empty DataFrame
    """
    csv_path = DATA_RAW / filename
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            
            # Rename price column if needed
            if 'Close' in df.columns:
                df = df[['Close']].rename(columns={'Close': column_name})
            elif 'close' in df.columns:
                df = df[['close']].rename(columns={'close': column_name})
            elif 'price' in df.columns:
                df = df[['price']].rename(columns={'price': column_name})
            
            print(f"[FALLBACK] Loaded {len(df)} rows from {csv_path}")
            return df
        
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path}: {e}")
    else:
        print(f"[INFO] Fallback file not found: {csv_path}")
        print(f"[INFO] Create {csv_path} with columns: date,Close")
    
    # Return empty DataFrame with correct structure
    return pd.DataFrame(columns=[column_name])


def load_all_benchmarks(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Load all benchmark indices and merge them.
    
    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD), defaults to today
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all benchmark indices
    """
    print("\n[BENCHMARKS] Loading all benchmark indices...")
    
    benchmarks = []
    
    # Load IMAT
    imat = load_imat(start, end)
    if not imat.empty:
        benchmarks.append(imat)
    
    # Load IAGRO
    iagro = load_iagro(start, end)
    if not iagro.empty:
        benchmarks.append(iagro)
    
    # Load IBOV
    ibov = load_ibov(start, end)
    if not ibov.empty:
        benchmarks.append(ibov)
    
    # Merge all benchmarks
    if benchmarks:
        result = pd.concat(benchmarks, axis=1)
        print(f"\n[OK] Benchmarks loaded: {list(result.columns)}")
        return result
    else:
        print("[WARN] No benchmarks loaded")
        return pd.DataFrame()

