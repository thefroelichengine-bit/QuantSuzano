"""Feature engineering module: merge, transforms, and lag features."""

import numpy as np
import pandas as pd

from .config import BUSINESS_FREQ
from .loaders import (
    load_climate,
    load_credit,
    load_equity,
    load_oil,
    load_ptax_sgs,
    load_pulp_usd,
    load_selic_sgs,
)
from .loaders_benchmarks import load_all_benchmarks


def build_features(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """
    Build feature matrix by merging all data sources and engineering features.
    
    Process:
    1. Load all data sources
    2. Merge on date
    3. Convert to business day frequency with forward fill
    4. Create pulp_brl (pulp_usd * ptax)
    5. Compute log returns for all numeric columns
    6. Create climate lags (15, 30, 60 days)
    
    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD)
    end : str, optional
        End date (YYYY-MM-DD)
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all features indexed by date
    """
    print("\n" + "=" * 60)
    print("BUILDING FEATURE MATRIX")
    print("=" * 60 + "\n")
    
    # Load all data sources
    dfs = []
    
    try:
        equity = load_equity(start=start, end=end)
        dfs.append(equity)
    except Exception as e:
        print(f"[ERROR] Failed to load equity: {e}")
    
    try:
        ptax = load_ptax_sgs(start=start, end=end)
        dfs.append(ptax)
    except Exception as e:
        print(f"[ERROR] Failed to load PTAX: {e}")
    
    try:
        selic = load_selic_sgs(start=start, end=end)
        dfs.append(selic)
    except Exception as e:
        print(f"[ERROR] Failed to load SELIC: {e}")
    
    try:
        pulp = load_pulp_usd()
        dfs.append(pulp)
    except Exception as e:
        print(f"[ERROR] Failed to load pulp: {e}")
    
    try:
        oil = load_oil(start=start, end=end)
        if not oil.empty:
            dfs.append(oil)
    except Exception as e:
        print(f"[ERROR] Failed to load oil: {e}")
    
    try:
        climate = load_climate()
        dfs.append(climate)
    except Exception as e:
        print(f"[ERROR] Failed to load climate: {e}")
    
    try:
        credit = load_credit()
        dfs.append(credit)
    except Exception as e:
        print(f"[ERROR] Failed to load credit: {e}")
    
    # Load benchmark indices
    try:
        benchmarks = load_all_benchmarks(start=start, end=end)
        if not benchmarks.empty:
            dfs.append(benchmarks)
    except Exception as e:
        print(f"[ERROR] Failed to load benchmarks: {e}")
    
    if not dfs:
        raise ValueError("No data sources loaded successfully!")
    
    print(f"\n[MERGE] Merging {len(dfs)} data sources...")
    
    # Merge all dataframes
    df = pd.concat(dfs, axis=1, join="outer")
    df = df.sort_index()
    
    print(f"[OK] Initial shape: {df.shape}")
    print(f"[OK] Date range: {df.index.min()} to {df.index.max()}")
    
    # Convert to business day frequency with forward fill
    print(f"\n[RESAMPLE] Converting to business day frequency ({BUSINESS_FREQ})...")
    df = df.asfreq(BUSINESS_FREQ).ffill()
    print(f"[OK] After resampling: {df.shape}")
    
    # Create pulp_brl if both pulp_usd and ptax exist
    if "pulp_usd" in df.columns and "ptax" in df.columns:
        print("\n[FEATURE] Creating pulp_brl = pulp_usd * ptax...")
        df["pulp_brl"] = df["pulp_usd"] * df["ptax"]
        print(f"[OK] pulp_brl created (range: {df['pulp_brl'].min():.2f} - {df['pulp_brl'].max():.2f})")
    
    # Create oil_brl if both oil_usd and ptax exist
    if "oil_usd" in df.columns and "ptax" in df.columns:
        print("\n[FEATURE] Creating oil_brl = oil_usd * ptax...")
        df["oil_brl"] = df["oil_usd"] * df["ptax"]
        print(f"[OK] oil_brl created (range: {df['oil_brl'].min():.2f} - {df['oil_brl'].max():.2f})")
    
    # Compute log returns for all numeric columns
    print("\n[FEATURE] Computing log returns...")
    
    return_cols = ["ptax", "selic", "pulp_brl", "oil_brl", "suzb", "credit", "precip_mm", "ndvi", "imat", "iagro", "ibov"]
    for col in return_cols:
        if col in df.columns:
            # Compute log return
            df[f"{col}_r"] = np.log(df[col]).diff()
            
            # Report stats
            non_null = df[f"{col}_r"].notna().sum()
            print(f"  [OK] {col}_r: {non_null} observations")
    
    # Create climate lags
    print("\n[FEATURE] Creating climate lags (15, 30, 60 days)...")
    
    for lag in [15, 30, 60]:
        if "precip_mm" in df.columns:
            df[f"precip_mm_l{lag}"] = df["precip_mm"].shift(lag)
            print(f"  [OK] precip_mm_l{lag}")
        
        if "ndvi" in df.columns:
            df[f"ndvi_l{lag}"] = df["ndvi"].shift(lag)
            print(f"  [OK] ndvi_l{lag}")
    
    # Momentum indicators
    if "suzb" in df.columns:
        print("\n[FEATURE] Creating momentum indicators...")
        
        # Momentum periods
        df["mom_5"] = df["suzb"].pct_change(5)
        df["mom_20"] = df["suzb"].pct_change(20)
        df["mom_60"] = df["suzb"].pct_change(60)
        print("  [OK] mom_5, mom_20, mom_60")
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)  # Avoid division by zero
            return 100 - (100 / (1 + rs))
        
        df["rsi_14"] = calculate_rsi(df["suzb"], 14)
        print("  [OK] rsi_14")
        
        # Volatility regime
        if "suzb_r" in df.columns:
            df["vol_20"] = df["suzb_r"].rolling(20).std()
            vol_median = df["vol_20"].median()
            df["vol_regime"] = np.where(df["vol_20"] > vol_median, 1, 0)  # 1=high, 0=low
            print(f"  [OK] vol_20, vol_regime (median: {vol_median:.6f})")
        
        # Trend strength (linear regression slope)
        from scipy.stats import linregress
        def calc_trend(prices):
            if len(prices) < 20 or prices.isna().any():
                return np.nan
            return linregress(range(len(prices)), prices).slope
        
        df["trend_20"] = df["suzb"].rolling(20).apply(calc_trend, raw=False)
        print("  [OK] trend_20")
        
        # EMA and gap
        df["ema_20"] = df["suzb"].ewm(span=20, adjust=False).mean()
        df["ema_gap"] = (df["suzb"] - df["ema_20"]) / df["suzb"]
        print("  [OK] ema_20, ema_gap")
        
        # Forward-looking target for momentum strategies
        df["target_fwd5"] = df["suzb"].pct_change(5).shift(-5)
        print("  [OK] target_fwd5 (5-day forward return)")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    return df


def get_feature_cols(df: pd.DataFrame, return_only: bool = True) -> list:
    """
    Get list of feature columns for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    return_only : bool
        If True, only return columns ending with '_r' (log returns)
    
    Returns
    -------
    list
        List of feature column names
    """
    if return_only:
        return [col for col in df.columns if col.endswith("_r") and col != "suzb_r"]
    else:
        return df.columns.tolist()


def clean_data(df: pd.DataFrame, drop_na: bool = True, winsorize: bool = False) -> pd.DataFrame:
    """
    Clean data by handling outliers and missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    drop_na : bool
        If True, drop rows with NaN values
    winsorize : bool
        If True, winsorize outliers at 1% and 99%
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if winsorize:
        print("\n[CLEAN] Winsorizing outliers (1%, 99%)...")
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            lower = df_clean[col].quantile(0.01)
            upper = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower, upper)
    
    if drop_na:
        before = len(df_clean)
        df_clean = df_clean.dropna()
        after = len(df_clean)
        print(f"\n[CLEAN] Dropped {before - after} rows with NaN values ({after} remaining)")
    
    return df_clean


if __name__ == "__main__":
    # Test feature engineering
    print("\n=== Testing Feature Engineering ===\n")
    
    try:
        df = build_features()
        print("\n[SUCCESS] Feature engineering successful!")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nLast 5 rows:\n{df.tail()}")
        print(f"\nDescriptive stats:\n{df.describe()}")
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        raise

