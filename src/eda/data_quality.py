"""
Data quality assessment and validation module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from .config import DATA_OUT


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in dataset.
    
    Returns DataFrame with missing value statistics.
    """
    missing_stats = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100,
        'dtype': df.dtypes
    })
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    print("\n[DATA QUALITY] Missing Values Analysis")
    print("=" * 60)
    if len(missing_stats) == 0:
        print("  No missing values detected!")
    else:
        print(missing_stats)
    
    return missing_stats


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, pd.Series]:
    """
    Detect outliers using IQR or Z-score method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    method : str
        'iqr' or 'zscore'
    threshold : float
        IQR multiplier or Z-score threshold
    
    Returns
    -------
    dict
        Dictionary mapping column names to boolean Series indicating outliers
    """
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\n[DATA QUALITY] Outlier Detection ({method.upper()} method)")
    print("=" * 60)
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        else:  # zscore
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = z_scores > threshold
        
        outliers[col] = outlier_mask
        n_outliers = outlier_mask.sum()
        pct_outliers = (n_outliers / len(df)) * 100
        
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers ({pct_outliers:.2f}%)")
    
    return outliers


def validate_consistency(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check data consistency (dates, ranges, relationships).
    
    Returns dict with validation results.
    """
    checks = {}
    
    print("\n[DATA QUALITY] Consistency Checks")
    print("=" * 60)
    
    # Check index is sorted
    checks['index_sorted'] = df.index.is_monotonic_increasing
    print(f"  Index sorted: {checks['index_sorted']}")
    
    # Check for duplicates
    checks['duplicates'] = df.index.duplicated().sum()
    print(f"  Duplicate dates: {checks['duplicates']}")
    
    # Check for negative prices (should not exist)
    price_cols = [col for col in df.columns if not col.endswith('_r')]
    for col in price_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            checks[f'{col}_negative'] = neg_count
            if neg_count > 0:
                print(f"  {col} negative values: {neg_count} [WARNING]")
    
    # Check return magnitudes (> 50% daily is suspicious)
    return_cols = [col for col in df.columns if col.endswith('_r')]
    for col in return_cols:
        if col in df.columns:
            extreme_count = (np.abs(df[col]) > 0.5).sum()
            checks[f'{col}_extreme'] = extreme_count
            if extreme_count > 0:
                print(f"  {col} extreme returns (>50%): {extreme_count} [WARNING]")
    
    return checks


def create_data_report(df: pd.DataFrame, output_path: Path = None) -> pd.DataFrame:
    """
    Generate comprehensive data quality report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    output_path : Path, optional
        Path to save report CSV
    
    Returns
    -------
    pd.DataFrame
        Summary report
    """
    if output_path is None:
        output_path = DATA_OUT / "data_quality_report.csv"
    
    print("\n[DATA QUALITY] Generating Comprehensive Report")
    print("=" * 60)
    
    report = []
    
    for col in df.columns:
        stats = {
            'variable': col,
            'dtype': str(df[col].dtype),
            'count': df[col].notna().sum(),
            'missing': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df)) * 100,
        }
        
        if df[col].dtype in [np.float64, np.int64]:
            stats.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'q25': df[col].quantile(0.25),
                'median': df[col].median(),
                'q75': df[col].quantile(0.75),
                'max': df[col].max(),
                'skew': df[col].skew(),
                'kurt': df[col].kurtosis(),
            })
        
        report.append(stats)
    
    report_df = pd.DataFrame(report)
    report_df.to_csv(output_path, index=False)
    
    print(f"  Report saved to: {output_path}")
    print(f"  Total variables: {len(report_df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Total observations: {len(df)}")
    
    return report_df


def run_full_quality_check(df: pd.DataFrame) -> Dict:
    """
    Run all data quality checks and generate report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    
    Returns
    -------
    dict
        Results from all quality checks
    """
    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 70)
    
    results = {}
    
    # Missing values
    results['missing'] = check_missing_values(df)
    
    # Outliers
    results['outliers'] = detect_outliers(df, method='iqr', threshold=3.0)
    
    # Consistency
    results['consistency'] = validate_consistency(df)
    
    # Generate report
    results['report'] = create_data_report(df)
    
    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT COMPLETE")
    print("=" * 70)
    
    return results

