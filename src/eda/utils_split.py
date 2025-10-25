"""Utilities for temporal data splitting without leakage."""

from typing import Tuple
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets temporally.
    
    Ensures no data leakage by using strict temporal ordering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame to split
    train_ratio : float
        Proportion for training (default 0.70)
    val_ratio : float
        Proportion for validation (default 0.15)
        Test proportion will be 1 - train_ratio - val_ratio
    
    Returns
    -------
    df_train : pd.DataFrame
        Training set (earliest data)
    df_val : pd.DataFrame
        Validation set (middle data)
    df_test : pd.DataFrame
        Test set (most recent data)
    
    Examples
    --------
    >>> df_train, df_val, df_test = temporal_split(df, 0.7, 0.15)
    >>> # Train on 70%, validate on 15%, test on 15%
    """
    n = len(df)
    
    # Calculate split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split using iloc to ensure no overlap
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\n[SPLIT] Temporal data split:")
    print(f"  Train: {len(df_train)} obs ({train_ratio*100:.0f}%) | {df_train.index.min()} to {df_train.index.max()}")
    print(f"  Val:   {len(df_val)} obs ({val_ratio*100:.0f}%) | {df_val.index.min()} to {df_val.index.max()}")
    print(f"  Test:  {len(df_test)} obs ({(1-train_ratio-val_ratio)*100:.0f}%) | {df_test.index.min()} to {df_test.index.max()}")
    
    return df_train, df_val, df_test


def walk_forward_splits(
    df: pd.DataFrame,
    train_size: int = 500,
    test_size: int = 50,
    step: int = 25,
) -> list:
    """
    Generate walk-forward train/test splits for backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame
    train_size : int
        Number of observations in each training window
    test_size : int
        Number of observations in each test window
    step : int
        Step size between consecutive windows
    
    Yields
    ------
    tuple
        (train_data, test_data) for each window
    
    Examples
    --------
    >>> for train, test in walk_forward_splits(df, 500, 50, 25):
    ...     model.fit(train)
    ...     metrics = evaluate(model, test)
    """
    splits = []
    
    for i in range(train_size, len(df) - test_size + 1, step):
        train_data = df.iloc[i - train_size:i].copy()
        test_data = df.iloc[i:i + test_size].copy()
        splits.append((train_data, test_data))
    
    print(f"\n[WALK-FORWARD] Generated {len(splits)} splits")
    print(f"  Train size: {train_size}, Test size: {test_size}, Step: {step}")
    
    return splits

