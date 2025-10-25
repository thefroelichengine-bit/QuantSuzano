"""Simple backtesting framework for signal validation."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .metrics import sharpe_ratio, sortino_ratio, max_drawdown


def simple_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    returns_col: str = "suzb_r",
    transaction_cost: float = 0.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Run simple backtest based on signals.
    
    Strategy:
    - signal = 1: Long position
    - signal = -1: Short position  
    - signal = 0: No position
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with signals and returns
    signal_col : str
        Column name for signals
    returns_col : str
        Column name for returns
    transaction_cost : float
        Transaction cost as fraction (e.g., 0.001 for 0.1%)
    
    Returns
    -------
    metrics : dict
        Dictionary with performance metrics
    results : pd.DataFrame
        DataFrame with position, returns, cumulative PnL
    """
    # Prepare data
    data = df[[signal_col, returns_col]].copy().dropna()
    
    # Shift signal to avoid look-ahead bias
    data["position"] = data[signal_col].shift(1).fillna(0)
    
    # Calculate strategy returns
    data["strategy_returns"] = data["position"] * data[returns_col]
    
    # Apply transaction costs on position changes
    position_change = data["position"].diff().abs()
    data["costs"] = position_change * transaction_cost
    data["strategy_returns_net"] = data["strategy_returns"] - data["costs"]
    
    # Cumulative returns
    data["cum_market_returns"] = data[returns_col].cumsum()
    data["cum_strategy_returns"] = data["strategy_returns_net"].cumsum()
    
    # Calculate metrics
    market_return = data["cum_market_returns"].iloc[-1]
    strategy_return = data["cum_strategy_returns"].iloc[-1]
    
    metrics = {
        "total_market_return": market_return,
        "total_strategy_return": strategy_return,
        "excess_return": strategy_return - market_return,
        "sharpe_ratio": sharpe_ratio(data["strategy_returns_net"]),
        "sortino_ratio": sortino_ratio(data["strategy_returns_net"]),
        "max_drawdown": max_drawdown(data["strategy_returns_net"]),
        "num_trades": position_change.sum(),
        "total_costs": data["costs"].sum(),
        "win_rate": (data["strategy_returns_net"] > 0).mean(),
    }
    
    return metrics, data


def zscore_backtest(
    df: pd.DataFrame,
    zscore_col: str = "zscore",
    returns_col: str = "suzb_r",
    threshold: float = 2.0,
    transaction_cost: float = 0.001,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Backtest strategy based on z-score thresholds.
    
    Strategy:
    - Buy when zscore < -threshold (undervalued)
    - Sell when zscore > +threshold (overvalued)
    - Close position when zscore crosses back toward 0
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with zscore and returns
    zscore_col : str
        Column name for z-scores
    returns_col : str
        Column name for returns
    threshold : float
        Z-score threshold for signals
    transaction_cost : float
        Transaction cost per trade
    
    Returns
    -------
    metrics : dict
        Performance metrics
    results : pd.DataFrame
        Detailed results
    """
    data = df[[zscore_col, returns_col]].copy().dropna()
    
    # Generate signals
    data["signal"] = 0
    data.loc[data[zscore_col] < -threshold, "signal"] = 1  # Long
    data.loc[data[zscore_col] > threshold, "signal"] = -1  # Short
    
    # Run backtest
    metrics, results = simple_backtest(
        data,
        signal_col="signal",
        returns_col=returns_col,
        transaction_cost=transaction_cost,
    )
    
    return metrics, results


def walk_forward_backtest(
    df: pd.DataFrame,
    model_func,
    train_size: int = 500,
    test_size: int = 50,
    step: int = 25,
) -> pd.DataFrame:
    """
    Perform walk-forward backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    model_func : callable
        Function that takes train data and returns fitted model
    train_size : int
        Training window size
    test_size : int
        Test window size
    step : int
        Step size between windows
    
    Returns
    -------
    pd.DataFrame
        Results from each walk-forward window
    """
    from .utils_split import walk_forward_splits
    
    results = []
    
    splits = walk_forward_splits(df, train_size, test_size, step)
    
    for i, (train_data, test_data) in enumerate(splits):
        print(f"[WF] Window {i+1}/{len(splits)}: {test_data.index.min()} to {test_data.index.max()}")
        
        # Fit model on train data
        model = model_func(train_data)
        
        # Predict on test data
        # (Implementation depends on model_func signature)
        # results.append(...)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test backtest functions
    print("\n=== Testing Backtest Module ===\n")
    
    # Create sample data
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    sample_df = pd.DataFrame({
        "zscore": np.random.randn(1000),
        "suzb_r": np.random.randn(1000) * 0.01,
        "signal": np.random.choice([-1, 0, 1], 1000),
    }, index=dates)
    
    # Run simple backtest
    metrics, results = simple_backtest(sample_df)
    print("[OK] Simple backtest completed")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Total Return: {metrics['total_strategy_return']:.4f}")
    
    # Run z-score backtest
    metrics_z, results_z = zscore_backtest(sample_df)
    print("[OK] Z-score backtest completed")
    print(f"  Sharpe: {metrics_z['sharpe_ratio']:.2f}")

