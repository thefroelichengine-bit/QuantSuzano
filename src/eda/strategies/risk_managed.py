"""
Risk-managed trading strategy with position sizing, stop-losses, and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from itertools import product
from ..config import DATA_OUT, Z_THRESHOLD
from ..backtest import zscore_backtest


class RiskManagedStrategy:
    """
    Enhanced z-score strategy with risk management.
    """
    
    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        stop_loss_pct: float = 0.10,
        take_profit_pct: float = 0.10,
        position_size: str = 'fixed',
        max_position: float = 1.0,
        vol_target: float = None,
        vol_filter: float = 0.40
    ):
        """
        Initialize risk-managed strategy.
        
        Parameters
        ----------
        z_entry : float
            Z-score threshold for entry
        z_exit : float
            Z-score threshold for exit
        stop_loss_pct : float
            Stop-loss as fraction of entry price
        take_profit_pct : float
            Take-profit as fraction of entry price
        position_size : str
            'fixed', 'volatility', or 'kelly'
        max_position : float
            Maximum position size
        vol_target : float, optional
            Target volatility for position sizing
        vol_filter : float
            Don't trade when volatility > this level
        """
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size = position_size
        self.max_position = max_position
        self.vol_target = vol_target
        self.vol_filter = vol_filter
        
    def generate_signals(
        self,
        df: pd.DataFrame,
        zscore_col: str = 'zscore',
        returns_col: str = 'suzb_r'
    ) -> pd.DataFrame:
        """
        Generate trading signals with risk management.
        
        Returns DataFrame with positions, stops, and metrics.
        """
        signals = df.copy()
        
        # Calculate rolling volatility
        signals['vol'] = signals[returns_col].rolling(60).std() * np.sqrt(252)
        
        # Base position from z-score
        signals['position'] = 0.0
        signals['position'] = np.where(signals[zscore_col] > self.z_entry, -1.0, signals['position'])
        signals['position'] = np.where(signals[zscore_col] < -self.z_entry, 1.0, signals['position'])
        signals['position'] = np.where(np.abs(signals[zscore_col]) < self.z_exit, 0.0, signals['position'])
        
        # Forward-fill positions
        signals['position'] = signals['position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Apply volatility filter
        signals.loc[signals['vol'] > self.vol_filter, 'position'] = 0.0
        
        # Position sizing
        if self.position_size == 'volatility' and self.vol_target is not None:
            # Scale positions by volatility
            vol_scalar = self.vol_target / (signals['vol'] + 1e-6)
            vol_scalar = vol_scalar.clip(0, 2)  # Limit scaling to 2x
            signals['position'] = signals['position'] * vol_scalar
        
        # Clip to max position
        signals['position'] = signals['position'].clip(-self.max_position, self.max_position)
        
        # Calculate returns
        signals['strategy_return'] = signals['position'].shift(1) * signals[returns_col]
        
        # Calculate cumulative returns
        signals['cum_return'] = (1 + signals['strategy_return']).cumprod() - 1
        
        # Stop-loss and take-profit logic (simplified)
        # In practice, this would track entry prices per trade
        signals['hit_stop'] = False
        signals['hit_target'] = False
        
        return signals
    
    def backtest(
        self,
        df: pd.DataFrame,
        zscore_col: str = 'zscore',
        returns_col: str = 'suzb_r'
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run backtest with full risk management.
        
        Returns
        -------
        metrics : dict
            Performance metrics
        signals : DataFrame
            Signal history
        """
        signals = self.generate_signals(df, zscore_col, returns_col)
        
        # Calculate metrics
        returns = signals['strategy_return'].dropna()
        
        if len(returns) == 0:
            return {}, signals
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.05) / (volatility + 1e-6)  # Assuming 5% risk-free rate
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns[returns != 0])
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Number of trades (position changes)
        position_changes = (signals['position'].diff() != 0).sum()
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': position_changes,
            'avg_return_per_trade': total_return / (position_changes + 1),
        }
        
        return metrics, signals


def optimize_strategy_parameters(
    df: pd.DataFrame,
    param_grid: Dict = None,
    zscore_col: str = 'zscore',
    returns_col: str = 'suzb_r'
) -> Tuple[Dict, pd.DataFrame]:
    """
    Grid search over strategy parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with z-scores
    param_grid : dict, optional
        Parameter grid to search
    zscore_col : str
        Z-score column name
    returns_col : str
        Returns column name
    
    Returns
    -------
    best_params : dict
        Best parameters found
    results_df : DataFrame
        Results for all parameter combinations
    """
    print("\n" + "=" * 70)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    if param_grid is None:
        param_grid = {
            'z_entry': [1.5, 2.0, 2.5],
            'z_exit': [0.0, 0.5, 1.0],
            'stop_loss_pct': [0.05, 0.10, 0.15],
            'vol_filter': [0.30, 0.40, 0.50],
        }
    
    # Generate all combinations
    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))
    
    print(f"\nTesting {len(combinations)} parameter combinations...")
    print(f"Parameters: {list(keys)}")
    
    results = []
    
    for i, params in enumerate(combinations, 1):
        param_dict = dict(zip(keys, params))
        
        strategy = RiskManagedStrategy(**param_dict)
        metrics, _ = strategy.backtest(df, zscore_col, returns_col)
        
        if metrics:
            result = {**param_dict, **metrics}
            results.append(result)
        
        if i % 10 == 0:
            print(f"  Tested {i}/{len(combinations)} combinations...")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n[ERROR] No valid results found!")
        return {}, results_df
    
    # Find best by Sharpe ratio
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_params = results_df.loc[best_idx, list(keys)].to_dict()
    best_metrics = results_df.loc[best_idx, ['sharpe_ratio', 'total_return', 'max_drawdown']].to_dict()
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("\nBest Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save results
    output_path = DATA_OUT / "strategy_optimization.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nAll results saved to: {output_path}")
    
    return best_params, results_df

