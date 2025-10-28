"""
Benchmark comparison module for strategy evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path
from ..config import DATA_OUT


class BenchmarkComparison:
    """
    Compare strategy performance against multiple benchmarks.
    """
    
    def __init__(self):
        self.benchmarks = {}
        self.strategy_returns = None
        self.metrics = {}
        
    def add_benchmark(self, name: str, returns: pd.Series):
        """Add a benchmark return series."""
        self.benchmarks[name] = returns
        
    def set_strategy_returns(self, returns: pd.Series):
        """Set the strategy return series."""
        self.strategy_returns = returns
        
    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate comprehensive metrics for strategy and all benchmarks.
        
        Returns DataFrame with metrics for each.
        """
        print("\n[BENCHMARK] Calculating metrics...")
        
        all_returns = {'Strategy': self.strategy_returns}
        all_returns.update(self.benchmarks)
        
        results = []
        
        for name, returns in all_returns.items():
            if returns is None or len(returns) == 0:
                continue
            
            returns = returns.dropna()
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.05) / (volatility + 1e-6)
            
            # Downside metrics
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = (annual_return - 0.05) / (downside_vol + 1e-6)
            
            # Drawdown
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
            calmar = annual_return / (abs(max_dd) + 1e-6)
            
            # Distribution
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Win rate
            win_rate = (returns > 0).sum() / len(returns)
            
            metrics_dict = {
                'name': name,
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_dd,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'win_rate': win_rate,
            }
            
            results.append(metrics_dict)
            self.metrics[name] = metrics_dict
        
        metrics_df = pd.DataFrame(results)
        
        print("\n  Metrics Summary:")
        print(metrics_df[['name', 'annual_return', 'sharpe_ratio', 'max_drawdown']].to_string(index=False))
        
        return metrics_df
    
    def calculate_relative_metrics(self, benchmark_name: str = 'Buy-and-Hold') -> Dict:
        """
        Calculate metrics relative to a specific benchmark (alpha, beta, etc.).
        
        Parameters
        ----------
        benchmark_name : str
            Name of benchmark to compare against
        
        Returns
        -------
        dict
            Relative metrics
        """
        if self.strategy_returns is None or benchmark_name not in self.benchmarks:
            return {}
        
        strat_ret = self.strategy_returns.dropna()
        bench_ret = self.benchmarks[benchmark_name].dropna()
        
        # Align indices
        common_idx = strat_ret.index.intersection(bench_ret.index)
        strat_ret = strat_ret.loc[common_idx]
        bench_ret = bench_ret.loc[common_idx]
        
        # Beta
        covariance = np.cov(strat_ret, bench_ret)[0, 1]
        benchmark_var = bench_ret.var()
        beta = covariance / (benchmark_var + 1e-6)
        
        # Alpha
        bench_annual_ret = (1 + bench_ret).prod() ** (252 / len(bench_ret)) - 1
        strat_annual_ret = (1 + strat_ret).prod() ** (252 / len(strat_ret)) - 1
        alpha = strat_annual_ret - beta * bench_annual_ret
        
        # Tracking error
        excess_returns = strat_ret - bench_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        excess_return_annual = excess_returns.mean() * 252
        information_ratio = excess_return_annual / (tracking_error + 1e-6)
        
        # Correlation
        correlation = strat_ret.corr(bench_ret)
        
        # Up/down capture
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0
        
        if up_periods.sum() > 0:
            up_capture = strat_ret[up_periods].mean() / bench_ret[up_periods].mean()
        else:
            up_capture = np.nan
            
        if down_periods.sum() > 0:
            down_capture = strat_ret[down_periods].mean() / bench_ret[down_periods].mean()
        else:
            down_capture = np.nan
        
        return {
            'benchmark': benchmark_name,
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation': correlation,
            'up_capture': up_capture,
            'down_capture': down_capture,
        }
    
    def plot_equity_curves(self, output_path: Path = None):
        """Plot cumulative returns for strategy and all benchmarks."""
        if output_path is None:
            output_path = DATA_OUT / "plots/strategies/benchmark_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot strategy
        if self.strategy_returns is not None:
            cum_ret = (1 + self.strategy_returns).cumprod()
            ax.plot(cum_ret.index, cum_ret, linewidth=2, label='Strategy', color='black')
        
        # Plot benchmarks
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.benchmarks)))
        for (name, returns), color in zip(self.benchmarks.items(), colors):
            cum_ret = (1 + returns).cumprod()
            ax.plot(cum_ret.index, cum_ret, linewidth=1.5, label=name, alpha=0.7, color=color)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (1 + R)', fontsize=12)
        ax.set_title('Strategy vs Benchmarks - Equity Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[BENCHMARK] Equity curve plot saved to: {output_path}")
    
    def save_comparison(self, output_path: Path = None):
        """Save full comparison to CSV."""
        if output_path is None:
            output_path = DATA_OUT / "benchmark_comparison.csv"
        
        metrics_df = self.calculate_metrics()
        metrics_df.to_csv(output_path, index=False)
        
        print(f"[BENCHMARK] Comparison saved to: {output_path}")
        
        return metrics_df


def compare_strategies(
    strategy_returns: pd.Series,
    benchmark_returns: Dict[str, pd.Series]
) -> Tuple[pd.DataFrame, BenchmarkComparison]:
    """
    Comprehensive strategy vs benchmark comparison.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy return series
    benchmark_returns : dict
        Dictionary of benchmark name -> return series
    
    Returns
    -------
    metrics_df : DataFrame
        Comparison metrics
    comparator : BenchmarkComparison
        Comparator object with all results
    """
    print("\n" + "=" * 70)
    print("STRATEGY VS BENCHMARK COMPARISON")
    print("=" * 70)
    
    comparator = BenchmarkComparison()
    comparator.set_strategy_returns(strategy_returns)
    
    for name, returns in benchmark_returns.items():
        comparator.add_benchmark(name, returns)
    
    # Calculate metrics
    metrics_df = comparator.calculate_metrics()
    
    # Calculate relative metrics vs first benchmark
    if len(benchmark_returns) > 0:
        first_benchmark = list(benchmark_returns.keys())[0]
        relative_metrics = comparator.calculate_relative_metrics(first_benchmark)
        
        print(f"\nRelative Metrics vs {first_benchmark}:")
        for k, v in relative_metrics.items():
            if k != 'benchmark':
                print(f"  {k}: {v:.4f}")
    
    # Generate plots
    comparator.plot_equity_curves()
    comparator.save_comparison()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON COMPLETE")
    print("=" * 70)
    
    return metrics_df, comparator

