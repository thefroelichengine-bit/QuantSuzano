"""Advanced visualization for model validation and diagnostics."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_predictions_by_split(
    df: pd.DataFrame,
    target_col: str = "suzb_r",
    pred_col: str = "synthetic_index",
    split_col: str = "split",
    outdir: Path = None,
):
    """
    Plot actual vs predicted values with train/val/test splits highlighted.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predictions and split labels
    target_col : str
        Column name for actual values
    pred_col : str
        Column name for predictions
    split_col : str
        Column name for split labels
    outdir : Path
        Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot by split
    for split_name in ['train', 'val', 'test']:
        split_data = df[df[split_col] == split_name]
        if len(split_data) > 0:
            ax.plot(split_data.index, split_data[target_col], 
                   label=f'{split_name.capitalize()} (Actual)', alpha=0.6, linewidth=1)
    
    # Plot predictions on top
    ax.plot(df.index, df[pred_col], label='Predicted', 
           color='red', linewidth=1.5, alpha=0.8)
    
    # Add split boundaries
    if split_col in df.columns:
        train_end = df[df[split_col] == 'train'].index.max()
        val_end = df[df[split_col] == 'val'].index.max()
        
        ax.axvline(train_end, color='gray', linestyle='--', alpha=0.5, label='Train/Val Split')
        ax.axvline(val_end, color='gray', linestyle=':', alpha=0.5, label='Val/Test Split')
    
    ax.set_title('Actual vs Predicted Returns (by Split)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "pred_vs_actual_splits.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {outdir / 'pred_vs_actual_splits.png'}")


def plot_residual_diagnostics(
    df: pd.DataFrame,
    error_col: str = "error",
    outdir: Path = None,
):
    """
    Create comprehensive residual diagnostic plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with residuals
    error_col : str
        Column name for residuals/errors
    outdir : Path
        Output directory
    """
    residuals = df[error_col].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram with normal curve
    ax = axes[0, 0]
    ax.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'N({mu:.4f}, {sigma:.4f})')
    
    ax.set_title('Residual Distribution')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals over time
    ax = axes[1, 0]
    ax.plot(df.index, df[error_col], alpha=0.6, linewidth=0.8)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.axhline(residuals.std(), color='orange', linestyle=':', alpha=0.7, label='+1 std')
    ax.axhline(-residuals.std(), color='orange', linestyle=':', alpha=0.7, label='-1 std')
    ax.set_title('Residuals Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Rolling std of residuals
    ax = axes[1, 1]
    rolling_std = df[error_col].rolling(60).std()
    rolling_std.plot(ax=ax, color='purple', linewidth=1.5)
    ax.set_title('Rolling Standard Deviation (60-day)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual Std Dev')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "residual_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {outdir / 'residual_diagnostics.png'}")


def plot_zscore_analysis(
    df: pd.DataFrame,
    zscore_col: str = "zscore",
    signal_col: str = "signal",
    outdir: Path = None,
):
    """
    Plot z-score analysis with signal zones.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with z-scores and signals
    zscore_col : str
        Column name for z-scores
    signal_col : str
        Column name for signals
    outdir : Path
        Output directory
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # Z-score plot
    df[zscore_col].plot(ax=ax1, color='darkblue', linewidth=1, alpha=0.7)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax1.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (+2)')
    ax1.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (-2)')
    
    # Fill extreme zones
    ax1.fill_between(df.index, 2, df[zscore_col], 
                     where=(df[zscore_col] > 2), color='red', alpha=0.2, label='Overbought')
    ax1.fill_between(df.index, -2, df[zscore_col], 
                     where=(df[zscore_col] < -2), color='green', alpha=0.2, label='Oversold')
    
    ax1.set_title('Z-Score of Residuals')
    ax1.set_ylabel('Z-Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Signal plot
    if signal_col in df.columns:
        df[signal_col].plot(ax=ax2, color='purple', linewidth=1.5, marker='o', markersize=3)
        ax2.set_title('Trading Signals')
        ax2.set_ylabel('Signal (-1: Short, 0: Neutral, 1: Long)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "zscore_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {outdir / 'zscore_analysis.png'}")


def plot_scatter_actual_vs_pred(
    df: pd.DataFrame,
    target_col: str = "suzb_r",
    pred_col: str = "synthetic_index",
    split_col: str = "split",
    outdir: Path = None,
):
    """
    Scatter plot of actual vs predicted with regression line.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and predicted values
    target_col : str
        Column name for actual values
    pred_col : str
        Column name for predictions
    split_col : str
        Column name for split labels
    outdir : Path
        Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot by split with different colors
    colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
    
    for split_name, color in colors.items():
        split_data = df[df[split_col] == split_name]
        if len(split_data) > 0:
            ax.scatter(split_data[target_col], split_data[pred_col], 
                      alpha=0.5, s=20, color=color, label=split_name.capitalize())
    
    # Add diagonal line (perfect prediction)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction')
    
    # Add regression line
    from scipy.stats import linregress
    valid_data = df[[target_col, pred_col]].dropna()
    slope, intercept, r_value, _, _ = linregress(valid_data[target_col], valid_data[pred_col])
    line_x = np.array(lims)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'g-', alpha=0.75, linewidth=2, 
           label=f'Fit (R={r_value:.3f})')
    
    ax.set_xlabel('Actual Returns')
    ax.set_ylabel('Predicted Returns')
    ax.set_title('Actual vs Predicted Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "scatter_actual_vs_pred.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {outdir / 'scatter_actual_vs_pred.png'}")


def plot_rolling_metrics(
    metrics_df: pd.DataFrame,
    outdir: Path = None,
):
    """
    Plot rolling performance metrics.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with rolling metrics
    outdir : Path
        Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    metric_cols = [col for col in metrics_df.columns if 'rolling' in col]
    
    for i, col in enumerate(metric_cols[:4]):
        ax = axes[i // 2, i % 2]
        metrics_df[col].plot(ax=ax, linewidth=1.5, color='steelblue')
        ax.set_title(col.replace('rolling_', '').replace('_', ' ').title())
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "rolling_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {outdir / 'rolling_metrics.png'}")


def plot_backtest_pnl(
    backtest_results: pd.DataFrame,
    outdir: Path = None,
):
    """
    Plot cumulative PnL from backtest.
    
    Parameters
    ----------
    backtest_results : pd.DataFrame
        Backtest results with cumulative returns
    outdir : Path
        Output directory
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    if 'cum_market_returns' in backtest_results.columns:
        backtest_results['cum_market_returns'].plot(
            ax=ax, label='Buy & Hold', linewidth=2, color='steelblue')
    
    if 'cum_strategy_returns' in backtest_results.columns:
        backtest_results['cum_strategy_returns'].plot(
            ax=ax, label='Strategy', linewidth=2, color='darkorange')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('Cumulative Returns: Strategy vs Buy & Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if outdir:
        plt.savefig(outdir / "backtest_pnl.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {outdir / 'backtest_pnl.png'}")


def generate_all_validation_plots(
    df: pd.DataFrame,
    backtest_results: pd.DataFrame = None,
    rolling_metrics: pd.DataFrame = None,
    outdir: Path = None,
):
    """
    Generate all validation plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame with predictions, errors, z-scores
    backtest_results : pd.DataFrame, optional
        Backtest results
    rolling_metrics : pd.DataFrame, optional
        Rolling metrics
    outdir : Path
        Output directory
    """
    if outdir is None:
        from ..config import PLOTS_DIR
        outdir = PLOTS_DIR
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("[GENERATING VALIDATION PLOTS]")
    print("=" * 70 + "\n")
    
    # Core plots
    plot_predictions_by_split(df, outdir=outdir)
    plot_residual_diagnostics(df, outdir=outdir)
    plot_zscore_analysis(df, outdir=outdir)
    plot_scatter_actual_vs_pred(df, outdir=outdir)
    
    # Optional plots
    if backtest_results is not None:
        plot_backtest_pnl(backtest_results, outdir=outdir)
    
    if rolling_metrics is not None:
        plot_rolling_metrics(rolling_metrics, outdir=outdir)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All validation plots generated!")
    print("=" * 70)


def plot_voting_patterns(
    signals_df: pd.DataFrame,
    outdir: Path = None,
):
    """
    Plot voting patterns showing how each model voted over time.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with model signals and voted signal
    outdir : Path
        Output directory
    """
    if outdir is None:
        from ..config import PLOTS_DIR
        outdir = PLOTS_DIR / "strategies"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Get signal columns
    signal_cols = [col for col in signals_df.columns if col.endswith('_signal')]
    if len(signal_cols) == 0:
        print("[WARNING] No signal columns found for voting plot")
        return
    
    fig, axes = plt.subplots(len(signal_cols) + 1, 1, figsize=(16, 3 * (len(signal_cols) + 1)), sharex=True)
    
    if len(signal_cols) == 1:
        axes = [axes]
    
    # Plot individual model signals
    for i, signal_col in enumerate(signal_cols):
        ax = axes[i]
        model_name = signal_col.replace('_signal', '')
        signal = signals_df[signal_col]
        
        # Plot signals as colored regions
        ax.fill_between(signal.index, -1.5, 1.5, where=(signal == 1), 
                       alpha=0.3, color='green', label='Long')
        ax.fill_between(signal.index, -1.5, 1.5, where=(signal == -1), 
                       alpha=0.3, color='red', label='Short')
        ax.plot(signal.index, signal, linewidth=1.5, alpha=0.7, color='black')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel(f'{model_name}')
        ax.set_ylim(-1.5, 1.5)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Plot voted signal
    ax = axes[-1]
    if 'voted_signal' in signals_df.columns:
        voted_signal = signals_df['voted_signal']
        ax.fill_between(voted_signal.index, -1.5, 1.5, where=(voted_signal == 1),
                       alpha=0.5, color='green', label='Voted Long')
        ax.fill_between(voted_signal.index, -1.5, 1.5, where=(voted_signal == -1),
                       alpha=0.5, color='red', label='Voted Short')
        ax.plot(voted_signal.index, voted_signal, linewidth=2, alpha=0.8, color='blue', label='Voted Signal')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Voted Signal')
    ax.set_xlabel('Date')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Ensemble Voting Patterns', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = outdir / "voting_patterns.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Voting patterns plot saved to: {output_path}")


def plot_risk_reward_analysis(
    signals_df: pd.DataFrame,
    outdir: Path = None,
):
    """
    Plot risk-reward ratio over time and execution decisions.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with risk_reward_ratio and executed_signal
    outdir : Path
        Output directory
    """
    if outdir is None:
        from ..config import PLOTS_DIR
        outdir = PLOTS_DIR / "strategies"
    outdir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Plot 1: Risk-reward ratio
    ax = axes[0]
    if 'risk_reward_ratio' in signals_df.columns:
        rr_ratio = signals_df['risk_reward_ratio'].dropna()
        ax.plot(rr_ratio.index, rr_ratio.values, linewidth=1.5, alpha=0.7, color='blue')
        ax.axhline(y=1.5, color='red', linestyle='--', linewidth=1, label='Threshold (1.5)')
        ax.fill_between(rr_ratio.index, 0, 1.5, alpha=0.2, color='red', label='Below Threshold')
        ax.fill_between(rr_ratio.index, 1.5, rr_ratio.max(), alpha=0.2, color='green', label='Above Threshold')
        ax.set_ylabel('Risk-Reward Ratio')
        ax.set_title('Risk-Reward Ratio Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Voted vs Executed signals
    ax = axes[1]
    if 'voted_signal' in signals_df.columns and 'executed_signal' in signals_df.columns:
        voted = signals_df['voted_signal']
        executed = signals_df['executed_signal']
        
        ax.plot(voted.index, voted.values, linewidth=1.5, alpha=0.5, color='gray', label='Voted Signal', linestyle='--')
        ax.plot(executed.index, executed.values, linewidth=2, alpha=0.8, color='blue', label='Executed Signal')
        ax.fill_between(executed.index, -1.5, 1.5, where=(executed == 1),
                       alpha=0.3, color='green', label='Executed Long')
        ax.fill_between(executed.index, -1.5, 1.5, where=(executed == -1),
                       alpha=0.3, color='red', label='Executed Short')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('Signal')
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Voted vs Executed Signals', fontsize=12, fontweight='bold')
    
    # Plot 3: Expected return and risk
    ax = axes[2]
    if 'expected_return' in signals_df.columns and 'expected_risk' in signals_df.columns:
        expected_return = signals_df['expected_return'].dropna()
        expected_risk = signals_df['expected_risk'].dropna()
        
        ax_twin = ax.twinx()
        line1 = ax.plot(expected_return.index, expected_return.values, 
                      linewidth=1.5, alpha=0.7, color='green', label='Expected Return')
        line2 = ax_twin.plot(expected_risk.index, expected_risk.values,
                           linewidth=1.5, alpha=0.7, color='red', label='Expected Risk')
        
        ax.set_ylabel('Expected Return', color='green')
        ax_twin.set_ylabel('Expected Risk', color='red')
        ax.set_xlabel('Date')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Expected Return vs Risk', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = outdir / "risk_reward_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Risk-reward analysis plot saved to: {output_path}")


def plot_ensemble_comparison(
    comparison_df: pd.DataFrame,
    outdir: Path = None,
):
    """
    Plot comparison between ensemble strategy and individual models.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with performance metrics for each model
    outdir : Path
        Output directory
    """
    if outdir is None:
        from ..config import PLOTS_DIR
        outdir = PLOTS_DIR / "strategies"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Select key metrics
    metric_cols = ['total_strategy_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    available_metrics = [col for col in metric_cols if col in comparison_df.columns]
    
    if len(available_metrics) == 0:
        print("[WARNING] No comparison metrics found")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    # Highlight ensemble
    ensemble_mask = comparison_df['model'].str.contains('Ensemble', case=False, na=False)
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Sort by metric
        sorted_df = comparison_df.sort_values(metric, ascending=False)
        
        # Plot bars
        colors = ['blue' if ensemble else 'gray' for ensemble in 
                 sorted_df['model'].str.contains('Ensemble', case=False, na=False)]
        
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors, alpha=0.7)
        
        # Labels
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['model'], fontsize=9)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = outdir / "ensemble_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Ensemble comparison plot saved to: {output_path}")


def generate_ensemble_plots(
    signals_df: pd.DataFrame,
    comparison_df: pd.DataFrame = None,
    outdir: Path = None,
):
    """
    Generate all ensemble strategy visualization plots.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        Ensemble signals DataFrame
    comparison_df : pd.DataFrame, optional
        Model comparison DataFrame
    outdir : Path, optional
        Output directory
    """
    if outdir is None:
        from ..config import PLOTS_DIR
        outdir = PLOTS_DIR / "strategies"
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("[ENSEMBLE PLOTS] Generating ensemble visualizations")
    print("=" * 70)
    
    # Voting patterns
    plot_voting_patterns(signals_df, outdir)
    
    # Risk-reward analysis
    if 'risk_reward_ratio' in signals_df.columns:
        plot_risk_reward_analysis(signals_df, outdir)
    
    # Comparison plot
    if comparison_df is not None:
        plot_ensemble_comparison(comparison_df, outdir)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All ensemble plots generated!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n=== Testing Validation Plots Module ===\n")
    print("[OK] Module loaded successfully")

