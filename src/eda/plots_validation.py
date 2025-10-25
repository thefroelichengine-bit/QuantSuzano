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


if __name__ == "__main__":
    print("\n=== Testing Validation Plots Module ===\n")
    print("[OK] Module loaded successfully")

