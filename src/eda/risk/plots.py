"""Visualization functions for risk analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

sns.set_style("whitegrid")


def plot_volatility_analysis(
    returns: pd.Series,
    vol_21d: pd.Series,
    vol_252d: pd.Series,
    output_dir: Path,
    figsize=(14, 10)
):
    """
    Create comprehensive volatility analysis plots.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    vol_21d : pd.Series
        21-day rolling volatility
    vol_252d : pd.Series
        252-day rolling volatility
    output_dir : Path
        Output directory
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # 1. Returns over time
    axes[0].plot(returns.index, returns * 100, linewidth=0.5, alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('Returns (%)')
    axes[0].set_title('Daily Returns')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Rolling volatility
    axes[1].plot(vol_21d.index, vol_21d * 100, label='21-day', linewidth=1.5)
    axes[1].plot(vol_252d.index, vol_252d * 100, label='252-day', linewidth=1.5)
    axes[1].set_ylabel('Volatility (% ann.)')
    axes[1].set_title('Rolling Volatility')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Volatility distribution
    vol_clean = vol_21d.dropna() * 100
    axes[2].hist(vol_clean, bins=50, alpha=0.7, edgecolor='black')
    axes[2].axvline(vol_clean.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {vol_clean.mean():.1f}%')
    axes[2].axvline(vol_clean.median(), color='blue', linestyle='--', 
                    linewidth=2, label=f'Median: {vol_clean.median():.1f}%')
    axes[2].set_xlabel('Volatility (% ann.)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Volatility Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'risk_volatility_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_path}")


def plot_var_analysis(
    returns: pd.Series,
    var_95: float,
    cvar_95: float,
    var_99: float,
    cvar_99: float,
    output_dir: Path,
    figsize=(14, 6)
):
    """
    Plot VaR and CVaR analysis.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    var_95, cvar_95 : float
        95% VaR and CVaR
    var_99, cvar_99 : float
        99% VaR and CVaR
    output_dir : Path
        Output directory
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    returns_pct = returns * 100
    
    # 1. Returns distribution with VaR markers
    ax1.hist(returns_pct, bins=50, alpha=0.7, edgecolor='black', density=True)
    
    ax1.axvline(-var_95 * 100, color='orange', linestyle='--', linewidth=2,
                label=f'VaR 95%: {var_95*100:.2f}%')
    ax1.axvline(-cvar_95 * 100, color='red', linestyle='--', linewidth=2,
                label=f'CVaR 95%: {cvar_95*100:.2f}%')
    ax1.axvline(-var_99 * 100, color='darkred', linestyle=':', linewidth=2,
                label=f'VaR 99%: {var_99*100:.2f}%')
    
    ax1.set_xlabel('Returns (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Returns Distribution with VaR Levels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart comparison
    metrics = ['VaR 95%', 'CVaR 95%', 'VaR 99%', 'CVaR 99%']
    values = [var_95 * 100, cvar_95 * 100, var_99 * 100, cvar_99 * 100]
    colors = ['orange', 'red', 'darkred', 'firebrick']
    
    ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Loss (%)')
    ax2.set_title('Value at Risk Metrics')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate labels
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / 'risk_var_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_path}")


def plot_drawdown_analysis(
    returns: pd.Series,
    output_dir: Path,
    figsize=(14, 10)
):
    """
    Create comprehensive drawdown visualizations.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    output_dir : Path
        Output directory
    figsize : tuple
        Figure size
    """
    from .drawdowns import calculate_drawdowns, calculate_drawdown_stats
    
    drawdowns = calculate_drawdowns(returns) * 100  # Convert to %
    dd_stats = calculate_drawdown_stats(returns)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    cum_returns = (1 + returns).cumprod()
    ax1.plot(cum_returns.index, cum_returns, linewidth=1.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Cumulative Returns')
    ax1.grid(True, alpha=0.3)
    
    # 2. Underwater plot
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdowns.index, drawdowns.values, 0,
                      where=(drawdowns < 0), color='red', alpha=0.3)
    ax2.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Mark max drawdown
    mdd_idx = drawdowns.idxmin()
    mdd_val = drawdowns.min()
    ax2.plot(mdd_idx, mdd_val, 'ro', markersize=10,
             label=f'Max DD: {abs(mdd_val):.2f}%')
    
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Underwater Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown distribution
    ax3 = fig.add_subplot(gs[2, 0])
    dd_negative = drawdowns[drawdowns < 0]
    ax3.hist(dd_negative, bins=30, alpha=0.7, edgecolor='black', color='red')
    ax3.axvline(dd_negative.mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {dd_negative.mean():.2f}%')
    ax3.set_xlabel('Drawdown (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Drawdown Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown statistics table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    stats_text = [
        f"Max Drawdown: {dd_stats['max_drawdown']*100:.2f}%",
        f"Max DD Date: {dd_stats['max_drawdown_date'].strftime('%Y-%m-%d')}",
        f"Average DD: {dd_stats['avg_drawdown']*100:.2f}%",
        f"Current DD: {dd_stats['current_drawdown']*100:.2f}%",
        f"Num Drawdowns: {dd_stats['num_drawdowns']}",
    ]
    
    if dd_stats['recovery_days'] is not None:
        stats_text.append(f"Recovery Days: {dd_stats['recovery_days']}")
    
    y_pos = 0.9
    for text in stats_text:
        ax4.text(0.1, y_pos, text, fontsize=11, verticalalignment='top',
                 fontfamily='monospace')
        y_pos -= 0.15
    
    ax4.set_title('Drawdown Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'risk_drawdown_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_path}")


def plot_risk_metrics_summary(
    metrics: dict,
    output_dir: Path,
    figsize=(12, 8)
):
    """
    Create summary dashboard of risk metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of risk metrics
    output_dir : Path
        Output directory
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Risk Metrics Summary', fontsize=16, fontweight='bold')
    
    # 1. Risk-adjusted ratios
    ax1 = axes[0, 0]
    ratios = ['Sharpe', 'Sortino', 'Calmar', 'Omega']
    ratio_values = [
        metrics.get('sharpe_ratio', 0),
        metrics.get('sortino_ratio', 0),
        metrics.get('calmar_ratio', 0),
        metrics.get('omega_ratio', 0),
    ]
    colors = ['green' if v > 0 else 'red' for v in ratio_values]
    ax1.barh(ratios, ratio_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Ratio Value')
    ax1.set_title('Risk-Adjusted Ratios')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Return metrics
    ax2 = axes[0, 1]
    returns_data = [
        ('Total Return', metrics.get('total_return', 0) * 100),
        ('Ann. Return', metrics.get('annualized_return', 0) * 100),
        ('Ann. Vol', metrics.get('annualized_volatility', 0) * 100),
    ]
    labels, values = zip(*returns_data)
    ax2.bar(labels, values, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Percent (%)')
    ax2.set_title('Return & Volatility')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Downside risk
    ax3 = axes[1, 0]
    downside_data = [
        ('Max DD', metrics.get('max_drawdown', 0) * 100),
        ('VaR 95%', metrics.get('var_95', 0) * 100),
        ('VaR 99%', metrics.get('var_99', 0) * 100),
    ]
    labels, values = zip(*downside_data)
    ax3.bar(labels, values, color='red', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Loss (%)')
    ax3.set_title('Downside Risk Metrics')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Distribution stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    dist_text = [
        f"Skewness: {metrics.get('skewness', 0):.3f}",
        f"Kurtosis: {metrics.get('kurtosis', 0):.3f}",
        f"Downside Dev: {metrics.get('downside_deviation', 0)*100:.2f}%",
    ]
    
    y_pos = 0.8
    for text in dist_text:
        ax4.text(0.1, y_pos, text, fontsize=12, verticalalignment='top',
                 fontfamily='monospace')
        y_pos -= 0.25
    
    ax4.set_title('Distribution Statistics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'risk_metrics_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_path}")

