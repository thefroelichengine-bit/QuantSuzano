"""
Comprehensive exploratory data analysis module.
Generates 30+ plots organized in subfolders.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from pathlib import Path
from typing import List, Optional
from .config import DATA_OUT

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_distributions(df: pd.DataFrame, output_dir: Path = None):
    """Generate distribution plots for all numeric variables."""
    if output_dir is None:
        output_dir = DATA_OUT / "plots/eda/univariate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA] Generating distribution plots...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Histogram with KDE
        axes[0].hist(df[col].dropna(), bins=50, density=True, alpha=0.7, edgecolor='black')
        df[col].plot(kind='kde', ax=axes[0], color='red', linewidth=2)
        axes[0].set_title(f'{col} - Distribution')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Density')
        
        # Q-Q plot
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title(f'{col} - Q-Q Plot')
        
        # Box plot
        axes[2].boxplot(df[col].dropna(), vert=True)
        axes[2].set_title(f'{col} - Box Plot')
        axes[2].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"dist_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(numeric_cols)} distribution plots to {output_dir}")


def plot_correlations(df: pd.DataFrame, output_dir: Path = None, windows: List[int] = [60, 120, 252]):
    """Generate correlation heatmaps and rolling correlations."""
    if output_dir is None:
        output_dir = DATA_OUT / "plots/eda/correlation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA] Generating correlation analysis...")
    
    # Overall correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix (Full Sample)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Rolling correlations
    if 'suzb_r' in df.columns:
        other_cols = [col for col in df.columns if col != 'suzb_r' and col.endswith('_r')]
        
        for window in windows:
            fig, axes = plt.subplots(len(other_cols), 1, figsize=(14, 3*len(other_cols)))
            if len(other_cols) == 1:
                axes = [axes]
            
            for idx, col in enumerate(other_cols):
                rolling_corr = df['suzb_r'].rolling(window=window).corr(df[col])
                axes[idx].plot(rolling_corr.index, rolling_corr, linewidth=1.5)
                axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
                axes[idx].set_title(f'Rolling Correlation: suzb_r vs {col} ({window} days)')
                axes[idx].set_ylabel('Correlation')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"rolling_corr_{window}d.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  Saved correlation plots to {output_dir}")


def plot_temporal_patterns(df: pd.DataFrame, output_dir: Path = None):
    """Analyze temporal patterns (seasonality, autocorrelation)."""
    if output_dir is None:
        output_dir = DATA_OUT / "plots/eda/temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA] Analyzing temporal patterns...")
    
    # ACF and PACF for return series
    return_cols = [col for col in df.columns if col.endswith('_r')]
    
    for col in return_cols:
        data = df[col].dropna()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        plot_acf(data, lags=40, ax=axes[0])
        axes[0].set_title(f'{col} - Autocorrelation Function (ACF)')
        
        plot_pacf(data, lags=40, ax=axes[1])
        axes[1].set_title(f'{col} - Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"acf_pacf_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # STL decomposition for price series
    price_cols = ['suzb', 'ibov'] if 'ibov' in df.columns else ['suzb']
    
    for col in price_cols:
        if col in df.columns and len(df[col].dropna()) > 365:
            data = df[col].dropna()
            
            try:
                stl = STL(data, period=252)  # Yearly seasonality
                result = stl.fit()
                
                fig, axes = plt.subplots(4, 1, figsize=(14, 10))
                
                axes[0].plot(data.index, data, linewidth=1)
                axes[0].set_title(f'{col} - Original Series')
                axes[0].set_ylabel('Price')
                
                axes[1].plot(result.trend.index, result.trend, linewidth=1.5, color='red')
                axes[1].set_title('Trend Component')
                axes[1].set_ylabel('Trend')
                
                axes[2].plot(result.seasonal.index, result.seasonal, linewidth=1, color='green')
                axes[2].set_title('Seasonal Component')
                axes[2].set_ylabel('Seasonal')
                
                axes[3].plot(result.resid.index, result.resid, linewidth=0.5, color='gray')
                axes[3].set_title('Residual Component')
                axes[3].set_ylabel('Residual')
                axes[3].set_xlabel('Date')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"stl_decomposition_{col}.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not perform STL decomposition for {col}: {e}")
    
    print(f"  Saved temporal pattern plots to {output_dir}")


def plot_bivariate_relationships(df: pd.DataFrame, target: str = 'suzb_r', output_dir: Path = None):
    """Generate scatter plots for bivariate relationships."""
    if output_dir is None:
        output_dir = DATA_OUT / "plots/eda/bivariate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[EDA] Analyzing bivariate relationships...")
    
    if target not in df.columns:
        print(f"  Warning: Target variable '{target}' not found")
        return
    
    feature_cols = [col for col in df.columns if col != target and col.endswith('_r')]
    
    # Individual scatter plots
    for col in feature_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter with regression line
        x = df[col].dropna()
        y = df[target].reindex(x.index).dropna()
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        ax.scatter(x, y, alpha=0.5, s=20)
        
        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x.sort_values(), p(x.sort_values()), "r--", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Calculate correlation
        corr = x.corr(y)
        ax.set_title(f'{target} vs {col} (corr={corr:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{target}_vs_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Scatter matrix for top features
    if len(feature_cols) > 0:
        top_features = feature_cols[:min(4, len(feature_cols))]
        scatter_cols = [target] + top_features
        
        fig = plt.figure(figsize=(14, 14))
        axes = pd.plotting.scatter_matrix(df[scatter_cols], alpha=0.5, figsize=(14, 14), diagonal='kde')
        
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(45)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        
        plt.suptitle('Scatter Matrix - Top Features', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved bivariate relationship plots to {output_dir}")


def run_comprehensive_eda(df: pd.DataFrame):
    """
    Run complete exploratory data analysis with all plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with datetime index
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Generate all plot categories
    plot_distributions(df)
    plot_correlations(df, windows=[60, 120, 252])
    plot_temporal_patterns(df)
    plot_bivariate_relationships(df, target='suzb_r')
    
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS COMPLETE")
    print(f"Plots saved to: {DATA_OUT / 'plots/eda/'}")
    print("=" * 70)

