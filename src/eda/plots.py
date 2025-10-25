"""Visualization module with matplotlib plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 10


def plot_levels(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot time series of level variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    outdir : Path
        Output directory for plots
    """
    print("\n📊 Plotting levels...")
    
    level_cols = ["ptax", "selic", "pulp_brl", "pulp_usd", "suzb", "credit"]
    existing_cols = [col for col in level_cols if col in df.columns]
    
    if not existing_cols:
        print("⚠ No level columns found")
        return
    
    fig, axes = plt.subplots(len(existing_cols), 1, figsize=(14, 3 * len(existing_cols)))
    
    if len(existing_cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, existing_cols):
        df[col].dropna().plot(ax=ax, title=f"{col.upper()} - Levels", color="steelblue")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "levels.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / 'levels.png'}")


def plot_returns(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot time series of log returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    outdir : Path
        Output directory for plots
    """
    print("\n📊 Plotting returns...")
    
    return_cols = [col for col in df.columns if col.endswith("_r")]
    
    if not return_cols:
        print("⚠ No return columns found")
        return
    
    fig, axes = plt.subplots(len(return_cols), 1, figsize=(14, 2.5 * len(return_cols)))
    
    if len(return_cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, return_cols):
        series = df[col].dropna()
        series.plot(ax=ax, title=f"{col.upper()} - Log Returns", color="darkorange", alpha=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / 'returns.png'}")


def plot_correlation_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot correlation heatmap of return variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    outdir : Path
        Output directory for plots
    """
    print("\n📊 Plotting correlation heatmap...")
    
    return_cols = [col for col in df.columns if col.endswith("_r")]
    
    if len(return_cols) < 2:
        print("⚠ Not enough return columns for correlation")
        return
    
    corr = df[return_cols].corr()
    
    # Save correlation matrix to CSV
    corr.to_csv(outdir / "correlation_matrix.csv")
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    
    # Set ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=20)
    
    # Annotate cells
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text = ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                fontsize=8,
            )
    
    ax.set_title("Correlation Matrix - Log Returns")
    plt.tight_layout()
    plt.savefig(outdir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / 'correlation_heatmap.png'}")
    print(f"✓ Saved: {outdir / 'correlation_matrix.csv'}")


def plot_rolling_correlation(
    df: pd.DataFrame, outdir: Path, col1: str = "suzb_r", col2: str = "pulp_brl_r", window: int = 60
) -> None:
    """
    Plot rolling correlation between two variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    outdir : Path
        Output directory for plots
    col1 : str
        First column name
    col2 : str
        Second column name
    window : int
        Rolling window size
    """
    print(f"\n📊 Plotting rolling correlation ({col1} vs {col2})...")
    
    if col1 not in df.columns or col2 not in df.columns:
        print(f"⚠ Columns {col1} or {col2} not found")
        return
    
    # Calculate rolling correlation
    rolling_corr = df[col1].rolling(window=window).corr(df[col2])
    
    fig, ax = plt.subplots(figsize=(14, 5))
    rolling_corr.plot(ax=ax, color="purple", linewidth=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling {window}-Day Correlation: {col1} vs {col2}")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(outdir / f"rolling_corr_{col1}_{col2}.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / f'rolling_corr_{col1}_{col2}.png'}")


def plot_synthetic_vs_actual(
    df: pd.DataFrame,
    synthetic: pd.Series,
    zscore: pd.Series,
    outdir: Path,
    target_col: str = "suzb_r",
) -> None:
    """
    Plot actual vs synthetic index with z-score bands.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    synthetic : pd.Series
        Synthetic index
    zscore : pd.Series
        Z-scores
    outdir : Path
        Output directory for plots
    target_col : str
        Target column name
    """
    print("\n📊 Plotting synthetic vs actual...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top panel: Actual vs Synthetic
    aligned_idx = df.index.intersection(synthetic.index)
    df.loc[aligned_idx, target_col].plot(
        ax=ax1, label="Actual", color="steelblue", linewidth=1.5
    )
    synthetic.loc[aligned_idx].plot(
        ax=ax1, label="Synthetic Index", color="darkorange", linewidth=1.5
    )
    ax1.set_title(f"{target_col.upper()}: Actual vs Synthetic Index")
    ax1.set_ylabel("Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Z-score
    zscore.plot(ax=ax2, color="darkgreen", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.axhline(2, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Threshold (+2)")
    ax2.axhline(-2, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Threshold (-2)")
    ax2.fill_between(zscore.index, 2, zscore, where=(zscore > 2), color="red", alpha=0.3)
    ax2.fill_between(zscore.index, -2, zscore, where=(zscore < -2), color="red", alpha=0.3)
    ax2.set_title("Z-Score (Spread / Rolling Std)")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "synthetic_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / 'synthetic_vs_actual.png'}")


def plot_signals(df: pd.DataFrame, signals: pd.Series, outdir: Path) -> None:
    """
    Plot trading signals on price chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    signals : pd.Series
        Trading signals
    outdir : Path
        Output directory for plots
    """
    print("\n📊 Plotting trading signals...")
    
    if "suzb" not in df.columns:
        print("⚠ Column 'suzb' not found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot price
    aligned_idx = df.index.intersection(signals.index)
    df.loc[aligned_idx, "suzb"].plot(ax=ax, color="steelblue", linewidth=2, label="SUZB3 Price")
    
    # Mark signals
    long_signals = signals[signals == 1]
    short_signals = signals[signals == -1]
    
    if len(long_signals) > 0:
        ax.scatter(
            long_signals.index,
            df.loc[long_signals.index, "suzb"],
            color="green",
            marker="^",
            s=100,
            label=f"Long Signal ({len(long_signals)})",
            zorder=5,
        )
    
    if len(short_signals) > 0:
        ax.scatter(
            short_signals.index,
            df.loc[short_signals.index, "suzb"],
            color="red",
            marker="v",
            s=100,
            label=f"Short Signal ({len(short_signals)})",
            zorder=5,
        )
    
    ax.set_title("Trading Signals on SUZB3 Price")
    ax.set_ylabel("Price (BRL)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "signals.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / 'signals.png'}")


def plot_distribution(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot distribution histograms of returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    outdir : Path
        Output directory for plots
    """
    print("\n📊 Plotting distributions...")
    
    return_cols = [col for col in df.columns if col.endswith("_r")]
    
    if not return_cols:
        print("⚠ No return columns found")
        return
    
    fig, axes = plt.subplots(
        len(return_cols), 1, figsize=(12, 3 * len(return_cols))
    )
    
    if len(return_cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, return_cols):
        series = df[col].dropna()
        
        ax.hist(series, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(series.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {series.mean():.4f}")
        ax.axvline(series.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {series.median():.4f}")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved: {outdir / 'distributions.png'}")


def generate_all_plots(
    df: pd.DataFrame,
    synthetic: pd.Series = None,
    zscore: pd.Series = None,
    signals: pd.Series = None,
    outdir: Path = None,
) -> None:
    """
    Generate all standard plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix
    synthetic : pd.Series, optional
        Synthetic index
    zscore : pd.Series, optional
        Z-scores
    signals : pd.Series, optional
        Trading signals
    outdir : Path, optional
        Output directory
    """
    from .config import PLOTS_DIR
    
    if outdir is None:
        outdir = PLOTS_DIR
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("GENERATING ALL PLOTS")
    print("=" * 60)
    
    plot_levels(df, outdir)
    plot_returns(df, outdir)
    plot_distribution(df, outdir)
    plot_correlation_heatmap(df, outdir)
    plot_rolling_correlation(df, outdir, col1="suzb_r", col2="pulp_brl_r")
    
    if synthetic is not None and zscore is not None:
        plot_synthetic_vs_actual(df, synthetic, zscore, outdir)
    
    if signals is not None:
        plot_signals(df, signals, outdir)
    
    print("\n" + "=" * 60)
    print(f"✓ All plots saved to: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    # Test plots
    print("\n=== Testing Plot Generation ===\n")
    
    from .features import build_features
    from .synthetic import fit_synthetic
    
    try:
        # Build features
        df = build_features()
        
        # Fit synthetic
        model, synthetic, zscore, signals = fit_synthetic(df)
        
        # Generate all plots
        generate_all_plots(df, synthetic, zscore, signals)
        
        print("\n✓ Plot generation successful!")
        
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
        raise

