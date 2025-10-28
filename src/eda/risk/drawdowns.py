"""Drawdown analysis and visualization."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def calculate_drawdowns(
    returns: pd.Series,
    cumulative: bool = False
) -> pd.Series:
    """
    Calculate drawdown series from returns.
    
    Parameters
    ----------
    returns : pd.Series
        Return series (or cumulative returns if cumulative=True)
    cumulative : bool
        Whether input is already cumulative returns
    
    Returns
    -------
    pd.Series
        Drawdown series (negative values)
    """
    if not cumulative:
        # Convert to cumulative wealth index
        wealth_index = (1 + returns).cumprod()
    else:
        wealth_index = 1 + returns
    
    # Running maximum
    running_max = wealth_index.cummax()
    
    # Drawdown
    drawdown = (wealth_index - running_max) / running_max
    
    return drawdown


def calculate_max_drawdown(
    returns: pd.Series,
    cumulative: bool = False
) -> float:
    """
    Calculate maximum drawdown (MDD).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    cumulative : bool
        Whether input is cumulative
    
    Returns
    -------
    float
        Maximum drawdown (positive number)
    """
    drawdowns = calculate_drawdowns(returns, cumulative)
    mdd = drawdowns.min()  # Most negative value
    return abs(mdd)


def calculate_drawdown_stats(
    returns: pd.Series,
    cumulative: bool = False
) -> dict:
    """
    Calculate comprehensive drawdown statistics.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    cumulative : bool
        Whether input is cumulative
    
    Returns
    -------
    dict
        Drawdown statistics
    """
    drawdowns = calculate_drawdowns(returns, cumulative)
    
    # Maximum drawdown
    mdd = abs(drawdowns.min())
    mdd_date = drawdowns.idxmin()
    
    # Average drawdown
    avg_dd = abs(drawdowns[drawdowns < 0].mean())
    
    # Median drawdown
    median_dd = abs(drawdowns[drawdowns < 0].median())
    
    # Number of drawdowns (periods below previous peak)
    in_drawdown = (drawdowns < 0).astype(int)
    dd_starts = (in_drawdown.diff() == 1).sum()
    
    # Current drawdown
    current_dd = abs(drawdowns.iloc[-1])
    
    # Recovery time (for max drawdown)
    if mdd_date in drawdowns.index:
        after_mdd = drawdowns.loc[mdd_date:]
        recovery_mask = after_mdd >= 0
        if recovery_mask.any():
            recovery_date = after_mdd[recovery_mask].index[0]
            recovery_days = (recovery_date - mdd_date).days
        else:
            recovery_date = None
            recovery_days = None  # Not yet recovered
    else:
        recovery_date = None
        recovery_days = None
    
    return {
        'max_drawdown': mdd,
        'max_drawdown_date': mdd_date,
        'recovery_date': recovery_date,
        'recovery_days': recovery_days,
        'avg_drawdown': avg_dd,
        'median_drawdown': median_dd,
        'num_drawdowns': dd_starts,
        'current_drawdown': current_dd,
    }


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    periods_per_year : int
        Periods per year for annualization
    
    Returns
    -------
    float
        Calmar ratio
    """
    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Max drawdown
    mdd = calculate_max_drawdown(returns)
    
    # Calmar ratio
    if mdd > 0:
        calmar = annualized_return / mdd
    else:
        calmar = np.nan
    
    return calmar


def identify_drawdown_periods(
    returns: pd.Series,
    cumulative: bool = False
) -> pd.DataFrame:
    """
    Identify all drawdown periods with start, trough, end dates.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    cumulative : bool
        Whether input is cumulative
    
    Returns
    -------
    pd.DataFrame
        Drawdown periods with columns: start, trough, end, depth, duration
    """
    drawdowns = calculate_drawdowns(returns, cumulative)
    
    # Identify drawdown periods
    in_drawdown = (drawdowns < 0).astype(int)
    dd_changes = in_drawdown.diff().fillna(0)
    
    periods = []
    start_idx = None
    
    for i, (date, change) in enumerate(dd_changes.items()):
        if change == 1:  # Start of drawdown
            start_idx = i
            start_date = date
        elif change == -1 and start_idx is not None:  # End of drawdown
            end_date = date
            dd_period = drawdowns.iloc[start_idx:i]
            
            trough_idx = dd_period.idxmin()
            trough_value = dd_period.min()
            duration = i - start_idx
            
            periods.append({
                'start': start_date,
                'trough': trough_idx,
                'end': end_date,
                'depth': abs(trough_value),
                'duration': duration,
            })
            
            start_idx = None
    
    # Handle if still in drawdown at end
    if start_idx is not None:
        dd_period = drawdowns.iloc[start_idx:]
        trough_idx = dd_period.idxmin()
        trough_value = dd_period.min()
        duration = len(drawdowns) - start_idx
        
        periods.append({
            'start': drawdowns.index[start_idx],
            'trough': trough_idx,
            'end': None,  # Ongoing
            'depth': abs(trough_value),
            'duration': duration,
        })
    
    return pd.DataFrame(periods)


def underwater_plot(
    returns: pd.Series,
    title: str = "Underwater Plot",
    figsize: Tuple = (12, 5)
) -> plt.Figure:
    """
    Create underwater plot showing drawdowns over time.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    drawdowns = calculate_drawdowns(returns) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(drawdowns.index, drawdowns.values, 0, 
                     where=(drawdowns < 0), color='red', alpha=0.3,
                     label='Drawdown')
    ax.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add max drawdown marker
    mdd_idx = drawdowns.idxmin()
    mdd_value = drawdowns.min()
    ax.plot(mdd_idx, mdd_value, 'ro', markersize=8, 
            label=f'Max DD: {abs(mdd_value):.2f}%')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_drawdown_distribution(
    returns: pd.Series,
    bins: int = 50,
    figsize: Tuple = (12, 5)
) -> plt.Figure:
    """
    Plot drawdown distribution histogram.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    drawdowns = calculate_drawdowns(returns) * 100
    dd_negative = drawdowns[drawdowns < 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(dd_negative, bins=bins, color='red', alpha=0.7, edgecolor='black')
    ax1.axvline(dd_negative.mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {dd_negative.mean():.2f}%')
    ax1.axvline(dd_negative.min(), color='darkred', linestyle='--', 
                linewidth=2, label=f'Max DD: {dd_negative.min():.2f}%')
    ax1.set_xlabel('Drawdown (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Drawdown Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    ax2.boxplot(dd_negative, vert=True)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Statistics')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


