"""Visualization functions for forecasting."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")


def plot_forecast(
    actual: pd.Series,
    forecast: pd.DataFrame,
    model_name: str = "ARIMA",
    output_dir: Path = None,
    figsize=(14, 6)
):
    """
    Plot actual data with forecast and confidence intervals.
    
    Parameters
    ----------
    actual : pd.Series
        Actual historical data
    forecast : pd.DataFrame
        Forecast with columns: forecast, lower, upper
    model_name : str
        Model name for title
    output_dir : Path
        Output directory (if None, displays instead of saving)
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    ax.plot(actual.index, actual, label='Actual', linewidth=1.5, color='black')
    
    # Plot forecast
    forecast_index = pd.date_range(
        start=actual.index[-1],
        periods=len(forecast) + 1,
        freq='B'
    )[1:]  # Exclude first date (overlap with actual)
    
    ax.plot(forecast_index, forecast['forecast'], 
            label='Forecast', linewidth=2, color='blue', linestyle='--')
    
    # Plot confidence interval
    ax.fill_between(
        forecast_index,
        forecast['lower'],
        forecast['upper'],
        alpha=0.3,
        color='blue',
        label='95% CI'
    )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.set_title(f'{model_name} Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / f'forecast_{model_name.lower()}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_path}")
    else:
        plt.show()


def plot_residual_diagnostics(
    residuals: pd.Series,
    model_name: str = "ARIMA",
    output_dir: Path = None,
    figsize=(14, 10)
):
    """
    Create residual diagnostic plots.
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    model_name : str
        Model name
    output_dir : Path
        Output directory
    figsize : tuple
        Figure size
    """
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Residuals over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(residuals.index, residuals, linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{model_name} Residuals Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram with normal distribution overlay
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay normal distribution
    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax2.plot(x, p, 'r-', linewidth=2, label='Normal')
    
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Density')
    ax2.set_title('Residual Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    ax3 = fig.add_subplot(gs[1, 1])
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. ACF
    ax4 = fig.add_subplot(gs[2, 0])
    plot_acf(residuals, lags=20, ax=ax4, alpha=0.05)
    ax4.set_title('Autocorrelation Function')
    
    # 5. PACF
    ax5 = fig.add_subplot(gs[2, 1])
    plot_pacf(residuals, lags=20, ax=ax5, alpha=0.05, method='ywm')
    ax5.set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / f'forecast_residuals_{model_name.lower()}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_path}")
    else:
        plt.show()


def plot_forecast_accuracy(
    actual: pd.Series,
    forecast: pd.Series,
    metrics: dict,
    model_name: str = "Model",
    output_dir: Path = None,
    figsize=(14, 8)
):
    """
    Plot forecast accuracy metrics and comparison.
    
    Parameters
    ----------
    actual : pd.Series
        Actual values
    forecast : pd.Series
        Forecasted values
    metrics : dict
        Accuracy metrics (MAE, RMSE, MAPE, etc.)
    model_name : str
        Model name
    output_dir : Path
        Output directory
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name} Forecast Accuracy', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Forecast
    ax1 = axes[0, 0]
    ax1.scatter(actual, forecast, alpha=0.5, s=20)
    
    # Add diagonal line (perfect forecast)
    min_val = min(actual.min(), forecast.min())
    max_val = max(actual.max(), forecast.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Forecast')
    
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Forecast')
    ax1.set_title('Actual vs Forecast')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Forecast errors
    ax2 = axes[0, 1]
    errors = actual - forecast
    ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(errors.mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {errors.mean():.4f}')
    ax2.set_xlabel('Forecast Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Time series comparison
    ax3 = axes[1, 0]
    ax3.plot(actual.index, actual, label='Actual', linewidth=1.5, alpha=0.7)
    ax3.plot(forecast.index, forecast, label='Forecast', 
             linewidth=1.5, alpha=0.7, linestyle='--')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.set_title('Actual vs Forecast Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    metrics_text = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics_text.append(f"{key}: {value:.4f}")
        else:
            metrics_text.append(f"{key}: {value}")
    
    y_pos = 0.9
    for text in metrics_text:
        ax4.text(0.1, y_pos, text, fontsize=11, verticalalignment='top',
                 fontfamily='monospace')
        y_pos -= 0.12
    
    ax4.set_title('Accuracy Metrics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / f'forecast_accuracy_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_path}")
    else:
        plt.show()

