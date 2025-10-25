"""Synthetic index module: OLS regression and z-score calculation."""

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .config import ROLL_Z, Z_THRESHOLD


def fit_synthetic(
    df: pd.DataFrame, target_col: str = "suzb_r"
) -> tuple[RegressionResultsWrapper, pd.Series, pd.Series, pd.Series]:
    """
    Fit synthetic index using OLS regression and compute z-scores.
    
    Process:
    1. Select all return features (*_r columns except target)
    2. Run OLS regression: target ~ const + features
    3. Calculate synthetic index = X @ betas
    4. Compute spread = target - synthetic
    5. Calculate rolling z-score (window = ROLL_Z)
    6. Generate signals where |z| > Z_THRESHOLD
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with return columns
    target_col : str
        Target column name (default: 'suzb_r')
    
    Returns
    -------
    model : RegressionResultsWrapper
        Fitted OLS model
    synthetic : pd.Series
        Synthetic index predictions
    zscore : pd.Series
        Z-scores of spread
    signals : pd.Series
        Trading signals (1: long, -1: short, 0: neutral)
    """
    print("\n" + "=" * 60)
    print("FITTING SYNTHETIC INDEX")
    print("=" * 60 + "\n")
    
    # Select feature columns (all _r columns except target)
    feature_cols = [
        col
        for col in df.columns
        if col.endswith("_r") and col != target_col
    ]
    
    print(f"📊 Target: {target_col}")
    print(f"📊 Features ({len(feature_cols)}): {feature_cols}\n")
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Drop NaN values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"✓ Valid observations: {len(X)}")
    print(f"✓ Date range: {X.index.min()} to {X.index.max()}\n")
    
    # Add constant
    X_with_const = add_constant(X)
    
    # Fit OLS
    print("🔧 Fitting OLS regression...")
    model = OLS(y, X_with_const).fit()
    
    print("✓ OLS fitted successfully\n")
    print("=" * 60)
    print("REGRESSION SUMMARY")
    print("=" * 60)
    print(model.summary())
    
    # Calculate synthetic index
    print("\n📈 Calculating synthetic index...")
    synthetic = pd.Series(
        model.predict(X_with_const),
        index=X.index,
        name="synthetic_index",
    )
    print(f"✓ Synthetic index computed ({len(synthetic)} values)")
    
    # Compute spread
    print("\n📉 Computing spread (actual - synthetic)...")
    spread = y.loc[synthetic.index] - synthetic
    spread.name = "spread"
    print(f"✓ Spread computed (mean: {spread.mean():.6f}, std: {spread.std():.6f})")
    
    # Calculate rolling z-score
    print(f"\n📊 Calculating rolling z-score (window={ROLL_Z})...")
    
    rolling_mean = spread.rolling(window=ROLL_Z, min_periods=ROLL_Z // 2).mean()
    rolling_std = spread.rolling(window=ROLL_Z, min_periods=ROLL_Z // 2).std()
    
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    
    # Remove infinite values
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    
    valid_zscores = zscore.notna().sum()
    print(f"✓ Z-scores computed ({valid_zscores} valid values)")
    print(f"  Min: {zscore.min():.2f}")
    print(f"  Max: {zscore.max():.2f}")
    print(f"  Mean: {zscore.mean():.2f}")
    print(f"  Std: {zscore.std():.2f}")
    
    # Generate signals
    print(f"\n🎯 Generating trading signals (threshold={Z_THRESHOLD})...")
    
    signals = pd.Series(0, index=zscore.index, name="signal")
    signals[zscore > Z_THRESHOLD] = -1  # Short (overvalued)
    signals[zscore < -Z_THRESHOLD] = 1  # Long (undervalued)
    
    long_signals = (signals == 1).sum()
    short_signals = (signals == -1).sum()
    neutral = (signals == 0).sum()
    
    print(f"✓ Signals generated:")
    print(f"  Long (z < -{Z_THRESHOLD}): {long_signals} ({100*long_signals/len(signals):.1f}%)")
    print(f"  Short (z > {Z_THRESHOLD}): {short_signals} ({100*short_signals/len(signals):.1f}%)")
    print(f"  Neutral: {neutral} ({100*neutral/len(signals):.1f}%)")
    
    return model, synthetic, zscore, signals


def analyze_coefficients(model: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Analyze regression coefficients with significance tests.
    
    Parameters
    ----------
    model : RegressionResultsWrapper
        Fitted OLS model
    
    Returns
    -------
    pd.DataFrame
        DataFrame with coefficients, std errors, t-stats, p-values
    """
    coef_df = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "t_statistic": model.tvalues,
            "p_value": model.pvalues,
            "significant": model.pvalues < 0.05,
        }
    )
    
    return coef_df


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    returns_col: str = "suzb_r",
    holding_period: int = 1,
) -> pd.DataFrame:
    """
    Simple backtest of trading signals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with returns
    signals : pd.Series
        Trading signals (-1, 0, 1)
    returns_col : str
        Column name for returns
    holding_period : int
        Number of periods to hold position
    
    Returns
    -------
    pd.DataFrame
        Backtest results with strategy returns
    """
    # Align signals and returns
    aligned = pd.DataFrame(
        {
            "signal": signals,
            "returns": df[returns_col],
        }
    ).dropna()
    
    # Shift signals to avoid look-ahead bias
    aligned["signal_shifted"] = aligned["signal"].shift(holding_period)
    
    # Calculate strategy returns
    aligned["strategy_returns"] = aligned["signal_shifted"] * aligned["returns"]
    
    # Cumulative returns
    aligned["cumulative_market"] = (1 + aligned["returns"]).cumprod() - 1
    aligned["cumulative_strategy"] = (1 + aligned["strategy_returns"].fillna(0)).cumprod() - 1
    
    return aligned


if __name__ == "__main__":
    # Test synthetic index
    print("\n=== Testing Synthetic Index ===\n")
    
    from .features import build_features
    
    try:
        # Build features
        df = build_features()
        
        # Fit synthetic index
        model, synthetic, zscore, signals = fit_synthetic(df)
        
        # Analyze coefficients
        print("\n" + "=" * 60)
        print("COEFFICIENT ANALYSIS")
        print("=" * 60)
        coef_analysis = analyze_coefficients(model)
        print(coef_analysis)
        
        # Simple backtest
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        backtest = backtest_signals(df, signals)
        
        total_return = backtest["cumulative_strategy"].iloc[-1]
        market_return = backtest["cumulative_market"].iloc[-1]
        
        print(f"Market return: {100*market_return:.2f}%")
        print(f"Strategy return: {100*total_return:.2f}%")
        print(f"Excess return: {100*(total_return - market_return):.2f}%")
        
        print("\n✓ Synthetic index test successful!")
        
    except Exception as e:
        print(f"❌ Synthetic index test failed: {e}")
        raise

