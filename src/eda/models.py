"""Time series models: VECM, Johansen cointegration, and ADF tests."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

warnings.filterwarnings("ignore", category=FutureWarning)


def run_adf_tests(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller tests for stationarity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series
    cols : list, optional
        List of columns to test. If None, test all numeric columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ADF test results
    """
    print("\n" + "=" * 60)
    print("AUGMENTED DICKEY-FULLER TESTS")
    print("=" * 60 + "\n")
    
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for col in cols:
        if col not in df.columns:
            print(f"⚠ Column '{col}' not found, skipping...")
            continue
        
        series = df[col].dropna()
        
        if len(series) < 10:
            print(f"⚠ Column '{col}' has too few observations ({len(series)}), skipping...")
            continue
        
        try:
            # Run ADF test
            adf_result = adfuller(series, maxlag=12, regression="c", autolag="AIC")
            
            statistic, pvalue, usedlag, nobs = adf_result[:4]
            critical_values = adf_result[4]
            
            # Determine stationarity
            is_stationary = pvalue < 0.05
            
            results.append(
                {
                    "column": col,
                    "observations": nobs,
                    "adf_statistic": statistic,
                    "p_value": pvalue,
                    "lags_used": usedlag,
                    "critical_1%": critical_values["1%"],
                    "critical_5%": critical_values["5%"],
                    "critical_10%": critical_values["10%"],
                    "stationary": is_stationary,
                }
            )
            
            status = "✓ Stationary" if is_stationary else "✗ Non-stationary"
            print(f"{col:20s}: {status} (ADF={statistic:.4f}, p={pvalue:.4f})")
        
        except Exception as e:
            print(f"❌ Error testing '{col}': {e}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print(f"Summary: {results_df['stationary'].sum()}/{len(results_df)} series are stationary")
    print("=" * 60)
    
    return results_df


def fit_vecm(
    df: pd.DataFrame,
    cols: tuple = ("pulp_brl", "suzb"),
    k_ar_diff: int = 2,
    coint_rank: int = 1,
    deterministic: str = "ci",
) -> tuple:
    """
    Fit Vector Error Correction Model (VECM) with Johansen cointegration test.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series
    cols : tuple
        Columns to include in VECM system
    k_ar_diff : int
        Number of lagged differences in the VECM
    coint_rank : int
        Cointegration rank
    deterministic : str
        Deterministic term specification ('ci': constant inside cointegration)
    
    Returns
    -------
    johansen_result
        Johansen cointegration test results
    vecm_model : VECMResults
        Fitted VECM model
    """
    print("\n" + "=" * 60)
    print("VECTOR ERROR CORRECTION MODEL (VECM)")
    print("=" * 60 + "\n")
    
    # Extract columns
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    vecm_df = df[list(cols)].dropna()
    
    print(f"📊 Variables: {list(cols)}")
    print(f"📊 Observations: {len(vecm_df)}")
    print(f"📊 Date range: {vecm_df.index.min()} to {vecm_df.index.max()}")
    print(f"📊 k_ar_diff: {k_ar_diff}")
    print(f"📊 coint_rank: {coint_rank}\n")
    
    # Johansen cointegration test
    print("🔬 Running Johansen cointegration test...")
    
    try:
        # det_order: -1=no deterministic, 0=constant, 1=trend
        det_order_map = {"n": -1, "co": 0, "ci": 0, "lo": 1, "li": 1}
        det_order = det_order_map.get(deterministic, 0)
        
        johansen_result = coint_johansen(vecm_df, det_order=det_order, k_ar_diff=k_ar_diff)
        
        print("✓ Johansen test completed\n")
        print("=" * 60)
        print("JOHANSEN COINTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"\nTrace Statistics:")
        for i, (trace_stat, crit_val) in enumerate(
            zip(johansen_result.lr1, johansen_result.cvt[:, 1])
        ):
            cointegrated = "✓" if trace_stat > crit_val else "✗"
            print(f"  r <= {i}: {trace_stat:.4f} (critical: {crit_val:.4f}) {cointegrated}")
        
        print(f"\nMax Eigenvalue Statistics:")
        for i, (max_stat, crit_val) in enumerate(
            zip(johansen_result.lr2, johansen_result.cvm[:, 1])
        ):
            cointegrated = "✓" if max_stat > crit_val else "✗"
            print(f"  r = {i}: {max_stat:.4f} (critical: {crit_val:.4f}) {cointegrated}")
        
        print(f"\nEigenvectors (cointegrating vectors):")
        print(johansen_result.evec)
        
    except Exception as e:
        print(f"❌ Johansen test failed: {e}")
        raise
    
    # Fit VECM
    print("\n🔧 Fitting VECM model...")
    
    try:
        vecm_model = VECM(
            vecm_df,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=deterministic,
        )
        
        vecm_result = vecm_model.fit()
        
        print("✓ VECM fitted successfully\n")
        print("=" * 60)
        print("VECM SUMMARY")
        print("=" * 60)
        print(vecm_result.summary())
        
        # Extract key results
        print("\n" + "=" * 60)
        print("KEY VECM RESULTS")
        print("=" * 60)
        
        print(f"\nCointegrating vectors (beta):")
        print(vecm_result.beta)
        
        print(f"\nAdjustment coefficients (alpha):")
        print(vecm_result.alpha)
        
        print(f"\nGamma (short-run dynamics):")
        print(vecm_result.gamma)
        
        return johansen_result, vecm_result
    
    except Exception as e:
        print(f"❌ VECM fitting failed: {e}")
        raise


def analyze_vecm_residuals(vecm_result) -> pd.DataFrame:
    """
    Analyze VECM residuals for diagnostics.
    
    Parameters
    ----------
    vecm_result : VECMResults
        Fitted VECM model
    
    Returns
    -------
    pd.DataFrame
        Residuals DataFrame
    """
    residuals = pd.DataFrame(
        vecm_result.resid,
        columns=vecm_result.names,
    )
    
    print("\n" + "=" * 60)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 60)
    print(f"\nResiduals shape: {residuals.shape}")
    print(f"\nResidual statistics:")
    print(residuals.describe())
    
    return residuals


def forecast_vecm(vecm_result, steps: int = 10) -> pd.DataFrame:
    """
    Generate forecasts from fitted VECM.
    
    Parameters
    ----------
    vecm_result : VECMResults
        Fitted VECM model
    steps : int
        Number of steps ahead to forecast
    
    Returns
    -------
    pd.DataFrame
        Forecasts DataFrame
    """
    print(f"\n🔮 Generating {steps}-step ahead forecast...")
    
    forecast = vecm_result.predict(steps=steps)
    
    forecast_df = pd.DataFrame(
        forecast,
        columns=vecm_result.names,
    )
    
    print("✓ Forecast generated")
    print(forecast_df)
    
    return forecast_df


if __name__ == "__main__":
    # Test VECM
    print("\n=== Testing VECM Models ===\n")
    
    from .features import build_features
    
    try:
        # Build features
        df = build_features()
        
        # Run ADF tests on key variables
        test_cols = ["ptax", "selic", "pulp_brl", "suzb", "pulp_brl_r", "suzb_r"]
        adf_results = run_adf_tests(df, cols=test_cols)
        print("\nADF Test Results:")
        print(adf_results)
        
        # Fit VECM
        johansen, vecm = fit_vecm(df, cols=("pulp_brl", "suzb"), k_ar_diff=2)
        
        # Analyze residuals
        residuals = analyze_vecm_residuals(vecm)
        
        # Forecast
        forecast = forecast_vecm(vecm, steps=5)
        
        print("\n✓ VECM test successful!")
        
    except Exception as e:
        print(f"❌ VECM test failed: {e}")
        raise

