"""CLI interface for EDA Suzano pipeline using Typer."""

import typer
import pandas as pd
from pathlib import Path

from .config import DATA_INT, DATA_OUT, PLOTS_DIR
from .features import build_features
from .synthetic import fit_synthetic, analyze_coefficients, backtest_signals
from .models import fit_vecm, run_adf_tests, analyze_vecm_residuals
from .plots import generate_all_plots

app = typer.Typer(
    name="eda",
    help="EDA Suzano - Exploratory Data Analysis CLI",
    add_completion=False,
)


@app.command()
def ingest(
    start: str = "2020-01-01",
    end: str = None,
):
    """
    Ingest data from all sources, merge, and save to interim.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[INGEST] Loading and merging data")
    typer.echo("=" * 70)
    
    try:
        # Build features
        df = build_features(start=start, end=end)
        
        # Create output directory
        DATA_INT.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        output_path = DATA_INT / "merged.parquet"
        df.to_parquet(output_path)
        
        typer.echo(f"\n[SUCCESS] Data saved to: {output_path}")
        typer.echo(f"   Shape: {df.shape}")
        typer.echo(f"   Columns: {len(df.columns)}")
        typer.echo(f"   Date range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during ingest: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def synthetic(
    input_file: str = None,
):
    """
    Fit synthetic index using OLS regression and compute z-scores.
    
    Args:
        input_file: Path to merged parquet file (default: data/interim/merged.parquet)
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[SYNTHETIC] Fitting OLS and computing z-scores")
    typer.echo("=" * 70)
    
    try:
        # Load data
        if input_file is None:
            input_file = DATA_INT / "merged.parquet"
        else:
            input_file = Path(input_file)
        
        typer.echo(f"\n[LOADING] Data from: {input_file}")
        df = pd.read_parquet(input_file)
        
        # Fit synthetic index
        model, synth, zscore, signals = fit_synthetic(df)
        
        # Analyze coefficients
        coef_df = analyze_coefficients(model)
        
        # Create output directory
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        
        # Save OLS summary
        summary_path = DATA_OUT / "ols_summary.txt"
        with open(summary_path, "w") as f:
            f.write(model.summary().as_text())
        typer.echo(f"\n[SAVED] OLS summary: {summary_path}")
        
        # Save coefficients
        coef_path = DATA_OUT / "ols_coefficients.csv"
        coef_df.to_csv(coef_path)
        typer.echo(f"[SAVED] Coefficients: {coef_path}")
        
        # Merge results with original data
        output_df = df.copy()
        output_df = output_df.join([synth, zscore.rename("zscore")])
        
        # Save synthetic results
        synthetic_path = DATA_OUT / "synthetic.parquet"
        output_df.to_parquet(synthetic_path)
        typer.echo(f"[SAVED] Synthetic data: {synthetic_path}")
        
        # Save signals
        signals_path = DATA_OUT / "signals.parquet"
        signals.to_frame("signal").to_parquet(signals_path)
        typer.echo(f"[SAVED] Signals: {signals_path}")
        
        # Run simple backtest
        backtest = backtest_signals(df, signals)
        backtest_path = DATA_OUT / "backtest.parquet"
        backtest.to_parquet(backtest_path)
        
        market_return = backtest["cumulative_market"].iloc[-1]
        strategy_return = backtest["cumulative_strategy"].iloc[-1]
        
        typer.echo(f"\n[BACKTEST] Results:")
        typer.echo(f"   Market return: {100*market_return:.2f}%")
        typer.echo(f"   Strategy return: {100*strategy_return:.2f}%")
        typer.echo(f"   Excess return: {100*(strategy_return - market_return):.2f}%")
        typer.echo(f"[SAVED] Backtest: {backtest_path}")
        
        typer.echo("\n[SUCCESS] Synthetic index pipeline completed!")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during synthetic: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def vecm(
    input_file: str = None,
    cols: str = "pulp_brl,suzb",
    k_ar_diff: int = 2,
):
    """
    Fit VECM model with Johansen cointegration test.
    
    Args:
        input_file: Path to merged parquet file
        cols: Comma-separated column names for VECM
        k_ar_diff: Number of lagged differences in VECM
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[VECM] Fitting Vector Error Correction Model")
    typer.echo("=" * 70)
    
    try:
        # Load data
        if input_file is None:
            input_file = DATA_INT / "merged.parquet"
        else:
            input_file = Path(input_file)
        
        typer.echo(f"\n[LOADING] Data from: {input_file}")
        df = pd.read_parquet(input_file)
        
        # Parse columns
        cols_list = tuple(cols.split(","))
        
        # Run ADF tests
        typer.echo(f"\n[TESTING] Running ADF tests on: {cols_list}")
        adf_results = run_adf_tests(df, cols=list(cols_list))
        
        # Fit VECM
        johansen_result, vecm_result = fit_vecm(df, cols=cols_list, k_ar_diff=k_ar_diff)
        
        # Analyze residuals
        residuals = analyze_vecm_residuals(vecm_result)
        
        # Create output directory
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        
        # Save VECM summary
        vecm_summary_path = DATA_OUT / "vecm_summary.txt"
        with open(vecm_summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("JOHANSEN COINTEGRATION TEST\n")
            f.write("=" * 70 + "\n\n")
            f.write("Trace Statistics:\n")
            for i, (trace_stat, crit_val) in enumerate(
                zip(johansen_result.lr1, johansen_result.cvt[:, 1])
            ):
                f.write(f"  r <= {i}: {trace_stat:.4f} (critical 5%: {crit_val:.4f})\n")
            
            f.write("\nMax Eigenvalue Statistics:\n")
            for i, (max_stat, crit_val) in enumerate(
                zip(johansen_result.lr2, johansen_result.cvm[:, 1])
            ):
                f.write(f"  r = {i}: {max_stat:.4f} (critical 5%: {crit_val:.4f})\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("VECM SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(vecm_result.summary().as_text())
        
        typer.echo(f"\n[SAVED] VECM summary: {vecm_summary_path}")
        
        # Save ADF results
        adf_path = DATA_OUT / "adf_tests.csv"
        adf_results.to_csv(adf_path, index=False)
        typer.echo(f"[SAVED] ADF tests: {adf_path}")
        
        # Save residuals
        residuals_path = DATA_OUT / "vecm_residuals.parquet"
        residuals.to_parquet(residuals_path)
        typer.echo(f"[SAVED] Residuals: {residuals_path}")
        
        typer.echo("\n[SUCCESS] VECM pipeline completed!")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during VECM: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def report(
    input_file: str = None,
    with_synthetic: bool = True,
):
    """
    Generate visualizations and report.
    
    Args:
        input_file: Path to merged parquet file
        with_synthetic: Include synthetic index plots
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[REPORT] Generating visualizations")
    typer.echo("=" * 70)
    
    try:
        # Load data
        if input_file is None:
            input_file = DATA_INT / "merged.parquet"
        else:
            input_file = Path(input_file)
        
        typer.echo(f"\n[LOADING] Data from: {input_file}")
        df = pd.read_parquet(input_file)
        
        # Create output directory
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load synthetic data if available
        synthetic = None
        zscore = None
        signals = None
        
        if with_synthetic:
            synthetic_path = DATA_OUT / "synthetic.parquet"
            signals_path = DATA_OUT / "signals.parquet"
            
            if synthetic_path.exists():
                typer.echo(f"[LOADING] Synthetic data from: {synthetic_path}")
                synth_df = pd.read_parquet(synthetic_path)
                
                if "synthetic_index" in synth_df.columns:
                    synthetic = synth_df["synthetic_index"]
                if "zscore" in synth_df.columns:
                    zscore = synth_df["zscore"]
            
            if signals_path.exists():
                typer.echo(f"[LOADING] Signals from: {signals_path}")
                signals_df = pd.read_parquet(signals_path)
                if "signal" in signals_df.columns:
                    signals = signals_df["signal"]
        
        # Generate plots
        generate_all_plots(df, synthetic, zscore, signals, PLOTS_DIR)
        
        typer.echo("\n[SUCCESS] Report generation completed!")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during report: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def all(
    start: str = "2020-01-01",
    end: str = None,
):
    """
    Run complete pipeline: ingest → synthetic → vecm → report.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[PIPELINE] RUNNING COMPLETE PIPELINE")
    typer.echo("=" * 70)
    
    # Run each step
    ingest(start=start, end=end)
    synthetic()
    vecm()
    report()
    
    typer.echo("\n" + "=" * 70)
    typer.echo("[SUCCESS] COMPLETE PIPELINE FINISHED!")
    typer.echo("=" * 70)
    typer.echo("\n[OUTPUTS]:")
    typer.echo(f"   - Data: {DATA_INT}/merged.parquet")
    typer.echo(f"   - Results: {DATA_OUT}/")
    typer.echo(f"   - Plots: {PLOTS_DIR}/")


@app.command()
def clean():
    """
    Clean generated files (interim and output directories).
    """
    typer.echo("\n[CLEAN] Cleaning generated files...")
    
    import shutil
    
    dirs_to_clean = [DATA_INT, DATA_OUT]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            typer.echo(f"[OK] Removed: {dir_path}")
    
    typer.echo("\n[SUCCESS] Clean completed!")


if __name__ == "__main__":
    app()

