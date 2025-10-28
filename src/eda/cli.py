"""CLI interface for EDA Suzano pipeline using Typer."""

import typer
import pandas as pd
import numpy as np
from pathlib import Path

from .config import DATA_INT, DATA_OUT, PLOTS_DIR, ROOT
from .features import build_features
from .synthetic import fit_synthetic, analyze_coefficients, backtest_signals
from .synthetic_robust import fit_synthetic_robust
from .models import fit_vecm, run_adf_tests, analyze_vecm_residuals, diag_residuals
from .plots import generate_all_plots
from .plots_validation import generate_all_validation_plots
from .backtest import zscore_backtest
from .metrics import rolling_metrics

# Pipeline imports
from .pipeline.orchestrator import DataPipeline
from .pipeline.monitoring import PipelineMonitor
from .pipeline.scheduler import DataScheduler
from .pipeline.alerting import AlertManager
from .pipeline.manual_upload import ManualUploadManager

# Risk management imports
from .risk import (
    calculate_historical_volatility,
    fit_garch_model,
    calculate_var,
    calculate_cvar,
    calculate_drawdowns,
    calculate_risk_metrics,
)

# Forecasting imports
from .forecasting import (
    fit_arima,
    auto_arima_select,
    forecast_arima,
)

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
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        output_path = DATA_OUT / "merged.parquet"
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
            input_file = DATA_OUT / "merged.parquet"
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
            input_file = DATA_OUT / "merged.parquet"
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
            input_file = DATA_OUT / "merged.parquet"
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
    typer.echo(f"   - Data: {DATA_OUT}/merged.parquet")
    typer.echo(f"   - Results: {DATA_OUT}/")
    typer.echo(f"   - Plots: {PLOTS_DIR}/")


@app.command()
def synthetic_robust(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    noise_alpha: float = 0.10,
    noise_mult: bool = False,
    cv_splits: int = 5,
):
    """
    Fit robust synthetic index with validation and anti-overfitting measures.
    
    Args:
        train_ratio: Training set proportion (default 0.70)
        val_ratio: Validation set proportion (default 0.15)
        noise_alpha: Noise injection level (default 0.10)
        noise_mult: Use multiplicative noise if True
        cv_splits: Number of CV splits for RidgeCV
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[SYNTHETIC ROBUST] Enhanced model with validation")
    typer.echo("=" * 70)
    
    try:
        # Load data
        input_file = DATA_OUT / "merged.parquet"
        typer.echo(f"\n[LOADING] Data from: {input_file}")
        df = pd.read_parquet(input_file)
        
        # Fit robust model
        model, output_df, metrics = fit_synthetic_robust(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            noise_alpha=noise_alpha,
            noise_mult=noise_mult,
            cv_splits=cv_splits,
        )
        
        # Create output directory
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        
        # Save outputs
        output_path = DATA_OUT / "synthetic_robust.parquet"
        output_df.to_parquet(output_path)
        typer.echo(f"\n[SAVED] Robust synthetic data: {output_path}")
        
        # Save metrics
        metrics_path = DATA_OUT / "metrics_robust.csv"
        pd.Series(metrics).to_csv(metrics_path)
        typer.echo(f"[SAVED] Metrics: {metrics_path}")
        
        # Save model info
        model_info_path = DATA_OUT / "model_info.txt"
        with open(model_info_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("RIDGE REGRESSION MODEL INFO\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best alpha (regularization): {model.alpha_:.6f}\n")
            f.write(f"CV score (neg MSE): {model.best_score_:.6f}\n\n")
            f.write("Train/Val/Test Metrics:\n")
            for key in ['R2', 'MAE', 'RMSE', 'IC']:
                train_val = metrics.get(f'train_{key}', np.nan)
                val_val = metrics.get(f'val_{key}', np.nan)
                test_val = metrics.get(f'test_{key}', np.nan)
                f.write(f"  {key}: train={train_val:.4f}, val={val_val:.4f}, test={test_val:.4f}\n")
        
        typer.echo(f"[SAVED] Model info: {model_info_path}")
        
        # Run residual diagnostics
        typer.echo("\n[DIAGNOSTICS] Running residual tests...")
        diag_path = DATA_OUT / "residual_diagnostics.txt"
        diag_residuals(output_df["error"], out_path=str(diag_path))
        
        # Run backtest
        typer.echo("\n[BACKTEST] Running z-score strategy backtest...")
        backtest_metrics, backtest_results = zscore_backtest(output_df)
        
        backtest_path = DATA_OUT / "backtest_robust.parquet"
        backtest_results.to_parquet(backtest_path)
        typer.echo(f"[SAVED] Backtest results: {backtest_path}")
        
        # Display backtest results
        typer.echo("\n[BACKTEST METRICS]:")
        for key, value in backtest_metrics.items():
            typer.echo(f"  {key}: {value:.4f}")
        
        # Generate validation plots
        typer.echo("\n[PLOTS] Generating validation plots...")
        generate_all_validation_plots(
            output_df,
            backtest_results=backtest_results,
            outdir=PLOTS_DIR,
        )
        
        # Calculate rolling metrics
        typer.echo("\n[METRICS] Calculating rolling metrics...")
        roll_metrics = rolling_metrics(
            output_df["suzb_r"],
            output_df["synthetic_index"],
            window=60,
        )
        roll_metrics_path = DATA_OUT / "metrics_rolling.csv"
        roll_metrics.to_csv(roll_metrics_path)
        typer.echo(f"[SAVED] Rolling metrics: {roll_metrics_path}")
        
        typer.echo("\n" + "=" * 70)
        typer.echo("[SUCCESS] Robust synthetic pipeline completed!")
        typer.echo("=" * 70)
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during robust synthetic: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def validate():
    """
    Generate comprehensive validation plots and metrics from existing results.
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[VALIDATE] Generating validation analysis")
    typer.echo("=" * 70)
    
    try:
        # Check for robust results first
        robust_path = DATA_OUT / "synthetic_robust.parquet"
        backtest_path = DATA_OUT / "backtest_robust.parquet"
        
        if robust_path.exists():
            typer.echo(f"\n[LOADING] Robust synthetic data from: {robust_path}")
            df = pd.read_parquet(robust_path)
            
            backtest_results = None
            if backtest_path.exists():
                typer.echo(f"[LOADING] Backtest results from: {backtest_path}")
                backtest_results = pd.read_parquet(backtest_path)
            
            # Generate all validation plots
            generate_all_validation_plots(
                df,
                backtest_results=backtest_results,
                outdir=PLOTS_DIR,
            )
            
            typer.echo("\n[SUCCESS] Validation plots generated!")
        else:
            typer.echo(f"\n[ERROR] Robust synthetic data not found. Run 'synthetic-robust' first.")
            raise typer.Exit(code=1)
            
    except Exception as e:
        typer.echo(f"\n[ERROR] Error during validation: {e}", err=True)
        raise typer.Exit(code=1)


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


@app.command()
def all_robust(
    start: str = "2020-01-01",
    end: str = None,
    train_ratio: float = 0.70,
    noise_alpha: float = 0.10,
):
    """
    Run complete robust pipeline: ingest → synthetic-robust → validate → vecm → report.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
        train_ratio: Training set proportion
        noise_alpha: Noise injection level
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[PIPELINE ROBUST] RUNNING ENHANCED PIPELINE")
    typer.echo("=" * 70)
    
    # Run each step
    ingest(start=start, end=end)
    synthetic_robust(train_ratio=train_ratio, noise_alpha=noise_alpha)
    validate()
    vecm()
    report()
    
    typer.echo("\n" + "=" * 70)
    typer.echo("[SUCCESS] ROBUST PIPELINE FINISHED!")
    typer.echo("=" * 70)
    typer.echo("\n[OUTPUTS]:")
    typer.echo(f"   - Data: {DATA_OUT}/merged.parquet")
    typer.echo(f"   - Results: {DATA_OUT}/")
    typer.echo(f"   - Robust Results: {DATA_OUT}/synthetic_robust.parquet")
    typer.echo(f"   - Metrics: {DATA_OUT}/metrics_robust.csv")
    typer.echo(f"   - Plots: {PLOTS_DIR}/")


# =============================================================================
# PRODUCTION PIPELINE COMMANDS
# =============================================================================

@app.command(name="pipeline-run")
def pipeline_run(
    sources: str = None,
    force_full: bool = False,
    no_cache: bool = False,
):
    """
    Run the production data pipeline to fetch and update all data sources.
    
    Args:
        sources: Comma-separated list of sources to update (default: all)
        force_full: Force full historical fetch instead of incremental
        no_cache: Disable scraper caching
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[PIPELINE] Starting production data pipeline")
    typer.echo("=" * 70)
    
    try:
        # Initialize pipeline
        pipeline = DataPipeline()
        
        # Parse sources
        source_list = None
        if sources:
            source_list = [s.strip() for s in sources.split(",")]
        
        # Run pipeline
        results = pipeline.run(
            sources=source_list,
            force_full=force_full,
            use_cache=not no_cache,
            continue_on_error=True
        )
        
        typer.echo(f"\n[SUCCESS] Pipeline completed!")
        typer.echo(f"   Updated {len(results)} sources")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Pipeline failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="pipeline-monitor")
def pipeline_monitor(
    export_csv: bool = False,
):
    """
    Monitor pipeline health and display status report.
    
    Args:
        export_csv: Export metrics to CSV
    """
    typer.echo("\n[MONITOR] Checking pipeline health...")
    
    try:
        pipeline = DataPipeline()
        monitor = PipelineMonitor(pipeline)
        
        # Generate report
        report = monitor.generate_status_report()
        typer.echo(report)
        
        # Export if requested
        if export_csv:
            output_file = DATA_OUT / "pipeline_metrics.csv"
            monitor.export_metrics_csv(output_file)
            typer.echo(f"\n[SAVED] Metrics exported to {output_file}")
        
        # Run health check
        is_healthy = monitor.run_health_check(alert_on_issues=False)
        
        if not is_healthy:
            raise typer.Exit(code=1)
    
    except Exception as e:
        typer.echo(f"\n[ERROR] Monitoring failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="pipeline-upload")
def pipeline_upload(
    file_path: str,
    source: str = "pulp_prices",
    date_column: str = "date",
):
    """
    Manually upload data from CSV/Excel file (for pulp prices, etc.).
    
    Args:
        file_path: Path to CSV or Excel file
        source: Data source name (default: pulp_prices)
        date_column: Name of date column (default: date)
    """
    typer.echo(f"\n[UPLOAD] Uploading file: {file_path}")
    
    try:
        pipeline = DataPipeline()
        uploader = ManualUploadManager(
            upload_dir=ROOT / "data" / "manual_uploads",
            version_manager=pipeline.version_manager
        )
        
        # Upload
        if source == "pulp_prices":
            df = uploader.upload_pulp_prices(Path(file_path))
        else:
            df = uploader.upload_file(
                Path(file_path),
                source_name=source,
                date_column=date_column
            )
        
        typer.echo(f"\n[SUCCESS] Uploaded {len(df)} rows for {source}")
        typer.echo(f"   Date range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Upload failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="pipeline-template")
def pipeline_template(
    source: str = "pulp_prices",
    output_path: str = None,
):
    """
    Create a template CSV file for manual uploads.
    
    Args:
        source: Data source name (default: pulp_prices)
        output_path: Output file path (default: data/manual_uploads/template_{source}.csv)
    """
    try:
        pipeline = DataPipeline()
        uploader = ManualUploadManager(
            upload_dir=ROOT / "data" / "manual_uploads",
            version_manager=pipeline.version_manager
        )
        
        # Define templates
        templates = {
            "pulp_prices": ["date", "price", "type"],
            "custom": ["date", "value"],
        }
        
        columns = templates.get(source, templates["custom"])
        
        if output_path is None:
            output_path = ROOT / "data" / "manual_uploads" / f"template_{source}.csv"
        else:
            output_path = Path(output_path)
        
        uploader.create_template(source, columns, output_path)
        
        typer.echo(f"\n[SUCCESS] Template created: {output_path}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Template creation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="pipeline-versions")
def pipeline_versions(
    source: str = None,
):
    """
    List version history for data sources.
    
    Args:
        source: Source name to filter (default: all sources)
    """
    try:
        pipeline = DataPipeline()
        
        if source:
            # Show detailed history for one source
            history = pipeline.version_manager.get_version_history(source)
            
            if len(history) == 0:
                typer.echo(f"\n[INFO] No versions found for {source}")
            else:
                typer.echo(f"\n[VERSIONS] History for {source}:")
                typer.echo(history.to_string())
        else:
            # Show summary for all sources
            typer.echo("\n[VERSIONS] All sources:")
            for source_name in pipeline.SOURCES.keys():
                versions = pipeline.version_manager.list_versions(source_name)
                if versions:
                    latest = versions[0]
                    typer.echo(f"\n  {source_name}:")
                    typer.echo(f"    Total versions: {len(versions)}")
                    typer.echo(f"    Latest: {latest.timestamp}")
                    typer.echo(f"    Rows: {latest.rows}")
                else:
                    typer.echo(f"\n  {source_name}: No versions")
    
    except Exception as e:
        typer.echo(f"\n[ERROR] Failed to list versions: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="pipeline-cleanup")
def pipeline_cleanup(
    keep_last: int = 10,
):
    """
    Clean up old data versions, keeping only the most recent N versions per source.
    
    Args:
        keep_last: Number of versions to keep (default: 10)
    """
    typer.echo(f"\n[CLEANUP] Cleaning up old versions (keeping last {keep_last})")
    
    try:
        pipeline = DataPipeline()
        pipeline.cleanup_old_versions(keep_last_n=keep_last)
        
        typer.echo(f"\n[SUCCESS] Cleanup completed")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Cleanup failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="scheduler-start")
def scheduler_start(
    run_immediately: bool = True,
):
    """
    Start the automated scheduler for periodic data updates.
    
    WARNING: This runs indefinitely until interrupted (Ctrl+C).
    
    Args:
        run_immediately: Run all jobs once before starting schedule (default: True)
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[SCHEDULER] Starting automated data pipeline scheduler")
    typer.echo("=" * 70)
    typer.echo("\nPress Ctrl+C to stop\n")
    
    try:
        pipeline = DataPipeline()
        alert_manager = AlertManager(console=True)
        scheduler = DataScheduler(pipeline, alert_manager)
        
        # Schedule all sources
        scheduler.schedule_all()
        
        # Start scheduler (blocks until Ctrl+C)
        scheduler.start(run_immediately=run_immediately)
        
    except KeyboardInterrupt:
        typer.echo("\n\n[SCHEDULER] Stopped by user")
    except Exception as e:
        typer.echo(f"\n[ERROR] Scheduler failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="scheduler-export-cron")
def scheduler_export_cron(
    output_file: str = None,
):
    """
    Export scheduler configuration as a cron-compatible shell script.
    
    Args:
        output_file: Output file path (default: scripts/pipeline_cron.sh)
    """
    try:
        if output_file is None:
            output_file = ROOT / "scripts" / "pipeline_cron.sh"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        pipeline = DataPipeline()
        scheduler = DataScheduler(pipeline)
        scheduler.schedule_all()
        
        scheduler.export_schedule_cron(output_file)
        
        typer.echo(f"\n[SUCCESS] Cron script exported to {output_file}")
        typer.echo("\nTo install:")
        typer.echo(f"  chmod +x {output_file}")
        typer.echo(f"  crontab -e")
        typer.echo(f"  # Add: @reboot {output_file}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Export failed: {e}", err=True)
        raise typer.Exit(code=1)


# =============================================================================
# RISK MANAGEMENT COMMANDS
# =============================================================================

@app.command(name="risk-analysis")
def risk_analysis(
    input_file: str = None,
    confidence: float = 0.95,
):
    """
    Comprehensive risk analysis: volatility, VaR, drawdowns, metrics.
    
    Args:
        input_file: Path to data file (default: data/interim/merged.parquet)
        confidence: Confidence level for VaR (default: 0.95)
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[RISK] Running comprehensive risk analysis")
    typer.echo("=" * 70)
    
    try:
        # Load data
        if input_file is None:
            input_file = DATA_INT / "merged.parquet"
        
        df = pd.read_parquet(input_file)
        returns = df['suzb_r'].dropna()
        
        typer.echo(f"\n[LOADING] {len(returns)} return observations")
        
        # 1. Volatility analysis
        typer.echo("\n[VOLATILITY] Calculating volatility metrics...")
        vol_21d = calculate_historical_volatility(returns, window=21)
        vol_252d = calculate_historical_volatility(returns, window=252)
        
        typer.echo(f"  Current volatility (21d): {vol_21d.iloc[-1]:.2%}")
        typer.echo(f"  Current volatility (252d): {vol_252d.iloc[-1]:.2%}")
        
        # 2. VaR/CVaR
        typer.echo(f"\n[VAR] Calculating Value at Risk ({confidence:.0%} confidence)...")
        var_hist = calculate_var(returns, confidence, method='historical')
        cvar_hist = calculate_cvar(returns, confidence, method='historical')
        
        typer.echo(f"  VaR (historical): {var_hist:.2%}")
        typer.echo(f"  CVaR (historical): {cvar_hist:.2%}")
        
        # 3. Drawdowns
        typer.echo("\n[DRAWDOWNS] Analyzing drawdowns...")
        from .risk.drawdowns import calculate_drawdown_stats
        dd_stats = calculate_drawdown_stats(returns)
        
        typer.echo(f"  Max drawdown: {dd_stats['max_drawdown']:.2%}")
        typer.echo(f"  Max DD date: {dd_stats['max_drawdown_date']}")
        typer.echo(f"  Average drawdown: {dd_stats['avg_drawdown']:.2%}")
        typer.echo(f"  Current drawdown: {dd_stats['current_drawdown']:.2%}")
        
        # 4. Comprehensive metrics
        typer.echo("\n[METRICS] Calculating risk metrics...")
        metrics = calculate_risk_metrics(returns)
        
        typer.echo(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        typer.echo(f"  Sortino ratio: {metrics['sortino_ratio']:.2f}")
        typer.echo(f"  Calmar ratio: {metrics['calmar_ratio']:.2f}")
        typer.echo(f"  Omega ratio: {metrics['omega_ratio']:.2f}")
        
        # Save results
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        results_path = DATA_OUT / "risk_analysis.csv"
        pd.Series(metrics).to_csv(results_path)
        
        typer.echo(f"\n[SAVED] Risk metrics: {results_path}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Risk analysis failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="forecast-arima")
def forecast_arima_cmd(
    input_file: str = None,
    horizon: int = 30,
    auto_select: bool = True,
):
    """
    ARIMA forecast for SUZB3 returns.
    
    Args:
        input_file: Path to data file
        horizon: Forecast horizon in days
        auto_select: Use automatic model selection
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[FORECAST] ARIMA time series forecasting")
    typer.echo("=" * 70)
    
    try:
        # Load data
        if input_file is None:
            input_file = DATA_INT / "merged.parquet"
        
        df = pd.read_parquet(input_file)
        returns = df['suzb_r'].dropna()
        
        typer.echo(f"\n[LOADING] {len(returns)} observations")
        
        if auto_select:
            typer.echo("\n[AUTO] Selecting best ARIMA model...")
            from .forecasting.classical import auto_arima_select
            
            best_model = auto_arima_select(
                returns,
                max_p=3,
                max_q=3,
                max_d=1,
                seasonal=False
            )
            
            typer.echo(f"  Best order: {best_model['order']}")
            typer.echo(f"  AIC: {best_model['aic']:.2f}")
            typer.echo(f"  BIC: {best_model['bic']:.2f}")
            
            result = best_model['result']
        
        else:
            typer.echo("\n[ARIMA] Fitting ARIMA(1,0,1)...")
            result, fitted, residuals = fit_arima(returns, order=(1, 0, 1))
        
        # Generate forecast
        typer.echo(f"\n[FORECAST] Generating {horizon}-day forecast...")
        forecast_df = forecast_arima(result, horizon=horizon)
        
        typer.echo(f"\n  Next 5 days forecast:")
        for i in range(min(5, horizon)):
            fc = forecast_df.iloc[i]
            typer.echo(f"    Day {i+1}: {fc['forecast']:.4f} [{fc['lower']:.4f}, {fc['upper']:.4f}]")
        
        # Save
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        forecast_path = DATA_OUT / "arima_forecast.csv"
        forecast_df.to_csv(forecast_path)
        
        typer.echo(f"\n[SAVED] Forecast: {forecast_path}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Forecast failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="fetch-fundamentals")
def fetch_fundamentals():
    """
    Fetch Suzano fundamental data (financials, ratios).
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[FUNDAMENTALS] Fetching Suzano company data")
    typer.echo("=" * 70)
    
    try:
        from .scrapers import registry
        
        # Get fundamentals scraper
        scraper = registry.get("fundamentals")
        
        # Fetch all financials
        typer.echo("\n[FETCHING] Financial statements...")
        financials = scraper._fetch_all_financials(scraper._fetch_data.__self__)
        
        # Calculate ratios
        typer.echo("\n[CALCULATING] Financial ratios...")
        ratios = scraper.calculate_financial_ratios(financials)
        
        # Get key metrics
        metrics = scraper.get_key_metrics_series(financials)
        
        # Save
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        
        if not ratios.empty:
            ratios_path = DATA_OUT / "financial_ratios.csv"
            ratios.to_csv(ratios_path)
            typer.echo(f"[SAVED] Ratios: {ratios_path}")
        
        if not metrics.empty:
            metrics_path = DATA_OUT / "key_metrics.csv"
            metrics.to_csv(metrics_path)
            typer.echo(f"[SAVED] Metrics: {metrics_path}")
            
            typer.echo(f"\n[SUMMARY] Latest metrics:")
            if len(metrics) > 0:
                latest = metrics.iloc[-1]
                for col in metrics.columns:
                    val = latest[col]
                    if not pd.isna(val):
                        typer.echo(f"  {col}: {val:,.0f}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Fundamentals fetch failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="fetch-macro")
def fetch_macro(
    source: str = "ibge",
    series: str = "ipca",
):
    """
    Fetch macro data (inflation, GDP, etc.).
    
    Args:
        source: Data source ('ibge' or 'bcb')
        series: Series name (e.g., 'ipca', 'gdp')
    """
    typer.echo("\n" + "=" * 70)
    typer.echo(f"[MACRO] Fetching {series} from {source}")
    typer.echo("=" * 70)
    
    try:
        from .scrapers import registry
        from datetime import datetime
        
        # Get scraper
        if source == "ibge":
            scraper = registry.get("ibge_macro")
        elif source == "bcb":
            scraper = registry.get("bcb_extended")
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Fetch
        end_date = datetime.now().strftime("%Y-%m-%d")
        df = scraper.fetch(
            start_date="2020-01-01",
            end_date=end_date,
            series=series
        )
        
        typer.echo(f"\n[SUCCESS] Fetched {len(df)} observations")
        typer.echo(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        # Save
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        output_path = DATA_OUT / f"macro_{source}_{series}.csv"
        df.to_csv(output_path)
        
        typer.echo(f"\n[SAVED] {output_path}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Macro fetch failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="data-quality")
def data_quality(
    input_file: str = None,
):
    """
    Run comprehensive data quality assessment.
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[DATA QUALITY] Running assessment")
    typer.echo("=" * 70)
    
    try:
        from .data_quality import run_full_quality_check
        
        # Load data
        if input_file:
            df = pd.read_parquet(input_file)
        else:
            feat_path = DATA_OUT / "merged.parquet"
            if not feat_path.exists():
                typer.echo("[ERROR] No merged.parquet found. Run 'ingest' first.", err=True)
                raise typer.Exit(code=1)
            df = pd.read_parquet(feat_path)
        
        # Run checks
        results = run_full_quality_check(df)
        
        typer.echo(f"\n[SUCCESS] Data quality report generated")
        typer.echo(f"  Report saved to: {DATA_OUT / 'data_quality_report.csv'}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Data quality check failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="eda-comprehensive")
def eda_comprehensive(
    input_file: str = None,
):
    """
    Run comprehensive exploratory data analysis with 30+ plots.
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[EDA] Running comprehensive analysis")
    typer.echo("=" * 70)
    
    try:
        from .exploratory import run_comprehensive_eda
        
        # Load data
        if input_file:
            df = pd.read_parquet(input_file)
        else:
            feat_path = DATA_OUT / "merged.parquet"
            if not feat_path.exists():
                typer.echo("[ERROR] No merged.parquet found. Run 'ingest' first.", err=True)
                raise typer.Exit(code=1)
            df = pd.read_parquet(feat_path)
        
        # Run EDA
        run_comprehensive_eda(df)
        
        typer.echo(f"\n[SUCCESS] Comprehensive EDA complete")
        typer.echo(f"  Plots saved to: {DATA_OUT / 'plots/eda/'}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Comprehensive EDA failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="automl-tpot")
def automl_tpot(
    input_file: str = None,
    generations: int = 10,
    population: int = 50,
):
    """
    Run TPOT AutoML to find best model pipeline.
    
    Args:
        input_file: Input parquet file (default: merged.parquet)
        generations: Number of GP generations (default: 10, use 50 for thorough search)
        population: Population size per generation
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[AUTOML] Starting TPOT optimization")
    typer.echo("=" * 70)
    
    try:
        from .automl import run_tpot_optimization
        
        # Load data
        if input_file:
            df = pd.read_parquet(input_file)
        else:
            feat_path = DATA_OUT / "merged.parquet"
            if not feat_path.exists():
                typer.echo("[ERROR] No merged.parquet found. Run 'ingest' first.", err=True)
                raise typer.Exit(code=1)
            df = pd.read_parquet(feat_path)
        
        # Run TPOT
        tpot_model, results = run_tpot_optimization(
            df,
            target_col='suzb_r',
            generations=generations,
            population_size=population
        )
        
        typer.echo(f"\n[SUCCESS] TPOT optimization complete")
        typer.echo(f"  Best pipeline saved to: {DATA_OUT / 'tpot_best_pipeline.py'}")
        typer.echo(f"  Results saved to: {DATA_OUT / 'tpot_results.csv'}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] TPOT optimization failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="model-comparison")
def model_comparison(
    input_file: str = None,
    include_ensemble: bool = True,
):
    """
    Compare multiple models (Ridge, XGBoost, LightGBM, etc.).
    
    Args:
        input_file: Input parquet file (default: merged.parquet)
        include_ensemble: Whether to create voting ensemble
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[MODEL COMPARISON] Testing multiple models")
    typer.echo("=" * 70)
    
    try:
        from .automl import compare_all_models
        
        # Load data
        if input_file:
            df = pd.read_parquet(input_file)
        else:
            feat_path = DATA_OUT / "merged.parquet"
            if not feat_path.exists():
                typer.echo("[ERROR] No merged.parquet found. Run 'ingest' first.", err=True)
                raise typer.Exit(code=1)
            df = pd.read_parquet(feat_path)
        
        # Run comparison
        comparator, results_df = compare_all_models(
            df,
            target_col='suzb_r',
            create_ensemble=include_ensemble
        )
        
        typer.echo(f"\n[SUCCESS] Model comparison complete")
        typer.echo(f"  Results saved to: {DATA_OUT / 'model_comparison.csv'}")
        typer.echo(f"  Plots saved to: {DATA_OUT / 'plots/models/'}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Model comparison failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="strategy-optimize")
def strategy_optimize(
    input_file: str = None,
):
    """
    Optimize strategy parameters using grid search.
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[STRATEGY] Optimizing parameters")
    typer.echo("=" * 70)
    
    try:
        from .strategies import optimize_strategy_parameters
        
        # Load data with z-scores
        if input_file:
            df = pd.read_parquet(input_file)
        else:
            synthetic_path = DATA_OUT / "synthetic_robust.parquet"
            if not synthetic_path.exists():
                typer.echo("[ERROR] No synthetic_robust.parquet found. Run 'all-robust' first.", err=True)
                raise typer.Exit(code=1)
            df = pd.read_parquet(synthetic_path)
        
        # Run optimization
        best_params, results_df = optimize_strategy_parameters(df)
        
        typer.echo(f"\n[SUCCESS] Strategy optimization complete")
        typer.echo(f"  Results saved to: {DATA_OUT / 'strategy_optimization.csv'}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Strategy optimization failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="benchmark-compare")
def benchmark_compare(
    strategy_file: str = None,
):
    """
    Compare strategy performance vs benchmarks.
    
    Args:
        strategy_file: Strategy results parquet (default: backtest_robust.parquet)
    """
    typer.echo("\n" + "=" * 70)
    typer.echo("[BENCHMARK] Comparing strategy vs benchmarks")
    typer.echo("=" * 70)
    
    try:
        from .strategies import compare_strategies
        from .loaders_benchmarks import load_all_benchmarks
        
        # Load strategy returns
        if strategy_file:
            df = pd.read_parquet(strategy_file)
        else:
            backtest_path = DATA_OUT / "backtest_robust.parquet"
            if not backtest_path.exists():
                typer.echo("[ERROR] No backtest_robust.parquet found. Run 'all-robust' first.", err=True)
                raise typer.Exit(code=1)
            df = pd.read_parquet(backtest_path)
        
        strategy_returns = df['strategy_return']
        
        # Load benchmarks
        bench_df = load_all_benchmarks(start="2020-01-01")
        
        # Calculate benchmark returns
        benchmark_returns = {}
        for col in ['imat', 'iagro', 'ibov']:
            if col in bench_df.columns:
                bench_returns = np.log(bench_df[col]).diff()
                benchmark_returns[col.upper()] = bench_returns.reindex(strategy_returns.index)
        
        # Add buy-and-hold SUZB3
        features_path = DATA_OUT / "merged.parquet"
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            if 'suzb_r' in features_df.columns:
                benchmark_returns['Buy-and-Hold'] = features_df['suzb_r'].reindex(strategy_returns.index)
        
        # Run comparison
        metrics_df, comparator = compare_strategies(strategy_returns, benchmark_returns)
        
        typer.echo(f"\n[SUCCESS] Benchmark comparison complete")
        typer.echo(f"  Results saved to: {DATA_OUT / 'benchmark_comparison.csv'}")
        typer.echo(f"  Plot saved to: {DATA_OUT / 'plots/strategies/benchmark_comparison.png'}")
        
    except Exception as e:
        typer.echo(f"\n[ERROR] Benchmark comparison failed: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

