"""Script to run ensemble strategy and capture output."""
import sys
import traceback

print("="*70)
print("RUNNING ENSEMBLE STRATEGY")
print("="*70)
print()

try:
    print("[1/5] Importing modules...")
    from eda.strategies import EnsembleStrategy
    import pandas as pd
    from pathlib import Path
    print("  ✓ Import successful")
    print()
    
    print("[2/5] Loading data...")
    data_path = Path("data/out/merged.parquet")
    if not data_path.exists():
        print(f"  ✗ ERROR: {data_path} not found")
        print("  Run 'python -m eda.cli ingest' first")
        sys.exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"  ✓ Data loaded: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print()
    
    print("[3/5] Initializing ensemble strategy...")
    strategy = EnsembleStrategy(
        voting_method='majority',
        risk_reward_threshold=1.5,
        z_threshold=2.0,
    )
    print("  ✓ Strategy initialized")
    print()
    
    print("[4/5] Fitting strategy...")
    strategy.fit(df, target_col='suzb_r')
    print("  ✓ Strategy fitted")
    print()
    
    print("[5/5] Generating signals and running backtest...")
    signals_df = strategy.generate_signals()
    metrics, backtest_results = strategy.backtest(signals_df=signals_df)
    
    print()
    print("="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    print("[SAVING] Saving results...")
    strategy.save_results(signals_df, metrics, backtest_results)
    print("  ✓ Results saved")
    print()
    
    print("="*70)
    print("SUCCESS! Ensemble strategy completed")
    print("="*70)
    print()
    print("Output files:")
    print("  - data/out/ensemble_signals.parquet")
    print("  - data/out/ensemble_backtest.parquet")
    print("  - data/out/ensemble_metrics.csv")
    print("  - data/out/plots/strategies/")
    
except Exception as e:
    print()
    print("="*70)
    print("ERROR OCCURRED")
    print("="*70)
    print(f"Error: {e}")
    print()
    traceback.print_exc()
    sys.exit(1)

