"""Test script to run ensemble strategy."""
import sys
print("Starting ensemble strategy test...", file=sys.stderr)
sys.stderr.flush()

try:
    from eda.strategies import EnsembleStrategy
    print("✓ EnsembleStrategy imported", file=sys.stderr)
    sys.stderr.flush()
    
    from eda.cli import strategy_ensemble
    print("✓ strategy_ensemble function imported", file=sys.stderr)
    sys.stderr.flush()
    
    # Try to run it
    import typer
    app = typer.Typer()
    app.command()(strategy_ensemble)
    
    print("✓ CLI command registered", file=sys.stderr)
    sys.stderr.flush()
    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("All imports successful!", file=sys.stderr)
sys.stderr.flush()

