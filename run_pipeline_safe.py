"""Safe runner script for the EDA pipeline that handles Windows encoding."""
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    # Set console to UTF-8 mode
    os.system('chcp 65001 > nul')
    
    # Redirect stdout/stderr to handle encoding
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Run the CLI
from eda.cli import app

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

