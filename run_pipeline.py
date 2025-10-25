"""Runner script for the EDA pipeline."""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Run the CLI
from eda.cli import app

if __name__ == "__main__":
    app()

