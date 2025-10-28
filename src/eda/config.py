"""Configuration module with paths and constants for EDA Suzano."""

import os
from pathlib import Path

# Project paths - detect workspace vs installed package
# If running from installed package, use current working directory
# Otherwise use the package location
_module_path = Path(__file__).resolve().parents[2]
_cwd_path = Path.cwd()

# Check if we're in the workspace (has pyproject.toml) or running from installed package
if (_cwd_path / "pyproject.toml").exists():
    ROOT = _cwd_path
elif (_cwd_path / "src" / "eda").exists():
    ROOT = _cwd_path
elif "QuantSuzano" in str(_cwd_path):
    ROOT = _cwd_path
else:
    # Fall back to module path if we can't detect workspace
    ROOT = _module_path

DATA_RAW = ROOT / "data" / "raw"
DATA_INT = ROOT / "data" / "interim"
DATA_OUT = ROOT / "data" / "out"
PLOTS_DIR = DATA_OUT / "plots"

# Constants
BUSINESS_FREQ = "B"  # Business days
ROLL_Z = 60  # Rolling window for z-score calculation
EQUITY_TICKER = "SUZB3.SA"  # Suzano ticker on Yahoo Finance

# BCB SGS series codes
SGS_PTAX = 1  # PTAX - Taxa de câmbio USD/BRL (compra)
SGS_SELIC = 432  # SELIC - Taxa de juros meta

# Signal thresholds
Z_THRESHOLD = 2.0  # |z-score| > 2 for trading signals

# Date range for data collection (can be overridden)
DEFAULT_START_DATE = "2020-01-01"

