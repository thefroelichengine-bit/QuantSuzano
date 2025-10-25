"""Configuration module with paths and constants for EDA Suzano."""

from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[2]
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

