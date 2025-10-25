# EDA Suzano - Setup Guide

## Quick Start (Windows)

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -e .
```

### 4. Run Complete Pipeline

```bash
python -m eda.cli all
```

This will:
- Fetch data from Yahoo Finance (SUZB3) and BCB API (PTAX, SELIC)
- Load CSV data (pulp, climate, credit)
- Build feature matrix with log returns and lags
- Fit synthetic index via OLS regression
- Calculate z-scores and trading signals
- Run VECM with Johansen cointegration test
- Generate all plots and reports

## Alternative: Using Make

### Windows (if Make is installed)

```bash
make help       # Show all available commands
make setup      # Create venv (manual activation needed)
make install    # Install dependencies
make all        # Run complete pipeline
```

### Linux/Mac

```bash
make setup && source .venv/bin/activate
make install
make all
```

## Individual Pipeline Steps

Run each step separately if needed:

```bash
# 1. Data ingestion
python -m eda.cli ingest

# 2. Synthetic index
python -m eda.cli synthetic

# 3. VECM modeling
python -m eda.cli vecm

# 4. Generate plots
python -m eda.cli report
```

## Open Jupyter Notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

Or using Make:

```bash
make notebook
```

## Expected Outputs

After running the pipeline, you'll find:

```
data/
├── interim/
│   └── merged.parquet              # Merged and processed data
├── out/
│   ├── ols_summary.txt            # OLS regression summary
│   ├── ols_coefficients.csv       # Regression coefficients
│   ├── synthetic.parquet          # Synthetic index & z-scores
│   ├── signals.parquet            # Trading signals
│   ├── backtest.parquet           # Backtest results
│   ├── vecm_summary.txt           # VECM summary
│   ├── adf_tests.csv              # ADF stationarity tests
│   ├── vecm_residuals.parquet     # VECM residuals
│   └── plots/
│       ├── levels.png             # Price levels
│       ├── returns.png            # Log returns
│       ├── correlation_heatmap.png
│       ├── rolling_corr_*.png
│       ├── synthetic_vs_actual.png
│       ├── signals.png
│       └── distributions.png
```

## Data Sources

### Real Data (APIs)
- **SUZB3**: Yahoo Finance via `yfinance`
- **PTAX**: Banco Central do Brasil SGS API (série 1)
- **SELIC**: Banco Central do Brasil SGS API (série 432)

### Placeholder Data (CSV)
Located in `data/raw/`:
- `pulp_usd.csv` - Celulose prices (replace with FOEX API)
- `climate.csv` - Precipitation and NDVI (replace with INMET)
- `credit.csv` - Credit index (replace with BCB credit series)

## Troubleshooting

### Import Errors

If you get import errors, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -e .`
3. You're in the project root directory

### API Connection Issues

If Yahoo Finance or BCB API fail:
- Check internet connection
- The loaders will attempt CSV fallbacks
- Create CSV files in `data/raw/` with required formats

### Missing Dependencies

Install specific packages if needed:
```bash
pip install pandas numpy matplotlib statsmodels yfinance typer requests
```

## Development

### Code Formatting

```bash
make format    # Format with black
make lint      # Lint with ruff
```

### Clean Generated Files

```bash
make clean
# or
python -m eda.cli clean
```

## Project Structure

```
eda-suzano/
├── src/eda/              # Main package
│   ├── __init__.py
│   ├── config.py         # Configuration
│   ├── loaders.py        # Data loaders
│   ├── features.py       # Feature engineering
│   ├── synthetic.py      # Synthetic index
│   ├── models.py         # VECM & tests
│   ├── plots.py          # Visualizations
│   └── cli.py            # CLI interface
├── notebooks/
│   └── EDA.ipynb         # Exploratory notebook
├── data/
│   ├── raw/              # Source data
│   ├── interim/          # Processed data
│   └── out/              # Results
├── pyproject.toml        # Dependencies
├── Makefile              # Build automation
└── README.md             # Documentation
```

## Next Steps

1. Replace CSV placeholders with real APIs
2. Explore Jupyter notebook for detailed analysis
3. Tune model parameters (rolling window, lags, etc.)
4. Implement production-grade backtesting
5. Add unit tests

## Support

For issues or questions:
1. Check README.md for documentation
2. Review code comments in `src/eda/`
3. Inspect notebook output for analysis insights

---

**Built with**: Python 3.11+, pandas, statsmodels, yfinance, matplotlib, typer

