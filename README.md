# QuantSuzano - The Froelich Engine

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange.svg)

## 📖 Overview

**QuantSuzano** is a comprehensive quantitative finance platform for analyzing Suzano (SUZB3) stock performance relative to pulp prices, exchange rates, macroeconomic factors, and climate data. The project implements **The Froelich Engine** - an ensemble voting strategy with risk-reward decision model that combines multiple machine learning models to generate trading signals.

### About The Froelich Engine

**The Froelich Engine** is named in honor of John Froelich, inventor of the tractor. Just as the tractor replaced unpredictability with control in agriculture, our engine replaces market "noise" with direct analysis of production fundamentals (climate data), revolutionizing investment in the sector.

**Strategy:** Uses Vector Error Correction Model (VECM) to explore cointegration between SUZB3 stock price and its fundamentals:
- Pulp commodity prices
- Climate-weighted index of cultivation areas

**Signals:** Long when stock is undervalued relative to fundamentals, Short when overvalued.

---

## ✨ Features

### 📊 Data Pipeline & Infrastructure
- **8 Data Scrapers**: Equity (Yahoo Finance), FX (PTAX), Rates (SELIC), Climate (NASA Power, INMET), Macro (IBGE), Fundamentals, Benchmarks
- **Production Pipeline**: Automated data collection with versioning, monitoring, scheduling, and alerting
- **Data Quality**: Comprehensive validation (missing values, outliers, duplicates, freshness)
- **Caching**: Smart caching with TTL to reduce API calls
- **Manual Upload**: Support for CSV/Excel uploads (pulp prices, etc.)

### 🤖 Machine Learning & Modeling
- **Multiple Models**: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Ensemble Voting**: Majority, weighted, or threshold-based voting across models
- **Robust Validation**: Train/val/test splits with anti-overfitting measures (noise injection, regularization)
- **VECM Analysis**: Cointegration testing with Johansen test
- **AutoML**: TPOT integration for automated pipeline optimization

### 📈 Trading Strategies
- **Ensemble Voting Strategy**: Multi-model signal generation with risk-reward filtering
- **Risk-Reward Decision Model**: Secondary model evaluates risk-reward ratios (threshold 1.5)
- **Z-Score Strategy**: Mean-reversion based on synthetic index deviations
- **Risk-Managed Strategy**: Position sizing, stop-loss, take-profit, volatility filtering
- **Benchmark Comparison**: Compare vs. IMAT, IAGRO, IBOV indices

### 🎯 Risk Management
- **Volatility Analysis**: Historical volatility, GARCH models, regime detection
- **VaR/CVaR**: Value at Risk and Conditional VaR (historical, parametric, Monte Carlo)
- **Drawdown Analysis**: Maximum drawdown, average drawdown, underwater plots
- **Risk Metrics**: Sharpe, Sortino, Calmar, Omega ratios

### 📉 Forecasting
- **ARIMA Models**: Manual and auto-selection (AIC/BIC optimization)
- **Multi-Horizon Forecasts**: Configurable days ahead with confidence intervals

### 📊 Visualization & Analysis
- **30+ Automated Plots**: Comprehensive EDA, validation plots, model comparisons
- **Strategy Visualization**: Voting patterns, risk-reward analysis, ensemble comparisons
- **Data Quality Reports**: CSV reports with quality metrics

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda

### Quick Start

1. **Clone the repository:**
```bash
git clone <repository-url>
cd QuantSuzano
```

2. **Install the package:**
```bash
# Using pip
pip install -e .

# Or using setup.py (Windows)
python setup.py install --user
```

3. **Configure (optional):**
```bash
# Copy example config
cp config.example.json config.json
# Edit config.json with your settings
```

### Dependencies

Core dependencies are automatically installed. Optional dependencies:

- **TPOT** (for AutoML): `pip install tpot`
- **XGBoost** (for XGBoost model): `pip install xgboost`
- **LightGBM** (for LightGBM model): `pip install lightgbm`

---

## 📚 Usage

### Command-Line Interface

The project includes a comprehensive CLI with 30+ commands:

#### Core Pipeline Commands

```bash
# Data ingestion
python -m eda.cli ingest

# Robust synthetic index
python -m eda.cli synthetic-robust

# VECM cointegration analysis
python -m eda.cli vecm

# Comprehensive EDA (30+ plots)
python -m eda.cli eda-comprehensive

# Risk analysis
python -m eda.cli risk-analysis

# ARIMA forecasting
python -m eda.cli forecast-arima --horizon 30
```

#### Ensemble Strategy Commands

```bash
# Run ensemble voting strategy with risk-reward
python -m eda.cli strategy-ensemble

# With custom parameters
python -m eda.cli strategy-ensemble \
    --voting-method weighted \
    --risk-reward-threshold 2.0 \
    --z-threshold 2.5

# Compare ensemble vs individual models
python -m eda.cli strategy-compare
```

#### Advanced Commands

```bash
# Model comparison (Ridge, XGBoost, LightGBM, etc.)
python -m eda.cli model-comparison

# TPOT AutoML (optional, requires tpot)
python -m eda.cli automl-tpot --generations 10 --population 50

# Strategy optimization
python -m eda.cli strategy-optimize

# Benchmark comparison
python -m eda.cli benchmark-compare
```

#### Production Pipeline Commands

```bash
# Run data pipeline
python -m eda.cli pipeline-run

# Monitor pipeline health
python -m eda.cli pipeline-monitor

# Start automated scheduler
python -m eda.cli scheduler-start

# Manual data upload
python -m eda.cli pipeline-upload pulp_prices.csv --source pulp_prices
```

### Batch Scripts (Windows)

For Windows users, convenient batch scripts are provided:

```batch
# Complete analysis pipeline
RUN_COMPLETE_ANALYSIS_FIXED.bat

# Ensemble strategy
RUN_ENSEMBLE_STRATEGY.bat

# Reinstall package
REINSTALL_PACKAGE.bat
```

### Python API

```python
from eda.strategies import EnsembleStrategy
import pandas as pd

# Load data
df = pd.read_parquet('data/out/merged.parquet')

# Initialize and fit strategy
strategy = EnsembleStrategy(
    voting_method='majority',
    risk_reward_threshold=1.5,
    z_threshold=2.0,
)
strategy.fit(df, target_col='suzb_r')

# Generate signals
signals_df = strategy.generate_signals()

# Run backtest
metrics, backtest_results = strategy.backtest(signals_df)

# Save results
strategy.save_results(signals_df, metrics, backtest_results)
```

---

## 📁 Project Structure

```
QuantSuzano/
├── src/eda/                    # Main package
│   ├── scrapers/               # Data source scrapers (8 scrapers)
│   │   ├── bcb_extended.py     # Brazilian Central Bank (PTAX, SELIC)
│   │   ├── yfinance_robust.py  # Yahoo Finance (SUZB3)
│   │   ├── nasa_power.py       # NASA climate data
│   │   ├── inmet_climate.py    # INMET climate data
│   │   ├── ibge_macro.py       # IBGE macroeconomic data
│   │   ├── fundamentals_suzano.py # Company fundamentals
│   │   └── ...
│   ├── pipeline/                # Production pipeline (8 modules)
│   │   ├── orchestrator.py     # Main pipeline coordinator
│   │   ├── versioning.py      # Data versioning
│   │   ├── monitoring.py       # Health monitoring
│   │   ├── scheduler.py        # Automated scheduling
│   │   ├── alerting.py         # Email/Slack alerts
│   │   └── ...
│   ├── strategies/             # Trading strategies
│   │   ├── voting.py           # Ensemble voting signal generation
│   │   ├── risk_reward.py      # Risk-reward decision model
│   │   ├── ensemble_strategy.py # Combined ensemble strategy
│   │   ├── risk_managed.py     # Risk-managed strategy
│   │   └── benchmarks.py       # Benchmark comparison
│   ├── automl/                 # AutoML & model comparison
│   │   ├── model_comparison.py # Compare multiple models
│   │   └── tpot_models.py     # TPOT AutoML (optional)
│   ├── risk/                   # Risk management (4 modules)
│   │   ├── volatility.py       # Volatility analysis
│   │   ├── var.py             # VaR/CVaR
│   │   ├── drawdowns.py       # Drawdown analysis
│   │   └── metrics.py          # Risk metrics
│   ├── forecasting/            # Time series forecasting
│   │   └── classical.py        # ARIMA models
│   ├── cli.py                  # CLI interface (30+ commands)
│   ├── features.py             # Feature engineering
│   ├── synthetic_robust.py     # Robust synthetic index
│   ├── models.py               # VECM & time series models
│   ├── backtest.py            # Backtesting framework
│   ├── plots.py               # Standard visualizations
│   ├── plots_validation.py    # Validation plots
│   └── ...
├── data/
│   ├── raw/                    # Raw CSV files (fallbacks)
│   ├── cache/                  # Scraper cache
│   ├── interim/                # Intermediate processed data
│   ├── out/                    # Final outputs
│   │   ├── plots/              # All visualizations
│   │   └── *.parquet           # Processed datasets
│   └── versions/               # Versioned data snapshots
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
├── config.example.json         # Configuration template
├── setup.py                    # Package setup
├── pyproject.toml              # Modern Python packaging
└── README.md                   # This file
```

---

## 🔧 Configuration

Copy `config.example.json` to `config.json` and customize:

```json
{
  "data": {
    "start_date": "2020-01-01",
    "business_frequency": "B",
    "rolling_window": 60,
    "z_threshold": 2.0
  },
  "scrapers": {
    "cache_ttl_hours": 24,
    "retry_attempts": 3,
    "rate_limit_seconds": 0.5
  },
  "sources": {
    "suzb3": {
      "enabled": true,
      "required": true,
      "ticker": "SUZB3.SA"
    },
    "pulp_prices": {
      "enabled": true,
      "manual_upload": true
    }
  }
}
```

---

## 📊 Output Files

### Main Results

- `data/out/merged.parquet` - Processed feature matrix
- `data/out/synthetic_robust.parquet` - Robust synthetic index results
- `data/out/ensemble_signals.parquet` - Ensemble voting signals
- `data/out/ensemble_backtest.parquet` - Full backtest results
- `data/out/ensemble_metrics.csv` - Performance metrics
- `data/out/ensemble_comparison.csv` - Model comparison

### Visualizations

- `data/out/plots/eda/` - 30+ exploratory plots
- `data/out/plots/models/` - Model performance comparisons
- `data/out/plots/strategies/` - Strategy visualizations
  - `voting_patterns.png` - How each model voted
  - `risk_reward_analysis.png` - Risk-reward ratios over time
  - `ensemble_comparison.png` - Performance comparison

### Reports

- `data/out/data_quality_report.csv` - Data quality assessment
- `data/out/risk_analysis.csv` - Risk metrics
- `data/out/vecm_summary.txt` - VECM cointegration results

---

## 🎯 Key Features Explained

### Ensemble Voting Strategy

1. **Training Phase**: Models trained on 70% of data (train set)
2. **Signal Generation**: Predictions on TEST data only (15% of data)
3. **Voting**: Multiple models vote on signals (majority, weighted, or threshold)
4. **Risk-Reward Filter**: Only execute trades with risk-reward ratio > 1.5
5. **Execution**: Final trades executed based on filtered signals

### Risk-Reward Decision Model

- **Expected Return**: Based on model prediction deviation from current level
- **Expected Risk**: Rolling volatility from uncorrelated market features
- **Features Used**: IBOV volatility, SELIC changes, PTAX volatility, climate data
- **NOT Used**: SUZB3-specific returns or directly correlated features
- **Threshold**: 1.5 (configurable)

### Data Sources

- **Equity**: SUZB3 (Yahoo Finance)
- **FX**: PTAX USD/BRL (Brazilian Central Bank)
- **Rates**: SELIC (Brazilian Central Bank)
- **Commodities**: Pulp prices (manual upload)
- **Climate**: NASA Power (Tres Lagoas, Imperatriz)
- **Macro**: IBGE (IPCA, GDP, etc.)
- **Benchmarks**: IMAT, IAGRO, IBOV

---

## 📈 Recent Results

### Test Period: June 5 - October 31, 2025 (107 days)

**Best Model: GradientBoosting**
- Return: +0.71%
- Sharpe Ratio: 0.626
- Max Drawdown: -1.01%
- Number of Trades: 6

**Ensemble Strategy (with Risk-Reward)**
- Return: 0.00% (all trades filtered by risk-reward threshold)
- Result: Preserved capital during market decline (-1.76%)

**Market Context**: Period of market decline (-1.76% for Buy & Hold SUZB3)

---

## 🧪 Testing

### Run Complete Analysis

```bash
# Windows
RUN_COMPLETE_ANALYSIS_FIXED.bat

# Or manually
python -m eda.cli ingest
python -m eda.cli synthetic-robust
python -m eda.cli vecm
python -m eda.cli risk-analysis
python -m eda.cli strategy-ensemble
```

### Validate Installation

```bash
python -c "from eda.strategies import EnsembleStrategy; print('OK')"
python -m eda.cli --help
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **John Froelich**: For the inspiration behind "The Froelich Engine"
- **Statsmodels**: For VECM and time series analysis
- **Scikit-learn**: For machine learning models
- **Yahoo Finance**: For equity data
- **Brazilian Central Bank**: For PTAX and SELIC data
- **NASA Power**: For climate data

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

## 🔗 Related Documentation

- [Complete Report (Portuguese)](RELATORIO_COMPLETO_QUANTSUZANO.md)
- [Configuration Guide](config.example.json)
- [Jupyter Notebooks](notebooks/)

---

## 📊 Version History

- **v0.1.0** (Current): Initial release with ensemble voting strategy and risk-reward decision model
  - 8 data scrapers
  - 7 machine learning models
  - Ensemble voting with risk-reward filtering
  - Production pipeline with monitoring
  - 30+ automated visualizations

---

**Built with ❤️ for quantitative finance**

