# QuantSuzano: Production-Grade EDA & Quant Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Reproducible exploratory data analysis and quantitative modeling pipeline for Suzano/Celulose market analysis.**

This project provides a comprehensive, production-ready data infrastructure for analyzing relationships between:
- Exchange rates (PTAX USD/BRL)
- Interest rates (SELIC)
- Climate data (temperature, precipitation, solar radiation)
- Pulp prices (FOEX BHKP)
- Suzano stock prices (SUZB3.SA)

## 🚀 Key Features

### Production-Grade Data Infrastructure
- ✅ **Automated Data Collection**: Scrapers for BCB, Yahoo Finance, NASA POWER
- ✅ **Incremental Updates**: Only fetch new data since last update
- ✅ **Data Validation**: Comprehensive quality checks
- ✅ **Version Control**: Track all data changes with rollback capability
- ✅ **Caching**: Reduce API calls with intelligent caching
- ✅ **Retry Logic**: Exponential backoff for failed requests
- ✅ **Rate Limiting**: Respect API limits automatically

### Monitoring & Alerting
- ✅ **Health Monitoring**: Real-time pipeline health checks
- ✅ **Alerting**: Email and Slack notifications
- ✅ **Scheduler**: Automated updates (cron-like)
- ✅ **Logging**: Comprehensive activity logs
- ✅ **Metrics**: Data freshness, error rates, etc.

### Robust Quantitative Modeling
- ✅ **Synthetic Index**: RidgeCV with temporal cross-validation
- ✅ **Anti-Overfitting**: Noise injection, regularization, proper train/val/test split
- ✅ **VECM Analysis**: Cointegration and long-run relationships
- ✅ **Z-Score Signals**: Mean-reversion trading strategy
- ✅ **Backtesting**: Simple PnL simulation with transaction costs
- ✅ **Comprehensive Metrics**: MAE, RMSE, R², Hit Ratio, IC, Directional Accuracy

### Developer Experience
- ✅ **CLI Interface**: Intuitive command-line tools
- ✅ **Type Hints**: Full type annotations
- ✅ **Documentation**: Comprehensive guides
- ✅ **Modular Design**: Easy to extend
- ✅ **Configuration**: JSON-based settings

## 📋 Requirements

- Python 3.11+
- Internet connection (for data fetching)

## 🔧 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/QuantSuzano.git
cd QuantSuzano

# Install dependencies
pip install -e .
```

### 2. Configure

```bash
# Copy example config
cp config.example.json config.json

# Edit config.json to set your preferences
# (optional: email/Slack alerts, data sources, etc.)
```

### 3. Fetch Data

```bash
# Fetch all data sources
python -m eda.cli pipeline-run

# Check status
python -m eda.cli pipeline-monitor
```

### 4. Run Analysis

```bash
# Run complete robust pipeline
python -m eda.cli all-robust

# Or run steps individually:
python -m eda.cli ingest
python -m eda.cli synthetic-robust
python -m eda.cli validate
python -m eda.cli vecm
python -m eda.cli report
```

### 5. View Results

Results are saved in:
- `data/out/plots/` - All visualizations
- `data/out/synthetic_robust.parquet` - Model results
- `data/out/metrics_robust.csv` - Performance metrics
- `data/out/vecm_summary.txt` - VECM analysis

## 📊 Data Sources

| Source | Type | Frequency | Status |
|--------|------|-----------|--------|
| **PTAX** (USD/BRL) | BCB API | Daily | ✅ Automated |
| **SELIC** (Interest Rate) | BCB API | Daily | ✅ Automated |
| **SUZB3** (Stock Price) | Yahoo Finance | Hourly | ✅ Automated |
| **Climate** | NASA POWER | Weekly | ✅ Automated |
| **Pulp Prices** | Manual Upload | Weekly | 📤 Manual |

See [DATA_SOURCES.md](DATA_SOURCES.md) for detailed information.

## 📁 Project Structure

```
QuantSuzano/
├── src/eda/                    # Main package
│   ├── scrapers/               # Data scrapers
│   │   ├── base.py             # Base scraper class
│   │   ├── bcb_extended.py     # BCB API scraper
│   │   ├── yfinance_robust.py  # Yahoo Finance scraper
│   │   ├── nasa_power.py       # NASA climate data
│   │   └── inmet_climate.py    # INMET weather data
│   ├── pipeline/               # Pipeline infrastructure
│   │   ├── orchestrator.py     # Main pipeline coordinator
│   │   ├── validator.py        # Data quality checks
│   │   ├── versioning.py       # Version management
│   │   ├── incremental.py      # Incremental updates
│   │   ├── scheduler.py        # Automated scheduling
│   │   ├── monitoring.py       # Health monitoring
│   │   ├── alerting.py         # Alert system
│   │   └── manual_upload.py    # Manual data uploads
│   ├── config.py               # Configuration
│   ├── loaders.py              # Data loaders (legacy)
│   ├── features.py             # Feature engineering
│   ├── synthetic.py            # OLS synthetic index
│   ├── synthetic_robust.py     # Robust synthetic index
│   ├── models.py               # VECM and time series models
│   ├── plots.py                # Basic plots
│   ├── plots_validation.py     # Validation plots
│   ├── backtest.py             # Backtesting logic
│   ├── metrics.py              # Performance metrics
│   ├── utils_split.py          # Train/val/test split
│   └── cli.py                  # CLI interface
├── data/
│   ├── raw/                    # Raw data (CSV)
│   ├── interim/                # Processed data
│   ├── out/                    # Analysis outputs
│   ├── versions/               # Versioned data
│   ├── cache/                  # Scraper cache
│   ├── manual_uploads/         # Manual data staging
│   └── logs/                   # Logs
├── notebooks/
│   └── EDA.ipynb               # Exploratory analysis
├── config.example.json         # Example configuration
├── pyproject.toml              # Dependencies
├── README.md                   # This file
├── DATA_SOURCES.md             # Data source documentation
├── PIPELINE_GUIDE.md           # Pipeline usage guide
└── SETUP_GUIDE.md              # Setup instructions

```

## 🎯 CLI Commands

### Pipeline Commands

```bash
# Fetch all data sources
python -m eda.cli pipeline-run

# Fetch specific sources
python -m eda.cli pipeline-run --sources ptax,selic

# Force full historical fetch
python -m eda.cli pipeline-run --force-full

# Monitor pipeline health
python -m eda.cli pipeline-monitor

# Upload manual data (pulp prices)
python -m eda.cli pipeline-upload --file-path data/manual_uploads/pulp_prices.csv

# Create upload template
python -m eda.cli pipeline-template --source pulp_prices

# View version history
python -m eda.cli pipeline-versions

# Clean up old versions
python -m eda.cli pipeline-cleanup --keep-last 10
```

### Scheduler Commands

```bash
# Start automated scheduler (runs indefinitely)
python -m eda.cli scheduler-start

# Export cron script
python -m eda.cli scheduler-export-cron
```

### Analysis Commands

```bash
# Ingest and build features
python -m eda.cli ingest --start 2020-01-01

# Fit robust synthetic index
python -m eda.cli synthetic-robust --train-ratio 0.70 --noise-alpha 0.10

# Generate validation plots
python -m eda.cli validate

# Fit VECM model
python -m eda.cli vecm

# Generate comprehensive report
python -m eda.cli report

# Run complete pipeline
python -m eda.cli all-robust --start 2020-01-01
```

## 📈 Methodology

### 1. Data Collection & Validation

- Automated fetching from multiple APIs
- Comprehensive validation (completeness, consistency, quality, freshness)
- Incremental updates to minimize API calls
- Versioning for reproducibility

### 2. Feature Engineering

- Log returns for all price series
- Derived features (e.g., pulp price in BRL)
- Climate lags (temperature, precipitation)
- Business day frequency resampling

### 3. Synthetic Index (Robust)

- **Model**: RidgeCV (regularized linear regression)
- **Target**: SUZB3 log returns
- **Features**: PTAX, SELIC, Climate, Pulp prices (all returns/changes)
- **Cross-Validation**: TimeSeriesSplit (5 folds)
- **Data Split**: 70% train, 15% validation, 15% test
- **Anti-Overfitting**: Noise injection (10%), regularization
- **Evaluation**: MAE, RMSE, R², Hit Ratio, IC, Directional Accuracy

### 4. Z-Score Analysis

- Rolling 60-day mean and std of (actual - predicted)
- Z-score = (error - rolling_mean) / rolling_std
- Trading signals: Long when z < -2, Short when z > +2

### 5. VECM (Vector Error Correction Model)

- Tests for cointegration (Johansen test)
- Estimates long-run equilibrium relationships
- Short-run dynamics and adjustment speeds

### 6. Backtesting

- Simple mean-reversion strategy based on z-scores
- Transaction costs: 0.1% per trade
- Metrics: Sharpe ratio, Sortino ratio, Max drawdown, Win rate

## 📊 Output Visualizations

The pipeline generates comprehensive visualizations:

- **Time Series**: Levels and returns for all variables
- **Correlations**: Heatmaps and pair plots
- **Predictions**: Actual vs predicted (train/val/test splits)
- **Residuals**: Diagnostics (ACF, Q-Q plots, distribution)
- **Z-Scores**: Time series with trading signals
- **Backtesting**: Cumulative PnL and drawdown
- **VECM**: Impulse responses and residual diagnostics

## 🔔 Alerting & Monitoring

### Email Alerts (Gmail)

1. Enable 2FA on Gmail
2. Create App Password: https://myaccount.google.com/apppasswords
3. Update `config.json`:

```json
"alerting": {
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your_email@gmail.com",
    "password": "your_app_password",
    "from_addr": "your_email@gmail.com",
    "to_addrs": ["recipient@example.com"]
  }
}
```

### Slack Alerts

1. Create webhook: https://api.slack.com/messaging/webhooks
2. Update `config.json`:

```json
"alerting": {
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }
}
```

## 🔄 Automated Scheduling

### Built-in Scheduler

```bash
python -m eda.cli scheduler-start
```

### Cron (Linux/Mac)

```bash
python -m eda.cli scheduler-export-cron
chmod +x scripts/pipeline_cron.sh
crontab -e
# Add: 0 0 * * * /path/to/scripts/pipeline_cron.sh
```

### Windows Task Scheduler

Create batch script and schedule in Task Scheduler.

## 📚 Documentation

- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Comprehensive pipeline usage
- [DATA_SOURCES.md](DATA_SOURCES.md) - Data source details
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup instructions
- [notebooks/EDA.ipynb](notebooks/EDA.ipynb) - Exploratory analysis

## 🛠️ Development

### Adding New Data Sources

1. Create scraper in `src/eda/scrapers/your_scraper.py`
2. Inherit from `BaseScraper`
3. Implement `_fetch_data()` method
4. Register in `src/eda/scrapers/__init__.py`
5. Add to `SOURCES` in `src/eda/pipeline/orchestrator.py`

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md#adding-new-data-sources) for details.

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

## 🐛 Troubleshooting

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md#troubleshooting) for common issues and solutions.

Quick checks:

```bash
# Check pipeline health
python -m eda.cli pipeline-monitor

# View recent versions
python -m eda.cli pipeline-versions

# Check logs
cat data/logs/alerts.jsonl
```

## 📝 To-Do / Roadmap

- [ ] Web dashboard (Streamlit/Dash)
- [ ] Real-time data streaming
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Machine learning model deployment
- [ ] More sophisticated trading strategies
- [ ] Multi-asset support
- [ ] Grafana/Prometheus integration

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sources: BCB (Central Bank of Brazil), Yahoo Finance, NASA POWER
- Libraries: pandas, statsmodels, scikit-learn, typer
- Inspiration: Quantitative finance community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for quantitative analysis and reproducible research**
