# Production Data Pipeline Guide

## Overview

The QuantSuzano pipeline provides a comprehensive, production-grade data infrastructure for automated data collection, validation, versioning, and monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ BCB API  │ Yahoo    │ NASA     │ INMET    │ Manual Upload  │
│ (PTAX,   │ Finance  │ POWER    │ Climate  │ (Pulp Prices)  │
│ SELIC)   │ (SUZB3)  │ (Climate)│          │                │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────────┬───────┘
     │          │          │          │              │
     └──────────┴──────────┴──────────┴──────────────┘
                         │
                    ┌────▼────┐
                    │ SCRAPERS│  (retry, cache, rate limit)
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │VALIDATOR│  (quality checks)
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │VERSIONING│ (change tracking)
                    └────┬────┘
                         │
                ┌────────┴────────┐
                │                 │
           ┌────▼────┐      ┌────▼────┐
           │SCHEDULER│      │PIPELINE │
           │         │      │MONITOR  │
           └─────────┘      └─────────┘
                │                │
                └────────┬───────┘
                         │
                    ┌────▼────┐
                    │ALERTING │  (email, slack)
                    └─────────┘
```

## Components

### 1. Scrapers (`src/eda/scrapers/`)

Base functionality:
- Retry logic with exponential backoff
- Response caching with TTL
- Rate limiting
- Error handling and logging

Available scrapers:
- **BCBExtendedScraper**: PTAX and SELIC from BCB
- **YFinanceRobustScraper**: Stock prices from Yahoo Finance
- **NASAPowerScraper**: Global climate data
- **INMETClimateScraper**: Brazilian weather data (optional)

### 2. Pipeline Orchestrator (`src/eda/pipeline/orchestrator.py`)

Coordinates:
- Multiple data sources
- Incremental updates
- Data validation
- Version management
- Error recovery

### 3. Data Validation (`src/eda/pipeline/validator.py`)

Checks:
- Completeness (no missing data)
- Consistency (correct types, sorted)
- Quality (outliers, duplicates)
- Freshness (data age)

### 4. Versioning (`src/eda/pipeline/versioning.py`)

Features:
- Change detection (hash-based)
- Version history tracking
- Rollback capability
- Metadata storage

### 5. Incremental Updates (`src/eda/pipeline/incremental.py`)

Strategy:
- Track last update timestamp
- Fetch only new data
- Merge with existing data
- Handle gaps and overlaps

### 6. Scheduler (`src/eda/pipeline/scheduler.py`)

Features:
- Cron-like scheduling
- Per-source update frequencies
- Automatic retry on failure
- Execution history

### 7. Monitoring (`src/eda/pipeline/monitoring.py`)

Provides:
- Data freshness checks
- Health score calculation
- Error rate monitoring
- Status reports

### 8. Alerting (`src/eda/pipeline/alerting.py`)

Channels:
- Email (SMTP)
- Slack (webhook)
- File logging
- Console

## Quick Start

### Install Dependencies

```bash
pip install -e .
```

### Configuration

Copy the example config:

```bash
cp config.example.json config.json
```

Edit `config.json` to set your preferences (email, slack, data sources, etc.).

### First Run

Fetch all data sources:

```bash
python -m eda.cli pipeline-run
```

This will:
1. Initialize the pipeline
2. Fetch data from all enabled sources
3. Validate the data
4. Save versioned copies
5. Export to CSV in `data/raw/`

### Monitor Status

Check pipeline health:

```bash
python -m eda.cli pipeline-monitor
```

Output:
```
======================================================================
DATA PIPELINE STATUS REPORT
Generated: 2025-10-25T10:30:00
======================================================================

OVERALL HEALTH: HEALTHY (Score: 95.0/100)
  Fresh sources:   5/5
  Stale sources:   0/5
  Missing sources: 0/5

DATA FRESHNESS:
  ✓ ptax                 fresh      (age: 2.5h)
  ✓ selic                fresh      (age: 2.5h)
  ✓ suzb3                fresh      (age: 0.3h)
  ✓ climate_nasa         fresh      (age: 24.1h)
  ✓ pulp_prices          fresh      (age: 72.0h)
...
```

### Manual Upload (Pulp Prices)

Generate template:

```bash
python -m eda.cli pipeline-template --source pulp_prices
```

Fill in the template and upload:

```bash
python -m eda.cli pipeline-upload --file-path data/manual_uploads/pulp_prices.csv
```

### View Version History

```bash
python -m eda.cli pipeline-versions
```

Or for a specific source:

```bash
python -m eda.cli pipeline-versions --source ptax
```

## Automated Scheduling

### Option 1: Built-in Scheduler

Start the scheduler (runs indefinitely):

```bash
python -m eda.cli scheduler-start
```

This will:
- Schedule all sources based on `config.json`
- Run updates automatically
- Send alerts on failures
- Keep running until Ctrl+C

### Option 2: Cron Jobs (Linux/Mac)

Export cron script:

```bash
python -m eda.cli scheduler-export-cron
```

Install:

```bash
chmod +x scripts/pipeline_cron.sh
crontab -e
# Add: @reboot /path/to/scripts/pipeline_cron.sh
```

### Option 3: Windows Task Scheduler

Create a batch script:

```batch
@echo off
cd C:\path\to\QuantSuzano
python -m eda.cli pipeline-run
```

Schedule in Task Scheduler to run daily.

### Option 4: Airflow/Prefect

Create a DAG/Flow that calls:

```python
from eda.pipeline.orchestrator import DataPipeline

pipeline = DataPipeline()
results = pipeline.run()
```

## CLI Commands Reference

### Pipeline Commands

| Command | Description |
|---------|-------------|
| `pipeline-run` | Fetch and update all data sources |
| `pipeline-monitor` | Display pipeline health status |
| `pipeline-upload` | Manually upload data file |
| `pipeline-template` | Create upload template |
| `pipeline-versions` | View version history |
| `pipeline-cleanup` | Delete old versions |

### Scheduler Commands

| Command | Description |
|---------|-------------|
| `scheduler-start` | Start automated scheduler |
| `scheduler-export-cron` | Export cron script |

### Analysis Commands (Original)

| Command | Description |
|---------|-------------|
| `ingest` | Build features from data |
| `synthetic` | Fit synthetic index (OLS) |
| `synthetic-robust` | Fit robust synthetic index (RidgeCV) |
| `validate` | Generate validation plots |
| `vecm` | Fit VECM model |
| `report` | Generate full report |
| `all-robust` | Run complete pipeline |

## Advanced Usage

### Custom Update Frequency

Run specific sources only:

```bash
python -m eda.cli pipeline-run --sources ptax,selic
```

Force full historical fetch:

```bash
python -m eda.cli pipeline-run --force-full
```

Disable caching:

```bash
python -m eda.cli pipeline-run --no-cache
```

### Alerting Setup

#### Email (Gmail)

1. Enable 2FA on your Gmail account
2. Create an App Password: https://myaccount.google.com/apppasswords
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

#### Slack

1. Create an incoming webhook: https://api.slack.com/messaging/webhooks
2. Update `config.json`:

```json
"alerting": {
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }
}
```

### Data Versioning

List all versions:

```bash
python -m eda.cli pipeline-versions
```

View history for specific source:

```bash
python -m eda.cli pipeline-versions --source ptax
```

Clean up old versions (keep last 5):

```bash
python -m eda.cli pipeline-cleanup --keep-last 5
```

### Programmatic Access

```python
from eda.pipeline.orchestrator import DataPipeline
from eda.pipeline.monitoring import PipelineMonitor

# Initialize
pipeline = DataPipeline()

# Fetch specific sources
results = pipeline.run(sources=["ptax", "selic"])

# Get merged data
df = pipeline.get_merged_data()

# Monitor
monitor = PipelineMonitor(pipeline)
health = monitor.get_health_score()
print(f"Health: {health['score']}/100")
```

## Data Directory Structure

```
data/
├── raw/                    # CSV exports (backward compatible)
│   ├── ptax.csv
│   ├── selic.csv
│   ├── suzb3.csv
│   ├── climate_nasa.csv
│   └── pulp_prices.csv
├── interim/                # Processed data
│   └── merged.parquet
├── out/                    # Analysis outputs
│   ├── plots/
│   ├── synthetic_robust.parquet
│   ├── metrics_robust.csv
│   └── ...
├── versions/               # Versioned data (Parquet)
│   ├── ptax_20241025_103000.parquet
│   ├── ptax_20241024_103000.parquet
│   ├── ...
│   └── versions.json       # Metadata
├── cache/                  # Scraper cache
│   ├── bcb_extended/
│   ├── yfinance/
│   └── nasa_power/
├── manual_uploads/         # Manual upload staging
│   ├── pulp_prices.csv
│   └── template_pulp_prices.csv
└── logs/                   # Alert logs
    └── alerts.jsonl
```

## Monitoring Best Practices

1. **Daily Health Checks**: Run `pipeline-monitor` daily via cron
2. **Alert Configuration**: Set up email/Slack for critical failures
3. **Version Retention**: Keep 10-20 versions per source
4. **Cache Management**: Clear cache monthly if disk space is limited
5. **Data Freshness**: Monitor for sources older than 48 hours
6. **Error Rates**: Investigate if error rate exceeds 10%

## Troubleshooting

### Issue: API Rate Limits

**Symptoms**: 429 errors, frequent retries

**Solutions**:
- Increase `rate_limit_seconds` in config
- Reduce update frequency
- Use caching more aggressively
- Contact API provider for higher limits

### Issue: Stale Data

**Symptoms**: Data older than expected

**Solutions**:
- Check scheduler is running
- Verify internet connection
- Check API status pages
- Review error logs

### Issue: Validation Failures

**Symptoms**: Warnings in validation report

**Solutions**:
- Review specific validation messages
- Adjust thresholds in config (e.g., `outlier_std`)
- Check if data source format changed
- Update scraper if needed

### Issue: High Error Rate

**Symptoms**: Many failed updates

**Solutions**:
- Check logs for specific errors
- Verify API credentials (if applicable)
- Test individual scrapers
- Increase retry attempts

### Issue: Disk Space

**Symptoms**: Out of disk space

**Solutions**:
- Run cleanup: `pipeline-cleanup --keep-last 5`
- Clear cache: `rm -rf data/cache/`
- Archive old data
- Adjust retention policy

## Performance Optimization

### Caching

Default TTL by scraper:
- BCB: 6 hours
- YFinance: 1 hour
- NASA: 1 week

Adjust in scraper initialization if needed.

### Parallel Fetching

The pipeline fetches sources sequentially by default. For parallel execution, modify `orchestrator.py` to use `concurrent.futures`:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(self.fetch_source, src) for src in sources]
    results = [f.result() for f in futures]
```

### Incremental Updates

The pipeline automatically uses incremental updates. To force full fetch:

```bash
python -m eda.cli pipeline-run --force-full
```

## Security

### Credentials

- Store in `config.json` (gitignored)
- Use environment variables for CI/CD:
  ```bash
  export EMAIL_PASSWORD="your_password"
  export SLACK_WEBHOOK="your_webhook"
  ```
- Use secrets managers (AWS Secrets Manager, Azure Key Vault, etc.)

### API Keys

Currently, all APIs are public and don't require keys. If adding authenticated APIs:

1. Store credentials securely
2. Use HTTPS only
3. Rotate keys regularly
4. Monitor usage

## Support

For issues or questions:

1. Check this guide
2. Review `DATA_SOURCES.md`
3. Check logs in `data/logs/`
4. Run diagnostics: `pipeline-monitor`
5. Open an issue on GitHub

## Roadmap

Potential enhancements:

- [ ] Web dashboard (Streamlit/Dash)
- [ ] Grafana integration
- [ ] Prometheus metrics
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Real-time data streaming
- [ ] Machine learning model deployment
- [ ] Backtesting framework integration
- [ ] Multi-asset support
- [ ] Cloud storage integration (S3, Azure Blob)

## License

See LICENSE file for details.

