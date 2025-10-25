# QuantSuzano Quick Start Guide

Get up and running with the production data pipeline in 5 minutes!

## Prerequisites

- Python 3.11+
- Internet connection (for API access)

## Step 1: Install Dependencies (30 seconds)

```bash
pip install -e .
```

This installs all required packages including the new ones (schedule, scikit-learn, seaborn).

## Step 2: Optional Configuration (1 minute)

For basic usage, no configuration needed! The pipeline works with defaults.

For alerts and customization:

```bash
cp config.example.json config.json
# Edit config.json to add email/Slack credentials
```

## Step 3: Fetch Real Data (2-3 minutes)

```bash
python -m eda.cli pipeline-run
```

This will:
- Fetch PTAX and SELIC from Brazilian Central Bank
- Fetch SUZB3 stock prices from Yahoo Finance
- Fetch climate data from NASA POWER
- Validate all data
- Save versions
- Export CSVs to `data/raw/`

**Expected Output:**
```
[PIPELINE] Starting production data pipeline
============================================================
[PIPELINE] Fetching: ptax
[BCB] Fetching series ptax (1)
[ptax] Successfully fetched 1200+ rows
[VALIDATOR] Validating ptax...
[VALIDATION PASSED]
[VERSION] Saved version: ptax_20251025_120000 (1200 rows)
...
[SUCCESS] Pipeline completed!
   Updated 4 sources
```

## Step 4: Upload Pulp Prices (1 minute)

Pulp prices need manual upload (FOEX has no public API).

**Create template:**
```bash
python -m eda.cli pipeline-template --source pulp_prices
```

**Fill in the template** (`data/manual_uploads/template_pulp_prices.csv`):
```csv
date,price,type
2023-01-01,820.0,BHKP
2023-02-01,825.0,BHKP
2023-03-01,830.0,BHKP
# ... add your data
```

**Upload:**
```bash
python -m eda.cli pipeline-upload --file-path data/manual_uploads/template_pulp_prices.csv
```

## Step 5: Check Status (10 seconds)

```bash
python -m eda.cli pipeline-monitor
```

**Expected Output:**
```
======================================================================
DATA PIPELINE STATUS REPORT
Generated: 2025-10-25T12:00:00
======================================================================

OVERALL HEALTH: HEALTHY (Score: 95.0/100)
  Fresh sources:   4/5
  Stale sources:   0/5
  Missing sources: 1/5  # pulp_prices until you upload

DATA FRESHNESS:
  ✓ ptax                 fresh      (age: 0.1h)
  ✓ selic                fresh      (age: 0.1h)
  ✓ suzb3                fresh      (age: 0.1h)
  ✓ climate_nasa         fresh      (age: 0.1h)
  ✗ pulp_prices          missing
...
```

## Step 6: Run Analysis (1-2 minutes)

```bash
python -m eda.cli all-robust
```

This runs the complete pipeline:
1. Ingest and merge data
2. Fit robust synthetic index (RidgeCV)
3. Generate validation plots
4. Fit VECM model
5. Generate comprehensive report

**Results saved to:**
- `data/out/plots/` - All visualizations
- `data/out/synthetic_robust.parquet` - Model results
- `data/out/metrics_robust.csv` - Performance metrics

## Next Steps

### Automate Updates

**Option A: Built-in scheduler (recommended for testing)**
```bash
python -m eda.cli scheduler-start
# Runs indefinitely, press Ctrl+C to stop
```

**Option B: Cron (recommended for production)**
```bash
python -m eda.cli scheduler-export-cron
chmod +x scripts/pipeline_cron.sh
crontab -e
# Add: 0 0 * * * /path/to/QuantSuzano/scripts/pipeline_cron.sh
```

### Configure Alerts

Edit `config.json`:

**Email (Gmail):**
```json
"alerting": {
  "email": {
    "enabled": true,
    "username": "your_email@gmail.com",
    "password": "your_app_password",  # Generate at https://myaccount.google.com/apppasswords
    "to_addrs": ["recipient@example.com"]
  }
}
```

**Slack:**
```json
"alerting": {
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }
}
```

### View Version History

```bash
# All sources
python -m eda.cli pipeline-versions

# Specific source
python -m eda.cli pipeline-versions --source ptax

# Cleanup old versions
python -m eda.cli pipeline-cleanup --keep-last 10
```

### Explore with Jupyter

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Troubleshooting

### Issue: "No module named 'eda'"

**Solution:**
```bash
pip install -e .
```

### Issue: API rate limits (Yahoo Finance)

**Solution:** Use caching and reduce update frequency:
```bash
# Force full fetch with cache
python -m eda.cli pipeline-run

# Check cache
ls data/cache/yfinance/
```

### Issue: Stale data

**Solution:**
```bash
# Force full refresh
python -m eda.cli pipeline-run --force-full

# Check what's old
python -m eda.cli pipeline-monitor
```

### Issue: Validation warnings

**Solution:** Usually safe to ignore unless error rate is high. Check specifics:
```bash
python -m eda.cli pipeline-monitor
```

## Common Commands Reference

| Task | Command |
|------|---------|
| Fetch all data | `python -m eda.cli pipeline-run` |
| Check status | `python -m eda.cli pipeline-monitor` |
| Upload manual data | `python -m eda.cli pipeline-upload --file-path FILE` |
| Run analysis | `python -m eda.cli all-robust` |
| Start scheduler | `python -m eda.cli scheduler-start` |
| View versions | `python -m eda.cli pipeline-versions` |

## File Locations

| Type | Location |
|------|----------|
| Raw data (CSV) | `data/raw/*.csv` |
| Versioned data | `data/versions/*.parquet` |
| Analysis results | `data/out/` |
| Plots | `data/out/plots/` |
| Logs | `data/logs/alerts.jsonl` |
| Cache | `data/cache/` |

## What to Monitor

**Daily:**
- Run `pipeline-monitor` to check health
- Ensure no sources are stale (>48h)

**Weekly:**
- Upload pulp prices (if manual)
- Review alerts (if configured)

**Monthly:**
- Clean up old versions: `pipeline-cleanup`
- Review error rates

## Getting Help

1. Check documentation:
   - [README.md](README.md) - Overview
   - [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Detailed usage
   - [DATA_SOURCES.md](DATA_SOURCES.md) - Data source info

2. Check pipeline health:
   ```bash
   python -m eda.cli pipeline-monitor
   ```

3. Check logs:
   ```bash
   cat data/logs/alerts.jsonl
   ```

4. Test individual components:
   ```bash
   python -m eda.cli pipeline-run --sources ptax
   ```

## Success Checklist

- [ ] Dependencies installed (`pip install -e .`)
- [ ] Data fetched (`pipeline-run`)
- [ ] Status healthy (`pipeline-monitor`)
- [ ] Pulp prices uploaded
- [ ] Analysis completed (`all-robust`)
- [ ] Results viewed (`data/out/plots/`)
- [ ] Optional: Alerts configured
- [ ] Optional: Scheduler running

## You're Ready!

The pipeline is now running and you have:
- ✅ Real data from 4+ sources
- ✅ Automated updates
- ✅ Version control
- ✅ Monitoring
- ✅ Comprehensive analysis

Enjoy your production-grade quantitative analysis platform! 🚀

