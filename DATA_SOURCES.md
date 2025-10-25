# Data Sources Documentation

## Overview

This document describes all data sources used in the QuantSuzano pipeline, their APIs, update frequencies, and requirements.

## Automated Sources (API/Web Scraping)

### 1. PTAX (USD/BRL Exchange Rate)

- **Source**: Brazilian Central Bank (BCB)
- **API**: SGS (Sistema Gerenciador de Séries Temporais)
- **Series Code**: 1
- **URL**: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados`
- **Frequency**: Daily (business days)
- **Update Frequency**: Every 24 hours
- **Authentication**: None required
- **Format**: JSON
- **Status**: ✅ Automated
- **Reliability**: High

**Implementation**: `src/eda/scrapers/bcb_extended.py`

### 2. SELIC (Interest Rate)

- **Source**: Brazilian Central Bank (BCB)
- **API**: SGS
- **Series Code**: 432
- **URL**: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados`
- **Frequency**: Daily
- **Update Frequency**: Every 24 hours
- **Authentication**: None required
- **Format**: JSON
- **Status**: ✅ Automated
- **Reliability**: High

**Implementation**: `src/eda/scrapers/bcb_extended.py`

### 3. SUZB3 (Suzano Stock Price)

- **Source**: Yahoo Finance
- **API**: yfinance Python library
- **Ticker**: SUZB3.SA
- **Frequency**: Daily
- **Update Frequency**: Every 1 hour (during market hours)
- **Authentication**: None required
- **Format**: DataFrame (OHLCV)
- **Status**: ✅ Automated
- **Reliability**: Medium (rate limits may apply)

**Notes**:
- Yahoo Finance may implement rate limiting
- Fallback to CSV available if API fails
- Only Close price used by default

**Implementation**: `src/eda/scrapers/yfinance_robust.py`

### 4. Climate Data (NASA POWER)

- **Source**: NASA POWER API
- **API**: Prediction Of Worldwide Energy Resources
- **URL**: `https://power.larc.nasa.gov/api/temporal/daily/point`
- **Locations**: 
  - Três Lagoas, MS (-20.75, -51.68)
  - Imperatriz, MA (-5.53, -47.48)
  - Suzano, SP (-23.55, -46.31)
  - Aracruz, ES (-19.82, -40.27)
- **Parameters**:
  - T2M: Temperature at 2m (°C)
  - T2M_MAX/MIN: Max/min temperature
  - PRECTOTCORR: Precipitation (mm/day)
  - RH2M: Relative humidity (%)
  - WS2M: Wind speed (m/s)
  - ALLSKY_SFC_SW_DWN: Solar radiation (MJ/m²/day)
- **Frequency**: Daily
- **Update Frequency**: Every 168 hours (weekly)
- **Authentication**: None required
- **Format**: JSON
- **Status**: ✅ Automated
- **Reliability**: High
- **Latency**: 1-2 days behind real-time

**Notes**:
- Data is gridded (0.5° x 0.5°)
- Historical data from 1981 onwards
- Aggregated across multiple mill locations

**Implementation**: `src/eda/scrapers/nasa_power.py`

### 5. Climate Data (INMET)

- **Source**: INMET (Instituto Nacional de Meteorologia)
- **API**: INMET Weather API
- **URL**: `https://apitempo.inmet.gov.br/estacao/{station}/{start}/{end}`
- **Stations**:
  - A742: Três Lagoas, MS
  - A201: Imperatriz, MA
  - A701: Suzano, SP
- **Frequency**: Hourly (aggregated to daily)
- **Update Frequency**: Every 24 hours
- **Authentication**: None required
- **Format**: JSON
- **Status**: ⚠️ Automated (may have restrictions)
- **Reliability**: Medium

**Notes**:
- INMET API may have usage restrictions or downtime
- Disabled by default, NASA POWER used as primary climate source
- Can be enabled in `config.json`

**Implementation**: `src/eda/scrapers/inmet_climate.py`

## Manual Sources (Require Upload)

### 6. Pulp Prices (FOEX)

- **Source**: FOEX (Fastmarkets RISI)
- **Type**: Manual upload
- **URL**: https://www.foex.fi/
- **Indices**:
  - BHKP: Bleached Hardwood Kraft Pulp (primary)
  - BSKP: Bleached Softwood Kraft Pulp
- **Frequency**: Weekly (typically)
- **Format**: CSV or Excel
- **Status**: 📤 Manual upload required
- **Reliability**: Depends on data availability

**Expected Schema**:
```csv
date,price,type
2024-01-01,850.5,BHKP
2024-01-08,852.0,BHKP
```

**Upload Command**:
```bash
python -m eda.cli pipeline-upload --file-path data/manual_uploads/pulp_prices.csv --source pulp_prices
```

**Template Generation**:
```bash
python -m eda.cli pipeline-template --source pulp_prices
```

**Notes**:
- FOEX data is subscription-based, no public API
- Prices typically in USD/ton
- Need to manually download and upload periodically
- Alternative free sources: Pulp and Paper Week, Bloomberg (if available)

**Implementation**: `src/eda/pipeline/manual_upload.py`

## Data Quality and Validation

Each data source goes through the following validation:

1. **Completeness**: No empty DataFrames, required columns present
2. **Consistency**: Correct data types, sorted chronologically
3. **Quality**: Outlier detection (>5σ), duplicate removal
4. **Freshness**: Age check (configurable threshold)

Validation is performed by `src/eda/pipeline/validator.py`.

## Update Frequencies

| Source | Frequency | Required | Automation |
|--------|-----------|----------|------------|
| PTAX | Daily | Yes | ✅ |
| SELIC | Daily | Yes | ✅ |
| SUZB3 | Hourly | Yes | ✅ |
| NASA Climate | Weekly | No | ✅ |
| INMET Climate | Daily | No | ⚠️ |
| Pulp Prices | Weekly/Manual | No | 📤 |

## API Limitations and Best Practices

### BCB API
- No explicit rate limits documented
- Recommended: 1 request per second
- Retry with exponential backoff on 502/503 errors

### Yahoo Finance
- Aggressive rate limiting
- Recommended: 1-2 requests per second
- Use caching extensively
- Fallback to CSV if rate limited

### NASA POWER
- No authentication required
- Large historical queries may be slow (30-60s)
- Cache results for at least 1 week

### INMET
- May have undocumented rate limits
- Test carefully before production use
- NASA POWER recommended as alternative

## Fallback Strategy

1. **Primary**: Fetch from API with retry logic
2. **Cache**: Use cached data if API fails (configurable TTL)
3. **Version**: Load latest versioned data if cache miss
4. **CSV Fallback**: Use static CSV if all else fails (yfinance, BCB)
5. **Alert**: Send alert if all methods fail

## Data Storage

### Raw Data
- Location: `data/raw/{source}.csv`
- Format: CSV with date index
- Updated by pipeline

### Versioned Data
- Location: `data/versions/{source}_{timestamp}.parquet`
- Format: Parquet (compressed)
- Metadata: `data/versions/versions.json`
- Retention: Last 10 versions per source (configurable)

### Cache
- Location: `data/cache/{scraper}/{hash}.pkl`
- Format: Pickle
- TTL: Configurable per scraper (1-168 hours)

## Adding New Data Sources

To add a new data source:

1. **Create Scraper** (if API available):
   ```python
   # src/eda/scrapers/my_scraper.py
   from .base import BaseScraper
   
   class MyDataScraper(BaseScraper):
       def _fetch_data(self, start_date, end_date, **kwargs):
           # Implement fetch logic
           pass
   ```

2. **Register Scraper**:
   ```python
   # src/eda/scrapers/__init__.py
   from .my_scraper import MyDataScraper
   registry.register("my_data", MyDataScraper)
   ```

3. **Add to Pipeline**:
   ```python
   # src/eda/pipeline/orchestrator.py
   SOURCES = {
       "my_data": {
           "scraper": "my_data",
           "method": "fetch",
           "update_frequency_hours": 24,
           "required": False,
       },
   }
   ```

4. **Test**:
   ```bash
   python -m eda.cli pipeline-run --sources my_data
   ```

## Monitoring

Monitor data pipeline health:

```bash
python -m eda.cli pipeline-monitor
```

View version history:

```bash
python -m eda.cli pipeline-versions --source ptax
```

Check scheduler status (if running):

```bash
python -m eda.cli pipeline-monitor --export-csv
```

## Troubleshooting

### API Rate Limits
- Increase `rate_limit_seconds` in scraper config
- Reduce `update_frequency_hours`
- Use caching more aggressively

### Stale Data
- Check internet connection
- Verify API endpoints are accessible
- Review error logs
- Check if manual upload needed (pulp prices)

### Validation Errors
- Review specific validation messages
- Adjust outlier threshold if needed
- Check data format matches expected schema

### Missing Dependencies
```bash
pip install -e .
```

## References

- BCB SGS API: https://dadosabertos.bcb.gov.br/
- NASA POWER: https://power.larc.nasa.gov/
- INMET: https://portal.inmet.gov.br/
- FOEX: https://www.foex.fi/
- yfinance: https://github.com/ranaroussi/yfinance

