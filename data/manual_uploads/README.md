# Manual Data Uploads

This directory is for data that cannot be automatically scraped or fetched from APIs.

## Primary Use Case: Pulp Prices (FOEX)

Pulp prices from FOEX (Fastmarkets RISI) require a paid subscription and have no public API.

### How to Upload Pulp Prices

1. **Generate Template** (first time only):
   ```bash
   python -m eda.cli pipeline-template --source pulp_prices
   ```

2. **Fill in the Template**:
   Open `template_pulp_prices.csv` and add your data:
   ```csv
   date,price,type
   2024-01-01,850.5,BHKP
   2024-01-08,852.0,BHKP
   2024-01-15,855.0,BHKP
   ```

   Fields:
   - `date`: Date in YYYY-MM-DD format
   - `price`: Price in USD/ton
   - `type`: Pulp type (BHKP, BSKP, etc.)

3. **Upload**:
   ```bash
   python -m eda.cli pipeline-upload --file-path data/manual_uploads/pulp_prices.csv
   ```

### Pulp Types

- **BHKP**: Bleached Hardwood Kraft Pulp (primary - used by default)
- **BSKP**: Bleached Softwood Kraft Pulp
- **NBSK**: Northern Bleached Softwood Kraft
- **BEK**: Bleached Eucalyptus Kraft

### Update Frequency

- Typical: Weekly
- Minimum recommended: Bi-weekly
- Data becomes stale after: 30 days

### Data Sources

Where to get pulp prices:

1. **FOEX** (Primary): https://www.foex.fi/
   - Subscription required
   - Most comprehensive and accurate
   - Industry standard

2. **Pulp and Paper Week**: https://www.ppw-online.com/
   - Alternative source
   - May have free trial

3. **Bloomberg** (if available):
   - Terminal subscribers can access pulp indices
   - Symbol: BHKPINDX

4. **Industry Reports**:
   - Suzano Investor Relations
   - Quarterly reports often include market price references

### Manual Upload Process

```bash
# Step 1: Download data from FOEX or other source
# Save as CSV with columns: date, price, type

# Step 2: Upload to pipeline
python -m eda.cli pipeline-upload \
    --file-path /path/to/your/pulp_prices.csv \
    --source pulp_prices

# Step 3: Verify upload
python -m eda.cli pipeline-versions --source pulp_prices

# Step 4: Re-run analysis
python -m eda.cli ingest
python -m eda.cli all-robust
```

### Example Data

See `template_pulp_prices.csv` for format.

Historical example:
```csv
date,price,type
2023-01-01,820.0,BHKP
2023-02-01,825.0,BHKP
2023-03-01,830.0,BHKP
2023-04-01,835.0,BHKP
2023-05-01,840.0,BHKP
2023-06-01,845.0,BHKP
```

### Troubleshooting

**Issue**: "Date column not found"
- Ensure first column is named `date` (lowercase)
- Check date format is YYYY-MM-DD

**Issue**: "Missing columns"
- Ensure you have `date`, `price`, and optionally `type`
- Check for extra commas or formatting issues

**Issue**: "Duplicate dates"
- Remove duplicate rows
- Upload will keep the most recent value

**Issue**: "Validation warnings"
- Check for missing values (NaN)
- Verify prices are reasonable (typically 700-1200 USD/ton for BHKP)
- Ensure dates are sequential

### Other Data Sources

This system can also handle other manual uploads:

```bash
# Generic data upload
python -m eda.cli pipeline-upload \
    --file-path your_data.csv \
    --source custom_source_name \
    --date-column date_column_name
```

All uploaded data will be:
- Validated
- Versioned
- Merged with other sources
- Available for analysis

### Automation Potential

If you have programmatic access to pulp price data:

1. Create a scraper in `src/eda/scrapers/foex_scraper.py`
2. Implement the fetch logic
3. Register in the pipeline
4. Enable in `config.json`

See `src/eda/scrapers/base.py` for the scraper interface.

