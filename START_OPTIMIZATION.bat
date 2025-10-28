@echo off
echo ================================================================
echo PHASE 1: Oil Data Integration + Momentum Features
echo ================================================================
echo.
echo This will:
echo   1. Reinstall package with oil price support
echo   2. Run data ingestion (includes WTI crude oil)
echo   3. Verify oil data was loaded
echo.
pause

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

echo.
echo [1/3] Reinstalling package...
%PYTHON_PATH% -m pip uninstall eda-suzano -y
%PYTHON_PATH% setup.py install --user
echo.

echo [2/3] Running data ingestion with oil prices...
%PYTHON_PATH% -m eda.cli ingest
echo.

echo [3/3] Checking data...
%PYTHON_PATH% -c "import pandas as pd; df = pd.read_parquet('data/out/merged.parquet'); print(f'Columns: {list(df.columns)}'); print(f'Has oil_usd: {\"oil_usd\" in df.columns}'); print(f'Has oil_brl: {\"oil_brl\" in df.columns}')"
echo.

echo ================================================================
echo Phase 1 Complete!
echo ================================================================
echo.
echo Next: We'll add momentum indicators to features.py
echo.
pause

