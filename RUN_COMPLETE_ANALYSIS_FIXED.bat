@echo off
echo ================================================================
echo QUANTSUZANO COMPLETE ANALYSIS PIPELINE
echo ================================================================
echo.
echo This will run the complete analysis from data ingestion to
echo advanced modeling and benchmarking.
echo.
echo Estimated time: 10-15 minutes
echo.
pause

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

echo.
echo ================================================================
echo STEP 1: DATA INGESTION
echo ================================================================
%PYTHON_PATH% -m eda.cli ingest
if errorlevel 1 (
    echo [ERROR] Data ingestion failed
    pause
    exit /b 1
)
echo [SUCCESS] Data ingested
echo.

echo ================================================================
echo STEP 2: DATA QUALITY ASSESSMENT
echo ================================================================
%PYTHON_PATH% -m eda.cli data-quality
if errorlevel 1 (
    echo [ERROR] Data quality check failed
    pause
    exit /b 1
)
echo [SUCCESS] Data quality assessment complete
echo.

echo ================================================================
echo STEP 3: COMPREHENSIVE EDA (30+ plots)
echo ================================================================
%PYTHON_PATH% -m eda.cli eda-comprehensive
if errorlevel 1 (
    echo [ERROR] EDA failed
    pause
    exit /b 1
)
echo [SUCCESS] Comprehensive EDA complete
echo.

echo ================================================================
echo STEP 4: SYNTHETIC INDEX WITH ROBUST VALIDATION
echo ================================================================
%PYTHON_PATH% -m eda.cli synthetic_robust
if errorlevel 1 (
    echo [ERROR] Synthetic index failed
    pause
    exit /b 1
)
echo [SUCCESS] Synthetic index complete
echo.

echo ================================================================
echo STEP 5: VECM COINTEGRATION ANALYSIS
echo ================================================================
%PYTHON_PATH% -m eda.cli vecm
if errorlevel 1 (
    echo [ERROR] VECM failed
    pause
    exit /b 1
)
echo [SUCCESS] VECM analysis complete
echo.

echo ================================================================
echo STEP 6: RISK ANALYSIS
echo ================================================================
%PYTHON_PATH% -m eda.cli risk-analysis
if errorlevel 1 (
    echo [ERROR] Risk analysis failed
    pause
    exit /b 1
)
echo [SUCCESS] Risk analysis complete
echo.

echo ================================================================
echo STEP 7: ARIMA FORECASTING
echo ================================================================
%PYTHON_PATH% -m eda.cli forecast-arima --target suzb_r --horizon 30
if errorlevel 1 (
    echo [ERROR] Forecasting failed
    pause
    exit /b 1
)
echo [SUCCESS] Forecasting complete
echo.

echo ================================================================
echo STEP 8: MODEL COMPARISON (Ridge, XGBoost, LightGBM)
echo ================================================================
%PYTHON_PATH% -m eda.cli model-comparison
if errorlevel 1 (
    echo [ERROR] Model comparison failed
    pause
    exit /b 1
)
echo [SUCCESS] Model comparison complete
echo.

echo ================================================================
echo STEP 9: STRATEGY OPTIMIZATION
echo ================================================================
%PYTHON_PATH% -m eda.cli strategy-optimize
if errorlevel 1 (
    echo [ERROR] Strategy optimization failed
    pause
    exit /b 1
)
echo [SUCCESS] Strategy optimization complete
echo.

echo ================================================================
echo STEP 10: BENCHMARK COMPARISON
echo ================================================================
%PYTHON_PATH% -m eda.cli benchmark-compare
if errorlevel 1 (
    echo [ERROR] Benchmark comparison failed
    pause
    exit /b 1
)
echo [SUCCESS] Benchmark comparison complete
echo.

echo ================================================================
echo STEP 11 (OPTIONAL): AUTOML WITH TPOT
echo ================================================================
echo This step can take 30-60 minutes. Skip? (Press Ctrl+C to skip)
pause
%PYTHON_PATH% -m eda.cli automl-tpot --generations 10 --population 50
if errorlevel 1 (
    echo [WARN] AutoML failed or was skipped
) else (
    echo [SUCCESS] AutoML complete
)
echo.

echo ================================================================
echo ANALYSIS COMPLETE!
echo ================================================================
echo.
echo All results saved to: data\out\
echo All plots saved to: data\out\plots\
echo.
echo Key outputs:
echo   - data\out\merged.parquet (processed data)
echo   - data\out\data_quality_report.csv
echo   - data\out\model_comparison.csv
echo   - data\out\strategy_optimization.csv
echo   - data\out\benchmark_comparison.csv
echo   - data\out\plots\eda\ (30+ exploratory plots)
echo   - data\out\plots\models\ (model performance)
echo   - data\out\plots\strategies\ (strategy comparison)
echo.
pause

