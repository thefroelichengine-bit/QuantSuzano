@echo off
echo ================================================================
echo RUNNING ENSEMBLE STRATEGY
echo ================================================================
echo.

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

cd /d "E:\AI stuff\QuantSuzano"

echo [1/3] Running ensemble strategy...
%PYTHON_PATH% -m eda.cli strategy-ensemble

if errorlevel 1 (
    echo.
    echo [ERROR] Strategy failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo SUCCESS! Results saved to data\out\
echo ================================================================
echo.
echo Output files:
echo   - ensemble_signals.parquet
echo   - ensemble_backtest.parquet
echo   - ensemble_metrics.csv
echo   - ensemble_comparison.csv
echo   - plots\strategies\
echo.
pause

