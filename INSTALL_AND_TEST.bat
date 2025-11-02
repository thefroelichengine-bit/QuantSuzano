@echo off
echo ================================================================
echo INSTALLING EDA PACKAGE AND TESTING ENSEMBLE STRATEGY
echo ================================================================

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

echo.
echo [1/3] Installing package in development mode...
%PYTHON_PATH% -m pip install -e "E:\AI stuff\QuantSuzano"
echo.

echo [2/3] Verifying installation...
%PYTHON_PATH% -c "import eda; print('[OK] EDA module imported'); from eda.strategies import EnsembleStrategy; print('[OK] EnsembleStrategy imported')"
echo.

echo [3/3] Testing CLI command...
%PYTHON_PATH% -m eda.cli --help
echo.

echo ================================================================
echo INSTALLATION COMPLETE
echo ================================================================
echo.
echo You can now run:
echo   python -m eda.cli strategy-ensemble
echo   python -m eda.cli strategy-compare
echo.
pause

