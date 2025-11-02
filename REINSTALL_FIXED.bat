@echo off
echo ================================================================
echo REINSTALLING EDA PACKAGE WITH FIXES
echo ================================================================

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

echo.
echo [1/3] Uninstalling old version...
%PYTHON_PATH% -m pip uninstall eda-suzano -y
echo.

echo [2/3] Installing new version with all fixes...
%PYTHON_PATH% setup.py install --user
echo.

echo [3/3] Verifying installation...
%PYTHON_PATH% -c "from eda.metrics import calc_information_coefficient; print('[OK] calc_information_coefficient imported'); from eda.strategies import EnsembleStrategy; print('[OK] EnsembleStrategy imported')"
echo.

echo ================================================================
echo REINSTALLATION COMPLETE
echo ================================================================
echo.
echo You can now run:
echo   python -m eda.cli strategy-ensemble
echo   or
echo   RUN_ENSEMBLE_STRATEGY.bat
echo.
pause

