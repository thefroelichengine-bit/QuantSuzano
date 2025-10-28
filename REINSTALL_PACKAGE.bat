@echo off
echo ================================================================
echo REINSTALLING EDA PACKAGE WITH FIXES
echo ================================================================

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

echo [1/2] Uninstalling old version...
%PYTHON_PATH% -m pip uninstall eda-suzano -y
echo.

echo [2/2] Installing new version with path fix...
%PYTHON_PATH% setup.py install --user
echo.

echo [3/3] Verifying installation...
%PYTHON_PATH% -c "from eda.config import ROOT; print(f'ROOT path detected: {ROOT}')"
echo.

echo ================================================================
echo REINSTALLATION COMPLETE
echo ================================================================
echo.
echo The package will now use the workspace directory for data files.
echo You can now run: RUN_COMPLETE_ANALYSIS_FIXED.bat
echo.
pause

