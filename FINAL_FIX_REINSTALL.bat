@echo off
echo ================================================================
echo FINAL FIX: Data Location Correction
echo ================================================================
echo.
echo This will reinstall the package with the correct data paths:
echo   - Changed: data/interim/merged.parquet
echo   - To: data/out/merged.parquet
echo.
pause

set PYTHON_PATH=C:\ProgramData\mambaforge\python.exe

echo [1/3] Uninstalling old version...
%PYTHON_PATH% -m pip uninstall eda-suzano -y
echo.

echo [2/3] Installing fixed version...
%PYTHON_PATH% setup.py install --user
echo.

echo [3/3] Testing the fix...
echo.
echo Running data ingestion...
%PYTHON_PATH% -m eda.cli ingest
echo.

echo Checking if data-quality can find the file...
%PYTHON_PATH% -m eda.cli data-quality
echo.

echo ================================================================
echo FINAL FIX COMPLETE
echo ================================================================
echo.
echo If data-quality ran successfully, the fix is working!
echo You can now run: RUN_COMPLETE_ANALYSIS_FIXED.bat
echo.
pause

