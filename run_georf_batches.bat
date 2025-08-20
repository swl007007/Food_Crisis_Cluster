@echo off
setlocal enabledelayedexpansion

REM GeoRF Memory Management Batch Script
REM This script iterates through time periods and forecasting scopes to avoid memory leakage
REM Each batch cleans up results before proceeding to the next run

echo ===== Starting GeoRF Batch Processing =====
echo This script will run GeoRF for multiple time periods and forecasting scopes
echo Each batch will clean up memory to prevent leakage issues
echo.

REM Counter for batch tracking
set batch_count=0

REM Time period 1: 2015-2016
echo ==========================================
echo Processing time period: 2015-2016
echo ==========================================
for /L %%f in (1,1,4) do (
    call :run_batch 2015 2016 %%f
)

REM Time period 2: 2017-2018
echo ==========================================
echo Processing time period: 2017-2018
echo ==========================================
for /L %%f in (1,1,4) do (
    call :run_batch 2017 2018 %%f
)

REM Time period 3: 2019-2020
echo ==========================================
echo Processing time period: 2019-2020
echo ==========================================
for /L %%f in (1,1,4) do (
    call :run_batch 2019 2020 %%f
)

REM Time period 4: 2021-2022
echo ==========================================
echo Processing time period: 2021-2022
echo ==========================================
for /L %%f in (1,1,4) do (
    call :run_batch 2021 2022 %%f
)

REM Time period 5: 2023-2024
echo ==========================================
echo Processing time period: 2023-2024
echo ==========================================
for /L %%f in (1,1,4) do (
    call :run_batch 2023 2024 %%f
)

echo.
echo ===== All batches completed successfully! =====
echo Total batches processed: !batch_count!
echo Time periods: 2015-2016, 2017-2018, 2019-2020, 2021-2022, 2023-2024
echo Forecasting scopes: 1, 2, 3, 4 (for each time period)
goto :end

REM Function to run a single batch
:run_batch
set /a batch_count+=1
set start_year=%1
set end_year=%2
set forecasting_scope=%3

echo.
echo ------ Batch !batch_count!: Years %start_year%-%end_year%, Forecasting Scope %forecasting_scope% ------
echo Running: python main_model_GF.py --start_year %start_year% --end_year %end_year% --forecasting_scope %forecasting_scope%

REM Run the Python script with current parameters
python main_model_GF.py --start_year %start_year% --end_year %end_year% --forecasting_scope %forecasting_scope%

REM Check if the Python script succeeded
if !errorlevel! neq 0 (
    echo ERROR: Python script failed for batch !batch_count!
    echo Parameters: start_year=%start_year%, end_year=%end_year%, forecasting_scope=%forecasting_scope%
    pause
    goto :end
)

echo Batch !batch_count! completed successfully

REM Clean up results folders and force garbage collection
echo Cleaning up results folders...
if exist "results_GeoRF*" (
    rmdir /s /q results_GeoRF* 2>nul
    echo Deleted results_GeoRF* folders
)

REM Clean up any temporary files
if exist "temp_*" (
    del /q temp_* 2>nul
    echo Deleted temp files
)

REM Clean up any pickle files that might be holding memory
if exist "*.pkl" (
    del /q *.pkl 2>nul
    echo Deleted pickle files
)

REM Clean up any matplotlib cache
if exist "__pycache__" (
    rmdir /s /q __pycache__ 2>nul
    echo Deleted __pycache__ folders
)

REM Force Windows to release memory and wait
echo Forcing memory cleanup...
timeout /t 3 /nobreak >nul

echo Memory cleanup completed for batch !batch_count!
echo.
goto :eof

:end
echo Script finished.
pause