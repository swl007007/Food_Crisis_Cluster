@echo off
setlocal enabledelayedexpansion
set PYTHONPATH=%~dp0

REM XGBoost Memory Management Batch Script
REM This script iterates through time periods and forecasting scopes to avoid memory leakage
REM Each batch cleans up results before proceeding to the next run

echo ===== Starting XGBoost Batch Processing =====
echo This script will run XGBoost for multiple time periods and forecasting scopes
echo Each batch will clean up memory to prevent leakage issues
echo.

REM Counter for batch tracking
set batch_count=0

REM Individual year processing for memory management
REM Each batch processes 1 year with 1 forecasting scope to minimize memory usage

REM Year 2015 - All forecasting scopes
echo ==========================================
echo Processing year: 2015
echo ==========================================
call :run_batch 2015 2015 1
call :run_batch 2015 2015 2
call :run_batch 2015 2015 3
call :run_batch 2015 2015 4

REM Year 2016 - All forecasting scopes
echo ==========================================
echo Processing year: 2016
echo ==========================================
call :run_batch 2016 2016 1
call :run_batch 2016 2016 2
call :run_batch 2016 2016 3
call :run_batch 2016 2016 4

REM Year 2017 - All forecasting scopes
echo ==========================================
echo Processing year: 2017
echo ==========================================
call :run_batch 2017 2017 1
call :run_batch 2017 2017 2
call :run_batch 2017 2017 3
call :run_batch 2017 2017 4

REM Year 2018 - All forecasting scopes
echo ==========================================
echo Processing year: 2018
echo ==========================================
call :run_batch 2018 2018 1
call :run_batch 2018 2018 2
call :run_batch 2018 2018 3
call :run_batch 2018 2018 4

REM Year 2019 - All forecasting scopes
echo ==========================================
echo Processing year: 2019
echo ==========================================
call :run_batch 2019 2019 1
call :run_batch 2019 2019 2
call :run_batch 2019 2019 3
call :run_batch 2019 2019 4

REM Year 2020 - All forecasting scopes
echo ==========================================
echo Processing year: 2020
echo ==========================================
call :run_batch 2020 2020 1
call :run_batch 2020 2020 2
call :run_batch 2020 2020 3
call :run_batch 2020 2020 4

REM Year 2021 - All forecasting scopes
echo ==========================================
echo Processing year: 2021
echo ==========================================
call :run_batch 2021 2021 1
call :run_batch 2021 2021 2
call :run_batch 2021 2021 3
call :run_batch 2021 2021 4

REM Year 2022 - All forecasting scopes
echo ==========================================
echo Processing year: 2022
echo ==========================================
call :run_batch 2022 2022 1
call :run_batch 2022 2022 2
call :run_batch 2022 2022 3
call :run_batch 2022 2022 4

REM Year 2023 - All forecasting scopes
echo ==========================================
echo Processing year: 2023
echo ==========================================
call :run_batch 2023 2023 1
call :run_batch 2023 2023 2
call :run_batch 2023 2023 3
call :run_batch 2023 2023 4

REM Year 2024 - All forecasting scopes
echo ==========================================
echo Processing year: 2024
echo ==========================================
call :run_batch 2024 2024 1
call :run_batch 2024 2024 2
call :run_batch 2024 2024 3
call :run_batch 2024 2024 4

echo.
echo ===== All XGBoost batches completed successfully! =====
echo Total batches processed: !batch_count!
echo Years processed: 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
echo Forecasting scopes: 1, 2, 3, 4 (for each year)
echo Memory-optimized: 1 year + 1 scope per batch (40 total batches)
goto :end

REM Function to run a single batch
:run_batch
set /a batch_count+=1
set start_year=%1
set end_year=%2
set forecasting_scope=%3

echo.
echo ------ XGBoost Batch !batch_count!: Years %start_year%-%end_year%, Forecasting Scope %forecasting_scope% ------

REM Pre-execution cleanup to ensure clean state
echo Performing pre-execution cleanup...
call :cleanup_xgboost_directories

echo Running: python app/main_model_XGB.py --start_year %start_year% --end_year %end_year% --forecasting_scope %forecasting_scope% --force_cleanup

REM Run the Python script with current parameters and force cleanup
python app/main_model_XGB.py --start_year %start_year% --end_year %end_year% --forecasting_scope %forecasting_scope% --force_cleanup

REM Check if the Python script succeeded
if !errorlevel! neq 0 (
    echo ERROR: XGBoost script failed for batch !batch_count!
    echo Parameters: start_year=%start_year%, end_year=%end_year%, forecasting_scope=%forecasting_scope%
    pause
    goto :end
)

echo XGBoost Batch !batch_count! completed successfully

REM Clean up XGBoost results folders and force garbage collection
echo Cleaning up XGBoost results folders...

REM Use robust cleanup with explicit enumeration and retry logic
call :cleanup_xgboost_directories

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
timeout /t 5 /nobreak >nul

echo Memory cleanup completed for XGBoost batch !batch_count!
echo.
goto :eof

:cleanup_xgboost_directories
echo Performing robust XGBoost directory cleanup...

REM Try to enumerate and delete each result_GeoXGB directory explicitly
for /d %%D in (result_GeoXGB*) do (
    echo Attempting to delete XGBoost directory: %%D
    
    REM First attempt - immediate deletion
    rmdir /s /q "%%D" 2>nul
    
    REM Verify deletion
    if exist "%%D" (
        echo Directory %%D still exists, trying alternative cleanup...
        
        REM Force close any open handles and retry
        timeout /t 2 /nobreak >nul
        
        REM Second attempt with explicit file deletion first
        del /s /f /q "%%D\*" 2>nul
        rmdir /s /q "%%D" 2>nul
        
        REM Final verification
        if exist "%%D" (
            echo WARNING: Failed to delete XGBoost directory %%D - may be locked by another process
        ) else (
            echo Successfully deleted XGBoost directory: %%D
        )
    ) else (
        echo Successfully deleted XGBoost directory: %%D
    )
)

REM Additional cleanup for numbered directories that might not match the pattern
for /L %%i in (0,1,50) do (
    if exist "result_GeoXGB_%%i" (
        echo Found numbered XGBoost directory result_GeoXGB_%%i, attempting deletion...
        rmdir /s /q "result_GeoXGB_%%i" 2>nul
        if not exist "result_GeoXGB_%%i" (
            echo Successfully deleted: result_GeoXGB_%%i
        )
    )
)

echo XGBoost directory cleanup attempt completed
goto :eof

:end
echo XGBoost batch script finished.
pause