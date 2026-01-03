@echo off
setlocal enabledelayedexpansion
set PYTHONPATH=%~dp0

REM XGBoost Monthly Evaluation Batch Script
REM This script processes all 12 months of each year in a single batch
REM Each batch cleans up memory to prevent leakage issues

echo ===== Starting XGBoost Monthly Batch Processing =====
echo This script will run XGBoost for multiple years with monthly evaluation
echo Active lag schedule: 4, 8, 12 months
echo Each year will evaluate all 12 months (Jan-Dec)
echo Each batch will clean up memory to prevent leakage issues
echo.

REM Counter for batch tracking
set batch_count=0

REM Process each year from 2013 to 2024 (12 years)
for /L %%y in (2013,1,2024) do (
    echo ==========================================
    echo Processing year: %%y (all 12 months^)
    echo ==========================================

    REM Process all 3 forecasting scopes for this year
    for /L %%f in (1,1,3) do (
        set /a batch_count+=1

        REM Set DESIRED_TERMS environment variable with all 12 months of this year
        set "DESIRED_TERMS=%%y-01,%%y-02,%%y-03,%%y-04,%%y-05,%%y-06,%%y-07,%%y-08,%%y-09,%%y-10,%%y-11,%%y-12"

        echo.
        echo ------ Batch !batch_count!/36: Year %%y, Forecasting Scope %%f ------
        echo Monthly evaluation: All 12 months of %%y

        REM Pre-execution cleanup
        echo Performing pre-execution cleanup...
        call :cleanup_xgboost_directories

        echo Running: python app/main_model_XGB.py --start_year %%y --end_year %%y --forecasting_scope %%f
        echo Environment: DESIRED_TERMS=!DESIRED_TERMS!

        REM Run the Python script
        python app/main_model_XGB.py --start_year %%y --end_year %%y --forecasting_scope %%f

        REM Check if script succeeded
        if !errorlevel! neq 0 (
            echo ERROR: XGBoost script failed for batch !batch_count!
            echo Parameters: start_year=%%y, end_year=%%y, forecasting_scope=%%f
            echo DESIRED_TERMS=!DESIRED_TERMS!
            pause
            goto :end
        )

        echo Batch !batch_count! completed successfully
        echo Results saved to: results_df_xgb_gp_fs%%f_%%y_%%y.csv

        REM Clean up results folders
        echo Cleaning up XGBoost results folders...
        call :cleanup_xgboost_directories

        REM Clean up temporary files
        if exist "temp_*" (
            del /q temp_* 2>nul
            echo Deleted temp files
        )

        REM Clean up pickle files
        if exist "*.pkl" (
            del /q *.pkl 2>nul
            echo Deleted pickle files
        )

        REM Clean up __pycache__
        if exist "__pycache__" (
            rmdir /s /q __pycache__ 2>nul
            echo Deleted __pycache__ folders
        )

        REM Force memory cleanup
        echo Forcing memory cleanup...
        timeout /t 5 /nobreak >nul

        echo Memory cleanup completed for XGBoost batch !batch_count!
        echo.
    )
)

echo.
echo ===== All XGBoost batches completed successfully! =====
echo Total batches processed: !batch_count!
echo Years processed: 2013-2024 (12 years, each with 12 months)
echo Forecasting scopes: 1, 2, 3 (4-month, 8-month, 12-month lags)
echo Total configurations: 12 years × 3 scopes = 36 batches
goto :end

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

REM Additional cleanup for numbered directories
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
