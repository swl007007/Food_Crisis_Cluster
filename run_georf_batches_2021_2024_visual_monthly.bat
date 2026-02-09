@echo off
setlocal enabledelayedexpansion
set PYTHONPATH=%~dp0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM GeoRF Monthly-by-Month Evaluation Batch Script (2021-2024 with Visual Debug)
REM This script processes EACH MONTH INDIVIDUALLY to prevent memory issues
REM Visual debug mode requires processing one month at a time due to high memory usage

echo ===== Starting GeoRF Monthly-by-Month Batch Processing (2021-2024, Visual Debug Mode) =====
echo This script will run GeoRF processing ONE MONTH AT A TIME
echo Active lag schedule: 4, 8, 12 months
echo Each month evaluated individually to manage memory
echo Visual debug mode: ENABLED (partition maps, metrics tracking, improvement visualization)
echo Results combined at the end of each year+scope
echo.

REM ============================================================================
REM VISUAL DEBUG CONFIGURATION
REM ============================================================================
echo Please ensure the following are enabled:
echo   1. config.py: VIS_DEBUG_MODE = True
echo   2. app/main_model_GF.py: track_partition_metrics = True
echo   3. app/main_model_GF.py: enable_metrics_maps = True
echo.
echo Press CTRL+C to cancel if not configured, or any key to continue...
pause

REM Counter for batch tracking
set batch_count=0
set total_batches=144
set monthly_result_count=0

echo.
echo Configuration summary:
echo   - Years: 2021-2024 (4 years)
echo   - Forecasting scopes: 3 (4-month, 8-month, 12-month lags)
echo   - Months per year: 12 (Jan-Dec)
echo   - Total batches: !total_batches! (4 years x 3 scopes x 12 months)
echo   - Visual outputs: Partition maps, metrics CSV, improvement maps
echo   - Memory management: One month at a time with cleanup
echo.

REM Process each year from 2021 to 2024
for /L %%y in (2021,1,2024) do (
    set "current_year=%%y"
    echo ==========================================
    echo Processing year: !current_year!
    echo ==========================================

    REM Process all 3 forecasting scopes for this year
    for /L %%f in (1,1,3) do (
        set "current_scope=%%f"

        echo.
        echo ====== Year !current_year!, Forecasting Scope !current_scope! ======
        echo Processing 12 months individually...

        REM Initialize monthly results tracking
        set "monthly_result_count=0"

        REM Process each month individually
        for %%m in (01 02 03 04 05 06 07 08 09 10 11 12) do (
            call :process_single_month !current_year! !current_scope! %%m

            REM Check if processing failed
            if !errorlevel! neq 0 (
                echo ERROR: Failed processing month %%m
                pause
                goto :end
            )
        )

        REM After all 12 months, combine results into yearly file
        echo.
        echo Combining monthly results for year !current_year!, scope !current_scope!...
        call :combine_monthly_results !current_year! !current_scope!

        echo Year !current_year!, Scope !current_scope! completed successfully
        echo.
    )
)

echo.
echo ===== All batches completed successfully! =====
echo Total monthly batches processed: !batch_count!
echo Years processed: 2021-2024 (4 years, each with 12 months)
echo Forecasting scopes: 1, 2, 3 (4-month, 8-month, 12-month lags)
echo Processing mode: One month at a time (144 total)
echo.
echo Visual outputs archived in folders: result_GeoRF_YEAR_fsN_YYYY-MM_visual
echo Combined results: results_df_gp_fsN_YEAR_YEAR.csv (12 months combined)
echo.
echo All temporary result_GeoRF_* directories have been cleaned up.
echo Archived visual folders are preserved for analysis.
goto :end

:process_single_month
REM Subroutine to process a single month with proper memory management
REM Args: %1 = year, %2 = forecasting scope, %3 = month (01-12)
set "proc_year=%1"
set "proc_scope=%2"
set "proc_month=%3"
set /a batch_count+=1

REM Set DESIRED_TERMS to only this single month
set "DESIRED_TERMS=%proc_year%-%proc_month%"

echo.
echo ------ Batch !batch_count!/!total_batches!: Year %proc_year%, Scope %proc_scope%, Month %proc_month% ------
echo Monthly evaluation: %proc_year%-%proc_month% ONLY
echo Visual debug: Enabled (creating partition maps and metrics)

REM Pre-execution cleanup
echo Performing pre-execution cleanup...
call :cleanup_directories

echo Running: python app/main_model_GF.py --start_year %proc_year% --end_year %proc_year% --forecasting_scope %proc_scope%
echo Environment: DESIRED_TERMS=%DESIRED_TERMS%
echo Visual outputs will be saved to: result_GeoRF_*/vis/

REM Run the Python script for this single month
python app/main_model_GF.py --start_year %proc_year% --end_year %proc_year% --forecasting_scope %proc_scope%

REM Check if script succeeded
if !errorlevel! neq 0 (
    echo ERROR: Python script failed for batch !batch_count!
    echo Parameters: start_year=%proc_year%, end_year=%proc_year%, forecasting_scope=%proc_scope%, month=%proc_month%
    echo DESIRED_TERMS=%DESIRED_TERMS%
    exit /b 1
)

echo Batch !batch_count! completed successfully

REM Extract this month's results and save separately
echo Extracting monthly results for %proc_year%-%proc_month%...
call :extract_monthly_results %proc_year% %proc_scope% %proc_month%

REM Archive important visual debug files
echo Archiving visual debug files for %proc_year%-%proc_month%...
call :archive_visual_files %proc_year% %proc_scope% %proc_month%

REM Clean up results folders
echo Cleaning up results folders...
call :cleanup_directories

REM Clean up temporary files
if exist "temp_*" (
    del /q temp_* 2>nul
)
if exist "*.pkl" (
    del /q *.pkl 2>nul
)
if exist "__pycache__" (
    rmdir /s /q __pycache__ 2>nul
)

REM Force memory cleanup
echo Forcing memory cleanup...
timeout /t 3 /nobreak >nul

echo Month %proc_month% processing completed
echo.
exit /b 0

:extract_monthly_results
REM Extract results from the single-month run and save to monthly file
REM Args: %1 = year, %2 = scope, %3 = month
set "ext_year=%1"
set "ext_scope=%2"
set "ext_month=%3"

REM Create monthly results directory if it doesn't exist
if not exist "monthly_results" mkdir "monthly_results"

REM Monthly result files
set "monthly_result_file=monthly_results\results_df_gp_fs%ext_scope%_%ext_year%_%ext_month%.csv"
set "monthly_pred_file=monthly_results\y_pred_test_gp_fs%ext_scope%_%ext_year%_%ext_month%.csv"

REM Find the most recent results file (should be from current run)
set "source_result_file="
for %%F in (results_df_gp_fs%ext_scope%_*.csv) do (
    set "source_result_file=%%F"
)

REM Find the most recent prediction file
set "source_pred_file="
for %%F in (y_pred_test_gp_fs%ext_scope%_*.csv) do (
    set "source_pred_file=%%F"
)

REM Copy results to monthly files
if defined source_result_file (
    if exist "!source_result_file!" (
        copy "!source_result_file!" "%monthly_result_file%" >nul 2>&1
        echo   - Saved: %monthly_result_file%
    )
)

if defined source_pred_file (
    if exist "!source_pred_file!" (
        copy "!source_pred_file!" "%monthly_pred_file%" >nul 2>&1
        echo   - Saved: %monthly_pred_file%
    )
)

REM Clean up the original files after copying
if defined source_result_file (
    if exist "!source_result_file!" del "!source_result_file!" 2>nul
)
if defined source_pred_file (
    if exist "!source_pred_file!" del "!source_pred_file!" 2>nul
)

goto :eof

:combine_monthly_results
REM Combine all monthly results into a single yearly file
REM Args: %1 = year, %2 = scope
set "comb_year=%1"
set "comb_scope=%2"

echo Combining 12 monthly result files into yearly summary...

REM Output files
set "combined_result_file=results_df_gp_fs%comb_scope%_%comb_year%_%comb_year%.csv"
set "combined_pred_file=y_pred_test_gp_fs%comb_scope%_%comb_year%_%comb_year%.csv"

REM Use Python to combine the monthly CSV files
python -c "import pandas as pd; import glob; pattern='monthly_results/results_df_gp_fs%comb_scope%_%comb_year%_*.csv'; files=sorted(glob.glob(pattern)); df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True); df.to_csv('%combined_result_file%', index=False); print(f'Combined {len(files)} monthly result files')"

python -c "import pandas as pd; import glob; pattern='monthly_results/y_pred_test_gp_fs%comb_scope%_%comb_year%_*.csv'; files=sorted(glob.glob(pattern)); df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True); df.to_csv('%combined_pred_file%', index=False); print(f'Combined {len(files)} monthly prediction files')"

echo   - Created: %combined_result_file%
echo   - Created: %combined_pred_file%

REM Clean up monthly files after combining
echo Cleaning up monthly intermediate files...
del /q "monthly_results\results_df_gp_fs%comb_scope%_%comb_year%_*.csv" 2>nul
del /q "monthly_results\y_pred_test_gp_fs%comb_scope%_%comb_year%_*.csv" 2>nul

goto :eof

:archive_visual_files
REM Subroutine to archive important visual debug files
REM Args: %1 = year, %2 = scope, %3 = month
set "archive_year=%1"
set "archive_scope=%2"
set "archive_month=%3"

REM Find the result_GeoRF directory - check common patterns in priority order
REM Priority: result_GeoRF (no number), result_GeoRF_0, result_GeoRF_1, result_GeoRF_2, etc.
set "source_dir="

REM First check result_GeoRF (no number suffix)
if exist "result_GeoRF" (
    if exist "result_GeoRF\vis" (
        set "source_dir=result_GeoRF"
        goto :found_source_dir
    )
)

REM Then check numbered versions: result_GeoRF_0 through result_GeoRF_20
for /L %%N in (0,1,20) do (
    if exist "result_GeoRF_%%N" (
        if exist "result_GeoRF_%%N\vis" (
            set "source_dir=result_GeoRF_%%N"
            goto :found_source_dir
        )
    )
)

REM If still not found, search for any result_GeoRF_* that's NOT archived (doesn't end with _visual)
for /d %%D in (result_GeoRF_*) do (
    set "temp_dir=%%D"
    REM Use string manipulation to check last 7 characters
    set "last_7=!temp_dir:~-7!"
    if not "!last_7!"=="_visual" (
        if exist "%%D\vis" (
            set "source_dir=%%D"
            goto :found_source_dir
        )
    )
)

echo WARNING: No valid result_GeoRF* directory with /vis folder found to archive
goto :eof

:found_source_dir
echo Found source directory: !source_dir!

REM Check if vis directory exists
if not exist "!source_dir!\vis" (
    echo WARNING: vis directory not found in !source_dir!
    goto :eof
)

REM Create descriptive archive folder name with month
set "archive_folder=result_GeoRF_%archive_year%_fs%archive_scope%_%archive_year%-%archive_month%_visual"
echo Creating archive folder: !archive_folder!

REM Create archive directory structure
if exist "!archive_folder!" (
    echo Archive folder already exists, removing old version...
    rmdir /s /q "!archive_folder!" 2>nul
)
mkdir "!archive_folder!"
mkdir "!archive_folder!\vis"

echo Copying files from !source_dir! to !archive_folder!...
set copied_count=0

REM Copy important files from /vis directory with verification
if exist "!source_dir!\vis\comprehensive_partition_metrics.csv" (
    copy "!source_dir!\vis\comprehensive_partition_metrics.csv" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\comprehensive_partition_metrics.csv" (
        echo   [OK] comprehensive_partition_metrics.csv
        set /a copied_count+=1
    ) else (
        echo   [FAILED] comprehensive_partition_metrics.csv
    )
) else (
    echo   [SKIP] comprehensive_partition_metrics.csv not found
)

if exist "!source_dir!\vis\final_f1_performance_map.png" (
    copy "!source_dir!\vis\final_f1_performance_map.png" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\final_f1_performance_map.png" (
        echo   [OK] final_f1_performance_map.png
        set /a copied_count+=1
    ) else (
        echo   [FAILED] final_f1_performance_map.png
    )
) else (
    echo   [SKIP] final_f1_performance_map.png not found
)

if exist "!source_dir!\vis\final_partition_map.png" (
    copy "!source_dir!\vis\final_partition_map.png" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\final_partition_map.png" (
        echo   [OK] final_partition_map.png
        set /a copied_count+=1
    ) else (
        echo   [FAILED] final_partition_map.png
    )
) else (
    echo   [SKIP] final_partition_map.png not found
)

if exist "!source_dir!\vis\overall_f1_improvement_map.png" (
    copy "!source_dir!\vis\overall_f1_improvement_map.png" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\overall_f1_improvement_map.png" (
        echo   [OK] overall_f1_improvement_map.png
        set /a copied_count+=1
    ) else (
        echo   [FAILED] overall_f1_improvement_map.png
    )
) else (
    echo   [SKIP] overall_f1_improvement_map.png not found
)

if exist "!source_dir!\vis\map_pct_err_all.png" (
    copy "!source_dir!\vis\map_pct_err_all.png" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\map_pct_err_all.png" (
        echo   [OK] map_pct_err_all.png
        set /a copied_count+=1
    ) else (
        echo   [FAILED] map_pct_err_all.png
    )
) else (
    echo   [SKIP] map_pct_err_all.png not found
)

if exist "!source_dir!\vis\map_pct_err_class1.png" (
    copy "!source_dir!\vis\map_pct_err_class1.png" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\map_pct_err_class1.png" (
        echo   [OK] map_pct_err_class1.png
        set /a copied_count+=1
    ) else (
        echo   [FAILED] map_pct_err_class1.png
    )
) else (
    echo   [SKIP] map_pct_err_class1.png not found
)

if exist "!source_dir!\vis\train_error_by_polygon.csv" (
    copy "!source_dir!\vis\train_error_by_polygon.csv" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\train_error_by_polygon.csv" (
        echo   [OK] train_error_by_polygon.csv
        set /a copied_count+=1
    ) else (
        echo   [FAILED] train_error_by_polygon.csv
    )
) else (
    echo   [SKIP] train_error_by_polygon.csv not found
)

REM Copy all score_details_*.csv files
set score_count=0
for %%F in ("!source_dir!\vis\score_details_*.csv") do (
    copy "%%F" "!archive_folder!\vis\" >nul 2>&1
    if exist "!archive_folder!\vis\%%~nxF" (
        echo   [OK] %%~nxF
        set /a copied_count+=1
        set /a score_count+=1
    ) else (
        echo   [FAILED] %%~nxF
    )
)
if !score_count! equ 0 (
    echo   [SKIP] No score_details_*.csv files found
)

REM Copy important files from parent directory
if exist "!source_dir!\correspondence_table_*.csv" (
    for %%F in ("!source_dir!\correspondence_table_*.csv") do (
        copy "%%F" "!archive_folder!\" >nul 2>&1
        if exist "!archive_folder!\%%~nxF" (
            echo   [OK] %%~nxF
            set /a copied_count+=1
        ) else (
            echo   [FAILED] %%~nxF
        )
    )
) else (
    echo   [SKIP] No correspondence_table_*.csv found
)

if exist "!source_dir!\log_print.txt" (
    copy "!source_dir!\log_print.txt" "!archive_folder!\" >nul 2>&1
    if exist "!archive_folder!\log_print.txt" (
        echo   [OK] log_print.txt
        set /a copied_count+=1
    ) else (
        echo   [FAILED] log_print.txt
    )
) else (
    echo   [SKIP] log_print.txt not found
)

echo Archive completed: !archive_folder!
echo Total files copied: !copied_count!

REM Verify archive folder still exists before cleanup
if exist "!archive_folder!" (
    echo Archive folder verified: !archive_folder!
) else (
    echo ERROR: Archive folder disappeared: !archive_folder!
)

goto :eof

:cleanup_directories
echo Performing robust directory cleanup...

REM Phase 1: Wildcard pattern cleanup (skip archived folders ending with _visual)
for /d %%D in (result_GeoRF*) do (
    call :check_and_delete_folder "%%D"
)

REM Phase 2: Explicit cleanup for base folder and numbered versions (0-20)
REM This provides redundancy if Phase 1 wildcard matching fails
REM Each deletion checks for _visual suffix to protect archived folders

if exist "result_GeoRF" (
    call :check_and_delete_folder "result_GeoRF"
)

for /L %%i in (0,1,20) do (
    if exist "result_GeoRF_%%i" (
        call :check_and_delete_folder "result_GeoRF_%%i"
    )
)

REM Clean up temp files and caches
del /f /q *.tmp 2>nul
for /d /r %%D in (__pycache__) do @if exist "%%D" rd /s /q "%%D" 2>nul

echo Cleanup completed
goto :eof

:check_and_delete_folder
REM Subroutine to check if a folder should be deleted or skipped
REM Args: %1 = folder name
set "folder_to_check=%~1"
REM Check if folder ends with _visual (archived folder)
REM Use string manipulation to check last 7 characters
set "last_7=%folder_to_check:~-7%"
if "%last_7%"=="_visual" (
    REM Folder ends with _visual - this IS archived, SKIP it
    echo   Skipping archived folder: %folder_to_check%
) else (
    REM Folder does NOT end with _visual - this is NOT archived, DELETE it
    echo   Attempting to delete: %folder_to_check%
    rmdir /s /q "%folder_to_check%" 2>nul
    if exist "%folder_to_check%" (
        timeout /t 2 /nobreak >nul
        del /s /f /q "%folder_to_check%\*" 2>nul
        rmdir /s /q "%folder_to_check%" 2>nul
    )
)
goto :eof

:end
echo Script finished.
pause
