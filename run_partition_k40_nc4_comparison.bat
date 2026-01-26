@echo off
REM ============================================================================
REM Partitioned (k40_nc4) vs Pooled RF Comparison - Windows Batch Script
REM ============================================================================
REM
REM This batch file runs the comparison script with default parameters.
REM
REM Usage:
REM   run_partition_k40_nc4_comparison.bat                 (no visuals)
REM   run_partition_k40_nc4_comparison.bat --visual        (with 4 maps)
REM
REM Customize by editing the variables below before running.
REM ============================================================================

echo.
echo ============================================================================
echo PARTITIONED (k40_nc4) VS POOLED RF COMPARISON
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Configuration (Edit as needed)
REM --------------------------------------------------------------------------

REM Python executable (adjust path if needed)
set PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe

REM Data paths
set DATA_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv
set PARTITION_MAP=cluster_mapping_k40_nc4.csv
set POLYGONS_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp

REM Output directory
set OUT_DIR=.\result_partition_k40_nc4_compare

REM Evaluation period
set START_MONTH=2021-01
set END_MONTH=2024-12

REM Model parameters
set TRAIN_WINDOW=36
set FORECASTING_SCOPE=3

REM --------------------------------------------------------------------------
REM Check if Python exists
REM --------------------------------------------------------------------------

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
    echo.
    echo Please edit this batch file and set PYTHON_EXE to your Python 3.12 installation.
    echo.
    pause
    exit /b 1
)

REM --------------------------------------------------------------------------
REM Parse command-line arguments
REM --------------------------------------------------------------------------

set VISUAL_FLAG=
if "%~1"=="--visual" set VISUAL_FLAG=--visual
if "%~1"=="-v" set VISUAL_FLAG=--visual

REM --------------------------------------------------------------------------
REM Display configuration
REM --------------------------------------------------------------------------

echo Configuration:
echo   Python:      %PYTHON_EXE%
echo   Data:        %DATA_PATH%
echo   Partition:   %PARTITION_MAP%
echo   Polygons:    %POLYGONS_PATH%
echo   Output:      %OUT_DIR%
echo   Period:      %START_MONTH% to %END_MONTH%
echo   Train Window: %TRAIN_WINDOW% months
echo   Forecasting Scope: %FORECASTING_SCOPE% (1=4mo, 2=8mo, 3=12mo lag)
echo   Visualization: %VISUAL_FLAG%
echo.
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Run comparison script
REM --------------------------------------------------------------------------

echo Running comparison...
echo.

"%PYTHON_EXE%" scripts\compare_partitioned_vs_pooled_rf_k40_nc4.py ^
  --data "%DATA_PATH%" ^
  --partition-map "%PARTITION_MAP%" ^
  --polygons "%POLYGONS_PATH%" ^
  --out-dir "%OUT_DIR%" ^
  --start-month "%START_MONTH%" ^
  --end-month "%END_MONTH%" ^
  --train-window %TRAIN_WINDOW% ^
  --forecasting-scope %FORECASTING_SCOPE% ^
  %VISUAL_FLAG%

REM --------------------------------------------------------------------------
REM Check exit code
REM --------------------------------------------------------------------------

if %ERRORLEVEL% neq 0 (
    echo.
    echo ============================================================================
    echo ERROR: Comparison script failed with exit code %ERRORLEVEL%
    echo ============================================================================
    echo.
    pause
    exit /b %ERRORLEVEL%
)

REM --------------------------------------------------------------------------
REM Success message
REM --------------------------------------------------------------------------

echo.
echo ============================================================================
echo COMPARISON COMPLETED SUCCESSFULLY
echo ============================================================================
echo.
echo Results saved to: %OUT_DIR%
echo.
echo CSV outputs:
echo   - metrics_monthly.csv
echo   - predictions_monthly.csv
echo   - metrics_admin0_overall.csv
echo   - run_manifest.json
echo.

if defined VISUAL_FLAG (
    echo Visualizations (4 maps):
    echo   - final_f1_performance_map.png
    echo   - map_pct_err_all.png
    echo   - map_pct_err_class1.png
    echo   - overall_f1_improvement_map.png
    echo.
) else (
    echo Visualizations: DISABLED
    echo   (Re-run with --visual flag to create 4 maps^)
    echo.
)

echo ============================================================================
echo.

pause
