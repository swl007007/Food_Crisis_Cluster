@echo off
REM ============================================================================
REM Partitioned (k40_nc4) vs Pooled XGBoost Comparison - Windows Batch Script
REM ============================================================================
REM
REM This batch file runs the XGBoost comparison script with default parameters.
REM
REM Usage:
REM   run_partition_k40_nc4_comparison_XGB.bat                 (no visuals, no contiguity)
REM   run_partition_k40_nc4_comparison_XGB.bat 1               (with contiguity, 2 iters)
REM   run_partition_k40_nc4_comparison_XGB.bat 1 3             (with contiguity, 3 iters)
REM   run_partition_k40_nc4_comparison_XGB.bat --visual        (with 4 maps, no contiguity)
REM   run_partition_k40_nc4_comparison_XGB.bat 1 2 --visual    (all features)
REM
REM Customize by editing the variables below before running.
REM ============================================================================

echo.
echo ============================================================================
echo PARTITIONED (k40_nc4) VS POOLED XGBOOST COMPARISON
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Configuration (Edit as needed)
REM --------------------------------------------------------------------------

REM Python executable (adjust path if needed)
set PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe

REM Data paths
set DATA_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_phase_change.csv
set PARTITION_MAP=cluster_mapping_k40_nc4.csv
set POLYGONS_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp

REM Month-specific partition maps (used when MONTH_IND=1)
set PARTITION_MAP_M2=cluster_mapping_k40_nc10_m2.csv
set PARTITION_MAP_M6=cluster_mapping_k40_nc2_m6.csv
set PARTITION_MAP_M10=cluster_mapping_k40_nc12_m10.csv

REM Output directory
set OUT_DIR=.\result_partition_k40_nc4_compare_XGB

REM Evaluation period
set START_MONTH=2021-01
set END_MONTH=2024-12

REM Model parameters
set TRAIN_WINDOW=36
set FORECASTING_SCOPE=2

REM Month-specific partitions (set to 1 to enable, 0 to disable)
set MONTH_IND=1

REM Contiguity refinement (set to 1 to enable, 0 to disable)
set CONTIGUITY=1

REM Refinement iterations (only used when CONTIGUITY=1)
set REFINE_ITERS=3

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
set MONTH_IND_FLAG=

REM Parse command-line arguments for CONTIGUITY and REFINE_ITERS
if not "%~1"=="" (
    if "%~1"=="--visual" set VISUAL_FLAG=--visual
    if "%~1"=="-v" set VISUAL_FLAG=--visual
    if "%~1"=="--month-ind" set MONTH_IND_FLAG=--month-ind
    REM Check if first arg is a number (CONTIGUITY)
    set "temp=%~1"
    set "temp=!temp:0=!"
    set "temp=!temp:1=!"
    if "!temp!"=="" if not "%~1"=="" set CONTIGUITY=%~1
)

if not "%~2"=="" (
    if "%~2"=="--visual" set VISUAL_FLAG=--visual
    if "%~2"=="-v" set VISUAL_FLAG=--visual
    if "%~2"=="--month-ind" set MONTH_IND_FLAG=--month-ind
    REM Check if second arg is a number (REFINE_ITERS)
    set "temp=%~2"
    set "temp=!temp:0=!"
    set "temp=!temp:1=!"
    set "temp=!temp:2=!"
    set "temp=!temp:3=!"
    set "temp=!temp:4=!"
    set "temp=!temp:5=!"
    set "temp=!temp:6=!"
    set "temp=!temp:7=!"
    set "temp=!temp:8=!"
    set "temp=!temp:9=!"
    if "!temp!"=="" if not "%~2"=="" set REFINE_ITERS=%~2
)

if "%~3"=="--visual" set VISUAL_FLAG=--visual
if "%~3"=="-v" set VISUAL_FLAG=--visual
if "%~3"=="--month-ind" set MONTH_IND_FLAG=--month-ind

REM Also check environment variable
if "%MONTH_IND%"=="1" set MONTH_IND_FLAG=--month-ind

REM --------------------------------------------------------------------------
REM Contiguity Refinement (if enabled)
REM --------------------------------------------------------------------------

set REFINED_DIR=%OUT_DIR%\refined
set ADJACENCY_CACHE=.\src\adjacency\polygon_adjacency_cache.pkl

if "%CONTIGUITY%"=="1" (
    echo.
    echo ============================================================================
    echo CONTIGUITY REFINEMENT ENABLED
    echo ============================================================================
    echo   Iterations: %REFINE_ITERS%
    echo   Adjacency cache: %ADJACENCY_CACHE%
    echo   Output: %REFINED_DIR%
    echo.

    REM Check if adjacency cache exists
    if not exist "%ADJACENCY_CACHE%" (
        echo ERROR: Adjacency cache not found at %ADJACENCY_CACHE%
        echo.
        pause
        exit /b 1
    )

    REM Refine month-specific partition maps
    echo Refining partition maps...

    "%PYTHON_EXE%" scripts\refine_partitions_contiguity.py --adj "%ADJACENCY_CACHE%" --in "%PARTITION_MAP_M2%" --out "%REFINED_DIR%" --iters %REFINE_ITERS%
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Refinement failed for %PARTITION_MAP_M2%
        pause
        exit /b %ERRORLEVEL%
    )

    "%PYTHON_EXE%" scripts\refine_partitions_contiguity.py --adj "%ADJACENCY_CACHE%" --in "%PARTITION_MAP_M6%" --out "%REFINED_DIR%" --iters %REFINE_ITERS%
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Refinement failed for %PARTITION_MAP_M6%
        pause
        exit /b %ERRORLEVEL%
    )

    "%PYTHON_EXE%" scripts\refine_partitions_contiguity.py --adj "%ADJACENCY_CACHE%" --in "%PARTITION_MAP_M10%" --out "%REFINED_DIR%" --iters %REFINE_ITERS%
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Refinement failed for %PARTITION_MAP_M10%
        pause
        exit /b %ERRORLEVEL%
    )

    echo Refinement complete.
    echo.

    REM Update partition map paths to use refined versions
    set PARTITION_MAP_M2=%REFINED_DIR%\cluster_mapping_k40_nc10_m2_refined_contig%REFINE_ITERS%.csv
    set PARTITION_MAP_M6=%REFINED_DIR%\cluster_mapping_k40_nc2_m6_refined_contig%REFINE_ITERS%.csv
    set PARTITION_MAP_M10=%REFINED_DIR%\cluster_mapping_k40_nc12_m10_refined_contig%REFINE_ITERS%.csv
) else (
    echo.
    echo Contiguity refinement: DISABLED
    echo.
)

REM --------------------------------------------------------------------------
REM Display configuration
REM --------------------------------------------------------------------------

echo Configuration:
echo   Python:      %PYTHON_EXE%
echo   Data:        %DATA_PATH%
echo   Partition:   %PARTITION_MAP%
if defined MONTH_IND_FLAG (
    echo   Month-specific partitions^: ENABLED
    echo     - Feb ^(m2^)^:  %PARTITION_MAP_M2%
    echo     - Jun ^(m6^)^:  %PARTITION_MAP_M6%
    echo     - Oct ^(m10^)^: %PARTITION_MAP_M10%
) else (
    echo   Month-specific partitions^: DISABLED
)
if "%CONTIGUITY%"=="1" (
    echo   Contiguity Refinement^: ENABLED ^(%REFINE_ITERS% iterations^)
) else (
    echo   Contiguity Refinement^: DISABLED
)
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

echo Running XGBoost comparison...
echo.

"%PYTHON_EXE%" scripts\compare_partitioned_vs_pooled_xgb_k40_nc4.py ^
  --data "%DATA_PATH%" ^
  --partition-map "%PARTITION_MAP%" ^
  --partition-map-m2 "%PARTITION_MAP_M2%" ^
  --partition-map-m6 "%PARTITION_MAP_M6%" ^
  --partition-map-m10 "%PARTITION_MAP_M10%" ^
  --polygons "%POLYGONS_PATH%" ^
  --out-dir "%OUT_DIR%" ^
  --start-month "%START_MONTH%" ^
  --end-month "%END_MONTH%" ^
  --train-window %TRAIN_WINDOW% ^
  --forecasting-scope %FORECASTING_SCOPE% ^
  %MONTH_IND_FLAG% ^
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
echo   - metrics_polygon_overall.csv
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
