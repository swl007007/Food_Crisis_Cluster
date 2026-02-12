@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Unified Partitioned (k40) vs Pooled Model Comparison - Windows Batch Script
REM ============================================================================
REM
REM Usage:
REM   run_partition_k40_comparison_unified.bat georf                (default settings)
REM   run_partition_k40_comparison_unified.bat geoxgb --visual      (with maps)
REM   run_partition_k40_comparison_unified.bat geodt 1 3 --visual   (contiguity + maps)
REM
REM Arguments:
REM   %1  = model type: georf, geoxgb, geodt (REQUIRED)
REM   Optional positional/flags:
REM     First numeric arg  = CONTIGUITY (0 or 1, default from config)
REM     Second numeric arg = REFINE_ITERS (default from config)
REM     --visual / -v      = Enable visualization maps
REM     --month-ind        = Enable month-specific partitions
REM
REM The script auto-discovers partition files from the experiment directory's
REM cluster_mapping_manifest.json (written by Stage 2). Falls back to pattern
REM matching if manifest is missing.
REM ============================================================================

echo.
echo ============================================================================
echo UNIFIED PARTITIONED (k40) VS POOLED MODEL COMPARISON
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Parse model type (required first argument)
REM --------------------------------------------------------------------------
if "%~1"=="" (
    echo ERROR: Model type is required.
    echo.
    echo Usage: %~nx0 ^<model_type^> [contiguity] [refine_iters] [--visual] [--month-ind]
    echo.
    echo Model types:
    echo   georf   - GeoRF Random Forest
    echo   geoxgb  - GeoXGB XGBoost
    echo   geodt   - GeoDT Decision Tree
    echo.
    pause
    exit /b 1
)

set "MODEL_TYPE=%~1"

REM --------------------------------------------------------------------------
REM Load model-specific configuration
REM --------------------------------------------------------------------------
call :load_model_config %MODEL_TYPE%
if !errorlevel! neq 0 (
    echo ERROR: Invalid model type "%MODEL_TYPE%"
    echo Valid options: georf, geoxgb, geodt
    pause
    exit /b 1
)

REM --------------------------------------------------------------------------
REM Python executable
REM --------------------------------------------------------------------------
set PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe

REM Common data paths
set DATA_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv
set POLYGONS_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp

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
REM Auto-discover partition maps from manifest or pattern
REM --------------------------------------------------------------------------
call :discover_partition_maps
if !errorlevel! neq 0 (
    echo ERROR: Failed to discover partition maps for %MODEL_DISPLAY%
    echo.
    echo Expected location: %EXPERIMENT_DIR%\knn_sparsification_results\
    echo Expected files: cluster_mapping_k40_nc*_general.csv
    echo.
    echo Please run Stage 2 (spatial_weighted_consensus_clustering.bat %MODEL_TYPE%^) first.
    pause
    exit /b 1
)

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
REM Parse additional command-line arguments
REM --------------------------------------------------------------------------
set VISUAL_FLAG=
set MONTH_IND_FLAG=

REM Shift past model type argument
shift

:parse_comparison_args
if "%~1"=="" goto :done_comparison_args
if "%~1"=="--visual" (
    set VISUAL_FLAG=--visual
    goto :next_comparison_arg
)
if "%~1"=="-v" (
    set VISUAL_FLAG=--visual
    goto :next_comparison_arg
)
if "%~1"=="--month-ind" (
    set MONTH_IND_FLAG=--month-ind
    goto :next_comparison_arg
)
REM Check if numeric - try CONTIGUITY first, then REFINE_ITERS
set "temp_val=%~1"
set "temp_check=!temp_val:0=!"
set "temp_check=!temp_check:1=!"
set "temp_check=!temp_check:2=!"
set "temp_check=!temp_check:3=!"
set "temp_check=!temp_check:4=!"
set "temp_check=!temp_check:5=!"
set "temp_check=!temp_check:6=!"
set "temp_check=!temp_check:7=!"
set "temp_check=!temp_check:8=!"
set "temp_check=!temp_check:9=!"
if "!temp_check!"=="" (
    if not defined PARSED_CONTIGUITY (
        set CONTIGUITY=%~1
        set PARSED_CONTIGUITY=1
    ) else (
        set REFINE_ITERS=%~1
    )
)
:next_comparison_arg
shift
goto :parse_comparison_args
:done_comparison_args

REM Also check environment variable for month-ind
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
    REM Extract base names for refined file naming
    for %%F in ("%PARTITION_MAP_M2%") do set "M2_BASE=%%~nF"
    for %%F in ("%PARTITION_MAP_M6%") do set "M6_BASE=%%~nF"
    for %%F in ("%PARTITION_MAP_M10%") do set "M10_BASE=%%~nF"
    set PARTITION_MAP_M2=%REFINED_DIR%\!M2_BASE!_refined_contig%REFINE_ITERS%.csv
    set PARTITION_MAP_M6=%REFINED_DIR%\!M6_BASE!_refined_contig%REFINE_ITERS%.csv
    set PARTITION_MAP_M10=%REFINED_DIR%\!M10_BASE!_refined_contig%REFINE_ITERS%.csv
) else (
    echo.
    echo Contiguity refinement: DISABLED
    echo.
)

REM --------------------------------------------------------------------------
REM Display configuration
REM --------------------------------------------------------------------------

echo Configuration:
echo   Model:       %MODEL_DISPLAY%
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

echo Running %MODEL_DISPLAY% comparison...
echo.

"%PYTHON_EXE%" %COMPARISON_SCRIPT% ^
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
  %LOWER_MODEL_FLAG% ^
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
echo %MODEL_DISPLAY% COMPARISON COMPLETED SUCCESSFULLY
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
exit /b 0

REM ============================================================================
REM MODEL CONFIGURATION SUBROUTINE
REM ============================================================================

:load_model_config
REM Sets all model-specific variables based on model type argument
REM Args: %1 = model type (georf, geoxgb, geodt)

if /i "%~1"=="georf" (
    set "MODEL_DISPLAY=GeoRF"
    set "EXPERIMENT_DIR=GeoRFExperiment"
    set "COMPARISON_SCRIPT=scripts\compare_partitioned_vs_pooled_rf_k40_nc4.py"
    set "LOWER_MODEL_FLAG="
    set "OUT_DIR=.\result_partition_k40_compare_GF"
    exit /b 0
)

if /i "%~1"=="geoxgb" (
    set "MODEL_DISPLAY=GeoXGB"
    set "EXPERIMENT_DIR=GeoXGBExperiment"
    set "COMPARISON_SCRIPT=scripts\compare_partitioned_vs_pooled_xgb_k40_nc4.py"
    set "LOWER_MODEL_FLAG="
    set "OUT_DIR=.\result_partition_k40_compare_XGB"
    exit /b 0
)

if /i "%~1"=="geodt" (
    set "MODEL_DISPLAY=GeoDT"
    set "EXPERIMENT_DIR=GeoDTExperiment"
    set "COMPARISON_SCRIPT=scripts\compare_partitioned_vs_pooled_rf_k40_nc4.py"
    set "LOWER_MODEL_FLAG=--lower-model dt"
    set "OUT_DIR=.\result_partition_k40_compare_DT"
    exit /b 0
)

REM Unknown model type
exit /b 1

REM ============================================================================
REM AUTO-DISCOVER PARTITION MAPS
REM ============================================================================

:discover_partition_maps
REM Try manifest first, then fall back to pattern matching
set "KNN_DIR=%EXPERIMENT_DIR%\knn_sparsification_results"
set "MANIFEST_FILE=%KNN_DIR%\cluster_mapping_manifest.json"

if exist "%MANIFEST_FILE%" (
    echo Discovering partition maps from manifest: %MANIFEST_FILE%

    REM Use Python to read manifest and set environment variables
    REM Each manifest entry is {"path": "...", "n_clusters": N} or null
    for /f "tokens=1,* delims==" %%A in ('"%PYTHON_EXE%" -c "import json; m=json.load(open(r'%MANIFEST_FILE%')); gp=lambda k: (m.get(k) or {}).get('path','') if isinstance(m.get(k),dict) else (m.get(k) or ''); print('PARTITION_MAP=' + gp('general')); print('PARTITION_MAP_M2=' + gp('m02')); print('PARTITION_MAP_M6=' + gp('m06')); print('PARTITION_MAP_M10=' + gp('m10'))"') do (
        set "%%A=%%B"
    )

    REM Validate general partition exists
    if not exist "!PARTITION_MAP!" (
        echo WARNING: Manifest general partition not found: !PARTITION_MAP!
        echo Falling back to pattern discovery...
        goto :discover_by_pattern
    )

    echo   General:  !PARTITION_MAP!
    echo   Feb ^(m2^): !PARTITION_MAP_M2!
    echo   Jun ^(m6^): !PARTITION_MAP_M6!
    echo   Oct ^(m10^): !PARTITION_MAP_M10!
    exit /b 0
)

:discover_by_pattern
echo Discovering partition maps by pattern in %KNN_DIR%\...

REM Find general partition
set "PARTITION_MAP="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_general.csv") do (
    set "PARTITION_MAP=%%F"
)

if not defined PARTITION_MAP (
    echo ERROR: No general partition found matching cluster_mapping_k40_nc*_general.csv
    exit /b 1
)

REM Find month-specific partitions
set "PARTITION_MAP_M2="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_m2.csv") do (
    set "PARTITION_MAP_M2=%%F"
)

set "PARTITION_MAP_M6="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_m6.csv") do (
    set "PARTITION_MAP_M6=%%F"
)

set "PARTITION_MAP_M10="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_m10.csv") do (
    set "PARTITION_MAP_M10=%%F"
)

echo   General:  !PARTITION_MAP!
if defined PARTITION_MAP_M2 (echo   Feb ^(m2^): !PARTITION_MAP_M2!) else (echo   Feb ^(m2^): NOT FOUND)
if defined PARTITION_MAP_M6 (echo   Jun ^(m6^): !PARTITION_MAP_M6!) else (echo   Jun ^(m6^): NOT FOUND)
if defined PARTITION_MAP_M10 (echo   Oct ^(m10^): !PARTITION_MAP_M10!) else (echo   Oct ^(m10^): NOT FOUND)

exit /b 0
