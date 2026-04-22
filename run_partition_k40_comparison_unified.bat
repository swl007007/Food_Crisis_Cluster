@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Unified Partitioned (k40) vs Pooled Model Comparison - Windows Batch Script
REM ============================================================================
REM
REM Usage:
REM   run_partition_k40_comparison_unified.bat all                   (run all 9 combos)
REM   run_partition_k40_comparison_unified.bat georf                 (single model, all scopes)
REM   run_partition_k40_comparison_unified.bat geoxgb --visual       (with maps)
REM   run_partition_k40_comparison_unified.bat geodt 1 3 --visual    (contiguity + maps)
REM
REM Arguments:
REM   %1  = model type: georf, geoxgb, geodt, or "all" (REQUIRED)
REM         "all" loops over georf, geoxgb, geodt x scopes 1,2,3
REM   Optional positional/flags:
REM     First numeric arg  = CONTIGUITY (0 or 1, default from config)
REM     Second numeric arg = REFINE_ITERS (default from config)
REM     --visual / -v      = Enable visualization maps
REM     --month-ind        = Enable month-specific partitions
REM     --scope N          = Run only scope N (1, 2, or 3); default: all three
REM
REM The script auto-discovers partition files from the experiment directory's
REM cluster_mapping_manifest.json (written by Stage 2). Falls back to pattern
REM matching if manifest is missing.
REM
REM After all runs complete, calls aggregate_results.py to build Table_Format.xlsx.
REM ============================================================================

echo.
echo ============================================================================
echo UNIFIED PARTITIONED (k40) VS POOLED MODEL COMPARISON
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Python executable
REM --------------------------------------------------------------------------
set PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
    pause
    exit /b 1
)

REM --------------------------------------------------------------------------
REM Parse model type (required first argument)
REM --------------------------------------------------------------------------
if "%~1"=="" (
    echo ERROR: Model type is required.
    echo.
    echo Usage: %~nx0 ^<model_type^|all^> [--scope N] [--visual] [--month-ind]
    echo.
    echo Model types: georf, geoxgb, geodt, all
    echo.
    pause
    exit /b 1
)

set "FIRST_ARG=%~1"

REM --------------------------------------------------------------------------
REM Parse flags shared by single-model and all-model modes
REM --------------------------------------------------------------------------
set VISUAL_FLAG=
set MONTH_IND_FLAG=
set SCOPE_FILTER=
set CONTIGUITY=1
set REFINE_ITERS=3
set PARSED_CONTIGUITY=

shift
:parse_global_args
if "%~1"=="" goto :done_global_args
if "%~1"=="--visual" ( set VISUAL_FLAG=--visual& goto :next_global_arg )
if "%~1"=="-v"       ( set VISUAL_FLAG=--visual& goto :next_global_arg )
if "%~1"=="--month-ind" ( set MONTH_IND_FLAG=--month-ind& goto :next_global_arg )
if "%~1"=="--scope" (
    shift
    set "SCOPE_FILTER=%~1"
    goto :next_global_arg
)
REM Numeric args -> CONTIGUITY then REFINE_ITERS
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
:next_global_arg
shift
goto :parse_global_args
:done_global_args

REM Also check environment variable for month-ind
if "%MONTH_IND%"=="1" set MONTH_IND_FLAG=--month-ind

REM --------------------------------------------------------------------------
REM Determine which scopes to iterate
REM --------------------------------------------------------------------------
if defined SCOPE_FILTER (
    set "SCOPES=%SCOPE_FILTER%"
) else (
    set "SCOPES=1 2 3"
)

REM --------------------------------------------------------------------------
REM "all" mode: loop over all 3 models, each with all scopes
REM --------------------------------------------------------------------------
if /i "%FIRST_ARG%"=="all" (
    echo MODE: Running ALL models x ALL scopes
    echo ============================================================================
    set "ALL_FAIL=0"
    for %%M in (georf geoxgb geodt) do (
        for %%S in (%SCOPES%) do (
            echo.
            echo ************************************************************
            echo  %%M  scope=%%S
            echo ************************************************************
            call :run_single_combo %%M %%S
            if !errorlevel! neq 0 (
                echo WARNING: %%M scope=%%S failed.
                set "ALL_FAIL=1"
            )
        )
    )
    goto :aggregate
)

REM --------------------------------------------------------------------------
REM Single-model mode: loop over scopes for the given model
REM --------------------------------------------------------------------------
call :load_model_config %FIRST_ARG%
if !errorlevel! neq 0 (
    echo ERROR: Invalid model type "%FIRST_ARG%"
    echo Valid options: georf, geoxgb, geodt, all
    pause
    exit /b 1
)

set "ALL_FAIL=0"
for %%S in (%SCOPES%) do (
    echo.
    echo ************************************************************
    echo  %FIRST_ARG%  scope=%%S
    echo ************************************************************
    call :run_single_combo %FIRST_ARG% %%S
    if !errorlevel! neq 0 (
        echo WARNING: %FIRST_ARG% scope=%%S failed.
        set "ALL_FAIL=1"
    )
)
goto :aggregate

REM ============================================================================
REM :run_single_combo  <model_type> <scope>
REM   Runs one comparison for the given model + forecasting scope.
REM   Skips if metrics_monthly.csv already exists in the output dir.
REM ============================================================================
:run_single_combo
set "RSC_MODEL=%~1"
set "RSC_SCOPE=%~2"

call :load_model_config %RSC_MODEL%
set "RSC_OUT_DIR=!OUT_DIR!_fs%RSC_SCOPE%"

REM Skip if results already exist
if exist "!RSC_OUT_DIR!\metrics_monthly.csv" (
    echo   SKIP: !RSC_OUT_DIR!\metrics_monthly.csv already exists.
    exit /b 0
)

call :discover_partition_maps
if !errorlevel! neq 0 (
    echo   ERROR: No partition maps for !MODEL_DISPLAY!. Run Stage 2 first.
    exit /b 1
)

REM Contiguity refinement
if "%CONTIGUITY%"=="1" (
    call :do_contiguity_refinement
    if !errorlevel! neq 0 exit /b 1
)

echo   Model: !MODEL_DISPLAY!  Scope: %RSC_SCOPE%  Out: !RSC_OUT_DIR!

"%PYTHON_EXE%" !COMPARISON_SCRIPT! ^
  --data "%DATA_PATH%" ^
  --partition-map "!PARTITION_MAP!" ^
  --partition-map-m2 "!PARTITION_MAP_M2!" ^
  --partition-map-m6 "!PARTITION_MAP_M6!" ^
  --partition-map-m10 "!PARTITION_MAP_M10!" ^
  --polygons "%POLYGONS_PATH%" ^
  --out-dir "!RSC_OUT_DIR!" ^
  --start-month "%START_MONTH%" ^
  --end-month "%END_MONTH%" ^
  --train-window %TRAIN_WINDOW% ^
  --forecasting-scope %RSC_SCOPE% ^
  !LOWER_MODEL_FLAG! ^
  %MONTH_IND_FLAG% ^
  %VISUAL_FLAG%

if !errorlevel! neq 0 (
    echo   ERROR: Comparison script failed for !MODEL_DISPLAY! scope=%RSC_SCOPE%
    exit /b 1
)

echo   OK: !MODEL_DISPLAY! scope=%RSC_SCOPE% complete.
exit /b 0

REM ============================================================================
REM AGGREGATE: collect all metrics_monthly.csv files and build the Excel table
REM ============================================================================
:aggregate
echo.
echo ============================================================================
echo AGGREGATING RESULTS INTO TABLE
echo ============================================================================

"%PYTHON_EXE%" other_outputs\aggregate_results.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Aggregation script failed.
)

echo.
echo ============================================================================
echo ALL DONE
echo ============================================================================

if "%ALL_FAIL%"=="1" (
    echo Some runs failed. Check output above for details.
)

pause
exit /b 0

REM ============================================================================
REM MODEL CONFIGURATION SUBROUTINE
REM ============================================================================

:load_model_config
set DATA_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_NGA.csv
set POLYGONS_PATH=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\Nigeria.shp
set START_MONTH=2021-01
set END_MONTH=2024-12
set TRAIN_WINDOW=36

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
exit /b 1

REM ============================================================================
REM CONTIGUITY REFINEMENT SUBROUTINE
REM ============================================================================

:do_contiguity_refinement
set "REFINED_DIR=!RSC_OUT_DIR!\refined"
set "ADJACENCY_CACHE=.\src\adjacency\polygon_adjacency_cache.pkl"
if not exist "%ADJACENCY_CACHE%" (
    echo ERROR: Adjacency cache not found at %ADJACENCY_CACHE%
    exit /b 1
)
for %%P in ("!PARTITION_MAP!" "!PARTITION_MAP_M2!" "!PARTITION_MAP_M6!" "!PARTITION_MAP_M10!") do (
    "%PYTHON_EXE%" scripts\refine_partitions_contiguity.py --adj "%ADJACENCY_CACHE%" --in %%P --out "!REFINED_DIR!" --iters %REFINE_ITERS%
    if !errorlevel! neq 0 exit /b 1
)
for %%F in ("!PARTITION_MAP!") do set "GEN_BASE=%%~nF"
for %%F in ("!PARTITION_MAP_M2!") do set "M2_BASE=%%~nF"
for %%F in ("!PARTITION_MAP_M6!") do set "M6_BASE=%%~nF"
for %%F in ("!PARTITION_MAP_M10!") do set "M10_BASE=%%~nF"
set "PARTITION_MAP=!REFINED_DIR!\!GEN_BASE!_refined_contig%REFINE_ITERS%.csv"
set "PARTITION_MAP_M2=!REFINED_DIR!\!M2_BASE!_refined_contig%REFINE_ITERS%.csv"
set "PARTITION_MAP_M6=!REFINED_DIR!\!M6_BASE!_refined_contig%REFINE_ITERS%.csv"
set "PARTITION_MAP_M10=!REFINED_DIR!\!M10_BASE!_refined_contig%REFINE_ITERS%.csv"
exit /b 0

REM ============================================================================
REM AUTO-DISCOVER PARTITION MAPS
REM ============================================================================

:discover_partition_maps
set "KNN_DIR=%EXPERIMENT_DIR%\knn_sparsification_results"
set "MANIFEST_FILE=%KNN_DIR%\cluster_mapping_manifest.json"

if exist "%MANIFEST_FILE%" (
    for /f "tokens=1,* delims==" %%A in ('"%PYTHON_EXE%" -c "import json; m=json.load(open(r'%MANIFEST_FILE%')); gp=lambda k: (m.get(k) or {}).get('path','') if isinstance(m.get(k),dict) else (m.get(k) or ''); print('PARTITION_MAP=' + gp('general')); print('PARTITION_MAP_M2=' + gp('m02')); print('PARTITION_MAP_M6=' + gp('m06')); print('PARTITION_MAP_M10=' + gp('m10'))"') do (
        set "%%A=%%B"
    )
    if not exist "!PARTITION_MAP!" goto :discover_by_pattern
    exit /b 0
)

:discover_by_pattern
set "PARTITION_MAP="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_general.csv") do set "PARTITION_MAP=%%F"
if not defined PARTITION_MAP (
    echo ERROR: No general partition found in %KNN_DIR%
    exit /b 1
)
set "PARTITION_MAP_M2="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_m2.csv") do set "PARTITION_MAP_M2=%%F"
set "PARTITION_MAP_M6="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_m6.csv") do set "PARTITION_MAP_M6=%%F"
set "PARTITION_MAP_M10="
for %%F in ("%KNN_DIR%\cluster_mapping_k40_nc*_m10.csv") do set "PARTITION_MAP_M10=%%F"
exit /b 0
