@echo off
setlocal
set "LAUNCHER_DIR=%~dp0"
for %%I in ("%LAUNCHER_DIR%..") do set "REPO_ROOT=%%~fI\"
set "PYTHONPATH=%REPO_ROOT%"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM Stage 1-pred: Diagnostic smoke-test for the standalone pure-prediction
REM pipeline (Jun 2026 nowcast, fs1 lag=4).
REM ============================================================================
REM
REM Usage:
REM   run_predict_2026_2027.bat georf
REM
REM Mirrors the operational surface of run_batches_2021_2024_visual_monthly.bat
REM but is intentionally light: it runs the canonical Stage 3-pred Python script
REM in --smoke mode (single target) so that any plumbing failure surfaces before
REM the full deliverable run. Outputs go to deliverables/predict_2026_2027/.
REM
REM Pre-requisites:
REM   1. The new combined CSV exists at:
REM      C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\
REM      1.Source Data\assembled_FEWSNET\
REM      FEWSNET_forecast_unadjusted_bm_2025_combined.csv
REM   2. A staged cluster mapping for fs1 has been produced by
REM      spatial_weighted_consensus_clustering_predict.bat georf.
REM ============================================================================

if "%~1"=="" (
    echo ERROR: model type is required.
    echo Usage: %~nx0 ^<model_type^>
    echo Supported: georf
    exit /b 1
)
set "MODEL_TYPE=%~1"
if /I not "%MODEL_TYPE%"=="georf" (
    echo ERROR: only "georf" is supported in the standalone prediction pipeline.
    exit /b 1
)

if not defined PYTHON_EXE set "PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set "STAGED_DIR=%REPO_ROOT%deliverables\predict_2026_2027\cluster_mappings"
set "PARTITION_MAP=%STAGED_DIR%\fs1_general.csv"
set "OUT_DIR=%REPO_ROOT%deliverables\predict_2026_2027\smoke"
set "GLOBAL_SHAPE=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"

if not exist "%PARTITION_MAP%" (
    echo ERROR: staged partition map not found at:
    echo   %PARTITION_MAP%
    echo Run spatial_weighted_consensus_clustering_predict.bat georf first.
    exit /b 1
)
if not exist "%GLOBAL_SHAPE%" (
    echo ERROR: global FEWSNET shapefile not found at:
    echo   %GLOBAL_SHAPE%
    echo Set GLOBAL_SHAPE in this launcher for explicit single-country analysis if needed.
    exit /b 1
)

echo ============================================================================
echo Stage 1-pred (smoke): GeoRF pure-prediction diagnostic
echo Partition map: %PARTITION_MAP%
echo Shapefile:     %GLOBAL_SHAPE%
echo Output dir:    %OUT_DIR%
echo ============================================================================

"%PYTHON_EXE%" "%REPO_ROOT%scripts\predict_partitioned_2026_2027.py" ^
    --partition-map "%PARTITION_MAP%" ^
    --polygons "%GLOBAL_SHAPE%" ^
    --out-dir "%OUT_DIR%" ^
    --smoke
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo Stage 1-pred FAILED with exit code %RC%.
    exit /b %RC%
)

echo Stage 1-pred smoke OK.
echo Next: run_partition_predict_unified.bat %MODEL_TYPE%
endlocal
