@echo off
setlocal
set PYTHONPATH=%~dp0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM Stage 3-pred: Canonical pure-prediction run for the standalone deliverables.
REM Trains pooled + per-cluster RFs on labels through the most recent FEWSNET
REM publication, then predicts:
REM   - Jun 2026 (nowcast, fs1 = lag 4 months)
REM   - Feb 2027 (12-month forecast, fs3 = lag 12 months)
REM
REM April 2026 / April 2027 are non-publication months in the FEWSNET cadence.
REM Jun 2026 is the closest near-term FEWSNET projection horizon and Feb 2027
REM is the next 12-month horizon from the 2026-02 publication.
REM ============================================================================
REM
REM Usage:
REM   run_partition_predict_unified.bat georf
REM
REM Outputs (under deliverables/predict_2026_2027/):
REM   D1: predictions_2026-06_2027-02_georf.xlsx   (per-polygon classification + uncertainty)
REM   D2: map_phase3plus_2026-06_georf.png         (Phase 3+ map for Jun 2026)
REM   D2: map_phase3plus_2027-02_georf.png         (Phase 3+ map for Feb 2027)
REM       run_manifest.json (reproducibility)
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

set "PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe"
set "REPO_ROOT=%~dp0"
set "STAGED_DIR=%REPO_ROOT%deliverables\predict_2026_2027\cluster_mappings"
REM The standalone script accepts a single partition map argument; for fs1/fs3
REM we rely on the canonical Stage 2-pred staging (fs1_general.csv) since the
REM Stage 2 spatial partitions are stable across forecast horizons. Use a
REM separate map per target only if Stage 2-pred staged distinct files.
set "PARTITION_MAP=%STAGED_DIR%\fs1_general.csv"
set "OUT_DIR=%REPO_ROOT%deliverables\predict_2026_2027"

if not exist "%PARTITION_MAP%" (
    echo ERROR: staged partition map not found at:
    echo   %PARTITION_MAP%
    echo Run spatial_weighted_consensus_clustering_predict.bat %MODEL_TYPE% first.
    exit /b 1
)

echo ============================================================================
echo Stage 3-pred: pure-prediction deliverables (Jun 2026 + Feb 2027)
echo Partition map: %PARTITION_MAP%
echo Output dir:    %OUT_DIR%
echo ============================================================================

"%PYTHON_EXE%" "%REPO_ROOT%scripts\predict_partitioned_2026_2027.py" ^
    --partition-map "%PARTITION_MAP%" ^
    --out-dir "%OUT_DIR%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo Stage 3-pred FAILED with exit code %RC%.
    exit /b %RC%
)

echo Stage 3-pred OK. Deliverables written to %OUT_DIR%.
endlocal
