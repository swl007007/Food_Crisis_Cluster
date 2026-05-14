@echo off
setlocal
set "LAUNCHER_DIR=%~dp0"
for %%I in ("%LAUNCHER_DIR%..") do set "REPO_ROOT=%%~fI\"
set "PYTHONPATH=%REPO_ROOT%"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM SCENARIO Prediction Overlay (Iran-oil + El-Nino), Jun 2026 + Feb 2027
REM ============================================================================
REM
REM This is a STAND-ALONE scenario overlay built on top of the standard
REM pure-prediction pipeline. It does NOT modify config.py, the standard
REM prediction script, or any standard deliverables.
REM
REM What it does:
REM   1) Re-runs the standard training procedure on real (non-mutated) data.
REM   2) On the prediction step, copies the target feature matrix and injects:
REM      - Iran-oil price shock (+200% on fs3, +100% on fs1) -> WFP_Price* + FAO_price*
REM      - El Nino weather z-shift (+1.5 fs3, +1.0 fs1)
REM      Applied ONLY to Greater Horn of Africa polygons (ETH,SOM,SDN,SSD,KEN,UGA;
REM      ERI and DJI are not present in the FEWSNET dataset and are skipped).
REM   3) Predicts at --scenario-threshold 0.40 by default (vs 0.50 standard).
REM      This is a scenario-only assumption, not the standard forecast threshold.
REM   4) Writes deliverables to: deliverables/predict_scenario_jun2026_feb2027/
REM
REM Usage:
REM   run_scenario_predict_jun2026_feb2027.bat
REM   run_scenario_predict_jun2026_feb2027.bat --smoke
REM ============================================================================

if not defined PYTHON_EXE set "PYTHON_EXE=C:\Users\swl00\AppData\Local\Microsoft\WindowsApps\python3.12.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set "STAGED_DIR=%REPO_ROOT%deliverables\predict_2026_2027\cluster_mappings"
set "PARTITION_MAP_FS1=%STAGED_DIR%\fs1_general.csv"
set "PARTITION_MAP_FS3=%STAGED_DIR%\fs3_general.csv"
set "OUT_DIR=%REPO_ROOT%deliverables\predict_scenario_jun2026_feb2027"

REM Continental shapefile (5718 polygons, 22 countries - full FEWSNET coverage).
REM config.py defaults to Nigeria.shp which would crop the maps to Nigeria only,
REM so we override here with the global shape used in production.
set "GLOBAL_SHAPE=C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"

if not exist "%PARTITION_MAP_FS1%" (
    echo ERROR: fs1 partition map not found at:
    echo   %PARTITION_MAP_FS1%
    echo Run spatial_weighted_consensus_clustering_predict.bat georf first.
    exit /b 1
)
if not exist "%PARTITION_MAP_FS3%" (
    echo ERROR: fs3 partition map not found at:
    echo   %PARTITION_MAP_FS3%
    echo Run spatial_weighted_consensus_clustering_predict.bat georf first.
    exit /b 1
)

echo ============================================================================
echo SCENARIO OVERLAY (synthetic; not part of standard pipeline)
echo fs1 partition map: %PARTITION_MAP_FS1%
echo fs3 partition map: %PARTITION_MAP_FS3%
echo Output dir:        %OUT_DIR%
echo ============================================================================

"%PYTHON_EXE%" "%REPO_ROOT%scripts\predict_scenario_2026_2027.py" ^
    --partition-map-fs1 "%PARTITION_MAP_FS1%" ^
    --partition-map-fs3 "%PARTITION_MAP_FS3%" ^
    --polygons "%GLOBAL_SHAPE%" ^
    --out-dir "%OUT_DIR%" ^
    %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo SCENARIO overlay FAILED with exit code %RC%.
    exit /b %RC%
)

echo SCENARIO overlay OK -- deliverables in %OUT_DIR%
endlocal
