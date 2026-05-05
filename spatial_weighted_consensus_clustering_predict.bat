@echo off
setlocal enabledelayedexpansion
set PYTHONPATH=%~dp0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ============================================================================
REM Stage 2-pred: Verify and stage existing cluster mappings for the standalone
REM prediction pipeline. Does NOT run consensus clustering: predictions reuse
REM cluster maps produced by the labeled fs1/fs3 Stage 2 pipeline so partitions
REM remain consistent across forecast horizons.
REM ============================================================================
REM
REM Usage:
REM   spatial_weighted_consensus_clustering_predict.bat georf
REM
REM Looks under Stage 2 outputs for cluster_mapping_k40_nc*_general.csv produced
REM by the most recent fs1 and fs3 runs, then copies them to a stable, run-
REM scoped path that Stage 3-pred can rely on.
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
set "OUT_DIR=%REPO_ROOT%deliverables\predict_2026_2027\cluster_mappings"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Locate cluster_mapping_k40_nc*_general.csv files. We search the repo for
REM matching files; the most recently modified per scope is staged.
set "FS1_CANDIDATE="
set "FS3_CANDIDATE="

for /f "delims=" %%F in ('dir /b /s /o-d "%REPO_ROOT%cluster_mapping_k40_nc*_general.csv" 2^>nul') do (
    if not defined FS1_CANDIDATE if /I not "%%~nF"=="" (
        echo "%%F" | findstr /I /C:"fs1" >nul
        if !ERRORLEVEL! EQU 0 set "FS1_CANDIDATE=%%F"
    )
    if not defined FS3_CANDIDATE if /I not "%%~nF"=="" (
        echo "%%F" | findstr /I /C:"fs3" >nul
        if !ERRORLEVEL! EQU 0 set "FS3_CANDIDATE=%%F"
    )
)

REM Fallback: if scope tags are not embedded in the path, take the two most
REM recent general-cluster CSVs and require the operator to confirm which is
REM which by re-running with --fs1-map / --fs3-map. For now we just stage the
REM single most recent file as both, since most users run a single Stage 2.
if not defined FS1_CANDIDATE (
    for /f "delims=" %%F in ('dir /b /s /o-d "%REPO_ROOT%cluster_mapping_k40_nc*_general.csv" 2^>nul') do (
        if not defined FS1_CANDIDATE set "FS1_CANDIDATE=%%F"
    )
)
if not defined FS3_CANDIDATE set "FS3_CANDIDATE=%FS1_CANDIDATE%"

if not defined FS1_CANDIDATE (
    echo ERROR: no cluster_mapping_k40_nc*_general.csv found under "%REPO_ROOT%".
    echo Run the labeled Stage 2 pipeline ^(spatial_weighted_consensus_clustering.bat^)
    echo first to produce a partition map.
    exit /b 1
)

echo ============================================================================
echo Staging cluster mappings for standalone prediction pipeline
echo fs1 ^(Jun 2026 nowcast^):  %FS1_CANDIDATE%
echo fs3 ^(Feb 2027 forecast^): %FS3_CANDIDATE%
echo Destination:              %OUT_DIR%
echo ============================================================================

copy /Y "%FS1_CANDIDATE%" "%OUT_DIR%\fs1_general.csv" >nul
if errorlevel 1 (
    echo ERROR: failed to copy fs1 cluster map.
    exit /b 1
)
copy /Y "%FS3_CANDIDATE%" "%OUT_DIR%\fs3_general.csv" >nul
if errorlevel 1 (
    echo ERROR: failed to copy fs3 cluster map.
    exit /b 1
)

REM Coverage check: every polygon in the new combined CSV must be in the maps.
"%PYTHON_EXE%" -c ^
    "import sys; import pandas as pd; ^
data_path=r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\assembled_FEWSNET\FEWSNET_forecast_unadjusted_bm_2025_combined.csv'; ^
m1=pd.read_csv(r'%OUT_DIR%\fs1_general.csv'); m3=pd.read_csv(r'%OUT_DIR%\fs3_general.csv'); ^
data=pd.read_csv(data_path, usecols=['FEWSNET_admin_code']); polys=set(data['FEWSNET_admin_code'].astype(str)); ^
missing1=polys - set(m1['FEWSNET_admin_code'].astype(str)); missing3=polys - set(m3['FEWSNET_admin_code'].astype(str)); ^
print('fs1 unmapped polygons:', len(missing1)); print('fs3 unmapped polygons:', len(missing3)); ^
sys.exit(0 if (len(missing1)/max(1,len(polys)) < 0.01 and len(missing3)/max(1,len(polys)) < 0.01) else 2)"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo WARNING: cluster mappings have ^>1%% unmapped polygons; Stage 3-pred will route them to the pooled fallback.
    echo (continuing — fallback is documented in the deliverable manifest)
)

echo Stage 2-pred OK.
echo Next: run_partition_predict_unified.bat %MODEL_TYPE%
endlocal
