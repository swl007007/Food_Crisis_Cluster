@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Spatial Weighted Consensus Clustering Pipeline (Stage 2)
REM ============================================================================
REM
REM Usage:
REM   spatial_weighted_consensus_clustering.bat georf
REM   spatial_weighted_consensus_clustering.bat geoxgb
REM   spatial_weighted_consensus_clustering.bat geodt
REM
REM This script runs the complete clustering pipeline (step1-step6) to produce
REM both GENERAL and MONTH-SPECIFIC (Feb, Jun, Oct) spatial partitions.
REM
REM Flow:
REM   Part 0 (gather):  Copy Stage 1 outputs into experiment directory
REM   Part 1 (shared):  step1 -> step3 (merge, link)
REM   Part 2 (general): step4(general) -> step5 -> step6(general)
REM   Part 3 (monthly): For each month (2, 6, 10):
REM                      step4(month) -> step5 -> step6(month)
REM
REM Outputs (in {ExperimentDir}/knn_sparsification_results/):
REM   - cluster_mapping_k40_nc{N}_general.csv
REM   - cluster_mapping_k40_nc{N}_m2.csv
REM   - cluster_mapping_k40_nc{N}_m6.csv
REM   - cluster_mapping_k40_nc{N}_m10.csv
REM   - cluster_mapping_manifest.json (paths to all partition files)
REM ============================================================================

echo.
echo ============================================================================
echo SPATIAL WEIGHTED CONSENSUS CLUSTERING PIPELINE
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Parse model type (required first argument)
REM --------------------------------------------------------------------------
if "%~1"=="" (
    echo ERROR: Model type is required.
    echo.
    echo Usage: %~nx0 ^<model_type^>
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

if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
    echo.
    echo Please edit this batch file and set PYTHON_EXE to your Python 3.12 installation.
    echo.
    pause
    exit /b 1
)

REM --------------------------------------------------------------------------
REM Display configuration
REM --------------------------------------------------------------------------
echo Model:            %MODEL_DISPLAY%
echo Experiment Dir:   %EXPERIMENT_DIR%
echo Results Subdir:   %RESULTS_SUBDIR%
echo.
echo Pipeline:
echo   Part 0 (gather):  Copy Stage 1 outputs into experiment directory
echo   Part 1 (shared):  step1 -^> step3 (merge, link)
echo   Part 2 (general): step4(general) -^> step5 -^> step6(general)
echo   Part 3 (monthly): step4(m) -^> step5 -^> step6(m) for m in {2, 6, 10}
echo.
echo Press CTRL+C to cancel, or any key to continue...
pause
echo.

REM ============================================================================
REM PART 0: GATHER STAGE 1 OUTPUTS INTO EXPERIMENT DIRECTORY
REM ============================================================================

echo ============================================================================
echo PART 0: GATHERING STAGE 1 OUTPUTS
echo ============================================================================
echo.

set "TARGET_RESULTS_DIR=%EXPERIMENT_DIR%\%RESULTS_SUBDIR%"

REM Create target directory if it doesn't exist
if not exist "%TARGET_RESULTS_DIR%" (
    echo Creating results directory: %TARGET_RESULTS_DIR%
    mkdir "%TARGET_RESULTS_DIR%"
)

REM Copy combined results CSVs (results_df_*) from repo root into results subdir
echo Copying results CSV files...

if /i "%MODEL_TYPE%"=="georf" set "RESULTS_GLOB=results_df_gp_fs*.csv"
if /i "%MODEL_TYPE%"=="geoxgb" set "RESULTS_GLOB=results_df_xgb_gp_fs*.csv"
if /i "%MODEL_TYPE%"=="geodt" set "RESULTS_GLOB=results_df_dt_gp_fs*.csv"

for %%F in (%RESULTS_GLOB%) do (
    copy "%%F" "%TARGET_RESULTS_DIR%\" >nul 2>&1
    echo   [OK] %%~nxF
)

REM Copy archived visual directories (with correspondence tables) into results subdir
echo Copying archived visual directories...

if /i "%MODEL_TYPE%"=="georf" set "ARCHIVE_GLOB=result_GeoRF_*_visual"
if /i "%MODEL_TYPE%"=="geoxgb" set "ARCHIVE_GLOB=result_GeoXGB_*_visual"
if /i "%MODEL_TYPE%"=="geodt" set "ARCHIVE_GLOB=result_GeoDT_*_visual"

for /d %%D in (%ARCHIVE_GLOB%) do (
    if not exist "%TARGET_RESULTS_DIR%\%%~nxD" (
        echo   Copying %%~nxD...
        xcopy /E /I /Y "%%D" "%TARGET_RESULTS_DIR%\%%~nxD" >nul 2>&1
        echo   [OK] %%~nxD
    ) else (
        echo   [EXISTS] %%~nxD already in target
    )
)

REM Validate: check target directory has at least one results CSV and one visual dir
REM (avoids unreliable set /a counter inside for blocks on some Windows versions)
set "HAS_RESULTS="
for %%F in ("%TARGET_RESULTS_DIR%\%RESULTS_GLOB%") do set "HAS_RESULTS=1"

set "HAS_ARCHIVES="
for /d %%D in ("%TARGET_RESULTS_DIR%\%ARCHIVE_GLOB%") do set "HAS_ARCHIVES=1"

if not defined HAS_RESULTS (
    echo.
    echo ERROR: No results CSV files found in %TARGET_RESULTS_DIR%
    echo Expected: %RESULTS_GLOB%
    echo Please run Stage 1 first: run_batches_2021_2024_visual_monthly.bat %MODEL_TYPE%
    pause
    exit /b 1
)
if not defined HAS_ARCHIVES (
    echo.
    echo ERROR: No archived visual directories found in %TARGET_RESULTS_DIR%
    echo Expected: %ARCHIVE_GLOB%
    echo Please run Stage 1 first: run_batches_2021_2024_visual_monthly.bat %MODEL_TYPE%
    pause
    exit /b 1
)

echo.
echo Stage 1 outputs verified in %TARGET_RESULTS_DIR%
echo.

echo ============================================================================
echo PART 0 COMPLETE - Stage 1 outputs gathered
echo ============================================================================
echo.

REM ============================================================================
REM PART 1: SHARED STEPS (step1 -> step3)
REM ============================================================================
REM NOTE: step2 (load_correspondence.py) was removed because its output
REM (correspondence_tables_loaded.pkl) is never consumed by downstream steps.
REM Step 3 reads merged_correspondence_tables.pkl directly from step 1.
REM ============================================================================

echo ============================================================================
echo PART 1: SHARED STEPS (step1, step3)
echo ============================================================================
echo.

REM Step 1: Merge results with correspondence tables
echo ------ Step 1: Merge Results with Correspondence Tables ------
"%PYTHON_EXE%" scripts\step1_merge_results.py --experiment-dir "%EXPERIMENT_DIR%" --model-type %MODEL_TYPE%
if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 1 failed
    pause
    exit /b %ERRORLEVEL%
)
echo Step 1 completed successfully.
echo.

REM Step 3: Create linked tables
echo ------ Step 3: Create Linked Tables ------
"%PYTHON_EXE%" scripts\step3_create_linked_tables.py --experiment-dir "%EXPERIMENT_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 3 failed
    pause
    exit /b %ERRORLEVEL%
)
echo Step 3 completed successfully.
echo.

echo ============================================================================
echo PART 1 COMPLETE - Shared artifacts generated (step1 + step3)
echo ============================================================================
echo.

REM ============================================================================
REM PART 2: GENERAL PARTITION (step4 -> step5 -> step6)
REM ============================================================================

echo ============================================================================
echo PART 2: GENERAL PARTITION (all months aggregated)
echo ============================================================================
echo.

set "GENERAL_SIM_DIR=similarity_matrices"
set "GENERAL_KNN_DIR=%EXPERIMENT_DIR%\knn_sparsification_results"

REM Step 4: Compute similarity matrix (general - no month filter)
echo ------ Step 4: Compute Similarity Matrix (General) ------
"%PYTHON_EXE%" scripts\step4_similarity_matrix.py --experiment-dir "%EXPERIMENT_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 4 ^(general^) failed
    pause
    exit /b %ERRORLEVEL%
)
echo Step 4 (general) completed successfully.
echo.

REM Step 5: KNN Sparsification + Eigengap Analysis (general)
echo ------ Step 5: KNN Sparsification + Eigengap (General) ------
"%PYTHON_EXE%" scripts\step5_sparsification.py --experiment-dir "%EXPERIMENT_DIR%" --similarity-dir "%GENERAL_SIM_DIR%" --suffix general
if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 5 ^(general^) failed
    pause
    exit /b %ERRORLEVEL%
)
echo Step 5 (general) completed successfully.
echo.

REM Step 6: Spectral Clustering (general)
echo ------ Step 6: Spectral Clustering (General) ------
"%PYTHON_EXE%" scripts\step6_complete_clustering_pipeline.py --experiment-dir "%EXPERIMENT_DIR%" --suffix general --similarity-dir "%GENERAL_SIM_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 6 ^(general^) failed
    pause
    exit /b %ERRORLEVEL%
)
echo Step 6 (general) completed successfully.
echo.

echo ============================================================================
echo PART 2 COMPLETE - General partition generated
echo ============================================================================
echo.

REM ============================================================================
REM PART 3: MONTH-SPECIFIC PARTITIONS (step4 -> step5 -> step6 per month)
REM ============================================================================

echo ============================================================================
echo PART 3: MONTH-SPECIFIC PARTITIONS (Feb, Jun, Oct)
echo ============================================================================
echo.

for %%M in (2 6 10) do (
    set "MONTH_NUM=%%M"
    if "%%M"=="2" set "MONTH_PADDED=02"
    if "%%M"=="6" set "MONTH_PADDED=06"
    if "%%M"=="10" set "MONTH_PADDED=10"

    set "MONTH_SIM_DIR=similarity_matrices_m!MONTH_PADDED!"
    set "MONTH_SUFFIX=m!MONTH_NUM!"

    echo.
    echo ====== Month %%M ^(m!MONTH_PADDED!^) ======
    echo.

    REM Step 4: Compute similarity matrix (month-specific)
    echo ------ Step 4: Compute Similarity Matrix ^(Month %%M^) ------
    "%PYTHON_EXE%" scripts\step4_similarity_matrix.py --experiment-dir "%EXPERIMENT_DIR%" --month %%M
    if !errorlevel! neq 0 (
        echo ERROR: Step 4 ^(month %%M^) failed
        pause
        exit /b !errorlevel!
    )
    echo Step 4 ^(month %%M^) completed successfully.
    echo.

    REM Step 5: KNN Sparsification + Eigengap Analysis (month-specific)
    echo ------ Step 5: KNN Sparsification + Eigengap ^(Month %%M^) ------
    "%PYTHON_EXE%" scripts\step5_sparsification.py --experiment-dir "%EXPERIMENT_DIR%" --similarity-dir "!MONTH_SIM_DIR!" --suffix "!MONTH_SUFFIX!"
    if !errorlevel! neq 0 (
        echo ERROR: Step 5 ^(month %%M^) failed
        pause
        exit /b !errorlevel!
    )
    echo Step 5 ^(month %%M^) completed successfully.
    echo.

    REM Step 6: Spectral Clustering (month-specific)
    echo ------ Step 6: Spectral Clustering ^(Month %%M^) ------
    "%PYTHON_EXE%" scripts\step6_complete_clustering_pipeline.py --experiment-dir "%EXPERIMENT_DIR%" --suffix "!MONTH_SUFFIX!" --similarity-dir "!MONTH_SIM_DIR!"
    if !errorlevel! neq 0 (
        echo ERROR: Step 6 ^(month %%M^) failed
        pause
        exit /b !errorlevel!
    )
    echo Step 6 ^(month %%M^) completed successfully.
    echo.
)

echo ============================================================================
echo PART 3 COMPLETE - Month-specific partitions generated
echo ============================================================================
echo.

REM ============================================================================
REM SUMMARY
REM ============================================================================

echo.
echo ============================================================================
echo ALL CLUSTERING COMPLETED SUCCESSFULLY for %MODEL_DISPLAY%
echo ============================================================================
echo.
echo Experiment directory: %EXPERIMENT_DIR%
echo.
echo Generated artifacts:
echo   Shared:
echo     - merged_correspondence_tables.pkl
echo     - linked_tables/ (main_index.csv, partitions/)
echo.
echo   Similarity matrices:
echo     - similarity_matrices/similarity_matrices.npz (general)
echo     - similarity_matrices_m02/similarity_matrices_m02.npz (February)
echo     - similarity_matrices_m06/similarity_matrices_m06.npz (June)
echo     - similarity_matrices_m10/similarity_matrices_m10.npz (October)
echo.
echo   KNN + Clustering:
echo     - knn_sparsification_results/knn_graph_k40_{general,m2,m6,m10}.npz
echo     - knn_sparsification_results/knn_analysis_report_k40_{general,m2,m6,m10}.json
echo     - knn_sparsification_results/cluster_mapping_k40_nc*_general.csv
echo     - knn_sparsification_results/cluster_mapping_k40_nc*_m2.csv
echo     - knn_sparsification_results/cluster_mapping_k40_nc*_m6.csv
echo     - knn_sparsification_results/cluster_mapping_k40_nc*_m10.csv
echo     - knn_sparsification_results/cluster_mapping_manifest.json
echo.
echo Next step: Run Stage 3 comparison
echo   run_partition_k40_comparison_unified.bat %MODEL_TYPE%
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
    set "RESULTS_SUBDIR=GeoRFResults"
    exit /b 0
)

if /i "%~1"=="geoxgb" (
    set "MODEL_DISPLAY=GeoXGB"
    set "EXPERIMENT_DIR=GeoXGBExperiment"
    set "RESULTS_SUBDIR=GeoXgboostResults"
    exit /b 0
)

if /i "%~1"=="geodt" (
    set "MODEL_DISPLAY=GeoDT"
    set "EXPERIMENT_DIR=GeoDTExperiment"
    set "RESULTS_SUBDIR=GeoDTResults"
    exit /b 0
)

REM Unknown model type
exit /b 1
