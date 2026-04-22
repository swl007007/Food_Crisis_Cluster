@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM Stage 2 of 3: Spatial Weighted Consensus Clustering Pipeline
REM ============================================================================
REM
REM Usage:
REM   spatial_weighted_consensus_clustering.bat georf
REM   spatial_weighted_consensus_clustering.bat geoxgb
REM   spatial_weighted_consensus_clustering.bat geodt
REM   spatial_weighted_consensus_clustering.bat georf --fs0-only
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
REM Note: step2_load_correspondence.py is not run by this batch. Its
REM output is unused by downstream steps, so the unified Stage 2 flow is
REM step1 -> step3 -> step4 -> step5 -> step6.
REM
REM --fs0-only: stand-alone fs0 (lag-1) pipeline. Consumes ONLY fs0 Stage 1
REM             artifacts and SKIPS Part 3 (month-specific partitions) because
REM             the fs0-only run produces too few candidate partitions. Leaves
REM             fs1+fs2+fs3 artifacts completely untouched.
REM
REM Default mode outputs (in {ExperimentDir}/knn_sparsification_results/):
REM   - cluster_mapping_k40_nc{N}_general.csv  (general, all months aggregated)
REM   - cluster_mapping_k40_nc{N}_m2.csv       (February-specific)
REM   - cluster_mapping_k40_nc{N}_m6.csv       (June-specific)
REM   - cluster_mapping_k40_nc{N}_m10.csv      (October-specific)
REM   - cluster_mapping_manifest.json          (paths to all partition files)
REM
REM --fs0-only mode outputs (Part 3 skipped, general only):
REM   - cluster_mapping_k40_nc{N}_general.csv  (fs0 general partition only)
REM   - cluster_mapping_manifest.json          (points to the general map only)
REM
REM Input globs (model + fs0-only aware):
REM   Default:    results_df_*_fs*.csv  +  result_Geo{Model}_*_visual
REM   --fs0-only: results_df_*_fs0_*.csv + result_Geo{Model}_*_fs0_*_visual
REM ============================================================================

echo.
echo ============================================================================
echo STAGE 2 OF 3: SPATIAL WEIGHTED CONSENSUS CLUSTERING PIPELINE
echo ============================================================================
echo.

REM --------------------------------------------------------------------------
REM Parse model type (required first argument)
REM --------------------------------------------------------------------------
if "%~1"=="" (
    echo ERROR: Model type is required.
    echo.
    echo Usage: %~nx0 ^<model_type^> [--fs0-only]
    echo.
    echo Model types:
    echo   georf   - GeoRF Random Forest
    echo   geoxgb  - GeoXGB XGBoost
    echo   geodt   - GeoDT Decision Tree
    echo.
    echo Options:
    echo   --fs0-only  Consume only fs0 Stage 1 outputs and skip Part 3
    echo.
    pause
    exit /b 1
)

set "MODEL_TYPE=%~1"
set "FS0_ONLY=0"

REM Parse optional flags
:parse_args
shift
if "%~1"=="" goto :done_args
if "%~1"=="--fs0-only" set "FS0_ONLY=1"
goto :parse_args
:done_args

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
if "%FS0_ONLY%"=="1" (
    echo Mode:             FS0-ONLY ^(stand-alone lag-1 pipeline^)
    set "STAGE1_HINT=run_batches_2021_2024_visual_monthly.bat %MODEL_TYPE% --fs0-only"
    set "RESULTS_GLOB_HINT=*fs0_*.csv"
    set "ARCHIVE_GLOB_HINT=result_Geo{Model}_*_fs0_*_visual"
    set "STAGE3_HINT=run_partition_k40_comparison_unified.bat %MODEL_TYPE% --fs0-only"
) else (
    echo Mode:             standard fs1+fs2+fs3 pipeline
    set "STAGE1_HINT=run_batches_2021_2024_visual_monthly.bat %MODEL_TYPE%"
    set "RESULTS_GLOB_HINT=*fs*.csv"
    set "ARCHIVE_GLOB_HINT=result_Geo{Model}_*_visual"
    set "STAGE3_HINT=run_partition_k40_comparison_unified.bat %MODEL_TYPE% --visual --month-ind"
)
echo Stage:            2 of 3 ^(consensus clustering^)
echo Stage 1 input:    %STAGE1_HINT%
echo.
echo Pipeline:
echo   Part 0 (gather):  Copy Stage 1 outputs into experiment directory
echo   Part 1 (shared):  step1 -^> step3 (merge, link)
echo   Part 2 (general): step4(general) -^> step5 -^> step6(general)
if "%FS0_ONLY%"=="1" (
    echo   Part 3 ^(monthly^): SKIPPED ^(fs0-only mode: insufficient candidate partitions^)
) else (
    echo   Part 3 ^(monthly^): step4^(m^) -^> step5 -^> step6^(m^) for m in {2, 6, 10}
)
echo.
echo Expected Stage 1 CSVs:     %RESULTS_GLOB_HINT%
echo Expected Stage 1 archives: %ARCHIVE_GLOB_HINT%
echo Stage 3 handoff:           cluster_mapping_manifest.json + cluster_mapping_k40_nc*.csv
echo Next stage command:        %STAGE3_HINT%
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

if "%FS0_ONLY%"=="1" (
    if /i "%MODEL_TYPE%"=="georf" set "RESULTS_GLOB=results_df_gp_fs0_*.csv"
    if /i "%MODEL_TYPE%"=="geoxgb" set "RESULTS_GLOB=results_df_xgb_gp_fs0_*.csv"
    if /i "%MODEL_TYPE%"=="geodt" set "RESULTS_GLOB=results_df_dt_gp_fs0_*.csv"
) else (
    if /i "%MODEL_TYPE%"=="georf" set "RESULTS_GLOB=results_df_gp_fs1_*.csv results_df_gp_fs2_*.csv results_df_gp_fs3_*.csv"
    if /i "%MODEL_TYPE%"=="geoxgb" set "RESULTS_GLOB=results_df_xgb_gp_fs1_*.csv results_df_xgb_gp_fs2_*.csv results_df_xgb_gp_fs3_*.csv"
    if /i "%MODEL_TYPE%"=="geodt" set "RESULTS_GLOB=results_df_dt_gp_fs1_*.csv results_df_dt_gp_fs2_*.csv results_df_dt_gp_fs3_*.csv"
)

for %%F in (%RESULTS_GLOB%) do (
    copy "%%F" "%TARGET_RESULTS_DIR%\" >nul 2>&1
    echo   [OK] %%~nxF
)

REM Copy archived visual directories (with correspondence tables) into results subdir
echo Copying archived visual directories...

if "%FS0_ONLY%"=="1" (
    if /i "%MODEL_TYPE%"=="georf" set "ARCHIVE_GLOB=result_GeoRF_*_fs0_*_visual"
    if /i "%MODEL_TYPE%"=="geoxgb" set "ARCHIVE_GLOB=result_GeoXGB_*_fs0_*_visual"
    if /i "%MODEL_TYPE%"=="geodt" set "ARCHIVE_GLOB=result_GeoDT_*_fs0_*_visual"
) else (
    if /i "%MODEL_TYPE%"=="georf" set "ARCHIVE_GLOB=result_GeoRF_*_fs1_*_visual result_GeoRF_*_fs2_*_visual result_GeoRF_*_fs3_*_visual"
    if /i "%MODEL_TYPE%"=="geoxgb" set "ARCHIVE_GLOB=result_GeoXGB_*_fs1_*_visual result_GeoXGB_*_fs2_*_visual result_GeoXGB_*_fs3_*_visual"
    if /i "%MODEL_TYPE%"=="geodt" set "ARCHIVE_GLOB=result_GeoDT_*_fs1_*_visual result_GeoDT_*_fs2_*_visual result_GeoDT_*_fs3_*_visual"
)

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
for %%F in (%RESULTS_GLOB%) do (
    if exist "%TARGET_RESULTS_DIR%\%%F" set "HAS_RESULTS=1"
)

set "HAS_ARCHIVES="
for %%D in (%ARCHIVE_GLOB%) do (
    if exist "%TARGET_RESULTS_DIR%\%%D" set "HAS_ARCHIVES=1"
)

if not defined HAS_RESULTS (
    echo.
    echo ERROR: No results CSV files found in %TARGET_RESULTS_DIR%
    echo Expected: %RESULTS_GLOB%
    echo Please run Stage 1 first: %STAGE1_HINT%
    pause
    exit /b 1
)
if not defined HAS_ARCHIVES (
    echo.
    echo ERROR: No archived visual directories found in %TARGET_RESULTS_DIR%
    echo Expected: %ARCHIVE_GLOB%
    echo Please run Stage 1 first: %STAGE1_HINT%
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
REM Skipped entirely in --fs0-only mode: fs0-only runs produce too few
REM candidate partitions to build month-specific consensus partitions.

if "%FS0_ONLY%"=="1" (
    echo ============================================================================
    echo PART 3 SKIPPED - FS0-ONLY mode ^(only general partition produced^)
    echo ============================================================================
    echo.
    goto :fs0_skip_part3
)

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

:fs0_skip_part3

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
if "%FS0_ONLY%"=="1" (
    echo   Similarity matrices:
    echo     - similarity_matrices/similarity_matrices.npz ^(general only^)
    echo.
    echo   KNN + Clustering:
    echo     - knn_sparsification_results/knn_graph_k40_general.npz
    echo     - knn_sparsification_results/knn_analysis_report_k40_general.json
    echo     - knn_sparsification_results/cluster_mapping_k40_nc*_general.csv
    echo     - knn_sparsification_results/cluster_mapping_manifest.json
    echo.
    echo Next step: Run Stage 3 comparison in fs0-only mode
    echo   %STAGE3_HINT%
) else (
    echo   Similarity matrices:
    echo     - similarity_matrices/similarity_matrices.npz ^(general^)
    echo     - similarity_matrices_m02/similarity_matrices_m02.npz ^(February^)
    echo     - similarity_matrices_m06/similarity_matrices_m06.npz ^(June^)
    echo     - similarity_matrices_m10/similarity_matrices_m10.npz ^(October^)
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
    echo   %STAGE3_HINT%
)
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
