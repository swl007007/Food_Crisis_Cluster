# Paper Reproducibility Package Design

## Purpose

Create a lightweight paper-facing reproducibility package for the current FEWS NET
GeoRF manuscript artifacts and align `README.md` with the manuscript and the
current no-leak result bundle.

The package is intended for fast review and replication. It must avoid requiring a
full Stage 1 partition-learning rerun by bundling the already generated Stage 2
consensus partition artifacts and the current Stage 3/final paper outputs.

## Evidence Checked During Design

- Manuscript PDF:
  `final_artifacts_in_paper_updated/Forecasting_FEWS_NET_Food_Security_Crises_Using_a_Geo_Aware_Spatial_Clustering_Model (1).pdf`
- Root workflow docs: `README.md`, `PIPELINE_WORKFLOW.md`,
  `CURRENT_RESULTS_REPRODUCTION.md`
- Runtime/config sources: `config.py`, `config_visual.py`,
  `run_batches_2018_2020_partition_learning_visual_monthly.bat`,
  `spatial_weighted_consensus_clustering.bat`,
  `run_partition_k40_comparison_unified.bat`
- Stage 2 manifests:
  `GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`,
  `GeoDTExperiment/knn_sparsification_results/cluster_mapping_manifest.json`
- Stage 3 manifests:
  `result_partition_k40_compare_{GF,DT}_fs{1,2,3}/run_manifest.json`
- Final artifacts: `final_artifacts_in_paper_updated/`
- Existing verifier: `scripts/verify_current_results_reproducibility.py`

The current verifier passed during design exploration. It confirmed that the live
Stage 3 folders, Stage 2 handoff artifacts, final artifacts, fixed-partition
ablation outputs, and no-leak archive are organized and have reproducibility
metadata.

## Accepted Differences Between Paper Text and Repo Artifacts

Do not treat these as defects:

- The manuscript reports the analysis panel as the actual FEWS NET release-month
  and valid-area subset. The assembled source CSV is larger because it contains
  the broader monthly panel and source feature inventory.
- The manuscript reports `200,060` observations, `5,716` areas, and 22 countries
  for the analysis scope. Stage 3 run manifests report `5,716` polygons and
  `62,189` predictions per forecast horizon for the evaluated 2021-2024
  February/June/October rows. The source data manifest records the larger
  assembled panel separately.
- Predictor-count differences such as 60 base predictors versus the manuscript's
  expanded engineered-variable count are acceptable when they reflect the
  distinction between base variables and generated lag/rolling/encoded features.
- Stage 3 manifests are configured for 2021-01 through 2024-12 but report
  `n_test_months_evaluated=12`, matching the manuscript's February, June, and
  October FEWS NET release-month evaluation.

## Scope

In scope:

- Update `README.md` so the first-read documentation matches the paper-facing
  workflow, geography, horizons, model roles, configuration, and artifact paths.
- Add a lightweight repo-internal package directory:
  `paper_reproducibility_package/`.
- Include frozen Stage 2 cluster maps and manifests for GeoRF and GeoDT.
- Include Stage 3 result summaries and refined maps needed to audit or regenerate
  paper tables and figures without Stage 1.
- Include final paper artifacts and artifact provenance files.
- Include final ablation workbook and ablation manifests/checksums, but not the
  full 4.7 GB ablation output tree.
- Add package-level manifests/checksums and a validation path.

Out of scope:

- Rerunning Stage 1, Stage 2, Stage 3, SHAP, ablation, or figure-generation jobs
  as part of package creation.
- Changing model code, active config constants, source data, or model behavior.
- Moving or archiving experimental code paths in this task. GeoXGB, fs0, and
  2026-2027 prediction/scenario scripts should be separated in documentation
  and excluded from the paper reproducibility package, but their file locations
  should remain unchanged until a later release-version migration task.
- Copying raw upstream source data or the large
  `FEWSNET_forecast_unadjusted_bm.csv` into the package.
- Copying the full `main_ablation_exclude_updated_stage3_fixed_partitions/`
  output tree.
- Promoting GeoXGB, fs0, or 2026-2027 forward/scenario prediction into the paper
  main workflow.

## README Alignment Requirements

The root `README.md` should become paper-first and should no longer read as a
general collection of every experiment.

Required changes:

- Replace the opening "Sub-Saharan Africa" wording with a scope such as
  "22 FEWS NET monitored countries" or "FEWS NET monitored countries across
  Africa, the Middle East, Asia, and Latin America."
- State that the paper-facing workflow is the no-leak GeoRF workflow with
  fixed partitions learned from 2018-2020 and evaluated on 2021-2024.
- Explain that GeoRF is the main paper model. GeoDT remains an auxiliary
  comparison/interpretability workflow used in the appendix.
- Keep GeoXGB visible only as a legacy/experimental implementation, not as a
  main workflow or performance claim.
- Move fs0 to an "Experimental and Extension Workflows" section, outside the
  Quick Start path.
- Move `prediction_pipeline/` and 2026-2027 forward/scenario prediction to the
  same extension section, clearly separated from the manuscript results.
- Replace the current XGBoost F1 performance claim with manuscript-aligned GeoRF
  summary numbers or a pointer to `main_month_ind_cont3.xlsx` and Table 1.
- Update the config summary to match current code and manifests:
  `ACTIVE_LAGS=(4, 8, 12)`, `TRAIN_WINDOW_MONTHS=36`, `DATA_MODE=unadjusted`,
  `FEWSNET_forecast_unadjusted_bm.csv`, polygon adjacency enabled,
  contiguity refinement with three iterations in Stage 3 artifacts,
  `K=40` as graph neighbors, `SIGMA=5.0`, GeoRF Stage 3 RF parameters
  `n_estimators=100`, `max_depth=None`, `random_state=5`, `n_jobs=1`.
- Make the quick path point to `paper_reproducibility_package/README.md`.
- Preserve the full three-stage pipeline as the slow complete regeneration path.

## Package Layout

Create this directory structure:

```text
paper_reproducibility_package/
|-- README.md
|-- MANIFEST.csv
|-- SHA256SUMS.txt
|-- SOURCE_DATA.md
|-- PAPER_ARTIFACT_MAP.md
|-- CONSISTENCY_AUDIT.md
|-- stage2_cluster_maps/
|   |-- georf/
|   `-- geodt/
|-- stage3_results/
|   |-- georf_fs1/
|   |-- georf_fs2/
|   |-- georf_fs3/
|   |-- geodt_fs1/
|   |-- geodt_fs2/
|   `-- geodt_fs3/
|-- paper_artifacts/
`-- ablation/
```

### Root Package Files

- `README.md`: describe quick replication, full regeneration, package contents,
  source-data requirements, and known accepted differences.
- `MANIFEST.csv`: one row per package file with package path, source path,
  artifact role, size, SHA-256, and notes.
- `SHA256SUMS.txt`: plain checksum file for command-line verification.
- `SOURCE_DATA.md`: record that the raw assembled CSV is not copied; include
  expected Windows/WSL paths, SHA-256, row/column count, date coverage, and
  redistribution caveat from the FEWS NET data manifest.
- `PAPER_ARTIFACT_MAP.md`: map manuscript tables/figures/appendix artifacts to
  package paths and source scripts/manifests where known.
- `CONSISTENCY_AUDIT.md`: summarize PDF-to-repo checks, including aligned
  items and accepted differences.

### Stage 2 Cluster Maps

Copy the contents needed to skip Stage 1 and avoid redoing consensus clustering:

- GeoRF:
  - `cluster_mapping_manifest.json`
  - `cluster_mapping_k40_nc17_general.csv`
  - `cluster_mapping_k40_nc13_m2.csv`
  - `cluster_mapping_k40_nc11_m6.csv`
  - `cluster_mapping_k40_nc16_m10.csv`
- GeoDT:
  - `cluster_mapping_manifest.json`
  - `cluster_mapping_k40_nc15_general.csv`
  - `cluster_mapping_k40_nc16_m2.csv`
  - `cluster_mapping_k40_nc15_m6.csv`
  - `cluster_mapping_k40_nc18_m10.csv`

Do not copy full Stage 1 monthly visual archives. The package README should
explain that these Stage 2 maps are frozen provenance inputs; a clean checkout can
copy them into the canonical `GeoRFExperiment/` and `GeoDTExperiment/`
`knn_sparsification_results/` directories before rerunning Stage 3.

### Stage 3 Results

For each of `result_partition_k40_compare_GF_fs1`,
`result_partition_k40_compare_GF_fs2`, `result_partition_k40_compare_GF_fs3`,
`result_partition_k40_compare_DT_fs1`,
`result_partition_k40_compare_DT_fs2`, and
`result_partition_k40_compare_DT_fs3`, copy:

- `run_manifest.json`
- `metrics_monthly.csv`
- `metrics_polygon_overall.csv`
- `predictions_monthly.csv`
- canonical refined partition CSV files referenced by `run_manifest.json` and
  month-specific refined maps required for paper figures or thresholded
  appendix provenance

Do not copy Stage 3 `vis/` folders unless a file is directly mapped to a paper
artifact. Paper-facing figures already live under `final_artifacts_in_paper_updated/`.

### Paper Artifacts

Copy the active final artifact folder into:

`paper_reproducibility_package/paper_artifacts/final_artifacts_in_paper_updated/`

This should include the manuscript PDF, final figures/tables, notes, JSON
manifests, artifact-source audit files, and the Nature Portfolio-style data
manifest. The folder is about 53 MB and is acceptable for the lightweight package.

### Ablation

Do not copy the full 4.7 GB
`main_ablation_exclude_updated_stage3_fixed_partitions/` tree.

Copy only:

- `final_artifacts_in_paper_updated/01_main_results/ablation_feature_exclude.xlsx`
  if it is not already available through the copied final artifacts
- `main_ablation_exclude_updated_stage3_fixed_partitions/ablation_run_manifest.json`
- `main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json`

The package README should state that full ablation regeneration requires the
source CSV and the existing ablation scripts:

- `scripts/create_feature_exclude_datasets_from_unadjusted.py`
- `scripts/run_feature_exclude_stage3_fixed_partitions.py`
- `scripts/build_feature_exclude_ablation_workbook.py`

## Quick Replication Workflows

The package should document three levels of replication:

1. **Immediate audit, no model rerun**: verify checksums, inspect manifests, and
   compare packaged final artifacts to Stage 3 outputs.
2. **Fast result regeneration from cached outputs**: regenerate paper-facing
   summary tables/figures from the included Stage 3 results and final artifact
   scripts where supported.
3. **Stage 3 rerun without Stage 1**: restore or reference the packaged Stage 2
   cluster maps, then run `run_partition_k40_comparison_unified.bat georf
   --visual --month-ind` and, if needed for appendix comparison,
   `run_partition_k40_comparison_unified.bat geodt --visual --month-ind`.

The full slow path remains the three-stage workflow in `PIPELINE_WORKFLOW.md`.

## Validation Requirements

Implementation should provide fresh verification evidence before completion:

- `python3 scripts/verify_current_results_reproducibility.py`
- A package validation command that checks all files listed in `MANIFEST.csv`
  exist and match `SHA256SUMS.txt`.
- `rg` checks confirming the root README no longer presents Sub-Saharan Africa,
  GeoXGB, fs0, or 2026-2027 prediction as the paper main workflow.
- `git diff --check`
- `git status --short` to confirm only intended files are modified or added.

The existing verifier may rewrite line endings in
`final_artifacts_in_paper_updated/artifact_source_audit.md`; implementation
should avoid committing unrelated verifier formatting churn unless the content
actually changes.

## Acceptance Criteria

- `README.md` is paper-first and clearly separates main manuscript results from
  GeoXGB, fs0, and forward/scenario prediction extensions.
- The README geography no longer says the paper scope is only Sub-Saharan
  Africa.
- `paper_reproducibility_package/` exists and contains package README,
  manifest, checksums, source-data note, consistency audit, Stage 2 maps, Stage
  3 core outputs, final paper artifacts, and lightweight ablation provenance.
- The package does not copy the raw source CSV or full 4.7 GB ablation output
  tree.
- The package gives a clear path for quick audit, fast table/figure replication,
  and optional Stage 3 rerun using cached Stage 2 maps.
- Package checksums validate.
- Existing current-result verifier still passes, or any failure is documented as
  unrelated to this packaging change.
