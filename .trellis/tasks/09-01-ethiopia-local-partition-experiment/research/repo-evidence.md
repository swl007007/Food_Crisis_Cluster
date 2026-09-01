# Repository Evidence

- Ethiopia cohort contract: `EthiopiaForecastingExperiment/README.md:24-33`,
  `docs/contract-draft.md:49-77`, `evidence/2026-08-31-initial-evidence.md:11-23`.
- Standard no-leak windows: `PIPELINE_WORKFLOW.md` and
  `run_partition_k40_comparison_unified.bat:170-173,353-361`.
- fs0 lag isolation: `src/utils/lag_schedules.py:16-55`; current launchers make
  fs0 mutually exclusive and skip month-specific processing:
  `spatial_weighted_consensus_clustering.bat:32-45,332-343` and
  `run_partition_k40_comparison_unified.bat:137-162,399-407`.
- Stage 1 hard-coded data path and shared flow:
  `app/main_model_GF.py:94-136,1217-1390`.
- Current feature behavior retains originals and lagged copies:
  `src/feature/feature.py:56-120`.
- Stage 2 combines all discovered plans into one similarity matrix:
  `scripts/step4_similarity_matrix.py:225-269`; spectral seed is fixed at 42 in
  `scripts/step6_complete_clustering_pipeline.py:185-188`.
- Stage 3 already accepts data/out paths and emits pooled/partitioned metrics:
  `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py:841-868,973-1207`.
- Existing threshold option tunes partitioned only, motivating symmetric
  experiment behavior: `compare_partitioned_vs_pooled_rf_k40_nc4.py:1040-1190`.
- FEWS NET same-row script explicitly forbids shifting:
  `scripts/fewsnet_baseline_same_row.py:3-10`; archived row shifts are not
  calendar-safe: `archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/fewsnet_baseline_evaluation.py:71-75`.
- Live read-only evidence: `FEWSNET.csv` exact Ethiopia filtering yields 55,120
  rows, 1,040 codes equal to the authoritative cohort. Recent releases are
  February/June/October; 2021-06 near/medium coverage is 1/1,040.
- Existing plot shape is scopes by precision/recall/F1 with pooled/partitioned
  series: `scripts/paper_artifacts/plot_monthly_performance_metrics.py:41-50,261-337`.
