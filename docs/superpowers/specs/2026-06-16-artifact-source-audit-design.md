# Artifact source audit and regeneration design

## Purpose

Remove phase-change data contamination from paper-facing GeoRF artifacts and
make future contamination fail closed. The active paper bundle is
`final_artifacts_in_paper_updated/`. Any artifact that is not provably derived
from the clean FEWSNET model panel must be treated as invalid until regenerated
or explicitly classified as static / shape-only.

Clean model-panel source:

`C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv`

Invalid model-panel source:

`C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_phase_change.csv`

## Scope

In scope:

- Audit all current paper-facing outputs under
  `final_artifacts_in_paper_updated/`.
- Audit provider result directories used by the paper bundle, including main
  GeoRF / GeoDT Stage 3 folders, thresholded GeoRF folders, and fixed-partition
  ablation providers.
- Fix script defaults that can accidentally use the phase-change panel for
  paper-facing GeoRF analysis.
- Regenerate every currently contaminated or suspicious artifact group from
  the clean panel source.
- Extend reproducibility verification so phase-change input cannot silently
  re-enter the final artifact bundle.

Out of scope:

- Reinterpreting phase-change results as an alternate analysis.
- Promoting phase-change outputs into appendix or supplementary materials.
- Retuning partitions, Stage 1 splitting, Stage 2 consensus clustering, or
  local RF hyperparameters.
- Expanding this cleanup to GeoXGB or additional baselines unless a paper
  artifact directly depends on them.

## Current Evidence

Known invalid outputs:

- `result_partition_k40_compare_GF_thresholded_fs1/`
- `result_partition_k40_compare_GF_thresholded_fs2/`
- `result_partition_k40_compare_GF_thresholded_fs3/`
- `final_artifacts_in_paper_updated/12_thresholded_georf_results/`

The thresholded provider manifests use
`FEWSNET_forecast_unadjusted_bm_phase_change.csv`, so the thresholded result
folders and the derived appendix artifacts are invalid.

Known suspicious outputs requiring regeneration:

- `final_artifacts_in_paper_updated/06_cluster_profiles/`
- `final_artifacts_in_paper_updated/10_false_negative_error_modes/`

Their generator scripts currently default to the phase-change panel. They use
clean main predictions but join panel covariates from the invalid source, so
their descriptive profiles and false-negative explanations must be regenerated.

Likely clean outputs based on existing manifests and script inspection:

- `result_partition_k40_compare_GF_fs1/`
- `result_partition_k40_compare_GF_fs2/`
- `result_partition_k40_compare_GF_fs3/`
- `result_partition_k40_compare_DT_fs1/`
- `result_partition_k40_compare_DT_fs2/`
- `result_partition_k40_compare_DT_fs3/`
- `main_ablation_exclude_updated_stage3_fixed_partitions/`
- `final_artifacts_in_paper_updated/03_class_prevalence/`
- `final_artifacts_in_paper_updated/04_error_analysis/`
- `final_artifacts_in_paper_updated/05_partition_diagnostics/`
- `final_artifacts_in_paper_updated/07_probability_uncertainty/`
- `final_artifacts_in_paper_updated/09_humanitarian_metrics/`
- `final_artifacts_in_paper_updated/11_threshold_free_metrics/`

These remain subject to the machine-readable audit. If a source cannot be
confirmed, it must be marked `needs_regeneration`, not assumed clean.

## Audit Design

Add a source-audit script that produces:

- `final_artifacts_in_paper_updated/artifact_source_audit.csv`
- `final_artifacts_in_paper_updated/artifact_source_audit.md`

The audit should classify each paper-facing artifact group and provider with
one of four statuses:

- `clean`: provenance explicitly points to the clean model panel or uses only
  clean prediction providers / raw outcome population data.
- `invalid_phase_change`: provenance explicitly points to the phase-change
  panel.
- `needs_regeneration`: provenance is incomplete, ambiguous, or uses a script
  that can default to phase-change input.
- `static_or_shape_only`: output does not use model-panel covariates or
  predictions, such as static diagrams or geography-only assets.

The audit should scan:

- all files under `final_artifacts_in_paper_updated/`;
- `run_manifest.json` files in GeoRF / GeoDT Stage 3 provider folders;
- thresholded GeoRF provider manifests;
- fixed-partition ablation dataset manifests;
- known generator scripts for artifact groups `03`, `04`, `06`, `07`, `09`,
  `10`, `11`, and `12`.

The audit should fail closed. Missing manifests, ambiguous input paths, or a
script defaulting to the phase-change panel should not pass as clean.

## Script Default Fixes

Patch paper-facing defaults from phase-change to the clean panel in:

- `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- `scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py`
- `scripts/analyze_georf_m2_cluster_profiles.py`
- `scripts/analyze_georf_false_negative_error_modes.py`

The `DATA_MODE == 'phase_change'` branches in `app/main_model_GF.py`,
`app/main_model_DT.py`, and `app/main_model_XGB.py` can remain as historical or
experimental paths. The cleanup only requires paper-facing defaults and
artifact generators to use the clean panel unless an explicit non-paper
experiment opts into phase-change.

## Required Regeneration

Regenerate these provider and artifact groups from
`FEWSNET_forecast_unadjusted_bm.csv`:

- `result_partition_k40_compare_GF_thresholded_fs1/`
- `result_partition_k40_compare_GF_thresholded_fs2/`
- `result_partition_k40_compare_GF_thresholded_fs3/`
- `final_artifacts_in_paper_updated/06_cluster_profiles/`
- `final_artifacts_in_paper_updated/10_false_negative_error_modes/`
- `final_artifacts_in_paper_updated/12_thresholded_georf_results/`

The regenerated thresholded GeoRF provider manifests must explicitly record the
clean panel source. The regenerated artifact notes should not mention
phase-change except in the audit report if historical invalidation needs to be
explained.

## Reproducibility Gate

Extend `scripts/verify_current_results_reproducibility.py` so verification
fails if:

- any paper-facing provider manifest uses the phase-change panel;
- thresholded GeoRF provider manifests do not end with
  `FEWSNET_forecast_unadjusted_bm.csv`;
- artifact groups `06`, `10`, or `12` are not classified as `clean`;
- `final_artifacts_in_paper_updated/artifact_source_audit.csv` contains
  `invalid_phase_change` or unresolved `needs_regeneration` rows for
  paper-facing artifacts.

The final verification should include:

- focused tests for the audit classifier and strict source-path matching;
- `python scripts/verify_current_results_reproducibility.py`;
- an `rg --no-ignore phase_change` scan over
  `final_artifacts_in_paper_updated/` and thresholded provider folders, with
  any remaining hits limited to code comments or explicit non-paper historical
  notes outside the final paper artifact content.

## Success Criteria

The cleanup is complete only when:

- all known invalid thresholded GeoRF results have been regenerated from the
  clean panel source;
- cluster profile and false-negative error-mode artifacts have been
  regenerated from the clean panel source;
- the source audit reports no paper-facing `invalid_phase_change` entries;
- remaining `needs_regeneration` entries, if any, are non-paper outputs or have
  a documented reason for exclusion from the final bundle;
- reproducibility verification fails on deliberate phase-change provenance and
  passes on the regenerated clean bundle.
