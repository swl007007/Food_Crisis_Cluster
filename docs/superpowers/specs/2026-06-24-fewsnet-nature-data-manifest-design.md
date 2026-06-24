# FEWSNET Nature Portfolio Data Manifest Design

## Purpose

Create a paper-facing Markdown data manifest for the current FEWSNET food-crisis forecasting workflow. The manifest should support Nature Portfolio-style data availability and reproducibility review by listing the minimum dataset needed to explain, verify, and rerun the paper results.

The deliverable is documentation only. It must not alter model code, source data, generated result folders, or workflow behavior.

Nature Portfolio policy reference: <https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards>

## Deliverable

Write the manifest to:

`final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md`

This location keeps the manifest with the paper methods and temporal-scope artifacts rather than with development-only specs.

## Scope

In scope:

- Primary assembled panel: `FEWSNET_forecast_unadjusted_bm.csv`.
- Local variable and provenance references:
  - `FEWSNET_forecast_unadjusted_bm_predictor_descriptives.md`
  - `variable_construction_notes_description.xlsx`
  - `assembled_IPCCH/metadata/variable_codebook_reorganized.csv`, used only for same-name variables shared with this FEWSNET panel.
- Source-data families represented in the assembled panel, including FEWS NET / IPC-like outcome and projection fields, FEWS NET admin boundaries, ACLED conflict features, AEZ and land/terrain features, remote-sensing weather and productivity features, SoilGrids / ISRIC soil features, WFP and FAO food-price fields, World Bank macro and food-price fields, population, nightlight, market-access, and derived z-score or adjusted fields.
- Derived reproducibility assets used by the current paper workflow:
  - Stage 2 cluster mapping manifests under `GeoRFExperiment/knn_sparsification_results/` and `GeoDTExperiment/knn_sparsification_results/`.
  - Stage 3 result folders `result_partition_k40_compare_{GF,DT}_fs{1,2,3}` and their `run_manifest.json` files.
  - Feature-exclude datasets under `main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/`.
  - Existing final paper artifact manifests under `final_artifacts_in_paper_updated/`.
  - FEWSNET baseline output files that feed the paper-facing comparison tables or figures.

Out of scope:

- Full recursive inventory of every raw upstream file in `Analysis/1.Source Data/`.
- Legacy GeoXGB outputs, unless a future paper section explicitly promotes them.
- `fs0` outputs, unless a future manuscript section uses the lag-1 experiment.
- Static/manual diagrams such as online-generated workflow or feature-engineering figures.
- Any public redistribution decision for restricted or third-party data.

## Confirmed Primary Dataset Metadata

The manifest should record the current on-disk metadata for:

`/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv`

Confirmed values from the design pass:

- SHA-256: `611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651`
- Size: `716303754` bytes
- Parsed CSV records: `1,029,240`
- Columns: `88`
- Physical text lines: `1,029,601`
- Date coverage: `2010-01` to `2024-12`
- Spatial coverage: `22` ISO3 countries and `5,718` FEWSNET admin codes
- Local modified time: `2025-11-05 09:41:30.068927800 -0500`

The manifest should report parsed records as the row count. The physical line count should be mentioned as a file-integrity caveat because it differs from parsed CSV records, likely due to quoted newlines or related CSV formatting.

## Manifest Fields

Each main table row should represent a data asset or artifact group, not every temporary file. Use these columns:

- `Data asset`
- `Role in reproduction`
- `Local path`
- `Source/provider`
- `License / access notes`
- `Checksum`
- `Rows x columns`
- `Date coverage / spatial coverage`
- `Generated or modified date`
- `Script / command`
- `Notes`

For file groups, the checksum field may point to the child manifest checksum or state that the component files are listed in an existing child manifest. For JSON, PNG, PDF, or shapefile assets, use `not tabular` or a relevant feature/file count instead of forcing row and column counts.

## License and Access Rules

Do not invent license statements. Use a three-level status:

- `Verified locally`: the local documentation or source file explicitly states license, DOI, citation, or access terms.
- `Provider identified; license pending verification`: the provider can be identified from local metadata or variable naming, but no local license file was found.
- `Derived from restricted/third-party sources`: the asset includes upstream data that may be restricted or subject to provider terms, so it should not be redistributed publicly until upstream licenses are reviewed.

The manifest should clearly mark ACLED, FEWS NET / IPC-style source data, WFP, FAO, and World Bank-derived indicators as requiring source-license verification before public redistribution unless the implementation finds explicit local license text.

## Document Structure

The Markdown manifest should contain:

1. **Purpose and Standard**: explain that this is a Nature Portfolio-style reproducibility manifest for the minimum dataset, not a full cache inventory.
2. **Primary Assembled Dataset**: document the FEWSNET CSV metadata, checksum, source families, and consuming scripts.
3. **Variable Provenance Summary**: summarize 88 columns by variable group and source family; state that the IPCCH codebook is only a same-variable cross-reference.
4. **Derived Reproducibility Assets**: list Stage 2 partition manifests, Stage 3 result groups, feature-exclude datasets, final artifact source manifests, and FEWSNET baseline outputs.
5. **License and Redistribution Notes**: separate confirmed local evidence from pending license verification.
6. **Reproduction Commands**: point to the three-stage no-leak workflow and `scripts/verify_current_results_reproducibility.py`.
7. **Known Caveats**: parsed rows vs physical lines, Windows/WSL path differences, legacy GeoXGB exclusion, fs0 exclusion, and static/manual figure exclusion.

## Evidence Sources

Implementation should derive values from existing local artifacts where possible:

- `CURRENT_RESULTS_REPRODUCTION.md`
- `PIPELINE_WORKFLOW.md`
- `run_partition_k40_comparison_unified.bat`
- `spatial_weighted_consensus_clustering.bat`
- `run_batches_2018_2020_partition_learning_visual_monthly.bat`
- `scripts/verify_current_results_reproducibility.py`
- Existing `run_manifest.json`, `cluster_mapping_manifest.json`, and `artifact_source_manifest.json` files
- `main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json`
- Source-data `AGENTS.md`
- FEWSNET predictor descriptives and IPCCH cross-reference codebook

## Acceptance Criteria

- The manifest includes source/provider, license/access notes, checksum, script/command, row-column count, and generated or modified date for the primary FEWSNET panel and major derived artifact groups.
- The primary FEWSNET CSV row count is reported as `1,029,240 x 88`, with the physical line-count discrepancy documented as a caveat.
- The primary FEWSNET CSV checksum is recorded exactly as `611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651`.
- Derived assets are connected to existing provider manifests or run manifests rather than described from memory.
- License notes do not overclaim public redistribution rights for third-party data.
- The manifest is concise enough for reviewer use while preserving the evidence needed to audit provenance.
- No model scripts, data files, result folders, or existing paper artifacts are modified except the new Markdown manifest.
