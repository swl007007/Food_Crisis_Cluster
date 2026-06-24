# FEWSNET Data Manifest for Nature Portfolio Reproducibility Review

## Purpose and Standard

This manifest is a Nature Portfolio-style reproducibility index for the current FEWSNET food-crisis forecasting workflow. It identifies the minimum dataset and derived artifacts needed to explain, audit, and rerun the paper-facing results; it is not a full recursive inventory of every raw cache or upstream source file under `Analysis/1.Source Data/`.

The entries below separate local reproducibility evidence from data-rights statements. Local checksums, row counts, timestamps, dimensions, and child manifests are reported as file-integrity evidence, not license evidence. Upstream licenses and redistribution rights remain subject to source-license verification, and this manifest does not claim public redistribution rights for restricted or third-party data.

## Primary Assembled Dataset

| Data asset | Role in reproduction | Local path | Source/provider | License / access notes | Checksum | Rows x columns | Date coverage / spatial coverage | Generated or modified date | Script / command | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `FEWSNET_forecast_unadjusted_bm.csv` | Primary assembled panel used by Stage 1 partition learning, Stage 2 clustering handoff, Stage 3 fixed-partition evaluation, paper helper scripts, and reproducibility verification. | WSL: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv`<br>Windows: `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv` | Assembled FEWSNET panel spanning FEWS NET geography/outcomes/projections, ACLED conflict exposure, AEZ and land/access/terrain/hydrology features, remote-sensing climate/productivity/nightlight fields, soil, macro, food-price, and population indicators. | Derived from restricted/third-party sources; do not claim public redistribution rights before source-license verification. | `611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651` | `1,029,240 x 88` | `2010-01 to 2024-12`; 22 ISO3 countries; 5,718 FEWSNET admin codes (`AFG`, `BDI`, `BFA`, `CMR`, `COD`, `ETH`, `GTM`, `HTI`, `KEN`, `MDG`, `MLI`, `MOZ`, `MWI`, `NER`, `NGA`, `SDN`, `SOM`, `SSD`, `TCD`, `UGA`, `YEM`, `ZWE`) | `2025-11-05 09:41:30 -0500` | `run_batches_2018_2020_partition_learning_visual_monthly.bat`; `spatial_weighted_consensus_clustering.bat`; `run_partition_k40_comparison_unified.bat`; paper helper scripts under `scripts/`; `scripts/verify_current_results_reproducibility.py` | Size is `716303754` bytes. Parsed CSV records are the reported row count. Physical text lines are `1,029,601`, which differs from parsed records and is tracked as a file-integrity caveat. |

The primary dataset provides the target and predictor columns consumed by the current no-leak paper workflow. It is the only assembled source panel required for the main GeoRF rerun commands listed below; derived manifests record how downstream partition maps, fixed-partition runs, ablations, and paper artifacts were produced from this panel.

## Variable Provenance Summary

All 88 FEWSNET panel columns are assigned exactly once to the source-family groups below: FEWS NET identifiers/geography/outcomes/projections (21 columns), ACLED conflict exposure (19), AEZ dummies (17, including `AEZ_15000`), land/access/terrain/hydrology (8), remote sensing climate/productivity/nightlight (8), soil (5), and macro/prices/population (10). The predictor descriptive group names used for paper summaries are `Conflict Metrics`, `AEZ Dummies`, `Land, Access, and Terrain`, `Climate, Productivity, and Soil`, and `Markets, Macro, and Population`.

| Variable group | Representative columns | Source/provider evidence | License / access note | Manifest treatment |
|---|---|---|---|---|
| FEWS NET geography, outcomes, and projections | `unit_name`, `ADMIN0`-`ADMIN3`, `FEWSNET_admin_code`, `fews_ipc`, `fews_proj_near`, `fews_proj_med`, `fews_ipc_crisis` | FEWSNET source files and admin-boundary folder under `Analysis/1.Source Data/Outcome/FEWSNET_IPC/` | Provider identified; license pending verification | Treated as core third-party FEWS NET-derived data; do not claim public redistribution rights. |
| ACLED conflict exposure | `distance_to_nearest_acled`, `event_count_*`, `sum_fatalities_*` | Source-data `AGENTS.md` maps `ACLED/` to conflict indicators. | Provider identified; license pending verification | Report as derived conflict features, not raw event redistribution. |
| Agroecological, terrain, hydrology, and market access | `AEZ_*`, `crop`, `range`, `distance_to_river`, `elevation`, `ruggedness`, `slope`, `market_access`, `market_distance` | Source-data folders and `variable_construction_notes_description.xlsx`. | Provider identified; license pending verification except locally documented DOI/license entries | Keep provider and local evidence separate from license claims. |
| Remote-sensing climate and productivity | `Rainf_f_tavg_mean`, `Tair_f_tavg_mean`, `Rainf_zscore`, `Tair_zscore`, `EVI`, `gpp_mean`, `nightlight`, `nightlight_sd` | Source-data folders and predictor descriptives | Provider identified; license pending verification | Explain z-scores as derived fields from upstream climate series. |
| Soil, macro, prices, and population | `sg_*`, `CPI`, `GDP`, `CC`, `gini`, `FAO_price`, `WFP_Price`, `WFP_Price_std`, `Food_CPI`, `Food_food_inflation`, `pop` | `ISRIC/`, `FAO/`, `WFP/`, `WBG/`, `Populationdensity/` source folders. | Provider identified; license pending verification | Mark public redistribution as dependent on upstream terms. |

`variable_construction_notes_description.xlsx` provides partial local construction support only: exact-name overlap is 15 of 88 panel columns. Source-data `AGENTS.md` supports source-family and folder provenance, not license rights.

## Derived Reproducibility Assets

| Data asset | Role in reproduction | Local path | Source/provider | License / access notes | Checksum | Rows x columns | Date coverage / spatial coverage | Generated or modified date | Script / command | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| GeoRF Stage 2 cluster mapping manifest | Records the fixed cluster mappings produced by the GeoRF Stage 2 spatial weighted consensus workflow. | `GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json` | Derived from the primary FEWSNET panel and GeoRF Stage 1/2 partition workflow. | Derived from restricted/third-party sources; checksum and provenance manifest are file-integrity evidence. | `05eb5f112deae583ff3ac83f955d347bf3d39d35aa19eac1d15321d9ea45be3c` | Not tabular child manifest | Stage 2 partition maps for downstream fixed-partition evaluation | Recorded in child manifest | `spatial_weighted_consensus_clustering.bat georf` | GeoRF Stage 2 handoff artifact. |
| GeoDT Stage 2 cluster mapping manifest | Records the fixed cluster mappings produced by the GeoDT Stage 2 spatial weighted consensus workflow. | `GeoDTExperiment/knn_sparsification_results/cluster_mapping_manifest.json` | Derived from the primary FEWSNET panel and GeoDT Stage 1/2 partition workflow. | Derived from restricted/third-party sources; checksum and provenance manifest are file-integrity evidence. | `e49be747772609c97faea2903f72f5eadcbf9076e2b53f3314243aa688ff2625` | Not tabular child manifest | Stage 2 partition maps for comparison workflow | Recorded in child manifest | `spatial_weighted_consensus_clustering.bat geodt` | GeoDT Stage 2 handoff artifact. |
| GeoRF Stage 3 result folders fs1-fs3 | Fixed-partition GeoRF evaluation outputs for paper-facing lags. | `result_partition_k40_compare_GF_fs1/run_manifest.json`<br>`result_partition_k40_compare_GF_fs2/run_manifest.json`<br>`result_partition_k40_compare_GF_fs3/run_manifest.json` | Derived from the primary FEWSNET panel and GeoRF Stage 2 cluster mappings. | Derived from restricted/third-party sources; run manifests and checksums are file-integrity evidence. | fs1 `49de1c0d74a25734c885399f27d29048398139d9779771ad0594e15f7e58e23b`<br>fs2 `9a091617ab7f43ce003687f6bb550a9cd983c94d5445418d4a9f268cab7dc771`<br>fs3 `993a297e392b2e43d64c40a0e58b236bb65ec8fb87685ae6e71fcdfb152200d7` | Each run reports `62189` predictions | Stage 3 fixed-partition evaluation; fs1 lag 4, fs2 lag 8, fs3 lag 12; `data_path=FEWSNET_forecast_unadjusted_bm.csv` | fs1 `2026-06-16T16:41:54.119227`<br>fs2 `2026-06-16T16:49:47.840412`<br>fs3 `2026-06-16T16:57:23.527637` | `run_partition_k40_comparison_unified.bat georf --visual --month-ind` | GeoRF Stage 3 is the main paper-facing fixed-partition evaluation. |
| GeoDT Stage 3 result folders fs1-fs3 | Fixed-partition GeoDT comparison outputs for branch/tree analysis. | `result_partition_k40_compare_DT_fs1/run_manifest.json`<br>`result_partition_k40_compare_DT_fs2/run_manifest.json`<br>`result_partition_k40_compare_DT_fs3/run_manifest.json` | Derived from the primary FEWSNET panel and GeoDT Stage 2 cluster mappings. | Derived from restricted/third-party sources; run manifests and checksums are file-integrity evidence. | fs1 `5953daf3268999a9d56fada1ee22bd110c97056bd2ff5766121649844c271bcd`<br>fs2 `b4eda69573bf14402a183659f3690f732228e702be6e53f7d2a6fe859ffb315a`<br>fs3 `9c9bb98a819ad8e112e620bce2307230ce5bfaa909189f7e732eb4128512e6da` | Each run reports `62189` predictions | Stage 3 fixed-partition comparison; fs1 lag 4, fs2 lag 8, fs3 lag 12; `data_path=FEWSNET_forecast_unadjusted_bm.csv` | fs1 `2026-06-15T17:17:50.244256`<br>fs2 `2026-06-15T17:20:15.806829`<br>fs3 `2026-06-15T17:22:42.826073` | `run_partition_k40_comparison_unified.bat geodt --visual --month-ind` | GeoDT Stage 3 is retained as a comparison workflow, not as the main GeoRF paper model. |
| Feature-exclude input datasets | Documents fixed-partition ablation inputs created by dropping source-family feature groups. | `main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json` | Derived from `C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv`. | Derived from restricted/third-party sources; child manifest and dataset checksums are file-integrity evidence. | `272d1964676c889063895dd73b3f096286f7f9e69e92362735af28387f55b28c` | Dataset-specific rows and columns listed below | Same panel coverage as primary FEWSNET panel | `2026-06-15T21:25:09.243118` | Feature-exclude ablation workflow under `main_ablation_exclude_updated_stage3_fixed_partitions/` | Feature-exclude variants are reproducibility assets, not new upstream data sources. |
| Monthly performance final-artifact manifest | Documents the paper-facing monthly performance artifacts in the main results folder. | `final_artifacts_in_paper_updated/01_main_results/monthly_performance_manifest.json` | Derived from Stage 3 GeoRF/GeoDT comparison outputs and FEWSNET baseline summaries. | Derived from restricted/third-party sources; child manifest checksum is file-integrity evidence. | `df4f8783cb7a45c881a2e65fa2d24c89f38e5a920cfb61da04916d5d93f833d1` | Child manifest | Paper main-result coverage | Recorded in child manifest | Paper helper scripts under `scripts/` | Monthly performance artifact group. |
| GeoRF partitioned SHAP final-artifact manifest | Documents partitioned SHAP artifacts used for GeoRF interpretation. | `final_artifacts_in_paper_updated/01_main_results/georf_partitioned_shap_manifest.json` | Derived from GeoRF Stage 3 outputs and model explanation artifacts. | Derived from restricted/third-party sources; child manifest checksum is file-integrity evidence. | `e52a8031cf7be6df1208185f401c5f5d9aed369ea7ac44f98624db33b6687953` | Child manifest | Paper main-result coverage | Recorded in child manifest | Paper helper scripts under `scripts/` | partitioned SHAP artifact group. |
| Cluster-profile final-artifact manifest | Documents cluster-profile artifacts used to describe learned spatial/feature partitions. | `final_artifacts_in_paper_updated/06_cluster_profiles/artifact_source_manifest.json` | Derived from Stage 2 cluster mappings and Stage 3 outputs. | Derived from restricted/third-party sources; child manifest checksum is file-integrity evidence. | `65182feb35612afc246a935e357f50d73f4f67ffe01998bdb1e37b64f0e7456d` | Child manifest | Paper cluster-profile coverage | Recorded in child manifest | Paper helper scripts under `scripts/` | Cluster-profile artifact group. |
| False-negative error-mode final-artifact manifest | Documents false-negative error-mode artifacts used for error analysis. | `final_artifacts_in_paper_updated/10_false_negative_error_modes/artifact_source_manifest.json` | Derived from Stage 3 predictions and outcome comparisons. | Derived from restricted/third-party sources; child manifest checksum is file-integrity evidence. | `5e9cc9025d09037deade8d0081f72e255568ce155284a4679e53a220dcea23de` | Child manifest | Paper false-negative analysis coverage | Recorded in child manifest | Paper helper scripts under `scripts/` | False-negative artifact group. |
| Thresholded GeoRF final-artifact manifest | Documents thresholded GeoRF appendix artifacts and their source manifests. | `final_artifacts_in_paper_updated/12_thresholded_georf_results/artifact_source_manifest.json` | Derived from GeoRF Stage 3 results. | Derived from restricted/third-party sources; child manifest checksum is file-integrity evidence. | `304d4b1c2aef370ee1c8acb358a078000e4b245bdaedb72f3cc0adce3a76ac66` | Child manifest | Paper thresholded GeoRF coverage | Recorded in child manifest | Paper helper scripts under `scripts/` | Thresholded GeoRF artifact group. |
| FEWSNET baseline comparison outputs | Baseline CSVs, main-result comparison table, and derived figure used for paper-facing baseline context. | `fewsnet_baseline_results/fewsnet_baseline_results_fs1.csv`<br>`fewsnet_baseline_results/fewsnet_baseline_results_fs2.csv`<br>`final_artifacts_in_paper_updated/01_main_results/table2_region_performance_partitioned_pooled_fewsnet.csv`<br>`final_artifacts_in_paper_updated/03_class_prevalence/fewsnet_crisis_stack_2018.png` | Derived from FEWS NET projection/outcome fields and paper result builders. | Derived from restricted/third-party sources; file checksums and figure dimensions are file-integrity evidence. | fs1 `4c90887265eee04f8bcccc27659a64084105b4752c526292de91cc45e6e7c975`<br>fs2 `bdc0fc77821dbc8d0868c99d35170b4caaa0406829c878953357fc3d73a63604`<br>table `3aafd0066ab5398bf8683a03c11aa109cd3647f71cb4536dee09f5f03ccb9fbc`<br>derived_figure `f87bbec0b33989e77c0d24cad0e732af01c4763844adefedcb6d85c23fe3d24c` | fs1 `39 x 6`; fs2 `39 x 6`; table `18 x 24`; derived_figure not tabular | Baseline comparison and class-prevalence figure coverage | File metadata captured locally | FEWSNET baseline and paper helper scripts under `scripts/` | `final_artifacts_in_paper_updated/03_class_prevalence/fewsnet_crisis_stack_2018.png` is a `derived_figure` entry: size `167609` bytes, dimensions `3570x1616`. |

Feature-exclude dataset details from the child manifest:

| Dataset | Rows x columns | SHA-256 | Dropped column count |
|---|---:|---|---:|
| `weather_exclude` | `1,029,240 x 84` | `2ee679fd2d2a7087d25ab771ffa65600da32b80301a1c013f7d6a5a3a4a3923f` | 4 |
| `agri_exclude` | `1,029,240 x 66` | `b839f9f8e7434e34d038f699a151e0a3a3e40385cc8eafc3bde3c5a8dc10d67f` | 22 |
| `conflict_exclude` | `1,029,240 x 69` | `716dba1a8ffe1f8de900e775ab7c39e9621cb1230a2f3718e6e9fc2bafb01b9d` | 19 |
| `econ_exclude` | `1,029,240 x 77` | `1ec36ec92bf25241e3f82a77115e6c0aa85d53c54cb8f36d1a0f251c9f627be2` | 11 |
| `food_prices_exclude` | `1,029,240 x 85` | `c6ff3329a265194aac50f5ca5616962ab28db1104043a32b9dd33c5f9fe7732f` | 3 |
| `geographic_exclude` | `1,029,240 x 80` | `cd734af1a88e7d2ec8a4020d88b444937c459ba52e2f07a5582b0aeb8ec68c8d` | 8 |
| `secondary_exclude` | `1,029,240 x 49` | `100e21fad3ad42faef319d368d9e6647ac29672c97c353555133ba5aabf9077c` | 39 |
| `lag_exclude` | `1,029,240 x 88` | `8e0961fbd5a7c17c3a522259ab573a92da51297a3623360a9473692747c9bdc0` | 0 |

## License and Redistribution Notes

Use these manifest status labels consistently:

| Status | Meaning in this manifest | Redistribution implication |
|---|---|---|
| Verified locally | Local documentation or a source file explicitly states license, DOI, citation, or access terms. Checksums, row counts, dimensions, and manifest metadata are file-integrity evidence, not this license/access status. | Use only where explicit local license/access evidence exists; this manifest found limited/no local explicit license evidence except any locally documented DOI/license entries. |
| Provider identified; license pending verification | The likely source/provider can be identified from local source folders, predictor names, or local metadata, but the applicable license/access terms have not been verified in this manifest. | Requires source-license verification before public redistribution or public data release. |
| Derived from restricted/third-party sources | The artifact contains or is derived from data families that may be restricted or governed by upstream provider terms. | Do not claim public redistribution rights until all upstream terms have been reviewed. |

FEWS NET source data, ACLED, WFP, FAO, World Bank-derived indicators, remote-sensing products, soil products, population, nightlight, and market-access inputs should be reviewed against their upstream terms before any public data package is released. Source-data `AGENTS.md` is used here only to support source-family/folder provenance; it is not license evidence.

## Reproduction Commands

Main GeoRF workflow:

```cmd
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
spatial_weighted_consensus_clustering.bat georf
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

GeoDT comparison workflow:

```cmd
run_batches_2018_2020_partition_learning_visual_monthly.bat geodt
spatial_weighted_consensus_clustering.bat geodt
run_partition_k40_comparison_unified.bat geodt --visual --month-ind
```

Verification:

```cmd
python scripts\verify_current_results_reproducibility.py
```

The listed commands assume the source CSV remains available at the recorded Windows/WSL source-data path and that the existing batch-file defaults are unchanged.

## Known Caveats

- Parsed CSV records are reported as `1,029,240 x 88`; physical text lines are `1,029,601`. The line-count difference is preserved as a file-integrity caveat because CSV quoting or embedded newline formatting can make physical text lines differ from parsed records.
- Windows/WSL paths both appear because the workflow is run from WSL while the primary source file lives under the mounted Windows profile path.
- GeoXGB outputs are legacy/experimental and are excluded from this minimum dataset unless a future paper section explicitly promotes them.
- fs0 lag-1 outputs are excluded from this manifest because the current paper-facing no-leak workflow uses fs1/fs2/fs3 unless a future manuscript section adds fs0.
- static/manual diagrams, including online-generated workflow or feature-engineering figures, are excluded unless listed above as a data-derived paper artifact.
- This is a minimum dataset manifest for reproducibility review, not a full source-data archive. Raw upstream-provider files and unrestricted public release packaging require a separate source-license verification pass.
