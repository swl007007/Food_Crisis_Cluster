# FEWSNET Nature Portfolio Data Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reviewer-facing Markdown data manifest for `FEWSNET_forecast_unadjusted_bm.csv` and the current paper reproducibility artifacts.

**Architecture:** Add one Markdown document under the paper methods artifact folder. Gather metadata from existing CSVs, JSON manifests, source-data codebooks, and workflow docs; do not add new production code or rerun model training.

**Tech Stack:** Markdown, bash coreutils (`sha256sum`, `stat`, `wc`), Python 3 standard library (`csv`, `json`, `hashlib`, `pathlib`), existing Food_Crisis_Cluster manifests.

---

## File Structure

- Create: `final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md`
  - Owns the paper-facing Nature Portfolio-style data manifest.
- Read only: `docs/superpowers/specs/2026-06-24-fewsnet-nature-data-manifest-design.md`
  - Approved design and acceptance criteria.
- Read only: `CURRENT_RESULTS_REPRODUCTION.md`, `PIPELINE_WORKFLOW.md`, root batch files, existing `run_manifest.json`, `cluster_mapping_manifest.json`, `artifact_source_manifest.json`, and source-data codebooks.
- Do not modify: model scripts, source CSVs, existing result folders, existing final paper artifacts, and the untracked manuscript PDF.

### Task 1: Reconfirm Primary Dataset Metadata

**Files:**
- Read: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv`
- Read: `docs/superpowers/specs/2026-06-24-fewsnet-nature-data-manifest-design.md`

- [ ] **Step 1: Run a metadata collector for the primary FEWSNET panel**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import csv
import hashlib
from datetime import datetime

path = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv")
sha = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        sha.update(chunk)

with path.open(newline="", encoding="utf-8", errors="replace") as handle:
    reader = csv.DictReader(handle)
    columns = reader.fieldnames or []
    parsed_rows = 0
    dates = []
    iso3 = set()
    admin_codes = set()
    for row in reader:
        parsed_rows += 1
        if row.get("date"):
            dates.append(row["date"])
        if row.get("ISO3"):
            iso3.add(row["ISO3"])
        if row.get("FEWSNET_admin_code"):
            admin_codes.add(row["FEWSNET_admin_code"])

physical_lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ")

print(f"sha256={sha.hexdigest()}")
print(f"size_bytes={path.stat().st_size}")
print(f"parsed_rows={parsed_rows}")
print(f"columns={len(columns)}")
print(f"physical_lines={physical_lines}")
print(f"date_coverage={min(dates)} to {max(dates)}")
print(f"iso3_count={len(iso3)}")
print(f"admin_code_count={len(admin_codes)}")
print(f"mtime={mtime}")
print("iso3=" + ",".join(sorted(iso3)))
PY
```

Expected output contains:

```text
sha256=611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651
size_bytes=716303754
parsed_rows=1029240
columns=88
physical_lines=1029601
date_coverage=2010-01 to 2024-12
iso3_count=22
admin_code_count=5718
```

- [ ] **Step 2: Compare the output with the approved spec**

Run:

```bash
rg -n "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651|1,029,240|88|1,029,601|2010-01|2024-12|5,718" docs/superpowers/specs/2026-06-24-fewsnet-nature-data-manifest-design.md
```

Expected: matches for the approved primary dataset metadata. If the live metadata differs, stop and explain the exact mismatch before writing the manifest.

### Task 2: Gather Existing Derived-Asset Evidence

**Files:**
- Read: `GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`
- Read: `GeoDTExperiment/knn_sparsification_results/cluster_mapping_manifest.json`
- Read: `result_partition_k40_compare_GF_fs1/run_manifest.json`
- Read: `result_partition_k40_compare_GF_fs2/run_manifest.json`
- Read: `result_partition_k40_compare_GF_fs3/run_manifest.json`
- Read: `result_partition_k40_compare_DT_fs1/run_manifest.json`
- Read: `result_partition_k40_compare_DT_fs2/run_manifest.json`
- Read: `result_partition_k40_compare_DT_fs3/run_manifest.json`
- Read: `main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json`
- Read: `final_artifacts_in_paper_updated/01_main_results/monthly_performance_manifest.json`
- Read: `final_artifacts_in_paper_updated/01_main_results/georf_partitioned_shap_manifest.json`
- Read: `final_artifacts_in_paper_updated/06_cluster_profiles/artifact_source_manifest.json`
- Read: `final_artifacts_in_paper_updated/10_false_negative_error_modes/artifact_source_manifest.json`
- Read: `final_artifacts_in_paper_updated/12_thresholded_georf_results/artifact_source_manifest.json`

- [ ] **Step 1: Print checksums for existing child manifests**

Run:

```bash
sha256sum \
  GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json \
  GeoDTExperiment/knn_sparsification_results/cluster_mapping_manifest.json \
  result_partition_k40_compare_GF_fs1/run_manifest.json \
  result_partition_k40_compare_GF_fs2/run_manifest.json \
  result_partition_k40_compare_GF_fs3/run_manifest.json \
  result_partition_k40_compare_DT_fs1/run_manifest.json \
  result_partition_k40_compare_DT_fs2/run_manifest.json \
  result_partition_k40_compare_DT_fs3/run_manifest.json \
  main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json \
  final_artifacts_in_paper_updated/01_main_results/monthly_performance_manifest.json \
  final_artifacts_in_paper_updated/01_main_results/georf_partitioned_shap_manifest.json \
  final_artifacts_in_paper_updated/06_cluster_profiles/artifact_source_manifest.json \
  final_artifacts_in_paper_updated/10_false_negative_error_modes/artifact_source_manifest.json \
  final_artifacts_in_paper_updated/12_thresholded_georf_results/artifact_source_manifest.json
```

Expected: one SHA-256 line per listed manifest, exit code 0.

- [ ] **Step 2: Print compact run-manifest summaries**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import json

paths = sorted(Path(".").glob("result_partition_k40_compare_*_fs*/run_manifest.json"))
for path in paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    if path.parent.name.startswith(("result_partition_k40_compare_GF_fs", "result_partition_k40_compare_DT_fs")):
        print(
            f"{path}: model_type={data.get('model_type')} "
            f"scope=fs{data.get('forecasting_scope')} "
            f"lag={data.get('active_lag_months')} "
            f"timestamp={data.get('timestamp')} "
            f"predictions={data.get('n_predictions')} "
            f"data_path={data.get('data_path')}"
        )
PY
```

Expected: six current main Stage 3 rows for GeoRF and GeoDT fs1-fs3, each using `FEWSNET_forecast_unadjusted_bm.csv`.

- [ ] **Step 3: Print feature-exclude dataset summaries**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import csv
import hashlib
import json

manifest_path = Path("main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/feature_exclude_dataset_manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"manifest_timestamp={manifest.get('timestamp')}")
print(f"source={manifest.get('source')}")
for name, entry in manifest["datasets"].items():
    path_text = entry["path"].replace("\\", "/")
    if path_text.startswith("C:/"):
        repo_rel = "main_ablation_exclude_updated_stage3_fixed_partitions/input_datasets/" + Path(path_text).name
        path = Path(repo_rel)
    else:
        path = Path(path_text)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    print(f"{name}: rows={rows} cols={len(header)} sha256={sha} dropped={','.join(entry.get('dropped_columns', []))}")
PY
```

Expected: one line for each feature-exclude dataset. `lag_exclude` reports `88` columns and no dropped columns.

### Task 3: Build Variable Provenance Summary

**Files:**
- Read: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm_predictor_descriptives.md`
- Read: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/variable_construction_notes_description.xlsx`
- Read: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/assembled_IPCCH/metadata/variable_codebook_reorganized.csv`
- Read: `/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/AGENTS.md`

- [ ] **Step 1: Print same-variable overlap with IPCCH codebook**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import csv
from collections import Counter

target = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/FEWSNET_forecast_unadjusted_bm.csv")
codebook = Path("/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/assembled_IPCCH/metadata/variable_codebook_reorganized.csv")

target_columns = next(csv.reader(target.open(newline="", encoding="utf-8", errors="replace")))
with codebook.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
    rows = {row["variable"]: row for row in csv.DictReader(handle) if row.get("variable")}

overlap = [col for col in target_columns if col in rows]
counts = Counter(rows[col]["category"] for col in overlap)
print(f"target_columns={len(target_columns)}")
print(f"codebook_variables={len(rows)}")
print(f"overlap={len(overlap)}")
for category, count in counts.most_common():
    print(f"{category}: {count}")
missing = [col for col in target_columns if col not in rows]
print("not_in_ipcch_codebook=" + ",".join(missing))
PY
```

Expected output includes `target_columns=88` and `overlap=63`. Use the category counts to support the manifest's variable provenance summary, and list the non-overlap FEWSNET-specific fields in the caveats or notes.

- [ ] **Step 2: Confirm source-data family descriptions**

Run:

```bash
sed -n '1,40p' "/mnt/c/Users/swl00/IFPRI Dropbox/Weilun Shi/Google fund/Analysis/1.Source Data/AGENTS.md"
```

Expected: table rows describing `ACLED/`, `FAO/`, `WFP/`, `WBG/`, remote-sensing, terrain, population, nightlight, market-access, and `ISRIC/` source folders.

### Task 4: Write the Markdown Manifest

**Files:**
- Create: `final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md`

- [ ] **Step 1: Create the manifest with the approved section structure**

Use `apply_patch` to add `final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md`.

The document must use this section order:

```markdown
# FEWSNET Data Manifest for Nature Portfolio Reproducibility Review

## Purpose and Standard

## Primary Assembled Dataset

## Variable Provenance Summary

## Derived Reproducibility Assets

## License and Redistribution Notes

## Reproduction Commands

## Known Caveats
```

- [ ] **Step 2: Add the primary dataset table**

Add a Markdown table with these columns:

```markdown
| Data asset | Role in reproduction | Local path | Source/provider | License / access notes | Checksum | Rows x columns | Date coverage / spatial coverage | Generated or modified date | Script / command | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
```

The first row must describe `FEWSNET_forecast_unadjusted_bm.csv` and include:

```text
611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651
1,029,240 x 88
2010-01 to 2024-12; 22 ISO3 countries; 5,718 FEWSNET admin codes
2025-11-05 09:41:30 -0500
```

State in the notes cell that physical text lines are `1,029,601` and parsed CSV records are the reported row count.

- [ ] **Step 3: Add variable provenance grouped by source family**

Create a table with these rows:

```markdown
| Variable group | Representative columns | Source/provider evidence | License / access note | Manifest treatment |
|---|---|---|---|---|
| FEWS NET geography, outcomes, and projections | `unit_name`, `ADMIN0`-`ADMIN3`, `FEWSNET_admin_code`, `fews_ipc`, `fews_proj_near`, `fews_proj_med`, `fews_ipc_crisis` | FEWSNET source files and admin-boundary folder under `Analysis/1.Source Data/Outcome/FEWSNET_IPC/` | Provider identified; license pending verification | Treated as core third-party FEWS NET-derived data; do not claim public redistribution rights. |
| ACLED conflict exposure | `distance_to_nearest_acled`, `event_count_*`, `sum_fatalities_*` | Source-data `AGENTS.md` maps `ACLED/` to conflict indicators; IPCCH codebook overlaps conflict categories | Provider identified; license pending verification | Report as derived conflict features, not raw event redistribution. |
| Agroecological, terrain, hydrology, and market access | `AEZ_*`, `crop`, `range`, `distance_to_river`, `elevation`, `ruggedness`, `slope`, `market_access`, `market_distance` | Source-data folders and `variable_construction_notes_description.xlsx`; IPCCH codebook overlap | Provider identified; license pending verification except locally documented DOI/license entries | Keep provider and local evidence separate from license claims. |
| Remote-sensing climate and productivity | `Rainf_f_tavg_mean`, `Tair_f_tavg_mean`, `Rainf_zscore`, `Tair_zscore`, `EVI`, `gpp_mean`, `nightlight`, `nightlight_sd` | Source-data folders and predictor descriptives | Provider identified; license pending verification | Explain z-scores as derived fields from upstream climate series. |
| Soil, macro, prices, and population | `sg_*`, `CPI`, `GDP`, `CC`, `gini`, `FAO_price`, `WFP_Price`, `WFP_Price_std`, `Food_CPI`, `Food_food_inflation`, `pop` | `ISRIC/`, `FAO/`, `WFP/`, `WBG/`, `Populationdensity/` source folders and IPCCH codebook overlap | Provider identified; license pending verification | Mark public redistribution as dependent on upstream terms. |
```

Add a sentence immediately below the table: `The IPCCH codebook is used only as a same-variable cross-reference; IPCCH-only variables are not imported into this FEWSNET manifest.`

- [ ] **Step 4: Add derived reproducibility asset rows**

Add rows for these asset groups in the derived-assets table:

```text
GeoRF Stage 2 cluster mapping manifest
GeoDT Stage 2 cluster mapping manifest
GeoRF Stage 3 result folders fs1-fs3
GeoDT Stage 3 result folders fs1-fs3
Feature-exclude input datasets
Monthly performance final-artifact manifest
GeoRF partitioned SHAP final-artifact manifest
Cluster-profile final-artifact manifest
False-negative error-mode final-artifact manifest
Thresholded GeoRF final-artifact manifest
FEWSNET baseline comparison outputs
```

For each row, include the child manifest path and SHA-256 captured in Task 2. For Stage 3 rows, include `run_manifest.json` timestamps, model family, scope/lag, and prediction counts from Task 2.

- [ ] **Step 5: Add reproduction commands**

Include these commands exactly:

```cmd
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
spatial_weighted_consensus_clustering.bat georf
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

Also include the GeoDT equivalent only as a comparison workflow:

```cmd
run_batches_2018_2020_partition_learning_visual_monthly.bat geodt
spatial_weighted_consensus_clustering.bat geodt
run_partition_k40_comparison_unified.bat geodt --visual --month-ind
```

Include the verification command:

```cmd
python scripts\verify_current_results_reproducibility.py
```

### Task 5: Validate the Manifest

**Files:**
- Validate: `final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md`

- [ ] **Step 1: Check required primary metadata appears**

Run:

```bash
rg -n "611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651|1,029,240 x 88|1,029,601|2010-01 to 2024-12|5,718 FEWSNET admin codes" final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md
```

Expected: all required primary metadata strings are found.

- [ ] **Step 2: Check required license statuses appear**

Run:

```bash
rg -n "Verified locally|Provider identified; license pending verification|Derived from restricted/third-party sources|do not claim public redistribution rights|source-license verification" final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md
```

Expected: each license-status category appears at least once.

- [ ] **Step 3: Check required artifact groups appear**

Run:

```bash
rg -n "GeoRF Stage 2|GeoDT Stage 2|GeoRF Stage 3|GeoDT Stage 3|Feature-exclude|Monthly performance|partitioned SHAP|Cluster-profile|False-negative|Thresholded GeoRF|FEWSNET baseline" final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md
```

Expected: each derived asset group appears.

- [ ] **Step 4: Check caveats appear**

Run:

```bash
rg -n "physical text lines|Windows/WSL|GeoXGB|fs0|static/manual|minimum dataset" final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md
```

Expected: caveats for line counts, paths, legacy GeoXGB, fs0, static figures, and minimum-dataset scope are present.

- [ ] **Step 5: Run Markdown whitespace validation**

Run:

```bash
git diff --check -- final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md
```

Expected: no output and exit code 0.

### Task 6: Commit the Manifest

**Files:**
- Commit: `final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md`

- [ ] **Step 1: Review git status**

Run:

```bash
git status --short
```

Expected: the new manifest is listed. The existing untracked manuscript PDF may also be listed and must not be staged unless the user explicitly requests it.

- [ ] **Step 2: Stage only the new manifest**

Run:

```bash
git add final_artifacts_in_paper_updated/02_methods_and_temporal_scope/fewsnet_data_manifest_nature_portfolio.md
```

Expected: command exits successfully.

- [ ] **Step 3: Commit the manifest**

Run:

```bash
git commit -m "add fewsnet data manifest"
```

Expected: commit succeeds and reports one new Markdown file.
