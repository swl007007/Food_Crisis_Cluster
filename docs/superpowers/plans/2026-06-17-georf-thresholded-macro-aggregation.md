# GeoRF Thresholded Macro Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute `final_artifacts_in_paper_updated/12_thresholded_georf_results` so its horizon-level paper-facing metrics use monthly macro means, matching `01_main_results`.

**Architecture:** Keep the thresholded GeoRF provider outputs unchanged. Change only the artifact builder's horizon aggregation from pooled polygon-month confusion counts to the mean of monthly precision, recall, and F1, while retaining summed TP/FP/FN/TN and support as diagnostic count fields.

**Tech Stack:** Python 3.12, pandas, pytest, existing GeoRF thresholded artifact builder.

---

### Task 1: Lock Horizon Aggregation To Monthly Macro Mean

**Files:**
- Modify: `src/tests/test_georf_thresholded_artifacts.py`
- Modify: `scripts/build_georf_thresholded_artifacts.py`

- [ ] **Step 1: Add a failing regression test**

Add a test that passes two monthly rows for one model where the micro F1 differs from the monthly mean F1. The expected horizon precision, recall, and F1 must equal the monthly means.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' -m pytest src/tests/test_georf_thresholded_artifacts.py -q
```

Expected: the new test fails because `build_horizon_metrics()` currently recomputes precision, recall, and F1 after summing TP/FP/FN.

- [ ] **Step 3: Update the artifact builder**

Change `_aggregate_model_metrics()` so `precision`, `recall`, and `f1` are `group["precision"].mean()`, `group["recall"].mean()`, and `group["f1"].mean()`. Keep `support`, `tp`, `fp`, `fn`, and `tn` as sums so diagnostics remain available.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' -m pytest src/tests/test_georf_thresholded_artifacts.py -q
```

Expected: all tests in `test_georf_thresholded_artifacts.py` pass.

- [ ] **Step 5: Commit the code change**

Run:

```bash
git add scripts/build_georf_thresholded_artifacts.py src/tests/test_georf_thresholded_artifacts.py
git commit -m "use macro aggregation for thresholded GeoRF"
```

### Task 2: Regenerate Artifact Group 12 And Verify Alignment

**Files:**
- Regenerate: `final_artifacts_in_paper_updated/12_thresholded_georf_results/*`
- Potentially modify: `final_artifacts_in_paper_updated/artifact_source_audit.csv`

- [ ] **Step 1: Rebuild group 12**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/build_georf_thresholded_artifacts.py
```

Expected: `georf_thresholded_compact_table.csv` and `georf_thresholded_horizon_metrics.csv` now report unthresholded pooled and partitioned values that match `01_main_results` for each forecasting horizon.

- [ ] **Step 2: Run regression and artifact checks**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' -m pytest src/tests/test_georf_stage3_manifest_provenance.py src/tests/test_georf_thresholded_artifacts.py src/tests/test_georf_thresholded_monthind_alignment.py src/tests/test_artifact_source_audit.py -q
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/verify_current_results_reproducibility.py
rg --no-ignore -n "FEWSNET_forecast_unadjusted_bm_phase_change|phase_change" final_artifacts_in_paper_updated result_partition_k40_compare_GF_thresholded_fs1 result_partition_k40_compare_GF_thresholded_fs2 result_partition_k40_compare_GF_thresholded_fs3
```

Expected: pytest and reproducibility verifier pass; the `rg` command exits with no matches.

- [ ] **Step 3: Commit regenerated artifacts**

Run:

```bash
git add final_artifacts_in_paper_updated/12_thresholded_georf_results final_artifacts_in_paper_updated/artifact_source_audit.csv final_artifacts_in_paper_updated/artifact_source_audit.md
git commit -m "rebuild thresholded GeoRF macro artifacts"
```
