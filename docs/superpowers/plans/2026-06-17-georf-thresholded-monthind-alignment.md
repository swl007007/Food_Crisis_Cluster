# GeoRF Thresholded Month-Specific Partition Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate and document `12_thresholded_georf_results` so thresholded GeoRF uses the same month-specific partition assignment regime as `01_main_results`.

**Architecture:** Keep Stage 3 model logic unchanged, but add provenance fields to the provider manifests and artifact manifest. Add a regression test that proves thresholded predictions use the refined `m2/m6/m10` maps rather than the general map for all months, then rerun the three thresholded GeoRF providers with `--month-ind`.

**Tech Stack:** Python 3.12, pandas, scikit-learn RF, imbalanced-learn SMOTE, pytest/unittest, existing GeoRF Stage 3 scripts.

---

## File Structure

- Modify `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
  - Responsibility: Stage 3 partitioned-vs-pooled provider generation.
  - Add manifest provenance helpers for partition map hashes, month-specific map paths, Python executable, and SMOTE availability.
- Modify `scripts/build_georf_thresholded_artifacts.py`
  - Responsibility: paper-facing summary builder for artifact group 12.
  - Add provider manifest details to `artifact_source_manifest.json`.
- Modify `src/tests/test_georf_thresholded_artifacts.py`
  - Responsibility: unit tests for the artifact builder.
  - Add coverage for provider manifest detail extraction.
- Create `src/tests/test_georf_thresholded_monthind_alignment.py`
  - Responsibility: regression test against live regenerated provider outputs.
  - Assert 12 thresholded predictions match the expected month-specific partition maps.
- Regenerate `result_partition_k40_compare_GF_thresholded_fs1`, `fs2`, and `fs3`.
- Regenerate `final_artifacts_in_paper_updated/12_thresholded_georf_results`.
- Regenerate `final_artifacts_in_paper_updated/artifact_source_audit.csv` and `.md` through the reproducibility verifier.

---

### Task 1: Add Provider Manifest Provenance Helpers

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- Test: `src/tests/test_georf_stage3_manifest_provenance.py`

- [ ] **Step 1: Write the failing unit tests**

Create `src/tests/test_georf_stage3_manifest_provenance.py`:

```python
import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py"
spec = importlib.util.spec_from_file_location("compare_partitioned_vs_pooled_rf_k40_nc4", SCRIPT_PATH)
stage3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3)


def test_file_sha256_returns_stable_digest(tmp_path: Path):
    sample = tmp_path / "sample.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")

    digest = stage3.file_sha256(sample)

    assert len(digest) == 64
    assert digest == stage3.file_sha256(sample)


def test_partition_map_provenance_records_month_specific_maps(tmp_path: Path):
    general = tmp_path / "general.csv"
    m2 = tmp_path / "m2.csv"
    m6 = tmp_path / "m6.csv"
    m10 = tmp_path / "m10.csv"
    for path in (general, m2, m6, m10):
        path.write_text("FEWSNET_admin_code,cluster_id\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        month_ind=True,
        partition_map=str(general),
        partition_map_m2=str(m2),
        partition_map_m6=str(m6),
        partition_map_m10=str(m10),
    )

    provenance = stage3.partition_map_provenance(args)

    assert provenance["month_ind_enabled"] is True
    assert provenance["partition_map_path"] == str(general)
    assert provenance["partition_map_m2_path"] == str(m2)
    assert provenance["partition_map_m6_path"] == str(m6)
    assert provenance["partition_map_m10_path"] == str(m10)
    assert len(provenance["partition_map_hashes"]["general"]) == 64
    assert len(provenance["partition_map_hashes"]["m2"]) == 64
    assert len(provenance["partition_map_hashes"]["m6"]) == 64
    assert len(provenance["partition_map_hashes"]["m10"]) == 64


def test_runtime_provenance_records_python_and_smote_fields():
    provenance = stage3.runtime_provenance()

    assert "python_executable" in provenance
    assert "python_version" in provenance
    assert "smote_available" in provenance
    assert "imblearn_version" in provenance
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
python3 -m pytest src/tests/test_georf_stage3_manifest_provenance.py -q
```

Expected: FAIL because `file_sha256`, `partition_map_provenance`, and `runtime_provenance` do not exist yet.

- [ ] **Step 3: Implement the helper functions**

In `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`, add `import hashlib` near the existing imports:

```python
import hashlib
```

Add these helpers after `_apply_partition_smote(...)`:

```python
def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for a file, or an empty string if it is missing."""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance() -> Dict[str, Any]:
    """Return runtime provenance needed to audit SMOTE and Python consistency."""
    if SMOTE is None:
        smote_available = False
        imblearn_version = None
    else:
        smote_available = True
        try:
            import imblearn

            imblearn_version = getattr(imblearn, "__version__", None)
        except Exception:
            imblearn_version = "unknown"

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "smote_available": smote_available,
        "imblearn_version": imblearn_version,
    }


def partition_map_provenance(args: argparse.Namespace) -> Dict[str, Any]:
    """Return partition-map provenance for general and month-specific modes."""
    map_paths = {
        "general": str(args.partition_map),
        "m2": str(args.partition_map_m2) if args.month_ind else None,
        "m6": str(args.partition_map_m6) if args.month_ind else None,
        "m10": str(args.partition_map_m10) if args.month_ind else None,
    }
    return {
        "month_ind_enabled": bool(args.month_ind),
        "partition_map_path": str(args.partition_map),
        "partition_map_m2_path": map_paths["m2"],
        "partition_map_m6_path": map_paths["m6"],
        "partition_map_m10_path": map_paths["m10"],
        "partition_map_hashes": {
            key: file_sha256(value) if value else ""
            for key, value in map_paths.items()
        },
    }
```

- [ ] **Step 4: Run the unit tests and verify they pass**

Run:

```bash
python3 -m pytest src/tests/test_georf_stage3_manifest_provenance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py src/tests/test_georf_stage3_manifest_provenance.py
git commit -m "record GeoRF stage3 provenance"
```

---

### Task 2: Write Provenance Into Stage 3 Run Manifests

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- Test: `src/tests/test_georf_stage3_manifest_provenance.py`

- [ ] **Step 1: Extend the failing unit test**

Append this test to `src/tests/test_georf_stage3_manifest_provenance.py`:

```python
def test_build_run_manifest_merges_runtime_and_partition_provenance(tmp_path: Path):
    general = tmp_path / "general.csv"
    m2 = tmp_path / "m2.csv"
    m6 = tmp_path / "m6.csv"
    m10 = tmp_path / "m10.csv"
    for path in (general, m2, m6, m10):
        path.write_text("FEWSNET_admin_code,cluster_id\n1,2\n", encoding="utf-8")

    args = argparse.Namespace(
        data="clean.csv",
        partition_map=str(general),
        partition_map_m2=str(m2),
        partition_map_m6=str(m6),
        partition_map_m10=str(m10),
        start_month="2021-01",
        end_month="2024-12",
        train_window=36,
        forecasting_scope=1,
        lower_model="rf",
        visual=False,
        month_ind=True,
        enable_validation_threshold=True,
        threshold_validation_months=6,
        threshold_lower_bound=0.05,
        threshold_upper_bound=0.95,
    )

    manifest = stage3.build_run_manifest(
        args=args,
        active_lag=4,
        test_months=[1, 2, 3],
        metrics_month_count=12,
        predictions_count=62189,
        n_polygons=5713,
        n_partitions=17,
        model_label="RF",
    )

    assert manifest["month_ind_enabled"] is True
    assert manifest["partition_map_m2_path"] == str(m2)
    assert len(manifest["partition_map_hashes"]["m10"]) == 64
    assert manifest["smote_available"] in {True, False}
    assert manifest["validation_threshold_enabled"] is True
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m pytest src/tests/test_georf_stage3_manifest_provenance.py::test_build_run_manifest_merges_runtime_and_partition_provenance -q
```

Expected: FAIL because `build_run_manifest` does not exist yet.

- [ ] **Step 3: Implement `build_run_manifest` and use it**

In `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`, add this helper before `main()`:

```python
def build_run_manifest(
    *,
    args: argparse.Namespace,
    active_lag: int,
    test_months: List[pd.Period],
    metrics_month_count: int,
    predictions_count: int,
    n_polygons: int,
    n_partitions: int,
    model_label: str,
) -> Dict[str, Any]:
    """Build the provider run manifest with explicit partition and runtime provenance."""
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "train_window_months": args.train_window,
        "forecasting_scope": args.forecasting_scope,
        "active_lag_months": active_lag,
        "n_test_months": len(test_months),
        "n_test_months_evaluated": int(metrics_month_count),
        "n_predictions": int(predictions_count),
        "n_polygons": int(n_polygons),
        "n_partitions": int(n_partitions),
        "rf_params": RF_PARAMS if args.lower_model == "rf" else DT_PARAMS,
        "model_type": model_label,
        "random_state": RANDOM_STATE,
        "visual_enabled": args.visual,
        "pipeline_version": "GeoRF_utilities_v1.0",
        "partition_learning_years": os.environ.get("NO_LEAK_PARTITION_LEARNING_YEARS", "2018-2020"),
        "evaluation_years": os.environ.get("NO_LEAK_EVALUATION_YEARS", "2021-2024"),
        "temporal_leakage_guard": "partitions learned before evaluation window",
        "main_model_scope": "GeoRF/GeoDT only; experimental XGBoost variant not included",
        "validation_threshold_enabled": bool(args.enable_validation_threshold),
        "threshold_selection_metric": "class_1_f1" if args.enable_validation_threshold else None,
        "threshold_validation_months": args.threshold_validation_months if args.enable_validation_threshold else None,
        "threshold_candidate_bounds": [args.threshold_lower_bound, args.threshold_upper_bound]
        if args.enable_validation_threshold
        else None,
        "threshold_default": DEFAULT_PARTITIONED_THRESHOLD if args.enable_validation_threshold else None,
    }
    manifest.update(partition_map_provenance(args))
    manifest.update(runtime_provenance())
    return manifest
```

Replace the existing inline `manifest = { ... }` block near the end of `main()` with:

```python
    manifest = build_run_manifest(
        args=args,
        active_lag=active_lag,
        test_months=test_months,
        metrics_month_count=metrics_df["test_month"].nunique() if not metrics_df.empty else 0,
        predictions_count=len(predictions_df) if not predictions_df.empty else 0,
        n_polygons=df["FEWSNET_admin_code"].nunique(),
        n_partitions=partition_df["cluster_id"].nunique(),
        model_label=model_label,
    )
```

- [ ] **Step 4: Run provenance tests**

Run:

```bash
python3 -m pytest src/tests/test_georf_stage3_manifest_provenance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py src/tests/test_georf_stage3_manifest_provenance.py
git commit -m "write GeoRF stage3 partition provenance"
```

---

### Task 3: Add Thresholded Artifact Manifest Details

**Files:**
- Modify: `scripts/build_georf_thresholded_artifacts.py`
- Modify: `src/tests/test_georf_thresholded_artifacts.py`

- [ ] **Step 1: Write failing builder tests**

Append these tests to `src/tests/test_georf_thresholded_artifacts.py`:

```python
import json


def test_load_provider_manifests_keeps_partition_and_runtime_details(tmp_path: Path):
    provider = tmp_path / "result_partition_k40_compare_GF_thresholded_fs1"
    provider.mkdir()
    manifest = {
        "data_path": r"C:\data\FEWSNET_forecast_unadjusted_bm.csv",
        "month_ind_enabled": True,
        "partition_map_path": "general.csv",
        "partition_map_m2_path": "m2.csv",
        "partition_map_m6_path": "m6.csv",
        "partition_map_m10_path": "m10.csv",
        "partition_map_hashes": {"general": "a" * 64, "m2": "b" * 64, "m6": "c" * 64, "m10": "d" * 64},
        "smote_available": True,
        "imblearn_version": "0.14.1",
        "python_executable": "python3.12.exe",
    }
    (provider / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    details = builder.load_provider_manifests(tmp_path, ["fs1"])

    assert details["fs1"]["month_ind_enabled"] is True
    assert details["fs1"]["partition_map_m2_path"] == "m2.csv"
    assert details["fs1"]["smote_available"] is True
    assert details["fs1"]["partition_map_hashes"]["m10"] == "d" * 64
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m pytest src/tests/test_georf_thresholded_artifacts.py::test_load_provider_manifests_keeps_partition_and_runtime_details -q
```

Expected: FAIL because `load_provider_manifests` does not exist.

- [ ] **Step 3: Implement provider manifest extraction**

In `scripts/build_georf_thresholded_artifacts.py`, replace `load_provider_sources(...)` with:

```python
def load_provider_manifests(source_dir: Path, scopes: list[str]) -> dict[str, dict]:
    """Return selected provenance fields from thresholded provider manifests."""
    details: dict[str, dict] = {}
    for scope in scopes:
        result_dir = source_dir / f"result_partition_k40_compare_GF_thresholded_{scope}"
        manifest_path = result_dir / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        details[scope] = {
            "data_path": str(manifest.get("data_path", "")),
            "month_ind_enabled": bool(manifest.get("month_ind_enabled", False)),
            "partition_map_path": manifest.get("partition_map_path"),
            "partition_map_m2_path": manifest.get("partition_map_m2_path"),
            "partition_map_m6_path": manifest.get("partition_map_m6_path"),
            "partition_map_m10_path": manifest.get("partition_map_m10_path"),
            "partition_map_hashes": manifest.get("partition_map_hashes", {}),
            "smote_available": manifest.get("smote_available"),
            "imblearn_version": manifest.get("imblearn_version"),
            "python_executable": manifest.get("python_executable"),
        }
    return details
```

Update `main(...)` so it writes both source paths and provider details:

```python
    provider_details = load_provider_manifests(args.source_dir, args.scopes)
    provider_sources = {
        scope: str(details.get("data_path", ""))
        for scope, details in provider_details.items()
    }
```

In the JSON payload for `artifact_source_manifest.json`, add:

```python
                "provider_details": provider_details,
```

- [ ] **Step 4: Run builder tests**

Run:

```bash
python3 -m pytest src/tests/test_georf_thresholded_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/build_georf_thresholded_artifacts.py src/tests/test_georf_thresholded_artifacts.py
git commit -m "record thresholded GeoRF provider details"
```

---

### Task 4: Add Live Month-Specific Alignment Regression Test

**Files:**
- Create: `src/tests/test_georf_thresholded_monthind_alignment.py`

- [ ] **Step 1: Write the failing regression test**

Create `src/tests/test_georf_thresholded_monthind_alignment.py`:

```python
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("fs1", "fs2", "fs3")
MONTH_MAPS = {
    2: "cluster_mapping_k40_nc13_m2_refined_contig3.csv",
    6: "cluster_mapping_k40_nc11_m6_refined_contig3.csv",
    10: "cluster_mapping_k40_nc16_m10_refined_contig3.csv",
}


def _load_map(scope: str, month: int) -> pd.Series:
    path = REPO_ROOT / f"result_partition_k40_compare_GF_{scope}" / "refined" / MONTH_MAPS[month]
    mapping = pd.read_csv(path)
    return mapping.set_index("FEWSNET_admin_code")["cluster_id"].astype(int)


def _prediction_admin_month_rows(scope: str, provider: str) -> pd.DataFrame:
    path = REPO_ROOT / provider / "predictions_monthly.csv"
    df = pd.read_csv(path, usecols=["FEWSNET_admin_code", "month_start", "partition_id"])
    df["month_start"] = pd.to_datetime(df["month_start"])
    return df.drop_duplicates(["FEWSNET_admin_code", "month_start"]).copy()


def test_thresholded_georf_predictions_use_month_specific_partitions():
    for scope in SCOPES:
        predictions = _prediction_admin_month_rows(scope, f"result_partition_k40_compare_GF_thresholded_{scope}")
        for month, _filename in MONTH_MAPS.items():
            month_rows = predictions[predictions["month_start"].dt.month == month].copy()
            expected = _load_map(scope, month)
            month_rows["expected_partition_id"] = month_rows["FEWSNET_admin_code"].map(expected)
            covered = month_rows["expected_partition_id"].notna()

            assert covered.any(), f"{scope} month {month} has no covered polygons"
            mismatches = (
                month_rows.loc[covered, "partition_id"].astype(int)
                != month_rows.loc[covered, "expected_partition_id"].astype(int)
            )
            assert not mismatches.any(), f"{scope} month {month} has {int(mismatches.sum())} partition mismatches"


def test_thresholded_georf_manifest_records_month_ind_and_runtime_provenance():
    for scope in SCOPES:
        manifest_path = REPO_ROOT / f"result_partition_k40_compare_GF_thresholded_{scope}" / "run_manifest.json"
        manifest = pd.read_json(manifest_path, typ="series").to_dict()

        assert manifest["month_ind_enabled"] is True
        assert manifest["partition_map_m2_path"].endswith("cluster_mapping_k40_nc13_m2_refined_contig3.csv")
        assert manifest["partition_map_m6_path"].endswith("cluster_mapping_k40_nc11_m6_refined_contig3.csv")
        assert manifest["partition_map_m10_path"].endswith("cluster_mapping_k40_nc16_m10_refined_contig3.csv")
        assert len(manifest["partition_map_hashes"]["m2"]) == 64
        assert "python_executable" in manifest
        assert "smote_available" in manifest
```

- [ ] **Step 2: Run the regression test and verify it fails before rerun**

Run:

```bash
python3 -m pytest src/tests/test_georf_thresholded_monthind_alignment.py -q
```

Expected: FAIL before regeneration because current thresholded providers use the general partition map and manifests do not yet record `month_ind_enabled`.

- [ ] **Step 3: Commit the failing regression test with implementation fixes already present**

Do not commit while the full test suite is failing because artifacts still need regeneration. Keep this file staged for the regeneration task.

---

### Task 5: Rerun Thresholded GeoRF Providers With `--month-ind`

**Files:**
- Regenerate: `result_partition_k40_compare_GF_thresholded_fs1/*`
- Regenerate: `result_partition_k40_compare_GF_thresholded_fs2/*`
- Regenerate: `result_partition_k40_compare_GF_thresholded_fs3/*`

- [ ] **Step 1: Run fs1 thresholded provider**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --data 'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv' \
  --partition-map result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --partition-map-m2 result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc13_m2_refined_contig3.csv \
  --partition-map-m6 result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc11_m6_refined_contig3.csv \
  --partition-map-m10 result_partition_k40_compare_GF_fs1/refined/cluster_mapping_k40_nc16_m10_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs1 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --train-window 36 \
  --forecasting-scope 1 \
  --month-ind \
  --enable-validation-threshold
```

Expected: command exits 0 and writes `metrics_monthly.csv`, `predictions_monthly.csv`, `metrics_polygon_overall.csv`, `threshold_provenance.csv`, and `run_manifest.json`.

- [ ] **Step 2: Run fs2 thresholded provider**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --data 'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv' \
  --partition-map result_partition_k40_compare_GF_fs2/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --partition-map-m2 result_partition_k40_compare_GF_fs2/refined/cluster_mapping_k40_nc13_m2_refined_contig3.csv \
  --partition-map-m6 result_partition_k40_compare_GF_fs2/refined/cluster_mapping_k40_nc11_m6_refined_contig3.csv \
  --partition-map-m10 result_partition_k40_compare_GF_fs2/refined/cluster_mapping_k40_nc16_m10_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs2 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --train-window 36 \
  --forecasting-scope 2 \
  --month-ind \
  --enable-validation-threshold
```

Expected: command exits 0 and writes the same five provider outputs for fs2.

- [ ] **Step 3: Run fs3 thresholded provider**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py \
  --data 'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm.csv' \
  --partition-map result_partition_k40_compare_GF_fs3/refined/cluster_mapping_k40_nc17_general_refined_contig3.csv \
  --partition-map-m2 result_partition_k40_compare_GF_fs3/refined/cluster_mapping_k40_nc13_m2_refined_contig3.csv \
  --partition-map-m6 result_partition_k40_compare_GF_fs3/refined/cluster_mapping_k40_nc11_m6_refined_contig3.csv \
  --partition-map-m10 result_partition_k40_compare_GF_fs3/refined/cluster_mapping_k40_nc16_m10_refined_contig3.csv \
  --out-dir result_partition_k40_compare_GF_thresholded_fs3 \
  --start-month 2021-01 \
  --end-month 2024-12 \
  --train-window 36 \
  --forecasting-scope 3 \
  --month-ind \
  --enable-validation-threshold
```

Expected: command exits 0 and writes the same five provider outputs for fs3.

- [ ] **Step 4: Run the month-specific alignment regression test**

Run:

```bash
python3 -m pytest src/tests/test_georf_thresholded_monthind_alignment.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4 and Task 5 together**

```bash
git add src/tests/test_georf_thresholded_monthind_alignment.py \
  result_partition_k40_compare_GF_thresholded_fs1 \
  result_partition_k40_compare_GF_thresholded_fs2 \
  result_partition_k40_compare_GF_thresholded_fs3
git commit -m "align thresholded GeoRF month partitions"
```

---

### Task 6: Rebuild Paper Artifact Group 12

**Files:**
- Regenerate: `final_artifacts_in_paper_updated/12_thresholded_georf_results/*`

- [ ] **Step 1: Rebuild the artifact group**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/build_georf_thresholded_artifacts.py
```

Expected: command exits 0 and writes:

```text
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_horizon_metrics.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_monthly_metrics.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_threshold_provenance.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_compact_table.csv
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_compact_table.md
final_artifacts_in_paper_updated/12_thresholded_georf_results/georf_thresholded_note.md
final_artifacts_in_paper_updated/12_thresholded_georf_results/artifact_source_manifest.json
```

- [ ] **Step 2: Verify the artifact manifest contains month-specific provider details**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("final_artifacts_in_paper_updated/12_thresholded_georf_results/artifact_source_manifest.json").read_text())
for scope, details in manifest["provider_details"].items():
    assert details["month_ind_enabled"] is True, scope
    assert details["partition_map_m2_path"], scope
    assert details["partition_map_m6_path"], scope
    assert details["partition_map_m10_path"], scope
    assert details["smote_available"] is True, scope
print("12 artifact manifest provider details verified")
PY
```

Expected: prints `12 artifact manifest provider details verified`.

- [ ] **Step 3: Commit Task 6**

```bash
git add final_artifacts_in_paper_updated/12_thresholded_georf_results scripts/build_georf_thresholded_artifacts.py src/tests/test_georf_thresholded_artifacts.py
git commit -m "rebuild thresholded GeoRF appendix"
```

---

### Task 7: Final Verification, Audit, and Push

**Files:**
- Regenerate through verifier: `final_artifacts_in_paper_updated/artifact_source_audit.csv`
- Regenerate through verifier: `final_artifacts_in_paper_updated/artifact_source_audit.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m pytest \
  src/tests/test_georf_stage3_manifest_provenance.py \
  src/tests/test_georf_thresholded_artifacts.py \
  src/tests/test_georf_thresholded_monthind_alignment.py \
  src/tests/test_artifact_source_audit.py \
  -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run the reproducibility verifier**

Run:

```bash
'/mnt/c/Users/swl00/AppData/Local/Microsoft/WindowsApps/python3.12.exe' scripts/verify_current_results_reproducibility.py
```

Expected: exits 0, reports no artifact-source audit failures, and rewrites `artifact_source_audit.csv/.md` if needed.

- [ ] **Step 3: Scan for phase-change contamination**

Run:

```bash
rg --no-ignore -n "FEWSNET_forecast_unadjusted_bm_phase_change|phase_change" \
  final_artifacts_in_paper_updated \
  result_partition_k40_compare_GF_thresholded_fs1 \
  result_partition_k40_compare_GF_thresholded_fs2 \
  result_partition_k40_compare_GF_thresholded_fs3
```

Expected: no output.

- [ ] **Step 4: Commit verifier outputs if changed**

Run:

```bash
git status --short
```

If `final_artifacts_in_paper_updated/artifact_source_audit.csv` or `.md` changed, commit them:

```bash
git add final_artifacts_in_paper_updated/artifact_source_audit.csv final_artifacts_in_paper_updated/artifact_source_audit.md
git commit -m "refresh artifact source audit"
```

Expected: either a clean tree or a small audit-output commit.

- [ ] **Step 5: Push all commits**

Run:

```bash
git push origin main
```

Expected: push succeeds.
