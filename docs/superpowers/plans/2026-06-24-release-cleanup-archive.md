# Release Cleanup and Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the release-facing repository surface so the main view exposes the GeoRF paper reproduction path, while archiving non-paper pipeline entry points and preserving verifier-dependent results.

**Architecture:** Move paper artifact generators into `scripts/paper_artifacts/`, move non-paper entry points into `archived/release_20260624_nonpaper_pipelines/`, and update references and documentation around the new layout. Keep low-level compatibility modules and result directories in place so the current reproducibility verifier remains stable. Finish by validating the package, verifier, archive manifest, and creating the annotated release tag.

**Tech Stack:** Git moves, Python 3 standard library (`csv`, `hashlib`, `pathlib`, `unittest`), existing pytest/unittest tests, Markdown documentation, Windows batch files retained for the paper path.

---

## File Structure

- Create: `src/tests/test_release_cleanup_contract.py`
  - Contract tests for the release layout, archive manifest, and release-facing README quickstart.
- Create: `scripts/paper_artifacts/__init__.py`
  - Marks the paper artifact script directory as importable.
- Move into `scripts/paper_artifacts/`
  - `scripts/audit_final_artifact_sources.py`
  - `scripts/analyze_georf_false_negative_error_modes.py`
  - `scripts/analyze_georf_humanitarian_population_metrics.py`
  - `scripts/analyze_georf_m2_cluster_profiles.py`
  - `scripts/analyze_georf_partition_stability.py`
  - `scripts/analyze_georf_probability_uncertainty.py`
  - `scripts/analyze_georf_threshold_free_metrics.py`
  - `scripts/build_feature_exclude_ablation_workbook.py`
  - `scripts/build_georf_partitioned_shap_heatmap.py`
  - `scripts/build_georf_thresholded_artifacts.py`
  - `scripts/create_region_performance_partitioned_pooled_fewsnet.py`
  - `scripts/paper_horizon_labels.py`
  - `scripts/plot_error_rate_grids.py`
  - `scripts/plot_fewsnet_crisis_stack_2018.py`
  - `scripts/plot_geodt_branch_1_vs_011_locations.py`
  - `scripts/plot_geodt_branch_tree_comparison.py`
  - `scripts/plot_georf_m2_adjacency_refinement.py`
  - `scripts/plot_georf_precision_recall_curves.py`
  - `scripts/plot_global_cluster_map_2x2_refined.py`
  - `scripts/plot_monthly_performance_metrics.py`
  - `scripts/plot_predictions_2024.py`
  - `scripts/plot_region_class_prevalence.py`
  - `scripts/plot_seasonal_performance.py`
  - `scripts/relabel_final_artifact_horizons.py`
- Move into `archived/release_20260624_nonpaper_pipelines/`
  - `prediction_pipeline/*.bat`
  - `scripts/predict_partitioned_2026_2027.py`
  - `scripts/predict_scenario_2026_2027.py`
  - `app/main_model_XGB.py`
  - `scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py`
  - `scripts/rename_xgb_results.py`
  - ignored legacy notebooks from `scripts/*.ipynb`
  - `app/final/*.py`
  - `demo/*.py`
  - `scripts/generate_actual_predicted_dashboard.py`
  - `scripts/plot_phase_change_monthly_performance.py`
  - `scripts/plot_f1_improvement_comparison.py`
  - `scripts/plot_f1_improvement_comparison_by_country.py`
  - `scripts/plot_score_details_maps.py`
  - `scripts/run_georf_2024.sh`
  - `regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py`
  - `regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html`
- Create: `archived/release_20260624_nonpaper_pipelines/README.md`
  - Explains the archive as historical provenance, not a maintained runnable surface.
- Create: `archived/release_20260624_nonpaper_pipelines/MANIFEST.csv`
  - Records old path, archive path, category, reason, size, and SHA-256.
- Modify: `scripts/verify_current_results_reproducibility.py`
  - Import paper artifact audit helpers from `scripts.paper_artifacts`.
- Modify: `scripts/build_paper_reproducibility_package.py`
  - Generated package docs must point non-paper workflows to the archive.
- Modify: `README.md`
  - Release quickstart shows package validation and GeoRF regeneration only.
- Modify: `PIPELINE_WORKFLOW.md`
  - GeoRF is primary; GeoDT is appendix artifact provenance; non-paper workflows are archive references.
- Modify: `CURRENT_RESULTS_REPRODUCTION.md`
  - Update paper artifact script paths and archive note.
- Modify: tests under `src/tests/`
  - Update imports and script paths to `scripts/paper_artifacts/`.
- Regenerate: `paper_reproducibility_package/`
  - Keep package manifest and checksums aligned with updated generated docs and final artifact audit outputs.
- Create: `RELEASE_MANIFEST.md`
  - Release scope, active commands, archived categories, verification evidence, and tag name.
- Create tag after all verification:
  - `v1.0-paper-reproducibility-20260624`

## Task 1: Add Release Cleanup Contract Tests

**Files:**
- Create: `src/tests/test_release_cleanup_contract.py`
- Read: `docs/superpowers/specs/2026-06-24-release-cleanup-archive-design.md`

- [ ] **Step 1: Create the release cleanup contract test**

Use `apply_patch` to add `src/tests/test_release_cleanup_contract.py`:

```python
"""Contract tests for the paper release cleanup layout."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "archived" / "release_20260624_nonpaper_pipelines"
PAPER_ARTIFACT_ROOT = REPO_ROOT / "scripts" / "paper_artifacts"

PAPER_ARTIFACT_SCRIPTS = [
    "audit_final_artifact_sources.py",
    "analyze_georf_false_negative_error_modes.py",
    "analyze_georf_humanitarian_population_metrics.py",
    "analyze_georf_m2_cluster_profiles.py",
    "analyze_georf_partition_stability.py",
    "analyze_georf_probability_uncertainty.py",
    "analyze_georf_threshold_free_metrics.py",
    "build_feature_exclude_ablation_workbook.py",
    "build_georf_partitioned_shap_heatmap.py",
    "build_georf_thresholded_artifacts.py",
    "create_region_performance_partitioned_pooled_fewsnet.py",
    "paper_horizon_labels.py",
    "plot_error_rate_grids.py",
    "plot_fewsnet_crisis_stack_2018.py",
    "plot_geodt_branch_1_vs_011_locations.py",
    "plot_geodt_branch_tree_comparison.py",
    "plot_georf_m2_adjacency_refinement.py",
    "plot_georf_precision_recall_curves.py",
    "plot_global_cluster_map_2x2_refined.py",
    "plot_monthly_performance_metrics.py",
    "plot_predictions_2024.py",
    "plot_region_class_prevalence.py",
    "plot_seasonal_performance.py",
    "relabel_final_artifact_horizons.py",
]

ARCHIVED_OLD_PATHS = {
    "prediction_pipeline/run_partition_predict_unified.bat",
    "prediction_pipeline/run_predict_2026_2027.bat",
    "prediction_pipeline/run_scenario_predict_jun2026_feb2027.bat",
    "prediction_pipeline/spatial_weighted_consensus_clustering_predict.bat",
    "scripts/predict_partitioned_2026_2027.py",
    "scripts/predict_scenario_2026_2027.py",
    "app/main_model_XGB.py",
    "scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py",
    "scripts/rename_xgb_results.py",
    "scripts/CH.ipynb",
    "scripts/baseline_comparison.ipynb",
    "scripts/baseline_comparison_2021.ipynb",
    "scripts/compare_unsplit_and_split_f1.ipynb",
    "scripts/examine_scope.ipynb",
    "scripts/genenerate_new_test.ipynb",
    "scripts/generate_test_data.ipynb",
    "scripts/seasonality.ipynb",
    "scripts/see_polygons.ipynb",
    "scripts/trial_phase_change.ipynb",
    "app/final/baseline_probit_regression.py",
    "app/final/fewsnet_baseline_evaluation.py",
    "app/final/georf_vs_baseline_comparison_plot.py",
    "demo/GeoRF_demo.py",
    "demo/data.py",
    "scripts/generate_actual_predicted_dashboard.py",
    "scripts/plot_phase_change_monthly_performance.py",
    "scripts/plot_f1_improvement_comparison.py",
    "scripts/plot_f1_improvement_comparison_by_country.py",
    "scripts/plot_score_details_maps.py",
    "scripts/run_georf_2024.sh",
    "regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py",
    "regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html",
}


def load_archive_manifest() -> dict[str, dict[str, str]]:
    manifest = ARCHIVE_ROOT / "MANIFEST.csv"
    assert manifest.is_file(), f"Missing archive manifest: {manifest}"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["old_path"]: row for row in rows}


def release_quickstart_text() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Quick Start for Paper Reproduction")
    end = readme.index("## Complete No-Leak Regeneration Path")
    return readme[start:end]


def test_paper_artifact_scripts_are_moved_under_paper_artifacts() -> None:
    assert (PAPER_ARTIFACT_ROOT / "__init__.py").is_file()
    for script in PAPER_ARTIFACT_SCRIPTS:
        assert (PAPER_ARTIFACT_ROOT / script).is_file(), script
        assert not (REPO_ROOT / "scripts" / script).exists(), script


def test_nonpaper_archive_manifest_contains_expected_moves() -> None:
    rows = load_archive_manifest()
    assert set(rows) >= ARCHIVED_OLD_PATHS
    for old_path in ARCHIVED_OLD_PATHS:
        row = rows[old_path]
        archive_path = REPO_ROOT / row["archive_path"]
        assert archive_path.is_file(), row
        assert row["category"] in {
            "prediction_pipeline",
            "scripts_prediction",
            "geoxgb_workflow",
            "notebooks_legacy",
            "legacy_misc",
        }
        assert row["sha256"]
        assert int(row["size_bytes"]) >= 0
        assert not (REPO_ROOT / old_path).exists(), old_path


def test_release_quickstart_is_paper_only() -> None:
    quickstart = release_quickstart_text()
    forbidden = [
        "GeoXGB",
        "fs0",
        "--fs0-only",
        "prediction_pipeline",
        "run_predict_2026_2027",
        "run_scenario_predict_jun2026_feb2027",
    ]
    for token in forbidden:
        assert token not in quickstart
    assert "validate_paper_reproducibility_package.py" in quickstart
    assert "verify_current_results_reproducibility.py" in quickstart


def test_release_manifest_names_archive_and_tag() -> None:
    manifest = REPO_ROOT / "RELEASE_MANIFEST.md"
    text = manifest.read_text(encoding="utf-8")
    assert "v1.0-paper-reproducibility-20260624" in text
    assert "archived/release_20260624_nonpaper_pipelines/" in text
    assert "GeoRF" in text
    assert "GeoDT appendix" in text
```

- [ ] **Step 2: Run the contract test and confirm it fails before cleanup**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py -q
```

Expected: FAIL because `scripts/paper_artifacts/`, the archive manifest, and `RELEASE_MANIFEST.md` do not exist yet.

- [ ] **Step 3: Commit the failing contract test**

Run:

```bash
git add src/tests/test_release_cleanup_contract.py
git commit -m "test release cleanup contract"
```

Expected: commit includes only `src/tests/test_release_cleanup_contract.py`.

## Task 2: Move Paper Artifact Scripts and Update Imports

**Files:**
- Create: `scripts/paper_artifacts/__init__.py`
- Move: the 24 paper artifact scripts listed in Task 1
- Modify: `scripts/verify_current_results_reproducibility.py`
- Modify: `src/tests/test_artifact_source_audit.py`
- Modify: `src/tests/test_build_feature_exclude_ablation_workbook.py`
- Modify: `src/tests/test_paper_horizon_labels.py`
- Modify: `src/tests/test_plot_predictions_2024.py`
- Modify: `src/tests/test_plot_monthly_performance_metrics.py`
- Modify: `src/tests/test_geodt_branch_tree_diagnostic.py`

- [ ] **Step 1: Create the paper artifact package directory**

Run:

```bash
mkdir -p scripts/paper_artifacts
touch scripts/paper_artifacts/__init__.py
```

- [ ] **Step 2: Move paper artifact scripts**

Run:

```bash
git mv scripts/audit_final_artifact_sources.py scripts/paper_artifacts/audit_final_artifact_sources.py
git mv scripts/analyze_georf_false_negative_error_modes.py scripts/paper_artifacts/analyze_georf_false_negative_error_modes.py
git mv scripts/analyze_georf_humanitarian_population_metrics.py scripts/paper_artifacts/analyze_georf_humanitarian_population_metrics.py
git mv scripts/analyze_georf_m2_cluster_profiles.py scripts/paper_artifacts/analyze_georf_m2_cluster_profiles.py
git mv scripts/analyze_georf_partition_stability.py scripts/paper_artifacts/analyze_georf_partition_stability.py
git mv scripts/analyze_georf_probability_uncertainty.py scripts/paper_artifacts/analyze_georf_probability_uncertainty.py
git mv scripts/analyze_georf_threshold_free_metrics.py scripts/paper_artifacts/analyze_georf_threshold_free_metrics.py
git mv scripts/build_feature_exclude_ablation_workbook.py scripts/paper_artifacts/build_feature_exclude_ablation_workbook.py
git mv scripts/build_georf_partitioned_shap_heatmap.py scripts/paper_artifacts/build_georf_partitioned_shap_heatmap.py
git mv scripts/build_georf_thresholded_artifacts.py scripts/paper_artifacts/build_georf_thresholded_artifacts.py
git mv scripts/create_region_performance_partitioned_pooled_fewsnet.py scripts/paper_artifacts/create_region_performance_partitioned_pooled_fewsnet.py
git mv scripts/paper_horizon_labels.py scripts/paper_artifacts/paper_horizon_labels.py
git mv scripts/plot_error_rate_grids.py scripts/paper_artifacts/plot_error_rate_grids.py
git mv scripts/plot_fewsnet_crisis_stack_2018.py scripts/paper_artifacts/plot_fewsnet_crisis_stack_2018.py
git mv scripts/plot_geodt_branch_1_vs_011_locations.py scripts/paper_artifacts/plot_geodt_branch_1_vs_011_locations.py
git mv scripts/plot_geodt_branch_tree_comparison.py scripts/paper_artifacts/plot_geodt_branch_tree_comparison.py
git mv scripts/plot_georf_m2_adjacency_refinement.py scripts/paper_artifacts/plot_georf_m2_adjacency_refinement.py
git mv scripts/plot_georf_precision_recall_curves.py scripts/paper_artifacts/plot_georf_precision_recall_curves.py
git mv scripts/plot_global_cluster_map_2x2_refined.py scripts/paper_artifacts/plot_global_cluster_map_2x2_refined.py
git mv scripts/plot_monthly_performance_metrics.py scripts/paper_artifacts/plot_monthly_performance_metrics.py
git mv scripts/plot_predictions_2024.py scripts/paper_artifacts/plot_predictions_2024.py
git mv scripts/plot_region_class_prevalence.py scripts/paper_artifacts/plot_region_class_prevalence.py
git mv scripts/plot_seasonal_performance.py scripts/paper_artifacts/plot_seasonal_performance.py
git mv scripts/relabel_final_artifact_horizons.py scripts/paper_artifacts/relabel_final_artifact_horizons.py
```

- [ ] **Step 3: Apply mechanical import and repo-root path updates**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("scripts/paper_artifacts")
for path in root.glob("*.py"):
    text = path.read_text(encoding="utf-8")
    text = text.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]")
    text = text.replace("from scripts.paper_horizon_labels", "from scripts.paper_artifacts.paper_horizon_labels")
    text = text.replace("from scripts import paper_horizon_labels", "from scripts.paper_artifacts import paper_horizon_labels")
    text = text.replace(
        'REPO_ROOT / "scripts" / "plot_region_class_prevalence.py"',
        'REPO_ROOT / "scripts" / "paper_artifacts" / "plot_region_class_prevalence.py"',
    )
    text = text.replace(
        'REPO_ROOT / "scripts" / "plot_seasonal_performance.py"',
        'REPO_ROOT / "scripts" / "paper_artifacts" / "plot_seasonal_performance.py"',
    )
    text = text.replace(
        '"entry_point": "scripts/plot_monthly_performance_metrics.py"',
        '"entry_point": "scripts/paper_artifacts/plot_monthly_performance_metrics.py"',
    )
    text = text.replace(
        "Reference: scripts/plot_error_rate_grids.py",
        "Reference: scripts/paper_artifacts/plot_error_rate_grids.py",
    )
    path.write_text(text, encoding="utf-8", newline="\n")

replacements = {
    "from scripts.audit_final_artifact_sources": "from scripts.paper_artifacts.audit_final_artifact_sources",
    "from scripts.build_feature_exclude_ablation_workbook": "from scripts.paper_artifacts.build_feature_exclude_ablation_workbook",
    '"scripts" / "paper_horizon_labels.py"': '"scripts" / "paper_artifacts" / "paper_horizon_labels.py"',
    '"scripts" / "relabel_final_artifact_horizons.py"': '"scripts" / "paper_artifacts" / "relabel_final_artifact_horizons.py"',
    '"scripts" / "plot_predictions_2024.py"': '"scripts" / "paper_artifacts" / "plot_predictions_2024.py"',
    '"scripts" / "plot_monthly_performance_metrics.py"': '"scripts" / "paper_artifacts" / "plot_monthly_performance_metrics.py"',
    '"scripts" / "plot_geodt_branch_tree_comparison.py"': '"scripts" / "paper_artifacts" / "plot_geodt_branch_tree_comparison.py"',
}
for path in [
    Path("scripts/verify_current_results_reproducibility.py"),
    Path("src/tests/test_artifact_source_audit.py"),
    Path("src/tests/test_build_feature_exclude_ablation_workbook.py"),
    Path("src/tests/test_paper_horizon_labels.py"),
    Path("src/tests/test_plot_predictions_2024.py"),
    Path("src/tests/test_plot_monthly_performance_metrics.py"),
    Path("src/tests/test_geodt_branch_tree_diagnostic.py"),
]:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8", newline="\n")
PY
```

- [ ] **Step 4: Update audit script paths for moved paper scripts**

Use `apply_patch` to update `scripts/paper_artifacts/audit_final_artifact_sources.py` so `SCRIPT_DEFAULTS` uses the new paper artifact paths while leaving the XGB script at its root path until Task 3 archives it:

```python
SCRIPT_DEFAULTS = [
    (
        "scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py",
        REPO_ROOT / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py",
    ),
    (
        "scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py",
        REPO_ROOT / "scripts" / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
    ),
    (
        "scripts/paper_artifacts/analyze_georf_m2_cluster_profiles.py",
        REPO_ROOT / "scripts" / "paper_artifacts" / "analyze_georf_m2_cluster_profiles.py",
    ),
    (
        "scripts/paper_artifacts/analyze_georf_false_negative_error_modes.py",
        REPO_ROOT / "scripts" / "paper_artifacts" / "analyze_georf_false_negative_error_modes.py",
    ),
]
```

- [ ] **Step 5: Update artifact source audit test script list for moved paper scripts**

Use `apply_patch` to update `src/tests/test_artifact_source_audit.py` in `test_paper_facing_script_defaults_do_not_reference_phase_change`:

```python
    scripts = [
        repo_root / "scripts" / "compare_partitioned_vs_pooled_rf_k40_nc4.py",
        repo_root / "scripts" / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
        repo_root / "scripts" / "paper_artifacts" / "analyze_georf_m2_cluster_profiles.py",
        repo_root / "scripts" / "paper_artifacts" / "analyze_georf_false_negative_error_modes.py",
    ]
```

- [ ] **Step 6: Run focused paper artifact tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  src/tests/test_build_feature_exclude_ablation_workbook.py \
  src/tests/test_paper_horizon_labels.py \
  src/tests/test_plot_predictions_2024.py \
  src/tests/test_plot_monthly_performance_metrics.py \
  src/tests/test_geodt_branch_tree_diagnostic.py \
  -q
```

Expected: tests pass after import and path updates.

- [ ] **Step 7: Commit paper artifact script layout**

Run:

```bash
git add scripts/paper_artifacts scripts/verify_current_results_reproducibility.py src/tests
git commit -m "move paper artifact scripts under dedicated package"
```

Expected: commit contains paper artifact moves, import updates, and focused test path updates only.

## Task 3: Archive Non-Paper Pipeline Entry Points

**Files:**
- Create: `archived/release_20260624_nonpaper_pipelines/README.md`
- Create: `archived/release_20260624_nonpaper_pipelines/MANIFEST.csv`
- Move: files listed under archive categories in this task

- [ ] **Step 1: Create archive directories**

Run:

```bash
mkdir -p archived/release_20260624_nonpaper_pipelines/prediction_pipeline
mkdir -p archived/release_20260624_nonpaper_pipelines/scripts_prediction
mkdir -p archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/app
mkdir -p archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/scripts
mkdir -p archived/release_20260624_nonpaper_pipelines/notebooks_legacy
mkdir -p archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final
mkdir -p archived/release_20260624_nonpaper_pipelines/legacy_misc/demo
mkdir -p archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts
mkdir -p archived/release_20260624_nonpaper_pipelines/legacy_misc/regional_ablation_results/Nigeria_experiments_global
mkdir -p archived/release_20260624_nonpaper_pipelines/legacy_misc/regional_ablation_results/actual_predicted_dashboard
```

- [ ] **Step 2: Move tracked non-paper entry points**

Run:

```bash
git mv prediction_pipeline/run_partition_predict_unified.bat archived/release_20260624_nonpaper_pipelines/prediction_pipeline/run_partition_predict_unified.bat
git mv prediction_pipeline/run_predict_2026_2027.bat archived/release_20260624_nonpaper_pipelines/prediction_pipeline/run_predict_2026_2027.bat
git mv prediction_pipeline/run_scenario_predict_jun2026_feb2027.bat archived/release_20260624_nonpaper_pipelines/prediction_pipeline/run_scenario_predict_jun2026_feb2027.bat
git mv prediction_pipeline/spatial_weighted_consensus_clustering_predict.bat archived/release_20260624_nonpaper_pipelines/prediction_pipeline/spatial_weighted_consensus_clustering_predict.bat
git mv scripts/predict_partitioned_2026_2027.py archived/release_20260624_nonpaper_pipelines/scripts_prediction/predict_partitioned_2026_2027.py
git mv scripts/predict_scenario_2026_2027.py archived/release_20260624_nonpaper_pipelines/scripts_prediction/predict_scenario_2026_2027.py
git mv app/main_model_XGB.py archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/app/main_model_XGB.py
git mv scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py
git mv scripts/rename_xgb_results.py archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/scripts/rename_xgb_results.py
git mv app/final/baseline_probit_regression.py archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/baseline_probit_regression.py
git mv app/final/fewsnet_baseline_evaluation.py archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/fewsnet_baseline_evaluation.py
git mv app/final/georf_vs_baseline_comparison_plot.py archived/release_20260624_nonpaper_pipelines/legacy_misc/app_final/georf_vs_baseline_comparison_plot.py
git mv demo/GeoRF_demo.py archived/release_20260624_nonpaper_pipelines/legacy_misc/demo/GeoRF_demo.py
git mv demo/data.py archived/release_20260624_nonpaper_pipelines/legacy_misc/demo/data.py
git mv scripts/generate_actual_predicted_dashboard.py archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts/generate_actual_predicted_dashboard.py
git mv scripts/plot_phase_change_monthly_performance.py archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts/plot_phase_change_monthly_performance.py
git mv scripts/plot_f1_improvement_comparison.py archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts/plot_f1_improvement_comparison.py
git mv scripts/plot_f1_improvement_comparison_by_country.py archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts/plot_f1_improvement_comparison_by_country.py
git mv scripts/plot_score_details_maps.py archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts/plot_score_details_maps.py
git mv scripts/run_georf_2024.sh archived/release_20260624_nonpaper_pipelines/legacy_misc/scripts/run_georf_2024.sh
git mv regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py archived/release_20260624_nonpaper_pipelines/legacy_misc/regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py
git mv regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html archived/release_20260624_nonpaper_pipelines/legacy_misc/regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html
```

- [ ] **Step 3: Update audit paths for the archived XGB script**

Use `apply_patch` to update `scripts/paper_artifacts/audit_final_artifact_sources.py` so the XGB `SCRIPT_DEFAULTS` entry points to the archive:

```python
    (
        "archived/release_20260624_nonpaper_pipelines/geoxgb_workflow/scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py",
        REPO_ROOT
        / "archived"
        / "release_20260624_nonpaper_pipelines"
        / "geoxgb_workflow"
        / "scripts"
        / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
    ),
```

Use `apply_patch` to update `src/tests/test_artifact_source_audit.py` so the XGB path in `test_paper_facing_script_defaults_do_not_reference_phase_change` is:

```python
        repo_root
        / "archived"
        / "release_20260624_nonpaper_pipelines"
        / "geoxgb_workflow"
        / "scripts"
        / "compare_partitioned_vs_pooled_xgb_k40_nc4.py",
```

- [ ] **Step 4: Move ignored legacy notebooks and force-track them in archive**

Run:

```bash
mv scripts/CH.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/CH.ipynb
mv scripts/baseline_comparison.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/baseline_comparison.ipynb
mv scripts/baseline_comparison_2021.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/baseline_comparison_2021.ipynb
mv scripts/compare_unsplit_and_split_f1.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/compare_unsplit_and_split_f1.ipynb
mv scripts/examine_scope.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/examine_scope.ipynb
mv scripts/genenerate_new_test.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/genenerate_new_test.ipynb
mv scripts/generate_test_data.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/generate_test_data.ipynb
mv scripts/seasonality.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/seasonality.ipynb
mv scripts/see_polygons.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/see_polygons.ipynb
mv scripts/trial_phase_change.ipynb archived/release_20260624_nonpaper_pipelines/notebooks_legacy/trial_phase_change.ipynb
git add -f archived/release_20260624_nonpaper_pipelines/notebooks_legacy/*.ipynb
```

Expected: ignored notebooks are now tracked under the archive because they are part of the historical release archive.

- [ ] **Step 5: Generate archive README and manifest**

Run:

```bash
python3 - <<'PY'
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

archive = Path("archived/release_20260624_nonpaper_pipelines")
rows = [
    ("prediction_pipeline/run_partition_predict_unified.bat", "prediction_pipeline/run_partition_predict_unified.bat", "prediction_pipeline", "2026-2027 forward prediction launcher"),
    ("prediction_pipeline/run_predict_2026_2027.bat", "prediction_pipeline/run_predict_2026_2027.bat", "prediction_pipeline", "2026-2027 smoke prediction launcher"),
    ("prediction_pipeline/run_scenario_predict_jun2026_feb2027.bat", "prediction_pipeline/run_scenario_predict_jun2026_feb2027.bat", "prediction_pipeline", "synthetic scenario launcher"),
    ("prediction_pipeline/spatial_weighted_consensus_clustering_predict.bat", "prediction_pipeline/spatial_weighted_consensus_clustering_predict.bat", "prediction_pipeline", "prediction-only Stage 2 launcher"),
    ("scripts/predict_partitioned_2026_2027.py", "scripts_prediction/predict_partitioned_2026_2027.py", "scripts_prediction", "prediction-only Python implementation"),
    ("scripts/predict_scenario_2026_2027.py", "scripts_prediction/predict_scenario_2026_2027.py", "scripts_prediction", "scenario prediction Python implementation"),
    ("app/main_model_XGB.py", "geoxgb_workflow/app/main_model_XGB.py", "geoxgb_workflow", "legacy GeoXGB entry point"),
    ("scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py", "geoxgb_workflow/scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py", "geoxgb_workflow", "legacy GeoXGB Stage 3 comparison"),
    ("scripts/rename_xgb_results.py", "geoxgb_workflow/scripts/rename_xgb_results.py", "geoxgb_workflow", "legacy GeoXGB utility"),
    ("scripts/CH.ipynb", "notebooks_legacy/CH.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/baseline_comparison.ipynb", "notebooks_legacy/baseline_comparison.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/baseline_comparison_2021.ipynb", "notebooks_legacy/baseline_comparison_2021.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/compare_unsplit_and_split_f1.ipynb", "notebooks_legacy/compare_unsplit_and_split_f1.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/examine_scope.ipynb", "notebooks_legacy/examine_scope.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/genenerate_new_test.ipynb", "notebooks_legacy/genenerate_new_test.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/generate_test_data.ipynb", "notebooks_legacy/generate_test_data.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/seasonality.ipynb", "notebooks_legacy/seasonality.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/see_polygons.ipynb", "notebooks_legacy/see_polygons.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("scripts/trial_phase_change.ipynb", "notebooks_legacy/trial_phase_change.ipynb", "notebooks_legacy", "ignored legacy notebook"),
    ("app/final/baseline_probit_regression.py", "legacy_misc/app_final/baseline_probit_regression.py", "legacy_misc", "legacy baseline entry point"),
    ("app/final/fewsnet_baseline_evaluation.py", "legacy_misc/app_final/fewsnet_baseline_evaluation.py", "legacy_misc", "legacy baseline entry point"),
    ("app/final/georf_vs_baseline_comparison_plot.py", "legacy_misc/app_final/georf_vs_baseline_comparison_plot.py", "legacy_misc", "legacy baseline plotting entry point"),
    ("demo/GeoRF_demo.py", "legacy_misc/demo/GeoRF_demo.py", "legacy_misc", "demo entry point outside paper release"),
    ("demo/data.py", "legacy_misc/demo/data.py", "legacy_misc", "demo helper outside paper release"),
    ("scripts/generate_actual_predicted_dashboard.py", "legacy_misc/scripts/generate_actual_predicted_dashboard.py", "legacy_misc", "exploratory dashboard generator"),
    ("scripts/plot_phase_change_monthly_performance.py", "legacy_misc/scripts/plot_phase_change_monthly_performance.py", "legacy_misc", "phase-change exploratory diagnostic"),
    ("scripts/plot_f1_improvement_comparison.py", "legacy_misc/scripts/plot_f1_improvement_comparison.py", "legacy_misc", "legacy GeoRF-vs-XGB figure"),
    ("scripts/plot_f1_improvement_comparison_by_country.py", "legacy_misc/scripts/plot_f1_improvement_comparison_by_country.py", "legacy_misc", "legacy GeoRF-vs-XGB country figure"),
    ("scripts/plot_score_details_maps.py", "legacy_misc/scripts/plot_score_details_maps.py", "legacy_misc", "visual debugging map generator"),
    ("scripts/run_georf_2024.sh", "legacy_misc/scripts/run_georf_2024.sh", "legacy_misc", "legacy ad-hoc shell runner"),
    ("regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py", "legacy_misc/regional_ablation_results/Nigeria_experiments_global/filter_nigeria_and_aggregate.py", "legacy_misc", "regional exploratory entry point"),
    ("regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html", "legacy_misc/regional_ablation_results/actual_predicted_dashboard/actual_predicted_dashboard_Nigeria.html", "legacy_misc", "regional exploratory dashboard output"),
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with (archive / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["old_path", "archive_path", "category", "reason", "size_bytes", "sha256"],
    )
    writer.writeheader()
    for old_path, relative_archive_path, category, reason in rows:
        archive_path = archive / relative_archive_path
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing archived file: {archive_path}")
        writer.writerow(
            {
                "old_path": old_path,
                "archive_path": (archive / relative_archive_path).as_posix(),
                "category": category,
                "reason": reason,
                "size_bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            }
        )

(archive / "README.md").write_text(
    """# Non-Paper Pipeline Archive for Paper Release 20260624

This archive preserves experimental and legacy entry points that are not part of
the paper release quickstart. Files here are historical provenance. They are not
maintained as runnable workflows from their archived paths.

The paper release main view keeps GeoRF paper reproduction first, retains GeoDT
as appendix artifact provenance, and excludes GeoXGB, fs0 launch guidance,
2026-2027 forward/scenario prediction, legacy notebooks, and regional
exploratory entry points.

See `MANIFEST.csv` for old paths, archive paths, categories, reasons, file
sizes, and SHA-256 checksums.
""",
    encoding="utf-8",
    newline="\n",
)
PY
```

- [ ] **Step 6: Run archive contract checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py::test_nonpaper_archive_manifest_contains_expected_moves -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_artifact_source_audit.py::test_paper_facing_script_defaults_do_not_reference_phase_change -q
```

Expected: both tests pass after archive moves and audit path updates.

- [ ] **Step 7: Commit archived non-paper entry points**

Run:

```bash
git add -A archived/release_20260624_nonpaper_pipelines prediction_pipeline scripts app demo regional_ablation_results src/tests/test_artifact_source_audit.py
git commit -m "archive non-paper release entry points"
```

Expected: commit contains archive moves, archive README, archive manifest, and the artifact-source-audit test adjustment.

## Task 4: Update Release Documentation and Package Text

**Files:**
- Modify: `README.md`
- Modify: `PIPELINE_WORKFLOW.md`
- Modify: `CURRENT_RESULTS_REPRODUCTION.md`
- Modify: `scripts/build_paper_reproducibility_package.py`
- Regenerate: `paper_reproducibility_package/`

- [ ] **Step 1: Update README release quickstart and workflow surface**

Use `apply_patch` to update `README.md` with these content rules:

````markdown
## Quick Start for Paper Reproduction

For fast review, start with the lightweight package:

```text
paper_reproducibility_package/README.md
paper_reproducibility_package/MANIFEST.csv
paper_reproducibility_package/SHA256SUMS.txt
```

Validate it from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Then verify the live repository result bundle:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```
````

Under `Complete No-Leak Regeneration Path`, show only the GeoRF commands:

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
spatial_weighted_consensus_clustering.bat georf
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

Add this paragraph after the GeoRF commands:

```markdown
GeoDT result directories and figures are retained as appendix and
interpretability provenance, but GeoDT is not part of the release quickstart.
Non-paper workflows are archived under
`archived/release_20260624_nonpaper_pipelines/`.
```

Remove the `Experimental and Extension Workflows` section from README and replace it with:

````markdown
## Non-Paper Workflow Archive

GeoXGB, fs0 lag-1 launch guidance, 2026-2027 forward/scenario prediction,
legacy notebooks, regional exploratory scripts, and legacy baseline/demo entry
points are preserved in:

```text
archived/release_20260624_nonpaper_pipelines/
```

Those files are historical provenance for the development repository. They are
not part of the release quickstart and are not maintained as runnable workflows
from their archived paths.
````

Update the directory tree to remove `prediction_pipeline/` and `app/main_model_XGB.py`, add `scripts/paper_artifacts/`, and add the archive root.

- [ ] **Step 2: Update PIPELINE_WORKFLOW paper-release framing**

Use `apply_patch` to edit `PIPELINE_WORKFLOW.md`:

- Replace the overview sentence with:

```markdown
This document describes the release-facing no-leak workflow for the GeoRF paper
results. Stage 1 learns GeoRF partition candidates on 2018-2020, Stage 2 learns
fixed consensus partitions from those outputs, and Stage 3 evaluates fixed
partitions on 2021-2024. GeoDT outputs are retained as appendix and
interpretability provenance. GeoXGB, fs0 launch guidance, and 2026-2027
forward/scenario prediction have been archived as non-paper workflows.
```

- Replace the pipeline flow summary table with GeoRF-only active commands.
- Replace the separate GeoRF-only prediction workflow and FS0-only sections with:

````markdown
## Non-Paper Workflow Archive

The former GeoXGB, fs0-only, 2026-2027 forward/scenario prediction, legacy
notebook, and regional exploratory entry points are preserved under:

```text
archived/release_20260624_nonpaper_pipelines/
```

These archived files are historical provenance and are not maintained as
release quickstart workflows.
````

- Keep detailed Stage 1, Stage 2, and Stage 3 sections only where they describe the GeoRF release path. If a section still references `geodt`, describe it as appendix provenance rather than a quickstart command.

- [ ] **Step 3: Update current results reproduction paths**

Use `apply_patch` to edit `CURRENT_RESULTS_REPRODUCTION.md`:

- Change commands that invoke moved paper artifact scripts from `python scripts/<name>.py` to `python -m scripts.paper_artifacts.<module_name>`.
- Add this note near the result inventory:

```markdown
Non-paper workflow entry points are archived under
`archived/release_20260624_nonpaper_pipelines/`. The archive is historical
provenance and is not required for `verify_current_results_reproducibility.py`.
```

- [ ] **Step 4: Update generated package documentation text**

Use `apply_patch` to edit `scripts/build_paper_reproducibility_package.py`:

- In `package_readme()`, replace:

```markdown
- It does not move or archive experimental scripts; release-version code
  migration is a separate future task.
```

with:

```markdown
- Experimental and legacy entry points are archived under
  `archived/release_20260624_nonpaper_pipelines/` in the source repository.
```

- In `consistency_audit()`, replace:

```markdown
GeoXGB, fs0 lag-1, and 2026-2027 forward/scenario prediction are not part of
this package. They remain in their current repo paths for future release-version
migration.
```

with:

```markdown
GeoXGB, fs0 lag-1 launch guidance, and 2026-2027 forward/scenario prediction
are not part of this package. Their entry points are preserved as historical
provenance under `archived/release_20260624_nonpaper_pipelines/`.
```

- Update any generated artifact-map references that still use root `scripts/<paper artifact>.py` paths to `scripts/paper_artifacts/<paper artifact>.py`.

- [ ] **Step 5: Regenerate package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_paper_reproducibility_package.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected: builder completes and validator reports `Package validation passed: <N> files checked.`

- [ ] **Step 6: Force-add regenerated package files hidden by gitignore**

Run:

```bash
git add -f paper_reproducibility_package
```

Expected: package manifest, checksums, generated docs, and copied final artifacts are staged if they changed.

- [ ] **Step 7: Run release documentation keyword checks**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
quick = readme[
    readme.index("## Quick Start for Paper Reproduction") : readme.index("## Complete No-Leak Regeneration Path")
]
for token in ["GeoXGB", "fs0", "--fs0-only", "prediction_pipeline", "run_predict_2026_2027"]:
    if token in quick:
        raise SystemExit(f"README quickstart still exposes {token}")

workflow = Path("PIPELINE_WORKFLOW.md").read_text(encoding="utf-8")
if "archived/release_20260624_nonpaper_pipelines/" not in workflow:
    raise SystemExit("PIPELINE_WORKFLOW.md does not point to the non-paper archive")
print("Release documentation keyword checks passed.")
PY
```

Expected: prints `Release documentation keyword checks passed.`

- [ ] **Step 8: Commit release documentation and regenerated package**

Run:

```bash
git add README.md PIPELINE_WORKFLOW.md CURRENT_RESULTS_REPRODUCTION.md scripts/build_paper_reproducibility_package.py paper_reproducibility_package
git commit -m "document paper release workflow surface"
```

Expected: commit contains docs, generated package updates, and package builder text updates.

## Task 5: Add Release Manifest

**Files:**
- Create: `RELEASE_MANIFEST.md`

- [ ] **Step 1: Create release manifest**

Use `apply_patch` to add `RELEASE_MANIFEST.md`:

````markdown
# Paper Reproducibility Release Manifest

## Release

- Tag: `v1.0-paper-reproducibility-20260624`
- Date: 2026-06-24
- Scope: GeoRF paper reproduction with GeoDT appendix artifact provenance

## Active Release Commands

Validate the lightweight package:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Verify the live result bundle:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

Run the full GeoRF regeneration path:

```batch
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
spatial_weighted_consensus_clustering.bat georf
run_partition_k40_comparison_unified.bat georf --visual --month-ind
```

## Retained Paper Assets

- `paper_reproducibility_package/`
- `final_artifacts_in_paper_updated/`
- `GeoRFExperiment/`
- `GeoDTExperiment/` for appendix provenance
- `result_partition_k40_compare_GF_fs1/`
- `result_partition_k40_compare_GF_fs2/`
- `result_partition_k40_compare_GF_fs3/`
- `result_partition_k40_compare_DT_fs1/` for appendix provenance
- `result_partition_k40_compare_DT_fs2/` for appendix provenance
- `result_partition_k40_compare_DT_fs3/` for appendix provenance
- `main_ablation_exclude_updated_stage3_fixed_partitions/`
- `result_partition_k40_compare_GF_thresholded_fs1/`
- `result_partition_k40_compare_GF_thresholded_fs2/`
- `result_partition_k40_compare_GF_thresholded_fs3/`

## Archived Non-Paper Entry Points

Non-paper workflow entry points are preserved under:

```text
archived/release_20260624_nonpaper_pipelines/
```

The archive includes GeoXGB workflow entry points, fs0 launch guidance sources,
2026-2027 forward/scenario prediction launchers, legacy notebooks, regional
exploratory entry points, and legacy baseline/demo entry points. These files are
historical provenance and are not maintained as runnable workflows from the
archive path.

## Verification Commands

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
git diff --check
git status --short
```
````

- [ ] **Step 2: Run release contract test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/tests/test_release_cleanup_contract.py -q
```

Expected: all release cleanup contract tests pass.

- [ ] **Step 3: Commit release manifest**

Run:

```bash
git add RELEASE_MANIFEST.md src/tests/test_release_cleanup_contract.py
git commit -m "add paper release manifest"
```

Expected: commit includes the release manifest and any final release-contract-test updates.

## Task 6: Full Verification and Release Tag

**Files:**
- Read/verify: all release files
- Create tag: `v1.0-paper-reproducibility-20260624`

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  src/tests/test_release_cleanup_contract.py \
  src/tests/test_artifact_source_audit.py \
  src/tests/test_build_feature_exclude_ablation_workbook.py \
  src/tests/test_paper_horizon_labels.py \
  src/tests/test_plot_predictions_2024.py \
  src/tests/test_plot_monthly_performance_metrics.py \
  src/tests/test_geodt_branch_tree_diagnostic.py \
  -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run package utility test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest src.tests.test_paper_reproducibility_package -v
```

Expected: all package utility tests pass.

- [ ] **Step 3: Validate paper reproducibility package**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_paper_reproducibility_package.py
```

Expected output:

```text
Package validation passed: <N> files checked.
```

- [ ] **Step 4: Run current result bundle verifier**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_current_results_reproducibility.py
```

Expected output ends with:

```text
Verification passed: current result bundle is organized and reproducibility metadata is present.
```

If this command rewrites only `final_artifacts_in_paper_updated/artifact_source_audit.md`, restore that file before final commit:

```bash
git restore final_artifacts_in_paper_updated/artifact_source_audit.md
```

- [ ] **Step 5: Verify package manifest is tracked**

Run:

```bash
python3 - <<'PY'
import csv
import subprocess
from pathlib import Path

root = Path("paper_reproducibility_package")
tracked = set(subprocess.check_output(["git", "ls-files", "paper_reproducibility_package"], text=True).splitlines())
with (root / "MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
    manifest = {str(root / row["package_path"]) for row in csv.DictReader(handle)}
manifest.add(str(root / "MANIFEST.csv"))
missing = sorted(manifest - tracked)
extra = sorted(tracked - manifest - {str(root / "SHA256SUMS.txt")})
if missing or extra:
    raise SystemExit(f"package tracking mismatch missing={missing[:5]} extra={extra[:5]}")
print("Package tracked-file check passed.")
PY
```

Expected: prints `Package tracked-file check passed.`

- [ ] **Step 6: Run final git checks**

Run:

```bash
git diff --check
git status --short
git log --oneline -10
```

Expected:

- `git diff --check` exits 0.
- `git status --short` is clean before tag creation.
- Recent commits include tests, paper artifact script move, archive move, docs/package updates, release manifest, and the design/plan commits.

- [ ] **Step 7: Create annotated release tag**

Run:

```bash
git tag -a v1.0-paper-reproducibility-20260624 -m "Paper reproducibility release 20260624"
git tag --list "v1.0-paper-reproducibility-20260624" -n
```

Expected: tag list shows `v1.0-paper-reproducibility-20260624 Paper reproducibility release 20260624`.

- [ ] **Step 8: Report final release state**

Run:

```bash
git status --short
git log --oneline -10
git tag --list "v1.0-paper-reproducibility-20260624" -n
```

Expected: worktree is clean and the release tag exists locally.
