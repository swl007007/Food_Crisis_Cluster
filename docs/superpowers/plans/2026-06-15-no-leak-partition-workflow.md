# No-Leak Partition Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active main workflow with a no-leak split where partitions are learned from 2018-2020 and final metrics are evaluated on 2021-2024, with GeoRF/GeoDT only.

**Architecture:** Keep the existing three-stage architecture and artifact naming patterns. Rename and narrow the active Stage 1 batch entrypoint, update Stage 2 and Stage 3 model allowlists and provenance, and add fast repository-contract tests that prevent reintroducing the old leaking Stage 1 path or GeoXGB main-result inclusion.

**Tech Stack:** Windows batch files, Python 3.12, standard-library `unittest`, existing pandas/openpyxl aggregation scripts, repository Markdown docs.

---

## Files And Responsibilities

- `run_batches_2018_2020_partition_learning_visual_monthly.bat`: new active Stage 1 main workflow entrypoint. Same output naming style as the old batch, but loops 2018-2020 and supports only `georf` and `geodt`.
- `run_batches_2021_2024_visual_monthly.bat`: remove from active root workflow by renaming with `git mv`; it must not remain at this path after implementation.
- `spatial_weighted_consensus_clustering.bat`: update usage, model validation, Stage 1 hints, and GeoXGB branches so Stage 2 main workflow consumes only GeoRF/GeoDT partition-learning outputs.
- `run_partition_k40_comparison_unified.bat`: keep evaluation window 2021-2024, narrow `all` mode to GeoRF/GeoDT, remove GeoXGB as main model, and write no-leak provenance into Stage 3 command flow.
- `other_outputs/aggregate_results.py`: remove GeoXGB from main aggregated workbooks.
- `other_outputs/generate_table.py`: remove GeoXGB from legacy/generated main table rows.
- `src/tests/test_no_leak_workflow_contract.py`: new fast contract tests for workflow naming, year split, model allowlist, and aggregation exclusion.
- `README.md`, `PIPELINE_WORKFLOW.md`, `INSTALL.md`: update user-facing main workflow docs to the no-leak split and GeoRF/GeoDT main scope.
- `docs/superpowers/specs/2026-06-15-no-leak-partition-workflow-design.md`: already written and committed; do not edit unless the design changes.

## Task 1: Add Failing Workflow Contract Tests

**Files:**
- Create: `src/tests/test_no_leak_workflow_contract.py`

- [ ] **Step 1: Create the contract test file**

Add this file exactly:

```python
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")


class NoLeakWorkflowContractTests(unittest.TestCase):
    def test_active_stage1_entrypoint_uses_partition_learning_window(self):
        new_path = REPO_ROOT / "run_batches_2018_2020_partition_learning_visual_monthly.bat"
        old_path = REPO_ROOT / "run_batches_2021_2024_visual_monthly.bat"

        self.assertTrue(new_path.exists(), "No-leak Stage 1 entrypoint is missing")
        self.assertFalse(old_path.exists(), "Old leaking Stage 1 entrypoint must not remain active")

        content = new_path.read_text(encoding="utf-8", errors="ignore").lower()
        self.assertIn("2018-2020", content)
        self.assertIn("for /l %%y in (2018,1,2020)", content)
        self.assertNotIn("for /l %%y in (2021,1,2024)", content)
        self.assertNotIn("geoxgb", content)
        self.assertNotIn("main_model_xgb.py", content)

    def test_stage2_main_workflow_excludes_geoxgb_and_points_to_new_stage1(self):
        content = read_text("spatial_weighted_consensus_clustering.bat").lower()

        self.assertIn("run_batches_2018_2020_partition_learning_visual_monthly.bat", content)
        self.assertNotIn("run_batches_2021_2024_visual_monthly.bat", content)
        self.assertNotIn("geoxgb", content)
        self.assertNotIn("geoxgbexperiment", content)

    def test_stage3_evaluates_2021_2024_but_all_mode_excludes_geoxgb(self):
        content = read_text("run_partition_k40_comparison_unified.bat").lower()

        self.assertIn("set start_month=2021-01", content)
        self.assertIn("set end_month=2024-12", content)
        self.assertIn("partition learning", content)
        self.assertIn("2018-2020", content)
        self.assertRegex(content, r"for %%m in \(georf geodt\) do")
        self.assertNotIn("for %%m in (georf geoxgb geodt) do", content)
        self.assertNotIn("compare_partitioned_vs_pooled_xgb", content)
        self.assertNotIn("result_partition_k40_compare_xgb", content)

    def test_main_aggregators_exclude_geoxgb(self):
        aggregate = read_text("other_outputs/aggregate_results.py")
        legacy_table = read_text("other_outputs/generate_table.py")

        self.assertNotIn("GeoXGB", aggregate)
        self.assertNotIn("_candidates(\"XGB\"", aggregate)
        self.assertNotIn("GeoXGB", legacy_table)
        self.assertNotIn("GeoXGBExperiment", legacy_table)

    def test_user_docs_do_not_advertise_old_stage1_or_geoxgb_main_workflow(self):
        docs = "\n".join(
            read_text(path)
            for path in ("README.md", "PIPELINE_WORKFLOW.md", "INSTALL.md")
            if (REPO_ROOT / path).exists()
        ).lower()

        self.assertNotIn("run_batches_2021_2024_visual_monthly.bat", docs)
        self.assertIn("run_batches_2018_2020_partition_learning_visual_monthly.bat", docs)
        self.assertNotRegex(docs, re.compile(r"model type[s]?:.*geoxgb"))
        self.assertNotIn("geoxgb partitions", docs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```bash
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected result before implementation:

```text
FAILED
```

At least one failure should mention the missing `run_batches_2018_2020_partition_learning_visual_monthly.bat` or the still-present old Stage 1 path.

- [ ] **Step 3: Commit the failing contract test**

Run:

```bash
git add src/tests/test_no_leak_workflow_contract.py
git commit -m "test: add no-leak workflow contract"
```

## Task 2: Replace Active Stage 1 Entrypoint

**Files:**
- Rename: `run_batches_2021_2024_visual_monthly.bat` -> `run_batches_2018_2020_partition_learning_visual_monthly.bat`
- Modify: `run_batches_2018_2020_partition_learning_visual_monthly.bat`

- [ ] **Step 1: Rename the active Stage 1 batch file**

Run:

```bash
git mv run_batches_2021_2024_visual_monthly.bat run_batches_2018_2020_partition_learning_visual_monthly.bat
```

- [ ] **Step 2: Update the header and usage**

In `run_batches_2018_2020_partition_learning_visual_monthly.bat`, replace the opening Stage 1 comments and usage model list so they describe 2018-2020 partition learning and only these model examples:

```bat
REM Stage 1 of 3: No-Leak Partition Candidate Learning Batch Script (2018-2020)
REM
REM Usage:
REM   run_batches_2018_2020_partition_learning_visual_monthly.bat georf
REM   run_batches_2018_2020_partition_learning_visual_monthly.bat geodt
REM   run_batches_2018_2020_partition_learning_visual_monthly.bat geodt --no-dt-rules
REM   run_batches_2018_2020_partition_learning_visual_monthly.bat georf --fs0-only
REM
REM This stage learns partition candidates on 2018-2020 only. Stage 3 applies
REM the learned partitions to the held-out 2021-2024 evaluation window.
```

- [ ] **Step 3: Remove GeoXGB from argument help**

In the missing-argument help block, make the model choices:

```bat
echo Model types:
echo   georf   - GeoRF Random Forest
echo   geodt   - GeoDT Decision Tree
```

And make invalid model messages say:

```bat
echo Valid options: georf, geodt
```

- [ ] **Step 4: Update year counts and hints**

Change standard and fs0 batch totals:

```bat
set "total_batches=36"
```

for fs0-only, and:

```bat
set "total_batches=108"
```

for standard fs1/fs2/fs3.

Update all user-facing year summaries to say:

```bat
echo   - Years: 2018-2020 (3 years)
echo Years processed: 2018-2020 (3 years, each with 12 months)
```

- [ ] **Step 5: Change the processing loop**

Replace:

```bat
for /L %%y in (2021,1,2024) do (
```

with:

```bat
for /L %%y in (2018,1,2020) do (
```

- [ ] **Step 6: Remove the GeoXGB model branch**

Delete the `geoxgb` block in `:load_model_config`:

```bat
if /i "%~1"=="geoxgb" (
    set "MODEL_DISPLAY=GeoXGB"
    set "ENTRYPOINT=app/main_model_XGB.py"
    set "RESULT_PREFIX=result_GeoXGB"
    set "RESULTS_CSV_PREFIX=results_df_xgb_gp_"
    set "PRED_CSV_PREFIX=y_pred_test_xgb_gp_"
    set "IS_DT=0"
    exit /b 0
)
```

- [ ] **Step 7: Run the Stage 1 contract test subset**

Run:

```bash
python3 -m unittest src/tests/test_no_leak_workflow_contract.py::NoLeakWorkflowContractTests.test_active_stage1_entrypoint_uses_partition_learning_window -v
```

If `unittest` does not accept pytest-style selectors in this environment, run:

```bash
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected: the Stage 1 test passes, while later tests may still fail.

- [ ] **Step 8: Commit Stage 1 rename and edits**

Run:

```bash
git add run_batches_2018_2020_partition_learning_visual_monthly.bat run_batches_2021_2024_visual_monthly.bat
git commit -m "fix: learn partitions on 2018 2020"
```

## Task 3: Narrow Stage 2 To GeoRF And GeoDT

**Files:**
- Modify: `spatial_weighted_consensus_clustering.bat`

- [ ] **Step 1: Update Stage 2 usage and docs in the batch file**

Replace examples with:

```bat
REM Usage:
REM   spatial_weighted_consensus_clustering.bat georf
REM   spatial_weighted_consensus_clustering.bat geodt
REM   spatial_weighted_consensus_clustering.bat georf --fs0-only
```

Set model choices help to:

```bat
echo Model types:
echo   georf   - GeoRF Random Forest
echo   geodt   - GeoDT Decision Tree
```

- [ ] **Step 2: Update Stage 1 hints**

Replace both Stage 1 hint assignments with the new script:

```bat
set "STAGE1_HINT=run_batches_2018_2020_partition_learning_visual_monthly.bat %MODEL_TYPE% --fs0-only"
```

and:

```bat
set "STAGE1_HINT=run_batches_2018_2020_partition_learning_visual_monthly.bat %MODEL_TYPE%"
```

- [ ] **Step 3: Remove GeoXGB result/archive glob branches**

Delete these branches:

```bat
if /i "%MODEL_TYPE%"=="geoxgb" set "RESULTS_GLOB=results_df_xgb_gp_fs0_*.csv"
if /i "%MODEL_TYPE%"=="geoxgb" set "RESULTS_GLOB=results_df_xgb_gp_fs1_*.csv results_df_xgb_gp_fs2_*.csv results_df_xgb_gp_fs3_*.csv"
if /i "%MODEL_TYPE%"=="geoxgb" set "ARCHIVE_GLOB=result_GeoXGB_*_fs0_*_visual"
if /i "%MODEL_TYPE%"=="geoxgb" set "ARCHIVE_GLOB=result_GeoXGB_*_fs1_*_visual result_GeoXGB_*_fs2_*_visual result_GeoXGB_*_fs3_*_visual"
```

- [ ] **Step 4: Remove GeoXGB load config**

Delete this block:

```bat
if /i "%~1"=="geoxgb" (
    set "MODEL_DISPLAY=GeoXGB"
    set "EXPERIMENT_DIR=GeoXGBExperiment"
    set "RESULTS_SUBDIR=GeoXgboostResults"
    exit /b 0
)
```

Make invalid model messages say:

```bat
echo Valid options: georf, geodt
```

- [ ] **Step 5: Run the Stage 2 contract test**

Run:

```bash
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected: the Stage 2 test no longer fails. Stage 3/docs/aggregation tests may still fail.

- [ ] **Step 6: Commit Stage 2 changes**

Run:

```bash
git add spatial_weighted_consensus_clustering.bat
git commit -m "fix: restrict consensus clustering to main models"
```

## Task 4: Narrow Stage 3 And Add No-Leak Provenance

**Files:**
- Modify: `run_partition_k40_comparison_unified.bat`

- [ ] **Step 1: Update Stage 3 usage**

Edit the header examples to remove GeoXGB and add the no-leak source note:

```bat
REM   run_partition_k40_comparison_unified.bat all                   (GeoRF + GeoDT)
REM   run_partition_k40_comparison_unified.bat georf --visual --month-ind
REM   run_partition_k40_comparison_unified.bat geodt 1 3 --visual
REM
REM Stage 3 evaluates 2021-2024 using partitions learned from the
REM 2018-2020 Stage 1 + Stage 2 partition-learning workflow.
```

Update model type help to:

```bat
echo Model types: georf, geodt, all
```

- [ ] **Step 2: Narrow all-mode loop**

Replace:

```bat
for %%M in (georf geoxgb geodt) do (
```

with:

```bat
for %%M in (georf geodt) do (
```

- [ ] **Step 3: Remove GeoXGB load config**

Delete the `geoxgb` branch in `:load_model_config`:

```bat
if /i "%~1"=="geoxgb" (
    set "MODEL_DISPLAY=GeoXGB"
    set "EXPERIMENT_DIR=GeoXGBExperiment"
    set "COMPARISON_SCRIPT=scripts\compare_partitioned_vs_pooled_xgb_k40_nc4.py"
    set "LOWER_MODEL_FLAG="
    set "OUT_DIR=.\result_partition_k40_compare_XGB"
    exit /b 0
)
```

Make invalid model messages say:

```bat
echo Valid options: georf, geodt, all
```

- [ ] **Step 4: Add partition/evaluation provenance variables**

Near `set START_MONTH=2021-01`, add:

```bat
set PARTITION_LEARNING_YEARS=2018-2020
set EVALUATION_YEARS=2021-2024
```

In the configuration echo block, add:

```bat
echo Partition learning: %PARTITION_LEARNING_YEARS% ^(fixed partitions from Stage 2^)
echo Evaluation window:  %START_MONTH% to %END_MONTH%
```

- [ ] **Step 5: Pass provenance to comparison scripts through environment**

Immediately before the `"%PYTHON_EXE%" !COMPARISON_SCRIPT! ^` call, add:

```bat
set NO_LEAK_PARTITION_LEARNING_YEARS=%PARTITION_LEARNING_YEARS%
set NO_LEAK_EVALUATION_YEARS=%EVALUATION_YEARS%
```

This lets Python scripts include the provenance in manifests in a later task without changing the CLI.

- [ ] **Step 6: Run Stage 3 contract test**

Run:

```bash
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected: the Stage 3 test no longer fails. Aggregation/docs tests may still fail.

- [ ] **Step 7: Commit Stage 3 changes**

Run:

```bash
git add run_partition_k40_comparison_unified.bat
git commit -m "fix: evaluate main models with no-leak partitions"
```

## Task 5: Record No-Leak Provenance In Stage 3 Manifests

**Files:**
- Modify: `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`

- [ ] **Step 1: Add manifest fields**

Find the `manifest = {` block and add these fields after `pipeline_version`:

```python
        'partition_learning_years': os.environ.get('NO_LEAK_PARTITION_LEARNING_YEARS', '2018-2020'),
        'evaluation_years': os.environ.get('NO_LEAK_EVALUATION_YEARS', '2021-2024'),
        'temporal_leakage_guard': 'partitions learned before evaluation window',
        'main_model_scope': 'GeoRF/GeoDT only; GeoXGB excluded from main results',
```

If the existing `pipeline_version` entry is last and has no trailing comma, add the comma first:

```python
        'pipeline_version': 'GeoRF_utilities_v1.0',
```

- [ ] **Step 2: Run Python syntax check**

Run:

```bash
python3 -m py_compile scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Commit manifest provenance**

Run:

```bash
git add scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py
git commit -m "fix: record no-leak evaluation provenance"
```

## Task 6: Remove GeoXGB From Main Aggregation

**Files:**
- Modify: `other_outputs/aggregate_results.py`
- Modify: `other_outputs/generate_table.py`

- [ ] **Step 1: Update `aggregate_results.py` candidate dictionaries**

Remove the GeoXGB entries from both `RESULT_CANDIDATES` dictionaries.

For fs0-only, keep:

```python
    RESULT_CANDIDATES = {
        ("GeoRF", 0): _candidates("GF", 0),
        ("GeoDT", 0): _candidates("DT", 0),
    }
```

For standard mode, keep:

```python
    RESULT_CANDIDATES = {
        ("GeoRF", 1): _candidates("GF", 1),
        ("GeoRF", 2): _candidates("GF", 2),
        ("GeoRF", 3): _candidates("GF", 3),
        ("GeoDT", 1): _candidates("DT", 1),
        ("GeoDT", 2): _candidates("DT", 2),
        ("GeoDT", 3): _candidates("DT", 3),
    }
```

- [ ] **Step 2: Update `aggregate_results.py` model rows**

Set the model list to:

```python
if FS0_ONLY:
    MODELS = ["GeoRF", "GeoDT"]
else:
    MODELS = ["GeoRF", "GeoDT", "FEWSNET (baseline)"]
```

- [ ] **Step 3: Update `generate_table.py` model rows**

Remove the GeoXGB load and rows from main aggregation. The resulting `MODEL_ROWS` blocks should be:

```python
if FS0_ONLY:
    MODEL_ROWS = [
        ("GeoRF", georf),
        ("GeoDT", geodt),
    ]
else:
    MODEL_ROWS = [
        ("GeoRF", georf),
        ("GeoDT", geodt),
        ("FEWSNET (baseline)", fewsnet),
    ]
```

Remove the line:

```python
geoxgb = load_model_data(os.path.join(BASE, "GeoXGBExperiment", "GeoXgboostResults"), "results_df_xgb_gp_")
```

- [ ] **Step 4: Run aggregation contract and syntax checks**

Run:

```bash
python3 -m py_compile other_outputs/aggregate_results.py other_outputs/generate_table.py
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected: aggregation test no longer fails. Docs tests may still fail.

- [ ] **Step 5: Commit aggregation changes**

Run:

```bash
git add other_outputs/aggregate_results.py other_outputs/generate_table.py
git commit -m "fix: exclude geoxgb from main aggregation"
```

## Task 7: Update User-Facing Main Workflow Docs

**Files:**
- Modify: `README.md`
- Modify: `PIPELINE_WORKFLOW.md`
- Modify: `INSTALL.md`

- [ ] **Step 1: Update README summary**

In `README.md`, replace the main title and opening description with GeoRF/GeoDT-only language:

```markdown
# GeoRF/GeoDT Food Crisis Prediction with No-Leak Spatial Consensus Clustering

Spatial transformation framework for food security crisis prediction using GeoRF and GeoDT with consensus-based spatial partitioning. The main results workflow learns partitions on 2018-2020 and evaluates fixed partitions on 2021-2024 to avoid temporal leakage. GeoXGB remains experimental and is excluded from main results.
```

- [ ] **Step 2: Update README quick-start commands**

Use these Stage 1 examples:

```markdown
run_batches_2018_2020_partition_learning_visual_monthly.bat georf
run_batches_2018_2020_partition_learning_visual_monthly.bat geodt
```

Use these Stage 2 examples:

```markdown
spatial_weighted_consensus_clustering.bat georf
spatial_weighted_consensus_clustering.bat geodt
```

Use these Stage 3 examples:

```markdown
run_partition_k40_comparison_unified.bat georf --visual --month-ind
run_partition_k40_comparison_unified.bat geodt --visual --month-ind
run_partition_k40_comparison_unified.bat all --visual --month-ind
```

- [ ] **Step 3: Update `PIPELINE_WORKFLOW.md`**

Replace the main workflow overview with these facts:

```markdown
Stage 1 partition learning: 2018-2020 monthly runs using `run_batches_2018_2020_partition_learning_visual_monthly.bat`.
Stage 2 consensus clustering: learns fixed partitions from Stage 1 outputs.
Stage 3 final evaluation: 2021-01 through 2024-12 using `run_partition_k40_comparison_unified.bat`.
Main model set: GeoRF and GeoDT. GeoXGB is excluded from main results.
```

Remove active command examples that run:

```text
run_batches_2021_2024_visual_monthly.bat
spatial_weighted_consensus_clustering.bat geoxgb
run_partition_k40_comparison_unified.bat geoxgb
```

- [ ] **Step 4: Update `INSTALL.md`**

Replace references to:

```text
run_batches_2021_2024_visual_monthly.bat <model>
```

with:

```text
run_batches_2018_2020_partition_learning_visual_monthly.bat <model>
```

and state that `<model>` is `georf` or `geodt` for main results.

- [ ] **Step 5: Run docs contract test**

Run:

```bash
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected: all contract tests pass unless remaining code references still need cleanup.

- [ ] **Step 6: Commit docs changes**

Run:

```bash
git add README.md PIPELINE_WORKFLOW.md INSTALL.md
git commit -m "docs: document no-leak main workflow"
```

## Task 8: Clean Main Workflow GeoXGB References In Support Scripts

**Files:**
- Modify as needed after scan: `other_outputs/plot_model_comparison.py`
- Modify as needed after scan: `scripts/plot_cluster_map_3x4.py`
- Modify as needed after scan: `scripts/plot_cluster_map_3x4_refined.py`
- Modify as needed after scan: `scripts/enable_visual_debug.py`

- [ ] **Step 1: Run focused scan**

Run:

```bash
rg -n "run_batches_2021_2024_visual_monthly|geoxgb|GeoXGB|result_partition_k40_compare_XGB" README.md PIPELINE_WORKFLOW.md INSTALL.md *.bat other_outputs scripts app specs --glob '!specs/001-*' --glob '!specs/002-*' --glob '!specs/003-*' --glob '!specs/004-*' --glob '!specs/005-*'
```

Expected: hits may remain in implementation modules such as `app/main_model_XGB.py` or `scripts/compare_partitioned_vs_pooled_xgb_k40_nc4.py`, but main workflow docs/scripts should not advertise GeoXGB as a current main model.

- [ ] **Step 2: Update plotting scripts that define main model order**

If `other_outputs/plot_model_comparison.py` still has:

```python
MODELS = ["GeoRF", "GeoXGB", "GeoDT"]
```

change it to:

```python
MODELS = ["GeoRF", "GeoDT"]
```

Remove GeoXGB entries from `RESULT_DIRS`, `COLORS`, and `MARKERS` in the same file.

- [ ] **Step 3: Update cluster map scripts that define main model order**

If `scripts/plot_cluster_map_3x4.py` or `scripts/plot_cluster_map_3x4_refined.py` still use:

```python
MODEL_ORDER = ("GeoDT", "GeoRF", "GeoXGB")
```

change to:

```python
MODEL_ORDER = ("GeoDT", "GeoRF")
```

Remove the GeoXGB entry from their model directory maps and update titles from `GeoDT / GeoRF / GeoXGB` to `GeoDT / GeoRF`.

- [ ] **Step 4: Update visual-debug helper script**

If `scripts/enable_visual_debug.py` still edits `app/main_model_XGB.py` as part of main workflow toggling, remove that file from the default main toggle set and update printed command suggestions to the new Stage 1 script with `georf` and `geodt` only.

- [ ] **Step 5: Run syntax checks for touched scripts**

Run only for files changed in this task. Example:

```bash
python3 -m py_compile other_outputs/plot_model_comparison.py scripts/plot_cluster_map_3x4.py scripts/plot_cluster_map_3x4_refined.py scripts/enable_visual_debug.py
```

Expected: no output and exit code 0.

- [ ] **Step 6: Commit support-script cleanup**

Run:

```bash
git add other_outputs/plot_model_comparison.py scripts/plot_cluster_map_3x4.py scripts/plot_cluster_map_3x4_refined.py scripts/enable_visual_debug.py
git commit -m "fix: remove geoxgb from main support scripts"
```

If a listed file did not need changes, omit it from `git add`.

## Task 9: Final Verification And Memory Note

**Files:**
- Modify: `/home/swl007007/.codex/memories/extensions/ad_hoc/notes/2026-06-15-food-crisis-main-workflow-controls.md` only if the user explicitly asks to update memory during implementation.

- [ ] **Step 1: Run contract tests**

Run:

```bash
python3 -m unittest discover -s src/tests -p 'test_no_leak_workflow_contract.py' -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run Python syntax checks**

Run:

```bash
python3 -m py_compile scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py other_outputs/aggregate_results.py other_outputs/generate_table.py
```

Add any Python support scripts touched in Task 8.

Expected: no output and exit code 0.

- [ ] **Step 3: Run legacy root filename scan**

Run:

```bash
find . -path './.git' -prune -o -path './.venv-geodt-diagnostic' -prune -o -type f -iname '*2021*2024*' -printf '%p\n'
```

Expected: no active root workflow file named `run_batches_2021_2024_visual_monthly.bat`. Historical result artifacts may require human review if the scan returns generated data paths.

- [ ] **Step 4: Run main workflow reference scan**

Run:

```bash
rg -n "run_batches_2021_2024_visual_monthly|for %%M in \\(georf geoxgb geodt\\)|Model types:.*geoxgb|GeoXGB.*main results|result_partition_k40_compare_XGB" README.md PIPELINE_WORKFLOW.md INSTALL.md *.bat other_outputs scripts
```

Expected: no active main workflow references. Any remaining hit must be either an explicitly experimental GeoXGB implementation script or a historical/excluded-artifact note.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional changes remain. Do not revert unrelated pre-existing dirty files unless the user explicitly asks.

- [ ] **Step 6: Final commit if needed**

If any final verification/documentation edits remain:

```bash
git add <changed-files>
git commit -m "chore: verify no-leak workflow cleanup"
```

## Self-Review Notes

- Spec coverage: Tasks cover Stage 1 2018-2020 partition learning, Stage 2 consumption/allowlist, Stage 3 2021-2024 evaluation, GeoXGB exclusion, docs cleanup, aggregation cleanup, and verification scans.
- Placeholder scan: No `TBD` or `TODO` placeholders are intentionally left in the plan.
- Type and path consistency: New Stage 1 filename is consistently `run_batches_2018_2020_partition_learning_visual_monthly.bat`; final evaluation remains `run_partition_k40_comparison_unified.bat`; main models are consistently `georf` and `geodt`.
