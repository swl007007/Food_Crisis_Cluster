# Temporal Data Splits Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact reviewer-response temporal split note and audit CSV under `final_artifacts_in_paper_updated/`.

**Architecture:** This is a documentation artifact change only. The Markdown note presents compact appendix-ready tables and formulas; the CSV companion enumerates the actually evaluated final-test months only: February, June, and October for 2021-2024 across fs1/fs2/fs3.

**Tech Stack:** Markdown, CSV, Python standard library/pandas for deterministic date generation, existing repository artifacts for verification.

---

## File Structure

- Create: `final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md`
  - Responsibility: bilingual reviewer note with Chinese audit first and English appendix-ready text last.
- Create: `final_artifacts_in_paper_updated/temporal_data_splits_table.csv`
  - Responsibility: audit companion with 36 computed rows for the evaluated 2021-2024 February/June/October target months and scopes `fs1`, `fs2`, `fs3`.
- Reference only: `docs/superpowers/specs/2026-06-16-temporal-data-splits-note-design.md`
  - Responsibility: approved design source.
- Reference only: `src/customize/customize.py`
  - Responsibility: rolling train/test date formula.
- Reference only: `GeoRFExperiment/*` and `GeoDTExperiment/*`
  - Responsibility: current Stage 2 plan counts and consensus map manifests.

---

### Task 1: Reconfirm Evidence Before Writing Final Artifacts

**Files:**
- Reference: `src/customize/customize.py`
- Reference: `GeoRFExperiment/linked_tables/main_index.csv`
- Reference: `GeoDTExperiment/linked_tables/main_index.csv`
- Reference: `GeoRFExperiment/knn_sparsification_results/cluster_mapping_manifest.json`
- Reference: `GeoDTExperiment/knn_sparsification_results/cluster_mapping_manifest.json`

- [ ] **Step 1: Verify rolling-window constants and validation settings**

Run:

```bash
sed -n '328,420p' src/customize/customize.py
sed -n '228,252p' config.py
```

Expected:

```text
train_end = test_month_start - pd.DateOffset(months=active_lag)
train_start = train_end - pd.DateOffset(months=train_window_months - 1)
train_mask = (dates >= train_start) & (dates < train_end)
test_mask = (dates >= test_month_start) & (dates < test_month_end)
VAL_RATIO = 0.20
GROUP_SPLIT ... random_state: 42
```

- [ ] **Step 2: Verify current Stage 2 plan counts**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd

for model in ["GeoRF", "GeoDT"]:
    exp = Path(f"{model}Experiment")
    main = pd.read_csv(exp / "linked_tables" / "main_index.csv")
    print(model, "rows", len(main))
    print("years", sorted(main["year"].unique().tolist()))
    print("months", sorted(main["month"].astype(int).unique().tolist()))
    print("scopes", sorted(main["forecasting_scope"].unique().tolist()))
    for summary_path in sorted(exp.glob("similarity_matrices*/summary_statistics*.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        print(summary_path, "n_plans", data["n_plans"], "month_filter", data["month_filter"])
    manifest = json.loads((exp / "knn_sparsification_results" / "cluster_mapping_manifest.json").read_text(encoding="utf-8"))
    print("maps", sorted(manifest.keys()))
PY
```

Expected:

```text
GeoRF rows 24
months [2, 6, 10]
GeoRFExperiment/similarity_matrices/summary_statistics.json n_plans 24 month_filter None
GeoRFExperiment/similarity_matrices_m02/summary_statistics_m02.json n_plans 8 month_filter 2
GeoRFExperiment/similarity_matrices_m06/summary_statistics_m06.json n_plans 8 month_filter 6
GeoRFExperiment/similarity_matrices_m10/summary_statistics_m10.json n_plans 8 month_filter 10
maps ['general', 'm02', 'm06', 'm10']
GeoDT rows 27
months [2, 6, 10]
GeoDTExperiment/similarity_matrices/summary_statistics.json n_plans 27 month_filter None
GeoDTExperiment/similarity_matrices_m02/summary_statistics_m02.json n_plans 9 month_filter 2
GeoDTExperiment/similarity_matrices_m06/summary_statistics_m06.json n_plans 9 month_filter 6
GeoDTExperiment/similarity_matrices_m10/summary_statistics_m10.json n_plans 9 month_filter 10
maps ['general', 'm02', 'm06', 'm10']
```

---

### Task 2: Generate the Companion CSV

**Files:**
- Create: `final_artifacts_in_paper_updated/temporal_data_splits_table.csv`

- [ ] **Step 1: Generate deterministic 36-row CSV**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("final_artifacts_in_paper_updated/temporal_data_splits_table.csv")
scope_lags = [("fs1", 4), ("fs2", 8), ("fs3", 12)]
rows = []

for year in range(2021, 2025):
    for month in (2, 6, 10):
        target = pd.Period(f"{year}-{month:02d}", freq="M")
        target_start = target.to_timestamp()
        target_end_exclusive = (target + 1).to_timestamp()
        calendar_month = int(target.month)
        if calendar_month == 2:
            stage3_partition_map_rule = "m2 month-specific partition"
            stage2_consensus_input_filter = "month == 2"
        elif calendar_month == 6:
            stage3_partition_map_rule = "m6 month-specific partition"
            stage2_consensus_input_filter = "month == 6"
        elif calendar_month == 10:
            stage3_partition_map_rule = "m10 month-specific partition"
            stage2_consensus_input_filter = "month == 10"
        else:
            raise AssertionError(f"Unexpected evaluated month: {calendar_month}")

        for scope, horizon in scope_lags:
            train_end = target_start - pd.DateOffset(months=horizon)
            train_start = train_end - pd.DateOffset(months=36 - 1)
            rows.append({
                "target_month": str(target),
                "forecasting_scope": scope,
                "horizon_months": horizon,
                "stage3_train_start_inclusive": train_start.date().isoformat(),
                "stage3_train_end_exclusive": train_end.date().isoformat(),
                "stage3_train_end_inclusive": (train_end - pd.DateOffset(days=1)).date().isoformat(),
                "stage3_final_test_start_inclusive": target_start.date().isoformat(),
                "stage3_final_test_end_exclusive": target_end_exclusive.date().isoformat(),
                "stage3_final_test_end_inclusive": (target_end_exclusive - pd.DateOffset(days=1)).date().isoformat(),
                "stage1_split_acceptance_validation_data": (
                    "Internal validation subset drawn from that Stage 1 run's rolling training window; "
                    "current GROUP_SPLIT uses val_ratio=0.20, min_val_per_group=1, "
                    "skip_singleton_groups=True, random_state=42."
                ),
                "stage2_consensus_data": (
                    "2018-2020 linked Stage 1 partition plans only; current artifacts contain "
                    "GeoRF general=24, GeoRF month-specific=8 per m2/m6/m10, "
                    "GeoDT general=27, GeoDT month-specific=9 per m2/m6/m10."
                ),
                "stage2_consensus_input_filter": stage2_consensus_input_filter,
                "stage3_partition_map_rule": stage3_partition_map_rule,
                "threshold_selection_data": (
                    "No separate threshold-selection data in standard Stage 3; comparison uses classifier hard predictions."
                ),
                "hyperparameter_selection_data": (
                    "GeoRF: fixed hyperparameters. GeoDT Stage 1: max_depth selected on the internal validation subset "
                    "using class-1 F1 when DT_MAX_DEPTH_CANDIDATES is enabled. Stage 3 comparison: fixed RF/DT parameters."
                ),
                "final_test_data": f"{target} target-month observations only",
            })

df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(out)
print(df.shape)
print(df.head(3).to_string(index=False))
print(df.tail(3).to_string(index=False))
PY
```

Expected:

```text
final_artifacts_in_paper_updated/temporal_data_splits_table.csv
(36, 16)
```

- [ ] **Step 2: Verify row count, scope coverage, and date sentinels**

Run:

```bash
python3 - <<'PY'
import pandas as pd

path = "final_artifacts_in_paper_updated/temporal_data_splits_table.csv"
df = pd.read_csv(path)
assert len(df) == 36, len(df)
assert df["target_month"].nunique() == 12, df["target_month"].nunique()
assert sorted(df["forecasting_scope"].unique().tolist()) == ["fs1", "fs2", "fs3"]
assert df["stage3_partition_map_rule"].value_counts().to_dict() == {
    "m2 month-specific partition": 12,
    "m6 month-specific partition": 12,
    "m10 month-specific partition": 12,
}, df["stage3_partition_map_rule"].value_counts().to_dict()
assert not df["stage3_partition_map_rule"].str.contains("general", case=False).any()

checks = {
    ("2021-02", "fs1"): ("2017-11-01", "2020-10-01", "2021-02-01", "2021-03-01"),
    ("2021-02", "fs2"): ("2017-07-01", "2020-06-01", "2021-02-01", "2021-03-01"),
    ("2021-02", "fs3"): ("2017-03-01", "2020-02-01", "2021-02-01", "2021-03-01"),
    ("2024-10", "fs1"): ("2021-07-01", "2024-06-01", "2024-10-01", "2024-11-01"),
    ("2024-10", "fs2"): ("2021-03-01", "2024-02-01", "2024-10-01", "2024-11-01"),
    ("2024-10", "fs3"): ("2020-11-01", "2023-10-01", "2024-10-01", "2024-11-01"),
}
for key, expected in checks.items():
    row = df[(df["target_month"] == key[0]) & (df["forecasting_scope"] == key[1])].iloc[0]
    got = (
        row["stage3_train_start_inclusive"],
        row["stage3_train_end_exclusive"],
        row["stage3_final_test_start_inclusive"],
        row["stage3_final_test_end_exclusive"],
    )
    assert got == expected, (key, got, expected)
print("CSV temporal split checks passed")
PY
```

Expected:

```text
CSV temporal split checks passed
```

---

### Task 3: Write the Compact Markdown Note

**Files:**
- Create: `final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md`

- [ ] **Step 1: Create Markdown note with Chinese audit and English appendix**

Use `apply_patch` to add `final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md` with this content:

```markdown
# Temporal Data Splits Schematic Note

## 中文审查记录

### 写作原则

本 note 只描述当前 no-leak GeoRF/GeoDT 主 workflow 已实现的数据使用方式。appendix 不枚举全部 configured candidate target-month x horizon 组合；appendix 使用公式表、stage-level table 和 target calendar month map-selection table 区分 configured candidate window 和 current evaluated result rows。当前 36 行核查表保存在同一文件夹的 `temporal_data_splits_table.csv`，作为 artifact-level audit companion。

### 数据窗口总览

- Stage 1 partition learning 只使用 2018-2020 target months，对每个运行月和 horizon 做 rolling temporal split。
- Stage 1 recursive split acceptance 使用该运行 rolling training window 内部的 validation subset；当前 `GROUP_SPLIT` 为 `val_ratio=0.20`、`min_val_per_group=1`、`skip_singleton_groups=True`、`random_state=42`。
- Stage 2 consensus clustering 只使用 Stage 1 产出的 linked partition plans 和对应 performance-derived weights，不使用 2021-2024 final test outcomes。
- 当前 Stage 2 产物中，GeoRF general consensus 使用 24 个 linked plans，m2/m6/m10 各使用 8 个 linked plans；GeoDT general consensus 使用 27 个 linked plans，m2/m6/m10 各使用 9 个 linked plans。
- Stage 3 configured candidate loop 覆盖 2021-01 到 2024-12，run manifest 记录 `n_test_months=48`。当前有 evaluated result rows 的 target months 只有 2021-2024 年的 February、June、October，run manifest 记录 `n_test_months_evaluated=12` per scope。对 target month `T` 和 horizon `h`，当前代码使用 `[T - h - 35 months, T - h)` 作为 rolling training mask，使用 `[T, T + 1 month)` 作为 final test mask。

### 不写入 appendix 的额外承诺

- 当前 standard Stage 3 comparison 没有单独的 threshold-selection data，也没有 probability calibration 或 threshold tuning；脚本使用 classifier hard predictions。
- GeoRF 当前不使用单独的 hyperparameter-selection data。GeoDT Stage 1 在 rolling training window 内部 validation subset 上用 class-1 F1 选择 `max_depth`。Stage 3 comparison 使用固定 RF/DT 参数。
- 本 note 不声称新增 k sensitivity、spatial-kernel sensitivity、或 2021-2024 final-test-period hyperparameter tuning。

## Appendix: Temporal Data-Use Schematic and Split Rules

The main workflow uses a no-leak temporal separation between partition learning, consensus construction, and final evaluation. Stage 1 learns recursive partition candidates on 2018-2020 target months. Stage 2 constructs fixed consensus maps from those Stage 1 partition plans. Stage 3 is configured with a candidate target-month loop from 2021-01 through 2024-12 (`n_test_months=48` in run manifests), but current evaluated result rows exist only for February, June, and October in each year (`n_test_months_evaluated=12` per scope).

```text
Stage 1: 2018-2020 partition-learning runs
    rolling temporal train window -> internal validation subset -> split acceptance
        |
        v
Stage 2: consensus maps from 2018-2020 linked plans only
    general map + month-specific maps for February, June, and October
        |
        v
Stage 3: configured 2021-01..2024-12 candidate loop
    current evaluated rows: February, June, October only
    rolling temporal train window -> fixed model parameters -> final target month T
```

### Table A1. Stage-Level Data Use

| Component | Data window | Data role | Current implementation detail | Leakage guard |
|---|---|---|---|---|
| Stage 1 partition learning | 2018-2020 target months | Learns recursive partition candidates | Each run uses a rolling temporal training window and an internal validation subset for split acceptance. | Final 2021-2024 test outcomes are not used to learn partition candidates. |
| Stage 2 consensus clustering | Linked Stage 1 plans from 2018-2020 only | Builds fixed general and month-specific consensus maps | Current artifacts contain GeoRF general=24 plans and GeoRF m2/m6/m10=8 plans each; GeoDT general=27 plans and GeoDT m2/m6/m10=9 plans each. | Consensus maps are created before Stage 3 final evaluation and do not use 2021-2024 outcomes. |
| Stage 3 final evaluation | Configured candidate window: 2021-01 through 2024-12; current evaluated result rows: February, June, October for 2021-2024 | Tests pooled and fixed-partition local models | Each evaluated target month uses the rolling training rule in Table A2 and tests on the target month only. | Fixed partitions from Stage 2 are applied; final test labels are used only for evaluation. |
| Threshold selection | Not a separate data split in the standard comparison | Not tuned in this workflow | Standard Stage 3 comparison uses classifier hard predictions. | No final-test-period threshold tuning is performed. |
| Hyperparameter selection | Model-specific | GeoRF fixed; GeoDT Stage 1 depth selection | GeoRF uses fixed hyperparameters. GeoDT Stage 1 selects `max_depth` on the internal validation subset using class-1 F1 when `DT_MAX_DEPTH_CANDIDATES` is enabled. Stage 3 comparison uses fixed RF/DT parameters. | GeoDT Stage 1 selection is confined to the Stage 1 rolling training window; Stage 3 does not tune on final test labels. |

### Table A2. Horizon-Specific Rolling Split Rule for Stage 3

For any final-evaluation target month `T` and horizon `h`, the implemented monthly split uses:

```text
train_end   = T - h months
train_start = train_end - 35 months
TRAIN mask  = [train_start, train_end)
TEST mask   = [T, T + 1 month)
```

| Forecasting scope | Horizon `h` | Stage 3 training period for target month `T` | Split-acceptance validation data | Final test data |
|---|---:|---|---|---|
| fs1 | 4 months | `[T - 39 months, T - 4 months)` | Stage 1 uses an internal validation subset from the corresponding Stage 1 rolling training window. Current group-aware split settings are `val_ratio=0.20`, `min_val_per_group=1`, `skip_singleton_groups=True`, `random_state=42`. | `[T, T + 1 month)` |
| fs2 | 8 months | `[T - 43 months, T - 8 months)` | Same validation rule as above, applied inside each Stage 1 partition-learning run. | `[T, T + 1 month)` |
| fs3 | 12 months | `[T - 47 months, T - 12 months)` | Same validation rule as above, applied inside each Stage 1 partition-learning run. | `[T, T + 1 month)` |

For example, the implemented rule maps evaluated `T=2021-02` and `fs1` to a Stage 3 training mask of `[2017-11-01, 2020-10-01)` and a final test mask of `[2021-02-01, 2021-03-01)`.

### Table A3. Stage 3 Partition Map Selection by Target Calendar Month

| Target calendar month | Stage 3 partition map used | Stage 2 consensus input filter |
|---|---|---|
| January | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| February | February-specific `m2` partition | `month == 2` |
| March | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| April | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| May | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| June | June-specific `m6` partition | `month == 6` |
| July | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| August | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| September | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| October | October-specific `m10` partition | `month == 10` |
| November | General partition if evaluated in a future run; no current final-test result rows | No month filter |
| December | General partition if evaluated in a future run; no current final-test result rows | No month filter |

A machine-readable audit companion, `temporal_data_splits_table.csv`, is included in this artifact folder. It enumerates the current evaluated result rows only: February, June, and October for 2021-2024 across fs1/fs2/fs3, for 36 rows using the same date formula. Non-2/6/10 months are configured candidates that would use the general partition if evaluated, but they do not have final test rows in the current results.
```

- [ ] **Step 2: Check the note does not overclaim unimplemented methods**

Run:

```bash
rg -n "tuned|calibration|AUC|log-loss|sensitivity|2021-2024 outcomes are used|all candidate rows are printed" final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md
```

Expected:

```text
Matches only in negative statements such as "not tuned", "no probability calibration", "not claim", or "not printed".
```

---

### Task 4: Verify Final Artifacts

**Files:**
- Verify: `final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md`
- Verify: `final_artifacts_in_paper_updated/temporal_data_splits_table.csv`

- [ ] **Step 1: Verify files exist and CSV schema is stable**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

md = Path("final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md")
csv = Path("final_artifacts_in_paper_updated/temporal_data_splits_table.csv")
assert md.exists(), md
assert csv.exists(), csv

df = pd.read_csv(csv)
expected_columns = [
    "target_month",
    "forecasting_scope",
    "horizon_months",
    "stage3_train_start_inclusive",
    "stage3_train_end_exclusive",
    "stage3_train_end_inclusive",
    "stage3_final_test_start_inclusive",
    "stage3_final_test_end_exclusive",
    "stage3_final_test_end_inclusive",
    "stage1_split_acceptance_validation_data",
    "stage2_consensus_data",
    "stage2_consensus_input_filter",
    "stage3_partition_map_rule",
    "threshold_selection_data",
    "hyperparameter_selection_data",
    "final_test_data",
]
assert df.columns.tolist() == expected_columns, df.columns.tolist()
assert len(df) == 36, len(df)
print("artifact schema checks passed")
PY
```

Expected:

```text
artifact schema checks passed
```

- [ ] **Step 2: Verify appendix compactness and bilingual structure**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md").read_text(encoding="utf-8")
assert "## 中文审查记录" in text
assert "## Appendix: Temporal Data-Use Schematic and Split Rules" in text
assert text.index("## 中文审查记录") < text.index("## Appendix: Temporal Data-Use Schematic and Split Rules")
assert "A machine-readable audit companion" in text
assert text.count("| fs1 |") == 1
assert text.count("| fs2 |") == 1
assert text.count("| fs3 |") == 1
print("markdown structure checks passed")
PY
```

Expected:

```text
markdown structure checks passed
```

- [ ] **Step 3: Run repository diff checks**

Run:

```bash
git diff --check -- final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md final_artifacts_in_paper_updated/temporal_data_splits_table.csv
git status --short
```

Expected:

```text
No whitespace errors.
?? final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md
?? final_artifacts_in_paper_updated/temporal_data_splits_table.csv
```

---

### Task 5: Commit Final Artifacts

**Files:**
- Commit: `final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md`
- Commit: `final_artifacts_in_paper_updated/temporal_data_splits_table.csv`

- [ ] **Step 1: Commit the final artifacts**

Run:

```bash
git add final_artifacts_in_paper_updated/temporal_data_splits_schematic_note.md final_artifacts_in_paper_updated/temporal_data_splits_table.csv
git commit -m "documents temporal data splits"
```

Expected:

```text
[main <hash>] documents temporal data splits
 2 files changed
```

- [ ] **Step 2: Confirm clean final state except expected branch-ahead commits**

Run:

```bash
git status --short --branch
```

Expected:

```text
## main...origin/main [ahead 2]
```
