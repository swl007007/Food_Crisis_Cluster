# Temporal Data Splits Schematic Note

## 中文审查记录

### 写作原则

本 note 只描述当前 no-leak GeoRF/GeoDT 主 workflow 已实现的数据使用方式。appendix 不枚举全部 configured candidate target-month x horizon 组合；appendix 使用公式表、stage-level table 和 target calendar month map-selection table 区分 configured candidate window 和 current evaluated result rows。当前 36 行核查表保存在同一文件夹的 `temporal_data_splits_table.csv`，作为 artifact-level audit companion。

### 数据窗口总览

- Stage 1 partition learning 只使用 2018-2020 target months，对每个运行月和 horizon 做 rolling temporal split。
- Stage 1 recursive split acceptance 使用该运行 rolling training window 内部的 validation subset；当前 `GROUP_SPLIT` 为 `val_ratio=0.20`、`min_val_per_group=1`、`skip_singleton_groups=True`、`random_state=42`。
- Stage 2 consensus clustering 只使用 Stage 1 产出的 linked partition plans 和对应 performance-derived weights，不使用 2021-2024 final test outcomes。
- 当前 Stage 2 产物中，GeoRF general consensus 使用 24 个 linked plans，m2/m6/m10 各使用 8 个 linked plans；GeoDT general consensus 使用 27 个 linked plans，m2/m6/m10 各使用 9 个 linked plans。
- Stage 3 configured candidate loop 覆盖 2021-01 到 2024-12，run manifest 记录 `n_test_months=48`。当前有 evaluated result rows 的 target months 只有 2021-2024 年的 February、June、October，run manifest 记录 `n_test_months_evaluated=12` per forecasting horizon。对 target month `T` 和 horizon `h`，当前代码使用 `[T - h - 35 months, T - h)` 作为 rolling training mask，使用 `[T, T + 1 month)` 作为 final test mask。
- 当前 `temporal_data_splits_table.csv` 只列出 actual evaluated rows：2021-2024 年 February、June、October x 4-month、8-month、12-month horizon，共 36 行。这些 rows 对应 m2/m6/m10 month-specific maps；general maps 只用于说明非 2/6/10 月如果在未来被 evaluation 覆盖时的配置规则。

### 不写入 appendix 的额外承诺

- 当前 standard Stage 3 comparison 没有单独的 threshold-selection data，也没有 probability calibration 或 threshold tuning；脚本使用 classifier hard predictions。
- GeoRF 当前不使用单独的 hyperparameter-selection data。GeoDT Stage 1 在 rolling training window 内部 validation subset 上用 class-1 F1 选择 `max_depth`。Stage 3 comparison 使用固定 RF/DT 参数。
- 本 note 不声称新增 k sensitivity、spatial-kernel sensitivity、或 2021-2024 final-test-period hyperparameter tuning。

## Appendix: Temporal Data-Use Schematic and Split Rules

The main workflow uses a no-leak temporal separation between partition learning, consensus construction, and final evaluation. Stage 1 learns recursive partition candidates on 2018-2020 target months. Stage 2 constructs fixed consensus maps from those Stage 1 partition plans. Stage 3 is configured with a candidate target-month loop from 2021-01 through 2024-12 (`n_test_months=48` in run manifests), but current evaluated result rows exist only for February, June, and October in each year (`n_test_months_evaluated=12` per forecasting horizon).

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

| Forecasting horizon | Horizon `h` | Stage 3 training period for target month `T` | Split-acceptance validation data | Final test data |
|---|---:|---|---|---|
| 4-month horizon | 4 months | `[T - 39 months, T - 4 months)` | Stage 1 uses an internal validation subset from the corresponding Stage 1 rolling training window. Current group-aware split settings are `val_ratio=0.20`, `min_val_per_group=1`, `skip_singleton_groups=True`, `random_state=42`. | `[T, T + 1 month)` |
| 8-month horizon | 8 months | `[T - 43 months, T - 8 months)` | Same validation rule as above, applied inside each Stage 1 partition-learning run. | `[T, T + 1 month)` |
| 12-month horizon | 12 months | `[T - 47 months, T - 12 months)` | Same validation rule as above, applied inside each Stage 1 partition-learning run. | `[T, T + 1 month)` |

For example, the implemented rule maps evaluated `T=2021-02` and the 4-month horizon to a Stage 3 training mask of `[2017-11-01, 2020-10-01)` and a final test mask of `[2021-02-01, 2021-03-01)`.

### Table A3. Stage 3 Partition Map Selection by Target Calendar Month

| Target calendar month | Stage 3 partition map used | Stage 2 consensus input filter | Current result-row status |
|---|---|---|---|
| January | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| February | February-specific `m2` partition | `month == 2` | Current evaluated result rows exist |
| March | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| April | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| May | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| June | June-specific `m6` partition | `month == 6` | Current evaluated result rows exist |
| July | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| August | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| September | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| October | October-specific `m10` partition | `month == 10` | Current evaluated result rows exist |
| November | General partition if evaluated in a future run | No month filter | No current final-test result rows |
| December | General partition if evaluated in a future run | No month filter | No current final-test result rows |

A machine-readable audit companion, `temporal_data_splits_table.csv`, is included in this artifact folder. It enumerates the current evaluated result rows only: February, June, and October for 2021-2024 across the 4-month, 8-month, and 12-month horizons, for 36 rows using the same date formula. These rows use the m2/m6/m10 month-specific maps. Non-2/6/10 months are configured candidates that would use the general partition if evaluated, but they do not have final test rows in the current results.
