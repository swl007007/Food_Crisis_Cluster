# FEWS NET Ethiopia pre-GeoRF data audit

## 审计边界

本审计使用 checksum 固定的 assembled panel，精确筛选 `ISO3 == "ETH"`，
分析其进入 `load_and_preprocess_data()` 之前的状态。结果为 187,200 行、
88 列、1,040 个 admin、180 个月（2010-01 至 2024-12），且
`(FEWSNET_admin_code, date)` 无重复。未运行 GeoRF pipeline。

`fewsnet_eth_pre_georf.csv.gz` 是这一 parsed slice 的确定性压缩导出；
权威来源仍是外部完整 panel 和 `SOURCE_MANIFEST.csv` 中的源 SHA-256。

## 特征与缺失概览

88 个字段包含 ID/行政层级、时间和坐标、冲突事件及 5/10 月窗口、17 个
AEZ dummy、遥感与环境、土壤/地形/市场可达性、价格与宏观、IPC 标签和
provider projection、人口以及气温/降水 z-score。完整逐列解释见
`column_profile.csv` 和 `pipeline_column_lineage.csv`。

原始面板是完整的 1,040 × 180 平衡 admin-month 网格；缺失来自字段本身，
不是丢失整行。≥20% effective missing 的字段如下：

```csv
column,feature_family,effective_missing_rate
fews_proj_near_ha,ipc_target_projection,0.772778
fews_proj_med_ha,ipc_target_projection,0.772778
fews_ha,ipc_target_projection,0.772206
fews_proj_med_adjusted,ipc_target_projection,0.750844
fews_proj_med,ipc_target_projection,0.750844
fews_proj_near,ipc_target_projection,0.750844
fews_ipc,ipc_target_projection,0.722756
fews_ipc_crisis,ipc_target_projection,0.722756
fews_ipc_adjusted,ipc_target_projection,0.722756
pop,agriculture_population,0.716667
gini,macro,0.600000
```

主要 pattern 是制度性时间块：IPC 标签只在 51/180 个月发布；`gini` 从
2016 起缺失；Food CPI 两列从 2023-07 起缺失；2024 年 GDP 缺失；WFP
价格存在早期/中期整块缺失。`Tair_zscore` 与 `Rainf_zscore` 各有 2,880
个 infinity，集中于固定 16 个 admin，production imputation 会把它们按
non-finite/missing 处理。详细见 missingness 与 nonfinite 系列 CSV。

## 简单过拟合检查

诊断 RF 排除了 IPC 标签、IPC phase、humanitarian-area 与 provider
projection 字段；预处理仅在训练集拟合 median imputer，threshold 只用
2019-2020 validation 选择。它仍保留 contemporaneous assembled-row
covariates，因此不是 forecast-time-safe baseline，也不是 GeoRF reproduction。

时间阻断结果：

```csv
candidate,split,n,prevalence,threshold,precision,recall,f1,average_precision,brier
unrestricted_with_admin_id,train,34219,0.164558,0.250000,0.982723,1.000000,0.991286,1.000000,0.006349
unrestricted_with_admin_id,validation,6240,0.145353,0.250000,0.527778,0.649394,0.582304,0.597483,0.089797
unrestricted_with_admin_id,test,11441,0.283542,0.250000,0.541298,0.678792,0.602298,0.622491,0.163594
unrestricted_without_admin_id,train,34219,0.164558,0.260000,0.985992,1.000000,0.992947,1.000000,0.006232
unrestricted_without_admin_id,validation,6240,0.145353,0.260000,0.537313,0.635061,0.582112,0.609261,0.089193
unrestricted_without_admin_id,test,11441,0.283542,0.260000,0.546871,0.670777,0.602520,0.633620,0.161189
regularized_without_admin_id,train,34219,0.164558,0.200000,0.584248,0.955070,0.724993,0.904031,0.057429
regularized_without_admin_id,validation,6240,0.145353,0.200000,0.491948,0.673649,0.568637,0.595640,0.093460
regularized_without_admin_id,test,11441,0.283542,0.200000,0.519190,0.679716,0.588706,0.628185,0.171973
```

核心信号：unrestricted/no-admin-ID probe 的 train F1 接近 1，而 temporal
test F1 约 0.584、AP 约 0.616、Brier 约 0.164；固定 regularization 将
train F1 降至约 0.760，但 test F1 仍约 0.581。这个结果支持“无约束 RF 在
该 assembled-row 诊断上存在强 capacity/memorization gap”，但不是正式
GeoRF 过拟合裁决。

随机 row split sensitivity：

```csv
split,n,threshold,f1,average_precision,brier
train,31140,0.360000,0.999659,1.000000,0.007693
validation,10380,0.360000,0.832490,0.914537,0.053175
test,10380,0.360000,0.827307,0.912816,0.053710
```

随机 test F1 约 0.831，显著高于 temporal test；且 train/test 都包含全部
1,040 个 admin。这说明随机行切分会严重乐观，不能作为时间或空间泛化证据。

冻结 GeoRF ETH temporal-test 结果（未重训）：

```csv
forecasting_horizon,model,n,precision,recall,f1,average_precision,brier
4,pooled,11441,0.708871,0.524661,0.603012,0.702749,0.133236
4,partitioned,11441,0.723290,0.580148,0.643859,0.737685,0.125060
4,partitioned_thresholded,11441,0.643883,0.746301,0.691319,0.737685,0.125060
8,pooled,11441,0.694107,0.493835,0.577089,0.668039,0.141514
8,partitioned,11441,0.737892,0.558878,0.636029,0.724681,0.130910
8,partitioned_thresholded,11441,0.644322,0.687423,0.665175,0.724681,0.130910
12,pooled,11441,0.645570,0.440197,0.523460,0.586810,0.157351
12,partitioned,11441,0.671147,0.492602,0.568178,0.654901,0.146852
12,partitioned_thresholded,11441,0.584651,0.615290,0.599579,0.654901,0.146852
```

稳定月份（排除只有 1 个标签的 2021-06）中，partitioned 相对 pooled 的
逐月 F1 胜场为：

```csv
scope,forecasting_horizon,stable_months_n_ge_30,partitioned_beats_pooled_months,thresholded_beats_partitioned_months,excluded_low_support_months
1,4,11,8,7,1
2,8,11,9,8,1
3,12,11,6,5,1
```

因此现有 held-out GeoRF 结果没有呈现“partitioned 在 ETH 上整体崩溃”，
但没有对应 GeoRF train/OOF metrics，不能直接计算 GeoRF train-test gap；
小 partition、低 partition stability、forecast-origin availability 仍是风险。

## 结论边界

- 已支持：原始 assembled ETH slice 的缺失主要是时间/字段块结构；存在
  non-finite z-score、常量/近常量和长尾特征。
- 已支持：无约束 lightweight RF 有明显 in-sample 到 temporal holdout gap；
  random row split 明显乐观。
- 未证明：正式 GeoRF 已过拟合。要形成该 finding，仍需相同 GeoRF 的
  train/validation/OOF 与 fixed temporal test 对照，以及 forecast-time-safe
  feature contract。
