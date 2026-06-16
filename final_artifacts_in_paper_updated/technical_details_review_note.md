# Technical Details Review Note

## 中文审查记录

### 写作原则

本 note 只描述当前代码和现有产物已经实现的技术细节。凡是审稿意见中提出、但当前代码没有实现或现有产物不能直接支撑的内容，不写入最后的英文 appendix，而是单独放入 `technical_details_unimplemented_patch_note.md` 作为内部 patch 候选。

证据入口如下：

- `config.py`
- `src/partition/transformation.py`
- `src/partition/partition_opt.py`
- `src/tests/sig_test.py`
- `scripts/step4_similarity_matrix.py`
- `scripts/step5_sparsification.py`
- `scripts/step6_complete_clustering_pipeline.py`
- `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py`
- `src/model/model_RF.py`
- `PIPELINE_WORKFLOW.md`

### 4. Recursive partitioning 逻辑

当前实现不是基于 RF 特征阈值的 exhaustive tree split，也不是直接优化 AUC、log loss 或 Gini impurity。Stage 1 的递归 partitioning 在每个 active branch 上使用 validation predictions 形成 polygon/group-level 的 crisis-class miss summary，然后用 scan-statistic 式的目标函数选择一个 binary split。

实现路径是 `src/partition/transformation.py` 调用 `get_class_wise_stat(...)` 和 `scan(...)`。在当前 crisis-focused 设置下，`GOVERNING_METRIC='class_1_f1'`，`SELECT_CLASS=np.array([1])`。代码中的 per-sample scoring 对真正 crisis case 且预测为 crisis 的样本记为 1，其他样本记为 0；因此 group summary 聚焦于 crisis class 的 missed/correctly recovered cases，而不是全部类别的平均准确率。

`src/partition/partition_opt.py` 中的 `get_c_b(...)` 构造 observed miss count `c` 和按 crisis-case base count 分配的 expected miss count `b`。`scan(...)` 对每个 group 计算

```text
g_i = sum_k [ c_ik log(q_k) + b_ik (1 - q_k) ]
```

其中 `q_k` 由 candidate subset 中 observed/expected miss ratio 更新。candidate subset 是按 `g_i` 排序后选取得分最高的一组 groups，并通过 coordinate descent 更新 `q` 和 group scores。当前 split balance 由 `FLEX_TYPE='n_group'` 和 `FLEX_RATIO=0.1` 控制。

spatial contiguity 不是 candidate split search 的硬约束。当前流程先按 scan statistic 得到 binary split，然后在 `CONTIGUITY=True` 时执行 polygon adjacency majority-vote refinement。polygon mode 优先使用 true adjacency matrix；如果没有 adjacency dict，才回退到 centroid distance neighbors。

### 5. Polygon-level error summaries

Stage 1 partition search 的 polygon/group summary 来自 branch validation set。每个 validation observation 按 `X_group` 聚合到 FEWSNET admin/group ID。当前实现使用 binary class predictions，不使用 predicted probabilities 来计算 split objective。

对 group `i`，当前 crisis-focused summary 可以理解为：

```text
base_i = validation crisis observations in group i
hit_i  = validation observations in group i with y=1 and y_pred=1
miss_i = base_i - hit_i
```

scan statistic 使用各 group 的 `miss_i` 相对于按 `base_i` 分配的 expected miss count 的偏离程度。这个 summary 是 validation observations 的计数聚合；没有额外按 polygon observation count 加权，也没有 probability-weighted error。

Stage 3 的 polygon-level metrics 在 `scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py` 中由 `predictions_monthly.csv` 聚合得到。每个 `FEWSNET_admin_code` 的 metrics 包括 `n`、class-1 precision、recall、F1、overall binary error rate 和 class-1 binary error rate。`pct_err_all` 是 `mean(y_true != y_pred)`；`pct_err_class1` 是在 `y_true == 1` 子样本上的 `mean(y_true != y_pred)`，如果没有 class-1 observations 则记为 0。

### 6. Split acceptance criteria

当前 split acceptance 由 `src/tests/sig_test.py` 和 `config.py` 控制。主要参数是：

```text
MIN_DEPTH = 1
MAX_DEPTH = 6
MIN_BRANCH_SAMPLE_SIZE = 0
MIN_SCAN_CLASS_SAMPLE = 0
SIGLVL = 0.1
MIN_CLASS_1_IMPROVEMENT_THRESHOLD = 0.01
CLASS_1_SIGNIFICANCE_TESTING = True
```

在 `sig_test_class_1_focused(...)` 中，child branches 的 class-1 focused score 与 parent/base score 比较。若 mean improvement 大于 `0.01`，split 接受；若 improvement 小于或等于 0，split 拒绝；介于两者之间时，使用 `mean_diff / std_err_mean` 的 fallback statistic，当前阈值为 `1.0`。

当前代码没有额外的 minimum positive cases rule，也没有独立的 precision-loss cap。precision 是否下降不是 split acceptance 的单独硬约束；当前接受逻辑以实现中的 class-1 score improvement 为准。

### 7. Crisis-class weighting

当前 crisis emphasis 主要通过两处实现：

1. `SELECT_CLASS=np.array([1])` 使 split statistics 聚焦 crisis class。
2. `GOVERNING_METRIC='class_1_f1'` 和 `CRISIS_FOCUSED_OPTIMIZATION=True` 让 partition scoring 使用 class-1 focused score。

RF 本身没有使用 sklearn `class_weight`，`src/model/model_RF.py` 中当前 `class_weight = None`。Stage 3 comparison script 的 `RF_PARAMS` 也没有设置 `class_weight`。标准 Stage 3 comparison 使用 hard class predictions，而不是 calibration 后的 probability threshold search。

### 8. Refinement、reassignment、islands 和 disconnected polygons

当前 recursive split 后可执行 contiguity refinement。`config.py` 中 `CONTIGUITY=True`、`REFINE_TIMES=3`、`CONTIGUITY_TYPE='polygon'`、`USE_ADJACENCY_MATRIX=True`。`swap_partition_polygon(...)` 使用 neighbor majority voting；如果当前 polygon 的 assignment 在 neighbors plus self 中占比低于 `4/9`，则切换到最常见的其他 partition。

isolated polygons 在 `PRESERVE_ISOLATED_POLYGONS=True` 时保留原 assignment。multi-cluster Stage 3 refinement 同样使用 adjacency-aware majority voting。当前 note 不新增 reassigned polygon 数量统计；若后续需要报告数量或比例，应从 refinement audit/report 或对应 CSV 重新计算。

Stage 2 spectral clustering 对 KNN graph 的最大 connected component 运行 clustering；不在最大 component 内的 nodes 被标记为 outliers，并用 geographic nearest-neighbor classifier 分配到已有 cluster。

### 9. Weighted co-occurrence matrix

Stage 2 的 weighted co-occurrence matrix 在 `scripts/step4_similarity_matrix.py` 构造。每个 monthly partition plan 的权重是：

```text
w_p = max(logit(F1_partitioned,p) - logit(F1_base,p), 0)
```

这里使用 `main_index.csv` 中的 `f1(1)` 和 `f1_base(1)`，也就是 class-1 F1。负 gains 会被截断为 0。对于任意两个 admin units，只要它们在某个 plan 中属于同一 partition，就给它们的 pairwise co-grouping similarity 加上该 plan 的 `w_p`。

general matrix 使用所有 in-scope plans；month-specific matrix 只过滤对应 month，例如 February、June、October。当前实现没有对 horizons/months 做额外 variance normalization，也没有 small-sample high-variance penalty。co-occurrence matrix 乘以 spatial kernel 后，再用 matrix maximum 做归一化。

### 10. Spatial proximity modulation

spatial proximity 在 `scripts/step4_similarity_matrix.py` 中实现。代码使用 FEWSNET admin unit 的 latitude/longitude，先转为 radians 后用 haversine distance，再转为 degrees。spatial kernel 是：

```text
K_ij = exp(-d_ij^2 / (2 sigma^2)), sigma = 5.0 degrees
```

final similarity 是 weighted co-grouping similarity 与 `K_ij` 的 elementwise product。这个 kernel 是 attenuation，不是 hard contiguity rule。因此 distant polygons 如果在高权重 partition plans 中反复 co-group，并且经过 kernel attenuation 和 KNN sparsification 后仍保持 strong similarity，仍可能进入同一 final cluster。当前结果中的 distant grouping 应解释为 performance-similarity grouping with spatial attenuation，而不是 strict contiguous regionalization。

### 11. k-nearest-neighbor sparsification

KNN sparsification 在 `scripts/step5_sparsification.py` 中实现，当前 `K_NEIGHBORS=40`。每一行保留 similarity 最大的 40 个 neighbors，然后用 `knn_mat.maximum(knn_mat.transpose())` 对称化 graph。`k40` 文件名中的 40 是 graph neighbor count，不是 cluster count。

当前代码没有运行 k sensitivity，也没有在本 note 中声称 k 是由 performance tuning 或 sensitivity analysis 选择的。它只报告当前实现使用固定 `k=40`。

### 12. Spectral clustering details

`scripts/step5_sparsification.py` 使用 normalized graph Laplacian：

```text
L = I - D^{-1/2} A D^{-1/2}
```

当前计算最多 20 个 smallest-magnitude eigenvalues，eigengap 是相邻 eigenvalues 的差，recommended cluster count 是最大 eigengap 的 index 加 1。ties 或 flat spectra 没有单独 rule；`np.argmax` 会选择第一个最大 gap。

`scripts/step6_complete_clustering_pipeline.py` 对最大 connected component 使用 sklearn `SpectralClustering(affinity='precomputed', assign_labels='kmeans', random_state=42, n_jobs=-1)`。最大 component 外的 outliers 用 `KNeighborsClassifier(n_neighbors=1, metric='euclidean')` 基于 latitude/longitude 预测 cluster label。

### 13. Local RF training rules

Stage 3 GeoRF comparison 中 pooled RF 和 local RF 使用同一组主要 hyperparameters：

```text
n_estimators = 100
max_depth = None
random_state = 5
n_jobs = 1
```

每个 partition 单独训练 local RF。若 partition training samples 少于 `MIN_PARTITION_SAMPLES=50`，或该 partition training window 中只出现一个 class，则该 partition 使用 pooled fallback。

当前 standard Stage 3 comparison 不做 probability calibration，也不调低或调高 prediction threshold；它使用 sklearn classifier 的 hard class predictions。

## Appendix: Technical Details of Recursive Spatial Partitioning and Consensus Clustering

The main workflow separates partition learning from final evaluation. Recursive partition candidates are learned on the 2018-2020 partition-learning window, and the fixed consensus partitions are then evaluated on 2021-2024. This prevents final test-period outcomes from being used to define the spatial partitions.

Within each partition-learning run, the recursive spatial partitioning algorithm evaluates active branches on validation observations and aggregates class-specific prediction summaries by FEWSNET administrative polygon or group ID. The current crisis-focused configuration sets the selected class to crisis cases and uses a class-1 focused governing score. For each active branch, validation observations are grouped by polygon ID, and the algorithm compares observed crisis misses with the expected number of crisis misses implied by the branch-level distribution of crisis cases. Candidate binary splits are generated by a scan-statistic search over polygon/group scores rather than by feature-threshold search. The scan statistic ranks groups by their contribution to a likelihood-ratio-style contrast,

```text
g_i = sum_k [ c_ik log(q_k) + b_ik (1 - q_k) ],
```

where `c_ik` is the observed miss count for selected class `k` in group `i`, `b_ik` is its expected miss count under proportional allocation, and `q_k` is iteratively updated from the candidate subset. The current implementation uses group-count balance with `FLEX_TYPE='n_group'` and `FLEX_RATIO=0.1`.

Split acceptance is based on the implemented class-1 focused score improvement. The current configuration uses `MIN_DEPTH=1`, `MAX_DEPTH=6`, `MIN_BRANCH_SAMPLE_SIZE=0`, `MIN_SCAN_CLASS_SAMPLE=0`, and `MIN_CLASS_1_IMPROVEMENT_THRESHOLD=0.01`. A split is accepted immediately when the mean class-1 focused improvement exceeds 0.01. Splits with non-positive mean improvement are rejected. Borderline positive cases use the current fallback statistic, `mean_diff / std_err_mean`, with threshold 1.0.

Spatial contiguity is applied as refinement after the scan-selected binary split. In polygon mode, the implementation uses a polygon adjacency dictionary when available and applies neighbor majority voting for `REFINE_TIMES=3` refinement passes. A polygon changes assignment when its current label has less than a 4/9 vote share among its neighbors plus itself; isolated polygons are preserved under the current configuration.

The consensus partition stage combines many monthly partition plans into a weighted co-occurrence matrix. For plan `p`, the implemented performance weight is

```text
w_p = max(logit(F1_partitioned,p) - logit(F1_base,p), 0),
```

where both F1 values are class-1 F1 values read from the linked plan index. Two polygons receive additional pairwise similarity when they appear in the same partition under a positively weighted plan. General consensus uses all in-scope plans; month-specific consensus filters the plan set to the selected month.

The co-occurrence matrix is spatially modulated by a Gaussian kernel computed from haversine distance between polygon centroids:

```text
K_ij = exp(-d_ij^2 / (2 sigma^2)), sigma = 5.0 degrees.
```

The final dense similarity matrix is the elementwise product of weighted co-occurrence similarity and this spatial kernel, normalized by its maximum value. The spatial kernel attenuates distant relationships but does not impose strict contiguity, so non-contiguous clusters can occur when repeated high-weight co-grouping remains strong after spatial attenuation.

The sparse graph used for clustering is built with fixed `k=40` nearest-neighbor sparsification. For each polygon, the 40 largest similarity entries are retained and the graph is symmetrized. The selected cluster count is determined from the eigengap of the normalized graph Laplacian,

```text
L = I - D^{-1/2} A D^{-1/2}.
```

The implementation computes up to 20 smallest-magnitude eigenvalues and selects the first largest eigengap. Final cluster labels are produced with scikit-learn `SpectralClustering` using precomputed affinity, k-means label assignment, and `random_state=42`. Clustering is applied to the largest connected component; nodes outside that component are assigned to the nearest learned cluster by one-nearest-neighbor classification in latitude/longitude space.

For final 2021-2024 evaluation, each consensus cluster receives a local Random Forest when enough training data are available. The Stage 3 GeoRF comparison uses `n_estimators=100`, `max_depth=None`, `random_state=5`, and `n_jobs=1` for both pooled and local Random Forests. A cluster falls back to the pooled model if it has fewer than 50 training samples or only one observed class in the training window. The reported polygon-level evaluation summaries aggregate hard binary predictions by `FEWSNET_admin_code`, including observation count, class-1 precision, class-1 recall, class-1 F1, overall binary error rate, and class-1 binary error rate.
