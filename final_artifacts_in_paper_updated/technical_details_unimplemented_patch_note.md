# Internal Patch Note: Reviewer Technical Detail Gaps

本文件只供内部排期和审稿回应准备使用，不放进 paper appendix。原则是：当前代码已经实现的内容进入 `technical_details_review_note.md` 的 appendix；当前没有实现、没有现成产物支撑、或会引出额外实验承诺的内容，只放在这里。

## 不进入 Appendix 的内容

### 1. k=40 sensitivity

当前 `scripts/step5_sparsification.py` 和 `scripts/step6_complete_clustering_pipeline.py` 固定使用 `K_NEIGHBORS=40`。尚未实现或汇总 k sensitivity。若后续审稿必须回应，可以增加 k grid，例如 20、30、40、50、60，比较 graph connectivity、eigengap、cluster count、Stage 3 F1/precision/recall 和 map stability。

### 2. Weighted co-occurrence 的 variance 或 small-sample penalty

当前 plan weight 是 `max(logit(F1_partitioned) - logit(F1_base), 0)`。没有按 validation sample size、monthly variance、horizon variance 或 small-sample uncertainty 做惩罚。若需要增强，可以在 `scripts/step4_similarity_matrix.py` 中增加 weight diagnostics，而不是直接改当前 paper appendix。

### 3. Spatial kernel sensitivity

当前 spatial kernel 使用 haversine distance 和 `sigma_degrees=5.0`。没有 distance-band sensitivity，也没有 alternate bandwidth comparison。appendix 只能说明当前 kernel 和它是 attenuation 而非 hard contiguity rule。

### 4. Probability calibration、log loss、AUC

当前 standard Stage 3 comparison 使用 hard class predictions。没有 probability calibration、log-loss optimization、AUC optimization 或 probability-threshold tuning进入主结果。scenario scripts 中的 threshold 设定不属于 standard comparison，不应混入本 appendix。

### 5. Precision-recall tradeoff formal rule

当前 split acceptance 使用 class-1 focused score improvement 和 fallback statistic。没有单独的 precision-loss cap，也没有 formal rule 说明 precision 的负效应可以被 recall gain 容忍到什么程度。appendix 不应声称存在这样的 rule。

### 6. Exhaustive contiguity-constrained split search

当前 candidate split search 是 scan-statistic heuristic，再做 adjacency majority refinement。它不是 exhaustive search，也不是 hard contiguity-constrained optimizer。appendix 不应使用会暗示 exhaustive spatial optimization 的措辞。

### 7. Minimum positive cases 和 stricter node-size rules

当前 `MIN_BRANCH_SAMPLE_SIZE=0`、`MIN_SCAN_CLASS_SAMPLE=0`，没有额外 minimum positive cases rule。若要增强审稿回应，需要先修改 config 或在 analysis artifact 中报告当前参数的含义。

### 8. Fallback frequency table

当前 Stage 3 code 会在 local partition 样本不足或 single-class 时使用 pooled fallback，但本轮 note 不新增 fallback frequency table。若 manuscript 需要量化 fallback 发生频率，应从 Stage 3 logs、run manifests 或重新 instrumented comparison outputs 汇总后再写入 paper。

### 9. Reassigned polygon count and refinement audit table

当前 contiguity refinement 会执行 fixed passes，并且部分 scripts 会写 audit/report 信息，但本轮不计算新表。若需要报告被重新分配 polygons 的数量或比例，应从 refinement output 重新聚合，并明确 general/m2/m6/m10、GeoRF/GeoDT 和 scope。

## 建议优先级

如果后续必须补实验，优先级建议是：

1. k sensitivity，因为审稿意见直接点名 k=40 会影响 connectivity、eigenspectrum、cluster count 和 boundaries。
2. fallback frequency table，因为它可以从 Stage 3 过程产物或轻量 instrumentation 得到，不一定需要完整重跑模型。
3. kernel bandwidth sensitivity，因为它直接对应 geographically distant grouping 的解释。
4. weight variance/small-sample diagnostics，因为它会影响 consensus matrix 的可信度，但可能需要更多设计。
