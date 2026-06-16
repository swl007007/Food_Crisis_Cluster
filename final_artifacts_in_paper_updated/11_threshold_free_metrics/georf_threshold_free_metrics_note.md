# GeoRF Threshold-Free and Fixed Operating-Point Metrics

中文审查说明：

该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。
所有指标都从现有 Stage 3 `y_prob_pooled` 和 `y_prob_partitioned` 概率输出计算，不重跑模型，也不改变主文 binary prediction rule。
PR-AUC 使用 average precision，衡量 crisis probability ranking 的整体 precision-recall 表现。
Recall at fixed precision 和 precision at fixed recall 是 post hoc operating-point diagnostics，用于展示现有概率排序在指定 precision 或 recall 约束下可达到的 tradeoff。
这些指标不表示本文已经进行了 threshold tuning；主结果仍然使用当前 hard predictions 的 precision、recall 和 F1。
如果某个 operating point 不可达到，表中保留空值。

Appendix text (English):

We report additional threshold-free and fixed operating-point diagnostics for the GeoRF pooled and partitioned RF models.
All metrics are computed from existing Stage 3 crisis-class probabilities and do not require model retraining or a different threshold-selection procedure.
PR-AUC is computed as average precision and summarizes the probability ranking across the precision-recall curve.
Recall at fixed precision and precision at fixed recall are post hoc operating-point diagnostics showing feasible tradeoffs under the existing probability scores.
The main binary results remain based on the implemented hard predictions; these appendix metrics are complementary ranking and sensitivity diagnostics.
