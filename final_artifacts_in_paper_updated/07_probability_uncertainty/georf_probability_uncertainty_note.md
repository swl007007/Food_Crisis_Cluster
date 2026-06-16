# GeoRF Probability and Uncertainty Diagnostics

中文审查说明：

该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。
Stage 3 现在导出 class-1 probabilities，同时保留原有 binary predictions。
标准 Stage 3 评估没有额外 threshold tuning；binary classification 使用 classifier 默认 hard prediction rule。
Brier score 和 reliability bins 使用 raw RF probabilities，不额外拟合 Platt scaling 或 isotonic calibration。
Brier score 越低越好；表中的 delta 仍定义为 partitioned minus pooled，因此 Brier delta 为负表示 partitioned 更好。
Paired bootstrap confidence intervals 使用 country-clustered resampling，默认 1000 次重复；region-specific CI 至少需要 3 个 countries。
这些 uncertainty summaries 是对已评估 polygon-month predictions 的不确定性诊断，不是未来事件的预测区间。

Appendix text (English):

We export class-1 probabilities for the GeoRF pooled and partitioned RF models and report probability diagnostics on the evaluated polygon-month observations.
The standard Stage 3 evaluation does not perform separate threshold tuning; hard classifications follow the classifier default decision rule.
Brier scores and reliability bins are computed from raw RF probabilities, with no additional calibration model fitted in this appendix.
Lower Brier scores are better; deltas are reported as partitioned minus pooled, so negative Brier deltas favor the partitioned model.
Paired confidence intervals use country-clustered bootstrap resampling so pooled and partitioned predictions are evaluated on the same resampled polygon-month observations.
Region-specific intervals repeat the same procedure within regions when enough countries are available.
