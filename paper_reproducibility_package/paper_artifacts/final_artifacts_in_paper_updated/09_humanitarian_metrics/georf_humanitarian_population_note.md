# GeoRF Population-Weighted Humanitarian Metrics

中文审查说明：

该 appendix 只针对 GeoRF pooled 和 partitioned/local RF 模型。
Population 来自 raw FEWSNET.csv 的 `pop` 字段，并按 `admin_code` 和 evaluated target month 与 Stage 3 polygon-month predictions 合并。
所有 population totals 都是 evaluated polygon-month observations 上的 population-month totals，不解释为 unique affected people。
Missed-crisis population 定义为实际 crisis 但模型预测 non-crisis 的 population-month 总和。
False-alert population 定义为实际 non-crisis 但模型预测 crisis 的 population-month 总和。
Population-weighted recall 使用 true-alert population 除以实际 crisis population；population-weighted precision 使用 true-alert population 除以预测 crisis population。
表中的 delta 定义为 partitioned minus pooled；missed-crisis / false-alert population 的负 delta 表示 partitioned 更低，recall / precision 的正 delta 表示 partitioned 更高。

Appendix text (English):

We report population-weighted humanitarian diagnostics for the GeoRF pooled and partitioned RF models.
Population is taken from the raw FEWSNET `pop` field and merged to evaluated polygon-month predictions by FEWSNET admin code and target month.
Population totals are population-month totals over evaluated polygon-month observations, not estimates of unique affected people.
Missed-crisis population is the population in observations where a crisis occurred but the model predicted non-crisis.
False-alert population is the population in observations where no crisis occurred but the model predicted crisis.
Population-weighted recall divides true-alert population by the total actual-crisis population, and population-weighted precision divides true-alert population by the total predicted-crisis population.
Deltas are reported as partitioned minus pooled.
