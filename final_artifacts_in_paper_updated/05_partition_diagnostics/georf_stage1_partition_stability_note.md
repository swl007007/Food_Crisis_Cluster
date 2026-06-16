# GeoRF Stage 1 Partition Stability Note

中文审查说明：

本诊断仅覆盖当前论文主模型 GeoRF 的第一阶段 partition plans，
不纳入 GeoDT。GeoDT 在本研究中主要用于展开和检查 branch differences，
不作为后续 partition stability 结论的默认模型。

稳定性基于 `GeoRFExperiment/linked_tables/main_index.csv` 中列出的
2018-2020 Stage 1 linked partition plans 计算。每一对 partition 先按
`FEWSNET_admin_code` 取共同且有效的 polygon；`s-1`、空值和缺失标签
被视为 out-of-scope，不进入 ARI/NMI 或 cluster-size 统计。

Pairwise comparison 分为三类主轴：同一月份和同一 forecasting horizon / lag
但不同年份为 across years；同一年和同一月份但不同 horizon 为 across horizons；
同一年和同一 horizon 但不同月份为 across months。其他组合保留为 mixed，
用于透明报告但不作为主要稳定性解释。

Adjusted Rand index (ARI) 和 normalized mutual information (NMI) 都对
cluster label permutation 不敏感，因此适合比较不同年度、月份和 horizon 下
重新学习得到的 partition labels。Cluster-size distribution 用于检查是否存在
少数超大 cluster 或大量极小 cluster 驱动的表观稳定性。

两个 Stage 1 plans 在当前 linked partition 表中全部为 `s-1`，
因此涉及这些 plans 的 pairwise ARI/NMI 被标记为不可计算，而不是解释为
低稳定性。

## Pairwise Stability Summary

| comparison_group | n_pairs_total | n_pairs_with_metric | n_pairs_without_common_valid | n_common_valid_min | n_common_valid_median | n_common_valid_max | adjusted_rand_index_mean | adjusted_rand_index_median | adjusted_rand_index_min | adjusted_rand_index_p25 | adjusted_rand_index_p75 | adjusted_rand_index_max | normalized_mutual_information_mean | normalized_mutual_information_median | normalized_mutual_information_min | normalized_mutual_information_p25 | normalized_mutual_information_p75 | normalized_mutual_information_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| across_years | 21 | 18 | 3 | 5159 | 5358 | 5365 | 0.1264 | 0.0180238 | -0.0959423 | 0.00232386 | 0.199617 | 0.500136 | 0.124445 | 0.0869823 | 0.00105425 | 0.0188972 | 0.238686 | 0.294423 |
| across_horizons | 21 | 18 | 3 | 5159 | 5360.5 | 5365 | 0.239392 | 0.156803 | -0.0998431 | 0.0517756 | 0.462424 | 0.581321 | 0.188905 | 0.185325 | 0.00800819 | 0.0288158 | 0.29352 | 0.517547 |
| across_months | 24 | 20 | 4 | 5159 | 5359 | 5365 | 0.136784 | 0.0480005 | -0.123061 | 0.0243834 | 0.113181 | 0.667948 | 0.123251 | 0.0710635 | 0.00178009 | 0.0240892 | 0.127144 | 0.435094 |
| mixed | 210 | 175 | 35 | 5080 | 5360 | 5365 | 0.131967 | 0.0726287 | -0.0737401 | 0.0186915 | 0.191583 | 0.76165 | 0.124765 | 0.0962835 | 0.00239448 | 0.0268534 | 0.203607 | 0.635766 |

## All-Out-of-Scope Stage 1 Plans

| plan | year | month | forecasting_horizon_months | source_scope | n_polygons | n_clusters |
| --- | --- | --- | --- | --- | --- | --- |
| GeoRF_2019_10_fs3 | 2019 | 10 | 12 | fs3 | 0 | 0 |
| GeoRF_2020_06_fs1 | 2020 | 6 | 4 | fs1 | 0 | 0 |

## Cluster-Size Summary by Forecasting Horizon / Lag

| forecasting_horizon_months | n_plans | median_n_clusters | median_n_polygons | median_cluster_size | median_largest_cluster_share |
| --- | --- | --- | --- | --- | --- |
| 4 | 6 | 3 | 5361.5 | 684 | 0.870223 |
| 8 | 9 | 5 | 5364 | 232 | 0.852819 |
| 12 | 9 | 4 | 5361 | 531.75 | 0.798621 |

Appendix text (English):

We assessed the stability of the GeoRF Stage 1 partition plans using pairwise
adjusted Rand index (ARI), normalized mutual information (NMI), and cluster-size
distributions. For each pair of Stage 1 partition plans, polygons were aligned by
FEWSNET administrative code, and invalid or out-of-scope assignments (`s-1`,
blank, or missing labels) were excluded. Pairwise comparisons were grouped as
across-year comparisons when month and forecasting horizon were fixed,
across-horizon comparisons when year and month were fixed, and across-month
comparisons when year and forecasting horizon were fixed. Remaining pairings
were retained as mixed comparisons for transparency. Cluster-size summaries were
computed within each Stage 1 plan after applying the same validity filter.
