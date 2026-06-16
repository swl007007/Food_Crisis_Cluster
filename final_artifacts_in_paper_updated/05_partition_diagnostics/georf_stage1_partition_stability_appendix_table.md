# Appendix Table: GeoRF Stage 1 Partition Stability

| Comparison axis | Total pairs | Pairs used for metrics | Median common polygons | ARI median [IQR] | NMI median [IQR] |
| --- | --- | --- | --- | --- | --- |
| Across years | 21 | 18/21 | 5358 | 0.018 [0.002, 0.200] | 0.087 [0.019, 0.239] |
| Across horizons | 21 | 18/21 | 5360 | 0.157 [0.052, 0.462] | 0.185 [0.029, 0.294] |
| Across months | 24 | 20/24 | 5359 | 0.048 [0.024, 0.113] | 0.071 [0.024, 0.127] |
| Mixed pairings | 210 | 175/210 | 5360 | 0.073 [0.019, 0.192] | 0.096 [0.027, 0.204] |

Note: ARI and NMI are computed on common FEWSNET administrative polygons with valid
Stage 1 partition assignments; out-of-scope `s-1` assignments are excluded.
