# GeoRF m2 Adjacency Refinement Note

中文审查说明：

该图使用 GeoRF 的 m2 consensus partition 作为一个实际 Stage 3 local RF
评估之前的 adjacency refinement 示例。左图展示 refinement 前的 cluster
assignment，中图展示 refinement 后的 assignment，右图只高亮发生 reassignment
的 polygons。

该 m2 mapping 共包含 5365 个 polygons，
3 次 deterministic refinement iteration 后共有 34 个
polygons 的最终 cluster assignment 发生变化，占 0.63%。
每次 iteration 的 reassignment move 数量为：29, 10, 6，
iteration-level moves 合计 45。

该图只汇报当前已经实现并用于 Stage 3 fixed-partition evaluation 的
adjacency refinement，不额外声称新的 split acceptance 或模型重训机制。

Appendix text (English):

As an example of the adjacency-refinement step applied before the Stage 3 local
RF evaluation, we show the GeoRF m2 consensus partition before and after
refinement. The refinement log records
45 polygon-iteration reassignment moves across
three deterministic iterations. In the final pre/post comparison,
34 of 5365 polygons (0.63%) changed cluster assignment. The right panel highlights the polygons
whose cluster assignment changed between the pre- and post-refinement maps.
