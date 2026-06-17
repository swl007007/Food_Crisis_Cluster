# GeoRF Validation-Selected Max-F1 Thresholding

中文审查说明：

该 appendix 只针对 GeoRF partitioned/local RF 模型的 thresholded diagnostic。
Threshold 在每个 rolling training window 的 validation subset 上选择，目标是最大化 class-1 F1。
选出的 threshold 只应用于随后 held-out target month 的 test probabilities；没有使用 test labels 选择 threshold。
原始 pooled 和 partitioned hard-prediction 结果保留，用于对照。
这些结果先写入独立 artifact folder，尚不覆盖主文 01-11 artifacts。

Appendix text (English):

We evaluate a validation-selected probability threshold for the GeoRF partitioned RF model.
For each forecasting horizon and target month, the threshold is selected on a validation subset from the rolling training window by maximizing class-1 F1, then applied to the held-out target-month probabilities.
The procedure does not use test labels for threshold selection.
Pooled and original partitioned hard-prediction results are retained as comparators.
