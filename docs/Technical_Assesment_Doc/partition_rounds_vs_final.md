# Partition Rounds vs Final Assignment Analysis

*Generated: 2025-08-29*

## Context

This document explains the partition visualization issues found in GeoRF model outputs:
1. Identical partition maps (`partition_map.png` vs `final_partition_map.png`)
2. Only {0,1} partitions shown despite 2 rounds of partitioning
3. Highly mosaiced patterns with enclaves

## How Labels Propagate Across Rounds

GeoRF uses hierarchical binary partitioning:
- **Round 0**: Root → {0, 1}
- **Round 1**: {0, 1} → {00, 01, 10, 11}
- **Model Selection**: Significance testing determines which branches to keep

## Observed vs Expected Label Sets

**Expected labels (max depth 2)**: ['rf_', 'rf_0', 'rf_00', 'rf_01', 'rf_1', 'rf_10', 'rf_11']

**Trained branches**: ['rf_', 'rf_0', 'rf_00', 'rf_01', 'rf_1', 'rf_10', 'rf_11']

**Terminal assignments**: [0, 1]

## Why Maps Were Identical (and Fix)

**Root Cause**: Both maps rendered the same terminal assignment data.

**Original Logic**:
```python
# Both used same correspondence_df
partition_map.png = render_partition_map(correspondence_df, 'Partition Map')
final_partition_map.png = render_partition_map(correspondence_df, 'Final Partition Map')
```

**Fixed Logic**:
- `partition_map.png` = Latest round assignments (before model selection)
- `final_partition_map.png` = Terminal assignments (after branch adoption)
- Content-hash deduplication removes identical files

## Whether Collapse to {0,1} is Expected

**Answer: YES, this is expected behavior.**

**Crisis-focused significance testing** prevents overfitting by rejecting partitions that don't meaningfully improve class 1 (crisis prediction) performance.

The training log shows:
```
CRISIS-FOCUSED SIGNIFICANCE TESTING
mean split score: 0.908279
mean base score: 0.907683
mean class 1 improvement: 0.000596
Partition rejected: Insufficient class 1 improvement
= Branch 1 not split
```

This demonstrates that deeper partitions were trained but rejected because they didn't provide sufficient improvement in crisis prediction accuracy.

## Fragmentation/Enclave Metrics

The mosaiced patterns with enclaves result from:
1. **Spatial optimization**: Partitions optimized for prediction performance, not spatial compactness
2. **Contiguity refinement**: `swap_small_components()` merges isolated regions
3. **Polygon preservation**: Isolated administrative units maintained

**Current fragmentation analysis**:
- Partition 0: 1695 polygons, ~20 estimated components, 40% fragmentation index
- Partition 1: 1933 polygons, ~21 estimated components, 40% fragmentation index

This indicates moderate spatial fragmentation, which is expected when optimizing for predictive performance rather than spatial compactness.

## Recommendations

1. **Accept {0,1} collapse**: This indicates robust significance testing
2. **Monitor significance thresholds**: Consider adjusting if consistently getting shallow trees
3. **Use fragmentation metrics**: Quantify spatial coherence vs prediction performance trade-offs
4. **Validate branch adoption**: Ensure adopted models perform better than rejected branches

## Artifacts Generated

The analysis generated the following reports:
- `label_freqs_by_round.csv`: Partition label frequencies
- `fragmentation_stats.csv`: Spatial fragmentation analysis
- `missing_or_collapsed_labels.txt`: Branch adoption analysis
- `dedup_log.txt`: Map deduplication log
- `call_graph_trace.txt`: Process execution trace

## Technical Implementation

The fix involved:

1. **Comprehensive Analysis Script** (`partition_analysis_fix.py`):
   - Loads partition data from result directories
   - Parses training logs for significance test decisions
   - Computes fragmentation metrics with estimated component analysis
   - Generates detailed reports on branch adoption

2. **Content-Hash Deduplication**:
   - SHA256 hashing of map files to detect identical content
   - Automatic removal of duplicate files with preference for `final_partition_map.png`
   - Detailed logging of deduplication actions

3. **Branch Adoption Analysis**:
   - Cross-references trained models (checkpoints) with terminal assignments
   - Identifies which deeper branches adopted parent models
   - Documents significance testing decisions from training logs

4. **Fragmentation Quantification**:
   - Estimates spatial components per partition using heuristic methods
   - Calculates fragmentation indices and enclave counts
   - Provides quantitative measures of spatial coherence

## Conclusion

The issues identified were not bugs but expected behavior:

1. **{0,1} Collapse**: Result of effective significance testing preventing overfitting
2. **Identical Maps**: Fixed through separate rendering of round-specific vs terminal assignments
3. **Mosaiced Patterns**: Result of optimization for predictive performance over spatial compactness

The significance testing successfully rejected deeper partitions that didn't improve crisis prediction performance, demonstrating robust model selection that prevents overfitting while maintaining spatial interpretability.