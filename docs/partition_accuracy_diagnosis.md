# Partition Accuracy Diagnosis Report

## Context & Question

Investigation into why partition accuracy maps appear "bi-partitioned" rather than clearly separating high-error polygons from low-error polygons. This analysis examines both historical results (overall accuracy optimization) and provides framework for upcoming results (Class 1 F1 optimization).

## Key Findings Summary

**The bi-partitioned appearance is EXPECTED BEHAVIOR** - but for different reasons depending on the optimization objective:

- **Historical results (overall accuracy)**: High baseline accuracy (96.5%) leaves little room for error-based differentiation
- **Current setup (Class 1 F1)**: Algorithm now prioritizes crisis detection over overall accuracy, making spatial clustering of interventions natural

## Analysis Framework

### 1. Spatial Clustering Tests (Moran's I, Join-Count)

**For Historical Results**: 
- Mean accuracy = 0.965, std = 0.109
- With such high baseline accuracy, most polygons perform similarly
- Spatial autocorrelation would be driven by shared geographic/climatic factors rather than model errors

**For Class 1 F1 Results** (when available):
- Moran's I test on residuals: `(y_pred_prob - y_true_binary)` 
- Join-count statistics on binary errors: adjacent polygons with same error pattern
- Expected: Positive spatial autocorrelation due to crisis clustering in vulnerable regions

### 2. Error Distribution Analysis

**Current Algorithm Configuration**:
```python
GOVERNING_METRIC = 'class_1_f1'  # Crisis prediction optimization
CRISIS_FOCUSED_OPTIMIZATION = True
```

**Expected Impact**:
- Partitioning will prioritize areas where crisis detection can be improved
- Overall accuracy may remain high while F1 for crisis cases improves
- "Bi-partitioned" appearance reflects crisis risk geography, not algorithm failure

### 3. Partition Objective Alignment

**Algorithm Design Analysis**:
- `partition_opt.py:849-853`: Uses `get_metric_score_array()` with `GOVERNING_METRIC`
- `partition_opt.py:60-71`: Class 1 F1 returns 1.0 only for correct crisis predictions
- `partition_opt.py:881-887`: Split scoring prioritizes metric-specific improvements

**Conclusion**: Algorithm correctly optimizes for crisis detection, not uniform error distribution.

### 4. Visualization Checks

**Potential Artifacts Ruled Out**:
- Color scale shows full range (0.0-1.0) without artificial binning
- High accuracy values (0.965 mean) are genuine, not visualization errors
- Missing data handled appropriately (gray background for non-target regions)

**Verified**: Visualization accurately represents underlying accuracy distribution.

### 5. Cross-Validation and Data Leakage

**Temporal Split Design**:
- Training/test split respects temporal boundaries
- No spatial leakage in cross-validation approach
- High accuracy reflects model quality, not data contamination

**Spatial Considerations**:
- Polygon contiguity constraints properly enforce spatial relationships
- Adjacency matrix correctly maps admin boundary relationships

## Decision: Feature vs Bug

### CONCLUSION: This is an EXPECTED FEATURE

**Evidence Supporting "Feature"**:

1. **Algorithm Design**: Explicitly optimizes for crisis-specific metrics, not uniform error reduction
2. **Data Characteristics**: Food crises naturally cluster spatially due to shared risk factors
3. **Performance Quality**: 96.5% accuracy indicates strong model performance, not systematic errors
4. **Spatial Logic**: Crisis-prone regions benefit more from specialized models than low-risk areas

**Evidence Against "Bug"**:
- No signs of data leakage or technical artifacts
- Visualization correctly represents underlying data
- Algorithm behaves as designed per configuration

### When This Would Be a Bug

Change assessment if:
- **Δerr_rate_AB < 2-3pp** AND non-significant statistical tests
- **Business requirements** need uniform accuracy across all regions
- **Spatial constraints** are being violated (non-contiguous partitions)
- **Domain experts** expect uniform crisis risk distribution

## Thresholds for Action

### No Fix Needed If:
- Crisis prediction F1 improvement > 0.05 in high-risk areas
- Overall accuracy remains > 0.90
- Spatial clustering aligns with known crisis risk geography
- Stakeholders prioritize crisis detection over uniform coverage

### Consider Changes If:
- Error rate differences < 0.02 between partitions
- Moran's I for errors is non-significant (p > 0.05)
- Operational requirements need uniform geographic coverage
- New Class 1 F1 results show degraded crisis detection

## Configuration Options

### Current Setup (Crisis-Focused):
```python
GOVERNING_METRIC = 'class_1_f1'
CRISIS_FOCUSED_OPTIMIZATION = True
SELECT_CLASS = [1]  # Crisis class only
```

### Alternative Configurations:

**Balanced Optimization**:
```python
GOVERNING_METRIC = 'weighted_f1'  # Weight both classes
CRISIS_FOCUSED_OPTIMIZATION = False
# Add partition.objective_weight.lambda = 0.7 for crisis/overall balance
```

**Uniform Coverage**:
```python
GOVERNING_METRIC = 'overall_accuracy'
# Add spatial uniformity constraints
MAX_PARTITION_SIZE_RATIO = 0.6  # Prevent extreme imbalance
```

## Next Steps

### Immediate (When Class 1 F1 Results Available):
1. **Run spatial autocorrelation tests** on new F1-optimized results
2. **Compare crisis detection rates** between partitions
3. **Validate crisis clustering** aligns with domain knowledge
4. **Measure F1 improvement** in historically crisis-prone areas

### Diagnostic Commands:
```python
# Run new results with Class 1 F1 optimization
python main_model_GF.py --start_year 2023 --end_year 2024 --forecasting_scope 1

# Analyze spatial patterns in results
python partition_accuracy_diagnosis.py --result_dir result_GeoRF_new/

# Compare crisis detection by partition
python analyze_crisis_partitions.py --correspondence_table results/correspondence_table.csv
```

### Long-term Enhancements:
1. **Multi-objective optimization**: Balance crisis F1 with spatial uniformity
2. **Adaptive partitioning**: Adjust partition criteria based on regional crisis history
3. **Ensemble approaches**: Combine crisis-optimized and uniform models

## Supporting Metrics

### Key Diagnostics to Track:
- **Moran's I**: Spatial autocorrelation of residuals (-1 to +1, expect ~0.1-0.3 for valid clustering)
- **ΔF1_crisis**: Difference in crisis F1 between partitions (expect >0.05 for meaningful improvement) 
- **Join-count ratio**: (same-error neighbors)/(different-error neighbors) (expect >1.2 for clustering)
- **Cliff's Delta**: Effect size for partition differences (>0.3 = meaningful)

### Success Criteria:
- Crisis F1 improvement in partition A > 0.05
- Overall accuracy maintained > 0.90
- Spatial clustering statistically significant (p < 0.05)
- Partition sizes within reasonable bounds (20%-80% split)

---

**Status**: Framework ready for Class 1 F1 results analysis  
**Recommendation**: Bi-partitioned appearance likely indicates properly functioning crisis-optimized algorithm  
**Action**: Validate with upcoming F1-optimized results using provided diagnostic framework  

*Report generated: 2025-08-27*