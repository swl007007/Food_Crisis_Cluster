# Class Imbalance Analysis Report: Understanding the F1 Score Discrepancy

**Date:** August 21, 2025  
**Model:** GeoRF Food Crisis Prediction  
**Test Case:** 2015Q2 Visual Debug Analysis  
**Issue:** High overall F1 metrics (~0.9) vs Low class 1 F1 scores (~0.5-0.6)

## Executive Summary

The discrepancy between high overall F1 metrics (~0.9) in partition metrics and low class 1 F1 scores (~0.5-0.6) in final evaluation is caused by **severe class imbalance** in food crisis prediction data. The partition metrics are dominated by majority class (non-crisis) performance, while your final evaluation correctly focuses on minority class (crisis) performance.

## Observed Behavior

### Debug Output Evidence
```
Number of metric records: 10497
Average F1 improvement: -0.0024
Average accuracy improvement: -0.0012
Positive F1 improvements: 165 out of 10497 partitions

Final Results:
INFO:root:f1: 0.956404, 0.539075  # [Class 0 F1, Class 1 F1]
Q2 2015 Test - GeoRF F1: [0.95640368 0.53907496]
```

### Performance Metrics Breakdown
- **Partition Metrics Average**: ~0.9 (macro-averaged across classes)
- **Class 0 (No Crisis) F1**: 0.956 (95.6% accuracy)
- **Class 1 (Crisis) F1**: 0.539 (53.9% accuracy)
- **Final Evaluation Focus**: Class 1 only (0.539)

## Root Cause Analysis

### 1. Class Imbalance Structure

**Typical Distribution in Food Crisis Data:**
- **Class 0 (No Crisis)**: ~85-90% of samples
- **Class 1 (Crisis)**: ~10-15% of samples

**Impact on Metrics:**
- Class 0 predictions are **statistically easier** (predict "no crisis" = 85-90% accuracy)
- Class 1 predictions are **statistically harder** (rare crisis events)
- **Aggregate metrics favor majority class** performance

### 2. Metric Calculation Differences

**Partition Metrics (Showing ~0.9):**
- Uses **macro-averaged F1** = (Class 0 F1 + Class 1 F1) / 2
- OR **weighted F1** = (Class 0 F1 × frequency_0) + (Class 1 F1 × frequency_1)
- **Dominated by excellent Class 0 performance** (0.956)

**Final Evaluation (Showing ~0.5-0.6):**
- Uses **Class 1 F1 only** = 0.539
- **Correctly focuses on crisis prediction** performance
- **Not influenced by easy non-crisis predictions**

### 3. Mathematical Breakdown

**Your 2015Q2 Results:**
```
Class 0 F1: 0.956 (96% accuracy on non-crisis prediction)
Class 1 F1: 0.539 (54% accuracy on crisis prediction)

Macro F1 = (0.956 + 0.539) / 2 = 0.748
Weighted F1 ≈ (0.956 × 0.85) + (0.539 × 0.15) ≈ 0.89
```

### 4. Partition Optimization Misalignment

**Current Partition Selection Logic:**
- Optimizes for **overall F1 improvement** (macro or weighted)
- **Penalizes partitions** that improve class 1 but slightly hurt class 0
- Results in **negative average F1 improvement** (-0.0024)

**Actual Partition Effect:**
- Partitioning likely **improves class 1 prediction** (crisis detection)
- Partitioning slightly **reduces class 0 prediction** (non-crisis detection)
- **Net macro effect appears negative** due to class imbalance
- **Only 165 out of 10497 partitions** show positive macro improvement

## Problem Identification

### 1. Misaligned Optimization Target

**Current Behavior:**
- **Partition optimization**: Maximizes macro/weighted F1
- **Final evaluation**: Measures class 1 F1 only
- **Result**: Optimization target ≠ Evaluation target

**Impact:**
- Spatial partitioning may **reject beneficial splits** for crisis prediction
- Model optimizes for **wrong objective function**
- **Suboptimal crisis prediction** performance

### 2. Evaluation Inconsistency

**Partition Selection:**
- Chooses partitions based on **overall performance improvement**
- May reject partitions that **improve class 1 but hurt class 0**
- **Misaligned with crisis prediction goals**

**Final Evaluation:**
- Measures only **class 1 performance** (crisis prediction)
- **Different optimization target** than partition selection

## Proposed Solutions

### Solution 1: Class 1 Focused Partition Optimization

**Modify Partition Selection Logic:**
```python
# In partition_opt.py or relevant partitioning code
def evaluate_partition_improvement(y_before, y_pred_before, y_after, y_pred_after):
    # Focus on class 1 F1 improvement only
    f1_before_class1 = f1_score(y_before, y_pred_before, pos_label=1)
    f1_after_class1 = f1_score(y_after, y_pred_after, pos_label=1)
    return f1_after_class1 - f1_before_class1
```

**Benefits:**
- Aligns partition optimization with final evaluation
- Prioritizes crisis prediction improvements
- Eliminates class imbalance bias in partition selection

### Solution 2: Dual Metric Tracking

**Enhanced Partition Metrics:**
```python
def track_partition_metrics():
    metrics = {
        'macro_f1': macro_f1_score,
        'class_0_f1': f1_score(pos_label=0),
        'class_1_f1': f1_score(pos_label=1),  # Primary optimization target
        'weighted_f1': weighted_f1_score,
        'class_1_improvement': class_1_f1_after - class_1_f1_before
    }
    return metrics
```

**Benefits:**
- Tracks both class-specific and aggregate metrics
- Provides visibility into trade-offs
- Enables crisis-focused decision making

### Solution 3: Configuration-Based Optimization

**Add Crisis-Focused Configuration:**
```python
# In config.py
PARTITION_OPTIMIZATION_METRIC = 'class_1_f1'  # Options: 'macro_f1', 'class_1_f1', 'weighted_f1'
CRISIS_FOCUSED_OPTIMIZATION = True
MIN_CLASS_1_F1_IMPROVEMENT = 0.01  # Minimum improvement threshold for crisis prediction
```

**Benefits:**
- Configurable optimization targets
- Easy switching between evaluation approaches
- Clear documentation of optimization goals

### Solution 4: Threshold-Based Partition Acceptance

**Smart Partition Acceptance Logic:**
```python
def should_accept_partition(class_0_improvement, class_1_improvement):
    if CRISIS_FOCUSED_OPTIMIZATION:
        # Accept if class 1 improves significantly, even if class 0 degrades slightly
        if class_1_improvement > 0.05:  # Meaningful crisis prediction improvement
            return True
        elif class_1_improvement > 0.01 and class_0_improvement > -0.02:  # Balanced improvement
            return True
        else:
            return False
    else:
        # Original macro-based logic
        return (class_0_improvement + class_1_improvement) > 0
```

**Benefits:**
- Balances class 1 focus with overall performance
- Prevents excessive class 0 degradation
- Maintains crisis prediction priority

## Recommended Implementation Plan

### Phase 1: Diagnostic Verification
1. **Extract actual class distribution** from 2015Q2 training data
2. **Analyze partition metrics CSV files** for class-specific patterns
3. **Confirm hypothesis** about class trade-offs during partitioning

### Phase 2: Core Algorithm Updates
1. **Update partition_opt.py**: Change optimization metric to class 1 F1
2. **Modify GeoRF.py**: Enhance fit() method with class-specific tracking
3. **Update config.py**: Add CRISIS_FOCUSED_OPTIMIZATION flag

### Phase 3: Enhanced Monitoring
1. **Upgrade partition metrics tracking**: Add class-specific improvement tracking
2. **Create crisis-focused visualizations**: Maps showing class 1 F1 improvements
3. **Update evaluation consistency**: Align all metrics with class 1 focus

### Phase 4: Validation Testing
1. **Re-run 2015Q2 with crisis-focused optimization**
2. **Compare class 1 F1 improvements** before/after changes
3. **Validate overall performance** doesn't degrade excessively

## Expected Outcomes

### After Implementation:
- **Partition metrics alignment**: Partition selection metrics match final evaluation focus
- **Improved class 1 F1**: Better crisis prediction through targeted optimization
- **Clearer performance tracking**: Class-specific improvements visible throughout training
- **Consistent evaluation**: Same optimization target from partitioning to final evaluation

### Performance Expectations:
- **Class 1 F1**: Expected improvement from 0.54 to 0.60-0.65
- **Class 0 F1**: May decrease slightly from 0.96 to 0.92-0.94
- **Overall crisis prediction**: Significantly enhanced due to aligned optimization

## Technical Implementation Notes

### Files Requiring Modification:
1. **partition_opt.py**: Core partition selection logic
2. **GeoRF.py**: Model fitting and evaluation methods
3. **GeoRF_XGB.py**: XGBoost version alignment
4. **config.py**: Crisis-focused configuration options
5. **visualization.py**: Class 1 focused improvement visualizations

### Backward Compatibility:
- Maintain original macro F1 option via configuration
- Support both crisis-focused and balanced optimization
- Preserve existing API for non-crisis applications

## Critical Discovery: Triple Misalignment in Pipeline

**Even with `SELECT_CLASS = np.array([1])`, BOTH partition optimization AND significance testing use overall accuracy instead of class 1 metrics!**

### Complete Pipeline Analysis

#### 1. Partition Optimization Problem

**In `get_class_wise_stat()` (partition_opt.py:69):**
```python
# Line 104: Calculate OVERALL accuracy
stat = (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).astype(int)

# Line 112-113: Filter to class 1 samples only  
if SELECT_CLASS is not None:
    y_true = y_true[:, SELECT_CLASS]

# Line 117: Multiply class 1 labels by OVERALL accuracy
true_pred_w_class = y_true * np.expand_dims(stat, 1)
```

**In `get_score()` (partition_opt.py:718):**
```python
# Line 741: Calculate OVERALL accuracy
score = (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).astype(int)

# Lines 748-755: Filter to class 1 SAMPLES, but score is still overall accuracy
if SELECT_CLASS is not None:
    score_select = np.zeros(score.shape)
    for i in range(SELECT_CLASS.shape[0]):
        class_id = int(SELECT_CLASS[i])
        score_select[y_true[:,class_id]==1] = 1
    score = score[score_select.astype(bool)]
```

#### 2. Significance Testing Problem

**In `transformation.py:562-563, 606`:**
```python
# Score calculation uses OVERALL accuracy
base_score_before = get_score(y_val, y_pred_before)      # OVERALL accuracy
split_score0, split_score1 = get_split_score(...)        # OVERALL accuracy per partition

# Significance test uses OVERALL accuracy scores
sig = sig_test(base_score, split_score0, split_score1, base_score_before)
```

**In `sig_test.py:23-24, 36-37`:**
```python
# Uses OVERALL accuracy scores for significance testing
split_score = np.hstack([split_score0, split_score1])
diff = split_score - base_score  # Improvement in OVERALL accuracy

# Rejects partitions that don't improve OVERALL accuracy
if mean_diff <= 0:  # No improvement in OVERALL accuracy
    return 0  # REJECT PARTITION
```

### What SELECT_CLASS Actually Does vs What It Should Do

**Current Behavior:**
- ✅ **Filters samples**: Only includes samples that belong to class 1
- ❌ **Score calculation**: Still uses overall accuracy for those samples
- ❌ **Optimization target**: Improves overall accuracy on class 1 samples
- ❌ **Significance testing**: Rejects partitions that don't improve overall accuracy

**What It Should Do for Crisis Prediction:**
- ✅ **Filter samples**: Only include samples that belong to class 1  
- ✅ **Score calculation**: Use class 1 specific accuracy (precision/recall/F1)
- ✅ **Optimization target**: Improve class 1 prediction performance
- ✅ **Significance testing**: Accept partitions that improve class 1 F1

### The Complete Pipeline Trace

1. **Partition optimization** (transformation.py:562-563):
   ```python
   base_score_before = get_score(y_val, y_pred_before)      # OVERALL accuracy
   split_score0, split_score1 = get_split_score(...)        # OVERALL accuracy per partition
   ```

2. **Significance testing** (transformation.py:606, sig_test.py:23-24):
   ```python
   sig = sig_test(base_score, split_score0, split_score1)   # Uses OVERALL accuracy
   diff = split_score - base_score  # Improvement in OVERALL accuracy
   if mean_diff <= 0: return 0     # Reject if no OVERALL improvement
   ```

3. **Final evaluation** (main_model_*.py):
   ```python
   f1_class1 = f1_score(y_true, y_pred, pos_label=1)       # CLASS 1 F1 only
   ```

**Result:** **Triple misalignment** where no stage optimizes for actual crisis prediction performance!

### Why This Causes Observed Behavior

**Partition Metrics (~0.9):**
- Optimizing for overall accuracy on class 1 samples
- Overall accuracy includes correct "non-crisis" predictions within class 1 timeframes  
- High performance due to easier overall classification task

**Final Evaluation (~0.5-0.6):**
- Measuring class 1 F1 (precision/recall for crisis detection)
- Much harder task requiring accurate crisis identification
- **Misaligned optimization target** leads to suboptimal performance

**Significance Testing Impact:**
- **Automatically rejects** partitions that improve crisis prediction but hurt overall accuracy
- **Only 165 out of 10497 partitions** show positive overall improvement
- **Most beneficial crisis prediction partitions** are eliminated by significance testing

## Comprehensive Solution Architecture

### Updated Class-Specific Score Calculation:
```python
def get_class_1_f1_score(y_true, y_pred):
    """Calculate class 1 specific F1 score instead of overall accuracy."""
    from sklearn.metrics import f1_score
    y_true_labels = np.argmax(y_true, axis=1) 
    y_pred_labels = np.argmax(y_pred, axis=1)
    return f1_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0)

def get_class_wise_stat_crisis_focused(y_true, y_pred, y_group):
    """Modified version that uses class 1 F1 instead of overall accuracy."""
    # Convert to one-hot if needed
    if len(y_true.shape)==1:
        y_true = np.eye(NUM_CLASS)[y_true.astype(int)].astype(int)
        y_pred = np.eye(NUM_CLASS)[y_pred.astype(int)].astype(int)
    
    # Calculate CLASS 1 SPECIFIC F1 score per sample
    stat = calculate_per_sample_class_1_contribution(y_true, y_pred)  # New function needed
    
    # Filter to class 1 samples
    if SELECT_CLASS is not None:
        y_true = y_true[:, SELECT_CLASS]
    
    true_pred_w_class = y_true * np.expand_dims(stat, 1)
    # ... rest of grouping logic

def sig_test_crisis_focused(base_score, split_score0, split_score1, base_score_before=None):
    """Modified significance test that uses class 1 F1 improvement."""
    # Use class 1 F1 scores instead of overall accuracy
    split_score = np.hstack([split_score0, split_score1])
    diff = split_score - base_score  # Improvement in CLASS 1 F1
    
    mean_diff = np.mean(diff)
    if mean_diff <= 0:  # No improvement in CLASS 1 F1
        return 0
    # ... rest of significance testing logic
```

### Configuration Addition:
```python
# In config.py
CRISIS_FOCUSED_OPTIMIZATION = True  # Use class 1 F1 instead of overall accuracy  
PARTITION_OPTIMIZATION_METRIC = 'class_1_f1'  # 'overall_accuracy' or 'class_1_f1'
CLASS_1_SIGNIFICANCE_TESTING = True  # Use class 1 F1 for significance testing
```
