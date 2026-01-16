---
**📋 MIGRATION NOTICE**

This document has been migrated to the new structured documentation system:
- **Current Usage**: See `CLAUDE.md` in root directory for user guide
- **Technical Design**: See `.ai/spatial-partitioning-optimization/design/architecture.md` for detailed design
- **Development Standards**: See `.ai/docs/` for foundation documents

For migration details, see `_migration_index.csv` and `_link_graph.md` in this directory.
---

# GeoRF Technical Design and Partition Algorithm

## Framework Overview

**GeoRF** is a spatially-explicit machine learning framework for acute food crisis prediction with:
- **Dual Model Support**: Random Forest and XGBoost with identical spatial partitioning logic
- **Monthly Evaluation**: All 12 months per year for fine-grained temporal analysis
- **Batch Processing**: 36 batches per model (12 years × 3 forecasting scopes)
- **4-Model Comparison**: Includes probit regression and FEWSNET baselines
- **Crisis Prediction Focus**: Class 1 (crisis) metrics - precision, recall, F1 score

## Partition Algorithm Details
  ================================
  
  
  - We compute c = y_true_value - true_pred_value, the observed errors per group/class.
  - We build b from get_c_b: it distributes the total errors (c_tot) across groups in proportion to their sample mass (base/base_tot). Intuitively b is what the error layout would look like if the parent branch were perfectly homogeneous.

  With c and b in place the loop maintains two ingredients:

  1. q – the relative risk (observed error / expected error) for each class inside the candidate hot-spot. It’s initialized by picking the top outliers per class and then re-estimated after every subset update (q = Σ c[s0] / Σ b[s0]). In spatial-scan language, q is the MLE of the “risk multiplier” inside the flagged subset.
  2. gscore (LTSS contribution) – for every group we compute
     g_i = Σ_class [ c_i * log(q_class) + b_i * (1 - q_class) ].
     That is the group’s contribution to the log-likelihood ratio comparing “this group lives in a shifted error regime (q ≠ 1)” vs the homogeneous null (q = 1). It’s exactly the form required by Linear-Time Subset Scanning (LTSS): once you calculate these per-item scores, the highest-scoring subset is the set of all records whose gscore exceeds some threshold.

  get_top_cells then sorts the g vector in descending order and chooses the top segment as s0 (the “needs a new branch” side) and the remainder as s1. Without flexing, it defaults to the top half, but FLEX_RATIO and flex_type tweak the subset size bounds so we don’t end up with a wildly unbalanced split. If flex_type uses sample counts ('n_sample'), the cumulative counts (cnt) derived
  from b_cnt are what enforce the allowable size range.
  So the sequence is:

  1. Aggregate validation stats per X_group.
  2. Build q=c/b.
  3. Iterate: compute g from current q, take the top-ranked groups (LTSS), recompute q from that subset, repeat.
  4. After convergence, s0/s1 define the binary partition that maximizes the log-likelihood ratio under the size constraints set by FLEX_RATIO.

  where c_i is the observed error mass for group i and b_i is what the error would look like there if the branch were perfectly homogeneous. A positive g_i means “this group is performing worse than the parent baseline under the current q.” get_top_cells then sorts those g_i values and, subject to the balance constraint from FLEX_RATIO / flex_type, collects the top segment into s0 (the
  candidate child branch) and the remainder into s1. That’s the LTSS step: once you have additive scores like g_i, the optimal subset with the highest total log-likelihood ratio is simply the set of items whose g_i exceed a threshold, so sorting and taking the top block gives you the best branch.

  Step 4 (parameter update)
  With that subset in hand, the algorithm recomputes the relative-risk parameters from the selected groups:

  q_class = Σ_{i∈s0} c_{i,class} / Σ_{i∈s0} b_{i,class}

  Those ratios are the MLE of “how much worse than baseline” the child branch behaves for each class. Once q is updated, the code loops: rebuild g using the new q, re-select s0, update q, and so on. In practice the subset stabilizes after one or two passes (because you’ve already picked the groups that inflate the log-likelihood the most), so the second iteration usually reproduces the
  same s0 and q, at which point you’re done. The comments about convergence thresholds are right—the code doesn’t currently stop on a tolerance, but the fixed-point behaviour means the loop naturally settles into a steady state.

Rank groups by their g_i under the current q, cut them according to the flex rules, recompute q from the chosen subset, and repeat until the ranking no longer changes.

  Contiguity smoothing and the later significance test happen on top of that, but the core split decision is exactly the LTSS ranking you described.

---

## Current Pipeline Architecture

### Evaluation System

**Monthly Evaluation (Not Quarterly):**
- Evaluates all 12 months per year (Jan-Dec)
- Provides fine-grained temporal analysis
- Controlled by DESIRED_TERMS environment variable in batch scripts
- Each result file contains data for all 12 months of that year

### Forecasting Scopes

The framework supports 3 forecasting scopes (not 4):

1. **Scope 1**: 4-month lag forecasting
2. **Scope 2**: 8-month lag forecasting
3. **Scope 3**: 12-month lag forecasting

### Batch Processing System

**GeoRF Batches:**
```
Total: 36 batches = 12 years × 3 scopes
Years: 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
Each year: All 12 months evaluated in single batch
Memory cleanup: Between batches with retry logic
Output: results_df_gp_fs{scope}_{year}_{year}.csv
```

**XGBoost Batches:**
```
Total: 36 batches = 12 years × 3 scopes
Years: Same 12 years as GeoRF
Each year: All 12 months evaluated in single batch
Memory cleanup: More aggressive due to XGBoost memory requirements
Output: results_df_xgb_gp_fs{scope}_{year}_{year}.csv
```

### 4-Model Comparison Framework

The framework includes comprehensive baseline comparisons:

**1. Probit Regression Baseline**
- Simple regression with lagged crisis variables
- Supports all 3 forecasting scopes
- File: `baseline_probit_results/baseline_probit_results_fs{1-3}.csv`

**2. FEWSNET Official Predictions Baseline**
- Uses official FEWS NET predictions
- Scope 1: `pred_near_lag1` (4-month forecasting)
- Scope 2: `pred_med_lag2` (8-month forecasting)
- Scope 3: Not supported (FEWSNET doesn't provide longer-term predictions)
- Monthly granularity processing
- File: `fewsnet_baseline_results/fewsnet_baseline_results_fs{1-2}.csv`

**3. GeoRF (Random Forest)**
- Geo-aware Random Forest with spatial partitioning
- Supports all 3 forecasting scopes
- Monthly evaluation (12 months per year)
- Files: `results_df_gp_fs{scope}_{year}_{year}.csv` (36 files total)

**4. XGBoost (GeoRF_XGB)**
- XGBoost with same spatial partitioning logic
- Supports all 3 forecasting scopes
- Monthly evaluation (12 months per year)
- Files: `results_df_xgb_gp_fs{scope}_{year}_{year}.csv` (36 files total)

### Crisis Prediction Focus (Class 1 Only)

All models report only Class 1 (crisis) metrics:

- **Precision (Class 1)**: `true_crisis_predictions / all_crisis_predictions`
- **Recall (Class 1)**: `true_crisis_predictions / all_actual_crises`
- **F1 Score (Class 1)**: Harmonic mean of precision and recall

**Why Class 1 Focus:**
- Crisis prediction is the primary objective
- Class imbalance makes overall accuracy misleading
- Policy relevance: False negatives (missed crises) have higher costs
- Alignment with operational food security early warning systems

### File Naming Convention

**GeoRF Results:**
- Pattern: `results_df_gp_fs{scope}_{year}_{year}.csv`
- Example: `results_df_gp_fs1_2024_2024.csv` (4-month lag, 2024, all 12 months)
- Suffix `gp` = GeoRF with polygon assignment

**XGBoost Results:**
- Pattern: `results_df_xgb_gp_fs{scope}_{year}_{year}.csv`
- Example: `results_df_xgb_gp_fs2_2023_2023.csv` (8-month lag, 2023, all 12 months)
- Suffix `xgb_gp` = XGBoost with polygon assignment

**Baseline Results:**
- Probit: `baseline_probit_results_fs{1-3}.csv`
- FEWSNET: `fewsnet_baseline_results_fs{1-2}.csv`

### Comparison Visualization

**Script:** `app/final/georf_vs_baseline_comparison_plot.py`

**Features:**
- Auto-detects all available result files using glob patterns
- Combines multiple batch files for each model
- Creates dynamic subplot grid for available forecasting scopes
- Shows time series for precision, recall, F1 score
- Handles missing data gracefully (e.g., FEWSNET scope 3)
- Outputs: `model_comparison_class1_focus.png`

**Summary Statistics:**
- Unweighted averages across time periods
- Separate statistics per forecasting scope
- Console output with detailed performance metrics

### Memory Management

**Batch Processing Benefits:**
- Prevents memory leakage during long temporal evaluations
- Cleans up result directories between batches
- Multiple cleanup attempts with retry logic for locked files
- Temporary file removal (__pycache__, pickle files)
- Windows memory release with timeout between batches

**Cleanup Strategy:**
- Pre-execution cleanup of old result directories
- Post-execution garbage collection
- Force cleanup flag passed to Python scripts
- Robust directory removal with error handling

### Production Configuration

**Default Settings:**
```python
assignment = 'polygons'              # FEWSNET admin boundaries
CONTIGUITY_TYPE = 'polygon'          # Polygon-based contiguity
USE_ADJACENCY_MATRIX = True          # True polygon adjacency (recommended)
MIN_DEPTH = 1                        # Minimum partition depth
MAX_DEPTH = 4                        # Maximum partition depth
N_JOBS = 32                          # Parallel processing cores
```

**Batch Scripts:**
- `run_georf_batches.bat`: Full GeoRF production run (36 batches)
- `run_xgboost_batches.bat`: Full XGBoost production run (36 batches)
- `test_georf_batches.bat`: Quick GeoRF testing (4 batches)
- `test_xgboost_batches.bat`: Quick XGBoost testing (4 batches)

### Complete Pipeline Execution

**Step 1: Run Baselines**
```bash
python app/final/baseline_probit_regression.py
python app/final/fewsnet_baseline_evaluation.py
```

**Step 2: Run Main Models**
```bash
run_georf_batches.bat      # 36 batches: 12 years × 3 scopes, monthly evaluation
run_xgboost_batches.bat    # 36 batches: 12 years × 3 scopes, monthly evaluation
```

**Step 3: Generate Comparison**
```bash
python app/final/georf_vs_baseline_comparison_plot.py
```

**Total Output:**
- 3 probit baseline files (scopes 1-3)
- 2 FEWSNET baseline files (scopes 1-2 only)
- 36 GeoRF result files (12 years × 3 scopes)
- 36 XGBoost result files (12 years × 3 scopes)
- 1 comparison visualization
- All with monthly granularity (12 months per year)

---

## Implementation Notes

### Monthly vs Quarterly
- **Previous**: Quarterly evaluation (Q1-Q4)
- **Current**: Monthly evaluation (Jan-Dec, all 12 months)
- **Control**: DESIRED_TERMS environment variable
- **Impact**: Finer temporal resolution, more detailed performance analysis

### Batch Count Changes
- **Previous GeoRF**: 20 batches (5 time periods × 4 scopes)
- **Current GeoRF**: 36 batches (12 years × 3 scopes)
- **Previous XGBoost**: 40 batches (10 years × 4 scopes)
- **Current XGBoost**: 36 batches (12 years × 3 scopes)
- **Reason**: Alignment with monthly evaluation, removal of scope 4

### Forecasting Scope Changes
- **Previous**: 4 scopes (3mo, 6mo, 9mo, 12mo)
- **Current**: 3 scopes (4mo, 8mo, 12mo)
- **FEWSNET**: Only scopes 1-2 supported

### File Naming Updates
- **GeoRF**: `gf` → `gp` (polygon assignment explicit)
- **XGBoost**: `xgb` → `xgb_gp` (polygon assignment explicit)
- **Year Range**: Both start and end year in filename (e.g., `2024_2024`)
- **Content**: Each file contains all 12 months of data