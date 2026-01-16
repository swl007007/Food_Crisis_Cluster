# Visual Debug Batch Files (2021-2024 Focus)

This directory contains specialized batch files for running GeoRF and XGBoost evaluations with full visual debug enabled, focusing on recent years (2021-2024) for faster iteration and comprehensive visual analysis.

## Quick Start

### 1. Enable Visual Debug Settings

Run the helper script to automatically configure visual debug mode:

```bash
python enable_visual_debug.py
```

This will enable:
- `config.py`: `VIS_DEBUG_MODE = True`
- `app/main_model_GF.py`: `track_partition_metrics = True`, `enable_metrics_maps = True`
- `app/main_model_XGB.py`: `track_partition_metrics = True`, `enable_metrics_maps = True`

### 2. Run Visual Debug Batches

**GeoRF (Random Forest) with Visual Debug:**
```bash
run_georf_batches_2021_2024_visual.bat
```

**XGBoost with Visual Debug:**
```bash
run_xgboost_batches_2021_2024_visual.bat
```

### 3. Disable Visual Debug (Optional)

When done, restore production settings:

```bash
python enable_visual_debug.py disable
```

---

## Batch File Details

### GeoRF Visual Debug Batch
**File:** `run_georf_batches_2021_2024_visual.bat`

**Configuration:**
- **Years:** 2021-2024 (4 years)
- **Evaluation:** Monthly (all 12 months per year)
- **Forecasting Scopes:** 3 (4-month, 8-month, 12-month lags)
- **Total Batches:** 12 (4 years × 3 scopes)
- **Visual Outputs:** Enabled

**Expected Outputs:**
- Result files: `results_df_gp_fs{1-3}_{2021-2024}_{2021-2024}.csv`
- Visual directories: `result_GeoRF_*/vis/`
- Metrics directories: `result_GeoRF_*/partition_metrics/`

### XGBoost Visual Debug Batch
**File:** `run_xgboost_batches_2021_2024_visual.bat`

**Configuration:**
- **Years:** 2021-2024 (4 years)
- **Evaluation:** Monthly (all 12 months per year)
- **Forecasting Scopes:** 3 (4-month, 8-month, 12-month lags)
- **Total Batches:** 12 (4 years × 3 scopes)
- **Visual Outputs:** Enabled

**Expected Outputs:**
- Result files: `results_df_xgb_gp_fs{1-3}_{2021-2024}_{2021-2024}.csv`
- Visual directories: `result_GeoXGB_*/vis/`
- Metrics directories: `result_GeoXGB_*/partition_metrics/`

---

## Visual Outputs

When visual debug is enabled, each batch generates:

### 1. Partition Maps
**Location:** `result_Geo*/vis/partition_*.png`

Shows the spatial partitions created by the hierarchical partitioning algorithm with color-coded regions.

**Example:** `partition_2024_Jan.png`

### 2. Performance Grids
**Location:** `result_Geo*/vis/performance_*.png`

Displays model performance metrics across spatial regions:
- Class 0 (non-crisis) performance
- Class 1 (crisis) performance
- Comparison with baseline Random Forest

**Example:** `performance_class1_2024_Jan.png`

### 3. Partition Metrics CSV
**Location:** `result_Geo*/partition_metrics/partition_metrics_round*.csv`

Contains detailed metrics for each spatial partition:
- F1 score before and after partition
- Accuracy before and after partition
- Improvement statistics
- Group-level performance

**Columns:**
- `group_id`: Spatial group identifier
- `f1_before`: F1 score before partition
- `f1_after`: F1 score after partition
- `f1_improvement`: F1 improvement from partition
- `accuracy_before`, `accuracy_after`, `accuracy_improvement`: Similar for accuracy

### 4. Improvement Maps
**Location:** `result_Geo*/partition_metrics/*_improvement.png`

Geographic visualization of performance improvements:
- Shows which regions benefited from partitioning
- Color-coded by improvement magnitude
- Highlights areas where local models outperform global model

**Example:** `f1_improvement_round0_branchroot.png`

---

## Comparison with Standard Batches

### Standard Production Batches
**Files:** `run_georf_batches.bat`, `run_xgboost_batches.bat`

- **Years:** 2013-2024 (12 years)
- **Total Batches:** 36 (12 years × 3 scopes)
- **Visual Debug:** Disabled
- **Speed:** Faster (no visualization overhead)
- **Use Case:** Production runs, performance evaluation

### Visual Debug Batches (This Set)
**Files:** `run_georf_batches_2021_2024_visual.bat`, `run_xgboost_batches_2021_2024_visual.bat`

- **Years:** 2021-2024 (4 years)
- **Total Batches:** 12 (4 years × 3 scopes)
- **Visual Debug:** Enabled
- **Speed:** Slower (comprehensive visualization)
- **Use Case:** Algorithm debugging, spatial analysis, presentation materials

---

## Execution Time Estimates

### With Visual Debug Enabled

**Per Batch (1 year, 1 scope, 12 months):**
- **GeoRF:** ~30-45 minutes (depending on partitioning depth)
- **XGBoost:** ~45-60 minutes (more complex models)

**Complete Runs:**
- **GeoRF (12 batches):** ~6-9 hours
- **XGBoost (12 batches):** ~9-12 hours

**Factors Affecting Speed:**
- Number of spatial partitions created
- Partition depth (MAX_DEPTH in config)
- Visualization complexity (number of maps generated)
- System resources (CPU, memory)

### Without Visual Debug (Production)

**Per Batch:**
- **GeoRF:** ~20-30 minutes
- **XGBoost:** ~30-40 minutes

Approximately **30-40% faster** without visual debug overhead.

---

## Troubleshooting

### Visual Debug Settings Not Taking Effect

**Check settings manually:**

1. **config.py** (around line 293):
   ```python
   VIS_DEBUG_MODE = True  # Should be True
   ```

2. **app/main_model_GF.py** (around line 1129-1130):
   ```python
   track_partition_metrics = True  # Should be True
   enable_metrics_maps = True      # Should be True
   ```

3. **app/main_model_XGB.py** (around line 1129-1130):
   ```python
   track_partition_metrics = True  # Should be True
   enable_metrics_maps = True      # Should be True
   ```

### No Visual Outputs Generated

**Possible causes:**
- Visual debug settings not enabled (run `enable_visual_debug.py`)
- Result directories cleaned up between batches (expected behavior)
- Partition depth too shallow (increase MAX_DEPTH in config.py)
- Errors during visualization (check log files)

**Solution:**
- Verify settings with `enable_visual_debug.py`
- Check `result_Geo*/log_print.txt` for visualization errors
- Copy visual outputs immediately after batch completes

### Memory Issues with Visual Debug

**Symptoms:**
- Python crashes during batch processing
- "MemoryError" messages
- System becomes unresponsive

**Solutions:**
1. Reduce parallel jobs: Lower `N_JOBS` in `config.py`
2. Process one scope at a time: Comment out scopes in batch file
3. Disable some visualizations: Set `enable_metrics_maps = False`
4. Increase system memory or use machine with more RAM

### Batch Script Pauses/Hangs

**At startup:**
- Expected behavior - script waits for user confirmation
- Press any key to continue after verifying settings

**During execution:**
- Check if Python script encountered an error
- Review `log_print.txt` in result directories
- Check system resources (Task Manager)

---

## Manual Configuration

If the `enable_visual_debug.py` script doesn't work, configure manually:

### Step 1: Edit config.py

Find and modify (around line 293):
```python
# Change from:
VIS_DEBUG_MODE = False

# To:
VIS_DEBUG_MODE = True
```

### Step 2: Edit app/main_model_GF.py

Find and modify (around line 1129-1130):
```python
# Change from:
track_partition_metrics = False
enable_metrics_maps = False

# To:
track_partition_metrics = True
enable_metrics_maps = True
```

### Step 3: Edit app/main_model_XGB.py

Find and modify (around line 1129-1130):
```python
# Change from:
track_partition_metrics = False
enable_metrics_maps = False

# To:
track_partition_metrics = True
enable_metrics_maps = True
```

---

## Output File Organization

### Result Files (Preserved)
These are **NOT** cleaned up between batches:

```
results_df_gp_fs1_2021_2021.csv
results_df_gp_fs1_2022_2022.csv
results_df_gp_fs1_2023_2023.csv
results_df_gp_fs1_2024_2024.csv
...
results_df_xgb_gp_fs1_2021_2021.csv
results_df_xgb_gp_fs1_2022_2022.csv
...
```

### Visual Output Directories (Cleaned)
These **ARE** cleaned up between batches:

```
result_GeoRF_*/
├── vis/
│   ├── partition_2024_Jan.png
│   ├── partition_2024_Feb.png
│   └── performance_class1_2024_Jan.png
└── partition_metrics/
    ├── partition_metrics_round0_branchroot.csv
    └── f1_improvement_round0_branchroot.png
```

**Important:** Copy any needed visualizations **immediately** after each batch completes, as these directories will be deleted before the next batch starts.

---

## Best Practices

### For Debugging/Analysis

1. **Enable visual debug** before starting
2. **Run one scope at a time** to carefully examine outputs
3. **Copy visualizations** immediately after each batch
4. **Review metrics CSV files** for quantitative analysis
5. **Check partition maps** for spatial coherence

### For Production Runs

1. **Disable visual debug** for faster execution
2. **Use standard batch files** (run_georf_batches.bat, run_xgboost_batches.bat)
3. **Process all years** for complete temporal coverage
4. **Enable only when needed** for specific analysis

### For Presentations/Papers

1. **Run visual debug batches** for specific years of interest
2. **Select representative months** for figure creation
3. **Archive all visualizations** in separate directory
4. **Document parameters** used for reproducibility

---

## Additional Notes

### DESIRED_TERMS Environment Variable

Both batch files set the `DESIRED_TERMS` environment variable to specify which months to evaluate:

```batch
set "DESIRED_TERMS=2024-01,2024-02,2024-03,2024-04,2024-05,2024-06,2024-07,2024-08,2024-09,2024-10,2024-11,2024-12"
```

This ensures all 12 months are evaluated in each batch.

### Memory Cleanup

Between batches, the scripts perform:
- Result directory cleanup (result_GeoRF_*, result_GeoXGB_*)
- Temporary file deletion (temp_*, *.pkl)
- Python cache cleanup (__pycache__)
- 5-second forced memory release

This prevents memory leakage during long batch runs.

### Batch Progress Tracking

The scripts display:
- Current batch number (e.g., "Batch 5/12")
- Year and forecasting scope being processed
- DESIRED_TERMS environment variable value
- Success/failure status of each batch

---

## Related Files

- `run_georf_batches.bat` - Full GeoRF production run (2013-2024, 36 batches)
- `run_xgboost_batches.bat` - Full XGBoost production run (2013-2024, 36 batches)
- `test_georf_batches.bat` - Quick GeoRF test (2 years, 4 batches)
- `test_xgboost_batches.bat` - Quick XGBoost test (2 years, 4 batches)
- `enable_visual_debug.py` - Helper script to toggle visual debug settings

---

## Questions or Issues?

If you encounter problems or have questions about these batch files:

1. Check this README for troubleshooting steps
2. Review the main documentation in `CLAUDE.md`
3. Examine log files in `result_Geo*/log_print.txt`
4. Verify visual debug settings are correctly enabled
