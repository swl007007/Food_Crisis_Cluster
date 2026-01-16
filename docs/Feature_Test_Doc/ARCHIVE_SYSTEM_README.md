# Visual Debug Archiving System

## Overview

The visual debug batch files now include an intelligent archiving system that:
1. **Preserves important visual debug files** from each run
2. **Renames folders with descriptive names** based on year, forecasting scope, and test date
3. **Cleans up temporary files** while keeping archived results
4. **Saves disk space** by only keeping essential files

## How It Works

### After Each Batch Completes

1. **Archive Creation**: A new folder is created with a descriptive name
   - Format: `result_GeoRF_YEAR_fsN_DATE_visual` or `result_GeoXGB_YEAR_fsN_DATE_visual`
   - Example: `result_GeoRF_2021_fs1_2021-06_visual`

2. **Important Files Copied**: Only essential files are preserved
   - From `/vis` directory (8 types of files)
   - From parent directory (2 files)

3. **Cleanup**: Original `result_GeoRF_*` or `result_GeoXGB_*` directories are deleted
   - Archived folders (ending with `_visual`) are **NOT** deleted
   - Temporary files are removed
   - Memory is released

## Archived Files

### From `/vis` Directory

| File | Description | Purpose |
|------|-------------|---------|
| `comprehensive_partition_metrics.csv` | Complete metrics for all partitions | Quantitative analysis of partition performance |
| `final_f1_performance_map.png` | F1 score map across regions | Visual summary of crisis prediction performance |
| `final_partition_map.png` | Final spatial partitions | Shows how space was divided |
| `overall_f1_improvement_map.png` | F1 improvements from partitioning | Shows benefit of spatial partitioning |
| `map_pct_err_all.png` | Overall error rates by region | Error distribution visualization |
| `map_pct_err_class1.png` | Crisis (class 1) errors by region | Crisis-specific error analysis |
| `train_error_by_polygon.csv` | Training errors per admin unit | Detailed error tracking |
| `score_details_*.csv` | Detailed scores for each partition round | Performance metrics at each depth |

### From Parent Directory

| File | Description | Purpose |
|------|-------------|---------|
| `correspondence_table_*.csv` | Admin unit to partition mapping | Links geographic units to model partitions |
| `log_print.txt` | Training and evaluation logs | Complete execution record |

## Folder Naming Convention

Archived folders follow this naming pattern:

```
result_[MODEL]_[YEAR]_fs[SCOPE]_[TEST_DATE]_visual
```

**Components:**
- `MODEL`: Either `GeoRF` (Random Forest) or `GeoXGB` (XGBoost)
- `YEAR`: Training year (2021-2024)
- `SCOPE`: Forecasting scope (1=4mo, 2=8mo, 3=12mo lag)
- `TEST_DATE`: Test date from correspondence table (e.g., 2021-06 for June 2021)
- `visual`: Suffix indicating this is a visual debug archive

**Examples:**
- `result_GeoRF_2021_fs1_2021-06_visual` - GeoRF, 2021, 4-month lag, tested on June 2021
- `result_GeoXGB_2023_fs2_2023-09_visual` - XGBoost, 2023, 8-month lag, tested on Sept 2023

## Expected Output Structure

After running the visual debug batches, you'll have:

```
Food_Crisis_Cluster/
├── result_GeoRF_2021_fs1_2021-01_visual/
│   ├── vis/
│   │   ├── comprehensive_partition_metrics.csv
│   │   ├── final_f1_performance_map.png
│   │   ├── final_partition_map.png
│   │   ├── overall_f1_improvement_map.png
│   │   ├── map_pct_err_all.png
│   │   ├── map_pct_err_class1.png
│   │   ├── train_error_by_polygon.csv
│   │   └── score_details_*.csv
│   ├── correspondence_table_2021-01.csv
│   └── log_print.txt
├── result_GeoRF_2021_fs2_2021-01_visual/
│   └── ... (same structure)
├── result_GeoRF_2021_fs3_2021-01_visual/
│   └── ... (same structure)
... (12 folders total for GeoRF: 4 years × 3 scopes)
├── result_GeoXGB_2021_fs1_2021-01_visual/
│   └── ... (same structure)
... (12 folders total for XGBoost: 4 years × 3 scopes)
├── results_df_gp_fs1_2021_2021.csv
├── results_df_gp_fs2_2021_2021.csv
└── ... (CSV result files preserved)
```

## Disk Space Management

### Space Saved

Original `result_GeoRF_*` directories can be **very large** (500MB - 2GB each) due to:
- Checkpoint files (trained models for each partition)
- Intermediate partition data
- Detailed contiguity refinement maps
- All partition round visualizations

Archived folders are **much smaller** (5-50MB each) because they only keep:
- Final summary visualizations
- Essential CSV files
- Logs for reproducibility

### Total Space Estimate

**Full Run (12 batches per model):**
- Without archiving: ~6-24 GB (all temporary files kept)
- With archiving: ~0.6-1.2 GB (only essential files)
- **Space savings: 80-90%**

## Files NOT Preserved

The following are intentionally **NOT** archived to save space:

### Checkpoint Files
- `checkpoints/rf_*` or `checkpoints/xgb_*` - Trained models for each partition
- **Reason**: Can be regenerated from data; very large files

### Intermediate Visualizations
- `contiguity_refinement_round_*_branch_*.png` - Contiguity refinement process
- `contiguity_swaps_round_*_branch_*.png` - Swap operations during refinement
- `partition_map_round_*_branch_*_scoped.png` - Intermediate partition maps
- `round*_branch*_f1_improvement_map.png` - Per-round improvement maps
- **Reason**: Intermediate steps; final results captured in summary files

### Partition Data
- `space_partitions/` - Detailed partition structure
- **Reason**: Captured in correspondence table and final maps

### Auxiliary Files
- `feature_name_reference.csv` - Feature metadata
- `val_coverage_by_group.csv` - Validation coverage statistics
- `grid_*.npy` - Grid representations
- `baseline_shap_*.png/csv` - SHAP analysis (if enabled)
- **Reason**: Auxiliary analysis; not core to visual debug

## Using Archived Results

### Finding Results for a Specific Run

Use the folder naming convention to locate results:

```bash
# Find all fs1 (4-month lag) results
ls -d result_GeoRF_*_fs1_*_visual

# Find all 2023 results
ls -d result_GeoRF_2023_*_visual

# Find all XGBoost results
ls -d result_GeoXGB_*_visual
```

### Comparing Across Forecasting Scopes

```bash
# Compare fs1 vs fs2 vs fs3 for 2021
ls -d result_GeoRF_2021_fs*_visual
```

### Batch Analysis of Archived Metrics

The preserved CSV files can be aggregated for analysis:

```python
import pandas as pd
import glob

# Load all comprehensive metrics
metrics_files = glob.glob("result_GeoRF_*_visual/vis/comprehensive_partition_metrics.csv")
all_metrics = pd.concat([pd.read_csv(f) for f in metrics_files])

# Load all correspondence tables
corr_files = glob.glob("result_GeoRF_*_visual/correspondence_table_*.csv")
all_correspondence = pd.concat([pd.read_csv(f) for f in corr_files])
```

## Troubleshooting

### Archive Folder Already Exists

If you see: `WARNING: Archive folder already exists, will overwrite`

**Cause**: A previous run with the same year/scope/date exists
**Solution**: The batch file will overwrite automatically; no action needed

### No result_GeoRF_* Directory Found

If you see: `WARNING: No result_GeoRF_* directory found to archive`

**Cause**: The Python script failed or didn't create output directory
**Solution**: Check Python script logs for errors; batch will continue to next iteration

### Archived Folder Still Deleted

If archived folders are being deleted:

**Check**: Ensure folder name ends with `_visual`
**Verify**: Cleanup subroutine skips folders matching `*_visual$` pattern

### Test Date Shows as "unknown"

If folder name is `result_GeoRF_2021_fs1_unknown_visual`:

**Cause**: No correspondence table found or filename doesn't match expected pattern
**Solution**: Folder is still valid; test date just couldn't be extracted from filename

## Best Practices

### 1. Monitor Disk Space

```bash
# Check size of archived folders
du -sh result_GeoRF_*_visual

# Total space used by archives
du -sh result_GeoRF_*_visual | awk '{sum+=$1} END {print sum}'
```

### 2. Periodic Cleanup

If disk space is limited, archive old results:

```bash
# Create backup archive of visual results
tar -czf visual_results_2021_2024.tar.gz result_Geo*_visual

# Verify archive integrity
tar -tzf visual_results_2021_2024.tar.gz

# Remove original folders after backup
rm -rf result_Geo*_visual
```

### 3. Selective Archiving

If you only need specific files, modify the `:archive_visual_files` subroutine in the batch file to skip certain files.

### 4. Cross-Reference with CSV Results

Always keep the main CSV result files (`results_df_*_fs*_*.csv`) alongside archived folders for complete analysis.

## Summary

The archiving system provides:
- ✅ **Space efficiency**: 80-90% space savings
- ✅ **Descriptive naming**: Easy identification of results
- ✅ **Essential preservation**: Key files for analysis retained
- ✅ **Automatic cleanup**: No manual intervention needed
- ✅ **Analysis-ready**: CSV files preserved for aggregation

For questions or issues with the archiving system, see `README_VISUAL_DEBUG_BATCHES.md` for additional troubleshooting steps.
