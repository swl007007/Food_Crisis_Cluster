# Monthly-by-Month Processing for Visual Debug Mode

## Problem

When running visual debug mode with all 12 months processed together, memory usage increases dramatically due to:
- Partition map generation for each test period
- Metrics tracking across all spatial regions
- Improvement visualizations
- Large checkpoint files and intermediate data

This causes **memory allocation errors** like:
```
Failed to allocate bitmap
MemoryError
```

## Solution: Monthly Processing

Process **one month at a time** instead of all 12 months together:

1. **Loop structure**: Year → Forecasting Scope → **Individual Month**
2. **After each month**:
   - Save `results_df` and `y_pred_test` for that month
   - Archive important visual files
   - Delete result directories
   - Clear memory
3. **After all 12 months**: Combine monthly results into yearly file

## New Batch Files

### GeoRF Monthly Processing
```bash
run_georf_batches_2021_2024_visual_monthly.bat
```

### XGBoost Monthly Processing
```bash
run_xgboost_batches_2021_2024_visual_monthly.bat
```

## How It Works

### Processing Flow

```
For each YEAR (2021-2024):
    For each SCOPE (1, 2, 3):
        For each MONTH (01-12):
            1. Set DESIRED_TERMS to single month (e.g., "2021-01")
            2. Run Python script for that month only
            3. Extract results → monthly_results/results_df_*_YEAR_MONTH.csv
            4. Archive visuals → result_GeoRF_YEAR_fsN_YEAR-MONTH_visual/
            5. Delete result_GeoRF_* directories
            6. Clear memory
        After all 12 months:
            7. Combine monthly CSV files → results_df_*_YEAR_YEAR.csv
            8. Clean up monthly intermediate files
```

### File Management

**During Processing (per month):**
```
monthly_results/
├── results_df_gp_fs1_2021_01.csv
├── results_df_gp_fs1_2021_02.csv
├── ... (12 files per year/scope)
├── y_pred_test_gp_fs1_2021_01.csv
└── ... (12 prediction files)
```

**After Combining:**
```
results_df_gp_fs1_2021_2021.csv         (Combined 12 months)
y_pred_test_gp_fs1_2021_2021.csv        (Combined predictions)
```

**Archived Visual Folders:**
```
result_GeoRF_2021_fs1_2021-01_visual/   (January)
result_GeoRF_2021_fs1_2021-02_visual/   (February)
...
result_GeoRF_2021_fs1_2021-12_visual/   (December)
```

## Batch Count Comparison

### Old Approach (All 12 months together)
- **Total batches**: 12 (4 years × 3 scopes)
- **Processing**: 12 months at once per batch
- **Memory**: Accumulates across all 12 months
- **Risk**: High memory errors

### New Approach (One month at a time)
- **Total batches**: 144 (4 years × 3 scopes × 12 months)
- **Processing**: 1 month per batch
- **Memory**: Cleared after each month
- **Risk**: Minimal memory errors

## Memory Management

### Per-Month Cleanup

After each month completes:

1. **Extract Results**
   ```
   Copy: results_df_*.csv → monthly_results/
   Copy: y_pred_test_*.csv → monthly_results/
   Delete: Original CSV files
   ```

2. **Archive Visuals**
   ```
   Copy important files → result_GeoRF_*_visual/
   8-11 files per month archived
   ```

3. **Cleanup Directories**
   ```
   Delete: result_GeoRF_* (original, not _visual)
   Delete: temp_* files
   Delete: *.pkl files
   Delete: __pycache__ directories
   ```

4. **Force Memory Release**
   ```
   3-second pause to allow OS memory cleanup
   ```

### Python Memory Cleanup

The Python scripts naturally clear memory between runs since each month is a separate process invocation.

## Execution Time

### Comparison

**Old Approach (12 months together):**
- Per batch: ~30-45 minutes
- Total: ~6-9 hours (12 batches)
- Memory errors: Likely

**New Approach (1 month at a time):**
- Per month: ~2-4 minutes
- Per year/scope: ~24-48 minutes (12 months)
- Total: ~6-12 hours (144 batches)
- Memory errors: Rare

**Net Effect**: Slightly longer runtime but **much more reliable**

## Output Files

### Final Results (Same as Before)

```
results_df_gp_fs1_2021_2021.csv         (All 12 months combined)
results_df_gp_fs2_2021_2021.csv
results_df_gp_fs3_2021_2021.csv
... (12 files total: 4 years × 3 scopes)

y_pred_test_gp_fs1_2021_2021.csv
... (12 files total)
```

### Visual Archives (More Granular)

**Old approach**: 12 folders (1 per year/scope)
```
result_GeoRF_2021_fs1_2021-06_visual/   (Last test month only)
```

**New approach**: 144 folders (1 per month)
```
result_GeoRF_2021_fs1_2021-01_visual/
result_GeoRF_2021_fs1_2021-02_visual/
...
result_GeoRF_2021_fs1_2021-12_visual/
```

**Benefit**: Visual outputs for **every month** instead of just the last one

## Using the Monthly Batch Files

### Quick Start

```bash
# 1. Enable visual debug
python enable_visual_debug.py

# 2. Run monthly processing
run_georf_batches_2021_2024_visual_monthly.bat

# OR for XGBoost
run_xgboost_batches_2021_2024_visual_monthly.bat

# 3. Disable visual debug when done (optional)
python enable_visual_debug.py disable
```

### Progress Monitoring

The batch file shows detailed progress:

```
------ Batch 13/144: Year 2021, Scope 2, Month 01 ------
Monthly evaluation: 2021-01 ONLY
Visual debug: Enabled
Running: python app/main_model_GF.py --start_year 2021 --end_year 2021 --forecasting_scope 2
Environment: DESIRED_TERMS=2021-01
...
Extracting monthly results for 2021-01...
  - Saved: monthly_results\results_df_gp_fs2_2021_01.csv
Archiving visual debug files for 2021-01...
  - Archived to: result_GeoRF_2021_fs2_2021-01_visual
Cleaning up results folders...
Month 01 processing completed
```

### After All Months Complete

```
Combining monthly results for year 2021, scope 2...
Combined 12 monthly result files
  - Created: results_df_gp_fs2_2021_2021.csv
  - Created: y_pred_test_gp_fs2_2021_2021.csv
Cleaning up monthly intermediate files...
Year 2021, Scope 2 completed successfully
```

## Disk Space Considerations

### Temporary Storage

During processing, monthly intermediate files accumulate:
- 12 CSV files per year/scope (small, ~1-10MB each)
- Automatically cleaned up after combining

### Final Storage

**Visual archives**: More folders but same total size
- Old: 12 folders × 50MB = 600MB
- New: 144 folders × 5MB = 720MB
- **Difference**: Marginal (~20% more)

**Result files**: Identical
- Same combined CSV files as before

## Troubleshooting

### Monthly CSV Files Not Combining

**Symptom**: No combined yearly file created

**Check**:
```bash
dir monthly_results\results_df_*_2021_*.csv
```

**Solution**: Ensure all 12 monthly files exist before combining runs

### Visual Archives Missing

**Symptom**: No archived folders created

**Check**: Python script completed successfully for that month
**Solution**: Review Python logs for errors

### Memory Still Too High

**Symptom**: Still getting memory errors with monthly processing

**Solutions**:
1. Reduce `MAX_DEPTH` in config.py (fewer partitions)
2. Set `N_JOBS=1` (disable parallel processing)
3. Disable some visualizations temporarily
4. Run on machine with more RAM

### Combining Script Fails

**Error**: `pandas not found` or `glob not found`

**Solution**: Ensure pandas is installed
```bash
pip install pandas
```

## Comparison Table

| Aspect | Old (12 months together) | New (1 month at a time) |
|--------|-------------------------|------------------------|
| **Total batches** | 12 | 144 |
| **Memory usage** | High (accumulates) | Low (cleared each month) |
| **Memory errors** | Common | Rare |
| **Execution time** | 6-9 hours | 6-12 hours |
| **Visual archives** | 12 folders (last month only) | 144 folders (all months) |
| **Result files** | Combined yearly | Combined yearly (identical) |
| **Granularity** | Yearly/scope | Monthly |
| **Reliability** | Low (memory issues) | High |

## Best Practices

### 1. Monitor First Month

Watch the first month closely to ensure:
- Python script completes successfully
- Results extracted correctly
- Visuals archived properly
- Cleanup works as expected

### 2. Check Intermediate Files

Periodically verify monthly results are accumulating:
```bash
dir monthly_results
```

Should see growing number of CSV files.

### 3. Verify Combined Results

After year/scope completes, verify combined file has 12 months of data:
```python
import pandas as pd
df = pd.read_csv('results_df_gp_fs1_2021_2021.csv')
print(df.shape)  # Should have rows for all 12 months
print(df['date'].unique())  # Should show all 12 months
```

### 4. Archive Management

With 144 visual archive folders, consider:
- Compressing older years: `tar -czf 2021_visual_archives.tar.gz result_Geo*_2021_*_visual`
- Selective preservation: Keep only key months (Q1, Q2, Q3, Q4)
- External backup: Copy to external drive or cloud storage

## Summary

The monthly processing approach:
- ✅ **Solves memory issues** by processing one month at a time
- ✅ **Provides more granular visual outputs** (144 folders vs 12)
- ✅ **Maintains identical final results** (same combined CSV files)
- ✅ **Adds minimal overhead** (~20% more execution time)
- ✅ **Much more reliable** for long runs with visual debug

Use these batch files when visual debug mode causes memory errors with the standard yearly processing approach.
