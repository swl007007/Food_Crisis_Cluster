# Quick Start Guide: Visual Debug Mode

## Problem: Memory Errors with Visual Debug

Getting errors like "Failed to allocate bitmap"? This is because visual debug mode creates many visualizations that consume large amounts of memory.

## Solution: Use Monthly Processing

Process **one month at a time** instead of all 12 months together.

---

## Quick Start (RECOMMENDED)

### Step 1: Enable Visual Debug
```bash
python enable_visual_debug.py
```

### Step 2: Run Monthly Processing
```bash
# GeoRF (Random Forest)
run_georf_batches_2021_2024_visual_monthly.bat

# OR XGBoost
run_xgboost_batches_2021_2024_visual_monthly.bat
```

### Step 3: Disable When Done (Optional)
```bash
python enable_visual_debug.py disable
```

---

## What You Get

### Final Result Files (Same as Before)
```
results_df_gp_fs1_2021_2021.csv         # All 12 months combined
results_df_gp_fs2_2021_2021.csv
results_df_gp_fs3_2021_2021.csv
... (12 files: 4 years × 3 scopes)

y_pred_test_gp_fs1_2021_2021.csv
... (12 files)
```

### Visual Archives (One Per Month!)
```
result_GeoRF_2021_fs1_2021-01_visual/   # January visuals
result_GeoRF_2021_fs1_2021-02_visual/   # February visuals
...
result_GeoRF_2021_fs1_2021-12_visual/   # December visuals

... (144 folders total: 4 years × 3 scopes × 12 months)
```

Each folder contains:
- Final partition map
- F1 performance map
- Overall improvement map
- Error maps (all classes and class 1)
- Comprehensive metrics CSV
- Score details for all partition rounds
- Correspondence table
- Training logs

---

## Processing Details

### Loop Structure
```
For each YEAR (2021-2024):
    For each SCOPE (fs1, fs2, fs3):
        For each MONTH (01-12):
            ✓ Process 1 month only
            ✓ Save results
            ✓ Archive visuals
            ✓ Delete temp files
            ✓ Clear memory
        ✓ Combine 12 monthly results into yearly file
```

### Total Batches
- **144 monthly batches** (4 years × 3 scopes × 12 months)
- Approximately **2-4 minutes per month**
- Total time: **6-12 hours** (GeoRF) or **9-15 hours** (XGBoost)

### Memory Management
- ✅ Each month runs as separate process
- ✅ Memory cleared between months
- ✅ Temp files deleted after each month
- ✅ Visual archives preserved
- ✅ Results automatically combined

---

## File Comparison

### Option 1: Yearly Processing (May Cause Errors)

**Batch files:**
- `run_georf_batches_2021_2024_visual.bat`
- `run_xgboost_batches_2021_2024_visual.bat`

**Characteristics:**
- 12 batches (4 years × 3 scopes)
- Processes 12 months at once
- ❌ High memory usage
- ❌ Likely to cause "Failed to allocate bitmap" error
- 12 visual archive folders

### Option 2: Monthly Processing (RECOMMENDED)

**Batch files:**
- `run_georf_batches_2021_2024_visual_monthly.bat`
- `run_xgboost_batches_2021_2024_visual_monthly.bat`

**Characteristics:**
- 144 batches (4 years × 3 scopes × 12 months)
- Processes 1 month at a time
- ✅ Low memory usage
- ✅ Reliable execution
- 144 visual archive folders (more detailed!)

---

## Progress Monitoring

Watch the console output to track progress:

```
------ Batch 13/144: Year 2021, Scope 2, Month 01 ------
Monthly evaluation: 2021-01 ONLY
Visual debug: Enabled
Running: python app/main_model_GF.py --start_year 2021 --end_year 2021 --forecasting_scope 2
Environment: DESIRED_TERMS=2021-01

Batch 13 completed successfully
Extracting monthly results for 2021-01...
  - Saved: monthly_results\results_df_gp_fs2_2021_01.csv
  - Saved: monthly_results\y_pred_test_gp_fs2_2021_01.csv
Archiving visual debug files for 2021-01...
  - Archived to: result_GeoRF_2021_fs2_2021-01_visual
Cleaning up results folders...
Month 01 processing completed
```

After all 12 months:
```
Combining monthly results for year 2021, scope 2...
Combined 12 monthly result files
  - Created: results_df_gp_fs2_2021_2021.csv
  - Created: y_pred_test_gp_fs2_2021_2021.csv
Year 2021, Scope 2 completed successfully
```

---

## Troubleshooting

### Still Getting Memory Errors?

Even with monthly processing, if you still see memory errors:

1. **Reduce partition depth**
   - Edit `config.py`
   - Change `MAX_DEPTH = 3` (or lower)

2. **Disable parallel processing**
   - Edit `config.py`
   - Change `N_JOBS = 1`

3. **Use machine with more RAM**
   - Visual debug needs at least 16GB RAM
   - 32GB+ recommended

### Visuals Not Being Created?

Check that visual debug is enabled:
```bash
python enable_visual_debug.py
```

Verify settings:
- `config.py`: `VIS_DEBUG_MODE = True`
- `app/main_model_GF.py`: `track_partition_metrics = True`
- `app/main_model_GF.py`: `enable_metrics_maps = True`

### Combined Files Missing?

If yearly combined files aren't created after 12 months:

1. Check monthly files exist:
   ```bash
   dir monthly_results
   ```

2. Manually combine if needed:
   ```python
   import pandas as pd
   import glob

   pattern = 'monthly_results/results_df_gp_fs1_2021_*.csv'
   files = sorted(glob.glob(pattern))
   df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
   df.to_csv('results_df_gp_fs1_2021_2021.csv', index=False)
   ```

---

## Disk Space Requirements

### During Processing
- **Monthly intermediate files**: ~100-500MB per year/scope
- **Automatically cleaned up** after combining

### Final Storage
- **Combined result files**: ~10-50MB per year/scope (12 files total)
- **Visual archives**: ~5-10MB per month (144 folders = ~720MB-1.4GB)
- **Total**: ~1-2GB for complete 2021-2024 visual debug run

---

## When to Use Each Mode

### Use Yearly Processing (`*_visual.bat`)
- ✅ You have 32GB+ RAM
- ✅ Only need final month's visuals
- ✅ Want faster execution (fewer batches)

### Use Monthly Processing (`*_monthly.bat`)
- ✅ Getting memory errors with yearly mode
- ✅ Want visuals for every month (more detailed)
- ✅ Have limited RAM (16GB or less)
- ✅ Need reliable execution

---

## Summary

**Recommended workflow:**

1. Enable visual debug: `python enable_visual_debug.py`
2. Run monthly processing: `run_georf_batches_2021_2024_visual_monthly.bat`
3. Wait 6-12 hours for completion
4. Get results:
   - Combined CSV files with all 12 months
   - 144 visual archive folders (1 per month)
5. Disable visual debug: `python enable_visual_debug.py disable`

**Result:** Reliable visual debug execution with comprehensive monthly outputs and no memory errors.

---

## More Information

- **Monthly processing details**: `README_MONTHLY_PROCESSING.md`
- **Visual debug batches**: `README_VISUAL_DEBUG_BATCHES.md`
- **Archive system**: `ARCHIVE_SYSTEM_README.md`
- **Main documentation**: `CLAUDE.md`
