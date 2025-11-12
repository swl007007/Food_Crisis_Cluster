# Remaining Changes for main_model_GF_visual_debug.py

## Completed
- ✅ DataFrame initialization (lines 141-147): Changed to 'month' column
- ✅ Monthly evaluation loop (lines 276-334): New monthly mode implemented
- ✅ Split function call (lines 327-334): Uses test_month, active_lag, train_window_months
- ✅ 2-layer model code indentation (lines 355-411): Fixed and updated

## Remaining (Quick Find/Replace)

### 1. Fix `else` block indentation (line 413)
**Current:**
```python
            else:
                # Single-layer model
```

**Change to:**
```python
                else:
                    # Single-layer model
```
Add 4 spaces to all lines in the else block (lines 413-537).

### 2. Update print statements (Find all and replace)
- Line 458: `Q{quarter} {test_year}` → `{test_month_period}`
- Line 476: `Q{quarter} {test_year}` → `{test_month_period}`
- Line 523: `Q{quarter} {test_year}` → `{test_month_period}`
- Line 527: `Q{quarter} {test_year}` → `{test_month_period}`
- Line 537: `Q{quarter} {test_year}` → `{test_month_period}`

### 3. Update results storage (line 545)
**Current:**
```python
'quarter': quarter,
```

**Change to:**
```python
'month': test_month,
```

### 4. Update predictions storage (lines 572-573, 595-596)
**Current:**
```python
'quarter': np.full(len(ytest), quarter, dtype=np.int8),
'month': np.full(len(ytest), quarter * 3, dtype=np.int8),
```

**Change to:**
```python
'month': np.full(len(ytest), test_month, dtype=np.int8),
```
Remove the 'quarter' line entirely.

### 5. Update correspondence table calls (lines 409, 535)
**Current:**
```python
create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, ...)
```

**Change to:**
```python
create_correspondence_table(df, years, dates, test_year, test_month, X_branch_id, ...)
```

### 6. Add legacy quarterly mode fallback (after line ~700)
Add before the closing of run_temporal_evaluation:
```python
    # === LEGACY QUARTERLY MODE FALLBACK ===
    else:
        print("\n[DEPRECATION WARNING] Using legacy quarterly mode.")
        print("Consider migrating to DESIRED_TERMS config for monthly evaluation.\n")
        # Use old quarterly loop...
        # (Can copy from main_model_GF.py lines 1027-1079)
```

### Quick Script for Batch Updates

```bash
# In the file main_model_GF_visual_debug.py, make these replacements:

# 1. Print statements
sed -i 's/Q{quarter} {test_year}/{test_month_period}/g' main_model_GF_visual_debug.py

# 2. Results storage
sed -i "s/'quarter': quarter,/'month': test_month,/g" main_model_GF_visual_debug.py

# 3. Predictions (remove quarter, update month)
sed -i "/'quarter': np.full(len(ytest), quarter, dtype=np.int8),/d" main_model_GF_visual_debug.py
sed -i "s/quarter \* 3/test_month/g" main_model_GF_visual_debug.py

# 4. Correspondence tables
sed -i 's/test_year, quarter, X_branch_id/test_year, test_month, X_branch_id/g' main_model_GF_visual_debug.py
```

## Summary
The file is ~80% complete. The monthly mode infrastructure is in place. The remaining changes are mostly find/replace operations to update variable names from `quarter` to `test_month` and fix indentation in the single-layer else block.

**Time estimate**: 10-15 minutes of careful find/replace or sed commands.
