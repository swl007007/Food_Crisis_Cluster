# GeoRF Monthly Evaluation Refactoring Status

**Date**: 2025-11-10
**Objective**: Refactor GeoRF pipelines from quarterly multi-month TEST sets to single-month TEST sets with lag-adjusted 36-month TRAIN windows.

---

## ✅ COMPLETED (6/11 files)

### 1. **config.py** ✅
- Added `TRAIN_WINDOW_MONTHS = 36`
- Added `ACTIVE_LAG = min(ACTIVE_LAGS)` with fallback
- Added `DESIRED_TERMS = ["2023-01"]` (monthly evaluation config)
- Added `_parse_month_term()` function (flexible string/datetime/Period parsing)
- Added `_validate_desired_terms()` function

### 2. **config_visual.py** ✅
- Imported temporal configuration from main config
- Set `ACTIVE_LAGS`, `TRAIN_WINDOW_MONTHS`, `ACTIVE_LAG`, `DESIRED_TERMS`

### 3. **src/customize/customize.py** ✅
**Modified `train_test_split_rolling_window()` function:**
- **New Monthly Mode**: When `test_month` parameter provided
  - TEST set = exactly one month
  - TRAIN end = `test_month_start - active_lag months`
  - TRAIN start = `train_end - (train_window_months - 1)`
  - Validates: `train_end < test_start`, disjoint sets, chronological order
  - Comprehensive logging with temporal boundaries
- **Legacy Quarterly Mode**: When `need_terms` parameter provided (deprecated)
- **Fallback Annual Mode**: When neither parameter provided

### 4. **app/main_model_GF.py** ✅
**Refactored `run_temporal_evaluation()` function:**
- **New monthly loop**: Reads `DESIRED_TERMS` from config, parses to pd.Period
- **Split call**: Uses `test_month`, `active_lag`, `train_window_months` parameters
- **Results storage**: Changed from `'quarter': quarter` to `'month': test_month`
- **Predictions storage**: Changed `'month': quarter*3` to `'month': test_month`
- **DataFrame initialization**: Changed columns from `'quarter'` to `'month'`
- **Summary statistics**: Changed groupby from `['year', 'quarter']` to `['year', 'month']`
- **Legacy fallback**: Added deprecated quarterly mode for backward compatibility
- **All print statements**: Updated from `Q{quarter}` to `{test_month_period}`
- **Correspondence tables**: Updated to use `test_month` instead of `quarter`

### 5. **src/utils/save_results.py** ✅
- Updated docstring to note monthly granularity
- Documented that DataFrames contain 'month' column (1-12), not 'quarter' (1-4)
- File naming remains unchanged (uses year range, which is compatible)
- No code changes needed (already generic)

### 6. **src/utils/lag_schedules.py** ✅
- Added `validate_lag_boundaries(active_lag, train_end, test_start)` function
- Added `assert_max_lag_valid(lags, active_lag, context)` function
- Both functions prevent data leakage by validating temporal constraints

---

## 📋 REMAINING WORK (5/11 files)

### 7. **app/main_model_GF_visual_debug.py** 🔄
**Pattern to apply** (same as main_model_GF.py):

```python
# At top of run_temporal_evaluation():
from config import DESIRED_TERMS, ACTIVE_LAG, TRAIN_WINDOW_MONTHS, _parse_month_term
import pandas as pd

# Replace quarterly loop (around line 340-380):
if DESIRED_TERMS is not None and len(DESIRED_TERMS) > 0:
    months_to_evaluate = [_parse_month_term(term) for term in DESIRED_TERMS]
    progress_bar = tqdm(total=len(months_to_evaluate), desc="Monthly Evaluation", unit="month")

    for i, test_month_period in enumerate(months_to_evaluate):
        test_year = test_month_period.year
        test_month = test_month_period.month

        # Call split with new parameters
        split_result = train_test_split_rolling_window(
            X, y, X_loc, X_group, years, dates,
            test_month=test_month_period,
            active_lag=ACTIVE_LAG,
            train_window_months=TRAIN_WINDOW_MONTHS,
            admin_codes=admin_codes
        )

        # ... rest of loop (update quarter → test_month everywhere)

# Update results storage:
new_result_row = {
    'year': test_year,
    'month': test_month,  # Changed
    # ... rest
}

pred_data = {
    'year': np.full(len(ytest), test_year, dtype=np.int16),
    'month': np.full(len(ytest), test_month, dtype=np.int8),  # Changed
    # ... rest
}

# Update DataFrame initialization:
results_df = pd.DataFrame(columns=['year', 'month', 'precision(1)', ...])  # Changed
y_pred_test = pd.DataFrame(columns=['year', 'month', 'adm_code', ...])  # Changed

# Update summary:
if 'month' in results_df.columns:  # Changed
    print(results_df.groupby(['year', 'month'])[['f1(1)', 'f1_base(1)']].mean())
```

**Search and replace**:
- All `Q{quarter}` → `{test_month_period}`
- All `'quarter': quarter` → `'month': test_month`
- All `'quarter'` columns → `'month'`
- All `.groupby(['year', 'quarter'])` → `.groupby(['year', 'month'])`

---

### 8. **app/main_model_XGB.py** 🔄
**Identical pattern as main_model_GF.py**:
- Import `DESIRED_TERMS`, `ACTIVE_LAG`, `TRAIN_WINDOW_MONTHS`, `_parse_month_term` from config
- Replace quarterly loop with monthly loop
- Update split function call with `test_month`, `active_lag`, `train_window_months`
- Change results storage from quarter to month
- Update DataFrame columns, print statements, and summary statistics

---

### 9. **app/main_model_XGB_visual_debug.py** 🔄
**Identical pattern as main_model_GF_visual_debug.py**:
- Apply same changes as main_model_XGB.py
- Update all quarter references to month
- Ensure consistency with visual debug features

---

### 10. **src/feature/feature.py** 🔄 (OPTIONAL - LOW PRIORITY)
**Changes needed**:
- **Line 146-152**: Remove quarterly terms derivation (if exists)
  ```python
  # OLD (remove):
  terms = (df['date'].dt.month - 1) // 3 + 1
  df['terms'] = terms

  # No replacement needed - months already in date column
  ```

- **Add lag assertion** (optional):
  ```python
  from src.utils.lag_schedules import assert_max_lag_valid
  from config import ACTIVE_LAG

  # Before creating lagged features:
  assert_max_lag_valid(ACTIVE_LAGS, ACTIVE_LAG, context="feature engineering")
  ```

**Search for**:
- Any code that derives `terms` or `quarter` from date
- Any hardcoded references to quarters (1-4)

---

### 11. **src/preprocess/preprocess.py** 🔄 (OPTIONAL - LOW PRIORITY)
**Changes needed** (optional enhancements):

```python
from src.utils.lag_schedules import assert_max_lag_valid

# In load_and_preprocess_data() or wherever lag features are created:
from config import ACTIVE_LAGS, ACTIVE_LAG

# Add assertion before creating lag features (lines ~185-195):
assert_max_lag_valid(ACTIVE_LAGS, ACTIVE_LAG, context="crisis lag features")

for lag in ACTIVE_LAGS:
    df[f'fews_ipc_crisis_lag_{lag}'] = ...
```

**Optional end_cutoff parameter** (advanced):
- Add parameter to filter TRAIN data beyond `train_end` date
- Prevents leakage in rolling window features
- Not strictly necessary if split function already filters data

---

## 🔍 VALIDATION CHECKLIST

After completing remaining files, verify:

### Data Integrity
- [ ] `train_end == test_month - ACTIVE_LAG` (exactly)
- [ ] `len(train_months) >= TRAIN_WINDOW_MONTHS` (usually equals)
- [ ] TRAIN, TEST sets are disjoint (no overlap)
- [ ] Sets are chronological (TRAIN < gap < TEST)

### Feature Leakage Prevention
- [ ] `max(ACTIVE_LAGS) <= ACTIVE_LAG` for all lag operations
- [ ] No features computed using data beyond `train_end`
- [ ] Rolling windows respect temporal boundaries

### IO Consistency
- [ ] All result CSVs have 'month' column (not 'quarter')
- [ ] Month values are 1-12 (not 3, 6, 9, 12)
- [ ] Filenames use year range (already compatible)
- [ ] No quarterly aggregation in saved files

### Runtime Behavior
- [ ] All DESIRED_TERMS months processed
- [ ] Progress bars show "month" units
- [ ] Logs display monthly boundaries correctly
- [ ] Error messages use month terminology

---

## 🧪 TESTING STRATEGY

### 1. **Quick Config Test**
```python
# Test config imports and parsing
python -c "from config import DESIRED_TERMS, ACTIVE_LAG, _parse_month_term; print('✓ Config OK')"
```

### 2. **Split Function Test**
```python
# Test monthly split with known dates
python -c "
from src.customize.customize import train_test_split_rolling_window
import pandas as pd
import numpy as np

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
X_loc = np.random.rand(1000, 2)
X_group = np.random.randint(0, 50, 1000)
dates = pd.date_range('2019-01-01', periods=1000, freq='D')
years = dates.year.values

result = train_test_split_rolling_window(
    X, y, X_loc, X_group, years, dates,
    test_month='2023-01',
    active_lag=4,
    train_window_months=36
)
print('✓ Split function OK')
"
```

### 3. **Integration Test**
```bash
# Run single month evaluation
python app/main_model_GF.py --start_year 2023 --end_year 2023 --forecasting_scope 1
```

Expected output:
```
MONTHLY EVALUATION MODE
Testing months: ['2023-01']
Active lag: 4 months
Train window: 36 months
================================================================================
MONTHLY SPLIT | TEST=2023-01 | LAG=4 months
TRAIN: 2019-09-01 to 2022-08-31 (36 months, XXXX samples)
TEST:  2023-01-01 to 2023-01-31 (1 month, XX samples)
```

### 4. **Output Validation**
```python
# Verify monthly results
import pandas as pd

results = pd.read_csv('results_df_gp_fs1_2023_2023.csv')
assert 'month' in results.columns
assert 'quarter' not in results.columns
assert results['month'].isin(range(1, 13)).all()
print('✓ Results format OK')

predictions = pd.read_csv('y_pred_test_gp_fs1_2023_2023.csv')
assert 'month' in predictions.columns
assert 'quarter' not in predictions.columns
print('✓ Predictions format OK')
```

---

## 📊 EXPECTED BEHAVIOR CHANGES

### Before (Quarterly)
- TEST = 3 months (e.g., Jan-Mar 2023)
- TRAIN = 3 years ending when TEST quarter begins
- Results: `{'year': 2023, 'quarter': 1, ...}`
- Predictions: `{'year': 2023, 'quarter': 1, 'month': 3, ...}`

### After (Monthly)
- TEST = 1 month (e.g., 2023-01)
- TRAIN = 36 months ending `ACTIVE_LAG` months before TEST
- Results: `{'year': 2023, 'month': 1, ...}`
- Predictions: `{'year': 2023, 'month': 1, ...}`

---

## 🚨 COMMON PITFALLS

1. **Indentation errors**: Ensure proper indentation in monthly loop (4 spaces per level)
2. **Variable naming**: Don't mix `quarter` and `test_month` variables
3. **DataFrame columns**: Remove 'quarter' from ALL DataFrame initializations
4. **Print statements**: Update ALL print/log statements to use monthly terminology
5. **Error handling**: Update exception messages to reference months, not quarters
6. **Correspondence tables**: Pass `test_month` (int 1-12), not `quarter` (int 1-4)

---

## 📁 FILES SUMMARY

| File | Status | Lines Changed | Complexity |
|------|--------|---------------|------------|
| config.py | ✅ Complete | ~60 added | Medium |
| config_visual.py | ✅ Complete | ~10 added | Low |
| src/customize/customize.py | ✅ Complete | ~150 modified | High |
| app/main_model_GF.py | ✅ Complete | ~200 modified | High |
| src/utils/save_results.py | ✅ Complete | ~20 modified | Low |
| src/utils/lag_schedules.py | ✅ Complete | ~60 added | Medium |
| **app/main_model_GF_visual_debug.py** | 🔄 **Pending** | ~200 needed | High |
| **app/main_model_XGB.py** | 🔄 **Pending** | ~200 needed | High |
| **app/main_model_XGB_visual_debug.py** | 🔄 **Pending** | ~200 needed | High |
| **src/feature/feature.py** | 🔄 **Pending (optional)** | ~10 needed | Low |
| **src/preprocess/preprocess.py** | 🔄 **Pending (optional)** | ~10 needed | Low |

**Total Progress**: 6/11 files complete (54%)
**Core functionality**: 100% complete (config + split + main GF pipeline)
**Remaining**: Apply pattern to 3 similar files + 2 optional enhancements

---

## 🎯 NEXT STEPS

### Immediate (Required)
1. Apply the monthly pattern to `main_model_GF_visual_debug.py` (copy logic from `main_model_GF.py`)
2. Apply the monthly pattern to `main_model_XGB.py` (same changes as GF)
3. Apply the monthly pattern to `main_model_XGB_visual_debug.py` (same changes as GF visual)

### Optional (Enhancements)
4. Remove quarterly terms derivation in `feature.py`
5. Add lag assertions in `preprocess.py`

### Testing
6. Run integration test with single month
7. Validate output CSVs have correct structure
8. Verify no quarterly references in logs

---

## 💡 TIPS FOR COMPLETION

1. **Use main_model_GF.py as reference**: The completed file shows the exact pattern to apply
2. **Search and replace carefully**: Use editor's find-replace for systematic updates
3. **Test incrementally**: After each file, run a quick syntax check
4. **Keep legacy mode**: The fallback ensures backward compatibility
5. **Document changes**: Add comments where significant logic changed

---

## ✅ SIGN-OFF

**Core Refactoring**: Complete
**Tested**: Config + Split function working
**Production Ready**: After completing 3 remaining main files
**Backward Compatible**: Yes (via legacy quarterly mode fallback)

---

*This refactoring establishes a robust single-month TEST evaluation framework with proper temporal boundaries and leakage prevention.*
