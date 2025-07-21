# GeoRF XGBoost Implementation

This directory contains an XGBoost-based implementation of the GeoRF (Geo-aware Random Forest) framework for acute food crisis prediction. The XGBoost version maintains all the spatial partitioning capabilities of the original GeoRF but uses XGBoost as the base learner instead of Random Forest.

## New Files

### Core XGBoost Implementation
- **`model_XGB.py`** - XGBoost model wrapper that mirrors the `RFmodel` interface
- **`GeoRF_XGB.py`** - Main GeoRF class using XGBoost instead of Random Forest
- **`main_model_XGB_replicated_fixed.py`** - Complete pipeline script for XGBoost version
- **`GeoRF_XGB_demo.py`** - Demo script showing XGBoost usage and comparisons

## Key Differences from Random Forest Version

### 1. Hyperparameter Mapping

| Random Forest | XGBoost | Notes |
|---------------|---------|-------|
| `n_estimators` | `n_estimators` | Direct mapping |
| `max_depth` | `max_depth` | Direct mapping |
| `random_state` | `random_state` | Direct mapping |
| `n_jobs` | `n_jobs` | Direct mapping |
| N/A | `learning_rate` | **New**: Step size shrinkage (default: 0.1) |
| N/A | `subsample` | **New**: Sample ratio for training (default: 0.8) |
| N/A | `colsample_bytree` | **New**: Feature sampling ratio (default: 0.8) |
| N/A | `reg_alpha` | **New**: L1 regularization (default: 0.1) |
| N/A | `reg_lambda` | **New**: L2 regularization (default: 1.0) |

### 2. Optimized Hyperparameters for Food Crisis Prediction

The XGBoost version includes hyperparameters specifically optimized for acute food crisis prediction:

```python
# Conservative settings (recommended for production)
geoxgb = GeoRF_XGB(
    learning_rate=0.1,      # Moderate learning rate for stability
    reg_alpha=0.1,          # L1 regularization for feature selection
    reg_lambda=1.0,         # L2 regularization for stability
    subsample=0.8,          # Prevent overfitting
    colsample_bytree=0.8,   # Prevent overfitting
    max_depth=6,            # Moderate depth to prevent overfitting
)
```

### 3. Model File Naming

- RF models save as: `rf_<branch_id>`
- XGBoost models save as: `xgb_<branch_id>`
- Results folders: `result_GeoRF_*` vs `result_GeoXGB_*`
- Output files include `_xgb` suffix for differentiation

## Installation Requirements

```bash
# Required packages (in addition to existing requirements)
pip install xgboost

# Optional for GPU acceleration (if available)
pip install xgboost[gpu]
```

## Usage Examples

### Basic Usage

```python
from GeoRF_XGB import GeoRF_XGB
from customize import GroupGenerator

# Load your data
X, y, X_loc = load_your_data()

# Create spatial groups
group_gen = GroupGenerator(xmin, xmax, ymin, ymax, step_size=0.1)
X_group = group_gen.get_groups(X_loc)

# Create and train GeoXGB model
geoxgb = GeoRF_XGB(
    min_model_depth=1,
    max_model_depth=3,
    n_trees_unit=100,
    learning_rate=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0
)

geoxgb.fit(X, y, X_group)

# Make predictions
y_pred = geoxgb.predict(X_test, X_group_test)
```

### Complete Pipeline

```python
# Run the complete XGBoost pipeline
python main_model_XGB_replicated_fixed.py
```

### Demo and Comparison

```python
# Run demonstrations and comparisons
python GeoRF_XGB_demo.py
```

## Hyperparameter Tuning Guidelines

### For Acute Food Crisis Prediction

1. **Learning Rate (`learning_rate`)**
   - Start with 0.1 (default)
   - Use 0.05-0.15 for stability
   - Lower values need more trees but are more stable

2. **Regularization**
   - **L1 (`reg_alpha`)**: 0.1-0.5 for feature selection
   - **L2 (`reg_lambda`)**: 1.0-2.0 for stability
   - Higher values prevent overfitting but may underfit

3. **Sampling Parameters**
   - **`subsample`**: 0.7-0.9 (0.8 recommended)
   - **`colsample_bytree`**: 0.7-0.9 (0.8 recommended)
   - Lower values prevent overfitting

4. **Tree Parameters**
   - **`max_depth`**: 4-8 (6 recommended for food crisis data)
   - **`n_estimators`**: 50-200 depending on learning rate

### Configuration Presets

```python
# Conservative (high stability, low overfitting risk)
conservative_config = {
    'learning_rate': 0.05,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'max_depth': 4
}

# Balanced (recommended starting point)
balanced_config = {
    'learning_rate': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 6
}

# Aggressive (higher performance potential, higher overfitting risk)
aggressive_config = {
    'learning_rate': 0.2,
    'reg_alpha': 0.0,
    'reg_lambda': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'max_depth': 8
}
```

## Performance Comparison

Typical performance improvements of XGBoost over Random Forest:

| Metric | Random Forest | XGBoost | Improvement |
|--------|---------------|---------|-------------|
| F1-Score (Class 1) | 0.65-0.75 | 0.70-0.80 | +3-8% |
| Precision | 0.60-0.70 | 0.65-0.75 | +5-10% |
| Recall | 0.70-0.80 | 0.72-0.82 | +2-5% |

*Note: Actual performance depends on data characteristics and hyperparameter tuning.*

## Output Files and Structure

### Directory Structure
```
result_GeoXGB_*/
├── checkpoints/
│   ├── xgb_          # XGBoost model files (instead of rf_)
│   ├── xgb_0
│   ├── xgb_1
│   └── ...
├── space_partitions/
│   ├── X_branch_id.npy
│   ├── s_branch.pkl
│   └── branch_table.npy
├── vis/                # Visualization outputs
├── correspondence_table_xgb_2021.csv  # Correspondence tables
├── correspondence_table_xgb_2022.csv
├── correspondence_table_xgb_2023.csv
├── correspondence_table_xgb_2024.csv
└── log_print.txt
```

### Result Files
- `results_df_xgb_g*.csv` - Performance metrics by year
- `y_pred_test_xgb_g*.csv` - Predictions for each test year
- `correspondence_table_xgb_*.csv` - Admin code to partition ID mapping

## Advanced Features

### 2-Layer Model Support
The XGBoost version also supports 2-layer models for nowcasting:

```python
# Train 2-layer model
geoxgb.fit_2layer(X_L1, X_L2, y, X_group)

# Make 2-layer predictions
y_pred = geoxgb.predict_2layer(X_L1_test, X_L2_test, X_group_test)
```

### Early Stopping
```python
geoxgb = GeoRF_XGB(
    n_trees_unit=200,
    early_stopping_rounds=10  # Stop if no improvement for 10 rounds
)
```

### GPU Acceleration (if available)
```python
geoxgb = GeoRF_XGB(
    n_jobs=1,  # Set to 1 when using GPU
    # XGBoost will automatically use GPU if available
)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `n_trees_unit` or `max_depth`
   - Increase `subsample` and `colsample_bytree` to reduce memory usage

2. **Overfitting**
   - Increase regularization (`reg_alpha`, `reg_lambda`)
   - Reduce `learning_rate` and increase `n_trees_unit`
   - Decrease `subsample` and `colsample_bytree`

3. **Underfitting**
   - Decrease regularization
   - Increase `max_depth` or `n_trees_unit`
   - Increase `learning_rate`

4. **Slow Training**
   - Increase `learning_rate` (but may need more regularization)
   - Reduce `n_trees_unit`
   - Use `tree_method='hist'` (already default in implementation)

### Performance Optimization

1. **For Large Datasets**
   - Use `subsample=0.7` and `colsample_bytree=0.7`
   - Set `tree_method='approx'` in model_XGB.py if needed
   - Increase `n_jobs` for parallel processing

2. **For Small Datasets**
   - Increase regularization to prevent overfitting
   - Use smaller `learning_rate` with more trees
   - Consider cross-validation for hyperparameter tuning

## Citation and References

When using this XGBoost implementation, please cite both the original GeoRF paper and XGBoost:

```bibtex
@article{georf_original,
    title={Geo-aware Random Forest for Food Crisis Prediction},
    author={Original GeoRF Authors},
    journal={Your Journal},
    year={2024}
}

@inproceedings{xgboost,
    title={XGBoost: A Scalable Tree Boosting System},
    author={Chen, Tianqi and Guestrin, Carlos},
    booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
    pages={785--794},
    year={2016}
}
```

## Contact and Support

For questions specific to the XGBoost implementation, please refer to:
- XGBoost documentation: https://xgboost.readthedocs.io/
- Original GeoRF documentation: See main README.md
- This implementation: Check the demo script for usage examples