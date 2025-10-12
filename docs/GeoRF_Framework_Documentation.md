# GeoRF: Spatial Transformation Framework with Dual Model Support

---
**📋 MIGRATION NOTICE**

This comprehensive documentation has been reorganized for better usability:
- **For Users**: See `CLAUDE.md` in root directory for running instructions
- **For Developers**: See `.ai/` directory for structured development documentation
- **For AI Assistants**: See `.ai/ai-context-structure.md` for complete structure guide

For migration details, see `_migration_index.csv` and `_link_graph.md` in this directory.
---

## Comprehensive Documentation

### Table of Contents
1. [Project Overview](#1-project-overview)
2. [Installation and Setup](#2-installation-and-setup)
3. [Architecture and Components](#3-architecture-and-components)
4. [Core Classes and Methods](#4-core-classes-and-methods)
5. [Data Pipeline](#5-data-pipeline)
6. [Configuration Guide](#6-configuration-guide)
7. [Usage Examples](#7-usage-examples)
8. [Advanced Features](#8-advanced-features)
9. [Output Structure](#9-output-structure)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

**GeoRF** is a Spatial Transformation Framework that implements geo-aware machine learning for acute food crisis prediction. The system handles spatial variability through hierarchical spatial partitioning, making it particularly effective for geographic data where regional variations are important.

The framework provides **two model implementations** that share identical spatial partitioning logic but use different base learners:
- **GeoRF (Random Forest)**: Uses scikit-learn's RandomForestClassifier as the base learner
- **GeoRF_XGB (XGBoost)**: Uses XGBoost as the base learner with identical spatial partitioning

Both implementations provide the same interface and capabilities, with XGBoost offering additional hyperparameter control and potentially improved performance for certain datasets.

### Key Features
- **Unified Spatial Partitioning**: Identical hierarchical binary partitioning logic for both model types
- **Dual Model Support**: 
  - **Random Forest Branch**: Traditional Random Forest with spatial partitioning
  - **XGBoost Branch**: XGBoost with enhanced hyperparameter control and regularization
- **Geo-aware Predictions**: Location-informed model selection for test points
- **Scalable Architecture**: Parallel processing support for large datasets
- **Flexible Grouping**: Grid-based, polygon-based, and K-means clustering spatial strategies
- **Spatial Contiguity**: Both grid-based and polygon-based contiguity refinement
- **Statistical Validation**: Significance testing for optimal partition depth
- **2-Layer Architecture**: Main prediction + error correction layers for nowcasting/forecasting
- **Comprehensive Visualization**: Spatial maps, performance grids, and partition metrics tracking
- **Memory Management**: Batch processing pipeline for large-scale temporal evaluation with automatic cleanup
- **Baseline Comparison Framework**: Built-in comparison with probit regression and FEWSNET official predictions
- **Crisis Prediction Focus**: Specialized metrics focusing on Class 1 (crisis) prediction performance

### Application Domain
Originally developed for food crisis prediction using satellite and ground-based data, but adaptable to any spatially-varying classification or regression task.

### Model Branch Comparison

| Feature | Random Forest Branch | XGBoost Branch |
|---------|---------------------|----------------|
| **Base Learner** | scikit-learn RandomForestClassifier | XGBoost |
| **Interface** | `GeoRF` class | `GeoRF_XGB` class |
| **Core Pipeline** | `main_model_GF.py` | `main_model_XGB.py` |
| **Spatial Logic** | ✅ Identical | ✅ Identical |
| **Hyperparameters** | Standard RF parameters | Enhanced: learning_rate, regularization, sampling |
| **Performance** | Baseline performance | Typically 3-8% higher F1-score |
| **Memory Usage** | Lower | Higher (requires more aggressive cleanup) |
| **Training Speed** | Faster | Slower (more hyperparameters to tune) |
| **Regularization** | Limited (max_depth, min_samples) | Extensive (L1, L2, sampling, early stopping) |
| **Output Directories** | `result_GeoRF_*` | `result_GeoXGB_*` |
| **Batch Processing** | 20 batches (5 time periods × 4 scopes) | 40 batches (10 years × 4 scopes, more granular) |
| **Memory Cleanup** | Moderate (time period-based) | Aggressive (yearly cleanup) |

**Recommendation**: Start with Random Forest for quick prototyping, then use XGBoost for production deployments where performance is critical.

---

## 2. Installation and Setup

### Dependencies
```bash
# Core Dependencies
numpy
scipy  
pandas
polars                    # High-performance data processing
scikit-learn             # Random Forest models
xgboost                  # XGBoost models
matplotlib               # Basic plotting
seaborn                  # Statistical visualization
geopandas                # Geospatial data handling
contextily               # Web map tiles for visualization
```

### File Structure
```
Food_Crisis_Cluster/
├── app/
│   ├── main_model_GF.py              # Food crisis pipeline (Random Forest)
│   ├── main_model_XGB.py             # Food crisis pipeline (XGBoost)
│   └── final/
│       ├── baseline_probit_regression.py # Probit baseline comparison
│       ├── fewsnet_baseline_evaluation.py # FEWSNET baseline evaluation
│       └── georf_vs_baseline_comparison_plot.py # 4-model comparison visualization
├── src/
│   ├── model/
│   │   ├── GeoRF.py                  # Main GeoRF Random Forest class
│   │   ├── GeoRF_XGB.py              # GeoRF XGBoost implementation
│   │   ├── model_RF.py               # Random Forest model wrapper
│   │   └── model_XGB.py              # XGBoost model wrapper
│   ├── partition/
│   │   ├── transformation.py         # Spatial partitioning algorithms
│   │   └── partition_opt.py          # Partition optimization and contiguity
│   ├── customize/
│   │   └── customize.py              # Group generation classes
│   ├── adjacency/
│   │   └── adjacency_utils.py        # Polygon adjacency matrix utilities
│   ├── initialization/
│   │   └── initialization.py         # Data initialization utilities
│   ├── helper/
│   │   └── helper.py                 # Utility functions
│   ├── metrics/
│   │   └── metrics.py                # Evaluation metrics
│   ├── vis/
│   │   └── visualization.py          # Visualization and mapping functions
│   ├── tests/
│   │   └── sig_test.py               # Statistical significance testing
│   └── utils/
│       ├── force_clean.py            # Memory cleanup utilities
│       └── save_results.py           # Result saving utilities
├── demo/
│   ├── GeoRF_demo.py                 # Demonstration script
│   └── data.py                       # Data loading utilities
├── config.py                         # Global configuration parameters
├── run_georf_batches.bat             # GeoRF batch processing script (20 batches)
├── run_xgboost_batches.bat           # XGBoost batch processing script (40 batches)
├── test_georf_batches.bat            # GeoRF testing script (4 batches)
├── test_xgboost_batches.bat          # XGBoost testing script (4 batches)
└── result_GeoRF*/                    # Random Forest output directories
└── result_GeoXGB*/                   # XGBoost output directories
```

---

## 3. Architecture and Components

### Core System Flow
```
Data Input → Group Generation → Spatial Partitioning → Local RF Training → Prediction → Evaluation
```

### Module Responsibilities

#### **GeoRF.py** - Random Forest Branch
- **Primary Class**: `GeoRF`
- **Purpose**: Random Forest implementation with spatial partitioning
- **Base Learner**: scikit-learn RandomForestClassifier
- **Key Features**: 
  - Standard and 2-layer variants
  - Spatial contiguity refinement (grid and polygon)
  - Model persistence and loading
  - Partition metrics tracking

#### **GeoRF_XGB.py** - XGBoost Branch  
- **Primary Class**: `GeoRF_XGB`
- **Purpose**: XGBoost implementation with identical spatial partitioning logic
- **Base Learner**: XGBoost Classifier/Regressor
- **Key Features**:
  - **Identical Interface**: Same `fit()`, `predict()`, `evaluate()` methods as GeoRF
  - **Enhanced Hyperparameters**: learning_rate, regularization, sampling controls
  - **Same Spatial Logic**: Uses identical partitioning algorithms from transformation.py
  - **Additional Controls**: Early stopping, GPU acceleration, advanced regularization

#### **customize.py** - Spatial Grouping
- **Primary Class**: `GroupGenerator`
- **Purpose**: Define minimum spatial units for partitioning
- **Key Features**:
  - Grid-based grouping
  - Administrative boundary grouping
  - K-means clustering groups
  - Missing value imputation

#### **transformation.py** - Spatial Partitioning
- **Primary Function**: `partition()`
- **Purpose**: Hierarchical binary partitioning of spatial groups
- **Algorithm**: Recursive optimization-based splitting

#### **model_RF.py** - Random Forest Implementation
- **Primary Class**: `RFmodel`
- **Purpose**: Wrapper for scikit-learn RandomForestClassifier
- **Features**: Multi-branch model management, spatial predictions

#### **adjacency_utils.py** - Polygon Adjacency Matrix (NEW)
- **Primary Functions**: `create_polygon_adjacency_matrix()`, `load_or_create_adjacency_matrix()`
- **Purpose**: Generate and manage true polygon boundary adjacency matrices
- **Features**:
  - **Shapefile Processing**: Converts shapefiles to adjacency matrices using geopandas
  - **Automatic Caching**: Caches matrices as `polygon_adjacency_cache.pkl` for performance
  - **True Spatial Adjacency**: Uses `touches()` method for actual polygon boundary relationships
  - **Validation Tools**: Compares adjacency vs distance-based neighbor detection
  - **Configuration Integration**: Seamlessly integrates with existing polygon contiguity system

---

## 4. Core Classes and Methods

### 4.1 Model Branch Classes

#### 4.1.1 GeoRF Class (Random Forest Branch)

```python
class GeoRF():
    def __init__(self, min_model_depth=1, max_model_depth=4, n_trees_unit=100, 
                 num_class=2, max_depth=None, random_state=5, n_jobs=32,
                 mode='classification', sample_weights_by_class=None)
```

#### 4.1.2 GeoRF_XGB Class (XGBoost Branch)

```python
class GeoRF_XGB():
    def __init__(self, min_model_depth=1, max_model_depth=4, n_trees_unit=100, 
                 num_class=2, max_depth=None, random_state=5, n_jobs=32,
                 mode='classification', sample_weights_by_class=None,
                 # XGBoost-specific hyperparameters
                 learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0,
                 subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=None)
```

**XGBoost-Specific Parameters:**
- `learning_rate` (float): Step size shrinkage to prevent overfitting (default: 0.1)
- `reg_alpha` (float): L1 regularization term for weights (default: 0.1)  
- `reg_lambda` (float): L2 regularization term for weights (default: 1.0)
- `subsample` (float): Subsample ratio of training instances (default: 0.8)
- `colsample_bytree` (float): Subsample ratio of features when constructing trees (default: 0.8)
- `early_stopping_rounds` (int): Validation metric needs to improve for this many rounds, or training stops

#### Primary Methods

##### **fit(X, y, X_group, X_set=None, val_ratio=0.2, print_to_file=True, contiguity_type='grid', polygon_contiguity_info=None, track_partition_metrics=False, correspondence_table_path=None)**
Train the GeoRF model with spatial partitioning.

**Parameters:**
- `X` (array-like): Feature matrix (n_samples × n_features)
- `y` (array-like): Target labels (n_samples,)
- `X_group` (array-like): Group assignments for each sample (n_samples,)
- `X_set` (array-like, optional): Train/validation split (0=train, 1=validation)
- `val_ratio` (float): Validation ratio if X_set not provided (default: 0.2)
- `print_to_file` (bool): Save training logs to file (default: True)
- `contiguity_type` (str): Type of contiguity refinement: 'grid' or 'polygon' (default: uses CONTIGUITY_TYPE from config)
- `polygon_contiguity_info` (dict, optional): Dictionary containing polygon contiguity information (required when contiguity_type='polygon')
- `track_partition_metrics` (bool): Enable partition performance tracking (default: False)
- `correspondence_table_path` (str, optional): Path to correspondence table for partition visualization

**Returns:**
- `self`: Trained GeoRF instance

**Example:**
```python
georf = GeoRF(max_model_depth=3, n_jobs=16)
georf.fit(X_train, y_train, X_group_train, val_ratio=0.2)

# With polygon contiguity
georf.fit(X_train, y_train, X_group_train, val_ratio=0.2,
          contiguity_type='polygon', polygon_contiguity_info=contiguity_info)
```

##### **predict(X, X_group, save_full_predictions=False)**
Make predictions using trained GeoRF model.

**Parameters:**
- `X` (array-like): Test features (n_test_samples × n_features)
- `X_group` (array-like): Group assignments for test samples
- `save_full_predictions` (bool): Save predictions to file

**Returns:**
- `y_pred` (array-like): Predicted labels (n_test_samples,)

##### **evaluate(Xtest, ytest, Xtest_group, eval_base=False, print_to_file=True)**
Evaluate GeoRF performance with metrics.

**Parameters:**
- `Xtest, ytest` (array-like): Test features and labels
- `Xtest_group` (array-like): Test group assignments
- `eval_base` (bool): Also evaluate base RF for comparison
- `print_to_file` (bool): Save evaluation logs

**Returns:**
- `pre, rec, f1` (array-like): Precision, recall, F1 scores per class
- `pre_base, rec_base, f1_base` (array-like): Base RF metrics (if eval_base=True)

##### **visualize_grid(Xtest, ytest, Xtest_group, step_size=0.1)**
Generate spatial visualizations of partitions and performance (for grid-based groups).

**Parameters:**
- `Xtest, ytest, Xtest_group`: Test data
- `step_size` (float): Grid cell size for visualization

### 4.2 2-Layer GeoRF Methods

##### **fit_2layer(X_L1, X_L2, y, X_group, val_ratio=0.2, contiguity_type='grid', polygon_contiguity_info=None)**
Train 2-layer GeoRF: main prediction + error correction.

**Parameters:**
- `X_L1` (array-like): Layer 1 features (main prediction)
- `X_L2` (array-like): Layer 2 features (error correction)
- `y, X_group`: Labels and group assignments
- `val_ratio`: Validation split ratio
- `contiguity_type` (str): Type of contiguity refinement: 'grid' or 'polygon' (default: uses CONTIGUITY_TYPE from config)
- `polygon_contiguity_info` (dict, optional): Dictionary containing polygon contiguity information (required when contiguity_type='polygon')

##### **predict_2layer(X_L1, X_L2, X_group, correction_strategy='flip')**
Make predictions using 2-layer model.

**Parameters:**
- `X_L1, X_L2`: Layer 1 and 2 test features
- `X_group`: Test group assignments
- `correction_strategy`: Error correction method ('flip' for binary)

##### **evaluate_2layer(X_L1_test, X_L2_test, y_test, X_group_test, X_L1_train=None, X_L2_train=None, y_train=None, X_group_train=None, correction_strategy='flip', print_to_file=True, contiguity_type='grid', polygon_contiguity_info=None)**
Evaluate 2-layer GeoRF against 2-layer base RF.

**Parameters:**
- `X_L1_test, X_L2_test`: Test features for both layers
- `y_test, X_group_test`: Test labels and group assignments
- `X_L1_train, X_L2_train, y_train, X_group_train` (optional): Training data for base model comparison
- `correction_strategy` (str): Error correction method ('flip' for binary)
- `print_to_file` (bool): Save evaluation logs to file
- `contiguity_type` (str): Type of contiguity refinement (for consistency)
- `polygon_contiguity_info` (dict, optional): Polygon contiguity information (for consistency)

**Returns:**
- `pre, rec, f1`: Precision, recall, F1 scores for 2-layer GeoRF
- `pre_base, rec_base, f1_base`: Precision, recall, F1 scores for 2-layer base RF

### 4.3 GroupGenerator Class

```python
class GroupGenerator():
    def __init__(self, xmin, xmax, ymin, ymax, step_size)
```

**Purpose**: Generate spatial groups (minimum spatial units) for partitioning.

##### **get_groups(X_loc)**
Generate group assignments from location coordinates.

**Parameters:**
- `X_loc` (array-like): Geographic coordinates (n_samples × 2) [lat, lon]

**Returns:**
- `X_group` (array-like): Group ID for each sample (n_samples,)

**Example:**
```python
# Create grid-based groups
group_gen = GroupGenerator(xmin=-10, xmax=50, ymin=-20, ymax=70, step_size=0.1)
X_group = group_gen.get_groups(X_loc)
```

### 4.4 Advanced Grouping Functions

##### **generate_kmeans_groups_from_admin_codes(df, features_for_clustering=None, n_clusters=50)**
Create K-means based groups from administrative codes.

**Parameters:**
- `df` (DataFrame): Data with 'FEWSNET_admin_code' column
- `features_for_clustering` (list): Features for clustering (default: ['lat', 'lon'])
- `n_clusters` (int): Number of clusters

**Returns:**
- `admin_to_group_map` (dict): Mapping from admin codes to group IDs
- `cluster_info` (dict): Clustering statistics and information

### 4.5 Polygon-Based Grouping

##### **PolygonGroupGenerator Class**
Generate groups for polygon-based spatial partitioning with contiguity support.

```python
class PolygonGroupGenerator():
    def __init__(self, polygon_centroids, polygon_group_mapping=None, 
                 neighbor_distance_threshold=None)
```

**Parameters:**
- `polygon_centroids` (array-like): Centroid coordinates (lat, lon) for each polygon
- `polygon_group_mapping` (dict, optional): Mapping from polygon indices to group IDs
- `neighbor_distance_threshold` (float, optional): Distance threshold for neighbors

**Methods:**
- `get_groups(X_polygon_ids)`: Generate group assignments from polygon IDs
- `get_contiguity_info()`: Get information needed for contiguity refinement

**Example:**
```python
# Create polygon group generator
polygon_gen = PolygonGroupGenerator(
    polygon_centroids=polygon_centroids,  # Shape: (n_polygons, 2)
    neighbor_distance_threshold=0.8
)

# Generate groups
X_group = polygon_gen.get_groups(X_polygon_ids)

# Get contiguity info for GeoRF
contiguity_info = polygon_gen.get_contiguity_info()
```

##### **generate_polygon_groups_from_centroids(X_polygon_ids, polygon_centroids, ...)**
Convenience function for polygon-based grouping.

**Parameters:**
- `X_polygon_ids` (array-like): Polygon IDs for each data point
- `polygon_centroids` (array-like): Centroid coordinates for each polygon
- `polygon_group_mapping` (dict, optional): Mapping from polygon indices to group IDs
- `neighbor_distance_threshold` (float, optional): Distance threshold for neighbors

**Returns:**
- `X_group` (array-like): Group assignments
- `polygon_generator` (PolygonGroupGenerator): Generator instance

### 4.6 Data Imputation

##### **impute_missing_values(X, strategy='max_plus', multiplier=100.0)**
Handle missing values using out-of-range imputation for decision trees.

**Parameters:**
- `X` (array-like): Data with missing values
- `strategy` (str): Imputation strategy ('max_plus', 'min_minus', 'extreme_high', 'extreme_low')
- `multiplier` (float): Multiplier for out-of-range values

### 4.7 Adjacency Matrix Utilities (NEW)

#### **Primary Functions**

##### **create_polygon_adjacency_matrix(shapefile_path, polygon_id_column=None)**
Generate adjacency matrix from shapefile using true polygon boundary relationships.

**Parameters:**
- `shapefile_path` (str): Path to shapefile containing polygon geometries
- `polygon_id_column` (str, optional): Column containing polygon IDs (uses row index if None)

**Returns:**
- `adjacency_dict` (dict): Dictionary mapping polygon IDs to lists of neighbor IDs
- `polygon_id_mapping` (dict): Mapping from internal indices to original polygon IDs
- `polygon_centroids` (array-like): Centroid coordinates for each polygon

##### **load_or_create_adjacency_matrix(shapefile_path=None, polygon_id_column=None, cache_dir=None, force_regenerate=False)**
Load cached adjacency matrix or create new one. Uses configuration parameters if arguments not provided.

**Parameters:**
- `shapefile_path` (str, optional): Path to shapefile (uses config.ADJACENCY_SHAPEFILE_PATH if None)
- `polygon_id_column` (str, optional): ID column (uses config.ADJACENCY_POLYGON_ID_COLUMN if None)
- `cache_dir` (str, optional): Cache directory (uses config.ADJACENCY_CACHE_DIR if None)
- `force_regenerate` (bool): Force regeneration even if cache exists

**Returns:**
- Same as `create_polygon_adjacency_matrix()`

##### **adjacency_dict_to_neighbors_dict(adjacency_dict)**
Convert adjacency dictionary format for compatibility with existing neighbor functions.

##### **validate_adjacency_matrix(adjacency_dict, polygon_centroids, distance_threshold=0.8)**
Validate adjacency matrix and compare with distance-based neighbor detection.

**Performance Benchmarks (FEWS Admin Boundaries):**
- **Distance-Based Neighbors**: 163,262 total connections, 28.55 average neighbors per polygon
- **Adjacency Matrix**: 23,224 total connections, 4.06 average neighbors per polygon  
- **Performance Improvement**: 86% reduction in neighbor relationships, faster contiguity refinement
- **Accuracy**: True polygon boundary relationships instead of approximate distance calculations

#### **Pipeline Validation Functions (Updated)**

##### **validate_polygon_contiguity(contiguity_info, X_group)** (in main_model_GF.py and main_model_XGB.py)
Updated validation function that automatically detects and validates the active contiguity approach.

**Smart Detection Logic:**
1. **Checks for adjacency_dict in contiguity_info** - if present, validates adjacency matrix approach
2. **Falls back to distance-based validation** - if adjacency matrix unavailable
3. **Provides context-appropriate messaging** - production vs fallback approach indicators
4. **Reports relevant statistics** - neighbor counts specific to the active method

**Expected Output (Production):**
```
=== Polygon Contiguity Validation ===
Using adjacency matrix-based neighbors (production approach)
Adjacency neighbor stats: min=0, max=16, mean=4.1
Note: 655 isolated polygons (normal for islands/enclaves)
Total adjacency connections: 23224
Centroid range: lat -34.90-37.55, lon -18.00-51.26
Group size stats: min=1, max=2847, mean=156.2
=== End Polygon Validation ===
```

**Legacy Cleaned:** No longer tests distance-based approach when adjacency matrix is active.

---

## 5. Data Pipeline

### 5.1 Data Requirements

GeoRF requires **4 input components** (unlike standard RF which needs only X, y):

1. **X**: Feature matrix (n_samples × n_features)
2. **y**: Target labels (n_samples,)
3. **X_loc**: Location coordinates (n_samples × 2) [latitude, longitude]
4. **X_group**: Group assignments (n_samples,) - generated from X_loc

### 5.2 Complete Workflow

```python
# 1. Data Loading
X, y, X_loc = load_demo_data()  # or your custom data loader

# 2. Spatial Range Calculation
xmin, xmax, ymin, ymax = get_spatial_range(X_loc)

# 3. Group Generation
group_gen = GroupGenerator(xmin, xmax, ymin, ymax, step_size=0.1)
X_group = group_gen.get_groups(X_loc)

# 4. Train-Test Split
(X_train, y_train, X_train_loc, X_train_group,
 X_test, y_test, X_test_loc, X_test_group) = train_test_split_all(
    X, y, X_loc, X_group, test_ratio=0.4)

# 5. Model Training
georf = GeoRF(min_model_depth=1, max_model_depth=4, n_jobs=32)
georf.fit(X_train, y_train, X_train_group, val_ratio=0.2)

# 6. Prediction
y_pred = georf.predict(X_test, X_test_group)

# 7. Evaluation
pre, rec, f1, pre_base, rec_base, f1_base = georf.evaluate(
    X_test, y_test, X_test_group, eval_base=True)
```

### 5.3 Data Loading Functions

##### **load_demo_data()**
Load demonstration dataset (crop classification data).

**Returns:**
- `X, y, X_loc`: Features, labels, and coordinates

##### **train_test_split_all(X, y, X_loc, X_group, test_ratio=0.4)**
Split all data components maintaining spatial consistency.

**Returns:**
- `(X_train, y_train, X_train_loc, X_train_group, X_test, y_test, X_test_loc, X_test_group)`

##### **train_test_split_rolling_window(X, y, X_loc, X_group, years, dates, test_year, input_terms, need_terms)**
Rolling window temporal splitting for quarterly evaluation with 5-year training windows.

---

## 6. Configuration Guide

### 6.1 Complete Evaluation Pipeline

The framework now includes a comprehensive 4-model comparison system for acute food crisis prediction:

#### **4-Model Evaluation Framework:**

1. **Probit Baseline**: Simple regression with lagged crisis variables (supports all 4 forecasting scopes)
2. **FEWSNET Baseline**: Official predictions from FEWS NET system (supports scopes 1-2 only)  
3. **GeoRF**: Geo-aware Random Forest with spatial partitioning (supports all 4 scopes)
4. **XGBoost**: XGBoost with same spatial partitioning framework (supports all 4 scopes)

#### **Forecasting Scopes:**
- **Scope 1**: 3-month lag forecasting (lag terms 1,2,3)
- **Scope 2**: 6-month lag forecasting (lag terms 2,3,4)
- **Scope 3**: 9-month lag forecasting (lag terms 3,4,5)  
- **Scope 4**: 12-month lag forecasting (lag terms 4,5,6)

#### **Crisis Prediction Focus (Class 1 Metrics Only):**
All models focus exclusively on crisis prediction (Class 1) performance:
- **Precision (Class 1)**: True crisis predictions / All crisis predictions
- **Recall (Class 1)**: True crisis predictions / All actual crises
- **F1 Score (Class 1)**: Harmonic mean of precision and recall

#### **Production Pipeline Configuration:**
```python
# Spatial assignment and contiguity (production defaults)
assignment = 'polygons'              # Uses FEWSNET admin boundaries  
CONTIGUITY_TYPE = 'polygon'          # Enables polygon-based contiguity
USE_ADJACENCY_MATRIX = True          # Uses true polygon adjacency (recommended)

# This combination provides:
# - True polygon boundary relationships (not distance approximations)
# - 86% reduction in neighbor connections (23K vs 163K)
# - Faster contiguity refinement and more accurate spatial relationships
```

**Assignment-Specific Contiguity Integration:**
- **'polygons'**: Uses adjacency matrix (if enabled) or distance fallback for FEWSNET admin boundaries
- **'country', 'AEZ', 'country_AEZ'**: Uses distance-based neighbors (no shapefiles available)
- **'geokmeans', 'all_kmeans'**: Uses distance-based neighbors for cluster centroids
- **'grid'**: Uses 8-neighbor grid contiguity (independent of adjacency matrix)

### 6.2 Running the Models

#### **Basic Random Forest GeoRF**
```bash
python demo/GeoRF_demo.py
```

#### **Food Crisis Pipeline - Single Execution**

**Random Forest (GeoRF):**
```bash
python app/main_model_GF.py --start_year 2023 --end_year 2024 --forecasting_scope 1
```

**XGBoost (GeoXGB):**
```bash
python app/main_model_XGB.py --start_year 2023 --end_year 2024 --forecasting_scope 1
```

**Command Line Arguments:**
- `--start_year`: Start year for evaluation (default: 2024)
- `--end_year`: End year for evaluation (default: 2024) 
- `--forecasting_scope`: Forecasting scope 1-4 (1=3mo, 2=6mo, 3=9mo, 4=12mo lag)

#### **RECOMMENDED: Batch Processing for Memory Management**

**GeoRF Batch Processing:**
```bash
# Full production run (5 time periods × 4 forecasting scopes = 20 batches)
run_georf_batches.bat

# Light testing (2 time periods × 2 forecasting scopes = 4 batches)
test_georf_batches.bat
```

**XGBoost Batch Processing:**
```bash
# Full production run (10 years × 4 forecasting scopes = 40 batches)
run_xgboost_batches.bat

# Light testing (2 years × 2 forecasting scopes = 4 batches)
test_xgboost_batches.bat
```

**Batch Processing Architecture:**

**GeoRF Batches (`run_georf_batches.bat`):**
- Processes 5 time periods: 2015-2016, 2017-2018, 2019-2020, 2021-2022, 2023-2024
- Each time period runs 4 forecasting scopes (1=3mo, 2=6mo, 3=9mo, 4=12mo lag)
- Total: 20 batches with memory optimization between each batch
- Uses robust directory cleanup with retry logic for locked files
- Includes pre-execution cleanup and post-execution garbage collection

**XGBoost Batches (`run_xgboost_batches.bat`):**
- Processes individual years: 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
- Each year runs 4 forecasting scopes (1=3mo, 2=6mo, 3=9mo, 4=12mo lag)
- Total: 40 batches with more granular memory management
- More aggressive memory cleanup due to XGBoost's higher memory requirements
- Single year processing to minimize memory footprint

**Batch Processing Benefits:**
- **Memory Cleanup**: Prevents memory leakage during long temporal evaluations
- **Error Handling**: Continues processing if individual batches fail
- **Unique Naming**: Generates files with `_{start_year}_{end_year}` suffixes to prevent overwrites
- **Progress Tracking**: Shows batch completion status and memory cleanup
- **Robust Directory Cleanup**: Multiple attempts with retry logic for locked directories
- **Cleanup Utilities**: Run `python -c "from src.utils.force_clean import force_cleanup_directories; force_cleanup_directories()"` manually when you need to purge old outputs
- **Temporary File Removal**: Cleans up temp files, pickle files, and __pycache__ directories
- **Windows Memory Release**: Forces OS-level memory cleanup with timeout between batches

#### **Baseline Comparisons and Model Evaluation**

**Probit Regression Baseline:**
```bash
python app/final/baseline_probit_regression.py
```

**FEWSNET Official Predictions Baseline:**
```bash
python app/final/fewsnet_baseline_evaluation.py
```

**4-Model Performance Comparison:**
```bash
python app/final/georf_vs_baseline_comparison_plot.py
```

#### **Complete Evaluation Pipeline**

**Full Pipeline Execution Order:**
1. **Run Baseline Models:**
   ```bash
   # Generate probit regression baseline results
   python app/final/baseline_probit_regression.py
   
   # Generate FEWSNET official predictions baseline results  
   python app/final/fewsnet_baseline_evaluation.py
   ```

2. **Run Main Models with Batch Processing:**
   ```bash
   # Generate GeoRF results (20 batches: 5 time periods × 4 scopes)
   run_georf_batches.bat
   
   # Generate XGBoost results (40 batches: 10 years × 4 scopes)
   run_xgboost_batches.bat
   ```

3. **Generate Comparison Analysis:**
   ```bash
   # Auto-detect all results and create 4-model comparison visualization
   python app/final/georf_vs_baseline_comparison_plot.py
   ```

**Pipeline Output Files:**

**Baseline Results:**
- `baseline_probit_results/baseline_probit_results_fs{1-4}.csv`
- `fewsnet_baseline_results/fewsnet_baseline_results_fs{1-2}.csv` (FEWSNET supports scopes 1-2 only)

**GeoRF Results (20 batch files from time periods):**
- `results_df_gf_fs{scope}_{start_year}_{end_year}.csv` - Main results with metrics
- `y_pred_test_gf_fs{scope}_{start_year}_{end_year}.csv` - Prediction arrays
- `result_GeoRF_{id}/` directories (cleaned up between batches)

**XGBoost Results (40 batch files from individual years):**  
- `results_df_xgb_fs{scope}_{start_year}_{end_year}.csv` - Main results with metrics
- `y_pred_test_xgb_fs{scope}_{start_year}_{end_year}.csv` - Prediction arrays
- `result_GeoXGB_{id}/` directories (cleaned up between batches)

**File Naming Convention:**
- `{model_type}`: `gf` (GeoRF), `xgb` (XGBoost), `baseline_probit`, `fewsnet_baseline`
- `fs{scope}`: Forecasting scope (fs1=3mo, fs2=6mo, fs3=9mo, fs4=12mo)
- `{start_year}_{end_year}`: Temporal range processed in the batch

**Comparison Output:**
- `model_comparison_class1_focus.png` - Dynamic 4-model performance visualization  
- Console output with detailed summary statistics for all models and forecasting scopes

**FEWSNET Baseline Details:**
- Uses `pred_near_lag1` for scope 1 (3-month forecasting)
- Uses `pred_med_lag2` for scope 2 (6-month forecasting)
- Converts monthly FEWSNET data to quarterly format for consistency
- Applies temporal lags by admin unit for proper forecasting evaluation
- Only supports scopes 1-2 as FEWSNET doesn't provide longer-term predictions

**Comparison Visualization Features:**
- Auto-detects available result files using glob patterns
- Combines multiple batch result files for GeoRF and XGBoost
- Creates dynamic subplot grid based on available forecasting scopes
- Shows precision, recall, and F1 score time series for each model
- Handles missing data gracefully with informative subplot messages
- Provides summary statistics with unweighted averages across time periods

**Crisis Prediction Focus:**
All models in the framework now focus exclusively on **Class 1 (crisis prediction)** metrics:

- **Precision (Class 1)**: True crisis predictions / All crisis predictions made by the model
- **Recall (Class 1)**: True crisis predictions / All actual crises in the data  
- **F1 Score (Class 1)**: Harmonic mean of precision and recall, balancing both metrics

**Why Class 1 Focus:**
- Crisis prediction is the primary objective (more important than predicting non-crisis)
- Class imbalance makes overall accuracy misleading
- Policy relevance: False negatives (missed crises) and false positives (unnecessary alerts) have different costs
- Alignment with operational food security early warning systems

**Model Performance Expectations:**
- **Random Forest GeoRF**: F1 scores typically 0.65-0.75 for crisis prediction
- **XGBoost GeoRF**: F1 scores typically 0.70-0.80 for crisis prediction (3-8% improvement)
- **Baseline Models**: Probit ~0.55-0.65, FEWSNET ~0.60-0.70 F1 scores
- **Performance varies by forecasting scope**: Shorter lags (3-month) typically perform better than longer lags (12-month)

*Note: The XGBoost version (`app/main_model_XGB.py`) provides complete feature parity with the Random Forest pipeline including checkpoint recovery, rolling window evaluation, and partition metrics tracking.*

### 6.2 Core Configuration Parameters (config.py)

#### **Model Structure Parameters**
```python
MIN_DEPTH = 1                    # Minimum partitioning depth
MAX_DEPTH = 4                    # Maximum partitioning depth  
N_JOBS = 32                      # Parallel processing cores
MODEL_CHOICE = 'RF'              # Model type ('RF' for Random Forest, 'XGB' for XGBoost)
MODE = 'classification'          # Task type ('classification' or 'regression')
NUM_CLASS = 2                    # Number of classes (binary classification)
```

#### **Training Parameters**
```python
TRAIN_RATIO = 0.6               # Training set ratio
VAL_RATIO = 0.20                # Validation ratio (from training set)
TEST_RATIO = 0.4                # Test set ratio
```

#### **Partitioning Parameters**
```python
MIN_BRANCH_SAMPLE_SIZE = 5      # Minimum samples to continue partitioning
MIN_SCAN_CLASS_SAMPLE = 10      # Minimum samples per class for optimization
FLEX_RATIO = 0.025              # Max partition size difference ratio
SIGLVL = 0.05                   # Significance level for statistical testing
ES_THRD = 0.8                   # Effect size threshold
MD_THRD = 0.001                 # Mean difference threshold
```

#### **Spatial Parameters**
```python
STEP_SIZE = 0.1                 # Grid cell size (degrees)
CONTIGUITY = True               # Enable spatial contiguity refinement
CONTIGUITY_TYPE = 'grid'        # Type of contiguity: 'grid' or 'polygon'
REFINE_TIMES = 3                # Number of contiguity refinement iterations
MIN_COMPONENT_SIZE = 5          # Minimum component size for refinement

# Polygon contiguity parameters (when CONTIGUITY_TYPE = 'polygon')
POLYGON_NEIGHBOR_DISTANCE_THRESHOLD = None  # Auto-calculate if None
POLYGON_CONTIGUITY_INFO = None  # Set to contiguity info dict when using polygon contiguity

# Adjacency matrix parameters (advanced polygon contiguity)
USE_ADJACENCY_MATRIX = True     # Use true polygon adjacency instead of distance-based
ADJACENCY_SHAPEFILE_PATH = r'...\FEWS_Admin_LZ_v3.shp'  # Path to shapefile
ADJACENCY_POLYGON_ID_COLUMN = 'FEWSNET_ID'  # ID column in shapefile
ADJACENCY_CACHE_DIR = None      # Cache directory (None = current directory)
ADJACENCY_FORCE_REGENERATE = False  # Force regeneration even if cache exists
```

### 6.2 Parameter Tuning Guidelines

#### **Partitioning Depth (MIN_DEPTH, MAX_DEPTH)**
- **MIN_DEPTH=1, MAX_DEPTH=2**: Conservative, fewer partitions, faster training
- **MIN_DEPTH=1, MAX_DEPTH=4**: Standard setting, good balance
- **MIN_DEPTH=2, MAX_DEPTH=5**: Aggressive partitioning, more local models

#### **Grid Size (STEP_SIZE)**
- **Large (1.0°)**: Fewer, larger groups; less spatial detail
- **Medium (0.1°)**: Standard setting, ~11km resolution
- **Small (0.01°)**: Fine detail, more groups, slower processing

#### **Statistical Testing**
- **SIGLVL=0.05**: Standard significance level
- **ES_THRD=0.8**: Effect size threshold (Cohen's d)
- **Higher values**: More conservative partitioning

---

## 7. Usage Examples

### 7.1 Basic Usage

#### 7.1.1 Random Forest Branch

```python
from GeoRF import GeoRF
from customize import GroupGenerator
from data import load_demo_data
from helper import get_spatial_range
from initialization import train_test_split_all

# Load data
X, y, X_loc = load_demo_data()

# Generate groups
xmin, xmax, ymin, ymax = get_spatial_range(X_loc)
group_gen = GroupGenerator(xmin, xmax, ymin, ymax, step_size=0.1)
X_group = group_gen.get_groups(X_loc)

# Split data
(X_train, y_train, X_train_loc, X_train_group,
 X_test, y_test, X_test_loc, X_test_group) = train_test_split_all(
    X, y, X_loc, X_group, test_ratio=0.4)

# Train and evaluate Random Forest branch
georf = GeoRF(max_model_depth=3)
georf.fit(X_train, y_train, X_train_group)
y_pred = georf.predict(X_test, X_test_group)
pre, rec, f1 = georf.evaluate(X_test, y_test, X_test_group)
```

#### 7.1.2 XGBoost Branch

```python
from GeoRF_XGB import GeoRF_XGB
# ... same data loading and preprocessing ...

# Train and evaluate XGBoost branch (identical interface)
geoxgb = GeoRF_XGB(
    max_model_depth=3,
    learning_rate=0.1,      # XGBoost-specific
    reg_alpha=0.1,          # L1 regularization  
    reg_lambda=1.0,         # L2 regularization
    subsample=0.8,          # Training instance sampling
    colsample_bytree=0.8    # Feature sampling
)

geoxgb.fit(X_train, y_train, X_train_group)  # Identical API
y_pred_xgb = geoxgb.predict(X_test, X_test_group)  # Identical API
pre_xgb, rec_xgb, f1_xgb = geoxgb.evaluate(X_test, y_test, X_test_group)  # Identical API
```

### 7.2 Advanced Usage with Custom Grouping

```python
# Using administrative boundaries (K-means clustering)
from customize import create_kmeans_groupgenerator_from_admin_codes

# Assuming df has 'FEWSNET_admin_code', 'lat', 'lon' columns
X_group, cluster_info, admin_map = create_kmeans_groupgenerator_from_admin_codes(
    df, features_for_clustering=['lat', 'lon'], n_clusters=50)

# Train with custom groups
georf = GeoRF(max_model_depth=4, n_jobs=16)
georf.fit(X, y, X_group)
```

### 7.3 Polygon-Based Grouping with Contiguity

```python
# Using polygon-based grouping with contiguity refinement
from customize import PolygonGroupGenerator

# Example: Administrative boundaries with centroids
polygon_centroids = np.array([
    [40.0, -100.0],  # Polygon 0 centroid (lat, lon)
    [40.0, -99.5],   # Polygon 1 centroid
    [40.5, -100.0],  # Polygon 2 centroid
    # ... more centroids
])

# Create polygon group generator
polygon_gen = PolygonGroupGenerator(
    polygon_centroids=polygon_centroids,
    neighbor_distance_threshold=0.8  # 0.8 degrees for neighbor detection
)

# Generate groups from polygon IDs
X_polygon_ids = np.array([0, 0, 1, 1, 2, 2, ...])  # Polygon ID for each data point
X_group = polygon_gen.get_groups(X_polygon_ids)

# Get contiguity info
contiguity_info = polygon_gen.get_contiguity_info()

# Train with polygon contiguity
georf = GeoRF(max_model_depth=3, n_jobs=16)
georf.fit(X_train, y_train, X_group_train,
          contiguity_type='polygon',
          polygon_contiguity_info=contiguity_info)

# Predict
y_pred = georf.predict(X_test, X_group_test)
```

### 7.4 2-Layer GeoRF Example

```python
# Prepare separate feature sets for each layer
X_L1 = X[:, :10]  # First 10 features for main prediction
X_L2 = X[:, 10:]  # Remaining features for error correction

# Train 2-layer model
georf = GeoRF(max_model_depth=3)
georf.fit_2layer(X_L1_train, X_L2_train, y_train, X_group_train)

# Train 2-layer model with polygon contiguity
georf.fit_2layer(X_L1_train, X_L2_train, y_train, X_group_train,
                 contiguity_type='polygon', polygon_contiguity_info=contiguity_info)

# Predict with both layers
y_pred = georf.predict_2layer(X_L1_test, X_L2_test, X_group_test)

# Evaluate against 2-layer base RF
pre, rec, f1, pre_base, rec_base, f1_base = georf.evaluate_2layer(
    X_L1_test, X_L2_test, y_test, X_group_test,
    X_L1_train, X_L2_train, y_train, X_group_train)
```

### 7.7 XGBoost Hyperparameter Tuning

#### 7.7.1 Configuration Presets for Food Crisis Prediction

```python
from GeoRF_XGB import GeoRF_XGB

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

# Use a configuration preset
geoxgb = GeoRF_XGB(max_model_depth=3, **balanced_config)
```

#### 7.7.2 Hyperparameter Tuning Guidelines

**For Acute Food Crisis Prediction:**

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
   - **`n_trees_unit`**: 50-200 depending on learning rate

#### 7.7.3 Typical Performance Improvements

| Metric | Random Forest Branch | XGBoost Branch | Improvement |
|--------|---------------------|----------------|-------------|
| F1-Score (Class 1) | 0.65-0.75 | 0.70-0.80 | +3-8% |
| Precision | 0.60-0.70 | 0.65-0.75 | +5-10% |
| Recall | 0.70-0.80 | 0.72-0.82 | +2-5% |

*Note: Actual performance depends on data characteristics and hyperparameter tuning.*

### 7.8 Missing Value Handling

```python
from customize import impute_missing_values

# Impute missing values using out-of-range strategy
X_imputed, imputer = impute_missing_values(
    X, strategy='max_plus', multiplier=100.0, verbose=True)

# Use imputed data for training
georf.fit(X_imputed_train, y_train, X_group_train)
```

### 7.9 Time-Based Splitting

```python
from customize import train_test_split_rolling_window

# Rolling window split for quarterly temporal validation
(X_train, y_train, X_train_loc, X_train_group,
 X_test, y_test, X_test_loc, X_test_group) = train_test_split_rolling_window(
    X, y, X_loc, X_group, years=year_array, test_year=2020)
```

---

## 8. Advanced Features

### 8.1 Spatial Contiguity Refinement

GeoRF can improve spatial contiguity of partitions using majority voting in local neighborhoods. Three types of contiguity are supported:

#### 8.1.1 Grid-Based Contiguity (Default)

For grid-based groups, uses 8-neighbor majority voting:

```python
# Enable in config.py
CONTIGUITY = True
CONTIGUITY_TYPE = 'grid'  # Default
REFINE_TIMES = 3
MIN_COMPONENT_SIZE = 5

# Automatic during training
georf.fit(X_train, y_train, X_group_train)  # Refinement applied automatically
```

#### 8.1.2 Polygon-Based Contiguity (Distance-Based)

For polygon-based groups (administrative boundaries, watersheds, etc.), uses centroid-based neighbor detection:

```python
# Enable polygon contiguity in config.py
CONTIGUITY = True
CONTIGUITY_TYPE = 'polygon'
USE_ADJACENCY_MATRIX = False    # Use distance-based neighbors
POLYGON_NEIGHBOR_DISTANCE_THRESHOLD = 0.8  # or None for auto-calculation
REFINE_TIMES = 3

# Create polygon group generator
from customize import PolygonGroupGenerator
polygon_gen = PolygonGroupGenerator(
    polygon_centroids=polygon_centroids,  # Shape: (n_polygons, 2)
    neighbor_distance_threshold=0.8
)

# Generate groups and get contiguity info
X_group = polygon_gen.get_groups(X_polygon_ids)
contiguity_info = polygon_gen.get_contiguity_info()

# Train with polygon contiguity
georf.fit(X_train, y_train, X_group_train,
          contiguity_type='polygon',
          polygon_contiguity_info=contiguity_info)
```

#### 8.1.3 Adjacency Matrix-Based Contiguity (Recommended for Production)

**NEW**: True polygon boundary adjacency using shapefile topology:

```python
# Enable adjacency matrix in config.py
CONTIGUITY = True
CONTIGUITY_TYPE = 'polygon'
USE_ADJACENCY_MATRIX = True     # Use true polygon adjacency
ADJACENCY_SHAPEFILE_PATH = r'path/to/admin_boundaries.shp'
ADJACENCY_POLYGON_ID_COLUMN = 'ADMIN_ID'  # ID column in shapefile
ADJACENCY_CACHE_DIR = None      # Cache in current directory
REFINE_TIMES = 3

# Automatic adjacency matrix loading
from adjacency_utils import load_or_create_adjacency_matrix
from customize import PolygonGroupGenerator

# Load or create adjacency matrix (cached automatically)
adjacency_dict, polygon_id_mapping, polygon_centroids = load_or_create_adjacency_matrix()

# Create polygon group generator with true adjacency
polygon_gen = PolygonGroupGenerator(
    polygon_centroids=polygon_centroids,
    adjacency_dict=adjacency_dict  # Use true polygon adjacency
)

# Generate groups and get contiguity info with adjacency matrix
X_group = polygon_gen.get_groups(X_polygon_ids)
contiguity_info = polygon_gen.get_contiguity_info()

# Train with true polygon adjacency
georf.fit(X_train, y_train, X_group_train,
          contiguity_type='polygon',
          polygon_contiguity_info=contiguity_info)
```

**Key Features of Each Contiguity Type:**

| Feature | Grid-Based | Distance-Based Polygon | Adjacency Matrix Polygon |
|---------|------------|----------------------|-------------------------|
| **Accuracy** | High for regular grids | Approximate | True polygon boundaries |
| **Performance** | Fast | Fast | Fastest (cached) |
| **Neighbor Count** | Fixed (8) | Variable (distance-based) | Variable (true adjacency) |
| **Setup Complexity** | Simple | Moderate | Complex (requires shapefile) |
| **Use Case** | Regular spatial grids | Approximate polygon analysis | Production polygon systems |

**Performance Comparison (FEWS Admin Boundaries):**
- **Distance-Based**: 163,262 connections, 28.55 avg neighbors
- **Adjacency Matrix**: 23,224 connections, 4.06 avg neighbors (86% reduction)
- **Benefits**: More accurate contiguity, faster refinement, no distance threshold tuning

**Adjacency Matrix Features:**
- **Automatic Caching**: Adjacency matrices cached as `polygon_adjacency_cache.pkl`
- **Backward Compatibility**: Falls back to distance-based if adjacency unavailable
- **True Spatial Relationships**: Uses geopandas `touches()` with boundary validation
- **Symmetric Adjacency**: Guaranteed symmetric neighbor relationships
- **Configuration Driven**: Easy switching between approaches via config flags
- **Smart Assignment Integration**: Only applies to 'polygons' assignment (FEWSNET admin boundaries)
- **Updated Validation**: Validation functions automatically detect and test the active approach

### 8.2 Partition Visualization

```python
# Generate spatial visualizations (grid-based groups only)
georf.visualize_grid(X_test, y_test, X_test_group, step_size=0.1)

# Visualizations saved to: result_GeoRF_*/vis/
# - Partition maps
# - Performance grids per class  
# - Difference maps (GeoRF vs Base RF)
# - Sample count maps
```

### 8.3 Model Persistence

```python
# Models automatically saved during training to:
# result_GeoRF_*/checkpoints/rf_* (individual RF models)
# result_GeoRF_*/space_partitions/ (partition definitions)

# Load trained model (manual process)
georf.model.load('')  # Load root branch
# s_branch loaded from: result_GeoRF_*/space_partitions/s_branch.pkl
```

### 8.4 Partition Metrics Tracking

GeoRF can track partition performance improvements during training:

```python
# Enable partition metrics tracking during training
georf = GeoRF(min_model_depth=1, max_model_depth=3)
georf.fit(X, y, X_group, 
          track_partition_metrics=True,
          correspondence_table_path='correspondence_table.csv')

# Metrics automatically saved to result_GeoRF_*/partition_metrics/
# - CSV files with group-wise F1/accuracy before/after each partition
# - Visualization maps showing performance improvements geographically
```

**Visualizing Partition Improvements:**
```python
from visualization import plot_metrics_improvement_map

# Plot F1 improvement map from saved metrics
plot_metrics_improvement_map(
    'result_GeoRF_27/partition_metrics/partition_metrics_round0_branchroot.csv',
    metric_type='f1_improvement',
    correspondence_table_path='correspondence_table.csv'
)
```

### 8.5 Partition Visualization and Debugging

```python
from visualization import plot_partition_map, plot_partition_map_from_result_dir

# Plot from correspondence table
plot_partition_map('result_GeoRF_27/correspondence_table_2021.csv')

# Plot from result directory (convenience function)
plot_partition_map_from_result_dir('result_GeoRF_27/', year='2021')
```

**Requirements for partition mapping:**
- Correspondence table with 'FEWSNET_admin_code' and 'partition_id' columns
- FEWS NET admin boundaries shapefile
- geopandas and contextily packages

### 8.6 Statistical Significance Testing

Partition creation uses statistical tests to determine optimal depth:

- **Cohen's d effect size**: Measures practical significance
- **T-test**: Tests statistical significance
- **Configurable thresholds**: ES_THRD, SIGLVL, MD_THRD

---

## 9. Output Structure

### 9.1 Directory Structure

```
result_GeoRF_*/                  # Random Forest results
├── checkpoints/               # Trained RF models
│   ├── rf_                   # Root model (all data)
│   ├── rf_0                  # Left branch after 1st split
│   ├── rf_1                  # Right branch after 1st split
│   ├── rf_00, rf_01          # After 2nd split
│   └── rf_000, rf_001, ...   # Deeper partitions
├── space_partitions/         # Partition definitions
│   ├── s_branch.pkl          # Group-to-branch mapping
│   ├── X_branch_id.npy       # Branch assignments
│   └── branch_table.npy      # Partition tree structure
├── partition_metrics/        # Partition performance tracking
│   ├── partition_metrics_round0_*.csv  # Performance before/after splits
│   └── improvement_maps_*.png # Geographic improvement visualization
├── vis/                      # Visualizations
│   ├── partition_*.png       # Partition maps
│   ├── performance_*.png     # Performance grids
│   └── diff_*.png           # Difference maps
├── correspondence_table_*.csv # Admin unit mappings
├── log_print.txt            # Training logs
├── log_print_eval.txt       # Evaluation logs
└── model.log               # Detailed model logs

result_GeoXGB_*/               # XGBoost results (identical structure to GeoRF)
├── checkpoints/               # Trained XGBoost models
│   ├── xgb_                  # Root model (instead of rf_)
│   ├── xgb_0, xgb_1         # Branch models
│   └── xgb_000, xgb_001, ... # Deeper partitions
├── space_partitions/         # Same partition structure
│   ├── s_branch.pkl          # Group-to-branch mapping
│   ├── X_branch_id.npy       # Branch assignments
│   └── branch_table.npy      # Partition tree structure
├── partition_metrics/        # NEW: Partition performance tracking
│   ├── partition_metrics_round0_*.csv
│   └── improvement_maps_*.png
├── vis/                      # Visualizations
├── correspondence_table_Q*.csv # Quarterly mappings (Q1_2024.csv, Q2_2024.csv, etc.)
├── log_print.txt            # Training logs
└── log_print_eval.txt       # Evaluation logs
```

### 9.2 Key Output Files

#### **s_branch.pkl**
Pandas DataFrame mapping group IDs to branch IDs:
```python
s_branch = pd.read_pickle('result_GeoRF_*/space_partitions/s_branch.pkl')
# Columns: group_id, branch_id (e.g., '', '0', '1', '00', '01', ...)
```

#### **branch_table.npy**
Binary tree structure showing which branches are split:
```python
branch_table = np.load('result_GeoRF_*/space_partitions/branch_table.npy')
# 1D array: 1 = split further, 0 = leaf node
```

#### **X_branch_id.npy**
Branch assignments for each data point:
```python
X_branch_id = np.load('result_GeoRF_*/space_partitions/X_branch_id.npy')
# String array with branch IDs for each sample
```

### 9.3 Performance Metrics

Standard classification metrics reported per class:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1 Score**: Harmonic mean of precision and recall

Results compared between:
- **GeoRF**: Spatially-partitioned model
- **Base RF**: Single global Random Forest

---

## 10. Updated XGBoost Pipeline Features

### 10.1 Complete Feature Parity

The updated `main_model_XGB.py` now provides complete feature parity with the Random Forest version:

#### **Checkpoint Recovery System (Removed)**
The legacy checkpoint-resume workflow has been retired. Both GeoRF and GeoXGB now
evaluate the requested quarters from scratch on each run, which keeps the code
paths aligned and avoids mismatches with older partial results.

#### **Rolling Window Temporal Evaluation**
```python
# 5-year training windows before each test quarter
(X_train, y_train, X_train_loc, X_train_group,
 X_test, y_test, X_test_loc, X_test_group) = train_test_split_rolling_window(
    X, y, X_loc, X_group, years, dates, test_year=test_year, 
    input_terms=input_terms, need_terms=quarter)
```

#### **Partition Metrics Tracking**
```python
# Enable in main() configuration
track_partition_metrics = True
enable_metrics_maps = True

# Creates detailed CSV files and geographic improvement maps
# Tracks F1/accuracy improvements before/after each partition round
```

#### **Memory Management for Large-Scale Evaluation**
```python
# Comprehensive cleanup for XGBoost models
# Handles quarterly evaluation across multiple years
# Progress tracking with memory usage monitoring
```

### 10.2 XGBoost-Specific Configuration

```python
# Complete hyperparameter control in main()
learning_rate = 0.1          # Step size shrinkage
reg_alpha = 0.1              # L1 regularization
reg_lambda = 1.0             # L2 regularization  
subsample = 0.8              # Training instance sampling
colsample_bytree = 0.8       # Feature sampling per tree

# Forecasting scope options
forecasting_scope = 1        # 1=3mo, 2=6mo, 3=9mo, 4=12mo lag

# Temporal evaluation range
start_year = 2024
end_year = 2024
desire_terms = None          # All quarters or specific (1-4)
```

### 10.3 Results and Output Files

```python
# Results follow same naming pattern as RF version
# Assignment method suffixes:
# - gp: polygons
# - gg: grid  
# - gc: country
# - gae: AEZ
# - gcae: country_AEZ
# - ggk: geokmeans
# - gak: all_kmeans

# Example batch result files:
# results_df_gf_fs1_2023_2024.csv      # GeoRF, 3mo lag, 2023-2024 time period
# results_df_xgb_fs2_2024_2024.csv     # XGBoost, 6mo lag, 2024 single year
# y_pred_test_gf_fs1_2023_2024.csv     # GeoRF predictions
# correspondence_table_Q1_2024.csv      # Quarterly admin code mapping
```

## 11. Troubleshooting

### 11.1 Common Issues

#### **Memory Issues**
- Reduce `N_JOBS` parameter
- Decrease `MAX_DEPTH` to limit partitions
- Increase `STEP_SIZE` to reduce number of groups
- Use smaller datasets for testing
- For XGBoost: disable GPU acceleration if memory-limited
- Consider using polars instead of pandas for large datasets

#### **Slow Performance**
- Increase `N_JOBS` for more parallel processing
- Reduce `MAX_DEPTH` for fewer partitions
- Increase `MIN_BRANCH_SAMPLE_SIZE` to stop early
- Use larger `STEP_SIZE` for fewer groups

#### **Poor Spatial Partitioning**
- Adjust `FLEX_RATIO` (0.01-0.1 range)
- Modify statistical thresholds: `SIGLVL`, `ES_THRD`
- Check group sizes - ensure adequate samples per group
- Verify spatial coordinates are correct

#### **Group Generation Issues**
- Check coordinate ranges: ensure xmin < xmax, ymin < ymax
- Verify `STEP_SIZE` creates reasonable number of groups
- For custom groups, ensure group IDs are consecutive integers starting from 0

#### **Polygon Contiguity Issues**
- **No neighbors found**: Check `neighbor_distance_threshold` - may be too small
- **Too many neighbors**: Increase `neighbor_distance_threshold` or use auto-calculation (None)
- **Contiguity not improving**: Verify polygon centroids are correctly calculated
- **Memory issues**: Reduce number of polygons or use larger distance threshold
- **Inconsistent results**: Ensure polygon_group_mapping is consistent across train/test

#### **Adjacency Matrix Issues (NEW)**

**Setup and Configuration:**
- **Shapefile not found**: Verify `ADJACENCY_SHAPEFILE_PATH` points to valid shapefile
- **Column not found**: Check `ADJACENCY_POLYGON_ID_COLUMN` exists in shapefile (uses row index if None)
- **Cache issues**: Delete `polygon_adjacency_cache.pkl` and set `ADJACENCY_FORCE_REGENERATE=True`
- **Permission errors**: Ensure write access to `ADJACENCY_CACHE_DIR` (or current directory)

**Performance Issues:**
- **Slow adjacency creation**: Use cached version; first generation can be slow for large shapefiles
- **Too many neighbors**: Normal for distance-based; adjacency matrix should reduce to ~4 neighbors per polygon
- **Too few neighbors**: Verify shapefile has proper polygon topology and boundary sharing

**Data Quality Issues:**
- **Isolated polygons**: 655 isolated polygons (11.5%) detected in FEWS data - normal for islands/enclaves
- **CRS warnings**: Consider reprojecting shapefile to appropriate coordinate system for accuracy
- **Geometry errors**: Use `gpd.read_file(shapefile).geometry.is_valid` to check polygon validity

**Integration Issues:**
- **Adjacency not used**: Verify `USE_ADJACENCY_MATRIX=True` and proper contiguity_info passing
- **Fallback to distance**: Check logs for adjacency loading errors; system falls back to distance-based
- **Performance not improved**: Adjacency matrix should show significant neighbor count reduction in logs

**Validation and Debugging:**

**Built-in Validation (Updated):**
The `validate_polygon_contiguity()` function in both main model files now automatically detects and validates the active contiguity approach:

```python
# Automatic validation during model execution
# In main_model_GF.py and main_model_XGB.py:
if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans']:
    validate_polygon_contiguity(contiguity_info, X_group)

# Function automatically detects and reports:
# - "Using adjacency matrix-based neighbors (production approach)" 
# - OR "Warning: Using distance-based neighbors (fallback)"
# - Neighbor statistics relevant to the active approach
# - Isolated polygon counts with appropriate context
```

**Manual Validation Tools:**
```python
from adjacency_utils import validate_adjacency_matrix, load_or_create_adjacency_matrix

# Load and validate adjacency matrix
adj_dict, id_mapping, centroids = load_or_create_adjacency_matrix()

# Compare with distance-based approach
validation_results = validate_adjacency_matrix(adj_dict, centroids, distance_threshold=0.8)

# Check neighbor statistics
print(f"Adjacency neighbors: {validation_results['adj_total_connections']}")
print(f"Distance neighbors: {validation_results['dist_total_connections']}")
print(f"Reduction ratio: {validation_results['connection_ratio']:.2f}")
```

**Current Pipeline Status:**
- **Production Default**: `assignment='polygons'` with `USE_ADJACENCY_MATRIX=True`
- **Expected Output**: "Using adjacency matrix-based neighbors (production approach)"
- **Expected Performance**: ~4 neighbors per polygon, ~23K total connections
- **Legacy Cleaned**: No more testing of unused distance-based approach in production

### 11.2 Debugging Tips

#### **Check Data Integrity**
```python
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")  
print(f"X_loc shape: {X_loc.shape}")
print(f"X_group unique values: {len(np.unique(X_group))}")
print(f"Coordinate ranges: lat={X_loc[:,0].min():.2f}-{X_loc[:,0].max():.2f}, "
      f"lon={X_loc[:,1].min():.2f}-{X_loc[:,1].max():.2f}")
```

#### **Monitor Training Progress**
```python
# Check training logs
tail -f result_GeoRF_*/log_print.txt

# Monitor partitioning depth
grep "depth" result_GeoRF_*/log_print.txt
```

#### **Validate Groups**
```python
# Check group sizes
group_sizes = pd.Series(X_group).value_counts()
print(f"Group size stats: min={group_sizes.min()}, max={group_sizes.max()}, "
      f"mean={group_sizes.mean():.1f}")

# Very small groups may cause issues
small_groups = group_sizes[group_sizes < 10]
if len(small_groups) > 0:
    print(f"Warning: {len(small_groups)} groups have <10 samples")
```

#### **Validate Polygon Contiguity**
```python
# Check polygon neighbor relationships
from partition_opt import get_polygon_neighbors

neighbors = get_polygon_neighbors(polygon_centroids, neighbor_distance_threshold=0.8)
neighbor_counts = [len(n) for n in neighbors.values()]

print(f"Neighbor stats: min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
      f"mean={np.mean(neighbor_counts):.1f}")

# Check for isolated polygons (no neighbors)
isolated_polygons = [poly_id for poly_id, neighs in neighbors.items() if len(neighs) == 0]
if len(isolated_polygons) > 0:
    print(f"Warning: {len(isolated_polygons)} isolated polygons: {isolated_polygons}")

# Check centroids
print(f"Centroid range: lat {polygon_centroids[:, 0].min():.2f}-{polygon_centroids[:, 0].max():.2f}, "
      f"lon {polygon_centroids[:, 1].min():.2f}-{polygon_centroids[:, 1].max():.2f}")
```

### 11.3 Model Branch-Specific Issues

#### **Random Forest Branch Issues**
- **Memory bottlenecks**: Use batch processing with 20 batches (time periods)
- **Poor performance**: Consider switching to XGBoost branch for better regularization
- **Fast prototyping**: Ideal for quick experimentation and baseline establishment

#### **XGBoost Branch Issues**

**Memory Issues:**
- XGBoost models can use significantly more memory than Random Forest
- Use 40-batch processing (single years) instead of 20-batch (time periods)
- Reduce `n_trees_unit` or use `subsample` and `colsample_bytree` < 0.8
- Monitor memory during large-scale quarterly evaluation
- Use checkpoint recovery to avoid recomputing completed quarters

**Performance Issues:**
- **Slow training**: Increase `learning_rate` but add more regularization
- **Overfitting**: Increase `reg_alpha` and `reg_lambda`, reduce `learning_rate`
- **Underfitting**: Decrease regularization, increase `n_trees_unit`
- **GPU issues**: Set `n_jobs=1` and verify CUDA installation
- **Hyperparameter sensitivity**: Start with balanced_config preset

**Checkpoint Recovery Issues:**
- Incomplete checkpoints: Check for existence of both `checkpoints/` and `space_partitions/` directories
- Partial results not loading: Verify filename patterns match current configuration
- Memory errors during resume: Clear old result directories or increase system memory

**Branch Selection Guidelines:**
- **Use Random Forest Branch when**: Quick prototyping, limited computational resources, baseline establishment
- **Use XGBoost Branch when**: Production deployments, performance optimization critical, sufficient computational resources
- **Performance expectation**: XGBoost typically provides 3-8% F1-score improvement over Random Forest

### 11.4 Performance Optimization

#### **For Large Datasets**
1. **Preprocessing**: Use feature selection to reduce dimensionality
2. **Sampling**: Train on subset, evaluate on full dataset
3. **Incremental**: Process data in chunks if memory-limited
4. **Distributed**: Use multiple machines with distributed frameworks
5. **XGBoost-specific**: Use `subsample` and `colsample_bytree` for memory efficiency
6. **Checkpoint recovery**: Leverage automatic resume functionality for long-running evaluations

#### **For Real-time Applications**
1. **Model Compression**: Save only necessary partition information
2. **Fast Grouping**: Pre-compute group assignments for common locations
3. **Simplified Models**: Reduce `n_trees_unit` for faster predictions
4. **XGBoost optimization**: Use higher `learning_rate` with fewer trees for speed
5. **Memory management**: Use checkpoint system to avoid retraining completed partitions

#### **For Temporal Evaluation**
1. **Rolling windows**: Use 5-year training windows for temporal consistency
2. **Quarterly processing**: Process quarters independently with checkpoint recovery
3. **Memory cleanup**: Automatic model cleanup between quarters
4. **Progress tracking**: Monitor evaluation progress across multiple years

---

## Conclusion

GeoRF provides a powerful unified framework for spatial machine learning that goes beyond traditional approaches by explicitly modeling spatial heterogeneity. The framework offers **two model branches** with identical spatial logic but different base learners, allowing users to choose the optimal approach for their specific requirements.

### Framework Strengths:
- **Unified Spatial Logic**: Identical hierarchical partitioning for both Random Forest and XGBoost
- **Branch Flexibility**: Choose between Random Forest (fast prototyping) and XGBoost (production performance)
- **Automatic Partitioning**: Data-driven determination of optimal spatial regions
- **Flexible Grouping**: Grid-based, polygon-based, and K-means clustering strategies
- **Advanced Contiguity Systems**: 
  - Grid-based contiguity for regular spatial data
  - Distance-based polygon contiguity for approximate relationships
  - **Adjacency matrix contiguity for true polygon boundary relationships (86% neighbor reduction)**
- **Performance Tracking**: Detailed partition metrics and improvement visualization
- **Statistical Rigor**: Significance testing for partition validation
- **Scalable Design**: Parallel processing and memory-optimized batch processing
- **True Spatial Accuracy**: Polygon adjacency matrices provide actual boundary relationships vs approximations
- **Comprehensive Evaluation**: Built-in comparison with multiple baseline models

### Model Branch Selection Guide:
- **Random Forest Branch**: Ideal for exploratory analysis, rapid prototyping, and resource-constrained environments
- **XGBoost Branch**: Recommended for production deployments where performance is critical and computational resources are sufficient

### Typical Use Cases:
1. **Rapid Development**: Start with Random Forest branch for quick insights and baseline establishment
2. **Production Deployment**: Switch to XGBoost branch for optimized performance in operational systems
3. **Comparative Analysis**: Use both branches to validate spatial partitioning effectiveness across different base learners
4. **Large-scale Evaluation**: Leverage batch processing pipeline for comprehensive temporal validation

The framework's design ensures that users can seamlessly transition between model branches while maintaining consistent spatial partitioning logic and evaluation protocols.

For further questions or advanced customizations, refer to the source code documentation and demo scripts provided with the framework.
