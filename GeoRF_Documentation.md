# GeoRF: Spatial Transformation Framework for Random Forest
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

**GeoRF** is a Spatial Transformation Framework for Random Forest that implements geo-aware machine learning for food crisis prediction and clustering analysis. The system handles spatial variability through hierarchical spatial partitioning, making it particularly effective for geographic data where regional variations are important.

### Key Features
- **Spatial Partitioning**: Hierarchical binary partitioning based on spatial/geographic groups
- **Dual Implementations**: Both Random Forest (GeoRF) and XGBoost (GeoRF_XGB) variants
- **Geo-aware Predictions**: Location-informed model selection for test points
- **Scalable Architecture**: Parallel processing support for large datasets
- **Flexible Grouping**: Grid-based, polygon-based, and K-means clustering spatial strategies
- **Spatial Contiguity**: Both grid-based and polygon-based contiguity refinement
- **Statistical Validation**: Significance testing for optimal partition depth
- **2-Layer Architecture**: Main prediction + error correction layers for nowcasting/forecasting
- **Comprehensive Visualization**: Spatial maps, performance grids, and partition metrics tracking

### Application Domain
Originally developed for food crisis prediction using satellite and ground-based data, but adaptable to any spatially-varying classification or regression task.

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
├── GeoRF.py                      # Main GeoRF Random Forest class
├── GeoRF_XGB.py                  # GeoRF XGBoost implementation
├── GeoRF_main.py                 # Basic Random Forest demo
├── GeoRF_demo.py                 # Demonstration script
├── main_model_GF.py              # Food crisis pipeline (Random Forest)
├── main_model_XGB.py             # Food crisis pipeline (XGBoost)
├── config.py                     # Global configuration parameters
├── data.py                       # Data loading utilities
├── customize.py                  # Group generation classes
├── model_RF.py                   # Random Forest model wrapper
├── model_XGB.py                  # XGBoost model wrapper
├── transformation.py             # Spatial partitioning algorithms
├── partition_opt.py              # Partition optimization and contiguity
├── train_branch.py               # Branch-specific training logic
├── initialization.py             # Data initialization utilities
├── helper.py                     # Utility functions
├── metrics.py                    # Evaluation metrics
├── visualization.py              # Visualization and mapping functions
├── sig_test.py                   # Statistical significance testing
├── baseline_probit_regression.py # Probit baseline comparison
├── metric_comparison_plot.py     # Performance comparison plots
├── static_table.py               # Result table generation
└── result_GeoRF*/                # Random Forest output directories
└── result_GeoXGB*/               # XGBoost output directories
```

---

## 3. Architecture and Components

### Core System Flow
```
Data Input → Group Generation → Spatial Partitioning → Local RF Training → Prediction → Evaluation
```

### Module Responsibilities

#### **GeoRF.py** - Random Forest Framework
- **Primary Class**: `GeoRF`
- **Purpose**: Main Random Forest interface providing standard ML methods (`fit()`, `predict()`, `evaluate()`)
- **Key Features**: 
  - Standard and 2-layer GeoRF variants
  - Spatial contiguity refinement (grid and polygon)
  - Model persistence and loading
  - Partition metrics tracking

#### **GeoRF_XGB.py** - XGBoost Framework  
- **Primary Class**: `GeoRF_XGB`
- **Purpose**: XGBoost implementation with same interface as GeoRF
- **Key Features**:
  - XGBoost-based local models
  - Same spatial partitioning as Random Forest variant
  - Hyperparameter optimization support
  - GPU acceleration capability

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

---

## 4. Core Classes and Methods

### 4.1 GeoRF Class

```python
class GeoRF():
    def __init__(self, min_model_depth=1, max_model_depth=4, n_trees_unit=100, 
                 num_class=2, max_depth=None, random_state=5, n_jobs=32,
                 mode='classification', sample_weights_by_class=None)
```

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

### 6.1 Running the Models

#### **Basic Random Forest GeoRF**
```bash
python GeoRF_main.py
```

#### **Food Crisis Pipeline (Random Forest)**
```bash
python main_model_GF.py
```

#### **Food Crisis Pipeline (XGBoost)**
```bash
python main_model_XGB.py
```

*Note: The XGBoost version (`main_model_XGB.py`) is now a complete replication of the Random Forest pipeline with identical functionality, checkpoint recovery, rolling window evaluation, and partition metrics tracking.*

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

# Train and evaluate
georf = GeoRF(max_model_depth=3)
georf.fit(X_train, y_train, X_train_group)
y_pred = georf.predict(X_test, X_test_group)
pre, rec, f1 = georf.evaluate(X_test, y_test, X_test_group)
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

### 7.7 XGBoost Implementation

```python
from GeoRF_XGB import GeoRF_XGB

# XGBoost variant with same interface as GeoRF
georf_xgb = GeoRF_XGB(max_model_depth=3, n_jobs=16)
georf_xgb.fit(X_train, y_train, X_group_train)
y_pred = georf_xgb.predict(X_test, X_group_test)

# XGBoost with comprehensive hyperparameter control
georf_xgb = GeoRF_XGB(
    max_model_depth=3,
    learning_rate=0.1,      # Step size shrinkage
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    subsample=0.8,          # Training instance sampling
    colsample_bytree=0.8,   # Feature sampling per tree
    n_estimators=100
)

# Complete XGBoost pipeline (equivalent to main_model_GF.py)
# python main_model_XGB.py
# Features: checkpoint recovery, rolling window evaluation, partition metrics
```

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

GeoRF can improve spatial contiguity of partitions using majority voting in local neighborhoods. Two types of contiguity are supported:

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

#### 8.1.2 Polygon-Based Contiguity

For polygon-based groups (administrative boundaries, watersheds, etc.), uses centroid-based neighbor detection:

```python
# Enable polygon contiguity in config.py
CONTIGUITY = True
CONTIGUITY_TYPE = 'polygon'
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

**Key Features of Polygon Contiguity:**
- **Centroid-based neighbors**: Uses polygon centroids to determine spatial neighbors
- **Adaptive thresholds**: Automatically calculates neighbor distance if not specified
- **Majority voting**: Same 4/9 threshold as grid-based system for consistency
- **Flexible mapping**: Supports one-to-one or one-to-many polygon-to-group mappings

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

#### **Checkpoint Recovery System**
```python
# Automatic detection of previous runs
completed_quarters, partial_results_files, checkpoint_dirs = get_checkpoint_info()
remaining_quarters = determine_remaining_quarters(completed_quarters, start_year, end_year, desire_terms)

# Automatic resume from last completed quarter
results_df, y_pred_test = load_partial_results(partial_results_files, assignment, nowcasting, max_depth, desire_terms, forecasting_scope)
```

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

# Example files:
# results_df_gp_d3_t1_fs1.csv      # Polygons, depth=3, Q1 only, 3mo lag
# y_pred_test_gp_d3_t1_fs1.csv     # Corresponding predictions
# correspondence_table_Q1_2024.csv  # Quarterly admin code mapping
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

### 11.3 XGBoost-Specific Issues

#### **XGBoost Memory Issues**
- XGBoost models can use more memory than Random Forest
- Reduce `n_trees_unit` or use `subsample` and `colsample_bytree` < 0.8
- Monitor memory during large-scale quarterly evaluation
- Use checkpoint recovery to avoid recomputing completed quarters

#### **XGBoost Performance Issues**
- Slow training: Increase `learning_rate` but add more regularization
- Overfitting: Increase `reg_alpha` and `reg_lambda`, reduce `learning_rate`
- Underfitting: Decrease regularization, increase `n_trees_unit`
- GPU issues: Set `n_jobs=1` and verify CUDA installation

#### **Checkpoint Recovery Issues**
- Incomplete checkpoints: Check for existence of both `checkpoints/` and `space_partitions/` directories
- Partial results not loading: Verify filename patterns match current configuration
- Memory errors during resume: Clear old result directories or increase system memory

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

GeoRF provides a powerful framework for spatial machine learning that goes beyond traditional approaches by explicitly modeling spatial heterogeneity. The hierarchical partitioning approach allows the model to adapt to different patterns across geographic regions while maintaining computational efficiency through parallel processing.

Key strengths:
- **Spatial Awareness**: Explicitly handles spatial non-stationarity
- **Dual Implementations**: Both Random Forest and XGBoost variants with identical interfaces
- **Automatic Partitioning**: Data-driven determination of optimal spatial regions
- **Flexible Grouping**: Grid-based, polygon-based, and K-means clustering strategies
- **Dual Contiguity Systems**: Both grid-based and polygon-based contiguity refinement
- **Performance Tracking**: Detailed partition metrics and improvement visualization
- **Statistical Rigor**: Significance testing for partition validation
- **Scalable Design**: Parallel processing and GPU acceleration support
- **Comprehensive Evaluation**: Built-in comparison with baseline models

For further questions or advanced customizations, refer to the source code documentation and example notebooks provided with the framework.