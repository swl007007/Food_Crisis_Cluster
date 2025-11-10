# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import sys
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    if '--dry-run' in sys.argv:
        class _NumpyStub:
            def array(self, value):  # type: ignore[override]
                return value

        np = _NumpyStub()  # type: ignore
    else:
        raise

#Note: Some of the parameters are for the deep learning version (commented).
#They are for now kept as part of the config file in case some conditions
#were used in some function definitions in the code repo.

#Model choices
MODEL_CHOICE = 'RF'
# Please refer to other code version in github for deep learning models
#Task (only classification version for RF has been tested)
MODE = 'classification'#'regression'

# Visualization guardrails (synced with config_visual)
VISUALIZE_ENFORCE_PARENT_SCOPE = True
VISUALIZE_HIDE_UNASSIGNED = True
PARTITIONING_VALIDATE_TERMINAL_LABELS = True

# Lag schedule (GeoRF + baseline alignment)
LEGACY_LAG_VALUES = {3, 6, 9}
DEFAULT_LAGS_MONTHS: Sequence[int] = (4, 8, 12)
LAGS_MONTHS = list(DEFAULT_LAGS_MONTHS)

# Admin code column aliases (for flexible data source compatibility)
ADMIN_CODE_ALIASES = ['FEWSNET_admin_code', 'admin_code', 'adm_code']

# Production mode: Skip baseline 5-fold cross-validation during training
# This validation is expensive and primarily useful for debugging/visualization
DISABLE_BASELINE_CV_MAP = True


def _validate_lag_schedule(lags: Sequence[int]) -> list[int]:
    resolved = list(lags)
    if list(dict.fromkeys(resolved)) != resolved:
        raise ValueError(f"Duplicated lag entries detected: {resolved}")
    if any(lag in LEGACY_LAG_VALUES for lag in resolved):
        raise ValueError(
            f"Legacy lag months detected ({LEGACY_LAG_VALUES}); GeoRF now requires {list(DEFAULT_LAGS_MONTHS)}."
        )
    if list(resolved) != list(DEFAULT_LAGS_MONTHS):
        raise ValueError(
            f"Lag schedule {resolved} unsupported; configure to {list(DEFAULT_LAGS_MONTHS)} to match FEWSNET baseline."
        )
    return resolved


LAGS_MONTHS = _validate_lag_schedule(LAGS_MONTHS)


#------------------GeoRF parameters------------------

#**************************ATTENTION NEEDED [1, total 3]**************************
#GeoRF structure parameters
#min and max depth for partitioning (actual number between min-max is auto-determined by significance testing)
MIN_DEPTH = 1  # Reduced to allow more splitting opportunities
MAX_DEPTH = 6  # Increased to allow deeper partitioning for more figures
# CRITICAL FIX 4: Adaptive N_JOBS to prevent memory fragmentation and bitmap allocation failures
# High n_jobs (32) can cause memory fragmentation and bitmap allocation errors
try:
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    if memory_gb < 8:
        N_JOBS = 1  # Very low memory systems - single threaded
        print(f"Low memory system ({memory_gb:.1f}GB): Using N_JOBS=1")
    elif memory_gb < 16:
        N_JOBS = 2  # Low memory systems 
        print(f"Limited memory system ({memory_gb:.1f}GB): Using N_JOBS=2")
    elif memory_gb < 32:
        N_JOBS = 4  # Medium memory systems
        print(f"Medium memory system ({memory_gb:.1f}GB): Using N_JOBS=4")
    elif memory_gb < 64:
        N_JOBS = 8  # Higher memory systems
        print(f"Higher memory system ({memory_gb:.1f}GB): Using N_JOBS=8")
    else:
        # Cap at 16 even for high-memory systems to prevent bitmap allocation issues
        N_JOBS = min(16, max(1, cpu_count // 2))
        print(f"High memory system ({memory_gb:.1f}GB): Using N_JOBS={N_JOBS}")
        
except ImportError:
    # Fallback if psutil is not available
    N_JOBS = 4  # Safe default to prevent memory issues
    print("psutil not available: Using safe default N_JOBS=4")
except Exception as e:
    # Fallback for any other error
    N_JOBS = 4  # Safe default
    print(f"Error detecting system specs: Using safe default N_JOBS=4. Error: {e}")

print(f"Final N_JOBS configuration: {N_JOBS}")
# Note: This will be further dynamically adjusted based on real-time memory pressure during RF training
#*********************************************************************************

#Detailed ***Optional*** specifications
MIN_BRANCH_SAMPLE_SIZE = 0 # Reduced to allow more partitioning
MIN_SCAN_CLASS_SAMPLE = 0   # Reduced to allow more partitioning opportunities
FLEX_RATIO = 0.1#affects max size difference between two partitions in each split
FLEX_OPTION = True
#FLEX_TYPE = 'n_sample'
#FLEX_TYPE = 'n_group_w_sample'#careful with threshold
FLEX_TYPE = 'n_group'
#MIN_GROUP_POS_SAMPLE_SIZE_FLEX = 10#minimum number of positive samples in a group
MIN_GROUP_POS_SAMPLE_SIZE_FLEX = 0
#For significance testing
#SIGLVL = 0.01#significance level. #0.05
SIGLVL = 0.1  # More lenient significance level to allow more splits
ES_THRD = 0.5  # Reduced effect size threshold to allow more splits
MD_THRD = 0.0005  # Reduced mean difference threshold


#------------------Training data related parameters------------------
#**************************ATTENTION NEEDED [2, total 3]**************************
#Train-val-test split
#Used as default function inputs
TRAIN_RATIO = 0.6
VAL_RATIO = 0.20#subset from training, e.g., 0.2 means 20% of training data will be set as validation ## validation set
TEST_RATIO = 1 - TRAIN_RATIO

# Group-aware validation split configuration
GROUP_SPLIT = {
    'enable': True,
    'min_val_per_group': 1,
    'skip_singleton_groups': True,
    'random_state': 42,
}

# Feature drop configuration (post temporal split, pre-training)
FEATURE_DROP = {
    'enable': True,
    'cols': ['year', 'month','fews_ha','IPC_admin_code','ISO_encoded','years'],
    'patterns': [],
}

# Backward compatible alias for lowercase access
feature_drop = FEATURE_DROP
#*********************************************************************************


#------------------Spatial range related parameters------------------
#**************************ATTENTION NEEDED [3, total 3]**************************
#Spatial range parameters used for the CONUS crop classification data.
#In the code now xmin, xmax, ymin, and ymax are derived based on data in preprocessing
xmin = -26.66055672814686
xmax = 38.22374755974924
xmax_raw = 38.22374755974924
ymax_raw = 73.33798817378396
xmax = xmax_raw
ymax = ymax_raw
ymin = -92.18212153785024
#The following is used in case the grouping is done by a grid
#X_DIM is kept as some visualization code used for testing purposes used those
X_DIM = np.array([xmax-xmin, ymax-ymin])#input img size in 2D
STEP_SIZE = 0.1#1 unit: degree
#*********************************************************************************

#The following is likely unused in the code shared (maybe in visualization or CONUS crop data preprocessing) and was used for specific visualizations
import math
X_DIM_RAW = np.array([xmax_raw, ymax_raw])#input img size in 2D
GRID_DIM = np.array([math.ceil(X_DIM[0]/STEP_SIZE), math.ceil(X_DIM[1]/STEP_SIZE)])
N_GROUPS = GRID_DIM[0] * GRID_DIM[1]

#Some of the paras are part of this specific implementation example, and can be changed.
#Two types of locations are used in this example for each data point:
#1. Its row and column ids in the original input frame (e.g., i,j in an image tile).
#2. Its grid cell's row and column ids in the grid (we overlay a grid on top of the original data to create unit groups. Each cell is a group).
GRID_COLS = [2,3]#The colunns to store the grid-based locations in X_loc (shape: Nx4). In this example the first two columns in X_loc store the original locations.
IMG_SIZE = 128

#------------------Additional parameters------------------
#Used if certain classes should be ignored in partitioning optimization or significance testing.
#Example: Imblanced binary classification where non-background pixels only accounts for a small proportion of samples.'''
#single vs multi
# multi = True
multi = False
if multi:
  #multi
  SELECT_CLASS = np.array([1,2,3,4,5,6,7])#multi-class (remove 0,9,10) and 8 (non-crop)
  NUM_CLASS = 9
  #in the data, there is only classes 0-8 (9 classes), not 0-10
  #use all
  # SELECT_CLASS = None # or np.array(range(NUM_CLASS))#select all classes
  #used in get_class_wise_stat() and get_score()
  #used locations marked by "#select_class"
else:
  #binary: one crop
  SELECT_CLASS = np.array([1])
  NUM_CLASS = 2

#For improved spatial contiguity of partitions
#This supports both grid-based and polygon-based contiguity
# CONTIGUITY = False
# REFINE_TIMES = 0
CONTIGUITY = True
REFINE_TIMES = 3
#MIN_COMPONENT_SIZE = 10
MIN_COMPONENT_SIZE = 5

#Contiguity type: 'grid' or 'polygon'
CONTIGUITY_TYPE = 'polygon'  # Default to polygon-based contiguity

#Polygon contiguity parameters (used when CONTIGUITY_TYPE = 'polygon')
POLYGON_NEIGHBOR_DISTANCE_THRESHOLD = 0.8  # Auto-calculate if None
POLYGON_CONTIGUITY_INFO = None  # Set this to polygon contiguity info dict when using polygon contiguity

# Adjacency matrix parameters for polygon-based contiguity
USE_ADJACENCY_MATRIX = True  # If True, use true polygon adjacency instead of distance-based neighbors
ADJACENCY_SHAPEFILE_PATH = r'C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp'
ADJACENCY_POLYGON_ID_COLUMN = 'admin_code'  # Column name for polygon identifiers in shapefile
ADJACENCY_CACHE_DIR = None  # Directory for caching adjacency matrix (None = current directory)
ADJACENCY_FORCE_REGENERATE = False  # If True, regenerate adjacency matrix even if cache exists

#Visualization debug control
VIS_DEBUG_MODE = False  # Set to False to disable all debug visualizations and metric tables to speed up execution

# FRAGMENTATION BUG FIX DOCUMENTATION (2025-08-28)
# Fixed critical bug where temporal records were being counted as spatial units
# Root cause: Spatial partitioning processed 70,228 temporal records but correspondence 
# table generation created entries for each temporal record instead of unique admin units
# Fix applied to: GeoRF.py, main_model_GF.py, main_model_GF_visual_debug.py, main_model_XGB.py
# Result: Visualization now shows correct admin unit counts (~3,600) instead of temporal record counts (~70,000)

# Crisis-focused optimization parameters
GOVERNING_METRIC = 'class_1_f1'  # Primary optimization target for crisis prediction
CRISIS_FOCUSED_OPTIMIZATION = True  # Enable class 1 prioritization throughout pipeline
PARTITION_OPTIMIZATION_METRIC = 'class_1_f1'  # Metric for partition selection
CLASS_1_SIGNIFICANCE_TESTING = True  # Use class 1 metrics in significance testing
MIN_CLASS_1_IMPROVEMENT_THRESHOLD = 0.01  # Minimum class 1 F1 improvement to accept partition

# Metric calculation modes
METRIC_CALCULATION_MODES = {
    'class_1_f1': 'f1_score_class1',
    'class_1_precision': 'precision_class1', 
    'class_1_recall': 'recall_class1',
    'balanced_accuracy': 'balanced_accuracy_score',
    'overall_accuracy': 'categorical_accuracy'  # Legacy default for backward compatibility
}

# Polygon preservation settings to prevent polygon disappearance
PRESERVE_ISOLATED_POLYGONS = True  # Keep polygons with no neighbors during contiguity refinement
VALIDATE_POLYGON_COUNTS = True     # Assert polygon count preservation at each stage
USE_OUTER_JOINS = True             # Prevent join-based losses in visualization
DIAGNOSTIC_POLYGON_TRACKING = True  # Log polygon counts at each stage

# Contiguity refinement safeguards  
MIN_ADJACENCY_THRESHOLD = 0.0      # Don't filter polygons by neighbor count
PRESERVE_ORIGINAL_ON_MAPPING_FAIL = True  # Fallback for mapping failures

# Final accuracy rendering configuration (requires VIS_DEBUG_MODE=True)
FINAL_ACCURACY_BACKEND = 'matplotlib'  # Visualization backend for accuracy maps
FINAL_ACCURACY_DPI = 200               # Output resolution for accuracy maps
FINAL_ACCURACY_MISSING_COLOR = '#dddddd'  # Color for polygons with missing accuracy data

#predefined groups such as US counties
#unused here
PREDEFINED_GROUPS = False
#PREDEFINED_GROUPS = False
if PREDEFINED_GROUPS:
  CONTIGUITY = False

#Not used here: for folder naming in specific testing cases over different settings
EVAL_EXT = ''

#----------------------Deep learning version only (not impacting results but might still be mentioned somewhere in function definitions)----------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#The following will only be used for deep learning models (do not comment out)
# MODEL_CHOICE = 'DNN'
# MODEL_CHOICE = 'LSTM'
# MODEL_CHOICE = 'UNET'

#*****************DO NOT DELETE***********************
ONEHOT = False
TIME_SERIES = False
#*****************DO NOT DELETE***********************

#Training related parameters - DL/default
PRETRAIN_EPOCH = 120
EPOCH_TRAIN = 120
BATCH_SIZE = 256*256
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0005
CLASSIFICATION_LOSS = 'categorical_crossentropy'
REGRESSION_LOSS = 'mean_squared_error'

INPUT_SIZE = 10*33+3#X.shape[1]#number of features
LAYER_SIZE = min(INPUT_SIZE, 10)


#Training related parameters: LSTM
if MODEL_CHOICE == 'LSTM':
  TIME_SERIES = True
  PRETRAIN_EPOCH = 60
  EPOCH_TRAIN = 60
  BATCH_SIZE = 256*256
  N_TIME = 33
  N_TIME_FEATURE = 10
  N_OTHER_FEATURE = 3
  LEARNING_RATE = 0.001
  CLASSIFICATION_LOSS = 'categorical_crossentropy'
  REGRESSION_LOSS = 'mean_squared_error'

#Training related parameters - UNet
if MODEL_CHOICE == 'UNET':
  PRETRAIN_EPOCH = 20#Stablize the model parameters before the partitioning starts
  EPOCH_TRAIN = 20#Number of epochs to train after each split (and equivalently, before the next split)
  BATCH_SIZE = 32
  LEARNING_RATE = 0.0001
  CLASSIFICATION_LOSS = 'categorical_crossentropy'
  REGRESSION_LOSS = 'mean_squared_error'
