#!/usr/bin/env python3
"""
Partitioned (k40_nc4) vs Pooled RF/DT Comparison Script

Evaluates monthly from 2021-01 through 2024-12:
- Pooled baseline: Single RF/DT trained on all partitions together
- Partitioned model: Separate RF/DT per partition (cluster_id from cluster_mapping_k40_nc4.csv)

Uses GeoRF pipeline utilities for proper feature preparation and temporal splitting.

Outputs:
- metrics_monthly.csv: Monthly precision/recall/F1 for both models
- predictions_monthly.csv: All predictions with uid/time/partition/y_true/y_pred
- metrics_polygon_overall.csv: Polygon-level aggregated metrics and F1 improvement
- run_manifest.json: Run parameters and diagnostics

Visualization (VISUAL=True only):
- final_f1_performance_map.png: Polygon F1 choropleth (pooled vs partitioned panels)
- map_pct_err_all.png: Overall error rate by polygon
- map_pct_err_class1.png: Class-1 specific error rate by polygon
- overall_f1_improvement_map.png: F1 delta (partitioned - pooled) by polygon

Author: Claude Code
Date: 2026-01-25
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GeoRF pipeline utilities
from src.preprocess.preprocess import load_and_preprocess_data
from src.feature.feature import prepare_features
from src.customize.customize import train_test_split_rolling_window
from src.utils.lag_schedules import forecasting_scope_to_lag
from src.metrics.metrics import get_prf, get_class_wise_accuracy
from config import LAGS_MONTHS

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration and Defaults
# ============================================================================

DEFAULT_DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_forecast_unadjusted_bm_phase_change.csv"
DEFAULT_PARTITION_MAP = "cluster_mapping_k40_nc4.csv"
DEFAULT_POLYGONS_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWS NET Admin Boundaries\FEWS_Admin_LZ_v3.shp"
DEFAULT_OUT_DIR = r".\result_partition_k40_nc4_compare"
DEFAULT_START_MONTH = "2021-01"
DEFAULT_END_MONTH = "2024-12"
DEFAULT_TRAIN_WINDOW = 36  # months
DEFAULT_FORECASTING_SCOPE = 1  # 1=4mo, 2=8mo, 3=12mo lag
RANDOM_STATE = 5  # MUST match main pipeline (GeoRF.py default)
SMOTE_K_NEIGHBORS = 5

# RF hyperparameters (MUST match main GeoRF pipeline for pooled baseline comparison)
# See src/model/GeoRF.py line 51 for main pipeline defaults
RF_PARAMS = {
    'n_estimators': 100,  # Main pipeline uses n_trees_unit=100 (NOT 500)
    'max_depth': None,     # Main pipeline uses max_depth=None (unlimited tree depth)
    'random_state': RANDOM_STATE,  # Main pipeline uses random_state=5 (NOT 42)
    'n_jobs': -1
    # NOTE: class_weight is NOT set in main pipeline (defaults to None, NOT 'balanced')
    # NOTE: min_samples_leaf defaults to 1 (not explicitly set in main pipeline)
}

# DT hyperparameters (MUST match main GeoDT pipeline for pooled baseline comparison)
# See src/model/model_DT.py for default constructor usage
DT_PARAMS = {
    'max_depth': None,
    'random_state': RANDOM_STATE
}

# Minimum samples per partition to train separate model (else fallback to pooled)
MIN_PARTITION_SAMPLES = 50

_SMOTE_WARNING_EMITTED = False


def _model_label(lower_model: str) -> str:
    """Return display label for supported lower model variants."""
    return 'DT' if lower_model == 'dt' else 'RF'


def _create_model(lower_model: str) -> Any:
    """Create a classifier instance for the selected lower model."""
    if lower_model == 'dt':
        return DecisionTreeClassifier(**DT_PARAMS)
    return RandomForestClassifier(**RF_PARAMS)


def _apply_partition_smote(
    X: np.ndarray,
    y: np.ndarray,
    partition_id: int,
    smote_k_neighbors: int = SMOTE_K_NEIGHBORS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to a single partition when possible, else return inputs."""

    global _SMOTE_WARNING_EMITTED

    if SMOTE is None:
        if not _SMOTE_WARNING_EMITTED:
            print("    SMOTE unavailable: install imbalanced-learn to enable oversampling for partition comparisons.")
            _SMOTE_WARNING_EMITTED = True
        return X, y

    unique_classes, counts = np.unique(y, return_counts=True)
    if unique_classes.shape[0] < 2:
        print(f"    SMOTE skipped for partition {partition_id}: only one class present.")
        return X, y

    minority_count = counts.min()
    if minority_count < 2:
        print(f"    SMOTE skipped for partition {partition_id}: minority class has fewer than 2 samples.")
        return X, y

    k_neighbors = min(smote_k_neighbors, minority_count - 1)
    if k_neighbors < 1:
        print(f"    SMOTE skipped for partition {partition_id}: insufficient samples for synthetic neighbors.")
        return X, y

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)

    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        synthetic = len(y_resampled) - len(y)
        if synthetic > 0:
            print(
                f"    SMOTE applied for partition {partition_id}: added {synthetic} synthetic samples (k_neighbors={k_neighbors})."
            )
        return X_resampled, y_resampled
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"    SMOTE failed for partition {partition_id}: {exc}. Proceeding without oversampling.")
        return X, y


# ============================================================================
# Data Loading and Setup
# ============================================================================

def load_partition_mapping(partition_map_path: str) -> pd.DataFrame:
    """Load cluster_mapping_k40_nc4.csv with FEWSNET_admin_code -> cluster_id mapping."""

    print(f"\nLoading partition mapping from {partition_map_path}")
    partition_df = pd.read_csv(partition_map_path)

    # Validate required columns
    required_cols = ['FEWSNET_admin_code', 'cluster_id']
    missing_cols = [col for col in required_cols if col not in partition_df.columns]
    if missing_cols:
        raise ValueError(f"Partition map missing required columns: {missing_cols}")

    # Keep only necessary columns
    partition_df = partition_df[['FEWSNET_admin_code', 'cluster_id']].copy()

    print(f"Loaded {len(partition_df)} admin units with {partition_df['cluster_id'].nunique()} partitions")
    print(f"Partition distribution:\n{partition_df['cluster_id'].value_counts().describe()}")

    return partition_df


def create_partition_group_array(df: pd.DataFrame, partition_df: pd.DataFrame) -> np.ndarray:
    """Create X_group array by mapping FEWSNET_admin_code to cluster_id."""

    print("\nMapping admin codes to partition IDs...")

    # Merge partition assignments into main dataframe
    df_with_partition = df.merge(
        partition_df,
        on='FEWSNET_admin_code',
        how='left'
    )

    # Check coverage
    unmapped = df_with_partition['cluster_id'].isna().sum()
    total = len(df_with_partition)
    pct_unmapped = 100 * unmapped / total

    print(f"Partition coverage: {total - unmapped}/{total} rows ({100 - pct_unmapped:.2f}%)")

    if pct_unmapped > 1.0:
        raise ValueError(
            f"Partition coverage insufficient: {pct_unmapped:.2f}% unmapped (threshold: 1%)\n"
            f"Ensure cluster_mapping_k40_nc4.csv covers all admin codes in the dataset."
        )

    # Fill unmapped with -1 (will be excluded from training)
    df_with_partition['cluster_id'].fillna(-1, inplace=True)

    # Convert to integer array
    X_group = df_with_partition['cluster_id'].astype(int).values

    print(f"Created X_group array: {len(np.unique(X_group))} unique partitions")

    return X_group, df_with_partition


# ============================================================================
# Model Training and Prediction
# ============================================================================

def train_pooled_model(X_train: np.ndarray, y_train: np.ndarray, lower_model: str = 'rf') -> Any:
    """Train a single pooled lower model on all training data."""

    print(f"  Training pooled {_model_label(lower_model)}: {len(y_train)} samples, {X_train.shape[1]} features")

    model = _create_model(lower_model)
    model.fit(X_train, y_train)

    return model


def train_partitioned_model(X_train: np.ndarray, y_train: np.ndarray,
                            X_group_train: np.ndarray,
                            lower_model: str = 'rf',
                            min_samples: int = MIN_PARTITION_SAMPLES) -> Dict[int, Any]:
    """Train separate lower models per partition (with pooled fallback for small partitions)."""

    unique_partitions = np.unique(X_group_train)
    unique_partitions = unique_partitions[unique_partitions >= 0]  # Exclude -1 (unmapped)

    print(f"  Training partitioned {_model_label(lower_model)}: {len(unique_partitions)} partitions")

    models = {}
    fallback_count = 0

    for partition_id in unique_partitions:
        partition_mask = X_group_train == partition_id
        X_partition = X_train[partition_mask]
        y_partition = y_train[partition_mask]

        if len(y_partition) >= min_samples:
            X_partition_balanced, y_partition_balanced = _apply_partition_smote(
                X_partition, y_partition, partition_id
            )
            model = _create_model(lower_model)
            model.fit(X_partition_balanced, y_partition_balanced)
            models[partition_id] = model
        else:
            models[partition_id] = None  # Will use pooled fallback
            fallback_count += 1

    if fallback_count > 0:
        print(f"    {fallback_count} partitions use pooled fallback (<{min_samples} samples)")

    return models


def predict_pooled(model: Any, X_test: np.ndarray) -> np.ndarray:
    """Predict using pooled model."""
    return model.predict(X_test)


def predict_partitioned(models: Dict[int, Any],
                       pooled_model: Any,
                       X_test: np.ndarray,
                       X_group_test: np.ndarray) -> np.ndarray:
    """Predict using partitioned models (with pooled fallback)."""

    y_pred = np.zeros(len(X_test), dtype=int)

    for partition_id, model in models.items():
        partition_mask = X_group_test == partition_id

        if partition_mask.sum() == 0:
            continue

        X_partition = X_test[partition_mask]

        if model is not None:
            # Use partition-specific model
            y_pred[partition_mask] = model.predict(X_partition)
        else:
            # Fallback to pooled model
            y_pred[partition_mask] = pooled_model.predict(X_partition)

    # Handle unmapped partitions (-1) with pooled model
    unmapped_mask = X_group_test == -1
    if unmapped_mask.sum() > 0:
        y_pred[unmapped_mask] = pooled_model.predict(X_test[unmapped_mask])

    return y_pred


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute binary classification metrics for class 1 (crisis).
    Uses same calculation as main GeoRF pipeline (get_prf from src.metrics.metrics).
    """
    # Use GeoRF pipeline's class-wise accuracy calculation
    true_class, total_class, pred_total = get_class_wise_accuracy(
        y_true, y_pred, prf=True
    )

    # Use GeoRF pipeline's precision/recall/F1 calculation
    # nan_option='mean' matches main pipeline default
    pre, rec, f1, _ = get_prf(true_class, total_class, pred_total, nan_option='mean')

    # Confusion matrix for additional metrics (keep existing logic)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Overall error rate
    pct_err_all = (y_true != y_pred).mean() if len(y_true) > 0 else 0.0

    # Class-1 error rate
    class1_mask = (y_true == 1)
    pct_err_class1 = (y_true[class1_mask] != y_pred[class1_mask]).mean() if class1_mask.sum() > 0 else 0.0

    return {
        'precision': pre[1],  # Class 1 only (crisis)
        'recall': rec[1],
        'f1': f1[1],
        'n': len(y_true),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'pct_err_all': pct_err_all,
        'pct_err_class1': pct_err_class1
    }


def compute_polygon_metrics(df: pd.DataFrame, uid_col: str = 'FEWSNET_admin_code') -> pd.DataFrame:
    """Compute polygon-level metrics from predictions."""

    polygon_list = []

    for polygon_id in sorted(df[uid_col].unique()):
        polygon_df = df[df[uid_col] == polygon_id]

        y_true = polygon_df['y_true'].values
        y_pred_pooled = polygon_df['y_pred_pooled'].values
        y_pred_partitioned = polygon_df['y_pred_partitioned'].values

        metrics_pooled = compute_binary_metrics(y_true, y_pred_pooled)
        metrics_partitioned = compute_binary_metrics(y_true, y_pred_partitioned)

        polygon_list.append({
            uid_col: polygon_id,
            'n': metrics_pooled['n'],
            'f1_pooled': metrics_pooled['f1'],
            'f1_partitioned': metrics_partitioned['f1'],
            'f1_diff': metrics_partitioned['f1'] - metrics_pooled['f1'],
            'precision_pooled': metrics_pooled['precision'],
            'precision_partitioned': metrics_partitioned['precision'],
            'recall_pooled': metrics_pooled['recall'],
            'recall_partitioned': metrics_partitioned['recall'],
            'pct_err_all_pooled': metrics_pooled['pct_err_all'],
            'pct_err_all_partitioned': metrics_partitioned['pct_err_all'],
            'pct_err_class1_pooled': metrics_pooled['pct_err_class1'],
            'pct_err_class1_partitioned': metrics_partitioned['pct_err_class1']
        })

    return pd.DataFrame(polygon_list)


# ============================================================================
# Visualization (Conditional, Exactly 4 Maps)
# ============================================================================

def create_visualizations(predictions_df: pd.DataFrame,
                         polygon_metrics_df: pd.DataFrame,
                         polygons_path: str,
                         out_dir: str,
                         visual: bool = False,
                         model_label: str = 'RF',
                         uid_col: str = 'FEWSNET_admin_code') -> None:
    """Create exactly 4 maps when VISUAL=True, else nothing."""

    if not visual:
        print("VISUAL=False, skipping all visualizations")
        return

    print("VISUAL=True, creating exactly 4 maps")

    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError as e:
        print(f"Visualization dependencies missing: {e}")
        print("Install with: pip install geopandas matplotlib")
        return

    vis_dir = Path(out_dir) / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load polygons
    print(f"Loading polygons from {polygons_path}")
    try:
        gdf = gpd.read_file(polygons_path)

        # Find admin code column (try multiple names)
        admin_col = None
        for col_name in ['admin_code', 'FEWSNET_admin_code', 'adm_code', 'area_id']:
            if col_name in gdf.columns:
                admin_col = col_name
                break

        if admin_col is None:
            raise ValueError("Shapefile must have admin_code, FEWSNET_admin_code, adm_code, or area_id column")

        # Rename to standard name for merging
        if admin_col != uid_col:
            gdf = gdf.rename(columns={admin_col: uid_col})

        gdf = gdf[[uid_col, 'geometry']].copy()

        # Ensure uid is string for merge
        gdf[uid_col] = gdf[uid_col].astype(str)
        polygon_metrics_df[uid_col] = polygon_metrics_df[uid_col].astype(str)

    except Exception as e:
        print(f"Failed to load polygons: {e}")
        return

    # Merge with polygon metrics
    merged = gdf.merge(polygon_metrics_df, on=uid_col, how='left')
    merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry')

    # Check merge success
    matched = merged['f1_pooled'].notna().sum()
    total_polygons = len(polygon_metrics_df)
    print(f"Matched {matched} / {total_polygons} polygons to geometries")

    # ========================================================================
    # Map 1: final_f1_performance_map.png (two panels: pooled vs partitioned)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Pooled F1
    merged_gdf.plot(column='f1_pooled', ax=axes[0], cmap='YlGnBu', vmin=0, vmax=1,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'F1 Score (Pooled)', 'shrink': 0.8})
    axes[0].set_title(f'Pooled {model_label} - F1 Score by Polygon', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')

    # Panel 2: Partitioned F1
    merged_gdf.plot(column='f1_partitioned', ax=axes[1], cmap='YlGnBu', vmin=0, vmax=1,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'F1 Score (Partitioned)', 'shrink': 0.8})
    axes[1].set_title(f'Partitioned {model_label} (k40_nc4) - F1 Score by Polygon', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    plt.tight_layout()
    out_path_1 = vis_dir / 'final_f1_performance_map.png'
    plt.savefig(out_path_1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_path_1}")
    plt.close()

    # ========================================================================
    # Map 2: map_pct_err_all.png (overall error rate, partitioned)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    merged_gdf.plot(column='pct_err_all_partitioned', ax=ax, cmap='Reds', vmin=0, vmax=1,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'Overall Error Rate', 'shrink': 0.8})
    ax.set_title(f'Partitioned {model_label} - Overall Error Rate by Polygon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    out_path_2 = vis_dir / 'map_pct_err_all.png'
    plt.savefig(out_path_2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_path_2}")
    plt.close()

    # ========================================================================
    # Map 3: map_pct_err_class1.png (class-1 error rate, partitioned)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    merged_gdf.plot(column='pct_err_class1_partitioned', ax=ax, cmap='Reds', vmin=0, vmax=1,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'Class-1 Error Rate', 'shrink': 0.8})
    ax.set_title(f'Partitioned {model_label} - Class-1 (Crisis) Error Rate by Polygon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    out_path_3 = vis_dir / 'map_pct_err_class1.png'
    plt.savefig(out_path_3, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_path_3}")
    plt.close()

    # ========================================================================
    # Map 4: overall_f1_improvement_map.png (F1 delta: partitioned - pooled)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    # Center colormap at zero for diverging colors
    vmin_diff = merged_gdf['f1_diff'].min()
    vmax_diff = merged_gdf['f1_diff'].max()
    abs_max = max(abs(vmin_diff), abs(vmax_diff))
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    merged_gdf.plot(column='f1_diff', ax=ax, cmap='RdBu', norm=norm,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'F1 Improvement (Partitioned - Pooled)', 'shrink': 0.8})
    ax.set_title('F1 Score Improvement: Partitioned vs Pooled by Polygon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    out_path_4 = vis_dir / 'overall_f1_improvement_map.png'
    plt.savefig(out_path_4, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_path_4}")
    plt.close()

    print("Exactly 4 maps created successfully")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Partitioned vs Pooled k40_nc4 Comparison')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, help='Path to panel dataset CSV')
    parser.add_argument('--partition-map', default=DEFAULT_PARTITION_MAP, help='Path to cluster_mapping CSV')
    parser.add_argument('--polygons', default=DEFAULT_POLYGONS_PATH, help='Path to admin boundaries shapefile')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory')
    parser.add_argument('--start-month', default=DEFAULT_START_MONTH, help='Start month (YYYY-MM)')
    parser.add_argument('--end-month', default=DEFAULT_END_MONTH, help='End month (YYYY-MM)')
    parser.add_argument('--train-window', type=int, default=DEFAULT_TRAIN_WINDOW, help='Training window (months)')
    parser.add_argument('--forecasting-scope', type=int, default=DEFAULT_FORECASTING_SCOPE,
                       choices=[1, 2, 3], help='Forecasting scope: 1=4mo, 2=8mo, 3=12mo lag')
    parser.add_argument('--visual', action='store_true', help='Enable visualization (creates exactly 4 maps)')
    parser.add_argument('--month-ind', action='store_true', help='Enable month-specific partitions')
    parser.add_argument('--partition-map-m2', default='cluster_mapping_k40_nc10_m2.csv', help='Partition map for February')
    parser.add_argument('--partition-map-m6', default='cluster_mapping_k40_nc2_m6.csv', help='Partition map for June')
    parser.add_argument('--partition-map-m10', default='cluster_mapping_k40_nc12_m10.csv', help='Partition map for October')
    parser.add_argument('--lower-model', choices=['rf', 'dt'], default='rf',
                        help='Lower-layer base model: rf (default) or dt')

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_label = _model_label(args.lower_model)

    # Compute active lag from forecasting scope
    active_lag = forecasting_scope_to_lag(args.forecasting_scope, LAGS_MONTHS)

    print("=" * 80)
    print(f"PARTITIONED (k40_nc4) VS POOLED {model_label} COMPARISON")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Partition map: {args.partition_map}")
    if args.month_ind:
        print(f"Month-specific partitions: ENABLED")
        print(f"  - Feb (m2): {args.partition_map_m2}")
        print(f"  - Jun (m6): {args.partition_map_m6}")
        print(f"  - Oct (m10): {args.partition_map_m10}")
    else:
        print(f"Month-specific partitions: DISABLED")
    print(f"Evaluation period: {args.start_month} to {args.end_month}")
    print(f"Train window: {args.train_window} months")
    print(f"Forecasting scope: {args.forecasting_scope} ({active_lag}-month lag)")
    print(f"Lower model: {model_label}")
    print(f"Visualization: {'ENABLED (4 maps)' if args.visual else 'DISABLED'}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    # ========================================================================
    # Step 1: Load and preprocess data using GeoRF pipeline
    # ========================================================================
    print("\n[Step 1/6] Loading and preprocessing data...")
    df = load_and_preprocess_data(args.data)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ========================================================================
    # Step 2: Load partition mapping and create X_group (only if not month-specific)
    # ========================================================================
    if not args.month_ind:
        print("\n[Step 2/6] Setting up partition groups...")
        partition_df = load_partition_mapping(args.partition_map)
        X_group, df_with_partition = create_partition_group_array(df, partition_df)
    else:
        print("\n[Step 2/6] Month-specific partitions enabled - will load per test month...")
        X_group = None  # Will be loaded per month
        df_with_partition = None

    # Create X_loc (latitude, longitude) - required by prepare_features
    if 'latitude' in df.columns and 'longitude' in df.columns:
        X_loc = df[['latitude', 'longitude']].values
    elif 'lat' in df.columns and 'lon' in df.columns:
        X_loc = df[['lat', 'lon']].values
    else:
        raise ValueError("Dataset must have latitude/longitude or lat/lon columns")

    print(f"Location matrix created: {X_loc.shape}")

    # ========================================================================
    # Step 3: Prepare features using GeoRF pipeline (creates lagged features!)
    # ========================================================================
    print(f"\n[Step 3/6] Preparing features with forecasting_scope={args.forecasting_scope}...")

    # Use temporary X_group for feature preparation if month-specific mode
    temp_X_group = X_group if X_group is not None else np.zeros(len(df), dtype=int)

    X, y, l1_index, l2_index, years, terms, dates, feature_columns = prepare_features(
        df, temp_X_group, X_loc, forecasting_scope=args.forecasting_scope
    )

    print(f"[OK] Features prepared: {X.shape}")
    print(f"  - Total features: {X.shape[1]}")
    print(f"  - L1 features (time-invariant): {len(l1_index)}")
    print(f"  - L2 features (time-variant + lagged): {len(l2_index)}")
    print(f"  - Target vector: {y.shape}")
    print(f"  - Active lag: {active_lag} months (features from t-{active_lag} used to predict t)")

    # Extract admin codes for prediction tracking
    admin_codes = df['FEWSNET_admin_code'].values
    # Note: partition_ids not needed here if month-specific mode (will be loaded per month)

    # ========================================================================
    # Step 4: Monthly evaluation loop using GeoRF train/test split
    # ========================================================================
    print(f"\n[Step 4/6] Evaluating monthly from {args.start_month} to {args.end_month}...")

    start_month = pd.Period(args.start_month, freq='M')
    end_month = pd.Period(args.end_month, freq='M')

    # Generate test months
    test_months = []
    current = start_month
    while current <= end_month:
        test_months.append(current)
        current += 1

    print(f"Test months: {len(test_months)} ({test_months[0]} to {test_months[-1]})")
    print("=" * 80)

    monthly_metrics = []
    all_predictions = []

    for i, test_month in enumerate(test_months, 1):
        print(f"\n[{i}/{len(test_months)}] Test month: {test_month}")
        print("-" * 80)

        # Load month-specific partition if MONTH_IND is enabled
        if args.month_ind:
            test_month_num = test_month.month

            if test_month_num == 2:
                current_partition_map = args.partition_map_m2
                print(f"  Using February-specific partition: {current_partition_map}")
            elif test_month_num == 6:
                current_partition_map = args.partition_map_m6
                print(f"  Using June-specific partition: {current_partition_map}")
            elif test_month_num == 10:
                current_partition_map = args.partition_map_m10
                print(f"  Using October-specific partition: {current_partition_map}")
            else:
                current_partition_map = args.partition_map
                print(f"  Using default partition: {current_partition_map}")

            # Reload partition mapping for this month
            partition_df = load_partition_mapping(current_partition_map)
            X_group, _ = create_partition_group_array(df, partition_df)

        try:
            # Baseline split: feed dummy groups so pooled model keeps full window
            baseline_groups = np.zeros_like(X_group)
            baseline_split = train_test_split_rolling_window(
                X, y, X_loc, baseline_groups, years, dates,
                test_month=test_month,
                active_lag=active_lag,
                train_window_months=args.train_window,
                admin_codes=admin_codes
            )

            if len(baseline_split) == 10:
                Xtrain_pooled, ytrain_pooled, _, _, Xtest, ytest, _, _, _, admin_codes_test = baseline_split
            else:
                Xtrain_pooled, ytrain_pooled, _, _, Xtest, ytest, _, _ = baseline_split
                admin_codes_test = None

            ytrain_pooled = ytrain_pooled.astype(int)
            ytest = ytest.astype(int)

            # Partitioned split: use real groups so filtering matches production logic
            partition_split = train_test_split_rolling_window(
                X, y, X_loc, X_group, years, dates,
                test_month=test_month,
                active_lag=active_lag,
                train_window_months=args.train_window,
                admin_codes=admin_codes
            )

            if len(partition_split) == 10:
                Xtrain_partitioned, ytrain_partitioned, _, Xtrain_group_partitioned, _, _, _, Xtest_group, _, _ = partition_split
            else:
                Xtrain_partitioned, ytrain_partitioned, _, Xtrain_group_partitioned, _, _, _, Xtest_group = partition_split

            ytrain_partitioned = ytrain_partitioned.astype(int)

            print(f"  Train (pooled): {len(ytrain_pooled)} samples | Train (partitioned): {len(ytrain_partitioned)} samples | Test: {len(ytest)} samples")

            if len(ytest) == 0:
                print("  Skipping: no test samples")
                continue

        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Train pooled model (full window)
        pooled_model = train_pooled_model(Xtrain_pooled, ytrain_pooled, args.lower_model)

        # Train partitioned models (filtered window)
        partitioned_models = train_partitioned_model(
            Xtrain_partitioned, ytrain_partitioned, Xtrain_group_partitioned, args.lower_model
        )

        # Predict
        y_pred_pooled = predict_pooled(pooled_model, Xtest)
        y_pred_partitioned = predict_partitioned(partitioned_models, pooled_model, Xtest, Xtest_group)

        # Compute metrics
        metrics_pooled = compute_binary_metrics(ytest, y_pred_pooled)
        metrics_partitioned = compute_binary_metrics(ytest, y_pred_partitioned)

        monthly_metrics.append({
            'test_month': str(test_month),
            'model': 'pooled',
            **metrics_pooled
        })

        monthly_metrics.append({
            'test_month': str(test_month),
            'model': 'partitioned',
            **metrics_partitioned
        })

        # Store predictions with admin codes and partition IDs
        if admin_codes_test is not None:
            pred_df = pd.DataFrame({
                'FEWSNET_admin_code': admin_codes_test,
                'month_start': test_month.to_timestamp(),
                'partition_id': Xtest_group,
                'y_true': ytest,
                'y_pred_pooled': y_pred_pooled,
                'y_pred_partitioned': y_pred_partitioned
            })
            all_predictions.append(pred_df)

        print(f"  Pooled:      Precision={metrics_pooled['precision']:.4f}, Recall={metrics_pooled['recall']:.4f}, F1={metrics_pooled['f1']:.4f}")
        print(f"  Partitioned: Precision={metrics_partitioned['precision']:.4f}, Recall={metrics_partitioned['recall']:.4f}, F1={metrics_partitioned['f1']:.4f}")

    print("\n" + "=" * 80)
    print("MONTHLY EVALUATION COMPLETE")
    print("=" * 80)

    # ========================================================================
    # Step 5: Save results
    # ========================================================================
    print("\n[Step 5/6] Saving results...")

    # 1. Monthly metrics
    metrics_df = pd.DataFrame(monthly_metrics)
    metrics_path = out_dir / 'metrics_monthly.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    # 2. Predictions
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = out_dir / 'predictions_monthly.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"  Saved: {predictions_path}")

        # 3. Polygon-level metrics
        polygon_metrics_df = compute_polygon_metrics(predictions_df)
        polygon_path = out_dir / 'metrics_polygon_overall.csv'
        polygon_metrics_df.to_csv(polygon_path, index=False)
        print(f"  Saved: {polygon_path}")
    else:
        print("  Warning: No predictions to save")
        predictions_df = pd.DataFrame()
        polygon_metrics_df = pd.DataFrame()

    # 4. Run manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'data_path': args.data,
        'partition_map_path': args.partition_map,
        'start_month': args.start_month,
        'end_month': args.end_month,
        'train_window_months': args.train_window,
        'forecasting_scope': args.forecasting_scope,
        'active_lag_months': active_lag,
        'n_test_months': len(test_months),
        'n_test_months_evaluated': len(monthly_metrics) // 2,
        'n_predictions': len(predictions_df) if not predictions_df.empty else 0,
        'n_polygons': int(df['FEWSNET_admin_code'].nunique()),
        'n_partitions': int(partition_df['cluster_id'].nunique()),
        'rf_params': RF_PARAMS if args.lower_model == 'rf' else DT_PARAMS,
        'model_type': model_label,
        'random_state': RANDOM_STATE,
        'visual_enabled': args.visual,
        'pipeline_version': 'GeoRF_utilities_v1.0'
    }

    manifest_path = out_dir / 'run_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    # ========================================================================
    # Step 6: Visualizations (conditional)
    # ========================================================================
    print(f"\n[Step 6/6] Visualization...")
    if not predictions_df.empty and not polygon_metrics_df.empty:
        create_visualizations(
            predictions_df, polygon_metrics_df, args.polygons, args.out_dir, args.visual, model_label=model_label
        )
    else:
        print("  Skipped: No predictions available for visualization")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {out_dir}")
    print(f"Test months evaluated: {manifest['n_test_months_evaluated']} / {manifest['n_test_months']}")
    print(f"Total predictions: {manifest['n_predictions']}")
    print(f"Polygons: {manifest['n_polygons']}")
    print(f"Partitions: {manifest['n_partitions']}")

    # Summary statistics
    if not metrics_df.empty:
        pooled_metrics = metrics_df[metrics_df['model'] == 'pooled']
        partitioned_metrics = metrics_df[metrics_df['model'] == 'partitioned']

        print(f"\nOverall Performance (Mean +- Std):")
        print(f"  Partitioned F1: {partitioned_metrics['f1'].mean():.4f} +- {partitioned_metrics['f1'].std():.4f}")
        print(f"  Pooled F1:      {pooled_metrics['f1'].mean():.4f} +- {pooled_metrics['f1'].std():.4f}")
        f1_diff = partitioned_metrics['f1'].values - pooled_metrics['f1'].values
        print(f"  F1 Improvement: {f1_diff.mean():.4f} +- {f1_diff.std():.4f}")
    else:
        print("No monthly metrics available for summary.")

    if args.visual:
        print(f"\nVisualizations: 4 maps created in {out_dir / 'vis'}")
    else:
        print(f"\nVisualizations: DISABLED (use --visual to enable)")

    print("\nDone.")


if __name__ == '__main__':
    main()
