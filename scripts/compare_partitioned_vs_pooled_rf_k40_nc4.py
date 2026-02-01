#!/usr/bin/env python3
"""
Partitioned (k40_nc4) RF Evaluation Script

Evaluates monthly from 2021-01 through 2024-12:
- Partitioned model: Separate RF per partition (cluster_id from cluster_mapping_k40_nc4.csv)

Uses GeoRF pipeline utilities for proper feature preparation and temporal splitting.

Outputs:
- metrics_monthly.csv: Monthly precision/recall/F1 for partitioned model
- predictions_monthly.csv: All predictions with uid/time/partition/y_true/y_pred
- metrics_polygon_overall.csv: Polygon-level aggregated metrics
- run_manifest.json: Run parameters and diagnostics

Visualization (VISUAL=True only):
- map_pct_err_all.png: Overall error rate by polygon
- map_pct_err_class1.png: Class-1 specific error rate by polygon

Author: Claude Code
Date: 2026-01-25
Updated: 2026-02-01 (Removed pooled baseline)
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

# RF hyperparameters (MUST match main GeoRF pipeline)
# See src/model/GeoRF.py line 51 for main pipeline defaults
RF_PARAMS = {
    'n_estimators': 100,  # Main pipeline uses n_trees_unit=100 (NOT 500)
    'max_depth': None,     # Main pipeline uses max_depth=None (unlimited tree depth)
    'random_state': RANDOM_STATE,  # Main pipeline uses random_state=5 (NOT 42)
    'n_jobs': -1
    # NOTE: class_weight is NOT set in main pipeline (defaults to None, NOT 'balanced')
    # NOTE: min_samples_leaf defaults to 1 (not explicitly set in main pipeline)
}

# Minimum samples per partition to train separate model
MIN_PARTITION_SAMPLES = 50


# ============================================================================
# Data Loading and Setup
# ============================================================================

def load_partition_mapping(partition_map_path: str) -> pd.DataFrame:
    """Load cluster_mapping_k40_nc4.csv with FEWSNET_admin_code → cluster_id mapping."""

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

def train_partitioned_rf(X_train: np.ndarray, y_train: np.ndarray,
                        X_group_train: np.ndarray,
                        min_samples: int = MIN_PARTITION_SAMPLES) -> Dict[int, RandomForestClassifier]:
    """Train separate RF per partition."""

    unique_partitions = np.unique(X_group_train)
    unique_partitions = unique_partitions[unique_partitions >= 0]  # Exclude -1 (unmapped)

    print(f"  Training partitioned RF: {len(unique_partitions)} partitions")

    # Train pooled model for fallback (small partitions and unmapped regions)
    print(f"  Training fallback RF for small partitions: {len(y_train)} samples")
    fallback_model = RandomForestClassifier(**RF_PARAMS)
    fallback_model.fit(X_train, y_train)

    models = {}
    fallback_count = 0

    for partition_id in unique_partitions:
        partition_mask = X_group_train == partition_id
        X_partition = X_train[partition_mask]
        y_partition = y_train[partition_mask]

        if len(y_partition) >= min_samples:
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_partition, y_partition)
            models[partition_id] = model
        else:
            models[partition_id] = None  # Will use fallback
            fallback_count += 1

    if fallback_count > 0:
        print(f"    {fallback_count} partitions use fallback (<{min_samples} samples)")

    # Store fallback model in dictionary with key -999
    models[-999] = fallback_model

    return models


def predict_partitioned(models: Dict[int, RandomForestClassifier],
                       X_test: np.ndarray,
                       X_group_test: np.ndarray) -> np.ndarray:
    """Predict using partitioned models (with fallback for small partitions and unseen partitions)."""

    fallback_model = models[-999]  # Get fallback model

    # FIX: Start with fallback predictions for ALL samples (ensures full coverage)
    y_pred = fallback_model.predict(X_test)

    # Override with partition-specific predictions where available
    for partition_id, model in models.items():
        if partition_id == -999:  # Skip fallback model entry
            continue

        partition_mask = X_group_test == partition_id

        if partition_mask.sum() == 0:
            continue

        if model is not None:
            # Use partition-specific model (override fallback)
            y_pred[partition_mask] = model.predict(X_test[partition_mask])
        # else: keep fallback prediction (already set)

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
    """Compute polygon-level metrics from predictions (partitioned-only)."""

    polygon_list = []

    for polygon_id in sorted(df[uid_col].unique()):
        polygon_df = df[df[uid_col] == polygon_id]

        y_true = polygon_df['y_true'].values
        y_pred_partitioned = polygon_df['y_pred_partitioned'].values

        metrics = compute_binary_metrics(y_true, y_pred_partitioned)

        polygon_list.append({
            uid_col: polygon_id,
            'n': metrics['n'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'pct_err_all': metrics['pct_err_all'],
            'pct_err_class1': metrics['pct_err_class1']
        })

    return pd.DataFrame(polygon_list)


# ============================================================================
# Visualization (Conditional, Partitioned-Only Maps)
# ============================================================================

def create_visualizations(predictions_df: pd.DataFrame,
                         polygon_metrics_df: pd.DataFrame,
                         polygons_path: str,
                         out_dir: str,
                         visual: bool = False,
                         uid_col: str = 'FEWSNET_admin_code') -> None:
    """Create 2 partitioned-only maps when VISUAL=True, else nothing."""

    if not visual:
        print("VISUAL=False, skipping all visualizations")
        return

    print("VISUAL=True, creating 2 partitioned-only maps")

    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
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
    matched = merged['f1'].notna().sum()
    total_polygons = len(polygon_metrics_df)
    print(f"Matched {matched} / {total_polygons} polygons to geometries")

    # ========================================================================
    # Map 1: map_pct_err_all.png (overall error rate, partitioned)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    merged_gdf.plot(column='pct_err_all', ax=ax, cmap='Reds', vmin=0, vmax=1,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'Overall Error Rate', 'shrink': 0.8})
    ax.set_title('Partitioned RF - Overall Error Rate by Polygon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    out_path_1 = vis_dir / 'map_pct_err_all.png'
    plt.savefig(out_path_1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_path_1}")
    plt.close()

    # ========================================================================
    # Map 2: map_pct_err_class1.png (class-1 error rate, partitioned)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    merged_gdf.plot(column='pct_err_class1', ax=ax, cmap='Reds', vmin=0, vmax=1,
                    legend=True, edgecolor='black', linewidth=0.2, missing_kwds={'color': 'lightgray'},
                    legend_kwds={'label': 'Class-1 Error Rate', 'shrink': 0.8})
    ax.set_title('Partitioned RF - Class-1 (Crisis) Error Rate by Polygon', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    out_path_2 = vis_dir / 'map_pct_err_class1.png'
    plt.savefig(out_path_2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out_path_2}")
    plt.close()

    print("2 partitioned-only maps created successfully")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Partitioned RF Evaluation')
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, help='Path to panel dataset CSV')
    parser.add_argument('--partition-map', default=DEFAULT_PARTITION_MAP, help='Path to cluster_mapping CSV')
    parser.add_argument('--polygons', default=DEFAULT_POLYGONS_PATH, help='Path to admin boundaries shapefile')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory')
    parser.add_argument('--start-month', default=DEFAULT_START_MONTH, help='Start month (YYYY-MM)')
    parser.add_argument('--end-month', default=DEFAULT_END_MONTH, help='End month (YYYY-MM)')
    parser.add_argument('--train-window', type=int, default=DEFAULT_TRAIN_WINDOW, help='Training window (months)')
    parser.add_argument('--forecasting-scope', type=int, default=DEFAULT_FORECASTING_SCOPE,
                       choices=[1, 2, 3], help='Forecasting scope: 1=4mo, 2=8mo, 3=12mo lag')
    parser.add_argument('--visual', action='store_true', help='Enable visualization (creates 2 maps)')
    parser.add_argument('--month-ind', action='store_true', help='Enable month-specific partitions')
    parser.add_argument('--partition-map-m2', default='cluster_mapping_k40_nc10_m2.csv', help='Partition map for February')
    parser.add_argument('--partition-map-m6', default='cluster_mapping_k40_nc2_m6.csv', help='Partition map for June')
    parser.add_argument('--partition-map-m10', default='cluster_mapping_k40_nc12_m10.csv', help='Partition map for October')

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute active lag from forecasting scope
    active_lag = forecasting_scope_to_lag(args.forecasting_scope, LAGS_MONTHS)

    print("=" * 80)
    print("PARTITIONED (k40_nc4) RF EVALUATION - POOLED BASELINE DISABLED")
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
    print(f"Visualization: {'ENABLED (2 maps)' if args.visual else 'DISABLED'}")
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

    print(f"✓ Features prepared: {X.shape}")
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
            # Use GeoRF's train_test_split_rolling_window with monthly mode
            split_result = train_test_split_rolling_window(
                X, y, X_loc, X_group, years, dates,
                test_month=test_month,
                active_lag=active_lag,
                train_window_months=args.train_window,
                admin_codes=admin_codes
            )

            # Unpack results (admin_codes included)
            if len(split_result) == 10:
                Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group, admin_codes_train, admin_codes_test = split_result
            else:
                Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group = split_result
                admin_codes_test = None

            ytrain = ytrain.astype(int)
            ytest = ytest.astype(int)

            print(f"  Train: {len(ytrain)} samples, Test: {len(ytest)} samples")

            if len(ytest) == 0:
                print(f"  Skipping: no test samples")
                continue

        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Train partitioned models (includes fallback model)
        partitioned_models = train_partitioned_rf(Xtrain, ytrain, Xtrain_group)

        # Predict using partitioned models
        y_pred_partitioned = predict_partitioned(partitioned_models, Xtest, Xtest_group)

        # Compute metrics
        metrics_partitioned = compute_binary_metrics(ytest, y_pred_partitioned)

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
                'y_pred_partitioned': y_pred_partitioned
            })
            all_predictions.append(pred_df)

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
        'n_test_months_evaluated': len(monthly_metrics),
        'n_predictions': len(predictions_df) if not predictions_df.empty else 0,
        'n_polygons': int(df['FEWSNET_admin_code'].nunique()),
        'n_partitions': int(partition_df['cluster_id'].nunique()),
        'rf_params': RF_PARAMS,
        'random_state': RANDOM_STATE,
        'visual_enabled': args.visual,
        'pipeline_version': 'GeoRF_utilities_v1.0',
        'model_type': 'partitioned_only'
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
        create_visualizations(predictions_df, polygon_metrics_df, args.polygons, args.out_dir, args.visual)
    else:
        print("  Skipped: No predictions available for visualization")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {out_dir}")
    print(f"Test months evaluated: {manifest['n_test_months_evaluated']} / {manifest['n_test_months']}")
    print(f"Total predictions: {manifest['n_predictions']}")
    print(f"Polygons: {manifest['n_polygons']}")
    print(f"Partitions: {manifest['n_partitions']}")

    # Summary statistics
    if not metrics_df.empty:
        partitioned_metrics = metrics_df[metrics_df['model'] == 'partitioned']

        print(f"\nOverall Performance (Mean ± Std):")
        print(f"  Partitioned F1: {partitioned_metrics['f1'].mean():.4f} ± {partitioned_metrics['f1'].std():.4f}")

    if args.visual:
        print(f"\nVisualizations: 2 maps created in {out_dir / 'vis'}")
    else:
        print(f"\nVisualizations: DISABLED (use --visual to enable)")

    print("\nDone.")


if __name__ == '__main__':
    main()
