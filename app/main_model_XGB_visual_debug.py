#!/usr/bin/env python3
"""
Replicated and debugged version of main_model_GF_main.ipynb - XGBoost Version

This script replicates the full functionality of the notebook for food crisis prediction
using GeoXGB (XGBoost-based GeoRF) with polygon-based contiguity support.

Key features:
1. Data preprocessing with polars and pandas
2. Multiple spatial grouping options (polygons, grid, country, AEZ, etc.)
3. Polygon-based contiguity with corrected setup
4. Time-based train-test splitting for temporal validation
5. Single-layer and 2-layer GeoXGB models (XGBoost instead of Random Forest)
6. Comprehensive evaluation and result saving
7. XGBoost-specific hyperparameters optimized for food crisis prediction

Date: 2025-08-12
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import sys
import gc
import glob
import warnings
import argparse

# Add parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# GeoXGB imports - CHANGED from GeoRF to GeoRF_XGB
from src.model.GeoRF_XGB import GeoRF_XGB
from src.customize.customize import *
from demo.data import load_demo_data
from src.helper.helper import get_spatial_range
from src.initialization.initialization import train_test_split_all
from src.customize.customize import train_test_split_rolling_window
from config_visual import *
from config import DEFAULT_LAGS_MONTHS
from src.utils.lag_schedules import (
    forecasting_scope_to_lag,
    log_lag_schedule,
    resolve_lag_schedule,
)
# Note: No imputation imports needed for XGBoost - handles missing values natively
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Import adjacency matrix utilities
if USE_ADJACENCY_MATRIX:
    from src.adjacency.adjacency_utils import load_or_create_adjacency_matrix

# Import redirected functions from src modules
from src.preprocess.preprocess import load_and_preprocess_data, setup_spatial_groups, handle_infinite_values
from src.feature.feature import prepare_features, validate_polygon_contiguity, create_correspondence_table
from src.utils.save_results import save_results

VIS_DEBUG_MODE = True
ARTIFACTS_ROOT = os.path.join('result_GeoRF')
ACTIVE_LAGS = resolve_lag_schedule(LAGS_MONTHS, context="config_visual.LAGS_MONTHS")
EXPECTED_LAG_SCHEDULE = list(DEFAULT_LAGS_MONTHS)
if ACTIVE_LAGS != EXPECTED_LAG_SCHEDULE:
    raise ValueError(
        f"GeoXGB visual debug requires FEWSNET-aligned lags {EXPECTED_LAG_SCHEDULE}; found {ACTIVE_LAGS}."
    )

# Configuration


DATA_MODE = 'unadjusted'  # Options: 'full', 'noconflict', 'nofoodprice', 'nomacro', 'nogis', 'min', 'unadjusted'


if DATA_MODE == 'full':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_v06252025.csv"
elif DATA_MODE == 'noconflict':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\noconf.csv"
elif DATA_MODE == 'nofoodprice':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\nofoodprice.csv"
elif DATA_MODE == 'nomacro':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\nomacro.csv"
elif DATA_MODE == 'nogis':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\nogis.csv"
elif DATA_MODE == 'min':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\minimal_viable_df.csv"
elif DATA_MODE == 'unadjusted':
    DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_unadjusted_bm.csv"
else:
    raise ValueError(f"Invalid DATA_MODE: {DATA_MODE}")


def run_temporal_evaluation(X, y, X_loc, X_group, years, dates, l1_index, l2_index, feature_columns,
                           assignment, contiguity_info, df, nowcasting=False, max_depth=None, input_terms=None, desire_terms=None,
                           track_partition_metrics=False, enable_metrics_maps=False, start_year=2015, end_year=2024, forecasting_scope=None, force_final_accuracy=False,
                           # XGBoost-specific hyperparameters
                           learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8):
    """
    Run temporal evaluation for all quarters from start_year to end_year using rolling window approach.
    Uses GeoXGB (XGBoost-based GeoRF) instead of Random Forest.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    X_loc : numpy.ndarray
        Location coordinates
    X_group : numpy.ndarray
        Group assignments
    years : numpy.ndarray
        Year values
    dates : pandas.Series or array-like
        Date values for precise temporal splitting
    l1_index : list
        L1 feature indices
    l2_index : list
        L2 feature indices
    assignment : str
        Spatial assignment method
    contiguity_info : dict or None
        Contiguity information
    df : pandas.DataFrame
        Original dataframe with FEWSNET_admin_code
    nowcasting : bool
        Whether to use 2-layer model
    max_depth : int or None
        Maximum depth for XGB models
    input_terms : numpy.ndarray
        Terms within each year (1-4 corresponding to quarters)
    desire_terms : int or None
        Specific quarter to evaluate (1-4), or None for all quarters
    track_partition_metrics : bool
        Whether to enable partition metrics tracking and visualization
    enable_metrics_maps : bool
        Whether to create maps showing F1/accuracy improvements
    forecasting_scope : int or None
        Forecasting scope (1-based index into active lag schedule)
    learning_rate : float
        XGBoost learning rate
    reg_alpha : float
        XGBoost L1 regularization
    reg_lambda : float
        XGBoost L2 regularization
    subsample : float
        XGBoost subsample ratio
    colsample_bytree : float
        XGBoost column subsample ratio
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Evaluation results with quarter information
    y_pred_test : pandas.DataFrame
        Prediction results with quarter information
    """
    print(f"Running temporal evaluation using GeoXGB (nowcasting={nowcasting})...")
    print(f"XGBoost hyperparameters: learning_rate={learning_rate}, reg_alpha={reg_alpha}, reg_lambda={reg_lambda}")
    print(f"                        subsample={subsample}, colsample_bytree={colsample_bytree}")
    
    # Initialize results tracking (class 1 only)
    results_df = pd.DataFrame(columns=[
        'year', 'quarter', 'precision(1)', 'recall(1)', 'f1(1)',
        'precision_base(1)', 'recall_base(1)', 'f1_base(1)',
        'num_samples(1)'
    ])
    
    y_pred_test = pd.DataFrame(columns=['year', 'quarter', 'month', 'adm_code', 'fews_ipc_crisis_pred', 'fews_ipc_crisis_true'])
    
    # Setup correspondence table path for partition metrics tracking
    correspondence_table_path = None
    if track_partition_metrics:
        print("Setting up partition metrics tracking...")
        
        # Create base correspondence table from the data
        correspondence_table_path = 'correspondence_table_metrics.csv'
        try:
            # Create correspondence table that properly maps X_group to FEWSNET_admin_code
            if not os.path.exists(correspondence_table_path):
                print("Creating correspondence table for partition metrics...")
                
                # Create proper correspondence table based on assignment method
                if assignment == 'polygons':
                    # For polygon assignment, X_group contains FEWSNET_admin_code values directly
                    # Create a direct mapping, but we need to verify the relationship
                    print(f"Debug: Creating polygon correspondence table")
                    print(f"X_group type: {type(X_group[0])}, sample values: {X_group[:5]}")
                    print(f"FEWSNET_admin_code sample values: {df['FEWSNET_admin_code'].head().tolist()}")
                    
                    # For polygon assignment, X_group should already BE the admin codes
                    # So we create a simple identity mapping
                    unique_admin_codes = df['FEWSNET_admin_code'].dropna().unique()
                    unique_groups = np.unique(X_group)
                    
                    print(f"Unique admin codes count: {len(unique_admin_codes)}")
                    print(f"Unique X_group values count: {len(unique_groups)}")
                    
                    # Check if X_group values are actually admin codes
                    admin_codes_set = set(unique_admin_codes)
                    group_codes_set = set(unique_groups)
                    
                    if admin_codes_set == group_codes_set:
                        print("X_group contains admin codes directly - creating identity mapping")
                        corr_df = pd.DataFrame({
                            'FEWSNET_admin_code': unique_admin_codes,
                            'X_group': unique_admin_codes
                        })
                    else:
                        print("X_group doesn't match admin codes - creating index-based mapping")
                        # Create mapping based on actual data relationships
                        unique_entries = []
                        for i in range(len(df)):
                            admin_code = df.iloc[i]['FEWSNET_admin_code']
                            group_id = X_group[i]
                            if not pd.isna(admin_code):
                                unique_entries.append({
                                    'FEWSNET_admin_code': admin_code,
                                    'X_group': group_id
                                })
                        
                        # Remove duplicates and create DataFrame
                        corr_df = pd.DataFrame(unique_entries).drop_duplicates()
                
                elif assignment in ['country', 'AEZ', 'country_AEZ']:
                    # For these assignments, create mapping from admin codes to group IDs
                    mapping_data = []
                    for i in range(len(df)):
                        admin_code = df.iloc[i]['FEWSNET_admin_code']
                        group_id = X_group[i]
                        if not pd.isna(admin_code):
                            mapping_data.append({
                                'FEWSNET_admin_code': admin_code,
                                'X_group': group_id
                            })
                    
                    corr_df = pd.DataFrame(mapping_data).drop_duplicates()
                
                elif assignment in ['geokmeans', 'all_kmeans']:
                    # For kmeans assignments, use the existing correspondence table if available
                    kmeans_table_path = f'correspondence_table_{assignment.replace("_", "")}.csv'
                    if os.path.exists(kmeans_table_path):
                        print(f"Using existing kmeans correspondence table: {kmeans_table_path}")
                        correspondence_table_path = kmeans_table_path
                        corr_df = None  # Don't create new table
                    else:
                        print(f"Warning: Expected kmeans correspondence table not found: {kmeans_table_path}")
                        correspondence_table_path = None
                        corr_df = None
                
                else:
                    # For grid assignment, create a simple mapping
                    mapping_data = []
                    for i in range(len(df)):
                        admin_code = df.iloc[i]['FEWSNET_admin_code']
                        group_id = X_group[i]
                        if not pd.isna(admin_code):
                            mapping_data.append({
                                'FEWSNET_admin_code': admin_code,
                                'X_group': group_id
                            })
                    
                    corr_df = pd.DataFrame(mapping_data).drop_duplicates()
                
                # Save the correspondence table if we created one
                if corr_df is not None and len(corr_df) > 0:
                    corr_df.to_csv(correspondence_table_path, index=False)
                    print(f"Created correspondence table with {len(corr_df)} entries: {correspondence_table_path}")
                    print(f"Sample entries:")
                    print(corr_df.head())
                else:
                    if corr_df is not None:  # Empty DataFrame
                        print("Warning: Could not create correspondence table - no valid data found")
                        correspondence_table_path = None
                    # else: using existing kmeans table, so correspondence_table_path is already set
                        
            else:
                print(f"Using existing correspondence table: {correspondence_table_path}")
                
        except Exception as e:
            print(f"Warning: Could not create correspondence table: {e}")
            print("Maps will not be generated, but CSV metrics will still be saved.")
            correspondence_table_path = None
    
    # Determine contiguity settings
    if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans']:
        contiguity_type = 'polygon'
        polygon_contiguity_info = contiguity_info
    else:
        contiguity_type = 'grid'
        polygon_contiguity_info = None
    
    # Run evaluation for all quarters from start_year to end_year using rolling window
    print(f"\nEvaluating all quarters from {start_year} to {end_year} using rolling window approach...")

    # Determine which quarters to evaluate based on desire_terms
    if desire_terms is None:
        quarters_to_evaluate = [1, 2, 3, 4]  # Evaluate all quarters
        print(f"Evaluating all quarters (Q1-Q4) for each year from {start_year} to {end_year}")
    else:
        quarters_to_evaluate = [desire_terms]  # Evaluate only specific quarter
        print(f"Evaluating only Q{desire_terms} for each year from {start_year} to {end_year}")

    print("Checkpoint recovery disabled: evaluating requested quarters from scratch.")

    remaining_quarters = [
        (year, quarter)
        for year in range(start_year, end_year + 1)
        for quarter in quarters_to_evaluate
    ]

    if not remaining_quarters:
        print("No quarters to evaluate with the provided configuration.")
        return results_df, y_pred_test

    # Create progress bar for requested quarterly evaluations
    progress_bar = tqdm(
        total=len(remaining_quarters),
        desc="GeoXGB Quarterly Evaluation",
        unit="quarter",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    # Loop through each quarter that needs to be evaluated
    for i, (test_year, quarter) in enumerate(remaining_quarters):
        progress_bar.set_description(f"GeoXGB Q{quarter} {test_year}")
        print(f"\n--- Evaluating Q{quarter} {test_year} (#{i+1}/{len(remaining_quarters)}) ---")
        
        # Memory monitoring at start of iteration
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Memory at start of Q{quarter} {test_year}: {start_memory:.1f} MB")
        
        try:
            # Train-test split with rolling window (5 years before quarter end)
            (Xtrain, ytrain, Xtrain_loc, Xtrain_group,
             Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_rolling_window(
                X, y, X_loc, X_group, years, dates, test_year=test_year, input_terms=input_terms, need_terms=quarter)
            
            ytrain = ytrain.astype(int)
            ytest = ytest.astype(int)
            
            print(f"Train samples: {len(ytrain)}, Test samples: {len(ytest)}")
            
            # Skip evaluation if no test samples
            if len(ytest) == 0:
                print(f"Warning: No test samples for Q{quarter} {test_year}. Skipping this quarter.")
                # Update progress bar even when skipping
                progress_bar.update(1)
                continue
            
            if nowcasting:
                # 2-layer model
                Xtrain_L1 = Xtrain[:, l1_index]
                Xtrain_L2 = Xtrain[:, l2_index]
                Xtest_L1 = Xtest[:, l1_index]
                Xtest_L2 = Xtest[:, l2_index]
                feature_names_L1 = [feature_columns[idx] for idx in l1_index]
            
                # Create and train 2-layer GeoXGB model - CHANGED from GeoRF to GeoRF_XGB
                geoxgb_2layer = GeoRF_XGB(
                    min_model_depth=MIN_DEPTH,
                    max_model_depth=MAX_DEPTH,
                    n_jobs=N_JOBS,
                    max_depth=max_depth,
                    # XGBoost-specific parameters
                    learning_rate=learning_rate,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree
                )
            
                # Train 2-layer model with optional metrics tracking
                if track_partition_metrics:
                    # Note: 2-layer fit doesn't support partition metrics yet, 
                    # but we can extend it later if needed
                    print("Note: Partition metrics tracking not yet supported for 2-layer models")
                
                geoxgb_2layer.fit_2layer(
                    Xtrain_L1, Xtrain_L2, ytrain, Xtrain_group,
                    val_ratio=VAL_RATIO,
                    contiguity_type=contiguity_type,
                    polygon_contiguity_info=polygon_contiguity_info,
                    feature_names_L1=feature_names_L1
                )
            
                # Get predictions
                ypred = geoxgb_2layer.predict_2layer(Xtest_L1, Xtest_L2, Xtest_group, correction_strategy='flip')
            
                # Evaluate
                (pre, rec, f1, pre_base, rec_base, f1_base) = geoxgb_2layer.evaluate_2layer(
                    X_L1_test=Xtest_L1,
                    X_L2_test=Xtest_L2,
                    y_test=ytest,
                    X_group_test=Xtest_group,
                    X_L1_train=Xtrain_L1,
                    X_L2_train=Xtrain_L2,
                    y_train=ytrain,
                    X_group_train=Xtrain_group,
                    correction_strategy='flip',
                    print_to_file=True,
                    contiguity_type=contiguity_type,
                    polygon_contiguity_info=polygon_contiguity_info
                )
            
                print(f"Q{quarter} {test_year} Test - 2-Layer GeoXGB F1: {f1}, 2-Layer Base XGB F1: {f1_base}")
            
                # Extract and save correspondence table for 2-layer model
                try:
                    X_branch_id_path = os.path.join(geoxgb_2layer.dir_space, 'X_branch_id.npy')
                    if os.path.exists(X_branch_id_path):
                        X_branch_id = np.load(X_branch_id_path)
                        create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, geoxgb_2layer.model_dir)
                except Exception as e:
                    print(f"Warning: Could not create correspondence table for Q{quarter} {test_year}: {e}")
            
            else:
                # Single-layer model
                geoxgb = GeoRF_XGB(  # CHANGED from GeoRF to GeoRF_XGB
                    min_model_depth=MIN_DEPTH,
                    max_model_depth=MAX_DEPTH,
                    n_jobs=N_JOBS,
                    max_depth=max_depth,
                    # XGBoost-specific parameters
                    learning_rate=learning_rate,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree
                )
            
            # Train model with optional partition metrics tracking
            if track_partition_metrics:
                print(f"Training GeoXGB with partition metrics tracking enabled")
                print(f"Correspondence table path: {correspondence_table_path}")
                print(f"Training set shape: {Xtrain.shape}, Groups shape: {Xtrain_group.shape}")
                print(f"Unique training groups: {len(np.unique(Xtrain_group))}")
                
                # Verify correspondence table exists and is readable
                if correspondence_table_path and os.path.exists(correspondence_table_path):
                    test_df = pd.read_csv(correspondence_table_path)
                    print(f"Correspondence table loaded successfully with {len(test_df)} entries")
                    print(f"Columns: {test_df.columns.tolist()}")
                    print(f"Sample entries:\n{test_df.head()}")
                else:
                    print(f"Warning: Correspondence table not found at {correspondence_table_path}")
            
            geoxgb.fit(
                Xtrain, ytrain, Xtrain_group,
                val_ratio=VAL_RATIO,
                contiguity_type=contiguity_type,
                polygon_contiguity_info=polygon_contiguity_info,
                track_partition_metrics=track_partition_metrics,
                correspondence_table_path=correspondence_table_path,
                feature_names=feature_columns,
                VIS_DEBUG_MODE=VIS_DEBUG_MODE
            )
            
            # Check if metrics were tracked
            if track_partition_metrics and hasattr(geoxgb, 'metrics_tracker'):
                if geoxgb.metrics_tracker is not None:
                        print(f"\nPartition metrics tracker found for Q{quarter} {test_year}")
                        
                        # Check if any metrics were actually recorded
                        if hasattr(geoxgb.metrics_tracker, 'all_metrics') and geoxgb.metrics_tracker.all_metrics:
                            print(f"Number of metric records: {len(geoxgb.metrics_tracker.all_metrics)}")
                            
                            # Show some sample metrics
                            for i, record in enumerate(geoxgb.metrics_tracker.all_metrics[:3]):
                                print(f"  Record {i}: Round {record.get('partition_round', 'N/A')}, "
                                      f"Branch {record.get('branch_id', 'N/A')}, "
                                      f"F1 improvement: {record.get('f1_improvement', 'N/A'):.4f}")
                        else:
                            print("No metrics records found in tracker")
                        
                        # Try to get summary
                        try:
                            summary = geoxgb.metrics_tracker.get_improvement_summary()
                            if summary:
                                print(f"\nPartition Metrics Summary for Q{quarter} {test_year}:")
                                print(f"  Total partitions tracked: {summary['total_partitions']}")
                                print(f"  Average F1 improvement: {summary['avg_f1_improvement']:.4f}")
                                print(f"  Average accuracy improvement: {summary['avg_accuracy_improvement']:.4f}")
                                print(f"  Positive F1 improvements: {summary['positive_f1_improvements']}")
                                print(f"  Positive accuracy improvements: {summary['positive_accuracy_improvements']}")
                            else:
                                print("Warning: No partition metrics summary available")
                        except Exception as e:
                            print(f"Error getting metrics summary: {e}")
                            
                        # Check if visualization files were created
                        if hasattr(geoxgb, 'model_dir'):
                            vis_dir = os.path.join(geoxgb.model_dir, 'vis')
                            metrics_dir = os.path.join(geoxgb.model_dir, 'partition_metrics')
                            
                            if os.path.exists(vis_dir):
                                vis_files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
                                print(f"Visualization files created: {len(vis_files)}")
                                if vis_files:
                                    print(f"  Sample files: {vis_files[:3]}")
                            else:
                                print("No visualization directory found")
                                
                            if os.path.exists(metrics_dir):
                                csv_files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
                                print(f"Metrics CSV files created: {len(csv_files)}")
                                if csv_files:
                                    print(f"  Sample files: {csv_files[:3]}")
                            else:
                                print("No metrics directory found")
                else:
                    print("Warning: Metrics tracker is None")
            else:
                if track_partition_metrics:
                    print("Warning: Metrics tracker not found on geoxgb object")
                
            # Get predictions
            ypred = geoxgb.predict(Xtest, Xtest_group)
            
            # Evaluate
            (pre, rec, f1, pre_base, rec_base, f1_base) = geoxgb.evaluate(
                Xtest, ytest, Xtest_group, eval_base=True, print_to_file=True,
                force_accuracy=force_final_accuracy and VIS_DEBUG_MODE,
                VIS_DEBUG_MODE=VIS_DEBUG_MODE
            )
            
            print(f"Q{quarter} {test_year} Test - GeoXGB F1: {f1}, Base XGB F1: {f1_base}")
            
            # Extract and save correspondence table for single-layer model
            try:
                X_branch_id_path = os.path.join(geoxgb.dir_space, 'X_branch_id.npy')
                if os.path.exists(X_branch_id_path):
                    X_branch_id = np.load(X_branch_id_path)
                    create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, geoxgb.model_dir)
            except Exception as e:
                print(f"Warning: Could not create correspondence table for Q{quarter} {test_year}: {e}")
            
            # Store results - MEMORY FIX: Use more efficient DataFrame appending
            nsample_class = np.bincount(ytest)
            
            # Create new result row as dictionary first to avoid intermediate DataFrame
            new_result_row = {
                'year': test_year,
                'quarter': quarter,
                'precision(0)': pre[0],
                'precision(1)': pre[1],
                'recall(0)': rec[0],
                'recall(1)': rec[1],
                'f1(0)': f1[0],
                'f1(1)': f1[1],
                'precision_base(0)': pre_base[0],
                'precision_base(1)': pre_base[1],
                'recall_base(0)': rec_base[0],
                'recall_base(1)': rec_base[1],
                'f1_base(0)': f1_base[0],
                'f1_base(1)': f1_base[1],
                'num_samples(0)': nsample_class[0],
                'num_samples(1)': nsample_class[1]
            }
            
            # Append row efficiently using pd.concat with list
            results_df = pd.concat([results_df, pd.DataFrame([new_result_row])], ignore_index=True)
            
            # Store predictions - MEMORY FIX: Use more efficient approach  
            try:
                # CRITICAL MEMORY FIX: Avoid unnecessary array copies that cause memory bloat
                # Create prediction data as dictionary first - avoid .copy() which doubles memory usage
                pred_data = {
                    'year': np.full(len(ytest), test_year, dtype=np.int16),  # Use smaller dtypes
                    'quarter': np.full(len(ytest), quarter, dtype=np.int8),
                    'month': np.full(len(ytest), quarter * 3, dtype=np.int8),  # Quarter end month (Mar, Jun, Sep, Dec)
                    'adm_code': np.zeros(len(ytest), dtype=np.int32),  # Placeholder - would need actual admin codes
                    'fews_ipc_crisis_pred': ypred,  # Don't copy - transfer ownership
                    'fews_ipc_crisis_true': ytest   # Don't copy - transfer ownership
                }
                
                # Append predictions efficiently
                y_pred_test = pd.concat([y_pred_test, pd.DataFrame(pred_data)], ignore_index=True)
                
            except Exception as e:
                print(f"Warning: Error storing predictions: {e}")
                # Fallback with placeholders - MEMORY OPTIMIZED
                pred_data = {
                    'year': np.full(len(ytest), test_year, dtype=np.int16),
                    'quarter': np.full(len(ytest), quarter, dtype=np.int8),
                    'month': np.full(len(ytest), quarter * 3, dtype=np.int8),  # Quarter end month (Mar, Jun, Sep, Dec)
                    'adm_code': np.zeros(len(ytest), dtype=np.int32),
                    'fews_ipc_crisis_pred': ypred,  # Transfer ownership, don't copy
                    'fews_ipc_crisis_true': ytest   # Transfer ownership, don't copy
                }
                y_pred_test = pd.concat([y_pred_test, pd.DataFrame(pred_data)], ignore_index=True)
            
            # CRITICAL MEMORY FIX: Add explicit cleanup for intermediate variables
            print("Cleaning up memory...")
            
            # Clean up intermediate variables immediately after storing results
            try:
                del new_result_row
                del pred_data
                del nsample_class
            except:
                pass
            
            # CRITICAL FIX 1: Clear PartitionMetricsTracker accumulation (major memory leak source)
            try:
                if 'geoxgb' in locals() and hasattr(geoxgb, 'metrics_tracker'):
                    if geoxgb.metrics_tracker is not None:
                        # Clear accumulated metrics data (can be hundreds of MB per quarter)
                        if hasattr(geoxgb.metrics_tracker, 'all_metrics'):
                            geoxgb.metrics_tracker.all_metrics.clear()
                        if hasattr(geoxgb.metrics_tracker, 'partition_history'):
                            geoxgb.metrics_tracker.partition_history.clear()
                        geoxgb.metrics_tracker = None
                        print("Cleared PartitionMetricsTracker data")
                if 'geoxgb_2layer' in locals() and hasattr(geoxgb_2layer, 'metrics_tracker'):
                    if geoxgb_2layer.metrics_tracker is not None:
                        if hasattr(geoxgb_2layer.metrics_tracker, 'all_metrics'):
                            geoxgb_2layer.metrics_tracker.all_metrics.clear()
                        if hasattr(geoxgb_2layer.metrics_tracker, 'partition_history'):
                            geoxgb_2layer.metrics_tracker.partition_history.clear()
                        geoxgb_2layer.metrics_tracker = None
                        print("Cleared 2-layer PartitionMetricsTracker data")
            except Exception as e:
                print(f"Warning: Could not clear metrics tracker: {e}")
            
            # CRITICAL FIX 2: Delete model objects completely and break circular references
            try:
                if 'geoxgb' in locals():
                    # ENHANCED model cleanup to prevent memory leaks
                    # Clear all model components explicitly
                    if hasattr(geoxgb, 'model') and geoxgb.model is not None:
                        if hasattr(geoxgb.model, 'model') and geoxgb.model.model is not None:
                            # Clear XGBoost internals that hold large arrays
                            # Use try-except to safely clear attributes that might not exist or cause errors
                            try:
                                if hasattr(geoxgb.model.model, '_Booster') and geoxgb.model.model._Booster is not None:
                                    geoxgb.model.model._Booster = None
                            except:
                                pass
                            try:
                                # Clear feature importances
                                if hasattr(geoxgb.model.model, 'feature_importances_') and geoxgb.model.model.feature_importances_ is not None:
                                    geoxgb.model.model.feature_importances_ = None
                            except:
                                pass
                            # Clear XGBoost model completely
                            geoxgb.model.model = None
                        geoxgb.model = None
                    
                    # Clear all directory references
                    geoxgb.dir_space = None
                    geoxgb.dir_ckpt = None
                    geoxgb.dir_vis = None
                    geoxgb.model_dir = None
                    
                    # Clear spatial partitioning data that can be large
                    if hasattr(geoxgb, 's_branch'):
                        geoxgb.s_branch = None
                    if hasattr(geoxgb, 'branch_table'):
                        geoxgb.branch_table = None
                    if hasattr(geoxgb, 'X_branch_id'):
                        geoxgb.X_branch_id = None
                    
                    # Clear any other potential large attributes
                    for attr in ['train_idx', 'val_idx', 'X_train', 'y_train', 'X_val', 'y_val']:
                        if hasattr(geoxgb, attr):
                            setattr(geoxgb, attr, None)
                    
                    geoxgb = None
                del geoxgb
                print("Cleared GeoXGB model and all references")
            except NameError:
                pass
            try:
                if 'geoxgb_2layer' in locals():
                    # ENHANCED cleanup for 2-layer model
                    # Clear both layer models 
                    if hasattr(geoxgb_2layer, 'georf_l1') and geoxgb_2layer.georf_l1 is not None:
                        # Clear L1 model internals
                        if hasattr(geoxgb_2layer.georf_l1, 'model') and geoxgb_2layer.georf_l1.model is not None:
                            if hasattr(geoxgb_2layer.georf_l1.model, 'model') and geoxgb_2layer.georf_l1.model.model is not None:
                                try:
                                    if hasattr(geoxgb_2layer.georf_l1.model.model, '_Booster') and geoxgb_2layer.georf_l1.model.model._Booster is not None:
                                        geoxgb_2layer.georf_l1.model.model._Booster = None
                                except:
                                    pass
                                geoxgb_2layer.georf_l1.model.model = None
                            geoxgb_2layer.georf_l1.model = None
                        geoxgb_2layer.georf_l1 = None
                    
                    if hasattr(geoxgb_2layer, 'georf_l2') and geoxgb_2layer.georf_l2 is not None:
                        # Clear L2 model internals
                        if hasattr(geoxgb_2layer.georf_l2, 'model') and geoxgb_2layer.georf_l2.model is not None:
                            if hasattr(geoxgb_2layer.georf_l2.model, 'model') and geoxgb_2layer.georf_l2.model.model is not None:
                                try:
                                    if hasattr(geoxgb_2layer.georf_l2.model.model, '_Booster') and geoxgb_2layer.georf_l2.model.model._Booster is not None:
                                        geoxgb_2layer.georf_l2.model.model._Booster = None
                                except:
                                    pass
                                geoxgb_2layer.georf_l2.model.model = None
                            geoxgb_2layer.georf_l2.model = None
                        geoxgb_2layer.georf_l2 = None
                    
                    if hasattr(geoxgb_2layer, 'model') and geoxgb_2layer.model is not None:
                        if hasattr(geoxgb_2layer.model, 'model') and geoxgb_2layer.model.model is not None:
                            try:
                                if hasattr(geoxgb_2layer.model.model, '_Booster') and geoxgb_2layer.model.model._Booster is not None:
                                    geoxgb_2layer.model.model._Booster = None
                            except:
                                pass
                            geoxgb_2layer.model.model = None
                        geoxgb_2layer.model = None
                    
                    # Clear directory references
                    geoxgb_2layer.dir_space = None
                    geoxgb_2layer.dir_ckpt = None
                    geoxgb_2layer.dir_vis = None
                    geoxgb_2layer.model_dir = None
                    
                    # Clear spatial partitioning data
                    if hasattr(geoxgb_2layer, 's_branch'):
                        geoxgb_2layer.s_branch = None
                    if hasattr(geoxgb_2layer, 'branch_table'):
                        geoxgb_2layer.branch_table = None
                        
                    geoxgb_2layer = None
                del geoxgb_2layer
                print("Cleared 2-layer GeoXGB model and all references")
            except NameError:
                pass
            
            # Delete training data
            try:
                del Xtrain
            except NameError:
                pass
            try:
                del ytrain
            except NameError:
                pass
            try:
                del Xtrain_loc
            except NameError:
                pass
            try:
                del Xtrain_group
            except NameError:
                pass
            
            # Delete test data
            try:
                del Xtest
            except NameError:
                pass
            try:
                del ytest
            except NameError:
                pass
            try:
                del Xtest_loc
            except NameError:
                pass
            try:
                del Xtest_group
            except NameError:
                pass
            
            # Delete layer-specific data
            try:
                del Xtrain_L1
            except NameError:
                pass
            try:
                del Xtrain_L2
            except NameError:
                pass
            try:
                del Xtest_L1
            except NameError:
                pass
            try:
                del Xtest_L2
            except NameError:
                pass
            
            # Delete predictions and other large arrays
            try:
                del ypred
            except NameError:
                pass
            try:
                del X_branch_id
            except NameError:
                pass
            
            # Delete any other potentially large variables
            try:
                del nsample_class
            except NameError:
                pass
            try:
                del pre, rec, f1, pre_base, rec_base, f1_base
            except NameError:
                pass
            
            # AGGRESSIVE memory cleanup to prevent hidden leaks
            import sys
            
            # CRITICAL FIX 2: Clear XGBoost internal caches and memory pools more aggressively
            try:
                # Clear XGBoost internal memory pools if available
                import xgboost as xgb
                if hasattr(xgb, '_lib') and hasattr(xgb._lib, 'XGBSetGlobalConfig'):
                    # Reset XGBoost global configuration to clear internal caches
                    try:
                        xgb._lib.XGBSetGlobalConfig('{"verbosity": 0}')
                    except:
                        pass
                print("Cleared XGBoost internal memory pools")
            except Exception as e:
                print(f"Warning: Could not clear XGBoost pools: {e}")
                
            # Clear numpy memory pools
            try:
                # Force numpy to release memory back to system (np already imported globally)
                np.random.seed()  # This can help clear internal state
            except:
                pass
            
            # CRITICAL FIX 3: DISABLE aggressive pandas cache clearing to prevent hashtable corruption
            # The KeyError 'int32' indicates that our cache clearing is corrupting pandas internal state
            # We'll use a much safer approach that only clears specific known-safe caches
            try:
                # Only clear the safest pandas caches to avoid corrupting internal hashtables
                print("Performing safe pandas cache cleanup...")
                
                # Clear only the most basic caches that are known to be safe
                try:
                    # Clear string interning which is generally safe
                    if hasattr(pd.core.dtypes.common, '_pandas_dtype_type_map'):
                        # Don't clear this as it can cause issues
                        pass
                    
                    # Force a small garbage collection instead of aggressive cache clearing
                    import gc
                    gc.collect()
                    
                    print("Applied safe pandas cleanup")
                except Exception as inner_e:
                    print(f"Warning: Safe pandas cleanup failed: {inner_e}")
                    
            except Exception as e:
                print(f"Warning: Could not perform pandas cleanup: {e}")
            
            # CRITICAL FIX 4: Add explicit DataFrame memory management to fix the real memory leak
            # The 1.3GB growth suggests DataFrames are not being properly garbage collected
            try:
                print("Forcing DataFrame garbage collection...")
                
                # Clear any DataFrames that might be lingering in local scope
                for var_name in list(locals().keys()):
                    if var_name.startswith('df') or 'frame' in var_name.lower():
                        try:
                            local_var = locals()[var_name]
                            if hasattr(local_var, 'values'):  # Likely a DataFrame
                                del locals()[var_name]
                        except:
                            pass
                
                # Aggressive garbage collection specifically for DataFrames
                import gc
                for obj in gc.get_objects():
                    try:
                        if hasattr(obj, '_mgr') and hasattr(obj, 'columns'):  # Likely a DataFrame
                            if hasattr(obj, '_clear_item_cache'):
                                obj._clear_item_cache()
                    except:
                        pass
                
                # Multiple garbage collection passes
                for i in range(3):
                    collected = gc.collect()
                    if collected == 0:
                        break
                        
                print("Completed DataFrame garbage collection")
                
            except Exception as e:
                print(f"Warning: DataFrame garbage collection failed: {e}")
                
            # Clear Python's internal object caches
            try:
                # Clear small int cache
                for i in range(-5, 257):
                    sys.intern(str(i))
                # Clear string interning cache (careful with this)
                sys.intern.cache_clear() if hasattr(sys.intern, 'cache_clear') else None
            except:
                pass
            
            # Force garbage collection after every quarter
            gc.collect()
            
            # Memory monitoring after cleanup with leak detection
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = end_memory - start_memory
            print(f"Memory after cleanup: {end_memory:.1f} MB (change: {memory_diff:+.1f} MB)")
            
            # Alert if significant memory leak detected
            if memory_diff > 500:  # More than 500MB growth per quarter
                print(f"WARNING: Large memory increase detected: {memory_diff:.1f} MB")
                print("This indicates a potential memory leak that needs investigation")
                print(f"Consider reducing n_jobs or disabling partition metrics tracking")
            elif memory_diff > 100:  # More than 100MB but less than 500MB
                print(f"NOTICE: Moderate memory increase: {memory_diff:.1f} MB")
                print("Memory growth within acceptable range but monitor if this persists")
            else:
                print(f"OK: Memory growth within normal range: {memory_diff:+.1f} MB")
            
            # Show DataFrame sizes for monitoring  
            print(f"DataFrame sizes: results_df={len(results_df)} rows, y_pred_test={len(y_pred_test)} rows")
            
        except Exception as e:
            print(f"Error evaluating Q{quarter} {test_year}: {str(e)}")
            print("Continuing to next quarter...")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always update progress bar, regardless of success or failure
            progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    return results_df, y_pred_test

def main():
    """
    Main function to run the complete GeoXGB pipeline.
    """
    print("=== Starting GeoXGB Food Crisis Prediction Pipeline ===")
    print(f"[PROD MODE: visuals {'ENABLED' if VIS_DEBUG_MODE else 'DISABLED'}]")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GeoXGB Food Crisis Prediction Pipeline')
    parser.add_argument('--start_year', type=int, default=2024, help='Start year for evaluation (default: 2024)')
    parser.add_argument('--end_year', type=int, default=2024, help='End year for evaluation (default: 2024)')
    scope_choices = list(range(1, len(ACTIVE_LAGS) + 1))
    scope_help = f"Forecasting scope (1-based index) for lag schedule {ACTIVE_LAGS} (default: {len(ACTIVE_LAGS)})"
    parser.add_argument('--forecasting_scope', type=int, default=len(ACTIVE_LAGS), choices=scope_choices,
                        help=scope_help)
    parser.add_argument('--force-final-accuracy', action='store_true',
                        help='Force generation of final accuracy maps even when VIS_DEBUG_MODE=False')
    args = parser.parse_args()
    
    # Configuration
    assignment = 'polygons'  # Change this to test different grouping methods
    nowcasting = False       # Set to True for 2-layer model
    max_depth = None  # Set to integer for specific XGB depth
    desire_terms = 1         # None=all quarters, 1=Q1 only, 2=Q2 only, 3=Q3 only, 4=Q4 only
    forecasting_scope = args.forecasting_scope
    active_lag = forecasting_scope_to_lag(forecasting_scope, ACTIVE_LAGS)
    log_path = log_lag_schedule(ACTIVE_LAGS, ARTIFACTS_ROOT)
    
    # XGBoost-specific hyperparameters optimized for food crisis prediction
    learning_rate = 0.1      # Learning rate (step size shrinkage)
    reg_alpha = 0.1          # L1 regularization term on weights
    reg_lambda = 1.0         # L2 regularization term on weights
    subsample = 0.8          # Subsample ratio of training instances
    colsample_bytree = 0.8   # Subsample ratio of columns when constructing each tree
    
    # Note: XGBoost handles missing values natively - no extreme value imputation needed
    
    # Partition Metrics Tracking Configuration
    track_partition_metrics = True # Enable partition metrics tracking and visualization
    enable_metrics_maps = True      # Create maps showing F1/accuracy improvements
    
    # Checkpoint Recovery Configuration
    # start year and end year from command line arguments
    start_year = 2015
    end_year = 2015
    
    print(f"Configuration:")
    print(f"  - Assignment method: {assignment}")
    print(f"  - Nowcasting (2-layer): {nowcasting}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Desired terms: {desire_terms} ({'All quarters (Q1-Q4)' if desire_terms is None else f'Q{desire_terms} only'})")
    print(f"  - Forecasting scope: {forecasting_scope} ({active_lag}-month lag)")
    print(f"  - Active lag schedule: {ACTIVE_LAGS} (logged to {log_path})")
    print(f"  - Rolling window: 5-year training windows before each test quarter")
    print(f"  - Track partition metrics: {track_partition_metrics}")
    print(f"  - Enable metrics maps: {enable_metrics_maps}")
    print(f"  - Start year: {start_year}, End year: {end_year}")
    print(f"  - XGBoost hyperparameters:")
    print(f"    * learning_rate: {learning_rate}")
    print(f"    * reg_alpha (L1): {reg_alpha}")
    print(f"    * reg_lambda (L2): {reg_lambda}")
    print(f"    * subsample: {subsample}")
    print(f"    * colsample_bytree: {colsample_bytree}")
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(DATA_PATH)
        
        # Step 2: Setup spatial groups
        X_group, X_loc, contiguity_info = setup_spatial_groups(df, assignment)
        
        # Step 3: Prepare features with forecasting scope
        X, y, l1_index, l2_index, years, terms, dates, feature_columns = prepare_features(df, X_group, X_loc, forecasting_scope=forecasting_scope)
        
        # Step 4: Validate polygon contiguity (if applicable) and track polygon counts
        if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans'] and contiguity_info is not None:
            validate_polygon_contiguity(contiguity_info, X_group)
            
            # Track polygon counts for disappearance diagnosis
            if DIAGNOSTIC_POLYGON_TRACKING:
                initial_polygon_count = len(np.unique(X_group))
                print(f"=== XGB POLYGON TRACKING ===")
                print(f"Initial polygon count after spatial setup: {initial_polygon_count}")
                print(f"X_group unique values: {len(np.unique(X_group))}")
                print(f"Data points: {len(X_group)}")
                print(f"Sample X_group values: {sorted(np.unique(X_group))[:20]}")
                
                # Check for missing polygon IDs in sequence
                unique_groups = sorted(np.unique(X_group))
                if len(unique_groups) > 1:
                    gaps = []
                    for i in range(1, len(unique_groups)):
                        if unique_groups[i] - unique_groups[i-1] > 1:
                            gaps.append((unique_groups[i-1], unique_groups[i]))
                    if gaps:
                        print(f"Warning: Gaps found in X_group sequence: {gaps[:5]}")
                    else:
                        print("No gaps found in X_group sequence")
                print("=" * 29)
        
        # Step 5: Run temporal evaluation with XGBoost hyperparameters
        results_df, y_pred_test = run_temporal_evaluation(
            X, y, X_loc, X_group, years, dates, l1_index, l2_index, feature_columns,
            assignment, contiguity_info, df, nowcasting, max_depth, input_terms=terms, desire_terms=desire_terms,
            track_partition_metrics=track_partition_metrics, enable_metrics_maps=enable_metrics_maps,
            start_year=start_year, end_year=end_year, forecasting_scope=forecasting_scope,
            force_final_accuracy=args.force_final_accuracy,
            # Pass XGBoost hyperparameters
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        )
        
        # Step 6: Filter results to class 1 only (if needed)
        class_1_columns = [col for col in results_df.columns if '(1)' in col or col in ['year', 'quarter']]
        if len(class_1_columns) < len(results_df.columns):
            print("Filtering results to class 1 metrics only...")
            results_df = results_df[class_1_columns].copy()
        
        # Step 7: Save results
        save_results(results_df, y_pred_test, assignment, nowcasting, max_depth, desire_terms=desire_terms, forecasting_scope=forecasting_scope, start_year=start_year, end_year=end_year)
        
        # Step 8: Display summary (class 1 only)
        print("\n=== Evaluation Summary (Class 1 Only) ===")
        if 'quarter' in results_df.columns:
            print("Results by Quarter:")
            print(results_df.groupby(['year', 'quarter'])[['f1(1)', 'f1_base(1)']].mean())
        else:
            print("Results by Year:")
            print(results_df.groupby('year')[['f1(1)', 'f1_base(1)']].mean())
        
        print("\n=== GeoXGB Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
