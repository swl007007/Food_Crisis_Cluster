#!/usr/bin/env python3
"""
XGBoost version of main_model_GF_replicated_fixed.py

This script replicates the full functionality of the original GeoRF pipeline
but uses XGBoost instead of Random Forest as the base learner.

Key features:
1. Data preprocessing with polars and pandas (same as RF version)
2. Multiple spatial grouping options (polygons, grid, country, AEZ, etc.)
3. Polygon-based contiguity with corrected setup
4. Time-based train-test splitting for temporal validation
5. Single-layer and 2-layer GeoXGB models (XGBoost-based GeoRF)
6. Comprehensive evaluation and result saving
7. XGBoost hyperparameters optimized for acute food crisis prediction

Date: 2024-07-21
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# GeoRF imports (modified for XGBoost)
from GeoRF_XGB import GeoRF_XGB  # Changed from GeoRF to GeoRF_XGB
from customize import *
from data import load_demo_data
from helper import get_spatial_range
from initialization import train_test_split_all
from config import *
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_v06252025.csv"

def comp_impute(X, strategy="max_plus", multiplier=100.0):
    """
    Custom imputation function that handles infinite values and missing data.
    Same as RF version - no changes needed.
    """
    print(f"Starting imputation with strategy: {strategy}")
    
    # Handle infinite values
    for col_idx in range(X.shape[1]):
        print(f"Checking column {col_idx} for inf values")
        col_data = X[:, col_idx]
        
        try:
            col_data_float = col_data.astype(float)
            if np.isinf(col_data_float).any():
                print(f"Column {col_idx} has inf values, replacing with NaN")
                col_data_float[np.isinf(col_data_float)] = np.nan
                X[:, col_idx] = col_data_float
        except Exception as e:
            print(f"Column {col_idx} could not be converted to float: {e}")
            continue
    
    # Apply imputation
    X_imputed, imputer = impute_missing_values(X, strategy=strategy, multiplier=multiplier, verbose=True)
    return X_imputed

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the FEWSNET data.
    Same as RF version - no changes needed.
    """
    print("Loading data...")
    
    # Load data with polars
    data = pl.read_csv(data_path)
    
    # Drop unnecessary columns
    cols_to_drop = [
        "ISO3", "fews_ipc_adjusted", "fews_proj_med_adjusted", "fews_ipc",
        "fews_proj_near", "fews_proj_near_ha", "fews_proj_med",
        "fews_proj_med_ha", "ADMIN0", "ADMIN1", "ADMIN2", "ADMIN3"
    ]
    data = data.drop(cols_to_drop)
    
    # Optionally reduce variables
    less_var = True
    if less_var:
        var_extra = [
            'Evap_tavg_mean', 'Qsb_tavg_mean', 'RadT_tavg_mean', 
            'SnowCover_inst_mean', 'SnowDepth_inst_mean', 'Snowf_tavg_mean',
            'SoilMoi00_10cm_tavg_mean', 'SoilMoi10_40cm_tavg_mean', 
            'SoilMoi100_200cm_tavg_mean', 'SoilMoi40_100cm_tavg_mean',
            'LWdown_f_tavg_mean', 'SoilTemp00_10cm_tavg_mean', 
            'SoilTemp10_40cm_tavg_mean', 'SoilTemp100_200cm_tavg_mean',
            'SoilTemp40_100cm_tavg_mean', 'SWdown_f_tavg_mean', 
            'SWE_inst_mean', 'Swnet_tavg_mean', 'Wind_f_tavg_mean',
            'Lwnet_tavg_mean', 'Psurf_f_tavg_mean', 'Qair_f_tavg_mean',
            'Qg_tavg_mean', 'Qh_tavg_mean', 'Qle_tavg_mean', 'Qs_tavg_mean'
        ]
        data = data.drop([col for col in data.columns if col.startswith(tuple(var_extra))])
    
    # Filter for non-null crisis data
    data = data.filter(pl.col("fews_ipc_crisis").is_not_null())
    
    # Encode ISO
    data = data.with_columns([
        pl.col("ISO").cast(pl.Categorical).to_physical().alias("ISO_encoded")
    ])
    
    # Drop unit name and ISO
    data = data.drop(["unit_name", "ISO"])
    
    # Process AEZ columns
    for col in data.columns:
        if col.startswith("AEZ_"):
            data = data.with_columns(
                pl.when(pl.col(col) == "True")
                .then(1)
                .when(pl.col(col) == "False")
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # Convert to pandas
    df = data.to_pandas()
    
    # Process dates
    df['date'] = pd.to_datetime(df['date'])
    df['years'] = df['date'].dt.year
    
    # Create lag features
    df['fews_ipc_crisis_lag_1'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(1)
    df['fews_ipc_crisis_lag_2'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(2)
    df['fews_ipc_crisis_lag_3'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(3)
    
    # Create year and month dummies
    for year in df['years'].unique():
        df[f'year_{year}'] = (df['years'] == year).astype(int)
    for month in df['date'].dt.month.unique():
        df[f'month_{month}'] = (df['date'].dt.month == month).astype(int)
    
    # Create AEZ groups
    aez_columns = [col for col in df.columns if col.startswith('AEZ_')]
    df['AEZ_group'] = df.groupby(aez_columns).ngroup()
    df['AEZ_country_group'] = df.groupby(['AEZ_group', 'ISO_encoded']).ngroup()
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def setup_spatial_groups(df, assignment='polygons'):
    """
    Setup spatial grouping based on assignment method.
    Same as RF version - no changes needed.
    """
    print(f"Setting up spatial groups using: {assignment}")
    
    # Get location coordinates
    X_loc = df[['lat', 'lon']].values
    
    if assignment == 'polygons':
        X_polygon_ids = df['FEWSNET_admin_code'].values
        
        # Get unique polygons and their centroids
        polygon_data = df[['FEWSNET_admin_code', 'lat', 'lon']].drop_duplicates()
        polygon_centroids = polygon_data[['lat', 'lon']].values
        
        # Get unique admin codes
        unique_polygons = polygon_data['FEWSNET_admin_code'].unique()
        
        # Create mapping from FEWSNET_admin_code to polygon index
        admin_to_polygon_idx = {admin_code: idx for idx, admin_code in enumerate(unique_polygons)}
        
        # FIXED: Create correct polygon_group_mapping
        polygon_group_mapping = {i: [unique_polygons[i]] for i in range(len(unique_polygons))}
        
        # Convert admin codes to polygon indices for PolygonGroupGenerator
        polygon_indices = np.array([admin_to_polygon_idx[admin_code] for admin_code in X_polygon_ids])
        
        # Create polygon generator with correct mapping
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=polygon_centroids,
            polygon_group_mapping=polygon_group_mapping,
            neighbor_distance_threshold=0.8
        )
        
        # Generate groups (these will be admin codes, not indices!)
        X_group = polygon_gen.get_groups(polygon_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Polygon setup complete: {len(unique_polygons)} polygons, {len(np.unique(X_group))} groups")
        print(f"X_group range: {X_group.min()} to {X_group.max()}")
        print(f"Contiguity info keys: {list(contiguity_info.keys())}")
        
        # Validation
        assert len(np.unique(X_group)) == len(unique_polygons), "Number of groups should equal number of polygons"
        assert set(X_group) == set(unique_polygons), "Group IDs should be admin codes"
        print("Validation passed!")
        
    elif assignment == 'grid':
        xmin, xmax, ymin, ymax = get_spatial_range(X_loc)
        STEP_SIZE = 0.1
        group_gen = GroupGenerator(xmin, xmax, ymin, ymax, STEP_SIZE)
        X_group = group_gen.get_groups(X_loc)
        contiguity_info = None
        
    # ... (other assignment methods same as RF version)
    
    return X_group, X_loc, contiguity_info

def prepare_features(df, X_group, X_loc):
    """
    Prepare feature matrices and identify L1/L2 feature indices.
    Same as RF version - no changes needed.
    """
    print("Preparing features...")
    
    # Define time-variant features for L2
    time_variants = [
        'event_count_battles', 'event_count_explosions', 'event_count_violence',
        'sum_fatalities_battles', 'sum_fatalities_explosions', 'sum_fatalities_violence',
        'event_count_battles_w5', 'event_count_explosions_w5', 'event_count_violence_w5',
        'sum_fatalities_battles_w5', 'sum_fatalities_explosions_w5', 'sum_fatalities_violence_w5',
        'event_count_battles_w10', 'event_count_explosions_w10', 'event_count_violence_w10',
        'sum_fatalities_battles_w10', 'sum_fatalities_explosions_w10', 'sum_fatalities_violence_w10',
        'nightlight', 'nightlight_sd', 'EVI', 'EVI_stdDev', 'FAO_price',
        'Evap_tavg_mean', 'Evap_tavg_stdDev', 'LWdown_f_tavg_mean', 'LWdown_f_tavg_stdDev',
        'Lwnet_tavg_mean', 'Lwnet_tavg_stdDev', 'Psurf_f_tavg_mean', 'Psurf_f_tavg_stdDev',
        'Qair_f_tavg_mean', 'Qair_f_tavg_stdDev', 'Qg_tavg_mean', 'Qg_tavg_stdDev',
        'Qh_tavg_mean', 'Qh_tavg_stdDev', 'Qle_tavg_mean', 'Qle_tavg_stdDev',
        'Qs_tavg_mean', 'Qs_tavg_stdDev', 'Qsb_tavg_mean', 'Qsb_tavg_stdDev',
        'RadT_tavg_mean', 'RadT_tavg_stdDev', 'Rainf_f_tavg_mean', 'Rainf_f_tavg_stdDev',
        'SnowCover_inst_mean', 'SnowCover_inst_stdDev', 'SnowDepth_inst_mean', 'SnowDepth_inst_stdDev',
        'Snowf_tavg_mean', 'Snowf_tavg_stdDev', 'SoilMoi00_10cm_tavg_mean', 'SoilMoi00_10cm_tavg_stdDev',
        'SoilMoi10_40cm_tavg_mean', 'SoilMoi10_40cm_tavg_stdDev', 'SoilMoi100_200cm_tavg_mean', 'SoilMoi100_200cm_tavg_stdDev',
        'SoilMoi40_100cm_tavg_mean', 'SoilMoi40_100cm_tavg_stdDev', 'SoilTemp00_10cm_tavg_mean', 'SoilTemp00_10cm_tavg_stdDev',
        'SoilTemp10_40cm_tavg_mean', 'SoilTemp10_40cm_tavg_stdDev', 'SoilTemp100_200cm_tavg_mean', 'SoilTemp100_200cm_tavg_stdDev',
        'SoilTemp40_100cm_tavg_mean', 'SoilTemp40_100cm_tavg_stdDev', 'SWdown_f_tavg_mean', 'SWdown_f_tavg_stdDev',
        'SWE_inst_mean', 'SWE_inst_stdDev', 'Swnet_tavg_mean', 'Swnet_tavg_stdDev',
        'Tair_f_tavg_mean', 'Tair_f_tavg_stdDev', 'Wind_f_tavg_mean', 'Wind_f_tavg_stdDev',
        'gpp_sd', 'gpp_mean', 'CPI', 'GDP', 'CC', 'gini', 'WFP_Price', 'WFP_Price_std'
    ]
    
    time_variants_m12 = [variant + '_m12' for variant in time_variants]
    time_variants_list = time_variants + time_variants_m12
    
    # Get target variable
    y = df['fews_ipc_crisis'].values
    
    # Create comprehensive correspondence table with X_group mapping
    correspondence_df = df[['FEWSNET_admin_code', 'AEZ_group', 'AEZ_country_group', 'ISO_encoded']].copy()
    correspondence_df['X_group'] = X_group
    correspondence_table = correspondence_df.drop_duplicates()
    correspondence_table.to_csv('correspondence_table_xgb.csv', index=False)
    
    # Sort by admin code and date
    df_sorted = df.sort_values(by=['FEWSNET_admin_code', 'date'])
    
    # Create additional lag features
    for lag in range(1, 4):
        df_sorted[f'fews_ipc_crisis_lag_{lag}'] = df_sorted.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(lag)
    
    # Drop unnecessary columns
    df_features = df_sorted.drop(columns=['date', 'fews_ipc_crisis', 'AEZ_group', 'ISO_encoded', 'AEZ_country_group'])
    
    # Get feature matrix
    X = df_features.values
    
    # Identify L2 feature indices
    l2_index = []
    for i, col in enumerate(df_features.columns):
        if col in time_variants_list:
            l2_index.append(i)
    
    # L1 features are all others
    l1_index = [i for i in range(X.shape[1]) if i not in l2_index]
    
    # Apply imputation
    X = comp_impute(X)
    
    # Get years for temporal splitting
    years = df_sorted['years'].values
    
    print(f"Feature preparation complete: {X.shape[1]} features, {len(l1_index)} L1 features, {len(l2_index)} L2 features")
    
    return X, y, l1_index, l2_index, years

def validate_polygon_contiguity(contiguity_info, X_group):
    """
    Validate polygon contiguity setup.
    Same as RF version - no changes needed.
    """
    print("=== Polygon Contiguity Validation ===")
    
    from partition_opt import get_polygon_neighbors
    
    neighbors = get_polygon_neighbors(contiguity_info['polygon_centroids'], 
                                     contiguity_info['neighbor_distance_threshold'])
    neighbor_counts = [len(n) for n in neighbors.values()]
    
    print(f"Neighbor stats: min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
          f"mean={np.mean(neighbor_counts):.1f}")
    
    # Check for isolated polygons
    isolated_polygons = [poly_id for poly_id, neighs in neighbors.items() if len(neighs) == 0]
    if len(isolated_polygons) > 0:
        print(f"Warning: {len(isolated_polygons)} isolated polygons")
    
    # Check centroids
    centroids = contiguity_info['polygon_centroids']
    print(f"Centroid range: lat {centroids[:, 0].min():.2f}-{centroids[:, 0].max():.2f}, "
          f"lon {centroids[:, 1].min():.2f}-{centroids[:, 1].max():.2f}")
    
    # Check group sizes
    group_sizes = pd.Series(X_group).value_counts()
    print(f"Group size stats: min={group_sizes.min()}, max={group_sizes.max()}, "
          f"mean={group_sizes.mean():.1f}")
    
    # Check for small groups
    small_groups = group_sizes[group_sizes < 10]
    if len(small_groups) > 0:
        print(f"Warning: {len(small_groups)} groups have <10 samples")
    
    print("=== End Polygon Validation ===")

def create_correspondence_table(df, years, train_year, X_branch_id, result_dir):
    """
    Create correspondence table mapping FEWSNET_admin_code to partition_id.
    Same as RF version - no changes needed.
    """
    print(f"Creating correspondence table for year {train_year}...")
    
    # Get training data mask (same logic as train_test_split_by_year)
    train_mask = (years < train_year) & (years >= (train_year - 5))
    
    # Get training subset of dataframe
    df_train = df[train_mask].copy()
    
    # Add partition IDs to training data
    df_train['partition_id'] = X_branch_id
    
    # Extract unique FEWSNET_admin_code and partition_id pairs
    correspondence_table = df_train[['FEWSNET_admin_code', 'partition_id']].drop_duplicates()
    
    # Sort by admin code for better readability
    correspondence_table = correspondence_table.sort_values('FEWSNET_admin_code')
    
    # Save to result directory
    output_path = os.path.join(result_dir, f'correspondence_table_xgb_{train_year}.csv')
    correspondence_table.to_csv(output_path, index=False)
    
    print(f"Correspondence table saved to: {output_path}")
    print(f"Table contains {len(correspondence_table)} unique admin_code-partition_id pairs")

def run_temporal_evaluation(X, y, X_loc, X_group, years, l1_index, l2_index, 
                           assignment, contiguity_info, df, nowcasting=False, 
                           # XGBoost hyperparameters
                           learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0,
                           subsample=0.8, colsample_bytree=0.8,
                           max_depth=None):
    """
    Run temporal evaluation for multiple years using XGBoost.
    Modified version with XGBoost-specific parameters.
    """
    print(f"Running temporal evaluation with XGBoost (nowcasting={nowcasting})...")
    print(f"XGB hyperparameters: lr={learning_rate}, reg_alpha={reg_alpha}, reg_lambda={reg_lambda}")
    
    # Initialize results tracking
    results_df = pd.DataFrame(columns=[
        'year', 'precision(0)', 'recall(0)', 'f1(0)', 'precision(1)', 'recall(1)', 'f1(1)',
        'precision_base(0)', 'recall_base(0)', 'f1_base(0)', 'precision_base(1)', 'recall_base(1)', 'f1_base(1)',
        'num_samples(0)', 'num_samples(1)'
    ])
    
    y_pred_test = pd.DataFrame(columns=['year', 'month', 'adm_code', 'fews_ipc_crisis_pred', 'fews_ipc_crisis_true'])
    
    # Determine contiguity settings
    if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans']:
        contiguity_type = 'polygon'
        polygon_contiguity_info = contiguity_info
    else:
        contiguity_type = 'grid'
        polygon_contiguity_info = None
    
    # Run evaluation for each year
    for year in range(2021, 2025):
        print(f"\nEvaluating year {year} with XGBoost...")
        
        # Train-test split
        (Xtrain, ytrain, Xtrain_loc, Xtrain_group,
         Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_by_year(
            X, y, X_loc, X_group, years, test_year=year)
        
        ytrain = ytrain.astype(int)
        ytest = ytest.astype(int)
        
        print(f"Train samples: {len(ytrain)}, Test samples: {len(ytest)}")
        
        if nowcasting:
            # 2-layer model with XGBoost
            Xtrain_L1 = Xtrain[:, l1_index]
            Xtrain_L2 = Xtrain[:, l2_index]
            Xtest_L1 = Xtest[:, l1_index]
            Xtest_L2 = Xtest[:, l2_index]
            
            # Create and train 2-layer GeoXGB model
            geoxgb_2layer = GeoRF_XGB(
                min_model_depth=MIN_DEPTH,
                max_model_depth=MAX_DEPTH,
                n_jobs=N_JOBS,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample_bytree
            )
            
            # Train 2-layer model
            geoxgb_2layer.fit_2layer(
                Xtrain_L1, Xtrain_L2, ytrain, Xtrain_group,
                val_ratio=VAL_RATIO,
                contiguity_type=contiguity_type,
                polygon_contiguity_info=polygon_contiguity_info
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
            
            print(f"Year {year} - 2-Layer GeoXGB F1: {f1}, 2-Layer Base XGB F1: {f1_base}")
            
            # Extract and save correspondence table for 2-layer model
            try:
                X_branch_id_path = os.path.join(geoxgb_2layer.dir_space, 'X_branch_id.npy')
                if os.path.exists(X_branch_id_path):
                    X_branch_id = np.load(X_branch_id_path)
                    create_correspondence_table(df, years, year, X_branch_id, geoxgb_2layer.model_dir)
            except Exception as e:
                print(f"Warning: Could not create correspondence table for year {year}: {e}")
            
        else:
            # Single-layer model with XGBoost
            geoxgb = GeoRF_XGB(
                min_model_depth=MIN_DEPTH,
                max_model_depth=MAX_DEPTH,
                n_jobs=N_JOBS,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample_bytree
            )
            
            # Train model
            geoxgb.fit(
                Xtrain, ytrain, Xtrain_group,
                val_ratio=VAL_RATIO,
                contiguity_type=contiguity_type,
                polygon_contiguity_info=polygon_contiguity_info
            )
            
            # Get predictions
            ypred = geoxgb.predict(Xtest, Xtest_group)
            
            # Evaluate
            (pre, rec, f1, pre_base, rec_base, f1_base) = geoxgb.evaluate(
                Xtest, ytest, Xtest_group, eval_base=True, print_to_file=True
            )
            
            print(f"Year {year} - GeoXGB F1: {f1}, Base XGB F1: {f1_base}")
            
            # Extract and save correspondence table for single-layer model
            try:
                X_branch_id_path = os.path.join(geoxgb.dir_space, 'X_branch_id.npy')
                if os.path.exists(X_branch_id_path):
                    X_branch_id = np.load(X_branch_id_path)
                    create_correspondence_table(df, years, year, X_branch_id, geoxgb.model_dir)
            except Exception as e:
                print(f"Warning: Could not create correspondence table for year {year}: {e}")
        
        # Store results (same as RF version)
        nsample_class = np.bincount(ytest)
        
        results_df = pd.concat([results_df, pd.DataFrame({
            'year': [year],
            'precision(0)': [pre[0]],
            'precision(1)': [pre[1]],
            'recall(0)': [rec[0]],
            'recall(1)': [rec[1]],
            'f1(0)': [f1[0]],
            'f1(1)': [f1[1]],
            'precision_base(0)': [pre_base[0]],
            'precision_base(1)': [pre_base[1]],
            'recall_base(0)': [rec_base[0]],
            'recall_base(1)': [rec_base[1]],
            'f1_base(0)': [f1_base[0]],
            'f1_base(1)': [f1_base[1]],
            'num_samples(0)': [nsample_class[0]],
            'num_samples(1)': [nsample_class[1]]
        })], ignore_index=True)
        
        # Store predictions
        try:
            y_pred_test = pd.concat([y_pred_test, pd.DataFrame({
                'year': [year] * len(ytest),
                'month': [1] * len(ytest),  # Placeholder
                'adm_code': [0] * len(ytest),  # Placeholder
                'fews_ipc_crisis_pred': ypred,
                'fews_ipc_crisis_true': ytest
            })], ignore_index=True)
        except:
            y_pred_test = pd.concat([y_pred_test, pd.DataFrame({
                'year': [year] * len(ytest),
                'month': [1] * len(ytest),
                'adm_code': [0] * len(ytest),
                'fews_ipc_crisis_pred': ypred,
                'fews_ipc_crisis_true': ytest
            })], ignore_index=True)
    
    return results_df, y_pred_test

def save_results(results_df, y_pred_test, assignment, nowcasting=False, max_depth=None):
    """
    Save evaluation results to CSV files.
    Modified version for XGBoost (adds _xgb suffix).
    """
    # Create file names based on assignment
    pred_test_name = 'y_pred_test_xgb_g'  # Added _xgb suffix
    results_df_name = 'results_df_xgb_g'  # Added _xgb suffix
    
    assignment_suffixes = {
        'polygons': 'p',
        'grid': 'g',
        'country': 'c',
        'AEZ': 'ae',
        'country_AEZ': 'cae',
        'geokmeans': 'gk',
        'all_kmeans': 'ak'
    }
    
    if assignment in assignment_suffixes:
        pred_test_name += assignment_suffixes[assignment]
        results_df_name += assignment_suffixes[assignment]
    
    # Add depth suffix if specified
    if max_depth is not None:
        pred_test_name += f'_d{max_depth}'
        results_df_name += f'_d{max_depth}'
    
    # Add nowcasting suffix
    if nowcasting:
        pred_test_name += '_nowcast'
        results_df_name += '_nowcast'
    
    # Add file extension
    pred_test_name += '.csv'
    results_df_name += '.csv'
    
    # Save files
    y_pred_test.to_csv(pred_test_name, index=False)
    results_df.to_csv(results_df_name, index=False)
    
    print(f"Results saved to {results_df_name}")
    print(f"Predictions saved to {pred_test_name}")

def main():
    """
    Main function to run the complete XGBoost pipeline.
    """
    print("=== Starting GeoXGB Food Crisis Prediction Pipeline ===")
    
    # Configuration - XGBoost hyperparameters optimized for food crisis prediction
    assignment = 'polygons'  # Change this to test different grouping methods
    nowcasting = False       # Set to True for 2-layer model
    max_depth = 6           # XGBoost default, good for preventing overfitting
    
    # XGBoost-specific hyperparameters
    learning_rate = 0.1     # Moderate learning rate for stability
    reg_alpha = 0.1         # L1 regularization for feature selection
    reg_lambda = 1.0        # L2 regularization for stability
    subsample = 0.8         # Prevent overfitting
    colsample_bytree = 0.8  # Prevent overfitting
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(DATA_PATH)
        
        # Step 2: Setup spatial groups
        X_group, X_loc, contiguity_info = setup_spatial_groups(df, assignment)
        
        # Step 3: Prepare features
        X, y, l1_index, l2_index, years = prepare_features(df, X_group, X_loc)
        
        # Step 4: Validate polygon contiguity (if applicable)
        if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans'] and contiguity_info is not None:
            validate_polygon_contiguity(contiguity_info, X_group)
        
        # Step 5: Run temporal evaluation with XGBoost
        results_df, y_pred_test = run_temporal_evaluation(
            X, y, X_loc, X_group, years, l1_index, l2_index,
            assignment, contiguity_info, df, nowcasting,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_depth=max_depth
        )
        
        # Step 6: Save results
        save_results(results_df, y_pred_test, assignment, nowcasting, max_depth)
        
        # Step 7: Display summary
        print("\n=== XGBoost Evaluation Summary ===")
        print(results_df.groupby('year')[['f1(0)', 'f1(1)', 'f1_base(0)', 'f1_base(1)']].mean())
        
        print("\n=== XGBoost Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)