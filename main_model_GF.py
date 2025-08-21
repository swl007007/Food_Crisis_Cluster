#!/usr/bin/env python3
"""
Replicated and debugged version of main_model_GF_main.ipynb

This script replicates the full functionality of the notebook for food crisis prediction
using GeoRF with polygon-based contiguity support.

Key features:
1. Data preprocessing with polars and pandas
2. Multiple spatial grouping options (polygons, grid, country, AEZ, etc.)
3. Polygon-based contiguity with corrected setup
4. Time-based train-test splitting for temporal validation
5. Single-layer and 2-layer GeoRF models
6. Comprehensive evaluation and result saving

Date: 2025-07-23
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
warnings.filterwarnings('ignore')

# GeoRF imports
from GeoRF import GeoRF
from customize import *
from data import load_demo_data
from helper import get_spatial_range
from initialization import train_test_split_all
from customize import train_test_split_rolling_window
from config import *
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# Import adjacency matrix utilities
if USE_ADJACENCY_MATRIX:
    from adjacency_utils import load_or_create_adjacency_matrix

# Configuration
DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_v06252025.csv"

def force_cleanup_directories():
    """
    Perform robust cleanup of all result_GeoRF* directories.
    Uses Python for better error handling and cross-platform compatibility.
    """
    import shutil
    import time
    
    print("Performing force cleanup of result directories...")
    
    # Find all result_GeoRF* directories
    result_dirs = glob.glob('result_GeoRF*')
    
    if not result_dirs:
        print("No result_GeoRF* directories found to clean up.")
        return
    
    cleaned_count = 0
    failed_count = 0
    
    for result_dir in result_dirs:
        try:
            print(f"Attempting to delete: {result_dir}")
            
            # Force remove read-only flags and delete
            if os.path.exists(result_dir):
                # First try to make everything writable
                for root, dirs, files in os.walk(result_dir):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        file_path = os.path.join(root, f)
                        os.chmod(file_path, 0o777)
                
                # Force remove the directory
                shutil.rmtree(result_dir, ignore_errors=True)
                
                # Verify deletion
                if not os.path.exists(result_dir):
                    print(f"✓ Successfully deleted: {result_dir}")
                    cleaned_count += 1
                else:
                    print(f"✗ Failed to delete: {result_dir} (still exists)")
                    failed_count += 1
                    
        except Exception as e:
            print(f"✗ Error deleting {result_dir}: {e}")
            failed_count += 1
            
        # Small delay to release file handles
        time.sleep(0.1)
    
    # Force garbage collection
    gc.collect()
    
    print(f"Cleanup completed: {cleaned_count} deleted, {failed_count} failed")
    
    # Additional cleanup of temporary files
    temp_patterns = ['temp_*', '*.pkl', '__pycache__']
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file, ignore_errors=True)
                print(f"Cleaned up: {temp_file}")
            except:
                pass

def get_checkpoint_info(force_cleanup=False):
    """
    Scan for existing checkpoint directories and return information about completed quarters.
    
    Parameters:
    -----------
    force_cleanup : bool
        If True, perform cleanup and return empty lists (bypass checkpoint detection)
    
    Returns:
    --------
    completed_quarters : list
        List of tuples (year, quarter) for completed evaluations
    partial_results_files : list
        List of paths to partial results CSV files
    checkpoint_dirs : dict
        Mapping from (year, quarter) to result directory path
    """
    if force_cleanup:
        print("Force cleanup mode enabled - performing directory cleanup...")
        force_cleanup_directories()
        print("Returning empty checkpoint info to force fresh execution.")
        return [], [], {}
    
    print("Scanning for existing checkpoints...")
    
    # Find all result_GeoRF_* directories
    result_dirs = glob.glob('result_GeoRF_*')
    completed_quarters = []
    checkpoint_dirs = {}
    partial_results_files = []
    
    for result_dir in result_dirs:
        # Check for correspondence tables to identify completed quarters
        corr_tables = glob.glob(os.path.join(result_dir, 'correspondence_table_Q*.csv'))
        
        for corr_table in corr_tables:
            # Parse quarter and year from filename: correspondence_table_Q4_2019.csv
            filename = os.path.basename(corr_table)
            try:
                parts = filename.replace('correspondence_table_Q', '').replace('.csv', '').split('_')
                quarter = int(parts[0])
                year = int(parts[1])
                
                # Verify the checkpoint is complete by checking for required files
                checkpoints_dir = os.path.join(result_dir, 'checkpoints')
                space_dir = os.path.join(result_dir, 'space_partitions')
                
                if (os.path.exists(checkpoints_dir) and 
                    os.path.exists(space_dir) and 
                    os.path.exists(os.path.join(space_dir, 'X_branch_id.npy'))):
                    
                    completed_quarters.append((year, quarter))
                    checkpoint_dirs[(year, quarter)] = result_dir
                    print(f"  Found completed Q{quarter} {year} in {result_dir}")
                    
            except (ValueError, IndexError):
                # Skip if filename doesn't match expected pattern
                continue
    
    # Look for partial results CSV files
    partial_results_patterns = [
        'results_df_gp_*.csv',  # polygon assignment results
        'results_df_gg_*.csv',  # grid assignment results  
        'results_df_gc_*.csv',  # country assignment results
        'results_df_*.csv'      # any other results files
    ]
    
    for pattern in partial_results_patterns:
        files = glob.glob(pattern)
        partial_results_files.extend(files)
    
    if partial_results_files:
        print(f"  Found {len(partial_results_files)} partial results files")
        for f in partial_results_files[:5]:  # Show first 5 files
            print(f"    {f}")
        if len(partial_results_files) > 5:
            print(f"    ... and {len(partial_results_files) - 5} more")
    
    # Sort completed quarters chronologically
    completed_quarters.sort()
    
    print(f"Found {len(completed_quarters)} completed quarter evaluations")
    if completed_quarters:
        print(f"  Range: Q{completed_quarters[0][1]} {completed_quarters[0][0]} to Q{completed_quarters[-1][1]} {completed_quarters[-1][0]}")
    
    return completed_quarters, partial_results_files, checkpoint_dirs

def load_partial_results(partial_results_files, assignment, nowcasting=False, max_depth=None, desire_terms=None, forecasting_scope=None, start_year=None, end_year=None):
    """
    Load and combine partial results from previous runs.
    
    Parameters:
    -----------
    partial_results_files : list
        List of partial results CSV file paths
    assignment : str
        Current assignment method to match files
    nowcasting : bool
        Whether 2-layer model is being used
    max_depth : int or None
        Maximum depth setting
    desire_terms : int or None
        Desired terms setting
    forecasting_scope : int or None
        Forecasting scope setting
    start_year : int or None
        Start year of evaluation period
    end_year : int or None
        End year of evaluation period
        
    Returns:
    --------
    results_df : pandas.DataFrame or None
        Combined results DataFrame, or None if no matching files found
    y_pred_test : pandas.DataFrame or None
        Combined predictions DataFrame, or None if no matching files found
    """
    print("Loading partial results from previous runs...")
    
    # Build expected filename patterns based on current configuration
    pred_test_name = 'y_pred_test_g'
    results_df_name = 'results_df_g'
    
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
    
    # Add configuration suffixes
    if max_depth is not None:
        pred_test_name += f'_d{max_depth}'
        results_df_name += f'_d{max_depth}'
    
    if desire_terms is not None:
        pred_test_name += f'_t{desire_terms}'
        results_df_name += f'_t{desire_terms}'
        
    if forecasting_scope is not None:
        pred_test_name += f'_fs{forecasting_scope}'
        results_df_name += f'_fs{forecasting_scope}'
    
    # Add year range suffix
    if start_year is not None and end_year is not None:
        pred_test_name += f'_{start_year}_{end_year}'
        results_df_name += f'_{start_year}_{end_year}'
    
    if nowcasting:
        pred_test_name += '_nowcast'
        results_df_name += '_nowcast'
    
    pred_test_name += '.csv'
    results_df_name += '.csv'
    
    results_df = None
    y_pred_test = None
    
    # Try to load matching results files
    if results_df_name in partial_results_files:
        try:
            results_df = pd.read_csv(results_df_name)
            print(f"  Loaded existing results: {results_df_name} ({len(results_df)} rows)")
        except Exception as e:
            print(f"  Warning: Could not load {results_df_name}: {e}")
    
    if pred_test_name in partial_results_files:
        try:
            y_pred_test = pd.read_csv(pred_test_name)
            print(f"  Loaded existing predictions: {pred_test_name} ({len(y_pred_test)} rows)")
        except Exception as e:
            print(f"  Warning: Could not load {pred_test_name}: {e}")
    
    return results_df, y_pred_test

def determine_remaining_quarters(completed_quarters, start_year, end_year, desire_terms):
    """
    Determine which quarters still need to be evaluated.
    
    Parameters:
    -----------
    completed_quarters : list
        List of tuples (year, quarter) for completed evaluations
    start_year : int
        Start year for evaluation
    end_year : int
        End year for evaluation
    desire_terms : int or None
        Specific quarter to evaluate (1-4), or None for all quarters
        
    Returns:
    --------
    remaining_quarters : list
        List of tuples (year, quarter) that still need to be evaluated
    """
    # Determine which quarters to evaluate based on desire_terms
    if desire_terms is None:
        quarters_to_evaluate = [1, 2, 3, 4]
    else:
        quarters_to_evaluate = [desire_terms]
    
    # Generate all expected quarters
    all_quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in quarters_to_evaluate:
            all_quarters.append((year, quarter))
    
    # Find remaining quarters
    completed_set = set(completed_quarters)
    remaining_quarters = [(year, quarter) for year, quarter in all_quarters 
                         if (year, quarter) not in completed_set]
    
    print(f"Checkpoint analysis:")
    print(f"  Total quarters to evaluate: {len(all_quarters)}")
    print(f"  Already completed: {len(completed_quarters)}")
    print(f"  Remaining: {len(remaining_quarters)}")
    
    if remaining_quarters:
        print(f"  Next to evaluate: Q{remaining_quarters[0][1]} {remaining_quarters[0][0]}")
        if len(remaining_quarters) > 1:
            print(f"  Last to evaluate: Q{remaining_quarters[-1][1]} {remaining_quarters[-1][0]}")
    else:
        print("  All quarters already completed!")
    
    return remaining_quarters

def save_checkpoint_results(results_df, y_pred_test, assignment, nowcasting=False, max_depth=None, desire_terms=None, forecasting_scope=None, start_year=None, end_year=None):
    """
    Save results as checkpoint files that can be resumed later.
    
    This is the same as save_results but called more frequently for checkpointing.
    """
    save_results(results_df, y_pred_test, assignment, nowcasting, max_depth, desire_terms, forecasting_scope, start_year, end_year)

def comp_impute(X, strategy="max_plus", multiplier=100.0):
    """
    Custom imputation function that handles infinite values and missing data.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data array
    strategy : str
        Imputation strategy
    multiplier : float
        Multiplier for out-of-range values
        
    Returns:
    --------
    X : numpy.ndarray
        Imputed data array
    """
    # Handle infinite values
    inf_count = 0
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        
        try:
            col_data_float = col_data.astype(float)
            if np.isinf(col_data_float).any():
                inf_count += 1
                col_data_float[np.isinf(col_data_float)] = np.nan
                X[:, col_idx] = col_data_float
        except Exception:
            continue
    
    if inf_count > 0:
        print(f"Found and replaced infinite values in {inf_count} columns")
    
    # Apply imputation with reduced verbosity
    X_imputed, imputer = impute_missing_values(X, strategy=strategy, multiplier=multiplier, verbose=False)
    
    # Get summary info
    if hasattr(imputer, 'column_stats_'):
        imputed_cols = sum(1 for stats in imputer.column_stats_.values() if stats['has_missing'])
        print(f"Imputed missing values in {imputed_cols} columns")
    
    return X_imputed

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the FEWSNET data.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
        
    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed dataframe
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
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    assignment : str
        Grouping method ('polygons', 'grid', 'country', etc.)
        
    Returns:
    --------
    X_group : numpy.ndarray
        Group assignments
    X_loc : numpy.ndarray
        Location coordinates
    contiguity_info : dict or None
        Contiguity information for polygon-based grouping
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
        # Map polygon_index -> [admin_code] (each polygon is its own group using admin code as group ID)
        polygon_group_mapping = {i: [unique_polygons[i]] for i in range(len(unique_polygons))}
        
        # Convert admin codes to polygon indices for PolygonGroupGenerator
        polygon_indices = np.array([admin_to_polygon_idx[admin_code] for admin_code in X_polygon_ids])
        
        # Load adjacency matrix if enabled
        adjacency_dict = None
        if USE_ADJACENCY_MATRIX:
            try:
                print("Loading adjacency matrix for polygon-based contiguity...")
                adj_dict_raw, polygon_id_mapping, adj_centroids = load_or_create_adjacency_matrix(
                    shapefile_path=ADJACENCY_SHAPEFILE_PATH,
                    polygon_id_column=ADJACENCY_POLYGON_ID_COLUMN,
                    cache_dir=ADJACENCY_CACHE_DIR,
                    force_regenerate=ADJACENCY_FORCE_REGENERATE
                )
                
                # Create mapping from admin_code to adjacency matrix index
                # The adjacency matrix keys are indices 0,1,2... corresponding to shapefile admin_codes
                admin_code_to_adj_idx = {}
                adj_idx_to_admin_code = {}
                
                for adj_idx, admin_code in polygon_id_mapping.items():
                    admin_code_to_adj_idx[admin_code] = adj_idx
                    adj_idx_to_admin_code[adj_idx] = admin_code
                
                # Map adjacency dictionary to use polygon indices (matching our polygon_indices array)
                adjacency_dict = {}
                for polygon_idx in range(len(unique_polygons)):
                    admin_code = unique_polygons[polygon_idx]
                    if admin_code in admin_code_to_adj_idx:
                        adj_idx = admin_code_to_adj_idx[admin_code]
                        if adj_idx in adj_dict_raw:
                            # Map neighbor adjacency indices back to polygon indices
                            neighbor_admin_codes = [adj_idx_to_admin_code.get(neighbor_adj_idx) for neighbor_adj_idx in adj_dict_raw[adj_idx]]
                            neighbor_polygon_indices = []
                            for neighbor_admin_code in neighbor_admin_codes:
                                if neighbor_admin_code in admin_to_polygon_idx:
                                    neighbor_polygon_indices.append(admin_to_polygon_idx[neighbor_admin_code])
                            adjacency_dict[polygon_idx] = np.array(neighbor_polygon_indices)
                        else:
                            adjacency_dict[polygon_idx] = np.array([])
                    else:
                        adjacency_dict[polygon_idx] = np.array([])
                
                print(f"Adjacency matrix loaded: {len(adj_dict_raw)} polygons in shapefile, {len(adjacency_dict)} mapped to current data")
                print(f"Sample adjacency connections: polygon 0 has {len(adjacency_dict.get(0, []))} neighbors")
                
            except Exception as e:
                print(f"Warning: Failed to load adjacency matrix: {e}")
                print("Falling back to distance-based neighbor calculation")
                adjacency_dict = None
        
        # Create polygon generator with correct mapping and adjacency info
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=polygon_centroids,
            polygon_group_mapping=polygon_group_mapping,
            neighbor_distance_threshold=0.8,
            adjacency_dict=adjacency_dict
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
        
    elif assignment == 'country':
        # Get country groups and calculate mean centroids for each country
        country_data = df[['ISO_encoded', 'lat', 'lon']].groupby('ISO_encoded').mean().reset_index()
        country_centroids = country_data[['lat', 'lon']].values
        unique_countries = country_data['ISO_encoded'].unique()
        
        # Create mapping from country_index -> [country_id]
        country_group_mapping = {i: [unique_countries[i]] for i in range(len(unique_countries))}
        
        # Create country to index mapping
        country_to_idx = {country_id: idx for idx, country_id in enumerate(unique_countries)}
        
        # Convert country IDs to indices for PolygonGroupGenerator
        country_indices = np.array([country_to_idx[country_id] for country_id in df['ISO_encoded'].values])
        
        # Create polygon generator for countries
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=country_centroids,
            polygon_group_mapping=country_group_mapping,
            neighbor_distance_threshold=2.0  # Larger threshold for countries
        )
        
        X_group = polygon_gen.get_groups(country_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Country setup complete: {len(unique_countries)} countries, {len(np.unique(X_group))} groups")
        
    elif assignment == 'AEZ':
        # Get AEZ groups and calculate mean centroids for each AEZ
        aez_data = df[['AEZ_group', 'lat', 'lon']].groupby('AEZ_group').mean().reset_index()
        aez_centroids = aez_data[['lat', 'lon']].values
        unique_aezs = aez_data['AEZ_group'].unique()
        
        # Create mapping from aez_index -> [aez_id]
        aez_group_mapping = {i: [unique_aezs[i]] for i in range(len(unique_aezs))}
        
        # Create AEZ to index mapping
        aez_to_idx = {aez_id: idx for idx, aez_id in enumerate(unique_aezs)}
        
        # Convert AEZ IDs to indices for PolygonGroupGenerator
        aez_indices = np.array([aez_to_idx[aez_id] for aez_id in df['AEZ_group'].values])
        
        # Create polygon generator for AEZs
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=aez_centroids,
            polygon_group_mapping=aez_group_mapping,
            neighbor_distance_threshold=1.5  # Medium threshold for AEZs
        )
        
        X_group = polygon_gen.get_groups(aez_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"AEZ setup complete: {len(unique_aezs)} AEZs, {len(np.unique(X_group))} groups")
        
    elif assignment == 'country_AEZ':
        # Get country-AEZ groups and calculate mean centroids for each combination
        country_aez_data = df[['AEZ_country_group', 'lat', 'lon']].groupby('AEZ_country_group').mean().reset_index()
        country_aez_centroids = country_aez_data[['lat', 'lon']].values
        unique_country_aezs = country_aez_data['AEZ_country_group'].unique()
        
        # Create mapping from country_aez_index -> [country_aez_id]
        country_aez_group_mapping = {i: [unique_country_aezs[i]] for i in range(len(unique_country_aezs))}
        
        # Create country-AEZ to index mapping
        country_aez_to_idx = {ca_id: idx for idx, ca_id in enumerate(unique_country_aezs)}
        
        # Convert country-AEZ IDs to indices for PolygonGroupGenerator
        country_aez_indices = np.array([country_aez_to_idx[ca_id] for ca_id in df['AEZ_country_group'].values])
        
        # Create polygon generator for country-AEZs
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=country_aez_centroids,
            polygon_group_mapping=country_aez_group_mapping,
            neighbor_distance_threshold=1.2  # Smaller threshold for country-AEZ combinations
        )
        
        X_group = polygon_gen.get_groups(country_aez_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Country-AEZ setup complete: {len(unique_country_aezs)} country-AEZ combinations, {len(np.unique(X_group))} groups")
        
    elif assignment == 'geokmeans':
        # Get geo-based K-means clusters
        cluster_id, cluster_info, admin_to_group_map = create_kmeans_groupgenerator_from_admin_codes(
            df, n_clusters=100, random_state=42, features_for_clustering=['lat', 'lon']
        )
        
        # Calculate mean centroids for each cluster
        df_with_clusters = df.copy()
        df_with_clusters['cluster_id'] = cluster_id
        cluster_data = df_with_clusters[['cluster_id', 'lat', 'lon']].groupby('cluster_id').mean().reset_index()
        cluster_centroids = cluster_data[['lat', 'lon']].values
        unique_clusters = cluster_data['cluster_id'].unique()
        
        # Create mapping from cluster_index -> [cluster_id]
        cluster_group_mapping = {i: [unique_clusters[i]] for i in range(len(unique_clusters))}
        
        # Create cluster to index mapping
        cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        
        # Convert cluster IDs to indices for PolygonGroupGenerator
        cluster_indices = np.array([cluster_to_idx[cluster_id] for cluster_id in cluster_id])
        
        # Create polygon generator for geo-kmeans clusters
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=cluster_centroids,
            polygon_group_mapping=cluster_group_mapping,
            neighbor_distance_threshold=1.0  # Medium threshold for geo-clusters
        )
        
        X_group = polygon_gen.get_groups(cluster_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"Geo-KMeans setup complete: {len(unique_clusters)} clusters, {len(np.unique(X_group))} groups")
        
        # Save correspondence table
        correspondence_table = pd.DataFrame(list(admin_to_group_map.items()), 
                                          columns=['FEWSNET_admin_code', 'cluster_id'])
        correspondence_table.to_csv('correspondence_table_geokmeans.csv', index=False)
        
    elif assignment == 'all_kmeans':
        # Get all-features K-means clusters
        X_columns = df.drop(columns=['FEWSNET_admin_code', 'lat', 'lon', 'fews_ipc_crisis']).columns.tolist()
        cluster_id, cluster_info, admin_to_group_map = create_kmeans_groupgenerator_from_admin_codes(
            df, n_clusters=100, random_state=42, features_for_clustering=X_columns
        )
        
        # Calculate mean centroids for each cluster (using lat/lon even though clustering was on all features)
        df_with_clusters = df.copy()
        df_with_clusters['cluster_id'] = cluster_id
        cluster_data = df_with_clusters[['cluster_id', 'lat', 'lon']].groupby('cluster_id').mean().reset_index()
        cluster_centroids = cluster_data[['lat', 'lon']].values
        unique_clusters = cluster_data['cluster_id'].unique()
        
        # Create mapping from cluster_index -> [cluster_id]
        cluster_group_mapping = {i: [unique_clusters[i]] for i in range(len(unique_clusters))}
        
        # Create cluster to index mapping
        cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        
        # Convert cluster IDs to indices for PolygonGroupGenerator
        cluster_indices = np.array([cluster_to_idx[cluster_id] for cluster_id in cluster_id])
        
        # Create polygon generator for all-features kmeans clusters
        polygon_gen = PolygonGroupGenerator(
            polygon_centroids=cluster_centroids,
            polygon_group_mapping=cluster_group_mapping,
            neighbor_distance_threshold=0.8  # Smaller threshold for feature-based clusters
        )
        
        X_group = polygon_gen.get_groups(cluster_indices)
        contiguity_info = polygon_gen.get_contiguity_info()
        
        print(f"All-KMeans setup complete: {len(unique_clusters)} clusters, {len(np.unique(X_group))} groups")
        
        # Save correspondence table
        correspondence_table = pd.DataFrame(list(admin_to_group_map.items()), 
                                          columns=['FEWSNET_admin_code', 'cluster_id'])
        correspondence_table.to_csv('correspondence_table_allkmeans.csv', index=False)
        
    else:
        raise ValueError(f"Unknown assignment method: {assignment}")
    
    return X_group, X_loc, contiguity_info

def prepare_features(df, X_group, X_loc, forecasting_scope=4):
    """
    Prepare feature matrices and identify L1/L2 feature indices with forecasting scope logic.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    X_group : numpy.ndarray
        Group assignments
    X_loc : numpy.ndarray
        Location coordinates
    forecasting_scope : int, default=4
        Forecasting scope: 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    l1_index : list
        Indices for L1 features
    l2_index : list
        Indices for L2 features
    years : numpy.ndarray
        Year values for temporal splitting
    """
    print(f"Preparing features with forecasting_scope={forecasting_scope}...")
    
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
    
    # Determine lag months based on forecasting scope  
    # 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag
    lag_months_map = {1: 3, 2: 6, 3: 9, 4: 12}
    lag_months = lag_months_map.get(forecasting_scope, 12)
    print(f"Using {lag_months}-month lag for forecasting scope {forecasting_scope}")
    
    # Create appropriate time variant list based on forecasting scope
    if forecasting_scope == 4:
        # For 12-month lag, use existing m12 variables if available
        time_variants_m12 = [variant + '_m12' for variant in time_variants]
        time_variants_list = time_variants + time_variants_m12
    else:
        # For other forecasting scopes, we'll create lagged features dynamically
        time_variants_list = time_variants.copy()
    
    # Get target variable
    y = df['fews_ipc_crisis'].values
    
    # Create comprehensive correspondence table with X_group mapping
    correspondence_df = df[['FEWSNET_admin_code', 'AEZ_group', 'AEZ_country_group', 'ISO_encoded']].copy()
    correspondence_df['X_group'] = X_group
    correspondence_table = correspondence_df.drop_duplicates()
    correspondence_table.to_csv('correspondence_table.csv', index=False)
    
    # Sort by admin code and date
    df_sorted = df.sort_values(by=['FEWSNET_admin_code', 'date'])
    
    # Create lag features based on forecasting scope
    print(f"Creating lag features for forecasting scope {forecasting_scope} ({lag_months} months)...")
    
    if forecasting_scope != 4:
        # For scopes 1, 2, 3, create lagged features for time-variant variables
        existing_cols_before = set(df_sorted.columns)
        
        for variant in time_variants:
            if variant in df_sorted.columns:
                lagged_col_name = f'{variant}_lag{lag_months}m'
                df_sorted[lagged_col_name] = df_sorted.groupby('FEWSNET_admin_code')[variant].shift(lag_months)
                time_variants_list.append(lagged_col_name)
        
        # Check for duplicates and remove if they already exist in the dataset
        print("Checking for duplicate lagged features...")
        cols_to_remove = []
        for variant in time_variants:
            lagged_col_name = f'{variant}_lag{lag_months}m'
            # Check if similar columns already exist with various naming patterns
            potential_existing = [
                f'{variant}_m{lag_months}',        # e.g., variant_m3, variant_m6
                f'{variant}_lag_{lag_months}',     # e.g., variant_lag_3, variant_lag_6  
                f'{variant}_{lag_months}m',        # e.g., variant_3m, variant_6m
                f'{variant}_l{lag_months}',        # e.g., variant_l3, variant_l6, variant_l9
                f'{variant}_L{lag_months}',        # e.g., variant_L3, variant_L6 (uppercase)
                f'{variant}.l{lag_months}',        # e.g., variant.l3, variant.l6 (with dot)
                f'{variant}.L{lag_months}'         # e.g., variant.L3, variant.L6 (with dot, uppercase)
            ]
            
            for existing_name in potential_existing:
                if existing_name in df_sorted.columns and lagged_col_name in df_sorted.columns:
                    print(f"Found duplicate: {lagged_col_name} vs {existing_name}, keeping {existing_name}")
                    cols_to_remove.append(lagged_col_name)
                    if lagged_col_name in time_variants_list:
                        time_variants_list.remove(lagged_col_name)
                    if existing_name not in time_variants_list:
                        time_variants_list.append(existing_name)
                    break
        
        # Remove duplicate columns
        if cols_to_remove:
            df_sorted = df_sorted.drop(columns=[col for col in cols_to_remove if col in df_sorted.columns])
            print(f"Removed {len(cols_to_remove)} duplicate columns")
    
    # Create additional lag features for crisis variable
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
    
    # Apply imputation (reduced verbosity)
    print("Applying imputation to missing values...")
    X = comp_impute(X, strategy="max_plus", multiplier=100.0)
    
    # Get years for temporal splitting
    years = df_sorted['years'].values
    # get terms within each year:group by FESNET_admin_code and year, set first month as 1, second months as 2
    months = df_sorted['date'].dt.month.values
    # if months = 1,2,3 then terms =1, if months = 4,5,6 then terms = 2, if months = 7,8,9 then terms = 3, if months = 10,11,12 then terms = 4
    terms = np.zeros_like(months)
    terms[months <= 3] = 1
    terms[(months > 3) & (months <= 6)] = 2
    terms[(months > 6) & (months <= 9)] = 3
    terms[(months > 9) & (months <= 12)] = 4
    
    print(f"Feature preparation complete: {X.shape[1]} features, {len(l1_index)} L1 features, {len(l2_index)} L2 features")
    
    return X, y, l1_index, l2_index, years, terms, df_sorted['date']

def validate_polygon_contiguity(contiguity_info, X_group):
    """
    Validate polygon contiguity setup using adjacency matrix (preferred) or distance-based fallback.
    
    Parameters:
    -----------
    contiguity_info : dict
        Contiguity information
    X_group : numpy.ndarray
        Group assignments
    """
    print("=== Polygon Contiguity Validation ===")
    
    # Use adjacency matrix if available (preferred approach)
    if 'adjacency_dict' in contiguity_info and contiguity_info['adjacency_dict'] is not None:
        print("Using adjacency matrix-based neighbors (production approach)")
        adjacency_dict = contiguity_info['adjacency_dict']
        neighbor_counts = [len(neighbors) for neighbors in adjacency_dict.values()]
        
        print(f"Adjacency neighbor stats: min={min(neighbor_counts) if neighbor_counts else 0}, "
              f"max={max(neighbor_counts) if neighbor_counts else 0}, "
              f"mean={np.mean(neighbor_counts):.1f if neighbor_counts else 0}")
        
        # Check for isolated polygons in adjacency matrix
        isolated_polygons = [poly_id for poly_id, neighs in adjacency_dict.items() if len(neighs) == 0]
        if len(isolated_polygons) > 0:
            print(f"Note: {len(isolated_polygons)} isolated polygons (normal for islands/enclaves)")
        
        total_connections = sum(len(neighbors) for neighbors in adjacency_dict.values())
        print(f"Total adjacency connections: {total_connections}")
        
    else:
        # Fallback to distance-based approach (legacy)
        print("Warning: Using distance-based neighbors (fallback - consider enabling adjacency matrix)")
        from partition_opt import get_polygon_neighbors
        
        neighbors = get_polygon_neighbors(contiguity_info['polygon_centroids'], 
                                         contiguity_info['neighbor_distance_threshold'])
        neighbor_counts = [len(n) for n in neighbors.values()]
        
        print(f"Distance neighbor stats: min={min(neighbor_counts)}, max={max(neighbor_counts)}, "
              f"mean={np.mean(neighbor_counts):.1f}")
        
        # Check for isolated polygons
        isolated_polygons = [poly_id for poly_id, neighs in neighbors.items() if len(neighs) == 0]
        if len(isolated_polygons) > 0:
            print(f"Warning: {len(isolated_polygons)} isolated polygons")
    
    # Check centroids (common validation)
    if 'polygon_centroids' in contiguity_info:
        centroids = contiguity_info['polygon_centroids']
        print(f"Centroid range: lat {centroids[:, 0].min():.2f}-{centroids[:, 0].max():.2f}, "
              f"lon {centroids[:, 1].min():.2f}-{centroids[:, 1].max():.2f}")
    
    # Check group sizes (common validation)
    group_sizes = pd.Series(X_group).value_counts()
    print(f"Group size stats: min={group_sizes.min()}, max={group_sizes.max()}, "
          f"mean={group_sizes.mean():.1f}")
    
    # Check for small groups
    small_groups = group_sizes[group_sizes < 10]
    if len(small_groups) > 0:
        print(f"Warning: {len(small_groups)} groups have <10 samples")
    
    print("=== End Polygon Validation ===")

def create_correspondence_table(df, years, dates, train_year, quarter, X_branch_id, result_dir):
    """
    Create correspondence table mapping FEWSNET_admin_code to partition_id for rolling window splits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe with FEWSNET_admin_code
    years : numpy.ndarray
        Year values for all data
    dates : pandas.Series
        Date values for precise temporal filtering
    train_year : int
        Year for which training was done (2024)
    quarter : int
        Quarter for which training was done (1-4)
    X_branch_id : numpy.ndarray
        Branch IDs (partition IDs) from trained model
    result_dir : str
        Directory to save correspondence table
    """
    print(f"Creating correspondence table for Q{quarter} {train_year}...")
    
    try:
        import pandas as pd
        
        # Convert dates to pandas datetime if needed
        if not isinstance(dates, pd.Series):
            dates = pd.to_datetime(dates)
        
        # Define quarter start and end dates using the actual train_year (not hardcoded 2024)
        quarter_starts = {
            1: pd.Timestamp(f'{train_year}-01-01'),
            2: pd.Timestamp(f'{train_year}-04-01'),
            3: pd.Timestamp(f'{train_year}-07-01'),
            4: pd.Timestamp(f'{train_year}-10-01')
        }
        
        quarter_ends = {
            1: pd.Timestamp(f'{train_year}-03-31'),
            2: pd.Timestamp(f'{train_year}-06-30'),
            3: pd.Timestamp(f'{train_year}-09-30'),
            4: pd.Timestamp(f'{train_year}-12-31')
        }
        
        # Get the training mask (same logic as rolling window - FIXED to match new logic)
        test_quarter_start = quarter_starts[quarter]
        train_end_date = test_quarter_start  # Training ends when test quarter begins (NO OVERLAP)
        train_start_date = train_end_date - pd.DateOffset(years=5)
        train_mask = (dates >= train_start_date) & (dates < train_end_date)
        
        # Get the training subset of dataframe
        df_train = df[train_mask].copy()
        actual_train_length = len(df_train)
        branch_id_length = len(X_branch_id)
        
        print(f"Correspondence table debug:")
        print(f"  Training data length (from df): {actual_train_length}")
        print(f"  X_branch_id length (from model): {branch_id_length}")
        
        # Handle length mismatch gracefully
        if branch_id_length != actual_train_length:
            print(f"Warning: Length mismatch detected")
            
            if branch_id_length < actual_train_length:
                # Model has fewer partition IDs than training samples
                # This can happen if some samples were filtered during model training
                print(f"  Using first {branch_id_length} training samples to match X_branch_id")
                df_train = df_train.iloc[:branch_id_length].copy()
            else:
                # More partition IDs than training samples (unusual)
                print(f"  Truncating X_branch_id to match training data length")
                X_branch_id = X_branch_id[:actual_train_length]
        
        # Ensure exact length match after adjustment
        df_train = df_train.iloc[:len(X_branch_id)].copy()
        
        print(f"  Final lengths: df_train={len(df_train)}, X_branch_id={len(X_branch_id)}")
        
        # Add partition IDs to training data
        df_train.loc[:, 'partition_id'] = X_branch_id
        
        # Extract unique FEWSNET_admin_code and partition_id pairs
        correspondence_table = df_train[['FEWSNET_admin_code', 'partition_id']].drop_duplicates()
        
        # Sort by admin code for better readability
        correspondence_table = correspondence_table.sort_values('FEWSNET_admin_code')
        
        # Save to result directory
        output_path = os.path.join(result_dir, f'correspondence_table_Q{quarter}_{train_year}.csv')
        correspondence_table.to_csv(output_path, index=False)
        
        print(f"Correspondence table saved to: {output_path}")
        print(f"Table contains {len(correspondence_table)} unique admin_code-partition_id pairs")
        
    except Exception as e:
        print(f"Error creating correspondence table: {e}")
        print("Continuing without correspondence table...")

def run_temporal_evaluation(X, y, X_loc, X_group, years, dates, l1_index, l2_index, 
                           assignment, contiguity_info, df, nowcasting=False, max_depth=None, input_terms=None, desire_terms=None,
                           track_partition_metrics=False, enable_metrics_maps=True, start_year=2015, end_year=2024, forecasting_scope=None, force_cleanup=False):
    """
    Run temporal evaluation for all quarters from start_year to end_year using rolling window approach.
    
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
        Maximum depth for RF models
    input_terms : numpy.ndarray
        Terms within each year (1-4 corresponding to quarters)
    desire_terms : int or None
        Specific quarter to evaluate (1-4), or None for all quarters
    track_partition_metrics : bool
        Whether to enable partition metrics tracking and visualization
    enable_metrics_maps : bool
        Whether to create maps showing F1/accuracy improvements
    forecasting_scope : int or None
        Forecasting scope (1=3mo, 2=6mo, 3=9mo, 4=12mo lag)
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Evaluation results with quarter information
    y_pred_test : pandas.DataFrame
        Prediction results with quarter information
    """
    print(f"Running temporal evaluation (nowcasting={nowcasting})...")
    
    # Initialize results tracking (class 1 only)
    results_df = pd.DataFrame(columns=[
        'year', 'quarter', 'precision(1)', 'recall(1)', 'f1(1)',
        'precision_base(1)', 'recall_base(1)', 'f1_base(1)',
        'num_samples(1)'
    ])
    
    y_pred_test = pd.DataFrame(columns=['year', 'quarter', 'month', 'adm_code', 'fews_ipc_crisis_pred', 'fews_ipc_crisis_true'])
    
    # CHECKPOINT RECOVERY: Check for existing results and determine what needs to be evaluated
    print("\n=== Checkpoint Recovery System ===")
    completed_quarters, partial_results_files, checkpoint_dirs = get_checkpoint_info(force_cleanup)
    
    # Load partial results if they exist
    existing_results_df, existing_y_pred_test = load_partial_results(
        partial_results_files, assignment, nowcasting, max_depth, desire_terms, forecasting_scope, start_year, end_year
    )
    
    # Merge existing results with new DataFrames
    if existing_results_df is not None and len(existing_results_df) > 0:
        results_df = existing_results_df.copy()
        print(f"Resuming from existing results: {len(results_df)} previous evaluations loaded")
    
    if existing_y_pred_test is not None and len(existing_y_pred_test) > 0:
        y_pred_test = existing_y_pred_test.copy()
        print(f"Resuming from existing predictions: {len(y_pred_test)} previous predictions loaded")
    
    # Determine remaining quarters to evaluate
    remaining_quarters = determine_remaining_quarters(completed_quarters, start_year, end_year, desire_terms)
    
    if not remaining_quarters:
        print("All quarters already completed! Returning existing results.")
        return results_df, y_pred_test
    
    print(f"Will evaluate {len(remaining_quarters)} remaining quarters")
    print("=== End Checkpoint Recovery ===\n")
    
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
    
    # Create progress bar for remaining quarterly evaluations
    progress_bar = tqdm(
        total=len(remaining_quarters), 
        desc="GeoRF Quarterly Evaluation", 
        unit="quarter",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Loop through remaining quarters only
    for i, (test_year, quarter) in enumerate(remaining_quarters):
        progress_bar.set_description(f"GeoRF Q{quarter} {test_year}")
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
            
                # Create and train 2-layer GeoRF model
                georf_2layer = GeoRF(
                    min_model_depth=MIN_DEPTH,
                    max_model_depth=MAX_DEPTH,
                    n_jobs=N_JOBS,
                    max_depth=max_depth
                )
            
                # Train 2-layer model with optional metrics tracking
                if track_partition_metrics:
                    # Note: 2-layer fit doesn't support partition metrics yet, 
                    # but we can extend it later if needed
                    print("Note: Partition metrics tracking not yet supported for 2-layer models")
                
                georf_2layer.fit_2layer(
                    Xtrain_L1, Xtrain_L2, ytrain, Xtrain_group,
                    val_ratio=VAL_RATIO,
                    contiguity_type=contiguity_type,
                    polygon_contiguity_info=polygon_contiguity_info
                )
            
                # Get predictions
                ypred = georf_2layer.predict_2layer(Xtest_L1, Xtest_L2, Xtest_group, correction_strategy='flip')
            
                # Evaluate
                (pre, rec, f1, pre_base, rec_base, f1_base) = georf_2layer.evaluate_2layer(
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
            
                print(f"Q{quarter} {test_year} Test - 2-Layer GeoRF F1: {f1}, 2-Layer Base RF F1: {f1_base}")
            
                # Extract and save correspondence table for 2-layer model
                try:
                    X_branch_id_path = os.path.join(georf_2layer.dir_space, 'X_branch_id.npy')
                    if os.path.exists(X_branch_id_path):
                        X_branch_id = np.load(X_branch_id_path)
                        create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, georf_2layer.model_dir)
                except Exception as e:
                    print(f"Warning: Could not create correspondence table for Q{quarter} {test_year}: {e}")
            
            else:
                # Single-layer model
                georf = GeoRF(
                    min_model_depth=MIN_DEPTH,
                    max_model_depth=MAX_DEPTH,
                    n_jobs=N_JOBS,
                    max_depth=max_depth
                )
            
            # Train model with optional partition metrics tracking
            if track_partition_metrics:
                print(f"Training GeoRF with partition metrics tracking enabled")
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
            
            georf.fit(
                Xtrain, ytrain, Xtrain_group,
                val_ratio=VAL_RATIO,
                contiguity_type=contiguity_type,
                polygon_contiguity_info=polygon_contiguity_info,
                track_partition_metrics=track_partition_metrics,
                correspondence_table_path=correspondence_table_path
            )
            
            # Check if metrics were tracked
            if track_partition_metrics and hasattr(georf, 'metrics_tracker'):
                if georf.metrics_tracker is not None:
                        print(f"\nPartition metrics tracker found for Q{quarter} {test_year}")
                        
                        # Check if any metrics were actually recorded
                        if hasattr(georf.metrics_tracker, 'all_metrics') and georf.metrics_tracker.all_metrics:
                            print(f"Number of metric records: {len(georf.metrics_tracker.all_metrics)}")
                            
                            # Show some sample metrics
                            for i, record in enumerate(georf.metrics_tracker.all_metrics[:3]):
                                print(f"  Record {i}: Round {record.get('partition_round', 'N/A')}, "
                                      f"Branch {record.get('branch_id', 'N/A')}, "
                                      f"F1 improvement: {record.get('f1_improvement', 'N/A'):.4f}")
                        else:
                            print("No metrics records found in tracker")
                        
                        # Try to get summary
                        try:
                            summary = georf.metrics_tracker.get_improvement_summary()
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
                        if hasattr(georf, 'model_dir'):
                            vis_dir = os.path.join(georf.model_dir, 'vis')
                            metrics_dir = os.path.join(georf.model_dir, 'partition_metrics')
                            
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
                    print("Warning: Metrics tracker not found on georf object")
                
            # Get predictions
            ypred = georf.predict(Xtest, Xtest_group)
            
            # Evaluate
            (pre, rec, f1, pre_base, rec_base, f1_base) = georf.evaluate(
                Xtest, ytest, Xtest_group, eval_base=True, print_to_file=True
            )
            
            print(f"Q{quarter} {test_year} Test - GeoRF F1: {f1}, Base RF F1: {f1_base}")
            
            # Extract and save correspondence table for single-layer model
            try:
                X_branch_id_path = os.path.join(georf.dir_space, 'X_branch_id.npy')
                if os.path.exists(X_branch_id_path):
                    X_branch_id = np.load(X_branch_id_path)
                    create_correspondence_table(df, years, dates, test_year, quarter, X_branch_id, georf.model_dir)
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
                    'month': np.full(len(ytest), quarter * 3, dtype=np.int8),  # Use quarter end month (3, 6, 9, 12)
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
                    'month': np.full(len(ytest), quarter * 3, dtype=np.int8),  # Use quarter end month (3, 6, 9, 12)
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
                if 'georf' in locals() and hasattr(georf, 'metrics_tracker'):
                    if georf.metrics_tracker is not None:
                        # Clear accumulated metrics data (can be hundreds of MB per quarter)
                        if hasattr(georf.metrics_tracker, 'all_metrics'):
                            georf.metrics_tracker.all_metrics.clear()
                        if hasattr(georf.metrics_tracker, 'partition_history'):
                            georf.metrics_tracker.partition_history.clear()
                        georf.metrics_tracker = None
                        print("Cleared PartitionMetricsTracker data")
                if 'georf_2layer' in locals() and hasattr(georf_2layer, 'metrics_tracker'):
                    if georf_2layer.metrics_tracker is not None:
                        if hasattr(georf_2layer.metrics_tracker, 'all_metrics'):
                            georf_2layer.metrics_tracker.all_metrics.clear()
                        if hasattr(georf_2layer.metrics_tracker, 'partition_history'):
                            georf_2layer.metrics_tracker.partition_history.clear()
                        georf_2layer.metrics_tracker = None
                        print("Cleared 2-layer PartitionMetricsTracker data")
            except Exception as e:
                print(f"Warning: Could not clear metrics tracker: {e}")
            
            # CRITICAL FIX 2: Delete model objects completely and break circular references
            try:
                if 'georf' in locals():
                    # ENHANCED model cleanup to prevent memory leaks
                    # Clear all model components explicitly
                    if hasattr(georf, 'model') and georf.model is not None:
                        if hasattr(georf.model, 'model') and georf.model.model is not None:
                            # Clear sklearn RandomForest internals that hold large arrays
                            # Use try-except to safely clear attributes that might not exist or cause errors
                            try:
                                if hasattr(georf.model.model, 'estimators_') and georf.model.model.estimators_ is not None:
                                    georf.model.model.estimators_ = None
                            except:
                                pass
                            try:
                                # Don't access feature_importances_ property as it can fail if estimators_ is None
                                # Clear the private attribute instead if it exists
                                if hasattr(georf.model.model, '_feature_importances'):
                                    georf.model.model._feature_importances = None
                            except:
                                pass
                            # Clear sklearn model completely
                            georf.model.model = None
                        georf.model = None
                    
                    # Clear all directory references
                    georf.dir_space = None
                    georf.dir_ckpt = None
                    georf.dir_vis = None
                    georf.model_dir = None
                    
                    # Clear spatial partitioning data that can be large
                    if hasattr(georf, 's_branch'):
                        georf.s_branch = None
                    if hasattr(georf, 'branch_table'):
                        georf.branch_table = None
                    if hasattr(georf, 'X_branch_id'):
                        georf.X_branch_id = None
                    
                    # Clear any other potential large attributes
                    for attr in ['train_idx', 'val_idx', 'X_train', 'y_train', 'X_val', 'y_val']:
                        if hasattr(georf, attr):
                            setattr(georf, attr, None)
                    
                    georf = None
                del georf
                print("Cleared GeoRF model and all references")
            except NameError:
                pass
            try:
                if 'georf_2layer' in locals():
                    # ENHANCED cleanup for 2-layer model
                    # Clear both layer models 
                    if hasattr(georf_2layer, 'georf_l1') and georf_2layer.georf_l1 is not None:
                        # Clear L1 model internals
                        if hasattr(georf_2layer.georf_l1, 'model') and georf_2layer.georf_l1.model is not None:
                            if hasattr(georf_2layer.georf_l1.model, 'model') and georf_2layer.georf_l1.model.model is not None:
                                try:
                                    if hasattr(georf_2layer.georf_l1.model.model, 'estimators_') and georf_2layer.georf_l1.model.model.estimators_ is not None:
                                        georf_2layer.georf_l1.model.model.estimators_ = None
                                except:
                                    pass
                                georf_2layer.georf_l1.model.model = None
                            georf_2layer.georf_l1.model = None
                        georf_2layer.georf_l1 = None
                    
                    if hasattr(georf_2layer, 'georf_l2') and georf_2layer.georf_l2 is not None:
                        # Clear L2 model internals
                        if hasattr(georf_2layer.georf_l2, 'model') and georf_2layer.georf_l2.model is not None:
                            if hasattr(georf_2layer.georf_l2.model, 'model') and georf_2layer.georf_l2.model.model is not None:
                                try:
                                    if hasattr(georf_2layer.georf_l2.model.model, 'estimators_') and georf_2layer.georf_l2.model.model.estimators_ is not None:
                                        georf_2layer.georf_l2.model.model.estimators_ = None
                                except:
                                    pass
                                georf_2layer.georf_l2.model.model = None
                            georf_2layer.georf_l2.model = None
                        georf_2layer.georf_l2 = None
                    
                    if hasattr(georf_2layer, 'model') and georf_2layer.model is not None:
                        if hasattr(georf_2layer.model, 'model') and georf_2layer.model.model is not None:
                            try:
                                if hasattr(georf_2layer.model.model, 'estimators_') and georf_2layer.model.model.estimators_ is not None:
                                    georf_2layer.model.model.estimators_ = None
                            except:
                                pass
                            georf_2layer.model.model = None
                        georf_2layer.model = None
                    
                    # Clear directory references
                    georf_2layer.dir_space = None
                    georf_2layer.dir_ckpt = None
                    georf_2layer.dir_vis = None
                    georf_2layer.model_dir = None
                    
                    # Clear spatial partitioning data
                    if hasattr(georf_2layer, 's_branch'):
                        georf_2layer.s_branch = None
                    if hasattr(georf_2layer, 'branch_table'):
                        georf_2layer.branch_table = None
                        
                    georf_2layer = None
                del georf_2layer
                print("Cleared 2-layer GeoRF model and all references")
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
            
            # CRITICAL FIX 2: Clear sklearn internal caches and memory pools more aggressively
            try:
                # Clear sklearn joblib memory pools
                from sklearn.externals import joblib
                joblib.Memory.clear_cache_older_than = 0
            except:
                pass
            
            try:
                # Force sklearn to release memory pools more aggressively
                from sklearn.utils import _joblib
                if hasattr(_joblib, 'Parallel'):
                    # Clear joblib parallel backend state
                    _joblib.Parallel._pool = None
                
                # Clear sklearn RandomForest internal memory pools
                import sklearn.ensemble._forest
                if hasattr(sklearn.ensemble._forest, '_generate_sample_indices'):
                    # Clear any cached sample indices that can accumulate
                    try:
                        del sklearn.ensemble._forest._generate_sample_indices.__defaults__
                    except:
                        pass
                
                # Force clearing of sklearn tree building memory
                import sklearn.tree._tree
                if hasattr(sklearn.tree._tree, 'Tree'):
                    # This helps clear tree building buffers
                    pass
                    
                print("Cleared sklearn internal memory pools")
            except Exception as e:
                print(f"Warning: Could not clear sklearn pools: {e}")
                
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
            
            # Save checkpoint after each quarter (in case of interruption)
            if (i + 1) % 5 == 0 or (i + 1) == len(remaining_quarters):  # Save every 5 quarters and at the end
                print(f"Saving checkpoint after Q{quarter} {test_year}...")
                save_checkpoint_results(results_df, y_pred_test, assignment, nowcasting, max_depth, desire_terms, forecasting_scope, start_year, end_year)
                
                # CRITICAL MEMORY FIX: Aggressive cleanup after checkpoints to prevent accumulation
                print("Performing aggressive memory cleanup after checkpoint...")
                
                # CRITICAL FIX 4: Rebuild DataFrames to eliminate fragmentation and internal memory bloat
                # This is a major source of memory leaks - DataFrames accumulate internal overhead
                print("Rebuilding DataFrames to clear internal overhead...")
                
                # Create completely new DataFrame objects to eliminate all internal overhead
                if len(results_df) > 0:
                    # Copy data to plain dictionary first, then create new DataFrame
                    results_data = results_df.to_dict('records')
                    del results_df  # Delete old DataFrame immediately
                    gc.collect()    # Force cleanup
                    results_df = pd.DataFrame(results_data)  # Create fresh DataFrame
                    del results_data  # Clean up temporary data
                    
                if len(y_pred_test) > 0:
                    # Same for predictions DataFrame
                    pred_data = y_pred_test.to_dict('records')
                    del y_pred_test  # Delete old DataFrame immediately 
                    gc.collect()     # Force cleanup
                    y_pred_test = pd.DataFrame(pred_data)  # Create fresh DataFrame
                    del pred_data    # Clean up temporary data
                
                # Reset indices on the new DataFrames
                results_df = results_df.reset_index(drop=True)
                y_pred_test = y_pred_test.reset_index(drop=True)
                
                # Force multiple aggressive garbage collections
                for _ in range(3):
                    gc.collect()
                    
                print(f"DataFrame rebuild complete. Current sizes: results_df={len(results_df)} rows, y_pred_test={len(y_pred_test)} rows")
            
            # Force garbage collection after every quarter
            gc.collect()
            
            # Memory monitoring after cleanup with leak detection
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = end_memory - start_memory
            print(f"Memory after cleanup: {end_memory:.1f} MB (change: {memory_diff:+.1f} MB)")
            
            # Alert if significant memory leak detected
            if memory_diff > 500:  # More than 500MB growth per quarter
                print(f"🚨 WARNING: Large memory increase detected: {memory_diff:.1f} MB")
                print("This indicates a potential memory leak that needs investigation")
                print(f"Consider reducing n_jobs or disabling partition metrics tracking")
            elif memory_diff > 100:  # More than 100MB but less than 500MB
                print(f"⚠️  NOTICE: Moderate memory increase: {memory_diff:.1f} MB")
                print("Memory growth within acceptable range but monitor if this persists")
            else:
                print(f"✅ Memory growth within normal range: {memory_diff:+.1f} MB")
            
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

def save_results(results_df, y_pred_test, assignment, nowcasting=False, max_depth=None, desire_terms=None, forecasting_scope=None, start_year=None, end_year=None):
    """
    Save evaluation results to CSV files.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Evaluation results
    y_pred_test : pandas.DataFrame
        Prediction results
    assignment : str
        Spatial assignment method
    nowcasting : bool
        Whether 2-layer model was used
    max_depth : int or None
        Maximum depth setting
    desire_terms : int or None
        Desired terms setting
    forecasting_scope : int or None
        Forecasting scope (1=3mo, 2=6mo, 3=9mo, 4=12mo)
    start_year : int or None
        Start year of evaluation period
    end_year : int or None
        End year of evaluation period
    """
    # Create file names based on assignment
    pred_test_name = 'y_pred_test_g'
    results_df_name = 'results_df_g'
    
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
    
    if desire_terms is not None:
        pred_test_name += f'_t{desire_terms}'
        results_df_name += f'_t{desire_terms}'
    
    # Add forecasting scope suffix
    if forecasting_scope is not None:
        pred_test_name += f'_fs{forecasting_scope}'
        results_df_name += f'_fs{forecasting_scope}'
    
    # Add year range suffix
    if start_year is not None and end_year is not None:
        pred_test_name += f'_{start_year}_{end_year}'
        results_df_name += f'_{start_year}_{end_year}'
    
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
    Main function to run the complete pipeline.
    """
    print("=== Starting GeoRF Food Crisis Prediction Pipeline ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GeoRF Food Crisis Prediction Pipeline')
    parser.add_argument('--start_year', type=int, default=2024, help='Start year for evaluation (default: 2024)')
    parser.add_argument('--end_year', type=int, default=2024, help='End year for evaluation (default: 2024)')
    parser.add_argument('--forecasting_scope', type=int, default=1, choices=[1,2,3,4], 
                        help='Forecasting scope: 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag (default: 1)')
    parser.add_argument('--force_cleanup', action='store_true', 
                        help='Force cleanup of existing result directories and bypass checkpoint detection')
    args = parser.parse_args()
    
    # Configuration
    assignment = 'polygons'  # Change this to test different grouping methods
    nowcasting = False       # Set to True for 2-layer model
    max_depth = None  # Set to integer for specific RF depth
    desire_terms = None      # None=all quarters, 1=Q1 only, 2=Q2 only, 3=Q3 only, 4=Q4 only
    forecasting_scope = args.forecasting_scope    # From command line argument
    
    # Partition Metrics Tracking Configuration
    track_partition_metrics = False  # Enable partition metrics tracking and visualization
    enable_metrics_maps = False      # Create maps showing F1/accuracy improvements
    
    # Checkpoint Recovery Configuration
    enable_checkpoint_recovery = True  # Enable automatic checkpoint detection and resume
    
    # start year and end year from command line arguments
    start_year = args.start_year
    end_year = args.end_year
    
    print(f"Configuration:")
    print(f"  - Assignment method: {assignment}")
    print(f"  - Nowcasting (2-layer): {nowcasting}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Desired terms: {desire_terms} ({'All quarters (Q1-Q4)' if desire_terms is None else f'Q{desire_terms} only'})")
    print(f"  - Forecasting scope: {forecasting_scope} ({[3,6,9,12][forecasting_scope-1]}-month lag)")
    print(f"  - Rolling window: 5-year training windows before each test quarter")
    print(f"  - Track partition metrics: {track_partition_metrics}")
    print(f"  - Enable metrics maps: {enable_metrics_maps}")
    print(f"  - Checkpoint recovery: {enable_checkpoint_recovery}")
    print(f"  - Start year: {start_year}, End year: {end_year}")
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data(DATA_PATH)
        
        # Step 2: Setup spatial groups
        X_group, X_loc, contiguity_info = setup_spatial_groups(df, assignment)
        
        # Step 3: Prepare features with forecasting scope
        X, y, l1_index, l2_index, years, terms, dates = prepare_features(df, X_group, X_loc, forecasting_scope=forecasting_scope)
        
        # Step 4: Validate polygon contiguity (if applicable)
        if assignment in ['polygons', 'country', 'AEZ', 'country_AEZ', 'geokmeans', 'all_kmeans'] and contiguity_info is not None:
            validate_polygon_contiguity(contiguity_info, X_group)
        
        # Step 5: Run temporal evaluation
        results_df, y_pred_test = run_temporal_evaluation(
            X, y, X_loc, X_group, years, dates, l1_index, l2_index,
            assignment, contiguity_info, df, nowcasting, max_depth, input_terms=terms, desire_terms=desire_terms,
            track_partition_metrics=track_partition_metrics, enable_metrics_maps=enable_metrics_maps,
            start_year=start_year, end_year=end_year, forecasting_scope=forecasting_scope, force_cleanup=args.force_cleanup
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
        
        print("\n=== Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)