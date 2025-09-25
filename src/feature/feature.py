
import numpy as np
import pandas as pd
import polars as pl
import os
import sys
from pathlib import Path
import warnings
# Add parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# GeoRF imports
from src.customize.customize import *
from src.tests.class_wise_metrics import *
from config_visual import *
from src.utils.force_clean import *
from src.preprocess.preprocess import *



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
    
    # Get feature matrix and preserve column names for downstream interpretable outputs
    feature_columns = [str(col) for col in df_features.columns]
    try:
        debug_path = Path('feature_columns_debug.csv')
        debug_df = pd.DataFrame({
            'feature_index': np.arange(len(feature_columns), dtype=int),
            'feature_name': feature_columns
        })
        debug_df.to_csv(debug_path, index=False)
    except Exception as debug_err:
        print(f'Warning: could not write feature_columns_debug.csv: {debug_err}')
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
    
    return X, y, l1_index, l2_index, years, terms, df_sorted['date'], feature_columns

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
        
        min_neighbors = min(neighbor_counts) if neighbor_counts else 0
        max_neighbors = max(neighbor_counts) if neighbor_counts else 0
        mean_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
        print(f"Adjacency neighbor stats: min={min_neighbors}, max={max_neighbors}, mean={mean_neighbors:.1f}")
        
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
