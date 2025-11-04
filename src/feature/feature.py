
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
from src.utils.lag_schedules import forecasting_scope_to_lag, resolve_lag_schedule

ACTIVE_LAGS = resolve_lag_schedule(LAGS_MONTHS, context="config_visual.LAGS_MONTHS")



def prepare_features(df, X_group, X_loc, forecasting_scope=len(ACTIVE_LAGS)):
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
    forecasting_scope : int, default=len(ACTIVE_LAGS)
        Forecasting scope (1-based index into active lag schedule)
        
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
    # df group by FEWSNET_admin_code, see which features are time-variant(has different value for one FEWSNET_admin_code within one year)
    features = df.columns.tolist()
    time_variants = []
    for feature in features:
        if feature not in ['FEWSNET_admin_code', 'date', 'fews_ipc_crisis', 'AEZ_group', 'ISO_encoded', 'AEZ_country_group']:
            if df.groupby(['FEWSNET_admin_code', df['date'].dt.year])[feature].nunique().max() > 2:
                time_variants.append(feature)
    
    lag_months = forecasting_scope_to_lag(forecasting_scope, ACTIVE_LAGS)
    print(f"Using {lag_months}-month lag for forecasting scope {forecasting_scope}")
    
    # Create appropriate time variant list based on forecasting scope
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

    
    for variant in time_variants:
        if variant in df_sorted.columns:
            lagged_col_name = f'{variant}_lag{lag_months}m'
            df_sorted[lagged_col_name] = df_sorted.groupby('FEWSNET_admin_code')[variant].shift(lag_months)
            time_variants_list.append(lagged_col_name)
    
    
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
    
    # Apply imputation (XGBoost pipelines skip because model handles NaNs)
    pipeline_flag = os.environ.get('GEORF_PIPELINE', '').strip().lower()
    skip_impute = pipeline_flag in {'xgb', 'geoxgb'}
    if skip_impute:
        print("Skipping comp_impute for GeoXGB pipeline (XGBoost handles missing values natively)...")
    else:
        print("Applying imputation to missing values...")
        X = comp_impute(X, strategy="max_plus", multiplier=100.0)

    # Coerce to numeric to allow finite checks (object dtype can appear when imputation is skipped)
    if not np.issubdtype(X.dtype, np.number):
        try:
            X = X.astype(np.float64)
        except (TypeError, ValueError) as err:
            raise ValueError("Feature matrix contains non-numeric entries; failed to convert to float before modeling.") from err

    # Ensure no infinities propagate into downstream models (XGBoost cannot handle them)
    inf_mask = np.isinf(X)
    if inf_mask.any():
        count_inf = int(np.sum(inf_mask))
        print(f"Detected {count_inf} +/-inf values; replacing with 0 before modeling.")
        X = np.where(inf_mask, 0.0, X)

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
