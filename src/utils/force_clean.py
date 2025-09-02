import pandas as pd
import os
import sys
import gc
import glob
import warnings

# Add parent directory to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# GeoRF imports
from src.customize.customize import *
from src.tests.class_wise_metrics import *
from config_visual import *

from tqdm import tqdm

# Import adjacency matrix utilities


# Configuration
DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_v06252025.csv"

def force_cleanup_directories(*, pipeline='gf'):
    """
    Perform robust cleanup of result directories.
    Uses Python for better error handling and cross-platform compatibility.
    
    Parameters:
    -----------
    pipeline : str, optional
        Pipeline type ('gf' for GeoRF, 'xgb' for XGBoost). Default is 'gf'.
    """
    import shutil
    import time
    
    print("Performing force cleanup of result directories...")
    
    # Find result directories based on pipeline
    if pipeline == 'xgb':
        result_dirs = glob.glob('result_GeoXGB*')
        dir_type = "result_GeoXGB*"
    else:
        result_dirs = glob.glob('result_GeoRF*')
        dir_type = "result_GeoRF*"
    
    if not result_dirs:
        print(f"No {dir_type} directories found to clean up.")
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