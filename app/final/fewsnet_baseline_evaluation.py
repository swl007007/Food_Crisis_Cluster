#!/usr/bin/env python3
"""
FEWSNET Official Predictions Baseline Evaluation

This script evaluates FEWSNET's own predictions as a baseline for comparison
with GeoRF and other models. It processes FEWSNET's near-term and medium-term
predictions with appropriate temporal lags.

Key features:
1. Loads FEWSNET official predictions from FEWSNET.csv
2. Converts monthly data to quarterly format
3. Applies appropriate lags:
   - pred_near_lag1 for forecasting scope 1 (3-month)
   - pred_med_lag2 for forecasting scope 2 (6-month)
4. Evaluates performance using class 1 metrics only
5. Saves results in format compatible with comparison script

Date: 2025-08-20
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import gc

# Configuration
FEWSNET_DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\Outcome\FEWSNET_IPC\FEWSNET.csv"

def month_to_quarter(month):
    """Convert month number to quarter number."""
    return (month - 1) // 3 + 1

def load_and_prepare_fewsnet_data():
    """
    Load FEWSNET data and prepare for evaluation.
    
    Returns:
    --------
    df : pandas.DataFrame
        Processed FEWSNET data with crisis indicators and temporal info
    """
    print("Loading FEWSNET data...")
    
    # Load data
    df = pd.read_csv(FEWSNET_DATA_PATH)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    print("FEWSNET data columns:", df.columns.tolist())
    print("Sample data:")
    print(df.head())
    
    # Convert admin_code to match our models (FEWSNET_admin_code)
    if 'admin_code' in df.columns:
        df['FEWSNET_admin_code'] = df['admin_code']
    
    # Add quarter information
    df['quarter'] = df['month'].apply(month_to_quarter)
    
    # Generate crisis indicators (>=3 = crisis)
    df['crisis_actual'] = df['fews_ipc'].apply(lambda x: 1 if x >= 3 else 0)
    df['pred_near'] = df['fews_proj_near'].apply(lambda x: 1 if pd.notna(x) and x >= 3 else 0)
    df['pred_med'] = df['fews_proj_med'].apply(lambda x: 1 if pd.notna(x) and x >= 3 else 0)
    
    # Apply lags by admin_code
    print("Applying temporal lags...")
    df = df.sort_values(['FEWSNET_admin_code', 'year', 'month'])
    df['pred_near_lag1'] = df.groupby('FEWSNET_admin_code')['pred_near'].shift(4)
    df['pred_med_lag2'] = df.groupby('FEWSNET_admin_code')['pred_med'].shift(8)
    
    # Filter to years >= 2015 to match other models
    print(f"Original data: {len(df)} rows")
    df = df[df['year'] >= 2013]
    print(f"After filtering >= 2013: {len(df)} rows")
    
    # Remove rows with missing target or predictions
    original_len = len(df)
    df = df.dropna(subset=['crisis_actual'])
    print(f"After removing missing crisis_actual: {len(df)} rows (removed {original_len - len(df)})")
    
    print(f"Target distribution: {df['crisis_actual'].value_counts().to_dict()}")
    print(f"Years range: {df['year'].min()} to {df['year'].max()}")
    print(f"Quarters distribution: {df['quarter'].value_counts().to_dict()}")
    
    return df

def evaluate_predictions_class1_only(y_true, y_pred):
    """
    Calculate evaluation metrics for class 1 only.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    metrics : dict
        Dictionary with class 1 metrics only
    """
    # Remove NaN values
    mask = pd.notna(y_pred) & pd.notna(y_true)
    y_true_clean = np.array(y_true)[mask]
    y_pred_clean = np.array(y_pred)[mask]
    
    if len(y_true_clean) == 0:
        return {
            'precision(1)': 0.0,
            'recall(1)': 0.0,
            'f1(1)': 0.0,
            'num_samples(1)': 0
        }
    
    # Calculate class 1 metrics
    precision_1 = precision_score(y_true_clean, y_pred_clean, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_true_clean, y_pred_clean, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true_clean, y_pred_clean, pos_label=1, zero_division=0)
    num_samples_1 = sum(y_true_clean == 1)
    
    return {
        'precision(1)': precision_1,
        'recall(1)': recall_1,
        'f1(1)': f1_1,
        'num_samples(1)': num_samples_1
    }

def generate_quarters(start_year, end_year):
    """
    Generate list of (year, quarter) tuples from start to end year.
    """
    quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            quarters.append((year, quarter))
    return quarters

def evaluate_fewsnet_scope(df, forecasting_scope, start_year=2015, end_year=2024):
    """
    Evaluate FEWSNET predictions for a specific forecasting scope.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        FEWSNET data with predictions and lags
    forecasting_scope : int
        1 for near-term (3-month), 2 for medium-term (6-month)
    start_year : int
        Start year for evaluation
    end_year : int
        End year for evaluation
        
    Returns:
    --------
    results_df : pandas.DataFrame
        Evaluation results by quarter
    """
    # Map forecasting scope to prediction column
    if forecasting_scope == 1:
        pred_col = 'pred_near_lag1'
        lag_months = 3
    elif forecasting_scope == 2:
        pred_col = 'pred_med_lag2'
        lag_months = 6
    else:
        raise ValueError(f"FEWSNET only supports forecasting scopes 1-2, got {forecasting_scope}")
    
    print(f"\nEvaluating FEWSNET forecasting scope {forecasting_scope} ({lag_months}-month lag)")
    
    # Generate quarters to evaluate
    eval_quarters = generate_quarters(start_year, end_year)
    print(f"Will evaluate {len(eval_quarters)} quarters from {start_year}Q1 to {end_year}Q4")
    
    # Initialize results storage
    results_list = []
    
    # Create progress bar
    progress_bar = tqdm(total=len(eval_quarters), desc=f"FEWSNET FS{forecasting_scope}")
    
    # Loop through quarters
    for i, (test_year, quarter) in enumerate(eval_quarters):
        progress_bar.set_description(f"FEWSNET FS{forecasting_scope} Q{quarter} {test_year}")
        print(f"\n--- Evaluating Q{quarter} {test_year} (#{i+1}/{len(eval_quarters)}) ---")
        
        try:
            # Filter data for this quarter
            quarter_data = df[(df['year'] == test_year) & (df['quarter'] == quarter)]
            
            print(f"Found {len(quarter_data)} records for Q{quarter} {test_year}")
            
            # Skip if no data for this quarter
            if len(quarter_data) == 0:
                print(f"Warning: No data for Q{quarter} {test_year}. Skipping.")
                progress_bar.update(1)
                continue
            
            # Get true and predicted values
            y_true = quarter_data['crisis_actual'].values
            y_pred = quarter_data[pred_col].values
            
            # Count valid predictions
            valid_mask = pd.notna(y_pred) & pd.notna(y_true)
            n_valid = sum(valid_mask)
            
            print(f"Valid predictions: {n_valid} out of {len(quarter_data)}")
            
            if n_valid == 0:
                print(f"Warning: No valid predictions for Q{quarter} {test_year}. Skipping.")
                progress_bar.update(1)
                continue
            
            # Evaluate predictions
            metrics = evaluate_predictions_class1_only(y_true, y_pred)
            
            # Add metadata
            metrics['year'] = test_year
            metrics['quarter'] = quarter
            
            # Print results
            print(f"Results for Q{quarter} {test_year}:")
            print(f"  Precision (class 1): {metrics['precision(1)']:.4f}")
            print(f"  Recall (class 1): {metrics['recall(1)']:.4f}")
            print(f"  F1 (class 1): {metrics['f1(1)']:.4f}")
            print(f"  Samples (class 1): {metrics['num_samples(1)']}")
            
            # Store results
            results_list.append(metrics)
            
        except Exception as e:
            print(f"Error processing Q{quarter} {test_year}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert results to DataFrame
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Sort by year and quarter
        results_df = results_df.sort_values(['year', 'quarter'])
        
        return results_df
    else:
        print("No results generated!")
        return None

def main():
    """
    Main function to run FEWSNET baseline evaluation.
    """
    print("=== FEWSNET Official Predictions Baseline Evaluation ===")
    
    # Parse command line arguments for consistency
    parser = argparse.ArgumentParser(description='FEWSNET Baseline Evaluation')
    parser.add_argument('--start_year', type=int, default=2013, help='Start year for evaluation (default: 2015)')
    parser.add_argument('--end_year', type=int, default=2024, help='End year for evaluation (default: 2024)')
    parser.add_argument('--forecasting_scope', type=int, choices=[1,2], 
                        help='Forecasting scope: 1=3mo lag, 2=6mo lag (default: run both)')
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  - Evaluation period: {args.start_year} to {args.end_year}")
    print(f"  - FEWSNET supports forecasting scopes 1-2 only")
    
    # Load and prepare data
    df = load_and_prepare_fewsnet_data()
    
    # Create output directory
    output_dir = "fewsnet_baseline_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which scopes to evaluate
    if args.forecasting_scope:
        scopes_to_run = [args.forecasting_scope]
    else:
        scopes_to_run = [1, 2]  # FEWSNET only supports scopes 1-2
    
    # Evaluate each scope
    for scope in scopes_to_run:
        print(f"\n{'='*60}")
        print(f"Processing Forecasting Scope {scope}")
        print(f"{'='*60}")
        
        try:
            # Evaluate this scope
            results_df = evaluate_fewsnet_scope(df, scope, args.start_year, args.end_year)
            
            if results_df is not None and len(results_df) > 0:
                # Save results
                output_file = os.path.join(output_dir, f"fewsnet_baseline_results_fs{scope}.csv")
                results_df.to_csv(output_file, index=False)
                print(f"\nResults saved to: {output_file}")
                
                # Print summary statistics
                print(f"\n=== Summary Statistics for Scope {scope} ===")
                print(f"Total quarters evaluated: {len(results_df)}")
                print(f"Average precision (class 1): {results_df['precision(1)'].mean():.4f}")
                print(f"Average recall (class 1): {results_df['recall(1)'].mean():.4f}")
                print(f"Average F1 (class 1): {results_df['f1(1)'].mean():.4f}")
                
            else:
                print(f"No results generated for scope {scope}!")
                
        except Exception as e:
            print(f"Error evaluating scope {scope}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== FEWSNET Baseline Evaluation Complete ===")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)