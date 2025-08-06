#!/usr/bin/env python3
"""
Baseline Probit Regression for Food Crisis Prediction

This script implements a simple probit regression baseline that mirrors the 
workflow in main_model_GF.py but uses only lagged crisis variables as features.

Key features:
1. Uses fews_ipc_crisis_lag_1, lag_2, and lag_3 as predictors
2. Uses 5-year rolling training window (same as main_model_GF)
3. Loops through 2015Q1 to 2024Q4 for temporal validation
4. Calculates precision, recall, and F1 score
5. Saves results in same format for comparison

Date: 2025-08-06
"""

import numpy as np
import pandas as pd
import polars as pl
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression  # Use logistic for probit-like behavior
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from customize import train_test_split_rolling_window
from tqdm import tqdm
import gc

# Configuration
DATA_PATH = r"C:\Users\swl00\IFPRI Dropbox\Weilun Shi\Google fund\Analysis\1.Source Data\FEWSNET_IPC_train_lag_forecast_v06252025.csv"

def load_and_prepare_data():
    """
    Load data and prepare features for probit regression baseline.
    """
    print("Loading data...")
    
    # Load data using polars (faster for large datasets)
    df = pl.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert to pandas for easier manipulation
    df = df.to_pandas()
    
    # Process dates and extract year/quarter (same as main_model_GF)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Keep only rows with valid fews_ipc_crisis values (filter early)
    print(f"Original data: {len(df)} rows")
    print(f"Missing fews_ipc_crisis values: {df['fews_ipc_crisis'].isna().sum()}")
    df = df.dropna(subset=['fews_ipc_crisis'])
    print(f"After filtering: {len(df)} rows with valid target values")
    print(f"Target distribution: {df['fews_ipc_crisis'].value_counts().to_dict()}")
    
    # Sort by admin code and date for proper lag calculation
    df = df.sort_values(['FEWSNET_admin_code', 'year', 'quarter'])
    
    # Create lag features for crisis variable (same as main_model_GF)
    print("Creating lag features...")
    df['fews_ipc_crisis_lag_1'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(1)
    df['fews_ipc_crisis_lag_2'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(2) 
    df['fews_ipc_crisis_lag_3'] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(3)
    
    # Define feature columns for baseline model
    feature_cols = ['fews_ipc_crisis_lag_1', 'fews_ipc_crisis_lag_2', 'fews_ipc_crisis_lag_3']
    
    print("Missing values in lag features:")
    for col in feature_cols:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing} missing out of {len(df)} ({missing/len(df)*100:.1f}%)")
    
    # Extract features and target
    X = df[feature_cols].values
    y = df['fews_ipc_crisis'].values.astype(int)
    
    # Extract location and grouping info (using same columns as main_model_GF)
    X_loc = df[['lat', 'lon']].values
    X_group = df['FEWSNET_admin_code'].values  # Use admin code as group identifier
    
    # Extract temporal info (same as main_model_GF)
    years = df['year'].values  
    dates = df['date'].values
    terms = df['quarter'].values  # This is the missing input_terms array!
    
    print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Feature names: {feature_cols}")
    print(f"Years range: {years.min()} to {years.max()}")
    print(f"Quarters distribution: {np.bincount(terms)}")
    
    return X, y, X_loc, X_group, years, dates, terms, feature_cols

def fit_probit_baseline(X_train, y_train):
    """
    Fit probit regression model using logistic regression as approximation.
    """
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Fit logistic regression (probit-like behavior)
    # Use L2 regularization for stability
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    model.fit(X_train_imputed, y_train)
    
    # Store imputer for prediction
    model.imputer = imputer
    
    return model

def predict_probit_baseline(model, X_test):
    """
    Make predictions using fitted probit model.
    """
    # Handle missing values
    X_test_imputed = model.imputer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_imputed)
    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
    
    return y_pred, y_pred_proba

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    Calculate evaluation metrics.
    """
    # Calculate metrics for both classes
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Get class counts
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    # Ensure we have metrics for both classes (0 and 1)
    metrics = {}
    for cls in [0, 1]:
        if cls < len(precision):
            metrics[f'precision({cls})'] = precision[cls]
            metrics[f'recall({cls})'] = recall[cls]
            metrics[f'f1({cls})'] = f1[cls]
        else:
            metrics[f'precision({cls})'] = 0.0
            metrics[f'recall({cls})'] = 0.0
            metrics[f'f1({cls})'] = 0.0
            
        # Count samples
        if cls in unique_true:
            idx = np.where(unique_true == cls)[0][0]
            metrics[f'num_samples({cls})'] = counts_true[idx]
        else:
            metrics[f'num_samples({cls})'] = 0
    
    return metrics

def generate_quarters(start_year, end_year):
    """
    Generate list of (year, quarter) tuples from start to end year.
    """
    quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            quarters.append((year, quarter))
    return quarters

def main():
    """
    Main function to run baseline probit regression evaluation.
    """
    print("=== Baseline Probit Regression for Food Crisis Prediction ===")
    
    # Load and prepare data
    X, y, X_loc, X_group, years, dates, terms, feature_names = load_and_prepare_data()
    
    # Generate quarters to evaluate (2015Q1 to 2024Q4)
    eval_quarters = generate_quarters(2015, 2024)
    print(f"Will evaluate {len(eval_quarters)} quarters from 2015Q1 to 2024Q4")
    
    # Initialize results storage
    results_list = []
    
    # Create progress bar
    progress_bar = tqdm(total=len(eval_quarters), desc="Baseline Probit")
    
    # Loop through quarters
    for i, (test_year, quarter) in enumerate(eval_quarters):
        progress_bar.set_description(f"Baseline Q{quarter} {test_year}")
        print(f"\n--- Evaluating Q{quarter} {test_year} (#{i+1}/{len(eval_quarters)}) ---")
        
        try:
            # Train-test split with rolling window (5 years before quarter end)
            # Use same function as main_model_GF for consistency
            (Xtrain, ytrain, Xtrain_loc, Xtrain_group,
             Xtest, ytest, Xtest_loc, Xtest_group) = train_test_split_rolling_window(
                X, y, X_loc, X_group, years, dates, 
                test_year=test_year, input_terms=terms, need_terms=quarter)
            
            ytrain = ytrain.astype(int)
            ytest = ytest.astype(int)
            
            print(f"Train samples: {len(ytrain)}, Test samples: {len(ytest)}")
            if len(ytrain) > 0:
                print(f"Train distribution: {np.bincount(ytrain)}")
            if len(ytest) > 0:
                print(f"Test distribution: {np.bincount(ytest)}")
            
            # Skip if no test samples
            if len(ytest) == 0:
                print(f"Warning: No test samples for Q{quarter} {test_year}. Skipping.")
                progress_bar.update(1)
                continue
                
            # Skip if no training samples  
            if len(ytrain) == 0:
                print(f"Warning: No training samples for Q{quarter} {test_year}. Skipping.")
                progress_bar.update(1)
                continue
            
            # Fit baseline probit model
            print("Fitting probit baseline model...")
            model = fit_probit_baseline(Xtrain, ytrain)
            
            # Make predictions
            print("Making predictions...")
            y_pred, y_pred_proba = predict_probit_baseline(model, Xtest)
            
            # Evaluate predictions
            metrics = evaluate_predictions(ytest, y_pred, y_pred_proba)
            
            # Add metadata
            metrics['year'] = test_year
            metrics['quarter'] = quarter
            
            # Print results
            print(f"Results for Q{quarter} {test_year}:")
            print(f"  Precision (class 0): {metrics['precision(0)']:.4f}")
            print(f"  Precision (class 1): {metrics['precision(1)']:.4f}")
            print(f"  Recall (class 0): {metrics['recall(0)']:.4f}")
            print(f"  Recall (class 1): {metrics['recall(1)']:.4f}")
            print(f"  F1 (class 0): {metrics['f1(0)']:.4f}")
            print(f"  F1 (class 1): {metrics['f1(1)']:.4f}")
            
            # Store results
            results_list.append(metrics)
            
            # Memory cleanup
            del model, Xtrain, ytrain, Xtest, ytest, y_pred, y_pred_proba
            gc.collect()
            
        except Exception as e:
            print(f"Error processing Q{quarter} {test_year}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert results to DataFrame and save
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # Sort by year and quarter
        results_df = results_df.sort_values(['year', 'quarter'])
        
        # Save results
        output_dir = "baseline_probit_results"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "baseline_probit_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total quarters evaluated: {len(results_df)}")
        print(f"Average precision (class 1): {results_df['precision(1)'].mean():.4f}")
        print(f"Average recall (class 1): {results_df['recall(1)'].mean():.4f}")
        print(f"Average F1 (class 1): {results_df['f1(1)'].mean():.4f}")
        
        return results_df
    else:
        print("No results generated!")
        return None

if __name__ == "__main__":
    results_df = main()