#!/usr/bin/env python3
"""
Baseline Probit Regression for Food Crisis Prediction

This script implements a simple probit regression baseline that mirrors the 
workflow in main_model_XGB.py but uses only lagged crisis variables as features.

Key features:
1. Uses lagged crisis variables as predictors with configurable forecasting scope
2. Forecasting scope options:
   - scope=1: lag 1,2,3 terms (3-month forecasting)
   - scope=2: lag 2,3,4 terms (6-month forecasting)  
   - scope=3: lag 3,4,5 terms (9-month forecasting)
   - scope=4: lag 4,5,6 terms (12-month forecasting)
3. Uses 5-year rolling training window (same as main_model_XGB)
4. Loops through configurable year range for temporal validation
5. Calculates precision, recall, and F1 score
6. Saves results with forecasting scope suffix for comparison

Date: 2025-08-13
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

def load_and_prepare_data(forecasting_scope=4):
    """
    Load data and prepare features for probit regression baseline.
    
    Parameters:
    -----------
    forecasting_scope : int, default=4
        Forecasting scope: 1=1,2,3 term lags, 2=2,3,4 term lags, 3=3,4,5 term lags, 4=4,5,6 term lags
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
    
    # Create lag features based on forecasting scope
    print(f"Creating lag features for forecasting_scope={forecasting_scope}...")
    
    # Define lag terms based on forecasting scope
    # scope=1: use lag 1,2,3 terms (3-month forecasting)
    # scope=2: use lag 2,3,4 terms (6-month forecasting)  
    # scope=3: use lag 3,4,5 terms (9-month forecasting)
    # scope=4: use lag 4,5,6 terms (12-month forecasting)
    lag_mapping = {
        1: [1, 2, 3],  # 3-month lag
        2: [2, 3, 4],  # 6-month lag
        3: [3, 4, 5],  # 9-month lag
        4: [4, 5, 6]   # 12-month lag
    }
    
    if forecasting_scope not in lag_mapping:
        raise ValueError(f"Invalid forecasting_scope: {forecasting_scope}. Must be 1, 2, 3, or 4.")
    
    lag_terms = lag_mapping[forecasting_scope]
    print(f"Using lag terms: {lag_terms} for {[3,6,9,12][forecasting_scope-1]}-month forecasting")
    
    # Create lag features
    feature_cols = []
    for lag in lag_terms:
        col_name = f'fews_ipc_crisis_lag_{lag}'
        df[col_name] = df.groupby('FEWSNET_admin_code')['fews_ipc_crisis'].shift(lag)
        feature_cols.append(col_name)
    
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

def evaluate_predictions_class1_only(y_true, y_pred, y_pred_proba=None):
    """
    Calculate evaluation metrics for class 1 only.
    """
    # Calculate class 1 metrics only
    precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Count class 1 samples
    num_samples_1 = sum(y_true == 1)
    
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

def main():
    """
    Main function to run baseline probit regression evaluation.
    """
    print("=== Baseline Probit Regression for Food Crisis Prediction ===")
    
    # Configuration
    forecasting_scope = 4  # 1=3mo lag, 2=6mo lag, 3=9mo lag, 4=12mo lag
    start_year = 2015
    end_year = 2024
    
    print(f"Configuration:")
    print(f"  - Forecasting scope: {forecasting_scope} ({[3,6,9,12][forecasting_scope-1]}-month lag)")
    print(f"  - Evaluation period: {start_year} to {end_year}")
    
    # Load and prepare data
    X, y, X_loc, X_group, years, dates, terms, feature_names = load_and_prepare_data(forecasting_scope=forecasting_scope)
    
    # Generate quarters to evaluate
    eval_quarters = generate_quarters(start_year, end_year)
    print(f"Will evaluate {len(eval_quarters)} quarters from {start_year}Q1 to {end_year}Q4")
    
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
            metrics = evaluate_predictions_class1_only(ytest, y_pred, y_pred_proba)
            
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
        
        # Save results with forecasting scope in filename
        output_dir = "baseline_probit_results"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"baseline_probit_results_fs{forecasting_scope}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary statistics
        print("\n=== Summary Statistics (Class 1 Only) ===")
        print(f"Total quarters evaluated: {len(results_df)}")
        print(f"Average precision (class 1): {results_df['precision(1)'].mean():.4f}")
        print(f"Average recall (class 1): {results_df['recall(1)'].mean():.4f}")
        print(f"Average F1 (class 1): {results_df['f1(1)'].mean():.4f}")
        print(f"Total class 1 samples: {results_df['num_samples(1)'].sum()}")
        
        return results_df
    else:
        print("No results generated!")
        return None

if __name__ == "__main__":
    results_df = main()