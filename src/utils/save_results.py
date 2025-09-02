import pandas as pd
import numpy as np 


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