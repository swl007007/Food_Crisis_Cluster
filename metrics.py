# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
# import tensorflow as tf
from config import *

def get_prf(true_class, total_class, pred_class, nan_option = 'mean', nan_value = -1):
  '''
  Args:
    nan_option: valid values: 'mean', 'zero', 'value' ...
    nan_value: only used at the end when outputing f1 scores, if nan_option is 'value'
  '''
  pre = true_class / pred_class
  rec = true_class / total_class

  if nan_option == 'mean':
    pre_fix = np.nan_to_num(pre, nan = np.nanmean(pre))
    rec_fix = np.nan_to_num(rec, nan = np.nanmean(rec))
  else:#put to zeros
    pre_fix = np.nan_to_num(pre)
    rec_fix = np.nan_to_num(rec)

  f1 = 2/(pre_fix**(-1) + rec_fix**(-1))
  # Ensure logical comparison uses proper precedence
  f1[(pre_fix == 0) & (rec_fix == 0)] = 0

  if nan_option == 'value':
    f1[total_class==0] = np.nan

  return pre, rec, f1, total_class


def get_overall_accuracy(y_true, y_pred):
    """Calculates overall accuracy without using sklearn's accuracy_score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        A tuple containing the number of correct predictions and the total number of predictions.
    """

    if len(y_true.shape) == 1:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        y_true = np.eye(NUM_CLASS)[y_true].astype(int)
        y_pred = np.eye(NUM_CLASS)[y_pred].astype(int)

    if len(y_true.shape) >= 3:
        y_true = y_true.reshape(-1, NUM_CLASS)
        y_pred = y_pred.reshape(-1, NUM_CLASS)

    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    total_predictions = y_true.shape[0]

    return correct_predictions, total_predictions


def get_class_wise_accuracy(y_true, y_pred, prf=False):
    """Calculates class-wise accuracy without using sklearn's accuracy_score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        prf: A boolean flag indicating whether to calculate precision, recall, and F1-score.

    Returns:
        A tuple containing the number of correct predictions per class and the total number of predictions per class.
    """

    if len(y_true.shape) >= 3:
        y_true = y_true.reshape(-1, NUM_CLASS)
        y_pred = y_pred.reshape(-1, NUM_CLASS)

    if len(y_true.shape) == 1:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        y_true = np.eye(NUM_CLASS)[y_true].astype(int)
        y_pred = np.eye(NUM_CLASS)[y_pred].astype(int)
    else:
        predicted_classes = np.argmax(y_pred, axis=1)
        y_pred = np.eye(NUM_CLASS)[predicted_classes].astype(int)

    correct_predictions = (y_true==y_pred) * (y_true == 1)
    true_per_class = np.sum(correct_predictions, axis=0)
    total_per_class = np.sum(y_true, axis=0)

    if prf:
        pred_w_class = np.argmax(y_pred, axis=1)
        pred_w_class = np.eye(NUM_CLASS)[pred_w_class].astype(int)
        pred_total = np.sum(pred_w_class, axis=0)
        return true_per_class, total_per_class, pred_total
    else:
        return true_per_class, total_per_class


def get_group_wise_metrics(y_true, y_pred, X_group, return_details=False):
    """
    Calculate F1 score and accuracy for each X_group.
    
    Args:
        y_true: True labels (1D array)
        y_pred: Predicted labels (1D array) 
        X_group: Group assignments (1D array)
        return_details: If True, return detailed per-class metrics
        
    Returns:
        dict: Group-wise metrics with group IDs as keys
    """
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame({
        'y_true': y_true.flatten() if len(y_true.shape) > 1 else y_true,
        'y_pred': y_pred.flatten() if len(y_pred.shape) > 1 else y_pred,
        'X_group': X_group.flatten() if len(X_group.shape) > 1 else X_group
    })
    
    group_metrics = {}
    
    for group_id in df['X_group'].unique():
        group_data = df[df['X_group'] == group_id]
        
        if len(group_data) == 0:
            continue
            
        y_true_group = group_data['y_true'].values
        y_pred_group = group_data['y_pred'].values
        
        # Skip if all predictions are the same class (F1 calculation issues)
        if len(np.unique(y_true_group)) == 1 and len(np.unique(y_pred_group)) == 1:
            if y_true_group[0] == y_pred_group[0]:
                f1 = 1.0
                accuracy = 1.0
                precision = 1.0 
                recall = 1.0
            else:
                f1 = 0.0
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
        else:
            try:
                # Use macro average for multi-class, binary for binary
                avg_method = 'binary' if NUM_CLASS == 2 else 'macro'
                f1 = f1_score(y_true_group, y_pred_group, average=avg_method, zero_division=0)
                accuracy = accuracy_score(y_true_group, y_pred_group)
                precision = precision_score(y_true_group, y_pred_group, average=avg_method, zero_division=0)
                recall = recall_score(y_true_group, y_pred_group, average=avg_method, zero_division=0)
            except Exception as e:
                print(f"Warning: Error calculating metrics for group {group_id}: {e}")
                f1 = 0.0
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
        
        group_metrics[group_id] = {
            'f1_score': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'n_samples': len(group_data),
            'n_classes': len(np.unique(y_true_group))
        }
        
        if return_details:
            # Add per-class metrics for detailed analysis
            try:
                f1_per_class = f1_score(y_true_group, y_pred_group, average=None, zero_division=0)
                precision_per_class = precision_score(y_true_group, y_pred_group, average=None, zero_division=0)
                recall_per_class = recall_score(y_true_group, y_pred_group, average=None, zero_division=0)
                
                group_metrics[group_id].update({
                    'f1_per_class': f1_per_class,
                    'precision_per_class': precision_per_class,
                    'recall_per_class': recall_per_class
                })
            except:
                pass
                
    return group_metrics


def calculate_metrics_improvement(metrics_before, metrics_after):
    """
    Calculate improvement in metrics after partitioning.
    
    Args:
        metrics_before: Group-wise metrics before partition
        metrics_after: Group-wise metrics after partition
        
    Returns:
        dict: Improvement metrics for each group
    """
    improvement = {}
    
    # Get all unique group IDs from both before and after
    all_groups = set(metrics_before.keys()) | set(metrics_after.keys())
    
    for group_id in all_groups:
        before = metrics_before.get(group_id, {'f1_score': 0.0, 'accuracy': 0.0})
        after = metrics_after.get(group_id, {'f1_score': 0.0, 'accuracy': 0.0})
        
        improvement[group_id] = {
            'f1_improvement': after['f1_score'] - before['f1_score'],
            'accuracy_improvement': after['accuracy'] - before['accuracy'],
            'f1_before': before['f1_score'],
            'f1_after': after['f1_score'],
            'accuracy_before': before['accuracy'],
            'accuracy_after': after['accuracy']
        }
    
    return improvement


class PartitionMetricsTracker:
    """
    Track metrics before and after each partition round for debugging and analysis.
    """
    
    def __init__(self, correspondence_table_path=None):
        self.partition_history = {}
        self.correspondence_table_path = correspondence_table_path
        self.all_metrics = []  # Add this to store all metrics records
        
    def record_partition_metrics(self, partition_round, branch_id, 
                               y_true, y_pred_before, y_pred_after, X_group,
                               partition_type="binary_split"):
        """
        Record metrics for a partition round.
        
        Args:
            partition_round: Round number (0, 1, 2, ...)
            branch_id: Branch identifier (e.g., '', '0', '01', etc.)
            y_true: True labels for validation set
            y_pred_before: Predictions before partition
            y_pred_after: Predictions after partition (concatenated from both branches)
            X_group: Group assignments
            partition_type: Type of partition ("binary_split", "no_split")
        """
        
        if partition_round not in self.partition_history:
            self.partition_history[partition_round] = {}
            
        # Calculate group-wise metrics
        metrics_before = get_group_wise_metrics(y_true, y_pred_before, X_group)
        metrics_after = get_group_wise_metrics(y_true, y_pred_after, X_group)
        improvement = calculate_metrics_improvement(metrics_before, metrics_after)
        
        self.partition_history[partition_round][branch_id] = {
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'improvement': improvement,
            'partition_type': partition_type,
            'n_samples': len(y_true),
            'n_groups': len(np.unique(X_group))
        }
        
        # Also populate all_metrics list for easier access
        for group_id, imp in improvement.items():
            self.all_metrics.append({
                'partition_round': partition_round,
                'branch_id': branch_id,
                'X_group': group_id,
                'f1_before': imp['f1_before'],
                'f1_after': imp['f1_after'],
                'f1_improvement': imp['f1_improvement'],
                'accuracy_before': imp['accuracy_before'],
                'accuracy_after': imp['accuracy_after'],
                'accuracy_improvement': imp['accuracy_improvement'],
                'partition_type': partition_type
            })
        
        print(f"Partition Round {partition_round}, Branch {branch_id}:")
        print(f"  - Average F1 before: {np.mean([m['f1_score'] for m in metrics_before.values()]):.4f}")
        print(f"  - Average F1 after: {np.mean([m['f1_score'] for m in metrics_after.values()]):.4f}")
        print(f"  - Average accuracy before: {np.mean([m['accuracy'] for m in metrics_before.values()]):.4f}")
        print(f"  - Average accuracy after: {np.mean([m['accuracy'] for m in metrics_after.values()]):.4f}")
        
    def save_metrics_to_csv(self, output_dir, filename_prefix="partition_metrics"):
        """
        Save all recorded metrics to CSV files.
        """
        import os
        
        for partition_round, branches in self.partition_history.items():
            for branch_id, data in branches.items():
                # Create DataFrame for this partition round and branch
                rows = []
                for group_id in data['improvement'].keys():
                    imp = data['improvement'][group_id]
                    rows.append({
                        'partition_round': partition_round,
                        'branch_id': branch_id,
                        'X_group': group_id,
                        'f1_before': imp['f1_before'],
                        'f1_after': imp['f1_after'],
                        'f1_improvement': imp['f1_improvement'],
                        'accuracy_before': imp['accuracy_before'],
                        'accuracy_after': imp['accuracy_after'],
                        'accuracy_improvement': imp['accuracy_improvement'],
                        'partition_type': data['partition_type']
                    })
                
                if rows:
                    df = pd.DataFrame(rows)
                    filename = f"{filename_prefix}_round{partition_round}_branch{branch_id or 'root'}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"Saved metrics to: {filepath}")
                    
    def get_improvement_summary(self):
        """
        Get summary statistics of improvements across all partitions.
        """
        all_improvements = []
        
        for partition_round, branches in self.partition_history.items():
            for branch_id, data in branches.items():
                for group_id, imp in data['improvement'].items():
                    imp_record = imp.copy()
                    imp_record.update({
                        'partition_round': partition_round,
                        'branch_id': branch_id,
                        'group_id': group_id
                    })
                    all_improvements.append(imp_record)
        
        if not all_improvements:
            return None
            
        df = pd.DataFrame(all_improvements)
        
        summary = {
            'total_partitions': len(all_improvements),
            'avg_f1_improvement': df['f1_improvement'].mean(),
            'avg_accuracy_improvement': df['accuracy_improvement'].mean(),
            'positive_f1_improvements': (df['f1_improvement'] > 0).sum(),
            'positive_accuracy_improvements': (df['accuracy_improvement'] > 0).sum(),
            'max_f1_improvement': df['f1_improvement'].max(),
            'max_accuracy_improvement': df['accuracy_improvement'].max(),
            'min_f1_improvement': df['f1_improvement'].min(),
            'min_accuracy_improvement': df['accuracy_improvement'].min()
        }
        
        return summary
