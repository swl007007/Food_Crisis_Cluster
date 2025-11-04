# @Author: xie (adapted for XGBoost)
# @Date:   2024-07-21
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2024-07-21
# @License: MIT License

"""XGBoost implementation replacing Random Forest in GeoRF framework.

Update notes:
1. Replaced RandomForestClassifier with XGBClassifier
2. Mapped RF hyperparameters to XGBoost equivalents where possible
3. Added XGBoost-specific parameters optimized for acute food crisis prediction
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import os

import sklearn
try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - SMOTE optional at runtime
    SMOTE = None

from sklearn.metrics import accuracy_score
from config import *
from src.helper.helper import get_X_branch_id_by_group
from src.metrics.metrics import *


class XGBmodel():
    """XGBoost model wrapper that mirrors the RFmodel interface."""
    
    def __init__(self, path, n_trees_unit,
                 max_new_forests=[1,1,1,1,1,1], num_class=NUM_CLASS, max_depth=None,
                 increase_thrd=0.05, random_state=5,
                 n_jobs=N_JOBS,
                 max_model_depth=MAX_DEPTH,
                 mode=MODE, name='XGB', type='static',
                 sample_weights_by_class=None,
                 use_smote=True,
                 smote_k_neighbors=5,
                 # XGBoost specific parameters
                 learning_rate=0.1,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 reg_alpha=0.1,
                 reg_lambda=1.0,
                 early_stopping_rounds=None):
        """
        Initialize XGBoost model with parameters mapped from RF and XGB-specific ones.
        
        Parameters:
        -----------
        path : str
            Folder path for intermediate models
        n_trees_unit : int  
            Number of estimators (mapped from RF n_estimators)
        max_new_forests : list
            Max number of new forests to add after each split (unused in XGB)
        num_class : int
            Total number of classes for the application
        max_depth : int or None
            Maximum depth of trees (mapped from RF)
        increase_thrd : float
            Performance improvement threshold (unused in XGB)
        random_state : int
            Random state for reproducibility (mapped from RF)
        n_jobs : int
            Number of parallel jobs (mapped from RF)
        max_model_depth : int
            Maximum model depth for GeoRF partitioning
        mode : str
            Mode of operation (classification/regression)
        name : str
            Model name identifier
        type : str
            Model type identifier
        sample_weights_by_class : array-like or None
            Sample weights by class
        learning_rate : float
            XGBoost learning rate (step size shrinkage)
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree
        reg_alpha : float
            L1 regularization term on weights
        reg_lambda : float
            L2 regularization term on weights  
        early_stopping_rounds : int or None
            Early stopping rounds for training
        """
        
        # RF-compatible parameters
        self.n_trees_unit = n_trees_unit  # maps to n_estimators
        self.max_new_forests = max_new_forests
        self.num_class = num_class
        self.max_depth = max_depth if max_depth is not None else 6  # XGB default
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mode = mode
        self.path = path
        self.max_model_depth = max_model_depth
        self.name = 'XGB'
        self.type = type
        self.sample_weights_by_class = sample_weights_by_class
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        self._smote_unavailable_warned = False
        self._smote_skip_logged = set()
        self._smote_failure_logged = set()

        # XGBoost-specific parameters optimized for food crisis prediction
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha  # L1 regularization helps with feature selection
        self.reg_lambda = reg_lambda  # L2 regularization for stability
        self.early_stopping_rounds = early_stopping_rounds
        
        # Model storage
        self.model = None

    def train(self, X, y, branch_id=None, mode=MODE, sample_weights=None, sample_weights_by_class=None):
        """Train XGBoost model on given data."""
        if branch_id is None:
            print('Error: branch_id is required for the XGB version.')
            
        depth = len(branch_id)
        X_balanced, y_balanced = self._apply_smote_if_needed(X, y, branch_id)
        self.model = self.get_new_xgb_model(X_balanced, y_balanced, sample_weights_by_class)

    def _apply_smote_if_needed(self, X, y, branch_id):
        """Balance training data with SMOTE when enabled and feasible."""

        def _log_skip(message):
            key = (branch_id, message)
            if key not in self._smote_skip_logged:
                print(message)
                self._smote_skip_logged.add(key)

        if not self.use_smote:
            return X, y

        if SMOTE is None:
            if not self._smote_unavailable_warned:
                print('SMOTE skipped: install imbalanced-learn to enable oversampling for XGB pipeline.')
                self._smote_unavailable_warned = True
            return X, y

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        unique_classes, counts = np.unique(y_arr, return_counts=True)
        if unique_classes.shape[0] < 2:
            _log_skip(f'SMOTE skipped for branch "{branch_id}": only one class present.')
            return X, y

        minority_count = counts.min()
        if minority_count < 2:
            _log_skip(f'SMOTE skipped for branch "{branch_id}": minority class has fewer than 2 samples.')
            return X, y

        k_neighbors = min(self.smote_k_neighbors, minority_count - 1)
        if k_neighbors < 1:
            _log_skip(f'SMOTE skipped for branch "{branch_id}": insufficient samples for synthetic neighbors.')
            return X, y

        smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)

        try:
            X_resampled, y_resampled = smote.fit_resample(X_arr, y_arr)
            synthetic_count = len(y_resampled) - len(y_arr)
            if synthetic_count > 0:
                print(f'SMOTE applied on branch "{branch_id}": added {synthetic_count} synthetic samples (k_neighbors={k_neighbors}).')
            return X_resampled, y_resampled
        except Exception as exc:
            key = (branch_id, str(exc))
            if key not in self._smote_failure_logged:
                print(f'SMOTE failed for branch "{branch_id}" ({exc}); proceeding without oversampling.')
                self._smote_failure_logged.add(key)
            return X, y

    def predict(self, X, prob=False):
        """Return predicted labels or probabilities."""
        if prob:
            y_pred_prob = self.model.predict_proba(X)
            return y_pred_prob
        
        return self.model.predict(X)

    def load(self, branch_id, fresh=True):
        """Load saved XGBoost model from disk."""
        filename_base = 'xgb_'
        with open(self.path + '/' + filename_base + branch_id, 'rb') as file:
            self.model = pickle.load(file)

    def save(self, branch_id):
        """Save current XGBoost model to disk."""
        filename = 'xgb_' + branch_id
        with open(self.path + '/' + filename, 'wb') as file:
            pickle.dump(self.model, file)

    def get_score(self, y_true, y_pred_prob):
        """Get accuracy score from predictions."""
        y_pred = np.argmax(y_pred_prob, axis=1)
        return accuracy_score(y_true, y_pred)

    def get_new_xgb_model(self, X, y, sample_weights_by_class):
        """Create and train new XGBoost model."""
        # Recover all classes (similar to RF implementation)
        X_pseudo, y_pseudo = self.get_pseudo_full_class_data(X.shape[1])
        X = np.vstack([X, X_pseudo])
        y = np.hstack([y, y_pseudo])
        
        # Determine objective based on number of classes
        if self.num_class == 2:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
        else:
            objective = 'multi:softprob'
            eval_metric = 'mlogloss'
        
        # Create XGBoost classifier with mapped and optimized parameters
        new_xgb = xgb.XGBClassifier(
            # Mapped from RF parameters
            n_estimators=self.n_trees_unit,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            
            # XGBoost-specific parameters optimized for food crisis prediction
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            
            # Objective and evaluation
            objective=objective,
            eval_metric=eval_metric,
            
            # Additional stability parameters
            scale_pos_weight=1,  # Can be adjusted for imbalanced datasets
            tree_method='hist',  # Faster training
            verbosity=0,  # Reduce output
        )
        
        # Get sample weights (use after full classes are recovered)
        if sample_weights_by_class is not None:
            sample_weights = self.get_sample_weights(y, sample_weights_by_class)
        else:
            sample_weights = None
        
        # Fit the model
        if sample_weights is not None:
            new_xgb.fit(X, y, sample_weight=sample_weights, verbose=False)
        else:
            new_xgb.fit(X, y, verbose=False)
            
        return new_xgb

    def get_sample_weights(self, y, sample_weights_by_class):
        """Get sample weights based on class weights."""
        sample_weights = sample_weights_by_class[y.astype(int)]
        return sample_weights

    def get_pseudo_full_class_data(self, n_features):
        """
        Create pseudo data to ensure all classes are represented.
        Sometimes a branch misses some of the classes in the original model.
        """
        X_pseudo = np.zeros((self.num_class, n_features))
        y_pseudo = np.array(range(0, self.num_class))
        return X_pseudo, y_pseudo

    def predict_test(self, X, y, X_group, s_branch, prf=True, X_branch_id=None, append_acc=False):
        """Make predictions and evaluate performance on test data."""
        true = 0
        total = 0
        true_class = np.zeros(self.num_class)
        total_class = np.zeros(self.num_class)
        total_pred = np.zeros(self.num_class)

        if X_branch_id is None:
            X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

        for branch_id in np.unique(X_branch_id):
            id_list = np.where(X_branch_id == branch_id)
            X_part = X[id_list]
            y_part = y[id_list]

            self.load(branch_id)
            y_pred = self.predict(X_part)

            if self.mode == 'classification':
                # Overall accuracy
                true_part, total_part = get_overall_accuracy(y_part, y_pred)
                true += true_part
                total += total_part

                # Class-wise accuracy
                true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(
                    y_part, y_pred, prf=True)
                true_class += true_class_part
                total_class += total_class_part
                total_pred += total_pred_part

        if prf:
            if append_acc:
                prf_result = list(get_prf(true_class, total_class, total_pred))
                prf_result.append(np.sum(true) / np.sum(total))
                return tuple(prf_result)
            else:
                return get_prf(true_class, total_class, total_pred)
        else:
            return true / total

    def predict_georf(self, X, X_group, s_branch, X_branch_id=None):
        """Make predictions using GeoRF partitioning strategy."""
        y_pred_full = np.zeros(X.shape[0])

        if X_branch_id is None:
            X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

        for branch_id in np.unique(X_branch_id):
            id_list = np.where(X_branch_id == branch_id)
            X_part = X[id_list]

            self.load(branch_id)
            y_pred = self.predict(X_part)

            y_pred_full[id_list] = y_pred

        return y_pred_full


def save_single(model, path, name='single'):
    """Save a single XGBoost model to disk."""
    filename = 'xgb_' + name
    with open(path + '/' + filename, 'wb') as file:
        pickle.dump(model, file)


def predict_test_group_wise(model, X, y, X_group, s_branch, prf=True, base=False, 
                           base_branch_id='', X_branch_id=None):
    """Make group-wise predictions and evaluations."""
    groups = np.unique(X_group)

    if prf:
        result = np.zeros((groups.shape[0], 4, model.num_class))
    else:
        result = np.zeros(groups.shape[0])

    total_number = np.zeros((groups.shape[0], model.num_class))
    groups[:] = 0

    if X_branch_id is None:
        X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

    cnt = 0
    for branch_id in np.unique(X_branch_id):
        id_list = np.where(X_branch_id == branch_id)
        X_part = X[id_list]
        y_part = y[id_list]
        X_group_part = X_group[id_list]

        if base:
            if cnt == 0:
                model.load(base_branch_id)
        else:  # partitioned
            model.load(branch_id)

        y_pred = model.predict(X_part)

        for group in np.unique(X_group_part):
            id_list_group = np.where(X_group_part == group)
            y_part_group = y_part[id_list_group]
            y_pred_group = y_pred[id_list_group]
            
            if model.mode == 'classification':
                # Overall accuracy
                true_part, total_part = get_overall_accuracy(y_part_group, y_pred_group)
                # Class-wise accuracy
                true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(
                    y_part_group, y_pred_group, prf=True)

                if prf:
                    result[cnt, :] = np.asarray(get_prf(true_class_part, total_class_part, total_pred_part))
                else:
                    result[cnt] = true_part / total_part

            groups[cnt] = group
            total_number[cnt, :] = total_class_part
            cnt += 1

    return result, groups, total_number
