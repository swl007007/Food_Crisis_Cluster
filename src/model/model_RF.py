# @Author: xie
# @Date:   2022-05-11
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-04-21
# @License: MIT License

'''Update notes:
1. Removing helper functions from tensorflow to build a pure RF-based version.
'''

import numpy as np
# import tensorflow as tf
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

from config import *
from src.helper.helper import get_X_branch_id_by_group
from src.metrics.metrics import *

try:
  from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - falls back when imbalanced-learn is missing
  SMOTE = None

# NUM_LAYERS = 8 #num_layers is the number of non-input layers

'''Model can be easily customized to different deep network architectures.
The following key functions need to be included, which are used in STAR training:
1. train()
2. model_complie()
3. predict(): this is the regular prediction function from X->y (using a single branch), which returns predicted labels
4. save(): save a branch
5. load(): load a branch

Optional (not called during STAR training, used either in model init() or final test):
6. predict_test(): this is for STAR's prediction, which make predictions based on which branch a sample belongs to, and returns performance metric values (can change)
'''

'''This is an example implementation for random forest.'''

class RFmodel():

  def __init__(self, path, n_trees_unit,
               max_new_forests = [1,1,1,1,1,1], num_class = NUM_CLASS, max_depth=None,
               increase_thrd = 0.05, random_state=5,
               n_jobs = N_JOBS,
               max_model_depth = MAX_DEPTH,
               mode=MODE, name = 'RF', type = 'static',
               sample_weights_by_class = None,
               use_smote=True,
               smote_k_neighbors=5):#, path = CKPT_FOLDER_PATH

    '''
    path: folder path for intermediate models
    n_trees_unit: number of trees in a unit random forest
    max_new_forests: a list containing the max number of new forests to add after each split, e.g., [1,2,4,8, ...]#unused
    num_class: total number of classes for the application (local training data may contain only a subset of classes)
    increase_thrd: when adding new forests at each new branch, we at unit-size forests one by one, until one of the following is met:
                    1. The relative performance improvement is less than increase_thrd
                    2. The number of forests added >= the corresponding value specified in max_new_forests
    '''

    #inputs
    self.n_trees_unit = n_trees_unit#number of trees for each model piece, see self.model
    self.max_new_forests = max_new_forests
    self.num_class = num_class
    self.max_depth = max_depth
    self.random_state = random_state
    self.n_jobs = n_jobs
    self.mode = mode
    self.path = path

    #define a list of models here
    self.max_model_depth = max_model_depth
    self.model = None#[None] * (2**self.max_model_depth)#[]#len(list)

    self.name = 'RF'
    self.type = type
    self.sample_weights_by_class = sample_weights_by_class
    self.use_smote = use_smote
    self.smote_k_neighbors = smote_k_neighbors
    self._smote_unavailable_warned = False
    self._smote_skip_logged = set()
    self._smote_failure_logged = set()


  def train(self, X, y, branch_id = None, mode = MODE, sample_weights = None, sample_weights_by_class = None):#, num_layers = NUM_LAYERS_DNN
    #branch_id is not always necessary, depending on the choice of ML model and design
    #here it is required, but for better compatibility with the earlier deep learning verison, leave it as "optional"
    #use the following condition to require the inclusion of branch_id here
    #the input sample_weights is not used at the moment (otherwise need to update subset functions in the general training code)

    if branch_id is None:
      print('Error: branch_id is required for the RF version.')
    depth = len(branch_id)
    # model_to_add = []
    X_balanced, y_balanced = self._apply_smote_if_needed(X, y, branch_id)
    self.model = self.get_new_forest(X_balanced, y_balanced, sample_weights_by_class)#self.

    # if mode == 'classification':
    # else:

  def _apply_smote_if_needed(self, X, y, branch_id):
    def _log_skip(message):
      key = (branch_id, message)
      if key not in self._smote_skip_logged:
        print(message)
        self._smote_skip_logged.add(key)

    if not self.use_smote:
      return X, y

    if SMOTE is None:
      if not self._smote_unavailable_warned:
        print('SMOTE skipped: install imbalanced-learn to enable oversampling.')
        self._smote_unavailable_warned = True
      return X, y

    # Ensure numpy arrays for SMOTE compatibility
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

  def predict(self, X, prob = False):
    """Return predicted labels or probabilities."""

    y_pred_prob = self.model.predict_proba(X)

    if prob:
      return y_pred_prob

    return self.model.predict(X)

  def load(self, branch_id, fresh = True):
    '''
    fresh: clear current model and load the new one
    '''
    filename_base = 'rf_'
    with open(self.path + '/' + filename_base + branch_id, 'rb') as file:
      self.model = pickle.load(file)
    # self.model = pickle.load(open(self.path + '/' + filename_base + branch_id, 'rb'))

  def save(self, branch_id):
    #only saves the current new forest (newly added one)
    filename = 'rf_' + branch_id
    # pickle.dump(self.model[-1], open(self.path + filename, 'wb'))
    with open(self.path + '/' + filename, 'wb') as file:
      pickle.dump(self.model, file)
    # pickle.dump(self.model, open(self.path + '/' + filename, 'wb'))

  def get_score(self, y_true, y_pred_prob):
    y_pred = np.argmax(y_pred_prob, axis=1)
    return accuracy_score(y_true, y_pred)

  def get_new_forest(self, X, y, sample_weights_by_class):
    import gc
    import psutil
    
    #recover all classes
    X_pseudo, y_pseudo = self.get_pseudo_full_class_data(X.shape[1])
    X = np.vstack([X, X_pseudo])
    y = np.hstack([y, y_pseudo])

    # class_weight = self.get_class_weights(y)

    # class_weight = self.get_class_weights_by_input_weights(sample_weights_by_class)
    class_weight = None

    # Force garbage collection before creating RF to free memory
    gc.collect()
    
    # ADAPTIVE MEMORY PRESSURE DETECTION
    # Check available memory and adjust n_jobs dynamically to prevent bitmap allocation failures
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent
        
        # Calculate estimated memory needs for RF training
        data_size_gb = (X.shape[0] * X.shape[1] * 8) / (1024**3)  # Rough estimate
        estimated_rf_memory = data_size_gb * self.n_jobs * 2  # Rough estimate of RF memory with parallelism
        
        # Adaptive n_jobs based on memory pressure
        if available_gb < estimated_rf_memory or used_percent > 85:
            # High memory pressure - use minimal parallelism
            adaptive_n_jobs = 1
            print(f"HIGH memory pressure detected (Available: {available_gb:.1f}GB, Used: {used_percent:.1f}%)")
            print(f"Using n_jobs=1 to prevent bitmap allocation failure")
        elif available_gb < estimated_rf_memory * 1.5 or used_percent > 75:
            # Medium memory pressure - use reduced parallelism
            adaptive_n_jobs = min(4, self.n_jobs)
            print(f"MEDIUM memory pressure detected (Available: {available_gb:.1f}GB, Used: {used_percent:.1f}%)")
            print(f"Using n_jobs={adaptive_n_jobs} to reduce memory pressure")
        else:
            # Low memory pressure - use full parallelism
            adaptive_n_jobs = self.n_jobs
            print(f"Low memory pressure (Available: {available_gb:.1f}GB, Used: {used_percent:.1f}%)")
            print(f"Using full n_jobs={adaptive_n_jobs}")
    except:
        # Fallback to original n_jobs if psutil fails
        adaptive_n_jobs = self.n_jobs
        print("Memory pressure detection failed, using original n_jobs")

    new_forest = RandomForestClassifier(n_estimators = self.n_trees_unit, max_depth=self.max_depth,
                                        random_state=self.random_state,
                                        n_jobs = adaptive_n_jobs,  # Use adaptive parallelism
                                        class_weight = class_weight)

    # new_forest.classes_ = np.array(range(0,NUM_CLASS))
    # new_forest.n_classes_ = self.num_class

    #get sample weights (use after full classes are recovered)
    if sample_weights_by_class is not None:
      sample_weights = self.get_sample_weights(y, sample_weights_by_class)
    else:
      sample_weights = None#default by sklearn
    # sample_weights = None

    try:
        new_forest.fit(X,y, sample_weights)
        print(f"Random Forest training successful with n_jobs={adaptive_n_jobs}")
    except Exception as e:
        if "bitmap" in str(e).lower() or "allocate" in str(e).lower() or "memory" in str(e).lower():
            print(f"MEMORY ERROR with n_jobs={adaptive_n_jobs}: {e}")
            print("Falling back to single-threaded execution...")
            
            # Delete failed model and cleanup
            del new_forest
            gc.collect()
            
            # Retry with single-threaded execution as last resort
            new_forest = RandomForestClassifier(n_estimators = self.n_trees_unit, max_depth=self.max_depth,
                                                random_state=self.random_state,
                                                n_jobs = 1,
                                                class_weight = class_weight)
            new_forest.fit(X,y, sample_weights)
            print("SUCCESS: Single-threaded Random Forest training completed")
        else:
            raise e
    
    return new_forest

  def get_class_weights(self, y):
    min_rf_class_sample = 5#classes with smaller than this number of samples will not receive weights
    max_class_weight = 100
    unique, counts = np.unique(y, return_counts=True)
    ratios = counts / (np.sum(counts) / unique[counts>min_rf_class_sample].shape[0])
    weights = np.zeros(unique.shape[0])
    weights[counts>min_rf_class_sample] = 1/ratios[counts>min_rf_class_sample] #filter out classes with smaller than 5 samples
    weights[weights>max_class_weight] = max_class_weight #avoid numerical instability
    class_weights = dict(zip(unique, weights))
    return class_weights

  def get_class_weights_by_input_weights(self, class_weights):
    if class_weights is not None:
      unique = np.array(range(NUM_CLASS))
      class_weights = dict(zip(unique, class_weights))
      return class_weights
    else:
      return None

  #prefixed sample weights, which might not be ideal for learning in partitions
  def get_sample_weights(self, y, sample_weights_by_class):
    sample_weights = sample_weights_by_class[y.astype(int)]
    return sample_weights

  def get_pseudo_full_class_data(self, n_features):
    '''
    Sometimes a branch misses some of the classes in the original model, which will create problems when comparing and integrating results
    '''
    X_pseudo = np.zeros((self.num_class, n_features))
    y_pseudo = np.array(range(0, self.num_class))
    return X_pseudo, y_pseudo

  def predict_test(self, X, y, X_group, s_branch, prf = True, X_branch_id = None, append_acc = False):
    #prob here is aggregated probability (does not sum to 1 without normalizing)
    '''
    group_branch contains the branch_id for each group
    '''
    true = 0
    total = 0
    true_class = np.zeros(self.num_class)
    total_class = np.zeros(self.num_class)
    total_pred =  np.zeros(self.num_class)

    if X_branch_id is None:
      X_branch_id = get_X_branch_id_by_group(X_group, s_branch)

    for branch_id in np.unique(X_branch_id):
      id_list = np.where(X_branch_id == branch_id)
      X_part = X[id_list]
      y_part = y[id_list]

      self.load(branch_id)
      y_pred = self.predict(X_part)

      if self.mode == 'classification':
        #overall
        true_part, total_part = get_overall_accuracy(y_part, y_pred)
        true += true_part
        total += total_part

        #class-wise, if needed
        true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part, y_pred, prf = True)
        true_class += true_class_part
        total_class += total_class_part
        total_pred += total_pred_part

    if prf:
      if append_acc:
        prf_result = list(get_prf(true_class, total_class, total_pred))
        prf_result.append(np.sum(true) / np.sum(total))
        return tuple(prf_result)
      else:
        return get_prf(true_class, total_class, total_pred)#acc, acc_class
    else:
      return true / total

  def predict_georf(self, X, X_group, s_branch, X_branch_id = None):
    #prob here is aggregated probability (does not sum to 1 without normalizing)
    '''
    group_branch contains the branch_id for each group
    '''

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



def save_single(model, path, name = 'single'):
    """Save a single RandomForest model to disk."""
    filename = 'rf_' + name
    with open(path + '/' + filename, 'wb') as file:
        pickle.dump(model, file)
    # pickle.dump(model, open(path + '/' + filename, 'wb'))


def predict_test_group_wise(model, X, y, X_group, s_branch, prf = True, base = False, base_branch_id = '', X_branch_id = None):
  #prob here is aggregated probability (does not sum to 1 without normalizing)
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
    else:#partitioned
      model.load(branch_id)

    y_pred = model.predict(X_part)

    for group in np.unique(X_group_part):
      id_list_group = np.where(X_group_part == group)
      y_part_group = y_part[id_list_group]
      y_pred_group = y_pred[id_list_group]
      if model.mode == 'classification':
        #overall
        true_part, total_part = get_overall_accuracy(y_part_group, y_pred_group)
        #class-wise, if needed
        true_class_part, total_class_part, total_pred_part = get_class_wise_accuracy(y_part_group, y_pred_group, prf = True)

        if prf:
          result[cnt, :] = np.asarray(get_prf(true_class_part, total_class_part, total_pred_part))
          # pre, rec, f1, _ = get_prf(true_class_part, total_class_part, total_pred_part)
          # result[cnt, :] = np.asarray([pre[1], rec[1], f1[1]])#only for class 1
        else:
          result[cnt] = true_part / total_part

      groups[cnt] = group
      # total_number[cnt, :] = total_pred_part#updated to true counts for visualization
      total_number[cnt, :] = total_class_part
      cnt += 1


  return result, groups, total_number
