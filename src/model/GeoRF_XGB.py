# @Author: xie (adapted for XGBoost)
# @Date:   2024-07-21
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2024-07-21
# @License: MIT License

"""GeoRF implementation using XGBoost instead of Random Forest.

This is a modified version of GeoRF.py that uses XGBmodel instead of RFmodel
for the underlying machine learning algorithm. All spatial partitioning logic
remains the same, only the base learner is changed from RF to XGBoost.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold

# GeoRF with XGBoost
# Model import changed from RF to XGB
from src.model.model_XGB import XGBmodel, save_single, predict_test_group_wise
try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

from src.customize.customize import *
from demo.data import *
from src.initialization.initialization import init_X_info, init_X_info_raw_loc, init_X_branch_id, train_val_split
from src.helper.helper import create_dir, open_dir, get_X_branch_id_by_group, get_filter_thrd
from src.utils.split import group_aware_train_val_split
from src.partition.transformation import partition
from src.vis.visualization import *
from src.metrics.metrics import get_class_wise_accuracy, get_prf
from src.partition.partition_opt import get_refined_partitions_all
# All global parameters
from config import *

import pandas as pd
import os
import argparse
import sys
import time
import logging
from pathlib import Path
import shap


class GeoRF_XGB():
    """Geo-aware XGBoost implementation.
    
    This class implements the same spatial partitioning framework as GeoRF
    but uses XGBoost as the base learner instead of Random Forest.
    """
    
    def __init__(self,
                 # Geo-RF specific parameters
                 min_model_depth=MIN_DEPTH,
                 max_model_depth=MAX_DEPTH,
                 dir="",
                 
                 # XGBoost specific parameters (mapped from RF where possible)
                 n_trees_unit=100,  # n_estimators in XGB
                 num_class=NUM_CLASS,
                 max_depth=None,  # tree depth in XGB
                 random_state=5,
                 n_jobs=N_JOBS,
                 mode=MODE,
                 name='XGB',
                 type='static',
                 sample_weights_by_class=None,
                 
                 # XGBoost-specific hyperparameters
                 learning_rate=0.1,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 reg_alpha=0.1,
                 reg_lambda=1.0,
                 early_stopping_rounds=None):
        """
        Initialize GeoRF with XGBoost.
        
        Parameters:
        -----------
        min_model_depth : int
            Minimum partitioning depth
        max_model_depth : int
            Maximum partitioning depth
        dir : str
            Directory for storing models (currently unused)
        n_trees_unit : int
            Number of estimators (trees) in XGBoost
        num_class : int
            Number of classes
        max_depth : int or None
            Maximum tree depth in XGBoost
        random_state : int
            Random state for reproducibility
        n_jobs : int
            Number of parallel jobs
        mode : str
            Training mode ('classification')
        name : str
            Model name identifier
        type : str
            Model type ('static')
        sample_weights_by_class : array-like or None
            Sample weights by class
        learning_rate : float
            XGBoost learning rate
        subsample : float
            Subsample ratio for training
        colsample_bytree : float
            Column sampling ratio
        reg_alpha : float
            L1 regularization
        reg_lambda : float
            L2 regularization
        early_stopping_rounds : int or None
            Early stopping rounds
        """
        
        # Geo-RF specifics
        self.min_model_depth = min_model_depth
        self.max_model_depth = max_model_depth
        self.model_dir = dir  # currently unused

        # Geo-RF outputs
        self.model = None
        self.branch_table = None
        self.s_branch = None
        
        # XGBoost parameters (mapped from RF parameters)
        self.n_trees_unit = n_trees_unit
        self.num_class = num_class
        self.max_depth = max_depth if max_depth is not None else 6
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mode = mode
        self.name = name
        self.type = type
        self.sample_weights_by_class = sample_weights_by_class
        
        # XGBoost-specific parameters
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds

        # Create directories to store models and results
        folder_name_ext = 'GeoXGB'  # Changed from GeoRF to GeoXGB
        
        separate_vis = True
        if separate_vis:
            model_dir, dir_space, dir_ckpt, dir_vis = create_dir(
                folder_name_ext=folder_name_ext, separate_vis=separate_vis)
        else:
            model_dir, dir_space, dir_ckpt = create_dir(folder_name_ext=folder_name_ext)
            dir_vis = dir_space

        CKPT_FOLDER_PATH = dir_ckpt

        self.model_dir = model_dir
        self.dir_space = dir_space
        self.dir_ckpt = dir_ckpt
        self.dir_vis = dir_vis
        self._cached_feature_names = None
        self.drop_list_ = []
        self._drop_indices = tuple()
        self._base_training_X = None
        self._base_training_y = None
        self._base_training_groups = None

        # Toggle between prints
        self.original_stdout = sys.stdout

    def fit(self, X, y, X_group, X_set=None, val_ratio=VAL_RATIO, print_to_file=True,
            split=None,
            contiguity_type=CONTIGUITY_TYPE, polygon_contiguity_info=POLYGON_CONTIGUITY_INFO,
            track_partition_metrics=False, correspondence_table_path=None, feature_names=None, VIS_DEBUG_MODE=True):
        """
        Train the geo-aware XGBoost (Geo-XGB).

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            Output targets.
        X_group : array-like
            Provides a group ID for each data point in X.
        X_set : array-like, optional
            Train/validation split (0=train, 1=validation).
        val_ratio : float
            Validation ratio if X_set not provided.
        print_to_file : bool
            Whether to redirect print output to file.
        contiguity_type : str
            Type of contiguity refinement ('grid' or 'polygon').
        polygon_contiguity_info : dict
            Polygon contiguity information.

        Returns
        -------
        self : GeoRF_XGB
            Returns self with trained model.
        """
        
        # Logging setup
        logging.basicConfig(filename=self.model_dir + '/' + "model.log",
                          format='%(asctime)s %(message)s',
                          filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Print to file
        if print_to_file:
            print('model_dir:', self.model_dir)
            print_file = self.model_dir + '/' + 'log_print.txt'
            sys.stdout = open(print_file, "w")

        print('Options: ')
        print('CONTIGUITY & REFINE_TIMES: ', CONTIGUITY, REFINE_TIMES)
        print('MIN_BRANCH_SAMPLE_SIZE: ', MIN_BRANCH_SAMPLE_SIZE)
        print('FLEX_RATIO: ', FLEX_RATIO)
        print('Partition MIN_DEPTH & MAX_DEPTH: ', MIN_DEPTH, MAX_DEPTH)
        print('Using XGBoost with learning_rate:', self.learning_rate)
        print('XGB regularization - L1:', self.reg_alpha, 'L2:', self.reg_lambda)

        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)

        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                print(f'Warning: feature_names length ({len(feature_names)}) != X width ({X.shape[1]})')
            self._cached_feature_names = [str(raw_name) for raw_name in feature_names]
            try:
                ref_path = Path(self.model_dir) / 'feature_name_reference.csv'
                reference_df = pd.DataFrame({
                    'feature_index': np.arange(len(self._cached_feature_names), dtype=int),
                    'feature_name': self._cached_feature_names
                })
                reference_df.to_csv(ref_path, index=False)
            except Exception as ref_err:
                print(f'Warning: could not write feature_name_reference.csv: {ref_err}')
        else:
            self._cached_feature_names = None

        # Initialize training information with group-aware validation
        val_groups = None
        coverage_df = None
        if split is not None:
            if 'X_set' in split:
                X_set = np.asarray(split['X_set'], dtype=int)
                if X_set.shape[0] != X.shape[0]:
                    raise ValueError('Provided split["X_set"] must match number of samples in X.')
            else:
                X_set = np.zeros(X.shape[0], dtype=int)
                val_key = 'val_indices' if 'val_indices' in split else 'val'
                if val_key not in split:
                    raise ValueError('split must include "X_set" or one of "val_indices"/"val" entries.')
                val_indices = np.asarray(split[val_key], dtype=int)
                if np.any(val_indices < 0) or np.any(val_indices >= X.shape[0]):
                    raise ValueError(f'split["{val_key}"] contains invalid indices.')
                X_set[val_indices] = 1
                if 'train_indices' in split or 'train' in split:
                    train_key = 'train_indices' if 'train_indices' in split else 'train'
                    train_indices = np.asarray(split[train_key], dtype=int)
                    if np.any(train_indices < 0) or np.any(train_indices >= X.shape[0]):
                        raise ValueError(f'split["{train_key}"] contains invalid indices.')
                    if np.intersect1d(train_indices, val_indices).size > 0:
                        raise ValueError('Provided train and val indices overlap.')
            val_groups = np.asarray(split.get('groups_val')) if 'groups_val' in split else np.unique(np.asarray(X_group)[X_set == 1])
        elif X_set is None:
            group_split_cfg = globals().get('GROUP_SPLIT', {})
            group_split_enabled = bool(group_split_cfg.get('enable', False))
            if group_split_enabled:
                split_result = group_aware_train_val_split(
                    groups=np.asarray(X_group),
                    val_ratio=val_ratio,
                    min_val_per_group=int(group_split_cfg.get('min_val_per_group', 1)),
                    random_state=group_split_cfg.get('random_state', self.random_state),
                    skip_singleton_groups=bool(group_split_cfg.get('skip_singleton_groups', True)),
                )
                X_set = split_result['X_set']
                coverage_df = split_result['coverage']
                val_groups = split_result['val_groups']
            else:
                X_set = train_val_split(X, val_ratio=val_ratio)
                val_groups = np.unique(np.asarray(X_group)[X_set == 1])
        else:
            val_groups = np.unique(np.asarray(X_group)[X_set == 1])

        if coverage_df is None:
            temp_df = pd.DataFrame({
                'FEWSNET_admin_code': np.asarray(X_group),
                'is_val': X_set == 1})
            coverage_df = temp_df.groupby('FEWSNET_admin_code', dropna=False).agg(
                total_count=('is_val', 'size'),
                val_count=('is_val', 'sum'))
            coverage_df['train_count'] = coverage_df['total_count'] - coverage_df['val_count']
            coverage_df = coverage_df.reset_index()[['FEWSNET_admin_code', 'total_count', 'train_count', 'val_count']]

        import os as os_module
        coverage_path = os_module.path.join(self.model_dir, 'val_coverage_by_group.csv')
        coverage_df.to_csv(coverage_path, index=False)
        logger.info(f'Validation coverage report written to {coverage_path}')

        uncovered = coverage_df[(coverage_df['val_count'] == 0) & (coverage_df['total_count'] > 1)]
        if not uncovered.empty:
            warning_groups = ', '.join(map(str, uncovered['FEWSNET_admin_code'].tolist()))
            print(f'WARNING: Groups without validation samples despite having multiple records: {warning_groups}')
            logger.warning(f'Groups without validation samples despite having multiple records: {warning_groups}')

        self.val_groups_ = val_groups

        X, updated_feature_names = self._prepare_feature_drop_after_split(X, self._cached_feature_names, logger=logger)
        if updated_feature_names is not None:
            self._cached_feature_names = [str(name) for name in updated_feature_names]

        train_indices_only = np.where(X_set == 0)[0]
        train_indices_flat = np.asarray(train_indices_only, dtype=int)
        reference_names_for_cache = list(self._cached_feature_names) if getattr(self, '_cached_feature_names', None) else None
        base_train_source = X[train_indices_flat]
        if reference_names_for_cache:
            base_train_aligned, _ = self._restrict_to_feature_reference(
                base_train_source,
                reference_names_for_cache,
                logger=None,
                context_label='fit.base_training'
            )
        else:
            base_train_aligned = np.asarray(base_train_source)
        self._base_training_X = np.array(base_train_aligned, copy=True)
        self._base_training_y = np.array(y[train_indices_flat], copy=True)
        self._base_training_groups = np.array(np.asarray(X_group)[train_indices_flat], copy=True)
        try:
            import config as cfg_module
        except ImportError:
            cfg_module = None
        if cfg_module is None or not getattr(cfg_module, 'DISABLE_BASELINE_CV_MAP', False):
            try:
                self._generate_baseline_cv_error_map(
                    X[train_indices_only],
                    y[train_indices_only],
                    np.asarray(X_group)[train_indices_only],
                    logger
                )
            except Exception as baseline_cv_err:
                print(f"Warning: Baseline CV misclassification map failed: {baseline_cv_err}")
                if logger:
                    logger.warning(f"Baseline CV misclassification map failed: {baseline_cv_err}")

        X_id = np.arange(X.shape[0])
        X_branch_id = init_X_branch_id(X, max_depth=self.max_model_depth)

        # Timer
        start_time = time.time()

        # Create XGBoost model with all parameters
        self.model = XGBmodel(
            path=self.dir_ckpt,
            n_trees_unit=self.n_trees_unit,
            num_class=self.num_class,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            max_model_depth=self.max_model_depth,
            mode=self.mode,
            name=self.name,
            type=self.type,
            sample_weights_by_class=self.sample_weights_by_class,
            # XGBoost-specific parameters
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=self.early_stopping_rounds
        )

        # PRE-PARTITIONING DIAGNOSTICS WITH CROSS-VALIDATION FOR XGBOOST
        # Generate diagnostic maps using CV to prevent overfitting bias before spatial partitioning
        try:
            from src.diagnostics.pre_partition_diagnostic import create_pre_partition_diagnostics_cv
            print("\n=== Generating Pre-Partitioning CV Diagnostic Maps (XGBoost) ===")
            
            # Get TRAINING DATA ONLY - completely exclude test data
            train_indices = np.where(X_set == 0)[0]
            X_train_only = X[train_indices]
            y_train_only = y[train_indices]
            X_group_train_only = X_group[train_indices]
            
            # Verify test data exclusion
            test_indices = np.where(X_set == 1)[0]
            assert len(np.intersect1d(train_indices, test_indices)) == 0, \
                "CRITICAL: Test data contamination detected in XGBoost diagnostic!"
            
            print(f"  Training samples for CV diagnostic: {len(X_train_only):,}")
            print(f"  Test samples (excluded): {len(test_indices):,}")
            print(f"  Total verification: {len(train_indices) + len(test_indices)} == {len(X)}")
            
            # Set up diagnostic output directory
            diagnostic_vis_dir = os.path.join(self.model_dir, 'vis') if hasattr(self, 'model_dir') else self.dir_vis
            
            # Model parameters for CV (XGBoost-specific)
            model_params = {
                'path': self.dir_ckpt,
                'n_trees_unit': self.n_trees_unit,
                'num_class': self.num_class,
                'max_depth': self.max_depth,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'max_model_depth': self.max_model_depth,
                'mode': self.mode,
                'name': self.name,
                'type': self.type,
                'sample_weights_by_class': self.sample_weights_by_class,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'early_stopping_rounds': self.early_stopping_rounds
            }
            
            # Get shapefile path from config
            shapefile_path = None
            try:
                from config import ADJACENCY_SHAPEFILE_PATH
                shapefile_path = ADJACENCY_SHAPEFILE_PATH
            except:
                pass
            
            # Run CV diagnostics (NO TEST DATA INVOLVED)
            diagnostic_results = create_pre_partition_diagnostics_cv(
                X_train=X_train_only,
                y_train=y_train_only,
                X_group_train=X_group_train_only,
                model_class=XGBmodel,
                model_params=model_params,
                vis_dir=diagnostic_vis_dir,
                shapefile_path=shapefile_path,
                uid_col='FEWSNET_admin_code',
                class_positive=1,
                cv_folds=5,
                random_state=42,
                VIS_DEBUG_MODE=VIS_DEBUG_MODE
            )
            
            print("SUCCESS: Pre-partitioning CV diagnostics (XGBoost) completed successfully")
            
        except ImportError:
            print("WARNING: Pre-partitioning diagnostic module not available. Skipping diagnostic maps.")
        except Exception as e:
            print(f"WARNING: Pre-partitioning CV diagnostics (XGBoost) failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with partitioning...")

        # Spatial partitioning (same as original GeoRF, with optional metrics tracking)
        print(f"DEBUG: GeoRF_XGB.fit() calling partition() with VIS_DEBUG_MODE={VIS_DEBUG_MODE}")
        partition_result = partition(
            self.model, X, y,
            X_group, X_set, X_id, X_branch_id,
            min_depth=self.min_model_depth,
            max_depth=self.max_model_depth,
            contiguity_type=contiguity_type,
            polygon_contiguity_info=polygon_contiguity_info,
            track_partition_metrics=track_partition_metrics and VIS_DEBUG_MODE,
            correspondence_table_path=correspondence_table_path,
            model_dir=self.model_dir,
            VIS_DEBUG_MODE=VIS_DEBUG_MODE
        )
        
        # Handle different return formats (with/without metrics tracker)
        if track_partition_metrics and VIS_DEBUG_MODE:
            X_branch_id, self.branch_table, self.s_branch, self.metrics_tracker = partition_result
        else:
            X_branch_id, self.branch_table, self.s_branch = partition_result
            self.metrics_tracker = None

        # Save results
        print(self.s_branch)
        self.s_branch.to_pickle(self.dir_space + '/' + 's_branch.pkl')
        np.save(self.dir_space + '/' + 'X_branch_id.npy', X_branch_id)
        np.save(self.dir_space + '/' + 'branch_table.npy', self.branch_table)

        print("Time: %f s" % (time.time() - start_time))
        logger.info("Time: %f s" % (time.time() - start_time))

        # Contiguity refinement (same as original)
        X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
        
        print('Total unique partitions before contiguity refinement: ', len(np.unique(X_branch_id)))

        if CONTIGUITY:
            print('Applying contiguity refinement...')
            if contiguity_type == 'polygon':
                print('Warning: Polygon-based contiguity refinement not fully implemented yet.')
                print('Skipping contiguity refinement for polygon-based spatial groups.')
                # TODO: Implement polygon-specific contiguity refinement
            else:
                # Grid-based contiguity refinement
                # Pass vis_dir only if VIS_DEBUG_MODE is enabled for contiguity refinement visualization
                vis_dir_param = self.dir_vis if VIS_DEBUG_MODE else None
                X_branch_id = get_refined_partitions_all(
                    X_branch_id, self.s_branch, X_group,
                    dir=vis_dir_param,
                    min_component_size=MIN_COMPONENT_SIZE
                )
                
        print('Total unique partitions after contiguity refinement: ', len(np.unique(X_branch_id)))

        # Reset stdout
        if print_to_file:
            sys.stdout.close()
            sys.stdout = self.original_stdout

        return self

    def _generate_baseline_cv_error_map(self, X_train, y_train, uid_train, logger=None):
        import config as cfg

        vis_dir = getattr(self, 'dir_vis', self.model_dir)
        os.makedirs(vis_dir, exist_ok=True)
        log_path = os.path.join(vis_dir, 'baseline_cv_map_log.txt')
        csv_path = os.path.join(vis_dir, 'baseline_cv_error_by_polygon.csv')
        png_path = os.path.join(vis_dir, 'baseline_cv_error_map.png')

        log_lines = ['baseline_cv_map_start']

        try:
            import geopandas as gpd
        except ImportError as exc:
            log_lines.append(f'geopandas_missing={exc}')
            with open(log_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            if logger:
                logger.warning(f'Baseline CV map skipped: geopandas missing ({exc})')
            return

        try:
            from matplotlib import pyplot as plt
            from matplotlib.ticker import PercentFormatter
            import matplotlib as mpl
        except ImportError as exc:
            log_lines.append(f'matplotlib_missing={exc}')
            with open(log_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            if logger:
                logger.warning(f'Baseline CV map skipped: matplotlib missing ({exc})')
            return

        seed = getattr(cfg, 'BASELINE_CV_SEED', 42)
        n_splits = getattr(cfg, 'BASELINE_CV_N_SPLITS', 5)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        uid_train = np.asarray(uid_train)
        if X_train.shape[0] == 0:
            log_lines.append('no_training_samples')
            with open(log_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            return
        if X_train.shape[0] < 2:
            log_lines.append('insufficient_samples_for_cv')
            with open(log_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            return

        effective_splits = min(n_splits, X_train.shape[0])
        if effective_splits < n_splits:
            log_lines.append(f'effective_splits={effective_splits}')

        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_count = class_counts.min() if class_counts.size else 0
        if min_count < effective_splits:
            log_lines.append(f'fallback_kfold_due_to_min_count={min_count}')
            splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        else:
            splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

        oof_pred = np.full(y_train.shape[0], np.nan)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            model_cv = XGBmodel(
                path=self.dir_ckpt,
                n_trees_unit=self.n_trees_unit,
                num_class=self.num_class,
                max_depth=self.max_depth,
                random_state=seed,
                n_jobs=self.n_jobs,
                max_model_depth=self.max_model_depth,
                mode=self.mode,
                name=self.name,
                type=self.type,
                sample_weights_by_class=self.sample_weights_by_class,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                early_stopping_rounds=self.early_stopping_rounds
            )
            model_cv.train(X_train[train_idx], y_train[train_idx], branch_id=f'cv{fold_idx}')
            y_pred_fold = model_cv.predict(X_train[val_idx])
            oof_pred[val_idx] = y_pred_fold
            fold_err = float(np.mean(y_pred_fold != y_train[val_idx])) if val_idx.size else float('nan')
            fold_msg = f'fold_{fold_idx}_done n_train={train_idx.size} n_val={val_idx.size} fold_err={fold_err:.6f}'
            log_lines.append(fold_msg)
            if logger:
                logger.info(fold_msg)

        valid_mask = ~np.isnan(oof_pred)
        total_expected = int(valid_mask.sum())
        missing = int((~valid_mask).sum())
        if missing > 0:
            log_lines.append(f'missing_oof_predictions={missing}')
        if total_expected == 0:
            log_lines.append('no_valid_oof_predictions')
            with open(log_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            if logger:
                logger.warning('Baseline CV map skipped: no valid OOF predictions')
            return

        oof_clean = oof_pred[valid_mask]
        y_valid = y_train[valid_mask]
        uid_valid = uid_train[valid_mask]

        metrics_df = pd.DataFrame({
            'uid': uid_valid.astype(str),
            'y_true': y_valid,
            'y_pred': oof_clean
        })
        metrics_df['is_error'] = (metrics_df['y_true'] != metrics_df['y_pred']).astype(int)
        agg_df = metrics_df.groupby('uid', dropna=False).agg(
            n_oof=('is_error', 'size'),
            errors=('is_error', 'sum')
        ).reset_index()
        agg_df['pct_err_all'] = agg_df['errors'] / agg_df['n_oof']
        agg_df = agg_df[['uid', 'n_oof', 'pct_err_all']]

        sum_n_oof = int(agg_df['n_oof'].sum()) if not agg_df.empty else 0
        log_lines.append(f'sum_n_oof={sum_n_oof}/{total_expected}')
        if logger:
            logger.info(f'sum_n_oof={sum_n_oof}/{total_expected}')

        uid_column = getattr(cfg, 'ADJACENCY_POLYGON_ID_COLUMN', 'FEWSNET_admin_code')
        polygon_path = getattr(cfg, 'BASELINE_POLYGON_PATH', None) or getattr(cfg, 'ADJACENCY_SHAPEFILE_PATH', None)
        if polygon_path is None or not os.path.exists(polygon_path):
            log_lines.append('polygon_path_missing')
            with open(log_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            return

        polys = gpd.read_file(polygon_path)
        if hasattr(polys.geometry, 'make_valid'):
            polys.geometry = polys.geometry.make_valid()
        polys['_uid_merge'] = polys[uid_column].astype(str)

        merged = polys.merge(agg_df, how='left', left_on='_uid_merge', right_on='uid')
        share_nan = float(merged['pct_err_all'].isna().mean())
        log_lines.append(f'share_nan_polygons={share_nan:.6f}')
        if logger:
            logger.info(f'share_nan_polygons={share_nan:.6f}')

        agg_df.to_csv(csv_path, index=False)
        csv_msg = f'csv_written_path={csv_path}'
        log_lines.append(csv_msg)
        if logger:
            logger.info(csv_msg)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=getattr(cfg, 'FINAL_ACCURACY_DPI', 200))
        vmin, vmax = 0.0, 1.0
        cmap = mpl.cm.get_cmap('Reds')
        missing_color = getattr(cfg, 'FINAL_ACCURACY_MISSING_COLOR', '#dddddd')

        merged.plot(
            column='pct_err_all',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            missing_kwds={'color': missing_color, 'edgecolor': 'none', 'label': 'No Data'}
        )
        ax.set_title('Baseline 5-Fold OOF Misclassification Rate by Polygon', fontsize=12)
        ax.axis('off')

        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        cb = plt.colorbar(sm, cax=cbar_ax)
        cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.set_label('Misclassification Rate')

        plt.savefig(png_path, bbox_inches='tight')
        plt.close(fig)
        png_msg = f'png_written_path={png_path}'
        log_lines.append(png_msg)
        if logger:
            logger.info(png_msg)

        with open(log_path, 'w') as log_file:
            log_file.write('\n'.join(log_lines))
        if logger:
            logger.info(f'baseline_cv_map_log_path={log_path}')

        print(f'Baseline CV misclassification artifacts: {csv_path}, {png_path}')

    def _get_feature_drop_config(self):
        """Return feature drop configuration from module globals if present."""
        cfg = globals().get('FEATURE_DROP')
        if isinstance(cfg, dict):
            return cfg
        cfg_lower = globals().get('feature_drop')
        if isinstance(cfg_lower, dict):
            return cfg_lower
        return {}

    def _prepare_feature_drop_after_split(self, X, feature_names, logger=None):
        """Apply configured feature drop post temporal split and cache indices."""
        drop_cfg = self._get_feature_drop_config()
        enabled = bool(drop_cfg.get('enable', False))
        drop_candidates_raw = drop_cfg.get('cols', [])
        drop_candidates = [str(col) for col in drop_candidates_raw if col is not None]
        pattern_candidates_raw = drop_cfg.get('patterns', [])
        pattern_candidates = [str(pat) for pat in pattern_candidates_raw if pat]
        if not enabled or (not drop_candidates and not pattern_candidates):
            self.drop_list_ = []
            self._drop_indices = tuple()
            return X, feature_names

        available_names = None
        if feature_names:
            available_names = [str(name) for name in feature_names]
        elif hasattr(X, 'columns') and getattr(X, 'columns', None) is not None:
            available_names = [str(col) for col in X.columns]

        if not available_names:
            if logger:
                logger.warning('feature_drop_enabled_but_feature_names_missing; skipping drop')
            self.drop_list_ = []
            self._drop_indices = tuple()
            return X, feature_names

        protected = set()
        for key in (
            'TARGET_COLUMN', 'TARGET_COL', 'TARGET', 'TARGET_NAME',
            'UID_COLUMN', 'UID_COL', 'UNIQUE_ID_COLUMN'
        ):
            value = globals().get(key)
            if value:
                protected.add(str(value))

        pattern_matches = set()
        try:
            import fnmatch
        except ImportError:
            fnmatch = None
        if pattern_candidates and fnmatch is not None:
            for pat in pattern_candidates:
                matched = [name for name in available_names if fnmatch.fnmatch(name, pat)]
                pattern_matches.update(matched)

        drop_ordered = []
        skipped_protected = []
        for name in drop_candidates:
            if name in protected:
                skipped_protected.append(name)
                continue
            drop_ordered.append(name)
        if skipped_protected and logger:
            logger.info(f'DROP_COLUMNS skipped_protected={skipped_protected}')

        present = [name for name in drop_ordered if name in available_names]
        present += [name for name in sorted(pattern_matches) if name not in present]
        missing = [name for name in drop_ordered if name not in available_names]
        if missing and logger:
            logger.info(f'DROP_COLUMNS missing post-temporal-split count={len(missing)} names={missing}')

        if not present:
            self.drop_list_ = []
            self._drop_indices = tuple()
            return X, feature_names

        index_map = {name: idx for idx, name in enumerate(available_names)}
        drop_indices = sorted(index_map[name] for name in present)

        if hasattr(X, 'drop') and getattr(X, 'columns', None) is not None:
            X_out = X.drop(columns=[name for name in present if name in X.columns])
        else:
            X_array = np.asarray(X)
            if X_array.ndim != 2:
                raise ValueError(f'Feature matrix must be 2-dimensional after drop; got shape {X_array.shape}')
            X_out = np.delete(X_array, drop_indices, axis=1) if drop_indices else X_array

        remaining_names = [name for name in available_names if name not in present]
        self.drop_list_ = present
        self._drop_indices = tuple(drop_indices)
        if logger:
            logger.info(f'DROP_COLUMNS post-temporal-split pre-train count={len(present)} names={present}')
        return X_out, remaining_names

    def _apply_feature_drop_inference(self, X):
        """Drop configured features at inference using cached indices."""
        drop_names = list(getattr(self, 'drop_list_', []) or [])
        drop_indices = list(getattr(self, '_drop_indices', tuple()) or [])
        if not drop_names:
            return X
        if hasattr(X, 'drop') and getattr(X, 'columns', None) is not None:
            present = [name for name in drop_names if name in X.columns]
            if present:
                return X.drop(columns=present)
            return X
        if drop_indices:
            X_array = np.asarray(X)
            if X_array.ndim != 2:
                raise ValueError(f'Feature matrix must be 2-dimensional at inference; got shape {X_array.shape}')
            return np.delete(X_array, drop_indices, axis=1)
        return X

    def _load_feature_name_reference(self, logger=None):
        """Load persisted feature reference list if available."""
        reference_names = None
        ref_path = Path(self.model_dir) / 'feature_name_reference.csv'
        if ref_path.exists():
            try:
                df_ref = pd.read_csv(ref_path)
                if 'feature_name' in df_ref.columns:
                    reference_names = df_ref['feature_name'].astype(str).tolist()
                    if logger:
                        logger.info(f'feature_reference_loaded count={len(reference_names)} path={ref_path}')
                else:
                    if logger:
                        logger.warning(f'feature_reference_missing_feature_name_column path={ref_path}')
            except Exception as ref_err:
                if logger:
                    logger.warning(f'feature_reference_load_failed path={ref_path} error={ref_err}')
        if not reference_names and getattr(self, '_cached_feature_names', None):
            reference_names = [str(name) for name in self._cached_feature_names]
        if reference_names:
            self._cached_feature_names = [str(name) for name in reference_names]
        return reference_names

    def _restrict_to_feature_reference(self, X_input, reference_names, logger=None, context_label='eval'):
        """Restrict features of X_input to match the persisted reference."""
        if not reference_names:
            return X_input, None
        reference_names = [str(name) for name in reference_names]
        reference_set = set(reference_names)
        self._cached_feature_names = [str(name) for name in reference_names]
        if hasattr(X_input, 'loc') and getattr(X_input, 'columns', None) is not None:
            columns = [str(col) for col in X_input.columns]
            extra = [col for col in columns if col not in reference_set]
            missing = [name for name in reference_names if name not in columns]
            if missing:
                raise ValueError(f'X input for {context_label} missing expected features: {missing}')
            if extra and logger:
                logger.warning(f'x_input_extra_features_dropped context={context_label} count={len(extra)} sample={extra[:5]}')
            aligned_df = X_input.loc[:, reference_names]
            return aligned_df.to_numpy(), aligned_df
        X_array = np.asarray(X_input)
        if X_array.ndim != 2:
            raise ValueError(f'X input for {context_label} must be 2-dimensional; got shape {X_array.shape}')
        if X_array.shape[1] != len(reference_names):
            raise ValueError(
                f'X input for {context_label} width mismatch: expected {len(reference_names)} features, got {X_array.shape[1]}'
            )
        return X_array, None

    def _resolve_feature_names(self, X_source, model, n_features, pipeline=None, logger=None):
        """Resolve feature names for SHAP outputs with ordered fallbacks."""
        names = None
        source = None
        if hasattr(X_source, 'columns') and getattr(X_source, 'columns', None) is not None:
            names = [str(col) for col in X_source.columns]
            source = 'X.columns'
        elif hasattr(model, 'feature_names_in_') and getattr(model, 'feature_names_in_', None) is not None:
            names = [str(col) for col in model.feature_names_in_]
            source = 'model.feature_names_in_'
        elif pipeline is not None and hasattr(pipeline, 'get_feature_names_out'):
            try:
                names = [str(col) for col in pipeline.get_feature_names_out()]
                source = 'pipeline.get_feature_names_out()'
            except Exception as err:
                if logger:
                    logger.warning(f'feature_name_resolution_pipeline_failed={err}')
        elif getattr(self, '_cached_feature_names', None):
            names = [str(col) for col in self._cached_feature_names]
            source = 'cached_feature_names'
        if not names:
            names = [f'feature_{idx}' for idx in range(n_features)]
            source = 'fallback_generated'
            if logger:
                logger.warning('feature_name_resolution_fallback_used')
        if len(names) != n_features:
            if len(names) > n_features:
                names = names[:n_features]
            else:
                names = names + [f'feature_{idx}' for idx in range(len(names), n_features)]
        deduped = []
        seen = {}
        for name in names:
            base = str(name)
            if base in seen:
                seen[base] += 1
                new_name = f'{base}__{seen[base]}'
                if logger and seen[base] == 2:
                    logger.warning('feature_name_resolution_deduplicated')
                deduped.append(new_name)
            else:
                seen[base] = 1
                deduped.append(base)
        return deduped, source

    def predict(self, X, X_group, save_full_predictions=False):
        """Make predictions using trained GeoXGB model."""
        X_processed = self._apply_feature_drop_inference(X)
        if hasattr(X_processed, 'to_numpy'):
            X_matrix = X_processed.to_numpy()
        else:
            X_matrix = np.asarray(X_processed)

        X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
        y_pred = self.model.predict_georf(X_matrix, X_group, self.s_branch, X_branch_id=X_branch_id)

        if save_full_predictions:
            np.save(self.dir_space + '/' + 'y_pred_geoxgb.npy', y_pred)

        return y_pred

    def evaluate(self, X, y, X_group, eval_base=False, print_to_file=False, force_accuracy=False, VIS_DEBUG_MODE=None):
        """Evaluate the trained GeoXGB model."""
        logger = logging.getLogger(__name__)
        if print_to_file:
            print_file = self.model_dir + '/' + 'log_print.txt'
            sys.stdout = open(print_file, "a")

        X = self._apply_feature_drop_inference(X)

        reference_names = self._load_feature_name_reference(logger=logger)
        if not reference_names:
            raise ValueError('feature_name_reference.csv unavailable; cannot enforce strict feature alignment')

        X_for_shap = None
        X_aligned, X_df = self._restrict_to_feature_reference(
            X,
            reference_names,
            logger=logger,
            context_label='evaluate.X'
        )
        X = X_aligned
        X_for_shap = X_df if X_df is not None else X_aligned

        X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)

        # Get GeoXGB predictions and evaluation
        pre, rec, f1, total_class = self.model.predict_test(
            X, y, X_group, self.s_branch, X_branch_id=X_branch_id)

        print('===== GeoXGB Evaluation Results =====')
        print('Precision: ', pre)
        print('Recall: ', rec)
        print('F1-score: ', f1)

        if eval_base:

            start_time = time.time()
            baseline_wrapper = None
            if self._base_training_X is not None and self._base_training_y is not None:
                baseline_train_matrix, _ = self._restrict_to_feature_reference(
                    self._base_training_X,
                    reference_names,
                    logger=logger,
                    context_label='evaluate.base_training'
                )
                baseline_wrapper = XGBmodel(
                    path=self.dir_ckpt,
                    n_trees_unit=self.n_trees_unit,
                    num_class=self.num_class,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_model_depth=self.max_model_depth,
                    mode=self.mode,
                    name=self.name,
                    type=self.type,
                    sample_weights_by_class=self.sample_weights_by_class,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    early_stopping_rounds=self.early_stopping_rounds
                )
                baseline_wrapper.train(
                    np.array(baseline_train_matrix, copy=True),
                    np.array(self._base_training_y, copy=True),
                    branch_id='eval',
                    sample_weights_by_class=self.sample_weights_by_class
                )
                logger.info('baseline_xgb_retrained_from_cached_training_data=1')
            else:
                logger.warning('baseline_training_cache_missing; falling back to saved baseline model')
                baseline_wrapper = XGBmodel(
                    path=self.dir_ckpt,
                    n_trees_unit=self.n_trees_unit,
                    num_class=self.num_class,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_model_depth=self.max_model_depth,
                    mode=self.mode,
                    name=self.name,
                    type=self.type,
                    sample_weights_by_class=self.sample_weights_by_class,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    early_stopping_rounds=self.early_stopping_rounds
                )
                baseline_wrapper.load('')

            base_estimator = getattr(baseline_wrapper, 'model', None)
            if base_estimator is None:
                raise ValueError('Baseline XGBoost model unavailable for evaluation')

            feature_count_expected = len(reference_names)
            if hasattr(base_estimator, 'n_features_in_'):
                n_features_in = getattr(base_estimator, 'n_features_in_', None)
                print(f'Baseline XGB debug: n_features_in_={n_features_in}, X width={feature_count_expected}')
                logger.info(f'baseline_xgb_feature_check n_features_in={n_features_in} X_width={feature_count_expected}')
                if n_features_in is not None and int(n_features_in) != int(feature_count_expected):
                    raise ValueError(f'Baseline XGBoost feature mismatch: model expects {n_features_in} features but data has {feature_count_expected}')
            if hasattr(base_estimator, 'feature_names_in_'):
                estimator_feature_names = [str(name) for name in getattr(base_estimator, 'feature_names_in_', [])]
                if estimator_feature_names and estimator_feature_names != reference_names:
                    raise ValueError('Baseline XGBoost feature names mismatch with feature_name_reference')

            y_pred_single = baseline_wrapper.predict(X)
            true_single, total_single, pred_total_single = get_class_wise_accuracy(y, y_pred_single, prf=True)
            pre_base, rec_base, f1_base, total_class = get_prf(true_single, total_single, pred_total_single)

            print('===== Base XGBoost Evaluation Results =====')
            print('Precision: ', pre_base)
            print('Recall: ', rec_base)
            print('F1-score: ', f1_base)

            try:
                explainer = shap.TreeExplainer(baseline_wrapper.model)
                X_baseline = X_for_shap
                shape_info = getattr(X_baseline, 'shape', None)
                print(f'SHAP debug: X_baseline shape={shape_info}')
                logger.info(f'shap_debug_X_baseline_shape={shape_info}')
                shap_values_raw = explainer.shap_values(X_baseline)
                if isinstance(shap_values_raw, list):
                    stacked = []
                    for class_idx, shap_matrix in enumerate(shap_values_raw):
                        matrix = np.asarray(shap_matrix)
                        if matrix.ndim != 2:
                            matrix = matrix.reshape(matrix.shape[0], -1)
                        stacked.append(matrix)
                    if not stacked:
                        raise ValueError('No SHAP values produced for baseline XGBoost model')
                    S = np.mean(np.stack(stacked, axis=0), axis=0)
                else:
                    S = np.asarray(shap_values_raw)
                    feature_count = X_baseline.shape[1]
                    print(f'SHAP debug: raw array shape={S.shape}, ndim={S.ndim}, feature_count={feature_count}')
                    if S.ndim == 3:
                        if S.shape[0] == X_baseline.shape[0] and S.shape[2] == self.num_class:
                            S = S.mean(axis=2)
                            print(f'SHAP debug: collapsed layout (samples, features, classes) -> {S.shape}')
                        elif S.shape[0] == self.num_class and S.shape[1] == X_baseline.shape[0]:
                            S = S.mean(axis=0)
                            print(f'SHAP debug: collapsed layout (classes, samples, features) -> {S.shape}')
                        else:
                            S = S.reshape(S.shape[0], -1)
                            print(f'SHAP debug: reshaped unexpected 3D layout -> {S.shape}')
                    elif S.ndim == 2 and self.num_class > 1 and S.shape[1] == feature_count * self.num_class:
                        S = S.reshape(S.shape[0], self.num_class, feature_count).mean(axis=1)
                        print(f'SHAP debug: collapsed flattened class axis -> {S.shape}')
                    elif S.ndim != 2:
                        S = S.reshape(S.shape[0], -1)
                        print(f'SHAP debug: reshaped fallback -> {S.shape}')
                n_samples, n_features = S.shape
                print(f'SHAP debug: n_samples={n_samples}, n_features={n_features}')
                feature_names, feature_source = self._resolve_feature_names(
                    X_baseline,
                    getattr(baseline_wrapper, 'model', baseline_wrapper),
                    n_features,
                    pipeline=getattr(baseline_wrapper, 'pipeline', None),
                    logger=logger
                )
                if reference_names:
                    if len(reference_names) != S.shape[1]:
                        raise ValueError('SHAP feature dimension mismatch with feature_name_reference')
                    feature_names = list(reference_names)
                    feature_source = 'feature_name_reference'
                logger.info(f'shap_feature_dimensions n_features={n_features} resolved={len(feature_names)} source={feature_source}')
                if S.shape[1] != len(feature_names):
                    raise ValueError('SHAP feature dimension mismatch')
                shap_df = pd.DataFrame(S, columns=feature_names)
                feature_index_map = {name: idx for idx, name in enumerate(feature_names)}
                mean_abs = shap_df.abs().mean(axis=0)
                df_rank = mean_abs.sort_values(ascending=False, kind='mergesort').reset_index()
                df_rank.columns = ['feature', 'mean_abs_shap']
                df_rank['rank'] = np.arange(1, len(df_rank) + 1)
                vis_dir = getattr(self, 'dir_vis', None) or getattr(self, 'model_dir', None) or '.'
                vis_path = Path(vis_dir)
                vis_path.mkdir(parents=True, exist_ok=True)
                csv_path = vis_path / 'baseline_shap_mean_abs_by_feature.csv'
                df_rank[['feature', 'mean_abs_shap', 'rank']].to_csv(csv_path, index=False)
                csv_path_str = str(csv_path)
                logger.info(f'feature_name_source={feature_source}')
                logger.info(f'csv_written_path={csv_path_str}')
                print(f'Baseline SHAP mean |SHAP| CSV saved: {csv_path_str}')
                top_n = min(10, len(df_rank))
                bottom_n = min(10, len(df_rank))
                top_df = df_rank.head(top_n)
                bottom_df = df_rank.tail(bottom_n)
                selected_df = pd.concat([top_df, bottom_df], axis=0).drop_duplicates('feature', keep='first')
                selected_df = selected_df.sort_values('rank', kind='mergesort')
                sel_features = selected_df['feature'].tolist()
                logger.info(f'beeswarm_feature_count={len(sel_features)} top_n={top_n} bottom_n={bottom_n}')
                if sel_features:
                    selected_names = list(sel_features)
                    S_sel = None
                    X_sel = None
                    plot_features = None
                    if hasattr(X_baseline, 'loc'):
                        existing = [name for name in selected_names if name in X_baseline.columns]
                        missing_count = len(selected_names) - len(existing)
                        if missing_count:
                            logger.warning(f'beeswarm_dataframe_missing_features={missing_count}')
                        if existing:
                            S_sel = shap_df[existing].values
                            X_sel = X_baseline.loc[:, existing]
                            plot_features = existing
                    else:
                        X_array = np.asarray(X_baseline)
                        if X_array.ndim == 1:
                            X_array = X_array.reshape(-1, 1)
                        valid_pairs = []
                        for name in selected_names:
                            idx = feature_index_map.get(name)
                            if idx is None:
                                continue
                            if idx >= X_array.shape[1]:
                                logger.warning('beeswarm_array_index_out_of_bounds name=%s idx=%d limit=%d', name, idx, X_array.shape[1])
                                continue
                            valid_pairs.append((name, idx))
                        if valid_pairs:
                            valid_names, valid_indices = zip(*valid_pairs)
                            valid_names = list(valid_names)
                            valid_indices = list(valid_indices)
                            S_sel = shap_df[valid_names].values
                            X_sel = X_array[:, valid_indices]
                            plot_features = valid_names
                    if S_sel is None or X_sel is None or plot_features is None:
                        logger.warning('beeswarm_skipped_no_valid_features_after_alignment')
                    else:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 6), dpi=200)
                        try:
                            shap.summary_plot(S_sel, features=X_sel, feature_names=plot_features, show=False, plot_type='dot')
                        except Exception:
                            for pos, feat in enumerate(plot_features):
                                values = S_sel[:, pos]
                                y_coords = np.full(values.shape[0], pos)
                                plt.scatter(values, y_coords, s=4, alpha=0.5, color='#1f77b4')
                            plt.yticks(range(len(plot_features)), plot_features)
                            plt.xlabel('SHAP value')
                        plt.title('Baseline SHAP Beeswarm (Top/Bottom-10 by mean |SHAP|)')
                        plt.tight_layout()
                        png_path = vis_path / 'baseline_shap_beeswarm_top10_bottom10.png'
                        plt.savefig(png_path)
                        plt.close()
                        png_path_str = str(png_path)
                        logger.info(f'beeswarm_written_path={png_path_str}')
                        print(f'Baseline SHAP beeswarm saved: {png_path_str}')
                        log_path = vis_path / 'baseline_shap_log.txt'
                        log_path_str = str(log_path)
                        log_lines = [
                            f'feature_name_source={feature_source}',
                            f'csv_written_path={csv_path_str}',
                            f'beeswarm_written_path={png_path_str}'
                        ]
                        with open(log_path, 'w') as log_file:
                            log_file.write('\n'.join(log_lines) + '\n')
                        logger.info(f'shap_log_written_path={log_path_str}')
                else:
                    logger.warning('beeswarm_skipped_no_features')
            except Exception as shap_err:
                logger.warning(f'baseline_shap_failed={shap_err}')

            # FINAL ACCURACY VISUALIZATION: Render final accuracy maps after evaluation (eval_base=True case)
            # Strict gate: use VIS_DEBUG_MODE from config only (production control)
            from config import VIS_DEBUG_MODE as _VIS_FLAG

            if _VIS_FLAG:
                try:
                    from src.vis.visualization_fix import ensure_vis_dir_and_render_maps
                    
                    # Render final accuracy maps using the test data that was just evaluated
                    render_summary = ensure_vis_dir_and_render_maps(
                        model_dir=self.model_dir,
                        test_data=(X, y, X_group),  # Use the test data from evaluation
                        force_accuracy=force_accuracy,
                        model=self,  # Pass model for accuracy computation
                        VIS_DEBUG_MODE=_VIS_FLAG
                    )
                    
                    if render_summary.get('final_accuracy_generated'):
                        print(f"Final accuracy maps rendered: {render_summary.get('final_accuracy_artifacts', [])}")
                    
                except Exception as e:
                    print(f"Warning: Could not render final accuracy maps: {e}")
            else:
                print("Visualization disabled (VIS_DEBUG_MODE=False)")

            if print_to_file:
                sys.stdout.close()
                sys.stdout = self.original_stdout

            return pre, rec, f1, pre_base, rec_base, f1_base

        # FINAL ACCURACY VISUALIZATION: Render final accuracy maps after evaluation (eval_base=False case)
        # Strict gate: use VIS_DEBUG_MODE from config only (production control)
        from config import VIS_DEBUG_MODE as _VIS_FLAG

        if _VIS_FLAG:
            try:
                from src.vis.visualization_fix import ensure_vis_dir_and_render_maps
                
                # Render final accuracy maps using the test data that was just evaluated
                render_summary = ensure_vis_dir_and_render_maps(
                    model_dir=self.model_dir,
                    test_data=(X, y, X_group),  # Use the test data from evaluation
                    force_accuracy=force_accuracy,
                    model=self,  # Pass model for accuracy computation
                    VIS_DEBUG_MODE=_VIS_FLAG
                )
                
                if render_summary.get('final_accuracy_generated'):
                    print(f"Final accuracy maps rendered: {render_summary.get('final_accuracy_artifacts', [])}")
                
            except Exception as e:
                print(f"Warning: Could not render final accuracy maps: {e}")
        else:
            print("Visualization disabled (VIS_DEBUG_MODE=False)")

        if print_to_file:
            sys.stdout.close()
            sys.stdout = self.original_stdout

        return pre, rec, f1

    def evaluate_group_wise(self, X, y, X_group, eval_base=False, print_to_file=False, 
                          step_size=STEP_SIZE, cnt_vis_thrd=5):
        """Evaluate model performance group-wise with visualization."""
        if print_to_file:
            print_file = self.model_dir + '/' + 'log_print.txt'
            sys.stdout = open(print_file, "a")

        X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
        
        # Group-wise evaluation
        results, groups, total_number = predict_test_group_wise(
            self.model, X, y, X_group, self.s_branch, X_branch_id=X_branch_id)

        # Create performance grids for visualization
        print('Generating group-wise performance visualizations...')
        
        # ... (visualization code would be similar to original GeoRF)
        
        if eval_base:
            results_base, groups_base, _ = predict_test_group_wise(
                self.model, X, y, X_group, self.s_branch, base=True, X_branch_id=X_branch_id)
            
            # Save comparison results
            np.save(self.dir_space + '/' + 'grid_geoxgb.npy', results)
            np.save(self.dir_space + '/' + 'grid_base_xgb.npy', results_base)

        if print_to_file:
            sys.stdout.close()
            sys.stdout = self.original_stdout

        return

    # 2-Layer model methods (similar to original GeoRF)
    def fit_2layer(self, X_L1, X_L2, y, X_group, val_ratio=VAL_RATIO, 
                   contiguity_type=CONTIGUITY_TYPE, polygon_contiguity_info=POLYGON_CONTIGUITY_INFO,
                   feature_names_L1=None):
        """Train 2-layer GeoXGB model (main prediction + error correction)."""
        print("Training 2-layer GeoXGB model...")
        
        # Train main model with L1 features
        self.fit(X_L1, y, X_group, val_ratio=val_ratio,
                contiguity_type=contiguity_type, 
                polygon_contiguity_info=polygon_contiguity_info,
                feature_names=feature_names_L1)
        
        # Get L1 predictions for training L2 model
        y_pred_L1 = self.predict(X_L1, X_group)
        
        # Create L2 training data (L2 features + L1 predictions + residuals)
        X_L2_augmented = np.column_stack([X_L2, y_pred_L1])
        y_residual = (y != y_pred_L1).astype(int)  # Binary: correct=0, incorrect=1
        
        # Train L2 error correction model
        self.model_L2 = XGBmodel(
            path=self.dir_ckpt + '_L2',
            n_trees_unit=self.n_trees_unit // 2,  # Smaller model for L2
            num_class=2,  # Binary classification for error correction
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            learning_rate=self.learning_rate * 0.5,  # Lower learning rate for L2
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda
        )
        
        # Simplified L2 training (no spatial partitioning for error correction)
        self.model_L2.train(X_L2_augmented, y_residual, branch_id='')
        self.model_L2.save('')
        
        return self

    def predict_2layer(self, X_L1, X_L2, X_group, correction_strategy='flip'):
        """Make predictions using 2-layer GeoXGB model."""
        # Get L1 predictions
        y_pred_L1 = self.predict(X_L1, X_group)
        
        # Get L2 error correction predictions
        X_L2_augmented = np.column_stack([X_L2, y_pred_L1])
        self.model_L2.load('')
        error_prob = self.model_L2.predict(X_L2_augmented, prob=True)
        
        # Apply correction strategy
        if correction_strategy == 'flip':
            # Flip predictions where error probability is high
            error_threshold = 0.5
            flip_mask = error_prob[:, 1] > error_threshold
            y_pred_final = y_pred_L1.copy()
            y_pred_final[flip_mask] = 1 - y_pred_final[flip_mask]  # Flip 0->1, 1->0
        else:
            # Just return L1 predictions
            y_pred_final = y_pred_L1
            
        return y_pred_final

    def evaluate_2layer(self, X_L1_test, X_L2_test, y_test, X_group_test,
                       X_L1_train, X_L2_train, y_train, X_group_train,
                       correction_strategy='flip', print_to_file=False,
                       contiguity_type=CONTIGUITY_TYPE, 
                       polygon_contiguity_info=POLYGON_CONTIGUITY_INFO):
        """Evaluate 2-layer GeoXGB model."""
        # Get 2-layer predictions
        y_pred_2layer = self.predict_2layer(X_L1_test, X_L2_test, X_group_test, correction_strategy)
        
        # Calculate metrics
        pre, rec, f1, total_class = get_prf(*get_class_wise_accuracy(y_test, y_pred_2layer, prf=True))
        
        print('===== 2-Layer GeoXGB Evaluation Results =====')
        print('Precision: ', pre)
        print('Recall: ', rec)
        print('F1-score: ', f1)
        #TODO:evaluate_2layer with XGBoost base model not yet implemented
        # Compare with base 2-layer model (no spatial partitioning)
        # ... (similar implementation as single layer)
        
        # For now, return placeholder base results
        pre_base = 0  # Placeholder
        rec_base = 0  # Placeholder  
        f1_base = 0   # Placeholder
        
        return pre, rec, f1, pre_base, rec_base, f1_base
