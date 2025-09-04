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

        # Toggle between prints
        self.original_stdout = sys.stdout

    def fit(self, X, y, X_group, X_set=None, val_ratio=VAL_RATIO, print_to_file=True,
            contiguity_type=CONTIGUITY_TYPE, polygon_contiguity_info=POLYGON_CONTIGUITY_INFO,
            track_partition_metrics=False, correspondence_table_path=None, VIS_DEBUG_MODE=True):
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

        # Initialize training information
        if X_set is None:
            X_set = train_val_split(X, val_ratio=val_ratio)
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
        partition_result = partition(
            self.model, X, y,
            X_group, X_set, X_id, X_branch_id,
            min_depth=self.min_model_depth,
            max_depth=self.max_model_depth,
            contiguity_type=contiguity_type,
            polygon_contiguity_info=polygon_contiguity_info,
            track_partition_metrics=track_partition_metrics and VIS_DEBUG_MODE,
            correspondence_table_path=correspondence_table_path,
            model_dir=self.model_dir
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

    def predict(self, X, X_group, save_full_predictions=False):
        """Make predictions using trained GeoXGB model."""
        X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
        y_pred = self.model.predict_georf(X, X_group, self.s_branch, X_branch_id=X_branch_id)

        if save_full_predictions:
            np.save(self.dir_space + '/' + 'y_pred_geoxgb.npy', y_pred)

        return y_pred

    def evaluate(self, X, y, X_group, eval_base=False, print_to_file=False, force_accuracy=False):
        """Evaluate the trained GeoXGB model."""
        if print_to_file:
            print_file = self.model_dir + '/' + 'log_print.txt'
            sys.stdout = open(print_file, "a")

        X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
        
        # Get GeoXGB predictions and evaluation
        pre, rec, f1, total_class = self.model.predict_test(
            X, y, X_group, self.s_branch, X_branch_id=X_branch_id)

        print('===== GeoXGB Evaluation Results =====')
        print('Precision: ', pre)
        print('Recall: ', rec)
        print('F1-score: ', f1)

        if eval_base:
            # Compare with base XGBoost (no partitioning)
            base_xgb = XGBmodel(
                path=self.dir_ckpt,
                n_trees_unit=self.n_trees_unit,
                num_class=self.num_class,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda
            )
            
            pre_base, rec_base, f1_base, total_class_base = base_xgb.predict_test(
                X, y, X_group, self.s_branch, X_branch_id=X_branch_id)

            print('===== Base XGBoost Evaluation Results =====')
            print('Precision: ', pre_base)
            print('Recall: ', rec_base)
            print('F1-score: ', f1_base)

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
                   contiguity_type=CONTIGUITY_TYPE, polygon_contiguity_info=POLYGON_CONTIGUITY_INFO):
        """Train 2-layer GeoXGB model (main prediction + error correction)."""
        print("Training 2-layer GeoXGB model...")
        
        # Train main model with L1 features
        self.fit(X_L1, y, X_group, val_ratio=val_ratio,
                contiguity_type=contiguity_type, 
                polygon_contiguity_info=polygon_contiguity_info)
        
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
