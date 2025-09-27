# @Author: xie
# @Date:   2025-06-20
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-06-20
# @License: MIT License

import numpy as np
import os
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold

#GeoRF
#Can be customized with the template

# from models import DNNmodel, LSTMmodel, UNetmodel#model is easily customizable
from src.model.model_RF import RFmodel, save_single, predict_test_group_wise#model is easily customizable
from sklearn.ensemble import RandomForestClassifier
# from customize import generate_groups_nonimg_input#can customize group definition
from src.customize.customize import *
from demo.data import *
from src.initialization.initialization import init_X_info, init_X_info_raw_loc, init_X_branch_id, train_val_split
from src.helper.helper import create_dir, open_dir, get_X_branch_id_by_group, get_filter_thrd
from src.utils.split import group_aware_train_val_split
from src.partition.transformation import partition
from src.vis.visualization import *
from src.metrics.metrics import get_class_wise_accuracy, get_prf
from src.partition.partition_opt import get_refined_partitions_all
#All global parameters
from config import *

import pandas as pd
import os
import argparse
import sys
import time
import logging
from pathlib import Path
# import shap value toolkit
import shap


#search for "!!!" for places to potentially update
class GeoRF():
	def __init__(self,
							 #Geo-RF specific paras
							 min_model_depth = MIN_DEPTH,
							 max_model_depth = MAX_DEPTH,#max number of levels in bi-partitioning hierarchy (e.g., max=1 means can partition at most once)
							 dir = "",
							 #RF specific paras
							 n_trees_unit = 100, num_class = NUM_CLASS, max_depth=None,#this is max tree depth in RF
               random_state=5,
               n_jobs = N_JOBS,
               mode=MODE, name = 'RF', type = 'static',
               sample_weights_by_class = None
							 #unused paras
							 # path,!!!generate this using above dir
							 # max_model_depth = MAX_DEPTH,#moved to above
							 #increase_thrd = 0.05,
							 # max_new_forests,
							):
		#Geo-RF specifics: inputs (not including basic RF paras)
		self.min_model_depth = min_model_depth
		self.max_model_depth = max_model_depth#max partitioning depth
		self.model_dir = dir#currently unused
		#not from user inputs

		#Geo-RF specifics: outputs
		self.model = None
		#this is used to store models from GeoRF
		#[None] * (2**self.max_model_depth)#[]#len(list)
		# self.X_branch_id = None#can be derived, no need to store
		self.branch_table = None
		self.s_branch = None
		#RF inputs
		self.n_trees_unit = n_trees_unit#number of trees for each model piece, see self.model
		self.num_class = num_class
		self.max_depth = max_depth
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.mode = mode#!!!may not be needed
		self.name = name
		self.type = type#static
		self.sample_weights_by_class = sample_weights_by_class
		#self.max_new_forests = max_new_forests#unused
		# self.path = path#!!!this may not be needed (can be generated using geo-RF path)

		#Create directories to store models and results
		if MODEL_CHOICE == 'RF':
			folder_name_ext = 'GeoRF'
		else:
			folder_name_ext = 'DL'#deep learning version

		separate_vis = True
		if separate_vis:
			#model_dir: main folder for the experiment
		  #dir: folder to store geo-rf related intermediate results such as space partitions
		  #dir_ckpt: store trained models for different local models
		  #dir_vis: for visualzation
			model_dir, dir_space, dir_ckpt, dir_vis = create_dir(folder_name_ext = folder_name_ext, separate_vis = separate_vis)
		else:
			model_dir, dir_space, dir_ckpt = create_dir(folder_name_ext = folder_name_ext)
			dir_vis = dir_space

		CKPT_FOLDER_PATH = dir_ckpt#might not be used

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

		#toggle between prints
		self.original_stdout = sys.stdout
		

	#Train GeoRF
	def fit(self, X, y, X_group, X_set = None, val_ratio = VAL_RATIO, print_to_file = True, 
	        split = None,
	        contiguity_type = CONTIGUITY_TYPE, polygon_contiguity_info = POLYGON_CONTIGUITY_INFO,
	        track_partition_metrics = False, correspondence_table_path = None, feature_names=None, VIS_DEBUG_MODE=True):#X_loc is unused
		"""
    Train the geo-aware random forest (Geo-RF).

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Output targets.
		X_group: array-like
				Provides a group ID for each data point in X. The groups are groupings of locations,
				which serve two important purposes:
				(1) Minimum spatial unit: A group is the minimum spatial unit for space-partitioning
				(or just data partitioning if non-spatial data). For example, a grid/fishnet can be used
				to generate groups,	where all data points in each grid cell belong to one group. As a
				minimum spatial unit,	all points in the same group will always be placed in the same
				spatial partition.
				(2) Test point model selection: Once Geo-RF is trained, the groups are used to determine
				which local model a test point should use. First, the group ID of a test point is determined
				by its location (e.g., based on grid cells), and then the corresponding partition ID of the
				group is used to determine the local RF to use for the prediction (all groups in a spatial
				partition share the same local model.).
		X_set : array-like
        Optional. One value per data point in X, with 0 for training and 1 for validation. If this
				is not provided, val_ratio will be used to randomly assign points to validation set. In Geo-RF,
				this left-out validation set is used by default to evaluate the necessity of new partitions. It
				can be used as a usual validation set, or if desired, a separate validation set can be used for
				other hyperparameter tuning, that are independent to this set, which is not used as training samples
				but their evaluations are used in the GeoRF branching process.
		val_ratio: float
				Optional. Used if X_set is not provided to assign samples to the validation set.
		contiguity_type: str
				Optional. Type of contiguity refinement: 'grid' or 'polygon'. Default uses CONTIGUITY_TYPE from config.
		polygon_contiguity_info: dict
				Optional. Dictionary containing polygon contiguity information (centroids, group mapping, neighbor threshold).
				Required when contiguity_type='polygon'. Default uses POLYGON_CONTIGUITY_INFO from config.
		track_partition_metrics: bool
				Optional. If True, tracks F1 and accuracy metrics before/after each partition round for debugging.
				Saves CSV files and creates maps showing performance improvements by X_group. Default: False.
		correspondence_table_path: str
				Optional. Path to correspondence table CSV mapping X_group to FEWSNET_admin_code.
				Required when track_partition_metrics=True for map visualization. Should have columns:
				'X_group' or 'FEWSNET_admin_code', 'partition_id'.
    Returns
    -------
		georf: GeoRF class object
				Returns GeoRF model parameters, trained results (e.g., partitions) and pointers to trained weights.
				The trained results are needed to make spatially-explicit predictions.
    """

		#Logging: testing purpose only
		logging.basicConfig(filename=self.model_dir + '/' + "model.log",
						format='%(asctime)s %(message)s',
						filemode='w')
		logger=logging.getLogger()
		logger.setLevel(logging.INFO)

		#print to file
		if print_to_file:
			print('model_dir:', self.model_dir)
			print_file = self.model_dir + '/' + 'log_print.txt'
			sys.stdout = open(print_file, "w")

		print('Options: ')
		print('CONTIGUITY & REFINE_TIMES: ', CONTIGUITY, REFINE_TIMES)
		print('MIN_BRANCH_SAMPLE_SIZE: ', MIN_BRANCH_SAMPLE_SIZE)
		print('FLEX_RATIO: ', FLEX_RATIO)
		print('Partition MIN_DEPTH & MAX_DEPTH: ', MIN_DEPTH, MAX_DEPTH)

		print('X.shape: ', X.shape)
		print('y.shape: ', y.shape)

		if feature_names is not None:
			if len(feature_names) != X.shape[1]:
				print(f'Warning: feature_names length ({len(feature_names)}) != X width ({X.shape[1]})')
			self._cached_feature_names = [str(raw_name) for raw_name in feature_names]
			self._write_feature_reference(self._cached_feature_names, logger=logger)
		else:
			self._cached_feature_names = None


		# #for debugging#X_loc removed here
		# print(np.min(X_loc[:,0]), np.max(X_loc[:,0]))
		# print(np.min(X_loc[:,1]), np.max(X_loc[:,1]))

		#Initialize location-related and training information. Can be customized.
    #X_id stores data points' ids in the original X, and is used as a reference.
    #X_set stores train-val-test assignments: train=0, val=1, test=2
    #X_branch_id stores branch_ids (or, partion ids) of each data points. All init to route branch ''. Dynamically updated during training.
    #X_group stores group assignment: customizable. In this example, groups are defined by grid cells in space.
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
		X_id = np.arange(X.shape[0])#the id is used to later refer back to the original X, and the related information
		X_branch_id = init_X_branch_id(X, max_depth = self.max_model_depth)
		# X_group, X_set, X_id, X_branch_id = init_X_info_raw_loc(X, y, X_loc, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, step_size = STEP_SIZE, predefined = PREDEFINED_GROUPS)

		X, updated_feature_names = self._prepare_feature_drop_after_split(X, self._cached_feature_names, logger=logger)
		if updated_feature_names is not None:
			self._cached_feature_names = [str(name) for name in updated_feature_names]
			self._write_feature_reference(self._cached_feature_names, logger=logger)

		# '''RF paras''' --> unused
		# max_new_forests = [1,1,1,1,1,1]
		# sample_weights_by_class = None#np.array([0.05, 0.95])#None#np.array([0.05, 0.95])#None

		#timer
		start_time = time.time()

		# PRE-PARTITIONING DIAGNOSTICS WITH CROSS-VALIDATION
		# Generate diagnostic maps using CV to prevent overfitting bias before spatial partitioning
		train_list_init = np.where(X_set == 0)
		train_indices_flat = train_list_init[0] if isinstance(train_list_init, tuple) else np.asarray(train_list_init)
		train_indices_flat = np.asarray(train_indices_flat, dtype=int)
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
					X[train_list_init],
					y[train_list_init],
					np.asarray(X_group)[train_list_init],
					logger
				)
			except Exception as baseline_cv_err:
				print(f"Warning: Baseline CV misclassification map failed: {baseline_cv_err}")
				if logger:
					logger.warning(f"Baseline CV misclassification map failed: {baseline_cv_err}")
		try:
			from src.diagnostics.pre_partition_diagnostic import create_pre_partition_diagnostics_cv
			print("\n=== Generating Pre-Partitioning CV Diagnostic Maps ===")
			
			# Get TRAINING DATA ONLY - completely exclude test data
			train_indices = np.where(X_set == 0)[0]
			X_train_only = X[train_indices]
			y_train_only = y[train_indices]
			X_group_train_only = X_group[train_indices]
			
			# Verify test data exclusion
			test_indices = np.where(X_set == 1)[0]
			assert len(np.intersect1d(train_indices, test_indices)) == 0, \
				"CRITICAL: Test data contamination detected in diagnostic!"
			
			print(f"  Training samples for CV diagnostic: {len(X_train_only):,}")
			print(f"  Test samples (excluded): {len(test_indices):,}")
			print(f"  Total verification: {len(train_indices) + len(test_indices)} == {len(X)}")
			
			# Set up diagnostic output directory
			import os as os_module
			diagnostic_vis_dir = os_module.path.join(self.model_dir, 'vis') if hasattr(self, 'model_dir') else self.dir_vis
			
			# Model parameters for CV
			model_params = {
				'dir_ckpt': self.dir_ckpt,
				'n_trees_unit': self.n_trees_unit,
				'max_depth': self.max_depth
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
				model_class=RFmodel,
				model_params=model_params,
				vis_dir=diagnostic_vis_dir,
				shapefile_path=shapefile_path,
				uid_col='FEWSNET_admin_code',
				class_positive=1,
				cv_folds=5,
				random_state=42,
				VIS_DEBUG_MODE=VIS_DEBUG_MODE
			)
			
			print("SUCCESS: Pre-partitioning CV diagnostics completed successfully")
			
		except ImportError:
			print("WARNING: Pre-partitioning diagnostic module not available. Skipping diagnostic maps.")
		except Exception as e:
			print(f"WARNING: Pre-partitioning CV diagnostics failed: {e}")
			import traceback
			traceback.print_exc()
			print("  Continuing with partitioning...")

		#Train to stablize before starting the first data partitioning
		if MODEL_CHOICE == 'RF':
			#RF
			self.model = RFmodel(self.dir_ckpt, self.n_trees_unit, max_depth = self.max_depth)#can add sample_weights_by_class
			self.model.train(X[train_list_init], y[train_list_init], branch_id = '', sample_weights_by_class = self.sample_weights_by_class)

		self.model.save('')#save root branch

		print("Time single: %f s" % (time.time() - start_time))
		logger.info("Time single: %f s" % (time.time() - start_time))

		#Spatial transformation (data partitioning, not necessarily for spatial data).
		#This will automatically partition data into subsets during training, so that each subset follows a homogeneous distribution.
		#format of branch_id: for example: '0010' refers to a branch after four bi-partitionings (four splits),
		  #and 0 or 1 shows the partition it belongs to after each split.
		  #'' is the root branch (before any split).
		#s_branch: another key output, that stores the group ids for all branches.
		#X_branch_id: contains the branch_id for each data point.
		#branch_table: shows which branches are further split and which are not.
		print("="*50)
		print(f"GEORF.PY DEBUG: GeoRF.fit() calling partition() with VIS_DEBUG_MODE={VIS_DEBUG_MODE}")
		print("="*50)
		partition_result = partition(self.model, X, y,
		                   X_group , X_set, X_id, X_branch_id,
		                   min_depth = self.min_model_depth, max_depth = self.max_model_depth,
		                   contiguity_type = contiguity_type, polygon_contiguity_info = polygon_contiguity_info,
		                   track_partition_metrics = track_partition_metrics and VIS_DEBUG_MODE, 
		                   correspondence_table_path = correspondence_table_path,
		                   model_dir = self.model_dir,
		                   VIS_DEBUG_MODE = VIS_DEBUG_MODE)#X_loc = X_loc is unused
		
		# Handle different return formats (with/without metrics tracker)
		if track_partition_metrics and VIS_DEBUG_MODE:
			X_branch_id, self.branch_table, self.s_branch, self.metrics_tracker = partition_result
		else:
			X_branch_id, self.branch_table, self.s_branch = partition_result
			self.metrics_tracker = None

		#Save s_branch
		print(self.s_branch)
		self.s_branch.to_pickle(self.dir_space + '/' + 's_branch.pkl')
		np.save(self.dir_space + '/' + 'X_branch_id.npy', X_branch_id)
		np.save(self.dir_space + '/' + 'branch_table.npy', self.branch_table)

		print("Time: %f s" % (time.time() - start_time))
		logger.info("Time: %f s" % (time.time() - start_time))

		#update branch_id for test data
		X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)#should be the same (previously fixed some potential inconsistency)

		#Optional: Improving Spatial Contiguity
		#The default function only works for groups defined by a grid, where majority voting in local neighborhoods are used to remove
		#fragmented partitions (e.g., one grid cell with a different partition ID from most of its neighbors).'''
		## GLOBAL_CONTIGUITY = False#unused (mentioned in visualization part later)
		if CONTIGUITY:
			# Pass vis_dir only if VIS_DEBUG_MODE is enabled for contiguity refinement visualization
			vis_dir_param = self.dir_vis if VIS_DEBUG_MODE else None
			X_branch_id = get_refined_partitions_all(X_branch_id, self.s_branch, X_group, dir = vis_dir_param, min_component_size = MIN_COMPONENT_SIZE, VIS_DEBUG_MODE=VIS_DEBUG_MODE)
		## 	GLOBAL_CONTIGUITY = True#unused

		# VISUALIZATION FIX: Always render essential maps regardless of conditions
		try:
			from src.vis.visualization_fix import ensure_vis_dir_and_render_maps
			
			# Create correspondence table for visualization with proper inheritance
			from config_visual import VALID_PARTITION_LABELS
			
			from src.merge.terminal import build_terminal
			print("CORRESPONDENCE TABLE BUILD:")
			print(f"  X_branch_id length (temporal records): {len(X_branch_id):,}")
			print(f"  X_group length (temporal records): {len(X_group):,}")
			print(f"  Unique admin units: {len(np.unique(X_group)):,}")
			correspondence_df, terminal_meta = build_terminal(X_group, X_branch_id)
			print(f"  Correspondence entries created: {len(correspondence_df):,}")
			print(f"  Collisions detected: {terminal_meta.get('n_collisions', 0)}")
			if terminal_meta.get('collisions'):
				print(f"  Collision sample: {terminal_meta['collisions']}")
			
			# Count partitions
			partition_count = len(self.s_branch) if hasattr(self, 's_branch') and self.s_branch is not None else 0
			
			# Generate frequency reports and trace logs
			if correspondence_df is not None and len(correspondence_df) > 0:
				# Create vis directory for reports
				import os
				vis_dir = os.path.join(self.model_dir, 'vis')
				os.makedirs(vis_dir, exist_ok=True)
				
				# Label frequency analysis
				partition_freqs = correspondence_df['partition_id'].value_counts().sort_index()
				branch_freqs = correspondence_df['branch_id'].value_counts()
				
				print(f"Partition label frequencies: {dict(partition_freqs)}")
				print(f"Branch ID frequencies: {dict(branch_freqs)}")
				
				# Save frequency table
				freq_data = []
				for pid in partition_freqs.index:
					freq_data.append({
						'label_type': 'partition_id',
						'label_value': pid,
						'frequency': partition_freqs[pid],
						'percentage': partition_freqs[pid] / len(correspondence_df) * 100
					})
				
				for bid in branch_freqs.index:
					freq_data.append({
						'label_type': 'branch_id', 
						'label_value': bid,
						'frequency': branch_freqs[bid],
						'percentage': branch_freqs[bid] / len(correspondence_df) * 100
					})
				
				freq_df = pd.DataFrame(freq_data)
				freq_path = os.path.join(vis_dir, 'label_freqs.csv')
				freq_df.to_csv(freq_path, index=False, encoding='utf-8')
				print(f"Label frequencies saved: {freq_path}")
				
				# Stage trace log
				trace_path = os.path.join(vis_dir, 'stage_trace.txt')
				with open(trace_path, 'a', encoding='utf-8') as f:
					import datetime
					timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
					f.write(f"\n=== PARTITION INHERITANCE ANALYSIS - {timestamp} ===\n")
					f.write(f"Model directory: {self.model_dir}\n")
					f.write(f"Total admin units: {len(correspondence_df)}\n")
					f.write(f"Partition count: {partition_count}\n")
					f.write(f"Branch count: {len(branch_freqs)}\n")
					f.write(f"Partition frequencies: {dict(partition_freqs)}\n")
					f.write(f"Branch frequencies: {dict(branch_freqs)}\n")
					
					# Check for problematic assignments
					unassigned_count = correspondence_df['partition_id'].isna().sum()
					if unassigned_count > 0:
						f.write(f"WARNING: {unassigned_count} unassigned partition IDs found\n")
					
					# Check for -1 partitions
					from config_visual import UNASSIGNED_LABELS
					problematic_partitions = correspondence_df[correspondence_df['partition_id'].isin(UNASSIGNED_LABELS)]
					if len(problematic_partitions) > 0:
						f.write(f"WARNING: {len(problematic_partitions)} polygons assigned to unassigned labels {UNASSIGNED_LABELS}\n")
						# Save dropped IDs
						dropped_path = os.path.join(vis_dir, 'dropped_or_unassigned_ids.txt')
						with open(dropped_path, 'w', encoding='utf-8') as drop_f:
							drop_f.write(f"# Polygons assigned to unassigned labels {UNASSIGNED_LABELS}\n")
							drop_f.write(f"# Generated: {timestamp}\n")
							for idx, row in problematic_partitions.iterrows():
								drop_f.write(f"{row['FEWSNET_admin_code']},{row['partition_id']},{row['branch_id']}\n")
						f.write(f"Dropped IDs saved: {dropped_path}\n")
				
				print(f"Stage trace updated: {trace_path}")
			
			# Render maps with comprehensive logging (only if VIS_DEBUG_MODE enabled)
			if VIS_DEBUG_MODE:
			# render_summary = ensure_vis_dir_and_render_maps(
				render_summary = ensure_vis_dir_and_render_maps(
					model_dir=self.model_dir,
					correspondence_df=correspondence_df,
					test_data=None,  # Test data would need to be passed from caller
					partition_count=partition_count,
					stage_info="post-training-with-fixes",
					model=self,  # Pass model for accuracy computation
					VIS_DEBUG_MODE=VIS_DEBUG_MODE
				)
				
				print(f"Visualization fix applied successfully: {len(render_summary['artifacts_rendered'])} maps rendered")
			else:
				print("Visualization disabled (VIS_DEBUG_MODE=False)")
			
		except Exception as vis_error:
			print(f"Warning: Visualization fix failed: {vis_error}")
			# Continue execution even if visualization fails

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
			model_cv = RFmodel(
				self.dir_ckpt,
				self.n_trees_unit,
				max_depth=self.max_depth,
				num_class=self.num_class,
				random_state=seed,
				n_jobs=self.n_jobs,
				sample_weights_by_class=self.sample_weights_by_class
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
			of = oof_pred[valid_mask]
			y_valid = y_train[valid_mask]
			uid_valid = uid_train[valid_mask]
		else:
			of = oof_pred
			y_valid = y_train
			uid_valid = uid_train

		metrics_df = pd.DataFrame({
			'uid': uid_valid.astype(str),
			'y_true': y_valid,
			'y_pred': of
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

	def _write_feature_reference(self, names, logger=None):
		if not names:
			return
		try:
			ref_path = Path(self.model_dir) / 'feature_name_reference.csv'
			reference_df = pd.DataFrame({
				'feature_index': np.arange(len(names), dtype=int),
				'feature_name': [str(name) for name in names]
			})
			reference_df.to_csv(ref_path, index=False)
		except Exception as ref_err:
			if logger:
				logger.warning(f'feature_reference_write_failed path={ref_path} error={ref_err}')

	def _get_feature_drop_config(self):
		cfg = globals().get('FEATURE_DROP')
		if isinstance(cfg, dict):
			return cfg
		cfg_lower = globals().get('feature_drop')
		if isinstance(cfg_lower, dict):
			return cfg_lower
		return {}

	def _prepare_feature_drop_after_split(self, X, feature_names, logger=None):
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
		for key in ('TARGET_COLUMN', 'TARGET_COL', 'TARGET', 'TARGET_NAME', 'UID_COLUMN', 'UID_COL', 'UNIQUE_ID_COLUMN'):
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
			X = X.drop(columns=[name for name in present if name in X.columns])
			X_out = X.to_numpy()
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

	def _resolve_feature_names(self, X_source, model, n_features, pipeline=None, logger=None):
		"""Resolve feature names with fallbacks and duplicate handling."""
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
		for idx, name in enumerate(names):
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

	def predict(self, X, X_group, save_full_predictions = False):
		"""
    Evaluating GeoRF and/or RF.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Output targets.
		X_group: array-like
				Same way of assignment as training. See detailed explanations in training.
		save_full_predictions: boolean
				Optional. If True, save predictions to file.
    Returns
    -------
		y_pred: array-like
				Returns predictions.
    """

		#Model assignment
		X_processed = self._apply_feature_drop_inference(X)
		if hasattr(X_processed, 'to_numpy'):
			X_matrix = X_processed.to_numpy()
		else:
			X_matrix = np.asarray(X_processed)

		X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
		y_pred = self.model.predict_georf(X_matrix, X_group, self.s_branch, X_branch_id = X_branch_id)

		if save_full_predictions:
			np.save(self.dir_space + '/' + 'y_pred_georf.npy', y_pred)

		return y_pred

	def evaluate(self, Xtest, ytest, Xtest_group, eval_base = False, print_to_file = True, force_accuracy = False, VIS_DEBUG_MODE=None):
		"""
		Evaluating GeoRF and/or RF.

		Parameters
		----------
		Xtest: array-like
			Input features.
		ytest: array-like
			Output targets.
		Xtest_group: array-like
			Same way of assignment as training. See detailed explanations in training.
		eval_base: boolean
			Optional. If True, base RF will be evaluated for comparison.
		print_to_file: boolean
			Optional. If True, prints will go to file.
		Returns
		-------
		pre, rec, f1: array-like
			Precision, recall and F1 scores. Separate for different classes in arrays.
		pre_single, rec_single, f1_single: array-like
			If eval_base is True, additionally returns results for the base RF model.
		"""

		# Logging
		logging.basicConfig(filename=self.model_dir + '/' + "model_eval.log",
						format='%(asctime)s %(message)s',
						filemode='w')
		logger=logging.getLogger()
		logger.setLevel(logging.INFO)

		# Print to file
		if print_to_file:
			print('model_dir:', self.model_dir)
			print('Printing to file.')
			print_file = self.model_dir + '/' + 'log_print_eval.txt'
			sys.stdout = open(print_file, "w")

		Xtest = self._apply_feature_drop_inference(Xtest)

		reference_names = self._load_feature_name_reference(logger=logger)
		if not reference_names:
			raise ValueError('feature_name_reference.csv unavailable; cannot enforce strict feature alignment')
		Xtest_for_shap = None
		Xtest_aligned, Xtest_df = self._restrict_to_feature_reference(
			Xtest,
			reference_names,
			logger=logger,
			context_label='evaluate.Xtest'
		)
		Xtest = Xtest_aligned
		Xtest_for_shap = Xtest_df if Xtest_df is not None else Xtest_aligned

		# Geo-RF
		start_time = time.time()

		# Model assignment
		Xtest_branch_id = get_X_branch_id_by_group(Xtest_group, self.s_branch)

		pre, rec, f1, total_class = self.model.predict_test(Xtest, ytest, Xtest_group, self.s_branch, X_branch_id = Xtest_branch_id)
		print('f1:', f1)
		log_print = ', '.join('%f' % value for value in f1)
		logger.info('f1: %s' % log_print)
		logger.info("Pred time: GeoRF: %f s" % (time.time() - start_time))

		# Base RF
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
				baseline_wrapper = RFmodel(
					self.dir_ckpt,
					self.n_trees_unit,
					max_depth=self.max_depth,
					num_class=self.num_class,
					random_state=self.random_state,
					n_jobs=self.n_jobs,
					sample_weights_by_class=self.sample_weights_by_class
				)
				baseline_wrapper.train(
					np.array(baseline_train_matrix, copy=True),
					np.array(self._base_training_y, copy=True),
					branch_id='eval',
					sample_weights_by_class=self.sample_weights_by_class
				)
				logger.info('baseline_rf_retrained_from_cached_training_data=1')
			else:
				logger.warning('baseline_training_cache_missing; falling back to saved baseline model')
				self.model.load('')
				baseline_wrapper = self.model
			base_estimator = getattr(baseline_wrapper, 'model', None)
			feature_count_expected = len(reference_names)
			if base_estimator is not None and hasattr(base_estimator, 'n_features_in_'):
				n_features_in = getattr(base_estimator, 'n_features_in_', None)
				print(f'Baseline RF debug: n_features_in_={n_features_in}, Xtest width={feature_count_expected}')
				logger.info(f'baseline_rf_feature_check n_features_in={n_features_in} Xtest_width={feature_count_expected}')
				if n_features_in is not None and int(n_features_in) != int(feature_count_expected):
					raise ValueError(f'Baseline RF feature mismatch: model expects {n_features_in} features but Xtest has {feature_count_expected}')
			if base_estimator is not None and hasattr(base_estimator, 'feature_names_in_'):
				estimator_feature_names = [str(name) for name in getattr(base_estimator, 'feature_names_in_', [])]
				if estimator_feature_names and estimator_feature_names != reference_names:
					raise ValueError('Baseline RF feature names mismatch with feature_name_reference')
			y_pred_single = baseline_wrapper.predict(Xtest)
			true_single, total_single, pred_total_single = get_class_wise_accuracy(ytest, y_pred_single, prf = True)
			pre_single, rec_single, f1_single, total_class = get_prf(true_single, total_single, pred_total_single)
   

			# Compute SHAP values for base RF with feature-aware outputs
			explainer = shap.TreeExplainer(baseline_wrapper.model)
			X_baseline = Xtest_for_shap
			shape_info = getattr(X_baseline, 'shape', None)
			print(f'SHAP debug: X_baseline shape={shape_info}')
			logger.info(f'shap_debug_X_baseline_shape={shape_info}')
			shap_values_raw = explainer.shap_values(X_baseline)
			if isinstance(shap_values_raw, list):
				print(f'SHAP debug: got list of length {len(shap_values_raw)}')
				for idx_tmp, shap_matrix in enumerate(shap_values_raw[:2]):
					try:
						print(f'SHAP debug: shap_matrix[{idx_tmp}] shape={np.asarray(shap_matrix).shape}')
					except Exception:
						pass
				stacked = []
				for class_idx, shap_matrix in enumerate(shap_values_raw):
					matrix = np.asarray(shap_matrix)
					if matrix.ndim != 2:
						matrix = matrix.reshape(matrix.shape[0], -1)
					stacked.append(matrix)
				if not stacked:
					raise ValueError('No SHAP values produced for baseline model')
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
			logger.info(f'feature_name_source={feature_source}')
			csv_path_str = str(csv_path)
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
							y = np.full(values.shape[0], pos)
							plt.scatter(values, y, s=4, alpha=0.5, color='#1f77b4')
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

			print('f1_base:', f1_single)
			log_print = ', '.join('%f' % value for value in f1_single)
			logger.info('f1_base: %s' % log_print)
			logger.info("Pred time: Base: %f s" % (time.time() - start_time))

			# Final accuracy visualization (eval_base=True)
			try:
				from src.vis.visualization_fix import ensure_vis_dir_and_render_maps
				# Use caller flag if provided; otherwise config (prod)
				if VIS_DEBUG_MODE is None:
					from config import VIS_DEBUG_MODE as _VIS_FLAG
				else:
					_VIS_FLAG = bool(VIS_DEBUG_MODE)
				render_summary = ensure_vis_dir_and_render_maps(
					model_dir=self.model_dir,
					test_data=(Xtest, ytest, Xtest_group),
					force_accuracy=force_accuracy,
					model=self,
					VIS_DEBUG_MODE=_VIS_FLAG
				)
				if render_summary.get('final_accuracy_generated'):
					print(f"Final accuracy maps rendered: {render_summary.get('final_accuracy_artifacts', [])}")
			except Exception as e:
				print(f"Warning: Could not render final accuracy maps: {e}")

			if print_to_file:
				sys.stdout.close()
				sys.stdout = self.original_stdout

			return pre, rec, f1, pre_single, rec_single, f1_single

		# Final accuracy visualization (eval_base=False)
		try:
			from src.vis.visualization_fix import ensure_vis_dir_and_render_maps
			if VIS_DEBUG_MODE is None:
				from config import VIS_DEBUG_MODE as _VIS_FLAG
			else:
				_VIS_FLAG = bool(VIS_DEBUG_MODE)
			render_summary = ensure_vis_dir_and_render_maps(
				model_dir=self.model_dir,
				test_data=(Xtest, ytest, Xtest_group),
				force_accuracy=force_accuracy,
				model=self,
				VIS_DEBUG_MODE=_VIS_FLAG
			)
			if render_summary.get('final_accuracy_generated'):
				print(f"Final accuracy maps rendered: {render_summary.get('final_accuracy_artifacts', [])}")
		except Exception as e:
			print(f"Warning: Could not render final accuracy maps: {e}")

		if print_to_file:
			sys.stdout.close()
			sys.stdout = self.original_stdout

		return pre, rec, f1

	def visualize_grid(self, Xtest, ytest, Xtest_group, step_size = STEP_SIZE):
		'''Visualization: temporary for testing purposes.
		Combine into a function later.

		Parameters
    ----------
    Xtest: array-like
        Input features.
    ytest: array-like
        Output targets.
		Xtest_group: array-like
				Same way of assignment as training. See detailed explanations in training.
		step_size: float
				Used to generate grid (must be same as the one used to generate grid-based groups).
    Returns
    -------
		None.
		'''

		#Model assignment
		Xtest_branch_id = get_X_branch_id_by_group(Xtest_group, self.s_branch)
		results, groups, total_number = predict_test_group_wise(self.model, Xtest, ytest, Xtest_group, self.s_branch, X_branch_id = Xtest_branch_id)

		#visualize partitions
		generate_vis_image(self.s_branch, Xtest_branch_id, max_depth = self.max_model_depth, dir = self.dir_vis, step_size = step_size)

		for class_id_input in SELECT_CLASS:
			class_id = int(class_id_input)
			ext = str(class_id)
			grid, vmin, vmax = generate_performance_grid(results, groups, class_id = class_id, step_size = step_size)
			print('X_DIM, grid.shape: ', X_DIM, grid.shape)
			grid_count, vmin_count, vmax_count = generate_count_grid(total_number, groups, class_id = class_id, step_size = step_size)
			generate_vis_image_for_all_groups(grid, dir = self.dir_vis, ext = '_star' + ext, vmin = vmin, vmax = vmax)
			generate_vis_image_for_all_groups(grid_count, dir = self.dir_vis, ext = '_count' + ext, vmin = vmin_count, vmax = vmax_count)

			results_base, groups_base, _ = predict_test_group_wise(self.model, Xtest, ytest, Xtest_group, self.s_branch, base = True, X_branch_id = Xtest_branch_id)
			grid_base, vmin_base, vmax_base = generate_performance_grid(results_base, groups_base, class_id = class_id, step_size = step_size)
			generate_vis_image_for_all_groups(grid_base, dir = self.dir_vis, ext = '_base' + ext, vmin = vmin_base, vmax = vmax_base)

			cnt_vis_thrd = get_filter_thrd(grid_count, ratio = 0.2)
			grid_diff, vmin_diff, vmax_diff = generate_diff_grid((grid - grid_base)*(grid_count>=cnt_vis_thrd), groups, step_size = step_size)
			generate_vis_image_for_all_groups(grid_diff, dir = self.dir_vis, ext = '_diff' + ext, vmin = vmin_diff, vmax = vmax_diff)

			np.save(self.dir_space + '/' + 'grid' + ext + '.npy', grid)
			np.save(self.dir_space + '/' + 'grid_base' + ext + '.npy', grid_base)
			np.save(self.dir_space + '/' + 'grid_count' + ext + '.npy', grid_count)

		return

	# Add this method to your GeoRF class in GeoRF.py    
	def fit_2layer(self, X_L1, X_L2, y, X_group, val_ratio=VAL_RATIO, 
	               contiguity_type=CONTIGUITY_TYPE, polygon_contiguity_info=POLYGON_CONTIGUITY_INFO):
		"""
		Train a 2-layer GeoRF model.
		
		Parameters
		----------
		X_L1 : array-like
			Input features for Layer 1 (main prediction)
		X_L2 : array-like  
			Input features for Layer 2 (error correction)
		y : array-like
			Output targets
		X_group : array-like
			Group assignments for spatial partitioning
		val_ratio : float
			Validation ratio for internal evaluation
		contiguity_type : str
			Type of contiguity refinement: 'grid' or 'polygon' (default: uses CONTIGUITY_TYPE from config)
		polygon_contiguity_info : dict
			Dictionary containing polygon contiguity information (required when contiguity_type='polygon')
			
		Returns
		-------
		self : GeoRF object
		"""
		print("Training 2-Layer GeoRF...")
		
		# Step 1: Train Layer 1 (main prediction model)
		print("Training Layer 1 (main prediction)...")
		self.georf_l1 = GeoRF(
			min_model_depth=self.min_model_depth,
			max_model_depth=self.max_model_depth,
			n_jobs=self.n_jobs,
			max_depth=self.max_depth
		)
		self.georf_l1.fit(X_L1, y, X_group, val_ratio=val_ratio, 
		                  contiguity_type=contiguity_type, polygon_contiguity_info=polygon_contiguity_info)
		
		# Step 2: Get Layer 1 predictions for training error model
		y_pred_l1 = self.georf_l1.predict(X_L1, X_group)
		error_targets = (y != y_pred_l1).astype(int)  # Binary error indicator
		
		# Step 3: Train Layer 2 (error correction model)
		print("Training Layer 2 (error correction)...")
		self.georf_l2 = GeoRF(
			min_model_depth=self.min_model_depth,
			max_model_depth=self.max_model_depth,
			n_jobs=self.n_jobs,
			max_depth=self.max_depth
		)
		self.georf_l2.fit(X_L2, error_targets, X_group, val_ratio=val_ratio, 
		                  contiguity_type=contiguity_type, polygon_contiguity_info=polygon_contiguity_info)
		
		# Store feature indices for prediction
		self.is_2layer = True
		
		return self
		
	def predict_2layer(self, X_L1, X_L2, X_group, correction_strategy='flip'):
		"""
		Make predictions using 2-layer GeoRF model.
		
		Parameters
		----------
		X_L1 : array-like
			Layer 1 features
		X_L2 : array-like
			Layer 2 features  
		X_group : array-like
			Group assignments
		correction_strategy : str
			How to apply error correction ('flip', 'confidence', etc.)
			
		Returns
		-------
		y_pred : array-like
			Combined predictions from both layers
		"""
		if not hasattr(self, 'is_2layer') or not self.is_2layer:
			raise ValueError("Model was not trained as 2-layer. Use fit_2layer() first.")
			
		# Get Layer 1 predictions
		y_pred_l1 = self.georf_l1.predict(X_L1, X_group)
		
		# Get Layer 2 error predictions
		error_pred = self.georf_l2.predict(X_L2, X_group)
		
		# Apply correction strategy
		if correction_strategy == 'flip':
			# Simple flip for binary classification
			y_pred_combined = y_pred_l1.copy()
			y_pred_combined[error_pred == 1] = 1 - y_pred_combined[error_pred == 1]
		else:
			# Add other strategies as needed
			y_pred_combined = y_pred_l1  # Default: no correction
			
		return y_pred_combined
	
	def evaluate_2layer(self, X_L1_test, X_L2_test, y_test, X_group_test, 
						X_L1_train=None, X_L2_train=None, y_train=None, X_group_train=None,
						correction_strategy='flip', print_to_file=True,
						contiguity_type=CONTIGUITY_TYPE, polygon_contiguity_info=POLYGON_CONTIGUITY_INFO):
		"""
		Evaluate 2-layer GeoRF model against 2-layer base RF model.
		
		Parameters
		----------
		X_L1_test, X_L2_test : array-like
			Test features for both layers
		y_test : array-like
			Test targets
		X_group_test : array-like
			Test group assignments
		X_L1_train, X_L2_train : array-like, optional
			Training features (needed for base model training)
		y_train : array-like, optional
			Training targets (needed for base model training)
		X_group_train : array-like, optional
			Training group assignments (needed for base model training)
		correction_strategy : str
			Error correction strategy
		print_to_file : bool
			Whether to print results to file
		contiguity_type : str
			Type of contiguity refinement: 'grid' or 'polygon' (default: uses CONTIGUITY_TYPE from config)
		polygon_contiguity_info : dict
			Dictionary containing polygon contiguity information (for consistency with other methods)
			
		Returns
		-------
		pre, rec, f1 : array-like
			Precision, recall, F1 for 2-layer GeoRF
		pre_base, rec_base, f1_base : array-like  
			Precision, recall, F1 for 2-layer base RF
		"""
		
		if print_to_file:
			print('Evaluating 2-Layer Models...')
			print_file = self.model_dir + '/' + 'log_print_eval_2layer.txt'
			original_stdout = sys.stdout
			sys.stdout = open(print_file, "w")
		
		# ============================================================
		# 2-Layer GeoRF Evaluation
		# ============================================================
		print("Evaluating 2-Layer GeoRF...")
		y_pred_georf = self.predict_2layer(X_L1_test, X_L2_test, X_group_test, correction_strategy)
		
		# Calculate GeoRF metrics
		true_georf, total_georf, pred_total_georf = get_class_wise_accuracy(y_test, y_pred_georf, prf=True)
		pre, rec, f1, total_class = get_prf(true_georf, total_georf, pred_total_georf)
		
		print('2-Layer GeoRF F1:', f1)
		
		# ============================================================
		# 2-Layer Base RF Evaluation  
		# ============================================================
		print("Training and evaluating 2-Layer Base RF...")
		
		if any(x is None for x in [X_L1_train, X_L2_train, y_train]):
			raise ValueError("Training data required for base model evaluation")
		
		# Train base Layer 1 (global RF, no partitioning)

		base_l1 = RandomForestClassifier(
			n_estimators=self.n_trees_unit,
			max_depth=self.max_depth,
			random_state=self.random_state,
			n_jobs=self.n_jobs
		)
		base_l1.fit(X_L1_train, y_train)
		
		# Get base Layer 1 predictions
		y_pred_base_l1_train = base_l1.predict(X_L1_train)
		y_pred_base_l1_test = base_l1.predict(X_L1_test)
		
		# Train base Layer 2 (global RF for error correction)
		error_targets_base = (y_train != y_pred_base_l1_train).astype(int)
		base_l2 = RandomForestClassifier(
			n_estimators=self.n_trees_unit,
			max_depth=self.max_depth,
			random_state=self.random_state,
			n_jobs=self.n_jobs
		)
		base_l2.fit(X_L2_train, error_targets_base)
		
		# Get base Layer 2 predictions and combine
		error_pred_base = base_l2.predict(X_L2_test)
		
		# Apply same correction strategy
		if correction_strategy == 'flip':
			y_pred_base = y_pred_base_l1_test.copy()
			y_pred_base[error_pred_base == 1] = 1 - y_pred_base[error_pred_base == 1]
		else:
			y_pred_base = y_pred_base_l1_test
		
		# Calculate base model metrics
		true_base, total_base, pred_total_base = get_class_wise_accuracy(y_test, y_pred_base, prf=True)
		pre_base, rec_base, f1_base, total_class = get_prf(true_base, total_base, pred_total_base)
		
		print('2-Layer Base RF F1:', f1_base)
		
		if print_to_file:
			sys.stdout.close()
			sys.stdout = original_stdout
			
		return pre, rec, f1, pre_base, rec_base, f1_base
