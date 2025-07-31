# @Author: xie
# @Date:   2025-06-20
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2025-06-20
# @License: MIT License

import numpy as np
from scipy import stats

#GeoRF
#Can be customized with the template

# from models import DNNmodel, LSTMmodel, UNetmodel#model is easily customizable
from model_RF import RFmodel, save_single, predict_test_group_wise#model is easily customizable
from sklearn.ensemble import RandomForestClassifier
# from customize import generate_groups_nonimg_input#can customize group definition
from customize import *
from data import *
from initialization import init_X_info, init_X_info_raw_loc, init_X_branch_id, train_val_split
from helper import create_dir, open_dir, get_X_branch_id_by_group, get_filter_thrd
from transformation import partition
from visualization import *
from metrics import get_class_wise_accuracy, get_prf
from partition_opt import get_refined_partitions_all
#All global parameters
from config import *

import pandas as pd
import os
import argparse
import sys
import time
import logging

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

		#toggle between prints
		self.original_stdout = sys.stdout
		

	#Train GeoRF
	def fit(self, X, y, X_group, X_set = None, val_ratio = VAL_RATIO, print_to_file = True, 
	        contiguity_type = CONTIGUITY_TYPE, polygon_contiguity_info = POLYGON_CONTIGUITY_INFO,
	        track_partition_metrics = False, correspondence_table_path = None):#X_loc is unused
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


		# #for debugging#X_loc removed here
		# print(np.min(X_loc[:,0]), np.max(X_loc[:,0]))
		# print(np.min(X_loc[:,1]), np.max(X_loc[:,1]))

		#Initialize location-related and training information. Can be customized.
    #X_id stores data points' ids in the original X, and is used as a reference.
    #X_set stores train-val-test assignments: train=0, val=1, test=2
    #X_branch_id stores branch_ids (or, partion ids) of each data points. All init to route branch ''. Dynamically updated during training.
    #X_group stores group assignment: customizable. In this example, groups are defined by grid cells in space.
		if X_set is None:
			X_set = train_val_split(X, val_ratio=val_ratio)
		X_id = np.arange(X.shape[0])#the id is used to later refer back to the original X, and the related information
		X_branch_id = init_X_branch_id(X, max_depth = self.max_model_depth)
		# X_group, X_set, X_id, X_branch_id = init_X_info_raw_loc(X, y, X_loc, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, step_size = STEP_SIZE, predefined = PREDEFINED_GROUPS)

		# '''RF paras''' --> unused
		# max_new_forests = [1,1,1,1,1,1]
		# sample_weights_by_class = None#np.array([0.05, 0.95])#None#np.array([0.05, 0.95])#None

		#timer
		start_time = time.time()

		#Train to stablize before starting the first data partitioning
		train_list_init = np.where(X_set == 0)
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
		partition_result = partition(self.model, X, y,
		                   X_group , X_set, X_id, X_branch_id,
		                   min_depth = self.min_model_depth, max_depth = self.max_model_depth,
		                   contiguity_type = contiguity_type, polygon_contiguity_info = polygon_contiguity_info,
		                   track_partition_metrics = track_partition_metrics, 
		                   correspondence_table_path = correspondence_table_path,
		                   model_dir = self.model_dir)#X_loc = X_loc is unused
		
		# Handle different return formats (with/without metrics tracker)
		if track_partition_metrics:
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
			X_branch_id = get_refined_partitions_all(X_branch_id, self.s_branch, X_group, dir = self.dir_vis, min_component_size = MIN_COMPONENT_SIZE)
		## 	GLOBAL_CONTIGUITY = True#unused

		if print_to_file:
			sys.stdout.close()
			sys.stdout = self.original_stdout

		return self

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
		X_branch_id = get_X_branch_id_by_group(X_group, self.s_branch)
		y_pred = self.model.predict_georf(X, X_group, self.s_branch, X_branch_id = X_branch_id)

		if save_full_predictions:
			np.save(self.dir_space + '/' + 'y_pred_georf.npy', y_pred)

		return y_pred

	def evaluate(self, Xtest, ytest, Xtest_group, eval_base = False, print_to_file = True):
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

		#Logging: testing purpose only.
		logging.basicConfig(filename=self.model_dir + '/' + "model_eval.log",
						format='%(asctime)s %(message)s',
						filemode='w')
		logger=logging.getLogger()
		logger.setLevel(logging.INFO)

		#print to file
		if print_to_file:
			print('model_dir:', self.model_dir)
			print('Printing to file.')
			print_file = self.model_dir + '/' + 'log_print_eval.txt'
			sys.stdout = open(print_file, "w")

		#Geo-RF
		start_time = time.time()

		#Model assignment
		Xtest_branch_id = get_X_branch_id_by_group(Xtest_group, self.s_branch)

		pre, rec, f1, total_class = self.model.predict_test(Xtest, ytest, Xtest_group, self.s_branch, X_branch_id = Xtest_branch_id)
		print('f1:', f1)
		log_print = ', '.join('%f' % value for value in f1)
		logger.info('f1: %s' % log_print)
		logger.info("Pred time: GeoRF: %f s" % (time.time() - start_time))

		#Base RF
		if eval_base:
			start_time = time.time()
			self.model.load('')
			y_pred_single = self.model.predict(Xtest)
			true_single, total_single, pred_total_single = get_class_wise_accuracy(ytest, y_pred_single, prf = True)
			pre_single, rec_single, f1_single, total_class = get_prf(true_single, total_single, pred_total_single)

			print('f1_base:', f1_single)
			log_print = ', '.join('%f' % value for value in f1_single)
			logger.info('f1_base: %s' % log_print)
			logger.info("Pred time: Base: %f s" % (time.time() - start_time))

			if print_to_file:
				sys.stdout.close()
				sys.stdout = self.original_stdout

			return pre, rec, f1, pre_single, rec_single, f1_single

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