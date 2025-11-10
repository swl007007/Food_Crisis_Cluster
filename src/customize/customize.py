# @Author: xie
# @Date:   2021-06-02
# @Email:  xie@umd.edu
# @Last modified by:   xie
# @Last modified time: 2022-05-11
# @License: MIT License

import numpy as np
from config import STEP_SIZE#this is the grid cell size (not step size for image patch generation for semantic segmentation)
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class OutOfRangeImputer:
    """
    Custom imputer that replaces missing values with values outside the data's min-max range.
    This allows decision trees to naturally separate missing data into different branches.
    
    Parameters:
    -----------
    strategy : str, default='max_plus'
        Strategy for imputation:
        - 'max_plus': Replace with max_value * multiplier
        - 'min_minus': Replace with min_value - (max_value - min_value) * multiplier
        - 'extreme_high': Replace with max_value + (max_value - min_value) * multiplier
        - 'extreme_low': Replace with min_value - (max_value - min_value) * multiplier
    
    multiplier : float, default=100.0
        Multiplier to ensure the imputed value is well outside the original range
    
    fallback_strategy : str, default='mean'
        Fallback strategy if all values in a column are missing
    """
    
    def __init__(self, strategy='max_plus', multiplier=100.0, fallback_strategy='mean'):
        self.strategy = strategy
        self.multiplier = multiplier
        self.fallback_strategy = fallback_strategy
        self.impute_values_ = {}
        self.column_stats_ = {}
        
    def _is_missing(self, values):
        """Check for missing values, handling different data types."""
        if values.dtype.kind in ['i', 'f']:  # Integer or float
            return np.isnan(values.astype(float))
        else:  # String/object types
            return pd.isna(values)
    
    def _to_numeric_safe(self, column_data):
        """Convert column to numeric, handling mixed types."""
        try:
            # Try direct conversion
            return pd.to_numeric(column_data, errors='coerce')
        except:
            # If that fails, convert to string first then numeric
            return pd.to_numeric(column_data.astype(str), errors='coerce')
    
    def fit(self, X):
        """
        Fit the imputer on X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
        """
        # Convert to DataFrame for easier handling of mixed types
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)
        
        for col in range(X_df.shape[1]):
            column_data = X_df.iloc[:, col]
            
            # Convert to numeric, coercing errors to NaN
            numeric_data = self._to_numeric_safe(column_data)
            
            # Check if column has any non-missing values
            missing_mask = self._is_missing(numeric_data)
            non_missing_mask = ~missing_mask
            
            if np.any(non_missing_mask):
                non_missing_values = numeric_data[non_missing_mask]
                min_val = np.min(non_missing_values)
                max_val = np.max(non_missing_values)
                range_val = max_val - min_val
                
                self.column_stats_[col] = {
                    'min': min_val,
                    'max': max_val,
                    'range': range_val,
                    'mean': np.mean(non_missing_values),
                    'has_missing': np.any(missing_mask),
                    'dtype': str(numeric_data.dtype)
                }
                
                # Calculate imputation value based on strategy
                if self.strategy == 'max_plus':
                    if max_val == 0:
                        impute_val = self.multiplier
                    else:
                        impute_val = max_val * self.multiplier
                        
                elif self.strategy == 'min_minus':
                    if range_val == 0:
                        impute_val = min_val - abs(min_val) * self.multiplier if min_val != 0 else -self.multiplier
                    else:
                        impute_val = min_val - range_val * self.multiplier
                        
                elif self.strategy == 'extreme_high':
                    if range_val == 0:
                        impute_val = max_val + abs(max_val) * self.multiplier if max_val != 0 else self.multiplier
                    else:
                        impute_val = max_val + range_val * self.multiplier
                        
                elif self.strategy == 'extreme_low':
                    if range_val == 0:
                        impute_val = min_val - abs(min_val) * self.multiplier if min_val != 0 else -self.multiplier
                    else:
                        impute_val = min_val - range_val * self.multiplier
                        
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                
                self.impute_values_[col] = impute_val
                
            else:
                # All values are missing - use fallback strategy
                print(f"Warning: Column {col} has all missing values. Using fallback strategy: {self.fallback_strategy}")
                if self.fallback_strategy == 'mean':
                    self.impute_values_[col] = 0.0  # Arbitrary value when all missing
                elif self.fallback_strategy == 'zero':
                    self.impute_values_[col] = 0.0
                else:
                    self.impute_values_[col] = float(self.fallback_strategy)
                
                self.column_stats_[col] = {
                    'min': np.nan,
                    'max': np.nan,
                    'range': np.nan,
                    'mean': np.nan,
                    'has_missing': True,
                    'dtype': str(numeric_data.dtype)
                }
        
        return self
    
    def transform(self, X):
        """
        Impute missing values in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_imputed : ndarray of shape (n_samples, n_features)
            Imputed data
        """
        if not hasattr(self, 'impute_values_'):
            raise ValueError("This OutOfRangeImputer instance is not fitted yet.")
        
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)
        
        X_imputed = X_df.copy()
        
        for col in range(X_imputed.shape[1]):
            if col in self.impute_values_:
                # Convert to numeric
                numeric_col = self._to_numeric_safe(X_imputed.iloc[:, col])
                missing_mask = self._is_missing(numeric_col)
                
                if np.any(missing_mask):
                    numeric_col.loc[missing_mask] = self.impute_values_[col]
                    X_imputed.iloc[:, col] = numeric_col
                    print(f"Column {col}: Imputed {np.sum(missing_mask)} missing values with {self.impute_values_[col]:.3f}")
        
        # Convert back to numpy array with proper dtype
        return X_imputed.astype(float).values
    
    def fit_transform(self, X):
        """
        Fit the imputer and transform X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        X_imputed : ndarray of shape (n_samples, n_features)
            Imputed data
        """
        return self.fit(X).transform(X)
    
    def get_imputation_summary(self):
        """
        Get a summary of the imputation process.
        
        Returns:
        --------
        summary : dict
            Dictionary containing imputation statistics for each column
        """
        summary = {}
        for col, stats in self.column_stats_.items():
            summary[col] = {
                'original_range': f"[{stats['min']:.3f}, {stats['max']:.3f}]" if not np.isnan(stats['min']) else "All missing",
                'imputed_value': self.impute_values_[col],
                'has_missing': stats['has_missing'],
                'strategy_used': self.strategy if stats['has_missing'] else 'No imputation needed'
            }
        return summary


# Example usage with your data
def impute_missing_values(X, strategy='max_plus', multiplier=100.0, verbose=True):
    """
    Convenience function to impute missing values using the OutOfRangeImputer.
    
    Parameters:
    -----------
    X : array-like
        Input data with potential missing values
    strategy : str
        Imputation strategy ('max_plus', 'min_minus', 'extreme_high', 'extreme_low')
    multiplier : float
        Multiplier for out-of-range values
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    X_imputed : ndarray
        Data with missing values imputed
    imputer : OutOfRangeImputer
        Fitted imputer object
    """
    
    # Create and fit the imputer
    imputer = OutOfRangeImputer(strategy=strategy, multiplier=multiplier)
    
    if verbose:
        print(f"Starting imputation with strategy: {strategy}, multiplier: {multiplier}")
        print(f"Original data shape: {X.shape}")
        print(f"Original data type: {type(X)}")
        if hasattr(X, 'dtype'):
            print(f"Original data dtype: {X.dtype}")
        
        # Convert to DataFrame for analysis
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)
        
        # Analyze data types
        print(f"\nData type analysis:")
        dtype_counts = X_df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Count missing values per column (handling mixed types)
        missing_counts = []
        total_missing = 0
        
        for col in range(X_df.shape[1]):
            try:
                # Try numeric conversion
                numeric_col = pd.to_numeric(X_df.iloc[:, col], errors='coerce')
                missing_count = pd.isna(numeric_col).sum()
            except:
                # Fallback for other types
                missing_count = pd.isna(X_df.iloc[:, col]).sum()
            
            missing_counts.append(missing_count)
            total_missing += missing_count
        
        print(f"\nTotal missing values: {total_missing}")
        
        if total_missing > 0:
            print("Missing values per column (showing first 10 columns with missing data):")
            missing_cols = [(col, count) for col, count in enumerate(missing_counts) if count > 0]
            for col, count in missing_cols[:10]:  # Show first 10
                percentage = (count / X.shape[0]) * 100
                print(f"  Column {col}: {count} ({percentage:.1f}%)")
            if len(missing_cols) > 10:
                print(f"  ... and {len(missing_cols) - 10} more columns with missing data")
    
    # Fit and transform the data
    X_imputed = imputer.fit_transform(X)
    
    if verbose:
        print("\nImputation Summary:")
        summary = imputer.get_imputation_summary()
        imputed_cols = [col for col, info in summary.items() if info['has_missing']]
        
        if len(imputed_cols) <= 10:
            for col in imputed_cols:
                info = summary[col]
                print(f"  Column {col}: {info['original_range']} → imputed with {info['imputed_value']:.3f}")
        else:
            print(f"  Imputed {len(imputed_cols)} columns total (showing first 5):")
            for col in imputed_cols[:5]:
                info = summary[col]
                print(f"    Column {col}: {info['original_range']} → imputed with {info['imputed_value']:.3f}")
            print(f"    ... and {len(imputed_cols) - 5} more columns")
        
        # Verify no missing values remain
        remaining_missing = pd.isna(X_imputed).sum()
        print(f"\nRemaining missing values after imputation: {remaining_missing}")
        print(f"Output data shape: {X_imputed.shape}")
        print(f"Output data type: {X_imputed.dtype}")
    
    return X_imputed, imputer



def train_test_split_rolling_window(X, y, X_loc, X_group, years, dates, test_year=2024, input_terms=None, need_terms=None, admin_codes=None):
    '''Rolling window temporal splitting for 2024 quarterly evaluation.

    This function implements a rolling window approach where:
    - For each 2024 quarter, uses 5 years of data before that quarter's end as training
    - Tests on the specific quarter of 2024
    - Enables more realistic temporal validation than fixed year-based splitting

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Labels.
    X_loc : array-like
        Location information for each sample.
    X_group : array-like
        Group ID for each sample.
    years : array-like
        1-D array containing year values.
    dates : array-like
        1-D array containing datetime values for precise temporal filtering.
    test_year : int, default=2024
        Year used for the test set. Currently fixed to 2024.
    input_terms : array-like, optional
        Terms within each year (1-4 corresponding to quarters).
    need_terms : int, optional
        Specific quarter to use as test set (1=Q1, 2=Q2, 3=Q3, 4=Q4).
        If None, uses traditional year-based splitting.
    admin_codes : array-like, optional
        Admin codes for each sample. If provided, will be split alongside other arrays.

    Returns
    -------
    Tuple containing train and test splits. If admin_codes is provided:
        Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group, admin_codes_train, admin_codes_test
    Otherwise:
        Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group
    '''
    # Use the provided test_year parameter
    
    if need_terms is None:
        # Training set: all years before test_year (fallback when need_terms is None)
        train_mask = (years < test_year) & (years >= (test_year - 3))  # Limit to last 3 years for training
        # Test set: only test_year
        test_mask = years == test_year

        Xtrain = X[train_mask]
        ytrain = y[train_mask]
        Xtrain_loc = X_loc[train_mask]
        Xtrain_group = X_group[train_mask]

        Xtest = X[test_mask]
        ytest = y[test_mask]
        Xtest_loc = X_loc[test_mask]
        Xtest_group = X_group[test_mask]

        # Split admin_codes if provided
        if admin_codes is not None:
            admin_codes_train = admin_codes[train_mask]
            admin_codes_test = admin_codes[test_mask]

        #Xtest_group unique values
        X_test_unique = np.unique(Xtest_group)

        # generate mask for Xtrain_group to only include groups present in Xtest_group
        train_group_mask = np.isin(Xtrain_group, X_test_unique)

        # apply mask to Xtrain, ytrain, Xtrain_loc, Xtrain_group
        Xtrain = Xtrain[train_group_mask]
        ytrain = ytrain[train_group_mask]
        Xtrain_loc = Xtrain_loc[train_group_mask]
        Xtrain_group = Xtrain_group[train_group_mask]

        # Apply group mask to admin_codes if provided
        if admin_codes is not None:
            admin_codes_train = admin_codes_train[train_group_mask]
        
    else:
        # Rolling window approach: 3 years before test term → test on specific 2024 term
        import pandas as pd
        
        # Convert dates to pandas datetime if not already
        if not isinstance(dates, pd.Series):
            dates = pd.to_datetime(dates)
        
        # Define quarter start and end dates for the test year
        quarter_starts = {
            1: pd.Timestamp(f'{test_year}-01-01'),
            2: pd.Timestamp(f'{test_year}-04-01'),
            3: pd.Timestamp(f'{test_year}-07-01'),
            4: pd.Timestamp(f'{test_year}-10-01')
        }
        
        quarter_ends = {
            1: pd.Timestamp(f'{test_year}-03-31'),
            2: pd.Timestamp(f'{test_year}-06-30'),
            3: pd.Timestamp(f'{test_year}-09-30'),
            4: pd.Timestamp(f'{test_year}-12-31')
        }
        
        # Get the start date for the test quarter (this is when test quarter begins)
        test_quarter_start = quarter_starts[need_terms]
        test_quarter_end = quarter_ends[need_terms]
        
        # Training set: 3 years of data ENDING BEFORE the test quarter starts
        # This ensures NO OVERLAP between training and test
        train_end_date = test_quarter_start  # Training ends when test quarter begins
        train_start_date = train_end_date - pd.DateOffset(years=3)

        # Training mask: includes data from 3 years ago UP TO (but not including) test quarter start
        train_mask = (dates >= train_start_date) & (dates < train_end_date)
        
        # Test set: only the specific quarter of 2024
        test_year_mask = years == test_year
        test_terms_mask = input_terms == need_terms
        test_mask = test_year_mask & test_terms_mask

        Xtrain = X[train_mask]
        ytrain = y[train_mask]
        Xtrain_loc = X_loc[train_mask]
        Xtrain_group = X_group[train_mask]

        Xtest = X[test_mask]
        ytest = y[test_mask]
        Xtest_loc = X_loc[test_mask]
        Xtest_group = X_group[test_mask]

        # Split admin_codes if provided
        if admin_codes is not None:
            admin_codes_train = admin_codes[train_mask]
            admin_codes_test = admin_codes[test_mask]

        # Ensure training groups overlap with test groups
        X_test_unique = np.unique(Xtest_group)
        train_group_mask = np.isin(Xtrain_group, X_test_unique)
        Xtrain = Xtrain[train_group_mask]
        ytrain = ytrain[train_group_mask]
        Xtrain_loc = Xtrain_loc[train_group_mask]
        Xtrain_group = Xtrain_group[train_group_mask]

        # Apply group mask to admin_codes if provided
        if admin_codes is not None:
            admin_codes_train = admin_codes_train[train_group_mask]

        print(f"Rolling Window Split for Q{need_terms} {test_year}:")
        print(f"  Training: {len(ytrain)} samples from {train_start_date.date()} to {train_end_date.date()} (5 years BEFORE test quarter)")
        print(f"  Test: {len(ytest)} samples from Q{need_terms} {test_year} ({test_quarter_start.date()} to {test_quarter_end.date()})")
        print(f"  No overlap: Training ends {train_end_date.date()}, Test starts {test_quarter_start.date()}")

        if admin_codes is not None:
            return Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group, admin_codes_train, admin_codes_test
        else:
            return Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group
        
    print(f"Train/Test split: {len(ytrain)} training samples (years < {test_year}), {len(ytest)} test samples (year = {test_year})")

    if admin_codes is not None:
        return Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group, admin_codes_train, admin_codes_test
    else:
        return Xtrain, ytrain, Xtrain_loc, Xtrain_group, Xtest, ytest, Xtest_loc, Xtest_group
class GroupGenerator():
  '''
  Generate groups (minimum spatial units) for partitioning in GeoRF.
  This generator is an example for grid-based group definitions,
  where a grid is overlaid on the study area and each grid cell defines one group.
  In general, any group definition can be used.

  The groups are groupings of locations, which serve two important purposes:
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
  '''
  def __init__(self, xmin, xmax, ymin, ymax, step_size):
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.step_size = step_size

  def get_groups(self, X_loc):
    '''
    Generate groups using locations of data points, and assign a group ID to each data point.

    Parameters
    ---------
    X_loc : array-like
        Same number of points as input features X. Stores the geographic coordinates (e.g., lat, lon) of each data point.
    Return
    -------
    X_group : array-like
        Provides a group ID assignment to each of the data point.
    '''

    X_loc[:,0] = X_loc[:,0] - self.xmin
    X_loc[:,1] = X_loc[:,1] - self.ymin

    #for debugging:
    # print('Old xmin, xmax:', self.xmin, self.xmax)
    # print('New xmin, xmax:', np.min(X_loc[:,0]), np.max(X_loc[:,0]))
    #
    # print('Old ymin, ymax:', self.ymin, self.ymax)
    # print('New ymin, ymax:', np.min(X_loc[:,1]), np.max(X_loc[:,1]))

    X_loc_grid = np.floor(X_loc/self.step_size)
    n_rows = np.max(X_loc_grid[:,0])+1
    n_cols = np.max(X_loc_grid[:,1])+1
    # print(n_rows, n_cols)
    X_group = X_loc_grid[:,0]*n_cols + X_loc_grid[:,1]

    return X_group



def generate_groups(X_loc):
  '''Create groups of data points.
  This is needed for data partitioning optimization, as it needs to calculalte statistics at the group level.
  In this example, we use each grid cell as a group.
  Users can customize their own groups for spatial or non-spatail data by modifying this function.

  Notes for customization:
  1. Each group should contain sufficient number of data points for statistics calculation (e.g., >50).
  2. Data points in each group should follow roughly the same distribution (e.g., nearby data points in space).

  Args in this example:
    X_loc: Locations of data points. First two values are pixel locations.
        The other two are grid cell locations (all pixels in a grid cell shares the same grid cell location).
  '''

  n_rows = np.max(X_loc[:,2])+1
  n_cols = np.max(X_loc[:,3])+1
  X_group = X_loc[:,2]*n_cols + X_loc[:,3]

  return X_group

def generate_groups_nonimg_input(X_loc, step_size):
  '''from RF test: might be the same as generate_groups_from_raw_loc()'''
  X_loc_grid = np.floor(X_loc/step_size)
  n_rows = np.max(X_loc_grid[:,0])+1
  n_cols = np.max(X_loc_grid[:,1])+1
  # print(n_rows, n_cols)
  X_group = X_loc_grid[:,0]*n_cols + X_loc_grid[:,1]
  return X_group

def get_locs_of_groups(X_group, X_loc):
  n_group = np.max(X_group).astype(int) + 1
  group_loc = np.zeros((n_group.astype(int), 2))
  for i in range(X_group.shape[0]):
    group_id = X_group[i]
    #for debugging
    if group_loc[group_id,0] > 0 and group_loc[group_id, 0] != X_loc[i, 0]:
      print('#Bug: same group with diff locs: ', i, group_id, group_loc[group_id, :], X_loc[i, :])
    group_loc[group_id, :] = X_loc[i, :]
    group_loc = group_loc.astype(int)

  return group_loc

def generate_groups_from_raw_loc(X_loc, step_size):# = STEP_SIZE
  '''Create groups of data points. This version only uses raw locations of each data point.
  '''
  # n_rows = np.max(X_loc[:,2])+1
  n_cols = np.floor(np.max(X_loc[:,1])/step_size)+1
  X_group = np.floor(X_loc[:,0]/step_size) * n_cols  + np.floor(X_loc[:,1]/step_size)#may need to convert to int here

  return X_group

def generate_groups_loc(X_DIM, step_size):#X_DIM, X_loc,  = STEP_SIZE
  '''Used to store row and columns ids of groups for spatial contiguity refinement if needed.
     Corresponds to groups from generate_groups_from_raw_loc
  '''
  n_cols = int(np.floor(X_DIM[1]/step_size)+1)
  n_rows = int(np.floor(X_DIM[0]/step_size)+1)
  #X_loc_grid = np.floor(X_loc/step_size)
  #n_rows = int(np.max(X_loc_grid[:,0])+1)
  #n_cols = int(np.max(X_loc_grid[:,1])+1)
  group_loc = -np.ones([n_cols * n_rows, 2])
  group_loc[:, 0] = np.floor(np.arange(n_cols * n_rows) / n_cols)
  group_loc[:, 1] = np.arange(n_cols * n_rows) % n_cols

  return group_loc.astype(int)


''' customized groups assignment using county assignment
'''
def generate_groups_counties(X_loc):
  import os
  import requests
  import geopandas as gpd

  county_file = 'CountyShp.zip'

  if not os.path.exists(county_file):
    census_url = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip'
    r = requests.get(census_url) # create HTTP response object
    with open('CountyShp.zip','wb') as f:
      f.write(r.content)

  county_gdf = gpd.read_file(county_file)
  county_gdf['id'] = county_gdf.index
  county_gdf = county_gdf[['geometry']]

  '''X_loc for RF have this offset needed to add to the X_loc to retrieve the actual lat and lon'''
  offset = np.array([24.54815386, -124.72499])
  X_loc_original = X_loc + offset

  X_loc_geom = gpd.points_from_xy(X_loc_original[:,1], X_loc_original[:,0])
  X_loc_original_gdf = gpd.GeoDataFrame(X_loc_original, geometry=X_loc_geom)
  X_loc_original_gdf.crs = 'EPSG:4269'

  X_loc_join = X_loc_original_gdf.sjoin(county_gdf, how='left')
  # X_loc_join[X_loc_join['index_right'].isna()].plot()
  '''points on the US boundaries are not assigned to any couty in the previous command. We use spatial join nearest to refine the join. '''
  X_loc_join[X_loc_join['index_right'].isna()] = X_loc_join[X_loc_join['index_right'].isna()].drop(columns='index_right').sjoin_nearest(county_gdf, how='left')
  X_loc_join['group'] = X_loc_join['index_right'].astype(int)
  X_loc_join = X_loc_join.drop(columns=['index_right'])

  X_group = X_loc_join['group'].values
  print('Assigned to County Map')
  # np.save('X_group_debug.npy', X_group)
  return X_group


def generate_kmeans_groups_from_admin_codes(df, features_for_clustering=None, n_clusters=50, random_state=42):
    """
    Generate K-means clustering groups based on FEWSNET_admin_code and associated features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing FEWSNET_admin_code and other features
    features_for_clustering : list, optional
        List of feature columns to use for clustering. If None, will use latitude and longitude.
        Default features could include: ['latitude', 'longitude', 'mean_feature1', 'mean_feature2', ...]
    n_clusters : int, default=50
        Number of clusters for K-means
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    admin_to_group_map : dict
        Dictionary mapping FEWSNET_admin_code to cluster group_id
    cluster_info : dict
        Additional information about clusters (centroids, feature stats, etc.)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    
    # Get unique admin codes and their characteristics
    admin_features = df.groupby('FEWSNET_admin_code').agg({
        'lat': 'mean',
        'lon': 'mean'
    }).reset_index()
    
    # Add other aggregated features if available
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['FEWSNET_admin_code', 'lat', 'lon']]
    
    if len(numeric_cols) > 0:
        additional_features = df.groupby('FEWSNET_admin_code')[numeric_cols].agg(['mean', 'std']).reset_index()
        additional_features.columns = ['FEWSNET_admin_code'] + [f"{col}_{stat}" for col, stat in additional_features.columns[1:]]
        admin_features = admin_features.merge(additional_features, on='FEWSNET_admin_code')
    
    # Select features for clustering
    if features_for_clustering is None:
        features_for_clustering = ['lat', 'lon']
    
    # Ensure all requested features are available
    available_features = [f for f in features_for_clustering if f in admin_features.columns]
    if not available_features:
        raise ValueError(f"None of the requested features {features_for_clustering} are available in the data")
    
    print(f"Using features for clustering: {available_features}")
    
    # Prepare clustering data
    clustering_data = admin_features[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_data_scaled)
    
    # Create mapping from admin code to group id
    admin_to_group_map = dict(zip(admin_features['FEWSNET_admin_code'], cluster_labels))
    
    # Calculate cluster statistics
    admin_features['cluster_id'] = cluster_labels
    cluster_stats = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = admin_features['cluster_id'] == cluster_id
        cluster_data = admin_features[cluster_mask]
        
        if len(cluster_data) > 0:
            cluster_stats[cluster_id] = {
                'n_admin_codes': len(cluster_data),
                'admin_codes': cluster_data['FEWSNET_admin_code'].tolist(),
                'centroid_lat': cluster_data['lat'].mean(),
                'centroid_lon': cluster_data['lon'].mean(),
                'lat_range': [cluster_data['lat'].min(), cluster_data['lat'].max()],
                'lon_range': [cluster_data['lon'].min(), cluster_data['lon'].max()]
            }
    
    cluster_info = {
        'n_clusters': n_clusters,
        'features_used': available_features,
        'cluster_stats': cluster_stats,
        'kmeans_model': kmeans,
        'scaler': scaler,
        'admin_features': admin_features
    }
    
    print(f"Created {n_clusters} clusters from {len(admin_features)} unique admin codes")
    print(f"Cluster sizes range from {min(len(stats['admin_codes']) for stats in cluster_stats.values())} to {max(len(stats['admin_codes']) for stats in cluster_stats.values())} admin codes")
    
    return admin_to_group_map, cluster_info


def assign_groups_from_admin_codes(df, admin_to_group_map):
    """
    Assign group IDs to dataframe rows based on FEWSNET_admin_code using the mapping from K-means clustering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing FEWSNET_admin_code column
    admin_to_group_map : dict
        Dictionary mapping FEWSNET_admin_code to group_id (from generate_kmeans_groups_from_admin_codes)
        
    Returns:
    --------
    X_group : numpy.ndarray
        Array of group IDs corresponding to each row in df
    """
    import numpy as np
    
    # Map admin codes to group IDs
    X_group = np.array([admin_to_group_map.get(admin_code, -1) for admin_code in df['FEWSNET_admin_code']])
    
    # Check for unmapped admin codes
    unmapped_count = np.sum(X_group == -1)
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} rows have admin codes not found in the clustering mapping")
        unique_unmapped = df[X_group == -1]['FEWSNET_admin_code'].unique()
        print(f"Unmapped admin codes: {unique_unmapped[:10]}{'...' if len(unique_unmapped) > 10 else ''}")
    
    return X_group


def create_kmeans_groupgenerator_from_admin_codes(df, features_for_clustering=None, n_clusters=50, random_state=42):
    """
    Complete workflow to create K-means based groups from FEWSNET_admin_code.
    This function combines clustering and group assignment.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing FEWSNET_admin_code and other features
    features_for_clustering : list, optional
        List of feature columns to use for clustering
    n_clusters : int, default=50
        Number of clusters for K-means
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    X_group : numpy.ndarray
        Array of group IDs for each row in df
    cluster_info : dict
        Information about the clustering results
    admin_to_group_map : dict
        Mapping from admin codes to group IDs
    """
    
    # Generate K-means clustering
    admin_to_group_map, cluster_info = generate_kmeans_groups_from_admin_codes(
        df, features_for_clustering, n_clusters, random_state
    )
    
    # Assign groups to all rows
    X_group = assign_groups_from_admin_codes(df, admin_to_group_map)
    
    return X_group, cluster_info, admin_to_group_map


class PolygonGroupGenerator():
  '''
  Generate groups for polygon-based spatial partitioning with contiguity support.
  This generator creates groups from polygon assignments and stores centroid information
  for spatial contiguity refinement.
  '''
  def __init__(self, polygon_centroids, polygon_group_mapping=None, neighbor_distance_threshold=None, adjacency_dict=None):
    '''
    Initialize PolygonGroupGenerator.
    
    Parameters:
    -----------
    polygon_centroids : array-like, shape (n_polygons, 2)
        Centroid coordinates (lat, lon) for each polygon
    polygon_group_mapping : dict, optional
        Mapping from polygon indices to group IDs. If None, each polygon becomes one group.
    neighbor_distance_threshold : float, optional
        Distance threshold for determining polygon neighbors in contiguity refinement
    adjacency_dict : dict, optional
        Pre-computed adjacency dictionary from true polygon boundaries.
        If provided, takes precedence over distance-based neighbor calculation.
    '''
    self.polygon_centroids = np.array(polygon_centroids)
    self.neighbor_distance_threshold = neighbor_distance_threshold
    self.adjacency_dict = adjacency_dict
    
    # Create polygon to group mapping if not provided
    if polygon_group_mapping is None:
      self.polygon_group_mapping = {i: i for i in range(len(polygon_centroids))}
    else:
      self.polygon_group_mapping = polygon_group_mapping
    
    # Create reverse mapping from group IDs to polygon indices
    self.group_to_polygon_mapping = {}
    for poly_idx, group_ids in self.polygon_group_mapping.items():
      if isinstance(group_ids, (list, np.ndarray)):
        for group_id in group_ids:
          if group_id not in self.group_to_polygon_mapping:
            self.group_to_polygon_mapping[group_id] = []
          self.group_to_polygon_mapping[group_id].append(poly_idx)
      else:
        if group_ids not in self.group_to_polygon_mapping:
          self.group_to_polygon_mapping[group_ids] = []
        self.group_to_polygon_mapping[group_ids].append(poly_idx)
    
    print(f"PolygonGroupGenerator initialized with {len(polygon_centroids)} polygons and {len(self.group_to_polygon_mapping)} groups")
  
  def get_groups(self, X_polygon_ids):
    '''
    Generate group assignments from polygon IDs.
    
    Parameters:
    -----------
    X_polygon_ids : array-like, shape (n_samples,)
        Polygon IDs for each data point
        
    Returns:
    --------
    X_group : array-like, shape (n_samples,)
        Group ID assignment for each data point
    '''
    X_group = np.zeros(len(X_polygon_ids), dtype=int)
    
    for i, poly_id in enumerate(X_polygon_ids):
      if poly_id in self.polygon_group_mapping:
        group_id = self.polygon_group_mapping[poly_id]
        if isinstance(group_id, (list, np.ndarray)):
          # If multiple groups per polygon, use the first one
          X_group[i] = group_id[0]
        else:
          X_group[i] = group_id
      else:
        # If polygon not in mapping, assign to polygon ID itself
        X_group[i] = poly_id
    
    return X_group
  
  def get_contiguity_info(self):
    '''
    Get information needed for polygon-based contiguity refinement.
    
    Returns:
    --------
    contiguity_info : dict
        Dictionary containing centroids, group mapping, neighbor threshold, and adjacency dict
    '''
    contiguity_info = {
      'polygon_centroids': self.polygon_centroids,
      'polygon_group_mapping': self.polygon_group_mapping,
      'neighbor_distance_threshold': self.neighbor_distance_threshold
    }
    
    # Add adjacency dictionary if available
    if self.adjacency_dict is not None:
      contiguity_info['adjacency_dict'] = self.adjacency_dict
    
    return contiguity_info


def generate_polygon_groups_from_centroids(X_polygon_ids, polygon_centroids, 
                                          polygon_group_mapping=None, neighbor_distance_threshold=None, adjacency_dict=None):
  '''
  Convenience function to generate polygon-based groups.
  
  Parameters:
  -----------
  X_polygon_ids : array-like, shape (n_samples,)
      Polygon IDs for each data point
  polygon_centroids : array-like, shape (n_polygons, 2)
      Centroid coordinates (lat, lon) for each polygon
  polygon_group_mapping : dict, optional
      Mapping from polygon indices to group IDs
  neighbor_distance_threshold : float, optional
      Distance threshold for determining polygon neighbors
  adjacency_dict : dict, optional
      Pre-computed adjacency dictionary from true polygon boundaries.
      If provided, takes precedence over distance-based neighbor calculation.
      
  Returns:
  --------
  X_group : array-like, shape (n_samples,)
      Group ID assignment for each data point
  polygon_generator : PolygonGroupGenerator
      Generator instance for further use
  '''
  polygon_generator = PolygonGroupGenerator(polygon_centroids, polygon_group_mapping, neighbor_distance_threshold, adjacency_dict)
  X_group = polygon_generator.get_groups(X_polygon_ids)
  
  return X_group, polygon_generator

