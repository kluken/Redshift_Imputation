from time import time
from tqdm import tqdm
import lightgbm as lgb
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from sklearn.metrics import mean_squared_error
from numpy.random import default_rng
import numpy as np

class BaseMICE:
    """Base class for the MICE implementation."""
    
    def __init__(self, max_iter=10, previous_loss = 0, tqdm_disable = False, rand_generator = default_rng(42)):
        self.max_iter = max_iter
        self.previous_loss = previous_loss
        self.tqdm_disable = tqdm_disable
        self.rand_generator = rand_generator
    
    
    def fill_missing_values(self, df):
        """Fills the missing values of a pandas DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data with missing values (nans).
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with imputed missing values.
        """
        nan_ids = np.argwhere(df.isna().values).tolist()
        columns_missing = df.isna().sum()
        columns_missing = columns_missing[columns_missing > 0]
        df_imputed = self.impute_initial_mean_or_mode(df)
        iter_results = []
        for iter in range(self.max_iter):
            df_imputed = self.transform(df_imputed, columns_missing, nan_ids, iter)
            # df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int
        return df_imputed
    
    def benchmark(self, df_original, df_missing, drop_columns_loss=None):
        """Benchmarks the fill method for missing values.
        
        Parameters
        ----------
        df_original : pandas.DataFrame
            Original data.
        df_missing : pandas.DataFrame
            Input data with missing values (nans).
        drop_columns_loss : list, optional
            Drop columns in the result DataFrame when 
            computing the loss.
            
        Returns
        -------
        List of Dictionarys
            List of dictionarys, where each dictionary are the results of a single iteration, holding the iter, time_seconds and loss
        pandas.DataFrame
            DataFrame with imputed missing values.
        """
        columns_missing = df_missing.isna().sum()
        columns_missing = columns_missing[columns_missing > 0]
        nan_ids = np.argwhere(df_missing.isna().values).tolist()
        df_imputed = self.impute_initial_mean_or_mode(df_missing)
        self.df_mean = df_imputed.copy()
        
        iter_results = []
        for iter in range(self.max_iter):
            time_start = time()
            df_imputed = self.transform(df_imputed, columns_missing, nan_ids, iter)
            time_stop = time() - time_start
            if drop_columns_loss:
                loss = self.compute_loss(df_original.drop(columns=drop_columns_loss, axis=1), 
                                         df_imputed.drop(columns=drop_columns_loss, axis=1))
            else:
                loss = self.compute_loss(df_original, df_imputed)
            if self.previous_loss == 0:
                self.previous_loss = loss
                # df_imputed.to_csv("imputed_df.csv", index = False)
            elif self.previous_loss < loss:
                self.previous_loss = loss
                # df_imputed.to_csv("imputed_df.csv", index = False)
            iter_results.append({
                "iter": iter,
                "time_seconds": time_stop, 
                "loss": loss
            })
        return iter_results, df_imputed
    
    def benchmark_mean_loss(self, df_original, df_missing, drop_columns_loss=None):
        """Computes the same iterations as benchmark() but only for the of the mean imputation method."""
        time_start = time()
        df_imputed = self.impute_initial_mean_or_mode(df_missing)
        time_stop = time() - time_start
        
        iter_results = []
        for iter in range(self.max_iter):
            if drop_columns_loss:
                loss = self.compute_loss(df_original.drop(columns=drop_columns_loss, axis=1), 
                                         df_imputed.drop(columns=drop_columns_loss, axis=1))
            else:
                loss = self.compute_loss(df_original, df_imputed)
            iter_results.append({
                "iter": iter,
                "time_seconds": time_stop, 
                "loss": loss
            })
        return iter_results
    
    def get_model(self, target):
        if is_numeric_dtype(target):
            model = lgb.LGBMRegressor()
        else:
            model = lgb.LGBMClassifier()
        return model
    
    def compute_loss(self, original_df, filled_df):
        """Computes the difference between the original and filled DataFrames."""
        return mean_squared_error(original_df, filled_df)
    
    def impute_initial_mean_or_mode(self, df):
        df_new = df.copy()
        for column in df:
            if is_numeric_dtype(df[column]):
                df_new[column] = df_new[column].fillna(df_new[column].mean())
            else:
                df_new[column] = df_new[column].fillna(df_new[column].mode())
        return df_new
    
    def transform(self, df, nan_ids):
        pass

class VanilaMICE(BaseMICE):
    """MICE implementation using value by value imputation."""
    
    method_name = "Vanila MICE"
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        random_ids = self.rand_generator.permutation(len(nan_ids)).tolist()
        for id in tqdm(random_ids, desc=f"{self.method_name}: Iter {iter_id + 1} / {self.max_iter}", position=0, disable=self.tqdm_disable):
            # Setup data
            row_id, col_id = nan_ids[id]
            target_column_name = df.columns[col_id]
            X = df.drop(columns=[target_column_name], axis=1)
            X = pd.get_dummies(X, drop_first=True)
            y = df[target_column_name]
            
            # Fit model
            model = self.get_model(y).fit(X.drop(index=row_id), y.drop(index=row_id))
            
            # Predict value
            df.iloc[row_id, col_id] = model.predict(X.iloc[row_id:row_id + 1, :])
        return df

class FastMICE(BaseMICE):
    """MICE implementation using column by column imputation."""
    
    method_name = "Fast MICE"
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        rand_column_ids = self.rand_generator.permutation(len(columns_missing)).tolist()
      
        for column_id in tqdm(rand_column_ids, desc=f"{self.method_name}: Iter {iter_id + 1} / {self.max_iter}", position=0, disable=self.tqdm_disable):
            target_column_name = columns_missing.index[column_id]
            X = df.drop(columns=[target_column_name], axis=1)
            X = pd.get_dummies(X, drop_first=True)
            y = df[target_column_name]
            column_nan_ids = [id[0] for id in nan_ids if id[1] == column_id]
            # Fit model
            tempX = X.drop(index=column_nan_ids)
            tempY = y.drop(index=column_nan_ids)
            tempModel = self.get_model(y)
            model = tempModel.fit(tempX, tempY)
            
            # Predict value
            df.iloc[column_nan_ids, column_id] = model.predict(X.iloc[column_nan_ids, :])
        return df

class SlowFastMICE(BaseMICE):
    """MICE implementation using value-by-value imputation in the first iteration 
    and column-by-column for the remaining iterations."""
    
    method_name = "Slow-Fast MICE"
    
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
        self.vanila_mice = VanilaMICE(self.max_iter)
        self.vanila_mice.method_name = self.method_name
        self.fast_mice = FastMICE(self.max_iter)
        self.fast_mice.method_name = self.method_name
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        if iter_id > 0:
            return self.fast_mice.transform(df, columns_missing, nan_ids, iter_id)
        else:
            return self.vanila_mice.transform(df, columns_missing, nan_ids, iter_id)

class FastSlowMICE(BaseMICE):
    """MICE implementation using column-by-column imputation in all but one iterations
    and value-by-value for the last iteration."""
    
    method_name = "Fast-Slow MICE"
    
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
        self.vanila_mice = VanilaMICE(self.max_iter)
        self.vanila_mice.method_name = self.method_name
        self.fast_mice = FastMICE(self.max_iter)
        self.fast_mice.method_name = self.method_name
    
    def transform(self, df: pd.DataFrame, columns_missing: list, nan_ids: list, iter_id: int):
        if iter_id + 1 == self.max_iter:
            return self.vanila_mice.transform(df, columns_missing, nan_ids, iter_id)
        else:
            return self.fast_mice.transform(df, columns_missing, nan_ids, iter_id)