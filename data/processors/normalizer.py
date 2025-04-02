import pandas as pd
import numpy as np
from data.processors.base_processor import BaseProcessor


class DataNormalizer(BaseProcessor):
    """Specialized normalizer for financial and time series data"""
    
    def __init__(self, 
                 price_cols=['open', 'high', 'low', 'close'],
                 volume_col='volume',
                 price_method='returns',
                 volume_method='log',
                 other_method='zscore'):
        """
        Initialize the financial data normalizer
        
        Args:
            price_cols: List of price columns
            volume_col: Volume column name
            price_method: Method for price columns ('returns', 'zscore', 'log', 'pct_change')
            volume_method: Method for volume column ('log', 'zscore', 'pct_of_avg')
            other_method: Default method for other numerical columns
        """
        self.price_cols = [col.lower() for col in price_cols]
        self.volume_col = volume_col.lower()
        self.price_method = price_method
        self.volume_method = volume_method
        self.other_method = other_method
        self._params = {}
        
    def fit(self, data):
        """Learn parameters from data"""
        df = data.copy()
        # Convert column names to lowercase for consistent processing
        df.columns = [col.lower() for col in df.columns]
        
        for col in df.columns:
            # Determine method based on column type
            if col in self.price_cols:
                self._fit_column(df, col, self.price_method)
            elif col == self.volume_col:
                self._fit_column(df, col, self.volume_method)
            elif df[col].dtype.kind in 'ifc':  # Numeric columns only
                self._fit_column(df, col, self.other_method)
                
        self._fitted_columns = df.columns.tolist()
        return self
    
    def _fit_column(self, data, col, method):
        """Fit a single column based on the specified method"""
        if method == 'zscore':
            self._params[f"{col}_mean"] = data[col].mean()
            self._params[f"{col}_std"] = data[col].std()
        elif method == 'minmax':
            self._params[f"{col}_min"] = data[col].min()
            self._params[f"{col}_max"] = data[col].max()
        elif method == 'robust':
            self._params[f"{col}_median"] = data[col].median()
            self._params[f"{col}_iqr"] = data[col].quantile(0.75) - data[col].quantile(0.25)
        elif method == 'pct_of_avg':
            self._params[f"{col}_avg"] = data[col].mean()
        # No parameters needed for returns, log, pct_change
        
    def transform(self, data):
        """Transform the data using appropriate methods"""
        df = data.copy()
        result = pd.DataFrame(index=df.index)
        
        # Convert column names to lowercase for consistent processing
        df.columns = [col.lower() for col in df.columns]
        
        for col in df.columns:
            # Skip raw price columns - never normalize these
            if col.endswith('_raw'):
                result[col] = df[col]
            elif col in self.price_cols:
                self._transform_column(df, result, col, self.price_method)
            elif col == self.volume_col:
                self._transform_column(df, result, col, self.volume_method)
            elif df[col].dtype.kind in 'ifc':  # Numeric columns only
                self._transform_column(df, result, col, self.other_method)
            else:
                # Keep non-numeric columns as is
                result[col] = df[col]
                
        return result
    
    def _transform_column(self, data, result, col, method):
        """Transform a single column based on the method"""
        if method == 'returns':
            # Daily returns calculation
            result[f"{col}_return"] = data[col].pct_change()
        elif method == 'pct_change':
            # Percentage change from first value
            first_value = data[col].iloc[0]
            if first_value != 0:
                result[col] = (data[col] / first_value - 1) * 100
            else:
                result[col] = 0
        elif method == 'log':
            # Log transformation with small constant to avoid log(0)
            result[col] = np.log(data[col] + 1e-8)
        elif method == 'zscore':
            mean = self._params[f"{col}_mean"]
            std = self._params[f"{col}_std"]
            if std > 0:
                result[col] = (data[col] - mean) / std
            else:
                result[col] = 0
        elif method == 'minmax':
            min_val = self._params[f"{col}_min"]
            max_val = self._params[f"{col}_max"]
            if max_val > min_val:
                result[col] = (data[col] - min_val) / (max_val - min_val)
            else:
                result[col] = 0.5
        elif method == 'robust':
            median = self._params[f"{col}_median"]
            iqr = self._params[f"{col}_iqr"]
            if iqr > 0:
                result[col] = (data[col] - median) / iqr
            else:
                result[col] = 0
        elif method == 'pct_of_avg':
            avg = self._params[f"{col}_avg"]
            if avg > 0:
                result[col] = data[col] / avg
            else:
                result[col] = 0
        else:
            # Default: copy as is
            result[col] = data[col]
    
    def inverse_transform(self, data):
        """Revert normalized data to original scale (where possible)"""
        # Implementation would vary based on transformation methods used
        # Some transformations (like returns) are not fully reversible
        # without additional context
        pass