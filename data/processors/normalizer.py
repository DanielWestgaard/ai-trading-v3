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
        
        # Handle potential NaN values during fitting
        for col in df.columns:
            if df[col].dtype.kind in 'ifc':  # Numeric columns only
                # Skip columns with all NaN values
                if df[col].isna().all():
                    continue
                
                # Fill NaN values temporarily for parameter estimation
                clean_col = df[col].fillna(df[col].mean() if df[col].count() > 0 else 0)
                
                # Determine method based on column type
                if col in self.price_cols:
                    self._fit_column_safely(clean_col, col, self.price_method)
                elif col == self.volume_col:
                    self._fit_column_safely(clean_col, col, self.volume_method)
                else:
                    self._fit_column_safely(clean_col, col, self.other_method)
                
        self._fitted_columns = df.columns.tolist()
        return self
    
    def _fit_column_safely(self, data_series, col, method):
        """Safely fit a single column based on the specified method, handling edge cases"""
        try:
            if method == 'zscore':
                self._params[f"{col}_mean"] = data_series.mean()
                # Handle zero standard deviation
                std = data_series.std()
                self._params[f"{col}_std"] = std if std > 0 else 1.0
            elif method == 'minmax':
                self._params[f"{col}_min"] = data_series.min()
                # Handle min == max case
                min_val = data_series.min()
                max_val = data_series.max()
                if min_val == max_val:
                    self._params[f"{col}_range"] = 1.0
                else:
                    self._params[f"{col}_range"] = max_val - min_val
                self._params[f"{col}_max"] = max_val
            elif method == 'robust':
                self._params[f"{col}_median"] = data_series.median()
                # Handle zero IQR case
                q1 = data_series.quantile(0.25)
                q3 = data_series.quantile(0.75)
                iqr = q3 - q1
                self._params[f"{col}_iqr"] = iqr if iqr > 0 else 1.0
            elif method == 'pct_of_avg':
                avg = data_series.mean()
                self._params[f"{col}_avg"] = avg if avg != 0 else 1.0
        except Exception as e:
            # Log and handle any unexpected errors during fitting
            import logging
            logging.warning(f"Error fitting normalization parameters for column {col}: {str(e)}")
            # Set fallback parameters to ensure transform doesn't fail
            if method == 'zscore':
                self._params[f"{col}_mean"] = 0
                self._params[f"{col}_std"] = 1
            elif method == 'minmax':
                self._params[f"{col}_min"] = 0
                self._params[f"{col}_max"] = 1
                self._params[f"{col}_range"] = 1
            elif method == 'robust':
                self._params[f"{col}_median"] = 0
                self._params[f"{col}_iqr"] = 1
            elif method == 'pct_of_avg':
                self._params[f"{col}_avg"] = 1
    
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
                self._transform_column_safely(df, result, col, self.price_method)
            elif col == self.volume_col:
                self._transform_column_safely(df, result, col, self.volume_method)
            elif df[col].dtype.kind in 'ifc':  # Numeric columns only
                self._transform_column_safely(df, result, col, self.other_method)
            else:
                # Keep non-numeric columns as is
                result[col] = df[col]
        
        # Handle any remaining NaNs that might have been introduced during transformation
        result = self._ensure_no_missing_values(result)
        
        # Final check for any non-finite values across all numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Replace any non-finite values with 0
            non_finite_mask = ~np.isfinite(result[col])
            if non_finite_mask.any():
                result[col] = np.where(np.isfinite(result[col]), result[col], 0)
        
        return result
    
    def _transform_column_safely(self, data, result, col, method):
        """Transform a single column based on the method with robust error handling"""
        try:
            # Handle missing values first by filling with mean or 0
            is_nan_mask = data[col].isna()
            temp_series = data[col].copy()
            
            if is_nan_mask.any():
                temp_series = temp_series.fillna(temp_series.mean() if temp_series.count() > 0 else 0)
            
            if method == 'returns':
                # Daily returns calculation
                transformed = temp_series.pct_change(fill_method=None)
                result[f"{col}_return"] = transformed
                # Fill NaN in first row with 0
                if is_nan_mask.any() or True:  # Always handle first row
                    first_row_mask = (result.index == result.index[0])
                    result.loc[first_row_mask, f"{col}_return"] = result.loc[first_row_mask, f"{col}_return"].fillna(0)
                    # Restore original NaNs if requested (usually not needed)
                    # result.loc[is_nan_mask, f"{col}_return"] = np.nan  
            elif method == 'pct_change':
                # Percentage change from first value
                first_value = temp_series.iloc[0]
                if first_value != 0:
                    transformed = (temp_series / first_value - 1) * 100
                else:
                    transformed = temp_series * 0  # All zeros if first value is zero
                result[col] = transformed
            elif method == 'log':
                # Log transformation with small constant to avoid log(0) or log(negative)
                min_val = temp_series.min()
                offset = 0 if min_val > 0 else abs(min_val) + 1e-8
                # Apply log transform safely
                safe_values = temp_series + offset
                try:
                    transformed = np.log(safe_values)
                    # Replace any potential infinities or NaNs with finite values
                    transformed = np.where(np.isfinite(transformed), transformed, 0)
                except:
                    # Fallback if log transformation fails
                    transformed = safe_values - min_val
                result[col] = transformed
            elif method == 'zscore':
                mean = self._params.get(f"{col}_mean", 0)
                std = self._params.get(f"{col}_std", 1)
                if std > 0:
                    transformed = (temp_series - mean) / std
                else:
                    transformed = temp_series * 0  # All zeros if std is zero
                result[col] = transformed
            elif method == 'minmax':
                min_val = self._params.get(f"{col}_min", 0)
                range_val = self._params.get(f"{col}_range", 1)
                if range_val > 0:
                    transformed = (temp_series - min_val) / range_val
                else:
                    transformed = temp_series * 0 + 0.5  # All 0.5 if range is zero
                result[col] = transformed
            elif method == 'robust':
                median = self._params.get(f"{col}_median", 0)
                iqr = self._params.get(f"{col}_iqr", 1)
                if iqr > 0:
                    transformed = (temp_series - median) / iqr
                else:
                    transformed = temp_series * 0  # All zeros if IQR is zero
                result[col] = transformed
            elif method == 'pct_of_avg':
                avg = self._params.get(f"{col}_avg", 1)
                if avg > 0:
                    transformed = temp_series / avg
                else:
                    transformed = temp_series * 0  # All zeros if avg is zero
                result[col] = transformed
            else:
                # Default: copy as is
                result[col] = data[col]
                
            # Ensure all values are finite - replace any non-finite values with 0
            # This is a critical step to ensure we never have NaN, inf, or -inf in the output
            if col in result.columns and result[col].dtype.kind in 'fc':  # Float or complex types
                result[col] = np.where(np.isfinite(result[col]), result[col], 0)
                
        except Exception as e:
            # If anything goes wrong, just copy the original data
            import logging
            logging.warning(f"Error transforming column {col} with method {method}: {str(e)}")
            result[col] = data[col]
            # Even in error case, ensure values are finite
            if col in result.columns and result[col].dtype.kind in 'fc':
                result[col] = np.where(np.isfinite(result[col]), result[col], 0)
        
    def _ensure_no_missing_values(self, df):
        """Make sure there are absolutely no missing values in the final result"""
        result = df.copy()
        
        # 1. Handle numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            nan_mask = result[col].isna()
            if nan_mask.any():
                # Try forward fill first
                result[col] = result[col].ffill()
                
                # If still has NaNs, try backward fill
                nan_mask = result[col].isna()
                if nan_mask.any():
                    result[col] = result[col].bfill()
                    
                # If STILL has NaNs, fill with column mean or 0
                nan_mask = result[col].isna()
                if nan_mask.any():
                    if result[col].count() > 0:  # If we have at least some non-NaN values
                        mean_val = result[col].mean()
                        result[col] = result[col].fillna(mean_val)
                    else:
                        # No valid values at all, fill with 0
                        result[col] = result[col].fillna(0)
                
                # Replace any non-finite values with 0
                non_finite_mask = ~np.isfinite(result[col])
                if non_finite_mask.any():
                    result[col] = np.where(np.isfinite(result[col]), result[col], 0)
        
        # 2. Handle non-numeric columns
        non_numeric_cols = result.select_dtypes(exclude=['number']).columns
        for col in non_numeric_cols:
            nan_mask = result[col].isna()
            if nan_mask.any():
                # Forward fill for non-numeric
                result[col] = result[col].ffill()
                
                # If still has NaNs, backward fill
                nan_mask = result[col].isna()
                if nan_mask.any():
                    result[col] = result[col].bfill()
                    
                # If STILL has NaNs, fill with most common value or a placeholder
                nan_mask = result[col].isna()
                if nan_mask.any():
                    if result[col].count() > 0:  # If we have at least some non-NaN values
                        most_common = result[col].mode()[0]
                        result[col] = result[col].fillna(most_common)
                    else:
                        # No valid values at all, fill with "UNKNOWN"
                        result[col] = result[col].fillna("UNKNOWN")
        
        return result
        
    def inverse_transform(self, data):
        """Revert normalized data to original scale (where possible)"""
        # Implementation would vary based on transformation methods used
        # Some transformations (like returns) are not fully reversible
        # without additional context
        pass