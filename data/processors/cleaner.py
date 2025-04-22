from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union, Tuple

from data.processors.base_processor import BaseProcessor


class DataCleaner(BaseProcessor):
    """Financial time series data cleaner optimized for OHLCV market data"""
    
    def __init__(self, 
                 price_cols: List[str] = ['open', 'high', 'low', 'close'],
                 volume_col: str = 'volume',
                 timestamp_col: str = 'timestamp',
                 missing_method: str = 'ffill',
                 outlier_method: str = 'winsorize',
                 outlier_threshold: float = 3.0,
                 ensure_ohlc_validity: bool = True,
                 resample_rule: Optional[str] = None,
                 preserve_original_case: bool = True,
                 handle_volume_outliers: bool = False):
        """
        Initialize the data cleaner with configuration parameters.
        
        Args:
            price_cols: Column names for price data (default OHLC)
            volume_col: Column name for volume data
            timestamp_col: Column name for timestamp
            missing_method: Method for handling missing values ('ffill', 'bfill', 'interpolate')
            outlier_method: Method for handling outliers ('zscore', 'iqr', 'winsorize', 'none')
            outlier_threshold: Threshold for outlier detection
            ensure_ohlc_validity: Fix invalid OHLC relationships
            resample_rule: Optional rule for resampling (e.g., '5min', '1h')
            preserve_original_case: Whether to preserve the original case of column names
            handle_volume_outliers: Whether to apply outlier handling to volume data
        """
        logging.info("Initiating data cleaning with configuration: %s", 
                     {"price_cols": price_cols, "volume_col": volume_col, 
                      "missing_method": missing_method, "outlier_method": outlier_method,
                      "outlier_threshold": outlier_threshold, "resample_rule": resample_rule})
        
        # Store original column names for reference
        self.original_price_cols = price_cols
        self.original_volume_col = volume_col
        self.original_timestamp_col = timestamp_col
        self.preserve_original_case = preserve_original_case
        self.handle_volume_outliers = handle_volume_outliers
        
        # Convert to lowercase for internal processing
        self.price_cols = [col.lower() for col in price_cols]
        self.volume_col = volume_col.lower()
        self.timestamp_col = timestamp_col.lower()
        self.missing_method = missing_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.ensure_ohlc_validity = ensure_ohlc_validity
        self.resample_rule = resample_rule
        
        # Parameters to be learned during fit
        self._stats = {}
        
        # Store original column to lowercase mapping
        self.column_mapping = {}
        
    def fit(self, data: pd.DataFrame) -> 'DataCleaner':
        """Learn parameters from the data for outlier detection"""
        logging.info("Fitting DataCleaner on dataset with shape %s", data.shape)
        df = data.copy()
        
        # Store original column names mapping before converting to lowercase
        if self.preserve_original_case:
            self.column_mapping = {col.lower(): col for col in df.columns}
        
        # Convert column names to lowercase for internal processing
        df.columns = [col.lower() for col in df.columns]
        logging.debug("Converted column names to lowercase: %s", list(df.columns))
        
        # Calculate statistics for outlier detection
        missing_cols = [col for col in self.price_cols if col not in df.columns]
        if missing_cols:
            logging.warning("Some specified price columns not found in data: %s", missing_cols)
            
        for col in self.price_cols:
            if col in df.columns:
                logging.debug("Calculating statistics for column: %s", col)
                
                try:
                    # Try to convert the column to numeric first, coercing non-numeric values to NaN
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    non_nan_values = numeric_values.dropna()
                    
                    if len(non_nan_values) > 0:
                        # Calculate statistics only on valid numeric values
                        self._stats[f"{col}_mean"] = non_nan_values.mean()
                        self._stats[f"{col}_std"] = non_nan_values.std()
                        self._stats[f"{col}_median"] = non_nan_values.median()
                        self._stats[f"{col}_q1"] = non_nan_values.quantile(0.25)
                        self._stats[f"{col}_q3"] = non_nan_values.quantile(0.75)
                        self._stats[f"{col}_iqr"] = self._stats[f"{col}_q3"] - self._stats[f"{col}_q1"]
                        
                        # Log if we had to drop non-numeric values
                        if len(non_nan_values) < len(df[col]):
                            logging.warning(f"Column '{col}' contains non-numeric values. Statistics calculated on {len(non_nan_values)}/{len(df[col])} values.")
                    else:
                        # No valid numeric values, set default statistics
                        logging.warning(f"Column '{col}' has no valid numeric values. Using default statistics.")
                        self._stats[f"{col}_mean"] = 0
                        self._stats[f"{col}_std"] = 1
                        self._stats[f"{col}_median"] = 0
                        self._stats[f"{col}_q1"] = 0
                        self._stats[f"{col}_q3"] = 0
                        self._stats[f"{col}_iqr"] = 0
                except Exception as e:
                    # Handle any other errors gracefully
                    logging.error(f"Error calculating statistics for column '{col}': {str(e)}. Using default values.")
                    self._stats[f"{col}_mean"] = 0
                    self._stats[f"{col}_std"] = 1
                    self._stats[f"{col}_median"] = 0
                    self._stats[f"{col}_q1"] = 0
                    self._stats[f"{col}_q3"] = 0
                    self._stats[f"{col}_iqr"] = 0
        
        # Also calculate statistics for volume if handling volume outliers
        if self.handle_volume_outliers and self.volume_col in df.columns:
            logging.debug("Calculating statistics for volume column: %s", self.volume_col)
            
            try:
                # Convert volume to numeric, handling non-numeric values
                numeric_values = pd.to_numeric(df[self.volume_col], errors='coerce')
                non_nan_values = numeric_values.dropna()
                
                if len(non_nan_values) > 0:
                    self._stats[f"{self.volume_col}_mean"] = non_nan_values.mean()
                    self._stats[f"{self.volume_col}_std"] = non_nan_values.std()
                    self._stats[f"{self.volume_col}_median"] = non_nan_values.median()
                    self._stats[f"{self.volume_col}_q1"] = non_nan_values.quantile(0.25)
                    self._stats[f"{self.volume_col}_q3"] = non_nan_values.quantile(0.75)
                    self._stats[f"{self.volume_col}_iqr"] = self._stats[f"{self.volume_col}_q3"] - self._stats[f"{self.volume_col}_q1"]
                    
                    # Log if we had to drop non-numeric values
                    if len(non_nan_values) < len(df[self.volume_col]):
                        logging.warning(f"Volume column contains non-numeric values. Statistics calculated on {len(non_nan_values)}/{len(df[self.volume_col])} values.")
                else:
                    logging.warning(f"Volume column has no valid numeric values. Using default statistics.")
                    self._stats[f"{self.volume_col}_mean"] = 0
                    self._stats[f"{self.volume_col}_std"] = 1
                    self._stats[f"{self.volume_col}_median"] = 0
                    self._stats[f"{self.volume_col}_q1"] = 0
                    self._stats[f"{self.volume_col}_q3"] = 0
                    self._stats[f"{self.volume_col}_iqr"] = 0
            except Exception as e:
                logging.error(f"Error calculating volume statistics: {str(e)}. Using default values.")
                self._stats[f"{self.volume_col}_mean"] = 0
                self._stats[f"{self.volume_col}_std"] = 1
                self._stats[f"{self.volume_col}_median"] = 0
                self._stats[f"{self.volume_col}_q1"] = 0
                self._stats[f"{self.volume_col}_q3"] = 0
                self._stats[f"{self.volume_col}_iqr"] = 0
        
        logging.info("DataCleaner fitting completed. Statistics calculated for %d columns.", 
                    len([col for col in self.price_cols if col in df.columns]) + 
                    (1 if self.handle_volume_outliers and self.volume_col in df.columns else 0))
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning operations to the data"""
        logging.info("Starting data transformation on dataset with shape %s", data.shape)
        df = data.copy()
        
        try:
            # Store original case before converting to lowercase 
            original_columns = df.columns.tolist()
            if self.preserve_original_case:
                self.column_mapping = {col.lower(): col for col in df.columns}
            
            # Convert column names to lowercase for consistent processing
            df.columns = [col.lower() for col in df.columns]
            
            # Identify price and volume columns with case-insensitive matching
            price_cols_in_data = [col for col in df.columns if col in self.price_cols]
            volume_col_in_data = self.volume_col if self.volume_col in df.columns else None
            
            # Convert price columns to numeric right away
            for col in price_cols_in_data:
                # Convert to numeric, coercing non-numeric values to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                non_numeric_count = df[col].isna().sum() - data[self.column_mapping.get(col, col) if self.preserve_original_case else col].isna().sum()
                if non_numeric_count > 0:
                    logging.warning(f"Converted {non_numeric_count} non-numeric values to NaN in column '{col}'")
            
            # Convert volume column to numeric as well
            if volume_col_in_data:
                df[volume_col_in_data] = pd.to_numeric(df[volume_col_in_data], errors='coerce')
            
            # Make a temporary copy of the timestamp column if we need to set it as index
            # This ensures we don't lose the original timestamp column
            has_timestamp = self.timestamp_col in df.columns
            if has_timestamp:
                orig_timestamp_name = self.timestamp_col
                
                # Convert timestamp to datetime if not already
                df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors='coerce')
                
                # Only set index temporarily for processing if needed for resampling
                if self.resample_rule:
                    logging.debug("Setting %s as temporary DataFrame index for resampling", self.timestamp_col)
                    df = df.set_index(self.timestamp_col)
            
            # Ensure time continuity by resampling if requested
            if self.resample_rule and isinstance(df.index, pd.DatetimeIndex):
                logging.info("Resampling data with rule: %s", self.resample_rule)
                original_shape = df.shape
                df = self._ensure_time_continuity(df)
                logging.info("Resampling changed shape from %s to %s", original_shape, df.shape)
            
            # Handle missing values
            missing_before = df.isna().sum().sum()
            logging.info("Handling missing values with method: %s. Total missing values: %d", 
                        self.missing_method, missing_before)
            df = self._handle_missing_values(df)
            missing_after = df.isna().sum().sum()
            logging.info("Missing values after handling: %d (reduction: %d)", 
                        missing_after, missing_before - missing_after)
            
            # Handle outliers
            if self.outlier_method != 'none':
                logging.info("Handling outliers with method: %s and threshold: %f", 
                            self.outlier_method, self.outlier_threshold)
                df = self._handle_outliers(df)
                logging.debug("Outlier handling completed")
            
            # Ensure OHLC validity (High ≥ Open ≥ Close ≥ Low)
            if self.ensure_ohlc_validity:
                logging.info("Ensuring OHLC validity")
                df = self._ensure_ohlc_validity(df)
                logging.debug("OHLC validity check completed")
            
            # Reset index if it was set for processing
            if self.resample_rule and isinstance(df.index, pd.DatetimeIndex):
                logging.debug("Resetting index to include timestamp column")
                df = df.reset_index()
            
            # Final check to ensure all price and volume columns are numeric
            for col in price_cols_in_data + ([volume_col_in_data] if volume_col_in_data else []):
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    logging.warning(f"Column {col} is still not numeric after processing. Forcing conversion.")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Restore original column case if needed
            if self.preserve_original_case:
                # Create new column mapping that includes any new columns created during processing
                new_mapping = {}
                for col in df.columns:
                    if col in self.column_mapping:
                        # Use original case for existing columns
                        new_mapping[col] = self.column_mapping[col]
                    else:
                        # Keep new columns as is
                        new_mapping[col] = col
                
                # Rename columns to original case
                df = df.rename(columns=new_mapping)
                logging.debug("Restored original column case")
                
                # Ensure all original price columns have numeric types in the output
                for orig_col in original_columns:
                    if orig_col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                        if orig_col in df.columns and not pd.api.types.is_numeric_dtype(df[orig_col]):
                            logging.warning(f"Forcing {orig_col} to numeric in final output")
                            df[orig_col] = pd.to_numeric(df[orig_col], errors='coerce').fillna(0)
                
            logging.info("Data transformation completed. Final shape: %s", df.shape)
            return df
        except Exception as e:
            logging.error("Transform failed: %s", str(e))
            # If transform fails, return a copy of input data with numeric price columns
            try:
                # Create a safe output with proper types
                safe_df = data.copy()
                for col in safe_df.columns:
                    if col.upper() in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
                        safe_df[col] = pd.to_numeric(safe_df[col], errors='coerce').fillna(0)
                return safe_df
            except:
                # If even that fails, return the input unchanged
                return data
    
    def _handle_missing_values(self, df: pd.DataFrame, column_specific_methods=None) -> pd.DataFrame:
        """Handle missing values differently for price and volume"""
        logging.debug("Handling missing values for %d columns", len(df.columns))
        
        # For price columns, use the specified method
        for col in self.price_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    logging.debug("Column %s has %d missing values (%.2f%%)", 
                                col, missing_count, (missing_count/len(df))*100)
                    
                if self.missing_method == 'ffill':
                    # Using direct ffill and bfill methods instead of deprecated fillna(method=...)
                    df[col] = df[col].ffill()
                    # Handle case where first values might be NaN
                    df[col] = df[col].bfill()
                elif self.missing_method == 'interpolate':
                    # Check if we're dealing with time series data with a datetime index
                    if isinstance(df.index, pd.DatetimeIndex):
                        # If we have a datetime index, use time-weighted interpolation
                        df[col] = df[col].interpolate(method='time')
                    else:
                        # For non-time indexed data, use linear interpolation instead
                        df[col] = df[col].interpolate(method='linear')
                    
                    # Handle endpoints
                    df[col] = df[col].ffill().bfill()
                
                remaining_missing = df[col].isna().sum()
                if remaining_missing > 0:
                    logging.warning("Column %s still has %d missing values after %s", 
                                col, remaining_missing, self.missing_method)
        
        # For volume, fill NaNs with 0 as it's often the correct value for no trading
        if self.volume_col in df.columns:
            missing_count = df[self.volume_col].isna().sum()
            if missing_count > 0:
                logging.debug("Volume column has %d missing values, filling with zeros", missing_count)
                df[self.volume_col] = df[self.volume_col].fillna(0)
        else:
            logging.debug("Volume column '%s' not found in data", self.volume_col)
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in price data"""
        logging.debug("Handling outliers using method: %s", self.outlier_method)
        
        if len(self._stats) == 0:
            logging.warning("No statistics available for outlier detection. Run fit() first.")
            return df
            
        # Process price columns
        for col in self.price_cols:
            if col not in df.columns:
                continue
            
            # Ensure the column is numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            logging.debug("Processing outliers for column: %s", col)
            
            try:
                # Store original dtype to preserve it
                original_dtype = df[col].dtype
                
                if self.outlier_method == 'zscore':
                    # Z-score method
                    mean = self._stats.get(f"{col}_mean", df[col].mean())
                    std = self._stats.get(f"{col}_std", df[col].std())
                    
                    # Ensure std is positive to avoid division by zero
                    if std <= 0:
                        std = 1.0
                        
                    outliers = abs((df[col] - mean) / std) > self.outlier_threshold
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        logging.info("Found %d outliers (%.2f%%) in column %s using zscore method", 
                                outlier_count, (outlier_count/len(df))*100, col)
                        
                        # Create replacement values
                        replacement_values = np.where(
                            df.loc[outliers, col] > mean,
                            mean + (self.outlier_threshold * std),
                            mean - (self.outlier_threshold * std)
                        )
                        
                        # Convert to the original dtype before assignment
                        df.loc[outliers, col] = pd.Series(replacement_values, index=df.loc[outliers].index).astype(original_dtype)
                
                elif self.outlier_method == 'iqr':
                    # IQR method
                    q1 = self._stats.get(f"{col}_q1", df[col].quantile(0.25))
                    q3 = self._stats.get(f"{col}_q3", df[col].quantile(0.75))
                    iqr = self._stats.get(f"{col}_iqr", q3 - q1)
                    
                    # Ensure IQR is positive
                    if iqr <= 0:
                        iqr = 1.0
                        
                    lower_bound = q1 - (self.outlier_threshold * iqr)
                    upper_bound = q3 + (self.outlier_threshold * iqr)
                    
                    # Count outliers before clipping
                    lower_outliers = (df[col] < lower_bound).sum()
                    upper_outliers = (df[col] > upper_bound).sum()
                    
                    if lower_outliers > 0 or upper_outliers > 0:
                        logging.info("Column %s: Found %d lower outliers and %d upper outliers using IQR method", 
                                col, lower_outliers, upper_outliers)
                    
                    # Clip values to bounds and convert to original dtype
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).astype(original_dtype)
                    
                elif self.outlier_method == 'winsorize':
                    # Winsorize (clip to percentiles)
                    lower_percentile = 1
                    upper_percentile = 99
                    
                    try:
                        # Handle empty series or all-NaN series
                        non_nan_values = df[col].dropna()
                        if len(non_nan_values) > 0:
                            lower_bound = np.percentile(non_nan_values, lower_percentile)
                            upper_bound = np.percentile(non_nan_values, upper_percentile)
                            
                            # Count outliers before clipping
                            lower_outliers = (df[col] < lower_bound).sum()
                            upper_outliers = (df[col] > upper_bound).sum()
                            
                            if lower_outliers > 0 or upper_outliers > 0:
                                logging.info("Column %s: Winsorizing %d lower outliers and %d upper outliers", 
                                        col, lower_outliers, upper_outliers)
                            
                            # Clip values to bounds and convert to original dtype
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).astype(original_dtype)
                        else:
                            logging.warning(f"Column {col} has no non-NaN values, skipping winsorization")
                    except Exception as e:
                        logging.error(f"Error winsorizing column {col}: {str(e)}")
            except Exception as e:
                logging.error(f"Error handling outliers for column {col}: {str(e)}")
        
        # Process volume column if requested
        if self.handle_volume_outliers and self.volume_col in df.columns:
            logging.debug("Processing outliers for volume column: %s", self.volume_col)
            
            # Ensure volume column is numeric
            if not pd.api.types.is_numeric_dtype(df[self.volume_col]):
                df[self.volume_col] = pd.to_numeric(df[self.volume_col], errors='coerce')
            
            try:
                # Store original dtype to preserve it
                original_dtype = df[self.volume_col].dtype
                
                if self.outlier_method == 'zscore':
                    # Z-score method - same approach as for price columns
                    mean = self._stats.get(f"{self.volume_col}_mean", df[self.volume_col].mean())
                    std = self._stats.get(f"{self.volume_col}_std", df[self.volume_col].std())
                    
                    # Ensure std is positive
                    if std <= 0:
                        std = 1.0
                        
                    outliers = abs((df[self.volume_col] - mean) / std) > self.outlier_threshold
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        logging.info("Found %d outliers (%.2f%%) in volume column using zscore method", 
                                outlier_count, (outlier_count/len(df))*100)
                        
                        replacement_values = np.where(
                            df.loc[outliers, self.volume_col] > mean,
                            mean + (self.outlier_threshold * std),
                            mean - (self.outlier_threshold * std)
                        )
                        
                        df.loc[outliers, self.volume_col] = pd.Series(
                            replacement_values, index=df.loc[outliers].index
                        ).astype(original_dtype)
                    
                elif self.outlier_method == 'iqr':
                    # Implementation for IQR method (similar to price columns)
                    # ... code similar to price column handling ...
                    q1 = self._stats.get(f"{self.volume_col}_q1", df[self.volume_col].quantile(0.25))
                    q3 = self._stats.get(f"{self.volume_col}_q3", df[self.volume_col].quantile(0.75))
                    iqr = self._stats.get(f"{self.volume_col}_iqr", q3 - q1)
                    
                    # Ensure IQR is positive
                    if iqr <= 0:
                        iqr = 1.0
                        
                    lower_bound = q1 - (self.outlier_threshold * iqr)
                    upper_bound = q3 + (self.outlier_threshold * iqr)
                    
                    # Count outliers before clipping
                    lower_outliers = (df[self.volume_col] < lower_bound).sum()
                    upper_outliers = (df[self.volume_col] > upper_bound).sum()
                    
                    if lower_outliers > 0 or upper_outliers > 0:
                        logging.info("Volume column: Found %d lower outliers and %d upper outliers using IQR method", 
                                lower_outliers, upper_outliers)
                    
                    # Clip values to bounds and convert to original dtype
                    df[self.volume_col] = df[self.volume_col].clip(
                        lower=lower_bound, upper=upper_bound
                    ).astype(original_dtype)
                    
                elif self.outlier_method == 'winsorize':
                    # Implementation for winsorizing (similar to price columns)
                    lower_percentile = 1
                    upper_percentile = 99
                    
                    try:
                        non_nan_values = df[self.volume_col].dropna()
                        if len(non_nan_values) > 0:
                            lower_bound = np.percentile(non_nan_values, lower_percentile)
                            upper_bound = np.percentile(non_nan_values, upper_percentile)
                            
                            # Count outliers before clipping
                            lower_outliers = (df[self.volume_col] < lower_bound).sum()
                            upper_outliers = (df[self.volume_col] > upper_bound).sum()
                            
                            if lower_outliers > 0 or upper_outliers > 0:
                                logging.info("Volume column: Winsorizing %d lower outliers and %d upper outliers", 
                                        lower_outliers, upper_outliers)
                            
                            # Clip values to bounds and convert to original dtype
                            df[self.volume_col] = df[self.volume_col].clip(
                                lower=lower_bound, upper=upper_bound
                            ).astype(original_dtype)
                        else:
                            logging.warning(f"Volume column has no non-NaN values, skipping winsorization")
                    except Exception as e:
                        logging.error(f"Error winsorizing volume column: {str(e)}")
            except Exception as e:
                logging.error(f"Error handling outliers for volume column: {str(e)}")
                
        return df
            
    def _ensure_time_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure regular time intervals by resampling"""
        logging.debug("Ensuring time continuity with resample rule: %s", self.resample_rule)
        
        # Make sure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            error_msg = "DataFrame index must be a DatetimeIndex to ensure time continuity"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Check for time gaps before resampling
        time_diff = df.index.to_series().diff().dt.total_seconds()
        max_gap = time_diff.max()
        if max_gap > 0:
            logging.info("Before resampling: max time gap is %.2f seconds (%.2f minutes)", 
                        max_gap, max_gap/60)
        
        # For OHLC data, use specific resampling
        has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
        
        if has_ohlc:
            logging.debug("Using OHLC-specific resampling")
            # Use proper OHLC resampling
            resampled = df.resample(self.resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            
            # Handle volume separately if it exists
            if self.volume_col in df.columns:
                logging.debug("Resampling volume column with sum aggregation")
                volume_resampled = df[self.volume_col].resample(self.resample_rule).sum()
                resampled[self.volume_col] = volume_resampled
                
            # Merge any other columns with appropriate aggregation
            other_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', self.volume_col]]
            if other_cols:
                logging.debug("Resampling %d additional columns with mean aggregation", len(other_cols))
                for col in other_cols:
                    resampled[col] = df[col].resample(self.resample_rule).mean()
                    
            logging.info("Resampling complete: shape changed from %s to %s", df.shape, resampled.shape)
            return resampled
        else:
            logging.debug("Using forward fill resampling for non-OHLC data")
            result = df.resample(self.resample_rule).ffill()
            logging.info("Resampling complete: shape changed from %s to %s", df.shape, result.shape)
            return result
    
    def _ensure_ohlc_validity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC relationships are valid: High ≥ Open ≥ Close ≥ Low"""
        
        # Check if all required columns exist
        price_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in price_cols):
            logging.debug("Skipping OHLC validity check, not all OHLC columns present")
            return df
        
        try:
            logging.debug("Checking OHLC validity")
            
            # Make sure all price columns are numeric
            for col in price_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for invalid relationships
            high_vals = df['high'].copy()
            low_vals = df['low'].copy()
            open_vals = df['open'].copy()
            close_vals = df['close'].copy()
            
            # Calculate max of open and close for each row safely
            max_open_close = pd.DataFrame({'open': open_vals, 'close': close_vals}).max(axis=1)
            min_open_close = pd.DataFrame({'open': open_vals, 'close': close_vals}).min(axis=1)
            
            # Find invalid high/low values
            invalid_high_mask = high_vals < max_open_close
            invalid_low_mask = low_vals > min_open_close
            
            invalid_high_count = invalid_high_mask.sum()
            invalid_low_count = invalid_low_mask.sum()
            
            if invalid_high_count > 0 or invalid_low_count > 0:
                logging.warning("Found %d invalid high values and %d invalid low values", 
                                invalid_high_count, invalid_low_count)
            
            # Fix high values that are too low - use row-wise maximum
            df.loc[invalid_high_mask, 'high'] = pd.DataFrame({
                'high': df.loc[invalid_high_mask, 'high'],
                'open': df.loc[invalid_high_mask, 'open'],
                'close': df.loc[invalid_high_mask, 'close']
            }).max(axis=1)
            
            # Fix low values that are too high - use row-wise minimum
            df.loc[invalid_low_mask, 'low'] = pd.DataFrame({
                'low': df.loc[invalid_low_mask, 'low'],
                'open': df.loc[invalid_low_mask, 'open'],
                'close': df.loc[invalid_low_mask, 'close']
            }).min(axis=1)
            
            logging.debug("OHLC validity check and correction completed")
        except Exception as e:
            logging.error(f"Error ensuring OHLC validity: {str(e)}")
            # Continue without fixing OHLC relationships if there's an error
            
        return df