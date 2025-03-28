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
                 resample_rule: Optional[str] = None):
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
        """
        logging.info("Initiating data cleaning with configuration: %s", 
                     {"price_cols": price_cols, "volume_col": volume_col, 
                      "missing_method": missing_method, "outlier_method": outlier_method,
                      "outlier_threshold": outlier_threshold, "resample_rule": resample_rule})
        
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
        
    def fit(self, data: pd.DataFrame) -> 'DataCleaner':
        """Learn parameters from the data for outlier detection"""
        logging.info("Fitting DataCleaner on dataset with shape %s", data.shape)
        df = data.copy()
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        logging.debug("Converted column names to lowercase: %s", list(df.columns))
        
        # Calculate statistics for outlier detection
        missing_cols = [col for col in self.price_cols if col not in df.columns]
        if missing_cols:
            logging.warning("Some specified price columns not found in data: %s", missing_cols)
            
        for col in self.price_cols:
            if col in df.columns:
                logging.debug("Calculating statistics for column: %s", col)
                self._stats[f"{col}_mean"] = df[col].mean()
                self._stats[f"{col}_std"] = df[col].std()
                self._stats[f"{col}_median"] = df[col].median()
                self._stats[f"{col}_q1"] = df[col].quantile(0.25)
                self._stats[f"{col}_q3"] = df[col].quantile(0.75)
                self._stats[f"{col}_iqr"] = self._stats[f"{col}_q3"] - self._stats[f"{col}_q1"]
        
        logging.info("DataCleaner fitting completed. Statistics calculated for %d columns.", 
                     len([col for col in self.price_cols if col in df.columns]))
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning operations to the data"""
        logging.info("Starting data transformation on dataset with shape %s", data.shape)
        df = data.copy()
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure the timestamp is the index
        if self.timestamp_col in df.columns:
            logging.debug("Setting %s as DataFrame index", self.timestamp_col)
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df.set_index(self.timestamp_col, inplace=True)
        else:
            logging.warning("Timestamp column '%s' not found in data", self.timestamp_col)
        
        # Ensure time continuity by resampling if requested
        if self.resample_rule:
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
        
        # Reset index if needed
        if self.timestamp_col not in df.columns:
            logging.debug("Resetting index to include timestamp column")
            df.reset_index(inplace=True)
            
        logging.info("Data transformation completed. Final shape: %s", df.shape)
        return df
    
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
                    df[col] = df[col].fillna(method='ffill')
                    # Handle case where first values might be NaN
                    df[col] = df[col].fillna(method='bfill')
                elif self.missing_method == 'interpolate':
                    df[col] = df[col].interpolate(method='time')
                    # Handle endpoints
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
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
            
        for col in self.price_cols:
            if col not in df.columns:
                continue
                
            logging.debug("Processing outliers for column: %s", col)
            
            if self.outlier_method == 'zscore':
                # Z-score method
                mean = self._stats[f"{col}_mean"]
                std = self._stats[f"{col}_std"]
                outliers = abs((df[col] - mean) / std) > self.outlier_threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logging.info("Found %d outliers (%.2f%%) in column %s using zscore method", 
                                outlier_count, (outlier_count/len(df))*100, col)
                    
                    # Replace outliers with threshold values
                    df.loc[outliers, col] = np.where(
                        df.loc[outliers, col] > mean,
                        mean + (self.outlier_threshold * std),
                        mean - (self.outlier_threshold * std)
                    )
                
            elif self.outlier_method == 'iqr':
                # IQR method
                q1 = self._stats[f"{col}_q1"]
                q3 = self._stats[f"{col}_q3"]
                iqr = self._stats[f"{col}_iqr"]
                lower_bound = q1 - (self.outlier_threshold * iqr)
                upper_bound = q3 + (self.outlier_threshold * iqr)
                
                # Count outliers before clipping
                lower_outliers = (df[col] < lower_bound).sum()
                upper_outliers = (df[col] > upper_bound).sum()
                
                if lower_outliers > 0 or upper_outliers > 0:
                    logging.info("Column %s: Found %d lower outliers and %d upper outliers using IQR method", 
                                col, lower_outliers, upper_outliers)
                
                # Clip values to bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif self.outlier_method == 'winsorize':
                # Winsorize (clip to percentiles)
                lower_percentile = 1
                upper_percentile = 99
                
                try:
                    lower_bound = np.percentile(df[col].dropna(), lower_percentile)
                    upper_bound = np.percentile(df[col].dropna(), upper_percentile)
                    
                    # Count outliers before clipping
                    lower_outliers = (df[col] < lower_bound).sum()
                    upper_outliers = (df[col] > upper_bound).sum()
                    
                    if lower_outliers > 0 or upper_outliers > 0:
                        logging.info("Column %s: Winsorizing %d lower outliers and %d upper outliers", 
                                    col, lower_outliers, upper_outliers)
                    
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                except Exception as e:
                    logging.error("Error winsorizing column %s: %s", col, str(e))
                
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
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logging.debug("Checking OHLC validity")
            
            # Check for invalid relationships
            invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            
            if invalid_high > 0 or invalid_low > 0:
                logging.warning("Found %d invalid high values and %d invalid low values", 
                               invalid_high, invalid_low)
            
            # Fix high values that are too low
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            
            # Fix low values that are too high
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
            logging.debug("OHLC validity check and correction completed")
        else:
            logging.debug("Skipping OHLC validity check, not all OHLC columns present")
            
        return df
    
    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate data quality metrics after cleaning"""
        logging.info("Calculating data quality metrics")
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]
        
        metrics = {
            "row_count": len(df),
            "missing_values": {col: df[col].isna().sum() for col in df.columns},
            "missing_percentage": {col: df[col].isna().mean() * 100 for col in df.columns},
        }
        
        # Add price-specific metrics if available
        price_cols = [col for col in self.price_cols if col in df.columns]
        if price_cols:
            # Calculate returns for volatility
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
                metrics["volatility"] = df['returns'].std() * np.sqrt(252 * 24 * 60 / 5)  # Annualized for 5-min data
                logging.debug("Calculated annualized volatility: %.4f", metrics["volatility"])
                
            # Count potential anomalies after cleaning
            if self.outlier_method != 'none':
                for col in price_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    potential_outliers = abs((df[col] - mean) / std) > self.outlier_threshold
                    metrics[f"{col}_potential_outliers"] = potential_outliers.sum()
                    if potential_outliers.sum() > 0:
                        logging.warning("Column %s still has %d potential outliers after cleaning", 
                                       col, potential_outliers.sum())
        
        logging.info("Data quality metrics calculation completed")
        return metrics

    def handle_gaps(self, df, max_gap_minutes=15):
        """Flag or handle large time gaps in data"""
        logging.info("Checking for time gaps with threshold of %d minutes", max_gap_minutes)
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.warning("DataFrame index is not DatetimeIndex, converting timestamps")
            if self.timestamp_col in df.columns:
                df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
                df.set_index(self.timestamp_col, inplace=True)
            else:
                logging.error("Cannot check for gaps: no timestamp column or DatetimeIndex available")
                return df
        
        df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60
        large_gaps = df['time_diff'] > max_gap_minutes
        gap_count = large_gaps.sum()
        
        if gap_count > 0:
            largest_gap = df['time_diff'].max()
            logging.warning("Found %d time gaps larger than %d minutes. Largest gap: %.2f minutes", 
                           gap_count, max_gap_minutes, largest_gap)
            
            # Detailed logging of gaps
            if gap_count < 10:  # Only log details for a reasonable number of gaps
                for idx in df[large_gaps].index:
                    gap_size = df.loc[idx, 'time_diff']
                    prev_time = idx - pd.Timedelta(minutes=gap_size)
                    logging.debug("Gap of %.2f minutes between %s and %s", 
                                 gap_size, prev_time, idx)
        else:
            logging.info("No time gaps larger than %d minutes found", max_gap_minutes)
            
        df['gap_flag'] = large_gaps
        return df