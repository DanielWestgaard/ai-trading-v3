from abc import ABC, abstractmethod
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
        df = data.copy()
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate statistics for outlier detection
        for col in self.price_cols:
            if col in df.columns:
                self._stats[f"{col}_mean"] = df[col].mean()
                self._stats[f"{col}_std"] = df[col].std()
                self._stats[f"{col}_median"] = df[col].median()
                self._stats[f"{col}_q1"] = df[col].quantile(0.25)
                self._stats[f"{col}_q3"] = df[col].quantile(0.75)
                self._stats[f"{col}_iqr"] = self._stats[f"{col}_q3"] - self._stats[f"{col}_q1"]
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning operations to the data"""
        df = data.copy()
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure the timestamp is the index
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df.set_index(self.timestamp_col, inplace=True)
        
        # Ensure time continuity by resampling if requested
        if self.resample_rule:
            df = self._ensure_time_continuity(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        if self.outlier_method != 'none':
            df = self._handle_outliers(df)
        
        # Ensure OHLC validity (High ≥ Open ≥ Close ≥ Low)
        if self.ensure_ohlc_validity:
            df = self._ensure_ohlc_validity(df)
        
        # Reset index if needed
        if self.timestamp_col not in df.columns:
            df.reset_index(inplace=True)
            
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values differently for price and volume"""
        # For price columns, use the specified method
        for col in self.price_cols:
            if col in df.columns:
                if self.missing_method == 'ffill':
                    df[col] = df[col].fillna(method='ffill')
                    # Handle case where first values might be NaN
                    df[col] = df[col].fillna(method='bfill')
                elif self.missing_method == 'interpolate':
                    df[col] = df[col].interpolate(method='time')
                    # Handle endpoints
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # For volume, fill NaNs with 0 as it's often the correct value for no trading
        if self.volume_col in df.columns:
            df[self.volume_col] = df[self.volume_col].fillna(0)
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in price data"""
        for col in self.price_cols:
            if col not in df.columns or len(self._stats) == 0:
                continue
                
            if self.outlier_method == 'zscore':
                # Z-score method
                mean = self._stats[f"{col}_mean"]
                std = self._stats[f"{col}_std"]
                outliers = abs((df[col] - mean) / std) > self.outlier_threshold
                
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
                
                # Clip values to bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif self.outlier_method == 'winsorize':
                # Winsorize (clip to percentiles)
                lower_percentile = 1
                upper_percentile = 99
                lower_bound = np.percentile(df[col].dropna(), lower_percentile)
                upper_bound = np.percentile(df[col].dropna(), upper_percentile)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
        return df
    
    def _ensure_time_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure regular time intervals by resampling"""
        # Make sure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex to ensure time continuity")
        
        # For OHLC data, use specific resampling
        has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
        
        if has_ohlc:
            # Use proper OHLC resampling
            resampled = df.resample(self.resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            
            # Handle volume separately if it exists
            if self.volume_col in df.columns:
                volume_resampled = df[self.volume_col].resample(self.resample_rule).sum()
                resampled[self.volume_col] = volume_resampled
                
            # Merge any other columns with appropriate aggregation
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', self.volume_col]:
                    resampled[col] = df[col].resample(self.resample_rule).mean()
                    
            return resampled
        else:
            # For non-OHLC data, use forward fill
            return df.resample(self.resample_rule).ffill()
    
    def _ensure_ohlc_validity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC relationships are valid: High ≥ Open ≥ Close ≥ Low"""
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Fix high values that are too low
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            
            # Fix low values that are too high
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
        return df
    
    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate data quality metrics after cleaning"""
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
                
            # Count potential anomalies after cleaning
            if self.outlier_method != 'none':
                for col in price_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    potential_outliers = abs((df[col] - mean) / std) > self.outlier_threshold
                    metrics[f"{col}_potential_outliers"] = potential_outliers.sum()
        
        return metrics