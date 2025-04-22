import pandas as pd
import numpy as np
from data.processors.base_processor import BaseProcessor
import logging


class FeaturePreparator(BaseProcessor):
    """
    Prepares features for model training with advanced handling of financial data characteristics.
    
    This component sits between feature generation and normalization in the pipeline,
    handling the complexities of financial feature preparation including window-based features,
    price transformations, and feature categorization.
    """
    
    def __init__(self, 
                 price_cols=['Open', 'High', 'Low', 'Close'],
                 volume_col='Volume',
                 timestamp_col='Date',
                 preserve_original_prices=True,
                 price_transform_method='returns',
                 trim_initial_periods=None,  # None = auto detect
                 min_data_points=1000,
                 feature_category_rules=None,
                 treatment_mode='advanced'):
        """
        Initialize the feature preparator.
        
        Args:
            price_cols: List of price column names
            volume_col: Volume column name
            timestamp_col: Timestamp column name
            preserve_original_prices: Whether to keep original prices alongside transforms
            price_transform_method: How to transform prices ('returns', 'log', 'pct_change', 'none')
            trim_initial_periods: Number of initial periods to trim (None = auto detect)
            min_data_points: Minimum required data points after preparation
            feature_category_rules: Custom rules for feature categorization
            treatment_mode: 'basic', 'advanced', or 'hybrid'
        """
        self.price_cols = [col.lower() for col in price_cols]
        self.volume_col = volume_col.lower()
        self.timestamp_col = timestamp_col.lower()
        self.preserve_original_prices = preserve_original_prices
        self.price_transform_method = price_transform_method
        self.trim_initial_periods = trim_initial_periods
        self.min_data_points = min_data_points
        self.treatment_mode = treatment_mode
        
        # Default feature category rules if none provided
        self.feature_category_rules = feature_category_rules or {
            'price': ['open', 'high', 'low', 'close'],
            'volume': ['volume'],
            'short_window': ['sma_5', 'sma_10', 'ema_5', 'ema_10', 'roc_1', 'roc_5'],
            'medium_window': ['sma_20', 'ema_20', 'rsi_14', 'macd', 'stoch_k', 'stoch_d', 'atr_14', 'volatility_5', 'volatility_10'],
            'long_window': ['sma_50', 'sma_200', 'ema_50', 'ema_200', 'volatility_20', 'volatility_30'],
            'categorical': ['day_of_week', 'hour_of_day', 'month', 'quarter', 'asian_session', 'european_session', 'us_session', 'day_'],
            'returns': ['_return', 'pct_change']
        }
        
        self._feature_categories = {}
        self._window_sizes = {}
        self._stats = {}
    
    def fit(self, data):
        """
        Analyze data to determine optimal feature preparation strategies.
        
        Args:
            data: DataFrame with features
        """
        df = data.copy()
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate statistics for each feature
        self._calculate_feature_stats(df)
        
        # Categorize features
        self._categorize_features(df.columns)
        
        # Detect window sizes based on NaN patterns if not manually specified
        self._detect_window_sizes(df)
        
        logging.info(f"Feature preparation strategy fitted. "
                    f"Found {len(self._feature_categories.get('short_window', []))} short window, "
                    f"{len(self._feature_categories.get('medium_window', []))} medium window, and "
                    f"{len(self._feature_categories.get('long_window', []))} long window features.")
        
        return self
    
    def transform(self, data):
        """
        Apply feature preparation based on the fitted strategy.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Prepared DataFrame ready for normalization and modeling
        """
        df = data.copy()
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Step 1: Transform prices if requested
        if self.price_transform_method != 'none':
            df = self._transform_prices(df)
        
        # Step 2: Handle features based on their category and treatment mode
        if self.treatment_mode == 'basic':  # Traditional approach (trim initial periods)
            # Basic mode: trim initial periods with NaNs
            df = self._basic_treatment(df)
        elif self.treatment_mode == 'advanced':
            # Advanced mode: customize treatment by feature category
            df = self._advanced_treatment(df)
        else:  # hybrid mode
            # Hybrid mode: balance between data retention and validity
            df = self._hybrid_treatment(df)
        
        # Step 3: Check if we have enough data left
        if len(df) < self.min_data_points:
            logging.warning(f"After preparation, only {len(df)} data points remain, "
                           f"which is less than the minimum required ({self.min_data_points}). "
                           f"Consider using a different treatment mode.")
        
        # Step 4: Final check for any remaining NaNs
        missing_counts = df.isna().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if not columns_with_missing.empty:
            logging.warning(f"There are still {len(columns_with_missing)} columns with missing values.")
            for col in columns_with_missing.index:
                logging.warning(f"  - {col}: {missing_counts[col]} missing values")
            
            # NEW: Force-fill any remaining missing values
            df = self._ensure_no_missing_values(df)
        
        return df
    
    def _ensure_no_missing_values(self, df):
        """Make sure there are absolutely no missing values in the final result"""
        result = df.copy()
        
        # 1. Handle numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Try forward fill first
            result[col] = result[col].ffill()
            
            # If still has NaNs, try backward fill
            if result[col].isna().any():
                result[col] = result[col].bfill()
                
            # If STILL has NaNs, fill with column mean or 0
            if result[col].isna().any():
                if result[col].count() > 0:  # If we have at least some non-NaN values
                    result[col] = result[col].fillna(result[col].mean())
                else:
                    # No valid values at all, fill with 0
                    result[col] = result[col].fillna(0)
        
        # 2. Handle non-numeric columns
        non_numeric_cols = result.select_dtypes(exclude=['number']).columns
        for col in non_numeric_cols:
            # Forward fill for non-numeric
            result[col] = result[col].ffill()
            
            # If still has NaNs, backward fill
            if result[col].isna().any():
                result[col] = result[col].bfill()
                
            # If STILL has NaNs, fill with most common value or a placeholder
            if result[col].isna().any():
                if result[col].count() > 0:  # If we have at least some non-NaN values
                    most_common = result[col].mode()[0]
                    result[col] = result[col].fillna(most_common)
                else:
                    # No valid values at all, fill with "UNKNOWN"
                    result[col] = result[col].fillna("UNKNOWN")
        
        # Verify all NaNs are gone
        remaining_nans = result.isna().sum().sum()
        if remaining_nans > 0:
            logging.warning(f"Failed to remove all NaNs. {remaining_nans} still remain. Using more aggressive method.")
            # Last resort - drop columns with NaNs
            cols_with_nans = result.columns[result.isna().any()].tolist()
            logging.warning(f"Dropping columns with remaining NaNs: {cols_with_nans}")
            result = result.drop(columns=cols_with_nans)
        
        return result
    
    def _calculate_feature_stats(self, df):
        """Calculate statistics for features to guide preparation"""
        # Store basic stats for each column
        for col in df.columns:
            non_nan_values = df[col].dropna()
            if len(non_nan_values) > 0:
                self._stats[f"{col}_first_valid_idx"] = df[col].first_valid_index()
                self._stats[f"{col}_nan_count"] = df[col].isna().sum()
                self._stats[f"{col}_nan_pct"] = (df[col].isna().sum() / len(df)) * 100
    
    def _categorize_features(self, columns):
        """Categorize features based on their names and characteristics"""
        # Initialize categories
        for category in self.feature_category_rules:
            self._feature_categories[category] = []
        
        # Categorize each column
        for col in columns:
            col_lower = col.lower()
            categorized = False
            
            # First try exact matches (for specific features like 'sma_50')
            for category, patterns in self.feature_category_rules.items():
                if col_lower in patterns:
                    self._feature_categories[category].append(col_lower)
                    categorized = True
                    break
            
            if not categorized:
                # Then try pattern matching for more general patterns
                for category, patterns in self.feature_category_rules.items():
                    # Use pattern matching, but be more careful with numeric suffixes
                    for pattern in patterns:
                        # If pattern doesn't contain underscore (like 'open', 'volume'), match exactly
                        if '_' not in pattern and pattern == col_lower:
                            self._feature_categories[category].append(col_lower)
                            categorized = True
                            break
                        # If pattern has underscore (like 'sma_', 'volatility_'), ensure it's a prefix match
                        elif '_' in pattern and col_lower.startswith(pattern):
                            self._feature_categories[category].append(col_lower)
                            categorized = True
                            break
                        # Generic substring matching for other patterns
                        elif pattern in col_lower:
                            self._feature_categories[category].append(col_lower)
                            categorized = True
                            break
                    
                    if categorized:
                        break
            
            # Handle any uncategorized columns
            if not categorized:
                if 'other' not in self._feature_categories:
                    self._feature_categories['other'] = []
                self._feature_categories['other'].append(col_lower)
    
    def _detect_window_sizes(self, df):
        """
        Auto-detect window sizes based on NaN patterns at the beginning of each feature
        """
        for window_type in ['short_window', 'medium_window', 'long_window']:
            if window_type in self._feature_categories:
                for col in self._feature_categories[window_type]:
                    if col in df.columns:
                        # Find the index of the first non-NaN value
                        first_valid_idx = df[col].first_valid_index()
                        if first_valid_idx is not None:
                            # Calculate the window size based on the first valid index
                            # Add 1 to convert from 0-based index to window size
                            if isinstance(first_valid_idx, int):
                                window_size = first_valid_idx + 1
                            else:
                                window_size = df.index.get_loc(first_valid_idx) + 1
                            self._window_sizes[col] = window_size
        
        # Calculate the maximum window size for each category
        for window_type in ['short_window', 'medium_window', 'long_window']:
            if window_type in self._feature_categories:
                category_cols = [col for col in self._feature_categories[window_type] if col in self._window_sizes]
                if category_cols:
                    self._window_sizes[f"max_{window_type}"] = max([self._window_sizes[col] for col in category_cols])
                else:
                    # Default values if detection failed
                    default_sizes = {'short_window': 10, 'medium_window': 20, 'long_window': 200}
                    self._window_sizes[f"max_{window_type}"] = default_sizes[window_type]
    
    def _transform_prices(self, df):
        """Transform price columns while preserving original values"""
        result = df.copy()
        
        # Convert result column names to lowercase for consistent processing
        column_case_map = {col.lower(): col for col in result.columns}
        result.columns = [col.lower() for col in result.columns]
        
        # Ensure we have price columns - use lowercase matching
        price_cols_in_data = []
        for price_col in self.price_cols:
            # Check if the column exists directly
            if price_col in result.columns:
                price_cols_in_data.append(price_col)
                continue
                
            # Look for price columns using pattern matching if exact match failed
            for col in result.columns:
                if price_col in col:
                    price_cols_in_data.append(col)
                    break
        
        if not price_cols_in_data:
            # Restore original column case and return if no price columns found
            result.columns = [column_case_map.get(col, col) for col in result.columns]
            return result
        
        # Preserve raw prices: These will never be normalized or modified
        for col in price_cols_in_data:
            result[f"{col}_raw"] = result[col].copy()
        
        # Make copies of original price columns with _original suffix
        for col in price_cols_in_data:
            result[f"{col}_original"] = result[col].copy()
        
        # Apply the specified transformation
        if self.price_transform_method == 'returns':
            # Calculate percentage returns
            for col in price_cols_in_data:
                # Explicitly specify fill_method=None to address the FutureWarning
                result[f"{col}_return"] = result[col].pct_change(fill_method=None)
                    
        elif self.price_transform_method == 'log':
            # Calculate log transformation
            for col in price_cols_in_data:
                # Add small constant to avoid log(0) or log(negative)
                min_val = result[col].min()
                offset = 0 if min_val > 0 else abs(min_val) + 1e-8
                # Handle possible negatives or zeros
                safe_values = result[col] + offset
                # Replace infinities or NaNs from log with NaN
                result[f"{col}_log"] = np.log(safe_values)
                    
        elif self.price_transform_method == 'pct_change':
            # Calculate percentage change from first value
            for col in price_cols_in_data:
                first_value = result[col].iloc[0]
                if first_value != 0:
                    result[f"{col}_pct_change"] = (result[col] / first_value - 1) * 100
                else:
                    result[f"{col}_pct_change"] = 0
                        
        elif self.price_transform_method == 'multi':
            # Apply multiple transformations for comparison
            for col in price_cols_in_data:
                # Returns
                result[f"{col}_return"] = result[col].pct_change(fill_method=None)
                    
                # Log transform
                min_val = result[col].min()
                offset = 0 if min_val > 0 else abs(min_val) + 1e-8
                # Handle possible negatives or zeros
                safe_values = result[col] + offset
                result[f"{col}_log"] = np.log(safe_values)
        
        # Original price columns are always preserved
        logging.info(f"Preserving original OHLC values for empirical feature selection")
        
        # Keep columns in lowercase without restoring original case
        # This ensures consistent column naming for downstream processing
        
        return result
    
    def _basic_treatment(self, df):
        """
        Basic treatment: handle missing values with a more resilient approach
        that preserves more data than simple trimming or dropping.
        """
        # Determine the maximum window size
        max_window_size = max(
            self._window_sizes.get('max_short_window', 10),
            self._window_sizes.get('max_medium_window', 20),
            self._window_sizes.get('max_long_window', 200)
        )
        
        # Use manually specified trim size if provided
        trim_size = self.trim_initial_periods or max_window_size
        
        # Trim the data if we have enough
        if len(df) > trim_size + self.min_data_points:
            logging.info(f"Trimming {trim_size} initial periods with potential NaN values.")
            result = df.iloc[trim_size:].reset_index(drop=True)
            
            # ENHANCED: Check if we still have NaNs after trimming and handle them
            if result.isna().sum().sum() > 0:
                logging.info(f"Still found NaNs after trimming. Applying additional filling.")
                result = self._fill_remaining_nans(result)
        else:
            # IMPROVED APPROACH: Instead of simple dropna(), use a more targeted strategy
            logging.warning(f"Not enough data to trim {trim_size} periods safely. "
                        f"Using targeted NaN handling instead.")
            
            result = self._fill_missing_values_by_type(df)
        
        return result
    
    def _fill_missing_values_by_type(self, df):
        """Enhanced method for handling missing values based on column type"""
        result = df.copy()
        
        # 1. Handle price columns - always critical
        price_cols = []
        for col in result.columns:
            if any(price in col.lower() for price in self.price_cols):
                price_cols.append(col)
                
        if price_cols:
            # Forward fill then backward fill for price data
            result[price_cols] = result[price_cols].ffill().bfill()
        
        # 2. Handle volume column
        volume_cols = [col for col in result.columns if self.volume_col in col.lower()]
        if volume_cols:
            # Fill missing volume with nearest values or zeros
            result[volume_cols] = result[volume_cols].ffill().bfill()
            result[volume_cols] = result[volume_cols].fillna(0)  # Any remaining NaNs become 0
        
        # 3. Handle technical indicators by category
        # Different handling for different types of indicators
        
        # Moving averages
        ma_cols = [col for col in result.columns if any(x in col.lower() for x in ['sma', 'ema'])]
        if ma_cols:
            # First try to fill forward (most accurate for time series)
            result[ma_cols] = result[ma_cols].ffill()
            
            # For any remaining NaNs, try to fill by taking average of price data
            for col in ma_cols:
                # Extract the window size from column name if possible (e.g., sma_20 -> 20)
                parts = col.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    window = int(parts[1])
                    
                    # Find the corresponding price column
                    price_type = 'close'  # Default
                    if 'open' in col:
                        price_type = 'open'
                    elif 'high' in col:
                        price_type = 'high'
                    elif 'low' in col:
                        price_type = 'low'
                    
                    # Find the actual price column in the data
                    price_col = None
                    for pcol in result.columns:
                        if price_type in pcol.lower() and 'raw' in pcol.lower():
                            price_col = pcol
                            break
                    
                    # If we found the price column, estimate the MA
                    if price_col is not None:
                        # Calculate rolling mean without NaN propagation
                        rolling_mean = result[price_col].rolling(window, min_periods=1).mean()
                        
                        # Fill NaN values in the indicator with the calculated values
                        mask = result[col].isna()
                        result.loc[mask, col] = rolling_mean[mask]
            
            # Any remaining NaNs, fill by backward fill and interpolation
            result[ma_cols] = result[ma_cols].bfill().interpolate(method='linear', limit_direction='both')
        
        # Oscillators (RSI, MACD, etc.)
        osc_cols = [col for col in result.columns if any(x in col.lower() for x in ['rsi', 'macd', 'stoch'])]
        if osc_cols:
            # First fill with ffill/bfill
            result[osc_cols] = result[osc_cols].ffill().bfill()
            
            # For RSI specifically, fill remaining NaNs with neutral values
            rsi_cols = [col for col in osc_cols if 'rsi' in col.lower()]
            for col in rsi_cols:
                result[col] = result[col].fillna(50)  # Neutral value for RSI
                
            # For other oscillators, fill with column mean
            for col in [c for c in osc_cols if c not in rsi_cols]:
                if result[col].isna().any():
                    if result[col].count() > 0:  # If we have non-NaN values
                        result[col] = result[col].fillna(result[col].mean())
                    else:
                        result[col] = result[col].fillna(0)  # No data at all
        
        # Volatility metrics
        vol_cols = [col for col in result.columns if any(x in col.lower() for x in ['atr', 'volatility', 'bollinger'])]
        if vol_cols:
            # Fill with ffill/bfill first
            result[vol_cols] = result[vol_cols].ffill().bfill()
            
            # Bollinger bands specifically
            bollinger_cols = [col for col in vol_cols if 'bollinger' in col.lower()]
            if bollinger_cols:
                # Find the price column
                price_col = None
                for pcol in result.columns:
                    if 'close' in pcol.lower() and 'raw' in pcol.lower():
                        price_col = pcol
                        break
                
                if price_col is not None:
                    # If middle band is missing, use price
                    middle_cols = [col for col in bollinger_cols if 'middle' in col.lower()]
                    for col in middle_cols:
                        mask = result[col].isna()
                        result.loc[mask, col] = result.loc[mask, price_col]
                    
                    # If upper/lower bands are missing, estimate from price and middle
                    for col in [c for c in bollinger_cols if c not in middle_cols]:
                        if 'upper' in col.lower():
                            mask = result[col].isna()
                            # Estimate upper as price + 5%
                            if any(mask):
                                middle_col = [c for c in middle_cols if c in result.columns][0] if middle_cols else price_col
                                result.loc[mask, col] = result.loc[mask, middle_col] * 1.05
                        elif 'lower' in col.lower():
                            mask = result[col].isna()
                            # Estimate lower as price - 5%
                            if any(mask):
                                middle_col = [c for c in middle_cols if c in result.columns][0] if middle_cols else price_col
                                result.loc[mask, col] = result.loc[mask, middle_col] * 0.95
            
            # For remaining volatility measures, use average volatility or small value
            for col in vol_cols:
                if result[col].isna().any():
                    if result[col].count() > 0:
                        avg_vol = result[col].mean()
                        result[col] = result[col].fillna(avg_vol)
                    else:
                        # Use a small positive value as fallback
                        result[col] = result[col].fillna(0.01)
        
        # 4. Handle any NaNs in derived price columns (returns, etc.)
        derived_price_cols = []
        for col in result.columns:
            if any(s in col.lower() for s in ['_return', '_log', '_pct_change']):
                derived_price_cols.append(col)
        
        if derived_price_cols:
            # First values are often NaN for returns
            result[derived_price_cols] = result[derived_price_cols].fillna(0)
        
        # 5. Handle pattern and categorical features
        pattern_cols = [col for col in result.columns if any(x in col.lower() for x in 
                                                          ['doji', 'hammer', 'engulfing', 'day_', 'hour_', 'month'])]
        if pattern_cols:
            # Fill categorical with most frequent value, boolean with False
            for col in pattern_cols:
                if result[col].dtype == bool:
                    result[col] = result[col].fillna(False)
                else:
                    # For categorical, use most frequent
                    if result[col].count() > 0:
                        most_frequent = result[col].mode()[0]
                        result[col] = result[col].fillna(most_frequent)
                    else:
                        # If no data, use a placeholder
                        result[col] = result[col].fillna("Unknown")
        
        # Final check - any remaining NaNs
        if result.isna().sum().sum() > 0:
            # Last resort interpolation for any numeric columns with remaining NaNs
            numeric_cols = result.select_dtypes(include=['number']).columns
            result[numeric_cols] = result[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            # Forward/backward fill for any non-numeric
            non_numeric_cols = result.select_dtypes(exclude=['number']).columns
            result[non_numeric_cols] = result[non_numeric_cols].ffill().bfill()
        
        return result
    
    def _fill_remaining_nans(self, df):
        """Fill any remaining NaNs after initial processing"""
        result = df.copy()
        
        # Fill any remaining NaNs in numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            # First try interpolation
            result[numeric_cols] = result[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            # Then try forward/backward fill for any remaining
            result[numeric_cols] = result[numeric_cols].ffill().bfill()
            
            # If still any NaNs, use column mean or 0
            for col in numeric_cols:
                if result[col].isna().any():
                    if result[col].count() > 0:
                        result[col] = result[col].fillna(result[col].mean())
                    else:
                        result[col] = result[col].fillna(0)
        
        # Handle non-numeric columns
        non_numeric_cols = result.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            # Forward/backward fill
            result[non_numeric_cols] = result[non_numeric_cols].ffill().bfill()
            
            # Fill remaining with most frequent or a placeholder
            for col in non_numeric_cols:
                if result[col].isna().any():
                    if result[col].count() > 0:
                        most_frequent = result[col].mode()[0]
                        result[col] = result[col].fillna(most_frequent)
                    else:
                        result[col] = result[col].fillna("UNKNOWN")
        
        return result
    
    def _advanced_treatment(self, df):
        """
        Advanced treatment: handle each feature category differently.
        Advanced Mode: Category-specific treatment
            Short-window features: Backfill for maximum retention
            Medium-window features: Partial backfill with statistical validity
            Long-window features: Strict trim for mathematical soundness
            Return values: Zero-filling for first row
            Original prices: Optional preservation alongside transforms
        """
        result = df.copy()
        
        # Handle returns - fill first NaN with 0
        if 'returns' in self._feature_categories:
            for col in self._feature_categories['returns']:
                if col in result.columns:
                    result[col] = result[col].fillna(0)
        
        # Handle categorical features - any NaNs should be filled with the most frequent value
        if 'categorical' in self._feature_categories:
            for col in self._feature_categories['categorical']:
                if col in result.columns and result[col].isna().any():
                    most_frequent = result[col].mode()[0]
                    result[col] = result[col].fillna(most_frequent)
        
        # Handle technical indicators
        
        # Short window features - backfill
        if 'short_window' in self._feature_categories:
            short_cols = [col for col in self._feature_categories['short_window'] if col in result.columns]
            result[short_cols] = result[short_cols].bfill().ffill()
        
        # Medium window features - only backfill if we have enough data
        if 'medium_window' in self._feature_categories:
            medium_cols = [col for col in self._feature_categories['medium_window'] if col in result.columns]
            if len(medium_cols) > 0:
                # Get maximum medium window size
                medium_window = self._window_sizes.get('max_medium_window', 20)
                
                # Only backfill after the window size
                if len(result) > medium_window + self.min_data_points:
                    # Split the data
                    init_data = result.iloc[:medium_window].copy()
                    main_data = result.iloc[medium_window:].copy()
                    
                    # Apply backfill to main data
                    main_data[medium_cols] = main_data[medium_cols].bfill()
                    
                    # Recombine
                    result = pd.concat([init_data, main_data], axis=0)
                else:
                    # Not enough data, use enhanced missing value handling
                    result[medium_cols] = self._fill_remaining_nans(result[medium_cols])
        
        # Long window features - preserve NaNs and trim
        if 'long_window' in self._feature_categories:
            long_cols = [col for col in self._feature_categories['long_window'] if col in result.columns]
            if len(long_cols) > 0:
                # Get maximum long window size
                long_window = self._window_sizes.get('max_long_window', 200)
                
                # Only keep rows where long window features are available
                if len(result) > long_window + self.min_data_points:
                    result = result.iloc[long_window:].reset_index(drop=True)
                else:
                    # Not enough data, use extended missing value handling
                    result[long_cols] = self._fill_missing_values_by_type(result[long_cols])
        
        # Final check for any remaining NaNs
        if result.isna().sum().sum() > 0:
            # Apply the enhanced nan filling
            result = self._fill_remaining_nans(result)
        
        return result.reset_index(drop=True)
    
    def _hybrid_treatment(self, df):
        """
        Hybrid treatment: balance between data retention and statistical validity.
        Hybrid Mode: Context-aware approach
            For small datasets: Maximizes data retention
            For medium datasets: Balances retention with validity
            For large datasets: Prioritizes statistical validity
        """
        result = df.copy()
        
        # 1. For very small datasets, preserve as much data as possible
        if len(df) < self.min_data_points * 1.5:
            # Handle all features to maximize data retention
            result = self._fill_missing_values_by_type(result)
            return result
        
        # 2. For medium-sized datasets, use a compromise approach
        if len(df) < self.min_data_points * 3:
            # Handle short and medium window features, drop long if needed
            short_cols = []
            medium_cols = []
            long_cols = []
            
            if 'short_window' in self._feature_categories:
                short_cols = [col for col in self._feature_categories['short_window'] if col in result.columns]
                result[short_cols] = result[short_cols].bfill().ffill()
            
            if 'medium_window' in self._feature_categories:
                medium_cols = [col for col in self._feature_categories['medium_window'] if col in result.columns]
                medium_window = self._window_sizes.get('max_medium_window', 20)
                if medium_window < len(result) // 3:
                    result[medium_cols] = result[medium_cols].bfill().ffill()
            
            if 'long_window' in self._feature_categories:
                long_cols = [col for col in self._feature_categories['long_window'] if col in result.columns]
                # Drop long window features if they would cause too much data loss
                if result[long_cols].isna().any(axis=1).sum() > len(result) // 3:
                    logging.warning(f"Dropped {len(long_cols)} long window features to preserve sufficient data.")
                    result = result.drop(columns=long_cols)
                    # Mark these columns as dropped
                    long_cols = []
            
            # Handle returns and categoricals
            if 'returns' in self._feature_categories:
                returns_cols = [col for col in self._feature_categories['returns'] if col in result.columns]
                result[returns_cols] = result[returns_cols].fillna(0)
            
            if 'categorical' in self._feature_categories:
                cat_cols = [col for col in self._feature_categories['categorical'] if col in result.columns]
                for col in cat_cols:
                    if result[col].isna().any():
                        most_frequent = result[col].mode()[0]
                        result[col] = result[col].fillna(most_frequent)
            
            # Handle any NaNs in columns we're keeping
            cols_to_keep = short_cols + medium_cols + long_cols
            if cols_to_keep:
                result[cols_to_keep] = self._fill_remaining_nans(result[cols_to_keep])
            
            # Finally, check for any remaining NaNs in the dataset
            if result.isna().sum().sum() > 0:
                result = self._fill_remaining_nans(result)
                
            return result
        
        # 3. For large datasets, use a modified advanced treatment with extra NaN checks
        result = self._advanced_treatment(df)
        
        # Add a final check to catch any remaining NaNs
        if result.isna().sum().sum() > 0:
            result = self._fill_remaining_nans(result)
            
        return result