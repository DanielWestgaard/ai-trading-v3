import pandas as pd
import numpy as np
from data.processors.base_processor import BaseProcessor


class FeatureGenerator(BaseProcessor):
    """Generates financial and technical features from OHLCV data"""
    
    def __init__(self, 
                 price_cols=['Open', 'High', 'Low', 'Close'],
                 volume_col='Volume',
                 timestamp_col='Date',
                 preserve_original_case=True):
        """
        Initialize feature generator with configuration.
        
        Args:
            price_cols: Column names for price data (OHLC)
            volume_col: Column name for volume data
            timestamp_col: Column name for timestamp
            preserve_original_case: Whether to preserve original column case
        """
        # Store original column names
        self.original_price_cols = price_cols
        self.original_volume_col = volume_col
        self.original_timestamp_col = timestamp_col
        self.preserve_original_case = preserve_original_case
        
        # Lowercase versions for internal consistency
        self.price_cols = [col.lower() for col in price_cols]
        self.volume_col = volume_col.lower()
        self.timestamp_col = timestamp_col.lower()
        
        # Column mapping for case preservation
        self.column_mapping = {}
        
    def fit(self, data):
        """Store necessary information about the data"""
        # Not much needed for fit in feature generation, but keeping for consistency
        if self.preserve_original_case:
            self.column_mapping = {col.lower(): col for col in data.columns}
        return self
        
    def transform(self, data):
        """Apply all feature generation to the data"""
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Standardize column names to lowercase for processing
        if self.preserve_original_case:
            self.column_mapping = {col.lower(): col for col in df.columns}
            df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have a proper datetime index for time-based features
        has_timestamp = self.timestamp_col in df.columns
        if has_timestamp:
            # Convert to datetime if not already
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            
            # Create a temporary datetime index
            df = df.set_index(self.timestamp_col)
        
        # Generate all features
        featured_df = self._add_technical_indicators(df)
        featured_df = self._add_volatility_metrics(featured_df)
        featured_df = self._add_price_patterns(featured_df)
        featured_df = self._add_time_features(featured_df)
        
        # Reset index to restore timestamp column
        if has_timestamp:
            featured_df = featured_df.reset_index()
        
        # Restore original column case if needed
        if self.preserve_original_case:
            # Create mapping for new columns (keep them lowercase)
            orig_cols = set(self.column_mapping.keys())
            for col in featured_df.columns:
                if col in orig_cols:
                    # Use original case for existing columns
                    featured_df = featured_df.rename(columns={col: self.column_mapping[col]})
        
        return featured_df
    
    def add_technical_indicators(self, data):
        """Public method to add only technical indicators"""
        df = data.copy()
        if self.preserve_original_case:
            df.columns = [col.lower() for col in df.columns]
        result = self._add_technical_indicators(df)
        
        # Restore original case
        if self.preserve_original_case:
            for col in self.column_mapping:
                if col in result.columns:
                    result = result.rename(columns={col: self.column_mapping[col]})
        
        return result
    
    def add_volatility_metrics(self, data):
        """Public method to add only volatility metrics"""
        df = data.copy()
        if self.preserve_original_case:
            df.columns = [col.lower() for col in df.columns]
        result = self._add_volatility_metrics(df)
        
        # Restore original case
        if self.preserve_original_case:
            for col in self.column_mapping:
                if col in result.columns:
                    result = result.rename(columns={col: self.column_mapping[col]})
        
        return result
    
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        result = df.copy()
        
        # Ensure we have the close price column
        if 'close' not in result.columns:
            return result
            
        close = result['close']
        
        # 1. Moving Averages
        for window in [5, 10, 20, 50, 200]:
            # Simple Moving Average (SMA)
            result[f'sma_{window}'] = close.rolling(window=window).mean()
            
            # Exponential Moving Average (EMA)
            result[f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()
        
        # 2. Relative Strength Index (RSI)
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # 4. Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        result['bollinger_upper'] = sma_20 + (std_20 * 2)
        result['bollinger_lower'] = sma_20 - (std_20 * 2)
        result['bollinger_middle'] = sma_20
        
        # 5. Price rate of change
        for window in [1, 5, 10, 20]:
            result[f'roc_{window}'] = close.pct_change(periods=window) * 100
        
        # 6. Stochastic Oscillator
        if all(col in result.columns for col in ['high', 'low']):
            high_14 = result['high'].rolling(window=14).max()
            low_14 = result['low'].rolling(window=14).min()
            result['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14))
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
        
        return result
    
    def _add_volatility_metrics(self, df):
        """Add volatility and risk metrics"""
        result = df.copy()
        
        # Ensure we have necessary price columns
        req_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in req_cols):
            return result
        
        # 1. Average True Range (ATR)
        high = result['high']
        low = result['low']
        close = result['close']
        
        tr1 = high - low  # Current high - current low
        tr2 = abs(high - close.shift())  # Current high - previous close
        tr3 = abs(low - close.shift())  # Current low - previous close
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr_14'] = true_range.rolling(window=14).mean()
        
        # 2. Historical Volatility (using standard deviation of returns)
        for window in [5, 10, 20, 30]:
            # Calculate daily returns
            returns = close.pct_change()
            # Annualized historical volatility (standard deviation of returns * sqrt(252))
            result[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # 3. Garman-Klass volatility estimator
        if 'open' in result.columns:
            open_price = result['open']
            # Garman-Klass volatility
            result['gk_volatility'] = 0.5 * np.log(high / low)**2 - (2*np.log(2)-1) * np.log(close / open_price)**2
            
        # 4. Normalized volatility (ATR divided by close price)
        result['normalized_atr'] = result['atr_14'] / close
        
        return result
    
    def _add_price_patterns(self, df):
        """Add price pattern recognition features"""
        result = df.copy()
        
        # Check for required columns
        if not all(col in result.columns for col in ['open', 'high', 'low', 'close']):
            return result
            
        open_price = result['open']
        high = result['high']
        low = result['low']
        close = result['close']
        
        # 1. Candlestick pattern indicators
        
        # Doji (open and close are almost equal)
        doji_threshold = 0.0005  # 0.05% threshold
        result['doji'] = (abs(open_price - close) / close) < doji_threshold
        
        # Hammer (lower shadow is at least twice the body)
        body = abs(close - open_price)
        lower_shadow = (open_price.clip(lower=close) - low)
        result['hammer'] = (lower_shadow >= 2 * body) & (body > 0)
        
        # Engulfing patterns
        prev_body = abs(open_price.shift() - close.shift())
        curr_body = abs(open_price - close)
        result['bullish_engulfing'] = (open_price < close) & (open_price.shift() > close.shift()) & (open_price <= close.shift()) & (close >= open_price.shift()) & (curr_body > prev_body)
        result['bearish_engulfing'] = (open_price > close) & (open_price.shift() < close.shift()) & (open_price >= close.shift()) & (close <= open_price.shift()) & (curr_body > prev_body)
        
        # 2. Support and resistance indicators
        for window in [10, 20]:
            # Local highs and lows
            result[f'resistance_{window}'] = high.rolling(window=window).max()
            result[f'support_{window}'] = low.rolling(window=window).min()
            
            # Distance to support/resistance
            result[f'dist_to_resistance_{window}'] = (result[f'resistance_{window}'] - close) / close
            result[f'dist_to_support_{window}'] = (close - result[f'support_{window}']) / close
        
        # 3. Price acceleration (change in rate of change)
        result['price_accel'] = close.pct_change().diff()
        
        return result
    
    def _add_time_features(self, df):
        """Add time-based features if we have a datetime index"""
        result = df.copy()
        
        # Check if we have a datetime index
        if not isinstance(result.index, pd.DatetimeIndex):
            return result
            
        # 1. Time-based features
        result['day_of_week'] = result.index.dayofweek
        result['hour_of_day'] = result.index.hour
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        
        # 2. Market session indicators (approximations)
        # Asia: 00:00-08:00 UTC, Europe: 08:00-16:00 UTC, US: 13:00-21:00 UTC
        hour = result.index.hour
        result['asian_session'] = (hour >= 0) & (hour < 8)
        result['european_session'] = (hour >= 8) & (hour < 16)
        result['us_session'] = (hour >= 13) & (hour < 21)
        result['market_overlap'] = ((hour >= 8) & (hour < 13)) | ((hour >= 13) & (hour < 16)) 
        
        # 3. Day of week indicators
        for day in range(5):  # 0-4 for Monday-Friday
            result[f'day_{day}'] = result.index.dayofweek == day
            
        return result