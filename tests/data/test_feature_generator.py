import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

from data.features.feature_generator import FeatureGenerator


class TestFeatureGenerator(unittest.TestCase):
    """Test suite for the FeatureGenerator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample OHLCV data for testing
        dates = pd.date_range(start='2023-01-01', periods=250, freq='1D')
        
        np.random.seed(42)  # For reproducibility
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 5, 250).cumsum(),
            'High': np.random.normal(102, 5, 250).cumsum(),
            'Low': np.random.normal(98, 5, 250).cumsum(),
            'Close': np.random.normal(101, 5, 250).cumsum(),
            'Volume': np.random.randint(1000, 5000, 250)
        })
        
        # Ensure High is always >= Open, Close and Low is always <= Open, Close
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.loc[i, 'Open'], self.sample_data.loc[i, 'Close'], self.sample_data.loc[i, 'High'])
            low = min(self.sample_data.loc[i, 'Open'], self.sample_data.loc[i, 'Close'], self.sample_data.loc[i, 'Low'])
            self.sample_data.loc[i, 'High'] = high
            self.sample_data.loc[i, 'Low'] = low
        
        # Create hourly data with different column names
        hourly_dates = pd.date_range(start='2023-01-01', periods=500, freq='1h')
        self.hourly_data = pd.DataFrame({
            'timestamp': hourly_dates,
            'price_open': np.random.normal(100, 2, 500).cumsum(),
            'price_high': np.random.normal(101, 2, 500).cumsum(),
            'price_low': np.random.normal(99, 2, 500).cumsum(),
            'price_close': np.random.normal(100.5, 2, 500).cumsum(),
            'volume': np.random.randint(100, 500, 500)
        })
        
        # Create a default generator instance
        self.generator = FeatureGenerator()
    
    def test_initialization(self):
        """Test that the generator initializes with correct parameters"""
        # Test default initialization
        generator = FeatureGenerator()
        self.assertEqual(generator.price_cols, ['open', 'high', 'low', 'close'])
        self.assertEqual(generator.volume_col, 'volume')
        self.assertEqual(generator.timestamp_col, 'date')
        self.assertTrue(generator.preserve_original_case)
        
        # Test custom initialization
        custom_generator = FeatureGenerator(
            price_cols=['price_open', 'price_high', 'price_low', 'price_close'],
            volume_col='vol',
            timestamp_col='time',
            preserve_original_case=False
        )
        self.assertEqual(custom_generator.price_cols, ['price_open', 'price_high', 'price_low', 'price_close'])
        self.assertEqual(custom_generator.volume_col, 'vol')
        self.assertEqual(custom_generator.timestamp_col, 'time')
        self.assertFalse(custom_generator.preserve_original_case)
    
    def test_case_preservation(self):
        """Test that original column case is preserved when requested"""
        # With case preservation (default)
        generator_preserve = FeatureGenerator(preserve_original_case=True)
        result_preserve = generator_preserve.transform(self.sample_data)
        
        # Check that original column names are preserved
        for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertIn(col, result_preserve.columns)
        
        # Without case preservation
        generator_no_preserve = FeatureGenerator(preserve_original_case=False)
        result_no_preserve = generator_no_preserve.transform(self.sample_data)
        
        # Check that columns are lowercase
        for col in ['date', 'open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, result_no_preserve.columns)
    
    def test_technical_indicators(self):
        """Test generation of technical indicators"""
        # Generate technical indicators only
        tech_indicators = self.generator.add_technical_indicators(self.sample_data)
        
        # Check that basic indicators were created
        expected_indicators = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bollinger_upper', 'bollinger_lower', 'bollinger_middle',
            'roc_1', 'roc_5', 'roc_10', 'roc_20',
            'stoch_k', 'stoch_d'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, tech_indicators.columns)
        
        # Test specific indicators
        # Check SMA calculation - 5-day simple moving average
        expected_sma5 = self.sample_data['Close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(
            tech_indicators['sma_5'],
            expected_sma5,
            check_names=False
        )
        
        # Check RSI has valid values
        rsi = tech_indicators['rsi_14'].dropna()
        self.assertTrue(all(rsi >= 0) and all(rsi <= 100))
    
    def test_volatility_metrics(self):
        """Test generation of volatility metrics"""
        # Generate volatility metrics only
        volatility_metrics = self.generator.add_volatility_metrics(self.sample_data)
        
        # Check that volatility metrics were created
        expected_metrics = [
            'atr_14', 'volatility_5', 'volatility_10', 'volatility_20', 'volatility_30',
            'gk_volatility', 'normalized_atr'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, volatility_metrics.columns)
        
        # Verify ATR calculation
        atr = volatility_metrics['atr_14'].dropna()
        self.assertTrue(all(atr >= 0))  # ATR should always be positive
    
    def test_price_patterns(self):
        """Test generation of price pattern features"""
        # Generate all features
        result = self.generator.transform(self.sample_data)
        
        # Check that price pattern features were created
        expected_patterns = [
            'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing',
            'resistance_10', 'support_10', 'dist_to_resistance_10', 'dist_to_support_10',
            'resistance_20', 'support_20', 'dist_to_resistance_20', 'dist_to_support_20',
            'price_accel'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, result.columns)
        
        # Check that binary pattern indicators are boolean
        for pattern in ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing']:
            self.assertTrue(result[pattern].isin([True, False, np.nan]).all())
    
    def test_time_features(self):
        """Test generation of time-based features"""
        # Use the transform method which will generate all features
        result = self.generator.transform(self.sample_data)
        
        # Check that time features were created
        expected_time_features = [
            'day_of_week', 'hour_of_day', 'month', 'quarter',
            'asian_session', 'european_session', 'us_session', 'market_overlap'
        ]
        
        for feature in expected_time_features:
            self.assertIn(feature, result.columns)
        
        # Verify day_of_week is between 0-6
        self.assertTrue(result['day_of_week'].dropna().between(0, 6).all())
        
        # Verify hour_of_day is between 0-23
        self.assertTrue(result['hour_of_day'].dropna().between(0, 23).all())
    
    def test_transform_all_features(self):
        """Test the generation of all features at once"""
        all_features = self.generator.transform(self.sample_data)
        
        # Basic check: we should have more columns than the original data
        self.assertGreater(len(all_features.columns), len(self.sample_data.columns))
        
        # Check for categories of features
        tech_indicators = ['sma_5', 'rsi_14', 'macd']
        volatility_metrics = ['atr_14', 'volatility_10']
        price_patterns = ['doji', 'hammer', 'resistance_10']
        time_features = ['day_of_week', 'month', 'asian_session']
        
        for feature in tech_indicators + volatility_metrics + price_patterns + time_features:
            self.assertIn(feature, all_features.columns)
        
        # Check that at least a certain percentage of data is non-NaN
        # Most features should be calculable after a few initial values
        non_nan_percentage = all_features.iloc[50:].notna().mean().mean()  # Skip first 50 rows
        self.assertGreater(non_nan_percentage, 0.9)  # At least 90% of data should be non-NaN
    
    def test_custom_column_names(self):
        """Test feature generation with custom column names"""
        # Create generator with custom column names
        custom_generator = FeatureGenerator(
            price_cols=['price_open', 'price_high', 'price_low', 'price_close'],
            volume_col='volume',
            timestamp_col='timestamp',
            preserve_original_case=True
        )
        
        # Generate features for hourly data
        result = custom_generator.transform(self.hourly_data)
        
        # Check that features were created despite different column names
        expected_features = [
            'sma_5', 'ema_10', 'rsi_14', 'macd',
            'atr_14', 'volatility_10',
            'doji', 'hammer',
            'day_of_week', 'hour_of_day'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns)
    
    def test_missing_columns(self):
        """Test behavior when some expected columns are missing"""
        # Create data with missing columns
        incomplete_data = self.sample_data.drop(columns=['High', 'Low'])
        
        # Should not raise an error
        try:
            result = self.generator.transform(incomplete_data)
            # Features that depend on missing columns should not be present
            self.assertNotIn('stoch_k', result.columns)
            self.assertNotIn('atr_14', result.columns)
        except Exception as e:
            self.fail(f"transform() raised {type(e).__name__} unexpectedly!")
    
    def test_fit_transform(self):
        """Test the fit_transform convenience method"""
        # Should be equivalent to calling fit() then transform()
        result = self.generator.fit_transform(self.sample_data)
        
        # Verify it contains the expected features
        expected_features = [
            'sma_5', 'ema_10', 'rsi_14', 'macd',
            'atr_14', 'volatility_10',
            'doji', 'hammer',
            'day_of_week', 'hour_of_day'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, result.columns)


if __name__ == '__main__':
    unittest.main()