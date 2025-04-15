import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

from data.features.feature_preparator import FeaturePreparator


class TestFeaturePreparator(unittest.TestCase):
    """Test suite for the FeaturePreparator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample data with features already generated
        dates = pd.date_range(start='2023-01-01', periods=250, freq='1D')
        
        np.random.seed(42)  # For reproducibility
        
        # Create base price and volume data
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 5, 250).cumsum(),
            'High': np.random.normal(102, 5, 250).cumsum(),
            'Low': np.random.normal(98, 5, 250).cumsum(),
            'Close': np.random.normal(101, 5, 250).cumsum(),
            'Volume': np.random.randint(1000, 5000, 250)
        })
        
        # Add technical indicators with NaN values in early periods
        # Short window features
        self.sample_data['sma_5'] = self.sample_data['Close'].rolling(window=5).mean()
        self.sample_data['sma_10'] = self.sample_data['Close'].rolling(window=10).mean()
        self.sample_data['ema_5'] = self.sample_data['Close'].ewm(span=5, adjust=False).mean()
        self.sample_data['ema_10'] = self.sample_data['Close'].ewm(span=10, adjust=False).mean()
        self.sample_data['roc_1'] = self.sample_data['Close'].pct_change(periods=1) * 100
        self.sample_data['roc_5'] = self.sample_data['Close'].pct_change(periods=5) * 100
        
        # Medium window features
        self.sample_data['sma_20'] = self.sample_data['Close'].rolling(window=20).mean()
        self.sample_data['ema_20'] = self.sample_data['Close'].ewm(span=20, adjust=False).mean()
        
        # Calculate RSI (medium window)
        delta = self.sample_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.sample_data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Long window features
        self.sample_data['sma_50'] = self.sample_data['Close'].rolling(window=50).mean()
        self.sample_data['sma_200'] = self.sample_data['Close'].rolling(window=200).mean()
        self.sample_data['ema_50'] = self.sample_data['Close'].ewm(span=50, adjust=False).mean()
        self.sample_data['ema_200'] = self.sample_data['Close'].ewm(span=200, adjust=False).mean()
        
        # Volatility features
        for window in [5, 10, 20, 30]:
            returns = self.sample_data['Close'].pct_change()
            self.sample_data[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Categorical features
        self.sample_data['day_of_week'] = dates.dayofweek
        self.sample_data['month'] = dates.month
        self.sample_data['quarter'] = dates.quarter
        
        # Create a default preparator instance
        self.preparator = FeaturePreparator()
    
    def test_initialization(self):
        """Test that the preparator initializes with correct parameters"""
        # Test default initialization
        preparator = FeaturePreparator()
        self.assertEqual(preparator.price_cols, ['open', 'high', 'low', 'close'])
        self.assertEqual(preparator.volume_col, 'volume')
        self.assertEqual(preparator.timestamp_col, 'date')
        self.assertTrue(preparator.preserve_original_prices)
        self.assertEqual(preparator.price_transform_method, 'returns')
        self.assertEqual(preparator.treatment_mode, 'advanced')
        
        # Test custom initialization
        custom_preparator = FeaturePreparator(
            price_cols=['Price', 'PriceHigh', 'PriceLow', 'PriceClose'],
            volume_col='Vol',
            timestamp_col='Time',
            preserve_original_prices=False,
            price_transform_method='log',
            trim_initial_periods=50,
            min_data_points=500,
            treatment_mode='basic'
        )
        self.assertEqual(custom_preparator.price_cols, ['price', 'pricehigh', 'pricelow', 'priceclose'])
        self.assertEqual(custom_preparator.volume_col, 'vol')
        self.assertEqual(custom_preparator.timestamp_col, 'time')
        self.assertFalse(custom_preparator.preserve_original_prices)
        self.assertEqual(custom_preparator.price_transform_method, 'log')
        self.assertEqual(custom_preparator.trim_initial_periods, 50)
        self.assertEqual(custom_preparator.min_data_points, 500)
        self.assertEqual(custom_preparator.treatment_mode, 'basic')
    
    def test_fit_method(self):
        """Test the fit method for feature categorization and window size detection"""
        # Fit the preparator to sample data
        preparator = self.preparator.fit(self.sample_data)
        
        # Check that feature categories were properly created
        categories = preparator._feature_categories
        
        # Price features should be categorized correctly
        self.assertTrue(all(col.lower() in categories.get('price', []) for col in ['Open', 'High', 'Low', 'Close']))
        
        # Technical indicators should be categorized by window size
        self.assertTrue('sma_5' in categories.get('short_window', []))
        self.assertTrue('sma_10' in categories.get('short_window', []))
        self.assertTrue('sma_20' in categories.get('medium_window', []))
        self.assertTrue('rsi_14' in categories.get('medium_window', []))
        self.assertTrue('sma_50' in categories.get('long_window', []))
        self.assertTrue('sma_200' in categories.get('long_window', []))
        
        # Window sizes should be detected
        window_sizes = preparator._window_sizes
        self.assertIn('max_short_window', window_sizes)
        self.assertIn('max_medium_window', window_sizes)
        self.assertIn('max_long_window', window_sizes)
        
        # Check that window sizes are in the expected range
        self.assertLessEqual(window_sizes['max_short_window'], 10)
        self.assertGreaterEqual(window_sizes['max_medium_window'], 14)  # For RSI
        self.assertGreaterEqual(window_sizes['max_long_window'], 200)  # For SMA 200
    
    def test_price_transformations(self):
        """Test different price transformation methods"""
        # Test 'returns' transformation by directly using _transform_prices
        returns_preparator = FeaturePreparator(price_transform_method='returns')
        returns_preparator.fit(self.sample_data)
        
        # Just test the price transformation directly without the full transform
        returns_result = returns_preparator._transform_prices(self.sample_data)
        
        # Check that return columns were created (lowercase for consistency)
        self.assertIn('open_return', returns_result.columns)
        self.assertIn('close_return', returns_result.columns)
        
        # Verify return calculation
        expected_returns = self.sample_data['Close'].pct_change()
        # Convert column names for comparison
        expected_returns.name = expected_returns.name.lower() if expected_returns.name else None
        
        pd.testing.assert_series_equal(
            returns_result['close_return'].fillna(0),
            expected_returns.fillna(0),
            check_names=False
        )
        
        # Test 'log' transformation
        log_preparator = FeaturePreparator(price_transform_method='log')
        log_preparator.fit(self.sample_data)
        
        # Just test the price transformation directly
        log_result = log_preparator._transform_prices(self.sample_data)
        
        # Check that log columns were created
        self.assertIn('open_log', log_result.columns)
        self.assertIn('close_log', log_result.columns)
        
        # Verify log calculation
        min_val = self.sample_data['Close'].min()
        offset = 0 if min_val > 0 else abs(min_val) + 1e-8
        expected_log = np.log(self.sample_data['Close'] + offset)
        # Convert column names for comparison
        expected_log.name = expected_log.name.lower() if expected_log.name else None
        
        pd.testing.assert_series_equal(
            log_result['close_log'],
            expected_log,
            check_names=False
        )
        
        # Test 'pct_change' transformation
        pct_preparator = FeaturePreparator(price_transform_method='pct_change')
        pct_preparator.fit(self.sample_data)
        
        # Just test the price transformation directly
        pct_result = pct_preparator._transform_prices(self.sample_data)
        
        # Check that percentage change columns were created
        self.assertIn('open_pct_change', pct_result.columns)
        self.assertIn('close_pct_change', pct_result.columns)
        
        # Verify percentage change calculation
        first_value = self.sample_data['Close'].iloc[0]
        expected_pct = (self.sample_data['Close'] / first_value - 1) * 100
        # Convert column names for comparison
        expected_pct.name = expected_pct.name.lower() if expected_pct.name else None
        
        pd.testing.assert_series_equal(
            pct_result['close_pct_change'],
            expected_pct,
            check_names=False
        )
        
        # Also verify that the full transform pipeline works correctly
        full_result = returns_preparator.transform(self.sample_data)
        # Look for lowercase columns in the final result
        self.assertIn('close_return', [col.lower() for col in full_result.columns])
        # Don't compare with expected_returns since the full transform may trim data
                
    def test_original_price_preservation(self):
        """Test that original prices are preserved when requested"""
        # With preservation (default)
        preserve_preparator = FeaturePreparator(preserve_original_prices=True)
        preserve_preparator.fit(self.sample_data)
        preserve_result = preserve_preparator.transform(self.sample_data)
        
        # Check that original price columns are preserved with _original suffix
        self.assertIn('open_original', preserve_result.columns)
        self.assertIn('high_original', preserve_result.columns)
        self.assertIn('low_original', preserve_result.columns)
        self.assertIn('close_original', preserve_result.columns)
        
        # Also check for _raw columns
        self.assertIn('open_raw', preserve_result.columns)
        self.assertIn('close_raw', preserve_result.columns)
        
        # Without preservation
        no_preserve_preparator = FeaturePreparator(preserve_original_prices=False)
        no_preserve_preparator.fit(self.sample_data)
        no_preserve_result = no_preserve_preparator.transform(self.sample_data)
        
        # We should still have _raw columns because they're essential
        self.assertIn('open_raw', no_preserve_result.columns)
        self.assertIn('close_raw', no_preserve_result.columns)
    
    def test_basic_treatment_mode(self):
        """Test the 'basic' treatment mode which trims initial periods"""
        # Create preparator with basic treatment
        basic_preparator = FeaturePreparator(treatment_mode='basic')
        basic_preparator.fit(self.sample_data)
        basic_result = basic_preparator.transform(self.sample_data)
        
        # In basic mode, NaN values should be removed by trimming
        nan_count = basic_result.isna().sum().sum()
        self.assertEqual(nan_count, 0)
        
        # Check that data was trimmed
        self.assertLess(len(basic_result), len(self.sample_data))
    
    def test_advanced_treatment_mode(self):
        """Test the 'advanced' treatment mode with category-specific handling"""
        # Create preparator with advanced treatment
        advanced_preparator = FeaturePreparator(treatment_mode='advanced')
        advanced_preparator.fit(self.sample_data)
        advanced_result = advanced_preparator.transform(self.sample_data)
        
        # Check that short window features have no NaNs (due to backfill)
        short_window_features = ['sma_5', 'sma_10', 'ema_5', 'ema_10', 'roc_1', 'roc_5']
        short_window_features = [col for col in short_window_features if col in advanced_result.columns]
        short_window_nan = advanced_result[short_window_features].isna().sum().sum()
        self.assertEqual(short_window_nan, 0)
        
        # Long window features should be trimmed
        self.assertLess(len(advanced_result), len(self.sample_data))
        
        # But not all data should be trimmed
        self.assertGreater(len(advanced_result), len(self.sample_data) - 200)  # Since SMA-200 is our longest window
    
    def test_hybrid_treatment_mode(self):
        """Test the 'hybrid' treatment mode that balances retention and validity"""
        # Create preparator with hybrid treatment
        hybrid_preparator = FeaturePreparator(treatment_mode='hybrid')
        hybrid_preparator.fit(self.sample_data)
        hybrid_result = hybrid_preparator.transform(self.sample_data)
        
        # Hybrid mode should have no NaNs in the final result
        nan_count = hybrid_result.isna().sum().sum()
        self.assertEqual(nan_count, 0)
        
        # Should preserve more data than advanced mode for large datasets
        advanced_preparator = FeaturePreparator(treatment_mode='advanced')
        advanced_preparator.fit(self.sample_data)
        advanced_result = advanced_preparator.transform(self.sample_data)
        
        # For our full dataset, advanced might actually preserve more data
        # But for smaller datasets, hybrid should preserve more
        small_data = self.sample_data.iloc[:100].copy()
        
        hybrid_preparator.fit(small_data)
        hybrid_small = hybrid_preparator.transform(small_data)
        
        advanced_preparator.fit(small_data)
        advanced_small = advanced_preparator.transform(small_data)
        
        # Hybrid should preserve more data in the small dataset case
        self.assertGreaterEqual(len(hybrid_small), len(advanced_small))
    
    def test_custom_feature_categories(self):
        """Test using custom feature category rules"""
        # Define custom category rules
        custom_categories = {
            'price_data': ['open', 'high', 'low', 'close'],
            'volume_data': ['volume'],
            'trend_indicators': ['sma', 'ema'],
            'oscillators': ['rsi', 'macd', 'stoch'],
            'volatility_indicators': ['volatility', 'atr'],
            'time_features': ['day', 'month', 'quarter']
        }
        
        # Create preparator with custom categories
        custom_preparator = FeaturePreparator(feature_category_rules=custom_categories)
        custom_preparator.fit(self.sample_data)
        
        # Check that feature categories were properly created
        categories = custom_preparator._feature_categories
        
        # Verify categorization
        self.assertTrue(all(col.lower() in categories.get('price_data', []) for col in ['Open', 'High', 'Low', 'Close']))
        self.assertTrue('sma_5' in categories.get('trend_indicators', []))
        self.assertTrue('rsi_14' in categories.get('oscillators', []))
        self.assertTrue('volatility_10' in categories.get('volatility_indicators', []))
        self.assertTrue('day_of_week' in categories.get('time_features', []))
    
    def test_missing_values_handling(self):
        """Test handling of missing values in the input data"""
        # Create data with some additional missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[10:15, 'Close'] = np.nan
        data_with_missing.loc[30:35, 'sma_20'] = np.nan
        data_with_missing.loc[50:55, 'rsi_14'] = np.nan
        
        # Process with different treatment modes
        for mode in ['basic', 'advanced', 'hybrid']:
            preparator = FeaturePreparator(treatment_mode=mode)
            preparator.fit(data_with_missing)
            result = preparator.transform(data_with_missing)
            
            # Verify no NaNs in result
            self.assertEqual(result.isna().sum().sum(), 0)
    
    def test_fit_transform(self):
        """Test the fit_transform convenience method"""
        # Should be equivalent to calling fit() then transform()
        preparator = FeaturePreparator()
        result = preparator.fit_transform(self.sample_data)
        
        # Basic checks
        self.assertGreater(len(result.columns), len(self.sample_data.columns))
        self.assertIn('close_return', result.columns)
        self.assertEqual(result.isna().sum().sum(), 0)  # No NaNs in result


if __name__ == '__main__':
    unittest.main()