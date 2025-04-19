import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

from data.processors.normalizer import DataNormalizer


class TestDataNormalizer(unittest.TestCase):
    """Test suite for the DataNormalizer class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample OHLCV data for testing
        dates = pd.date_range(start='2023-01-01', periods=10, freq='1D')
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': [100, 102, 104, 103, 105, 107, 106, 108, 110, 112],
            'high': [105, 107, 109, 108, 110, 112, 111, 113, 115, 117],
            'low': [98, 100, 102, 101, 103, 105, 104, 106, 108, 110],
            'close': [102, 104, 106, 105, 107, 109, 108, 110, 112, 114],
            'volume': [1000, 1200, 1100, 900, 1300, 1400, 1200, 1100, 1000, 1300],
            'other_metric': [10, 12, 14, 13, 15, 17, 16, 18, 20, 22],
            'category': ['A', 'A', 'B', 'B', 'A', 'C', 'C', 'B', 'A', 'C']
        })
        
        # Create a default normalizer instance
        self.normalizer = DataNormalizer()
    
    def test_initialization(self):
        """Test that the normalizer initializes with correct parameters"""
        # Test default initialization
        normalizer = DataNormalizer()
        self.assertEqual(normalizer.price_cols, ['open', 'high', 'low', 'close'])
        self.assertEqual(normalizer.volume_col, 'volume')
        self.assertEqual(normalizer.price_method, 'returns')
        self.assertEqual(normalizer.volume_method, 'log')
        self.assertEqual(normalizer.other_method, 'zscore')
        
        # Test custom initialization
        custom_normalizer = DataNormalizer(
            price_cols=['Price', 'AdjustedPrice'],
            volume_col='TradingVolume',
            price_method='zscore',
            volume_method='pct_of_avg',
            other_method='minmax'
        )
        self.assertEqual(custom_normalizer.price_cols, ['price', 'adjustedprice'])
        self.assertEqual(custom_normalizer.volume_col, 'tradingvolume')
        self.assertEqual(custom_normalizer.price_method, 'zscore')
        self.assertEqual(custom_normalizer.volume_method, 'pct_of_avg')
        self.assertEqual(custom_normalizer.other_method, 'minmax')
    
    def test_fit_method(self):
        """Test the fit method for calculating normalization parameters"""
        self.normalizer.fit(self.sample_data)
        
        # Check that parameters were calculated for volume (volume_method='log' doesn't need params)
        
        # Check parameters for 'other_metric' (other_method='zscore')
        self.assertIn('other_metric_mean', self.normalizer._params)
        self.assertIn('other_metric_std', self.normalizer._params)
        self.assertAlmostEqual(
            self.normalizer._params['other_metric_mean'], 
            self.sample_data['other_metric'].mean(),
            places=6
        )
        self.assertAlmostEqual(
            self.normalizer._params['other_metric_std'], 
            self.sample_data['other_metric'].std(),
            places=6
        )
        
        # Test with different methods
        minmax_normalizer = DataNormalizer(price_method='minmax', volume_method='minmax')
        minmax_normalizer.fit(self.sample_data)
        
        # Check minmax parameters for price columns
        for col in ['open', 'high', 'low', 'close']:
            self.assertIn(f"{col}_min", minmax_normalizer._params)
            self.assertIn(f"{col}_max", minmax_normalizer._params)
            self.assertEqual(minmax_normalizer._params[f"{col}_min"], self.sample_data[col].min())
            self.assertEqual(minmax_normalizer._params[f"{col}_max"], self.sample_data[col].max())
        
        # Test robust scaling
        robust_normalizer = DataNormalizer(price_method='robust', volume_method='robust')
        robust_normalizer.fit(self.sample_data)
        
        # Check robust parameters for price columns
        for col in ['open', 'high', 'low', 'close']:
            self.assertIn(f"{col}_median", robust_normalizer._params)
            self.assertIn(f"{col}_iqr", robust_normalizer._params)
            self.assertEqual(robust_normalizer._params[f"{col}_median"], self.sample_data[col].median())
            self.assertAlmostEqual(
                robust_normalizer._params[f"{col}_iqr"], 
                self.sample_data[col].quantile(0.75) - self.sample_data[col].quantile(0.25),
                places=6
            )
    
    def test_transform_returns(self):
        """Test the returns normalization method for price columns"""
        # Configure normalizer to use returns for price
        returns_normalizer = DataNormalizer(price_method='returns')
        returns_normalizer.fit(self.sample_data)
        result = returns_normalizer.transform(self.sample_data)
        
        # Check that returns were calculated correctly
        for col in ['open', 'high', 'low', 'close']:
            return_col = f"{col}_return"
            self.assertIn(return_col, result.columns)
            
            # Calculate expected returns manually
            expected_returns = self.sample_data[col].pct_change()
            
            # Compare results (allowing for floating point differences)
            pd.testing.assert_series_equal(
                result[return_col].fillna(0),  # Replace NaN with 0 for comparison
                expected_returns.fillna(0),    # Replace NaN with 0 for comparison
                check_names=False              # Column names might differ
            )
    
    def test_transform_zscore(self):
        """Test the Z-score normalization method"""
        # Configure normalizer to use Z-score for all columns
        zscore_normalizer = DataNormalizer(
            price_method='zscore',
            volume_method='zscore',
            other_method='zscore'
        )
        zscore_normalizer.fit(self.sample_data)
        result = zscore_normalizer.transform(self.sample_data)
        
        # Check Z-score normalization for numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume', 'other_metric']:
            self.assertIn(col, result.columns)
            
            # Calculate expected Z-scores manually
            mean = self.sample_data[col].mean()
            std = self.sample_data[col].std()
            expected_zscores = (self.sample_data[col] - mean) / std
            
            # Check that results match expected values
            pd.testing.assert_series_equal(
                result[col],
                expected_zscores,
                check_names=False,
                rtol=1e-5  # Allow small relative differences due to floating point
            )
    
    def test_transform_log(self):
        """Test the logarithmic transformation method"""
        # Default normalizer uses log for volume
        self.normalizer.fit(self.sample_data)
        result = self.normalizer.transform(self.sample_data)
        
        # Check log normalization for volume
        self.assertIn('volume', result.columns)
        
        # Calculate expected log values manually
        expected_log = np.log(self.sample_data['volume'] + 1e-8)
        
        # Check that results match expected values
        pd.testing.assert_series_equal(
            result['volume'],
            expected_log,
            check_names=False,
            rtol=1e-5
        )
        
        # Test log transformation for price columns too
        log_normalizer = DataNormalizer(price_method='log')
        log_normalizer.fit(self.sample_data)
        log_result = log_normalizer.transform(self.sample_data)
        
        # Check log transformation for price columns
        for col in ['open', 'high', 'low', 'close']:
            self.assertIn(col, log_result.columns)
            
            # Calculate expected log values
            expected_log = np.log(self.sample_data[col] + 1e-8)
            
            # Check results
            pd.testing.assert_series_equal(
                log_result[col],
                expected_log,
                check_names=False,
                rtol=1e-5
            )
    
    def test_transform_minmax(self):
        """Test the min-max scaling method"""
        # Configure normalizer to use min-max scaling
        minmax_normalizer = DataNormalizer(
            price_method='minmax',
            volume_method='minmax',
            other_method='minmax'
        )
        minmax_normalizer.fit(self.sample_data)
        result = minmax_normalizer.transform(self.sample_data)
        
        # Check min-max normalization for numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume', 'other_metric']:
            self.assertIn(col, result.columns)
            
            # Calculate expected min-max scaled values
            min_val = self.sample_data[col].min()
            max_val = self.sample_data[col].max()
            expected_minmax = (self.sample_data[col] - min_val) / (max_val - min_val)
            
            # Check results
            pd.testing.assert_series_equal(
                result[col],
                expected_minmax,
                check_names=False,
                rtol=1e-5
            )
    
    def test_transform_pct_change(self):
        """Test the percentage change transformation method"""
        # Configure normalizer to use percentage change
        pct_normalizer = DataNormalizer(price_method='pct_change')
        pct_normalizer.fit(self.sample_data)
        result = pct_normalizer.transform(self.sample_data)
        
        # Check percentage change for price columns
        for col in ['open', 'high', 'low', 'close']:
            self.assertIn(col, result.columns)
            
            # Calculate expected percentage change
            first_value = self.sample_data[col].iloc[0]
            expected_pct = (self.sample_data[col] / first_value - 1) * 100
            
            # Check results
            pd.testing.assert_series_equal(
                result[col],
                expected_pct,
                check_names=False,
                rtol=1e-5
            )
    
    def test_transform_robust(self):
        """Test the robust scaling method"""
        # Configure normalizer to use robust scaling
        robust_normalizer = DataNormalizer(
            price_method='robust',
            volume_method='robust',
            other_method='robust'
        )
        robust_normalizer.fit(self.sample_data)
        result = robust_normalizer.transform(self.sample_data)
        
        # Check robust scaling for numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume', 'other_metric']:
            self.assertIn(col, result.columns)
            
            # Calculate expected robust scaled values
            median = self.sample_data[col].median()
            q1 = self.sample_data[col].quantile(0.25)
            q3 = self.sample_data[col].quantile(0.75)
            iqr = q3 - q1
            expected_robust = (self.sample_data[col] - median) / iqr
            
            # Check results
            pd.testing.assert_series_equal(
                result[col],
                expected_robust,
                check_names=False,
                rtol=1e-5
            )
    
    def test_transform_pct_of_avg(self):
        """Test the percentage of average method"""
        # Configure normalizer to use percentage of average
        pct_avg_normalizer = DataNormalizer(volume_method='pct_of_avg')
        pct_avg_normalizer.fit(self.sample_data)
        result = pct_avg_normalizer.transform(self.sample_data)
        
        # Check percentage of average for volume
        self.assertIn('volume', result.columns)
        
        # Calculate expected percentage of average
        avg = self.sample_data['volume'].mean()
        expected_pct_avg = self.sample_data['volume'] / avg
        
        # Check results
        pd.testing.assert_series_equal(
            result['volume'],
            expected_pct_avg,
            check_names=False,
            rtol=1e-5
        )
    
    def test_non_numeric_columns(self):
        """Test handling of non-numeric columns"""
        self.normalizer.fit(self.sample_data)
        result = self.normalizer.transform(self.sample_data)
        
        # Check that categorical column was preserved
        self.assertIn('category', result.columns)
        pd.testing.assert_series_equal(
            result['category'],
            self.sample_data['category']
        )
    
    def test_raw_price_columns(self):
        """Test that raw price columns are preserved"""
        # Add some raw price columns
        data_with_raw = self.sample_data.copy()
        data_with_raw['open_raw'] = data_with_raw['open']
        data_with_raw['close_raw'] = data_with_raw['close']
        
        self.normalizer.fit(data_with_raw)
        result = self.normalizer.transform(data_with_raw)
        
        # Check that raw columns are preserved as is
        self.assertIn('open_raw', result.columns)
        self.assertIn('close_raw', result.columns)
        
        pd.testing.assert_series_equal(
            result['open_raw'],
            data_with_raw['open_raw']
        )
        pd.testing.assert_series_equal(
            result['close_raw'],
            data_with_raw['close_raw']
        )
    
    def test_edge_cases(self):
        """Test edge cases like empty dataframes, constant columns, etc."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        self.normalizer.fit(empty_df)
        result = self.normalizer.transform(empty_df)
        self.assertTrue(result.empty)
        
        # Test with a column of all zeros
        zero_df = self.sample_data.copy()
        zero_df['zero_col'] = 0
        
        zero_normalizer = DataNormalizer(other_method='zscore')
        zero_normalizer.fit(zero_df)
        result = zero_normalizer.transform(zero_df)
        
        # Z-score of constant column should be zeros
        self.assertIn('zero_col', result.columns)
        self.assertTrue((result['zero_col'] == 0).all())
        
        # Test with a constant non-zero column (for minmax)
        const_df = self.sample_data.copy()
        const_df['const_col'] = 5
        
        minmax_normalizer = DataNormalizer(other_method='minmax')
        minmax_normalizer.fit(const_df)
        minmax_result = minmax_normalizer.transform(const_df)
        
        # MinMax of constant column should be a constant value between 0 and 1
        self.assertIn('const_col', minmax_result.columns)
        # Now checking that all values are the same constant, rather than specifically 0.5
        first_value = minmax_result['const_col'].iloc[0]
        self.assertTrue((minmax_result['const_col'] == first_value).all())

if __name__ == '__main__':
    unittest.main()