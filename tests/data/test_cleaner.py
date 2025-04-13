import os
import sys
import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from unittest.mock import patch

# Import the components we want to test
from data.processors.cleaner import DataCleaner

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestDataCleaner(unittest.TestCase):
    """Test suite for the DataCleaner class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample OHLCV data with various issues
        dates = pd.date_range(start='2023-01-01', periods=10, freq='min')
        
        # Regular data with some issues
        self.sample_data = pd.DataFrame({
            'Timestamp': dates,
            'Open': [100, 105, np.nan, 103, 106, 107, 105, 104, 102, 101],
            'High': [110, 115, 108, 108, 116, 117, 115, 114, 112, 111],
            'Low': [95, 100, 97, 98, 101, 102, 100, 99, 97, 96],
            'Close': [105, 107, 102, 104, 110, 112, 108, 105, 100, 98],
            'Volume': [1000, 1200, np.nan, 800, 1500, 1300, 1100, 900, 700, 1000]
        })
        
        # Data with OHLC validity issues
        self.invalid_ohlc = pd.DataFrame({
            'Timestamp': dates,
            'Open': [100, 105, 103, 103, 106, 107, 105, 104, 102, 101],
            'High': [95, 115, 108, 108, 105, 117, 115, 103, 112, 111],  # Some high values < open
            'Low': [105, 100, 97, 108, 101, 102, 100, 99, 97, 96],      # Some low values > open
            'Close': [105, 107, 102, 104, 110, 112, 108, 105, 100, 98],
            'Volume': [1000, 1200, 900, 800, 1500, 1300, 1100, 900, 700, 1000]
        })
        
        # Data with outliers
        self.outlier_data = pd.DataFrame({
            'Timestamp': dates,
            'Open': [100, 105, 103, 103, 106, 107, 105, 104, 102, 500],  # Outlier in last row
            'High': [110, 115, 108, 108, 116, 117, 115, 114, 112, 600],  # Outlier in last row
            'Low': [95, 100, 97, 98, 101, 102, 100, 99, 97, 96],
            'Close': [105, 107, 102, 104, 110, 112, 108, 105, 100, 550],  # Outlier in last row
            'Volume': [1000, 1200, 900, 800, 1500, 1300, 1100, 900, 700, 10000]  # Outlier
        })
        
        # Data with time gaps (missing days)
        irregular_dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            # Gap: Jan 4th is missing
            datetime(2023, 1, 5),
            datetime(2023, 1, 6),
            # Gap: Jan 7th and 8th are missing
            datetime(2023, 1, 9),
            datetime(2023, 1, 10)
        ]
        
        self.time_gap_data = pd.DataFrame({
            'Timestamp': irregular_dates,
            'Open': [100, 105, 103, 106, 107, 102, 101],
            'High': [110, 115, 108, 116, 117, 112, 111],
            'Low': [95, 100, 97, 101, 102, 97, 96],
            'Close': [105, 107, 102, 110, 112, 100, 98],
            'Volume': [1000, 1200, 900, 1500, 1300, 700, 1000]
        })
        
        self.volume_outlier_data = pd.DataFrame({
            'Timestamp': dates,
            'Open': [100, 105, 103, 103, 106, 107, 105, 104, 102, 101],
            'High': [110, 115, 108, 108, 116, 117, 115, 114, 112, 111],
            'Low': [95, 100, 97, 98, 101, 102, 100, 99, 97, 96],
            'Close': [105, 107, 102, 104, 110, 112, 108, 105, 100, 98],
            'Volume': [1000, 1200, 900, 800, 1500, 1300, 1100, 900, 700, 20000]  # Outlier in last row
        })
        
        # Create default cleaner instance
        self.cleaner = DataCleaner()
    
    def test_initialization(self):
        """Test that the cleaner initializes with correct parameters"""
        # Test default initialization
        cleaner = DataCleaner()
        self.assertEqual(cleaner.price_cols, ['open', 'high', 'low', 'close'])
        self.assertEqual(cleaner.volume_col, 'volume')
        self.assertEqual(cleaner.timestamp_col, 'timestamp')
        self.assertEqual(cleaner.missing_method, 'ffill')
        
        # Test custom initialization
        custom_cleaner = DataCleaner(
            price_cols=['Price', 'AdjustedPrice'],
            volume_col='TradingVolume',
            timestamp_col='Date',
            missing_method='interpolate',
            outlier_method='zscore',
            outlier_threshold=2.5
        )
        self.assertEqual(custom_cleaner.price_cols, ['price', 'adjustedprice'])
        self.assertEqual(custom_cleaner.volume_col, 'tradingvolume')
        self.assertEqual(custom_cleaner.timestamp_col, 'date')
        self.assertEqual(custom_cleaner.missing_method, 'interpolate')
        self.assertEqual(custom_cleaner.outlier_method, 'zscore')
        self.assertEqual(custom_cleaner.outlier_threshold, 2.5)
    
    def test_fit_method(self):
        """Test the fit method for calculating statistics"""
        # Fit the cleaner to the sample data
        self.cleaner.fit(self.sample_data)
        
        # Check that statistics were calculated for price columns
        self.assertIn('open_mean', self.cleaner._stats)
        self.assertIn('high_mean', self.cleaner._stats)
        self.assertIn('low_mean', self.cleaner._stats)
        self.assertIn('close_mean', self.cleaner._stats)
        
        # Verify correct calculation of statistics
        self.assertAlmostEqual(self.cleaner._stats['open_mean'], 
                               self.sample_data['Open'].mean(), 
                               places=2)
        self.assertAlmostEqual(self.cleaner._stats['high_mean'], 
                               self.sample_data['High'].mean(), 
                               places=2)
    
    def test_handle_missing_values(self):
        """Test the handling of missing values"""
        # Fit and transform the data - setting preserve_original_case to False for this test
        cleaner = DataCleaner(missing_method='ffill', preserve_original_case=False)
        transformed_data = cleaner.fit_transform(self.sample_data)
        
        # Check that there are no NaN values in the result
        self.assertEqual(transformed_data.isna().sum().sum(), 0)
        
        # Verify forward fill worked correctly for price data
        # The third row had NaN for Open, should be filled with previous value
        self.assertEqual(transformed_data['open'].iloc[2], self.sample_data['Open'].iloc[1])
        
        # Test interpolation method
        cleaner_interp = DataCleaner(missing_method='interpolate', preserve_original_case=False)
        transformed_interp = cleaner_interp.fit_transform(self.sample_data)
        
        # Check that there are no NaN values in the result
        self.assertEqual(transformed_interp.isna().sum().sum(), 0)
        
        # Volume should be filled with 0 (different from price columns)
        # Check if NaN volume was filled with 0
        orig_volume_nan_idx = self.sample_data['Volume'].isna()
        self.assertTrue((transformed_data['volume'][orig_volume_nan_idx] == 0).all())
    
    def test_handle_outliers(self):
        """Test the detection and handling of outliers"""
        # Test with different outlier methods
        
        # 1. Z-score method - setting preserve_original_case to False for this test
        cleaner_zscore = DataCleaner(outlier_method='zscore', outlier_threshold=2.0, preserve_original_case=False)
        result_zscore = cleaner_zscore.fit_transform(self.outlier_data)
        
        # The last row had an extreme value (500) for Open which should be capped
        # Check that the outlier was reduced
        self.assertLess(result_zscore['open'].iloc[-1], self.outlier_data['Open'].iloc[-1])
        
        # 2. IQR method
        cleaner_iqr = DataCleaner(outlier_method='iqr', outlier_threshold=1.5, preserve_original_case=False)
        result_iqr = cleaner_iqr.fit_transform(self.outlier_data)
        
        # Check that the outlier was reduced
        self.assertLess(result_iqr['open'].iloc[-1], self.outlier_data['Open'].iloc[-1])
        
        # 3. Winsorize method
        cleaner_winsorize = DataCleaner(outlier_method='winsorize', preserve_original_case=False)
        result_winsorize = cleaner_winsorize.fit_transform(self.outlier_data)
        
        # Check that the outlier was reduced
        self.assertLess(result_winsorize['open'].iloc[-1], self.outlier_data['Open'].iloc[-1])
    
    def test_ensure_ohlc_validity(self):
        """Test ensuring OHLC validity (High ≥ Open ≥ Close ≥ Low)"""
        # Setting preserve_original_case to False for this test
        cleaner = DataCleaner(ensure_ohlc_validity=True, preserve_original_case=False)
        result = cleaner.fit_transform(self.invalid_ohlc)
        
        # Check that High is always the maximum
        for i in range(len(result)):
            row_max = max(result['open'].iloc[i], result['close'].iloc[i])
            self.assertGreaterEqual(result['high'].iloc[i], row_max)
        
        # Check that Low is always the minimum
        for i in range(len(result)):
            row_min = min(result['open'].iloc[i], result['close'].iloc[i])
            self.assertLessEqual(result['low'].iloc[i], row_min)
    
    def test_time_continuity(self):
        """Test ensuring time continuity through resampling"""
        # Set timestamp column as the index
        time_data = self.time_gap_data.copy()
        time_data = time_data.set_index('Timestamp')
        
        # Create cleaner with daily resampling
        cleaner = DataCleaner(resample_rule='1D')
        
        # Patch the method since we need to test internal implementation
        with patch.object(cleaner, '_ensure_time_continuity') as mock_method:
            mock_method.return_value = time_data  # Just return original data for this test
            cleaner.fit_transform(time_data)
            
            # Verify the method was called
            mock_method.assert_called_once()
        
        # Now test the actual functionality (need to handle case with datetime index)
        # Create a new test with data already having a datetime index
        cleaner_real = DataCleaner(resample_rule='1D')
        
        # Create a sample with a datetime index to directly test _ensure_time_continuity
        sample_with_index = self.time_gap_data.copy()
        sample_with_index = sample_with_index.set_index('Timestamp')
        
        # Call the method directly
        result = cleaner_real._ensure_time_continuity(sample_with_index)
        
        # Check that the result has more rows (filled gaps)
        self.assertGreater(len(result), len(sample_with_index))
        
        # Check that we now have a continuous date range
        expected_dates = pd.date_range(start=sample_with_index.index.min(),
                                      end=sample_with_index.index.max(),
                                      freq='1D')
        self.assertEqual(len(result), len(expected_dates))
    
    def test_preserve_column_case(self):
        """Test that original column case is preserved when requested"""
        # With case preservation (default)
        cleaner_preserve = DataCleaner(preserve_original_case=True)
        result_preserve = cleaner_preserve.fit_transform(self.sample_data)
        
        # Check that original column names are maintained
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            self.assertIn(col, result_preserve.columns)
        
        # Without case preservation
        cleaner_no_preserve = DataCleaner(preserve_original_case=False)
        result_no_preserve = cleaner_no_preserve.fit_transform(self.sample_data)
        
        # Check that columns are lowercase
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, result_no_preserve.columns)
    
    def test_edge_cases(self):
        """Test edge cases like empty dataframes, missing columns, etc."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        cleaner = DataCleaner()
        
        # This should not raise an error, but return empty dataframe
        result = cleaner.fit_transform(empty_df)
        self.assertTrue(result.empty)
        
        # Test with missing price columns
        incomplete_df = pd.DataFrame({
            'Timestamp': pd.date_range(start='2023-01-01', periods=5, freq='1D'),
            'Open': [100, 101, 102, 103, 104],
            # Missing High, Low, Close
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Should not raise an error
        incomplete_result = cleaner.fit_transform(incomplete_df)
        self.assertEqual(len(incomplete_result), 5)
    
    def test_end_to_end(self):
        """Test the complete data cleaning process end-to-end"""
        # Create a complex test case with multiple issues
        dates = pd.date_range(start='2023-01-01', periods=15, freq='min')
        
        complex_data = pd.DataFrame({
            'Timestamp': dates,
            'Open': [100, 105, np.nan, 103, 106, 107, 105, 104, 102, 101, 103, 104, 105, 106, 107],
            'High': [95, 115, 108, 108, 105, 117, 115, 103, 112, 111, 113, 114, 115, 116, 400],  # Invalid High and outlier
            'Low': [105, 100, 97, 108, 101, 102, 100, 99, 97, 96, 98, 99, 100, 101, 102],      # Invalid Low
            'Close': [105, 107, 102, 104, 110, 112, 108, 105, 100, 98, 99, 100, 101, 102, 103],
            'Volume': [1000, 1200, np.nan, 800, 1500, 1300, 1100, 900, 700, 1000, 1100, 1200, 1300, 1400, 20000]  # Missing and outlier
        })
        
        # Configure cleaner with all options enabled - but note that volume outliers won't be handled
        # because _handle_outliers only processes price_cols
        complete_cleaner = DataCleaner(
            missing_method='ffill',
            outlier_method='zscore',
            outlier_threshold=2.5,
            ensure_ohlc_validity=True,
            preserve_original_case=True
        )
        
        result = complete_cleaner.fit_transform(complex_data)
        
        # Check that result maintains expected properties
        self.assertEqual(len(result), len(complex_data))  # Same number of rows
        self.assertEqual(result.isna().sum().sum(), 0)    # No missing values
        
        # Check OHLC validity
        for i in range(len(result)):
            self.assertGreaterEqual(result['High'].iloc[i], result['Open'].iloc[i])
            self.assertGreaterEqual(result['High'].iloc[i], result['Close'].iloc[i])
            self.assertLessEqual(result['Low'].iloc[i], result['Open'].iloc[i])
            self.assertLessEqual(result['Low'].iloc[i], result['Close'].iloc[i])
        
        # Check that extreme outliers were handled for High (price column)
        self.assertLess(result['High'].iloc[-1], complex_data['High'].iloc[-1])
        
        # MODIFY THIS LINE: Don't check Volume outlier since the current implementation
        # doesn't handle volume outliers (it only processes price_cols)
        # Instead, verify Volume is the same as in the original data
        self.assertEqual(result['Volume'].iloc[-1], complex_data['Volume'].iloc[-1])
        
    def test_volume_outlier_handling(self):
        """Test that volume outliers are handled when requested"""
        # Create cleaner with volume outlier handling enabled
        cleaner = DataCleaner(
            outlier_method='zscore',
            outlier_threshold=2.0,
            handle_volume_outliers=True,
            preserve_original_case=True
        )
        
        # Fit and transform the data
        result = cleaner.fit_transform(self.volume_outlier_data)
        
        # Check that the volume outlier was reduced
        self.assertLess(result['Volume'].iloc[-1], self.volume_outlier_data['Volume'].iloc[-1])
        
        # Create cleaner with volume outlier handling disabled (default)
        cleaner_no_volume = DataCleaner(
            outlier_method='zscore',
            outlier_threshold=2.0,
            handle_volume_outliers=False,
            preserve_original_case=True
        )
        
        # Fit and transform the data
        result_no_volume = cleaner_no_volume.fit_transform(self.volume_outlier_data)
        
        # Check that the volume outlier was NOT changed
        self.assertEqual(result_no_volume['Volume'].iloc[-1], self.volume_outlier_data['Volume'].iloc[-1])
    
    def test_end_to_end_with_volume_outliers(self):
        """Test the complete data cleaning process with volume outlier handling"""
        # Create a complex test case with multiple issues including volume outliers
        dates = pd.date_range(start='2023-01-01', periods=15, freq='min')
        
        complex_data = pd.DataFrame({
            'Timestamp': dates,
            'Open': [100, 105, np.nan, 103, 106, 107, 105, 104, 102, 101, 103, 104, 105, 106, 107],
            'High': [95, 115, 108, 108, 105, 117, 115, 103, 112, 111, 113, 114, 115, 116, 400],  # Invalid High and outlier
            'Low': [105, 100, 97, 108, 101, 102, 100, 99, 97, 96, 98, 99, 100, 101, 102],      # Invalid Low
            'Close': [105, 107, 102, 104, 110, 112, 108, 105, 100, 98, 99, 100, 101, 102, 103],
            'Volume': [1000, 1200, np.nan, 800, 1500, 1300, 1100, 900, 700, 1000, 1100, 1200, 1300, 1400, 20000]  # Missing and outlier
        })
        
        # Configure cleaner with all options enabled, including volume outlier handling
        complete_cleaner = DataCleaner(
            missing_method='ffill',
            outlier_method='zscore',
            outlier_threshold=2.5,
            ensure_ohlc_validity=True,
            preserve_original_case=True,
            handle_volume_outliers=True
        )
        
        result = complete_cleaner.fit_transform(complex_data)
        
        # Check that result maintains expected properties
        self.assertEqual(len(result), len(complex_data))  # Same number of rows
        self.assertEqual(result.isna().sum().sum(), 0)    # No missing values
        
        # Check OHLC validity
        for i in range(len(result)):
            self.assertGreaterEqual(result['High'].iloc[i], result['Open'].iloc[i])
            self.assertGreaterEqual(result['High'].iloc[i], result['Close'].iloc[i])
            self.assertLessEqual(result['Low'].iloc[i], result['Open'].iloc[i])
            self.assertLessEqual(result['Low'].iloc[i], result['Close'].iloc[i])
        
        # Check that extreme outliers were handled for both High and Volume
        self.assertLess(result['High'].iloc[-1], complex_data['High'].iloc[-1])
        self.assertLess(result['Volume'].iloc[-1], complex_data['Volume'].iloc[-1])



if __name__ == '__main__':
    unittest.main()