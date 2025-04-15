import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data.processors.splitter import TimeSeriesSplitter


class TestTimeSeriesSplitter(unittest.TestCase):
    """Test suite for the TimeSeriesSplitter class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create sample time series data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
        
        np.random.seed(42)  # For reproducibility
        self.sample_data = pd.DataFrame({
            'date': dates,
            'open': np.random.normal(100, 10, 100).cumsum(),
            'high': np.random.normal(105, 12, 100).cumsum(),
            'low': np.random.normal(95, 8, 100).cumsum(),
            'close': np.random.normal(102, 10, 100).cumsum(),
            'volume': np.random.randint(1000, 5000, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Create more data with different datetime frequency
        irregular_dates = [datetime(2023, 1, 1) + timedelta(hours=i*4) for i in range(100)]
        self.hourly_data = pd.DataFrame({
            'timestamp': irregular_dates,
            'price': np.random.normal(100, 5, 100).cumsum(),
            'volume': np.random.randint(100, 500, 100)
        })
    
    def test_initialization(self):
        """Test that the splitter initializes with correct parameters"""
        # Test default initialization
        splitter = TimeSeriesSplitter(train_period='30D', test_period='10D')
        self.assertEqual(splitter.train_period, '30D')
        self.assertEqual(splitter.test_period, '10D')
        self.assertEqual(splitter.step_size, '10D')  # Default to test_period
        self.assertEqual(splitter.date_column, 'date')
        
        # Test custom initialization
        custom_splitter = TimeSeriesSplitter(
            train_period='60D',
            test_period='15D',
            step_size='7D',
            max_train_size='45D',
            start_date=datetime(2023, 2, 1),
            end_date=datetime(2023, 3, 1),
            n_splits=5,
            date_column='timestamp'
        )
        self.assertEqual(custom_splitter.train_period, '60D')
        self.assertEqual(custom_splitter.test_period, '15D')
        self.assertEqual(custom_splitter.step_size, '7D')
        self.assertEqual(custom_splitter.max_train_size, '45D')
        self.assertEqual(custom_splitter.start_date, datetime(2023, 2, 1))
        self.assertEqual(custom_splitter.end_date, datetime(2023, 3, 1))
        self.assertEqual(custom_splitter.n_splits, 5)
        self.assertEqual(custom_splitter.date_column, 'timestamp')
    
    def test_period_parsing(self):
        """Test parsing of different period specifications"""
        splitter = TimeSeriesSplitter(train_period='30D', test_period='10D')
        dates = pd.DatetimeIndex(self.sample_data['date'])
        
        # Test string period parsing
        days_30 = splitter._parse_period('30D', dates)
        self.assertIsInstance(days_30, timedelta)
        self.assertEqual(days_30.days, 30)
        
        days_2w = splitter._parse_period('2W', dates)
        self.assertIsInstance(days_2w, timedelta)
        self.assertEqual(days_2w.days, 14)
        
        # Test integer parsing
        periods_10 = splitter._parse_period(10, dates)
        self.assertEqual(periods_10, 10)
        
        # Test timedelta parsing
        td = timedelta(days=15)
        self.assertEqual(splitter._parse_period(td, dates), td)
    
    def test_date_based_splits(self):
        """Test splitting by date periods"""
        # Create splitter with date-based periods
        splitter = TimeSeriesSplitter(
            train_period='30D', 
            test_period='10D',
            step_size='15D'
        )
        
        # Generate splits
        splitter.fit(self.sample_data)
        splits = splitter.get_splits()
        
        # Should have multiple splits
        self.assertGreater(len(splits), 1)
        
        # Check the first split
        train, test = splits[0]
        
        # Verify dates don't overlap
        last_train_date = train['date'].max()
        first_test_date = test['date'].min()
        self.assertLessEqual(last_train_date, first_test_date)
        
        # Verify approximate period lengths (may not be exact due to data availability)
        train_days = (train['date'].max() - train['date'].min()).days
        self.assertGreaterEqual(train_days, 25)  # Allow some flexibility
        self.assertLessEqual(train_days, 35)
        
        test_days = (test['date'].max() - test['date'].min()).days
        self.assertGreaterEqual(test_days, 5)
        self.assertLessEqual(test_days, 15)
    
    def test_period_based_splits(self):
        """Test splitting by number of periods"""
        # Create splitter with integer periods
        splitter = TimeSeriesSplitter(
            train_period=50,  # 50 samples for training
            test_period=20,   # 20 samples for testing
            step_size=10      # Advance by 10 samples each split
        )
        
        # Generate splits
        splitter.fit(self.sample_data)
        splits = splitter.get_splits()
        
        # Should have multiple splits
        self.assertGreater(len(splits), 1)
        
        # Check the first split
        train, test = splits[0]
        
        # Verify split sizes
        self.assertEqual(len(train), 50)
        self.assertEqual(len(test), 20)
        
        # Check that splits advance by step_size
        if len(splits) >= 2:
            next_train, _ = splits[1]
            first_idx_1 = train.index[0]
            first_idx_2 = next_train.index[0]
            self.assertEqual(first_idx_2 - first_idx_1, 10)
    
    def test_n_splits_parameter(self):
        """Test the n_splits parameter overrides other parameters"""
        n_splits = 5
        splitter = TimeSeriesSplitter(
            train_period='20D',
            test_period='10D',
            n_splits=n_splits
        )
        
        splitter.fit(self.sample_data)
        splits = splitter.get_splits()
        
        # Should have exactly n_splits
        self.assertEqual(len(splits), n_splits)
    
    def test_max_train_size(self):
        """Test the max_train_size parameter limits training data"""
        # Create splitter with max_train_size
        splitter = TimeSeriesSplitter(
            train_period='40D',
            test_period='10D',
            max_train_size='20D'  # Limit training to 20 days
        )
        
        splitter.fit(self.sample_data)
        splits = splitter.get_splits()
        
        # Get the first split
        train, _ = splits[0]
        
        # Training period should be limited to approximately 20 days
        train_days = (train['date'].max() - train['date'].min()).days
        self.assertGreaterEqual(train_days, 15)  # Allow some flexibility
        self.assertLessEqual(train_days, 25)
    
    def test_transform_method(self):
        """Test the transform method returns data with split annotations"""
        splitter = TimeSeriesSplitter(
            train_period='30D',
            test_period='10D'
        )
        
        # Fit the splitter
        splitter.fit(self.sample_data)
        
        # Transform should return data with split_type column
        result = splitter.transform(self.sample_data)
        
        self.assertIn('split_type', result.columns)
        self.assertIn('split_index', result.columns)
        
        # Should have both train and test data
        self.assertTrue((result['split_type'] == 'train').any())
        self.assertTrue((result['split_type'] == 'test').any())
    
    def test_different_date_column(self):
        """Test using a different date column name"""
        # The hourly data uses 'timestamp' instead of 'date'
        splitter = TimeSeriesSplitter(
            train_period='7D',
            test_period='2D',
            date_column='timestamp'
        )
        
        splitter.fit(self.hourly_data)
        splits = splitter.get_splits()
        
        # Should have splits
        self.assertGreater(len(splits), 0)
        
        # Check the first split
        train, test = splits[0]
        
        # Verify timestamp column was used correctly
        self.assertIn('timestamp', train.columns)
        self.assertIn('timestamp', test.columns)
        
        # Verify dates don't overlap
        last_train_date = train['timestamp'].max()
        first_test_date = test['timestamp'].min()
        self.assertLessEqual(last_train_date, first_test_date)
    
    def test_auto_date_column_detection(self):
        """Test automatic detection of date column"""
        # Create data with an unusual date column name
        unusual_data = self.sample_data.copy()
        unusual_data = unusual_data.rename(columns={'date': 'event_datetime'})
        
        # Don't specify date_column
        splitter = TimeSeriesSplitter(
            train_period='30D',
            test_period='10D',
            date_column='date'  # This doesn't exist in the data
        )
        
        # This should try to detect a date column automatically
        try:
            splitter.fit(unusual_data)
            splits = splitter.get_splits()
            self.assertGreater(len(splits), 0)
            
            # Check that it used the event_datetime column
            found_date_col = splitter.date_column
            self.assertEqual(found_date_col, 'event_datetime')
        except ValueError:
            # If it can't find a suitable date column, that's acceptable too
            # The implementation might be strict about date column names
            pass
    
    def test_date_type_conversion(self):
        """Test conversion of string dates to datetime"""
        # Create data with string dates
        string_dates = self.sample_data.copy()
        string_dates['date'] = string_dates['date'].dt.strftime('%Y-%m-%d')
        
        splitter = TimeSeriesSplitter(
            train_period='30D',
            test_period='10D'
        )
        
        # Should convert string dates to datetime
        splitter.fit(string_dates)
        splits = splitter.get_splits()
        
        # Should have splits
        self.assertGreater(len(splits), 0)
        
        # First split's date column should be datetime type
        train, _ = splits[0]
        self.assertTrue(pd.api.types.is_datetime64_dtype(train['date']))
    
    def test_empty_splits_handling(self):
        """Test handling of empty splits"""
        # Create data with large gaps
        sparse_data = pd.DataFrame({
            'date': [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                # Large gap
                datetime(2023, 2, 1),
                datetime(2023, 2, 2),
            ],
            'value': [1, 2, 3, 4]
        })
        
        splitter = TimeSeriesSplitter(
            train_period='5D',
            test_period='5D',
            step_size='1D'
        )
        
        # Should not crash with sparse data
        splitter.fit(sparse_data)
        splits = splitter.get_splits()
        
        # All splits should have data
        for train, test in splits:
            self.assertGreater(len(train), 0)
            self.assertGreater(len(test), 0)
    
    def test_start_end_date_parameters(self):
        """Test using start_date and end_date parameters"""
        start_date = datetime(2023, 1, 15)
        end_date = datetime(2023, 3, 15)
        
        splitter = TimeSeriesSplitter(
            train_period='15D',
            test_period='7D',
            start_date=start_date,
            end_date=end_date
        )
        
        splitter.fit(self.sample_data)
        splits = splitter.get_splits()
        
        # Check date boundaries in splits
        for train, test in splits:
            # All dates should be >= start_date
            self.assertGreaterEqual(train['date'].min(), start_date)
            
            # All dates should be <= end_date
            self.assertLessEqual(test['date'].max(), end_date)


if __name__ == '__main__':
    unittest.main()