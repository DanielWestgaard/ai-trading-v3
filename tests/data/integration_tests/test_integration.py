import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta

from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer
from data.features.feature_selector import FeatureSelector
from data.processors.splitter import TimeSeriesSplitter
from data.pipelines.data_pipeline import DataPipeline


class TestIntegration(unittest.TestCase):
    """Integration tests for verifying component interactions in the data pipeline"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create realistic OHLCV data for testing with price trends and patterns
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1D')
        
        np.random.seed(42)  # For reproducibility
        
        # Create price data with trends, cycles and some patterns
        price_trend = np.cumsum(np.random.normal(0.05, 1, 200))  # Trending component
        price_cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 200))  # Cyclical component
        price_noise = np.random.normal(0, 2, 200)  # Random noise
        
        # Combine components for a realistic price series
        close_prices = 100 + price_trend + price_cycle + price_noise
        
        # Create realistic OHLC data with proper relationships
        opens = close_prices.copy()
        opens[1:] = close_prices[:-1]  # Open is previous close
        opens[0] = 100  # First open
        
        # Add some random intraday movement to create high and low
        intraday_range = np.abs(np.random.normal(0, 2, 200))
        highs = np.maximum(opens, close_prices) + intraday_range
        lows = np.minimum(opens, close_prices) - intraday_range
        
        # Create volume with some correlation to price movement
        price_changes = np.abs(close_prices[1:] - close_prices[:-1])
        volume = 1000 + price_changes * 100 + np.random.normal(0, 500, 199)
        volume = np.append(volume, 1000)  # Add volume for first day
        
        # Create DataFrame
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volume
        })
        
        # Save test data to file for pipeline tests
        os.makedirs(os.path.join(self.temp_dir.name, 'raw'), exist_ok=True)
        self.test_data_path = os.path.join(self.temp_dir.name, 'raw', 'BTCUSD_D1_20230101_20230720.csv')
        self.sample_data.to_csv(self.test_data_path, index=False)
        
        # Initialize components
        self.cleaner = DataCleaner()
        self.feature_generator = FeatureGenerator()
        self.feature_preparator = FeaturePreparator(min_data_points=10)  # Lower threshold for test data
        self.normalizer = DataNormalizer()
        self.feature_selector = FeatureSelector(save_visualizations=False)
        self.splitter = TimeSeriesSplitter(train_period='30D', test_period='10D')
    
    def tearDown(self):
        """Clean up after each test method"""
        self.temp_dir.cleanup()
    
    def test_cleaner_to_feature_generator(self):
        """Test that FeatureGenerator can properly handle data cleaned by DataCleaner"""
        # 1. Clean the data
        self.cleaner.fit(self.sample_data)
        cleaned_data = self.cleaner.transform(self.sample_data)
        
        # Verify cleaned data structure
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(len(cleaned_data), len(self.sample_data))
        self.assertEqual(cleaned_data.isna().sum().sum(), 0)  # No NaNs
        
        # 2. Generate features from cleaned data
        self.feature_generator.fit(cleaned_data)
        featured_data = self.feature_generator.transform(cleaned_data)
        
        # Verify featured data structure
        self.assertIsInstance(featured_data, pd.DataFrame)
        self.assertEqual(len(featured_data), len(cleaned_data))
        self.assertGreater(len(featured_data.columns), len(cleaned_data.columns))
        
        # Verify essential technical indicators were created
        expected_features = ['sma_5', 'rsi_14', 'macd', 'bollinger_upper', 'atr_14']
        for feature in expected_features:
            self.assertIn(feature, featured_data.columns)
    
    def test_feature_generator_to_preparator(self):
        """Test that FeaturePreparator correctly processes features from FeatureGenerator"""
        # 1. Generate features
        self.feature_generator.fit(self.sample_data)
        featured_data = self.feature_generator.transform(self.sample_data)
        
        # 2. Prepare features
        self.feature_preparator.fit(featured_data)
        prepared_data = self.feature_preparator.transform(featured_data)
        
        # Verify prepared data structure
        self.assertIsInstance(prepared_data, pd.DataFrame)
        self.assertEqual(prepared_data.isna().sum().sum(), 0)  # No NaNs after preparation
        
        # Verify price transformations
        for price_col in ['open', 'close', 'high', 'low']:
            self.assertIn(f'{price_col}_return', prepared_data.columns)
            self.assertIn(f'{price_col}_raw', prepared_data.columns)
            self.assertIn(f'{price_col}_original', prepared_data.columns)
    
    def test_preparator_to_normalizer(self):
        """Test that DataNormalizer correctly normalizes prepared features"""
        # 1. Generate and prepare features
        self.feature_generator.fit(self.sample_data)
        featured_data = self.feature_generator.transform(self.sample_data)
        
        self.feature_preparator.fit(featured_data)
        prepared_data = self.feature_preparator.transform(featured_data)
        
        # 2. Normalize the prepared data
        self.normalizer.fit(prepared_data)
        normalized_data = self.normalizer.transform(prepared_data)
        
        # Verify normalized data structure
        self.assertIsInstance(normalized_data, pd.DataFrame)
        self.assertEqual(len(normalized_data), len(prepared_data))
        self.assertEqual(normalized_data.isna().sum().sum(), 0)  # No NaNs
        
        # Verify raw price columns are preserved
        for price_col in ['open_raw', 'close_raw', 'high_raw', 'low_raw']:
            if price_col in prepared_data.columns:
                self.assertIn(price_col, normalized_data.columns)
                # Raw columns should be identical before and after normalization
                pd.testing.assert_series_equal(
                    normalized_data[price_col],
                    prepared_data[price_col]
                )
    
    def test_normalizer_to_feature_selector(self):
        """Test that FeatureSelector correctly selects important features from normalized data"""
        # 1. Generate, prepare, and normalize data
        self.feature_generator.fit(self.sample_data)
        featured_data = self.feature_generator.transform(self.sample_data)
        
        self.feature_preparator.fit(featured_data)
        prepared_data = self.feature_preparator.transform(featured_data)
        
        self.normalizer.fit(prepared_data)
        normalized_data = self.normalizer.transform(prepared_data)
        
        # 2. Select features
        # Configure selector with a smaller number of splits for faster testing
        self.feature_selector = FeatureSelector(
            n_splits=2,
            save_visualizations=False,
            selection_method='threshold',
            importance_threshold=0.01
        )
        self.feature_selector.fit(normalized_data)
        selected_data = self.feature_selector.transform(normalized_data)
        
        # Verify selected data structure
        self.assertIsInstance(selected_data, pd.DataFrame)
        self.assertLessEqual(len(selected_data.columns), len(normalized_data.columns))
        self.assertEqual(len(selected_data), len(normalized_data))
        
        # Verify feature importance analysis was performed
        self.assertIsNotNone(self.feature_selector.importance_df)
        self.assertGreater(len(self.feature_selector.importance_df), 0)
        
        # Verify essential columns are always included
        essential_columns = ['open_raw', 'high_raw', 'low_raw', 'close_raw']
        for col in essential_columns:
            if col in normalized_data.columns:
                self.assertIn(col, selected_data.columns)
    
    def test_splitter_with_feature_selection(self):
        """Test that TimeSeriesSplitter works correctly with FeatureSelector for time-based importance analysis"""
        # Set up the splitter with small periods for test data
        splitter = TimeSeriesSplitter(
            train_period='30D',
            test_period='10D',
            date_column='Date'
        )
        
        # 1. Split the data
        splitter.fit(self.sample_data)
        splits = splitter.get_splits()
        
        # Verify splits
        self.assertGreater(len(splits), 0)
        
        # 2. Take the first split for demonstration
        train_data, test_data = splits[0]
        
        # 3. Generate features on the training data
        self.feature_generator.fit(train_data)
        train_featured = self.feature_generator.transform(train_data)
        
        # Create the target column (close_return) by calculating future returns
        # This is typically done by FeaturePreparator but we're using FeatureGenerator directly
        if 'Close' in train_featured.columns:
            train_featured['close_return'] = train_featured['Close'].pct_change(1).shift(-1)
            # Fill the last row's NaN with 0 or the mean
            train_featured['close_return'] = train_featured['close_return'].fillna(0)
        elif 'close' in train_featured.columns:
            train_featured['close_return'] = train_featured['close'].pct_change(1).shift(-1)
            train_featured['close_return'] = train_featured['close_return'].fillna(0)
        
        # 4. Run feature selection on training data
        selector = FeatureSelector(
            n_splits=2,
            save_visualizations=False,
            # Specify a target column that exists and a minimum number of features
            target_col='close_return',
            min_features=3,
            max_features=10
        )
        
        # Verify feature selection runs without errors on the training split
        try:
            selector.fit(train_featured)
            train_selected = selector.transform(train_featured)
            
            # Verify selected features are available
            self.assertIsNotNone(selector.selected_features)
            self.assertGreater(len(selector.selected_features), 0)
            
            # 5. Apply the same feature selection to test data
            test_featured = self.feature_generator.transform(test_data)
            
            # Create the same target column in test data
            if 'Close' in test_featured.columns:
                test_featured['close_return'] = test_featured['Close'].pct_change(1).shift(-1)
                test_featured['close_return'] = test_featured['close_return'].fillna(0)
            elif 'close' in test_featured.columns:
                test_featured['close_return'] = test_featured['close'].pct_change(1).shift(-1)
                test_featured['close_return'] = test_featured['close_return'].fillna(0)
            
            test_selected = selector.transform(test_featured)
            
            # Verify that the test data has the same selected features
            self.assertEqual(set(train_selected.columns), set(test_selected.columns))
            
        except Exception as e:
            self.fail(f"Feature selection on time series split failed with error: {str(e)}")
    
    def test_missing_data_resilience(self):
        """Test resilience of the pipeline to missing data"""
        # Create data with intentional missing values
        missing_data = self.sample_data.copy()
        
        # Add missing values in different patterns
        # 1. Random missing values - but not too many
        random_missing = np.random.choice([True, False], size=missing_data.shape, p=[0.03, 0.97])
        random_missing[:, 0] = False  # Don't remove date values
        missing_data = missing_data.mask(random_missing)
        
        # 2. Chunks of missing values
        missing_data.loc[50:55, 'Close'] = np.nan
        missing_data.loc[30:35, 'High'] = np.nan
        missing_data.loc[70:75, 'Volume'] = np.nan
        
        # Save missing data to file
        missing_path = os.path.join(self.temp_dir.name, 'raw', 'missing_data.csv')
        missing_data.to_csv(missing_path, index=False)
        
        # Run pipeline with different treatment modes
        for mode in ['basic', 'advanced', 'hybrid']:
            # Create pipeline with this treatment mode
            pipeline = DataPipeline(feature_treatment_mode=mode)
            pipeline.feature_preparator.min_data_points = 10  # Lower threshold for test data
            
            # Run pipeline
            try:
                result_df, result_path = pipeline.run(
                    target_path=os.path.join(self.temp_dir.name, f'missing_{mode}'),
                    raw_data=missing_path,
                    save_intermediate=False,
                    run_feature_selection=False
                )
                
                # Verify result has no missing values
                self.assertEqual(result_df.isna().sum().sum(), 0, 
                                f"Treatment mode '{mode}' left missing values")
                
                # Verify we have a reasonable amount of data left
                self.assertGreater(len(result_df), 10, 
                                  f"Treatment mode '{mode}' removed too much data")
                
            except Exception as e:
                self.fail(f"Pipeline failed with missing data using '{mode}' mode: {str(e)}")
    
    def test_outlier_handling(self):
        """Test handling of outliers throughout the pipeline"""
        # Create data with outliers
        outlier_data = self.sample_data.copy()
        
        # Add outliers to price and volume
        outlier_data.loc[50, 'Close'] = outlier_data['Close'].mean() + 10 * outlier_data['Close'].std()
        outlier_data.loc[75, 'Open'] = outlier_data['Open'].mean() - 8 * outlier_data['Open'].std()
        outlier_data.loc[100, 'Volume'] = outlier_data['Volume'].mean() + 15 * outlier_data['Volume'].std()
        
        # Ensure High/Low reflect the outliers
        outlier_data.loc[50, 'High'] = outlier_data.loc[50, 'Close'] + 1
        outlier_data.loc[75, 'Low'] = outlier_data.loc[75, 'Open'] - 1
        
        # Save outlier data to file
        outlier_path = os.path.join(self.temp_dir.name, 'raw', 'outlier_data.csv')
        outlier_data.to_csv(outlier_path, index=False)
        
        # Run pipeline with outlier handling
        pipeline = DataPipeline()
        pipeline.cleaner = DataCleaner(
            outlier_method='zscore',
            outlier_threshold=3.0,
            handle_volume_outliers=True
        )
        pipeline.feature_preparator.min_data_points = 10  # Lower threshold for test data
        
        # Run pipeline
        result_df, result_path = pipeline.run(
            target_path=os.path.join(self.temp_dir.name, 'outliers'),
            raw_data=outlier_path,
            save_intermediate=True,
            run_feature_selection=False
        )
        
        # Load the cleaned data to check if outliers were handled
        cleaned_file = os.path.join(self.temp_dir.name, 'outliers', 'clean', 'clean_BTCUSD_D1_20230101_20230720.csv')
        if os.path.exists(cleaned_file):
            cleaned_df = pd.read_csv(cleaned_file)
            
            # Verify outliers were mitigated
            # Check if the extreme values were reduced
            self.assertLess(cleaned_df.loc[50, 'Close'], outlier_data.loc[50, 'Close'])
            self.assertGreater(cleaned_df.loc[75, 'Open'], outlier_data.loc[75, 'Open'])
            
            # If volume outlier handling is enabled, also check volume
            if pipeline.cleaner.handle_volume_outliers:
                self.assertLess(cleaned_df.loc[100, 'Volume'], outlier_data.loc[100, 'Volume'])
    
    def test_nonstandard_column_names(self):
        """Test that the pipeline can handle non-standard column names"""
        # Create data with non-standard column names
        nonstandard_data = self.sample_data.copy()
        nonstandard_data = nonstandard_data.rename(columns={
            'Date': 'Timestamp',
            'Open': 'price_open',
            'High': 'price_high',
            'Low': 'price_low',
            'Close': 'price_close',
            'Volume': 'trading_volume'
        })
        
        # Create pipeline with custom column names
        pipeline = DataPipeline(
            timestamp_column='Timestamp'  # Use the constructor parameter
        )
        pipeline.cleaner = DataCleaner(
            price_cols=['price_open', 'price_high', 'price_low', 'price_close'],
            volume_col='trading_volume',
            timestamp_col='Timestamp'
        )
        pipeline.feature_generator = FeatureGenerator(
            price_cols=['price_open', 'price_high', 'price_low', 'price_close'],
            volume_col='trading_volume',
            timestamp_col='Timestamp'
        )
        pipeline.feature_preparator = FeaturePreparator(
            price_cols=['price_open', 'price_high', 'price_low', 'price_close'],
            volume_col='trading_volume',
            timestamp_col='Timestamp',
            min_data_points=10
        )
        pipeline.normalizer = DataNormalizer(
            price_cols=['price_open', 'price_high', 'price_low', 'price_close'],
            volume_col='trading_volume'
        )
        
        # Save data to test file
        nonstandard_path = os.path.join(self.temp_dir.name, 'raw', 'nonstandard_columns.csv')
        nonstandard_data.to_csv(nonstandard_path, index=False)
        
        # Run pipeline
        try:
            result_df, result_path = pipeline.run(
                target_path=os.path.join(self.temp_dir.name, 'nonstandard'),
                raw_data=nonstandard_path,
                save_intermediate=False,
                run_feature_selection=False
            )
            
            # Verify result structure
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertGreater(len(result_df.columns), len(nonstandard_data.columns))
            
            # Verify some expected derived columns exist
            self.assertTrue(any('return' in col.lower() for col in result_df.columns))
                
        except Exception as e:
            self.fail(f"Pipeline failed with non-standard column names: {str(e)}")
    
    def test_pipeline_with_custom_configurations(self):
        """Test pipeline with custom configurations for all components"""
        # Configure a highly customized pipeline
        pipeline = DataPipeline(
            feature_treatment_mode='hybrid',
            price_transform_method='log',
            normalization_method='minmax',
            feature_selection_method='top_n',
            feature_importance_threshold=0.05,
            target_column='open_return'
        )
        
        # Customize each component
        pipeline.cleaner = DataCleaner(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
            missing_method='interpolate',
            outlier_method='iqr',
            outlier_threshold=2.0,
            handle_volume_outliers=True
        )
        
        pipeline.feature_generator = FeatureGenerator(
            preserve_original_case=True
        )
        
        pipeline.feature_preparator = FeaturePreparator(
            preserve_original_prices=True,
            price_transform_method='log',
            treatment_mode='hybrid',
            min_data_points=10
        )
        
        pipeline.normalizer = DataNormalizer(
            price_method='minmax',
            volume_method='minmax',
            other_method='minmax'
        )
        
        # Run pipeline
        try:
            result_df, result_path = pipeline.run(
                target_path=os.path.join(self.temp_dir.name, 'custom'),
                raw_data=self.test_data_path,
                save_intermediate=True,
                run_feature_selection=False
            )
            
            # Verify result structure
            self.assertIsInstance(result_df, pd.DataFrame)
            
            # Check for log-transformed price columns
            self.assertTrue(any('log' in col.lower() for col in result_df.columns))
            
        except Exception as e:
            self.fail(f"Pipeline failed with custom configurations: {str(e)}")


if __name__ == '__main__':
    unittest.main()