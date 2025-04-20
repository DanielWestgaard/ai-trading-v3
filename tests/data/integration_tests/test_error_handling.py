import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
import logging
import time
import warnings

from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer
from data.features.feature_selector import FeatureSelector
from data.processors.splitter import TimeSeriesSplitter
from data.pipelines.data_pipeline import DataPipeline
import config.constants.data_config as data_config
import config.constants.system_config as sys_config


class TestErrorHandling(unittest.TestCase):
    """Test cases specifically for error handling in the data pipeline components"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create the testing storage directory if it doesn't exist
        os.makedirs(data_config.TEST_DUMMY_PATH, exist_ok=True)
        
        # Create subdirectories for raw and processed
        self.test_raw_dir = os.path.join(data_config.TEST_DUMMY_PATH, 'raw')
        self.test_processed_dir = os.path.join(data_config.TEST_DUMMY_PATH, 'processed')
        os.makedirs(self.test_raw_dir, exist_ok=True)
        os.makedirs(self.test_processed_dir, exist_ok=True)
        
        # Create a temporary directory for additional test outputs
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
        
        # Override system config paths temporarily for testing
        self.original_raw_dir = sys_config.CAPCOM_RAW_DATA_DIR
        self.original_processed_dir = sys_config.CAPCOM_PROCESSED_DATA_DIR
        sys_config.CAPCOM_RAW_DATA_DIR = self.test_raw_dir
        sys_config.CAPCOM_PROCESSED_DATA_DIR = self.test_processed_dir
        
        # Save test data to file for pipeline tests with a proper parseable filename
        self.test_symbol = "BTCUSD"
        self.test_timeframe = "D1"
        self.test_start_date = "20230101"
        self.test_end_date = "20230720"
        
        self.test_data_filename = f"raw_{self.test_symbol}_{self.test_timeframe}_{self.test_start_date}_{self.test_end_date}.csv"
        self.test_data_path = os.path.join(self.test_raw_dir, self.test_data_filename)
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
        # Restore original config paths
        sys_config.CAPCOM_RAW_DATA_DIR = self.original_raw_dir
        sys_config.CAPCOM_PROCESSED_DATA_DIR = self.original_processed_dir
        
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_missing_critical_columns(self):
        """Test how components handle missing critical columns like price or date"""
        # Create data missing 'Close' column
        missing_close = self.sample_data.copy().drop(columns=['Close'])
        
        # Test FeatureGenerator with missing Close
        try:
            self.feature_generator.fit(missing_close)
            featured_missing_close = self.feature_generator.transform(missing_close)
            # Check if it gracefully handled missing 'Close'
            self.assertIn('high', featured_missing_close.columns.str.lower())
            self.assertNotIn('close', featured_missing_close.columns.str.lower())
            # Check if indicators that depend on Close are absent
            self.assertNotIn('rsi_14', featured_missing_close.columns)
        except Exception as e:
            self.fail(f"Feature generator failed with missing Close column: {str(e)}")
        
        # Create data missing date column
        missing_date = self.sample_data.copy().drop(columns=['Date'])
        
        # Test cleaner with missing date column
        try:
            self.cleaner.fit(missing_date)
            cleaned_missing_date = self.cleaner.transform(missing_date)
            # Should still clean price data even without date
            self.assertEqual(cleaned_missing_date.isna().sum().sum(), 0)
        except Exception as e:
            self.fail(f"Cleaner failed with missing Date column: {str(e)}")
        
        # Test pipeline with missing Volume
        missing_volume = self.sample_data.copy().drop(columns=['Volume'])
        missing_volume_path = os.path.join(self.test_raw_dir, "missing_volume.csv")
        missing_volume.to_csv(missing_volume_path, index=False)
        
        missing_vol_dir = os.path.join(self.test_processed_dir, 'missing_volume')
        os.makedirs(missing_vol_dir, exist_ok=True)
        
        try:
            pipeline = DataPipeline()
            result_df, result_path = pipeline.run(
                target_path=missing_vol_dir,
                raw_data=missing_volume_path,
                run_feature_selection=False
            )
            # Pipeline should complete even without Volume
            self.assertIsInstance(result_df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Pipeline failed with missing Volume column: {str(e)}")
    
    def test_empty_or_small_dataset(self):
        """Test behavior with empty or very small datasets"""
        # Create an empty DataFrame with the same columns
        empty_df = pd.DataFrame(columns=self.sample_data.columns)
        
        # Test feature generator with empty data
        with self.assertRaises(Exception):
            # Should raise an exception as you can't calculate features on empty data
            self.feature_generator.fit(empty_df)
        
        # Create a very small dataset (just 3 rows)
        tiny_df = self.sample_data.iloc[:3].copy()
        tiny_path = os.path.join(self.test_raw_dir, "tiny_dataset.csv")
        tiny_df.to_csv(tiny_path, index=False)
        
        # Test feature selector with tiny dataset
        with self.assertRaises(ValueError):
            # Should raise ValueError because dataset is too small for feature importance
            selector = FeatureSelector(n_splits=2)
            selector.fit(tiny_df)
        
        # Test splitter with tiny dataset
        try:
            # Should create fewer splits than requested, but not fail
            self.splitter.fit(tiny_df)
            splits = self.splitter.get_splits()
            self.assertLessEqual(len(splits), 1)  # Might create 0 or 1 split
        except Exception as e:
            self.fail(f"Splitter failed with tiny dataset: {str(e)}")
    
    def test_nan_heavy_dataset(self):
        """Test with datasets containing high proportions of NaN values"""
        # Create data with heavy NaN presence
        heavy_nan_df = self.sample_data.copy()
        
        # Make 50% of values NaN (excluding Date column)
        mask = np.random.choice([True, False], size=heavy_nan_df.shape, p=[0.5, 0.5])
        mask[:, 0] = False  # Preserve Date column
        heavy_nan_df = heavy_nan_df.mask(mask)
        
        heavy_nan_path = os.path.join(self.test_raw_dir, "heavy_nan.csv")
        heavy_nan_df.to_csv(heavy_nan_path, index=False)
        
        # Test cleaner
        try:
            self.cleaner.fit(heavy_nan_df)
            cleaned_heavy_nan = self.cleaner.transform(heavy_nan_df)
            # Should fill all NaNs
            self.assertEqual(cleaned_heavy_nan.isna().sum().sum(), 0)
        except Exception as e:
            self.fail(f"Cleaner failed with NaN-heavy data: {str(e)}")
        
        # Test complete pipeline
        nan_heavy_dir = os.path.join(self.test_processed_dir, 'nan_heavy')
        os.makedirs(nan_heavy_dir, exist_ok=True)
        
        try:
            pipeline = DataPipeline()
            result_df, result_path = pipeline.run(
                target_path=nan_heavy_dir,
                raw_data=heavy_nan_path,
                run_feature_selection=False
            )
            # Pipeline should complete with filled values
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(result_df.isna().sum().sum(), 0)
        except Exception as e:
            self.fail(f"Pipeline failed with NaN-heavy data: {str(e)}")
    
    def test_corrupted_values(self):
        """Test with inappropriate values in the dataset"""
        # Create data with inappropriate values
        corrupted_df = self.sample_data.copy()
        
        # Add negative prices
        corrupted_df.loc[10, 'Close'] = -50.0
        corrupted_df.loc[11, 'Open'] = -30.0
        
        # Add extremely large values
        corrupted_df.loc[20, 'Volume'] = 1e12
        
        # Add zero values
        corrupted_df.loc[30, 'High'] = 0.0
        corrupted_df.loc[30, 'Low'] = 0.0
        
        corrupted_path = os.path.join(self.test_raw_dir, "corrupted_values.csv")
        corrupted_df.to_csv(corrupted_path, index=False)
        
        # Test preparator's price transforms with negative values
        try:
            self.feature_preparator.price_transform_method = 'log'
            self.feature_preparator.fit(corrupted_df)
            prepared_corrupted = self.feature_preparator.transform(corrupted_df)
            # Should handle log of negative values without error
            self.assertFalse(prepared_corrupted.isna().any().any())
        except Exception as e:
            self.fail(f"Feature preparator failed with corrupted values: {str(e)}")
        
        # Test normalizer with extreme values
        try:
            self.normalizer.fit(corrupted_df)
            normalized_corrupted = self.normalizer.transform(corrupted_df)
            # Should produce finite values always
            self.assertTrue(np.isfinite(normalized_corrupted.select_dtypes(include=['number']).values).all())
        except Exception as e:
            self.fail(f"Normalizer failed with corrupted values: {str(e)}")
        
        # Test cleaner with invalid OHLC relationships
        corrupted_df.loc[40, 'High'] = 50.0
        corrupted_df.loc[40, 'Low'] = 100.0  # Low > High (invalid relationship)
        
        try:
            self.cleaner.ensure_ohlc_validity = True
            self.cleaner.fit(corrupted_df)
            cleaned_corrupted = self.cleaner.transform(corrupted_df)
            # Should fix the invalid relationship
            self.assertGreaterEqual(cleaned_corrupted.loc[40, 'High'], cleaned_corrupted.loc[40, 'Low'])
        except Exception as e:
            self.fail(f"Cleaner failed to fix invalid OHLC relationships: {str(e)}")
    
    def test_edge_case_transformations(self):
        """Test edge cases in data transformations that could cause errors"""
        # Create data with edge cases
        edge_case_df = self.sample_data.copy()
        
        # Case 1: All identical values (zero standard deviation)
        edge_case_df.loc[0:10, 'Close'] = 100.0  # Identical values
        
        # Case 2: First value is zero (causes issues in percent change)
        edge_case_df.loc[0, 'Open'] = 0.0
        
        # Case 3: Sudden extreme value changes (challenges outlier detection)
        edge_case_df.loc[50, 'High'] = edge_case_df.loc[49, 'High'] * 100
        
        edge_case_path = os.path.join(self.test_raw_dir, "edge_cases.csv")
        edge_case_df.to_csv(edge_case_path, index=False)
        
        # Test normalizer with zero variance data
        try:
            self.normalizer.fit(edge_case_df)
            normalized_edge = self.normalizer.transform(edge_case_df)
            # Should not have NaN values even with zero std
            self.assertFalse(normalized_edge.isna().any().any())
        except Exception as e:
            self.fail(f"Normalizer failed with zero variance data: {str(e)}")
        
        # Test feature preparator with zero-starting data
        try:
            self.feature_preparator.price_transform_method = 'pct_change'
            self.feature_preparator.fit(edge_case_df)
            prepared_edge = self.feature_preparator.transform(edge_case_df)
            # Should handle percent change from zero
            self.assertFalse(prepared_edge.isna().any().any())
        except Exception as e:
            self.fail(f"Feature preparator failed with zero-starting data: {str(e)}")
        
        # Test full pipeline with edge cases
        edge_dir = os.path.join(self.test_processed_dir, 'edge_cases')
        os.makedirs(edge_dir, exist_ok=True)
        
        try:
            pipeline = DataPipeline()
            result_df, result_path = pipeline.run(
                target_path=edge_dir,
                raw_data=edge_case_path,
                run_feature_selection=False
            )
            # Pipeline should handle all edge cases
            self.assertIsInstance(result_df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Pipeline failed with edge case data: {str(e)}")
    
    def test_data_type_handling(self):
        """Test handling of different data types and mixed types"""
        # Create data with mixed types
        mixed_types_df = self.sample_data.copy()
        
        # Add string values in numeric columns
        mixed_types_df.loc[5, 'Close'] = 'ERROR'
        mixed_types_df.loc[15, 'Volume'] = 'NA'
        
        # Add boolean column
        mixed_types_df['IsTradingDay'] = True
        mixed_types_df.loc[::7, 'IsTradingDay'] = False  # Every 7th day is not a trading day
        
        # Add mixed date formats
        mixed_types_df.loc[25, 'Date'] = '01-15-2023'  # Different format
        
        mixed_path = os.path.join(self.test_raw_dir, "mixed_types.csv")
        mixed_types_df.to_csv(mixed_path, index=False)
        
        # Test cleaner with mixed types - should handle type conversions
        try:
            test_cleaner = DataCleaner()
            # Suppress warnings during this test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_cleaner.fit(mixed_types_df)
                cleaned_mixed = test_cleaner.transform(mixed_types_df)
            
            # After cleaning, should have numeric values where needed
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_mixed['Close']))
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_mixed['Volume']))
        except Exception as e:
            self.fail(f"Cleaner failed with mixed types: {str(e)}")
        
        # Test pipeline with mixed data types
        mixed_dir = os.path.join(self.test_processed_dir, 'mixed_types')
        os.makedirs(mixed_dir, exist_ok=True)
        
        try:
            pipeline = DataPipeline()
            # Suppress warnings during pipeline run
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Expect pipeline to handle type conversions
                result_df, result_path = pipeline.run(
                    target_path=mixed_dir,
                    raw_data=mixed_path,
                    run_feature_selection=False
                )
            # Should produce a valid result
            self.assertIsInstance(result_df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Pipeline failed with mixed data types: {str(e)}")
    
    def test_inconsistent_column_case(self):
        """Test handling of inconsistent column casing"""
        # Create data with mixed case columns
        mixed_case_df = self.sample_data.copy()
        
        # Rename columns with mixed case
        mixed_case_df = mixed_case_df.rename(columns={
            'Open': 'OPEN',
            'High': 'high',
            'Low': 'LOW',
            'Close': 'close',
            'Volume': 'Volume'
        })
        
        mixed_case_path = os.path.join(self.test_raw_dir, "mixed_case.csv")
        mixed_case_df.to_csv(mixed_case_path, index=False)
        
        # Test components with mixed case columns
        try:
            # Feature generator should handle mixed case
            self.feature_generator.fit(mixed_case_df)
            featured_mixed = self.feature_generator.transform(mixed_case_df)
            
            # Should have technical indicators regardless of original case
            self.assertIn('rsi_14', featured_mixed.columns)
            
            # Pipeline should handle mixed case
            mixed_case_dir = os.path.join(self.test_processed_dir, 'mixed_case')
            os.makedirs(mixed_case_dir, exist_ok=True)
            
            pipeline = DataPipeline()
            result_df, result_path = pipeline.run(
                target_path=mixed_case_dir,
                raw_data=mixed_case_path,
                run_feature_selection=False
            )
            self.assertIsInstance(result_df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Failed with inconsistent column case: {str(e)}")
    
    def test_feature_selector_robustness(self):
        """Test feature selector with challenging data conditions"""
        # Create data with perfect multicollinearity (extreme correlation)
        collinear_df = self.sample_data.copy()
        
        # Add perfectly correlated features
        collinear_df['Close_Copy'] = collinear_df['Close']
        collinear_df['Close_Noise'] = collinear_df['Close'] + np.random.normal(0, 0.1, len(collinear_df))
        collinear_df['Close_Scaled'] = collinear_df['Close'] * 2
        
        # Create a feature with no variation
        collinear_df['Constant'] = 100
        
        collinear_path = os.path.join(self.test_raw_dir, "collinear.csv")
        collinear_df.to_csv(collinear_path, index=False)
        
        # Test Feature Selector
        try:
            # Add a target column for feature importance analysis
            collinear_df['close_return'] = collinear_df['Close'].pct_change().shift(-1).fillna(0)
            
            # Create feature selector with a small number of splits for speed
            selector = FeatureSelector(
                n_splits=2,
                target_col='close_return',
                preserve_target=True,
                selection_method='threshold',
                importance_threshold=0.01
            )
            
            selector.fit(collinear_df)
            selected_df = selector.transform(collinear_df)
            
            # Verify it can handle perfect collinearity
            self.assertIsInstance(selected_df, pd.DataFrame)
            
            # Check if constant column was removed
            self.assertNotIn('Constant', selected_df.columns)
            
        except Exception as e:
            self.fail(f"Feature selector failed with collinear data: {str(e)}")
    
    def test_recovering_from_component_failure(self):
        """Test pipeline's ability to recover when components partially fail"""
        # Create a problematic dataset
        problem_df = self.sample_data.copy()
        
        # Add a few NaN values to test recovery
        problem_df.loc[0, 'Close'] = np.nan
        problem_df.loc[1, 'Open'] = np.nan
        problem_df.loc[2, 'High'] = np.nan
        
        # Add incompatible values
        problem_df.loc[10, 'Volume'] = -100  # Negative volume
        
        problem_path = os.path.join(self.test_raw_dir, "problem_data.csv")
        problem_df.to_csv(problem_path, index=False)
        
        # Test feature preparator's recovery from NaNs
        try:
            self.feature_generator.fit(problem_df)
            featured_problem = self.feature_generator.transform(problem_df)
            
            # Feature preparator should handle NaNs that were propagated
            self.feature_preparator.fit(featured_problem)
            prepared_problem = self.feature_preparator.transform(featured_problem)
            
            # Should have no NaNs after preparation
            self.assertEqual(prepared_problem.isna().sum().sum(), 0)
            
        except Exception as e:
            self.fail(f"Components failed to recover from problematic data: {str(e)}")
    
    def test_handling_out_of_bounds_parameters(self):
        """Test components with out-of-bounds parameters"""
        # Test DateCleaner with extreme outlier threshold
        try:
            extreme_cleaner = DataCleaner(outlier_threshold=100.0)  # Very high threshold
            extreme_cleaner.fit(self.sample_data)
            cleaned_extreme = extreme_cleaner.transform(self.sample_data)
            self.assertIsInstance(cleaned_extreme, pd.DataFrame)
        except Exception as e:
            self.fail(f"Cleaner failed with extreme outlier threshold: {str(e)}")
        
        # Test feature preparator with extreme parameters
        try:
            extreme_preparator = FeaturePreparator(min_data_points=5000)  # More rows than we have
            extreme_preparator.fit(self.sample_data)
            # Should handle this gracefully and use as much data as possible
            prepared_extreme = extreme_preparator.transform(self.sample_data)
            self.assertIsInstance(prepared_extreme, pd.DataFrame)
        except Exception as e:
            self.fail(f"Feature preparator failed with extreme min_data_points: {str(e)}")
    
    def test_multiple_component_failures(self):
        """Test pipeline resilience when multiple components have issues"""
        # Create particularly challenging data
        challenging_df = self.sample_data.copy()
        
        # 1. Add mixed data types
        challenging_df.loc[5, 'Close'] = 'ERROR'
        
        # 2. Add NaNs
        challenging_df.loc[10:15, 'Open'] = np.nan
        
        # 3. Add outliers
        challenging_df.loc[20, 'High'] = challenging_df['High'].mean() * 10
        
        # 4. Add invalid OHLC relationship
        challenging_df.loc[25, 'Low'] = challenging_df.loc[25, 'High'] * 1.5
        
        challenging_path = os.path.join(self.test_raw_dir, "challenging.csv")
        challenging_df.to_csv(challenging_path, index=False)
        
        # Test if pipeline can handle all of these issues together
        challenging_dir = os.path.join(self.test_processed_dir, 'challenging')
        os.makedirs(challenging_dir, exist_ok=True)
        
        try:
            # Suppress warnings for this test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                pipeline = DataPipeline()
                result_df, result_path = pipeline.run(
                    target_path=challenging_dir,
                    raw_data=challenging_path,
                    run_feature_selection=False
                )
            
            # Pipeline should overcome all issues
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(result_df.isna().sum().sum(), 0)  # No NaNs
            self.assertTrue(pd.api.types.is_numeric_dtype(result_df.select_dtypes(include=['number'])))  # All numeric data is numeric
            
        except Exception as e:
            self.fail(f"Pipeline failed with multiple challenges: {str(e)}")


if __name__ == '__main__':
    unittest.main()