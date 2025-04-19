import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the class we're testing
from data.pipelines.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Test suite for the DataPipeline class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample OHLCV data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1D')
        
        np.random.seed(42)  # For reproducibility
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 5, 100).cumsum(),
            'High': np.random.normal(102, 5, 100).cumsum(),
            'Low': np.random.normal(98, 5, 100).cumsum(),
            'Close': np.random.normal(101, 5, 100).cumsum(),
            'Volume': np.random.randint(1000, 5000, 100)
        })
        
        # Add a close_return column to simulate what would be created during processing
        self.sample_data['close_return'] = self.sample_data['Close'].pct_change()
        self.sample_data['high_return'] = self.sample_data['High'].pct_change()
        
        # Ensure High is always >= Open, Close and Low is always <= Open, Close
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.loc[i, 'Open'], self.sample_data.loc[i, 'Close'], self.sample_data.loc[i, 'High'])
            low = min(self.sample_data.loc[i, 'Open'], self.sample_data.loc[i, 'Close'], self.sample_data.loc[i, 'Low'])
            self.sample_data.loc[i, 'High'] = high
            self.sample_data.loc[i, 'Low'] = low
        
        # Create directories
        self.data_dir = os.path.join(self.temp_dir.name, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Save sample data to a raw data file with a filename that contains metadata
        # Use a proper parseable filename format
        self.sample_data_path = os.path.join(self.raw_dir, 'raw_EURUSD_D1_20230101_20230410.csv')
        self.sample_data.to_csv(self.sample_data_path, index=False)
        
        # Setup metadata dictionary that will be returned by the mocked extract_file_metadata
        self.fake_metadata = {
            'instrument': 'EURUSD',
            'timeframe': 'D1',
            'start_date': '20230101',
            'end_date': '20230410',
            'is_raw': True,
            'is_processed': False,
            'is_meta': False,
            'base_name': 'EURUSD_D1_20230101_20230410'
        }
        
        # Create patcher for extract_file_metadata and all related functions
        self.metadata_patcher = patch('utils.data_utils.extract_file_metadata', 
                                      return_value=self.fake_metadata)
        self.mock_extract_metadata = self.metadata_patcher.start()
        
        # Also patch generate_derived_filename to ensure it doesn't rely on extract_file_metadata
        def mock_generate_derived_filename(processed_file, file_type, extension='csv'):
            return f"{file_type}_{self.fake_metadata['base_name']}.{extension}"
        
        self.derived_filename_patcher = patch('utils.data_utils.generate_derived_filename', 
                                             side_effect=mock_generate_derived_filename)
        self.mock_derive_filename = self.derived_filename_patcher.start()
        
        # Also patch get_derived_file_path
        def mock_get_derived_file_path(processed_file, file_type, base_dir=None, sub_dir=None, extension='csv'):
            if not base_dir:
                base_dir = os.path.dirname(processed_file)
            
            # Create directory path
            if sub_dir:
                dir_path = os.path.join(base_dir, sub_dir)
            else:
                dir_path = base_dir
            
            # Ensure the directory exists
            os.makedirs(dir_path, exist_ok=True)
            
            # Return the full file path
            filename = f"{file_type}_{self.fake_metadata['base_name']}.{extension}"
            return os.path.join(dir_path, filename)
        
        self.derived_path_patcher = patch('utils.data_utils.get_derived_file_path', 
                                        side_effect=mock_get_derived_file_path)
        self.mock_derive_path = self.derived_path_patcher.start()
        
        # Create a sample feature metadata dataframe for mocking
        self.feature_metadata_df = pd.DataFrame({
            'column': ['close', 'high', 'low', 'open', 'volume', 'close_return'],
            'category': ['price', 'price', 'price', 'price', 'volume', 'returns'],
            'nan_count': [0, 0, 0, 0, 0, 0],
            'nan_pct': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        # Create a sample processed dataframe to return from transforms
        self.processed_sample = self.sample_data.copy()
        # Add some feature columns to simulate processing
        self.processed_sample['sma_5'] = self.processed_sample['Close'].rolling(5).mean()
        self.processed_sample['rsi_14'] = np.random.uniform(0, 100, len(self.processed_sample))
        self.processed_sample['open_raw'] = self.processed_sample['Open']
        self.processed_sample['high_raw'] = self.processed_sample['High']
        self.processed_sample['low_raw'] = self.processed_sample['Low']
        self.processed_sample['close_raw'] = self.processed_sample['Close']
    
    def tearDown(self):
        """Clean up after each test method"""
        self.metadata_patcher.stop()
        self.derived_filename_patcher.stop()
        self.derived_path_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the pipeline initializes with correct parameters"""
        # Test default initialization
        pipeline = DataPipeline()
        self.assertEqual(pipeline.feature_preparator.treatment_mode, 'advanced')
        self.assertEqual(pipeline.feature_preparator.price_transform_method, 'returns')
        self.assertEqual(pipeline.normalizer.other_method, 'zscore')
        self.assertEqual(pipeline.feature_selection_method, 'threshold')
        self.assertEqual(pipeline.feature_importance_threshold, 0.01)
        self.assertEqual(pipeline.target_column, 'close_return')
        self.assertTrue(pipeline.preserve_target)
        
        # Test custom initialization
        custom_pipeline = DataPipeline(
            feature_treatment_mode='basic',
            price_transform_method='log',
            normalization_method='minmax',
            feature_selection_method='top_n',
            feature_importance_threshold=0.05,
            target_column='open_return',
            preserve_target=False
        )
        self.assertEqual(custom_pipeline.feature_preparator.treatment_mode, 'basic')
        self.assertEqual(custom_pipeline.feature_preparator.price_transform_method, 'log')
        self.assertEqual(custom_pipeline.normalizer.other_method, 'minmax')
        self.assertEqual(custom_pipeline.feature_selection_method, 'top_n')
        self.assertEqual(custom_pipeline.feature_importance_threshold, 0.05)
        self.assertEqual(custom_pipeline.target_column, 'open_return')
        self.assertFalse(custom_pipeline.preserve_target)
    
    def test_run_minimal(self):
        """Test running the pipeline with minimal options"""
        # Create pipeline with small min_data_points to accommodate test data
        pipeline = DataPipeline()
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Raw data needs to be a path to a CSV file containing the Date column
        # We're using self.sample_data_path which contains our test data
        
        # Run pipeline with minimal configuration
        with patch('pandas.read_csv', return_value=self.sample_data), \
             patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df):
            
            result_df, result_path = pipeline.run(
                target_path=self.processed_dir,
                raw_data=self.sample_data_path,
                save_intermediate=False,
                run_feature_selection=False
            )
            
            # Check that we got results
            self.assertIsNotNone(result_df)
            self.assertIsNotNone(result_path)
            self.assertTrue(isinstance(result_df, pd.DataFrame))
            
            # Basic sanity check on result data
            self.assertGreater(len(result_df), 0)
    
    def test_run_with_intermediate(self):
        """Test running the pipeline with saving intermediate results"""
        # Create pipeline with small min_data_points
        pipeline = DataPipeline()
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Run pipeline with intermediate saving
        with patch('pandas.read_csv', return_value=self.sample_data), \
             patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df):
            
            result_df, result_path = pipeline.run(
                target_path=self.processed_dir,
                raw_data=self.sample_data_path,
                save_intermediate=True,
                run_feature_selection=False
            )
            
            # Check that we got results
            self.assertIsNotNone(result_df)
            self.assertIsNotNone(result_path)
            
            # Check that intermediate directories were created
            expected_dirs = ['clean', 'features', 'prepared', 'normalized']
            for dir_name in expected_dirs:
                dir_path = os.path.join(self.processed_dir, dir_name)
                self.assertTrue(os.path.exists(dir_path), f"Directory {dir_path} does not exist")
    
    def test_run_with_feature_selection(self):
        """Test running the pipeline with feature selection"""
        # Create pipeline with small min_data_points
        pipeline = DataPipeline()
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Create a list for selected features that will survive the mocking process
        selected_features = [
            'open_raw', 'high_raw', 'low_raw', 'close_raw',
            'open_original', 'high_original', 'low_original', 'close_original',
            'close_return', 'sma_5', 'sma_10', 'rsi_14'
        ]
        
        # Create mock for feature selector
        mock_selector = MagicMock()
        mock_selector.selected_features = selected_features
        
        # Setup mocks to properly track state through the pipeline
        with patch('pandas.read_csv', return_value=self.processed_sample), \
             patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df), \
             patch('data.features.feature_selector.FeatureSelector.fit', return_value=mock_selector) as mock_fit, \
             patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform, \
             patch('data.features.feature_selector.FeatureSelector.__init__', return_value=None) as mock_init:
            
            # Make sure feature_selector is properly set on the pipeline
            def set_mock_selector(*args, **kwargs):
                pipeline.feature_selector = mock_selector
                return mock_selector
            
            mock_fit.side_effect = set_mock_selector
            
            # For transform, return a non-empty subset
            def transform_side_effect(df):
                # Create a non-empty result dataframe 
                result = pd.DataFrame()
                
                # Add at least some columns that exist in the input dataframe
                existing_cols = [col for col in selected_features if col in df.columns]
                if existing_cols:
                    result = df[existing_cols].copy()
                
                # If target column exists, ensure it's included
                if 'close_return' in df.columns and 'close_return' not in result.columns:
                    result['close_return'] = df['close_return']
                
                # Ensure we're not returning an empty dataframe
                if len(result) == 0 or len(result.columns) == 0:
                    # Fallback to some basic columns
                    result = df[['Date', 'Open', 'Close', 'close_return']].copy()
                
                return result
            
            mock_transform.side_effect = transform_side_effect
            
            # Run pipeline with feature selection
            result_df, result_path = pipeline.run(
                target_path=self.processed_dir,
                raw_data=self.sample_data_path,
                save_intermediate=True,
                run_feature_selection=True
            )
        
        # Check that we got results
        self.assertIsNotNone(result_df)
        self.assertIsNotNone(result_path)
        self.assertGreater(len(result_df), 0)
        
        # Check that the features directory was created
        features_dir = os.path.join(self.processed_dir, 'features')
        self.assertTrue(os.path.exists(features_dir))
    
    def test_different_treatment_modes(self):
        """Test running the pipeline with different feature treatment modes"""
        # Test all treatment modes
        for mode in ['basic', 'advanced', 'hybrid']:
            # Create pipeline with this mode and small min_data_points
            pipeline = DataPipeline(feature_treatment_mode=mode)
            pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
            
            # Create a unique target directory for this test
            target_dir = os.path.join(self.processed_dir, f"treatment_{mode}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Run pipeline
            with patch('pandas.read_csv', return_value=self.sample_data), \
                 patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df):
                
                result_df, result_path = pipeline.run(
                    target_path=target_dir,
                    raw_data=self.sample_data_path,
                    save_intermediate=False,
                    run_feature_selection=False
                )
                
                # Check that we got results
                self.assertIsNotNone(result_df)
                self.assertIsNotNone(result_path)
                self.assertTrue(isinstance(result_df, pd.DataFrame))
                
                # Basic sanity check on result data
                self.assertGreater(len(result_df), 0)
    
    def test_different_price_transforms(self):
        """Test running the pipeline with different price transformation methods"""
        # Test all price transform methods
        for method in ['returns', 'log', 'pct_change', 'multi']:
            # Create pipeline with this method and small min_data_points
            pipeline = DataPipeline(price_transform_method=method)
            pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
            
            # Create a unique target directory for this test
            target_dir = os.path.join(self.processed_dir, f"transform_{method}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Run pipeline
            with patch('pandas.read_csv', return_value=self.sample_data), \
                 patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df):
                
                result_df, result_path = pipeline.run(
                    target_path=target_dir,
                    raw_data=self.sample_data_path,
                    save_intermediate=False,
                    run_feature_selection=False
                )
                
                # Check that we got results
                self.assertIsNotNone(result_df)
                self.assertIsNotNone(result_path)
                self.assertGreater(len(result_df), 0)
                
                # Check for transformed price columns based on method
                if method == 'returns':
                    self.assertTrue(any(col.lower().endswith('_return') for col in result_df.columns))
                elif method == 'log':
                    self.assertTrue(any(col.lower().endswith('_log') for col in result_df.columns))
                elif method == 'pct_change':
                    self.assertTrue(any(col.lower().endswith('_pct_change') for col in result_df.columns))
                elif method == 'multi':
                    self.assertTrue(any(col.lower().endswith('_return') for col in result_df.columns))
                    self.assertTrue(any(col.lower().endswith('_log') for col in result_df.columns))
    
    def test_different_normalization_methods(self):
        """Test running the pipeline with different normalization methods"""
        # Test all normalization methods
        for method in ['zscore', 'minmax', 'robust']:
            # Create pipeline with this method and small min_data_points
            pipeline = DataPipeline(normalization_method=method)
            pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
            
            # Create a unique target directory for this test
            target_dir = os.path.join(self.processed_dir, f"normalize_{method}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Run pipeline
            with patch('pandas.read_csv', return_value=self.sample_data), \
                 patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df):
                
                result_df, result_path = pipeline.run(
                    target_path=target_dir,
                    raw_data=self.sample_data_path,
                    save_intermediate=False,
                    run_feature_selection=False
                )
                
                # Check that we got results
                self.assertIsNotNone(result_df)
                self.assertIsNotNone(result_path)
                
                # Basic sanity check - normalization shouldn't change the number of rows
                self.assertGreater(len(result_df), 0)
    
    def test_metadata_creation(self):
        """Test creation of feature metadata directly"""
        # Create pipeline 
        pipeline = DataPipeline()
        
        # Create a simple test DataFrame
        test_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200],
            'close_return': [0.01, 0.02, 0.01]
        })
        
        # DIRECTLY test the feature metadata creation method
        metadata_df = pipeline._create_feature_metadata(test_df)
        
        # Print for debugging
        print(f"Metadata categories: {metadata_df['category'].unique().tolist() if not metadata_df.empty else []}")
        print(f"Metadata columns: {metadata_df.columns.tolist()}")
        
        # Verify metadata content directly
        self.assertIsNotNone(metadata_df)
        self.assertGreater(len(metadata_df), 0)
        self.assertTrue('column' in metadata_df.columns, 
                    f"'column' not found in columns: {metadata_df.columns.tolist()}")
        self.assertTrue('category' in metadata_df.columns)
        
        # Verify it correctly categorized the columns - using the actual categories from the implementation
        raw_price_cols = metadata_df[metadata_df['category'] == 'raw_price']['column'].tolist()
        self.assertTrue(len(raw_price_cols) > 0, 
                    f"No raw price columns found. Categories: {metadata_df['category'].unique().tolist()}")
        
        # Check transformed price
        transformed_price_cols = metadata_df[metadata_df['category'] == 'transformed_price']['column'].tolist()
        self.assertTrue(len(transformed_price_cols) > 0, 
                    f"No transformed price columns found. Categories: {metadata_df['category'].unique().tolist()}")
        
        # Check volume
        volume_cols = metadata_df[metadata_df['category'] == 'volume']['column'].tolist()
        self.assertTrue(len(volume_cols) > 0, 
                    f"No volume columns found. Categories: {metadata_df['category'].unique().tolist()}")
        
    def test_error_handling_missing_file(self):
        """Test error handling when input file doesn't exist"""
        # Create pipeline
        pipeline = DataPipeline()
        
        # Try to run with a non-existent file
        non_existent_file = os.path.join(self.temp_dir.name, 'does_not_exist.csv')
        
        # This should raise an error
        with self.assertRaises(FileNotFoundError):
            # When we don't mock read_csv, it will try to read an actual file
            pipeline.run(
                target_path=self.processed_dir,
                raw_data=non_existent_file,
                save_intermediate=False,
                run_feature_selection=False
            )
    
    def test_target_column_handling(self):
        """Test proper handling of target column"""
        # Create pipeline with custom target column and small min_data_points
        pipeline = DataPipeline(target_column='high_return')
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Create a list for selected features that will survive the mocking process
        selected_features = [
            'open_raw', 'high_raw', 'low_raw', 'close_raw',
            'open_original', 'high_original', 'low_original', 'close_original',
            'high_return'  # Include the target column
        ]
        
        # Create mock selector
        mock_selector = MagicMock()
        mock_selector.selected_features = selected_features
        
        # Prepare a processed sample with the high_return column
        processed_data = self.processed_sample.copy()
        
        # Patch pandas.read_csv to return our sample data
        with patch('pandas.read_csv', return_value=processed_data), \
             patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df), \
             patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
             patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform, \
             patch('data.features.feature_selector.FeatureSelector.__init__', return_value=None) as mock_init:
            
            # Make sure the mock selector is set on the pipeline
            def set_mock_selector(*args, **kwargs):
                pipeline.feature_selector = mock_selector
                return mock_selector
                
            mock_fit.side_effect = set_mock_selector
            
            # Return non-empty dataframe from transform that includes the target
            def transform_side_effect(df):
                # Make sure we return some rows with the target column
                result = pd.DataFrame()
                
                # Add key columns
                for col in ['Date', 'Open', 'High', 'Low', 'Close', 'high_return']:
                    if col in df.columns:
                        result[col] = df[col]
                
                # Add selected features that exist
                for col in selected_features:
                    if col in df.columns and col not in result.columns:
                        result[col] = df[col]
                
                return result
            
            mock_transform.side_effect = transform_side_effect
            
            # Run pipeline with feature selection
            result_df, _ = pipeline.run(
                target_path=self.processed_dir,
                raw_data=self.sample_data_path,
                save_intermediate=False,
                run_feature_selection=True
            )
        
        # Basic check that we got a non-empty result
        self.assertIsNotNone(result_df)
        self.assertGreater(len(result_df), 0)
    
    def test_feature_selector_configuration(self):
        """Test feature selector configuration with different methods"""
        # Test different feature selection methods
        for method, threshold in [('threshold', 0.01), ('top_n', None), ('cumulative', 0.9)]:
            # Create pipeline with this method
            pipeline = DataPipeline(
                feature_selection_method=method,
                feature_importance_threshold=threshold
            )
            
            # The feature selector is created during run() in the pipeline,
            # so we need to patch and run before checking the selector config
            with patch('pandas.read_csv', return_value=self.sample_data), \
                 patch.object(pipeline, '_create_feature_metadata', return_value=self.feature_metadata_df), \
                 patch('data.features.feature_selector.FeatureSelector.__init__') as mock_init, \
                 patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
                 patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
                
                # Make the init not do anything but return the mock
                mock_init.return_value = None
                
                # Mock fit to set selected_features
                def set_selected_features(*args, **kwargs):
                    pipeline.feature_selector = MagicMock()
                    pipeline.feature_selector.selected_features = ['col1', 'col2', 'col3']
                    return MagicMock()
                
                mock_fit.side_effect = set_selected_features
                
                # Set up transform to return a dataframe
                mock_transform.return_value = self.sample_data[['Date', 'Open', 'Close', 'close_return']]
                
                # Just call the run method to create the feature selector
                try:
                    pipeline.run(
                        target_path=self.processed_dir,
                        raw_data=self.sample_data_path,
                        save_intermediate=False,
                        run_feature_selection=True
                    )
                except:
                    # Ignore any errors, we just need to check the initialization
                    pass
                
                # Check that the feature selector was initialized with the right parameters
                mock_init.assert_called()
                
                # Check the method and threshold in the call
                for call_args in mock_init.call_args_list:
                    args, kwargs = call_args
                    if 'selection_method' in kwargs:
                        self.assertEqual(kwargs['selection_method'], method)
                    if 'importance_threshold' in kwargs and threshold is not None:
                        self.assertEqual(kwargs['importance_threshold'], threshold)
    
    def test_integration_all_components(self):
        """Integration test to verify all components work together"""
        # Create special test directory
        integration_dir = os.path.join(self.processed_dir, 'integration')
        os.makedirs(integration_dir, exist_ok=True)
        
        # Create pipeline with all options enabled and small min_data_points
        pipeline = DataPipeline(
            feature_treatment_mode='hybrid',
            price_transform_method='multi',
            normalization_method='robust',
            feature_selection_method='threshold',
            feature_importance_threshold=0.01
        )
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Create mock feature metadata
        feature_metadata = pd.DataFrame({
            'column': ['open', 'high', 'low', 'close', 'volume', 'close_return'],
            'category': ['price', 'price', 'price', 'price', 'volume', 'returns'],
            'nan_count': [0, 0, 0, 0, 0, 0],
            'nan_pct': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        # Create a real metadata file
        metadata_file = os.path.join(integration_dir, f"meta_{self.fake_metadata['base_name']}.csv")
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        feature_metadata.to_csv(metadata_file, index=False)
        
        # Create mock for feature selector
        mock_selector = MagicMock()
        mock_selector.selected_features = [
            'open_raw', 'high_raw', 'low_raw', 'close_raw',
            'open_original', 'high_original', 'low_original', 'close_original',
            'close_return', 'sma_5', 'sma_10', 'rsi_14'
        ]
        
        # Mock feature selection to avoid actual ML computation
        with patch('pandas.read_csv', return_value=self.processed_sample), \
             patch.object(pipeline, '_create_feature_metadata', return_value=feature_metadata), \
             patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
             patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
            
            # Set up the mock.fit to return our mock_selector
            def mock_fit_side_effect(*args, **kwargs):
                pipeline.feature_selector = mock_selector
                return mock_selector
                
            mock_fit.side_effect = mock_fit_side_effect
            
            # Return non-empty dataframe from transform
            def transform_side_effect(df):
                # Return a subset of the input dataframe
                # Get a subset of columns including target if possible
                result = pd.DataFrame()
                
                # Include key columns
                for col in ['Date', 'Open', 'High', 'Low', 'Close', 'close_return']:
                    if col in df.columns:
                        result[col] = df[col]
                
                # Add some technical indicators if they exist
                for col in ['sma_5', 'rsi_14', 'open_raw', 'close_raw']:
                    if col in df.columns:
                        result[col] = df[col]
                
                return result
            
            mock_transform.side_effect = transform_side_effect
            
            # Run pipeline with all options
            result_df, result_path = pipeline.run(
                target_path=integration_dir,
                raw_data=self.sample_data_path,
                save_intermediate=True,
                run_feature_selection=True
            )
        
        # Check that we got results
        self.assertIsNotNone(result_df)
        self.assertIsNotNone(result_path)
        self.assertGreater(len(result_df), 0)
        
        # Output file should exist
        self.assertTrue(os.path.exists(result_path))
        
        # Check intermediate directories
        expected_dirs = ['clean', 'features', 'prepared', 'normalized']
        for dir_name in expected_dirs:
            dir_path = os.path.join(integration_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_path} does not exist")


if __name__ == '__main__':
    unittest.main()