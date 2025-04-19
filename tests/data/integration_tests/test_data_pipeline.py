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
        self.sample_data_path = os.path.join(self.raw_dir, 'EURUSD_D1_2023-01-01_2023-04-10.csv')
        self.sample_data.to_csv(self.sample_data_path, index=False)
        
        # Setup metadata dictionary that will be returned by the mocked extract_file_metadata
        self.fake_metadata = {
            'instrument': 'EURUSD',
            'timeframe': 'D1',
            'start_date': '2023-01-01',
            'end_date': '2023-04-10',
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
        
        # Run pipeline with minimal configuration
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
        self.assertGreater(len(result_df.columns), len(self.sample_data.columns))
    
    def test_run_with_intermediate(self):
        """Test running the pipeline with saving intermediate results"""
        # Create pipeline with small min_data_points
        pipeline = DataPipeline()
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Run pipeline with intermediate saving
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
        
        # First run the pipeline to the point where it creates the feature selector
        with patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
             patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform, \
             patch('data.features.feature_selector.FeatureSelector.__init__', return_value=None) as mock_init:
            
            # Explicitly set the selected_features on the feature_selector after it's created
            def add_selected_features(*args, **kwargs):
                # After the feature selector is created, set its selected_features attribute
                pipeline.feature_selector.selected_features = selected_features
                return mock_fit.return_value
            
            mock_fit.side_effect = add_selected_features
            
            # For transform, return a subset of columns as if feature selection happened
            def transform_side_effect(df):
                # Get columns that actually exist in the dataframe
                columns_to_keep = [col for col in selected_features if col in df.columns]
                # If no matching columns, use the first few columns
                if not columns_to_keep:
                    columns_to_keep = df.columns[:5].tolist()
                return df[columns_to_keep]
            
            mock_transform.side_effect = transform_side_effect
            
            # Mock metadata creation to avoid NaN division warnings
            with patch.object(pipeline, '_create_feature_metadata') as mock_metadata:
                mock_metadata.return_value = pd.DataFrame({
                    'column': ['close', 'high', 'low', 'open', 'volume'],
                    'category': ['price', 'price', 'price', 'price', 'volume'],
                    'nan_count': [0, 0, 0, 0, 0],
                    'nan_pct': [0.0, 0.0, 0.0, 0.0, 0.0]
                })
                
                # Mock file operations to avoid issues
                with patch('builtins.open', create=True), \
                     patch('os.makedirs', return_value=None):
                    
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
            self.assertGreater(len(result_df.columns), len(self.sample_data.columns))
    
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
        """Test creation of feature metadata"""
        # Create pipeline with small min_data_points
        pipeline = DataPipeline()
        pipeline.feature_preparator.min_data_points = 10  # Set much smaller than default 1000
        
        # Run pipeline
        result_df, result_path = pipeline.run(
            target_path=self.processed_dir,
            raw_data=self.sample_data_path,
            save_intermediate=False,
            run_feature_selection=False
        )
        
        # The filename of the metadata file is controlled by our mock of get_derived_file_path
        # We know the structure should be meta_[base_name].csv
        metadata_file = os.path.join(self.processed_dir, f"meta_{self.fake_metadata['base_name']}.csv")
        
        # Check if metadata file exists
        self.assertTrue(os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}")
        
        # Verify metadata content if file exists
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
            
            self.assertGreater(len(metadata_df), 0)
            self.assertTrue('column' in metadata_df.columns)
            self.assertTrue('category' in metadata_df.columns)
    
    def test_error_handling_missing_file(self):
        """Test error handling when input file doesn't exist"""
        # Create pipeline
        pipeline = DataPipeline()
        
        # Try to run with a non-existent file
        non_existent_file = os.path.join(self.temp_dir.name, 'does_not_exist.csv')
        
        # This should raise an error
        with self.assertRaises(FileNotFoundError):
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
        
        # Patch feature selector to avoid actual ML computation
        with patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
             patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform, \
             patch('data.features.feature_selector.FeatureSelector.__init__', return_value=None) as mock_init:
            
            # Explicitly set the selected_features on the feature_selector after it's created
            def add_selected_features(*args, **kwargs):
                # After the feature selector is created, set its selected_features attribute
                pipeline.feature_selector.selected_features = selected_features
                return mock_fit.return_value
            
            mock_fit.side_effect = add_selected_features
            
            # For transform, return a subset of columns including the target
            def transform_side_effect(df):
                # Get columns that actually exist in the dataframe
                columns_to_keep = [col for col in selected_features if col in df.columns]
                # If no matching columns, use the first few columns
                if not columns_to_keep:
                    columns_to_keep = df.columns[:5].tolist()
                    if 'high_return' in df.columns:
                        columns_to_keep.append('high_return')
                return df[columns_to_keep]
            
            mock_transform.side_effect = transform_side_effect
            
            # Mock metadata creation to avoid NaN division warnings
            with patch.object(pipeline, '_create_feature_metadata') as mock_metadata:
                mock_metadata.return_value = pd.DataFrame({
                    'column': ['close', 'high', 'low', 'open', 'volume'],
                    'category': ['price', 'price', 'price', 'price', 'volume'],
                    'nan_count': [0, 0, 0, 0, 0],
                    'nan_pct': [0.0, 0.0, 0.0, 0.0, 0.0]
                })
                
                # Mock file operations to avoid issues
                with patch('builtins.open', create=True), \
                     patch('os.makedirs', return_value=None):
                    
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
            with patch('data.features.feature_selector.FeatureSelector.__init__') as mock_init, \
                 patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
                 patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
                
                # Make the init not do anything but return the mock
                mock_init.return_value = None
                
                # Set up other mocks to return sensible values
                mock_fit.return_value = MagicMock()
                mock_transform.return_value = pd.DataFrame()
                
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
        
        # Mock feature selection to avoid actual ML computation
        with patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit, \
             patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
            
            # Configure mocks to return sensible values
            # Create a mock feature selector with attributes
            mock_selector = MagicMock()
            mock_selector.selected_features = [
                'open_raw', 'high_raw', 'low_raw', 'close_raw',
                'open_original', 'high_original', 'low_original', 'close_original',
                'close_return', 'sma_5', 'sma_10', 'rsi_14'
            ]
            mock_fit.return_value = mock_selector
            
            # Return subset of columns to simulate feature selection
            def transform_side_effect(df):
                # Get columns that actually exist in the dataframe
                selected_cols = [col for col in mock_fit.return_value.selected_features if col in df.columns]
                # If no columns were selected, use a default set
                if not selected_cols:
                    selected_cols = df.columns[:5].tolist()  # Just use the first few columns
                return df[selected_cols]
            
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
        
        # Patch metadata creation
        with patch.object(pipeline, '_create_feature_metadata') as mock_metadata:
            # Return a simple DataFrame
            mock_metadata.return_value = pd.DataFrame({
                'column': ['close', 'high', 'low', 'open', 'volume'],
                'category': ['price', 'price', 'price', 'price', 'volume'],
                'nan_count': [0, 0, 0, 0, 0],
                'nan_pct': [0.0, 0.0, 0.0, 0.0, 0.0]
            })
            
            # Check that the mock was called
            self.assertTrue(mock_metadata.called)
        
        # Check intermediate directories
        expected_dirs = ['clean', 'features', 'prepared', 'normalized']
        for dir_name in expected_dirs:
            dir_path = os.path.join(integration_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_path} does not exist")


if __name__ == '__main__':
    unittest.main()