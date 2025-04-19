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
    
    def tearDown(self):
        """Clean up after each test method"""
        self.temp_dir.cleanup()
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    @patch('config.constants.data_config.TESTING_RAW_FILE')
    def test_initialization(self, mock_testing_raw, mock_processed_dir, mock_raw_dir):
        """Test that the pipeline initializes with correct parameters"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        mock_testing_raw.return_value = self.sample_data_path
        
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_run_minimal(self, mock_processed_dir, mock_raw_dir):
        """Test running the pipeline with minimal options"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Create pipeline
        pipeline = DataPipeline()
        
        # Run pipeline with minimal configuration
        with patch('pandas.read_csv', return_value=self.sample_data):
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_run_with_intermediate(self, mock_processed_dir, mock_raw_dir):
        """Test running the pipeline with saving intermediate results"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Create pipeline
        pipeline = DataPipeline()
        
        # Run pipeline with intermediate saving
        with patch('pandas.read_csv', return_value=self.sample_data):
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_run_with_feature_selection(self, mock_processed_dir, mock_raw_dir):
        """Test running the pipeline with feature selection"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Create pipeline
        pipeline = DataPipeline()
        
        # Mock the feature selector to avoid actual machine learning computations
        with patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit:
            # Configure mock to return the selector itself
            mock_fit.return_value = pipeline.feature_selector
            
            # Also mock the transform method
            with patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
                # Configure mock to return a subset of columns
                def transform_side_effect(df):
                    # Just return a subset of columns as if feature selection happened
                    essential_cols = ['open_raw', 'high_raw', 'low_raw', 'close_raw',
                                      'open_original', 'high_original', 'low_original', 'close_original']
                    # Add some technical features
                    for col in df.columns:
                        if any(pattern in col.lower() for pattern in ['sma', 'ema', 'rsi']):
                            essential_cols.append(col)
                    # Include target
                    if 'close_return' in df.columns:
                        essential_cols.append('close_return')
                    # Filter to actually existing columns
                    existing_cols = [col for col in essential_cols if col in df.columns]
                    return df[existing_cols]
                
                mock_transform.side_effect = transform_side_effect
                
                # Mock the feature importance getter
                pipeline.feature_selector.selected_features = ['open_raw', 'high_raw', 'low_raw', 'close_raw',
                                                             'open_original', 'high_original', 'low_original', 'close_original',
                                                             'sma_5', 'rsi_14', 'close_return']
                
                # Run pipeline with feature selection
                with patch('pandas.read_csv', return_value=self.sample_data):
                    result_df, result_path = pipeline.run(
                        target_path=self.processed_dir,
                        raw_data=self.sample_data_path,
                        save_intermediate=True,
                        run_feature_selection=True
                    )
        
        # Check that we got results
        self.assertIsNotNone(result_df)
        self.assertIsNotNone(result_path)
        
        # Check that the features directory was created
        features_dir = os.path.join(self.processed_dir, 'features')
        self.assertTrue(os.path.exists(features_dir))
        
        # Check that at least some features were selected
        self.assertLess(len(result_df.columns), len(self.sample_data.columns) + 20)  # Rough estimate
        
        # Metadata file should have been created
        metadata_files = [f for f in os.listdir(self.processed_dir) if f.startswith('meta_')]
        self.assertGreater(len(metadata_files), 0)
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_different_treatment_modes(self, mock_processed_dir, mock_raw_dir):
        """Test running the pipeline with different feature treatment modes"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Test all treatment modes
        for mode in ['basic', 'advanced', 'hybrid']:
            # Create pipeline with this mode
            pipeline = DataPipeline(feature_treatment_mode=mode)
            
            # Create a unique target directory for this test
            target_dir = os.path.join(self.processed_dir, f"treatment_{mode}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Run pipeline
            with patch('pandas.read_csv', return_value=self.sample_data):
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_different_price_transforms(self, mock_processed_dir, mock_raw_dir):
        """Test running the pipeline with different price transformation methods"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Test all price transform methods
        for method in ['returns', 'log', 'pct_change', 'multi']:
            # Create pipeline with this method
            pipeline = DataPipeline(price_transform_method=method)
            
            # Create a unique target directory for this test
            target_dir = os.path.join(self.processed_dir, f"transform_{method}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Run pipeline
            with patch('pandas.read_csv', return_value=self.sample_data):
                result_df, result_path = pipeline.run(
                    target_path=target_dir,
                    raw_data=self.sample_data_path,
                    save_intermediate=False,
                    run_feature_selection=False
                )
            
            # Check that we got results
            self.assertIsNotNone(result_df)
            self.assertIsNotNone(result_path)
            
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_different_normalization_methods(self, mock_processed_dir, mock_raw_dir):
        """Test running the pipeline with different normalization methods"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Test all normalization methods
        for method in ['zscore', 'minmax', 'robust']:
            # Create pipeline with this method
            pipeline = DataPipeline(normalization_method=method)
            
            # Create a unique target directory for this test
            target_dir = os.path.join(self.processed_dir, f"normalize_{method}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Run pipeline
            with patch('pandas.read_csv', return_value=self.sample_data):
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_metadata_creation(self, mock_processed_dir, mock_raw_dir):
        """Test creation of feature metadata"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Create pipeline
        pipeline = DataPipeline()
        
        # Run pipeline
        with patch('pandas.read_csv', return_value=self.sample_data):
            result_df, result_path = pipeline.run(
                target_path=self.processed_dir,
                raw_data=self.sample_data_path,
                save_intermediate=False,
                run_feature_selection=False
            )
        
        # Check that metadata file was created
        metadata_files = [f for f in os.listdir(self.processed_dir) if f.startswith('meta_')]
        self.assertGreater(len(metadata_files), 0)
        
        # Verify metadata content
        metadata_path = os.path.join(self.processed_dir, metadata_files[0])
        metadata_df = pd.read_csv(metadata_path)
        
        self.assertGreater(len(metadata_df), 0)
        self.assertTrue('column' in metadata_df.columns)
        self.assertTrue('category' in metadata_df.columns)
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_error_handling_missing_file(self, mock_processed_dir, mock_raw_dir):
        """Test error handling when input file doesn't exist"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
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
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_target_column_handling(self, mock_processed_dir, mock_raw_dir):
        """Test proper handling of target column"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Create pipeline with custom target column
        pipeline = DataPipeline(target_column='high_return')
        
        # Patch feature selector to avoid actual ML computation
        with patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit:
            mock_fit.return_value = pipeline.feature_selector
            
            with patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
                # Configure mock to include target column
                def transform_side_effect(df):
                    # Make sure target column is included
                    cols_to_keep = ['open_raw', 'high_raw', 'low_raw', 'close_raw', 'high_return']
                    existing_cols = [col for col in cols_to_keep if col in df.columns]
                    return df[existing_cols]
                
                mock_transform.side_effect = transform_side_effect
                
                # Run pipeline with feature selection
                with patch('pandas.read_csv', return_value=self.sample_data):
                    result_df, _ = pipeline.run(
                        target_path=self.processed_dir,
                        raw_data=self.sample_data_path,
                        save_intermediate=False,
                        run_feature_selection=True
                    )
        
        # Check that target column is in the result (should be created during transformation)
        self.assertTrue('high_return' in result_df.columns 
                     or any(col.lower() == 'high_return' for col in result_df.columns))
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_feature_selector_configuration(self, mock_processed_dir, mock_raw_dir):
        """Test feature selector configuration with different methods"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Test different feature selection methods
        for method, threshold in [('threshold', 0.01), ('top_n', None), ('cumulative', 0.9)]:
            # Create pipeline with this method
            pipeline = DataPipeline(
                feature_selection_method=method,
                feature_importance_threshold=threshold
            )
            
            # Verify the feature selector is configured correctly
            self.assertEqual(pipeline.feature_selector.selection_method, method)
            
            if threshold is not None:
                self.assertEqual(pipeline.feature_selector.importance_threshold, threshold)
    
    @patch('config.constants.system_config.CAPCOM_RAW_DATA_DIR')
    @patch('config.constants.system_config.CAPCOM_PROCESSED_DATA_DIR')
    def test_integration_all_components(self, mock_processed_dir, mock_raw_dir):
        """Integration test to verify all components work together"""
        # Configure mocks
        mock_raw_dir.return_value = self.raw_dir
        mock_processed_dir.return_value = self.processed_dir
        
        # Create special test directory
        integration_dir = os.path.join(self.processed_dir, 'integration')
        os.makedirs(integration_dir, exist_ok=True)
        
        # Create pipeline with all options enabled
        pipeline = DataPipeline(
            feature_treatment_mode='hybrid',
            price_transform_method='multi',
            normalization_method='robust',
            feature_selection_method='threshold',
            feature_importance_threshold=0.01
        )
        
        # Mock feature selection to avoid actual ML computation
        with patch('data.features.feature_selector.FeatureSelector.fit') as mock_fit:
            mock_fit.return_value = pipeline.feature_selector
            
            with patch('data.features.feature_selector.FeatureSelector.transform') as mock_transform:
                # Return most columns to simulate feature selection
                def transform_side_effect(df):
                    # Select a subset of columns as if feature selection happened
                    return df.iloc[:, :min(len(df.columns), 20)]
                
                mock_transform.side_effect = transform_side_effect
                
                # Run pipeline with all options
                with patch('pandas.read_csv', return_value=self.sample_data):
                    result_df, result_path = pipeline.run(
                        target_path=integration_dir,
                        raw_data=self.sample_data_path,
                        save_intermediate=True,
                        run_feature_selection=True
                    )
        
        # Check that we got results and all pipeline stages were executed
        self.assertIsNotNone(result_df)
        self.assertIsNotNone(result_path)
        
        # Output file should exist
        self.assertTrue(os.path.exists(result_path))
        
        # Metadata file should exist
        metadata_files = [f for f in os.listdir(integration_dir) if f.startswith('meta_')]
        self.assertGreater(len(metadata_files), 0)
        
        # Check intermediate directories
        expected_dirs = ['clean', 'features', 'prepared', 'normalized']
        for dir_name in expected_dirs:
            dir_path = os.path.join(integration_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_path} does not exist")


if __name__ == '__main__':
    unittest.main()