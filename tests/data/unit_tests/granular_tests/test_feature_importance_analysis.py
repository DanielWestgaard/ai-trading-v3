import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from unittest.mock import patch, MagicMock

from data.features.feature_selector import FeatureSelector


class TestFeatureImportanceAnalysis(unittest.TestCase):
    """Dedicated test suite for the analyze_feature_importance method"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create temp directory for output files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample data with features
        np.random.seed(42)  # For reproducibility
        n_samples = 200  # Smaller dataset for faster tests
        n_features = 30
        
        # Generate dates
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1D')
        
        # Create feature matrix
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                         columns=[f'feature_{i}' for i in range(n_features)])
        
        # Add price columns
        X['open'] = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
        X['high'] = X['open'] + np.abs(np.random.normal(0, 0.5, n_samples))
        X['low'] = X['open'] - np.abs(np.random.normal(0, 0.5, n_samples))
        X['close'] = X['open'] + np.random.normal(0, 0.5, n_samples)
        X['volume'] = np.random.randint(1000, 5000, n_samples)
        
        # Add technical indicators (some with known correlation to target)
        X['sma_5'] = X['close'].rolling(window=5).mean()
        X['sma_10'] = X['close'].rolling(window=10).mean()
        X['rsi_14'] = np.random.normal(50, 10, n_samples)
        
        # Generate target (future close return)
        # Make some features strongly correlated with target
        target = 0.4 * X['feature_0'] + 0.3 * X['sma_5'] - 0.2 * X['feature_10']
        # Add some noise
        target += np.random.normal(0, 0.5, n_samples)
        X['close_return'] = target
        
        # Alternative target
        X['alt_return'] = target * 0.8 + np.random.normal(0, 0.3, n_samples)
        
        # Add date column
        X['date'] = dates
        
        # Add timestamp alternative
        X['timestamp'] = dates
        
        # Add some constant columns
        X['constant_1'] = 1.0
        X['constant_2'] = 'constant'
        
        # Store the cleaned data (first 20 rows contain NaNs from rolling windows)
        self.sample_data = X.iloc[20:].copy().reset_index(drop=True)
        
        # Create a version with NaN values for testing
        X_with_nan = X.copy()
        # Add NaN values to some features
        X_with_nan.loc[40:45, 'feature_2'] = np.nan
        X_with_nan.loc[60:65, 'sma_10'] = np.nan
        # Add NaN values to target
        X_with_nan.loc[80:85, 'close_return'] = np.nan
        self.data_with_nan = X_with_nan.iloc[20:].copy().reset_index(drop=True)
        
        # Create a default feature selector
        self.selector = FeatureSelector(
            target_col='close_return',
            output_dir=self.temp_dir.name,
            save_visualizations=False,
            n_splits=2  # Use fewer splits for faster testing
        )
        
        # Save sample data to CSV for file path tests
        self.csv_path = os.path.join(self.temp_dir.name, 'sample_data.csv')
        self.sample_data.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up after each test method"""
        self.temp_dir.cleanup()
    
    def test_basic_functionality(self):
        """Test basic functionality of analyze_feature_importance"""
        # Call the method with default parameters
        importance_df, fold_df = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='close_return',
            n_splits=2
        )
        
        # Check that the returned objects are DataFrames
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIsInstance(fold_df, pd.DataFrame)
        
        # Check that importance_df has expected structure
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        self.assertIn('category', importance_df.columns)
        
        # Check that fold_df has expected structure
        self.assertIn('fold', fold_df.columns)
        self.assertIn('mse', fold_df.columns)
        self.assertIn('mae', fold_df.columns)
        self.assertIn('r2', fold_df.columns)
        
        # We know feature_0, sma_5, and feature_10 should be important
        important_features = ['feature_0', 'sma_5', 'feature_10']
        
        # Check that at least one of our known important features is in the top 10
        top_features = importance_df.nlargest(10, 'importance')['feature'].values
        found_important = [feat for feat in important_features if feat in top_features]
        self.assertGreaterEqual(len(found_important), 1, 
                               f"At least one of {important_features} should be in the top 10 features")
        
        # Check that importance values are non-negative and sum to a reasonable value
        self.assertTrue(all(importance_df['importance'] >= 0), "Feature importance scores should be non-negative")
        self.assertGreater(importance_df['importance'].sum(), 0, "Sum of importance scores should be positive")
        
        # Check that feature categories were assigned
        categories = importance_df['category'].unique()
        self.assertGreaterEqual(len(categories), 3)  # Should have multiple categories
        
    def test_date_column_variations(self):
        """Test handling of different date column formats"""
        # Create versions with different date column names
        data_date = self.sample_data.copy()
        data_date.rename(columns={'date': 'Date'}, inplace=True)
        
        data_timestamp = self.sample_data.copy()
        data_timestamp.drop(columns=['date'], inplace=True)  # Keep only timestamp
        
        data_no_date = self.sample_data.copy()
        data_no_date.drop(columns=['date', 'timestamp'], inplace=True)
        
        # Test with 'Date' column
        importance_df1, _ = self.selector.analyze_feature_importance(
            data=data_date,
            target_col='close_return',
            n_splits=2
        )
        
        # Test with 'timestamp' column
        importance_df2, _ = self.selector.analyze_feature_importance(
            data=data_timestamp,
            target_col='close_return',
            n_splits=2
        )
        
        # Test with no date column
        importance_df3, _ = self.selector.analyze_feature_importance(
            data=data_no_date,
            target_col='close_return',
            n_splits=2
        )
        
        # Check that all approaches return valid DataFrames
        self.assertIsInstance(importance_df1, pd.DataFrame)
        self.assertIsInstance(importance_df2, pd.DataFrame)
        self.assertIsInstance(importance_df3, pd.DataFrame)
        
        # Check that they return similar number of features
        self.assertGreater(len(importance_df1), 10)
        self.assertGreater(len(importance_df2), 10)
        self.assertGreater(len(importance_df3), 10)
    
    def test_target_column_variations(self):
        """Test handling of different target columns"""
        # Test with specified target
        importance_df1, _ = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='close_return',
            n_splits=2
        )
        
        # Test with alternative target
        importance_df2, _ = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='alt_return',
            n_splits=2
        )
        
        # Test with missing target, but alternative available
        data_without_close_return = self.sample_data.copy()
        data_without_close_return.drop(columns=['close_return'], inplace=True)
        
        importance_df3, _ = self.selector.analyze_feature_importance(
            data=data_without_close_return,
            target_col='close_return',  # This doesn't exist, should use alt_return
            n_splits=2
        )
        
        # Check that all approaches return valid DataFrames
        self.assertIsInstance(importance_df1, pd.DataFrame)
        self.assertIsInstance(importance_df2, pd.DataFrame)
        self.assertIsInstance(importance_df3, pd.DataFrame)
        
        # The important features should be similar for close_return and alt_return
        # since we made alt_return a function of close_return
        top_features1 = importance_df1.head(5)['feature'].values
        top_features2 = importance_df2.head(5)['feature'].values
        
        # At least some overlap in the top features
        common_features = [f for f in top_features1 if f in top_features2]
        self.assertGreaterEqual(len(common_features), 1)
    
    def test_missing_values(self):
        """Test handling of missing values in features and target"""
        # Test with NaN values in features and target
        importance_df, fold_df = self.selector.analyze_feature_importance(
            data=self.data_with_nan,
            target_col='close_return',
            n_splits=2
        )
        
        # Check that the method ran and returned DataFrames
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIsInstance(fold_df, pd.DataFrame)
        
        # Check that features with NaN values were dropped
        nan_features = ['feature_2', 'sma_10']
        for feature in nan_features:
            self.assertNotIn(feature, importance_df['feature'].values)
    
    def test_constant_features(self):
        """Test handling of constant features"""
        # Call the method on data with constant features
        importance_df, _ = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='close_return',
            n_splits=2
        )
        
        # Constant features should not be in the result
        self.assertNotIn('constant_1', importance_df['feature'].values)
        self.assertNotIn('constant_2', importance_df['feature'].values)
        
        # Create data with only constant features (plus target)
        constant_data = pd.DataFrame({
            'constant_1': [1.0] * len(self.sample_data),
            'constant_2': [2.0] * len(self.sample_data),
            'constant_3': [3.0] * len(self.sample_data),
            'close_return': self.sample_data['close_return'].values,
            'date': self.sample_data['date'].values
        })
        
        # Test should still run and return empty feature importance
        importance_df, _ = self.selector.analyze_feature_importance(
            data=constant_data,
            target_col='close_return',
            n_splits=2
        )
        
        # Should return empty DataFrame with proper structure
        self.assertEqual(len(importance_df), 0)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
    def test_lookback_parameter(self):
        """Test the lookback parameter for creating future targets"""
        # Standard lookback = 1
        importance_df1, _ = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='close_return',
            lookback=1,
            n_splits=2
        )
        
        # Larger lookback = 5
        importance_df2, _ = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='close_return',
            lookback=5,
            n_splits=2
        )
        
        # Both should return valid DataFrames
        self.assertIsInstance(importance_df1, pd.DataFrame)
        self.assertIsInstance(importance_df2, pd.DataFrame)
        
        # The importance order might change with different lookback periods
        # Just check they both produced results
        self.assertGreater(len(importance_df1), 10)
        self.assertGreater(len(importance_df2), 10)
    
    def test_column_categorization(self):
        """Test that features are properly categorized"""
        importance_df, _ = self.selector.analyze_feature_importance(
            data=self.sample_data,
            target_col='close_return',
            n_splits=2
        )
        
        # Check that category is assigned to each feature
        self.assertIn('category', importance_df.columns)
        self.assertEqual(len(importance_df[importance_df['category'].isna()]), 0)
        
        # Check specific categories for some known column types
        feature_categories = dict(zip(importance_df['feature'], importance_df['category']))
        
        # Check price columns
        if 'open' in feature_categories:
            self.assertIn(feature_categories['open'], ['Price', 'Price Other'])
            
        # Check moving averages
        if 'sma_5' in feature_categories:
            self.assertEqual(feature_categories['sma_5'], 'Moving Averages')
            
        # Check volume
        if 'volume' in feature_categories:
            self.assertEqual(feature_categories['volume'], 'Volume')
            
        # Check oscillators
        if 'rsi_14' in feature_categories:
            self.assertEqual(feature_categories['rsi_14'], 'Oscillators')

    def test_large_feature_set(self):
        """Test performance with a large feature set"""
        # Create data with many features
        np.random.seed(42)
        n_samples = 100
        n_features = 100  # More features than samples
        
        # Create feature matrix
        large_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add target
        large_data['close_return'] = np.random.randn(n_samples)
        large_data['date'] = pd.date_range(start='2023-01-01', periods=n_samples)
        
        # Test should run without errors
        importance_df, _ = self.selector.analyze_feature_importance(
            data=large_data,
            target_col='close_return',
            n_splits=2
        )
        
        # Should return valid DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertGreater(len(importance_df), 50)  # Should have most features


if __name__ == '__main__':
    unittest.main()