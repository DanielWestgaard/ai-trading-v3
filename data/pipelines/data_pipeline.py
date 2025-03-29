import os
import pandas as pd
import logging

from data.loaders.broker_loader import CapitalComLoader
from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer
import config.data_config as data_config
import config.system_config as sys_config
from utils import data_utils


class DataPipeline:
    """Coordinates the entire data processing pipeline with enhanced feature preparation"""
    
    def __init__(self, 
                 preserve_original_prices=True,
                 feature_treatment_mode='advanced',
                 normalization_method='zscore'):
        """
        Initialize the data pipeline with configuration.
        
        Args:
            preserve_original_prices: Whether to keep original price columns alongside transforms
            feature_treatment_mode: How to handle features ('basic', 'advanced', 'hybrid')
            normalization_method: Method for normalization ('zscore', 'minmax', 'robust')
        """
        # Configure the cleaner
        self.cleaner = DataCleaner(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
        )
        
        # Configure the feature generator
        self.feature_generator = FeatureGenerator()
        
        # Configure the feature preparator (NEW)
        self.feature_preparator = FeaturePreparator(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
            preserve_original_prices=preserve_original_prices,
            price_transform_method='returns',
            treatment_mode=feature_treatment_mode
        )
        
        # Configure the normalizer
        self.normalizer = DataNormalizer(other_method=normalization_method)
        
    def _get_loader(self, loader_config):
        # Factory method to get appropriate loader
        if loader_config['type'] == 'capital_com':
            return CapitalComLoader()
        # other loaders...
        
    def run(self, source=None, target_path=sys_config.CAPCOM_PROCESSED_DATA_DIR, 
            raw_data=data_config.TESTING_RAW_FILE, save_intermediate=False):
        """
        Execute the full pipeline with enhanced feature preparation.
        
        Args:
            source: Source configuration for data loading
            target_path: Directory to save processed data
            raw_data: Path to raw data file
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Fully processed DataFrame ready for model training
        """
        # Configure logging
        logging.info("Starting data pipeline execution")
        
        # 1. Load data
        logging.info("Loading raw data")
        raw_data_df = pd.read_csv(raw_data, parse_dates=['Date'])
        logging.info(f"Loaded raw data with shape: {raw_data_df.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(raw_data_df, "processed", "raw_data.csv")
        
        # 2. Clean data
        logging.info("Cleaning data")
        self.cleaner.fit(raw_data_df)
        clean_data = self.cleaner.transform(raw_data_df)
        logging.info(f"Cleaned data shape: {clean_data.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(clean_data, "processed", "clean_data.csv")
        
        # 3. Generate features
        logging.info("Generating features")
        self.feature_generator.fit(clean_data)
        featured_data = self.feature_generator.transform(clean_data)
        logging.info(f"Generated features. New shape: {featured_data.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(featured_data, "processed", "featured_data.csv")
        
        # 4. NEW STEP: Prepare features
        logging.info("Preparing features for modeling")
        self.feature_preparator.fit(featured_data)
        prepared_data = self.feature_preparator.transform(featured_data)
        logging.info(f"Prepared features. New shape: {prepared_data.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(prepared_data, "processed", "prepared_data.csv")
        
        # 5. Normalize
        logging.info("Normalizing data")
        self.normalizer.fit(prepared_data)
        normalized_data = self.normalizer.transform(prepared_data)
        logging.info(f"Normalized data. Final shape: {normalized_data.shape}")
        
        # 6. Save processed data
        if target_path:
            data_utils.save_data_file(normalized_data, "processed", "final_model_data.csv")
            logging.info(f"Saved final processed data to {target_path}")
        
        # Return the fully processed data
        return normalized_data


# Function to analyze the impact of different feature preparation strategies
def analyze_preparation_strategies(raw_data_path, strategies=None):
    """
    Analyze the impact of different feature preparation strategies.
    
    Args:
        raw_data_path: Path to raw data
        strategies: List of strategy configurations to test
        
    Returns:
        DataFrame comparing the results of different strategies
    """
    if strategies is None:
        strategies = [
            {'name': 'Preserve Prices + Advanced', 'preserve_prices': True, 'mode': 'advanced'},
            {'name': 'Preserve Prices + Basic', 'preserve_prices': True, 'mode': 'basic'},
            {'name': 'Transform Only + Advanced', 'preserve_prices': False, 'mode': 'advanced'},
            {'name': 'Hybrid Approach', 'preserve_prices': True, 'mode': 'hybrid'}
        ]
    
    results = []
    
    for strategy in strategies:
        # Create pipeline with this strategy
        pipeline = DataPipeline(
            preserve_original_prices=strategy['preserve_prices'],
            feature_treatment_mode=strategy['mode']
        )
        
        # Run pipeline
        try:
            result = pipeline.run(raw_data=raw_data_path, save_intermediate=False)
            
            # Calculate metrics
            metrics = {
                'strategy': strategy['name'],
                'rows_retained': len(result),
                'columns_count': result.shape[1],
                'missing_values': result.isna().sum().sum(),
                'original_prices_included': any(col in result.columns for col in ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close']),
                'feature_categories': {
                    'price': sum(1 for col in result.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])),
                    'returns': sum(1 for col in result.columns if 'return' in col.lower()),
                    'technical': sum(1 for col in result.columns if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd'])),
                    'volatility': sum(1 for col in result.columns if any(x in col.lower() for x in ['atr', 'volatility'])),
                    'patterns': sum(1 for col in result.columns if any(x in col.lower() for x in ['doji', 'hammer', 'engulfing'])),
                    'time': sum(1 for col in result.columns if any(x in col.lower() for x in ['day_', 'hour_', 'month', 'session']))
                }
            }
            
            results.append(metrics)
            logging.info(f"Strategy '{strategy['name']}' resulted in {len(result)} rows and {result.shape[1]} columns")
            
        except Exception as e:
            logging.error(f"Strategy '{strategy['name']}' failed with error: {str(e)}")
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(results)
    
    # Print summary
    print("\nFeature Preparation Strategy Comparison:")
    print(comparison[['strategy', 'rows_retained', 'columns_count', 'missing_values', 'original_prices_included']])
    
    return comparison