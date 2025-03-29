import os
import pandas as pd

from data.loaders.broker_loader import CapitalComLoader
from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.processors.normalizer import DataNormalizer
import config.data_config as data_config
import config.system_config as sys_config
from utils import data_utils


class DataPipeline:
    """Coordinates the entire data processing pipeline"""
    def __init__(self, normalization_method='zscore'):
        self.cleaner = DataCleaner(
            # Specify the exact column names from your raw data
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
        )
        self.feature_generator = FeatureGenerator()
        # Initialize normalizer with specific method
        self.normalizer = DataNormalizer(other_method=normalization_method)
        
    def _get_loader(self, loader_config):
        # Factory method to get appropriate loader
        if loader_config['type'] == 'capital_com':
            return CapitalComLoader()
        # other loaders...
        
    def run(self, source=None, target_path=sys_config.CAPCOM_PROCESSED_DATA_DIR, 
            raw_data=data_config.TESTING_RAW_FILE, save_intermediate=True):
        """Execute the full pipeline"""
        # 1. Load data
        print("1. Loading data...")
        raw_data_df = pd.read_csv(raw_data, parse_dates=['Date'])
        
        # 2. Clean data
        print("2. Cleaning data...")
        self.cleaner.fit(raw_data_df)
        clean_data = self.cleaner.transform(raw_data_df)
        
        if save_intermediate and target_path:
            data_utils.save_data_file(clean_data, "processed", "testing_clean.csv")
        
        # 3. Generate features
        # print("3. Generating features...")
        # featured_data = self.feature_generator.add_technical_indicators(clean_data)
        # featured_data = self.feature_generator.add_volatility_metrics(featured_data)
        
        # if save_intermediate and target_path:
        #     data_utils.save_data_file(featured_data, "featured", "testing_featured.csv")
        
        # 4. Normalize
        print("4. Normalizing data...")
        self.normalizer.fit(clean_data)  #featured_data
        normalized_data = self.normalizer.transform(clean_data)  #featured_data
        
        # 5. Save processed data
        if target_path:
            data_utils.save_data_file(normalized_data, "processed", "testing_normalized.csv")
            
        # Return the normalized data
        return normalized_data
