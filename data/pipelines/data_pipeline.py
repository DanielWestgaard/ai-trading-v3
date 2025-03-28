import os
import pandas as pd

from data.loaders.broker_loader import CapitalComLoader
from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.processors.normalizer import DataNormalizer
import config.data_config as data_config
import config.system_config as sys_config


class DataPipeline:
    """Coordinates the entire data processing pipeline"""
    def __init__(self):
        #self.config = config
        #self.loader = self._get_loader(config['loader'])
        self.cleaner = DataCleaner(
            # Specify the exact column names from your raw data
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
        )
        self.feature_generator = FeatureGenerator()
        self.normalizer = DataNormalizer()
        
    def _get_loader(self, loader_config):
        # Factory method to get appropriate loader
        if loader_config['type'] == 'capital_com':
            return CapitalComLoader()
        # other loaders...
        
    def run(self, source=None, target_path=sys_config.CAPCOM_PROCESSED_DATA_DIR, raw_data=data_config.TESTING_RAW_FILE):
        """Execute the full pipeline"""
        # 1. Load data
        #raw_data = self.loader.load(source)
        raw_data = pd.read_csv(raw_data, parse_dates=['Date'])
        
        # 2. Clean data
        self.cleaner.fit(raw_data)
        clean_data = self.cleaner.transform(raw_data)
        
        # 3. Generate features
        #featured_data = self.feature_generator.add_technical_indicators(clean_data)
        #featured_data = self.feature_generator.add_volatility_metrics(featured_data)
        
        # 4. Normalize
        #normalized_data = self.normalizer.standardize(featured_data)
        
        # 5. Save processed data
        if target_path:
            clean_data.to_csv(os.path.join(target_path, 'test.csv'), index=False)
            
        #return normalized_data
        return clean_data