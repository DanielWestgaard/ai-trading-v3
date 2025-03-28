import pandas as pd

from data.loaders.broker_loader import CapitalComLoader
from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.processors.normalizer import DataNormalizer


class DataPipeline:
    """Coordinates the entire data processing pipeline"""
    def __init__(self, config):
        self.config = config
        #self.loader = self._get_loader(config['loader'])
        self.cleaner = DataCleaner()
        self.feature_generator = FeatureGenerator()
        self.normalizer = DataNormalizer()
        
    def _get_loader(self, loader_config):
        # Factory method to get appropriate loader
        if loader_config['type'] == 'capital_com':
            return CapitalComLoader()
        # other loaders...
        
    def run(self, source, target_path=None):
        """Execute the full pipeline"""
        # 1. Load data
        #raw_data = self.loader.load(source)
        raw_data = pd.read_csv()
        
        # 2. Clean data
        clean_data = self.cleaner.handle_missing_values(raw_data)
        clean_data = self.cleaner.remove_outliers(clean_data)
        
        # 3. Generate features
        featured_data = self.feature_generator.add_technical_indicators(clean_data)
        featured_data = self.feature_generator.add_volatility_metrics(featured_data)
        
        # 4. Normalize
        normalized_data = self.normalizer.standardize(featured_data)
        
        # 5. Save processed data
        if target_path:
            normalized_data.to_csv(target_path)
            
        return normalized_data