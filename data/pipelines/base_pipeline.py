from data.features.feature_generator import FeatureGenerator
from data.processors.cleaner import DataCleaner
from data.processors.normalizer import DataNormalizer


class BasePipeline:
    def __init__(self, config):
        self.config = config
        self.cleaner = DataCleaner()
        self.feature_generator = FeatureGenerator()
        self.normalizer = DataNormalizer()
        
    def process_data(self, data):
        """Common processing steps"""
        # 1. Clean data
        clean_data = self.cleaner.transform(data)
        
        # 2. Generate features
        featured_data = self.feature_generator.transform(clean_data)
        
        # 3. Normalize
        return self.normalizer.transform(featured_data)