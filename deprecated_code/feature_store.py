import pandas as pd


class FeatureStore:
    """Manages saving and loading feature sets"""
    def __init__(self, base_path):
        self.base_path = base_path
        
    def save_features(self, data, symbol, timeframe, feature_set_name):
        """Save generated features to storage"""
        path = f"{self.base_path}/{symbol}_{timeframe}_{feature_set_name}.parquet"
        data.to_parquet(path)
        
    def load_features(self, symbol, timeframe, feature_set_name):
        """Load pre-generated features"""
        path = f"{self.base_path}/{symbol}_{timeframe}_{feature_set_name}.parquet"
        return pd.read_parquet(path)