from data.processors.base_processor import BaseProcessor


class DataNormalizer(BaseProcessor):
    def __init__(self, outlier_method='zscore', missing_method='ffill'):
        self.outlier_method = outlier_method
        self.missing_method = missing_method
        
    def fit(self, data):
        # Calculate any parameters needed (e.g., z-score thresholds)
        self._mean = data.mean()
        self._std = data.std()
        return self
        
    def transform(self, data):
        # Apply cleaning based on parameters
        result = self._handle_missing_values(data)
        result = self._remove_outliers(result)
        return result