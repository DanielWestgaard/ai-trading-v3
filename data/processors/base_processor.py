class BaseProcessor:
    def fit(self, data):
        """Learn parameters from data"""
        return self
        
    def transform(self, data):
        """Apply transformation using learned parameters"""
        return data
        
    def fit_transform(self, data):
        """Convenience method"""
        return self.fit(data).transform(data)