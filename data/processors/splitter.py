# IMPLEMENTED IN time_series_ml.py

class TimeSeriesSplitter:
    """Handles time series-specific train/validation/test splitting"""
    def train_test_split(self, data, test_size=0.2, validation_size=0.2):
        pass
        
    def walk_forward_split(self, data, train_window, test_window, step=1):
        pass