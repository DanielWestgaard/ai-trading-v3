import pandas as pd

from data.features.feature_generator import FeatureGenerator
from data.loaders.streaming_loader import StreamingDataLoader
from data.processors.cleaner import DataCleaner
from data.processors.normalizer import DataNormalizer


class LiveDataPipeline:
    """Pipeline for processing streaming market data"""
    def __init__(self, config, model, action_callback=None):
        self.config = config
        self.model = model
        self.action_callback = action_callback
        
        # Initialize components
        self.streamer = StreamingDataLoader(broker=config['broker'], callback=self.process_tick)
        self.cleaner = DataCleaner()
        self.feature_generator = FeatureGenerator()
        self.normalizer = DataNormalizer()
        
        # For maintaining recent history for feature calculation
        self.history_buffer = pd.DataFrame()
        
    def start(self, credentials, symbols):
        """Start the live data processing pipeline"""
        self.streamer.connect(credentials, symbols)
        self.streamer.start()
        
    def process_tick(self, new_data):
        """Process new incoming data point(s)"""
        # 1. Add to history buffer
        self.history_buffer = pd.concat([self.history_buffer, pd.DataFrame([new_data])])
        self.history_buffer = self.history_buffer.tail(self.config['history_size'])
        
        # 2. Clean (might be simpler for live data)
        clean_data = self.cleaner.handle_missing_values(self.history_buffer)
        
        # 3. Generate features (same as historical pipeline)
        featured_data = self.feature_generator.add_technical_indicators(clean_data)
        featured_data = self.feature_generator.add_volatility_metrics(featured_data)
        
        # 4. Normalize (same as historical pipeline)
        normalized_data = self.normalizer.standardize(featured_data)
        
        # 5. Make prediction with the model
        prediction = self.model.predict(normalized_data.tail(1))
        
        # 6. Take action based on prediction
        if self.action_callback:
            self.action_callback(prediction, new_data)