# loaders/streaming_loader.py
from data.loaders.base_loader import BaseDataLoader


class StreamingDataLoader(BaseDataLoader):
    """Handles real-time data streams from brokers"""
    def __init__(self, broker='capital_com', callback=None):
        self.broker = broker
        self.callback = callback
        self.connection = None
        
    def connect(self, credentials, symbols):
        """Establish connection to the data stream"""
        if self.broker == 'capital_com':
            # Set up WebSocket or API connection
            self.connection = "" #CapitalComWebSocket(credentials)
            for symbol in symbols:
                self.connection.subscribe(symbol)
        
    def start(self):
        """Start receiving data"""
        self.connection.on_data(self._process_incoming)
        self.connection.start()
        
    def _process_incoming(self, data):
        """Process incoming tick/candle and forward to callback"""
        if self.callback:
            self.callback(data)