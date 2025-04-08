class TimeframeResampler:
    def __init__(self, base_timeframe=5, target_timeframe=60):  # 5-min to 1-hour
        self.base_timeframe = base_timeframe
        self.target_timeframe = target_timeframe
        self.ratio = target_timeframe // base_timeframe
        self.buffer = {}  # symbol -> list of data points
        
    def add_bar(self, symbol, timestamp, data):
        """Add a new bar to the buffer."""
        if symbol not in self.buffer:
            self.buffer[symbol] = []
            
        self.buffer[symbol].append((timestamp, data))
        
        # Keep only enough bars for resampling
        if len(self.buffer[symbol]) > self.ratio * 2:  # Keep extra for safety
            self.buffer[symbol] = self.buffer[symbol][-self.ratio*2:]
    
    def should_make_decision(self, timestamp):
        """Check if this is a decision point (e.g., hourly boundary)."""
        # For hourly decisions on 5-min data
        return timestamp.minute % self.target_timeframe == 0 and timestamp.second == 0
    
    def get_resampled_data(self, symbol):
        """Get resampled data for the higher timeframe."""
        if symbol not in self.buffer or len(self.buffer[symbol]) < self.ratio:
            return None
            
        recent_bars = self.buffer[symbol][-self.ratio:]
        
        # Create OHLC from the data
        opens = [bar[1]['open_raw'] for bar in recent_bars]
        highs = [bar[1]['high_raw'] for bar in recent_bars]
        lows = [bar[1]['low_raw'] for bar in recent_bars]
        closes = [bar[1]['close_raw'] for bar in recent_bars]
        
        resampled = {
            'open_raw': opens[0],
            'high_raw': max(highs),
            'low_raw': min(lows),
            'close_raw': closes[-1],
            # Add more resampled data as needed
        }
        
        return resampled