class TimeframeResampler:
    def __init__(self, base_timeframe=5, target_timeframe=60, feature_generator=None):  # 5-min to 1-hour
        self.base_timeframe = base_timeframe
        self.target_timeframe = target_timeframe
        self.ratio = target_timeframe // base_timeframe
        self.buffer = {}  # symbol -> list of data points
        self.feature_generator = feature_generator  # Option to pass in the feature generator
        
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
        """
        Get resampled data for the higher timeframe with all features preserved.
        
        Instead of just returning the OHLC values, this enhanced version
        preserves all the features needed by the model.
        """
        import logging
        
        if symbol not in self.buffer or len(self.buffer[symbol]) < self.ratio:
            return None
            
        # Get the most recent bars for this timeframe
        recent_bars = self.buffer[symbol][-self.ratio:]
        
        # Extract all fields from the first bar to understand what data we have
        first_bar_data = recent_bars[0][1]
        
        # Initialize resampled data with basic OHLC
        try:
            # Extract OHLC data
            opens = []
            highs = []
            lows = []
            closes = []
            
            # Try to get values using multiple possible field names
            for bar in recent_bars:
                bar_data = bar[1]
                
                # Get open price (try multiple possible field names)
                if 'open_raw' in bar_data:
                    opens.append(bar_data['open_raw'])
                elif 'Open_raw' in bar_data:
                    opens.append(bar_data['Open_raw'])
                elif 'open' in bar_data:
                    opens.append(bar_data['open'])
                elif 'Open' in bar_data:
                    opens.append(bar_data['Open'])
                else:
                    # Log available fields for debugging
                    logging.warning(f"No open price field found in bar data. Available fields: {list(bar_data.keys())[:10]}")
                    opens.append(0)  # Default value
                
                # Similar approach for high, low, close
                if 'high_raw' in bar_data:
                    highs.append(bar_data['high_raw'])
                elif 'High_raw' in bar_data:
                    highs.append(bar_data['High_raw'])
                elif 'high' in bar_data:
                    highs.append(bar_data['high'])
                elif 'High' in bar_data:
                    highs.append(bar_data['High'])
                else:
                    highs.append(0)
                
                if 'low_raw' in bar_data:
                    lows.append(bar_data['low_raw'])
                elif 'Low_raw' in bar_data:
                    lows.append(bar_data['Low_raw'])
                elif 'low' in bar_data:
                    lows.append(bar_data['low'])
                elif 'Low' in bar_data:
                    lows.append(bar_data['Low'])
                else:
                    lows.append(0)
                
                if 'close_raw' in bar_data:
                    closes.append(bar_data['close_raw'])
                elif 'Close_raw' in bar_data:
                    closes.append(bar_data['Close_raw'])
                elif 'close' in bar_data:
                    closes.append(bar_data['close'])
                elif 'Close' in bar_data:
                    closes.append(bar_data['Close'])
                else:
                    closes.append(0)
            
            # Create basic resampled data
            resampled = {
                'open_raw': opens[0],
                'high_raw': max(highs),
                'low_raw': min(lows),
                'close_raw': closes[-1],
                'open': opens[0],
                'high': max(highs),
                'low': min(lows),
                'close': closes[-1],
                # Add other fields that are needed
                'open_original': opens[0],
                'high_original': max(highs),
                'low_original': min(lows),
                'close_original': closes[-1],
                'volume': sum([bar[1].get('volume', 0) for bar in recent_bars])
            }
            
            # If epic and resolution are available, include them
            if 'epic' in first_bar_data:
                resampled['epic'] = first_bar_data['epic']
            if 'resolution' in first_bar_data:
                resampled['resolution'] = first_bar_data['resolution']
            
            # Use the timestamp from the most recent bar
            resampled['timestamp'] = recent_bars[-1][0]
            
            # Instead of trying to preserve all technical indicators (which would be incorrect
            # when resampling), we should approach this in one of two ways:
            
            # Option 1: Signal the caller that they need to recalculate features
            resampled['needs_feature_calculation'] = True
            
            # Option 2: If we have a feature generator, recalculate here
            # This is less preferred because the feature calculation should
            # ideally happen in the strategy with full context
            
            return resampled
            
        except Exception as e:
            logging.error(f"Error in get_resampled_data: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def get_higher_timeframe_data(self, symbol):
        """
        Alternative approach: return list of raw OHLC data for recalculation.
        This allows the caller to recalculate all technical indicators properly.
        """
        if symbol not in self.buffer or len(self.buffer[symbol]) < self.ratio:
            return None
        
        # Get all the required candles for this timeframe
        candles = self.buffer[symbol][-self.ratio:]
        
        # Let the caller handle the actual feature calculation
        return candles