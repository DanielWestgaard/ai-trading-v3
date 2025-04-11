import logging


class SignalFilter:
    def __init__(self, lookback_window=10, consensus_threshold=0.7):
        self.lookback_window = lookback_window
        self.consensus_threshold = consensus_threshold
        self.prediction_history = {}  # symbol -> list of predictions
        
    def add_prediction(self, symbol, timestamp, prediction, confidence):
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
            
        self.prediction_history[symbol].append((timestamp, prediction, confidence))
        
        # Keep only lookback window size
        if len(self.prediction_history[symbol]) > self.lookback_window:
            self.prediction_history[symbol] = self.prediction_history[symbol][-self.lookback_window:]
    
    def should_generate_signal(self, symbol, direction):
        """Check if we should generate a signal based on consensus."""
        if symbol not in self.prediction_history or len(self.prediction_history[symbol]) < self.lookback_window:
            logging.debug(f"No predictions for {symbol} yet")
            return False
            
        # Calculate consensus in the requested direction
        recent_predictions = [p[1] for p in self.prediction_history[symbol]]
        if direction == 1:  # Bullish
            consensus = sum(1 for p in recent_predictions if p == 1) / len(recent_predictions)
            logging.debug(f"BUY consensus for {symbol}: {consensus:.2f}, threshold: {self.consensus_threshold}")
        else:  # Bearish
            consensus = sum(1 for p in recent_predictions if p == -1) / len(recent_predictions)
            logging.debug(f"SELL consensus for {symbol}: {consensus:.2f}, threshold: {self.consensus_threshold}")
            
        return consensus >= self.consensus_threshold