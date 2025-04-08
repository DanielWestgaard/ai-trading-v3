# backtesting/strategies/indicator_crossover_strategy.py
import pandas as pd
from typing import List, Optional

from core.strategies.base_strategy import BaseStrategy
from backtesting.events import SignalEvent, SignalType
import logging

class IndicatorCrossoverStrategy(BaseStrategy):
    """Strategy that uses pre-calculated technical indicators."""
    
    def __init__(self, symbols, params=None):
        # Default parameters
        default_params = {
            'fast_indicator': 'ema_20',  # Use existing EMAs instead of calculating new ones
            'slow_indicator': 'sma_50',
            'signal_threshold': 0.001
        }
        
        # Merge params
        if params:
            merged_params = default_params.copy()
            merged_params.update(params)
            params = merged_params
        else:
            params = default_params
        
        super().__init__(symbols, params)
        self.current_position = {symbol: 0 for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy."""
        logging.info(f"Initializing {self.__class__.__name__} with indicators: "
                         f"{self.params['fast_indicator']} and {self.params['slow_indicator']}")
    
    def generate_signals(self, market_data, portfolio):
        signals = []
        
        for symbol, data in market_data.items():
            if symbol not in self.symbols:
                continue
            
            # Get indicator values
            fast_indicator = self.params['fast_indicator']
            slow_indicator = self.params['slow_indicator']
            
            # Check if indicators exist in the data
            if fast_indicator not in data.data or slow_indicator not in data.data:
                if not hasattr(self, 'missing_indicators_logged'):
                    logging.warning(f"Indicators not found: {fast_indicator} or {slow_indicator}")
                    logging.info(f"Available columns: {list(data.data.keys())}")
                    self.missing_indicators_logged = True
                continue
            
            fast_value = data.data[fast_indicator]
            slow_value = data.data[slow_indicator]
            
            # Check for indicator crossover
            current_position = self.current_position.get(symbol, 0)
            threshold = self.params.get('signal_threshold', 0.001)
            
            # Calculate indicator difference
            indicator_diff = fast_value - slow_value
            
            # Generate signals based on crossover
            signal = None
            if indicator_diff > threshold and current_position <= 0:
                # Fast indicator above slow indicator -> BUY signal
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY if current_position == 0 else SignalType.REVERSE,
                    timestamp=data.timestamp,
                    reason=f"Indicator Crossover: {fast_indicator} > {slow_indicator}",
                    metadata={
                        'fast_value': fast_value,
                        'slow_value': slow_value,
                        'difference': indicator_diff
                    }
                )
                self.current_position[symbol] = 1
                
            elif indicator_diff < -threshold and current_position >= 0:
                # Fast indicator below slow indicator -> SELL signal
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL if current_position == 0 else SignalType.REVERSE,
                    timestamp=data.timestamp,
                    reason=f"Indicator Crossover: {fast_indicator} < {slow_indicator}",
                    metadata={
                        'fast_value': fast_value,
                        'slow_value': slow_value,
                        'difference': indicator_diff
                    }
                )
                self.current_position[symbol] = -1
            
            if signal:
                signals.append(signal)
                logging.info(f"Generated signal: {signal}")
        
        return signals