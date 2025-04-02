import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from backtesting.strategies.base_strategy import BaseStrategy
from backtesting.events import SignalEvent, SignalType


class SimpleMovingAverageCrossover(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, symbols, params=None, logger=None):
        """
        Initialize MA Crossover strategy.
        
        Args:
            symbols: List of symbols to trade
            params: Strategy parameters
                - short_window: Short MA window (default: 10)
                - long_window: Long MA window (default: 50)
                - use_sma: Use Simple Moving Average (default: True)
                - use_close: Use close price (default: True)
                - signal_threshold: Threshold for generating signals (default: 0.0)
            logger: Custom logger
        """
        # Default parameters
        default_params = {
            'short_window': 10,
            'long_window': 50,
            'use_sma': True,    # If False, use EMA
            'use_close': True,  # If False, use typical price (H+L+C)/3
            'signal_threshold': 0.0  # Minimum difference to generate signal
        }
        
        # Merge provided params with defaults
        if params:
            merged_params = default_params.copy()
            merged_params.update(params)
            params = merged_params
        else:
            params = default_params
        
        super().__init__(symbols, params, logger)
        
        # Strategy state
        self.current_position = {symbol: 0 for symbol in symbols}  # 1=long, -1=short, 0=flat
        self.historical_data = {symbol: pd.DataFrame() for symbol in symbols}
        self.ma_values = {symbol: {'short': None, 'long': None} for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy."""
        self.logger.info(f"Initializing MA Crossover strategy with params: {self.params}")
        self.logger.info(f"Short window: {self.params['short_window']}, Long window: {self.params['long_window']}")
    
    def calculate_moving_averages(self, data, symbol):
        """
        Calculate moving averages for a symbol.
        
        Args:
            data: Market data
            symbol: Market symbol
            
        Returns:
            Tuple of (short_ma, long_ma)
        """
        # Get price data
        if self.params['use_close']:
            price = data.get('Close', data.get('close'))
            # If not found, try with _original suffix
            if price is None:
                price = data.get('close_original')    
            # If still not found, print available columns for debugging
            if price is None:
                self.logger.warning(f"Price data not found. Available columns: {list(data.keys())}")
                return None, None
        else:
            # Use typical price (H+L+C)/3
            high = data.get('High', data.get('high'))
            low = data.get('Low', data.get('low'))
            close = data.get('Close', data.get('close'))
            
            if high is not None and low is not None and close is not None:
                price = (high + low + close) / 3
            else:
                price = close  # Fallback to close if other prices not available
        
        # Store data in historical DataFrame
        if symbol not in self.historical_data:
            self.historical_data[symbol] = pd.DataFrame()
        
        timestamp = data.get('Date', data.get('date', datetime.now()))
        
        # Create a new row as a DataFrame and use concat instead of append
        new_row = pd.DataFrame({'timestamp': [timestamp], 'price': [price]})
        
        # Use concat instead of append (which is deprecated in newer pandas versions)
        self.historical_data[symbol] = pd.concat([self.historical_data[symbol], new_row], ignore_index=True)
        
        # We need at least long_window data points
        if len(self.historical_data[symbol]) < self.params['long_window']:
            return None, None
        
        # Calculate MAs
        if self.params['use_sma']:
            short_ma = self.historical_data[symbol]['price'].rolling(
                window=self.params['short_window']).mean().iloc[-1]
            long_ma = self.historical_data[symbol]['price'].rolling(
                window=self.params['long_window']).mean().iloc[-1]
        else:
            # Use EMA
            short_ma = self.historical_data[symbol]['price'].ewm(
                span=self.params['short_window']).mean().iloc[-1]
            long_ma = self.historical_data[symbol]['price'].ewm(
                span=self.params['long_window']).mean().iloc[-1]
        
        return short_ma, long_ma
    
    def generate_signals(self, market_data, portfolio) -> List[Optional[SignalEvent]]:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            market_data: Dictionary of current market data
            portfolio: Current portfolio state
            
        Returns:
            List of signal events or None
        """
        signals = []
        
        for symbol, data in market_data.items():
            if symbol not in self.symbols:
                continue
            
            # Calculate moving averages
            short_ma, long_ma = self.calculate_moving_averages(data.data, symbol)
            
            # Store current MA values
            self.ma_values[symbol]['short'] = short_ma
            self.ma_values[symbol]['long'] = long_ma
            
            # Not enough data yet
            if short_ma is None or long_ma is None:
                continue
            
            # Get current position
            current_position = self.current_position.get(symbol, 0)
            
            # Calculate signal based on MA crossover
            signal = None
            timestamp = data.timestamp
            
            # Check if the difference exceeds the threshold
            ma_diff = short_ma - long_ma
            
            if ma_diff > self.params['signal_threshold'] and current_position <= 0:
                # Short MA crossed above Long MA -> BUY signal
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY if current_position == 0 else SignalType.REVERSE,
                    timestamp=timestamp,
                    reason=f"MA Crossover: Short {short_ma:.2f} > Long {long_ma:.2f}",
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'ma_diff': ma_diff
                    }
                )
                self.current_position[symbol] = 1
                
            elif ma_diff < -self.params['signal_threshold'] and current_position >= 0:
                # Short MA crossed below Long MA -> SELL signal
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL if current_position == 0 else SignalType.REVERSE,
                    timestamp=timestamp,
                    reason=f"MA Crossover: Short {short_ma:.2f} < Long {long_ma:.2f}",
                    metadata={
                        'short_ma': short_ma,
                        'long_ma': long_ma,
                        'ma_diff': ma_diff
                    }
                )
                self.current_position[symbol] = -1
            
            if signal:
                signals.append(signal)
                self.logger.info(f"Generated signal: {signal}")
        
        return signals


class MACDStrategy(BaseStrategy):
    """MACD (Moving Average Convergence Divergence) Strategy."""
    
    def __init__(self, symbols, params=None, logger=None):
        """
        Initialize MACD strategy.
        
        Args:
            symbols: List of symbols to trade
            params: Strategy parameters
                - fast_period: Fast EMA period (default: 12)
                - slow_period: Slow EMA period (default: 26)
                - signal_period: Signal EMA period (default: 9)
                - use_close: Use close price (default: True)
                - signal_threshold: Threshold for generating signals (default: 0.0)
            logger: Custom logger
        """
        # Default parameters
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'use_close': True,  # If False, use typical price (H+L+C)/3
            'signal_threshold': 0.0  # Minimum histogram difference to generate signal
        }
        
        # Merge provided params with defaults
        if params:
            merged_params = default_params.copy()
            merged_params.update(params)
            params = merged_params
        else:
            params = default_params
        
        super().__init__(symbols, params, logger)
        
        # Strategy state
        self.current_position = {symbol: 0 for symbol in symbols}  # 1=long, -1=short, 0=flat
        self.historical_data = {symbol: pd.DataFrame() for symbol in symbols}
        self.macd_values = {symbol: {'macd': None, 'signal': None, 'hist': None} for symbol in symbols}
    
    def initialize(self):
        """Initialize strategy."""
        self.logger.info(f"Initializing MACD strategy with params: {self.params}")
        self.logger.info(f"Fast period: {self.params['fast_period']}, Slow period: {self.params['slow_period']}, Signal period: {self.params['signal_period']}")
    
    def calculate_macd(self, data, symbol):
        """
        Calculate MACD for a symbol.
        
        Args:
            data: Market data
            symbol: Market symbol
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Get price data
        if self.params['use_close']:
            price = data.get('Close', data.get('close'))
        else:
            # Use typical price (H+L+C)/3
            high = data.get('High', data.get('high'))
            low = data.get('Low', data.get('low'))
            close = data.get('Close', data.get('close'))
            
            if high is not None and low is not None and close is not None:
                price = (high + low + close) / 3
            else:
                price = close  # Fallback to close if other prices not available
        
        # Store data in historical DataFrame
        timestamp = data.get('Date', data.get('date', datetime.now()))
        
        # Create a new row as a DataFrame and use concat instead of append
        new_row = pd.DataFrame({'timestamp': [timestamp], 'price': [price]})
        
        # Use concat instead of append (which is deprecated in newer pandas versions)
        self.historical_data[symbol] = pd.concat([self.historical_data[symbol], new_row], ignore_index=True)
        
        # We need at least slow_period + signal_period data points
        min_periods = self.params['slow_period'] + self.params['signal_period']
        if len(self.historical_data[symbol]) < min_periods:
            return None, None, None
        
        # Calculate MACD
        fast_ema = self.historical_data[symbol]['price'].ewm(span=self.params['fast_period']).mean()
        slow_ema = self.historical_data[symbol]['price'].ewm(span=self.params['slow_period']).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.params['signal_period']).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def generate_signals(self, market_data, portfolio) -> List[Optional[SignalEvent]]:
        """
        Generate trading signals based on MACD crossover.
        
        Args:
            market_data: Dictionary of current market data
            portfolio: Current portfolio state
            
        Returns:
            List of signal events or None
        """
        signals = []
        
        for symbol, data in market_data.items():
            if symbol not in self.symbols:
                continue
            
            # Calculate MACD
            macd_line, signal_line, histogram = self.calculate_macd(data.data, symbol)
            
            # Store current MACD values
            self.macd_values[symbol]['macd'] = macd_line
            self.macd_values[symbol]['signal'] = signal_line
            self.macd_values[symbol]['hist'] = histogram
            
            # Not enough data yet
            if macd_line is None or signal_line is None or histogram is None:
                continue
            
            # Get current position
            current_position = self.current_position.get(symbol, 0)
            
            # Calculate signal based on MACD histogram
            signal = None
            timestamp = data.timestamp
            
            # Check if the histogram exceeds the threshold
            if histogram > self.params['signal_threshold'] and current_position <= 0:
                # Histogram is positive and above threshold -> BUY signal
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY if current_position == 0 else SignalType.REVERSE,
                    timestamp=timestamp,
                    reason=f"MACD: Histogram {histogram:.6f} > {self.params['signal_threshold']}",
                    metadata={
                        'macd': macd_line,
                        'signal': signal_line,
                        'histogram': histogram
                    }
                )
                self.current_position[symbol] = 1
                
            elif histogram < -self.params['signal_threshold'] and current_position >= 0:
                # Histogram is negative and below threshold -> SELL signal
                signal = self.create_signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL if current_position == 0 else SignalType.REVERSE,
                    timestamp=timestamp,
                    reason=f"MACD: Histogram {histogram:.6f} < -{self.params['signal_threshold']}",
                    metadata={
                        'macd': macd_line,
                        'signal': signal_line,
                        'histogram': histogram
                    }
                )
                self.current_position[symbol] = -1
            
            if signal:
                signals.append(signal)
                self.logger.info(f"Generated signal: {signal}")
        
        return signals