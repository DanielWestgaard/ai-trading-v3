from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from backtesting.events import SignalEvent, SignalType


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, symbols: List[str], params: Dict[str, Any] = None, logger=None):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of symbols to trade
            params: Strategy parameters
            logger: Custom logger (if None, creates default)
        """
        self.symbols = symbols
        self.params = params or {}
        self.logger = logger or self._setup_logger()
        
        # State tracking
        self.current_positions = {symbol: 0 for symbol in symbols}
        self.signals_generated = []
        self.custom_state = {}  # For strategy-specific state tracking
        
        # Initialize strategy
        self.initialize()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def initialize(self):
        """
        Initialize strategy state. Called once at strategy creation.
        Override this method to set up any initial state or calculations.
        """
        self.logger.info(f"Initializing {self.__class__.__name__} for symbols {self.symbols}")
    
    @abstractmethod
    def generate_signals(self, data_point, portfolio) -> List[Optional[SignalEvent]]:
        """
        Generate trading signals based on market data.
        
        Args:
            data_point: Current market data point
            portfolio: Current portfolio state
            
        Returns:
            List of signal events or None
        """
        pass
    
    def create_signal(self, 
                     symbol: str, 
                     signal_type: SignalType, 
                     timestamp: datetime,
                     strength: float = 1.0, 
                     reason: str = None, 
                     metadata: Dict[str, Any] = None) -> SignalEvent:
        """
        Create a signal event.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            timestamp: Signal timestamp
            strength: Signal strength (0.0 to 1.0)
            reason: Reason for the signal
            metadata: Additional signal information
            
        Returns:
            Signal event
        """
        signal = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            strategy_id=self.__class__.__name__,
            reason=reason,
            metadata=metadata
        )
        
        # Track the signal
        self.signals_generated.append(signal)
        
        return signal
    
    def update_position(self, symbol: str, quantity: float):
        """
        Update the strategy's tracked position for a symbol.
        
        Args:
            symbol: Market symbol
            quantity: New position quantity
        """
        if symbol in self.current_positions:
            self.current_positions[symbol] = quantity
            self.logger.debug(f"Updated position for {symbol}: {quantity}")
        else:
            self.logger.warning(f"Symbol {symbol} not in strategy symbol list")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.params.copy()
    
    def set_parameters(self, params: Dict[str, Any]):
        """
        Set strategy parameters.
        
        Args:
            params: New parameters
        """
        self.params.update(params)
        self.logger.info(f"Updated strategy parameters: {params}")
    
    def on_backtest_start(self):
        """Called when a backtest is started."""
        self.logger.info(f"Starting backtest with {self.__class__.__name__}")
    
    def on_backtest_end(self):
        """Called when a backtest is completed."""
        self.logger.info(f"Completed backtest with {self.__class__.__name__}")
        self.logger.info(f"Generated {len(self.signals_generated)} signals")