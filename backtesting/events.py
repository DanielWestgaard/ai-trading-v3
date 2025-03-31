from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class EventType(Enum):
    """Types of events in the backtesting system."""
    MARKET = "MARKET"  # New market data (e.g., price bar)
    SIGNAL = "SIGNAL"  # Strategy generated signal
    ORDER = "ORDER"    # Order sent to broker
    FILL = "FILL"      # Order has been filled
    
    # Custom event types
    RISK = "RISK"      # Risk management event (e.g., stop loss triggered)
    CUSTOM = "CUSTOM"  # User-defined event


class Event(ABC):
    """Base class for all events in the backtesting system."""
    
    def __init__(self, event_type: EventType, timestamp: Optional[datetime] = None):
        """
        Initialize base event.
        
        Args:
            event_type: Type of the event
            timestamp: When the event occurred (default: current time)
        """
        self.type = event_type
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self):
        return f"{self.type.value} Event at {self.timestamp}"


class MarketEvent(Event):
    """Event for new market data."""
    
    def __init__(self, 
                timestamp: datetime, 
                symbol: str, 
                data: Dict[str, Any]):
        """
        Initialize market event.
        
        Args:
            timestamp: When the event occurred
            symbol: Market symbol (e.g., 'AAPL', 'EURUSD')
            data: Market data (e.g., OHLCV)
        """
        super().__init__(EventType.MARKET, timestamp)
        self.symbol = symbol
        self.data = data
    
    def __str__(self):
        return f"MARKET Event: {self.symbol} at {self.timestamp}"


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    
    # Advanced signal types
    SCALE_IN = "SCALE_IN"         # Add to existing position
    SCALE_OUT = "SCALE_OUT"       # Reduce existing position
    REVERSE = "REVERSE"           # Change from long to short or vice versa
    HEDGE = "HEDGE"               # Add a hedge position
    CUSTOM = "CUSTOM"             # Custom signal type


class SignalEvent(Event):
    """Event for strategy-generated signals."""
    
    def __init__(self, 
                timestamp: datetime,
                symbol: str, 
                signal_type: SignalType, 
                strength: float = 1.0,
                strategy_id: str = "default",
                reason: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize signal event.
        
        Args:
            timestamp: When the signal was generated
            symbol: Market symbol
            signal_type: Type of signal
            strength: Signal strength (0.0 to 1.0)
            strategy_id: ID of the strategy generating the signal
            reason: Reason for the signal
            metadata: Additional signal information
        """
        super().__init__(EventType.SIGNAL, timestamp)
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.strategy_id = strategy_id
        self.reason = reason
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"SIGNAL: {self.signal_type.value} {self.symbol} (strength: {self.strength}) from {self.strategy_id}"


class OrderType(Enum):
    """Types of orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    """Order sides (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderEvent(Event):
    """Event for order submission."""
    
    def __init__(self, 
                timestamp: datetime,
                symbol: str, 
                order_type: OrderType,
                order_side: OrderSide, 
                quantity: float,
                price: Optional[float] = None,
                expiry: Optional[datetime] = None,
                order_id: Optional[str] = None,
                parent_id: Optional[str] = None,
                stop_price: Optional[float] = None,
                trailing_amount: Optional[float] = None,
                signal_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize order event.
        
        Args:
            timestamp: When the order was created
            symbol: Market symbol
            order_type: Type of order
            order_side: Buy or sell
            quantity: Order quantity
            price: Limit price (required for LIMIT and STOP_LIMIT)
            expiry: When the order expires (optional)
            order_id: Unique order identifier
            parent_id: Parent order ID (for OCO or bracket orders)
            stop_price: Stop price (for STOP and STOP_LIMIT orders)
            trailing_amount: Trailing amount (for TRAILING_STOP orders)
            signal_id: ID of the signal that generated this order
            metadata: Additional order information
        """
        super().__init__(EventType.ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.order_side = order_side
        self.quantity = quantity
        self.price = price
        self.expiry = expiry
        self.order_id = order_id
        self.parent_id = parent_id
        self.stop_price = stop_price
        self.trailing_amount = trailing_amount
        self.signal_id = signal_id
        self.metadata = metadata or {}
    
    def __str__(self):
        price_str = f" @ {self.price}" if self.price else ""
        stop_str = f" stop @ {self.stop_price}" if self.stop_price else ""
        return f"ORDER: {self.order_side.value} {self.quantity} {self.symbol} ({self.order_type.value}){price_str}{stop_str}"


class FillEvent(Event):
    """Event for order fills."""
    
    def __init__(self, 
                timestamp: datetime,
                symbol: str, 
                order_side: OrderSide,
                quantity: float,
                fill_price: float,
                commission: float = 0.0,
                order_id: Optional[str] = None,
                fill_id: Optional[str] = None,
                partial: bool = False,
                remaining: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize fill event.
        
        Args:
            timestamp: When the fill occurred
            symbol: Market symbol
            order_side: Buy or sell
            quantity: Filled quantity
            fill_price: Price at which the order was filled
            commission: Commission/fees for the fill
            order_id: ID of the filled order
            fill_id: Unique fill identifier
            partial: Whether this is a partial fill
            remaining: Remaining quantity to be filled
            metadata: Additional fill information
        """
        super().__init__(EventType.FILL, timestamp)
        self.symbol = symbol
        self.order_side = order_side
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
        self.order_id = order_id
        self.fill_id = fill_id
        self.partial = partial
        self.remaining = remaining
        self.metadata = metadata or {}
    
    def __str__(self):
        partial_str = " (Partial)" if self.partial else ""
        return f"FILL: {self.order_side.value} {self.quantity} {self.symbol} @ {self.fill_price}{partial_str}"