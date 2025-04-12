import numpy as np
import time
import logging
from typing import Dict, Optional, Any

from core.events import OrderEvent, FillEvent, OrderSide
from core.execution.execution_interface import BaseExecutionHandler

class SimulationExecutionHandler(BaseExecutionHandler):
    """Simulates order execution during backtesting."""
    
    def __init__(self, 
                slippage_model="fixed",
                slippage_params=None,
                rejection_probability=0.0,
                fill_latency_ms=0,
                partial_fills=False,
                logger=None):
        """
        Initialize execution handler.
        
        Args:
            slippage_model: Type of slippage model
            slippage_params: Parameters for slippage model
            rejection_probability: Probability of order rejection
            fill_latency_ms: Simulated fill latency in milliseconds
            partial_fills: Whether to simulate partial fills
            logger: Custom logger
        """
        self.slippage_model = slippage_model
        self.slippage_params = slippage_params or {}
        self.rejection_probability = rejection_probability
        self.fill_latency_ms = fill_latency_ms
        self.partial_fills = partial_fills
        self.logger = logger or logging.getLogger(__name__)
    
    # Copy the execute_order, _apply_slippage, and _calculate_commission methods 
    # from the original ExecutionHandler class
    
    def execute_order(self, order: OrderEvent, market_data: Dict[str, Any]) -> Optional[FillEvent]:
        """
        Simulate order execution.
        
        Args:
            order: Order to execute
            market_data: Current market data
            
        Returns:
            Fill event or None if order rejected
        """
        # Check for rejection
        if np.random.random() < self.rejection_probability:
            logging.info(f"Order rejected: {order}")
            return None
        
        # Simulate latency
        if self.fill_latency_ms > 0:
            time.sleep(self.fill_latency_ms / 1000.0)
        
        # Get market data for the symbol
        symbol_data = market_data.get(order.symbol)
        if not symbol_data:
            logging.warning(f"No market data for symbol {order.symbol}, cannot execute order")
            return None
        
        # Determine fill price with slippage
        if isinstance(symbol_data, dict):
            # Look for price columns in priority order
            price_columns = [
                'close_raw',        # First priority: raw unmodified price
                'Close_raw',        
                'close_original',   # Second priority: original prices 
                'Close_original',
                'close',            # Last resort: normalized prices
                'Close'
            ]
            
            # Try each column until we find a valid price
            base_price = None
            for col in price_columns:
                if col in symbol_data:
                    base_price = symbol_data[col]
                    logging.info(f"Using {col} price for {order.symbol}: {base_price}")
                    break
            
            if base_price is None:
                logging.warning(f"No price column found for {order.symbol}. Available columns: {list(symbol_data.keys())}")
                return None
                
            # Ensure price is positive and reasonable for forex
            if base_price <= 0:
                logging.warning(f"Invalid non-positive price: {base_price} for {order.symbol}")
                return None
        else:
            # Assume it's a MarketEvent
            base_price = symbol_data.data.get('close_original', 
                        symbol_data.data.get('Close_original',
                        symbol_data.data.get('Close', 
                        symbol_data.data.get('close'))))
            
            if base_price is None:
                # Log available keys to help debug
                if hasattr(symbol_data, 'data') and isinstance(symbol_data.data, dict):
                    logging.warning(f"No close price found in market data for {order.symbol}. Available keys: {list(symbol_data.data.keys())}")
                else:
                    logging.warning(f"No close price found in market data for {order.symbol}")
                return None
        
        # Apply slippage to the price
        fill_price = self._apply_slippage(base_price, order)
        
        # Create fill event
        fill = FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            order_side=order.order_side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=self._calculate_commission(order, fill_price),
            order_id=order.order_id,
            fill_id=f"fill_{order.order_id}",
            partial=False,
            remaining=0.0
        )
        
        logging.info(f"Order executed: {order} -> {fill}")
        return fill
    
    def _apply_slippage(self, base_price: float, order: OrderEvent) -> float:
        """
        Apply slippage to the base price.
        
        Args:
            base_price: Base price
            order: Order
            
        Returns:
            Price with slippage applied
        """
        if self.slippage_model == "fixed":
            # Fixed slippage in percentage
            slippage_percent = self.slippage_params.get("percent", 0.0)
            
            if order.order_side == OrderSide.BUY:
                # Buy orders get filled at a higher price
                return base_price * (1 + slippage_percent / 100.0)
            else:
                # Sell orders get filled at a lower price
                return base_price * (1 - slippage_percent / 100.0)
        
        elif self.slippage_model == "variable":
            # Variable slippage based on order size
            base_slippage = self.slippage_params.get("base_percent", 0.0)
            size_factor = self.slippage_params.get("size_factor", 0.0)
            
            # Scale slippage based on order size
            slippage_percent = base_slippage + (size_factor * order.quantity)
            
            if order.order_side == OrderSide.BUY:
                return base_price * (1 + slippage_percent / 100.0)
            else:
                return base_price * (1 - slippage_percent / 100.0)
        
        elif self.slippage_model == "probabilistic":
            # Probabilistic slippage model
            min_slippage = self.slippage_params.get("min_percent", 0.0)
            max_slippage = self.slippage_params.get("max_percent", 0.0)
            
            # Random slippage between min and max
            slippage_percent = np.random.uniform(min_slippage, max_slippage)
            
            if order.order_side == OrderSide.BUY:
                return base_price * (1 + slippage_percent / 100.0)
            else:
                return base_price * (1 - slippage_percent / 100.0)
        
        else:
            # No slippage model or unknown model
            return base_price
    
    def _calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        """
        Calculate commission for an order.
        
        Args:
            order: Order
            fill_price: Fill price
            
        Returns:
            Commission amount
        """
        # Simple percentage commission model
        commission_percent = self.slippage_params.get("commission_percent", 0.1)
        return (commission_percent / 100.0) * fill_price * order.quantity
        