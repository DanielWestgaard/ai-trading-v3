from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from backtesting.events import OrderEvent, FillEvent, OrderSide, OrderType


class Position:
    """Class representing a trading position."""
    
    def __init__(self, 
                symbol: str, 
                entry_price: float, 
                quantity: float,
                entry_time: datetime,
                direction: str,
                order_id: Optional[str] = None,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                position_id: Optional[str] = None,
                metadata: Dict[str, Any] = None):
        """
        Initialize a position.
        
        Args:
            symbol: Market symbol
            entry_price: Average entry price
            quantity: Position size
            entry_time: When the position was entered
            direction: "LONG" or "SHORT"
            order_id: ID of the order that created this position
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            position_id: Unique position identifier
            metadata: Additional position information
        """
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.direction = direction
        self.order_id = order_id
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_id = position_id or f"{symbol}_{entry_time.strftime('%Y%m%d%H%M%S')}_{direction}"
        self.metadata = metadata or {}
        
        # Position tracking
        self.is_open = True
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.exit_reason = None
        
    def __str__(self):
        status = "OPEN" if self.is_open else "CLOSED"
        pnl_str = f", PnL: {self.pnl:.2f}" if not self.is_open else ""
        return f"{status} {self.direction} {self.quantity} {self.symbol} @ {self.entry_price}{pnl_str}"
    
    def update_unrealized_pnl(self, current_price: float):
        """
        Update unrealized P&L based on current market price.
        
        Args:
            current_price: Current market price
        """
        if self.direction == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def close(self, exit_price: float, exit_time: datetime, reason: str = "MANUAL"):
        """
        Close the position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit time
            reason: Reason for closing
        
        Returns:
            Realized P&L
        """
        if not self.is_open:
            return 0.0
        
        self.is_open = False
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate realized P&L
        if self.direction == "LONG":
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        self.pnl = self.realized_pnl
        self.unrealized_pnl = 0.0
        
        return self.realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the position to a dictionary.
        
        Returns:
            Dictionary representation of the position
        """
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "is_open": self.is_open,
            "pnl": self.pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "exit_reason": self.exit_reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "order_id": self.order_id,
            **self.metadata
        }


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, 
                initial_capital: float = 10000.0,
                leverage: float = 1.0,
                commission_model: Any = None,
                logger=None):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Starting capital
            leverage: Portfolio leverage
            commission_model: Model for calculating commissions
            logger: Custom logger
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission_model = commission_model
        self.logger = logger or self._setup_logger()
        
        # Portfolio state
        self.cash = initial_capital
        self.equity = initial_capital
        self.positions = {}  # symbol -> Position object
        self.position_history = []
        self.open_orders = {}  # order_id -> OrderEvent
        
        # Portfolio history tracking
        self.history = []
        self.transactions = []
        
        # Initialize portfolio history with starting values
        self._record_portfolio_state(datetime.now())
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.Portfolio")
        logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def _record_portfolio_state(self, timestamp: datetime):
        """
        Record current portfolio state to history.
        
        Args:
            timestamp: Current timestamp
        """
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        state = {
            "timestamp": timestamp,
            "cash": self.cash,
            "equity": self.equity,
            "unrealized_pnl": unrealized_pnl,
            "open_positions": len(self.positions),
            "leverage_used": self._calculate_leverage_used()
        }
        
        self.history.append(state)
    
    def _calculate_leverage_used(self) -> float:
        """
        Calculate current leverage used.
        
        Returns:
            Current leverage used
        """
        if self.equity <= 0:
            return 0.0
        
        total_position_value = sum(
            abs(pos.quantity * pos.entry_price) for pos in self.positions.values()
        )
        
        return total_position_value / self.equity if self.equity > 0 else 0.0
    
    def _calculate_commission(self, fill_event: FillEvent) -> float:
        """
        Calculate commission for a fill event.
        
        Args:
            fill_event: Fill event
            
        Returns:
            Commission amount
        """
        if self.commission_model:
            return self.commission_model.calculate(fill_event)
        
        # Simple default commission model
        return fill_event.fill_price * fill_event.quantity * 0.001  # 0.1% commission
    
    def process_fill(self, fill_event: FillEvent):
        """
        Process a fill event.
        
        Args:
            fill_event: Fill event
        """
        symbol = fill_event.symbol
        fill_price = fill_event.fill_price
        quantity = fill_event.quantity
        timestamp = fill_event.timestamp
        commission = self._calculate_commission(fill_event)
        
        # Determine if this is opening a new position or closing/modifying an existing one
        is_buy = fill_event.order_side == OrderSide.BUY
        direction = "LONG" if is_buy else "SHORT"
        
        # Check if we already have a position for this symbol
        existing_position = self.positions.get(symbol)
        
        if existing_position:
            # Existing position - determine if increasing, decreasing, or reversing
            if (is_buy and existing_position.direction == "LONG") or \
               (not is_buy and existing_position.direction == "SHORT"):
                # Increasing position
                self._increase_position(existing_position, quantity, fill_price, timestamp, commission)
            else:
                # Decreasing or closing position
                self._decrease_position(existing_position, quantity, fill_price, timestamp, commission, fill_event.order_id)
        else:
            # New position
            self._open_position(symbol, quantity, fill_price, direction, timestamp, commission, fill_event.order_id)
        
        # Update portfolio state
        self._record_portfolio_state(timestamp)
        
        # Log the fill
        self.logger.info(f"Processed fill: {fill_event}")
    
    def _open_position(self, symbol, quantity, price, direction, timestamp, commission, order_id=None):
        """Open a new position."""
        # Create new position
        position = Position(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            entry_time=timestamp,
            direction=direction,
            order_id=order_id
        )
        
        # Update cash (subtract the position value + commission)
        position_value = price * quantity
        self.cash -= position_value + commission
        
        # Add to positions
        self.positions[symbol] = position
        
        # Record transaction
        self.transactions.append({
            "timestamp": timestamp,
            "type": "OPEN",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "order_id": order_id,
            "position_id": position.position_id
        })
        
        self.logger.info(f"Opened new position: {position}")
    
    def _increase_position(self, position, additional_quantity, price, timestamp, commission):
        """Increase an existing position."""
        # Calculate new average entry price
        total_quantity = position.quantity + additional_quantity
        new_entry_price = ((position.entry_price * position.quantity) + 
                          (price * additional_quantity)) / total_quantity
        
        # Update position
        position.entry_price = new_entry_price
        position.quantity = total_quantity
        
        # Update cash
        self.cash -= price * additional_quantity + commission
        
        # Record transaction
        self.transactions.append({
            "timestamp": timestamp,
            "type": "INCREASE",
            "symbol": position.symbol,
            "quantity": additional_quantity,
            "price": price,
            "commission": commission,
            "order_id": None,
            "position_id": position.position_id
        })
        
        self.logger.info(f"Increased position: {position.symbol}, new quantity: {total_quantity}")
    
    def _decrease_position(self, position, quantity_to_decrease, price, timestamp, commission, order_id=None):
        """Decrease or close an existing position."""
        if quantity_to_decrease >= position.quantity:
            # Closing the entire position
            realized_pnl = position.close(price, timestamp, "TRADE")
            
            # Update cash (add the position value - commission)
            self.cash += price * position.quantity - commission
            
            # Add realized P&L to equity
            self.equity += realized_pnl - commission
            
            # Record to position history
            self.position_history.append(position.to_dict())
            
            # Remove from open positions
            del self.positions[position.symbol]
            
            # Record transaction
            self.transactions.append({
                "timestamp": timestamp,
                "type": "CLOSE",
                "symbol": position.symbol,
                "quantity": position.quantity,
                "price": price,
                "commission": commission,
                "pnl": realized_pnl,
                "order_id": order_id,
                "position_id": position.position_id
            })
            
            self.logger.info(f"Closed position: {position.symbol}, PnL: {realized_pnl}")
        else:
            # Partially reducing the position
            # Calculate P&L for the closed portion
            if position.direction == "LONG":
                partial_pnl = (price - position.entry_price) * quantity_to_decrease
            else:  # SHORT
                partial_pnl = (position.entry_price - price) * quantity_to_decrease
            
            # Update position
            position.quantity -= quantity_to_decrease
            
            # Update cash
            self.cash += price * quantity_to_decrease - commission
            
            # Add partial realized P&L to equity
            self.equity += partial_pnl - commission
            
            # Record transaction
            self.transactions.append({
                "timestamp": timestamp,
                "type": "DECREASE",
                "symbol": position.symbol,
                "quantity": quantity_to_decrease,
                "price": price,
                "commission": commission,
                "pnl": partial_pnl,
                "order_id": order_id,
                "position_id": position.position_id
            })
            
            self.logger.info(f"Decreased position: {position.symbol}, new quantity: {position.quantity}")
    
    def update_portfolio(self, market_data: Dict[str, Dict[str, Any]]):
        """
        Update portfolio state based on current market data.
        
        Args:
            market_data: Dictionary mapping symbols to market data
        """
        timestamp = None
        total_unrealized_pnl = 0.0
        
        for symbol, position in list(self.positions.items()):
            if symbol in market_data:
                data = market_data[symbol]
                timestamp = data.get("timestamp", datetime.now())
                
                # Get current price (typically close price)
                current_price = data.get("Close", data.get("close", None))
                if current_price is None:
                    self.logger.warning(f"No price data found for {symbol}")
                    continue
                
                # Update unrealized P&L
                position.update_unrealized_pnl(current_price)
                total_unrealized_pnl += position.unrealized_pnl
                
                # Check for stop loss / take profit
                self._check_exit_conditions(position, current_price, timestamp)
        
        # Update equity
        self.equity = self.cash + total_unrealized_pnl
        
        # Record portfolio state
        if timestamp:
            self._record_portfolio_state(timestamp)
    
    def _check_exit_conditions(self, position, current_price, timestamp):
        """
        Check for stop loss / take profit conditions.
        
        Args:
            position: Position to check
            current_price: Current market price
            timestamp: Current timestamp
        """
        if not position.is_open:
            return
        
        # Check stop loss
        if position.stop_loss is not None:
            if (position.direction == "LONG" and current_price <= position.stop_loss) or \
               (position.direction == "SHORT" and current_price >= position.stop_loss):
                # Stop loss triggered
                realized_pnl = position.close(current_price, timestamp, "STOP_LOSS")
                
                # Update cash and equity
                self.cash += current_price * position.quantity
                self.equity += realized_pnl
                
                # Record to position history
                self.position_history.append(position.to_dict())
                
                # Remove from open positions
                del self.positions[position.symbol]
                
                # Record transaction
                self.transactions.append({
                    "timestamp": timestamp,
                    "type": "STOP_LOSS",
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "price": current_price,
                    "commission": 0.0,  # Simplified - no commission on stop loss
                    "pnl": realized_pnl,
                    "position_id": position.position_id
                })
                
                self.logger.info(f"Stop loss triggered for {position.symbol}, PnL: {realized_pnl}")
                return
        
        # Check take profit
        if position.take_profit is not None:
            if (position.direction == "LONG" and current_price >= position.take_profit) or \
               (position.direction == "SHORT" and current_price <= position.take_profit):
                # Take profit triggered
                realized_pnl = position.close(current_price, timestamp, "TAKE_PROFIT")
                
                # Update cash and equity
                self.cash += current_price * position.quantity
                self.equity += realized_pnl
                
                # Record to position history
                self.position_history.append(position.to_dict())
                
                # Remove from open positions
                del self.positions[position.symbol]
                
                # Record transaction
                self.transactions.append({
                    "timestamp": timestamp,
                    "type": "TAKE_PROFIT",
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "price": current_price,
                    "commission": 0.0,  # Simplified - no commission on take profit
                    "pnl": realized_pnl,
                    "position_id": position.position_id
                })
                
                self.logger.info(f"Take profit triggered for {position.symbol}, PnL: {realized_pnl}")
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve as a DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        return df
    
    def get_open_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dictionary mapping symbols to Position objects
        """
        return self.positions.copy()
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all closed positions.
        
        Returns:
            List of position dictionaries
        """
        return self.position_history.copy()
    
    def get_transactions(self) -> List[Dict[str, Any]]:
        """
        Get all transactions.
        
        Returns:
            List of transaction dictionaries
        """
        return self.transactions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol if it exists.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Position object or None
        """
        return self.positions.get(symbol)
    
    def get_position_value(self, symbol: str) -> float:
        """
        Get current value of a position.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Position value or 0.0 if no position
        """
        position = self.positions.get(symbol)
        return position.quantity * position.entry_price if position else 0.0
    
    def get_available_funds(self) -> float:
        """
        Get available funds for new positions.
        
        Returns:
            Available funds
        """
        # Simple implementation - just use cash
        return max(0.0, self.cash)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get summary of portfolio state.
        
        Returns:
            Portfolio summary dictionary
        """
        return {
            "equity": self.equity,
            "cash": self.cash,
            "initial_capital": self.initial_capital,
            "return": (self.equity / self.initial_capital - 1) * 100,
            "open_positions": len(self.positions),
            "closed_positions": len(self.position_history),
            "unrealized_pnl": sum(pos.unrealized_pnl for pos in self.positions.values()),
            "realized_pnl": sum(pos.get("pnl", 0) for pos in self.position_history),
            "leverage_used": self._calculate_leverage_used()
        }