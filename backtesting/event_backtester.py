from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time

from backtesting.base_backtester import BaseBacktester
from core.events import Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent, OrderSide, OrderType
from backtesting.data.market_data import MarketData
from core.execution.simulation_execution import SimulationExecutionHandler


class EventDrivenBacktester(BaseBacktester):
    """Event-driven backtesting engine."""
    
    def __init__(self, 
                initial_capital: float = 10000.0,
                portfolio_cls=None,
                performance_tracker_cls=None,
                execution_handler=None,  # Now accepts any BaseExecutionHandler
                commission_model=None,
                slippage_model="fixed",
                slippage_params=None):
        """
        Initialize the event-driven backtester.
        
        Args:
            initial_capital: Starting capital for the portfolio
            portfolio_cls: Custom portfolio class
            performance_tracker_cls: Custom performance tracker class
            execution_handler: Custom execution handler (must implement BaseExecutionHandler)
            commission_model: Commission model
            slippage_model: Slippage model type (used only if no execution_handler provided)
            slippage_params: Parameters for slippage model (used only if no execution_handler provided)
        """
        super().__init__(
            initial_capital=initial_capital,
            portfolio_cls=portfolio_cls,
            performance_tracker_cls=performance_tracker_cls,
            execution_handler_cls=None)  # We'll handle execution handler differently
        
        # Create execution handler if not provided
        if execution_handler is None:
            self.execution_handler = SimulationExecutionHandler(
                slippage_model=slippage_model,
                slippage_params=slippage_params)
        else:
            # Use the provided execution handler
            self.execution_handler = execution_handler
        
        # Set commission model
        self.commission_model = commission_model
    
    def run(self, strategy, market_data: MarketData, **kwargs):
        """
        Run the backtest.
        
        Args:
            strategy: Trading strategy to test
            market_data: Market data provider
            **kwargs: Additional arguments
                - start_date: Start date for backtest
                - end_date: End date for backtest
                - verbose: Verbosity level
                
        Returns:
            Dictionary of backtest results
        """
        # Extract kwargs
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        verbose = kwargs.get('verbose', False)
        
        # Reset market data
        market_data.reset()
        
        # Initialize tracking variables
        self.is_running = True
        self.current_date = None
        self.events = []
        self.results = {}
        self.trade_history = []
        
        # Log start of backtest
        logging.info(f"Starting backtest with {strategy.__class__.__name__}")
        logging.info(f"Initial capital: {self.initial_capital}")
        logging.info(f"Market data: {len(market_data.get_symbols())} symbols, {market_data.get_length()} data points")
        
        # Initialize strategy
        strategy.on_backtest_start()
        
        # Main event loop
        while market_data.has_more_data() and self.is_running:
            # Get next market data
            market_events = market_data.get_next()
            
            if not market_events:
                logging.debug("No market events, continuing")
                continue
            
            # Store current market data as instance variable for order execution
            self.current_market_data = {symbol: event.data for symbol, event in market_events.items()}
            
            # Update current date from first market event
            first_symbol = next(iter(market_events))
            self.current_date = market_events[first_symbol].timestamp
            
            # Check date range
            if start_date and self.current_date < start_date:
                continue
            if end_date and self.current_date > end_date:
                break
            
            if verbose:
                logging.info(f"Processing data for {self.current_date}")
            else:
                logging.debug(f"Processing data for {self.current_date}")
            
            # Add market events to queue
            for symbol, event in market_events.items():
                self.add_event(event)
            
            # Process all events
            self._process_events(market_events)
            
            # Update portfolio with current market data
            market_data_dict = {symbol: event.data for symbol, event in market_events.items()}
            self.portfolio.update_portfolio(market_data_dict)
            
            # Generate signals from strategy
            signals = strategy.generate_signals(market_events, self.portfolio)
            
            # Process signals
            if signals:
                for signal in signals:
                    self.process_signal(signal)
            
            # Process any remaining events in the queue
            while self.events:
                event = self.events.pop(0)
                self.process_event(event)
        
        # Finalize backtest
        strategy.on_backtest_end()
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance()
        
        # Create results
        self.results = {
            "portfolio_history": self.portfolio.get_equity_curve(),
            "trade_history": self.portfolio.get_position_history(),
            "performance_metrics": performance_metrics,
            "final_equity": self.portfolio.equity,
            "return": (self.portfolio.equity / self.initial_capital - 1) * 100,
            "total_trades": len(self.portfolio.get_position_history()),
            "strategy": strategy.__class__.__name__,
            "strategy_params": strategy.get_parameters()
        }
        
        logging.info(f"Backtest completed for {strategy.__class__.__name__}")
        logging.info(f"Final equity: {self.portfolio.equity:.2f} (Return: {self.results['return']:.2f}%)")
        logging.info(f"Total trades: {self.results['total_trades']}")
        
        return self.results
    
    def _process_events(self, market_events):
        """
        Process all events in the queue.
        
        Args:
            market_events: Current market events
        """
        # First process any events already in the queue
        while self.events:
            event = self.events.pop(0)
            self.process_event(event)
    
    def process_event(self, event: Event):
        """
        Process an event based on its type.
        
        Args:
            event: Event to process
        """
        if event.type == EventType.MARKET:
            # Market events are already handled
            pass
        
        elif event.type == EventType.SIGNAL:
            self.process_signal(event)
        
        elif event.type == EventType.ORDER:
            self.process_order(event)
        
        elif event.type == EventType.FILL:
            self.process_fill(event)
    
    def process_signal(self, signal: SignalEvent):
        """
        Process a signal event by generating an order.
        
        Args:
            signal: Signal event
        """
        logging.debug(f"Processing signal: {signal}")
        
        # Create corresponding order
        if signal.signal_type.value in ["BUY", "SELL", "REVERSE"]:
            # Determine order side
            if signal.signal_type.value == "BUY":
                order_side = OrderSide.BUY
            elif signal.signal_type.value == "SELL":
                order_side = OrderSide.SELL
            elif signal.signal_type.value == "REVERSE":
                # For REVERSE, check current position and do the opposite
                current_position = self.portfolio.get_position(signal.symbol)
                if current_position and current_position.direction == "LONG":
                    order_side = OrderSide.SELL
                else:
                    order_side = OrderSide.BUY
            
            # Simple fixed quantity for now
            quantity = 1.0
            
            # Create order
            order = OrderEvent(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                order_type=OrderType.MARKET,
                order_side=order_side,
                quantity=quantity,
                signal_id=id(signal)
            )
            
            # Add order to queue
            self.add_event(order)
            logging.debug(f"Created order from signal: {order}")
        
        # Add this new code to handle exit signals
        elif signal.signal_type.value in ["EXIT_LONG", "EXIT_SHORT"]:
            # For EXIT signals, determine the correct side to exit
            order_side = OrderSide.SELL if signal.signal_type.value == "EXIT_LONG" else OrderSide.BUY
            
            # Get current position to determine quantity
            current_position = self.portfolio.get_position(signal.symbol)
            quantity = current_position.quantity if current_position else 0
            
            if quantity > 0:
                # Create order to close position
                order = OrderEvent(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    order_type=OrderType.MARKET,
                    order_side=order_side,
                    quantity=quantity,
                    signal_id=id(signal)
                )
                
                # Add order to queue
                self.add_event(order)
                logging.debug(f"Created exit order from signal: {order}")
                
    def process_order(self, order: OrderEvent):
        """
        Process an order event by simulating execution.
        
        Args:
            order: Order event
        """
        logging.debug(f"Processing order: {order}")
        
        # Get current market data for the symbol
        if hasattr(self, 'current_market_data') and self.current_market_data:
            market_data = self.current_market_data
        else:
            logging.warning("No current market data available, using empty dict")
            market_data = {}
        
        # Execute the order
        fill = self.execution_handler.execute_order(order, market_data)
        
        if fill:
            # Add fill event to queue
            self.add_event(fill)
        else:
            logging.warning(f"Order not filled: {order}")
    
    def process_fill(self, fill: FillEvent):
        """
        Process a fill event by updating the portfolio.
        
        Args:
            fill: Fill event
        """
        logging.debug(f"Processing fill: {fill}")
        
        # Update portfolio
        self.portfolio.process_fill(fill)
        
        # Record the trade
        self._record_trade(fill)
    
    def _record_trade(self, fill: FillEvent):
        """
        Record a trade in the trade history.
        
        Args:
            fill: Fill event
        """
        # Get position if available
        position = None
        if hasattr(self.portfolio, 'get_position'):
            position = self.portfolio.get_position(fill.symbol)
        
        # Calculate PnL if this is a closing trade
        pnl = 0.0
        if position and hasattr(position, 'pnl'):
            pnl = position.pnl
        
        trade = {
            "symbol": fill.symbol,
            "timestamp": fill.timestamp,
            "quantity": fill.quantity,
            "price": fill.fill_price,
            "side": fill.order_side.value,
            "commission": fill.commission,
            "order_id": fill.order_id,
            "pnl": pnl  # Add PnL field
        }
        
        self.trade_history.append(trade)