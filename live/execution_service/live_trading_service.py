import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import threading
from queue import Queue

from core.events import SignalEvent, SignalType
from core.execution.live_execution import LiveExecutionHandler
from core.portfolio.portfolio import Portfolio
from core.risk.risk_manager import RiskManager
from broker.capital_com.capitalcom import CapitalCom
from live.live_data_handling.live_data_handler import LiveDataHandler


class LiveTradingService:
    """
    Service to manage live trading operations.
    Connects the model-based strategy to the CapitalCom broker.
    """
    
    def __init__(self, 
                 strategy, 
                 broker: CapitalCom,
                 data_handler: LiveDataHandler,
                 risk_manager: Optional[RiskManager] = None,
                 initial_capital: float = 10000.0,
                 max_active_positions: int = 3,
                 logger=None):
        """
        Initialize the live trading service.
        
        Args:
            strategy: Trading strategy instance
            broker: Broker client instance
            data_handler: Live data handler
            risk_manager: Risk manager instance
            initial_capital: Initial capital
            max_active_positions: Maximum number of active positions
            logger: Custom logger
        """
        self.strategy = strategy
        self.broker = broker
        self.data_handler = data_handler
        self.risk_manager = risk_manager or RiskManager()
        self.logger = logger or logging.getLogger(__name__)
        
        # Portfolio tracking
        print("Have capital ", initial_capital)
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.active_positions = {}  # symbol -> position details
        self.max_active_positions = max_active_positions
        
        # Create execution handler
        self.execution_handler = LiveExecutionHandler(broker_client=broker)
        
        # Service state
        self.is_running = False
        self.should_stop = False
        self.trading_thread = None
        
        # Signal queue for processing
        self.signal_queue = Queue()
        
    def start(self):
        """Start the trading service."""
        if self.is_running:
            self.logger.warning("Trading service is already running")
            return
            
        self.logger.info("Starting live trading service")
        
        # Initialize strategy
        self.strategy.initialize()
        self.strategy.on_backtest_start()  # Reuse this method for initialization
        
        # Start trading thread
        self.should_stop = False
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.is_running = True
        self.logger.info("Live trading service started")
    
    def stop(self):
        """Stop the trading service and close all positions."""
        if not self.is_running:
            self.logger.warning("Trading service is not running")
            return
            
        self.logger.info("Stopping live trading service")
        
        # Signal thread to stop
        self.should_stop = True
        
        # Wait for trading thread to exit
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10.0)
        
        # Close all open positions
        self._close_all_positions()
        
        # Call strategy cleanup
        self.strategy.on_backtest_end()  # Reuse this method for cleanup
        
        self.is_running = False
        self.logger.info("Live trading service stopped")
    
    def _trading_loop(self):
        """Main trading loop."""
        self.logger.info("Trading loop started")
        
        last_summary_time = time.time()
        
        while not self.should_stop:
            try:
                # Process any pending signals
                self._process_signals()
                
                # Get latest market data
                market_data = self._get_latest_market_data()
                if not market_data:
                    time.sleep(1)
                    continue
                
                # Update portfolio with latest market data
                self._update_portfolio(market_data)
                
                # Generate new signals from strategy
                signals = self.strategy.generate_signals(market_data, self.portfolio)
                
                # Add signals to queue
                if signals:
                    for signal in signals:
                        self.signal_queue.put(signal)
                    self.logger.info(f"Added {len(signals)} signals to queue")
                
                # Log a summary every minute
                if time.time() - last_summary_time > 60:
                    self._log_trading_summary()
                    last_summary_time = time.time()
                
                # Sleep to prevent CPU usage
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Sleep longer on error
        
        self.logger.info("Trading loop exited")
    
    def _process_signals(self):
        """Process signals from the queue."""
        # Process up to 10 signals per cycle
        for _ in range(10):
            if self.signal_queue.empty():
                break
                
            try:
                signal = self.signal_queue.get(block=False)
                self._execute_signal(signal)
                self.signal_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
    
    def _execute_signal(self, signal: SignalEvent):
        """
        Execute a trading signal.
        
        Args:
            signal: Signal event to execute
        """
        self.logger.info(f"Executing signal: {signal}")
        
        try:
            if signal.signal_type == SignalType.BUY:
                self._open_long_position(signal)
            elif signal.signal_type == SignalType.SELL:
                self._open_short_position(signal)
            elif signal.signal_type == SignalType.EXIT_LONG:
                self._close_position(signal.symbol)
            elif signal.signal_type == SignalType.EXIT_SHORT:
                self._close_position(signal.symbol)
            else:
                self.logger.warning(f"Unsupported signal type: {signal.signal_type}")
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    def _open_long_position(self, signal: SignalEvent):
        """
        Open a long position.
        
        Args:
            signal: Buy signal
        """
        # Check if we already have a position for this symbol
        if signal.symbol in self.active_positions:
            self.logger.info(f"Already have a position for {signal.symbol}, skipping")
            return
            
        # Check if we've reached the maximum positions
        if len(self.active_positions) >= self.max_active_positions:
            self.logger.info(f"Maximum positions reached ({self.max_active_positions}), skipping")
            return
        
        # Determine position size using risk manager
        current_price = self._get_current_price(signal.symbol)
        if not current_price:
            self.logger.warning(f"Could not get current price for {signal.symbol}, skipping")
            return
        
        # Calculate position size (with risk manager if available)
        position_size = 1.0  # Default size
        if self.risk_manager:
            position_size = self.risk_manager.position_sizer.calculate_position_size(
                symbol=signal.symbol,
                signal_type="BUY",
                current_price=current_price,
                portfolio=self.portfolio
            )
        
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if self.risk_manager and self.risk_manager.auto_stop_loss:
            stop_loss = self.risk_manager._calculate_stop_loss(
                symbol=signal.symbol,
                signal_type="BUY",
                current_price=current_price,
                market_data=None
            )
        
        if self.risk_manager and self.risk_manager.auto_take_profit:
            take_profit = self.risk_manager._calculate_take_profit(
                symbol=signal.symbol,
                signal_type="BUY",
                current_price=current_price,
                stop_loss=stop_loss,
                market_data=None
            )
        
        # Execute order through broker
        try:
            self.logger.info(f"Placing buy order for {signal.symbol}: {position_size} units at {current_price}")
            result = self.broker.place_market_order(
                symbol=signal.symbol,
                direction="BUY",
                size=position_size,
                stop_level=stop_loss,
                profit_level=take_profit
            )
            
            if result:
                self.logger.info(f"Buy order placed successfully for {signal.symbol}")
                
                # Track the position
                self.active_positions[signal.symbol] = {
                    'direction': 'LONG',
                    'size': position_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'deal_id': result  # Store the deal reference/ID
                }
                
                # Update strategy's position tracking
                self.strategy.update_position(signal.symbol, position_size)
            else:
                self.logger.warning(f"Failed to place buy order for {signal.symbol}")
        except Exception as e:
            self.logger.error(f"Error placing buy order: {e}")
    
    def _open_short_position(self, signal: SignalEvent):
        """
        Open a short position.
        
        Args:
            signal: Sell signal
        """
        # Check if we already have a position for this symbol
        if signal.symbol in self.active_positions:
            self.logger.info(f"Already have a position for {signal.symbol}, skipping")
            return
            
        # Check if we've reached the maximum positions
        if len(self.active_positions) >= self.max_active_positions:
            self.logger.info(f"Maximum positions reached ({self.max_active_positions}), skipping")
            return
        
        # Determine position size using risk manager
        current_price = self._get_current_price(signal.symbol)
        if not current_price:
            self.logger.warning(f"Could not get current price for {signal.symbol}, skipping")
            return
        
        # Calculate position size (with risk manager if available)
        position_size = 1.0  # Default size
        if self.risk_manager:
            position_size = self.risk_manager.position_sizer.calculate_position_size(
                symbol=signal.symbol,
                signal_type="SELL",
                current_price=current_price,
                portfolio=self.portfolio
            )
        
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if self.risk_manager and self.risk_manager.auto_stop_loss:
            stop_loss = self.risk_manager._calculate_stop_loss(
                symbol=signal.symbol,
                signal_type="SELL",
                current_price=current_price,
                market_data=None
            )
        
        if self.risk_manager and self.risk_manager.auto_take_profit:
            take_profit = self.risk_manager._calculate_take_profit(
                symbol=signal.symbol,
                signal_type="SELL",
                current_price=current_price,
                stop_loss=stop_loss,
                market_data=None
            )
        
        # Execute order through broker
        try:
            self.logger.info(f"Placing sell order for {signal.symbol}: {position_size} units at {current_price}")
            result = self.broker.place_market_order(
                symbol=signal.symbol,
                direction="SELL",
                size=position_size,
                stop_level=stop_loss,
                profit_level=take_profit
            )
            
            if result:
                self.logger.info(f"Sell order placed successfully for {signal.symbol}")
                
                # Track the position
                self.active_positions[signal.symbol] = {
                    'direction': 'SHORT',
                    'size': position_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'deal_id': result  # Store the deal reference/ID
                }
                
                # Update strategy's position tracking
                self.strategy.update_position(signal.symbol, -position_size)
            else:
                self.logger.warning(f"Failed to place sell order for {signal.symbol}")
        except Exception as e:
            self.logger.error(f"Error placing sell order: {e}")
    
    def _close_position(self, symbol: str):
        """
        Close a position.
        
        Args:
            symbol: Symbol to close position for
        """
        if symbol not in self.active_positions:
            self.logger.warning(f"No active position for {symbol}")
            return
        
        position = self.active_positions[symbol]
        
        try:
            self.logger.info(f"Closing position for {symbol}")
            
            # Use the dealId if available
            if 'deal_id' in position:
                result = self.broker.close_position(dealId=position['deal_id'])
            else:
                # Otherwise close based on the symbol
                # This is less reliable as there could be multiple positions
                result = self.broker.close_all_orders()
            
            if result:
                self.logger.info(f"Position closed successfully for {symbol}")
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                # Update strategy's position tracking
                self.strategy.update_position(symbol, 0)
            else:
                self.logger.warning(f"Failed to close position for {symbol}")
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _close_all_positions(self):
        """Close all open positions."""
        self.logger.info(f"Closing all positions: {len(self.active_positions)} positions")
        
        try:
            # Use broker's close_all_orders method
            result = self.broker.close_all_orders()
            
            if result:
                self.logger.info("All positions closed successfully")
                
                # Clear active positions
                for symbol in list(self.active_positions.keys()):
                    # Update strategy's position tracking
                    self.strategy.update_position(symbol, 0)
                
                self.active_positions = {}
            else:
                self.logger.warning("Failed to close all positions")
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if not available
        """
        latest_data = self.data_handler.get_latest_data()
        
        if not latest_data:
            return None
            
        if isinstance(latest_data, dict):
            # Direct access to latest data point
            if latest_data.get('epic') == symbol:
                return latest_data.get('close', latest_data.get('c'))
        
        # If we can't get price from latest data, try broker
        # (not implemented in this stub)
        
        return None
    
    def _get_latest_market_data(self) -> Dict[str, Any]:
        """
        Get latest market data for all symbols.
        
        Returns:
            Dictionary mapping symbols to market data
        """
        # Get latest data from data handler
        latest_data = self.data_handler.get_latest_data()
        
        if not latest_data:
            return {}
            
        # Format as expected by strategy
        # In the real implementation, you'd need to adapt this to match
        # what your strategy expects (typically a dict of symbol -> MarketEvent)
        from collections import namedtuple
        MarketEvent = namedtuple('MarketEvent', ['timestamp', 'data'])
        
        if isinstance(latest_data, dict):
            # Single data point
            symbol = latest_data.get('epic')
            timestamp = latest_data.get('datetime', datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            return {
                symbol: MarketEvent(
                    timestamp=timestamp,
                    data=latest_data
                )
            }
        else:
            # Multiple data points
            market_data = {}
            for data_point in latest_data:
                symbol = data_point.get('epic')
                timestamp = data_point.get('datetime', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)   
                market_data[symbol] = MarketEvent(
                    timestamp=timestamp,
                    data=data_point
                )
            return market_data
    
    def _update_portfolio(self, market_data: Dict[str, Any]):
        """
        Update portfolio with latest market data.
        
        Args:
            market_data: Dictionary of latest market data
        """
        # Extract just the data part for portfolio update
        data_dict = {}
        for symbol, event in market_data.items():
            data_dict[symbol] = event.data
        
        # Update portfolio
        self.portfolio.update_portfolio(data_dict)
    
    def _log_trading_summary(self):
        """Log a summary of current trading status."""
        positions_info = [
            f"{symbol}: {pos['direction']} {pos['size']} units" 
            for symbol, pos in self.active_positions.items()
        ]
        
        positions_str = ', '.join(positions_info) if positions_info else "None"
        
        self.logger.info(f"Trading summary - Active positions: {positions_str}")
        self.logger.info(f"Portfolio equity: {self.portfolio.equity:.2f}, Cash: {self.portfolio.cash:.2f}")
        
        # Get pending signals count
        pending_signals = self.signal_queue.qsize()
        self.logger.info(f"Pending signals: {pending_signals}")
    
    def modify_position(self, symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """
        Modify an existing position's stop loss or take profit.
        
        Args:
            symbol: Symbol to modify position for
            stop_loss: New stop loss level
            take_profit: New take profit level
        """
        if symbol not in self.active_positions:
            self.logger.warning(f"No active position for {symbol}")
            return
        
        position = self.active_positions[symbol]
        
        try:
            self.logger.info(f"Modifying position for {symbol}: SL={stop_loss}, TP={take_profit}")
            
            if 'deal_id' in position:
                result = self.broker.modify_position(
                    dealId=position['deal_id'],
                    stop_level=stop_loss,
                    profit_level=take_profit
                )
                
                if result:
                    self.logger.info(f"Position modified successfully for {symbol}")
                    
                    # Update tracked position
                    if stop_loss is not None:
                        self.active_positions[symbol]['stop_loss'] = stop_loss
                    if take_profit is not None:
                        self.active_positions[symbol]['take_profit'] = take_profit
                else:
                    self.logger.warning(f"Failed to modify position for {symbol}")
            else:
                self.logger.warning(f"No deal ID for {symbol}, cannot modify")
        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")