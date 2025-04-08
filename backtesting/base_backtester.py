from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from core.events import Event, SignalEvent, OrderEvent, FillEvent
from core.portfolio.portfolio import Portfolio
from core.performance.performance import PerformanceTracker


class BaseBacktester(ABC):
    """Abstract base class for backtesting engines."""
    
    def __init__(self, 
                initial_capital: float = 10000.0,
                portfolio_cls=None,
                performance_tracker_cls=None,
                execution_handler_cls=None):
        """
        Initialize the backtester with configuration.
        
        Args:
            initial_capital: Starting capital for the portfolio
            portfolio_cls: Custom portfolio class (if None, uses default)
            performance_tracker_cls: Custom performance tracker (if None, uses default)
            execution_handler_cls: Custom execution handler (if None, uses default)
            logger: Custom logger (if None, creates default)
        """
        self.initial_capital = initial_capital
        
        # Create portfolio
        self.portfolio_cls = portfolio_cls or Portfolio
        self.portfolio = self.portfolio_cls(initial_capital=initial_capital, logger=logging)
        
        # Create performance tracker
        self.performance_tracker_cls = performance_tracker_cls or PerformanceTracker
        self.performance_tracker = self.performance_tracker_cls(logger=logging)
        
        # Create execution handler
        self.execution_handler_cls = execution_handler_cls
        self.execution_handler = None if not execution_handler_cls else execution_handler_cls(logger=logging)
        
        # Initialize event queue
        self.events = []
        
        # Results containers
        self.results = {}
        self.trade_history = []
        
        # State tracking
        self.current_date = None
        self.is_running = False
    
    @abstractmethod
    def run(self, strategy, data, **kwargs):
        """
        Run the backtest.
        
        Args:
            strategy: The trading strategy to backtest
            data: Historical data to use for the backtest
            **kwargs: Additional arguments for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        pass
    
    @abstractmethod
    def process_event(self, event: Event):
        """
        Process an event in the event queue.
        
        Args:
            event: The event to process
        """
        pass
    
    def add_event(self, event: Event):
        """
        Add an event to the event queue.
        
        Args:
            event: The event to add
        """
        self.events.append(event)
        logging.debug(f"Added {event.type} event to queue, queue length: {len(self.events)}")
    
    def generate_signals(self, strategy, data_point):
        """
        Generate trading signals using the strategy.
        
        Args:
            strategy: The trading strategy
            data_point: The current market data point
            
        Returns:
            List of signal events
        """
        signals = strategy.generate_signals(data_point, self.portfolio)
        
        if signals:
            for signal in signals:
                self.add_event(signal)
                logging.info(f"Generated signal: {signal}")
        
        return signals
    
    def calculate_performance(self):
        """
        Calculate performance metrics after the backtest.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_tracker.calculate_metrics(
            self.portfolio.history, 
            self.trade_history
        )
    
    def generate_report(self, output_path=None):
        """
        Generate a backtest report with key statistics and visualizations.
        
        Args:
            output_path: Path to save the report (if None, returns report data)
            
        Returns:
            Report data if output_path is None
        """
        performance_metrics = self.calculate_performance()
        
        # Combine performance metrics with other results
        report_data = {
            "performance_metrics": performance_metrics,
            "trade_summary": self._generate_trade_summary(),
            "equity_curve": self.portfolio.get_equity_curve(),
            "drawdowns": self.performance_tracker.calculate_drawdowns(),
            "monthly_returns": self.performance_tracker.calculate_period_returns("M"),
        }
        
        self.results.update(report_data)
        
        # If output path provided, save report to file
        if output_path:
            self._save_report(report_data, output_path)
            logging.info(f"Backtest report saved to {output_path}")
            return None
        
        return report_data
    
    def _generate_trade_summary(self):
        """Generate summary statistics for all trades."""
        if not self.trade_history:
            return {"no_trades": True}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        if len(trades_df) == 0:
            return {"no_trades": True}
            
        # Calculate trade statistics
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        summary = {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            "total_pnl": trades_df["pnl"].sum(),
            "avg_profit": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
            "largest_profit": trades_df["pnl"].max(),
            "largest_loss": trades_df["pnl"].min(),
            "profit_factor": (winning_trades["pnl"].sum() * -1) / losing_trades["pnl"].sum() 
                             if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0 else float('inf'),
            "avg_trade_duration": (trades_df["exit_time"] - trades_df["entry_time"]).mean()
                                 if "exit_time" in trades_df.columns and "entry_time" in trades_df.columns else None
        }
        
        return summary
    
    def _save_report(self, report_data, output_path):
        """Save report data to the specified path."""
        # Implementation depends on desired format (JSON, HTML, PDF, etc.)
        pass