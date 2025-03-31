#!/usr/bin/env python
"""
Example script for running a backtest using the backtesting framework.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative
from backtesting.backtest_runner import BacktestRunner
from backtesting.strategies.simple_ma_crossover import SimpleMovingAverageCrossover, MACDStrategy
from backtesting.data.market_data import CSVMarketData, PipelineMarketData
import utils.logging_utils as log_utils
import config.system_config as sys_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a backtest')
    
    # Required arguments
    parser.add_argument('--config', '-c', type=str, help='Path to backtest configuration file')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory for results')
    parser.add_argument('--backtest-id', '-id', type=str, help='Backtest identifier')
    parser.add_argument('--data-path', '-d', type=str, help='Path to data file')
    parser.add_argument('--strategy', '-s', type=str, help='Strategy class name (MA or MACD)')
    parser.add_argument('--symbol', type=str, help='Symbol to trade')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--short-window', type=int, default=20, help='Short moving average window')
    parser.add_argument('--long-window', type=int, default=50, help='Long moving average window')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logging = log_utils.setup_logging(log_to_file=False, log_level=sys_config.DEFAULT_LOG_LEVEL)
    
    # Create output directory
    output_dir = args.output_dir or os.path.join(os.getcwd(), 'backtest_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create backtest runner
    runner = BacktestRunner(
        config_path=args.config,
        output_dir=output_dir,
        log_level=sys_config.DEBUG_LOG_LEVEL
    )
    
    # If config file was provided, run backtest from config
    if args.config:
        logging.info(f"Running backtest from configuration: {args.config}")
        
        # Get backtest ID from config or command line
        backtest_id = args.backtest_id or runner.config.get('backtest_id')
        if not backtest_id:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backtest_id = f"backtest_{timestamp}"
        
        # Create backtest from config
        runner.create_backtest(
            backtest_id=backtest_id,
            strategy=runner.config.get('strategy', {}),
            market_data=runner.config.get('market_data', {}),
            initial_capital=runner.config.get('initial_capital', 10000.0),
            risk_manager_config=runner.config.get('risk_manager'),
            slippage_model=runner.config.get('slippage_model', 'fixed'),
            slippage_params=runner.config.get('slippage_params')
        )
    else:
        # Create backtest from command line arguments
        logging.info("Creating backtest from command line arguments")
        
        # Create backtest ID if not provided
        backtest_id = args.backtest_id
        if not backtest_id:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_name = args.strategy or 'MA'
            symbol = args.symbol or 'UNKNOWN'
            backtest_id = f"{strategy_name}_{symbol}_{timestamp}"
        
        # Check if data path is provided
        if not args.data_path:
            logging.error("Data path is required when not using a config file")
            return
        
        # Create market data
        if os.path.isdir(args.data_path):
            # Assume it's a directory with CSV files
            market_data = CSVMarketData(
                symbols=[args.symbol] if args.symbol else ['UNKNOWN'],
                csv_dir=args.data_path
            )
        else:
            # Assume it's a processed data file
            market_data = PipelineMarketData(
                processed_data_path=args.data_path,
                symbols=[args.symbol] if args.symbol else None
            )
        
        # Create strategy
        strategy_class = MACDStrategy if args.strategy == 'MACD' else SimpleMovingAverageCrossover
        strategy = strategy_class(
            symbols=[args.symbol] if args.symbol else market_data.get_symbols(),
            params={
                'short_window': args.short_window,
                'long_window': args.long_window
            }
        )
        
        # Create backtest
        runner.create_backtest(
            backtest_id=backtest_id,
            strategy=strategy,
            market_data=market_data,
            initial_capital=args.initial_capital
        )
    
    # Run the backtest
    logging.info(f"Running backtest: {runner.current_backtest}")
    results = runner.run_backtest()
    
    # Display summary
    runner.display_summary_table()
    
    logging.info(f"Backtest completed. Results saved to {os.path.join(output_dir, runner.current_backtest)}")


if __name__ == '__main__':
    main()