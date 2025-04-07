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

from backtesting.strategies.model_based_strategy import ModelBasedStrategy

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


def main(config_file:str=None):
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logging = log_utils.setup_logging(name="backtesting", type="BACKTEST", log_to_file=False, log_level=sys_config.DEFAULT_LOG_LEVEL)
    
    # Create output directory
    output_dir = args.output_dir or os.path.join(os.getcwd(), 'backtest_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create backtest runner
    runner = BacktestRunner(
        config_path=args.config or config_file,
        output_dir=output_dir
    )
    
    # If config file was provided, run backtest from config
    if args.config or config_file:
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
    
    
def run_backtest(model, data_path, backtest_config):
    """Run a backtest using the trained model."""
    logging.info("Setting up backtest with configuration:")
    logging.info(json.dumps(backtest_config, indent=2))
    
    # Create backtest runner
    runner = BacktestRunner(
        output_dir=backtest_config.get('output_dir', 'backtest_results')
    )
    
    # Set up market data
    market_data_config = backtest_config.get('market_data', {})
    market_data = PipelineMarketData(
        processed_data_path=data_path,
        symbols=market_data_config.get('symbols', ['GBPUSD']),
        date_col=market_data_config.get('date_col', 'date')
    )
    
    # Set up model-based strategy
    required_features = model.features
    logging.info(f"Using {len(required_features)} features for model predictions")
    
    strategy = ModelBasedStrategy(
        symbols=market_data_config.get('symbols', ['GBPUSD']),
        model=model,
        prediction_threshold=backtest_config.get('prediction_threshold', 0.55),
        confidence_threshold=backtest_config.get('confidence_threshold', 0.0),
        lookback_window=backtest_config.get('lookback_window', 1),
        required_features=required_features
    )
    
    # Set up and run backtest
    backtest_id = backtest_config.get('backtest_id', f'model_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Create backtest
    runner.create_backtest(
        backtest_id=backtest_id,
        strategy=strategy,
        market_data=market_data,
        initial_capital=backtest_config.get('initial_capital', 10000.0),
        risk_manager_config=backtest_config.get('risk_manager'),
        slippage_model=backtest_config.get('slippage_model', 'fixed'),
        slippage_params=backtest_config.get('slippage_params')
    )
    
    # Run backtest
    results = runner.run_backtest(backtest_id)
    
    # Display summary
    runner.display_summary_table(backtest_id)
    
    return results, runner

def main_backtest_trained_model(model, DATA_PATH):
    # Backtest configuration
    backtest_config = {
        'backtest_id': f'xgboost_gbpusd_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'market_data': {
            'symbols': ['GBPUSD'],
            'date_col': 'date'
        },
        'initial_capital': 10000.0,
        'prediction_threshold': 0.55,
        'confidence_threshold': 0.6,
        'lookback_window': 1,
        'risk_manager': {
            'position_sizing_method': 'percent',
            'position_sizing_params': {
                'percent': 2.0  # Use 2% of equity per trade
            }
        }
    }
    
    # Run backtest
    results, runner = run_backtest(model, DATA_PATH, backtest_config)
    
    # Final results
    backtest_id = backtest_config['backtest_id']
    logging.info(f"Completed backtest: {backtest_id}")
    
    # Compare with baseline
    baseline_id = "indicator_crossover_gbpusd"  # Replace with your baseline strategy
    
    try:
        comparison = runner.compare_backtests([backtest_id, baseline_id])
        logging.info(f"Comparison with baseline:\n{comparison}")
        
        # Plot comparison
        runner.plot_equity_comparison(
            [backtest_id, baseline_id],
            title="Model vs Indicator Strategy",
            save_path=f"backtest_results/{backtest_id}/reports/comparison.png"
        )
    except Exception as e:
        logging.warning(f"Could not compare with baseline: {str(e)}")
