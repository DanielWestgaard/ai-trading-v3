#!/usr/bin/env python
"""
Example showing integration between the data pipeline and backtesting framework.

This script demonstrates how to:
1. Process raw data with the data pipeline
2. Pass the processed data to the backtesting system
3. Run multiple strategy backtests on the same data
4. Compare the results
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipelines.data_pipeline import DataPipeline
from backtesting.backtest_runner import BacktestRunner
from backtesting.strategies.simple_ma_crossover import SimpleMovingAverageCrossover, MACDStrategy
from backtesting.data.market_data import PipelineMarketData
from backtesting.risk_manager import RiskManager


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def process_data(raw_data_path, output_dir):
    """
    Process raw data using the data pipeline.
    
    Args:
        raw_data_path: Path to raw data file
        output_dir: Output directory for processed data
        
    Returns:
        Path to processed data file
    """
    logging.info(f"Processing raw data: {raw_data_path}")
    
    # Configure the data pipeline
    pipeline = DataPipeline(
        feature_treatment_mode='advanced',
        price_transform_method='returns',
        normalization_method='zscore',
        feature_selection_method='threshold',
        feature_importance_threshold=0.01,
        target_column='close_return'
    )
    
    # Run the pipeline
    processed_data, processed_file_path = pipeline.run(
        raw_data=raw_data_path,
        target_path=output_dir,
        save_intermediate=True,
        run_feature_selection=True
    )
    
    logging.info(f"Data processing complete. Output: {processed_file_path}")
    return processed_file_path


def run_backtests(processed_data_path, output_dir, symbol='UNKNOWN'):
    """
    Run multiple strategy backtests on the processed data.
    
    Args:
        processed_data_path: Path to processed data file
        output_dir: Output directory for backtest results
        symbol: Symbol name
        
    Returns:
        Backtest runner instance with results
    """
    logging.info(f"Running backtests on processed data: {processed_data_path}")
    
    # Create backtest runner
    runner = BacktestRunner(output_dir=output_dir)
    
    # Create market data
    market_data = PipelineMarketData(
        processed_data_path=processed_data_path,
        symbols=[symbol]
    )
    
    # Create risk manager with different position sizing strategies
    risk_manager_fixed = RiskManager(
        position_sizing_method='fixed',
        position_sizing_params={'size': 1.0},
        auto_stop_loss=True,
        stop_loss_method='percent',
        stop_loss_params={'percent': 2.0}
    )
    
    risk_manager_percent = RiskManager(
        position_sizing_method='percent',
        position_sizing_params={'percent': 10.0},
        auto_stop_loss=True,
        stop_loss_method='percent',
        stop_loss_params={'percent': 2.0}
    )
    
    risk_manager_risk = RiskManager(
        position_sizing_method='risk',
        position_sizing_params={'risk_percent': 1.0, 'stop_loss_percent': 2.0},
        auto_stop_loss=True,
        stop_loss_method='percent',
        stop_loss_params={'percent': 2.0}
    )
    
    # Create and run MA strategy backtest with different parameters
    ma_strategy_20_50 = SimpleMovingAverageCrossover(
        symbols=[symbol],
        params={
            'short_window': 20,
            'long_window': 50,
            'use_sma': True
        }
    )
    
    ma_strategy_10_30 = SimpleMovingAverageCrossover(
        symbols=[symbol],
        params={
            'short_window': 10,
            'long_window': 30,
            'use_sma': True
        }
    )
    
    # Create and run MACD strategy backtest
    macd_strategy = MACDStrategy(
        symbols=[symbol],
        params={
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
    )
    
    # Run MA 20/50 backtest with fixed position sizing
    runner.create_backtest(
        backtest_id=f"MA_20_50_fixed_{symbol}",
        strategy=ma_strategy_20_50,
        market_data=market_data,
        initial_capital=10000.0,
        risk_manager=risk_manager_fixed
    )
    runner.run_backtest()
    
    # Run MA 10/30 backtest with percent position sizing
    runner.create_backtest(
        backtest_id=f"MA_10_30_percent_{symbol}",
        strategy=ma_strategy_10_30,
        market_data=market_data,
        initial_capital=10000.0,
        risk_manager=risk_manager_percent
    )
    runner.run_backtest()
    
    # Run MACD backtest with risk-based position sizing
    runner.create_backtest(
        backtest_id=f"MACD_risk_{symbol}",
        strategy=macd_strategy,
        market_data=market_data,
        initial_capital=10000.0,
        risk_manager=risk_manager_risk
    )
    runner.run_backtest()
    
    return runner


def compare_and_visualize(runner, output_dir):
    """
    Compare and visualize backtest results.
    
    Args:
        runner: BacktestRunner instance with results
        output_dir: Output directory for comparison results
    """
    logging.info("Comparing backtest results")
    
    # Compare all backtests
    backtest_ids = list(runner.results.keys())
    
    comparison_df = runner.compare_backtests(
        backtest_ids=backtest_ids,
        output_path=os.path.join(output_dir, 'backtest_comparison.csv')
    )
    
    # Print comparison table
    print("\n" + "="*100)
    print("Backtest Comparison")
    print("="*100)
    print(comparison_df)
    print("="*100)
    
    # Plot equity comparison
    fig = runner.plot_equity_comparison(
        backtest_ids=backtest_ids,
        title='Strategy Comparison',
        save_path=os.path.join(output_dir, 'equity_comparison.png')
    )
    
    # Show key metrics as a bar chart
    metrics_to_plot = ['return', 'max_drawdown_pct', 'sharpe_ratio', 'win_rate_pct']
    
    for metric in metrics_to_plot:
        if metric in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            ax = comparison_df.plot(
                kind='bar',
                x='backtest_id',
                y=metric,
                title=f'Comparison: {metric}',
                rot=45
            )
            
            # Add value labels on top of bars
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom'
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
    
    logging.info(f"Comparison results saved to {output_dir}")


def main():
    """Main function."""
    setup_logging()
    
    # Configuration
    raw_data_path = "storage/capital_com/raw/raw_GBPUSD_m5_20240101_20250101.csv"
    symbol = "GBPUSD"
    
    # Create output directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "storage", "capital_com", "processed")
    results_dir = os.path.join(base_dir, "backtest_results", f"pipeline_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Process data if raw data file exists
    if os.path.exists(raw_data_path):
        processed_file_path = process_data(raw_data_path, processed_dir)
    else:
        # If raw data doesn't exist, look for a processed file
        logging.warning(f"Raw data file not found: {raw_data_path}")
        processed_files = [f for f in os.listdir(processed_dir) if f.startswith(f"processed_{symbol}")]
        
        if processed_files:
            processed_file_path = os.path.join(processed_dir, processed_files[0])
            logging.info(f"Using existing processed file: {processed_file_path}")
        else:
            logging.error("No processed data found. Please provide raw data or processed file.")
            return
    
    # Run backtests
    runner = run_backtests(processed_file_path, results_dir, symbol)
    
    # Compare and visualize results
    compare_and_visualize(runner, results_dir)
    
    logging.info("Pipeline and backtest integration test completed successfully")


if __name__ == '__main__':
    main()