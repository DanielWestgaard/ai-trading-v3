#!/usr/bin/env python
"""
Example script to create a strategy backtest with custom analysis and visualization.
This would typically be done in a Jupyter notebook but is shown as a Python script.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_runner import BacktestRunner
from backtesting.strategies.simple_ma_crossover import SimpleMovingAverageCrossover
from backtesting.data.market_data import CSVMarketData
from backtesting.visualization import BacktestVisualizer


# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


def load_data(csv_dir, symbol, date_col='Date'):
    """Load market data from CSV file."""
    print(f"Loading data for {symbol}...")
    
    market_data = CSVMarketData(
        symbols=[symbol],
        csv_dir=csv_dir,
        date_col=date_col
    )
    
    print(f"Loaded {market_data.get_length()} data points")
    return market_data


def optimize_strategy_parameters(market_data, symbol, initial_capital=10000.0):
    """
    Find the best MA parameters by running multiple backtests.
    
    Args:
        market_data: Market data
        symbol: Symbol to trade
        initial_capital: Initial capital
        
    Returns:
        DataFrame with optimization results
    """
    print("Optimizing strategy parameters...")
    
    # Define parameter ranges to test
    short_windows = [5, 10, 15, 20, 25]
    long_windows = [30, 40, 50, 60, 70]
    
    # Create backtest runner
    runner = BacktestRunner(
        output_dir='backtest_results/optimization',
        log_level=30  # WARNING level to reduce output
    )
    
    results = []
    
    # Run backtests for each parameter combination
    for short_window in short_windows:
        for long_window in long_windows:
            # Skip invalid combinations
            if short_window >= long_window:
                continue
            
            # Create strategy
            strategy = SimpleMovingAverageCrossover(
                symbols=[symbol],
                params={
                    'short_window': short_window,
                    'long_window': long_window,
                    'use_sma': True
                }
            )
            
            # Create and run backtest
            backtest_id = f"MA_{short_window}_{long_window}"
            runner.create_backtest(
                backtest_id=backtest_id,
                strategy=strategy,
                market_data=market_data,
                initial_capital=initial_capital
            )
            
            backtest_results = runner.run_backtest(
                save_results=False,
                generate_reports=False
            )
            
            # Extract key metrics
            metrics = backtest_results.get('performance_metrics', {})
            
            results.append({
                'short_window': short_window,
                'long_window': long_window,
                'return': backtest_results.get('return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'win_rate_pct': metrics.get('win_rate_pct', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_trades': backtest_results.get('total_trades', 0)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best parameters based on Sharpe ratio
    best_params = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    print(f"Optimization complete. Best parameters:")
    print(f"Short Window: {best_params['short_window']}")
    print(f"Long Window: {best_params['long_window']}")
    print(f"Return: {best_params['return']:.2f}%")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    
    return results_df, best_params


def visualize_optimization_results(results_df):
    """
    Visualize optimization results as heatmaps.
    
    Args:
        results_df: DataFrame with optimization results
    """
    # Create heatmaps for key metrics
    metrics = ['return', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Pivot data for heatmap
        pivot_data = results_df.pivot_table(
            index='short_window',
            columns='long_window',
            values=metric
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap='viridis' if metric not in ['max_drawdown_pct'] else 'viridis_r',
            ax=axes[i]
        )
        
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.savefig('optimization_heatmaps.png', dpi=300)
    plt.show()


def run_final_backtest(market_data, symbol, short_window, long_window, initial_capital=10000.0):
    """
    Run a backtest with the optimal parameters.
    
    Args:
        market_data: Market data
        symbol: Symbol to trade
        short_window: Short MA window
        long_window: Long MA window
        initial_capital: Initial capital
        
    Returns:
        BacktestRunner instance with results
    """
    print(f"Running final backtest with parameters: Short={short_window}, Long={long_window}")
    
    # Create strategy
    strategy = SimpleMovingAverageCrossover(
        symbols=[symbol],
        params={
            'short_window': short_window,
            'long_window': long_window,
            'use_sma': True
        }
    )
    
    # Create backtest runner
    runner = BacktestRunner(
        output_dir='backtest_results/final',
        log_level=20  # INFO level
    )
    
    # Create and run backtest
    runner.create_backtest(
        backtest_id=f"MA_{short_window}_{long_window}_final",
        strategy=strategy,
        market_data=market_data,
        initial_capital=initial_capital
    )
    
    runner.run_backtest()
    
    # Display summary
    runner.display_summary_table()
    
    return runner


def custom_analysis(runner):
    """
    Perform custom analysis on the backtest results.
    
    Args:
        runner: BacktestRunner instance with results
    """
    print("Performing custom analysis...")
    
    # Get results
    backtest_id = runner.current_backtest
    results = runner.results[backtest_id]
    
    # Get portfolio history and trade history
    portfolio_history = results['portfolio_history']
    trade_history = pd.DataFrame(results['trade_history']) if 'trade_history' in results else None
    
    # Analyze monthly returns
    if 'timestamp' in portfolio_history.columns:
        portfolio_history['month'] = portfolio_history['timestamp'].dt.strftime('%Y-%m')
        
        monthly_returns = portfolio_history.groupby('month').apply(
            lambda x: (x['equity'].iloc[-1] / x['equity'].iloc[0] - 1) * 100
        ).reset_index()
        monthly_returns.columns = ['month', 'return']
        
        print("\nMonthly Returns:")
        print(monthly_returns.sort_values('return', ascending=False))
        
        # Plot monthly returns
        plt.figure(figsize=(12, 6))
        plt.bar(monthly_returns['month'], monthly_returns['return'])
        plt.title('Monthly Returns (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('monthly_returns.png', dpi=300)
        plt.show()
    
    # Analyze trades
    if trade_history is not None and len(trade_history) > 0:
        if 'entry_time' in trade_history.columns and 'exit_time' in trade_history.columns:
            # Convert to datetime
            trade_history['entry_time'] = pd.to_datetime(trade_history['entry_time'])
            trade_history['exit_time'] = pd.to_datetime(trade_history['exit_time'])
            
            # Calculate holding period
            trade_history['holding_period'] = (
                trade_history['exit_time'] - trade_history['entry_time']
            ).dt.total_seconds() / (60 * 60 * 24)  # Convert to days
            
            # Analyze holding period vs. return
            plt.figure(figsize=(10, 6))
            plt.scatter(
                trade_history['holding_period'],
                trade_history['pnl'],
                alpha=0.7
            )
            plt.title('Holding Period vs. P&L')
            plt.xlabel('Holding Period (days)')
            plt.ylabel('P&L')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('holding_period_vs_pnl.png', dpi=300)
            plt.show()
            
            # Analyze trade entry times
            trade_history['hour'] = trade_history['entry_time'].dt.hour
            
            # Count trades by hour
            trades_by_hour = trade_history.groupby('hour').size()
            
            # Calculate win rate by hour
            win_rate_by_hour = trade_history[trade_history['pnl'] > 0].groupby('hour').size() / trades_by_hour
            
            # Plot win rate by hour
            plt.figure(figsize=(12, 6))
            win_rate_by_hour.plot(kind='bar', color='green', alpha=0.7)
            plt.title('Win Rate by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Win Rate')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('win_rate_by_hour.png', dpi=300)
            plt.show()


def main():
    """Main function."""
    # Configuration
    csv_dir = 'data/sample'
    symbol = 'EURUSD'
    initial_capital = 10000.0
    
    # Load market data
    market_data = load_data(csv_dir, symbol)
    
    # Optimize strategy parameters
    results_df, best_params = optimize_strategy_parameters(market_data, symbol, initial_capital)
    
    # Visualize optimization results
    visualize_optimization_results(results_df)
    
    # Run final backtest with best parameters
    runner = run_final_backtest(
        market_data=market_data,
        symbol=symbol,
        short_window=int(best_params['short_window']),
        long_window=int(best_params['long_window']),
        initial_capital=initial_capital
    )
    
    # Custom analysis
    custom_analysis(runner)
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()