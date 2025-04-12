import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
from typing import Dict, List, Optional, Union, Any
import importlib

from backtesting.event_backtester import EventDrivenBacktester
from backtesting.data.market_data import CSVMarketData, DataFrameMarketData, PipelineMarketData
from core.portfolio.portfolio import Portfolio
from core.performance.performance import PerformanceTracker
from backtesting.visualization import BacktestVisualizer
from core.risk.risk_manager import RiskManager
import config.constants.system_config as sys_config


class BacktestRunner:
    """
    Main class for running backtests and managing backtest results.
    
    This class provides a convenient interface for:
    1. Loading data and strategies
    2. Configuring and running backtests
    3. Analyzing and visualizing results
    4. Comparing multiple backtest runs
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                output_dir: Optional[str] = None):
        """
        Initialize the backtest runner.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for storing results
            log_level: Logging level
        """
        
        # Setup output directory
        self.output_dir = output_dir or sys_config.BACKTEST_RESTULTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Output directory: {self.output_dir}")
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            self.load_config(config_path)
        
        # Initialize containers for backtests and results
        self.backtests = {}
        self.results = {}
        self.current_backtest = None
        
        # Components
        self.visualizer = BacktestVisualizer(savefig_dir=self.output_dir)
    
    def load_config(self, config_path: str):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            self.config = {}
    
    def save_config(self, config_path: Optional[str] = None):
        """
        Save current configuration to a JSON file.
        
        Args:
            config_path: Path to save configuration (default: config.json in output_dir)
        """
        if not config_path:
            config_path = os.path.join(self.output_dir, 'config.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
    
    def _load_class_from_path(self, class_path: str) -> type:
        """
        Dynamically load a class from its import path.
        
        Args:
            class_path: Import path in format 'module.submodule.ClassName'
            
        Returns:
            Class type
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            logging.error(f"Error loading class {class_path}: {str(e)}")
            raise
    
    def load_strategy(self, strategy_config: Dict[str, Any]):
        """
        Load a strategy from configuration.
        
        Args:
            strategy_config: Strategy configuration dictionary
                - class_path: Import path to strategy class
                - parameters: Strategy parameters
                
        Returns:
            Strategy instance
        """
        class_path = strategy_config.get('class_path')
        parameters = strategy_config.get('parameters', {})
        
        if not class_path:
            raise ValueError("Strategy configuration must include 'class_path'")
        
        try:
            strategy_class = self._load_class_from_path(class_path)
            symbols = parameters.pop('symbols', ['SPY'])
            strategy = strategy_class(symbols=symbols, params=parameters)
            logging.info(f"Loaded strategy: {strategy_class.__name__}")
            return strategy
        except Exception as e:
            logging.error(f"Error creating strategy: {str(e)}")
            raise
    
    def load_market_data(self, data_config: Dict[str, Any]):
        """
        Load market data from configuration.
        
        Args:
            data_config: Market data configuration dictionary
                - type: Data type ('csv', 'dataframe', 'pipeline')
                - parameters: Data source parameters
                
        Returns:
            MarketData instance
        """
        data_type = data_config.get('type', 'csv')
        parameters = data_config.get('parameters', {})
        
        try:
            if data_type == 'csv':
                csv_dir = parameters.get('csv_dir')
                symbols = parameters.get('symbols', ['SPY'])
                date_col = parameters.get('date_col', 'Date')
                
                if not csv_dir:
                    raise ValueError("CSV data configuration must include 'csv_dir'")
                
                return CSVMarketData(
                    symbols=symbols,
                    csv_dir=csv_dir,
                    date_col=date_col
                )
                
            elif data_type == 'dataframe':
                # For DataFrameMarketData, we expect dataframes to be provided separately
                raise NotImplementedError("DataFrameMarketData must be created programmatically")
                
            elif data_type == 'pipeline':
                processed_data_path = parameters.get('processed_data_path')
                symbols = parameters.get('symbols')
                date_col = parameters.get('date_col', 'Date')
                
                if not processed_data_path:
                    raise ValueError("Pipeline data configuration must include 'processed_data_path'")
                
                return PipelineMarketData(
                    processed_data_path=processed_data_path,
                    symbols=symbols,
                    date_col=date_col
                )
                
            else:
                raise ValueError(f"Unknown market data type: {data_type}")
                
        except Exception as e:
            logging.error(f"Error loading market data: {str(e)}")
            raise
    
    def create_backtest(self, 
                       backtest_id: str,
                       strategy,
                       market_data,
                       initial_capital: float = 10000.0,
                       risk_manager_config: Optional[Dict[str, Any]] = None,
                       commission_model=None,
                       slippage_model="fixed",
                       slippage_params=None):
        """
        Create a new backtest configuration.
        
        Args:
            backtest_id: Identifier for the backtest
            strategy: Strategy instance or configuration
            market_data: MarketData instance or configuration
            initial_capital: Initial capital
            risk_manager_config: Risk manager configuration
            commission_model: Commission model
            slippage_model: Slippage model type
            slippage_params: Slippage model parameters
            
        Returns:
            Backtest configuration dictionary
        """
        # Load strategy if configuration is provided
        if isinstance(strategy, dict):
            strategy = self.load_strategy(strategy)
        
        # Load market data if configuration is provided
        if isinstance(market_data, dict):
            market_data = self.load_market_data(market_data)
        
        # Create risk manager
        risk_manager = None
        if risk_manager_config:
            position_sizing_method = risk_manager_config.get('position_sizing_method', 'percent')
            position_sizing_params = risk_manager_config.get('position_sizing_params')
            max_position_size = risk_manager_config.get('max_position_size')
            max_portfolio_risk = risk_manager_config.get('max_portfolio_risk', 20.0)
            
            risk_manager = RiskManager(
                position_sizing_method=position_sizing_method,
                position_sizing_params=position_sizing_params,
                max_position_size=max_position_size,
                max_portfolio_risk=max_portfolio_risk
            )
        
        # Create backtest configuration
        backtest_config = {
            'id': backtest_id,
            'strategy': strategy,
            'market_data': market_data,
            'initial_capital': initial_capital,
            'risk_manager': risk_manager,
            'commission_model': commission_model,
            'slippage_model': slippage_model,
            'slippage_params': slippage_params,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store the backtest
        self.backtests[backtest_id] = backtest_config
        self.current_backtest = backtest_id
        
        logging.info(f"Created backtest: {backtest_id}")
        return backtest_config
    
    def run_backtest(self, 
                   backtest_id: Optional[str] = None,
                   save_results: bool = True,
                   generate_reports: bool = True):
        """
        Run a backtest and analyze results.
        
        Args:
            backtest_id: Identifier for the backtest (default: current_backtest)
            save_results: Whether to save results to disk
            generate_reports: Whether to generate performance reports
            
        Returns:
            Dictionary of backtest results
        """
        # Use current backtest if not specified
        if backtest_id is None:
            backtest_id = self.current_backtest
        
        if backtest_id not in self.backtests:
            raise ValueError(f"Backtest not found: {backtest_id}")
        
        backtest_config = self.backtests[backtest_id]
        
        # Extract components
        strategy = backtest_config['strategy']
        market_data = backtest_config['market_data']
        initial_capital = backtest_config['initial_capital']
        risk_manager = backtest_config.get('risk_manager')
        commission_model = backtest_config.get('commission_model')
        slippage_model = backtest_config.get('slippage_model', 'fixed')
        slippage_params = backtest_config.get('slippage_params')
        
        # Create backtest engine
        engine = EventDrivenBacktester(
            initial_capital=initial_capital,
            commission_model=commission_model,
            slippage_model=slippage_model,
            slippage_params=slippage_params
        )
        
        # Run the backtest
        logging.info(f"Running backtest: {backtest_id}")
        start_time = time.time()
        
        results = engine.run(
            strategy=strategy,
            market_data=market_data
        )
        
        execution_time = time.time() - start_time
        logging.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        # Add execution time to results
        results['execution_time'] = execution_time
        
        # Store the results
        self.results[backtest_id] = results
        
        # Generate reports if requested
        if generate_reports:
            self._generate_performance_reports(backtest_id)
        
        # Save results if requested
        if save_results:
            self._save_results(backtest_id)
        
        return results
    
    def _generate_performance_reports(self, backtest_id: str):
        """
        Generate performance reports for a backtest.
        
        Args:
            backtest_id: Backtest identifier
        """
        if backtest_id not in self.results:
            logging.warning(f"No results found for backtest: {backtest_id}")
            return
        
        results = self.results[backtest_id]
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, backtest_id, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Set visualizer to save to report directory
        self.visualizer.savefig_dir = report_dir
        
        # Generate equity curve plot
        if 'portfolio_history' in results:
            self.visualizer.plot_equity_curve(
                equity_data=results['portfolio_history'],
                title=f'Equity Curve - {backtest_id}',
                save_filename='equity_curve.png'
            )
        
        # Generate drawdown plot
        if 'portfolio_history' in results:
            self.visualizer.plot_drawdowns(
                equity_data=results['portfolio_history'],
                save_filename='drawdowns.png'
            )
        
        # Generate monthly returns heatmap
        if 'portfolio_history' in results:
            self.visualizer.plot_monthly_returns(
                equity_data=results['portfolio_history'],
                save_filename='monthly_returns.png'
            )
        
        # Generate return distribution
        if 'portfolio_history' in results:
            self.visualizer.plot_return_distribution(
                equity_data=results['portfolio_history'],
                save_filename='return_distribution.png'
            )
        
        # Generate trade analysis
        if 'trade_history' in results:
            self.visualizer.plot_trade_analysis(
                trades_data=results['trade_history'],
                save_filename='trade_analysis.png'
            )
        
        # Generate performance dashboard
        if 'portfolio_history' in results:
            self.visualizer.create_performance_dashboard(
                equity_data=results['portfolio_history'],
                trades_data=results.get('trade_history'),
                save_filename='performance_dashboard.png'
            )
        
        logging.info(f"Generated performance reports for {backtest_id}")
    
    def _save_results(self, backtest_id: str):
        """
        Save backtest results to disk.
        
        Args:
            backtest_id: Backtest identifier
        """
        if backtest_id not in self.results:
            logging.warning(f"No results found for backtest: {backtest_id}")
            return
        
        results = self.results[backtest_id]
        
        # Create results directory
        results_dir = os.path.join(self.output_dir, backtest_id, 'data')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save portfolio history
        if 'portfolio_history' in results:
            portfolio_history = results['portfolio_history']
            if isinstance(portfolio_history, pd.DataFrame):
                portfolio_history.to_csv(os.path.join(results_dir, 'portfolio_history.csv'), index=False)
        
        # Save trade history
        if 'trade_history' in results:
            trade_history = results['trade_history']
            if isinstance(trade_history, list) and trade_history:
                pd.DataFrame(trade_history).to_csv(os.path.join(results_dir, 'trade_history.csv'), index=False)
        
        # Save performance metrics
        if 'performance_metrics' in results:
            performance_metrics = results['performance_metrics']
            with open(os.path.join(results_dir, 'performance_metrics.json'), 'w') as f:
                json.dump(performance_metrics, f, indent=4)
        
        # Save strategy parameters
        backtest_config = self.backtests[backtest_id]
        strategy = backtest_config['strategy']
        
        if hasattr(strategy, 'get_parameters'):
            strategy_params = strategy.get_parameters()
            with open(os.path.join(results_dir, 'strategy_params.json'), 'w') as f:
                json.dump(strategy_params, f, indent=4)
        
        # Save summary results
        summary = {
            'backtest_id': backtest_id,
            'strategy': strategy.__class__.__name__,
            'initial_capital': backtest_config['initial_capital'],
            'final_equity': results.get('final_equity'),
            'return': results.get('return'),
            'total_trades': results.get('total_trades'),
            'execution_time': results.get('execution_time'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logging.info(f"Saved results for {backtest_id}")
    
    def compare_backtests(self, 
                        backtest_ids: List[str],
                        metrics: Optional[List[str]] = None,
                        output_path: Optional[str] = None):
        """
        Compare multiple backtests.
        
        Args:
            backtest_ids: List of backtest identifiers
            metrics: List of metrics to compare (default: standard metrics)
            output_path: Path to save comparison results
            
        Returns:
            DataFrame with comparison results
        """
        if not backtest_ids:
            logging.warning("No backtests to compare")
            return pd.DataFrame()
        
        # Default metrics to compare
        if metrics is None:
            metrics = [
                'return', 'annualized_return_pct', 'max_drawdown_pct', 
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'total_trades', 'win_rate_pct', 'profit_factor'
            ]
        
        # Collect results
        comparison_data = []
        
        for backtest_id in backtest_ids:
            if backtest_id not in self.results:
                logging.warning(f"No results found for backtest: {backtest_id}")
                continue
            
            results = self.results[backtest_id]
            backtest_config = self.backtests[backtest_id]
            
            # Extract metrics
            metrics_dict = {'backtest_id': backtest_id}
            
            # Basic metrics
            metrics_dict['strategy'] = backtest_config['strategy'].__class__.__name__
            metrics_dict['initial_capital'] = backtest_config['initial_capital']
            metrics_dict['final_equity'] = results.get('final_equity')
            metrics_dict['return'] = results.get('return')
            metrics_dict['total_trades'] = results.get('total_trades')
            
            # Performance metrics
            perf_metrics = results.get('performance_metrics', {})
            for metric in metrics:
                if metric in perf_metrics:
                    metrics_dict[metric] = perf_metrics[metric]
            
            comparison_data.append(metrics_dict)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison if output path provided
        if output_path and not comparison_df.empty:
            comparison_df.to_csv(output_path, index=False)
            logging.info(f"Saved backtest comparison to {output_path}")
        
        return comparison_df
    
    def plot_equity_comparison(self, 
                             backtest_ids: List[str],
                             title: str = 'Equity Curve Comparison',
                             figsize: tuple = (12, 8),
                             save_path: Optional[str] = None):
        """
        Plot equity curves for multiple backtests.
        
        Args:
            backtest_ids: List of backtest identifiers
            title: Plot title
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not backtest_ids:
            logging.warning("No backtests to compare")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each equity curve
        for backtest_id in backtest_ids:
            if backtest_id not in self.results:
                logging.warning(f"No results found for backtest: {backtest_id}")
                continue
            
            results = self.results[backtest_id]
            backtest_config = self.backtests[backtest_id]
            
            if 'portfolio_history' in results:
                equity_data = results['portfolio_history']
                if isinstance(equity_data, pd.DataFrame) and 'equity' in equity_data.columns:
                    # Plot equity curve
                    ax.plot(
                        equity_data['timestamp'],
                        equity_data['equity'],
                        linewidth=2,
                        label=f"{backtest_id} ({backtest_config['strategy'].__class__.__name__})"
                    )
        
        # Format the plot
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add dollar signs to y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved equity comparison to {save_path}")
        
        return fig
    
    def get_backtest_summary(self, backtest_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of backtest results.
        
        Args:
            backtest_id: Backtest identifier (default: current_backtest)
            
        Returns:
            Dictionary with backtest summary
        """
        # Use current backtest if not specified
        if backtest_id is None:
            backtest_id = self.current_backtest
        
        if backtest_id not in self.results:
            logging.warning(f"No results found for backtest: {backtest_id}")
            return {}
        
        results = self.results[backtest_id]
        backtest_config = self.backtests[backtest_id]
        
        # Create summary
        summary = {
            'backtest_id': backtest_id,
            'strategy': backtest_config['strategy'].__class__.__name__,
            'initial_capital': backtest_config['initial_capital'],
            'final_equity': results.get('final_equity'),
            'return': results.get('return'),
            'total_trades': results.get('total_trades'),
            'execution_time': results.get('execution_time'),
        }
        
        # Add performance metrics
        perf_metrics = results.get('performance_metrics', {})
        for key, value in perf_metrics.items():
            summary[key] = value
        
        return summary
    
    def display_summary_table(self, backtest_id: Optional[str] = None):
        """
        Display a formatted summary table for a backtest.
        
        Args:
            backtest_id: Backtest identifier (default: current_backtest)
        """
        summary = self.get_backtest_summary(backtest_id)
        
        if not summary:
            return
        
        # Print header
        print("\n" + "="*50)
        print(f"Backtest Summary: {summary['backtest_id']}")
        print("="*50)
        
        # Print strategy info
        print(f"Strategy: {summary['strategy']}")
        print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"Final Equity: ${summary['final_equity']:,.2f}")
        print(f"Return: {summary['return']:.2f}%")
        print(f"Total Trades: {summary['total_trades']}")
        
        # Print performance metrics if available
        print("\nPerformance Metrics:")
        print("-"*50)
        
        metrics_to_show = [
            ('annualized_return_pct', 'Annualized Return', '%'),
            ('max_drawdown_pct', 'Maximum Drawdown', '%'),
            ('sharpe_ratio', 'Sharpe Ratio', ''),
            ('sortino_ratio', 'Sortino Ratio', ''),
            ('calmar_ratio', 'Calmar Ratio', ''),
            ('win_rate_pct', 'Win Rate', '%'),
            ('profit_factor', 'Profit Factor', ''),
            ('avg_trade', 'Average Trade', '$'),
            ('avg_win', 'Average Win', '$'),
            ('avg_loss', 'Average Loss', '$'),
            ('max_win_streak', 'Max Win Streak', ''),
            ('max_loss_streak', 'Max Loss Streak', '')
        ]
        
        for key, label, unit in metrics_to_show:
            if key in summary:
                value = summary[key]
                if unit == '$':
                    print(f"{label}: ${value:,.2f}")
                elif unit == '%':
                    print(f"{label}: {value:.2f}%")
                else:
                    print(f"{label}: {value:.2f}")
        
        print("="*50)
        
    def load_results(self, results_dir: str) -> Dict[str, Any]:
        """
        Load saved backtest results from disk.
        
        Args:
            results_dir: Path to results directory
            
        Returns:
            Dictionary of loaded results
        """
        try:
            # Load summary
            summary_path = os.path.join(results_dir, 'data', 'summary.json')
            if not os.path.exists(summary_path):
                logging.warning(f"Summary file not found: {summary_path}")
                return {}
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            backtest_id = summary['backtest_id']
            
            # Load portfolio history
            portfolio_history_path = os.path.join(results_dir, 'data', 'portfolio_history.csv')
            if os.path.exists(portfolio_history_path):
                portfolio_history = pd.read_csv(portfolio_history_path)
                if 'timestamp' in portfolio_history.columns:
                    portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
            else:
                portfolio_history = None
            
            # Load trade history
            trade_history_path = os.path.join(results_dir, 'data', 'trade_history.csv')
            if os.path.exists(trade_history_path):
                trade_history = pd.read_csv(trade_history_path)
            else:
                trade_history = []
            
            # Load performance metrics
            metrics_path = os.path.join(results_dir, 'data', 'performance_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    performance_metrics = json.load(f)
            else:
                performance_metrics = {}
            
            # Create results
            results = {
                'backtest_id': backtest_id,
                'portfolio_history': portfolio_history,
                'trade_history': trade_history,
                'performance_metrics': performance_metrics,
                'final_equity': summary.get('final_equity'),
                'return': summary.get('return'),
                'total_trades': summary.get('total_trades'),
                'execution_time': summary.get('execution_time')
            }
            
            # Store the results
            self.results[backtest_id] = results
            
            logging.info(f"Loaded results for {backtest_id}")
            return results
            
        except Exception as e:
            logging.error(f"Error loading results: {str(e)}")
            return {}