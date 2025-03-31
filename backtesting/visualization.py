import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime
import os
import logging


class BacktestVisualizer:
    """Visualizes backtest results and performance metrics."""
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 style: str = 'seaborn-v0_8-darkgrid',
                 palette: str = 'viridis',
                 savefig_dir: Optional[str] = None,
                 dpi: int = 300,
                 logger=None):
        """
        Initialize the backtest visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
            palette: Seaborn color palette
            savefig_dir: Directory to save figures
            dpi: DPI for saved figures
            logger: Custom logger
        """
        self.figsize = figsize
        self.style = style
        self.palette = palette
        self.savefig_dir = savefig_dir
        self.dpi = dpi
        self.logger = logger or self._setup_logger()
        
        # Set the plotting style
        plt.style.use(style)
        sns.set_palette(palette)
        
        # Create save directory if specified
        if self.savefig_dir and not os.path.exists(self.savefig_dir):
            os.makedirs(self.savefig_dir)
            self.logger.info(f"Created directory for figures: {self.savefig_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.BacktestVisualizer")
        logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def _save_figure(self, fig, filename: str):
        """Save figure if savefig_dir is specified."""
        if self.savefig_dir:
            filepath = os.path.join(self.savefig_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved figure to {filepath}")
    
    def plot_equity_curve(self, 
                         equity_data,
                         benchmark_data=None,
                         title: str = 'Equity Curve',
                         show_drawdowns: bool = True,
                         figsize=None,
                         save_filename: Optional[str] = 'equity_curve.png'):
        """
        Plot portfolio equity curve.
        
        Args:
            equity_data: DataFrame or list with equity curve data
            benchmark_data: DataFrame or list with benchmark data (optional)
            title: Plot title
            show_drawdowns: Whether to highlight drawdown periods
            figsize: Figure size (defaults to self.figsize)
            save_filename: Filename to save the figure (if savefig_dir is set)
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting equity curve")
        
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Validate data
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning("No valid equity data for plotting")
            return
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Calculate returns and drawdowns if not present
        if 'return' not in equity_df.columns:
            equity_df['return'] = equity_df['equity'].pct_change()
        
        if 'drawdown' not in equity_df.columns:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Plot equity curve
        equity_line = ax.plot(
            equity_df['timestamp'], 
            equity_df['equity'],
            linewidth=2,
            label='Strategy Equity'
        )
        
        # Plot benchmark if provided
        if benchmark_data is not None:
            if isinstance(benchmark_data, list):
                benchmark_df = pd.DataFrame(benchmark_data)
            else:
                benchmark_df = benchmark_data
            
            # Validate benchmark data
            if len(benchmark_df) > 0 and all(col in benchmark_df.columns for col in ['timestamp', 'value']):
                # Ensure timestamp is datetime
                benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
                
                # Normalize benchmark to match starting equity
                start_equity = equity_df['equity'].iloc[0]
                norm_factor = start_equity / benchmark_df['value'].iloc[0]
                benchmark_df['normalized'] = benchmark_df['value'] * norm_factor
                
                ax.plot(
                    benchmark_df['timestamp'],
                    benchmark_df['normalized'],
                    linestyle='--',
                    linewidth=1.5,
                    label='Benchmark'
                )
        
        # Highlight drawdown periods if requested
        if show_drawdowns:
            # Find drawdown periods
            is_drawdown = equity_df['drawdown'] < 0
            drawdown_start = is_drawdown & ~is_drawdown.shift(1).fillna(False)
            drawdown_end = ~is_drawdown & is_drawdown.shift(1).fillna(False)
            
            # Convert to list of (start, end) timestamp tuples
            starts = equity_df[drawdown_start]['timestamp'].tolist()
            ends = equity_df[drawdown_end]['timestamp'].tolist()
            
            # Add current timestamp to ends if we're in a drawdown
            if len(starts) > len(ends):
                ends.append(equity_df['timestamp'].iloc[-1])
            
            # Highlight each drawdown period
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, alpha=0.2, color='red')
        
        # Format axes
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis with dates
        date_format = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add annotation with overall return
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        ax.annotate(
            f'Return: {total_return:.2f}%',
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_filename:
            self._save_figure(fig, save_filename)
        
        return fig
    
    def plot_drawdowns(self,
                      equity_data,
                      top_n: int = 5,
                      figsize=None,
                      show_annotations: bool = True,
                      save_filename: Optional[str] = 'drawdowns.png'):
        """
        Plot drawdowns from the equity curve.
        
        Args:
            equity_data: DataFrame or list with equity curve data
            top_n: Number of largest drawdowns to highlight
            figsize: Figure size (defaults to self.figsize)
            show_annotations: Whether to annotate drawdowns
            save_filename: Filename to save the figure (if savefig_dir is set)
            
        Returns:
            Matplotlib figure
        """
        self.logger.info(f"Plotting top {top_n} drawdowns")
        
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Validate data
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning("No valid equity data for plotting drawdowns")
            return
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Calculate drawdowns if not present
        if 'drawdown' not in equity_df.columns:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Create the figure
        fig, axes = plt.subplots(2, 1, figsize=figsize or self.figsize, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot equity curve in top subplot
        axes[0].plot(
            equity_df['timestamp'],
            equity_df['equity'],
            linewidth=2
        )
        
        # Plot drawdowns in bottom subplot
        drawdown_line = axes[1].fill_between(
            equity_df['timestamp'],
            equity_df['drawdown'] * 100,
            0,
            where=equity_df['drawdown'] < 0,
            color='red',
            alpha=0.4
        )
        
        # Find and highlight top drawdowns
        if top_n > 0:
            # Identify drawdown periods
            equity_df['drawdown_id'] = (equity_df['drawdown'] == 0).cumsum()
            drawdown_periods = []
            
            for dd_id, period in equity_df[equity_df['drawdown'] < 0].groupby('drawdown_id'):
                start_idx = period.index[0]
                end_idx = period.index[-1]
                max_drawdown_idx = period['drawdown'].idxmin()
                
                drawdown_periods.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'max_drawdown_idx': max_drawdown_idx,
                    'start_date': equity_df.loc[start_idx, 'timestamp'],
                    'end_date': equity_df.loc[end_idx, 'timestamp'],
                    'max_drawdown_date': equity_df.loc[max_drawdown_idx, 'timestamp'],
                    'max_drawdown': equity_df.loc[max_drawdown_idx, 'drawdown'],
                    'duration_days': (equity_df.loc[end_idx, 'timestamp'] - equity_df.loc[start_idx, 'timestamp']).days
                })
            
            # Sort drawdown periods by depth
            drawdown_periods.sort(key=lambda x: x['max_drawdown'])
            top_drawdowns = drawdown_periods[:top_n]
            
            # Highlight top drawdowns
            colors = plt.cm.viridis(np.linspace(0, 1, top_n))
            
            for i, dd in enumerate(top_drawdowns):
                # Highlight period in equity curve
                axes[0].axvspan(
                    dd['start_date'],
                    dd['end_date'],
                    alpha=0.2,
                    color=colors[i]
                )
                
                # Highlight in drawdown plot
                axes[1].axvspan(
                    dd['start_date'],
                    dd['end_date'],
                    alpha=0.3,
                    color=colors[i]
                )
                
                # Add annotation if requested
                if show_annotations:
                    # Annotate the depth and duration
                    axes[1].annotate(
                        f"{dd['max_drawdown']*100:.1f}%\n{dd['duration_days']}d",
                        xy=(dd['max_drawdown_date'], dd['max_drawdown'] * 100),
                        xytext=(0, -30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
        
        # Format axes
        axes[0].set_title('Equity Curve with Top Drawdowns', fontsize=16)
        axes[0].set_ylabel('Equity', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Format y-axis as currency
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format drawdown axis
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Format x-axis with dates
        date_format = mdates.DateFormatter('%Y-%m-%d')
        for ax in axes:
            ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_filename:
            self._save_figure(fig, save_filename)
        
        return fig
    
    def plot_monthly_returns(self,
                           equity_data,
                           figsize=None,
                           cmap: str = 'RdYlGn',
                           save_filename: Optional[str] = 'monthly_returns.png'):
        """
        Plot monthly returns as a heatmap.
        
        Args:
            equity_data: DataFrame or list with equity curve data
            figsize: Figure size (defaults to self.figsize)
            cmap: Colormap for the heatmap
            save_filename: Filename to save the figure (if savefig_dir is set)
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting monthly returns heatmap")
        
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Validate data
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning("No valid equity data for plotting monthly returns")
            return
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Extract dates
        equity_df['year'] = equity_df['timestamp'].dt.year
        equity_df['month'] = equity_df['timestamp'].dt.month
        
        # Calculate monthly returns
        monthly_equity = equity_df.groupby(['year', 'month'])['equity'].last().reset_index()
        monthly_equity['prev_equity'] = monthly_equity['equity'].shift(1)
        monthly_equity['return'] = monthly_equity['equity'] / monthly_equity['prev_equity'] - 1
        
        # Create a pivot table for the heatmap
        heatmap_data = monthly_equity.pivot_table(
            index='year',
            columns='month',
            values='return'
        )
        
        # Replace month numbers with names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        heatmap_data.columns = [month_names[m] for m in heatmap_data.columns]
        
        # Calculate yearly returns
        yearly_returns = []
        for year in heatmap_data.index:
            year_data = monthly_equity[monthly_equity['year'] == year]
            if len(year_data) > 0:
                start_equity = year_data['equity'].iloc[0] / (1 + year_data['return'].iloc[0])
                end_equity = year_data['equity'].iloc[-1]
                yearly_return = end_equity / start_equity - 1
                yearly_returns.append(yearly_return)
            else:
                yearly_returns.append(np.nan)
        
        # Add yearly returns as a column
        heatmap_data['Year'] = yearly_returns
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Create the heatmap
        sns.heatmap(
            heatmap_data * 100,  # Convert to percentage
            cmap=cmap,
            annot=True,
            fmt='.1f',
            center=0,
            linewidths=1,
            ax=ax,
            cbar_kws={'label': 'Monthly Return (%)'}
        )
        
        # Format axes
        ax.set_title('Monthly Returns (%)', fontsize=16)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_filename:
            self._save_figure(fig, save_filename)
        
        return fig
    
    def plot_return_distribution(self,
                               equity_data,
                               period: str = 'D',
                               kde: bool = True,
                               figsize=None,
                               save_filename: Optional[str] = 'return_distribution.png'):
        """
        Plot the distribution of returns.
        
        Args:
            equity_data: DataFrame or list with equity curve data
            period: Time period for returns ('D' for daily, 'W' for weekly, 'M' for monthly)
            kde: Whether to plot the kernel density estimate
            figsize: Figure size (defaults to self.figsize)
            save_filename: Filename to save the figure (if savefig_dir is set)
            
        Returns:
            Matplotlib figure
        """
        self.logger.info(f"Plotting {period} return distribution")
        
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Validate data
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning("No valid equity data for plotting return distribution")
            return
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Calculate returns for the specified period
        equity_df = equity_df.set_index('timestamp')
        
        if period == 'D':
            returns = equity_df['equity'].pct_change().dropna()
        else:
            # Resample to period and calculate returns
            resampled = equity_df['equity'].resample(period).last()
            returns = resampled.pct_change().dropna()
        
        # Convert to percentage
        returns *= 100
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Plot histogram with KDE
        sns.histplot(
            returns,
            kde=kde,
            bins=50,
            ax=ax
        )
        
        # Add vertical lines for mean and median
        mean_return = returns.mean()
        median_return = returns.median()
        
        ax.axvline(
            mean_return,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Mean: {mean_return:.2f}%'
        )
        
        ax.axvline(
            median_return,
            color='blue',
            linestyle='-.',
            linewidth=1.5,
            label=f'Median: {median_return:.2f}%'
        )
        
        # Add vertical line at zero
        ax.axvline(
            0,
            color='black',
            linewidth=1.0
        )
        
        # Format axes
        period_name = {
            'D': 'Daily',
            'W': 'Weekly',
            'M': 'Monthly'
        }.get(period, period)
        
        ax.set_title(f'{period_name} Return Distribution', fontsize=16)
        ax.set_xlabel(f'{period_name} Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add statistics annotation
        stats_text = (
            f"Mean: {mean_return:.2f}%\n"
            f"Median: {median_return:.2f}%\n"
            f"Std Dev: {returns.std():.2f}%\n"
            f"Min: {returns.min():.2f}%\n"
            f"Max: {returns.max():.2f}%\n"
            f"Positive: {(returns > 0).mean()*100:.1f}%"
        )
        
        ax.annotate(
            stats_text,
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            verticalalignment='top'
        )
        
        # Add legend
        ax.legend()
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_filename:
            self._save_figure(fig, save_filename)
        
        return fig
    
    def plot_trade_analysis(self,
                          trades_data,
                          figsize=None,
                          save_filename: Optional[str] = 'trade_analysis.png'):
        """
        Plot trade analysis charts.
        
        Args:
            trades_data: DataFrame or list with trade data
            figsize: Figure size (defaults to self.figsize)
            save_filename: Filename to save the figure (if savefig_dir is set)
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Plotting trade analysis")
        
        # Convert to DataFrame if needed
        if isinstance(trades_data, list):
            trades_df = pd.DataFrame(trades_data)
        else:
            trades_df = trades_data
        
        # Validate data
        if len(trades_df) == 0:
            self.logger.warning("No trade data for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize or (14, 10))
        
        # 1. PnL Distribution
        if 'pnl' in trades_df.columns:
            sns.histplot(
                trades_df['pnl'],
                kde=True,
                ax=axes[0, 0]
            )
            
            # Add vertical line at zero
            axes[0, 0].axvline(
                0,
                color='black',
                linewidth=1.0
            )
            
            axes[0, 0].set_title('P&L Distribution', fontsize=14)
            axes[0, 0].set_xlabel('P&L', fontsize=12)
        else:
            axes[0, 0].set_title('P&L Distribution (No data)', fontsize=14)
        
        # 2. Cumulative P&L
        if 'pnl' in trades_df.columns and 'exit_time' in trades_df.columns:
            # Sort trades by exit time
            sorted_trades = trades_df.sort_values('exit_time')
            
            # Calculate cumulative P&L
            sorted_trades['cumulative_pnl'] = sorted_trades['pnl'].cumsum()
            
            axes[0, 1].plot(
                range(len(sorted_trades)),
                sorted_trades['cumulative_pnl'],
                linewidth=2
            )
            
            axes[0, 1].set_title('Cumulative P&L', fontsize=14)
            axes[0, 1].set_xlabel('Trade #', fontsize=12)
            axes[0, 1].set_ylabel('Cumulative P&L', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].set_title('Cumulative P&L (No data)', fontsize=14)
        
        # 3. Win/Loss by Symbol
        if 'symbol' in trades_df.columns and 'pnl' in trades_df.columns:
            # Calculate win rate by symbol
            symbol_results = trades_df.groupby('symbol').agg({
                'pnl': ['count', 'mean', 'sum'],
                'symbol': 'first'
            })
            
            symbol_results.columns = ['count', 'avg_pnl', 'total_pnl', 'symbol']
            symbol_results['win_rate'] = trades_df[trades_df['pnl'] > 0].groupby('symbol').size() / symbol_results['count']
            
            # Sort by total P&L
            symbol_results = symbol_results.sort_values('total_pnl', ascending=False)
            
            # Plot
            bar_colors = ['green' if x > 0 else 'red' for x in symbol_results['total_pnl']]
            
            axes[1, 0].bar(
                symbol_results.index,
                symbol_results['total_pnl'],
                color=bar_colors
            )
            
            axes[1, 0].set_title('P&L by Symbol', fontsize=14)
            axes[1, 0].set_xlabel('Symbol', fontsize=12)
            axes[1, 0].set_ylabel('Total P&L', fontsize=12)
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].set_title('P&L by Symbol (No data)', fontsize=14)
        
        # 4. Win Rate by Month
        if 'exit_time' in trades_df.columns and 'pnl' in trades_df.columns:
            # Ensure exit_time is datetime
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Extract year and month
            trades_df['year_month'] = trades_df['exit_time'].dt.strftime('%Y-%m')
            
            # Calculate monthly win rate
            monthly_stats = trades_df.groupby('year_month').agg({
                'pnl': ['count', 'mean', 'sum']
            })
            
            monthly_stats.columns = ['count', 'avg_pnl', 'total_pnl']
            monthly_stats['win_rate'] = trades_df[trades_df['pnl'] > 0].groupby('year_month').size() / monthly_stats['count']
            
            # Sort by year-month
            monthly_stats = monthly_stats.sort_index()
            
            # Plot win rate
            axes[1, 1].bar(
                monthly_stats.index,
                monthly_stats['win_rate'] * 100,
                color='blue',
                alpha=0.7
            )
            
            # Add line for overall win rate
            overall_win_rate = (trades_df['pnl'] > 0).mean() * 100
            axes[1, 1].axhline(
                overall_win_rate,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label=f'Overall: {overall_win_rate:.1f}%'
            )
            
            axes[1, 1].set_title('Win Rate by Month', fontsize=14)
            axes[1, 1].set_xlabel('Month', fontsize=12)
            axes[1, 1].set_ylabel('Win Rate (%)', fontsize=12)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].set_ylim(0, 100)
        else:
            axes[1, 1].set_title('Win Rate by Month (No data)', fontsize=14)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if requested
        if save_filename:
            self._save_figure(fig, save_filename)
        
        return fig
    
    def create_performance_dashboard(self,
                                  equity_data,
                                  trades_data=None,
                                  benchmark_data=None,
                                  figsize=None,
                                  save_filename: Optional[str] = 'performance_dashboard.png'):
        """
        Create a comprehensive performance dashboard.
        
        Args:
            equity_data: DataFrame or list with equity curve data
            trades_data: DataFrame or list with trade data (optional)
            benchmark_data: DataFrame or list with benchmark data (optional)
            figsize: Figure size (defaults to larger size)
            save_filename: Filename to save the figure (if savefig_dir is set)
            
        Returns:
            Matplotlib figure
        """
        self.logger.info("Creating performance dashboard")
        
        # Use larger figure size for dashboard
        dashboard_figsize = figsize or (16, 12)
        
        # Create figure with gridspec for flexible layout
        fig = plt.figure(figsize=dashboard_figsize)
        gs = fig.add_gridspec(3, 3)
        
        # Plot equity curve (top row, spans all columns)
        ax_equity = fig.add_subplot(gs[0, :])
        self._plot_equity_in_dashboard(ax_equity, equity_data, benchmark_data)
        
        # Plot drawdown (middle row, spans 2 columns)
        ax_drawdown = fig.add_subplot(gs[1, :2])
        self._plot_drawdown_in_dashboard(ax_drawdown, equity_data)
        
        # Plot return distribution (middle row, right column)
        ax_dist = fig.add_subplot(gs[1, 2])
        self._plot_return_dist_in_dashboard(ax_dist, equity_data)
        
        # Plot monthly returns heatmap (bottom row, spans 2 columns)
        ax_monthly = fig.add_subplot(gs[2, :2])
        self._plot_monthly_returns_in_dashboard(ax_monthly, equity_data)
        
        # Plot trade statistics (bottom row, right column)
        ax_trades = fig.add_subplot(gs[2, 2])
        self._plot_trade_stats_in_dashboard(ax_trades, trades_data)
        
        # Add title to the dashboard
        fig.suptitle('Backtest Performance Dashboard', fontsize=20, y=0.98)
        
        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        
        # Save if requested
        if save_filename:
            self._save_figure(fig, save_filename)
        
        return fig
    
    def _plot_equity_in_dashboard(self, ax, equity_data, benchmark_data=None):
        """Helper to plot equity curve in the dashboard."""
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Check data validity
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            ax.set_title('Equity Curve (No data)', fontsize=14)
            return
        
        # Ensure timestamp column
        if 'timestamp' not in equity_df.columns:
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Plot equity curve
        ax.plot(
            equity_df['timestamp'],
            equity_df['equity'],
            linewidth=2,
            label='Strategy'
        )
        
        # Plot benchmark if provided
        if benchmark_data is not None:
            if isinstance(benchmark_data, list):
                benchmark_df = pd.DataFrame(benchmark_data)
            else:
                benchmark_df = benchmark_data
            
            if len(benchmark_df) > 0 and all(col in benchmark_df.columns for col in ['timestamp', 'value']):
                # Normalize benchmark to match starting equity
                start_equity = equity_df['equity'].iloc[0]
                norm_factor = start_equity / benchmark_df['value'].iloc[0]
                benchmark_df['normalized'] = benchmark_df['value'] * norm_factor
                
                ax.plot(
                    benchmark_df['timestamp'],
                    benchmark_df['normalized'],
                    linestyle='--',
                    linewidth=1.5,
                    label='Benchmark'
                )
        
        # Format the plot
        ax.set_title('Equity Curve', fontsize=14)
        ax.set_ylabel('Equity', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Calculate and display key metrics
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        # Add annotation with return
        ax.annotate(
            f'Return: {total_return:.2f}%',
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    def _plot_drawdown_in_dashboard(self, ax, equity_data):
        """Helper to plot drawdowns in the dashboard."""
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Check data validity
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            ax.set_title('Drawdowns (No data)', fontsize=14)
            return
        
        # Ensure timestamp column
        if 'timestamp' not in equity_df.columns:
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Calculate drawdowns if not present
        if 'drawdown' not in equity_df.columns:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Plot drawdowns
        ax.fill_between(
            equity_df['timestamp'],
            equity_df['drawdown'] * 100,
            0,
            where=equity_df['drawdown'] < 0,
            color='red',
            alpha=0.4
        )
        
        # Format the plot
        ax.set_title('Drawdowns', fontsize=14)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add annotation with max drawdown
        max_drawdown = equity_df['drawdown'].min() * 100
        ax.annotate(
            f'Max Drawdown: {max_drawdown:.2f}%',
            xy=(0.02, 0.05),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    def _plot_return_dist_in_dashboard(self, ax, equity_data):
        """Helper to plot return distribution in the dashboard."""
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Check data validity
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            ax.set_title('Return Distribution (No data)', fontsize=14)
            return
        
        # Calculate returns if not present
        if 'return' not in equity_df.columns:
            equity_df['return'] = equity_df['equity'].pct_change()
        
        # Convert to percentage
        returns = equity_df['return'] * 100
        
        # Plot histogram with KDE
        sns.histplot(
            returns.dropna(),
            kde=True,
            bins=20,
            ax=ax
        )
        
        # Add vertical line for mean
        mean_return = returns.mean()
        ax.axvline(
            mean_return,
            color='red',
            linestyle='--',
            linewidth=1.5
        )
        
        # Add vertical line at zero
        ax.axvline(
            0,
            color='black',
            linewidth=1.0
        )
        
        # Format the plot
        ax.set_title('Daily Return Distribution', fontsize=14)
        ax.set_xlabel('Return (%)', fontsize=12)
        
        # Add stats annotation
        stats_text = (
            f"Mean: {mean_return:.2f}%\n"
            f"Std: {returns.std():.2f}%\n"
            f"Min: {returns.min():.2f}%\n"
            f"Max: {returns.max():.2f}%\n"
            f"Pos: {(returns > 0).mean()*100:.1f}%"
        )
        
        ax.annotate(
            stats_text,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            verticalalignment='top'
        )
    
    def _plot_monthly_returns_in_dashboard(self, ax, equity_data):
        """Helper to plot monthly returns in the dashboard."""
        # Convert to DataFrame if needed
        if isinstance(equity_data, list):
            equity_df = pd.DataFrame(equity_data)
        else:
            equity_df = equity_data
        
        # Check data validity
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            ax.set_title('Monthly Returns (No data)', fontsize=14)
            return
        
        # Ensure timestamp column
        if 'timestamp' not in equity_df.columns:
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Extract year and month
        equity_df['year'] = equity_df['timestamp'].dt.year
        equity_df['month'] = equity_df['timestamp'].dt.month
        
        # Calculate monthly returns
        monthly_equity = equity_df.groupby(['year', 'month'])['equity'].last().reset_index()
        monthly_equity['prev_equity'] = monthly_equity['equity'].shift(1)
        monthly_equity['return'] = monthly_equity['equity'] / monthly_equity['prev_equity'] - 1
        
        # Create a pivot table for the heatmap
        try:
            heatmap_data = monthly_equity.pivot_table(
                index='year',
                columns='month',
                values='return'
            )
            
            # Replace month numbers with names
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            heatmap_data.columns = [month_names[m] for m in heatmap_data.columns]
            
            # Create the heatmap
            sns.heatmap(
                heatmap_data * 100,  # Convert to percentage
                cmap='RdYlGn',
                annot=True,
                fmt='.1f',
                center=0,
                linewidths=1,
                ax=ax,
                cbar_kws={'label': 'Return (%)'}
            )
            
            # Format the plot
            ax.set_title('Monthly Returns (%)', fontsize=14)
        except Exception as e:
            self.logger.warning(f"Error creating monthly returns heatmap: {str(e)}")
            ax.set_title('Monthly Returns (Error in calculation)', fontsize=14)
    
    def _plot_trade_stats_in_dashboard(self, ax, trades_data):
        """Helper to plot trade statistics in the dashboard."""
        if trades_data is None or len(trades_data) == 0:
            ax.set_title('Trade Statistics (No data)', fontsize=14)
            ax.axis('off')
            return
        
        # Convert to DataFrame if needed
        if isinstance(trades_data, list):
            trades_df = pd.DataFrame(trades_data)
        else:
            trades_df = trades_data
        
        # Calculate trade statistics
        if 'pnl' in trades_df.columns:
            total_trades = len(trades_df)
            winning_trades = (trades_df['pnl'] > 0).sum()
            losing_trades = (trades_df['pnl'] < 0).sum()
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Calculate profit factor and expected payoff
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_trade = trades_df['pnl'].mean()
            
            # Average holding period
            avg_holding_period = None
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['holding_period'] = (
                    pd.to_datetime(trades_df['exit_time']) - 
                    pd.to_datetime(trades_df['entry_time'])
                ).dt.total_seconds() / (60 * 60 * 24)  # Convert to days
                
                avg_holding_period = trades_df['holding_period'].mean()
            
            # Create stats table
            ax.axis('off')
            stats_text = (
                f"Trade Statistics\n\n"
                f"Total Trades: {total_trades}\n"
                f"Winning Trades: {winning_trades} ({win_rate*100:.1f}%)\n"
                f"Losing Trades: {losing_trades} ({(1-win_rate)*100:.1f}%)\n"
                f"Profit Factor: {profit_factor:.2f}\n"
                f"Avg Win: ${avg_win:.2f}\n"
                f"Avg Loss: ${avg_loss:.2f}\n"
                f"Avg Trade: ${avg_trade:.2f}\n"
            )
            
            if avg_holding_period is not None:
                stats_text += f"Avg Holding: {avg_holding_period:.1f} days\n"
            
            ax.text(
                0.5, 0.5,
                stats_text,
                ha='center',
                va='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=1", fc="white", ec="gray", alpha=0.8)
            )
        else:
            ax.set_title('Trade Statistics (Incomplete data)', fontsize=14)
            ax.axis('off')