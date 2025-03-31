import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging


class PerformanceTracker:
    """Tracks and analyzes performance metrics for backtests."""
    
    def __init__(self, risk_free_rate=0.0, logger=None):
        """
        Initialize the performance tracker.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0%)
            logger: Custom logger
        """
        self.risk_free_rate = risk_free_rate / 100.0  # Convert percentage to decimal
        self.logger = logger or self._setup_logger()
        
        # Cache for calculated metrics
        self._metrics_cache = {}
        self._drawdown_cache = None
        self._period_returns_cache = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.PerformanceTracker")
        logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def calculate_metrics(self, equity_curve: List[Dict[str, Any]], 
                         trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: Portfolio equity history
            trades: Trade history
            
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Calculating performance metrics")
        
        # Reset cache
        self._metrics_cache = {}
        
        # Convert to DataFrame if needed
        if isinstance(equity_curve, list):
            equity_df = pd.DataFrame(equity_curve)
        else:
            equity_df = equity_curve
        
        if len(equity_df) == 0:
            self.logger.warning("Empty equity curve, cannot calculate metrics")
            return {"error": "Empty equity curve"}
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Sort by timestamp
        equity_df = equity_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns
        if 'equity' in equity_df.columns:
            equity_df['return'] = equity_df['equity'].pct_change()
            equity_df['log_return'] = np.log(equity_df['equity'] / equity_df['equity'].shift(1))
            
            # Cumulative returns
            initial_equity = equity_df['equity'].iloc[0]
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity / initial_equity) - 1
            
            # Calculate daily, monthly, and annualized returns
            trading_days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
            trading_days = max(1, trading_days)  # Avoid division by zero
            
            annualized_return = (1 + total_return) ** (365 / trading_days) - 1
            
            # Volatility metrics
            daily_volatility = equity_df['return'].std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Sharpe and Sortino ratios
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility != 0 else 0
            
            # Calculate Sortino ratio (downside deviation)
            downside_returns = equity_df[equity_df['return'] < 0]['return']
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
            
            # Calculate drawdowns
            drawdowns = self.calculate_drawdowns(equity_df)
            max_drawdown = drawdowns['max_drawdown']
            max_drawdown_duration = drawdowns['max_duration_days']
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Store basic metrics
            metrics = {
                "initial_equity": initial_equity,
                "final_equity": final_equity,
                "total_return_pct": total_return * 100,
                "annualized_return_pct": annualized_return * 100,
                "trading_days": trading_days,
                "trading_months": trading_days / 30,  # Approximate
                "annualized_volatility_pct": annualized_volatility * 100,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown_pct": max_drawdown * 100,
                "max_drawdown_duration": max_drawdown_duration,
                "calmar_ratio": calmar_ratio,
                "risk_free_rate_pct": self.risk_free_rate * 100
            }
            
            # Add trade metrics if available
            if trades:
                trade_metrics = self._calculate_trade_metrics(trades)
                metrics.update(trade_metrics)
            
            # Cache metrics
            self._metrics_cache = metrics
            
            return metrics
        else:
            self.logger.error("No 'equity' column in equity curve")
            return {"error": "No equity data found"}
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trade-specific metrics.
        
        Args:
            trades: Trade history
            
        Returns:
            Dictionary of trade metrics
        """
        if not trades:
            return {"trades": 0}
        
        # Convert to DataFrame
        if isinstance(trades, list):
            trades_df = pd.DataFrame(trades)
        else:
            trades_df = trades
        
        # Calculate basic trade statistics
        num_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / num_trades if num_trades > 0 else 0
        
        # Calculate P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        avg_win = winning_trades['pnl'].mean() if num_winning > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if num_losing > 0 else 0
        
        # Profit factor and expected payoff
        gross_profits = winning_trades['pnl'].sum() if num_winning > 0 else 0
        gross_losses = abs(losing_trades['pnl'].sum()) if num_losing > 0 else 0
        
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        expected_payoff = avg_pnl if num_trades > 0 else 0
        
        # Consecutive wins/losses
        if 'exit_time' in trades_df.columns:
            trades_df = trades_df.sort_values('exit_time')
        
        if 'pnl' in trades_df.columns:
            win_loss_streaks = (trades_df['pnl'] > 0).astype(int).diff().ne(0).cumsum()
            streak_counts = win_loss_streaks.value_counts()
            
            max_win_streak = 0
            max_loss_streak = 0
            
            for streak_id, trades_in_streak in trades_df.groupby(win_loss_streaks):
                is_winning = trades_in_streak['pnl'].iloc[0] > 0
                streak_length = len(trades_in_streak)
                
                if is_winning and streak_length > max_win_streak:
                    max_win_streak = streak_length
                elif not is_winning and streak_length > max_loss_streak:
                    max_loss_streak = streak_length
        else:
            max_win_streak = 0
            max_loss_streak = 0
        
        # Average holding period
        avg_holding_period = None
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['holding_period'] = (
                pd.to_datetime(trades_df['exit_time']) - 
                pd.to_datetime(trades_df['entry_time'])
            ).dt.total_seconds() / (60 * 60 * 24)  # Convert to days
            
            avg_holding_period = trades_df['holding_period'].mean()
        
        # Compile trade metrics
        trade_metrics = {
            "total_trades": num_trades,
            "winning_trades": num_winning,
            "losing_trades": num_losing,
            "win_rate_pct": win_rate * 100,
            "total_pnl": total_pnl,
            "average_pnl": avg_pnl,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "expected_payoff": expected_payoff,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak
        }
        
        if avg_holding_period is not None:
            trade_metrics["avg_holding_period_days"] = avg_holding_period
        
        return trade_metrics
    
    def calculate_drawdowns(self, equity_curve=None) -> Dict[str, Any]:
        """
        Calculate drawdown statistics.
        
        Args:
            equity_curve: Equity curve data
            
        Returns:
            Dictionary of drawdown metrics
        """
        # Use cached drawdowns if available and no new equity curve provided
        if self._drawdown_cache is not None and equity_curve is None:
            return self._drawdown_cache
        
        # Convert to DataFrame if needed
        if isinstance(equity_curve, list):
            equity_df = pd.DataFrame(equity_curve)
        else:
            equity_df = equity_curve
        
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning("No valid equity data for drawdown calculation")
            return {
                "max_drawdown": 0.0,
                "max_duration_days": 0,
                "current_drawdown": 0.0
            }
        
        # Calculate drawdowns
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        
        # Find the maximum drawdown
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate drawdown duration
        equity_df['is_drawdown'] = equity_df['drawdown'] < 0
        equity_df['drawdown_id'] = (equity_df['is_drawdown'] != equity_df['is_drawdown'].shift()).cumsum()
        
        # Group by drawdown periods and calculate duration
        drawdown_periods = equity_df[equity_df['is_drawdown']].groupby('drawdown_id')
        
        max_duration = 0
        max_duration_drawdown = 0
        
        for period_id, period_data in drawdown_periods:
            if 'timestamp' in period_data.columns:
                start_date = period_data['timestamp'].iloc[0]
                end_date = period_data['timestamp'].iloc[-1]
                duration_days = (end_date - start_date).days
                
                period_max_drawdown = period_data['drawdown'].min()
                
                if duration_days > max_duration:
                    max_duration = duration_days
                    max_duration_drawdown = period_max_drawdown
        
        # Current drawdown
        current_drawdown = equity_df['drawdown'].iloc[-1]
        
        # Calculate underwater periods
        underwater_df = equity_df[equity_df['drawdown'] < 0]
        underwater_pct = len(underwater_df) / len(equity_df) if len(equity_df) > 0 else 0
        
        # Calculate average drawdown
        avg_drawdown = underwater_df['drawdown'].mean() if len(underwater_df) > 0 else 0
        
        # Prepare results
        results = {
            "max_drawdown": max_drawdown,
            "max_duration_days": max_duration,
            "max_duration_drawdown": max_duration_drawdown,
            "current_drawdown": current_drawdown,
            "underwater_pct": underwater_pct * 100,
            "avg_drawdown": avg_drawdown
        }
        
        # Cache results
        self._drawdown_cache = results
        
        return results
    
    def calculate_period_returns(self, period='M', equity_curve=None) -> pd.DataFrame:
        """
        Calculate returns for specific periods (e.g., monthly).
        
        Args:
            period: Period frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            equity_curve: Equity curve data
            
        Returns:
            DataFrame of period returns
        """
        # Use cached period returns if available
        if period in self._period_returns_cache and equity_curve is None:
            return self._period_returns_cache[period]
        
        # Convert to DataFrame if needed
        if isinstance(equity_curve, list):
            equity_df = pd.DataFrame(equity_curve)
        else:
            equity_df = equity_curve
        
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning(f"No valid equity data for {period} return calculation")
            return pd.DataFrame()
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Resample to the specified period
        period_data = equity_df.set_index('timestamp', drop=False)
        
        # Calculate period returns
        if period == 'D':
            # Daily returns
            period_returns = period_data['equity'].pct_change().dropna()
            return_df = pd.DataFrame({
                'timestamp': period_returns.index,
                'return': period_returns.values
            })
        else:
            # Weekly, monthly, or other periods
            period_start = period_data.resample(period).first()
            period_end = period_data.resample(period).last()
            
            period_returns = pd.DataFrame({
                'start_date': period_start.index,
                'end_date': period_end.index,
                'start_equity': period_start['equity'],
                'end_equity': period_end['equity']
            }).dropna()
            
            period_returns['return'] = (
                period_returns['end_equity'] / period_returns['start_equity'] - 1
            )
            
            # For monthly returns, add year and month columns
            if period == 'M':
                period_returns['year'] = period_returns['start_date'].dt.year
                period_returns['month'] = period_returns['start_date'].dt.month
                period_returns['month_name'] = period_returns['start_date'].dt.strftime('%b')
            
            return_df = period_returns
        
        # Cache results
        self._period_returns_cache[period] = return_df
        
        return return_df
    
    def calculate_rolling_metrics(self, window=30, equity_curve=None) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            window: Rolling window size in days
            equity_curve: Equity curve data
            
        Returns:
            DataFrame of rolling metrics
        """
        # Convert to DataFrame if needed
        if isinstance(equity_curve, list):
            equity_df = pd.DataFrame(equity_curve)
        else:
            equity_df = equity_curve
        
        if len(equity_df) == 0 or 'equity' not in equity_df.columns:
            self.logger.warning("No valid equity data for rolling metrics calculation")
            return pd.DataFrame()
        
        # Make sure we have timestamp column
        if 'timestamp' not in equity_df.columns:
            self.logger.warning("No timestamp column in equity curve, using index")
            equity_df['timestamp'] = pd.date_range(
                start='2000-01-01', periods=len(equity_df), freq='D'
            )
        
        # Ensure timestamp is datetime
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        # Calculate returns if not already present
        if 'return' not in equity_df.columns:
            equity_df['return'] = equity_df['equity'].pct_change()
        
        # Set up rolling window
        rolling_df = equity_df.set_index('timestamp', drop=False)
        
        # Calculate rolling metrics
        rolling_return = rolling_df['equity'].pct_change(window)
        rolling_volatility = rolling_df['return'].rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio (approximate)
        risk_free_daily = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_return = rolling_df['return'] - risk_free_daily
        rolling_sharpe = (
            excess_return.rolling(window).mean() * 252 /
            (rolling_df['return'].rolling(window).std() * np.sqrt(252))
        )
        
# Rolling drawdown
        rolling_df['rolling_peak'] = rolling_df['equity'].rolling(window).max()
        rolling_df['rolling_drawdown'] = (rolling_df['equity'] - rolling_df['rolling_peak']) / rolling_df['rolling_peak']
        
        # Combine rolling metrics
        rolling_metrics = pd.DataFrame({
            'timestamp': rolling_df.index,
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_df['rolling_drawdown']
        })
        
        return rolling_metrics
    
    def calculate_monthly_return_table(self, equity_curve=None) -> pd.DataFrame:
        """
        Calculate monthly returns for display in a table format.
        
        Args:
            equity_curve: Equity curve data
            
        Returns:
            DataFrame of monthly returns in a year x month table format
        """
        # Get monthly returns
        monthly_returns = self.calculate_period_returns('M', equity_curve)
        
        if len(monthly_returns) == 0:
            return pd.DataFrame()
        
        # Pivot to get year x month table
        monthly_table = monthly_returns.pivot_table(
            index='year',
            columns='month_name',
            values='return'
        )
        
        # Calculate yearly returns
        yearly_returns = monthly_returns.groupby('year')['return'].apply(
            lambda x: np.prod(1 + x) - 1
        )
        
        # Add yearly returns column
        monthly_table['Year'] = yearly_returns
        
        # Calculate monthly averages
        monthly_avgs = monthly_returns.groupby('month')['return'].mean()
        
        # Create a new row for monthly averages
        monthly_avgs_row = pd.DataFrame(
            [monthly_avgs.values], 
            columns=[pd.to_datetime(f'2000-{m}-01').strftime('%b') for m in monthly_avgs.index],
            index=['Average']
        )
        
        # Combine with monthly table
        monthly_table = pd.concat([monthly_table, monthly_avgs_row])
        
        return monthly_table * 100  # Convert to percentage