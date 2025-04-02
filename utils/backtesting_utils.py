
# Metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate the Sharpe ratio of a return series."""
    import numpy as np
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_drawdown(equity_curve):
    """Calculate the drawdown series and maximum drawdown."""
    import numpy as np
    
    # Calculate the running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage terms
    drawdown = (equity_curve - running_max) / running_max
    
    return drawdown, drawdown.min()

def calculate_performance_metrics(portfolio_history):
    """Calculate comprehensive performance metrics from portfolio history."""
    import pandas as pd
    import numpy as np
    
    # Convert to DataFrame if it's a list
    if isinstance(portfolio_history, list):
        history_df = pd.DataFrame(portfolio_history)
        history_df.set_index('timestamp', inplace=True)
    else:
        history_df = portfolio_history
        
    # Calculate daily returns
    daily_returns = history_df['total_value'].pct_change().dropna()
    
    # Calculate various metrics
    total_return = (history_df['total_value'].iloc[-1] / 
                    history_df['total_value'].iloc[0]) - 1
    
    # Calculate drawdown
    drawdown_series, max_drawdown = calculate_drawdown(history_df['total_value'])
    
    # Annual metrics (assuming 252 trading days)
    trading_days = len(daily_returns)
    years = trading_days / 252
    
    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
    }
    

# Plotting

def plot_equity_curve(results, benchmark=None, figsize=(12, 6)):
    """Plot equity curve with optional benchmark comparison."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(figsize=figsize)
    
    if isinstance(results, dict) and 'equity_curve' in results:
        equity_curve = results['equity_curve']
    else:
        equity_curve = results
    
    if isinstance(equity_curve, list):
        equity_curve = pd.DataFrame(equity_curve)
        
    plt.plot(equity_curve['total_value'], label='Strategy')
    
    if benchmark is not None:
        # Normalize benchmark to same starting point
        benchmark_norm = benchmark / benchmark.iloc[0] * equity_curve['total_value'].iloc[0]
        plt.plot(benchmark_norm, label='Benchmark', alpha=0.7)
        
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    
    return plt