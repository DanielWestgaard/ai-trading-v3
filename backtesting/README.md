```
backtesting/
├── __init__.py
├── config.py
├── engine/
│   ├── __init__.py
│   ├── backtester.py             # The central component that orchestrates the backtesting process
│   ├── portfolio.py              # Manages positions and tracks portfolio performance
│   └── performance.py
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py          # Defines the base interface for all trading strategies
│   ├── moving_average.py         # Example strategy implementation
│   └── mean_reversion.py
├── risk/
│   ├── __init__.py
│   ├── position_sizer.py         # Handles the sizing of positions based on risk parameters
│   └── risk_manager.py           # Handles overall risk management and trade filtering
├── data_adapters/  # New module for integration with existing data pipeline
│   ├── __init__.py
│   ├── capital_adapter.py
│   └── pipeline_adapter.py
└── examples/
    ├── simple_ma_backtest.py
    └── portfolio_backtest.py
```

# Understanding Backtesting: A Complete Guide

## What is Backtesting?

Backtesting is the process of testing a trading strategy using historical data to see how it would have performed in the past. It's a critical step before deploying any trading strategy with real money.

## Why Backtest?

- **Validate Strategy Performance**: Determine if a strategy actually works before risking real capital
- **Optimize Parameters**: Find the best parameters for your strategy (like moving average periods)
- **Understand Risk**: Measure metrics like maximum drawdown and volatility
- **Identify Weaknesses**: Discover market conditions where your strategy struggles

## Our Backtesting Framework Architecture

Our framework follows an event-driven architecture, which is industry standard for backtesting systems:

<div style="text-align: center">
<pre>
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Market Data  │ ──> │   Strategy   │ ──> │ Risk Manager │ ──> │   Portfolio  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       └────────────────────┴────────────────────┴──────────────────>│
                                                                      │
                                                                      ▼
                                                            ┌──────────────────┐
                                                            │   Performance    │
                                                            │    Analysis      │
                                                            └──────────────────┘
</pre>
</div>

### Key Components

1. **Market Data Module** (`market_data.py`)
   - Loads historical price data from various sources (CSV, processed data, APIs)
   - Creates market events with bar data (OHLCV) at each time step
   - Manages the time iteration through historical data

2. **Event System** (`events.py`)
   - The heart of the framework - everything communicates through events
   - Main event types: MarketEvent, SignalEvent, OrderEvent, and FillEvent
   - Events flow through the system in a specific order

3. **Strategy Module** (`strategies/`)
   - Receives market data and generates trading signals
   - Implements specific trading logic (e.g., moving average crossover)
   - Can be extended with new strategy classes

4. **Risk Management** (`risk_manager.py`)
   - Determines position sizing based on risk rules
   - Sets stop-loss and take-profit levels
   - Controls portfolio-level risk exposure

5. **Portfolio Module** (`portfolio.py`)
   - Tracks positions, cash balance, and overall equity
   - Processes executed trades and updates portfolio state
   - Calculates unrealized and realized P&L

6. **Performance Analysis** (`performance.py`)
   - Calculates performance metrics (returns, Sharpe ratio, drawdowns)
   - Provides data for visualizations and reporting
   - Helps evaluate strategy effectiveness

7. **Visualization** (`visualization.py`)
   - Creates charts and graphs to analyze performance
   - Generates equity curves, drawdown charts, return distributions
   - Helps identify patterns and understand strategy behavior

8. **Backtest Runner** (`backtest_runner.py`)
   - Coordinates the entire backtesting process
   - Manages configurations and runs multiple backtests
   - Handles result storage and comparison

## Event-Driven Flow

The system follows this sequence of events:

1. **Market Event** is generated from historical data
2. Strategy receives Market Event and may generate **Signal Event**
3. Risk Manager processes Signal Event and creates **Order Event**
4. Order gets executed (with simulated slippage and commission) creating **Fill Event**
5. Portfolio processes Fill Event and updates positions and equity
6. Process repeats for each time step in the historical data

## Understanding the Moving Average Crossover Strategy

This is one of the simplest technical trading strategies:

```
If Short-Term Moving Average crosses ABOVE Long-Term Moving Average → BUY
If Short-Term Moving Average crosses BELOW Long-Term Moving Average → SELL
```

For example, with a 20-day and 50-day Moving Average:
- When the 20-day MA goes from below to above the 50-day MA, it's considered a bullish signal
- When the 20-day MA goes from above to below the 50-day MA, it's considered a bearish signal

## Extending the Framework for ML Models

To incorporate a trained machine learning model (like a .keras model), you would:

1. **Create a New Strategy Class**:

```python
class MLModelStrategy(BaseStrategy):
    def __init__(self, symbols, model_path, params=None, logger=None):
        super().__init__(symbols, params, logger)
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
    def generate_signals(self, market_data, portfolio):
        # Extract features for the model
        features = self._prepare_features(market_data)
        
        # Get prediction from model
        prediction = self.model.predict(features)
        
        # Convert prediction to trading signal
        signals = self._convert_prediction_to_signals(prediction, market_data)
        
        return signals
```

2. **Feature Engineering**:
   - Extract the same features your model was trained on
   - Normalize/scale features in the same way as during training
   - Ensure feature order matches what the model expects

3. **Prediction Interpretation**:
   - Convert model output (probability, regression value, etc.) to actual trading signals
   - Apply any threshold or decision rules for signal generation
   - Include risk parameters like confidence levels

## Performance Metrics to Consider

For a comprehensive evaluation, look at:

1. **Return Metrics**:
   - Total Return
   - Annualized Return
   - Risk-Adjusted Return (Sharpe, Sortino ratios)

2. **Risk Metrics**:
   - Maximum Drawdown
   - Volatility
   - Value at Risk (VaR)

3. **Trade Metrics**:
   - Win Rate
   - Profit Factor
   - Average Win vs. Average Loss
   - Holding Period

## Common Backtesting Pitfalls

1. **Overfitting**: Optimizing strategy parameters too much for historical data
2. **Look-Ahead Bias**: Accidentally using future data in your strategy
3. **Survivorship Bias**: Testing only on stocks that exist today
4. **Transaction Costs**: Ignoring fees, spreads, slippage
5. **Data Quality**: Poor quality or insufficient historical data

## Example: Implementing a Keras ML Model Strategy

Here's a more detailed example of how to implement a ML model strategy:





</thinking>

# Understanding the Backtesting Framework: A Complete Guide for Beginners

As someone new to backtesting and ML/AI, I'll break down our backtesting system from the fundamentals up, explaining how it works and how to extend it.

## What is Backtesting?

Backtesting is the process of testing a trading strategy using historical data to see how it would have performed in the past. It lets you evaluate strategies without risking real money.

## Why Is Backtesting Important?

- **Validate Strategy Performance**: Determine if a strategy actually works
- **Optimize Parameters**: Find the best parameters (like moving average periods)
- **Understand Risk**: Measure metrics like maximum drawdown and volatility
- **Identify Weaknesses**: Discover market conditions where your strategy struggles

## Our Backtesting Framework Architecture

Our framework uses an event-driven architecture, which is the industry standard:

```
Market Data → Strategy → Risk Manager → Portfolio → Performance Analysis
```

### Core Components and Their Functions

1. **Event System** (`events.py`)
   - The heart of the framework where all components communicate through events
   - Main event types: MarketEvent, SignalEvent, OrderEvent, and FillEvent
   - Each event triggers specific actions in other components

2. **Market Data Module** (`market_data.py`)
   - Loads historical price data from your processed files
   - Creates market events at each time step
   - Manages time iteration through historical data

3. **Strategy Module** (`strategies/`)
   - Contains strategies like SimpleMovingAverageCrossover
   - Receives market data and generates trading signals
   - This is where your trading logic lives

4. **Risk Management** (`risk_manager.py`)
   - Determines position sizing (how much to trade)
   - Sets stop-loss and take-profit levels
   - Controls overall portfolio risk

5. **Portfolio Module** (`portfolio.py`)
   - Tracks positions, cash balance, and equity
   - Processes trades and updates portfolio state
   - Calculates P&L (profit and loss)

6. **Performance Analysis** (`performance.py`)
   - Calculates metrics like Sharpe ratio and drawdowns
   - Evaluates strategy effectiveness
   - Provides data for reporting

7. **Visualization** (`visualization.py`)
   - Creates charts for analyzing performance
   - Generates equity curves and drawdown charts
   - Helps visualize strategy behavior

8. **Backtest Runner** (`backtest_runner.py`)
   - Coordinates the entire backtesting process
   - Manages configurations and runs tests
   - Handles result storage

## The Moving Average Crossover Strategy Explained

The strategy you're currently testing is one of the most fundamental technical strategies:

```
When Short-Term MA (20-day) crosses ABOVE Long-Term MA (50-day) → BUY
When Short-Term MA (20-day) crosses BELOW Long-Term MA (50-day) → SELL
```

This strategy attempts to identify trend changes:
- When the faster MA crosses above the slower MA, it suggests prices are trending upward
- When the faster MA crosses below the slower MA, it suggests prices are trending downward

It's popular because it's simple and effective in trending markets, though it can generate false signals in sideways markets.

## How to Backtest a Keras ML Model

To integrate a trained Keras model with our backtesting framework:

1. **Create a ML Strategy Class**:
   I've created a template for you in `ml_model_strategy.py` that:
   - Loads a Keras model
   - Prepares features in the correct format
   - Converts model predictions into trading signals

2. **Configure Your ML Backtest**:
   Use the `ml_model_backtest_config.json` configuration I've provided, which includes:
   - Path to your model file
   - Feature columns your model expects
   - Settings for prediction interpretation
   - Risk management parameters

3. **Run the Backtest**:
   ```bash
   python run_backtest.py --config config/ml_model_backtest_config.json
   ```

## Key Concepts to Understand

### 1. Event-Driven Architecture
In our system, everything happens through events passing between components:
- **Market Event**: New price data (like a candlestick)
- **Signal Event**: Strategy's decision to buy or sell
- **Order Event**: Details of the trade to execute
- **Fill Event**: Confirmation of an executed trade

### 2. Risk Management
Risk management is critical in trading:
- **Position Sizing**: How much capital to allocate to each trade
- **Stop-Loss**: Automatic exit points to limit losses
- **Portfolio Risk**: Managing overall exposure

### 3. Performance Metrics
Evaluate your strategy with these key metrics:
- **Return**: Total and annualized return
- **Risk-Adjusted Return**: Sharpe ratio, Sortino ratio
- **Drawdown**: Maximum drop from peak to trough
- **Trade Statistics**: Win rate, profit factor, etc.

## Common Backtesting Pitfalls

1. **Overfitting**: Creating a strategy that works perfectly on historical data but fails on new data
2. **Look-Ahead Bias**: Accidentally using future data in your strategy
3. **Survivorship Bias**: Testing only on data that exists today
4. **Ignoring Costs**: Unrealistic assumptions about fees and slippage
5. **Insufficient Testing**: Not testing across different market conditions

## Extending the Framework

You can extend this framework in many ways:

1. **New Strategies**: 
   - Create technical indicator strategies
   - Implement pattern recognition
   - Add sentiment analysis

2. **Enhanced Risk Management**:
   - Dynamic position sizing
   - Adaptive stop-loss levels
   - Portfolio optimization

3. **ML/AI Integration**:
   - Classification models (up/down prediction)
   - Regression models (price target prediction)
   - Reinforcement learning agents

## ML Model Integration

For your specific interest in integrating a .keras model, here's the detailed process:

1. **Train your model separately** using your feature engineering pipeline
2. **Save the model** in Keras format
3. **Use the ML strategy template** I provided
4. **Configure the features** to match your training data
5. **Run the backtest** to see how your model would perform in real trading

The key advantage of our framework is that once your model is integrated, you get all the risk management, portfolio tracking, and performance analysis "for free" - you don't need to re-implement these components.

Would you like me to explain any specific component in more detail?