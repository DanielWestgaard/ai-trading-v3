# Core Trading System Components

## Overview

The `core` directory contains the foundational components shared between the backtesting and live trading systems. These components implement the essential trading logic, risk management, position handling, and portfolio tracking that are common to both environments.

## Directory Structure

```
core/
├── events.py                # Event definitions and handling
├── execution/               # Order execution interfaces
│   ├── execution_interface.py
│   ├── live_execution.py
│   └── simulation_execution.py
├── performance/             # Performance analysis tools
│   └── performance.py
├── portfolio/               # Portfolio tracking
│   └── portfolio.py
├── risk/                    # Risk management components
│   ├── position_management.py
│   └── risk_manager.py
├── signal_filter.py         # Signal filtering and consensus
├── strategies/              # Trading strategy implementations
│   ├── base_strategy.py
│   ├── debug_strategy.py
│   ├── indicator_based_strategy.py
│   ├── model_based_strategy.py
│   └── simple_ma_crossover.py
└── timeframe_resampler.py   # Timeframe conversion utilities
```

## Core Components

### 1. Event System (`events.py`)

The event system provides a standardized way for components to communicate with each other:

- **MarketEvent**: New market data (e.g., price bar)
- **SignalEvent**: Strategy generated trading signal (Buy/Sell)
- **OrderEvent**: Order to be sent to broker
- **FillEvent**: Order has been filled

Events flow through the system in a defined sequence, creating a clean separation between components.

### 2. Execution Components

The execution module provides interfaces and implementations for order execution:

- **ExecutionInterface**: Base class defining the execution API
- **LiveExecution**: Connects to real broker APIs for live trading
- **SimulationExecution**: Simulates order execution for backtesting

### 3. Performance Tracking

The `PerformanceTracker` class calculates and monitors trading performance metrics:

- Equity curve and drawdowns
- Returns (daily, monthly, annualized)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Trade statistics (win rate, profit factor, etc.)

### 4. Portfolio Management

The `Portfolio` class tracks positions and account metrics:

- **Position**: Represents a trading position with entry/exit details
- **Portfolio**: Manages multiple positions and calculates:
  - Equity and cash balances
  - Unrealized and realized P&L
  - Position values and exposures

### 5. Risk Management

The risk module handles position sizing and risk controls:

- **PositionManager**: Manages position entry, exit, and holding periods
- **RiskManager**: Controls:
  - Position sizing methods (percent of equity, fixed risk, Kelly criterion)
  - Stop-loss and take-profit calculation
  - Position correlation limits
  - Maximum portfolio risk
  - Risk-adjusted position sizing

### 6. Signal Filtering

The `SignalFilter` reduces false positives in trading signals:

- Tracks prediction history
- Applies consensus thresholds
- Filters based on signal strength and confidence
- Prevents overtrading

### 7. Strategy Implementations

The strategies module contains different trading strategy implementations:

- **BaseStrategy**: Abstract base class defining the strategy interface
- **ModelBasedStrategy**: Uses ML models for prediction and signal generation
- **IndicatorBasedStrategy**: Uses technical indicators for trading decisions
- **SimpleMAcrossover**: Classic moving average crossover strategy
- **DebugStrategy**: For testing and debugging the system

### 8. Timeframe Resampling

The `TimeframeResampler` converts between different timeframes:

- Aggregates lower timeframe bars into higher timeframe bars (e.g., 1-minute → 1-hour)
- Determines when trading decisions should be made based on timeframe boundaries
- Handles OHLCV data properly when resampling

## Data Flow and Interaction

The core components interact in a structured pipeline:

1. **Data Input**:
   - Market data enters the system as MarketEvents

2. **Strategy Processing**:
   - Strategies analyze market data
   - Signal generation (SignalEvents)

3. **Risk Assessment**:
   - Risk Manager evaluates signals
   - Position sizing calculation
   - Stop-loss and take-profit determination

4. **Order Creation**:
   - Orders are created (OrderEvents)
   - Sent to execution handler

5. **Execution**:
   - Orders are executed by the execution handler
   - Fill events (FillEvents) are generated

6. **Portfolio Update**:
   - Portfolio tracks positions and equity
   - Performance is calculated

## Design Principles

The core modules follow several key design principles:

1. **Event-Driven Architecture**: Components communicate via events, reducing coupling
2. **Separation of Concerns**: Each component has a single responsibility
3. **Modularity**: Components can be replaced or extended without affecting others
4. **Strategy Independence**: Strategies are isolated from execution details
5. **Shared Risk Management**: Consistent risk management across systems

## Key Interfaces

### Strategy Interface

```python
def generate_signals(self, market_data, portfolio):
    """Generate trading signals based on market data and portfolio state."""
    pass
```

### Execution Interface

```python
def execute_order(self, order_event):
    """Execute an order and return a fill event."""
    pass
```

### Risk Manager Interface

```python
def process_signal(self, signal, portfolio, market_data=None):
    """Process a signal and create an order based on risk parameters."""
    pass
```

## Implementation Notes

- The system uses **abstract base classes** to define interfaces
- Event objects are immutable to prevent side effects
- **Factory patterns** create appropriate components based on configuration
- Risk management is strictly separated from signal generation
- Performance tracking is passive and doesn't affect trading decisions

## Usage Example

Here's how these components work together in a typical workflow:

```python
# Create a strategy
strategy = ModelBasedStrategy(
    symbols=['GBPUSD'],
    model=xgb_model,
    prediction_threshold=0.55
)

# Create a risk manager
risk_manager = RiskManager(
    position_sizing_method='percent',
    position_sizing_params={'percent': 2.0},
    auto_stop_loss=True
)

# Process a market event
signals = strategy.generate_signals(market_data, portfolio)

# Process signals through risk manager
for signal in signals:
    order = risk_manager.process_signal(signal, portfolio, market_data)
    
    # Execute the order
    fill = execution_handler.execute_order(order)
    
    # Update portfolio
    portfolio.process_fill(fill)

# Track performance
performance = performance_tracker.calculate_metrics(portfolio.get_equity_curve())
```

This modular design allows the same core components to be used effectively in both backtesting and live trading environments.