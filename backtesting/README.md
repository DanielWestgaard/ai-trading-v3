```
backtesting/
├── __init__.py
├── config.py
├── engine/
│   ├── __init__.py
│   ├── backtester.py             # The central component that orchestrates the backtesting process
│   ├── portfolio.py              # Manages positions and tracks portfolio performance
│   └── performance.py
├── data/
│   ├── __init__.py
│   ├── data_handler.py           # Provides a standardized interface for working with financial data
│   ├── data_loader.py            # Responsible for loading data from various sources
│   └── data_processor.py
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py          # Defines the base interface for all trading strategies
│   ├── moving_average.py         # Example strategy implementation
│   └── mean_reversion.py
├── risk/
│   ├── __init__.py
│   ├── position_sizer.py         # Handles the sizing of positions based on risk parameters
│   └── risk_manager.py           # Handles overall risk management and trade filtering
└── examples/
    ├── simple_ma_backtest.py
    └── portfolio_backtest.py
```