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