# ai-trading-v3
This repo will build on the learnings from v2.

# Learning from previous try
I have a quite large project (v2). I came really close to deploying a model on a demo account, but stopped as I struggled to understand my own code. 
1. Lack of proper planning for the system as a whole
2. Lack of planning for organizing (files and folders)
3. Lack of following said structure
4. Relying too much on AI assistant to create this project for me.
    - This led to me not fully understanding the project and code I was writing/copying. So when going back to add or modify code, there would be duplicates and "code-types" spread across multiple files (like code related to positioning spread across three files).
5. Properly importing files and methods
6. Doing "everything at once"
    - My approach to this was wrong. I made everything "as good and as complex" as I could, and when I wanted to test it out on a demo account or modify something, it would be too complex to understand and a real pain to work with.

# Project Structure
```
algotrading/
├── config/                  # Configuration files
│   ├── model_config.yaml    # Model hyperparameters
│   ├── market_config.yaml   # Market-specific settings (forex, indices)
│   └── system_config.yaml   # System-wide settings
│
├── data/                    # Data management
│   ├── loaders/             # Data acquisition modules
│   ├── processors/          # Data cleaning and feature engineering
│   ├── features/            # Feature definitions and generators
│   └── storage/             # Data storage (CSV, parquet files)
│
├── models/                  # Trading models
│   ├── base.py              # Abstract model class
│   ├── gbdt_models/         # Gradient boosting models (XGBoost, LightGBM)
│   ├── nn_models/           # Neural network models (LSTM, GRU)
│   ├── ensemble/            # Model combination strategies
│   └── definitions/         # Model architecture definitions
│
├── strategies/              # Trading strategies
│   ├── base.py              # Strategy interface
│   ├── entry/               # Entry signal strategies
│   ├── exit/                # Exit signal strategies
│   └── position_sizing/     # Position sizing algorithms
│
├── execution/               # Order execution
│   ├── broker_api/          # Broker integration
│   ├── order_types/         # Different order implementations
│   └── execution_algos/     # Smart execution algorithms
│
├── risk/                    # Risk management
│   ├── position_risk.py     # Per-position risk calculations
│   ├── portfolio_risk.py    # Portfolio-level risk management
│   └── limits.py            # Trading limits and circuit breakers
│
├── backtesting/             # Backtesting framework
│   ├── engine.py            # Backtesting engine
│   ├── metrics.py           # Performance metrics
│   └── visualizations.py    # Performance visualizations
│
├── live/                    # Live trading infrastructure
│   ├── runners/             # Trading session managers
│   ├── monitoring/          # System monitoring
│   └── alerts/              # Alert system
│
├── utils/                   # Utility functions
│   ├── logging_utils.py     # Logging utilities
│   ├── time_utils.py        # Time manipulation utilities
│   └── validation.py        # Input validation
│
├── notebooks/               # Jupyter notebooks for research
│   ├── exploratory/         # Data exploration
│   ├── model_development/   # Model tuning and testing
│   └── analysis/            # Performance analysis
│
├── logs/                    # Logging files
│   ├── backtest/            # Backtest logs
│   └── live/                # Live trading logs
│
├── tests/                   # Testing
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── data/                # Test data
│
├── model_storage/           # Trained model storage
│   ├── production/          # Production-ready models
│   │   ├── forex/           # Forex market models
│   │   │   ├── v1/          # Version 1 models
│   │   │   └── v2/          # Version 2 models
│   │   └── indices/         # Indices market models
│   │       ├── v1/          # Version 1 models
│   │       └── v2/          # Version 2 models
│   ├── staging/             # Models under evaluation
│   ├── archive/             # Previous models for reference
│   └── metadata/            # Model performance metadata
│
├── docs/                    # Documentation
│   ├── architecture/        # System architecture docs
│   ├── models/              # Model documentation
│   └── operations/          # Operational guides
│
├── scripts/                 # Utility scripts
│   ├── setup.py             # Environment setup
│   ├── backtest.py          # Backtest runner
│   └── deploy.py            # Deployment script
│
├── .env                     # Environment variables (never commit to git)
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
└── main.py                  # Entry point
```