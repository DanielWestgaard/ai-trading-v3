# Machine Learning Trading Strategy Framework

## Overview

This framework provides a comprehensive solution for developing, testing, and implementing machine learning-based trading strategies. It combines modern ML techniques with robust backtesting capabilities to create data-driven trading systems.

The framework follows modular design principles, allowing components to be easily extended or replaced while maintaining interoperability. Its primary use case is algorithmic trading strategy development using predictive models trained on financial time series data.

## System Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│                 │     │                   │     │                     │
│  Data Pipeline  │────►│  Model Training   │────►│  Trading Strategy   │
│                 │     │                   │     │                     │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
        │                        │                          │
        │                        │                          │
        ▼                        ▼                          ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│                 │     │                   │     │                     │
│  Feature        │     │  Model Evaluation │     │  Backtesting Engine │
│  Engineering    │     │  & Selection      │     │                     │
│                 │     │                   │     │                     │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
```

## Key Components

### 1. Models

- **BaseModel** (`base_model.py`): Abstract base class that defines the common interface for all prediction models
- **ModelFactory** (`model_factory.py`): Factory class for creating model instances based on specified type
- **Model implementations**:
  - **XGBoostModel** (`xgboost_model.py`): Implementation using XGBoost algorithm
  - **RandomForestModel** (`random_forest_model.py`): Implementation using Random Forest algorithm

### 2. Model Training and Evaluation

- **ModelTrainer** (`model_trainer.py`): Class for training and evaluating prediction models
  - Handles data preparation, training, and validation
  - Supports cross-validation for robust performance estimation
  - Provides model evaluation metrics and visualization tools

### 3. Feature Selection

- **ModelFeatureSelector** (`feature_selector.py`): Utility for selecting optimal features using multiple techniques:
  - Importance-based selection
  - Correlation-based selection
  - Recursive feature elimination
  - Sequential feature selection
  - Correlation filtering to reduce multicollinearity

### 4. Trading Strategy

- **ModelBasedStrategy** (`model_based_strategy.py`): Trading strategy implementation based on ML model predictions
  - Generates trading signals based on model confidence levels
  - Tracks prediction history and position state
  - Implements signal generation logic with configurable thresholds

### 5. Backtesting

- **BacktestRunner** (referenced in `run_model_backtest.py`): Engine for running backtests using trained models
  - Handles market data integration
  - Manages portfolio and position sizing
  - Computes performance metrics and generates reports

## Workflow

The typical workflow using this framework is:

1. **Data Preparation**: Load and clean financial time series data
2. **Feature Engineering**: Create features for model training
3. **Feature Selection**: Select optimal features using various methods
4. **Model Training**: Train prediction models with selected features
5. **Model Evaluation**: Evaluate models using cross-validation and performance metrics
6. **Strategy Configuration**: Configure trading strategy based on model outputs
7. **Backtesting**: Run backtests to evaluate strategy performance
8. **Analysis**: Analyze results and refine the approach

The `run_model_backtest.py` script demonstrates this complete workflow.

## Machine Learning Approaches

The framework supports multiple ML approaches:

### Classification Models

- **Use case**: Predict market direction (up/down)
- **Target variable**: Binary classification (1 for upward movement, 0 for downward)
- **Output**: Probability of upward movement
- **Signal generation**: Based on probability threshold and confidence level

### Regression Models

- **Use case**: Predict price change magnitude
- **Target variable**: Continuous value (e.g., return percentage)
- **Output**: Predicted price change
- **Signal generation**: Based on the sign and magnitude of prediction

### Supported Algorithms

1. **XGBoost**
   - Gradient boosting implementation known for performance and accuracy
   - Handles complex non-linear relationships
   - Built-in feature importance calculation

2. **Random Forest**
   - Ensemble learning method using multiple decision trees
   - Resistant to overfitting
   - Provides feature importance metrics

## Feature Selection Techniques

The framework includes multiple feature selection methods:

1. **Importance-based Selection**
   - Uses model's feature importance scores
   - Selects features above a threshold or top N features

2. **Correlation-based Selection**
   - Selects features with high correlation to the target
   - Removes highly correlated features to reduce redundancy

3. **Recursive Feature Elimination (RFE)**
   - Iteratively removes least important features
   - Uses cross-validation to find optimal feature subset

4. **Sequential Feature Selection**
   - Builds feature set incrementally by adding/removing features
   - Evaluates performance at each step to optimize selection

## Backtesting Methodology

The backtesting engine implements:

1. **Signal Generation**
   - Converts model predictions to actionable trading signals
   - Applies confidence thresholds to filter low-conviction signals

2. **Position Management**
   - Tracks positions and generates entry/exit/reverse signals
   - Maintains prediction and position history

3. **Performance Evaluation**
   - Calculates key metrics (returns, drawdowns, Sharpe ratio, etc.)
   - Compares strategy performance against baselines

4. **Risk Management**
   - Configurable position sizing
   - Implements risk controls based on portfolio metrics

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### Example Usage

```python
# Load and prepare data
data = load_data("path/to/financial_data.csv")

# Configure and train model
model_config = {
    'model_type': 'xgboost',
    'prediction_type': 'classification',
    'target': 'close_return',
    'features': None,  # Auto-select features
    'prediction_horizon': 1,
    'test_size': 0.2,
    'cross_validate': True
}

# Train model
trainer = ModelTrainer(**model_config)
model = trainer.train(data)

# Configure and run backtest
backtest_config = {
    'backtest_id': 'xgboost_strategy_test',
    'market_data': {'symbols': ['SYMBOL']},
    'initial_capital': 10000.0,
    'prediction_threshold': 0.55,
    'confidence_threshold': 0.6
}

# Run backtest
runner = BacktestRunner()
results = runner.run_backtest(model, data, backtest_config)
```

## Advanced Usage

### Custom Feature Selection

```python
# Create feature selector
selector = ModelFeatureSelector(
    selection_method='importance',
    target='close_return',
    n_features=20,
    importance_threshold=0.01
)

# Select features
selected_features = selector.select_features(data, model_type='xgboost')

# Use selected features for model training
model_config['features'] = selected_features
```

### Cross-Validation

```python
# Configure cross-validation
cv_params = {
    'train_period': '3M',  # 3 months training window
    'test_period': '1M',   # 1 month test window
    'step_size': '7D'      # 7 day step between CV folds
}

# Run cross-validation
cv_results = trainer.cross_validate(data, **cv_params)
```

### Performance Comparison

```python
# Compare strategy against baseline
comparison = runner.compare_backtests(
    ['xgboost_strategy', 'baseline_strategy']
)

# Plot equity curves
runner.plot_equity_comparison(
    ['xgboost_strategy', 'baseline_strategy'],
    title="Model vs Baseline"
)
```

## Extending the Framework

### Adding New Models

1. Create a new class inheriting from `BaseModel`
2. Implement required methods: `fit()`, `predict()`, `predict_proba()`
3. Add the model to `ModelFactory`

### Creating Custom Strategies

1. Inherit from `BaseStrategy` or extend `ModelBasedStrategy`
2. Override `generate_signals()` method with custom logic
3. Configure the strategy in backtest configuration

## Best Practices

1. **Feature engineering** is often more important than model selection
2. Use **cross-validation** to avoid overfitting to specific market conditions
3. Implement proper **walk-forward testing** for realistic performance estimation
4. Monitor **feature importance** to understand what drives the model
5. Consider **ensemble approaches** combining multiple models
6. Pay attention to **data leakage** issues in financial time series
7. Implement **realistic transaction costs** in backtests
8. Test strategies across **different market regimes**