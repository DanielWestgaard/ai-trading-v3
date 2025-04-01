# Data Processing Pipeline

The data processing pipeline is designed for financial time series data, particularly for OHLCV (Open, High, Low, Close, Volume) market data. It provides a robust framework for cleaning, feature generation, normalization, and feature selection to prepare market data for algorithmic trading models.

## Pipeline Architecture

The pipeline follows a modular design with the following components:

1. **Data Loading**: Supports various data sources including CSV files and broker APIs (Capital.com)
2. **Data Cleaning**: Handles missing values, outliers, and ensures OHLC validity
3. **Feature Generation**: Creates technical indicators, volatility metrics, price patterns, and time features
4. **Feature Preparation**: Transforms raw prices, manages window-based features, and prepares data for modeling
5. **Normalization**: Applies appropriate scaling methods for different data types
6. **Feature Selection**: Uses machine learning techniques to identify the most important features

## Key Components

### DataCleaner
- Handles missing values with methods like forward-fill or interpolation
- Detects and manages outliers using z-score, IQR, or winsorization
- Ensures OHLC relationship validity (High ≥ Open ≥ Close ≥ Low)
- Supports time continuity through resampling

### FeatureGenerator
- Calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Computes volatility metrics (ATR, historical volatility)
- Identifies price patterns (Doji, Hammer, Engulfing)
- Generates time-based features (day of week, market sessions)

### FeaturePreparator
- Transforms raw prices to returns or other derived metrics
- Handles window-based features with different treatment modes
- Preserves original prices for empirical feature selection

### DataNormalizer
- Applies appropriate scaling methods for different data types
- Supports z-score, min-max, robust scaling, and log transformations

### FeatureSelector
- Uses Random Forest feature importance with time series cross-validation
- Produces visualizations of feature importance and category distribution
- Supports multiple selection methods (threshold, top-N, cumulative importance)

### DataPipeline
- Coordinates the entire process from raw data to processed feature set
- Manages file organization with consistent naming conventions
- Provides options for saving intermediate results and visualizations

## Usage Example

```python
from data.pipelines.data_pipeline import DataPipeline

# Initialize pipeline with configuration
pipeline = DataPipeline(
    feature_treatment_mode='advanced',
    price_transform_method='returns',
    normalization_method='zscore',
    feature_selection_method='threshold',
    feature_importance_threshold=0.01,
    target_column='close_return'
)

# Process data
processed_data, output_path = pipeline.run(
    raw_data='path/to/raw_data.csv',
    target_path='output/directory',
    save_intermediate=True,
    run_feature_selection=True
)
```

## Future Improvements and Additions

1. **Streaming Data Support**: Enhance real-time data processing capabilities for live trading
2. **Feature Store**: Implement a centralized feature repository for model training and inference
3. **Adaptive Feature Selection**: Develop methods to dynamically adjust feature importance based on market conditions
4. **GPU Acceleration**: Optimize computational performance for large datasets using GPU processing
5. **AutoML Integration**: Incorporate automated feature engineering and parameter tuning
6. **Market Regime Detection**: Add capabilities to identify different market states and adapt processing accordingly
7. **Data Quality Monitoring**: Implement continuous validation of data quality and feature drift
8. **Alternative Data Sources**: Expand support for sentiment data, economic indicators, and other alternative data
9. **Cross-Asset Features**: Develop features that capture relationships between different markets and asset classes

The pipeline's modular design makes it straightforward to extend with new data sources, feature engineering techniques, and optimization methods as trading strategies evolve.

# Understanding Walk-Forward Testing in Your ML Trading System
## Where This Fits in Your Pipeline
Your current system has this flow:
```
CopyRaw Data → Data Cleaning → Feature Generation → Feature Preparation → Normalization → Feature Selection → Processed Data
```
The walk-forward testing component sits after this pipeline as a bridge between your data processing and your trading strategy:
```
CopyRaw Data → Data Pipeline → Processed Data → Walk-Forward Testing → Trading Strategy → Backtesting
```
It's essentially a framework that ensures your model training and evaluation follow best practices for time series data.

## Working with a Keras Model
Integrating a Keras model is straightforward! Here's what that would look like
```
import tensorflow as tf
from data.features.time_series_ml import WalkForwardAnalysis, MLModelStrategy

# 1. DEFINE YOUR KERAS MODEL FUNCTIONS
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, data, features, target):
    X = data[features].values
    y = data[target].values
    # Normalize inputs if needed
    model.fit(X, y, epochs=50, verbose=0)
    return model

def predict(model, data, features):
    X = data[features].values
    # Normalize inputs if needed
    return model.predict(X).flatten()

# 2. RUN WALK-FORWARD ANALYSIS
wfa = WalkForwardAnalysis(
    train_period='1M',
    test_period='1W',
    output_dir='results/keras_analysis'
)

results = wfa.run_model_analysis(
    data=processed_data,
    features=features,
    target_column='target_return',
    model_factory=create_model,
    train_func=train_model,
    predict_func=predict
)

# 3. USING A SAVED KERAS MODEL
# If you already have a trained .keras model:
def load_saved_model():
    return tf.keras.models.load_model('models/my_model.keras')

# Then use with the strategy framework:
def create_strategy(model):
    return MLModelStrategy(
        model=model,
        features=features,
        prediction_type='regression',
        threshold=0.0005  # Adjust based on your model's output range
    )

# Run backtesting with your saved model
results = wfa.run_strategy_analysis(
    data=processed_data,
    features=features,
    model_factory=load_saved_model,  # This will load your saved model
    train_func=lambda m, d, f: m,    # No training needed, model is pre-trained
    strategy_factory=create_strategy
)
```