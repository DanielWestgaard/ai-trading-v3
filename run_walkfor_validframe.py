import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Import the walk-forward testing module
from data.features.time_series_ml import WalkForwardAnalysis, MLModelStrategy

# Path configuration
DATA_DIR = "data/storage/capital_com/processed"
OUTPUT_DIR = "analysis/walk_forward_results"
MODEL_DIR = "models/walk_forward"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def run_walk_forward_analysis(data_file, model_type="rf"):
    """
    Run walk-forward analysis on the given data file.
    
    Args:
        data_file: Path to the processed data file
        model_type: Type of model to use ('rf' for RandomForest, 'gb' for GradientBoosting)
    """
    print(f"Starting walk-forward analysis for {data_file}")
    
    # Load data
    data_path = os.path.join(DATA_DIR, data_file)
    data = pd.read_csv(data_path, parse_dates=['date'])
    
    # Extract instrument and time frame info for output naming
    instrument = 'unknown'
    timeframe = 'unknown'
    
    # Try to extract from filename
    filename_parts = data_file.split('_')
    if len(filename_parts) >= 3:
        instrument = filename_parts[1]
        timeframe = filename_parts[2]
    
    # Create output directories for this specific analysis
    analysis_output_dir = os.path.join(OUTPUT_DIR, f"{instrument}_{timeframe}_{model_type}")
    analysis_model_dir = os.path.join(MODEL_DIR, f"{instrument}_{timeframe}_{model_type}")
    
    os.makedirs(analysis_output_dir, exist_ok=True)
    os.makedirs(analysis_model_dir, exist_ok=True)
    
    # Set up WalkForwardAnalysis
    wfa = WalkForwardAnalysis(
        train_period='1M',  # 1 month of training data
        test_period='1W',   # 1 week of testing
        step_size='1W',     # Step forward 1 week each time
        output_dir=analysis_output_dir,
        model_store_path=analysis_model_dir,
        date_column='date'  # Specify the date column name
    )
    
    # Define features for model
    # These should be features that exist in your processed data
    # Use a subset of features that are common and reliable
    features = [
        'ema_200', 'sma_50', 'rsi_14', 'macd_signal',
        'volatility_20', 'atr_14', 'day_of_week'
    ]
    
    # Verify that the selected features actually exist in the data
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Warning: The following features are missing: {missing_features}")
        print(f"Available columns: {data.columns}")
        # Filter out missing features
        features = [f for f in features if f not in missing_features]
        if not features:
            print("No valid features remaining. Aborting analysis.")
            return
    
    print(f"Using {len(features)} features: {features}")
    
    # Define model factory based on model_type
    def create_model():
        if model_type == "rf":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gb":
            return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Define training function
    def train_model(model, train_data, features, target):
        # Create a copy of the data to avoid SettingWithCopyWarning
        train_df = train_data.copy()
        
        # Get feature values
        X = train_df[features]
        y = train_df[target]
        
        # Handle NaN values if any
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Scale features for better model performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        model.fit(X_scaled, y)
        
        # Store scaler in model for later use
        model.scaler = scaler
        
        return model
    
    # Define prediction function
    def predict(model, test_data, features):
        # Create a copy of the data
        test_df = test_data.copy()
        
        # Get feature values
        X = test_df[features]
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Scale features using the stored scaler
        X_scaled = model.scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        return predictions
    
    # Run the model analysis
    print("Running model analysis...")
    model_results = wfa.run_model_analysis(
        data=data,
        features=features,
        create_target=True,  # Automatically create a target column
        target_type='return',
        source_column='close_original',  # Use the preserved original close price
        horizon=10,  # Predict return 10 periods ahead (e.g., 50 minutes for M5 data)
        model_factory=create_model,
        train_func=train_model,
        predict_func=predict,
        prediction_type='regression'
    )
    
    # Print summary of results
    print("\nModel Analysis Results:")
    if 'overall_performance' in model_results and 'overall' in model_results['overall_performance']:
        metrics = model_results['overall_performance']['overall']
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
        print(f"  R²: {metrics.get('r2', 'N/A'):.6f}")
        print(f"  Direction Accuracy: {metrics.get('direction_accuracy', 'N/A'):.2%}")
    
    # Run strategy backtesting
    print("\nRunning strategy backtesting...")
    
    # Define strategy factory
    def create_strategy(model):
        return MLModelStrategy(
            model=model,
            features=features,
            prediction_type='regression',
            threshold=0.0001  # Signal threshold - adjust based on your needs
        )
    
    # Get the target column name that was created
    target_column = f"target_close_original_10"  # This matches our earlier settings
    
    # Define training function for strategy (simpler version that expects the target column)
    def train_for_strategy(model, train_data, features):
        return train_model(model, train_data, features, target_column)
    
    # Run the strategy analysis
    strategy_results = wfa.run_strategy_analysis(
        data=data,
        features=features,
        model_factory=create_model,
        train_func=train_for_strategy,
        strategy_factory=create_strategy,
        initial_capital=10000.0
    )
    
    # Print summary of backtest results
    print("\nStrategy Backtest Results:")
    if 'aggregate_performance' in strategy_results and 'avg' in strategy_results['aggregate_performance']:
        metrics = strategy_results['aggregate_performance']['avg']
        print(f"  Avg Return: {metrics.get('annualized_return_pct', 'N/A'):.2f}%")
        print(f"  Avg Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"  Avg Drawdown: {metrics.get('max_drawdown_pct', 'N/A'):.2f}%")
    
    print(f"\nAnalysis complete! Results saved to {analysis_output_dir}")
    return model_results, strategy_results


def analyze_multiple_timeframes():
    """Run analysis on multiple processed data files."""
    # Get all processed data files
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith('processed_') and f.endswith('.csv')]
    
    results = {}
    
    for file in data_files:
        print(f"\n{'='*50}")
        print(f"Analyzing {file}")
        print(f"{'='*50}\n")
        
        try:
            model_results, strategy_results = run_walk_forward_analysis(file)
            
            # Store results for comparison
            instrument_timeframe = "_".join(file.split("_")[1:3])
            results[instrument_timeframe] = {
                'model': model_results.get('overall_performance', {}),
                'strategy': strategy_results.get('aggregate_performance', {})
            }
            
        except Exception as e:
            print(f"Error analyzing {file}: {str(e)}")
    
    # Create a comparison report
    if results:
        comparison_df = pd.DataFrame([
            {
                'instrument_timeframe': k,
                'direction_accuracy': v['model'].get('overall', {}).get('direction_accuracy', float('nan')),
                'rmse': v['model'].get('overall', {}).get('rmse', float('nan')),
                'return_pct': v['strategy'].get('avg', {}).get('annualized_return_pct', float('nan')),
                'sharpe_ratio': v['strategy'].get('avg', {}).get('sharpe_ratio', float('nan')),
                'max_drawdown_pct': v['strategy'].get('avg', {}).get('max_drawdown_pct', float('nan'))
            }
            for k, v in results.items()
        ])
        
        # Save comparison report
        comparison_path = os.path.join(OUTPUT_DIR, 'timeframe_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison report saved to {comparison_path}")


def analyze_different_models():
    """Compare different model types on the same data."""
    # Select a specific data file
    data_file = 'processed_GBPUSD_m5_20240101_20250101.csv'
    
    model_types = ['rf', 'gb']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} model on {data_file}")
        print(f"{'='*50}\n")
        
        try:
            run_walk_forward_analysis(data_file, model_type=model_type)
        except Exception as e:
            print(f"Error with {model_type} model: {str(e)}")


if __name__ == "__main__":
    # Check if the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found!")
    else:
        # Run analysis on a specific file
        run_walk_forward_analysis('processed_GBPUSD_m5_20240101_20250101.csv')
        
        # Uncomment these to run more comprehensive analyses
        # analyze_multiple_timeframes()
        # analyze_different_models()