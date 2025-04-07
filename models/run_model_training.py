# run_model_backtest.py
import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Set up logging
from utils.logging_utils import setup_logging
logger = setup_logging(name="model_backtest", type="model_backtest", log_to_file=False)

# Import custom modules
from models.model_trainer import ModelTrainer
from models.model_factory import ModelFactory
from backtesting.backtest_runner import BacktestRunner
from backtesting.strategies.model_based_strategy import ModelBasedStrategy
from backtesting.data.market_data import PipelineMarketData
from config import data_config


def load_data(data_path):
    """Load and prepare data."""
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Basic data checks
    logger.info(f"Data loaded with shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def train_model(df, model_config):
    """Train a model with the provided configuration."""
    logger.info("Training model with configuration:")
    logger.info(json.dumps(model_config, indent=2))
    
    # Create model trainer
    trainer = ModelTrainer(
        model_type=model_config.get('model_type', 'xgboost'),
        features=model_config.get('features'),
        target=model_config.get('target', 'close_return'),
        prediction_type=model_config.get('prediction_type', 'classification'),
        lookback_periods=model_config.get('lookback_periods', 10),
        prediction_horizon=model_config.get('prediction_horizon', 1),
        test_size=model_config.get('test_size', 0.2),
        validation_size=model_config.get('validation_size', 0.1),
        n_splits=model_config.get('n_splits', 5),
        scale_features=model_config.get('scale_features', True),
        #model_dir=model_config.get('model_dir', 'model_storage'),
        #results_dir=model_config.get('results_dir', 'ml_model_results'),
        model_params=model_config.get('model_params', {})
    )
    
    # Train the model
    model = trainer.train(df)
    
    # Cross-validate if requested
    if model_config.get('cross_validate', False):
        logger.info("Performing cross-validation")
        cv_results = trainer.cross_validate(df)
        logger.info(f"Cross-validation results: {cv_results}")
    
    return model, trainer

def run_backtest(model, data_path, backtest_config):
    """Run a backtest using the trained model."""
    logger.info("Setting up backtest with configuration:")
    logger.info(json.dumps(backtest_config, indent=2))
    
    # Create backtest runner
    runner = BacktestRunner(
        output_dir=backtest_config.get('output_dir', 'backtest_results')
    )
    
    # Set up market data
    market_data_config = backtest_config.get('market_data', {})
    market_data = PipelineMarketData(
        processed_data_path=data_path,
        symbols=market_data_config.get('symbols', ['GBPUSD']),
        date_col=market_data_config.get('date_col', 'date')
    )
    
    # Set up model-based strategy
    required_features = model.features
    logger.info(f"Using {len(required_features)} features for model predictions")
    
    strategy = ModelBasedStrategy(
        symbols=market_data_config.get('symbols', ['GBPUSD']),
        model=model,
        prediction_threshold=backtest_config.get('prediction_threshold', 0.55),
        confidence_threshold=backtest_config.get('confidence_threshold', 0.0),
        lookback_window=backtest_config.get('lookback_window', 1),
        required_features=required_features
    )
    
    # Set up and run backtest
    backtest_id = backtest_config.get('backtest_id', f'model_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Create backtest
    runner.create_backtest(
        backtest_id=backtest_id,
        strategy=strategy,
        market_data=market_data,
        initial_capital=backtest_config.get('initial_capital', 10000.0),
        risk_manager_config=backtest_config.get('risk_manager'),
        slippage_model=backtest_config.get('slippage_model', 'fixed'),
        slippage_params=backtest_config.get('slippage_params')
    )
    
    # Run backtest
    results = runner.run_backtest(backtest_id)
    
    # Display summary
    runner.display_summary_table(backtest_id)
    
    return results, runner

def main(data_path=None, model_config=None):
    """Main function to run the entire workflow."""
    logger.info("Starting model training and backtesting workflow")
    
    # Configuration
    DATA_PATH = data_path or data_config.TESTING_PROCESSED_DATA  # "data/storage/capital_com/processed/processed_GBPUSD_m5_20240101_20250101.csv"
    
    # Model configuration
    model_config = model_config or {
        'model_type': 'xgboost',
        'prediction_type': 'classification',
        'target': 'close_return',  # Be explicit
        'features': None,
        'prediction_horizon': 1,
        'test_size': 0.2,
        'cross_validate': True,
        'scale_features': True,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
    }
    
    # Backtest configuration
    backtest_config = {
        'backtest_id': f'xgboost_gbpusd_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'market_data': {
            'symbols': ['GBPUSD'],
            'date_col': 'date'
        },
        'initial_capital': 10000.0,
        'prediction_threshold': 0.55,
        'confidence_threshold': 0.6,
        'lookback_window': 1,
        'risk_manager': {
            'position_sizing_method': 'percent',
            'position_sizing_params': {
                'percent': 2.0  # Use 2% of equity per trade
            }
        }
    }
    
    try:
        # Load data
        df = load_data(DATA_PATH)
        
        # Train model
        model, trainer = train_model(df, model_config)
        
        # Run backtest
        # results, runner = run_backtest(model, DATA_PATH, backtest_config)
        
        # Final results
        # backtest_id = backtest_config['backtest_id']
        # logger.info(f"Completed backtest: {backtest_id}")
        
        # # Compare with baseline
        # baseline_id = "indicator_crossover_gbpusd"  # Replace with your baseline strategy
        
        # try:
        #     comparison = runner.compare_backtests([backtest_id, baseline_id])
        #     logger.info(f"Comparison with baseline:\n{comparison}")
            
        #     # Plot comparison
        #     runner.plot_equity_comparison(
        #         [backtest_id, baseline_id],
        #         title="Model vs Indicator Strategy",
        #         save_path=f"backtest_results/{backtest_id}/reports/comparison.png"
        #     )
        # except Exception as e:
        #     logger.warning(f"Could not compare with baseline: {str(e)}")
        
        logger.info("Workflow completed successfully")
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()