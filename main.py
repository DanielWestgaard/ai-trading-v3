import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data.pipelines.data_pipeline import DataPipeline
from data.processors.cleaner import DataCleaner
import utils.logging_utils as log_utils
from broker.capital_com.capitalcom import CapitalCom
import config.data_config as data_config
import config.system_config as sys_config
import models.run_model_training as run_model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid trading strategy models with uncertainty quantification')
    
    parser.add_argument('--broker-func', action='store_true', default=False, help='Test all broker functionality')
    parser.add_argument('--data-pipeline', action='store_true', default=False, help='Run data processing pipeline')
    parser.add_argument('--backtest', action='store_true', default=False, help='Run backtesting')
    parser.add_argument('--walk-forward-analysis', action='store_true', default=False, 
                        help='Run Walk-Forward Testing and Cross-Validation Implementation with Backtesting system.')
    parser.add_argument('--train-model', action='store_true', default=True, help='Train model')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger = log_utils.setup_logging(log_to_file=False, log_level=sys_config.DEBUG_LOG_LEVEL)
    
    if args.broker_func:
        broker = CapitalCom()
        broker.start_session()
        # broker.session_details(print_answer=True)
        # broker.switch_active_account(print_answer=True)
        # broker.list_all_accounts(print_answer=True)
        # broker.get_historical_data(epic="SILVER", resolution="MINUTE_5",
        #                            from_date="2020-02-24T00:00:00", to_date="2020-02-24T01:00:00",
        #                            print_answer=True)
        broker.fetch_and_save_historical_prices(epic="GBPUSD", resolution="MINUTE_5",
                                                from_date="2024-01-01T00:00:00", to_date="2025-01-01T01:00:00",
                                                print_answer=False)
        
        broker.end_session()

    if args.data_pipeline:
        data_pipeline = DataPipeline()
        # Run pipeline with intermediate saves for inspection
        result, saved_file = data_pipeline.run(save_intermediate=False)
      
    if args.walk_forward_analysis:
        from data.features.time_series_ml import WalkForwardAnalysis

        # Load your processed data
        data = pd.read_csv("data/storage/capital_com/processed/processed_GBPUSD_m5_20240101_20250101.csv", 
                        parse_dates=['date'])

        # Initialize the walk-forward analyzer
        wfa = WalkForwardAnalysis(
            train_period='1M',   # 1 month training window
            test_period='1W',    # 1 week testing window
            output_dir='walk_forward_results'
        )

        # Define your model creation and training functions
        def create_model():
            return RandomForestRegressor(n_estimators=100)

        def train_model(model, data, features, target):
            X = data[features]
            y = data[target]
            return model.fit(X, y)

        def predict(model, data, features):
            X = data[features]
            return model.predict(X)

        features = [
            'ema_200', 'sma_50', 'rsi_14', 'macd_signal',  # Technical indicators
            'open_return', 'high_return', 'low_return', 'close_return'  # Normalized returns
        ]
        
        # Run the analysis with automated target creation
        results = wfa.run_model_analysis(
            data=data,
            features=features,
            create_target=True,
            target_type='return',
            source_column='close_raw',
            horizon=10,
            model_factory=create_model,
            train_func=train_model,
            predict_func=predict
        )
        
    if args.train_model:
        model_config = {
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
        
        run_model.main(data_path=data_config.TESTING_PROCESSED_DATA, model_config=model_config)

    if args.backtest:
        # Call run_backtest.py or integrate it into here
        pass

if __name__ == "__main__":
    exit(main())