import os
import sys
import logging
import time
from datetime import datetime, timedelta
import argparse
import pandas as pd
import yaml
import signal
import threading

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.capital_com.capitalcom import CapitalCom
from models.model_factory import ModelFactory
from core.strategies.model_based_strategy import ModelBasedStrategy
from core.risk.risk_manager import RiskManager, PositionSizer
from live.execution_service.live_trading_service import LiveTradingService
from live.live_data_handling.live_data_handler import LiveDataHandler
import config.constants.system_config as sys_config


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_model(model_path, model_type='xgboost'):
    """Load a trained model."""
    logging.info(f"Loading model from {model_path}")
    
    model = ModelFactory.create_model(model_type)
    success = model.load(model_path)
    
    if not success:
        logging.error(f"Failed to load model from {model_path}")
        return None
    
    logging.info(f"Model loaded from {model_path}")
    return model


def create_strategy(model, symbols, config):
    """Create trading strategy."""
    strategy_config = config.get('strategy', {})
    
    # Extract feature list if available
    features = None
    if hasattr(model, 'features'):
        features = model.features
    
    # Create strategy
    strategy = ModelBasedStrategy(
        symbols=symbols,
        model=model,
        prediction_threshold=strategy_config.get('prediction_threshold', 0.55),
        confidence_threshold=strategy_config.get('confidence_threshold', 0.0),
        lookback_window=strategy_config.get('lookback_window', 1),
        required_features=features,
        params=strategy_config
    )
    
    return strategy


def create_risk_manager(config):
    """Create risk manager."""
    risk_config = config.get('risk', {})
    
    # Create position sizer
    position_sizer = PositionSizer(
        method=risk_config.get('position_sizing_method', 'percent'),
        params=risk_config.get('position_sizing_params', {'percent': 5.0})
    )
    
    # Create risk manager
    risk_manager = RiskManager(
        position_sizer=position_sizer,
        max_position_size=risk_config.get('max_position_size'),
        max_correlated_positions=risk_config.get('max_correlated_positions', 3),
        max_portfolio_risk=risk_config.get('max_portfolio_risk', 20.0),
        auto_stop_loss=risk_config.get('auto_stop_loss', True),
        stop_loss_method=risk_config.get('stop_loss_method', 'percent'),
        stop_loss_params=risk_config.get('stop_loss_params'),
        auto_take_profit=risk_config.get('auto_take_profit', True),
        take_profit_method=risk_config.get('take_profit_method', 'percent'),
        take_profit_params=risk_config.get('take_profit_params')
    )
    
    return risk_manager

def warmup_system(symbols, timeframe, broker: CapitalCom, data_handler: LiveDataHandler):
    # Fetch historical data for warm-up
    logging.info(f"Fetching historical data for warm-up period")
    for symbol in symbols:
        # Calculate date range for historical data
        lookback_bars = 250  # More than needed to ensure enough after processing
        to_date = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        
        # Calculate from_date based on the timeframe
        timeframe_minutes = {"MINUTE": 1, "MINUTE_5": 5, "MINUTE_15": 15, 
                            "MINUTE_30": 30, "HOUR": 60, "HOUR_4": 240, 
                            "DAY": 1440}.get(timeframe, 60)
        
        minutes_to_subtract = timeframe_minutes * lookback_bars
        from_date = (datetime.now() - timedelta(minutes=minutes_to_subtract, hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
        
        logging.info(f"Fetching historical data for {symbol} from {from_date} to {to_date}")
        
        historical_data = broker.get_historical_data(
            epic=symbol,
            resolution=timeframe,
            from_date=from_date,
            to_date=to_date,
            max=lookback_bars
        )
        
        # Format and warm up the data handler
        if "prices" in historical_data:
            formatted_history = []
            for candle in historical_data["prices"]:
                # Calculate OHLC midpoints
                open_price = (candle.get("openPrice", {}).get("bid", 0) + 
                             candle.get("openPrice", {}).get("ask", 0)) / 2
                high_price = (candle.get("highPrice", {}).get("bid", 0) + 
                             candle.get("highPrice", {}).get("ask", 0)) / 2
                low_price = (candle.get("lowPrice", {}).get("bid", 0) + 
                            candle.get("lowPrice", {}).get("ask", 0)) / 2
                close_price = (candle.get("closePrice", {}).get("bid", 0) + 
                              candle.get("closePrice", {}).get("ask", 0)) / 2
                
                formatted_candle = {
                    "epic": symbol,
                    "resolution": timeframe,
                    "t": pd.to_datetime(candle.get("snapshotTime")).timestamp() * 1000,
                    "datetime": pd.to_datetime(candle.get("snapshotTime")),
                    # Include standard OHLC fields
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    # Include raw fields that match what the model expects
                    "open_raw": open_price,
                    "high_raw": high_price,
                    "low_raw": low_price,
                    "close_raw": close_price,
                    # Also include _original versions for consistency
                    "open_original": open_price,
                    "high_original": high_price,
                    "low_original": low_price,
                    "close_original": close_price,
                    "volume": candle.get("lastTradedVolume", 0)
                }
                formatted_history.append(formatted_candle)
            
            # Warm up the data handler with historical data
            data_handler.warm_up_with_historical_data(formatted_history)
            logging.info(f"Successfully retrieved {lookback_bars} previous bars.")
        else:
            logging.warning(f"No historical data received for {symbol}")

def run_live_trading(config_path, model_path, duration_hours=None):
    """
    Run live trading with the specified configuration and model.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model
        duration_hours: Duration to run in hours (None for indefinite)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load model
    model_type = config.get('model_type', 'xgboost')
    model = load_model(model_path, model_type)
    
    if not model:
        logging.error("Failed to load model. Exiting.")
        return
    
    # Get trading symbols
    symbols = config.get('symbols', ['GBPUSD'])
    
    # Initialize components
    try:
        # Initialize broker
        broker = CapitalCom()
        
        # Start broker session
        logging.info("Starting session with Capital.com...")
        broker.start_session()
        
        # Switch to demo account
        broker.switch_active_account(account_name=config.get('account_name'))
        logging.info(f"Switched to account: {config.get('account_name', 'default')}")
        
        # Create data handler with pipeline components
        feature_generator = None  # Initialize if needed
        feature_preparator = None  # Initialize if needed
        data_normalizer = None  # Initialize if needed
        
        # Create live data handler
        output_dir = sys_config.LIVE_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        data_handler = LiveDataHandler(
            model=model,
            output_file=os.path.join(output_dir, f"live_data_{symbols[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            feature_generator=feature_generator,
            feature_preparator=feature_preparator,
            data_normalizer=data_normalizer,
            model_features=model.features if hasattr(model, 'features') else None
        )
        
        # Create strategy
        strategy = create_strategy(model, symbols, config)
        
        # Create risk manager
        risk_manager = create_risk_manager(config)
        
        # Create trading service
        trading_service = LiveTradingService(
            strategy=strategy,
            broker=broker,
            data_handler=data_handler,
            risk_manager=risk_manager,
            initial_capital=config.get('initial_capital', 10000.0),
            max_active_positions=config.get('max_active_positions', 3)
        )
        
        # Subscribe to market data
        for symbol in symbols:
            logging.info(f"Subscribing to market data for {symbol}")
            timeframe = config.get('timeframe', 'MINUTE')
            
            # Warming up the system with enough (lookback) bars to ensure enough data for features to be calculated
            warmup_system(symbols=symbols, timeframe=timeframe, broker=broker, data_handler=data_handler)
                
            # Custom message handler to forward data to the data handler
            def custom_message_handler(ws, message):
                data_handler.process_message(message)
            
            # Subscribe to market data
            broker.sub_live_market_data(
                symbol=symbol,
                timeframe=timeframe,
                message_handler=custom_message_handler
            )
        
        # Start data handler
        data_handler.start()
        
        # Start trading service
        trading_service.start()
        
        # Run for specified duration or indefinitely
        if duration_hours:
            duration_seconds = duration_hours * 60 * 60
            logging.info(f"Running live trading for {duration_hours} hours")
            time.sleep(duration_seconds)
        else:
            # Set up signal handler for graceful shutdown
            def signal_handler(sig, frame):
                logging.info("Received shutdown signal, stopping trading...")
                trading_service.stop()
                broker.end_session()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            logging.info("Running indefinitely (Ctrl+C to stop)")
            
            # Keep the main thread alive
            while True:
                time.sleep(60)
        
    except Exception as e:
        logging.error(f"Error in live trading: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # Cleanup
        if 'trading_service' in locals():
            trading_service.stop()
        
        if 'broker' in locals():
            broker.end_session()
        
        logging.info("Live trading session ended. Waiting for ping to close (may take up to a minute)...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live trading with a trained model")
    parser.add_argument("--config", type=str, default="config/live_trading_config.yaml", help="Path to configuration file")
    parser.add_argument("--model", type=str, required=False, help="Path to trained model")
    parser.add_argument("--duration", type=float, help="Duration to run in hours")
    
    args = parser.parse_args()
    
    run_live_trading(args.config, args.model or "model_registry/model_storage/xgboost_20250403_102300_20250403_102301.pkl", args.duration)