# Could be moved to deprecated, as the live_trading_runner.py is utilizing the newly added live execution logic with actual trades?
import logging
import time
import json
import os
from threading import Event
import pandas as pd

from broker.capital_com.capitalcom import CapitalCom
from live.live_data_handling.live_data_handler import LiveDataHandler
from models.model_factory import ModelFactory
from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer
import config.system_config as sys_config


def load_model_and_features(model_path: str = None):
    """
    Load a trained model and extract its expected features.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Tuple of (model, feature_list)
    """
    # Default model path if none provided
    model_path = model_path or "model_registry/model_storage/xgboost_20250403_102300_20250403_102301.pkl"
    
    # Load the model
    model_type = "xgboost"  # or "random_forest" depending on what's already saved
    model = ModelFactory.create_model(model_type)
    
    # Load the saved model from disk
    success = model.load(model_path)
    if not success:
        logging.error(f"Could not find model at {model_path}")
        return None, None
    
    logging.info(f"Successfully loaded model from {model_path}")
    
    # Try to extract feature list
    feature_list = None
    try:
        # Check if features are stored in the model
        if hasattr(model, 'features') and model.features:
            feature_list = model.features
            logging.info(f"Found {len(feature_list)} features embedded in the model")
        else:
            # Try to find feature importance
            feature_importance = model.get_feature_importance(plot=False)
            if isinstance(feature_importance, dict) and len(feature_importance) > 0:
                feature_list = list(feature_importance.keys())
                logging.info(f"Extracted {len(feature_list)} features from feature importance")
    except Exception as e:
        logging.warning(f"Could not extract features from model: {e}")
    
    # If we couldn't find features in the model, look for a feature file
    if not feature_list:
        try:
            # Try to find a features file based on model path
            base_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).split('.')[0]
            features_file = os.path.join(base_dir, f"{model_name}_features.txt")
            
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    feature_list = [line.strip() for line in f.readlines() if line.strip()]
                logging.info(f"Loaded {len(feature_list)} features from {features_file}")
            else:
                # Look for any features file in the directory
                features_files = [f for f in os.listdir(base_dir) if f.endswith('_features.txt')]
                if features_files:
                    features_file = os.path.join(base_dir, features_files[0])
                    with open(features_file, 'r') as f:
                        feature_list = [line.strip() for line in f.readlines() if line.strip()]
                    logging.info(f"Found and loaded {len(feature_list)} features from {features_file}")
        except Exception as e:
            logging.warning(f"Could not load features from file: {e}")
    
    if not feature_list:
        logging.warning("No feature list found. The model will attempt to use all available features.")
    
    return model, feature_list


def setup_data_pipeline():
    """Set up data pipeline components for live data processing."""
    
    # Create the feature generator
    feature_generator = FeatureGenerator(
        price_cols=['Open', 'High', 'Low', 'Close'],
        volume_col='Volume',
        timestamp_col='Date',
        preserve_original_case=True
    )
    
    # Create the feature preparator
    feature_preparator = FeaturePreparator(
        price_cols=['Open', 'High', 'Low', 'Close'],
        volume_col='Volume',
        timestamp_col='Date',
        preserve_original_prices=True,
        price_transform_method='returns',
        treatment_mode='basic'  # Basic mode for live data is usually better
    )
    
    # Create the normalizer
    data_normalizer = DataNormalizer(
        price_cols=['open', 'high', 'low', 'close'],
        volume_col='volume',
        price_method='returns',
        volume_method='log',
        other_method='zscore'
    )
    
    return feature_generator, feature_preparator, data_normalizer


def subscribe_and_process_data(broker, model, feature_list=None, symbol="GBPUSD", timeframe="MINUTE", duration_minutes=None):
    """
    Subscribe to live market data and process it in real-time with the data pipeline.
    
    Args:
        broker: Initialized broker instance
        model: Trained model for predictions
        feature_list: List of features expected by the model
        symbol: Trading symbol/epic to subscribe to
        timeframe: Resolution for the data
        duration_minutes: How long to run (None for indefinite)
    """
    # Create output directories
    output_dir = os.path.join(sys_config.LIVE_DATA_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'live_data_{symbol}_{timeframe}_{timestamp}.csv')
    
    # Set up data pipeline components
    feature_generator, feature_preparator, data_normalizer = setup_data_pipeline()
    
    # Create data handler with pipeline components
    data_handler = LiveDataHandler(
        model=model,
        output_file=output_file,
        feature_generator=feature_generator,
        feature_preparator=feature_preparator,
        data_normalizer=data_normalizer,
        model_features=feature_list
    )
    
    # Start the data handler
    data_handler.start()
    
    # Create a stop event for controlled termination
    stop_event = Event()
    
    # Set up message tracking
    message_count = 0
    last_message_time = time.time()
    
    # Define custom message handler that forwards to our data handler
    def custom_message_handler(ws, message):
        """Called when a message is received from the WebSocket."""
        nonlocal message_count, last_message_time
        
        # Forward message to data handler
        data_handler.process_message(message)
        
        # Count OHLC messages
        if "ohlc.event" in message:
            message_count += 1
            last_message_time = time.time()
        
        # Print status every 20 OHLC messages
        if message_count > 0 and message_count % 20 == 0:  # Perhaps change to every 5 message so it's easier to follow the status?
            logging.info(f"Processed {message_count} market data messages")
            
            # Also check data buffer status
            with data_handler.lock:
                buffer_size = len(data_handler.recent_data)
                bid_cache_size = len(data_handler.bid_cache)
                ask_cache_size = len(data_handler.ask_cache)
            
            logging.info(f"Buffer status: {buffer_size} processed points, "
                       f"{bid_cache_size} pending bids, {ask_cache_size} pending asks")
            
            # Log latest data point if available
            latest = data_handler.get_latest_data()
            if latest:
                logging.info(f"Latest data: {latest.get('datetime')} | "
                          f"OHLC: {latest.get('open'):.5f}/{latest.get('high'):.5f}/"
                          f"{latest.get('low'):.5f}/{latest.get('close'):.5f}")
    
    try:
        logging.info(f"Subscribing to {symbol} {timeframe} data with integrated data pipeline...")
        logging.info(f"Data will be saved to: {output_file}")
        
        # Subscribe with custom message handler
        ws = broker.sub_live_market_data(
            symbol=symbol, 
            timeframe=timeframe,
            message_handler=custom_message_handler
        )
        
        # Set timeout if specified
        if duration_minutes:
            def stop_after_timeout():
                logging.info(f"Stopping after {duration_minutes} minutes...")
                stop_event.set()
                
            # Schedule the stop
            import threading
            timer = threading.Timer(duration_minutes * 60, stop_after_timeout)
            timer.daemon = True
            timer.start()
        
        # Wait for termination signal with watchdog
        try:
            while not stop_event.is_set():
                # Check if we're still receiving data
                if message_count > 0 and time.time() - last_message_time > 120:
                    logging.warning("No messages received for 2 minutes, possible connection issue")
                
                # Force a file save every 60 seconds
                if message_count > 0 and time.time() % 60 < 1:
                    data_handler._save_to_file()
                
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down...")
            stop_event.set()
            
    except Exception as e:
        logging.error(f"Error in data subscription: {e}")
        
    finally:
        # Final save before stopping
        if data_handler.recent_data:
            data_handler._save_to_file()
            
        # Clean up
        data_handler.stop()
        logging.info("Data handler stopped")


def main(model_path: str = None, duration_minutes: int = None):
    """
    Main function to run live market data integration.
    
    Args:
        model_path: Path to the trained model
        duration_minutes: How long to run in minutes (None for indefinite)
    """    
    # Load the model and extract features
    model, feature_list = load_model_and_features(model_path)
    
    if not model:
        logging.error("Failed to load model. Exiting.")
        return
    
    # Initialize broker
    broker = CapitalCom()
    
    try:
        # Start session
        logging.info("Starting session with Capital.com...")
        broker.start_session()
        
        # Switch to the correct active demo account
        broker.switch_active_account(print_answer=False)
        logging.info("Switched to demo account")
        
        # Subscribe to live market data and process with data pipeline
        subscribe_and_process_data(
            broker=broker,
            model=model,
            feature_list=feature_list,
            symbol="GBPUSD",
            timeframe="MINUTE",
            duration_minutes=duration_minutes
        )
        
    except Exception as e:
        logging.error(f"Error in live integration: {e}")
        
    finally:
        # Close session
        logging.info("Ending session with Capital.com...")
        broker.end_session()


if __name__ == "__main__":
    # Run for 60 minutes by default
    main(duration_minutes=60)