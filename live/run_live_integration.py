import logging
import time
import json
import os
from threading import Event

from broker.capital_com.capitalcom import CapitalCom
from live.live_data_handling.live_data_handler import LiveDataHandler
from models.model_factory import ModelFactory
import config.system_config as sys_config


def load_model(model_path: str = None):
    """Load a trained model from disk."""
    model_type = "xgboost"  # or "random_forest" depending on what's already saved
    model = ModelFactory.create_model(model_type)
    model_path = model_path or "model_registry/model_storage/xgboost_20250403_102300_20250403_102301.pkl"
    
    # Load the saved model from disk
    success = model.load(model_path)
    if success:
        logging.info(f"Successfully loaded model from {model_path}")
        return model
    else:
        logging.error(f"Could not find model at {model_path}")
        return None


def subscribe_and_process_data(broker, model, symbol="GBPUSD", timeframe="MINUTE", duration_minutes=None):
    """
    Subscribe to live market data and process it in real-time.
    
    Args:
        broker: Initialized broker instance
        model: Trained model for predictions
        symbol: Trading symbol/epic to subscribe to
        timeframe: Resolution for the data
        duration_minutes: How long to run (None for indefinite)
    """
    # Create data handler with custom output file name that includes timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f'live_data_{symbol}_{timeframe}_{timestamp}.csv'
    
    # Create full path for output file
    output_dir = sys_config.LIVE_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    
    # Create data handler
    data_handler = LiveDataHandler(
        model=model,
        output_file=output_path
    )
    
    # Start the data handler
    data_handler.start()
    
    # Create a stop event for controlled termination
    stop_event = Event()
    
    # Set up message count tracking
    message_count = 0
    last_message_time = time.time()
    
    # Define custom message handler that forwards to our data handler
    def custom_message_handler(ws, message):
        """Called when a message is received from the WebSocket."""
        nonlocal message_count, last_message_time
        
        # Forward message to data handler
        data_handler.process_message(message)
        
        # Debug logging
        if "ohlc.event" in message:
            message_count += 1
            last_message_time = time.time()
        
        # Print status every 10 OHLC messages (5 minutes)
        if message_count > 0 and message_count % 10 == 0:
            latest = data_handler.get_latest_data()
            if latest:
                logging.info(f"Latest midpoint data: {latest['datetime']} | " +
                           f"O: {latest['o']:.5f} | H: {latest['h']:.5f} | " +
                           f"L: {latest['l']:.5f} | C: {latest['c']:.5f} | " +
                           f"Spread: {latest['spread']:.6f}")
                           
                # Also log cached data sizes
                logging.debug(f"Cached data: {len(data_handler.bid_cache)} bids, " +
                             f"{len(data_handler.ask_cache)} asks, " +
                             f"{len(data_handler.recent_data)} processed points")
            else:
                logging.warning("No processed data available yet")
    
    try:
        logging.info(f"Subscribing to {symbol} {timeframe} data...")
        logging.info(f"Data will be saved to: {output_path}")
        
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
                # Check if we're still receiving data (watchdog)
                if message_count > 0 and time.time() - last_message_time > 120:
                    logging.warning("No messages received for 2 minutes, possible connection issue")
                
                # Force a file save every 60 seconds
                if message_count > 0 and message_count % 30 == 0:
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
    
    # Load the model
    model = load_model(model_path)
    
    # Initialize broker
    broker = CapitalCom()
    
    try:
        # Start session
        logging.info("Starting session with Capital.com...")
        broker.start_session()
        
        # Switch to the correct active demo account
        broker.switch_active_account(print_answer=False)
        logging.info("Switched to demo account")
        
        # Subscribe to live market data and process
        subscribe_and_process_data(
            broker=broker,
            model=model,
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