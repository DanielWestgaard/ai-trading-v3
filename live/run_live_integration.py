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
    # Create data handler
    data_handler = LiveDataHandler(
        model=model,
        output_file=f'live_data_{symbol}_{timeframe}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    )
    
    # Start the data handler
    data_handler.start()
    
    # Create a stop event for controlled termination
    stop_event = Event()
    
    # Define custom message handler that forwards to our data handler
    def custom_message_handler(ws, message):
        """Called when a message is received from the WebSocket."""
        data_handler.process_message(message)
        
        # Print a sample of the processed data periodically
        if hasattr(custom_message_handler, 'counter'):
            custom_message_handler.counter += 1
        else:
            custom_message_handler.counter = 1
            
        if custom_message_handler.counter % 10 == 0:
            latest = data_handler.get_latest_data()
            if latest:
                logging.info(f"Latest midpoint data: {latest['datetime']} | Open: {latest['o']:.5f} | Close: {latest['c']:.5f} | Spread: {latest['spread']:.6f}")
    
    try:
        logging.info(f"Subscribing to {symbol} {timeframe} data...")
        
        # Subscribe with custom message handler
        ws = broker.sub_live_market_data_with_handler(
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
        
        # Wait for termination signal
        try:
            while not stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down...")
            stop_event.set()
            
    except Exception as e:
        logging.error(f"Error in data subscription: {e}")
        
    finally:
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