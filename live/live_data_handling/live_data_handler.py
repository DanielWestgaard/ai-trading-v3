import json
import logging
import time
import pandas as pd
from threading import Thread, Lock
from queue import Queue
from typing import Dict, List, Optional, Callable


class LiveDataHandler:
    """
    Handles processing of live market data from WebSocket connections.
    Calculates midpoint prices from bid/ask pairs and prepares data for model consumption.
    """
    
    def __init__(self, model=None, strategy=None, output_file='live/temp_live/live_market_data.csv'):
        """
        Initialize the LiveDataHandler.
        
        Args:
            model: The prediction model (optional)
            strategy: Trading strategy to execute trades (optional)
            output_file: CSV file to periodically save data to
        """
        self.data_queue = Queue(maxsize=1000)
        self.recent_data = []  # In-memory buffer for recent data
        self.unpaired_bids = {}  # Store bids waiting for matching asks
        self.unpaired_asks = {}  # Store asks waiting for matching bids
        self.output_file = output_file
        self.model = model
        self.strategy = strategy
        self.lock = Lock()  # To protect shared data
        self.is_running = False
        self.processor_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the data processor thread."""
        if not self.is_running:
            self.is_running = True
            self.processor_thread = Thread(target=self._data_processor, daemon=True)
            self.processor_thread.start()
            self.logger.info("LiveDataHandler processor thread started")
    
    def stop(self):
        """Stop the data processor thread."""
        self.is_running = False
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5.0)
            self.logger.info("LiveDataHandler processor thread stopped")
    
    def process_message(self, message: str):
        """
        Process incoming WebSocket message.
        
        Args:
            message: The WebSocket message as a string
        """
        try:
            data = json.loads(message)
            
            # Only process OHLC events
            if data.get("destination") != "ohlc.event":
                return
            
            payload = data.get("payload", {})
            price_type = payload.get("priceType")
            timestamp = payload.get("t")
            
            if not (price_type and timestamp):
                return
                
            # Store message based on type
            if price_type == "bid":
                self._handle_price_message(payload, "bid")
            elif price_type == "ask":
                self._handle_price_message(payload, "ask")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def _handle_price_message(self, data: Dict, price_type: str):
        """
        Handle a single price message (bid or ask) and try to match it.
        
        Args:
            data: The payload data from the message
            price_type: Either "bid" or "ask"
        """
        timestamp = data.get("t")
        epic = data.get("epic")
        
        # Store this message
        if price_type == "bid":
            self.unpaired_bids[timestamp] = data
            # Check if we have a matching ask
            if timestamp in self.unpaired_asks:
                self._create_midpoint(self.unpaired_bids[timestamp], self.unpaired_asks[timestamp])
                # Clean up after matching
                del self.unpaired_bids[timestamp]
                del self.unpaired_asks[timestamp]
        else:  # ask
            self.unpaired_asks[timestamp] = data
            # Check if we have a matching bid
            if timestamp in self.unpaired_bids:
                self._create_midpoint(self.unpaired_bids[timestamp], self.unpaired_asks[timestamp])
                # Clean up after matching
                del self.unpaired_bids[timestamp]
                del self.unpaired_asks[timestamp]
        
        # Clean up old unmatched messages (older than 30 seconds)
        self._cleanup_old_messages()
    
    def _create_midpoint(self, bid_data: Dict, ask_data: Dict):
        """
        Create midpoint data from matching bid and ask, and add to processing queue.
        
        Args:
            bid_data: The bid price data
            ask_data: The ask price data
        """
        # Ensure bid and ask are for the same instrument and timeframe
        if bid_data["epic"] != ask_data["epic"] or bid_data["resolution"] != ask_data["resolution"]:
            return
            
        # Calculate midpoint values
        midpoint = {
            "epic": bid_data["epic"],
            "resolution": bid_data["resolution"],
            "t": bid_data["t"],
            "datetime": pd.to_datetime(bid_data["t"], unit='ms'),
            "o": (bid_data["o"] + ask_data["o"]) / 2,
            "h": (bid_data["h"] + ask_data["h"]) / 2,
            "l": (bid_data["l"] + ask_data["l"]) / 2,
            "c": (bid_data["c"] + ask_data["c"]) / 2,
            "spread": ask_data["c"] - bid_data["c"]  # Capture spread information
        }
        
        # Add to queue for processing
        self.data_queue.put(midpoint)
    
    def _cleanup_old_messages(self, max_age_seconds: int = 30):
        """
        Remove old unmatched messages to prevent memory leaks.
        
        Args:
            max_age_seconds: Maximum age in seconds to keep unmatched messages
        """
        current_time = int(time.time() * 1000)  # Convert to milliseconds
        
        # Remove old bids
        bid_timestamps = list(self.unpaired_bids.keys())
        for ts in bid_timestamps:
            if current_time - ts > max_age_seconds * 1000:
                del self.unpaired_bids[ts]
        
        # Remove old asks
        ask_timestamps = list(self.unpaired_asks.keys())
        for ts in ask_timestamps:
            if current_time - ts > max_age_seconds * 1000:
                del self.unpaired_asks[ts]
    
    def _data_processor(self):
        """Consumer thread that processes the midpoint data and feeds to model."""
        while self.is_running:
            try:
                # Get data from queue (with timeout to check is_running periodically)
                try:
                    midpoint_data = self.data_queue.get(timeout=1.0)
                except Queue.Empty:
                    continue
                
                with self.lock:
                    # Add to in-memory buffer
                    self.recent_data.append(midpoint_data)
                    
                    # If buffer grows too large, trim it
                    if len(self.recent_data) > 1000:  # Keep last 1000 points
                        self.recent_data = self.recent_data[-1000:]
                
                # Periodically save to disk
                if len(self.recent_data) % 100 == 0:
                    self._save_to_file()
                
                # Feed to model for prediction if model exists
                if self.model:
                    try:
                        # Convert the recent data to the format expected by your model
                        model_input = self._prepare_model_input()
                        prediction = self.model.predict(model_input)
                        
                        # Execute trade if strategy exists and prediction meets criteria
                        if self.strategy and prediction:
                            self.strategy.evaluate_signal(prediction)
                    except Exception as e:
                        self.logger.error(f"Error in model prediction or strategy execution: {e}")
                
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in data processor: {e}")
    
    def _prepare_model_input(self):
        """
        Prepare recent data for model consumption.
        This method should be customized based on your model's input requirements.
        """
        with self.lock:
            # Convert list of dictionaries to pandas DataFrame
            df = pd.DataFrame(self.recent_data)
            
            # Ensure datetime is the index
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            
            # Add any necessary technical indicators or features here
            
            return df
    
    def _save_to_file(self):
        """Save recent data to disk."""
        try:
            with self.lock:
                df = pd.DataFrame(self.recent_data)
            
            # Save with header only if file doesn't exist
            import os
            file_exists = os.path.isfile(self.output_file)
            df.to_csv(self.output_file, mode='a', header=not file_exists, index=False)
            self.logger.info(f"Saved {len(df)} records to {self.output_file}")
        except Exception as e:
            self.logger.error(f"Error saving data to file: {e}")
    
    def get_latest_data(self, n: int = 1):
        """
        Get the latest n data points.
        
        Args:
            n: Number of latest data points to retrieve
            
        Returns:
            List of the latest n data points or DataFrame if n > 1
        """
        with self.lock:
            if n == 1 and self.recent_data:
                return self.recent_data[-1]
            elif n > 1:
                latest = self.recent_data[-n:] if len(self.recent_data) >= n else self.recent_data
                return pd.DataFrame(latest)
            return None