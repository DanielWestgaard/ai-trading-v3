import json
import logging
import time
import numpy as np
import pandas as pd
from threading import Thread, Lock
from queue import Queue, Empty
from typing import Dict, List, Optional, Callable
from datetime import datetime


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
        self.bid_cache = {}  # Store bids by timestamp
        self.ask_cache = {}  # Store asks by timestamp
        self.output_file = output_file
        self.model = model
        self.strategy = strategy
        self.lock = Lock()  # To protect shared data
        self.is_running = False
        self.processor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Configure the logger
        self.logger.setLevel(logging.DEBUG)
    
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
            
            # Skip ping responses and subscription confirmations
            if data.get("destination") == "ping" or data.get("destination") == "OHLCMarketData.subscribe":
                return
                
            # Only process OHLC events
            if data.get("destination") != "ohlc.event":
                return
            
            payload = data.get("payload", {})
            price_type = payload.get("priceType")
            timestamp = payload.get("t")
            
            if not (price_type and timestamp):
                return
            
            self.logger.debug(f"Processing {price_type} message with timestamp {timestamp}")
                
            # Process based on message type
            if price_type == "bid":
                self._store_price_data(payload, "bid")
            elif price_type == "ask":
                self._store_price_data(payload, "ask")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def _store_price_data(self, data: Dict, price_type: str):
        """
        Store price data and attempt to match with opposite type.
        
        Args:
            data: The payload data
            price_type: Either "bid" or "ask"
        """
        timestamp = data.get("t")
        
        # Store this price data
        if price_type == "bid":
            self.bid_cache[timestamp] = data
            self.logger.debug(f"Stored bid data for timestamp {timestamp}")
        else:
            self.ask_cache[timestamp] = data
            self.logger.debug(f"Stored ask data for timestamp {timestamp}")
        
        # Try to match and create midpoint
        if price_type == "bid" and timestamp in self.ask_cache:
            self._create_midpoint(self.bid_cache[timestamp], self.ask_cache[timestamp])
            self.logger.debug(f"Matched bid with existing ask for timestamp {timestamp}")
            
            # Clean up after matching
            del self.bid_cache[timestamp]
            del self.ask_cache[timestamp]
            
        elif price_type == "ask" and timestamp in self.bid_cache:
            self._create_midpoint(self.bid_cache[timestamp], self.ask_cache[timestamp])
            self.logger.debug(f"Matched ask with existing bid for timestamp {timestamp}")
            
            # Clean up after matching
            del self.bid_cache[timestamp]
            del self.ask_cache[timestamp]
        
        # Clean up old unmatched data periodically
        if len(self.bid_cache) > 100 or len(self.ask_cache) > 100:
            self._cleanup_old_data()
    
    def _create_midpoint(self, bid_data: Dict, ask_data: Dict):
        """
        Create midpoint data from matching bid and ask, and add to processing queue.
        
        Args:
            bid_data: The bid price data
            ask_data: The ask price data
        """
        try:
            # Ensure bid and ask are for the same instrument and timeframe
            if bid_data["epic"] != ask_data["epic"] or bid_data["resolution"] != ask_data["resolution"]:
                self.logger.warning(f"Mismatched epic or resolution: {bid_data['epic']} vs {ask_data['epic']}")
                return
                
            # Calculate midpoint values
            midpoint = {
                "epic": bid_data["epic"],
                "resolution": bid_data["resolution"],
                "t": bid_data["t"],
                "datetime": datetime.fromtimestamp(bid_data["t"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "o": (bid_data["o"] + ask_data["o"]) / 2,
                "h": (bid_data["h"] + ask_data["h"]) / 2,
                "l": (bid_data["l"] + ask_data["l"]) / 2,
                "c": (bid_data["c"] + ask_data["c"]) / 2,
                "spread": ask_data["c"] - bid_data["c"]  # Capture spread information
            }
            
            self.logger.debug(f"Created midpoint for {midpoint['epic']} at {midpoint['datetime']}")
            
            # Add to queue for processing
            self.data_queue.put(midpoint)
            self.logger.debug(f"Added midpoint to queue. Queue size: ~{self.data_queue.qsize()}")
            
        except Exception as e:
            self.logger.error(f"Error creating midpoint: {e}")
    
    def _cleanup_old_data(self, max_age_seconds: int = 300):
        """
        Clean up old data from bid and ask caches.
        
        Args:
            max_age_seconds: Maximum age in seconds to keep unmatched data
        """
        current_time = int(time.time() * 1000)
        
        # Clean old bids
        old_bids = [ts for ts in self.bid_cache.keys() if current_time - ts > max_age_seconds * 1000]
        for ts in old_bids:
            del self.bid_cache[ts]
            
        # Clean old asks
        old_asks = [ts for ts in self.ask_cache.keys() if current_time - ts > max_age_seconds * 1000]
        for ts in old_asks:
            del self.ask_cache[ts]
            
        self.logger.debug(f"Cleaned up {len(old_bids)} old bids and {len(old_asks)} old asks")
    
    def _data_processor(self):
        """Consumer thread that processes the midpoint data and feeds to model."""
        self.logger.info("Data processor thread started")
        
        while self.is_running:
            try:
                # Get data from queue (with timeout to check is_running periodically)
                try:
                    midpoint_data = self.data_queue.get(timeout=1.0)
                    self.logger.debug(f"Got midpoint from queue: {midpoint_data['datetime']}")
                except Empty:
                    continue
                
                with self.lock:
                    # Add to in-memory buffer
                    self.recent_data.append(midpoint_data)
                    self.logger.debug(f"Added midpoint to recent_data. Size: {len(self.recent_data)}")
                    
                    # If buffer grows too large, trim it
                    if len(self.recent_data) > 1000:
                        self.recent_data = self.recent_data[-1000:]
                
                # Save to file every 10 data points (instead of 100)
                if len(self.recent_data) % 10 == 0:
                    self._save_to_file()
                
                # Feed to model for prediction if model exists and we have enough data
                if self.model and len(self.recent_data) >= 10:  # Need sufficient history
                    try:
                        # Convert the recent data to the format expected by your model
                        model_input = self._prepare_model_input()
                        
                        if model_input is not None and not model_input.empty:
                            self.logger.debug(f"Running prediction on data with shape {model_input.shape}")
                            
                            # Log column types to help debug type issues
                            self.logger.debug(f"Column dtypes: {model_input.dtypes}")
                            
                            # Ensure we have purely numeric data for XGBoost
                            for col in model_input.columns:
                                if not pd.api.types.is_numeric_dtype(model_input[col]):
                                    self.logger.warning(f"Column {col} is not numeric: {model_input[col].dtype}")
                                    model_input[col] = pd.to_numeric(model_input[col], errors='coerce')
                            
                            # Replace any NaN or infinite values
                            model_input.replace([np.inf, -np.inf], np.nan, inplace=True)
                            model_input.fillna(0, inplace=True)
                            
                            # Run prediction
                            prediction = self.model.predict(model_input)
                            self.logger.info(f"Model prediction: {prediction}")
                            
                            # Execute trade if strategy exists and prediction meets criteria
                            if self.strategy and prediction is not None:
                                self.strategy.evaluate_signal(prediction)
                        else:
                            self.logger.warning("Prepared model input is None or empty")
                    except Exception as e:
                        self.logger.error(f"Error in model prediction or strategy execution: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in data processor: {e}")
    
    def _prepare_model_input(self):
        """
        Prepare recent data for model consumption.
        Formats data to match XGBoost's expected input format.
        """
        with self.lock:
            if not self.recent_data or len(self.recent_data) < 5:  # Need enough data points
                self.logger.warning("Not enough data points for model input")
                return None
                
            # Convert list of dictionaries to pandas DataFrame
            df = pd.DataFrame(self.recent_data)
            
            # Keep only numeric columns needed for prediction
            numeric_cols = ['o', 'h', 'l', 'c', 'spread']
            
            # Ensure all required columns exist
            if not all(col in df.columns for col in numeric_cols):
                self.logger.warning(f"Missing required columns. Available: {df.columns.tolist()}")
                return None
                
            # Filter to only numeric columns
            df_features = df[numeric_cols].copy()
            
            # Convert timestamp to datetime for time-based features
            if 't' in df.columns:
                df_features['hour'] = pd.to_datetime(df['t'], unit='ms').dt.hour
                df_features['minute'] = pd.to_datetime(df['t'], unit='ms').dt.minute
                df_features['day_of_week'] = pd.to_datetime(df['t'], unit='ms').dt.dayofweek
            
            # Add derived features if needed
            df_features['hl_diff'] = df_features['h'] - df_features['l']
            df_features['oc_diff'] = df_features['c'] - df_features['o']
            
            # Calculate percentage changes (returns) for relevant columns
            for col in ['o', 'h', 'l', 'c']:
                df_features[f'{col}_pct_change'] = df_features[col].pct_change()
            
            # Drop the first row which will have NaN from pct_change
            df_features = df_features.iloc[1:].copy()
            
            # Fill any remaining NaN values
            df_features.fillna(0, inplace=True)
            
            # Ensure all columns are numeric
            for col in df_features.columns:
                if df_features[col].dtype == 'object':
                    self.logger.warning(f"Converting column {col} from {df_features[col].dtype} to numeric")
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    
            # Fill any NaN values created by the conversion
            df_features.fillna(0, inplace=True)
            
            self.logger.debug(f"Prepared model input with shape {df_features.shape} and columns {df_features.columns.tolist()}")
            
            return df_features
    
    def _save_to_file(self):
        """Save recent data to disk."""
        try:
            with self.lock:
                if not self.recent_data:
                    self.logger.warning("No data to save")
                    return
                    
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
            Latest data point or DataFrame of latest points
        """
        with self.lock:
            self.logger.debug(f"Getting latest data. recent_data size: {len(self.recent_data)}")
            if not self.recent_data:
                self.logger.warning("No data in recent_data buffer")
                return None
                
            if n == 1:
                return self.recent_data[-1]
            else:
                latest = self.recent_data[-n:] if len(self.recent_data) >= n else self.recent_data
                return pd.DataFrame(latest)