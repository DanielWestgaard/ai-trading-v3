import json
import logging
import time
import pandas as pd
import numpy as np
from threading import Thread, Lock
from queue import Queue, Empty
from typing import Dict, List, Optional, Callable
from datetime import datetime

from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer


class LiveDataHandler:
    """
    Handles processing of live market data from WebSocket connections.
    Processes bid/ask pairs into midpoint prices and prepares data for model prediction.
    """
    
    def __init__(self, model=None, strategy=None, output_file='live/temp_live/live_market_data.csv',
                 feature_generator=None, feature_preparator=None, data_normalizer=None,
                 model_features=None):
        """
        Initialize the LiveDataHandler.
        
        Args:
            model: The prediction model (optional)
            strategy: Trading strategy to execute trades (optional)
            output_file: CSV file to periodically save data to
            feature_generator: FeatureGenerator for technical indicators
            feature_preparator: FeaturePreparator for feature handling
            data_normalizer: DataNormalizer for normalizing features
            model_features: List of features expected by the model
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
        
        # Data processing components
        self.feature_generator = feature_generator or FeatureGenerator()
        self.feature_preparator = feature_preparator or FeaturePreparator(
            price_transform_method='returns', treatment_mode='basic')
        self.data_normalizer = data_normalizer or DataNormalizer(other_method='zscore')
        
        # Expected features for the model
        self.model_features = model_features
        
        # Data processing state
        self.min_data_points = 25  # Minimum data points needed before generating features
        self.initialized_processing = False
        
        # Configure the logger
        # self.logger.setLevel(logging.INFO)
    
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
                "open": (bid_data["o"] + ask_data["o"]) / 2,
                "high": (bid_data["h"] + ask_data["h"]) / 2,
                "low": (bid_data["l"] + ask_data["l"]) / 2,
                "close": (bid_data["c"] + ask_data["c"]) / 2,
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
                if self.model and len(self.recent_data) >= self.min_data_points:
                    try:
                        # Convert the recent data to the format expected by your model
                        model_input = self._prepare_model_input()
                        
                        if model_input is not None and not model_input.empty:
                            self.logger.debug(f"Running prediction on data with shape {model_input.shape}")
                            
                            # Run prediction
                            prediction = self.model.predict(model_input)
                            self.logger.info(f"Model prediction: {prediction}")
                            
                            # Execute trade if strategy exists and prediction meets criteria
                            if self.strategy and prediction is not None:
                                self.strategy.evaluate_signal(prediction)
                        else:
                            self.logger.debug("Prepared model input is None or empty")
                    except Exception as e:
                        self.logger.error(f"Error in model prediction or strategy execution: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in data processor: {e}")
    
    def _prepare_model_input(self):
        """
        Prepare recent data for model consumption using the data pipeline.
        
        Returns:
            DataFrame with properly formatted features for model prediction
        """
        with self.lock:
            # Ensure we have enough data
            if len(self.recent_data) < self.min_data_points:
                self.logger.warning(f"Not enough data points for feature generation. Need at least {self.min_data_points}")
                return None
                
            # Convert list of dictionaries to pandas DataFrame
            df = pd.DataFrame(self.recent_data)
            
            # Rename columns to match expected data pipeline format
            column_mapping = {
                'datetime': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                't': 'timestamp'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Make sure Date is datetime type
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['datetime'])
                
            # Initialize data processing components if needed
            if not self.initialized_processing:
                self.logger.info("Initializing data processing components")
                # Fit the feature generator and preparator on initial data
                self.feature_generator.fit(df)
                
                # Don't fit the preparator and normalizer yet since we need feature generation first
                self.initialized_processing = True
        
        # Apply feature generation
        try:
            # Generate technical features
            df_with_features = self.feature_generator.transform(df)
            self.logger.debug(f"Generated features. New shape: {df_with_features.shape}")
            
            # Prepare features (transform prices, handle NaNs, etc.)
            if not hasattr(self.feature_preparator, '_stats') or not self.feature_preparator._stats:
                # First time, need to fit
                self.feature_preparator.fit(df_with_features)
                
            df_prepared = self.feature_preparator.transform(df_with_features)
            self.logger.debug(f"Prepared features. New shape: {df_prepared.shape}")
            
            # Normalize the data
            if not hasattr(self.data_normalizer, '_params') or not self.data_normalizer._params:
                # First time, need to fit
                self.data_normalizer.fit(df_prepared)
                
            df_normalized = self.data_normalizer.transform(df_prepared)
            self.logger.debug(f"Normalized features. Final shape: {df_normalized.shape}")
            
            # Select only the features expected by the model (if specified)
            if self.model_features is not None:
                # Find which expected features are actually in our data
                available_features = [f for f in self.model_features if f in df_normalized.columns]
                
                if not available_features:
                    self.logger.error(f"None of the expected model features are available in processed data")
                    self.logger.debug(f"Available columns: {df_normalized.columns.tolist()}")
                    self.logger.debug(f"Expected features: {self.model_features}")
                    return None
                    
                if len(available_features) < len(self.model_features):
                    missing = set(self.model_features) - set(available_features)
                    self.logger.warning(f"Missing {len(missing)} expected features: {missing}")
                
                df_features = df_normalized[available_features].copy()
            else:
                # No specific features provided, use all numeric columns
                df_features = df_normalized.select_dtypes(include=['number'])
            
            # Safety check - ensure all columns are numeric
            for col in df_features.columns:
                if df_features[col].dtype == 'object':
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                    
            # Replace NaN values with 0
            df_features = df_features.fillna(0)
            
            # Remove infinity values
            df_features = df_features.replace([np.inf, -np.inf], 0)
            
            self.logger.debug(f"Final feature set for model: {df_features.shape[1]} features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error preparing model input: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _save_to_file(self):
        """Save recent data to disk."""
        try:
            with self.lock:
                if not self.recent_data:
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