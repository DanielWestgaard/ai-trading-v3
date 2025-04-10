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
    
    Includes special handling for forex data without volume and limited data points.
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
        
        # Create a custom preparator for live data
        self.feature_preparator = feature_preparator or self._create_live_data_preparator()
        
        self.data_normalizer = data_normalizer or DataNormalizer(other_method='zscore')
        
        # Expected features for the model
        self.model_features = model_features
        
        # Data processing state
        self.startup_mode = True
        self.min_data_points = 15  # Minimum data points needed before generating features
        self.initialized_processing = False
        self.have_run_prediction = False  # Track if we've successfully run a prediction
        
        # Missing feature handling
        self.add_synthetic_volume = True
        self.synthetic_fields = {}  # Store synthetic field generation functions
        
        # Configure the logger
        # self.logger.setLevel(logging.INFO)
    
    def _create_live_data_preparator(self):
        """Create a feature preparator with settings optimized for live data."""
        return FeaturePreparator(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
            preserve_original_prices=True,
            price_transform_method='returns',
            treatment_mode='hybrid',  # Use hybrid mode which is more flexible
            trim_initial_periods=5,   # Smaller trim period for live data
            min_data_points=20        # Much lower threshold for live data
        )
    
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
                "datetime": datetime.fromtimestamp(bid_data["t"] / 1000),  # Store as actual datetime object, not string
                "open": (bid_data["o"] + ask_data["o"]) / 2,
                "high": (bid_data["h"] + ask_data["h"]) / 2,
                "low": (bid_data["l"] + ask_data["l"]) / 2,
                "close": (bid_data["c"] + ask_data["c"]) / 2,
                "spread": ask_data["c"] - bid_data["c"]  # Capture spread information
            }
            
            # Add synthetic volume if needed
            if self.add_synthetic_volume:
                # Generate volume based on price movement and spread
                high_low_range = midpoint["high"] - midpoint["low"]
                midpoint["volume"] = int(high_low_range * 10000000)  # Scale to reasonable volume for forex
            
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
                            
                            # Log column types to help debug type issues
                            self.logger.debug(f"Column dtypes: {model_input.dtypes}")
                            
                            # Run prediction
                            prediction = self.model.predict(model_input)
                            self.logger.info(f"Model prediction: {prediction}")
                            self.have_run_prediction = True
                            
                            # Execute trade if strategy exists and prediction meets criteria
                            if self.strategy and prediction is not None:
                                self.strategy.evaluate_signal(prediction)
                        else:
                            if not self.have_run_prediction:
                                # Only log this warning if we haven't successfully run a prediction yet
                                self.logger.warning("Prepared model input is None or empty - still working on data processing")
                    except Exception as e:
                        self.logger.error(f"Error in model prediction or strategy execution: {e}")
                        self.logger.error(f"Model input shape: {model_input.shape}")
                        self.logger.error(f"Model input columns: {model_input.columns.tolist()}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in data processor: {e}")
    
    def _prepare_model_input(self):
        """
        Prepare recent data for model consumption, ensuring exact feature name matching.
        Fixes the feature_names mismatch error by creating all expected columns.
        
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
        
        # Generate any missing fields needed by the model
        df = self._add_missing_fields(df)
        
        # Apply feature generation
        try:
            # Generate technical features
            df_with_features = self.feature_generator.transform(df)
            self.logger.debug(f"Generated features. New shape: {df_with_features.shape}")
            
            # Custom preprocessing for live data
            df_prepared = self._simplified_feature_preparation(df_with_features)
            
            # CRITICAL FIX: Add missing '_raw' and '_original' columns that the model expects
            # First, check what's missing
            if self.model_features:
                # Find which expected features are missing from our processed data
                missing_features = set(self.model_features) - set(df_prepared.columns)
                
                if missing_features:
                    self.logger.debug(f"Adding {len(missing_features)} missing expected features: {missing_features}")
                    
                    # Process the missing features by category
                    for feature in missing_features:
                        # Handle *_raw features - these should be the unmodified values
                        if feature.endswith('_raw'):
                            base_feature = feature.replace('_raw', '')
                            if base_feature.lower() in df_prepared.columns:
                                # Use the lowercase version
                                df_prepared[feature] = df_prepared[base_feature.lower()].copy()
                            elif base_feature in df_prepared.columns:
                                # Use the exact case version
                                df_prepared[feature] = df_prepared[base_feature].copy()
                            else:
                                # If we can't find the base feature, look for its capitalized version
                                capitalized = base_feature.capitalize()
                                if capitalized in df_prepared.columns:
                                    df_prepared[feature] = df_prepared[capitalized].copy()
                                else:
                                    # Last resort - just use a reasonable value
                                    self.logger.warning(f"Could not find appropriate value for {feature}, using zeros")
                                    df_prepared[feature] = 0.0
                        
                        # Handle *_original features - these should be the unmodified values
                        elif feature.endswith('_original'):
                            base_feature = feature.replace('_original', '')
                            if base_feature.lower() in df_prepared.columns:
                                # Use the lowercase version
                                df_prepared[feature] = df_prepared[base_feature.lower()].copy()
                            elif base_feature in df_prepared.columns:
                                # Use the exact case version
                                df_prepared[feature] = df_prepared[base_feature].copy()
                            else:
                                # If we can't find the base feature, look for its capitalized version
                                capitalized = base_feature.capitalize()
                                if capitalized in df_prepared.columns:
                                    df_prepared[feature] = df_prepared[capitalized].copy()
                                else:
                                    # Last resort - just use a reasonable value
                                    self.logger.warning(f"Could not find appropriate value for {feature}, using zeros")
                                    df_prepared[feature] = 0.0
                        
                        # Handle simple missing features
                        elif feature == 'volume' and 'volume' not in df_prepared.columns:
                            # Create synthetic volume if needed
                            high_low_diff = df_prepared['high'] - df_prepared['low'] if 'high' in df_prepared.columns and 'low' in df_prepared.columns else 0.01
                            df_prepared['volume'] = (high_low_diff * 1000000).astype(int)
                        
                        # Any other missing features
                        else:
                            # For other features, just add zeros - not ideal but better than failing
                            self.logger.warning(f"Adding zeros for missing feature: {feature}")
                            df_prepared[feature] = 0.0
            
            # If we have model features, extract the required ones in the EXACT order
            if self.model_features is not None:
                # Check which expected features are actually available
                available_features = [f for f in self.model_features if f in df_prepared.columns]
                
                # Log any missing features
                missing = set(self.model_features) - set(available_features)
                if missing:
                    self.logger.warning(f"Missing {len(missing)} expected features: {missing}")
                
                # CRITICAL: Create DataFrame with EXACT columns and order to match model expectations
                if len(self.model_features) > 0:
                    # Initialize DataFrame with all expected features as zeros
                    model_input = pd.DataFrame(0, index=range(len(df_prepared)), columns=self.model_features)
                    
                    # Fill with actual values where available
                    for feature in self.model_features:
                        if feature in df_prepared.columns:
                            model_input[feature] = df_prepared[feature].values
                else:
                    # Fallback if model_features is empty
                    model_input = df_prepared
            else:
                # No specific features provided, use all numeric columns
                model_input = df_prepared.select_dtypes(include=['number'])
            
            # Safety check - ensure all columns are numeric
            for col in model_input.columns:
                if not pd.api.types.is_numeric_dtype(model_input[col]):
                    model_input[col] = pd.to_numeric(model_input[col], errors='coerce')
                    
            # Replace NaN values with 0
            model_input = model_input.fillna(0)
            
            # Remove infinity values
            model_input = model_input.replace([np.inf, -np.inf], 0)
            
            # Log column types to help debug
            self.logger.debug(f"Column dtypes: {model_input.dtypes}")
            
            if not model_input.empty:
                self.logger.debug(f"Final feature set for model: {model_input.shape[1]} features, {model_input.shape[0]} rows")
            
            return model_input
            
        except Exception as e:
            self.logger.error(f"Error preparing model input: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def _add_missing_fields(self, df):
        """
        Add any missing fields that might be needed by the model.
        
        Args:
            df: DataFrame to enhance
            
        Returns:
            Enhanced DataFrame
        """
        # Add volume if missing but expected
        if 'Volume' not in df.columns and 'volume' not in df.columns and 'volume' in self.model_features:
            self.logger.info("Adding synthetic volume data")
            
            # Calculate synthetic volume based on price range
            high_low_diff = df['High'] - df['Low'] if 'High' in df.columns else df['high'] - df['low']
            df['Volume'] = (high_low_diff * 1000000).astype(int)
            
            # Add some randomness to make it look more realistic
            import random
            random_factor = np.array([random.uniform(0.8, 1.2) for _ in range(len(df))])
            df['Volume'] = (df['Volume'] * random_factor).astype(int)
        
        return df
    
    def _simplified_feature_preparation(self, df):
        """
        A simplified version of feature preparation designed for live data.
        
        This avoids the complex logic of the feature preparator that can remove too much data.
        
        Args:
            df: DataFrame with generated features
            
        Returns:
            Prepared DataFrame
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # 1. Calculate returns for price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in result.columns:
                result[f'{col.lower()}_return'] = result[col].pct_change()
        
        # 2. Create return for main close price
        if 'Close' in result.columns:
            result['close_return'] = result['Close'].pct_change()
        
        # 3. Handle NaN values from calculations
        result = result.fillna(method='bfill').fillna(method='ffill')
        
        # 4. Simple Z-score normalization for technical indicators
        numeric_cols = result.select_dtypes(include=['number']).columns
        technical_indicators = [col for col in numeric_cols if col not in price_cols + 
                              ['Date', 'timestamp', 't', 'Volume', 'spread']]
        
        for col in technical_indicators:
            # Skip if already a return
            if 'return' in col:
                continue
                
            # Apply z-score normalization
            mean = result[col].mean()
            std = result[col].std()
            if std > 0:
                result[col] = (result[col] - mean) / std
        
        return result
    
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
            
            # Make directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
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
    
    def warm_up_with_historical_data(self, historical_data):
        """
        Initialize the data handler with historical data before starting live processing.
        
        Args:
            historical_data: List of OHLCV dictionaries with historical market data
        """
        logging.info(f"Warming up data handler with {len(historical_data)} historical data points")
        
        # Convert historical data to the expected format
        for data_point in historical_data:
            # Ensure we have the datetime as an actual datetime object
            if 'timestamp' in data_point and 'datetime' not in data_point:
                data_point['datetime'] = datetime.fromtimestamp(data_point['timestamp'] / 1000)
            elif 't' in data_point and 'datetime' not in data_point:
                data_point['datetime'] = datetime.fromtimestamp(data_point['t'] / 1000)
            
            # Add to recent data
            self.recent_data.append(data_point)
        
        # Save the initial data to the output file
        self._save_to_file()
        
        logging.info(f"Data handler warmed up with {len(self.recent_data)} data points")