# backtesting/strategies/model_based_strategy.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Tuple, Optional, Any

from core.risk.position_management import PositionManager
from backtesting.signal_filter import SignalFilter
from core.strategies.base_strategy import BaseStrategy
from core.events import SignalEvent, SignalType
from backtesting.timeframe_resampler import TimeframeResampler

class ModelBasedStrategy(BaseStrategy):
    """Trading strategy based on ML model predictions."""
    
    def __init__(self, 
                symbols: List[str],
                model,
                prediction_threshold: float = 0.55,
                confidence_threshold: float = 0.0,
                lookback_window: int = 1,
                required_features: List[str] = None,
                params: Dict[str, Any] = None,
                **kwargs):
        """
        Initialize the model-based strategy.
        
        Args:
            symbols: List of symbols to trade
            model: Trained model instance
            prediction_threshold: Threshold for prediction to generate signal (classification)
            confidence_threshold: Minimum confidence needed for a signal
            lookback_window: Number of past predictions to consider
            required_features: Features required by the model
            params: Additional strategy parameters
        """
        self.model = model
        self.prediction_threshold = prediction_threshold
        self.confidence_threshold = confidence_threshold
        self.lookback_window = lookback_window
        
        # Get required features from model if not provided
        if required_features is None and hasattr(model, 'features'):
            self.required_features = model.features
        else:
            self.required_features = required_features or []
        
        # Log the required features
        if self.required_features:
            logging.info(f"Model requires {len(self.required_features)} features")
            logging.debug(f"Model features (first 10): {self.required_features[:10]}")
        
        # Initialize our new modules
        self.signal_filter = SignalFilter(
            lookback_window=kwargs.get('lookback_window', 10),
            consensus_threshold=kwargs.get('consensus_threshold', 0.7)
        )
        
        self.position_manager = PositionManager(
            min_hold_bars=kwargs.get('min_hold_bars', 12),  # 1 hour minimum
            max_hold_bars=kwargs.get('max_hold_bars', 288)  # 1 day maximum
        )
        
        self.resampler = TimeframeResampler(
            base_timeframe=1,  # 1-minute data
            target_timeframe=kwargs.get('decision_timeframe', 5)  # 5 for 5 min decisions, 60 for 1-hour decisions 
        )
        
        # We'll initialize the feature generator in generate_signals when needed
        # This allows us to lazy-load it only when necessary
        
        # Initialize with base strategy
        super().__init__(symbols, params)
        
        # Store predictions and positions
        self.predictions = {symbol: [] for symbol in symbols}
        self.prediction_history = {symbol: [] for symbol in symbols}
        self.position_history = {symbol: 0 for symbol in symbols}
    
    def initialize(self):
        """Initialize the strategy."""
        logging.info("Initializing model-based strategy")
        
        # Check if model is loaded
        if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted:
            logging.warning("Model is not fitted. Strategy may not generate signals.")
    
    def on_backtest_start(self):
        """
        Called at the start of a backtest.
        """
        logging.info("Backtest starting with model-based strategy")
        
        # Reset prediction history
        self.predictions = {symbol: [] for symbol in self.symbols}
        self.prediction_history = {symbol: [] for symbol in self.symbols}
        self.position_history = {symbol: 0 for symbol in self.symbols}
    
    def generate_signals(self, market_data, portfolio):
        """
        Generate trading signals based on model predictions.
        
        Args:
            market_data: Dictionary mapping symbols to market data
            portfolio: Portfolio instance
            
        Returns:
            List of signal events
        """
        import pandas as pd
        import numpy as np
        from data.features.feature_generator import FeatureGenerator
        
        signals = []
        logging.debug(f"generate_signals called with {len(market_data)} symbols")
        
        # Synchronize position tracking with actual portfolio positions
        self.synchronize_positions(portfolio)
        
        # Initialize feature generator if needed (first time only)
        if not hasattr(self, 'feature_generator'):
            self.feature_generator = FeatureGenerator()
            logging.info("Initialized feature generator for strategy")
        
        for symbol, data in market_data.items():
            logging.debug(f"Processing symbol {symbol} with timestamp {data.timestamp}")
            if symbol not in self.symbols:
                continue
            
            # Update position tracking
            self.position_manager.on_bar(symbol, data.timestamp)
            
            # Add data to resampler
            self.resampler.add_bar(symbol, data.timestamp, data.data)
            
            # Only make decisions at the higher timeframe boundaries
            should_decide = self.resampler.should_make_decision(data.timestamp)
            logging.debug(f"Resampler decision: {should_decide}")
            
            if not should_decide:
                continue
            
            try:
                # Extract the basic OHLC data
                ohlc_data = {}
                
                # Try to get OHLC values with various possible naming conventions
                for base_field, variations in {
                    'open': ['open', 'Open', 'open_raw', 'Open_raw'],
                    'high': ['high', 'High', 'high_raw', 'High_raw'],
                    'low': ['low', 'Low', 'low_raw', 'Low_raw'],
                    'close': ['close', 'Close', 'close_raw', 'Close_raw']
                }.items():
                    for field in variations:
                        if field in data.data:
                            ohlc_data[base_field] = data.data[field]
                            break
                
                # Check if we have all required OHLC fields
                if len(ohlc_data) < 4:
                    logging.warning(f"Missing basic OHLC data for {symbol}. Found: {list(ohlc_data.keys())}")
                    continue
                
                # Create a small dataframe with this candle for feature generation
                candle_df = pd.DataFrame({
                    'Date': [data.timestamp],
                    'Open': [ohlc_data['open']],
                    'High': [ohlc_data['high']],
                    'Low': [ohlc_data['low']],
                    'Close': [ohlc_data['close']],
                    'Volume': [data.data.get('volume', 0)]
                })
                
                # Generate features directly
                try:
                    featured_df = self.feature_generator.transform(candle_df)
                    logging.debug(f"Feature generation successful. Created {len(featured_df.columns)} features.")
                    
                    # Check if we have enough features for the model
                    if len(featured_df.columns) < 20:  # Arbitrary threshold
                        logging.warning(f"Not enough features generated: {len(featured_df.columns)}. Expected more than 20.")
                        continue
                    
                    # Extract features for the model
                    feature_dict = {}
                    
                    # Ensure all required features are present
                    for feature in self.required_features:
                        # Try to find the feature in the dataframe (case-insensitive)
                        found = False
                        for col in featured_df.columns:
                            if col.lower() == feature.lower():
                                feature_dict[feature] = featured_df[col].iloc[0]
                                found = True
                                break
                        
                        # If not found, add a default value
                        if not found:
                            feature_dict[feature] = 0.0
                    
                    # Create feature array for prediction
                    X = np.array([list(feature_dict.values())])
                    
                    # Make prediction
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(X)
                        confidence = proba[0, 1]  # Probability of positive class
                    else:
                        # For models without predict_proba
                        pred = self.model.predict(X)[0]
                        confidence = abs(pred)
                    
                    # Store prediction and confidence
                    prediction = 1 if confidence > self.prediction_threshold else -1
                    self.predictions[symbol].append((data.timestamp, prediction, confidence))
                    
                    # Keep only the last N predictions
                    if len(self.predictions[symbol]) > self.lookback_window:
                        self.predictions[symbol] = self.predictions[symbol][-self.lookback_window:]
                        
                    # Add prediction to filter
                    self.signal_filter.add_prediction(symbol, data.timestamp, prediction, confidence)
                    
                    # Log prediction
                    logging.info(f"Model prediction for {symbol}: {prediction} with confidence {confidence:.4f}")
                    
                    # Get ACTUAL current position from portfolio
                    current_position = 0
                    portfolio_position = portfolio.get_position(symbol)
                    if portfolio_position:
                        current_position = 1 if portfolio_position.direction == "LONG" else -1
                    
                    # Generate signal based on prediction and confidence
                    signal = None
                    
                    # No position - check for entry
                    if current_position == 0:
                        should_buy = self.signal_filter.should_generate_signal(symbol, 1)
                        should_sell = self.signal_filter.should_generate_signal(symbol, -1)
                        
                        if should_buy:  # Bullish consensus
                            logging.info(f"SIGNAL GENERATION: BUY conditions met for {symbol}")
                            signal = self.create_signal(symbol, SignalType.BUY, data.timestamp)
                            self.position_manager.open_position(symbol, data.timestamp, 1)
                        elif should_sell:  # Bearish consensus
                            logging.info(f"SIGNAL GENERATION: SELL conditions met for {symbol}")
                            signal = self.create_signal(symbol, SignalType.SELL, data.timestamp)
                            self.position_manager.open_position(symbol, data.timestamp, -1)
                    
                    # Long position - check for exit
                    elif current_position > 0:
                        if (self.position_manager.can_exit(symbol) and 
                            self.signal_filter.should_generate_signal(symbol, -1)):
                            logging.info(f"SIGNAL GENERATION: EXIT LONG conditions met for {symbol}")
                            signal = self.create_signal(symbol, SignalType.EXIT_LONG, data.timestamp)
                            self.position_manager.close_position(symbol, data.timestamp)
                        elif self.position_manager.should_exit(symbol):  # Force exit after max hold time
                            logging.info(f"SIGNAL GENERATION: Force EXIT LONG (max hold time) conditions met for {symbol}")
                            signal = self.create_signal(symbol, SignalType.EXIT_LONG, data.timestamp)
                            self.position_manager.close_position(symbol, data.timestamp)
                    
                    # Short position - check for exit
                    elif current_position < 0:
                        if (self.position_manager.can_exit(symbol) and 
                            self.signal_filter.should_generate_signal(symbol, 1)):
                            logging.info(f"SIGNAL GENERATION: EXIT SHORT conditions met for {symbol}")
                            signal = self.create_signal(symbol, SignalType.EXIT_SHORT, data.timestamp)
                            self.position_manager.close_position(symbol, data.timestamp)
                        elif self.position_manager.should_exit(symbol):  # Force exit after max hold time
                            logging.info(f"SIGNAL GENERATION: Force EXIT SHORT (max hold time) conditions met for {symbol}")
                            signal = self.create_signal(symbol, SignalType.EXIT_SHORT, data.timestamp)
                            self.position_manager.close_position(symbol, data.timestamp)
                    
                    # Add signal to list if generated
                    if signal:
                        signals.append(signal)
                        logging.info(f"Generated signal: {signal}")
                        
                    # Record prediction history
                    self.prediction_history[symbol].append({
                        'timestamp': data.timestamp,
                        'prediction': prediction,
                        'confidence': confidence,
                        'signal_generated': signal is not None,
                        'position': self.position_history[symbol]
                    })
                    
                except Exception as e:
                    logging.error(f"Error in feature generation or prediction: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    
            except Exception as e:
                logging.error(f"Error processing symbol {symbol}: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        return signals

    def on_backtest_end(self):
        """
        Called at the end of a backtest.
        """
        logging.info("Backtest completed with model-based strategy")
        
        # Calculate prediction accuracy if possible
        for symbol in self.symbols:
            if len(self.prediction_history[symbol]) > 0:
                df = pd.DataFrame(self.prediction_history[symbol])
                
                # Calculate basic metrics
                total_predictions = len(df)
                signals_generated = df['signal_generated'].sum()
                
                logging.info(f"Symbol: {symbol} - Total predictions: {total_predictions}, Signals generated: {signals_generated}")
    
    def get_parameters(self):
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of parameters
        """
        params = super().get_parameters()
        
        # Add model-specific parameters
        params.update({
            'model_type': self.model.__class__.__name__,
            'prediction_threshold': self.prediction_threshold,
            'confidence_threshold': self.confidence_threshold,
            'lookback_window': self.lookback_window,
            'required_features': self.required_features
        })
        
        return params
    
    def get_prediction_history(self, symbol=None):
        """
        Get prediction history.
        
        Args:
            symbol: Symbol to get history for (if None, return all)
            
        Returns:
            DataFrame of prediction history
        """
        if symbol:
            if symbol not in self.prediction_history:
                return pd.DataFrame()
            return pd.DataFrame(self.prediction_history[symbol])
        
        # Combine all symbols
        all_history = []
        for sym, history in self.prediction_history.items():
            df = pd.DataFrame(history)
            if len(df) > 0:
                df['symbol'] = sym
                all_history.append(df)
        
        if not all_history:
            return pd.DataFrame()
        
        return pd.concat(all_history, ignore_index=True)
    
    def synchronize_positions(self, portfolio):
        """Synchronize strategy position tracking with actual portfolio positions."""
        for symbol in self.symbols:
            position = portfolio.get_position(symbol)
            if position is None:
                self.position_history[symbol] = 0
            else:
                # If position direction is LONG, set to 1, if SHORT, set to -1
                self.position_history[symbol] = 1 if position.direction == "LONG" else -1