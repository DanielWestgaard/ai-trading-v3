# backtesting/strategies/model_based_strategy.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Tuple, Optional, Any

from backtesting.strategies.base_strategy import BaseStrategy
from backtesting.events import SignalEvent, SignalType

class ModelBasedStrategy(BaseStrategy):
    """Trading strategy based on ML model predictions."""
    
    def __init__(self, 
                 symbols: List[str],
                 model,
                 prediction_threshold: float = 0.55,
                 confidence_threshold: float = 0.0,
                 lookback_window: int = 1,
                 required_features: List[str] = None,
                 params: Dict[str, Any] = None):
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
        self.required_features = required_features or []
        
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
        signals = []
        # Synchronize position tracking with actual portfolio positions
        self.synchronize_positions(portfolio)
        
        for symbol, data in market_data.items():
            if symbol not in self.symbols:
                continue
            
            # Check if we have the required features
            missing_features = [f for f in self.required_features if f not in data.data]
            if missing_features:
                if not hasattr(self, 'missing_features_logged'):
                    logging.warning(f"Missing required features: {missing_features}")
                    self.missing_features_logged = True
                continue
            
            # Prepare features for the model
            features = {f: data.data[f] for f in self.required_features if f in data.data}
            
            # Skip if any feature is None or NaN
            if any(pd.isna(value) for value in features.values()):
                continue
            
            # Create feature array for prediction
            X = np.array([list(features.values())])
            
            # Make prediction
            try:
                # Get prediction probabilities
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
                
                # Get current position
                current_position = self.position_history.get(symbol, 0)
                
                # Get ACTUAL current position from portfolio instead of relying on self.position_history
                current_position = 0
                portfolio_position = portfolio.get_position(symbol)
                if portfolio_position:
                    current_position = 1 if portfolio_position.direction == "LONG" else -1
                
                # Generate signal based on prediction and confidence
                signal = None
                
                # Check if confidence exceeds threshold
                if confidence >= self.confidence_threshold:
                    # Long signal
                    if prediction == 1 and current_position <= 0:
                        signal_type = SignalType.BUY if current_position == 0 else SignalType.REVERSE
                        reason = f"Model prediction: UP (confidence: {confidence:.4f})"
                        signal = self.create_signal(
                            symbol=symbol,
                            signal_type=signal_type,
                            timestamp=data.timestamp,
                            reason=reason,
                            metadata={
                                'confidence': confidence,
                                'prediction': prediction
                            }
                        )
                    
                    # Short signal
                    elif prediction == -1 and current_position >= 0:
                        signal_type = SignalType.SELL if current_position == 0 else SignalType.REVERSE
                        reason = f"Model prediction: DOWN (confidence: {confidence:.4f})"
                        signal = self.create_signal(
                            symbol=symbol,
                            signal_type=signal_type,
                            timestamp=data.timestamp,
                            reason=reason,
                            metadata={
                                'confidence': confidence,
                                'prediction': prediction
                            }
                        )
                
                # Add this new code to exit positions when prediction changes
                # Exit long position when prediction becomes negative
                elif current_position > 0 and prediction == -1:
                    signal = self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        timestamp=data.timestamp,
                        reason=f"Exit long: prediction changed to DOWN (confidence: {confidence:.4f})",
                        metadata={
                            'confidence': confidence,
                            'prediction': prediction
                        }
                    )
                
                # Exit short position when prediction becomes positive
                elif current_position < 0 and prediction == 1:
                    signal = self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_SHORT,
                        timestamp=data.timestamp,
                        reason=f"Exit short: prediction changed to UP (confidence: {confidence:.4f})",
                        metadata={
                            'confidence': confidence,
                            'prediction': prediction
                        }
                    )
                
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
                logging.error(f"Error making prediction: {str(e)}")
        
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