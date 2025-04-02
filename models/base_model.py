# models/base_model.py
from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional, Any

class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, 
                 name: str,
                 features: List[str] = None,
                 target: str = 'close_return',
                 prediction_type: str = 'classification',
                 lookback_periods: int = 10,
                 prediction_horizon: int = 1,
                 model_dir: str = 'model_storage'):
        """
        Initialize the base model.
        
        Args:
            name: Model name/identifier
            features: List of feature columns to use
            target: Target column name to predict
            prediction_type: 'classification' or 'regression'
            lookback_periods: Number of periods to look back for features
            prediction_horizon: Number of periods ahead to predict
            model_dir: Directory to save/load model files
        """
        self.name = name
        self.features = features
        self.target = target
        self.prediction_type = prediction_type
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self.model_dir = model_dir
        self.model = None
        self.feature_importance = None
        self.is_fitted = False
        self.metadata = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance': {}
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Generate predictions.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X, **kwargs):
        """
        Generate probability predictions (for classification models).
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Probability predictions
        """
        pass
    
    def predict_direction(self, X, threshold=0.5, **kwargs):
        """
        Predict market direction (up/down).
        
        Args:
            X: Input features
            threshold: Decision threshold for classification
            **kwargs: Additional prediction parameters
            
        Returns:
            Direction predictions (1 for up, -1 for down)
        """
        if self.prediction_type == 'classification':
            probs = self.predict_proba(X, **kwargs)
            return np.where(probs[:, 1] > threshold, 1, -1)
        else:
            preds = self.predict(X, **kwargs)
            return np.where(preds > 0, 1, -1)
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping feature names to importance values
        """
        if self.feature_importance is None:
            self.logger.warning("Feature importance not available")
            return {}
        
        if isinstance(self.feature_importance, pd.Series):
            return self.feature_importance.to_dict()
        
        feature_names = self.features if self.features else [f"feature_{i}" for i in range(len(self.feature_importance))]
        return dict(zip(feature_names, self.feature_importance))
    
    def save(self, filepath=None):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model (default: auto-generated)
            
        Returns:
            Path where the model was saved
        """
        if not self.is_fitted:
            self.logger.warning("Cannot save unfitted model")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_dir, f"{self.name}_{timestamp}.pkl")
        
        # Update metadata
        self.metadata['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model as pickle file
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'features': self.features,
                'target': self.target,
                'prediction_type': self.prediction_type,
                'lookback_periods': self.lookback_periods,
                'prediction_horizon': self.prediction_horizon,
                'feature_importance': self.feature_importance,
                'metadata': self.metadata
            }, f)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.features = data['features']
            self.target = data['target']
            self.prediction_type = data['prediction_type']
            self.lookback_periods = data['lookback_periods']
            self.prediction_horizon = data['prediction_horizon']
            self.feature_importance = data['feature_importance']
            self.metadata = data['metadata']
            self.is_fitted = True
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_fitted:
            self.logger.warning("Cannot evaluate unfitted model")
            return {}
        
        # Get predictions
        if self.prediction_type == 'classification':
            y_pred = self.predict(X_test)
            y_prob = self.predict_proba(X_test)
            
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_prob[:, 1])
            }
        else:
            y_pred = self.predict(X_test)
            
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Add directional accuracy for regression
            direction_pred = np.sign(y_pred)
            direction_true = np.sign(y_test)
            metrics['direction_accuracy'] = np.mean(direction_pred == direction_true)
        
        # Update metadata with performance metrics
        self.metadata['performance'] = metrics
        
        return metrics
