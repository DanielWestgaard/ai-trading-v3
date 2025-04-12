# models/tree/xgboost_model.py
import os
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from typing import Dict, List, Union, Tuple, Optional, Any
import config.constants.system_config as sys_congig

class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, 
                 name: str = 'xgboost_model',
                 features: List[str] = None,
                 target: str = 'close_return',
                 prediction_type: str = 'classification',
                 lookback_periods: int = 10,
                 prediction_horizon: int = 1,
                 model_dir: str = sys_congig.SAVED_MODELS_DIR,
                 **xgb_params):
        """
        Initialize XGBoost model.
        
        Args:
            name: Model name/identifier
            features: List of feature columns to use
            target: Target column name to predict
            prediction_type: 'classification' or 'regression'
            lookback_periods: Number of periods to look back for features
            prediction_horizon: Number of periods ahead to predict
            model_dir: Directory to save/load model files
            **xgb_params: Parameters for the XGBoost model
        """
        super().__init__(
            name=name,
            features=features,
            target=target,
            prediction_type=prediction_type,
            lookback_periods=lookback_periods,
            prediction_horizon=prediction_horizon,
            model_dir=model_dir
        )
        
        # Default XGBoost parameters - potential for optimizing parameters?
        self.xgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic' if prediction_type == 'classification' else 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with user-provided parameters
        self.xgb_params.update(xgb_params)
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        import xgboost as xgb
        
        # Update parameters with kwargs
        fit_params = self.xgb_params.copy()
        fit_params.update(kwargs)
        
        # Create and train the model
        if self.prediction_type == 'classification':
            self.model = xgb.XGBClassifier(**fit_params)
        else:
            self.model = xgb.XGBRegressor(**fit_params)
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            
            if self.features:
                self.feature_importance = pd.Series(
                    self.feature_importance, 
                    index=self.features if len(self.features) == X_train.shape[1] else None
                )
        
        self.is_fitted = True
        self.logger.info(f"XGBoost model trained with {X_train.shape[0]} samples")
        
        return self.model
    
    def predict(self, X, **kwargs):
        """
        Generate predictions with the XGBoost model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            self.logger.warning("Model not fitted, cannot predict")
            return None
        
        return self.model.predict(X)
    
    def predict_proba(self, X, **kwargs):
        """
        Generate probability predictions with the XGBoost model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            self.logger.warning("Model not fitted, cannot predict probabilities")
            return None
        
        if self.prediction_type == 'classification':
            return self.model.predict_proba(X)
        else:
            # For regression, create pseudo-probabilities based on prediction value
            preds = self.model.predict(X)
            # Convert to probabilities centered around 0.5
            # Scaling factor can be adjusted based on the range of predictions
            scaling_factor = 1.0
            probs = 1 / (1 + np.exp(-scaling_factor * preds))
            return np.column_stack((1 - probs, probs))
    
    def get_feature_importance(self, plot=False, top_n=20):
        """
        Get feature importance from the trained XGBoost model.
        
        Use this method AFTER MODEL TRAINING to:
        1. Interpret which features the model found most useful
        2. Understand the model's decision-making process
        3. Debug model behavior and identify potential issues
        4. Generate reports and visualizations for stakeholders
        
        For feature selection BEFORE training as part of the data pipeline,
        use the FeatureSelector class from data.features.feature_selector.
        
        Args:
            plot: Whether to plot feature importance
            top_n: Number of top features to show in plot
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        feature_importance = super().get_feature_importance()
        
        if plot and self.features and len(feature_importance) > 0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Convert to Series for plotting
            if not isinstance(self.feature_importance, pd.Series):
                importance_series = pd.Series(feature_importance)
            else:
                importance_series = self.feature_importance
            
            # Sort and select top N features
            importance_series = importance_series.sort_values(ascending=False).head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x=importance_series.values, y=importance_series.index)
            plt.title(f'Top {top_n} Feature Importance - {self.name}')
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.model_dir, f"{self.name}_feature_importance.png")
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return feature_importance