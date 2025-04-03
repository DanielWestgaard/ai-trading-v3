# models/tree/random_forest_model.py
import os
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from typing import Dict, List, Union, Tuple, Optional, Any

class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self, 
                 name: str = 'random_forest_model',
                 features: List[str] = None,
                 target: str = 'close_return',
                 prediction_type: str = 'classification',
                 lookback_periods: int = 10,
                 prediction_horizon: int = 1,
                 model_dir: str = 'model_storage',
                 **rf_params):
        """
        Initialize Random Forest model.
        
        Args:
            name: Model name/identifier
            features: List of feature columns to use
            target: Target column name to predict
            prediction_type: 'classification' or 'regression'
            lookback_periods: Number of periods to look back for features
            prediction_horizon: Number of periods ahead to predict
            model_dir: Directory to save/load model files
            **rf_params: Parameters for the Random Forest model
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
        
        # Default Random Forest parameters
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with user-provided parameters
        self.rf_params.update(rf_params)
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Update parameters with kwargs
        fit_params = self.rf_params.copy()
        fit_params.update(kwargs)
        
        # Create and train the model
        if self.prediction_type == 'classification':
            self.model = RandomForestClassifier(**fit_params)
        else:
            self.model = RandomForestRegressor(**fit_params)
        
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
        self.logger.info(f"Random Forest model trained with {X_train.shape[0]} samples")
        
        return self.model
    
    def predict(self, X, **kwargs):
        """
        Generate predictions with the Random Forest model.
        
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
        Generate probability predictions with the Random Forest model.
        
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
            # For regression, create pseudo-probabilities
            # This is similar to the XGBoost implementation
            preds = self.model.predict(X)
            scaling_factor = 1.0
            probs = 1 / (1 + np.exp(-scaling_factor * preds))
            return np.column_stack((1 - probs, probs))
    
    def get_feature_importance(self, plot=False, top_n=20):
        """
        Get feature importance from the trained Random Forest model.
        
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

    def get_tree_visualization(self, tree_index=0, max_depth=3):
        """
        Generate a visualization of one of the trees in the forest.
        
        Args:
            tree_index: Index of tree to visualize
            max_depth: Maximum depth of tree to display
            
        Returns:
            Path to saved visualization
        """
        if not self.is_fitted:
            self.logger.warning("Model not fitted, cannot visualize trees")
            return None
        
        try:
            from sklearn.tree import export_graphviz
            import graphviz
            
            # Get the selected tree
            estimator = self.model.estimators_[tree_index]
            
            # Create feature names if not provided
            if self.features is None:
                feature_names = [f'feature_{i}' for i in range(estimator.n_features_in_)]
            else:
                feature_names = self.features
            
            # Set class names for classification
            if self.prediction_type == 'classification':
                class_names = ['down', 'up']
            else:
                class_names = None
            
            # Export tree to dot format
            dot_data = export_graphviz(
                estimator,
                out_file=None,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True,
                max_depth=max_depth
            )
            
            # Convert to graphviz object
            graph = graphviz.Source(dot_data)
            
            # Save the visualization
            save_path = os.path.join(self.model_dir, f"{self.name}_tree_{tree_index}.pdf")
            graph.render(os.path.splitext(save_path)[0])
            
            self.logger.info(f"Tree visualization saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error generating tree visualization: {str(e)}")
            return None