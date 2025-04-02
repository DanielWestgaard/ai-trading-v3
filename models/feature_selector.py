# Is this even needed??

# models/feature_selector.py
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime

class ModelFeatureSelector:
    """Utility for selecting optimal features for prediction models."""
    
    def __init__(self, 
                 selection_method: str = 'importance',
                 target: str = 'close_return',
                 prediction_type: str = 'classification',
                 n_features: int = 20,
                 importance_threshold: float = 0.01,
                 min_features: int = 5,
                 correlation_threshold: float = 0.7,
                 cv_folds: int = 5,
                 output_dir: str = 'ml_model_results'):
        """
        Initialize the feature selector.
        
        Args:
            selection_method: Method for feature selection 
                              ('importance', 'correlation', 'recursive', 'sequential')
            target: Target column name
            prediction_type: 'classification' or 'regression'
            n_features: Number of features to select (for top-n methods)
            importance_threshold: Minimum importance threshold for features
            min_features: Minimum number of features to select
            correlation_threshold: Threshold for correlation filtering
            cv_folds: Number of cross-validation folds
            output_dir: Directory to save results
        """
        self.selection_method = selection_method
        self.target = target
        self.prediction_type = prediction_type
        self.n_features = n_features
        self.importance_threshold = importance_threshold
        self.min_features = min_features
        self.correlation_threshold = correlation_threshold
        self.cv_folds = cv_folds
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.ModelFeatureSelector")
        
        # Selected features and importance scores
        self.selected_features = []
        self.feature_scores = {}
    
    def select_features(self, data: pd.DataFrame, model_type: str = 'xgboost') -> List[str]:
        """
        Select optimal features for the given data and model type.
        
        Args:
            data: Input DataFrame with features and target
            model_type: Type of model to use for feature selection
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features using method: {self.selection_method}")
        
        # Create binary classification target if needed
        if self.prediction_type == 'classification':
            target_col = self._prepare_classification_target(data)
        else:
            # For regression, just use the target column
            target_col = self.target
        
        # Check if target column exists
        if target_col not in data.columns:
            self.logger.error(f"Target column '{target_col}' not found in data")
            return []
        
        # Filter out date columns and the target
        exclude_cols = [self.target, target_col, 'date', 'timestamp']
        date_cols = [col for col in data.columns if data[col].dtype == 'datetime64[ns]' 
                     or 'date' in col.lower() or 'time' in col.lower()]
        exclude_cols.extend(date_cols)
        
        # Get available features
        features = [col for col in data.columns if col not in exclude_cols]
        self.logger.info(f"Starting with {len(features)} available features")
        
        # Apply feature selection based on method
        if self.selection_method == 'importance':
            selected = self._select_by_importance(data[features], data[target_col], model_type)
        elif self.selection_method == 'correlation':
            selected = self._select_by_correlation(data[features], data[target_col])
        elif self.selection_method == 'recursive':
            selected = self._select_by_recursive_elimination(data[features], data[target_col], model_type)
        elif self.selection_method == 'sequential':
            selected = self._select_by_sequential(data[features], data[target_col], model_type)
        else:
            self.logger.warning(f"Unknown selection method: {self.selection_method}, using importance")
            selected = self._select_by_importance(data[features], data[target_col], model_type)
        
        # Ensure we have at least min_features
        if len(selected) < self.min_features and len(features) >= self.min_features:
            self.logger.warning(f"Only {len(selected)} features selected, using top {self.min_features}")
            if self.feature_scores:
                # Sort features by importance score
                sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [f[0] for f in sorted_features[:self.min_features]]
            else:
                # If no scores available, just take the first min_features
                selected = features[:self.min_features]
        
        self.selected_features = selected
        self.logger.info(f"Selected {len(selected)} features")
        
        # Save feature importance visualization
        self._visualize_feature_importance()
        
        return selected
    
    def _prepare_classification_target(self, data: pd.DataFrame) -> str:
        """Prepare classification target column."""
        target_col = f"{self.target}_class"
        
        if target_col not in data.columns:
            # Generate target based on future returns (shifted)
            self.logger.info(f"Creating classification target from {self.target}")
            data[target_col] = (data[self.target].shift(-1) > 0).astype(int)
        
        return target_col
    
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> List[str]:
        """Select features based on feature importance from a model."""
        self.logger.info("Selecting features by importance")
        
        try:
            # Train a model to get feature importance
            if model_type == 'xgboost':
                import xgboost as xgb
                
                # Configure model
                if self.prediction_type == 'classification':
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='binary:logistic',
                        random_state=42
                    )
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='reg:squarederror',
                        random_state=42
                    )
            
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # Configure model
                if self.prediction_type == 'classification':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
            
            else:
                raise ValueError(f"Unsupported model type for importance: {model_type}")
            
            # Train model
            model.fit(X, y)
            
            # Get feature importances
            importance = model.feature_importances_
            
            # Create DataFrame for visualization
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Save feature importance for later
            for _, row in importance_df.iterrows():
                self.feature_scores[row['feature']] = row['importance']
            
            # Select features based on threshold or top N
            if self.importance_threshold > 0:
                selected = importance_df[importance_df['importance'] >= self.importance_threshold]['feature'].tolist()
            else:
                selected = importance_df.head(self.n_features)['feature'].tolist()
            
            # Save importance to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"feature_importance_{timestamp}.csv")
            importance_df.to_csv(save_path, index=False)
            self.logger.info(f"Feature importance saved to {save_path}")
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error in importance-based selection: {str(e)}")
            return list(X.columns)[:self.n_features]
    
    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on correlation with target."""
        self.logger.info("Selecting features by correlation with target")
        
        try:
            # Combine features and target
            data = pd.concat([X, y], axis=1)
            
            # Calculate correlation with target
            target_name = y.name
            correlations = data.corr()[target_name].sort_values(ascending=False)
            
            # Remove target correlation with itself
            correlations = correlations.drop(target_name)
            
            # Save correlations for later
            for feature, corr in correlations.items():
                self.feature_scores[feature] = abs(corr)
            
            # Filter features with high absolute correlation
            selected = correlations[abs(correlations) >= self.correlation_threshold].index.tolist()
            
            # If too few features, take top N by absolute correlation
            if len(selected) < self.min_features:
                selected = correlations.abs().sort_values(ascending=False).head(self.n_features).index.tolist()
            
            # Now remove highly correlated features from the selection
            selected = self._remove_correlated_features(X[selected])
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error in correlation-based selection: {str(e)}")
            return list(X.columns)[:self.n_features]
    
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.9) -> List[str]:
        """Remove highly correlated features from the selection."""
        self.logger.info(f"Removing highly correlated features (threshold: {threshold})")
        
        # Calculate feature correlation matrix
        corr_matrix = X.corr().abs()
        
        # Extract upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        self.logger.info(f"Dropped {len(to_drop)} correlated features")
        
        # Keep remaining features
        return [feature for feature in X.columns if feature not in to_drop]
    
    def _select_by_recursive_elimination(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> List[str]:
        """Select features by recursive feature elimination."""
        self.logger.info("Selecting features by recursive feature elimination")
        
        try:
            from sklearn.feature_selection import RFECV
            
            # Train a model to use for RFE
            if model_type == 'xgboost':
                import xgboost as xgb
                
                # Configure model
                if self.prediction_type == 'classification':
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
            
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # Configure model
                if self.prediction_type == 'classification':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
            
            else:
                raise ValueError(f"Unsupported model type for RFE: {model_type}")
            
            # Create cross-validation based RFE
            rfe = RFECV(
                estimator=model,
                step=1,
                cv=self.cv_folds,
                scoring='roc_auc' if self.prediction_type == 'classification' else 'neg_mean_squared_error',
                min_features_to_select=self.min_features
            )
            
            # Fit RFE
            rfe.fit(X, y)
            
            # Get selected features
            selected = X.columns[rfe.support_].tolist()
            
            # Save feature importance using ranking
            for i, feature in enumerate(X.columns):
                # Lower rank means more important
                # Convert to a score where higher is better
                self.feature_scores[feature] = 1.0 / rfe.ranking_[i] if rfe.ranking_[i] > 0 else 0
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error in RFE-based selection: {str(e)}")
            return list(X.columns)[:self.n_features]
    
    def _select_by_sequential(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> List[str]:
        """Select features by sequential feature selection."""
        self.logger.info("Selecting features by sequential feature selection")
        
        try:
            from sklearn.feature_selection import SequentialFeatureSelector
            
            # Train a model to use for sequential selection
            if model_type == 'xgboost':
                import xgboost as xgb
                
                # Configure model
                if self.prediction_type == 'classification':
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
            
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                
                # Configure model
                if self.prediction_type == 'classification':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
            
            else:
                raise ValueError(f"Unsupported model type for sequential selection: {model_type}")
            
            # Calculate max features as percentage if n_features is less than 1
            max_features = self.n_features
            if max_features < 1:
                max_features = max(int(len(X.columns) * max_features), self.min_features)
            
            # Create sequential selector
            selector = SequentialFeatureSelector(
                estimator=model,
                n_features_to_select=max_features,
                direction='forward',
                scoring='roc_auc' if self.prediction_type == 'classification' else 'neg_mean_squared_error',
                cv=self.cv_folds
            )
            
            # Fit selector
            selector.fit(X, y)
            
            # Get selected features
            selected = X.columns[selector.get_support()].tolist()
            
            # All selected features get equal importance score
            for feature in selected:
                self.feature_scores[feature] = 1.0
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error in sequential selection: {str(e)}")
            return list(X.columns)[:self.n_features]
    
    def _visualize_feature_importance(self):
        """Visualize feature importance of selected features."""
        if not self.feature_scores or not self.selected_features:
            return
        
        try:
            # Create DataFrame for visualization
            scores = {feature: self.feature_scores.get(feature, 0) for feature in self.selected_features}
            vis_df = pd.DataFrame({
                'feature': list(scores.keys()),
                'importance': list(scores.values())
            })
            
            # Sort by importance
            vis_df = vis_df.sort_values('importance', ascending=False)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='importance', y='feature', data=vis_df)
            plt.title(f'Feature Importance ({self.selection_method.capitalize()} Method)')
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"feature_importance_plot_{timestamp}.png")
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance visualization: {str(e)}")
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_scores
    
    def save_features_list(self, filepath=None) -> str:
        """Save list of selected features to a file."""
        if not self.selected_features:
            self.logger.warning("No features selected to save")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"selected_features_{timestamp}.txt")
        
        try:
            with open(filepath, 'w') as f:
                f.write('\n'.join(self.selected_features))
            
            self.logger.info(f"Selected features saved to {filepath}")
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error saving features list: {str(e)}")
            return None