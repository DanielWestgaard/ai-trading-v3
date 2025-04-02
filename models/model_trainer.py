# models/model_trainer.py
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from models.model_factory import ModelFactory

class ModelTrainer:
    """Class for training and evaluating prediction models."""
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 features: List[str] = None,
                 target: str = 'close_return',
                 prediction_type: str = 'classification',
                 lookback_periods: int = 10,
                 prediction_horizon: int = 1,
                 test_size: float = 0.2,
                 validation_size: float = 0.1,
                 n_splits: int = 5,
                 scale_features: bool = True,
                 model_dir: str = 'model_storage',
                 results_dir: str = 'ml_model_results',
                 model_params: Dict[str, Any] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('xgboost', 'random_forest', 'lstm', etc.)
            features: List of feature columns to use
            target: Target column name to predict
            prediction_type: 'classification' or 'regression'
            lookback_periods: Number of periods to look back for features
            prediction_horizon: Number of periods ahead to predict
            test_size: Fraction of data to use for testing
            validation_size: Fraction of training data to use for validation
            n_splits: Number of splits for time series cross-validation
            scale_features: Whether to scale features
            model_dir: Directory to save/load model files
            results_dir: Directory to save results
            model_params: Parameters for the model
        """
        self.model_type = model_type
        self.features = features
        self.target = target
        self.prediction_type = prediction_type
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.validation_size = validation_size
        self.n_splits = n_splits
        self.scale_features = scale_features
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.model_params = model_params or {}
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.ModelTrainer")
        
        # Initialize the model
        self.model = None
        self.scaler = StandardScaler() if scale_features else None
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training and testing.
        
        Args:
            data: Input DataFrame with features and target
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Make a copy of the data
        df = data.copy()
        
        # Verify target column exists or find a suitable alternative
        if self.target not in df.columns:
            self.logger.warning(f"Target column '{self.target}' not found. Looking for alternatives...")
            
            # Option 1: Look for any return column
            return_cols = [col for col in df.columns if 'return' in col.lower()]
            if return_cols:
                self.target = return_cols[0]
                self.logger.info(f"Using '{self.target}' as target instead.")
            
            # Option 2: Look for close_original and create a return
            elif 'close_original' in df.columns:
                self.logger.info("Using 'close_original' to create a return target.")
                df['close_return'] = df['close_original'].pct_change()
                self.target = 'close_return'
                
                # Drop first row with NaN return
                df = df.dropna(subset=[self.target])
            
            # Option 3: Look for close_raw
            elif 'close_raw' in df.columns:
                self.logger.info("Using 'close_raw' to create a return target.")
                df['close_return'] = df['close_raw'].pct_change()
                self.target = 'close_return'
                
                # Drop first row with NaN return
                df = df.dropna(subset=[self.target])
            
            else:
                raise ValueError(f"Could not find suitable target column in dataset. Available columns: {df.columns.tolist()}")
        
        # Create binary classification target if needed
        if self.prediction_type == 'classification':
            # Generate target based on future returns
            # For n periods ahead, shift target back by n periods
            future_target = df[self.target].shift(-self.prediction_horizon)
            df['target_class'] = (future_target > 0).astype(int)
            target_col = 'target_class'
        else:
            # For regression, just use the future value
            df['target_value'] = df[self.target].shift(-self.prediction_horizon)
            target_col = 'target_value'
        
        # Drop NaN values created by shift
        df = df.dropna(subset=[target_col])
        
        # Select features if provided, otherwise use all available features
        if self.features is None:
            # Exclude the original target and the derived targets
            exclude_cols = [self.target, 'target_class', 'target_value', 'date', 'timestamp']
            self.features = [col for col in df.columns if col not in exclude_cols]
            
            # Remove any date or timestamp columns
            date_cols = [col for col in self.features if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower()]
            self.features = [col for col in self.features if col not in date_cols]
            
            self.logger.info(f"Auto-selected {len(self.features)} features")
        
        # Extract features and target
        X = df[self.features]
        y = df[target_col]
        
        # Time-based train/test split
        train_size = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features if requested
        if self.scale_features and self.scaler:
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        self.logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
        
    def train(self, data: pd.DataFrame, **kwargs) -> Any:
        """
        Train the model on the provided data.
        
        Args:
            data: Input DataFrame with features and target
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        # Initialize the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model_type}_{timestamp}"
        
        self.model = ModelFactory.create_model(
            self.model_type,
            name=model_name,
            features=self.features,
            target=self.target,
            prediction_type=self.prediction_type,
            lookback_periods=self.lookback_periods,
            prediction_horizon=self.prediction_horizon,
            model_dir=self.model_dir,
            **self.model_params
        )
        
        # Combine kwargs with model params
        fit_params = self.model_params.copy()
        fit_params.update(kwargs)
        
        # Train the model
        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train, **fit_params)
        
        # Evaluate the model
        metrics = self.model.evaluate(X_test, y_test)
        self.logger.info(f"Model evaluation metrics: {metrics}")
        
        # Save the model
        model_path = self.model.save()
        
        # Save feature importance plot
        self.model.get_feature_importance(plot=True)
        
        # Generate and save prediction plots
        self._generate_prediction_plots(X_test, y_test)
        
        return self.model
    
    def cross_validate(self, data: pd.DataFrame, **kwargs) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation using proper financial time series splitting.
        
        Args:
            data: Input DataFrame with features and target
            **kwargs: Additional training parameters
                - train_period: Length of training period (default: '3M')
                - test_period: Length of test period (default: '1M')
                - step_size: Increment between successive training sets (default: same as test_period)
                
        Returns:
            Dictionary with cross-validation metrics
        """
        from data.features.time_series_ml import TimeSeriesSplit
        
        # Extract CV parameters
        train_period = kwargs.pop('train_period', '3M')
        test_period = kwargs.pop('test_period', '1M')
        step_size = kwargs.pop('step_size', None)
        
        # Create a splitter
        splitter = TimeSeriesSplit(
            train_period=train_period,
            test_period=test_period,
            step_size=step_size,
            n_splits=self.n_splits
        )
        
        # Prepare data first to prepare target column
        _, _, target_col = self.prepare_data_for_cv(data)
        
        # Generate splits
        splits = splitter.split(data)
        
        # Initialize containers for metrics
        cv_metrics = {}
        
        for i, (train_data, test_data) in enumerate(splits):
            self.logger.info(f"Cross-validation fold {i+1}/{len(splits)}")
            
            # Extract features and target
            X_train = train_data[self.features]
            y_train = train_data[target_col]
            X_test = test_data[self.features]
            y_test = test_data[target_col]
            
            # Scale features if requested
            if self.scale_features:
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
            
            # Initialize the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_cv{i+1}_{timestamp}"
            
            model = ModelFactory.create_model(
                self.model_type,
                name=model_name,
                features=self.features,
                target=target_col,
                prediction_type=self.prediction_type,
                lookback_periods=self.lookback_periods,
                prediction_horizon=self.prediction_horizon,
                model_dir=self.model_dir,
                **self.model_params
            )
            
            # Train the model
            model.fit(X_train, y_train, **kwargs)
            
            # Evaluate the model
            fold_metrics = model.evaluate(X_test, y_test)
            
            # Store metrics
            for metric, value in fold_metrics.items():
                if metric not in cv_metrics:
                    cv_metrics[metric] = []
                cv_metrics[metric].append(value)
        
        # Calculate average metrics
        avg_metrics = {
            f"avg_{metric}": np.mean(values) for metric, values in cv_metrics.items()
        }
        
        # Add standard deviation
        std_metrics = {
            f"std_{metric}": np.std(values) for metric, values in cv_metrics.items()
        }
        
        # Combine all metrics
        all_metrics = {**cv_metrics, **avg_metrics, **std_metrics}
        
        self.logger.info(f"Cross-validation results: {avg_metrics}")
        
        # Save CV results
        self._save_cv_results(all_metrics)
        
        return all_metrics
    
    def prepare_data_for_cv(self, data: pd.DataFrame) -> Tuple[List[str], str, str]:
        """
        Prepare data for cross-validation by determining features and target.
        
        Args:
            data: Input DataFrame with features and target
            
        Returns:
            features, original_target_name, processed_target_name
        """
        # Create binary classification target if needed
        if self.prediction_type == 'classification':
            # Generate target based on future returns
            # For n periods ahead, shift target back by n periods
            future_target = data[self.target].shift(-self.prediction_horizon)
            data['target_class'] = (future_target > 0).astype(int)
            target_col = 'target_class'
        else:
            # For regression, just use the future value
            data['target_value'] = data[self.target].shift(-self.prediction_horizon)
            target_col = 'target_value'
        
        # Select features if provided, otherwise use all available features
        if self.features is None:
            # Exclude the original target and the derived targets
            exclude_cols = [self.target, 'target_class', 'target_value']
            self.features = [col for col in data.columns if col not in exclude_cols]
            
            # Remove any date or timestamp columns
            date_cols = [col for col in self.features if data[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower()]
            self.features = [col for col in self.features if col not in date_cols]
            
            self.logger.info(f"Auto-selected {len(self.features)} features for cross-validation")
        
        return self.features, self.target, target_col
        
    def _generate_prediction_plots(self, X_test, y_test):
        """Generate prediction vs actual plots."""
        if self.model is None:
            return
        
        # Create predictions
        y_pred = self.model.predict(X_test)
        
        if self.prediction_type == 'classification':
            # For classification, plot probability predictions
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot actual values
            plt.scatter(X_test.index, y_test, c='blue', alpha=0.5, label='Actual')
            
            # Plot probability predictions
            plt.scatter(X_test.index, y_prob, c='red', alpha=0.5, label='Predicted Probability')
            
            plt.title(f'{self.model_type.capitalize()} Classification Predictions')
            plt.xlabel('Date')
            plt.ylabel('Probability / Actual (0/1)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            # For regression, plot actual vs predicted
            plt.figure(figsize=(12, 6))
            
            # Plot actual values
            plt.plot(X_test.index, y_test, c='blue', alpha=0.7, label='Actual')
            
            # Plot predicted values
            plt.plot(X_test.index, y_pred, c='red', alpha=0.7, label='Predicted')
            
            plt.title(f'{self.model_type.capitalize()} Regression Predictions')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.results_dir, f"{self.model_type}_predictions_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Prediction plot saved to {save_path}")
    
    def _save_cv_results(self, metrics):
        """Save cross-validation results."""
        # Convert metrics to DataFrame
        results = {}
        for metric, values in metrics.items():
            if isinstance(values, list):
                for i, value in enumerate(values):
                    results[f"{metric}_fold{i+1}"] = value
            else:
                results[metric] = values
        
        results_df = pd.DataFrame([results])
        
        # Add metadata
        results_df['model_type'] = self.model_type
        results_df['features_count'] = len(self.features)
        results_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.results_dir, f"{self.model_type}_cv_results_{timestamp}.csv")
        results_df.to_csv(save_path, index=False)
        
        self.logger.info(f"Cross-validation results saved to {save_path}")