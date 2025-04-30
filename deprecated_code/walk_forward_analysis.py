# Important Note: This has not been unified or implemented with the current verison of backtesting!
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import json

from sklearn.model_selection import TimeSeriesSplit

from backtesting.backtest_runner import BacktestRunner


class ModelPerformanceTracker:
    """
    Tracks and analyzes model performance across different time periods and market regimes.
    Optimized for your trading strategy testing needs.
    """
    
    def __init__(self, output_dir=None):
        """Initialize performance tracker."""
        self.predictions = []
        self.metrics = {}
        self.performance_by_regime = {}
        self.output_dir = output_dir
        logging = self._setup_logger()
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.ModelPerformanceTracker")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def add_predictions(self, fold_id: int, test_data: pd.DataFrame, 
                      predictions: np.ndarray, dates: np.ndarray,
                      actuals: np.ndarray, prediction_type: str = 'regression',
                      model_meta: Optional[Dict[str, Any]] = None):
        """
        Add prediction results from a test fold.
        
        Args:
            fold_id: ID of the fold
            test_data: Test data DataFrame
            predictions: Model predictions
            dates: Dates corresponding to predictions
            actuals: Actual values
            prediction_type: 'regression' or 'classification'
            model_meta: Additional model metadata
        """
        # Store prediction data
        pred_data = {
            'fold_id': fold_id,
            'dates': dates,
            'predictions': predictions,
            'actuals': actuals,
            'prediction_type': prediction_type,
            'model_meta': model_meta or {}
        }
        
        # Check if the test data has a market regime column
        if 'market_regime' in test_data.columns:
            pred_data['market_regime'] = test_data['market_regime'].values
        
        self.predictions.append(pred_data)
        
        # Calculate metrics for this fold
        metrics = self._calculate_metrics(predictions, actuals, prediction_type)
        self.metrics[fold_id] = metrics
        
        # Log summary
        logging.info(f"Fold {fold_id} metrics: {metrics}")
        
        # Save predictions to CSV if output_dir is specified
        if self.output_dir:
            # Create a DataFrame with the predictions
            fold_df = pd.DataFrame({
                'date': dates,
                'actual': actuals,
                'prediction': predictions,
            })
            
            # Add any market regime information if available
            if 'market_regime' in pred_data:
                fold_df['market_regime'] = pred_data['market_regime']
            
            # Save to CSV
            fold_csv_path = os.path.join(self.output_dir, f'fold_{fold_id}_predictions.csv')
            fold_df.to_csv(fold_csv_path, index=False)
            logging.info(f"Saved fold {fold_id} predictions to {fold_csv_path}")
        
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, actuals: np.ndarray, 
                          prediction_type: str = 'regression') -> Dict[str, float]:
        """
        Calculate performance metrics based on prediction type.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            prediction_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of metrics
        """
        if prediction_type == 'regression':
            # Regression metrics
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actuals, predictions)
            mae = np.mean(np.abs(actuals - predictions))
            
            # Direction accuracy for regression
            direction_correct = np.sum((np.diff(actuals) > 0) == (np.diff(predictions) > 0))
            direction_accuracy = direction_correct / (len(actuals) - 1) if len(actuals) > 1 else np.nan
            
            return {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            }
        
        elif prediction_type == 'classification':
            # Classification metrics
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
            recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
            f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def calculate_overall_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics across all folds.
        
        Returns:
            Dictionary of overall metrics
        """
        if not self.predictions:
            return {}
        
        # Combine all predictions and actuals
        all_predictions = np.concatenate([p['predictions'] for p in self.predictions])
        all_actuals = np.concatenate([p['actuals'] for p in self.predictions])
        prediction_type = self.predictions[0]['prediction_type']
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_predictions, all_actuals, prediction_type)
        
        # Calculate average and std of fold metrics
        fold_metrics = pd.DataFrame(self.metrics).T
        avg_metrics = fold_metrics.mean(axis=0).to_dict()
        std_metrics = fold_metrics.std(axis=0).to_dict()
        
        # Combine metrics
        result = {
            'overall': overall_metrics,
            'avg_fold': {f"avg_{k}": v for k, v in avg_metrics.items()},
            'std_fold': {f"std_{k}": v for k, v in std_metrics.items()}
        }
        
        # Save overall metrics to CSV if output_dir is specified
        if self.output_dir:
            # Create a DataFrame with all predictions
            all_dates = np.concatenate([p['dates'] for p in self.predictions])
            
            # Sort by date
            sort_indices = np.argsort(all_dates)
            all_dates = all_dates[sort_indices]
            all_predictions = all_predictions[sort_indices]
            all_actuals = all_actuals[sort_indices]
            
            # Create DataFrame
            all_df = pd.DataFrame({
                'date': all_dates,
                'actual': all_actuals,
                'prediction': all_predictions,
            })
            
            # Save to CSV
            all_csv_path = os.path.join(self.output_dir, 'all_predictions.csv')
            all_df.to_csv(all_csv_path, index=False)
            logging.info(f"Saved all predictions to {all_csv_path}")
            
            # Save metrics to JSON
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(result, f, indent=4)
            logging.info(f"Saved metrics to {metrics_path}")
        
        return result
    
    def analyze_by_market_regime(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance broken down by market regime.
        
        Returns:
            Dictionary mapping market regimes to performance metrics
        """
        regime_predictions = {}
        regime_actuals = {}
        
        # Group predictions by regime
        for pred_data in self.predictions:
            if 'market_regime' not in pred_data:
                continue
            
            regimes = pred_data['market_regime']
            predictions = pred_data['predictions']
            actuals = pred_data['actuals']
            
            for regime, pred, actual in zip(regimes, predictions, actuals):
                if regime not in regime_predictions:
                    regime_predictions[regime] = []
                    regime_actuals[regime] = []
                
                regime_predictions[regime].append(pred)
                regime_actuals[regime].append(actual)
        
        # Calculate metrics for each regime
        regime_metrics = {}
        
        if not self.predictions or not regime_predictions:
            logging.warning("No market regime data available for analysis")
            return {}
            
        prediction_type = self.predictions[0]['prediction_type']
        
        for regime in regime_predictions:
            preds = np.array(regime_predictions[regime])
            acts = np.array(regime_actuals[regime])
            
            metrics = self._calculate_metrics(preds, acts, prediction_type)
            regime_metrics[regime] = metrics
            
            logging.info(f"Regime '{regime}' metrics: {metrics}")
        
        self.performance_by_regime = regime_metrics
        
        # Save regime metrics to CSV if output_dir is specified
        if self.output_dir:
            # Create a DataFrame with regime metrics
            regime_df = pd.DataFrame(regime_metrics).T
            regime_df.index.name = 'regime'
            
            # Save to CSV
            regime_csv_path = os.path.join(self.output_dir, 'regime_metrics.csv')
            regime_df.to_csv(regime_csv_path)
            logging.info(f"Saved regime metrics to {regime_csv_path}")
        
        return regime_metrics
    
    def plot_predictions(self, output_path=None, figsize: tuple = (15, 8)) -> plt.Figure:
        """
        Plot actual vs predicted values across all folds.
        
        Args:
            output_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.predictions:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Combine all data, sorting by date
        all_dates = []
        all_actuals = []
        all_predictions = []
        all_fold_ids = []
        
        for pred_data in self.predictions:
            all_dates.extend(pred_data['dates'])
            all_actuals.extend(pred_data['actuals'])
            all_predictions.extend(pred_data['predictions'])
            all_fold_ids.extend([pred_data['fold_id']] * len(pred_data['dates']))
        
        # Convert to DataFrame for easier handling
        results_df = pd.DataFrame({
            'date': all_dates,
            'actual': all_actuals,
            'prediction': all_predictions,
            'fold_id': all_fold_ids
        }).sort_values('date')
        
        # Plot actuals
        ax.plot(results_df['date'], results_df['actual'], label='Actual', color='black')
        
        # Plot predictions colored by fold
        fold_ids = results_df['fold_id'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(fold_ids)))
        
        for i, fold_id in enumerate(sorted(fold_ids)):
            fold_data = results_df[results_df['fold_id'] == fold_id]
            ax.plot(fold_data['date'], fold_data['prediction'], 
                   label=f'Predictions (Fold {fold_id})', 
                   color=colors[i], alpha=0.7)
        
        ax.set_title('Actual vs Predicted Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if path provided or output_dir is set
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved predictions plot to {output_path}")
        elif self.output_dir:
            plot_path = os.path.join(self.output_dir, 'predictions_plot.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved predictions plot to {plot_path}")
        
        return fig
    
    def plot_metrics_over_time(self, metric_name: str = None, output_path=None, figsize: tuple = (15, 8)) -> plt.Figure:
        """
        Plot how metrics change over time across folds.
        
        Args:
            metric_name: Specific metric to plot (if None, use first available)
            output_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.metrics:
            return None
        
        # Get available metrics
        available_metrics = list(next(iter(self.metrics.values())).keys())
        
        if not metric_name or metric_name not in available_metrics:
            metric_name = available_metrics[0]
            logging.info(f"Using metric: {metric_name}")
        
        # Extract fold dates (using first date in each fold)
        fold_dates = []
        fold_metrics = []
        
        for fold_id, metrics in sorted(self.metrics.items()):
            fold_data = next(p for p in self.predictions if p['fold_id'] == fold_id)
            fold_dates.append(fold_data['dates'][0])
            fold_metrics.append(metrics[metric_name])
        
        # Plot metrics over time
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fold_dates, fold_metrics, marker='o', linestyle='-')
        
        ax.set_title(f'{metric_name} Over Time')
        ax.set_xlabel('Time (Start of Test Period)')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        
        # Add overall average
        avg_metric = np.mean(fold_metrics)
        ax.axhline(avg_metric, color='red', linestyle='--', 
                  label=f'Average: {avg_metric:.4f}')
        ax.legend()
        
        # Save if path provided or output_dir is set
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved metrics plot to {output_path}")
        elif self.output_dir:
            plot_path = os.path.join(self.output_dir, f'{metric_name}_over_time.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved metrics plot to {plot_path}")
        
        return fig


class WalkForwardAnalysis:
    """
    Implements Walk-Forward Analysis for validating trading strategies and ML models.
    Designed to work with your existing pipeline components.
    """
    
    def __init__(self, 
                 train_period: Union[str, int, timedelta] = '1Y',
                 test_period: Union[str, int, timedelta] = '3M',
                 step_size: Optional[Union[str, int, timedelta]] = None,
                 max_train_size: Optional[Union[str, int, timedelta]] = None,
                 n_splits: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 model_store_path: Optional[str] = None,
                 date_column: str = 'date'):
        """
        Initialize Walk-Forward Analysis.
        
        Args:
            train_period: Length of the training period
            test_period: Length of the test period
            step_size: Increment between successive training sets (default: same as test_period)
            max_train_size: Maximum training set size (if None, no limit)
            n_splits: Number of splits (if specified, overrides other parameters)
            output_dir: Directory to save results
            model_store_path: Directory to save trained models
            date_column: Name of the date/timestamp column
        """
        self.splitter = TimeSeriesSplit(
            train_period=train_period,
            test_period=test_period,
            step_size=step_size,
            max_train_size=max_train_size,
            n_splits=n_splits,
            date_column=date_column
        )
        
        self.date_column = date_column
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.model_store_path = model_store_path
        if model_store_path and not os.path.exists(model_store_path):
            os.makedirs(model_store_path)
        
        self.performance_tracker = ModelPerformanceTracker(output_dir=output_dir)
        logging = self._setup_logger()
        
        # Store analysis results
        self.results = {
            'config': {
                'train_period': str(train_period),
                'test_period': str(test_period),
                'step_size': str(step_size) if step_size else None,
                'max_train_size': str(max_train_size) if max_train_size else None,
                'n_splits': n_splits
            },
            'models': {},
            'performance': {},
            'backtest_results': {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.WalkForwardAnalysis")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger

    def create_target_column(self, data: pd.DataFrame, target_type='return', 
                           source_column='close_original', horizon=1) -> pd.DataFrame:
        """
        Create a target column for prediction based on future values.
        
        Args:
            data: DataFrame containing features
            target_type: Type of target ('return', 'direction', 'price')
            source_column: Source column to use for target creation
            horizon: Number of periods to look ahead
            
        Returns:
            DataFrame with added target column
        """
        result = data.copy()
        
        # Find a suitable source column if the provided one doesn't exist
        if source_column not in result.columns:
            # Try to find a price column
            price_cols = [col for col in result.columns if 'close' in col.lower()]
            if price_cols:
                source_column = price_cols[0]
                logging.info(f"Using '{source_column}' as source column for target creation")
            else:
                raise ValueError(f"Source column '{source_column}' not found")
        
        target_column = f'target_{source_column}_{horizon}'
        
        if target_type == 'return':
            # Future return
            result[target_column] = result[source_column].shift(-horizon).pct_change(-horizon)
            logging.info(f"Created future return target '{target_column}'")
            
        elif target_type == 'direction':
            # Future direction (1 for up, -1 for down, 0 for no change)
            future_price = result[source_column].shift(-horizon)
            result[target_column] = np.sign(future_price - result[source_column])
            logging.info(f"Created future direction target '{target_column}'")
            
        elif target_type == 'price':
            # Future price
            result[target_column] = result[source_column].shift(-horizon)
            logging.info(f"Created future price target '{target_column}'")
            
        elif target_type == 'normalized_return':
            # Z-score normalized return
            future_return = result[source_column].shift(-horizon).pct_change(-horizon)
            mean = future_return.mean()
            std = future_return.std()
            result[target_column] = (future_return - mean) / std
            logging.info(f"Created normalized future return target '{target_column}'")
        
        # Drop rows with NaN targets
        na_count = result[target_column].isna().sum()
        if na_count > 0:
            logging.info(f"Dropping {na_count} rows with NaN target values")
            result = result.dropna(subset=[target_column])
        
        return result, target_column
    
    def run_model_analysis(self, 
                         data: pd.DataFrame,
                         features: List[str] = None,
                         target_column: str = None,
                         create_target: bool = False,
                         target_type: str = 'return', 
                         source_column: str = 'close_original',
                         horizon: int = 1,
                         model_factory: Callable[[], Any] = None,
                         train_func: Callable[[Any, pd.DataFrame, List[str], str], Any] = None,
                         predict_func: Callable[[Any, pd.DataFrame, List[str]], np.ndarray] = None,
                         prediction_type: str = 'regression',
                         save_models: bool = True) -> Dict[str, Any]:
        """
        Run walk-forward analysis with a machine learning model.
        
        Args:
            data: DataFrame containing features and target
            features: List of feature columns
            target_column: Name of the target column
            create_target: Whether to create a target column
            target_type: Type of target to create ('return', 'direction', 'price')
            source_column: Column to use for target creation
            horizon: Number of periods to look ahead for target
            model_factory: Function to create a new model instance
            train_func: Function to train the model (model, data, features, target) -> trained_model
            predict_func: Function to make predictions (model, data, features) -> predictions
            prediction_type: 'regression' or 'classification'
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of analysis results
        """
        # Create a working copy of the data
        df = data.copy()
        
        # Create target column if requested
        if create_target or target_column is None:
            df, target_column = self.create_target_column(
                df, target_type=target_type, source_column=source_column, horizon=horizon
            )
        elif target_column not in df.columns:
            # Try to find a suitable target column
            possible_targets = [col for col in df.columns if 'target' in col.lower() or 'return' in col.lower()]
            if possible_targets:
                target_column = possible_targets[0]
                logging.info(f"Using '{target_column}' as target column")
            else:
                # Create a target as a fallback
                df, target_column = self.create_target_column(
                    df, target_type=target_type, source_column=source_column, horizon=horizon
                )
        
        # Select features if not provided
        if features is None:
            # Exclude target and date columns
            exclude_cols = [self.date_column, target_column]
            features = [col for col in df.columns if col not in exclude_cols]
            logging.info(f"Using all {len(features)} columns as features")
        
        # Log key information
        logging.info(f"Running walk-forward analysis with {len(features)} features")
        logging.info(f"Target column: {target_column} (type: {prediction_type})")
        logging.info(f"Data shape: {df.shape} rows, {df.shape[1]} columns")
        
        # Visualize the splits
        if self.output_dir:
            try:
                fig = self.splitter.plot_splits(df)
                splits_viz_path = os.path.join(self.output_dir, 'splits_visualization.png')
                fig.savefig(splits_viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logging.info(f"Saved splits visualization to {splits_viz_path}")
            except Exception as e:
                logging.warning(f"Could not create splits visualization: {str(e)}")
        
        # Store configuration
        self.results['features'] = features
        self.results['target'] = target_column
        self.results['prediction_type'] = prediction_type
        
        # Generate train/test splits
        splits = self.splitter.split(df)
        logging.info(f"Created {len(splits)} train/test splits")
        
        # Run analysis for each split
        for fold_id, (train_data, test_data) in enumerate(splits):
            logging.info(f"Processing fold {fold_id+1}/{len(splits)}")
            
            # Safety check: make sure all features are in the data
            missing_features = [f for f in features if f not in train_data.columns]
            if missing_features:
                logging.warning(f"Missing features in training data: {missing_features}")
                features = [f for f in features if f not in missing_features]
                if len(features) == 0:
                    logging.error("No valid features left after filtering")
                    continue
            
            # Create model
            model = model_factory()
            
            # Train model
            logging.info(f"Training model on {len(train_data)} samples")
            try:
                model = train_func(model, train_data, features, target_column)
            except Exception as e:
                logging.error(f"Error training model: {str(e)}")
                continue
            
            # Save trained model if requested
            if save_models and self.model_store_path:
                model_filename = f"model_fold_{fold_id}.joblib"
                model_path = os.path.join(self.model_store_path, model_filename)
                joblib.dump(model, model_path)
                logging.info(f"Saved model to {model_path}")
                
                self.results['models'][fold_id] = {
                    'path': model_path,
                    'train_start': train_data[self.date_column].min().strftime('%Y-%m-%d'),
                    'train_end': train_data[self.date_column].max().strftime('%Y-%m-%d'),
                    'test_start': test_data[self.date_column].min().strftime('%Y-%m-%d'),
                    'test_end': test_data[self.date_column].max().strftime('%Y-%m-%d')
                }
            
            # Make predictions
            logging.info(f"Making predictions on {len(test_data)} samples")
            try:
                predictions = predict_func(model, test_data, features)
            except Exception as e:
                logging.error(f"Error making predictions: {str(e)}")
                continue
            
            # Extract test dates and actuals
            test_dates = test_data[self.date_column].values
            test_actuals = test_data[target_column].values
            
            # Track performance
            metrics = self.performance_tracker.add_predictions(
                fold_id=fold_id,
                test_data=test_data,
                predictions=predictions,
                dates=test_dates,
                actuals=test_actuals,
                prediction_type=prediction_type
            )
            
            self.results['performance'][fold_id] = metrics
        
        # Calculate overall performance
        overall_metrics = self.performance_tracker.calculate_overall_metrics()
        self.results['overall_performance'] = overall_metrics
        
        # Analyze by market regime if applicable
        if any('market_regime' in p for p in self.performance_tracker.predictions):
            regime_metrics = self.performance_tracker.analyze_by_market_regime()
            self.results['regime_performance'] = regime_metrics
        
        # Generate and save plots
        if self.output_dir:
            # Plot predictions
            pred_fig = self.performance_tracker.plot_predictions()
            if pred_fig:
                plt.close(pred_fig)
            
            # Plot metrics over time
            metrics_fig = self.performance_tracker.plot_metrics_over_time()
            if metrics_fig:
                plt.close(metrics_fig)
            
            # Save results
            results_path = os.path.join(self.output_dir, 'model_analysis_results.json')
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
            logging.info(f"Saved analysis results to {results_path}")
        
        return self.results
    
    def run_strategy_analysis(self, 
                            data: pd.DataFrame,
                            features: List[str] = None,
                            model_factory: Callable[[], Any] = None,
                            train_func: Callable[[Any, pd.DataFrame, List[str], str], Any] = None,
                            strategy_factory: Callable[[Any], Any] = None,
                            initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run walk-forward analysis with a trading strategy, using backtests for evaluation.
        
        Args:
            data: DataFrame containing features and price data
            features: List of feature columns
            model_factory: Function to create a new model instance
            train_func: Function to train the model
            strategy_factory: Function to create a strategy from a model
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary of analysis results
        """
        logging.info(f"Running walk-forward strategy analysis")
        
        # Create a working copy of the data
        df = data.copy()
        
        # Select features if not provided
        if features is None:
            # Exclude date column and any price columns that end with _raw
            raw_price_cols = [col for col in df.columns if col.endswith('_raw')]
            exclude_cols = [self.date_column] + raw_price_cols
            features = [col for col in df.columns if col not in exclude_cols]
            logging.info(f"Using {len(features)} columns as features")
        
        # Generate train/test splits
        splits = self.splitter.split(df)
        logging.info(f"Created {len(splits)} train/test splits")
        
        # Run analysis for each split
        for fold_id, (train_data, test_data) in enumerate(splits):
            logging.info(f"Processing fold {fold_id+1}/{len(splits)}")
            
            # Create and train model
            model = model_factory()
            try:
                model = train_func(model, train_data, features)
                logging.info(f"Trained model for fold {fold_id}")
            except Exception as e:
                logging.error(f"Error training model for fold {fold_id}: {str(e)}")
                continue
            
            # Save trained model if requested
            if self.model_store_path:
                model_filename = f"model_fold_{fold_id}.joblib"
                model_path = os.path.join(self.model_store_path, model_filename)
                joblib.dump(model, model_path)
                logging.info(f"Saved model to {model_path}")
                
                self.results['models'][fold_id] = {
                    'path': model_path,
                    'train_start': train_data[self.date_column].min().strftime('%Y-%m-%d'),
                    'train_end': train_data[self.date_column].max().strftime('%Y-%m-%d'),
                    'test_start': test_data[self.date_column].min().strftime('%Y-%m-%d'),
                    'test_end': test_data[self.date_column].max().strftime('%Y-%m-%d')
                }
            
            # Create strategy from model
            strategy = strategy_factory(model)
            
            # Run backtest on test data
            backtest_result = self._run_backtest(fold_id, strategy, test_data, initial_capital)
            
            if backtest_result:
                self.results['backtest_results'][fold_id] = backtest_result
        
        # Analyze overall backtest performance
        if self.results['backtest_results']:
            self._analyze_backtest_performance()
        else:
            logging.warning("No successful backtests to analyze")
        
        # Save results
        if self.output_dir:
            results_path = os.path.join(self.output_dir, 'strategy_analysis_results.json')
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
            logging.info(f"Saved analysis results to {results_path}")
            
            # Generate and save visualizations
            self._save_strategy_plots()
        
        return self.results
    
    def _run_backtest(self, 
                     fold_id: int, 
                     strategy: Any, 
                     test_data: pd.DataFrame, 
                     initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run a backtest for a single fold.
        
        Args:
            fold_id: ID of the fold
            strategy: Strategy to test
            test_data: Test data for backtesting
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        logging.info(f"Running backtest for fold {fold_id}")
        
        # Set up market data
        from backtesting.data.market_data import DataFrameMarketData
        
        # Check if we have required price columns
        required_columns = ['open', 'high', 'low', 'close']
        
        # Try to map required columns to available columns
        column_mapping = {}
        for req_col in required_columns:
            # Check for original price columns
            orig_col = f"{req_col}_original"
            raw_col = f"{req_col}_raw"
            
            if orig_col in test_data.columns:
                column_mapping[req_col] = orig_col
            elif raw_col in test_data.columns:
                column_mapping[req_col] = raw_col
            elif req_col in test_data.columns:
                column_mapping[req_col] = req_col
            else:
                # Try case-insensitive matching
                matches = [col for col in test_data.columns if col.lower() == req_col.lower()]
                if matches:
                    column_mapping[req_col] = matches[0]
        
        # Check if we have all required columns
        missing_cols = [col for col in required_columns if col not in column_mapping]
        if missing_cols:
            logging.error(f"Missing required price columns for backtesting: {missing_cols}")
            return None
        
        # Create a copy of the data with required columns
        backtest_data = test_data.copy()
        for req_col, mapped_col in column_mapping.items():
            backtest_data[req_col] = backtest_data[mapped_col]
        
        # Determine the symbol from the data or use a default
        symbol = backtest_data.get('symbol', [None])[0]
        if symbol is None:
            symbol = "UNKNOWN"
        
        # Create market data object
        symbols = [symbol]
        data_dict = {symbol: backtest_data}
        market_data = DataFrameMarketData(data=data_dict, date_col=self.date_column)
        
        # Set backtest output directory
        backtest_output_dir = os.path.join(self.output_dir, 'backtests') if self.output_dir else None
        
        # Create backtest ID
        backtest_id = f"walk_forward_fold_{fold_id}"
        
        # Create backtest runner and run backtest
        try:
            runner = BacktestRunner(output_dir=backtest_output_dir)
            
            runner.create_backtest(
                backtest_id=backtest_id,
                strategy=strategy,
                market_data=market_data,
                initial_capital=initial_capital
            )
            
            results = runner.run_backtest(backtest_id=backtest_id)
            
            # Store results summary
            summary = runner.get_backtest_summary(backtest_id)
            performance_metrics = results.get('performance_metrics', {})
            
            result_data = {
                'summary': summary,
                'performance_metrics': performance_metrics
            }
            
            logging.info(f"Backtest for fold {fold_id} completed successfully")
            return result_data
            
        except Exception as e:
            logging.error(f"Error running backtest for fold {fold_id}: {str(e)}")
            return None
    
    def _analyze_backtest_performance(self) -> Dict[str, Any]:
        """
        Analyze performance across all backtest folds.
        
        Returns:
            Dictionary of aggregated performance metrics
        """
        # Extract key metrics from all folds
        metrics_by_fold = {}
        fold_summaries = {}
        
        for fold_id, results in self.results['backtest_results'].items():
            metrics = results.get('performance_metrics', {})
            summary = results.get('summary', {})
            
            metrics_by_fold[fold_id] = metrics
            fold_summaries[fold_id] = summary
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_by_fold).T
        summary_df = pd.DataFrame(fold_summaries).T
        
        # Calculate aggregate statistics
        if not metrics_df.empty:
            avg_metrics = metrics_df.mean(axis=0).to_dict()
            std_metrics = metrics_df.std(axis=0).to_dict()
            min_metrics = metrics_df.min(axis=0).to_dict()
            max_metrics = metrics_df.max(axis=0).to_dict()
            
            aggregate_metrics = {
                'avg': avg_metrics,
                'std': std_metrics,
                'min': min_metrics,
                'max': max_metrics
            }
            
            self.results['aggregate_performance'] = aggregate_metrics
            
            # Log key metrics
            logging.info(f"Aggregate performance metrics:")
            
            # Use try/except since column names might vary
            try:
                logging.info(f"  Average annualized return: {avg_metrics.get('annualized_return_pct', 'N/A')}%")
                logging.info(f"  Average Sharpe ratio: {avg_metrics.get('sharpe_ratio', 'N/A')}")
                logging.info(f"  Average max drawdown: {avg_metrics.get('max_drawdown_pct', 'N/A')}%")
            except Exception as e:
                logging.warning(f"Error logging metrics: {str(e)}")
            
            return aggregate_metrics
        else:
            logging.warning("No performance metrics available for aggregation")
            return {}
    
    def _save_strategy_plots(self):
        """Generate and save strategy analysis plots."""
        if not self.output_dir or not self.results['backtest_results']:
            return
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Compare equity curves across folds
        try:
            # Collect equity curves
            equity_data = {}
            for fold_id, results in self.results['backtest_results'].items():
                summary = results.get('summary', {})
                if 'equity_curve' in summary:
                    equity_data[fold_id] = pd.DataFrame(summary['equity_curve'])
            
            if equity_data:
                # Create equity curve comparison
                fig, ax = plt.subplots(figsize=(15, 8))
                
                for fold_id, equity_df in equity_data.items():
                    if 'timestamp' in equity_df.columns and 'equity' in equity_df.columns:
                        # Normalize to start at 100%
                        initial_equity = equity_df['equity'].iloc[0]
                        equity_df['normalized'] = equity_df['equity'] / initial_equity * 100
                        
                        ax.plot(equity_df['timestamp'], equity_df['normalized'], 
                              label=f"Fold {fold_id}")
                
                ax.set_title('Normalized Equity Curves Across Folds')
                ax.set_xlabel('Date')
                ax.set_ylabel('Equity (%)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Save the plot
                equity_path = os.path.join(self.output_dir, 'equity_curves_comparison.png')
                fig.savefig(equity_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logging.info(f"Saved equity curves comparison to {equity_path}")
        
        except Exception as e:
            logging.error(f"Error creating equity curves plot: {str(e)}")
        
        # Create performance metrics comparison if available
        try:
            if 'aggregate_performance' in self.results:
                metrics = ['annualized_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']
                available_metrics = []
                
                # Find which metrics are actually available
                for metric in metrics:
                    if metric in self.results['aggregate_performance']['avg']:
                        available_metrics.append(metric)
                
                if available_metrics:
                    # Collect metrics by fold
                    metrics_by_fold = {}
                    for fold_id, results in self.results['backtest_results'].items():
                        metrics_dict = {}
                        perf_metrics = results.get('performance_metrics', {})
                        for metric in available_metrics:
                            metrics_dict[metric] = perf_metrics.get(metric, np.nan)
                        metrics_by_fold[fold_id] = metrics_dict
                    
                    metrics_df = pd.DataFrame(metrics_by_fold).T
                    
                    # Create plots for each metric
                    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(15, 4*len(available_metrics)))
                    if len(available_metrics) == 1:
                        axes = [axes]
                    
                    for i, metric in enumerate(available_metrics):
                        if metric in metrics_df.columns:
                            axes[i].bar(metrics_df.index, metrics_df[metric])
                            axes[i].set_title(f'{metric} by Fold')
                            axes[i].set_xlabel('Fold')
                            axes[i].set_ylabel(metric)
                            axes[i].grid(True, alpha=0.3)
                            
                            # Add average line
                            if metric in self.results['aggregate_performance']['avg']:
                                avg_value = self.results['aggregate_performance']['avg'][metric]
                                axes[i].axhline(avg_value, color='red', linestyle='--', 
                                              label=f'Average: {avg_value:.2f}')
                                axes[i].legend()
                    
                    fig.tight_layout()
                    metrics_path = os.path.join(self.output_dir, 'performance_by_fold.png')
                    fig.savefig(metrics_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    logging.info(f"Saved performance metrics plot to {metrics_path}")
        
        except Exception as e:
            logging.error(f"Error creating performance metrics plot: {str(e)}")
