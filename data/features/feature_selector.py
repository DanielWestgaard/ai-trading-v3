from data.processors.base_processor import BaseProcessor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os
import time
import matplotlib.pyplot as plt

import config.system_config as sys_config


class FeatureSelector(BaseProcessor):
    """
    Selects the most important features based on feature importance scores.
    Using Random Forest feature importance with time series cross-validation
    
    This component fits at the end of the pipeline, after normalization,
    to select the most relevant features for modeling.
    """
    
    def __init__(self, 
                 target_col='close_return',
                 selection_method='threshold',
                 n_features=None,
                 importance_threshold=0.01,
                 min_features=10,
                 max_features=100,
                 category_balance=True,
                 categories_to_keep=None,
                 lookback=1,
                 n_splits=5,
                 save_visualizations=True,
                 output_dir=None):
        """
        Initialize the feature selector.
        
        Args:
            target_col: Target column for prediction (typically 'close_return')
            selection_method: Method for feature selection ('threshold', 'top_n', 'cumulative')
            n_features: Number of features to select when using 'top_n' method
            importance_threshold: Minimum importance threshold when using 'threshold' method
            min_features: Minimum number of features to select
            max_features: Maximum number of features to select
            category_balance: Whether to balance selection across feature categories
            categories_to_keep: List of feature categories to always include
            lookback: Number of periods to look back when creating prediction target
            n_splits: Number of folds for time series cross-validation
            save_visualizations: Whether to save feature importance visualizations
            output_dir: Directory to save visualizations and metadata
        """
        self.target_col = target_col
        self.selection_method = selection_method
        self.n_features = n_features
        self.importance_threshold = importance_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.category_balance = category_balance
        self.categories_to_keep = categories_to_keep or ['price', 'transformed_price', 'volume']
        self.lookback = lookback
        self.n_splits = n_splits
        self.save_visualizations = save_visualizations
        self.output_dir = output_dir
        
        # State to be learned during fit
        self.selected_features = None
        self.importance_df = None
        self.performance_df = None
        self._feature_categories = {}
        
    def fit(self, data):
        """
        Analyze feature importance and select the most important features.
        
        Args:
            data: DataFrame with features to analyze
            
        Returns:
            self: The fitted selector
        """
        logging.info(f"Starting feature selection with target '{self.target_col}' using {self.selection_method} method")
        
        # First, validate the input data and target column
        self._validate_data(data)
        
        # 1. Analyze feature importance using the instance method
        try:
            logging.info(f"Analyzing feature importance for target '{self.target_col}' with {self.n_splits} splits")
            importance_df, fold_df = self.analyze_feature_importance(
                data=data,
                target_col=self.target_col,
                lookback=self.lookback,
                n_splits=self.n_splits
            )
            logging.info(f"Feature importance analysis complete. Found {len(importance_df)} features.")
        except Exception as e:
            logging.error(f"Feature importance analysis failed: {str(e)}")
            # Print more detailed traceback
            import traceback
            logging.error(traceback.format_exc())
            raise
        
        self.importance_df = importance_df
        self.performance_df = fold_df
        
        # 2. Save visualizations if requested
        if self.save_visualizations and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Feature importance bar chart
            viz_file = os.path.join(sys_config.CAPCOM_VIS_DATA_DIR, 'feature_importance.png')
            self.visualize_feature_importance(
                importance_df=importance_df,
                top_n=30,
                output_path=viz_file,
                show_figure=False
            )
            logging.info(f"Saved feature importance visualization to {viz_file}")
            
            # Category importance pie chart
            cat_viz_file = os.path.join(self.output_dir, 'category_importance.png')
            self.visualize_category_importance(
                importance_df=importance_df,
                output_path=cat_viz_file,
                show_figure=False
            )
            logging.info(f"Saved category importance visualization to {cat_viz_file}")
            
            # Save importance scores to CSV
            csv_file = os.path.join(self.output_dir, 'feature_importance.csv')
            importance_df.to_csv(csv_file, index=False)
            logging.info(f"Saved feature importance scores to {csv_file}")
            
            # Save cross-validation performance
            perf_file = os.path.join(self.output_dir, 'model_performance.csv')
            fold_df.to_csv(perf_file, index=False)
        
        # 3. Select features based on the specified method
        self.selected_features = self._select_features(importance_df)
        
        # 4. Log selection results
        n_selected = len(self.selected_features)
        logging.info(f"Selected {n_selected} features using '{self.selection_method}' method")
        
        # 5. Group selected features by category for analysis
        self._categorize_selected_features()
        for category, features in self._feature_categories.items():
            if features:
                logging.info(f"  - {category}: {len(features)} features")
        
        return self
    
    def transform(self, data):
        """
        Filter the data to include only selected features.
        
        Args:
            data: DataFrame with all features
            
        Returns:
            DataFrame with only selected features
        """
        if self.selected_features is None:
            logging.warning("No features have been selected. Run fit() first.")
            return data
        
        # Ensure all selected features are in the data
        missing_features = [f for f in self.selected_features if f not in data.columns]
        if missing_features:
            logging.warning(f"Some selected features are missing from the data: {missing_features}")
            # Keep only features that are actually in the data
            actual_features = [f for f in self.selected_features if f in data.columns]
        else:
            actual_features = self.selected_features
        
        # Select the columns
        result = data[actual_features].copy()
        
        # Make sure to include the date/timestamp column if it exists
        for date_col in ['Date', 'date', 'timestamp', 'Timestamp']:
            if date_col in data.columns and date_col not in result.columns:
                result[date_col] = data[date_col]
                break
        
        logging.info(f"Transformed data to include {len(actual_features)} selected features")
        return result
    
    def _select_features(self, importance_df):
        """
        Select features based on the specified method.
        
        Args:
            importance_df: DataFrame with feature importance scores
            
        Returns:
            List of selected feature names
        """
        if self.selection_method == 'top_n':
            # Select top N features
            n = self.n_features or min(self.max_features, len(importance_df) // 2)
            selected = importance_df.head(n)['feature'].tolist()
            
        elif self.selection_method == 'threshold':
            # Select features above importance threshold
            selected = importance_df[importance_df['importance'] >= self.importance_threshold]['feature'].tolist()
            
            # Ensure we have at least min_features
            if len(selected) < self.min_features:
                selected = importance_df.head(self.min_features)['feature'].tolist()
                
            # Cap at max_features
            if len(selected) > self.max_features:
                selected = importance_df.head(self.max_features)['feature'].tolist()
                
        elif self.selection_method == 'cumulative':
            # Select features until cumulative importance reaches threshold (e.g., 90%)
            threshold = self.importance_threshold if 0 < self.importance_threshold <= 1 else 0.9
            
            # Calculate cumulative importance
            sorted_df = importance_df.sort_values('importance', ascending=False).copy()
            total_importance = sorted_df['importance'].sum()
            sorted_df['cumulative'] = sorted_df['importance'].cumsum() / total_importance
            
            # Select features until threshold is reached
            selected = sorted_df[sorted_df['cumulative'] <= threshold]['feature'].tolist()
            
            # Always include at least one feature
            if not selected and len(sorted_df) > 0:
                selected = [sorted_df.iloc[0]['feature']]
                
            # Ensure we have at least min_features
            if len(selected) < self.min_features:
                selected = sorted_df.head(self.min_features)['feature'].tolist()
                
            # Cap at max_features
            if len(selected) > self.max_features:
                selected = sorted_df.head(self.max_features)['feature'].tolist()
                
        else:
            # Default to top 20% of features
            n = min(self.max_features, max(self.min_features, int(len(importance_df) * 0.2)))
            selected = importance_df.head(n)['feature'].tolist()
        
        # Apply category balancing if requested
        if self.category_balance and 'category' in importance_df.columns:
            selected = self._balance_feature_categories(importance_df, selected)
        
        return selected
    
    def _balance_feature_categories(self, importance_df, initial_selection):
        """
        Balance feature selection across categories.
        
        Args:
            importance_df: DataFrame with feature importance scores
            initial_selection: Initial list of selected features
            
        Returns:
            List of selected feature names with better category balance
        """
        # Make a copy of the input DataFrame and mark initially selected features
        df = importance_df.copy()
        df['selected'] = df['feature'].isin(initial_selection)
        
        # Count features by category
        category_counts = df[df['selected']].groupby('category').size()
        
        # Identify underrepresented categories
        mean_features_per_cat = max(1, len(initial_selection) // len(df['category'].unique()))
        underrepresented = [cat for cat in df['category'].unique() 
                           if cat not in category_counts or category_counts[cat] < mean_features_per_cat]
        
        # Add top features from underrepresented categories
        additional_features = []
        
        for cat in underrepresented:
            # If the category is in our categories_to_keep list, add more features
            cat_features = df[(df['category'] == cat) & (~df['selected'])].head(
                mean_features_per_cat if cat in self.categories_to_keep else max(1, mean_features_per_cat // 2)
            )['feature'].tolist()
            
            additional_features.extend(cat_features)
        
        # Combine with initial selection (limited by max_features)
        all_features = initial_selection + [f for f in additional_features if f not in initial_selection]
        
        if len(all_features) > self.max_features:
            # Calculate how many features to keep from each source
            keep_from_initial = int(self.max_features * (len(initial_selection) / len(all_features)))
            keep_from_additional = self.max_features - keep_from_initial
            
            # Take the top features from each source
            all_features = initial_selection[:keep_from_initial] + additional_features[:keep_from_additional]
        
        return all_features
    
    def _categorize_selected_features(self):
        """Categorize selected features for analysis"""
        if not hasattr(self, 'importance_df') or 'category' not in self.importance_df.columns:
            return
        
        # Reset feature categories
        self._feature_categories = {}
        
        # Group selected features by category
        selected_df = self.importance_df[self.importance_df['feature'].isin(self.selected_features)]
        
        for category, group in selected_df.groupby('category'):
            self._feature_categories[category] = group['feature'].tolist()
            
    def _validate_data(self, data):
        """
        Validate input data for feature selection.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data doesn't meet requirements for feature selection
        """
        # Check that we have data
        if data is None or len(data) == 0:
            raise ValueError("Input data is empty")
            
        # Check that we have enough data points
        if len(data) < 100:  # Arbitrary minimum for time series cross-validation
            logging.warning(f"Dataset has only {len(data)} rows. Feature selection may not be reliable.")
            
        # Check for target column
        if self.target_col not in data.columns:
            # Look for alternatives if possible
            possible_targets = [col for col in data.columns if 'return' in col.lower()]
            if possible_targets:
                original_target = self.target_col
                self.target_col = possible_targets[0]
                logging.warning(f"Target column '{original_target}' not found. Using '{self.target_col}' instead.")
            else:
                raise ValueError(f"Target column '{self.target_col}' not found and no alternative is available")
                
        # Check that target is numeric
        if not pd.api.types.is_numeric_dtype(data[self.target_col]):
            raise ValueError(f"Target column '{self.target_col}' must be numeric")
            
        # Check that we have sufficient numeric features
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        numeric_features = [col for col in numeric_cols if col != self.target_col]
        
        if len(numeric_features) < 2:
            raise ValueError(f"Not enough numeric features found. Need at least 2, found {len(numeric_features)}")
            
        # Log numeric feature count
        logging.info(f"Found {len(numeric_features)} numeric features for analysis")

    def analyze_feature_importance(self, data, target_col='close_return', lookback=1, n_splits=5, log_level=logging.INFO):
        """
        Analyze feature importance using a Random Forest model and time series cross-validation.
        
        Args:
            data: Either a DataFrame or path to a CSV file
            target_col: Target column to predict (e.g., 'close_return' for price returns)
            lookback: Number of periods to look back when creating prediction target
            n_splits: Number of folds for time series cross-validation
            log_level: Logging level (INFO, DEBUG, etc.)
            
        Returns:
            DataFrame with feature importance scores and performance metrics DataFrame
        """
        # Configure logging
        logging.getLogger().setLevel(log_level)
        
        # Start timing
        start_time = time.time()
        logging.info(f"Starting feature importance analysis (target: {target_col}, lookback: {lookback})")
        
        # Handle different input types
        if isinstance(data, str):
            # It's a file path
            logging.info(f"Loading data from file: {data}")
            try:
                df = pd.read_csv(data, parse_dates=['Date'])
                logging.debug(f"Successfully loaded data with 'Date' column")
            except KeyError:
                # Try lowercase column name
                logging.debug(f"'Date' column not found, trying 'date'")
                df = pd.read_csv(data, parse_dates=['date'])
        else:
            # Assume it's already a DataFrame
            logging.info(f"Using provided DataFrame")
            df = data.copy()
        
        logging.info(f"Data loaded successfully with shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
        
        # Log column information
        num_cols = len(df.columns)
        logging.debug(f"Column names (showing first 10 of {num_cols}): {', '.join(df.columns[:10])}")
        
        # Categorize columns
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])]
        raw_price_cols = [col for col in price_cols if 'original' in col.lower()]
        transformed_price_cols = [col for col in price_cols if any(x in col.lower() for x in ['return', 'log', 'pct'])]
        technical_cols = [col for col in df.columns if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bollinger'])]
        other_cols = [col for col in df.columns if col not in price_cols + technical_cols]
        
        logging.info(f"Column categories:")
        logging.info(f"  - Raw price columns: {len(raw_price_cols)}")
        logging.info(f"  - Transformed price columns: {len(transformed_price_cols)}")
        logging.info(f"  - Technical indicators: {len(technical_cols)}")
        logging.info(f"  - Other columns: {len(other_cols)}")
        
        # Ensure date column is properly formatted
        date_col = None
        for col_name in ['Date', 'date', 'timestamp', 'time']:
            if col_name in df.columns:
                date_col = col_name
                logging.debug(f"Found date column: {date_col}")
                break
        
        if date_col:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                logging.debug(f"Converting {date_col} to datetime")
                df[date_col] = pd.to_datetime(df[date_col])
        else:
            logging.warning("No date column found. Time series cross-validation may not work correctly.")
        
        # Check if target column exists
        if target_col not in df.columns:
            available_return_cols = [col for col in df.columns if 'return' in col.lower()]
            if available_return_cols:
                original_target = target_col
                target_col = available_return_cols[0]
                logging.warning(f"Target column '{original_target}' not found. Using '{target_col}' instead.")
            else:
                error_msg = f"Target column '{target_col}' not found and no alternative return columns available."
                logging.error(error_msg)
                raise ValueError(error_msg)
        
        # Log target column statistics
        logging.info(f"Target column: {target_col}")
        logging.info(f"Target statistics: mean={df[target_col].mean():.6f}, std={df[target_col].std():.6f}")
        logging.debug(f"Target range: min={df[target_col].min():.6f}, max={df[target_col].max():.6f}")
        
        # Create target variable (future returns)
        df[f'target_{target_col}'] = df[target_col].shift(-lookback)
        na_before = df[f'target_{target_col}'].isna().sum()
        df.dropna(subset=[f'target_{target_col}'], inplace=True)
        na_after = df[f'target_{target_col}'].isna().sum()
        logging.info(f"Created future target with lookback={lookback}. Dropped {na_before} rows with NaN targets.")
        logging.info(f"Target shape after NaN removal: {df.shape}")
        
        # Prepare features and target
        drop_cols = [col for col in [date_col, f'target_{target_col}', target_col] if col in df.columns]
        logging.debug(f"Dropping columns for model training: {drop_cols}")
        X = df.drop(drop_cols, axis=1)
        y = df[f'target_{target_col}']
        
        logging.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
        
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        
        if non_numeric_cols:
            logging.warning(f"Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        
        X = X[numeric_cols]  # Keep only numeric columns
        
        # Check for remaining NaN values
        na_counts = X.isna().sum()
        cols_with_na = na_counts[na_counts > 0]
        
        if not cols_with_na.empty:
            logging.warning(f"Found {len(cols_with_na)} columns with NaN values")
            for col, count in cols_with_na.items():
                logging.debug(f"  - {col}: {count} NaNs ({count/len(X)*100:.2f}%)")
        
        # Drop any remaining NaN values
        shape_before = X.shape
        X = X.dropna(axis=1, how='any')
        shape_after = X.shape
        if shape_before[1] != shape_after[1]:
            logging.warning(f"Dropped {shape_before[1] - shape_after[1]} columns with NaN values")
        
        # Get feature names after dropping NaN columns
        feature_names = X.columns.tolist()
        logging.info(f"Final feature set: {len(feature_names)} features")
        logging.debug(f"Feature list (first 10): {feature_names[:10]}")
        
        # Initialize time series cross-validation
        logging.info(f"Setting up time series cross-validation with {n_splits} splits")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize results tracking
        importance_scores = np.zeros(len(feature_names))
        fold_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            fold_start_time = time.time()
            logging.info(f"Processing fold {fold+1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            logging.debug(f"  Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            # Scale features
            logging.debug(f"  Scaling features")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            logging.debug(f"  Training Random Forest model")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            logging.debug(f"  Evaluating model performance")
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            fold_scores.append({
                'fold': fold + 1,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
            
            # Accumulate feature importance
            importance_scores += model.feature_importances_
            
            fold_time = time.time() - fold_start_time
            logging.info(f"  Fold {fold+1} results: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f} (took {fold_time:.2f}s)")
        
        # Average feature importance across folds
        importance_scores /= n_splits
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Log fold scores
        fold_df = pd.DataFrame(fold_scores)
        logging.info(f"Cross-validation performance summary:")
        logging.info(f"  MSE: {fold_df['mse'].mean():.6f} (±{fold_df['mse'].std():.6f})")
        logging.info(f"  MAE: {fold_df['mae'].mean():.6f} (±{fold_df['mae'].std():.6f})")
        logging.info(f"  R²: {fold_df['r2'].mean():.6f} (±{fold_df['r2'].std():.6f})")
        
        # Log top features
        logging.info("Top 20 features by importance:")
        for i, row in importance_df.head(20).iterrows():
            logging.info(f"  {i+1}. {row['feature']}: {row['importance']:.6f}")
        
        # Calculate feature importance by category
        def get_feature_category(feature_name):
            feature_lower = feature_name.lower()
            if 'original' in feature_lower:
                return 'Raw Price'
            elif any(x in feature_lower for x in ['open', 'high', 'low', 'close']):
                if any(x in feature_lower for x in ['return', 'log', 'pct']):
                    return 'Transformed Price'
                else:
                    return 'Price Other'
            elif 'volume' in feature_lower:
                return 'Volume'
            elif any(x in feature_lower for x in ['sma', 'ema']):
                return 'Moving Averages'
            elif any(x in feature_lower for x in ['rsi', 'macd', 'stoch']):
                return 'Oscillators'
            elif any(x in feature_lower for x in ['atr', 'volatility']):
                return 'Volatility'
            elif any(x in feature_lower for x in ['doji', 'hammer', 'engulfing']):
                return 'Patterns'
            elif any(x in feature_lower for x in ['day', 'hour', 'month', 'session']):
                return 'Time'
            else:
                return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(get_feature_category)
        category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        logging.info("Feature importance by category:")
        for category, importance in category_importance.items():
            logging.info(f"  {category}: {importance:.6f} ({importance*100:.2f}%)")
        
        total_time = time.time() - start_time
        logging.info(f"Feature importance analysis completed in {total_time:.2f} seconds")
        
        return importance_df, fold_df

    def visualize_feature_importance(self, importance_df, top_n=30, output_path=None, show_figure=True):
        """
        Visualize feature importance with category coloring.
        
        Args:
            importance_df: DataFrame with feature importance scores
            top_n: Number of top features to show
            output_path: Path to save visualization
            show_figure: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        logging.info(f"Creating feature importance visualization for top {top_n} features")
        
        # Get top N features
        top_features = importance_df.head(top_n).copy()
        
        # Determine feature categories if not already present
        if 'category' not in top_features.columns:
            def categorize_feature(feature_name):
                feature_lower = feature_name.lower()
                if 'original' in feature_lower:
                    return 'Raw Price'
                elif any(x in feature_lower for x in ['open', 'high', 'low', 'close']):
                    if any(x in feature_lower for x in ['return', 'log', 'pct']):
                        return 'Transformed Price'
                    else:
                        return 'Price Other'
                elif 'volume' in feature_lower:
                    return 'Volume'
                elif any(x in feature_lower for x in ['sma', 'ema']):
                    return 'Moving Averages'
                elif any(x in feature_lower for x in ['rsi', 'macd', 'stoch']):
                    return 'Oscillators'
                elif any(x in feature_lower for x in ['atr', 'volatility']):
                    return 'Volatility'
                elif any(x in feature_lower for x in ['doji', 'hammer', 'engulfing']):
                    return 'Patterns'
                elif any(x in feature_lower for x in ['day', 'hour', 'month']):
                    return 'Time'
                else:
                    return 'Other'
            
            top_features['category'] = top_features['feature'].apply(categorize_feature)
        
        # Log category breakdown of top features
        category_counts = top_features['category'].value_counts()
        logging.info(f"Top {top_n} features by category:")
        for category, count in category_counts.items():
            logging.info(f"  {category}: {count} features")
        
        # Define colors for categories
        category_colors = {
            'Raw Price': '#1f77b4',
            'Transformed Price': '#ff7f0e',
            'Price Other': '#2ca02c',
            'Volume': '#d62728',
            'Moving Averages': '#9467bd',
            'Oscillators': '#8c564b',
            'Volatility': '#e377c2',
            'Patterns': '#7f7f7f',
            'Time': '#bcbd22',
            'Other': '#17becf'
        }
        
        # Create figure
        plt.figure(figsize=(12, max(8, top_n * 0.3)))
        
        # Plot horizontal bar chart
        bars = plt.barh(
            y=top_features['feature'],
            width=top_features['importance'],
            color=[category_colors.get(cat, '#7f7f7f') for cat in top_features['category']]
        )
        
        # Add category labels to bars
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.001, 
                bar.get_y() + bar.get_height()/2, 
                top_features.iloc[i]['category'],
                va='center', 
                fontsize=8
            )
        
        # Add legend
        legend_categories = sorted(set(top_features['category']))
        handles = [plt.Rectangle((0,0), 1, 1, color=category_colors.get(cat, '#7f7f7f')) 
                for cat in legend_categories]
        plt.legend(handles, legend_categories, loc='lower right')
        
        # Add labels and title
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Features by Importance')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved feature importance visualization to {output_path}")
        
        # Show or close
        if show_figure:
            logging.info("Displaying feature importance visualization")
        else:
            plt.close()
        
        # Always return the plot object
        return plt

    def visualize_category_importance(self, importance_df, output_path=None, show_figure=True):
        """
        Create a pie chart of feature importance by category.
        
        Args:
            importance_df: DataFrame with feature importance scores and categories
            output_path: Path to save visualization
            show_figure: Whether to display the figure
            
        Returns:
            Matplotlib figure object
        """
        logging.info("Creating feature importance by category visualization")
        
        # Add category if not present
        if 'category' not in importance_df.columns:
            def categorize_feature(feature_name):
                feature_lower = feature_name.lower()
                if 'original' in feature_lower:
                    return 'Raw Price'
                elif any(x in feature_lower for x in ['open', 'high', 'low', 'close']):
                    if any(x in feature_lower for x in ['return', 'log', 'pct']):
                        return 'Transformed Price'
                    else:
                        return 'Price Other'
                elif 'volume' in feature_lower:
                    return 'Volume'
                elif any(x in feature_lower for x in ['sma', 'ema']):
                    return 'Moving Averages'
                elif any(x in feature_lower for x in ['rsi', 'macd', 'stoch']):
                    return 'Oscillators'
                elif any(x in feature_lower for x in ['atr', 'volatility']):
                    return 'Volatility'
                elif any(x in feature_lower for x in ['doji', 'hammer', 'engulfing']):
                    return 'Patterns'
                elif any(x in feature_lower for x in ['day', 'hour', 'month']):
                    return 'Time'
                else:
                    return 'Other'
                    
            importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        # Aggregate by category
        category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        # Define colors
        category_colors = {
            'Raw Price': '#1f77b4',
            'Transformed Price': '#ff7f0e',
            'Price Other': '#2ca02c',
            'Volume': '#d62728',
            'Moving Averages': '#9467bd',
            'Oscillators': '#8c564b',
            'Volatility': '#e377c2', 
            'Patterns': '#7f7f7f',
            'Time': '#bcbd22',
            'Other': '#17becf'
        }
        
        # Extract colors in the correct order
        colors = [category_colors.get(cat, '#7f7f7f') for cat in category_importance.index]
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            category_importance, 
            labels=category_importance.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        # Styling
        plt.axis('equal')
        plt.title('Feature Importance by Category')
        
        # Make percentage labels more readable
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
            autotext.set_color('white')
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved category importance visualization to {output_path}")
        
        # Show or close
        if show_figure:
            logging.info("Displaying category importance visualization")
        else:
            plt.close()
        
        # Always return the plot object
        return plt