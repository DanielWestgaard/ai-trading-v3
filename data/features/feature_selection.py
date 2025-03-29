import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# This code would be added later when implementing feature selection


def analyze_feature_importance(data_file, target_col='close_return', lookback=1, n_splits=5):
    """
    Analyze feature importance using a Random Forest model and time series cross-validation.
    
    Args:
        data_file: Path to the processed data CSV
        target_col: Target column to predict (e.g., 'close_return' for price returns)
        lookback: Number of periods to look back when creating prediction target
        n_splits: Number of folds for time series cross-validation
        
    Returns:
        DataFrame with feature importance scores
    """
    # Load data
    df = pd.read_csv(data_file, parse_dates=['Date'])
    logging.info(f"Loaded data with shape: {df.shape}")
    
    # Create target variable (future returns)
    df[f'target_{target_col}'] = df[target_col].shift(-lookback)
    df.dropna(subset=[f'target_{target_col}'], inplace=True)
    
    # Prepare features and target
    X = df.drop(['Date', f'target_{target_col}', target_col], axis=1)
    y = df[f'target_{target_col}']
    
    # Drop any remaining NaN values
    X = X.dropna(axis=1, how='any')
    
    # Get feature names after dropping NaN columns
    feature_names = X.columns.tolist()
    logging.info(f"Using {len(feature_names)} features after dropping NaN columns")
    
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize results tracking
    importance_scores = np.zeros(len(feature_names))
    fold_scores = []
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        fold_scores.append({
            'fold': fold + 1,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
        
        # Accumulate feature importance
        importance_scores += model.feature_importances_
    
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
    logging.info(f"Cross-validation performance:")
    logging.info(f"MSE: {fold_df['mse'].mean():.6f} (±{fold_df['mse'].std():.6f})")
    logging.info(f"MAE: {fold_df['mae'].mean():.6f} (±{fold_df['mae'].std():.6f})")
    logging.info(f"R²: {fold_df['r2'].mean():.6f} (±{fold_df['r2'].std():.6f})")
    
    # Log top features
    logging.info("Top 20 features by importance:")
    for i, row in importance_df.head(20).iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.6f}")
    
    return importance_df, fold_df


def visualize_feature_importance(importance_df, top_n=30, output_path=None):
    """
    Visualize feature importance with category coloring.
    
    Args:
        importance_df: DataFrame with feature importance scores
        top_n: Number of top features to show
        output_path: Path to save visualization
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Determine feature categories
    def categorize_feature(feature_name):
        feature_lower = feature_name.lower()
        if any(x in feature_lower for x in ['open', 'high', 'low', 'close']):
            if any(x in feature_lower for x in ['return', 'log', 'pct']):
                return 'Transformed Price'
            else:
                return 'Raw Price'
        elif 'volume' in feature_lower:
            return 'Volume'
        elif any(x in feature_lower for x in ['sma', 'ema', 'rsi', 'macd']):
            return 'Technical'
        elif any(x in feature_lower for x in ['atr', 'volatility']):
            return 'Volatility'
        elif any(x in feature_lower for x in ['doji', 'hammer', 'engulfing']):
            return 'Patterns'
        elif any(x in feature_lower for x in ['day', 'hour', 'month']):
            return 'Time'
        else:
            return 'Other'
    
    top_features['category'] = top_features['feature'].apply(categorize_feature)
    
    # Define colors for categories
    category_colors = {
        'Raw Price': '#1f77b4',
        'Transformed Price': '#ff7f0e',
        'Volume': '#2ca02c',
        'Technical': '#d62728',
        'Volatility': '#9467bd',
        'Patterns': '#8c564b',
        'Time': '#e377c2',
        'Other': '#7f7f7f'
    }
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot horizontal bar chart
    bars = plt.barh(
        y=top_features['feature'],
        width=top_features['importance'],
        color=[category_colors[cat] for cat in top_features['category']]
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
    handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in category_colors.values()]
    plt.legend(handles, category_colors.keys(), loc='lower right')
    
    # Add labels and title
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Features by Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved feature importance visualization to {output_path}")
    else:
        plt.show()
    
    return plt


# Function to select optimal feature subset based on importance
def select_optimal_features(importance_df, data_file, target_col='close_return', 
                           feature_count_range=range(10, 101, 10), 
                           lookback=1, n_splits=5):
    """
    Find the optimal number of features using importance ranking.
    
    Args:
        importance_df: DataFrame with feature importance scores
        data_file: Path to processed data file
        target_col: Target column to predict
        feature_count_range: Range of feature counts to test
        lookback: Prediction horizon
        n_splits: Number of cross-validation splits
        
    Returns:
        DataFrame with performance metrics for different feature counts
    """
    # Load data
    df = pd.read_csv(data_file, parse_dates=['Date'])
    
    # Create target variable
    df[f'target_{target_col}'] = df[target_col].shift(-lookback)
    df.dropna(subset=[f'target_{target_col}'], inplace=True)
    
    # Initialize results
    results = []
    
    # Get all feature names in order of importance
    all_features = importance_df['feature'].tolist()
    
    # Loop through different feature counts
    for n_features in feature_count_range:
        # Select top N features
        selected_features = all_features[:n_features]
        
        # Prepare features and target
        X = df[selected_features]
        y = df[f'target_{target_col}']
        
        # Initialize time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Track performance metrics
        fold_mse = []
        fold_mae = []
        fold_r2 = []
        
        # Perform cross-validation
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            fold_mse.append(mean_squared_error(y_test, y_pred))
            fold_mae.append(mean_absolute_error(y_test, y_pred))
            fold_r2.append(r2_score(y_test, y_pred))
        
        # Record results
        results.append({
            'n_features': n_features,
            'mse': np.mean(fold_mse),
            'mse_std': np.std(fold_mse),
            'mae': np.mean(fold_mae),
            'mae_std': np.std(fold_mae),
            'r2': np.mean(fold_r2),
            'r2_std': np.std(fold_r2)
        })
        
        logging.info(f"Tested {n_features} features: MSE={np.mean(fold_mse):.6f}, R²={np.mean(fold_r2):.6f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal feature count
    best_idx = results_df['r2'].idxmax()
    optimal_n_features = results_df.iloc[best_idx]['n_features']
    optimal_features = all_features[:int(optimal_n_features)]
    
    logging.info(f"Optimal feature count: {optimal_n_features}")
    logging.info(f"Best R²: {results_df.iloc[best_idx]['r2']:.6f}")
    
    return results_df, optimal_features