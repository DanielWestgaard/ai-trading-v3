import os
import pandas as pd
import logging

from data.loaders.broker_loader import CapitalComLoader
from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer
from data.features.feature_selector import FeatureSelector  # Import the new class
import config.data_config as data_config
import config.system_config as sys_config
from utils import data_utils


class DataPipeline:
    """Coordinates the entire data processing pipeline with empirical feature approach"""
    
    def __init__(self, 
                 feature_treatment_mode='advanced',
                 price_transform_method='returns',
                 normalization_method='zscore',
                 feature_selection_method='threshold',
                 feature_importance_threshold=0.01,
                 target_column='close_return'):
        """
        Initialize the data pipeline with configuration.
        
        Args:
            feature_treatment_mode: How to handle features ('basic', 'advanced', 'hybrid')
            price_transform_method: How to transform prices ('returns', 'log', 'multi', 'none')
            normalization_method: Method for normalization ('zscore', 'minmax', 'robust')
            feature_selection_method: Method for feature selection ('threshold', 'top_n', 'cumulative')
            feature_importance_threshold: Importance threshold for feature selection
            target_column: Target column for prediction (influences feature selection)
        """
        # Configure the cleaner
        self.cleaner = DataCleaner(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
        )
        
        # Configure the feature generator
        self.feature_generator = FeatureGenerator()
        
        # Configure the feature preparator - always preserve original prices
        self.feature_preparator = FeaturePreparator(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col='Date',
            preserve_original_prices=True,  # Always True for empirical approach
            price_transform_method=price_transform_method,
            treatment_mode=feature_treatment_mode
        )
        
        # Configure the normalizer
        self.normalizer = DataNormalizer(other_method=normalization_method)
        
        # Configure the feature selector
        self.feature_selector = FeatureSelector(
            target_col=target_column,
            selection_method=feature_selection_method,
            importance_threshold=feature_importance_threshold,
            save_visualizations=True,
            output_dir=os.path.join(sys_config.CAPCOM_PROCESSED_DATA_DIR, 'feature_selection')
        )
        
    def run(self, source=None, target_path=sys_config.CAPCOM_PROCESSED_DATA_DIR, 
            raw_data=data_config.TESTING_RAW_FILE, save_intermediate=False,
            run_feature_selection=True):  # Added parameter
        """Execute the full pipeline with empirical feature approach"""
        # Configure logging
        logging.info("Starting data pipeline execution with empirical feature approach")
        
        # 1. Load data
        logging.info("Loading raw data")
        raw_data_df = pd.read_csv(raw_data, parse_dates=['Date'])
        logging.info(f"Loaded raw data with shape: {raw_data_df.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(raw_data_df, "raw", "raw_data.csv")
        
        # 2. Clean data
        logging.info("Cleaning data")
        self.cleaner.fit(raw_data_df)
        clean_data = self.cleaner.transform(raw_data_df)
        logging.info(f"Cleaned data shape: {clean_data.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(clean_data, "clean", "clean_data.csv")
        
        # 3. Generate features
        logging.info("Generating features")
        self.feature_generator.fit(clean_data)
        featured_data = self.feature_generator.transform(clean_data)
        logging.info(f"Generated features. New shape: {featured_data.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(featured_data, "featured", "featured_data.csv")
        
        # 4. Prepare features
        logging.info("Preparing features for modeling (keeping raw OHLC)")
        self.feature_preparator.fit(featured_data)
        prepared_data = self.feature_preparator.transform(featured_data)
        logging.info(f"Prepared features. New shape: {prepared_data.shape}")
        
        # Log the price-related columns for reference
        price_cols = [col for col in prepared_data.columns 
                     if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])]
        logging.info(f"Price-related columns ({len(price_cols)}): {', '.join(price_cols[:10])}...")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(prepared_data, "prepared", "prepared_data.csv")
        
        # 5. Normalize
        logging.info("Normalizing data")
        self.normalizer.fit(prepared_data)
        normalized_data = self.normalizer.transform(prepared_data)
        logging.info(f"Normalized data. Shape: {normalized_data.shape}")
        
        if save_intermediate and target_path:
            data_utils.save_data_file(normalized_data, "normalized", "normalized_data.csv")
            
        # 6. Feature selection (optional)
        selected_data = normalized_data
        if run_feature_selection:
            logging.info("Performing feature selection")
            self.feature_selector.fit(normalized_data)
            selected_data = self.feature_selector.transform(normalized_data)
            logging.info(f"Selected features. Final shape: {selected_data.shape}")
        
        # 7. Save processed data
        if target_path:
            file_path = data_utils.save_financial_data(selected_data, "processed", raw_filename=raw_data)
            # Also save a metadata file with column descriptions
            self._save_feature_metadata(selected_data, target_path, file_path)
            logging.info(f"Saved final processed data to {file_path}")
            
            # Save selected features list for future reference
            if run_feature_selection and hasattr(self.feature_selector, 'selected_features'):
                features_file = os.path.join(os.path.dirname(file_path), 'selected_features.txt')
                with open(features_file, 'w') as f:
                    f.write('\n'.join(self.feature_selector.selected_features))
                logging.info(f"Saved list of {len(self.feature_selector.selected_features)} selected features to {features_file}")
        
        # Return the fully processed data and path
        return selected_data, file_path
    
    def _save_feature_metadata(self, df, target_path, processed_filename):
        """Save metadata about the features for reference"""
        # Categorize features
        feature_categories = {
            'raw_price': [],
            'transformed_price': [],
            'volume': [],
            'technical': [],
            'volatility': [],
            'patterns': [],
            'time': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['open', 'high', 'low', 'close']):
                if any(x in col_lower for x in ['return', 'log', 'pct']):
                    feature_categories['transformed_price'].append(col)
                else:
                    feature_categories['raw_price'].append(col)
            elif 'volume' in col_lower:
                feature_categories['volume'].append(col)
            elif any(x in col_lower for x in ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stoch']):
                feature_categories['technical'].append(col)
            elif any(x in col_lower for x in ['atr', 'volatility']):
                feature_categories['volatility'].append(col)
            elif any(x in col_lower for x in ['doji', 'hammer', 'engulfing', 'support', 'resistance']):
                feature_categories['patterns'].append(col)
            elif any(x in col_lower for x in ['day', 'hour', 'month', 'session']):
                feature_categories['time'].append(col)
        
        # Create metadata DataFrame
        metadata = []
        for category, cols in feature_categories.items():
            for col in cols:
                metadata.append({
                    'column': col,
                    'category': category,
                    'nan_count': df[col].isna().sum(),
                    'nan_pct': (df[col].isna().sum() / len(df)) * 100
                })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df = metadata_df.sort_values(['category', 'column'])
        
        filename = os.path.basename(processed_filename)
        # Save metadata
        metadata_path = os.path.join(target_path, f"meta_{filename}")
        metadata_df.to_csv(metadata_path, index=False)
        logging.info(f"Saved feature metadata to {metadata_path}")
        
        # Log feature category summary
        category_summary = metadata_df.groupby('category').size().to_dict()
        logging.info(f"Feature category summary: {category_summary}")
        
        return metadata_df