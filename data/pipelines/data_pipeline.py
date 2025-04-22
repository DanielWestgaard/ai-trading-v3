import os
import pandas as pd
import logging

from data.loaders.broker_loader import CapitalComLoader
from data.processors.cleaner import DataCleaner
from data.features.feature_generator import FeatureGenerator
from data.features.feature_preparator import FeaturePreparator
from data.processors.normalizer import DataNormalizer
from data.features.feature_selector import FeatureSelector
import config.constants.data_config as data_config
import config.constants.system_config as sys_config
from utils import data_utils


class DataPipeline:
    """Coordinates the entire data processing pipeline with improved file organization"""
    
    def __init__(self, 
                 feature_treatment_mode='advanced',
                 price_transform_method='returns',
                 normalization_method='zscore',
                 feature_selection_method='threshold',
                 feature_importance_threshold=0.01,
                 target_column='close_return',
                 timestamp_column='Date',  # Add timestamp_column parameter
                 preserve_target=True):
        """
        Initialize the data pipeline with configuration.
        
        Args:
            feature_treatment_mode: How to handle features ('basic', 'advanced', 'hybrid')
            price_transform_method: How to transform prices ('returns', 'log', 'multi', 'none')
            normalization_method: Method for normalization ('zscore', 'minmax', 'robust')
            feature_selection_method: Method for feature selection ('threshold', 'top_n', 'cumulative')
            feature_importance_threshold: Importance threshold for feature selection
            target_column: Target column for prediction (influences feature selection)
            timestamp_column: Column name for timestamp/date data
            preserve_target: Whether to always preserve the target column
        """
        # Store timestamp column name
        self.timestamp_column = timestamp_column
        
        # Configure the cleaner
        self.cleaner = DataCleaner(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col=self.timestamp_column,
        )
        
        # Configure the feature generator
        self.feature_generator = FeatureGenerator(
            timestamp_col=self.timestamp_column
        )
        
        # Configure the feature preparator - always preserve original prices
        self.feature_preparator = FeaturePreparator(
            price_cols=['Open', 'High', 'Low', 'Close'],
            volume_col='Volume',
            timestamp_col=self.timestamp_column,
            preserve_original_prices=True,
            price_transform_method=price_transform_method,
            treatment_mode=feature_treatment_mode
        )
        
        # Configure the normalizer
        self.normalizer = DataNormalizer(other_method=normalization_method)
        
        # Feature selector will be configured during run() when we know the output directory
        self.feature_selector = None
        self.feature_selection_method = feature_selection_method
        self.feature_importance_threshold = feature_importance_threshold
        self.target_column = target_column
        self.preserve_target = preserve_target
        
    def run(self, target_path=sys_config.CAPCOM_PROCESSED_DATA_DIR, 
                raw_data=data_config.TESTING_RAW_FILE, save_intermediate=False,
                run_feature_selection=True):
        """Execute the full pipeline with improved file organization"""
        # Configure logging
        logging.info("Starting data pipeline execution with empirical feature approach")
        
        # 1. Load data
        logging.info(f"Loading raw data from {raw_data}")
        try:
            # Use the configured timestamp column for parsing dates
            raw_data_df = pd.read_csv(raw_data, parse_dates=[self.timestamp_column])
            logging.info(f"Loaded raw data with shape: {raw_data_df.shape}")
        except KeyError as e:
            # Try a fallback approach if the configured column isn't found
            logging.warning(f"Column {self.timestamp_column} not found in CSV. Attempting to load without date parsing.")
            raw_data_df = pd.read_csv(raw_data)
            
            # Look for date/time columns
            date_cols = [col for col in raw_data_df.columns if any(
                date_str in col.lower() for date_str in ['date', 'time', 'timestamp'])]
            
            if date_cols:
                # Update the timestamp column to what was found
                self.timestamp_column = date_cols[0]
                logging.info(f"Using '{self.timestamp_column}' as timestamp column")
                
                # Parse the found date column
                raw_data_df[self.timestamp_column] = pd.to_datetime(raw_data_df[self.timestamp_column])
                
                # Update all components to use this column
                self.cleaner.timestamp_col = self.timestamp_column.lower()
                self.feature_generator.timestamp_col = self.timestamp_column.lower()
                self.feature_preparator.timestamp_col = self.timestamp_column.lower()
            else:
                logging.warning("No date/time column found in the data")
        
        # Extract metadata from raw filename for consistent naming
        raw_metadata = data_utils.extract_file_metadata(raw_data)
        if not raw_metadata:
            logging.warning("Could not extract metadata from raw filename. Using default naming.")
            instrument = "UNKNOWN"
            timeframe = "UNKNOWN"
            date_range = "UNKNOWN"
        else:
            instrument = raw_metadata['instrument']
            timeframe = raw_metadata['timeframe']
            date_range = f"{raw_metadata['start_date']}_{raw_metadata['end_date']}"
            logging.info(f"Processing {instrument} {timeframe} data from {raw_metadata['start_date']} to {raw_metadata['end_date']}")
        
        # Ensure target directories exist
        os.makedirs(target_path, exist_ok=True)
        
        if save_intermediate:
            # Save raw data with consistent naming
            raw_save_path = os.path.join(sys_config.CAPCOM_RAW_DATA_DIR, f"raw_{instrument}_{timeframe}_{date_range}.csv")
            raw_data_df.to_csv(raw_save_path, index=False)
            logging.info(f"Saved raw data to {raw_save_path}")
        
        # 2. Clean data
        logging.info("Cleaning data")
        self.cleaner.fit(raw_data_df)
        clean_data = self.cleaner.transform(raw_data_df)
        logging.info(f"Cleaned data shape: {clean_data.shape}")
        
        if save_intermediate:
            clean_file_path = os.path.join(target_path, 'clean', f"clean_{instrument}_{timeframe}_{date_range}.csv")
            os.makedirs(os.path.dirname(clean_file_path), exist_ok=True)
            clean_data.to_csv(clean_file_path, index=False)
            logging.info(f"Saved cleaned data to {clean_file_path}")
        
        # 3. Generate features
        logging.info("Generating features")
        self.feature_generator.fit(clean_data)
        featured_data = self.feature_generator.transform(clean_data)
        logging.info(f"Generated features. New shape: {featured_data.shape}")
        
        if save_intermediate:
            featured_file_path = os.path.join(target_path, 'features', f"featured_{instrument}_{timeframe}_{date_range}.csv")
            os.makedirs(os.path.dirname(featured_file_path), exist_ok=True)
            featured_data.to_csv(featured_file_path, index=False)
            logging.info(f"Saved featured data to {featured_file_path}")
        
        # 4. Prepare features
        logging.info("Preparing features for modeling")
        self.feature_preparator.fit(featured_data)
        prepared_data = self.feature_preparator.transform(featured_data)
        logging.info(f"Prepared features. New shape: {prepared_data.shape}")
        
        if save_intermediate:
            prepared_file_path = os.path.join(target_path, 'prepared', f"prepared_{instrument}_{timeframe}_{date_range}.csv")
            os.makedirs(os.path.dirname(prepared_file_path), exist_ok=True)
            prepared_data.to_csv(prepared_file_path, index=False)
            logging.info(f"Saved prepared data to {prepared_file_path}")
        
        # 5. Normalize
        logging.info("Normalizing data")
        self.normalizer.fit(prepared_data)
        normalized_data = self.normalizer.transform(prepared_data)
        logging.info(f"Normalized data. Shape: {normalized_data.shape}")
        
        if save_intermediate:
            normalized_file_path = os.path.join(target_path, 'normalized', f"normalized_{instrument}_{timeframe}_{date_range}.csv")
            os.makedirs(os.path.dirname(normalized_file_path), exist_ok=True)
            normalized_data.to_csv(normalized_file_path, index=False)
            logging.info(f"Saved normalized data to {normalized_file_path}")
        
        # 6. Save processed data first so we have the filename for feature selection outputs
        processed_filename = f"processed_{instrument}_{timeframe}_{date_range}.csv"
        processed_file_path = os.path.join(target_path, processed_filename)
        
        # 7. Feature selection (optional)
        selected_data = normalized_data
        if run_feature_selection:
            # Create feature directories
            features_dir = os.path.join(target_path, 'features')
            os.makedirs(features_dir, exist_ok=True)
            
            # Configure feature selector with consistent file naming
            self.feature_selector = FeatureSelector(
                target_col=self.target_column,
                selection_method=self.feature_selection_method,
                importance_threshold=self.feature_importance_threshold,
                save_visualizations=True,
                output_dir=features_dir,
                processed_file_path=processed_file_path,
                preserve_target=self.preserve_target
            )
            
            logging.info("Performing feature selection")
            self.feature_selector.fit(normalized_data)
            selected_data = self.feature_selector.transform(normalized_data)
            logging.info(f"Selected features. Final shape: {selected_data.shape}")
            
            # Ensure target column is preserved
            if self.preserve_target and self.target_column not in selected_data.columns:
                if self.target_column in normalized_data.columns:
                    logging.warning(f"Target column '{self.target_column}' was removed during feature selection. Re-adding it.")
                    
                    # Handle case where selected_data is empty
                    if selected_data.empty and not normalized_data.empty:
                        # Create a dataframe with just the target column
                        selected_data = pd.DataFrame({
                            self.target_column: normalized_data[self.target_column]
                        })
                    else:
                        # Just add the column to existing dataframe
                        selected_data[self.target_column] = normalized_data[self.target_column]
                else:
                    logging.error(f"Target column '{self.target_column}' not found in normalized data. Cannot preserve it.")
        
        # 8. Final check for proper data types in price and numeric columns
        try:
            # Identify financial columns that should be numeric
            financial_cols = [col for col in selected_data.columns if any(
                x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume', 'price', 'return'])]
            
            # Convert them to numeric with coercion
            for col in financial_cols:
                if not pd.api.types.is_numeric_dtype(selected_data[col]):
                    logging.warning(f"Column {col} is not numeric in final output. Converting to numeric.")
                    selected_data[col] = pd.to_numeric(selected_data[col], errors='coerce').fillna(0)
            
            # Also ensure other numeric columns have proper numeric dtypes
            numeric_cols = selected_data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if selected_data[col].dtype.kind not in 'ifc':  # integer, float, complex
                    selected_data[col] = pd.to_numeric(selected_data[col], errors='coerce').fillna(0)
        except Exception as e:
            logging.error(f"Error during final type conversion: {str(e)}")
        
        # Save the processed data
        selected_data.to_csv(processed_file_path, index=False)
        logging.info(f"Saved final processed data to {processed_file_path}")
        
        # 9. Save metadata about features
        try:
            metadata_path = data_utils.get_derived_file_path(
                processed_file_path, 
                'meta', 
                extension='csv'
            )
            
            metadata_df = self._create_feature_metadata(selected_data)
            metadata_df.to_csv(metadata_path, index=False)
            logging.info(f"Saved feature metadata to {metadata_path}")
        except Exception as e:
            logging.error(f"Error creating feature metadata: {str(e)}")
        
        # 10. Save selected features list if feature selection was run
        if run_feature_selection and hasattr(self.feature_selector, 'selected_features') and self.feature_selector.selected_features:
            try:
                features_file = data_utils.get_derived_file_path(
                    processed_file_path,
                    'selected_features',
                    sub_dir='features',
                    extension='txt'
                )
                
                os.makedirs(os.path.dirname(features_file), exist_ok=True)
                with open(features_file, 'w') as f:
                    if isinstance(self.feature_selector.selected_features, list):
                        f.write('\n'.join(self.feature_selector.selected_features))
                    else:
                        # If somehow it's not a list, convert to string first
                        f.write(str(self.feature_selector.selected_features))
                    
                logging.info(f"Saved list of selected features to {features_file}")
            except Exception as e:
                logging.error(f"Error saving selected features list: {str(e)}")
        
        # Return the fully processed data and path
        return selected_data, processed_file_path
    
    def _create_feature_metadata(self, df):
        """Create metadata about the features for reference"""
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
        
        try:
            # Process each column and categorize it
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
            
            # Create a DataFrame with the metadata
            metadata_df = pd.DataFrame(metadata)
            
            # Sort by category and column name
            if not metadata_df.empty:
                metadata_df = metadata_df.sort_values(['category', 'column'])
                
                # Log feature category summary
                category_summary = metadata_df.groupby('category').size().to_dict()
                logging.info(f"Feature category summary: {category_summary}")
            else:
                # Create a minimal metadata DataFrame with required columns if no features were categorized
                metadata_df = pd.DataFrame(columns=['column', 'category', 'nan_count', 'nan_pct'])
            
            return metadata_df
            
        except Exception as e:
            # If anything fails, return a minimal valid DataFrame
            logging.error(f"Error creating feature metadata: {str(e)}")
            return pd.DataFrame({
                'column': df.columns.tolist(),
                'category': ['other'] * len(df.columns),
                'nan_count': [0] * len(df.columns),
                'nan_pct': [0.0] * len(df.columns)
            })