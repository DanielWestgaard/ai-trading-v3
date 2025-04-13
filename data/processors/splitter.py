import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

from data.processors.base_processor import BaseProcessor


class TimeSeriesSplitter(BaseProcessor):
    """
    Time Series Split Processor that respects temporal order.
    Designed to work with your existing data pipeline.
    """
    
    def __init__(self, 
                 train_period: Union[str, int, timedelta],
                 test_period: Union[str, int, timedelta],
                 step_size: Union[str, int, timedelta] = None,
                 max_train_size: Optional[Union[str, int, timedelta]] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 n_splits: Optional[int] = None,
                 date_column: str = 'date'):
        """
        Initialize Time Series Splitter processor.
        
        Args:
            train_period: Length of the training period
                         (string like '1Y', '6M', '30D', or number of periods, or timedelta)
            test_period: Length of the test period
                        (string like '1M', '10D', or number of periods, or timedelta)
            step_size: Increment between successive training sets
                      (default: same as test_period)
            max_train_size: Maximum training set size (if None, no limit)
            start_date: Start date for the data (if None, use earliest date in data)
            end_date: End date for the data (if None, use latest date in data)
            n_splits: Number of splits (if specified, overrides other parameters)
            date_column: Name of the date/timestamp column
        """
        logging.info("Initializing TimeSeriesSplitter with train_period=%s, test_period=%s", 
                    train_period, test_period)
        
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size if step_size is not None else test_period
        self.max_train_size = max_train_size
        self.start_date = start_date
        self.end_date = end_date
        self.n_splits = n_splits
        self.date_column = date_column
        
        # Store the generated splits
        self.splits = None
        self.current_split_index = 0
    
    def _parse_period(self, period: Union[str, int, timedelta], 
                    reference_index: pd.DatetimeIndex) -> Union[int, timedelta]:
        """
        Parse a period specification to either a number of samples or a timedelta.
        
        Args:
            period: Period specification
            reference_index: Reference DatetimeIndex to determine period length
            
        Returns:
            Number of samples or timedelta
        """
        if isinstance(period, (int, timedelta)):
            return period
        
        if isinstance(period, str):
            if period.endswith(('D', 'W', 'M', 'Q', 'Y')):
                # For pandas offset strings like '30D', '12M', etc.
                try:
                    # Convert to a timedelta for consistency
                    if period.endswith('D'):
                        return timedelta(days=int(period[:-1]))
                    elif period.endswith('W'):
                        return timedelta(days=7 * int(period[:-1]))
                    elif period.endswith('M'):
                        return timedelta(days=30 * int(period[:-1]))  # Approximation
                    elif period.endswith('Q'):
                        return timedelta(days=90 * int(period[:-1]))  # Approximation
                    elif period.endswith('Y'):
                        return timedelta(days=365 * int(period[:-1]))  # Approximation
                    else:
                        raise ValueError(f"Unrecognized period format: {period}")
                except ValueError:
                    logging.warning(f"Could not parse period '{period}' as timedelta. Using default 30 days.")
                    return timedelta(days=30)
            else:
                # Try to interpret as a number of periods
                try:
                    return int(period)
                except ValueError:
                    logging.warning(f"Could not parse period: {period}. Using default 30 days.")
                    return timedelta(days=30)
        
        logging.warning(f"Unrecognized period type: {type(period)}. Using default 30 days.")
        return timedelta(days=30)

    def fit(self, data: pd.DataFrame) -> 'TimeSeriesSplitter':
        """
        Generate and store train/test splits respecting temporal order.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            Self (fitted splitter)
        """
        logging.info("Generating time series splits for data with shape %s", data.shape)
        
        # Generate splits
        self.splits = self._generate_splits(data)
        self.current_split_index = 0
        
        logging.info("Created %d time series splits", len(self.splits))
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the current split data with a 'split_type' column indicating train or test.
        If called repeatedly, will cycle through all available splits.
        
        Args:
            data: DataFrame containing time series data (ignored if splits already generated)
            
        Returns:
            DataFrame with 'split_type' column indicating 'train' or 'test'
        """
        if self.splits is None:
            logging.warning("No splits available. Fitting the splitter first.")
            self.fit(data)
        
        if not self.splits:
            logging.warning("No valid splits were generated. Returning original data.")
            return data
        
        # Get the current split and prepare to return the next one on the next call
        train_data, test_data = self.splits[self.current_split_index]
        self.current_split_index = (self.current_split_index + 1) % len(self.splits)
        
        # Mark the data as train or test
        train_data = train_data.copy()
        test_data = test_data.copy()
        train_data['split_type'] = 'train'
        test_data['split_type'] = 'test'
        
        # Combine and return
        combined = pd.concat([train_data, test_data])
        combined['split_index'] = self.current_split_index
        
        logging.info("Returning split %d of %d (train: %d rows, test: %d rows)",
                   self.current_split_index, len(self.splits), 
                   len(train_data), len(test_data))
        
        return combined
    
    def _generate_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits respecting temporal order.
        
        Args:
            data: DataFrame containing time series data
            
        Returns:
            List of (train_data, test_data) tuples
        """
        # Ensure data is sorted by date
        df = data.copy()
        
        # Make sure date column exists
        if self.date_column not in df.columns:
            # Try to find a date column
            date_cols = [col for col in df.columns if any(date_str in col.lower() for 
                                                        date_str in ['date', 'time', 'timestamp'])]
            if date_cols:
                self.date_column = date_cols[0]
                logging.info(f"Using '{self.date_column}' as date column")
            else:
                raise ValueError(f"Date column '{self.date_column}' not found and no alternative available")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        df = df.sort_values(self.date_column).copy()
        
        # Extract the DatetimeIndex
        dates = pd.DatetimeIndex(df[self.date_column])
        
        # Set start and end dates if not specified
        start_date = self.start_date or dates.min()
        end_date = self.end_date or dates.max()
        
        # Parse periods based on the data's frequency
        train_period = self._parse_period(self.train_period, dates)
        test_period = self._parse_period(self.test_period, dates)
        step_size = self._parse_period(self.step_size or test_period, dates)
        max_train_size = self._parse_period(self.max_train_size, dates) if self.max_train_size else None
        
        splits = []
        
        # Handle different types of periods
        if isinstance(train_period, int) and isinstance(test_period, int):
            # Use integer indices for train/test splits
            n_samples = len(df)
            indices = np.arange(n_samples)
            
            if self.n_splits is not None:
                # If n_splits is specified, calculate step size to achieve the desired number of splits
                total_range = n_samples - train_period - test_period
                if total_range <= 0:
                    raise ValueError("Data has too few samples for the specified train and test periods")
                step_size = max(1, total_range // (self.n_splits - 1)) if self.n_splits > 1 else test_period
            
            for i in range(0, n_samples - test_period - train_period + 1, step_size):
                train_end = i + train_period
                test_end = train_end + test_period
                
                # Limit training size if specified
                train_start = i
                if max_train_size is not None and isinstance(max_train_size, int):
                    train_start = max(0, train_end - max_train_size)
                
                train_indices = indices[train_start:train_end]
                test_indices = indices[train_end:test_end]
                
                if len(train_indices) == 0 or len(test_indices) == 0:
                    logging.warning(f"Skipping empty split at index {i}")
                    continue
                
                train_data = df.iloc[train_indices].copy()
                test_data = df.iloc[test_indices].copy()
                
                logging.info(f"Split {len(splits)+1}: Train {len(train_data)} samples, Test {len(test_data)} samples")
                splits.append((train_data, test_data))
        
        else:
            # Use datetime for train/test splits
            if isinstance(train_period, int):
                # Convert periods to timedeltas based on average time difference
                avg_diff = (dates[-1] - dates[0]) / (len(dates) - 1)
                train_period = avg_diff * train_period
            
            if isinstance(test_period, int):
                avg_diff = (dates[-1] - dates[0]) / (len(dates) - 1)
                test_period = avg_diff * test_period
            
            if isinstance(step_size, int):
                avg_diff = (dates[-1] - dates[0]) / (len(dates) - 1)
                step_size = avg_diff * step_size
            
            if isinstance(max_train_size, int):
                avg_diff = (dates[-1] - dates[0]) / (len(dates) - 1)
                max_train_size = avg_diff * max_train_size
            
            # Calculate splits based on n_splits if specified
            if self.n_splits is not None:
                # Ensure both periods are timedeltas for consistent date arithmetic
                if not isinstance(train_period, timedelta):
                    logging.warning(f"Converting train_period to timedelta from {type(train_period)}")
                    train_period = timedelta(days=30)  # Default fallback
                    
                if not isinstance(test_period, timedelta):
                    logging.warning(f"Converting test_period to timedelta from {type(test_period)}")
                    test_period = timedelta(days=10)  # Default fallback
                
                # Calculate total range in days
                total_days = (end_date - start_date).days - train_period.days - test_period.days
                
                if total_days <= 0:
                    raise ValueError("Time range too small for the specified periods")
                    
                # Calculate days between splits
                days_between_splits = total_days / (self.n_splits - 1) if self.n_splits > 1 else test_period.days
                step_size = timedelta(days=int(days_between_splits))
                
                logging.info(f"Calculated step size of {step_size.days} days for {self.n_splits} splits")
            
            # Generate splits
            current_train_start = start_date
            
            while True:
                # Ensure both periods are timedeltas for consistent date arithmetic
                if not isinstance(train_period, timedelta):
                    logging.warning(f"Converting train_period to timedelta from {type(train_period)}")
                    train_period = timedelta(days=30)  # Default fallback
                    
                if not isinstance(test_period, timedelta):
                    logging.warning(f"Converting test_period to timedelta from {type(test_period)}")
                    test_period = timedelta(days=10)  # Default fallback
                    
                if not isinstance(step_size, timedelta):
                    logging.warning(f"Converting step_size to timedelta from {type(step_size)}")
                    step_size = test_period  # Default to test period size
                
                train_end = current_train_start + train_period
                test_end = train_end + test_period
                
                if test_end > end_date:
                    break
                
                # Limit training size if specified
                if max_train_size is not None:
                    if not isinstance(max_train_size, timedelta):
                        logging.warning(f"Converting max_train_size to timedelta from {type(max_train_size)}")
                        max_train_size = train_period  # Default to full train period
                        
                    actual_train_start = max(start_date, train_end - max_train_size)
                else:
                    actual_train_start = current_train_start
                
                train_mask = (df[self.date_column] >= actual_train_start) & (df[self.date_column] < train_end)
                test_mask = (df[self.date_column] >= train_end) & (df[self.date_column] < test_end)
                
                train_data = df[train_mask].copy()
                test_data = df[test_mask].copy()
                
                if len(train_data) == 0 or len(test_data) == 0:
                    logging.warning(f"Skipping empty split at {train_end}")
                    current_train_start += step_size
                    continue
                
                splits.append((train_data, test_data))
                logging.info(f"Split {len(splits)}: Train {actual_train_start} to {train_end}, "
                            f"Test {train_end} to {test_end}")
                
                current_train_start += step_size
        
        logging.info(f"Created {len(splits)} time series splits")
        
        return splits
    
    def get_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get all generated splits.
        
        Returns:
            List of (train_data, test_data) tuples
        """
        if self.splits is None:
            logging.warning("Splits not yet generated. Call fit() first.")
            return []
        return self.splits
    
    def plot_splits(self, data: pd.DataFrame = None, value_column: str = 'close', 
                  figsize: tuple = (15, 8)):
        """
        Visualize the train/test splits.
        
        Args:
            data: DataFrame containing time series data (if None, uses the data from fit)
            value_column: Name of the value column to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.splits is None and data is None:
            logging.warning("No data available for visualization. Call fit() first or provide data.")
            return None
            
        # If splits not generated but data provided, generate them now
        if self.splits is None and data is not None:
            self.fit(data)
            
        # Get first split if splits exist but no data provided
        if data is None:
            if not self.splits:
                logging.warning("No splits available for visualization.")
                return None
                
            # Use the first split's data to reconstruct the full dataset
            train_data, test_data = self.splits[0]
            data = pd.concat([train_data, test_data])
            
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Find a suitable value column if the specified one doesn't exist
        if value_column not in df.columns:
            # Try to find a suitable price column
            price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'price', 'open'])]
            if price_cols:
                value_column = price_cols[0]
                logging.info(f"Using '{value_column}' instead of '{value_column}'")
            else:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                    logging.info(f"Using '{value_column}' as fallback")
                else:
                    raise ValueError("No suitable numeric column found for visualization")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot full dataset
        ax.plot(df[self.date_column], df[value_column], label='Full Dataset', color='grey', alpha=0.5)
        
        # Plot each split
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.splits)))
        
        for i, (train, test) in enumerate(self.splits):
            ax.scatter(train[self.date_column], train[value_column], color=colors[i], marker='.', s=30, alpha=0.5,
                      label=f'Train {i+1}' if i < 3 or i == len(self.splits)-1 else '')
            ax.scatter(test[self.date_column], test[value_column], color=colors[i], marker='x', s=30,
                      label=f'Test {i+1}' if i < 3 or i == len(self.splits)-1 else '')
        
        if len(self.splits) > 4:
            ax.plot([], [], ' ', label=f'... and {len(self.splits) - 4} more splits')
        
        ax.set_title('Time Series Cross-Validation Splits')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_column)
        ax.legend()
        
        return fig