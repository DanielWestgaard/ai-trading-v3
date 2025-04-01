from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import glob

from backtesting.events import MarketEvent


class MarketData(ABC):
    """Abstract base class for market data handlers."""
    
    def __init__(self, symbols: List[str], logger=None):
        """
        Initialize the market data handler.
        
        Args:
            symbols: List of symbols to track
            logger: Custom logger (if None, creates default)
        """
        self.symbols = symbols
        self.current_idx = 0
        self.data = {}
        
    @abstractmethod
    def load_data(self):
        """Load market data."""
        pass
    
    @abstractmethod
    def get_next(self) -> Optional[Dict[str, MarketEvent]]:
        """
        Get the next market data points.
        
        Returns:
            Dictionary mapping symbols to MarketEvent objects, or None if no more data
        """
        pass
    
    def reset(self):
        """Reset the data iterator."""
        self.current_idx = 0
        logging.info("Reset market data iterator")
    
    def get_length(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of data points
        """
        return 0  # Override in subclasses
    
    def get_dates(self) -> List[datetime]:
        """
        Get list of all dates in the dataset.
        
        Returns:
            List of datetime objects
        """
        return []  # Override in subclasses
    
    def get_symbols(self) -> List[str]:
        """
        Get list of symbols.
        
        Returns:
            List of symbols
        """
        return self.symbols
    
    def has_more_data(self) -> bool:
        """
        Check if there is more data available.
        
        Returns:
            True if more data is available, False otherwise
        """
        return self.current_idx < self.get_length()


class CSVMarketData(MarketData):
    """Market data handler for CSV files."""
    
    def __init__(self, 
                symbols: List[str], 
                csv_dir: str, 
                date_col: str = 'Date',
                dtypes: Dict[str, Any] = None,
                date_format: str = None,
                logger=None):
        """
        Initialize CSV market data handler.
        
        Args:
            symbols: List of symbols to track
            csv_dir: Directory containing CSV files
            date_col: Name of date column
            dtypes: Data types for columns
            date_format: Format string for parsing dates
            logger: Custom logger
        """
        super().__init__(symbols, logger)
        self.csv_dir = csv_dir
        self.date_col = date_col
        self.dtypes = dtypes
        self.date_format = date_format
        self.data = {}
        self.dates = []
        self.load_data()
    
    def load_data(self):
        """Load market data from CSV files."""
        logging.info(f"Loading market data from {self.csv_dir}")
        
        for symbol in self.symbols:
            # Look for CSV files matching the symbol
            pattern = os.path.join(self.csv_dir, f"*{symbol}*.csv")
            files = glob.glob(pattern)
            
            if not files:
                logging.warning(f"No CSV files found for symbol {symbol}")
                continue
            
            # Use the first file found for the symbol
            file_path = files[0]
            logging.info(f"Loading {symbol} data from {file_path}")
            
            try:
                # Load the CSV file
                if self.dtypes:
                    df = pd.read_csv(file_path, dtype=self.dtypes)
                else:
                    df = pd.read_csv(file_path)
                
                # Convert date column to datetime
                if self.date_format:
                    df[self.date_col] = pd.to_datetime(df[self.date_col], format=self.date_format)
                else:
                    df[self.date_col] = pd.to_datetime(df[self.date_col])
                
                # Sort by date
                df = df.sort_values(by=self.date_col)
                
                # Store the data
                self.data[symbol] = df
                
                logging.info(f"Loaded {len(df)} data points for {symbol}")
            except Exception as e:
                logging.error(f"Error loading data for {symbol}: {str(e)}")
        
        # Align dates across all symbols
        self._align_dates()
    
    def _align_dates(self):
        """Align dates across all symbols to ensure consistent iteration."""
        if not self.data:
            logging.warning("No data loaded, cannot align dates")
            return
        
        # Get dates from each symbol
        symbol_dates = {}
        for symbol, df in self.data.items():
            symbol_dates[symbol] = set(df[self.date_col].dt.date)
        
        # Find common dates if there are multiple symbols
        if len(symbol_dates) > 1:
            common_dates = set.intersection(*symbol_dates.values())
            logging.info(f"Found {len(common_dates)} common dates across all symbols")
        else:
            # If there's only one symbol, use all its dates
            symbol = list(self.data.keys())[0]
            common_dates = symbol_dates[symbol]
        
        # Sort the common dates
        self.dates = sorted(list(common_dates))
        
        # Filter data to only include common dates
        for symbol, df in self.data.items():
            self.data[symbol] = df[df[self.date_col].dt.date.isin(self.dates)].reset_index(drop=True)
            logging.info(f"Filtered {symbol} data to {len(self.data[symbol])} points with common dates")
    
    def get_next(self) -> Optional[Dict[str, MarketEvent]]:
        """
        Get the next market data points.
        
        Returns:
            Dictionary mapping symbols to MarketEvent objects, or None if no more data
        """
        if not self.has_more_data() or not self.dates:
            return None
        
        # Get the date for the current index
        if self.current_idx < len(self.dates):
            current_date = self.dates[self.current_idx]
        else:
            return None
        
        # Convert date to datetime for event timestamp
        current_datetime = datetime.combine(current_date, datetime.min.time())
        
        # Prepare market events for all symbols
        events = {}
        for symbol, df in self.data.items():
            # Find data for the current date
            date_data = df[df[self.date_col].dt.date == current_date]
            
            if len(date_data) > 0:
                # Create a market event with the data
                row_dict = date_data.iloc[0].to_dict()
                
                # Create MarketEvent
                events[symbol] = MarketEvent(
                    timestamp=current_datetime,
                    symbol=symbol,
                    data=row_dict
                )
        
        # Increment the index for next time
        self.current_idx += 1
        
        return events
    
    def get_length(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of data points (days)
        """
        return len(self.dates)
    
    def get_dates(self) -> List[datetime]:
        """
        Get list of all dates in the dataset.
        
        Returns:
            List of datetime objects
        """
        return [datetime.combine(date, datetime.min.time()) for date in self.dates]


class DataFrameMarketData(MarketData):
    """Market data handler for pre-loaded pandas DataFrames."""
    
    def __init__(self, 
                data: Dict[str, pd.DataFrame], 
                date_col: str = 'Date',
                logger=None):
        """
        Initialize DataFrame market data handler.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            date_col: Name of date column
            logger: Custom logger
        """
        symbols = list(data.keys())
        super().__init__(symbols, logger)
        self.date_col = date_col
        self.data = data
        self.dates = []
        self._align_dates()
    
    def load_data(self):
        """Data is already loaded, just validate it."""
        if not self.data:
            logging.warning("No data provided")
            return
        
        for symbol, df in self.data.items():
            if self.date_col not in df.columns:
                logging.error(f"Date column '{self.date_col}' not found in data for {symbol}")
            else:
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                    logging.info(f"Converting {self.date_col} to datetime for {symbol}")
                    self.data[symbol][self.date_col] = pd.to_datetime(df[self.date_col])
    
    def _align_dates(self):
        """Align dates across all symbols to ensure consistent iteration."""
        if not self.data:
            logging.warning("No data loaded, cannot align dates")
            return
        
        # Get dates from each symbol
        symbol_dates = {}
        for symbol, df in self.data.items():
            symbol_dates[symbol] = set(df[self.date_col].dt.date)
        
        # Find common dates if there are multiple symbols
        if len(symbol_dates) > 1:
            common_dates = set.intersection(*symbol_dates.values())
            logging.info(f"Found {len(common_dates)} common dates across all symbols")
        else:
            # If there's only one symbol, use all its dates
            symbol = list(self.data.keys())[0]
            common_dates = symbol_dates[symbol]
        
        # Sort the common dates
        self.dates = sorted(list(common_dates))
        
        # Filter data to only include common dates
        for symbol, df in self.data.items():
            self.data[symbol] = df[df[self.date_col].dt.date.isin(self.dates)].reset_index(drop=True)
            logging.info(f"Filtered {symbol} data to {len(self.data[symbol])} points with common dates")
    
    def get_next(self) -> Optional[Dict[str, MarketEvent]]:
        """Implementation identical to CSVMarketData."""
        if not self.has_more_data() or not self.dates:
            return None
        
        # Get the date for the current index
        if self.current_idx < len(self.dates):
            current_date = self.dates[self.current_idx]
        else:
            return None
        
        # Convert date to datetime for event timestamp
        current_datetime = datetime.combine(current_date, datetime.min.time())
        
        # Prepare market events for all symbols
        events = {}
        for symbol, df in self.data.items():
            # Find data for the current date
            date_data = df[df[self.date_col].dt.date == current_date]
            
            if len(date_data) > 0:
                # Create a market event with the data
                row_dict = date_data.iloc[0].to_dict()
                
                # Create MarketEvent
                events[symbol] = MarketEvent(
                    timestamp=current_datetime,
                    symbol=symbol,
                    data=row_dict
                )
        
        # Increment the index for next time
        self.current_idx += 1
        
        return events
    
    def get_length(self) -> int:
        return len(self.dates)
    
    def get_dates(self) -> List[datetime]:
        return [datetime.combine(date, datetime.min.time()) for date in self.dates]


# Integration with the data pipeline
class PipelineMarketData(DataFrameMarketData):
    """Market data handler that integrates with the existing data pipeline."""
    
    def __init__(self, 
                processed_data_path: str,
                symbols: List[str] = None,
                date_col: str = 'Date'):
        """
        Initialize pipeline market data handler.
        
        Args:
            processed_data_path: Path to processed data file
            symbols: List of symbols (can be None if single symbol data)
            date_col: Name of date column
            logger: Custom logger
        """
        self.processed_data_path = processed_data_path
        self.date_col = date_col
        
        # Load processed data
        df = self._load_processed_data()
        
        # If no symbols provided, assume single symbol data
        if symbols is None:
            # Extract symbol from filename
            filename = os.path.basename(processed_data_path)
            parts = filename.split('_')
            if len(parts) > 1:
                # Try to extract symbol from filename
                symbol = parts[1]
                symbols = [symbol]
            else:
                # Use a default symbol
                symbol = "UNKNOWN"
                symbols = [symbol]
            
            logging.info(f"No symbols provided, using {symbol}")
            
            # Create data dictionary with the single symbol
            data = {symbol: df}
        else:
            # Create a copy of the same data for each symbol
            # This is a simplification - in a real system, you'd have different data for each symbol
            data = {symbol: df.copy() for symbol in symbols}
        
        # Initialize parent class
        super().__init__(data, date_col)
    
    def _load_processed_data(self) -> pd.DataFrame:
        """Load processed data from file."""
        logging.info(f"Loading processed data from {self.processed_data_path}")
        
        try:
            df = pd.read_csv(self.processed_data_path)
            
            # Convert date column to datetime
            if self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
            else:
                logging.warning(f"Date column '{self.date_col}' not found in processed data")
            
            logging.info(f"Loaded {len(df)} data points from processed data")
            return df
        except Exception as e:
            logging.error(f"Error loading processed data: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()