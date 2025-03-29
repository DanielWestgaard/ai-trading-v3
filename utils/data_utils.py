import datetime
import logging
import re
import os

import pandas as pd

import config.market_config as mark_config
import config.system_config as sys_config

def generate_filename(symbol, timeframe, start_date, end_date, is_raw=True, 
                     data_source=None, processing_info=None, extension='csv'):
    """
    Generate a standardized filename for financial data.
    
    Parameters:
    -----------
    symbol : str
        The trading symbol or asset identifier
    timeframe : str
        The timeframe of the data (e.g., '1min', '1h', '1d')
    start_date : datetime.datetime or str
        The start date of the data
    end_date : datetime.datetime or str
        The end date of the data
    is_raw : bool, default=True
        Flag indicating if the data is raw or processed
    data_source : str, optional
        The source of the data (e.g., 'yahoo', 'bloomberg')
    processing_info : str, optional
        Additional information about processing steps (only used if is_raw=False)
    extension : str, default='csv'
        The file extension without the dot
    """
    # Sanitize symbol (remove non-alphanumeric characters except underscore)
    symbol = re.sub(r'[^\w]', '_', symbol)
    
    # Standardize timeframe format
    timeframe = timeframe.lower().replace('minute', 'm').replace('hour', 'h').replace('day', 'd').replace('_', '')
    
    # Convert dates to datetime objects if they are strings
    if isinstance(start_date, str):
        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%Y-%m-%dT%H:%M:%S']:  #2020-02-24T00:00:00
                try:
                    start_date = datetime.datetime.strptime(start_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not parse start_date: {start_date}")
        except ValueError as e:
            raise ValueError(f"Could not parse start_date: {e}")
            
    if isinstance(end_date, str):
        try:
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%Y-%m-%dT%H:%M:%S']:
                try:
                    end_date = datetime.datetime.strptime(end_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not parse end_date: {end_date}")
        except ValueError as e:
            raise ValueError(f"Could not parse end_date: {e}")
    
    # Format dates as YYYYMMDD
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    # Base name components
    components = [symbol, timeframe, start_str, end_str]
    
    # Add data source if provided
    if data_source:
        sanitized_source = re.sub(r'[^\w]', '_', data_source)
        components.append(sanitized_source)
    
    base_name = "_".join(components)
    
    # Add prefix based on whether the data is raw or processed
    if is_raw:
        prefix = "raw"
    else:
        prefix = "processed"
        # Add processing info if provided and data is processed
        if processing_info:
            sanitized_info = re.sub(r'[^\w]', '_', processing_info)
            prefix = f"{prefix}_{sanitized_info}"
    
    # Create the final filename
    filename = f"{prefix}_{base_name}.{extension}"
    
    return filename

# More enhanced and should be more widely used
def save_financial_data(data, file_type, **kwargs):
    """
    Save financial data as either raw or processed files.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The data to be saved
    file_type : str
        Type of data: 'raw' or 'processed'
    **kwargs :
        For raw files:
            symbol, timeframe, start_date, end_date : required for filename generation
            data_source, extension : optional for filename generation
            location : optional directory to save file
            
        For processed files:
            raw_filename : required - the original raw filename (can be a full path)
            processing_info : optional - additional processing information
            location : optional directory to save file
    
    Returns:
    --------
    str
        The path to the saved file
    """
    import os
    import logging
    import re
    
    # Generate filename based on file_type
    if file_type == 'raw':
        # Validate required parameters for raw files
        required_params = ['symbol', 'timeframe', 'start_date', 'end_date']
        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters for raw file: {', '.join(missing)}")
        
        # Generate raw filename
        filename = generate_filename(
            symbol=kwargs['symbol'],
            timeframe=kwargs['timeframe'],
            start_date=kwargs['start_date'],
            end_date=kwargs['end_date'],
            is_raw=True,
            data_source=kwargs.get('data_source'),
            extension=kwargs.get('extension', 'csv')
        )
        
        # Set location for raw files
        location = kwargs.get('location', sys_config.CAPCOM_RAW_DATA_DIR)
        
    elif file_type == 'processed':
        # Validate we have a raw filename for processed files
        if 'raw_filename' not in kwargs or not kwargs['raw_filename']:
            raise ValueError("Raw filename is required for processed files")
        
        # Extract just the filename if a full path is provided
        raw_filename = os.path.basename(kwargs['raw_filename'])
        
        # Validate it's a raw filename
        if not raw_filename.startswith("raw_"):
            raise ValueError("Raw filename must start with 'raw_'")
        
        # Convert to processed filename
        filename = raw_filename.replace("raw_", "processed_", 1)
        
        # Add processing info if provided
        if processing_info := kwargs.get('processing_info'):
            # Split into base name and extension
            name_parts = filename.rsplit('.', 1)
            base_name = name_parts[0]
            extension = name_parts[1] if len(name_parts) > 1 else 'csv'
            
            # Sanitize and add processing info
            sanitized_info = re.sub(r'[^\w]', '_', processing_info)
            filename = f"{base_name}_{sanitized_info}.{extension}"
        
        # Set location for processed files
        location = kwargs.get('location', sys_config.CAPCOM_PROCESSED_DATA_DIR)
        
    else:
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'raw' or 'processed'")
    
    # Ensure directory exists
    os.makedirs(location, exist_ok=True)
    
    # Create full file path
    file_path = os.path.join(location, filename)
    
    # Save data to file
    data.to_csv(file_path, index=False)
    
    # Log and return file path
    logging.info(f"File saved at: {file_path}")
    return file_path

def get_file_path(filename, base_dir=None, create_dirs=True):
    """
    Generate a full path for a file and optionally create the directories.
    
    Parameters:
    -----------
    filename : str
        The generated filename
    base_dir : str, optional
        The base directory to save files. Defaults to current working directory.
    create_dirs : bool, default=True
        Whether to create directories if they don't exist
    """
    if not base_dir:
        base_dir = os.getcwd()
    
    # Determine if it's raw or processed from the filename
    if filename.startswith("raw_"):
        sub_dir = "raw_data"
    elif filename.startswith("processed_"):
        sub_dir = "processed_data"
    else:
        sub_dir = "other_data"
    
    # Create the directory path
    dir_path = os.path.join(base_dir, sub_dir)
    
    # Create directories if they don't exist and create_dirs is True
    if create_dirs and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Return the full file path
    return os.path.join(dir_path, filename)

# Have a more enhanced version, but sometimes simpler might be better to use at that time
def save_data_file(data:pd.DataFrame, type:str, filename:str, location:str=None):
    logging.info("Saving file...")
    
    if location is None:
        if type == 'processed':
            location = sys_config.CAPCOM_PROCESSED_DATA_DIR
        else:  # raw
            location = sys_config.CAPCOM_RAW_DATA_DIR
    else:
        location = location
        
    # Ensure the directory exists
    os.makedirs(location, exist_ok=True)
    # Create the full file path
    file_path = os.path.join(location, filename)
    
    # Write content to file
    data.to_csv(file_path, index=False)

    logging.info(f"File saved at: {file_path}")

# Example usage
if __name__ == "__main__":
    # Raw data example
    raw_filename = generate_filename(
        symbol="AAPL",
        timeframe="MINUTE_5",
        start_date="2023-01-01",
        end_date="2023-12-31",
        is_raw=True,
        data_source="yahoo"
    )
    print(f"Raw data filename: {raw_filename}")
    # Output: raw_AAPL_1d_20230101_20231231_yahoo.csv
    
    # Processed data example
    processed_filename = generate_filename(
        symbol="AAPL",
        timeframe="1day",
        start_date="2020-02-24T00:00:00",
        end_date="2020-02-24T00:00:00",
        is_raw=False,
        data_source="yahoo",
        processing_info="normalized"
    )
    print(f"Processed data filename: {processed_filename}")
    # Output: processed_normalized_AAPL_1d_20230101_20231231_yahoo.csv