from datetime import datetime
import http.client
import json
import os
import sys
import logging
import time
import base64
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from dateutil.relativedelta import relativedelta
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config.utils import load_secrets, setup_logging
import config.market_config as config
from utils.broker_utils import convert_json_to_ohlcv_csv


# Global variables
conn = http.client.HTTPSConnection(config.BASE_DEMO_URL)

def historical_prices(X_SECURITY_TOKEN, CST,
                      epic, resolution, from_date, to_date,  # format 2022-02-24T00:00:00
                      max=1000, print_answer=True):
    """
    Returns historical prices for a particular instrument. The maximum number of the values in answer is max = 1000.
    
    Args:
        epic: Instrument epic/symbol. Ex. GOLD, GBPUSD.
        resolution: Defines the resolution of requested prices. Possible values are MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK.
        from_date: 
        to_date
        max: 
        The maximum number of the values in answer. Default = 10, max = 1000
    Returns:

    """
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST
    }
    conn.request("GET", f"/api/v1/prices/{epic}?resolution={resolution}&max={max}&from={from_date}&to={to_date}", payload, headers)
    #conn.request("GET", "/api/v1/prices/SILVER?resolution=MINUTE&max=10&from=2020-02-24T00:00:00&to=2020-02-24T01:00:00", payload, headers)
    res = conn.getresponse()
    data = res.read()

    parsed_data = json.loads(data.decode("utf-8"))
    print(json.dumps(parsed_data, indent=4))
    return parsed_data


# These functions are not related to the capital.com API historical market prices, but handling it
def extended_historical_prices(X_SECURITY_TOKEN, CST, epic, resolution, from_date, to_date, print_answer=False):
    """
    Returns historical prices for a particular instrument over an extended period,
    automatically splitting requests to handle Capital.com's 1000 value limit.
    
    Args:
        X_SECURITY_TOKEN: Security token for API authentication
        CST: CST value for API authentication
        epic: Instrument epic/symbol. Ex. GOLD, GBPUSD
        resolution: Defines the resolution of requested prices (MINUTE, MINUTE_5, MINUTE_15, 
                   MINUTE_30, HOUR, HOUR_4, DAY, WEEK)
        from_date: Start date in format "YYYY-MM-DDTHH:MM:SS"
        to_date: End date in format "YYYY-MM-DDTHH:MM:SS"
        print_answer: Whether to print individual API responses
        
    Returns:
        DataFrame containing all the historical price data for the specified period
    """
    # Parse date strings to datetime objects for easier manipulation
    from_datetime = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S")
    
    # Define time chunks based on resolution to approximate 1000 values per request
    chunk_sizes = {
        "MINUTE": relativedelta(days=1),         # ~1440 minutes in a day, so less than a day
        "MINUTE_5": relativedelta(days=3),       # ~288 intervals in a day, so ~3 days
        "MINUTE_15": relativedelta(days=10),     # ~96 intervals in a day, so ~10 days
        "MINUTE_30": relativedelta(days=20),     # ~48 intervals in a day, so ~20 days
        "HOUR": relativedelta(days=41),          # ~24 intervals in a day, so ~41 days
        "HOUR_4": relativedelta(days=166),       # ~6 intervals in a day, so ~166 days
        "DAY": relativedelta(years=2, months=9), # ~365 days in a year, so ~2.7 years
        "WEEK": relativedelta(years=19)          # ~52 weeks in a year, so ~19 years
    }
    
    # Adjust chunk size based on resolution
    chunk_size = chunk_sizes.get(resolution, relativedelta(days=7))
    
    # Initialize variables
    current_from = from_datetime
    all_data = []
    
    # Process the date range in chunks
    while current_from < to_datetime:
        # Calculate the end date for this chunk
        current_to = min(current_from + chunk_size, to_datetime)
        
        # Format dates for API call
        current_from_str = current_from.strftime("%Y-%m-%dT%H:%M:%S")
        current_to_str = current_to.strftime("%Y-%m-%dT%H:%M:%S")
        
        if print_answer:
            print(f"Fetching data from {current_from_str} to {current_to_str}")
        
        # Call the original function
        chunk_data = historical_prices(
            X_SECURITY_TOKEN, 
            CST, 
            epic, 
            resolution, 
            current_from_str, 
            current_to_str, 
            max=1000, 
            print_answer=print_answer
        )
        
        # Add data to our collection if the API call was successful
        if "prices" in chunk_data:
            all_data.extend(chunk_data["prices"])
        else:
            print(f"Warning: Failed to get data for period {current_from_str} to {current_to_str}")
            print(f"API Response: {chunk_data}")
        
        # Move to the next chunk
        current_from = current_to
    
    # Convert to DataFrame for easier manipulation
    if all_data:
        # Assuming the API returns data with timestamps, open, high, low, close values
        df = pd.DataFrame(all_data)
        
        # Sort by timestamp to ensure chronological order
        if "snapshotTime" in df.columns:
            df = df.sort_values(by="snapshotTime")
        
        return df
    else:
        print("No data returned from API.")
        return pd.DataFrame()

def fetch_and_save_historical_prices(X_SECURITY_TOKEN, CST, epic, resolution, 
                                    from_date, to_date, output_file="BigBoyTest_historical_prices.csv",
                                    print_answer=False, save_raw_data=False):
    """
    Fetches historical price data over an extended period and saves it to a CSV file.
    
    Args:
        X_SECURITY_TOKEN: Security token for API authentication
        CST: CST value for API authentication
        epic: Instrument epic/symbol. Ex. GOLD, GBPUSD
        resolution: Defines the resolution of requested prices
        from_date: Start date in format "YYYY-MM-DDTHH:MM:SS"
        to_date: End date in format "YYYY-MM-DDTHH:MM:SS"
        output_file: Path to the output CSV file
        print_answer: Whether to print individual API responses
        save_raw_data: Save a copy of the raw JSON data (for debugging)
        
    Returns:
        Dictionary containing all the raw price data (with 'prices' key)
    """
    # Fetch the data using our extended function
    logging.info(f"Fetching historical data for {epic} from {from_date} to {to_date}")
    
    # Process in chunks and collect all data
    current_from = datetime.strptime(from_date, "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime(to_date, "%Y-%m-%dT%H:%M:%S")
    
    # Define time chunks based on resolution to approximate 1000 values per request
    chunk_sizes = {
        "MINUTE": relativedelta(days=1),         # ~1440 minutes in a day, so less than a day
        "MINUTE_5": relativedelta(days=3),       # ~288 intervals in a day, so ~3 days
        "MINUTE_15": relativedelta(days=10),     # ~96 intervals in a day, so ~10 days
        "MINUTE_30": relativedelta(days=20),     # ~48 intervals in a day, so ~20 days
        "HOUR": relativedelta(days=41),          # ~24 intervals in a day, so ~41 days
        "HOUR_4": relativedelta(days=166),       # ~6 intervals in a day, so ~166 days
        "DAY": relativedelta(years=2, months=9), # ~365 days in a year, so ~2.7 years
        "WEEK": relativedelta(years=19)          # ~52 weeks in a year, so ~19 years
    }
    
    # Adjust chunk size based on resolution
    chunk_size = chunk_sizes.get(resolution, relativedelta(days=7))
    
    # Initialize the collection of all price data
    all_prices = []
    
    # Process the date range in chunks
    while current_from < to_datetime:
        # Calculate the end date for this chunk
        current_to = min(current_from + chunk_size, to_datetime)
        
        # Format dates for API call
        current_from_str = current_from.strftime("%Y-%m-%dT%H:%M:%S")
        current_to_str = current_to.strftime("%Y-%m-%dT%H:%M:%S")
        
        logging.info(f"Fetching chunk from {current_from_str} to {current_to_str}")
        
        # Call the original function
        chunk_data = historical_prices(
            X_SECURITY_TOKEN, 
            CST, 
            epic, 
            resolution, 
            current_from_str, 
            current_to_str, 
            max=1000, 
            print_answer=print_answer
        )
        
        # Add data to our collection if the API call was successful
        if "prices" in chunk_data:
            all_prices.extend(chunk_data["prices"])
            logging.info(f"Added {len(chunk_data['prices'])} data points from this chunk")
        else:
            logging.warning(f"Failed to get data for period {current_from_str} to {current_to_str}")
            logging.warning(f"API Response: {chunk_data}")
        
        # Move to the next chunk
        current_from = current_to
    
    # Prepare the complete data object
    complete_data = {
        "prices": all_prices,
        "instrument": epic,
        "resolution": resolution,
        "from": from_date,
        "to": to_date
    }
    
    # Save the raw JSON data if requested
    if save_raw_data:
        raw_json_file = f"{output_file.split('.')[0]}_raw.json"
        with open(raw_json_file, 'w') as f:
            json.dump(complete_data, f, indent=4)
        logging.info(f"Raw data saved to {raw_json_file}")
    
    # Save the data to OHLCV CSV format
    if all_prices:
        logging.info(f"Converting {len(all_prices)} data points to OHLCV format")
        convert_json_to_ohlcv_csv(complete_data, output_file)
        logging.info(f"OHLCV data saved to {output_file}")
    else:
        logging.error("No data to save.")
    
    return complete_data
