import csv
import os
import logging
from datetime import datetime
import sys
import json

# Add the parent directory to the path for imports
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.system_config as config


def load_secrets_alpaca(file_path="live_market_data/secrets/secrets.txt"):
    desired_keys={"alpaca_secret_key_paper", "alpaca_api_key_paper"}
    secrets = {}
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "=" in line:  # Ensure valid format
                key, value = line.split("=", 1)  # Split at the first '='
                if key in desired_keys:
                    secrets[key] = value
    
    api_key = secrets.get('alpaca_api_key_paper')
    secret_key = secrets.get('alpaca_secret_key_paper')

    return secrets, api_key, secret_key

def load_secrets(file_path="live_market_data/secrets/secrets.txt"):
    """
    Utility functions to extract the following keys: API_KEY_CAP, PASSWORD_CAP, EMAIL.
    
    Returns:
        secrets{}, API_KEY_CAP, PASSWORD_CAP, EMAIL
    """
    desired_keys = {"API_KEY_CAP", "PASSWORD_CAP", "EMAIL", "ENCRYPTION_KEY_CAP"}
    secrets = {}
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "=" in line:  # Ensure valid format
                key, value = line.split("=", 1)  # Split at the first '='
                if key in desired_keys:
                    secrets[key] = value
    
    api_key = secrets.get('API_KEY_CAP')
    password = secrets.get('PASSWORD_CAP')
    email = secrets.get('EMAIL')
    encryption_key = secrets.get('ENCRYPTION_KEY_CAP')

    return secrets, api_key, password, email

def setup_logging(type='standard', save_to_file=True):
    """
    Configure logging for the project.
    Logs are saved to the logs directory specified in config if save_to_file is True.
    
    Args:
        type (str): Type of logger to create
        save_to_file (bool): Whether to save logs to a file
    
    Returns:
        Logger
        Path to the created log file or None if save_to_file is False
    """
    # Only manage existing log files if we're saving to file
    if save_to_file:
        manage_log_files()
        
        # Create logs directory if it doesn't exist
        os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Initialize log_filepath
    log_filepath = None
    
    # Create handlers list starting with StreamHandler
    handlers = [logging.StreamHandler()]
    
    # Add FileHandler if we're saving to file
    if save_to_file:
        # Create log filename with timestamp
        timestamp = get_timestamp()
        log_filename = f"{type}_{timestamp}.log"
        log_filepath = os.path.join(config.LOGS_DIR, log_filename)
        handlers.append(logging.FileHandler(log_filepath))
    
    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=config.LOGGING_LEVEL,
        format=config.LOGGING_FORMAT,
        handlers=handlers
    )
    
    logger = logging.getLogger(type)
    logger.info("Logging configured successfully!")
    if save_to_file:
        logger.info(f"Logging to file: {log_filepath}")
    
    # Store log path in global variable for easy access
    global current_log_path
    current_log_path = log_filepath
    
    return logger, log_filepath

# Global variable to store current log file path
current_log_path = None

def get_current_log_path():
    """
    Get the path to the current log file.
    
    Returns:
        Path to current log file or None if logging not initialized
    """
    return current_log_path

def manage_log_files():
    """
    Manage log files by archiving old ones if there are too many.
    """
    if not config.ARCHIVE_LOGS:
        return
        
    try:
        log_files = [f for f in os.listdir(config.LOGS_DIR) 
                     if f.endswith(".log")]
        
        # If we have more than the maximum allowed log files
        if len(log_files) > config.MAX_LOG_FILES:
            # Sort by modification time (oldest first)
            log_files.sort(key=lambda f: os.path.getmtime(os.path.join(config.LOGS_DIR, f)))
            
            # Create archives directory if it doesn't exist
            archives_dir = os.path.join(config.LOGS_DIR, "archives")
            os.makedirs(archives_dir, exist_ok=True)
            
            # Move oldest files to archives
            files_to_archive = log_files[:len(log_files) - config.MAX_LOG_FILES]
            for file in files_to_archive:
                old_path = os.path.join(config.LOGS_DIR, file)
                new_path = os.path.join(archives_dir, file)
                os.rename(old_path, new_path)
                
            logging.info(f"Archived {len(files_to_archive)} old log files")
    except Exception as e:
        logging.warning(f"Error managing log files: {e}")

def get_timestamp():
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Related to handling
def on_open(ws, subscription_message):
    print("Connection opened")
    # Send the subscription message
    ws.send(json.dumps(subscription_message))

def on_message(ws, message):
    parsed = json.dumps(message, indent=4)
    print(f"Received: {message}")

def on_message_improved(ws, message):
    try:
        # Parse the JSON message
        data = json.loads(message)
        
        # Print the formatted JSON with indentation for readability
        formatted_json = json.dumps(data, indent=4)
        
        # Add a separator line for visual clarity between messages
        print("\n" + "-"*50)
        print(formatted_json)
        print("-"*50)
    except Exception as e:
        print(f"Error processing message: {e}")
        print(f"Original message: {message}")

def on_message_pretty(ws, message):
    try:
        # Parse the JSON message
        data = json.loads(message)
        
        # Check if it's an OHLC event
        if data.get("destination") == "ohlc.event":
            # Extract the payload
            payload = data.get("payload", {})
            
            # Convert timestamp to readable datetime
            timestamp_ms = payload.get("t")
            if timestamp_ms:
                timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_time = "N/A"
            
            # Create a formatted output
            print("\n" + "="*50)
            print(f"OHLC Update for {payload.get('epic')}")
            print(f"Time: {formatted_time}")
            print(f"Resolution: {payload.get('resolution')}")
            print(f"Type: {payload.get('type')}")
            print(f"Price Type: {payload.get('priceType')}")
            print("-"*20)
            print(f"Open:  {payload.get('o')}")
            print(f"High:  {payload.get('h')}")
            print(f"Low:   {payload.get('l')}")
            print(f"Close: {payload.get('c')}")
            print("="*50)
        else:
            # For other types of messages, just print them nicely formatted
            print(f"\nReceived message:\n{json.dumps(data, indent=4)}")
    except Exception as e:
        print(f"Error processing message: {e}")
        print(f"Original message: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")



# Related to API

def get_account_id_by_name(json_data, account_name=config.ACCOUNT_TEST):
    """ Retrieves the account ID based on the account name. """
    # Parse the JSON string to a Python dictionary
    parsed_data = json.loads(json_data)
    
    # Iterate through accounts to find the matching accountName
    for account in parsed_data["accounts"]:
        if account["accountName"] == account_name:
            logging.info(f"Found accountId matching account {account_name}!")
            return account["accountId"]
    
    logging.info(f"Did not find accountId that matched account {account_name}.")
    # Return None if no match is found
    return None

def extract_deal_reference(json_data, key_string_to_extract):
    """ Extracting and returning the deal reference / dealID from confirmed positions. """
    # Parse the JSON string to a Python dictionary
    parsed_data = json.loads(json_data)
    # Extract the dealReference value
    value = parsed_data.get(key_string_to_extract)  # "dealReference"
    #logging.info(f"Successfully extracted {key_string_to_extract}: {value}")
    
    return value

def process_positions(json_response):
    """
    Process a trading positions JSON response.
    
    Args:
        json_response (str or dict): JSON string or dictionary containing positions data
    
    Returns:
        list: List of all deal IDs
    """
    # Parse the JSON if it's a string
    if isinstance(json_response, str):
        data = json.loads(json_response)
    else:
        data = json_response
    
    # Get the positions array
    positions = data.get('positions', [])
    
    # Count active positions
    position_count = len(positions)
    print(f"Number of active positions: {position_count}")
    
    # Extract all deal IDs
    deal_ids = []
    for position in positions:
        if 'position' in position and 'dealId' in position['position']:
            deal_ids.append(position['position']['dealId'])
    
    return deal_ids

# Not sure where to place this method. Was thinking in the data processing pipeline, 
# but could also be more of a utility func for when taking in live market data.
def convert_json_to_ohlcv_csv(json_data, output_file="test.csv"):
    """
    Converts the provided JSON price data to OHLCV format and saves it to a CSV file.
    
    Args:
        json_data (dict or str): The JSON data containing price information
        output_file (str): Path to the output CSV file
    """
    # If json_data is a string, parse it to a dictionary
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        # Create CSV writer
        fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Process each price entry
        for price in data['prices']:
            # Extract UTC timestamp (using snapshotTimeUTC for consistency)
            timestamp = price['snapshotTimeUTC']
            
            # Use bid prices (could be modified to use ask or average)
            open_price = price['openPrice']['bid']
            high_price = price['highPrice']['bid']
            low_price = price['lowPrice']['bid']
            close_price = price['closePrice']['bid']
            volume = price['lastTradedVolume']
            
            # Write the row to the CSV
            writer.writerow({
                'Date': timestamp,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
    
    logging.info(f"Data successfully converted and saved to {output_file}")

def parse_live_market_data():
    pass