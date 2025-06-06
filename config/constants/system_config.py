# Logging configuration
import logging
import os


# Logs
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%d-%m-%Y %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai-trading-v3/
BASE_LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DIFFERENT_LOG_DIRS = ['training', 'data', 'trash', "trades", "performance", "backtesting"]
BASE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'storage')
BASE_CONFIG_DIR = os.path.join(BASE_DIR, 'config')
BACKTEST_RESTULTS_DIR = os.path.join(BASE_DIR, 'backtest_results')
LIVE_DATA_DIR = os.path.join(BASE_DIR, 'live', 'temp_live')

# Capital.com related to data storage
CAPCOM_BASE_DATA_DIR = os.path.join(BASE_DATA_DIR, 'capital_com')
CAPCOM_RAW_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'raw')
CAPCOM_PROCESSED_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'processed')
CAPCOM_VIS_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'visualizations')

# Model storage
MODEL_BASE_DIR = os.path.join(BASE_DIR, 'model_registry')
SAVED_MODELS_DIR = os.path.join(MODEL_BASE_DIR, 'model_storage')
ML_MODEL_RESULTS_DIR = os.path.join(MODEL_BASE_DIR, 'ml_model_results')
MODEL_METADATA_DIR = os.path.join(MODEL_BASE_DIR, 'metadata')

# Backtesting config
BACKTESTING_CONFIG_PATH = os.path.join(BASE_CONFIG_DIR, 'backtesting', 'indicator_strategy_config.yaml')

# Live config
LIVE_TRADING_CONFIG_PATH = os.path.join(BASE_CONFIG_DIR, 'live', 'live_trading_config.yaml')

# Environments
DEV_ENV = "development"
TEST_ENV = "test"
PROD_ENV = "production"
