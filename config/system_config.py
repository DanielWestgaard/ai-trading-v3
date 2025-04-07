# Logging configuration
import logging
import os


# Logs
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%d-%m-%Y %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai-trading-v3/
BASE_LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DIFFERENT_LOG_DIRS = ['training', 'data', 'trash', "trades", "performance", "backtesting"]
BASE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'storage')
BASE_CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# Capital.com related to data storage
CAPCOM_BASE_DATA_DIR = os.path.join(BASE_DATA_DIR, 'capital_com')
CAPCOM_RAW_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'raw')
CAPCOM_PROCESSED_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'processed')
CAPCOM_VIS_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'visualizations')

# Model storage
MODEL_BASE_DIR = os.path.join(BASE_DIR, 'model_related_storage')
SAVED_MODELS_DIR = os.path.join(MODEL_BASE_DIR, 'model_storage')
ML_MODEL_RESULTS_DIR = os.path.join(MODEL_BASE_DIR, 'ml_model_results')
MODEL_METADATA_DIR = os.path.join(MODEL_BASE_DIR, 'metadata')

# Backtesting config
CONFIG_BACKTESTING_PATH = os.path.join(BASE_CONFIG_DIR, 'indicator_strategy_config.json')

# Environments
DEV_ENV = "development"
TEST_ENV = "test"
PROD_ENV = "production"
