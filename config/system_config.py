# Logging configuration
import logging
import os


# Logs
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%d-%m-%Y %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO
DEBUG_LOG_LEVEL = logging.DEBUG

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DIFFERENT_LOG_DIRS = ['training', 'data', 'trash', "trades", "performance"]
BASE_DATA_DIR = os.path.join(BASE_DIR, 'data', 'storage')

# Capital.com related to data storage
CAPCOM_BASE_DATA_DIR = os.path.join(BASE_DATA_DIR, 'capital_com')
CAPCOM_RAW_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'raw')
CAPCOM_PROCESSED_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'processed')
CAPCOM_VIS_DATA_DIR = os.path.join(CAPCOM_BASE_DATA_DIR, 'visualizations')

# Environments
DEV_ENV = "development"
TEST_ENV = "test"
PROD_ENV = "production"
