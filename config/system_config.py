# Logging configuration
import logging
import os


# Logs
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%d-%m-%Y %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DIFFERENT_LOG_DIRS = ['training', 'data', 'trash', "trades", "performance"]

# Environments
DEV_ENV = "development"
TEST_ENV = "test"
PROD_ENV = "production"