import os
import config.constants.system_config as sys_config

# API related variables
encryption = "false"  # Want to use password encryption? true for yes, false for no, TODO: implement this
sleepTime = 5  # seconds wait before ending session, only for testing
# Alpaca
ALP_LIVE_URL = "https://api.alpaca.markets"
ALP_PAPER_URL = "https://paper-api.alpaca.markets"
# Capital.com
BASE_URL = "api-capital.backend-capital.com"  # Live account
BASE_DEMO_URL = "demo-api-capital.backend-capital.com"  # Demo accounts
WEBSOCKET_URL = "wss://api-streaming-capital.backend-capital.com/connect"
ACCOUNT_TEST = "USD_testing"
ACCOUNT_MODEL = "USD_model"
CAPCOM_RESPONSE_JSON_DIR = os.path.join(sys_config.BASE_DIR, 'broker', 'capital_com', 'rest_api', 'saved_responses')

PRICE_DECIMALS = 5