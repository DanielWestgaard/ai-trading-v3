from abc import ABC
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime

from broker.base_interface import BaseBroker
import utils.broker_utils as br_util
import config.market_config as mark_config
import broker.capital_com.rest_api.account as account
import broker.capital_com.rest_api.session as session
import broker.capital_com.rest_api.trading as trading
import broker.capital_com.rest_api.markets_info as markets_info


class CapitalCom(BaseBroker):
    """Base class for broker implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the broker with configuration."""
        logging.info("Inside broker")
        
        try: 
            # Getting secrets
            self.secrets, self.api_key, self.password, self.email = br_util.load_secrets()
            # Encrypting password
            self.enc_pass = session.encrypt_password(self.password, self.api_key)
        except Exception as e:
            logging.info(f"Could not initiate BaseBroker with secrets and/or encrypted password. Error: {e}")
            
    # =================== SESSION METHODS ==================
    
    def start_session(self, email=None, password=None, api_key=None, use_encryption=True, print_answer=False):
        """Starting a session with the broker."""
        self.body, self.headers_dict, self.x_security_token, self.cst = session.start_session(email=email or self.email,
                                     password=password or self.enc_pass,
                                     api_key=api_key or self.api_key,
                                     use_encryption=use_encryption,
                                     print_answer=print_answer)
        return 
        
    def end_session(self, X_SECURITY_TOKEN=None, CST=None):
        return session.end_session(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst)
    
    def session_details(self, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        return session.session_details(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
    
    def switch_active_account(self, account_id=None, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        if account_id is None or self.all_accounts is None:
            logging.info("AccountID and/or all_accounts is None. Initializing them now.")
            self.all_accounts = account.list_all_accounts(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
            self.account_id = br_util.get_account_id_by_name(self.all_accounts, mark_config.ACCOUNT_TEST)
        return session.switch_active_account(self.account_id, X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
    
    # ==================== DATA METHODS ====================
    
    def get_historical_data(self, epic:str, resolution:str, 
                            from_date:str, to_date:str,  # format 2022-02-24T00:00:00
                            X_SECURITY_TOKEN=None, CST=None,
                            max=1000, print_answer=False):
        """Fetch historical OHLCV data."""
        return markets_info.historical_prices(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,
                                              epic=epic, resolution=resolution, from_date=from_date, to_date=to_date,
                                              max=max, print_answer=print_answer)
    
    # ==================== ACCOUNT METHODS ====================
    
    def list_all_accounts(self, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        self.all_accounts = account.list_all_accounts(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
        return self.all_accounts
    
    def get_account_balance(self) -> Dict:
        """Get account balance information."""
        logging.info("Getting account balance")
        return {}  # Return empty dict
    
    def get_positions(self) -> List[Dict]:
        """Get current open positions."""
        logging.info("Getting positions")
        return []  # Return empty list
    
    def get_orders(self) -> List[Dict]:
        """Get current pending orders."""
        logging.info("Getting orders")
        return []  # Return empty list
    
    # ==================== TRADING METHODS ====================
    
    def place_market_order(self, 
                          symbol: str, 
                          side: str, 
                          quantity: float,
                          take_profit: Optional[float] = None,
                          stop_loss: Optional[float] = None) -> Dict:
        """Place a market order."""
        logging.info(f"Placing market order for {symbol}")
        return {}  # Return empty dict
    
    def place_limit_order(self,
                         symbol: str,
                         side: str,
                         quantity: float,
                         price: float,
                         take_profit: Optional[float] = None,
                         stop_loss: Optional[float] = None) -> Dict:
        """Place a limit order."""
        logging.info(f"Placing limit order for {symbol}")
        return {}  # Return empty dict
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        logging.info(f"Canceling order {order_id}")
        return True
    
    def modify_position(self,
                       position_id: str,
                       take_profit: Optional[float] = None,
                       stop_loss: Optional[float] = None) -> bool:
        """Modify an existing position."""
        logging.info(f"Modifying position {position_id}")
        return True