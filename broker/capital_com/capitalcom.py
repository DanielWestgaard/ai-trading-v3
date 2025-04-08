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
            try:
                # Encrypting password
                self.enc_pass = session.encrypt_password(self.password, self.api_key)
            except Exception as e:
                logging.warning(f"Could not initiate BaseBroker with encrypted password. Error: {e}")
        except Exception as e:
            logging.warning(f"Could not initiate BaseBroker with secrets. Error: {e}")
            
    # =================== SESSION METHODS ==================
    
    def start_session(self, email=None, password=None, api_key=None, use_encryption=True, print_answer=False):
        """Starting a session with the broker."""
        self.body, self.headers_dict, self.x_security_token, self.cst = session.start_session(email=email or self.email,
                                     password=password or self.enc_pass,
                                     api_key=api_key or self.api_key,
                                     use_encryption=use_encryption,
                                     print_answer=print_answer)
        return self.body, self.headers_dict, self.x_security_token, self.cst  # Just in case I need to use them outside
        
    def end_session(self, X_SECURITY_TOKEN=None, CST=None):
        """End the session with the broker."""
        return session.end_session(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst)
    
    def session_details(self, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        return session.session_details(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
    
    def switch_active_account(self, account_id=None, account_name=None, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        if account_id is None or self.all_accounts is None:
            logging.info("AccountID and/or all_accounts is None. Initializing them now.")
            self.all_accounts = account.list_all_accounts(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
            self.account_id, self.account_name = br_util.get_account_id_by_name(self.all_accounts, account_name=account_name or mark_config.ACCOUNT_TEST)
        if self.account_id is None and self.account_name is None:
            logging.error("Error switching active account! Unable to switch!")
            return None
        return session.switch_active_account(self.account_id, self.account_name, X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst, print_answer=print_answer)
    
    # ==================== DATA METHODS ====================
    
    def get_historical_data(self, epic:str, resolution:str, 
                            from_date:str, to_date:str,  # format 2022-02-24T00:00:00
                            X_SECURITY_TOKEN=None, CST=None,
                            max=1000, print_answer=False):
        """Fetch historical OHLCV data."""
        return markets_info.historical_prices(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,
                                              epic=epic, resolution=resolution, from_date=from_date, to_date=to_date,
                                              max=max, print_answer=print_answer)
    
    def fetch_and_save_historical_prices(self, epic:str, resolution:str, 
                                    from_date:str, to_date:str,  # format 2022-02-24T00:00:00
                                    output_file=None,
                                    X_SECURITY_TOKEN=None, CST=None,
                                    print_answer=False, save_raw_data=False):
        return markets_info.fetch_and_save_historical_prices(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,
                                                             epic=epic, resolution=resolution, from_date=from_date, to_date=to_date,
                                                             output_file=output_file, print_answer=print_answer, save_raw_data=save_raw_data)
    
    # ==================== ACCOUNT METHODS ====================
    
    def list_all_accounts(self, X_SECURITY_TOKEN=None, CST=None, print_answer=False):
        """Returns a list of accounts belonging to the logged-in client."""
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
    
    def all_positions(self, X_SECURITY_TOKEN=None, CST=None, print_answer=True):
        """Returns all open positions for the active account."""
        return trading.all_positions(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,print_answer=print_answer)
    
    def place_market_order(self, symbol, direction, size, stop_amount=None, profit_amount=None, stop_level=None, profit_level=None,
                            X_SECURITY_TOKEN=None, CST=None,
                            print_answer=True):
        """
        Create orders and positions.
        Note: The deal reference you get as "confirmation" from successfully creating a new position
        is not the same dealReference the order has (when active) and not the same as dealId.
        
        Args:
            symbol: Instrument epic identifier. Ex. SILVER
            direction: Deal direction. Must be BUY or SELL
            size: Deal size. Ex. 1
            stop_amount: Loss amount when a stop loss will be triggered. Ex. 4
            profit_amount: Profit amount when a take profit will be triggered. Ex. 20
            print_answer: If true, prints response body and headers. Default is False
        
        Return:
            Deal Reference / deal ID
        """
        return trading.create_new_position(X_SECURITY_TOKEN=X_SECURITY_TOKEN or self.x_security_token, CST=CST or self.cst,print_answer=print_answer,
                                        symbol=symbol, direction=direction, size=size, stop_amount=stop_amount, profit_amount=profit_amount, stop_level=stop_level, profit_level=profit_level)
    
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