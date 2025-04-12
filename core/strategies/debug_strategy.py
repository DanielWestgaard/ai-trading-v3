# backtesting/strategies/debug_strategy.py
import pandas as pd
from backtesting.strategies.base_strategy import BaseStrategy

class DebugStrategy(BaseStrategy):
    """Debug strategy to diagnose data issues."""
    
    def __init__(self, symbols, params=None, logger=None):
        super().__init__(symbols, params, logger)
        self.data_analyzed = False
    
    def generate_signals(self, market_data, portfolio):
        if not self.data_analyzed:
            self.logger.info("=== DEBUGGING DATA CONTENTS ===")
            
            for symbol, data in market_data.items():
                self.logger.info(f"Symbol: {symbol}")
                self.logger.info(f"Available columns: {list(data.data.keys())}")
                
                # Check for various price columns 
                price_cols_found = []
                for col in data.data.keys():
                    if any(name in col.lower() for name in ['close', 'open', 'high', 'low', 'price']):
                        price_cols_found.append(col)
                        self.logger.info(f"Potential price column: {col} = {data.data[col]}")
                
                if not price_cols_found:
                    self.logger.warning("No price columns found in the data!")
                
                # Print first 5 numeric columns as samples
                self.logger.info("Sample numeric columns:")
                count = 0
                for col, val in data.data.items():
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        self.logger.info(f"  {col}: {val}")
                        count += 1
                        if count >= 5:
                            break
            
            self.data_analyzed = True
        
        return []  # No signals