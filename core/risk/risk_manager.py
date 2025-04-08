from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging
from datetime import datetime

from core.events import OrderEvent, OrderType, OrderSide


class PositionSizer:
    """Determines position size based on risk parameters."""
    
    def __init__(self, 
                 method: str = 'fixed',
                 params: Dict[str, Any] = None,
                 logger=None):
        """
        Initialize position sizer.
        
        Args:
            method: Position sizing method
                - 'fixed': Fixed position size
                - 'percent': Percentage of portfolio equity
                - 'risk': Fixed risk percentage
                - 'kelly': Kelly criterion
                - 'volatility': Volatility-adjusted
            params: Parameters for the sizing method
            logger: Custom logger
        """
        self.method = method
        self.params = params or {}
        self.logger = logger or self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.PositionSizer")
        logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def calculate_position_size(self, 
                               symbol: str, 
                               signal_type: str,
                               current_price: float,
                               portfolio: Any,
                               risk_params: Dict[str, Any] = None,
                               market_data: Dict[str, Any] = None) -> float:
        """
        Calculate position size based on the chosen method.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal (BUY, SELL)
            current_price: Current market price
            portfolio: Portfolio object with current state
            risk_params: Additional risk parameters
            market_data: Market data for volatility calculations
            
        Returns:
            Position size in units
        """
        if self.method == 'fixed':
            return self._fixed_size(symbol, signal_type, current_price, portfolio, risk_params)
        
        elif self.method == 'percent':
            return self._percent_equity(symbol, signal_type, current_price, portfolio, risk_params)
        
        elif self.method == 'risk':
            return self._fixed_risk(symbol, signal_type, current_price, portfolio, risk_params, market_data)
        
        elif self.method == 'kelly':
            return self._kelly_criterion(symbol, signal_type, current_price, portfolio, risk_params)
        
        elif self.method == 'volatility':
            return self._volatility_adjusted(symbol, signal_type, current_price, portfolio, risk_params, market_data)
        
        else:
            self.logger.warning(f"Unknown position sizing method: {self.method}, using fixed size")
            return self._fixed_size(symbol, signal_type, current_price, portfolio, risk_params)
    
    def _fixed_size(self, symbol, signal_type, current_price, portfolio, risk_params):
        """
        Fixed position size.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            current_price: Current market price
            portfolio: Portfolio object
            risk_params: Additional risk parameters
            
        Returns:
            Fixed position size
        """
        # Use risk_params if provided, otherwise fall back to self.params
        params = risk_params or self.params
        
        # Get fixed size parameter (default: 1.0)
        fixed_size = params.get('size', 1.0)
        
        return fixed_size
    
    def _percent_equity(self, symbol, signal_type, current_price, portfolio, risk_params):
        """
        Percentage of portfolio equity.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            current_price: Current market price
            portfolio: Portfolio object
            risk_params: Additional risk parameters
            
        Returns:
            Position size based on portfolio percentage
        """
        # Use risk_params if provided, otherwise fall back to self.params
        params = risk_params or self.params
        
        # Get percentage parameter (default: 10%)
        percent = params.get('percent', 10.0) / 100.0
        
        # Calculate position size
        equity = portfolio.equity
        position_value = equity * percent
        
        # Convert to units
        if current_price > 0:
            position_size = position_value / current_price
        else:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}, using fixed size 1.0")
            position_size = 1.0
        
        return position_size
    
    def _fixed_risk(self, symbol, signal_type, current_price, portfolio, risk_params, market_data):
        """
        Fixed risk percentage with stop loss.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            current_price: Current market price
            portfolio: Portfolio object
            risk_params: Additional risk parameters
            market_data: Market data for calculating stop loss
            
        Returns:
            Position size based on risk percentage
        """
        # Use risk_params if provided, otherwise fall back to self.params
        params = risk_params or self.params
        
        # Get risk percentage parameter (default: 2%)
        risk_percent = params.get('risk_percent', 2.0) / 100.0
        
        # Get stop loss distance (default: 2%)
        stop_loss_percent = params.get('stop_loss_percent', 2.0) / 100.0
        
        # Calculate stop loss price
        if signal_type.upper() in ['BUY', 'LONG']:
            stop_loss = current_price * (1 - stop_loss_percent)
        else:  # SELL, SHORT
            stop_loss = current_price * (1 + stop_loss_percent)
        
        # Calculate position size
        equity = portfolio.equity
        risk_amount = equity * risk_percent
        
        # Calculate risk per unit
        risk_per_unit = abs(current_price - stop_loss)
        
        # Calculate position size
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            self.logger.warning(f"Invalid risk per unit for {symbol}: {risk_per_unit}, using fixed size 1.0")
            position_size = 1.0
        
        return position_size
    
    def _kelly_criterion(self, symbol, signal_type, current_price, portfolio, risk_params):
        """
        Kelly criterion position sizing.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            current_price: Current market price
            portfolio: Portfolio object
            risk_params: Additional risk parameters
            
        Returns:
            Position size based on Kelly criterion
        """
        # Use risk_params if provided, otherwise fall back to self.params
        params = risk_params or self.params
        
        # Kelly parameters
        win_rate = params.get('win_rate', 0.5)
        win_loss_ratio = params.get('win_loss_ratio', 1.0)
        
        # Apply Kelly fraction
        kelly_fraction = params.get('kelly_fraction', 0.5)  # Half-Kelly is more conservative
        
        # Calculate Kelly percentage
        if win_loss_ratio > 0:
            kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        else:
            kelly_percent = 0
        
        # Apply fraction and constraints
        kelly_percent = max(0, min(kelly_percent * kelly_fraction, 0.2))  # Cap at 20%
        
        # Calculate position size
        equity = portfolio.equity
        position_value = equity * kelly_percent
        
        # Convert to units
        if current_price > 0:
            position_size = position_value / current_price
        else:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}, using fixed size 1.0")
            position_size = 1.0
        
        return position_size
    
    def _volatility_adjusted(self, symbol, signal_type, current_price, portfolio, risk_params, market_data):
        """
        Volatility-adjusted position sizing.
        
        Args:
            symbol: Market symbol
            signal_type: Type of signal
            current_price: Current market price
            portfolio: Portfolio object
            risk_params: Additional risk parameters
            market_data: Market data for volatility calculation
            
        Returns:
            Position size adjusted for volatility
        """
        # Use risk_params if provided, otherwise fall back to self.params
        params = risk_params or self.params
        
        # Get target volatility parameter (default: 1%)
        target_volatility = params.get('target_volatility', 1.0) / 100.0
        
        # Get portfolio volatility scaling (default: True)
        portfolio_scaling = params.get('portfolio_scaling', True)
        
        # Get target risk parameter (default: 5%)
        target_risk = params.get('target_risk', 5.0) / 100.0
        
        # Calculate market volatility
        volatility = 0.02  # Default 2% daily volatility
        
        # Extract volatility from market data if available
        if market_data and symbol in market_data:
            data = market_data[symbol]
            
            if isinstance(data, dict):
                # Check if volatility is directly available
                if 'volatility' in data:
                    volatility = data['volatility']
                # If not, check for volatility fields from the feature generator
                elif 'volatility_20' in data:
                    volatility = data['volatility_20']
                elif 'atr_14' in data:
                    # ATR as percentage of price
                    volatility = data['atr_14'] / current_price
        
        # Adjust for portfolio volatility if enabled
        if portfolio_scaling and hasattr(portfolio, 'get_volatility'):
            portfolio_vol = portfolio.get_volatility()
            if portfolio_vol > 0:
                # Scale based on portfolio vol vs target vol
                scale_factor = target_volatility / portfolio_vol
                target_risk *= scale_factor
        
        # Calculate position size based on volatility
        equity = portfolio.equity
        position_value = equity * target_risk / volatility
        
        # Convert to units
        if current_price > 0:
            position_size = position_value / current_price
        else:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}, using fixed size 1.0")
            position_size = 1.0
        
        return position_size


class RiskManager:
    """
    Manages risk and position sizing for the backtesting system.
    
    The risk manager is responsible for:
    1. Determining position size based on risk parameters
    2. Setting stop-loss and take-profit levels
    3. Filtering orders based on risk rules
    4. Managing portfolio-level risk constraints
    """
    
    def __init__(self, 
                 position_sizer=None,
                 position_sizing_method: str = 'percent',
                 position_sizing_params: Dict[str, Any] = None,
                 max_position_size: Optional[float] = None,
                 max_correlated_positions: int = 5,
                 max_portfolio_risk: float = 20.0,  # 20% max portfolio risk
                 auto_stop_loss: bool = True,
                 stop_loss_method: str = 'percent',
                 stop_loss_params: Dict[str, Any] = None,
                 auto_take_profit: bool = True,
                 take_profit_method: str = 'percent',
                 take_profit_params: Dict[str, Any] = None,
                 logger=None):
        """
        Initialize the risk manager.
        
        Args:
            position_sizer: Custom position sizer object
            position_sizing_method: Method for position sizing
            position_sizing_params: Parameters for position sizing
            max_position_size: Maximum position size
            max_correlated_positions: Maximum correlated positions
            max_portfolio_risk: Maximum portfolio risk percentage
            auto_stop_loss: Whether to set stop loss automatically
            stop_loss_method: Method for setting stop loss
            stop_loss_params: Parameters for stop loss
            auto_take_profit: Whether to set take profit automatically
            take_profit_method: Method for setting take profit
            take_profit_params: Parameters for take profit
            logger: Custom logger
        """
        self.logger = logger or self._setup_logger()
        
        # Position sizing
        self.position_sizer = position_sizer or PositionSizer(
            method=position_sizing_method,
            params=position_sizing_params or {
                'percent': 10.0,  # Default to 10% of equity
                'risk_percent': 1.0,  # Default to 1% risk per trade
                'stop_loss_percent': 2.0  # Default to 2% stop loss
            },
            logger=self.logger
        )
        
        # Risk constraints
        self.max_position_size = max_position_size
        self.max_correlated_positions = max_correlated_positions
        self.max_portfolio_risk = max_portfolio_risk / 100.0  # Convert to decimal
        
        # Stop loss settings
        self.auto_stop_loss = auto_stop_loss
        self.stop_loss_method = stop_loss_method
        self.stop_loss_params = stop_loss_params or {
            'percent': 2.0,  # Default 2% stop loss
            'atr_multiple': 2.0,  # Default 2x ATR
            'min_percent': 1.0,  # Minimum 1%
            'max_percent': 5.0  # Maximum 5%
        }
        
        # Take profit settings
        self.auto_take_profit = auto_take_profit
        self.take_profit_method = take_profit_method
        self.take_profit_params = take_profit_params or {
            'percent': 5.0,  # Default 5% take profit
            'risk_reward_ratio': 2.0  # Default 2:1 risk/reward
        }
        
        # Risk state tracking
        self.position_risk = {}  # symbol -> risk amount
        self.portfolio_risk = 0.0  # Current portfolio risk
        self.correlated_groups = {}  # group_id -> list of symbols
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger(f"{__name__}.RiskManager")
        logger.setLevel(logging.INFO)
        
        # Add handlers if they don't exist
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def process_signal(self, 
                      signal,
                      portfolio,
                      market_data=None) -> Optional[OrderEvent]:
        """
        Process a signal and create an order based on risk parameters.
        
        Args:
            signal: Signal event
            portfolio: Portfolio object
            market_data: Market data for price and volatility
            
        Returns:
            Order event or None if signal is rejected
        """
        symbol = signal.symbol
        signal_type = signal.signal_type.value
        
        # Get current market price
        current_price = self._get_current_price(symbol, market_data)
        if current_price is None:
            self.logger.warning(f"Could not get current price for {symbol}, rejecting signal")
            return None
        
        # Check if portfolio can accept new positions
        if not self._check_portfolio_capacity(symbol, portfolio):
            self.logger.info(f"Portfolio capacity reached, rejecting signal for {symbol}")
            return None
        
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            symbol=symbol,
            signal_type=signal_type,
            current_price=current_price,
            portfolio=portfolio,
            market_data=market_data
        )
        
        # Apply maximum position size constraint
        if self.max_position_size is not None:
            position_size = min(position_size, self.max_position_size)
        
        # Skip if position size is too small
        if position_size <= 0:
            self.logger.info(f"Position size too small for {symbol}, rejecting signal")
            return None
        
        # Calculate stop loss and take profit levels
        stop_loss = None
        take_profit = None
        
        if self.auto_stop_loss:
            stop_loss = self._calculate_stop_loss(
                symbol=symbol,
                signal_type=signal_type,
                current_price=current_price,
                market_data=market_data
            )
        
        if self.auto_take_profit:
            take_profit = self._calculate_take_profit(
                symbol=symbol,
                signal_type=signal_type,
                current_price=current_price,
                stop_loss=stop_loss,
                market_data=market_data
            )
        
        # Calculate risk for this position
        position_risk = self._calculate_position_risk(
            symbol=symbol,
            signal_type=signal_type,
            current_price=current_price,
            position_size=position_size,
            stop_loss=stop_loss
        )
        
        # Check if adding this position would exceed portfolio risk limit
        if not self._check_portfolio_risk(position_risk, portfolio):
            self.logger.info(f"Position would exceed portfolio risk limit, rejecting signal for {symbol}")
            return None
        
        # Create order event
        order_side = OrderSide.BUY if signal_type in ['BUY', 'LONG'] else OrderSide.SELL
        
        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type=OrderType.MARKET,
            order_side=order_side,
            quantity=position_size,
            stop_price=stop_loss,
            signal_id=id(signal),
            metadata={
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_risk': position_risk
            }
        )
        
        self.logger.info(f"Created order from signal: {order}, "
                       f"stop_loss: {stop_loss}, take_profit: {take_profit}")
        
        return order
    
    def _get_current_price(self, symbol, market_data) -> Optional[float]:
        """Get current price from market data."""
        if market_data is None:
            return None
        
        # Handle different data formats
        if symbol in market_data:
            data = market_data[symbol]
            
            if isinstance(data, dict):
                # Try to get close price first
                price = data.get('close', data.get('Close'))
                
                # If no close price, try other price fields
                if price is None:
                    for field in ['price', 'last', 'bid', 'ask', 'mid']:
                        price = data.get(field)
                        if price is not None:
                            break
                
                return price
            
            elif hasattr(data, 'data'):
                # Try MarketEvent object
                return data.data.get('close', data.data.get('Close'))
        
        return None
    
    def _check_portfolio_capacity(self, symbol, portfolio) -> bool:
        """Check if portfolio has capacity for a new position."""
        # Check number of open positions
        open_positions = portfolio.positions
        
        # If symbol already has a position, always allow sizing
        if symbol in open_positions:
            return True
        
        # Check number of correlated positions
        if self.max_correlated_positions > 0:
            # Find correlation group for this symbol
            group_id = None
            for group, symbols in self.correlated_groups.items():
                if symbol in symbols:
                    group_id = group
                    break
            
            if group_id is not None:
                # Count positions in this correlation group
                group_positions = sum(1 for s in self.correlated_groups[group_id] if s in open_positions)
                
                if group_positions >= self.max_correlated_positions:
                    self.logger.info(f"Maximum correlated positions reached for group {group_id}")
                    return False
        
        return True
    
    def _calculate_stop_loss(self, symbol, signal_type, current_price, market_data) -> Optional[float]:
        """Calculate stop loss level based on the chosen method."""
        if self.stop_loss_method == 'percent':
            # Get stop loss percentage
            stop_loss_percent = self.stop_loss_params.get('percent', 2.0) / 100.0
            
            # Calculate stop level
            if signal_type in ['BUY', 'LONG']:
                return current_price * (1 - stop_loss_percent)
            else:  # SELL, SHORT
                return current_price * (1 + stop_loss_percent)
        
        elif self.stop_loss_method == 'atr':
            # Get ATR multiple
            atr_multiple = self.stop_loss_params.get('atr_multiple', 2.0)
            
            # Get ATR value
            atr = 0.02 * current_price  # Default 2% of price if no ATR available
            
            if market_data and symbol in market_data:
                data = market_data[symbol]
                
                if isinstance(data, dict):
                    if 'atr_14' in data:
                        atr = data['atr_14']
                    elif 'ATR' in data:
                        atr = data['ATR']
            
            # Calculate stop level
            if signal_type in ['BUY', 'LONG']:
                return current_price - (atr * atr_multiple)
            else:  # SELL, SHORT
                return current_price + (atr * atr_multiple)
        
        elif self.stop_loss_method == 'support_resistance':
            # This would require more complex analysis of support/resistance levels
            # Default to percent method for now
            stop_loss_percent = self.stop_loss_params.get('percent', 2.0) / 100.0
            
            if signal_type in ['BUY', 'LONG']:
                return current_price * (1 - stop_loss_percent)
            else:  # SELL, SHORT
                return current_price * (1 + stop_loss_percent)
        
        else:
            return None
    
    def _calculate_take_profit(self, symbol, signal_type, current_price, stop_loss, market_data) -> Optional[float]:
        """Calculate take profit level based on the chosen method."""
        if self.take_profit_method == 'percent':
            # Get take profit percentage
            take_profit_percent = self.take_profit_params.get('percent', 5.0) / 100.0
            
            # Calculate take profit level
            if signal_type in ['BUY', 'LONG']:
                return current_price * (1 + take_profit_percent)
            else:  # SELL, SHORT
                return current_price * (1 - take_profit_percent)
        
        elif self.take_profit_method == 'risk_reward':
            # Get risk-reward ratio
            risk_reward_ratio = self.take_profit_params.get('risk_reward_ratio', 2.0)
            
            # Stop loss must be available
            if stop_loss is None:
                return None
            
            # Calculate risk amount
            risk_amount = abs(current_price - stop_loss)
            
            # Calculate take profit level
            if signal_type in ['BUY', 'LONG']:
                return current_price + (risk_amount * risk_reward_ratio)
            else:  # SELL, SHORT
                return current_price - (risk_amount * risk_reward_ratio)
        
        else:
            return None
    
    def _calculate_position_risk(self, symbol, signal_type, current_price, position_size, stop_loss) -> float:
        """Calculate the risk amount for a position in currency terms."""
        # If stop loss is provided, use it to calculate risk
        if stop_loss is not None:
            risk_per_unit = abs(current_price - stop_loss)
            return risk_per_unit * position_size
        
        # Otherwise, use default risk percentage
        default_risk_percent = self.stop_loss_params.get('percent', 2.0) / 100.0
        return current_price * position_size * default_risk_percent
    
    def _check_portfolio_risk(self, position_risk, portfolio) -> bool:
        """
        Check if adding a position would exceed the portfolio risk limit.
        
        Args:
            position_risk: Risk amount for the new position
            portfolio: Portfolio object
            
        Returns:
            True if position can be added, False otherwise
        """
        equity = portfolio.equity
        
        # Current risk
        current_risk = sum(self.position_risk.values())
        
        # New total risk with this position
        total_risk = current_risk + position_risk
        
        # Calculate as percentage of equity
        risk_percent = total_risk / equity if equity > 0 else float('inf')
        
        # Check against limit
        return risk_percent <= self.max_portfolio_risk
    
    def update_position_risk(self, symbol, risk_amount=None):
        """
        Update the risk amount for a position.
        
        Args:
            symbol: Market symbol
            risk_amount: New risk amount (None to remove)
        """
        if risk_amount is None:
            # Remove position risk
            if symbol in self.position_risk:
                del self.position_risk[symbol]
        else:
            # Update position risk
            self.position_risk[symbol] = risk_amount
        
        # Recalculate total risk
        self.portfolio_risk = sum(self.position_risk.values())
    
    def set_correlated_symbols(self, group_id: str, symbols: List[str]):
        """
        Set a group of correlated symbols.
        
        Args:
            group_id: Correlation group identifier
            symbols: List of correlated symbols
        """
        self.correlated_groups[group_id] = symbols
        self.logger.info(f"Set correlation group {group_id} with {len(symbols)} symbols")
    
    def apply_portfolio_risk_rules(self, portfolio) -> List[Dict[str, Any]]:
        """
        Apply portfolio-wide risk rules and return actions to take.
        
        Args:
            portfolio: Portfolio object
            
        Returns:
            List of risk actions to take
        """
        equity = portfolio.equity
        actions = []
        
        # Check if portfolio risk exceeds maximum
        total_risk = sum(self.position_risk.values())
        risk_percent = total_risk / equity if equity > 0 else 0
        
        if risk_percent > self.max_portfolio_risk:
            self.logger.warning(f"Portfolio risk {risk_percent*100:.2f}% exceeds maximum {self.max_portfolio_risk*100:.2f}%")
            
            # Calculate how much risk to reduce
            excess_risk = total_risk - (self.max_portfolio_risk * equity)
            
            # Find positions to reduce, starting with highest risk
            positions_by_risk = sorted(
                self.position_risk.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            risk_to_reduce = excess_risk
            for symbol, risk in positions_by_risk:
                if risk_to_reduce <= 0:
                    break
                
                # Calculate reduction
                reduction_amount = min(risk, risk_to_reduce)
                reduction_percent = reduction_amount / risk
                
                # Add action
                actions.append({
                    'type': 'reduce_position',
                    'symbol': symbol,
                    'percent': reduction_percent,
                    'reason': 'portfolio_risk_exceeded'
                })
                
                risk_to_reduce -= reduction_amount
        
        return actions