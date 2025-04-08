from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from core.events import OrderEvent, FillEvent


class BaseExecutionHandler(ABC):
    """Base interface for all execution handlers (both sim and live)."""
    
    @abstractmethod
    def execute_order(self, order: OrderEvent, market_data: Dict[str, Any]) -> Optional[FillEvent]:
        """Execute an order and return a fill event if successful."""
        pass