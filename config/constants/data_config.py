from dataclasses import dataclass
import os
from typing import List, Dict, Any, Optional
import config.constants.system_config as sys_config

# Directories
TESTING_RAW_FILE = os.path.join(sys_config.CAPCOM_RAW_DATA_DIR, 'raw_GBPUSD_m5_20240101_20250101.csv')
TESTING_PROCESSED_DATA = os.path.join(sys_config.CAPCOM_PROCESSED_DATA_DIR, 'processed_GBPUSD_m5_20240101_20250101.csv')
TEST_DUMMY_PATH = os.path.join(sys_config.BASE_DATA_DIR, 'testing')

DEFAULT_PIPELINE_CONFIG = {
    'loader': {
        'type': 'capital_com',
        'params': {'resolution': 'MINUTE_5'}
    },
    'cleaner': {
        'outlier_method': 'zscore',
        'outlier_threshold': 3.0,
        'missing_value_method': 'ffill'
    },
    'features': {
        'indicators': ['sma_20', 'sma_50', 'ema_14', 'rsi_14', 'macd'],
        'volatility_metrics': ['atr_14', 'bbands_20_2'],
        'time_features': ['hour', 'day_of_week', 'month']
    },
    'normalization': {
        'method': 'standardize',
        'window': 100  # For rolling normalization
    },
    'split': {
        'test_size': 0.2,
        'validation_size': 0.1,
        'method': 'chronological'  # or 'walk_forward'
    }
}

@dataclass
class DataProcessingConfig:
    cleaner_config: Dict[str, Any]
    normalizer_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    
@dataclass
class DataSourceConfig:
    source_type: str
    symbols: List[str]
    timeframes: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    credentials_path: Optional[str] = None