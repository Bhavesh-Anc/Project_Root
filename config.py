"""Configuration file for AI Stock Advisor Pro"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Secrets:
    """Store API keys and sensitive configuration"""
    NEWS_API_KEY: str = "e910f3c66a6a4a40922a2a265a06897e"  # Add your NewsAPI key here if available
    ALPHA_VANTAGE_API_KEY: str = "L3UIBO60Q48REOXX"  # Add your Alpha Vantage key here if available
    
    def __post_init__(self):
        # Try to load from environment variables if available
        self.NEWS_API_KEY = os.getenv('NEWS_API_KEY', self.NEWS_API_KEY)
        self.ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', self.ALPHA_VANTAGE_API_KEY)

# Global secrets instance
secrets = Secrets()

# Database configuration
DATABASE_CONFIG = {
    'stock_data_db': 'data/stock_data.db',
    'realtime_data_db': 'data/realtime_data.db',
    'model_monitor_db': 'data/model_monitor.db',
    'cache_duration_hours': 24,
    'backup_enabled': True
}

# Model configuration
MODEL_CONFIG = {
    'default_horizons': ['next_week', 'next_month', 'next_quarter'],
    'model_cache_dir': 'model_cache',
    'feature_cache_dir': 'feature_cache_v2',
    'max_models_per_ticker': 3,
    'early_stopping_patience': 10,
    'hyperparameter_trials': 50
}

# Data configuration
DATA_CONFIG = {
    'default_period': '20y',
    'max_period': '25y',
    'intervals': ['1d', '1wk', '1mo'],
    'default_interval': '1d',
    'max_workers': 8,
    'retry_attempts': 7,
    'retry_delay': 2,
    'exponential_backoff': True,
    'cache_enabled': True,
    'cache_duration_hours': 24,
    'batch_size': 12,
    'request_delay': 0.3,
    'timeout': 90,
    'validate_data': True,
    'use_database': True,
    'use_async': True,
    'fallback_tickers': True,
    'data_quality_threshold': 0.75
}

# Feature engineering configuration 
FEATURE_CONFIG = {
    'lookback_periods': [5, 10, 20, 50],
    'technical_indicators': [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 
        'stochastic', 'atr', 'cci', 'williams_r', 'obv'
    ],
    'price_features': True,
    'volume_features': True,
    'volatility_features': True,
    'momentum_features': True,
    'trend_features': True,
    'pattern_features': True,
    'market_microstructure': True,
    'sentiment_features': True,
    'target_horizons': ['next_week', 'next_month', 'next_quarter', 'next_year'],
    'feature_selection_enabled': True,
    'parallel_processing': True,
    'cache_features': True,
    'advanced_features': True
}

# Application configuration
APP_CONFIG = {
    'page_title': 'AI Stock Advisor Pro',
    'max_tickers': 100,
    'default_investment_amount': 500000,
    'risk_free_rate': 0.06,  # 6% risk-free rate assumption
    'trading_cost': 0.001,   # 0.1% trading cost assumption
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'log_file': 'logs/stock_advisor.log'
}

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'logs', 
        'model_cache',
        'feature_cache_v2',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories on import
create_directories()