# config.py
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

# Application configuration
APP_CONFIG = {
    'page_title': 'AI Stock Advisor Pro',
    'max_tickers': 100,
    'default_selected_tickers': 10,  # Default number of stocks to select
    'min_selected_tickers': 1,       # Minimum stocks that must be selected
    'max_selected_tickers': 50,      # Maximum stocks that can be selected
    'default_investment_amount': 500000,
    'risk_free_rate': 0.06,  # 6% risk-free rate assumption
    'trading_cost': 0.001,   # 0.1% trading cost assumption
}

# ADD THIS NEW SECTION - Stock selection defaults
STOCK_SELECTION_CONFIG = {
    'default_stocks': [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS"
    ],
    'popular_stocks': [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
        "HDFC.NS", "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS", "SUNPHARMA.NS", "NESTLEIND.NS"
    ],
    'sector_leaders': {
        'Banking': ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS"],
        'IT': ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        'Oil & Gas': ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS"],
        'FMCG': ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
        'Auto': ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"]
    }
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