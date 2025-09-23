# config.py - Complete Configuration for AI Stock Advisor Pro
"""Configuration file for AI Stock Advisor Pro with all fixes applied"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class Secrets:
    """Store API keys and sensitive configuration"""
    NEWS_API_KEY: str = "e910f3c66a6a4a40922a2a265a06897e"  # NewsAPI key
    ALPHA_VANTAGE_API_KEY: str = "L3UIBO60Q48REOXX"  # Alpha Vantage key
    
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
    'backtest_db': 'data/backtests.db',
    'cache_duration_hours': 24,
    'backup_enabled': True,
    'auto_vacuum': True,
    'connection_timeout': 30.0
}

# Model configuration
MODEL_CONFIG = {
    'default_horizons': ['next_week', 'next_month', 'next_quarter'],
    'model_cache_dir': 'model_cache',
    'feature_cache_dir': 'feature_cache_v2',
    'max_models_per_ticker': 3,
    'early_stopping_patience': 10,
    'hyperparameter_trials': 50,
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'random_state': 42,
    'parallel_training': True,
    'max_workers': 4
}

# Application configuration
APP_CONFIG = {
    'page_title': 'AI Stock Advisor Pro - Complete Edition',
    'max_tickers': 100,
    'default_selected_tickers': 10,
    'min_selected_tickers': 1,
    'max_selected_tickers': 50,
    'default_investment_amount': 500000,
    'risk_free_rate': 0.06,  # 6% risk-free rate
    'trading_cost': 0.001,   # 0.1% trading cost
    'session_timeout_minutes': 60,
    'max_memory_usage_gb': 4.0
}

# Stock selection defaults - ENHANCED
STOCK_SELECTION_CONFIG = {
    'default_stocks': [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS"
    ],
    'popular_stocks': [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
        "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "ASIANPAINT.NS", "MARUTI.NS"
    ],
    'sector_mappings': {
        'Banking': ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
        'Technology': ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        'Industrial': ["RELIANCE.NS", "LT.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "COALINDIA.NS", "ONGC.NS"],
        'FMCG': ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "TITAN.NS"],
        'Auto': ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS"],
        'Financial Services': ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFC.NS"],
        'Pharma': ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
        'Telecom': ["BHARTIARTL.NS"],
        'Cement': ["ULTRACEMCO.NS", "SHREECEM.NS"],
        'Chemicals': ["UPL.NS"]
    }
}

# Enhanced data configuration
DATA_CONFIG = {
    'max_period': '5y',
    'default_period': '2y',
    'use_database': True,
    'validate_data': True,
    'min_data_points': 100,
    'parallel_downloads': True,
    'max_workers': 4,
    'retry_attempts': 3,
    'timeout_seconds': 30,
    'backup_sources': ['yfinance'],
    'data_quality_checks': True,
    'auto_cleanup': True,
    'cache_duration_hours': 24,
    'real_time_updates': False
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
    'sentiment_features': False,  # Disabled by default
    'target_horizons': ['next_week', 'next_month', 'next_quarter'],
    'feature_selection_enabled': True,
    'max_features': 50,
    'parallel_processing': True,
    'cache_features': True,
    'advanced_features': True,
    'feature_scaling': True
}

# Enhanced model configuration
ENHANCED_MODEL_CONFIG = {
    'ensemble_size': 3,
    'enable_stacking': True,
    'cross_validation_folds': 5,
    'test_size': 0.2,
    'random_state': 42,
    'feature_selection': True,
    'max_features': 50,
    'hyperparameter_tuning': True,
    'model_types': ['random_forest', 'gradient_boosting', 'logistic_regression'],
    'target_horizons': ['next_week', 'next_month', 'next_quarter'],
    'return_thresholds': {
        'next_week': 0.02,    # 2% weekly return threshold
        'next_month': 0.05,   # 5% monthly return threshold
        'next_quarter': 0.10  # 10% quarterly return threshold
    },
    'cache_models': True,
    'model_cache_dir': 'model_cache',
    'parallel_training': True,
    'max_workers': 4,
    'early_stopping': True,
    'validation_split': 0.2,
    'model_monitoring': True
}

# Backtesting configuration
BACKTEST_CONFIG = {
    'initial_capital': 1000000,  # 10 lakh default
    'transaction_cost_pct': 0.001,  # 0.1% per trade
    'slippage_pct': 0.0005,  # 0.05% slippage
    'min_position_size': 10000,  # Minimum 10k per position
    'max_position_size': 200000,  # Maximum 2 lakh per position
    'max_positions': 20,  # Maximum concurrent positions
    'rebalance_frequency': 'monthly',
    'benchmark': 'NIFTY50',
    'risk_free_rate': 0.06,  # 6% annual
    'max_drawdown_limit': 0.15,  # Stop trading if drawdown > 15%
    'position_sizing_method': 'equal_weight',
    'lookback_window': 252,  # Days for rolling calculations
    'confidence_level': 0.95,  # For VaR calculations
    'enable_enhanced_backtesting': True,
    'save_results': True
}

# Enhanced backtesting configuration
ENHANCED_BACKTEST_CONFIG = {
    'max_correlation': 0.7,
    'max_drawdown_limit': 0.15,
    'var_confidence': 0.95,
    'stress_test_frequency': 20,  # days
    'position_sizing_method': 'kelly',
    'enable_dynamic_hedging': True,
    'enable_correlation_monitoring': True,
    'enable_stress_testing': True,
    'rebalance_on_risk_breach': True,
    'prediction_confidence_threshold': 0.6,
    'ensemble_weight_decay': 0.1,
    'signal_aggregation_method': 'weighted_average'
}

# Risk management configuration
RISK_CONFIG = {
    'max_portfolio_risk': 0.02,  # 2% max portfolio risk per trade
    'max_correlation': 0.7,      # Maximum correlation between positions
    'max_drawdown': 0.15,        # Maximum portfolio drawdown
    'var_confidence': 0.95,      # VaR confidence level
    'max_concentration': 0.20,   # Maximum position concentration
    'kelly_fraction': 0.25,      # Kelly criterion fraction
    'risk_parity_lookback': 252,   # Risk parity lookback period
    'volatility_target': 0.15,   # Target portfolio volatility
    'stress_scenarios': [
        'market_crash_2008', 'covid_2020', 'dotcom_2000', 'custom_stress'
    ],
    'monte_carlo_simulations': 1000,
    'confidence_intervals': [0.95, 0.99],
    'hedge_threshold': 0.05,     # Hedge when portfolio correlation > threshold
    'rebalance_frequency': 5,      # Rebalance every N days
    'correlation_window': 60,      # Rolling correlation window
    'enable_risk_monitoring': True,
    'risk_alerts': True
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'default_theme': 'plotly_white',
    'color_scheme': {
        'positive': '#26C6DA',
        'negative': '#EF5350',
        'neutral': '#66BB6A',
        'primary': '#1976D2',
        'secondary': '#7B1FA2'
    },
    'chart_height': 400,
    'chart_width': None,  # Auto-width
    'animation_enabled': True,
    'interactive_charts': True,
    'export_formats': ['png', 'html', 'pdf'],
    'default_timeframe': '1Y'
}

# Performance and optimization configuration
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_ttl_hours': 24,
    'parallel_processing': True,
    'max_workers': 4,
    'memory_limit_gb': 4.0,
    'cpu_limit_percent': 80,
    'enable_profiling': False,
    'log_performance_metrics': True,
    'optimize_for_memory': True,
    'enable_gpu_acceleration': False  # For future ML acceleration
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/ai_stock_advisor.log',
    'max_file_size_mb': 10,
    'backup_count': 5,
    'console_logging': True,
    'file_logging': True,
    'log_data_operations': True,
    'log_model_operations': True,
    'log_api_calls': False  # Set to True for debugging
}

# API configuration (for external data sources)
API_CONFIG = {
    'rate_limits': {
        'yfinance': {'requests_per_minute': 100},
        'news_api': {'requests_per_minute': 60},
        'alpha_vantage': {'requests_per_minute': 5}
    },
    'timeout_seconds': 30,
    'retry_attempts': 3,
    'retry_delay_seconds': 1,
    'enable_rate_limiting': True,
    'user_agent': 'AI-Stock-Advisor-Pro/1.0'
}

# Security configuration
SECURITY_CONFIG = {
    'enable_ssl': True,
    'session_security': True,
    'input_validation': True,
    'sql_injection_protection': True,
    'xss_protection': True,
    'rate_limiting': True,
    'api_key_encryption': True,
    'secure_headers': True
}

# Development and debugging configuration
DEBUG_CONFIG = {
    'debug_mode': False,
    'verbose_logging': False,
    'show_stack_traces': False,
    'enable_profiler': False,
    'memory_monitoring': False,
    'performance_tracking': False,
    'mock_data_mode': False,  # For testing without real data
    'disable_external_apis': False  # For offline testing
}

# Export configuration
EXPORT_CONFIG = {
    'supported_formats': ['xlsx', 'csv', 'json', 'pdf'],
    'default_format': 'xlsx',
    'include_charts': True,
    'include_raw_data': False,
    'compress_exports': True,
    'max_export_size_mb': 50,
    'export_timeout_seconds': 120
}

# Notification configuration
NOTIFICATION_CONFIG = {
    'enable_notifications': False,  # Disabled by default
    'email_notifications': False,
    'sms_notifications': False,
    'push_notifications': False,
    'slack_notifications': False,
    'discord_notifications': False,
    'notification_frequency': 'daily',
    'alert_thresholds': {
        'high_risk': 0.8,
        'significant_movement': 0.05,
        'model_accuracy_drop': 0.1
    }
}

# Cloud and deployment configuration
DEPLOYMENT_CONFIG = {
    'environment': 'development',  # development, staging, production
    'cloud_provider': None,  # aws, azure, gcp, None for local
    'container_deployment': False,
    'auto_scaling': False,
    'load_balancing': False,
    'cdn_enabled': False,
    'backup_strategy': 'local',
    'monitoring_enabled': False,
    'health_checks': True
}

# Integration configuration
INTEGRATION_CONFIG = {
    'enable_slack_integration': False,
    'enable_teams_integration': False,
    'enable_discord_integration': False,
    'enable_telegram_integration': False,
    'webhook_endpoints': [],
    'api_endpoints': [],
    'third_party_apis': {
        'portfolio_management': False,
        'trading_platforms': False,
        'news_feeds': False,
        'social_sentiment': False
    }
}

# Machine learning specific configuration
ML_CONFIG = {
    'auto_feature_selection': True,
    'feature_importance_threshold': 0.01,
    'model_selection_metric': 'roc_auc',
    'cross_validation_strategy': 'time_series',
    'hyperparameter_optimization': 'optuna',
    'ensemble_methods': ['voting', 'stacking', 'blending'],
    'model_interpretability': True,
    'model_monitoring': True,
    'auto_retrain_threshold': 0.1,  # Retrain if performance drops by 10%
    'feature_drift_detection': True,
    'data_drift_detection': True
}

# Consolidated configuration dictionary
CONFIG = {
    'secrets': secrets,
    'database': DATABASE_CONFIG,
    'model': MODEL_CONFIG,
    'app': APP_CONFIG,
    'stock_selection': STOCK_SELECTION_CONFIG,
    'data': DATA_CONFIG,
    'features': FEATURE_CONFIG,
    'enhanced_model': ENHANCED_MODEL_CONFIG,
    'backtest': BACKTEST_CONFIG,
    'enhanced_backtest': ENHANCED_BACKTEST_CONFIG,
    'risk': RISK_CONFIG,
    'visualization': VISUALIZATION_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'logging': LOGGING_CONFIG,
    'api': API_CONFIG,
    'security': SECURITY_CONFIG,
    'debug': DEBUG_CONFIG,
    'export': EXPORT_CONFIG,
    'notification': NOTIFICATION_CONFIG,
    'deployment': DEPLOYMENT_CONFIG,
    'integration': INTEGRATION_CONFIG,
    'ml': ML_CONFIG
}

# Configuration validation functions
def validate_config() -> Dict[str, bool]:
    """Validate configuration settings"""
    validation_results = {}
    
    try:
        # Validate essential directories exist
        essential_dirs = ['data', 'logs', 'model_cache', 'feature_cache_v2']
        for dir_name in essential_dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                    validation_results[f'{dir_name}_created'] = True
                except Exception as e:
                    validation_results[f'{dir_name}_error'] = str(e)
            else:
                validation_results[f'{dir_name}_exists'] = True
        
        # Validate model configuration
        validation_results['model_config_valid'] = (
            isinstance(ENHANCED_MODEL_CONFIG['ensemble_size'], int) and
            ENHANCED_MODEL_CONFIG['ensemble_size'] > 0
        )
        
        # Validate risk configuration
        validation_results['risk_config_valid'] = (
            0 < RISK_CONFIG['max_portfolio_risk'] < 1 and
            0 < RISK_CONFIG['max_correlation'] <= 1
        )
        
        # Validate data configuration
        validation_results['data_config_valid'] = (
            DATA_CONFIG['min_data_points'] > 0 and
            DATA_CONFIG['cache_duration_hours'] > 0
        )
        
    except Exception as e:
        validation_results['validation_error'] = str(e)
    
    return validation_results

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration"""
    return {
        'total_configurations': len(CONFIG),
        'database_enabled': DATABASE_CONFIG['backup_enabled'],
        'caching_enabled': PERFORMANCE_CONFIG['enable_caching'],
        'parallel_processing': PERFORMANCE_CONFIG['parallel_processing'],
        'risk_management': RISK_CONFIG['enable_risk_monitoring'],
        'debug_mode': DEBUG_CONFIG['debug_mode'],
        'environment': DEPLOYMENT_CONFIG['environment']
    }

def update_config(section: str, key: str, value: Any) -> bool:
    """Safely update configuration values"""
    try:
        if section in CONFIG and isinstance(CONFIG[section], dict):
            CONFIG[section][key] = value
            return True
        return False
    except Exception:
        return False

# Export main configuration objects
__all__ = [
    'CONFIG',
    'secrets',
    'DATABASE_CONFIG',
    'MODEL_CONFIG', 
    'APP_CONFIG',
    'STOCK_SELECTION_CONFIG',
    'DATA_CONFIG',
    'FEATURE_CONFIG',
    'ENHANCED_MODEL_CONFIG',
    'BACKTEST_CONFIG',
    'ENHANCED_BACKTEST_CONFIG',
    'RISK_CONFIG',
    'VISUALIZATION_CONFIG',
    'PERFORMANCE_CONFIG',
    'validate_config',
    'get_config_summary',
    'update_config'
]

# Configuration initialization
if __name__ == "__main__":
    print("AI Stock Advisor Pro - Configuration System")
    print("=" * 50)
    
    # Validate configuration
    validation_results = validate_config()
    print(f"Configuration validation results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    # Display configuration summary
    summary = get_config_summary()
    print(f"\nConfiguration summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal configuration sections: {len(CONFIG)}")
    print("Configuration system initialized successfully!")