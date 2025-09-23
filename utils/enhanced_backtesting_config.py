# utils/enhanced_backtesting_config.py
"""
Enhanced Backtesting Configuration Module
Provides configuration classes and utilities for advanced backtesting
Author: AI Stock Advisor Pro Team
Version: 2.0 - Enhanced Edition
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import logging
from enum import Enum

# ==================== ENUMS ====================

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGETING = "volatility_targeting"
    FIXED_AMOUNT = "fixed_amount"
    MARKET_CAP_WEIGHTED = "market_cap_weighted"

class RebalanceFrequency(Enum):
    """Rebalancing frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"

class BenchmarkType(Enum):
    """Benchmark types"""
    NIFTY_50 = "^NSEI"
    BANK_NIFTY = "^NSEBANK"
    NIFTY_IT = "^CNXIT"
    NIFTY_MIDCAP = "^NSMIDCP"
    CUSTOM = "custom"

# ==================== CONFIGURATION CLASSES ====================

@dataclass
class CapitalManagementConfig:
    """Capital management configuration"""
    initial_capital: float = 1000000.0
    minimum_position_size: float = 10000.0
    maximum_position_size: float = 200000.0
    reserve_cash_percentage: float = 0.05
    leverage_limit: float = 1.0
    margin_requirement: float = 0.25

@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_drawdown_limit: float = 0.15
    max_daily_loss_limit: float = 0.02
    max_position_correlation: float = 0.7
    var_confidence_level: float = 0.95
    stress_test_frequency: int = 5
    max_sector_exposure: float = 0.3
    max_single_position: float = 0.2
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_trailing_stop: bool = False

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    min_prediction_confidence: float = 0.6
    profit_target_percentage: float = 0.25
    stop_loss_percentage: float = 0.10
    min_holding_days: int = 3
    max_holding_days: int = 60
    enable_short_selling: bool = False
    market_timing_enabled: bool = False
    sector_rotation_enabled: bool = False

@dataclass
class TransactionConfig:
    """Transaction cost configuration"""
    brokerage_percentage: float = 0.0005
    exchange_charges: float = 0.00005
    gst_percentage: float = 0.18
    stt_percentage: float = 0.001
    stamp_duty_percentage: float = 0.00015
    sebi_charges: float = 0.000001
    slippage_percentage: float = 0.0005
    
    def total_transaction_cost(self) -> float:
        """Calculate total transaction cost percentage"""
        base_cost = (self.brokerage_percentage + self.exchange_charges + 
                    self.stt_percentage + self.stamp_duty_percentage + self.sebi_charges)
        return base_cost * (1 + self.gst_percentage) + self.slippage_percentage

@dataclass
class BacktestTimeConfig:
    """Backtest time configuration"""
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=730))
    end_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    trading_hours_only: bool = True
    skip_holidays: bool = True
    warm_up_period: int = 60  # Days for model warm-up

@dataclass
class PerformanceTargetsConfig:
    """Performance targets configuration"""
    target_annual_return: float = 0.20
    target_sharpe_ratio: float = 1.5
    target_sortino_ratio: float = 2.0
    max_acceptable_drawdown: float = 0.12
    target_win_rate: float = 0.55
    target_profit_factor: float = 1.8
    benchmark_type: BenchmarkType = BenchmarkType.NIFTY_50
    benchmark_symbol: str = "^NSEI"

@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration"""
    num_simulations: int = 1000
    confidence_intervals: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95, 0.99])
    random_seed: Optional[int] = None
    include_path_dependency: bool = True
    shock_magnitude: float = 0.05
    correlation_shocks: bool = True

@dataclass 
class EnhancedBacktestConfig:
    """Comprehensive backtesting configuration"""
    
    # Sub-configurations
    capital_management: CapitalManagementConfig = field(default_factory=CapitalManagementConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    transactions: TransactionConfig = field(default_factory=TransactionConfig)
    timing: BacktestTimeConfig = field(default_factory=BacktestTimeConfig)
    targets: PerformanceTargetsConfig = field(default_factory=PerformanceTargetsConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    
    # Position sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_PARITY
    
    # Additional settings
    enable_detailed_logging: bool = True
    save_trade_details: bool = True
    calculate_attribution: bool = True
    generate_reports: bool = True
    
    # Model configuration
    model_update_frequency: int = 30  # Days
    feature_importance_tracking: bool = True
    model_drift_detection: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Capital management validations
        if self.capital_management.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.capital_management.minimum_position_size >= self.capital_management.maximum_position_size:
            errors.append("Minimum position size must be less than maximum position size")
        
        # Risk management validations
        if not (0 < self.risk_management.max_drawdown_limit < 1):
            errors.append("Max drawdown limit must be between 0 and 1")
        
        if not (0 < self.risk_management.var_confidence_level < 1):
            errors.append("VaR confidence level must be between 0 and 1")
        
        # Strategy validations
        if not (0 < self.strategy.min_prediction_confidence < 1):
            errors.append("Min prediction confidence must be between 0 and 1")
        
        if self.strategy.stop_loss_percentage >= self.strategy.profit_target_percentage:
            errors.append("Stop loss should be smaller than profit target")
        
        # Time validations
        if self.timing.start_date >= self.timing.end_date:
            errors.append("Start date must be before end date")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    
    def to_json(self, filepath: str = None) -> str:
        """Convert configuration to JSON"""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create configuration from dictionary"""
        # Handle nested configurations
        if 'capital_management' in config_dict:
            config_dict['capital_management'] = CapitalManagementConfig(**config_dict['capital_management'])
        if 'risk_management' in config_dict:
            config_dict['risk_management'] = RiskManagementConfig(**config_dict['risk_management'])
        if 'strategy' in config_dict:
            config_dict['strategy'] = StrategyConfig(**config_dict['strategy'])
        if 'transactions' in config_dict:
            config_dict['transactions'] = TransactionConfig(**config_dict['transactions'])
        if 'timing' in config_dict:
            timing_dict = config_dict['timing']
            if isinstance(timing_dict.get('start_date'), str):
                timing_dict['start_date'] = datetime.fromisoformat(timing_dict['start_date'])
            if isinstance(timing_dict.get('end_date'), str):
                timing_dict['end_date'] = datetime.fromisoformat(timing_dict['end_date'])
            if isinstance(timing_dict.get('rebalance_frequency'), str):
                timing_dict['rebalance_frequency'] = RebalanceFrequency(timing_dict['rebalance_frequency'])
            config_dict['timing'] = BacktestTimeConfig(**timing_dict)
        if 'targets' in config_dict:
            targets_dict = config_dict['targets']
            if isinstance(targets_dict.get('benchmark_type'), str):
                targets_dict['benchmark_type'] = BenchmarkType(targets_dict['benchmark_type'])
            config_dict['targets'] = PerformanceTargetsConfig(**targets_dict)
        if 'monte_carlo' in config_dict:
            config_dict['monte_carlo'] = MonteCarloConfig(**config_dict['monte_carlo'])
        
        # Handle enums
        if isinstance(config_dict.get('position_sizing_method'), str):
            config_dict['position_sizing_method'] = PositionSizingMethod(config_dict['position_sizing_method'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str = None, filepath: str = None):
        """Create configuration from JSON"""
        if filepath:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def get_total_transaction_cost(self) -> float:
        """Get total transaction cost percentage"""
        return self.transactions.total_transaction_cost()
    
    def get_effective_capital(self) -> float:
        """Get effective capital after reserves"""
        return self.capital_management.initial_capital * (1 - self.capital_management.reserve_cash_percentage)
    
    def is_valid_trade_date(self, date: datetime) -> bool:
        """Check if date is valid for trading"""
        if not self.timing.trading_hours_only:
            return True
        
        # Simple weekday check (can be enhanced with holiday calendar)
        return date.weekday() < 5  # Monday = 0, Sunday = 6

# ==================== PRESET CONFIGURATIONS ====================

class ConfigurationPresets:
    """Predefined configuration presets for different strategies"""
    
    @staticmethod
    def conservative_config() -> EnhancedBacktestConfig:
        """Conservative trading configuration"""
        config = EnhancedBacktestConfig()
        
        # Conservative capital management
        config.capital_management.reserve_cash_percentage = 0.15
        config.capital_management.leverage_limit = 0.8
        
        # Conservative risk management
        config.risk_management.max_drawdown_limit = 0.10
        config.risk_management.max_daily_loss_limit = 0.015
        config.risk_management.max_single_position = 0.15
        
        # Conservative strategy
        config.strategy.min_prediction_confidence = 0.7
        config.strategy.profit_target_percentage = 0.15
        config.strategy.stop_loss_percentage = 0.08
        config.strategy.min_holding_days = 5
        
        return config
    
    @staticmethod
    def aggressive_config() -> EnhancedBacktestConfig:
        """Aggressive trading configuration"""
        config = EnhancedBacktestConfig()
        
        # Aggressive capital management
        config.capital_management.reserve_cash_percentage = 0.02
        config.capital_management.leverage_limit = 1.5
        
        # Aggressive risk management
        config.risk_management.max_drawdown_limit = 0.25
        config.risk_management.max_daily_loss_limit = 0.03
        config.risk_management.max_single_position = 0.25
        
        # Aggressive strategy
        config.strategy.min_prediction_confidence = 0.55
        config.strategy.profit_target_percentage = 0.35
        config.strategy.stop_loss_percentage = 0.12
        config.strategy.min_holding_days = 1
        config.strategy.enable_short_selling = True
        
        return config
    
    @staticmethod
    def balanced_config() -> EnhancedBacktestConfig:
        """Balanced trading configuration (default)"""
        return EnhancedBacktestConfig()  # Default values are balanced
    
    @staticmethod
    def day_trading_config() -> EnhancedBacktestConfig:
        """Day trading configuration"""
        config = EnhancedBacktestConfig()
        
        # Day trading specific
        config.timing.rebalance_frequency = RebalanceFrequency.DAILY
        
        # Fast execution
        config.strategy.min_holding_days = 0
        config.strategy.max_holding_days = 1
        config.strategy.profit_target_percentage = 0.05
        config.strategy.stop_loss_percentage = 0.02
        config.strategy.enable_trailing_stop = True
        
        # Higher transaction costs for frequent trading
        config.transactions.brokerage_percentage = 0.001
        config.transactions.slippage_percentage = 0.001
        
        return config
    
    @staticmethod
    def swing_trading_config() -> EnhancedBacktestConfig:
        """Swing trading configuration"""
        config = EnhancedBacktestConfig()
        
        # Medium-term holding
        config.strategy.min_holding_days = 3
        config.strategy.max_holding_days = 21
        config.strategy.profit_target_percentage = 0.12
        config.strategy.stop_loss_percentage = 0.06
        
        # Weekly rebalancing
        config.timing.rebalance_frequency = RebalanceFrequency.WEEKLY
        
        return config
    
    @staticmethod
    def long_term_config() -> EnhancedBacktestConfig:
        """Long-term investment configuration"""
        config = EnhancedBacktestConfig()
        
        # Long-term holding
        config.strategy.min_holding_days = 30
        config.strategy.max_holding_days = 365
        config.strategy.profit_target_percentage = 0.50
        config.strategy.stop_loss_percentage = 0.20
        
        # Quarterly rebalancing
        config.timing.rebalance_frequency = RebalanceFrequency.QUARTERLY
        
        # Lower transaction costs due to less frequent trading
        config.transactions.brokerage_percentage = 0.0003
        config.transactions.slippage_percentage = 0.0002
        
        return config

# ==================== CONFIGURATION BUILDER ====================

class ConfigurationBuilder:
    """Builder pattern for creating custom configurations"""
    
    def __init__(self):
        self.config = EnhancedBacktestConfig()
    
    def with_capital(self, initial_capital: float, reserve_percentage: float = 0.05) -> 'ConfigurationBuilder':
        """Set capital management parameters"""
        self.config.capital_management.initial_capital = initial_capital
        self.config.capital_management.reserve_cash_percentage = reserve_percentage
        return self
    
    def with_risk_limits(self, max_drawdown: float, max_position: float = 0.2) -> 'ConfigurationBuilder':
        """Set risk management parameters"""
        self.config.risk_management.max_drawdown_limit = max_drawdown
        self.config.risk_management.max_single_position = max_position
        return self
    
    def with_strategy(self, min_confidence: float, profit_target: float, stop_loss: float) -> 'ConfigurationBuilder':
        """Set strategy parameters"""
        self.config.strategy.min_prediction_confidence = min_confidence
        self.config.strategy.profit_target_percentage = profit_target
        self.config.strategy.stop_loss_percentage = stop_loss
        return self
    
    def with_timeframe(self, start_date: datetime, end_date: datetime, 
                      rebalance_freq: RebalanceFrequency = RebalanceFrequency.MONTHLY) -> 'ConfigurationBuilder':
        """Set timeframe parameters"""
        self.config.timing.start_date = start_date
        self.config.timing.end_date = end_date
        self.config.timing.rebalance_frequency = rebalance_freq
        return self
    
    def with_position_sizing(self, method: PositionSizingMethod) -> 'ConfigurationBuilder':
        """Set position sizing method"""
        self.config.position_sizing_method = method
        return self
    
    def with_benchmark(self, benchmark_type: BenchmarkType) -> 'ConfigurationBuilder':
        """Set benchmark"""
        self.config.targets.benchmark_type = benchmark_type
        self.config.targets.benchmark_symbol = benchmark_type.value
        return self
    
    def build(self) -> EnhancedBacktestConfig:
        """Build the final configuration"""
        return self.config

# ==================== CONFIGURATION VALIDATOR ====================

class ConfigurationValidator:
    """Validate backtesting configurations"""
    
    @staticmethod
    def validate_comprehensive(config: EnhancedBacktestConfig) -> Dict[str, List[str]]:
        """Comprehensive configuration validation"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Capital validation
        if config.capital_management.initial_capital < 100000:
            issues['warnings'].append("Initial capital below ₹1,00,000 may limit diversification")
        
        if config.capital_management.reserve_cash_percentage > 0.2:
            issues['warnings'].append("High cash reserve (>20%) may reduce returns")
        
        # Risk validation
        if config.risk_management.max_drawdown_limit > 0.3:
            issues['errors'].append("Max drawdown limit >30% is extremely risky")
        
        if config.risk_management.max_single_position > 0.3:
            issues['warnings'].append("Single position >30% reduces diversification")
        
        # Strategy validation
        if config.strategy.profit_target_percentage / config.strategy.stop_loss_percentage < 1.5:
            issues['warnings'].append("Risk-reward ratio <1.5:1 may not be optimal")
        
        if config.strategy.min_prediction_confidence < 0.5:
            issues['warnings'].append("Low minimum confidence may include poor predictions")
        
        # Timeframe validation
        days_difference = (config.timing.end_date - config.timing.start_date).days
        if days_difference < 90:
            issues['warnings'].append("Short backtest period (<3 months) may not be reliable")
        
        if days_difference > 1825:  # 5 years
            issues['info'].append("Long backtest period (>5 years) - ensure data quality")
        
        # Transaction cost validation
        total_cost = config.get_total_transaction_cost()
        if total_cost > 0.005:  # >0.5%
            issues['warnings'].append(f"High transaction costs ({total_cost:.3%}) may impact returns")
        
        return issues
    
    @staticmethod
    def suggest_improvements(config: EnhancedBacktestConfig) -> List[str]:
        """Suggest configuration improvements"""
        suggestions = []
        
        # Risk-reward optimization
        if config.strategy.profit_target_percentage / config.strategy.stop_loss_percentage < 2:
            suggestions.append("Consider increasing profit target or decreasing stop loss for better risk-reward ratio")
        
        # Diversification suggestions
        if config.risk_management.max_single_position > 0.2:
            suggestions.append("Consider reducing maximum single position size for better diversification")
        
        # Capital efficiency
        if config.capital_management.reserve_cash_percentage > 0.1:
            suggestions.append("Consider reducing cash reserves to improve capital efficiency")
        
        # Monte Carlo settings
        if config.monte_carlo.num_simulations < 1000:
            suggestions.append("Increase Monte Carlo simulations to 1000+ for more reliable stress testing")
        
        return suggestions

# ==================== USAGE EXAMPLES ====================

def create_sample_configurations():
    """Create sample configurations for different use cases"""
    
    configs = {}
    
    # Conservative long-term investor
    configs['conservative'] = ConfigurationPresets.conservative_config()
    
    # Aggressive day trader
    configs['aggressive'] = ConfigurationPresets.aggressive_config()
    
    # Custom configuration using builder
    custom_config = (ConfigurationBuilder()
                    .with_capital(2000000, 0.08)
                    .with_risk_limits(0.12, 0.18)
                    .with_strategy(0.65, 0.22, 0.09)
                    .with_position_sizing(PositionSizingMethod.KELLY_CRITERION)
                    .with_benchmark(BenchmarkType.NIFTY_50)
                    .build())
    
    configs['custom'] = custom_config
    
    return configs

# ==================== EXPORT ====================

__all__ = [
    'EnhancedBacktestConfig',
    'CapitalManagementConfig',
    'RiskManagementConfig', 
    'StrategyConfig',
    'TransactionConfig',
    'BacktestTimeConfig',
    'PerformanceTargetsConfig',
    'MonteCarloConfig',
    'ConfigurationPresets',
    'ConfigurationBuilder',
    'ConfigurationValidator',
    'PositionSizingMethod',
    'RebalanceFrequency',
    'BenchmarkType'
]

if __name__ == "__main__":
    # Demo usage
    print("Creating sample configurations...")
    configs = create_sample_configurations()
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"Initial Capital: ₹{config.capital_management.initial_capital:,}")
        print(f"Max Drawdown: {config.risk_management.max_drawdown_limit:.1%}")
        print(f"Min Confidence: {config.strategy.min_prediction_confidence:.1%}")
        
        # Validate configuration
        issues = ConfigurationValidator.validate_comprehensive(config)
        if issues['errors']:
            print(f"❌ Errors: {len(issues['errors'])}")
        if issues['warnings']:
            print(f"⚠️ Warnings: {len(issues['warnings'])}")