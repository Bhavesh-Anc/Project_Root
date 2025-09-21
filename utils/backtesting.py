# utils/backtesting.py - Enhanced with Comprehensive Risk Management
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor
import pickle

# Import our new risk management system
from utils.risk_management import (
    ComprehensiveRiskManager, RiskConfig, CorrelationAnalyzer, 
    DrawdownTracker, PositionSizer, StressTester
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

@dataclass
class MLStrategy:
    """Machine Learning Strategy for backtesting"""
    
    def __init__(self, models: Dict, featured_data: Dict, horizon: str = 'next_month'):
        """
        Initialize ML Strategy
        
        Args:
            models: Dictionary of trained models {ticker: model_dict}
            featured_data: Dictionary of featured dataframes {ticker: df}
            horizon: Prediction horizon ('next_month', 'next_week', etc.)
        """
        self.models = models
        self.featured_data = featured_data
        self.horizon = horizon
        self.lookback_window = 30  # Days for signal generation
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_date: datetime) -> Dict[str, float]:
        """
        Generate trading signals for given date
        
        Args:
            data: Dictionary of price data {ticker: df}
            current_date: Current trading date
            
        Returns:
            Dictionary of signals {ticker: signal_strength} where signal_strength is between -1 and 1
        """
        signals = {}
        
        try:
            for ticker in self.models.keys():
                if ticker not in data:
                    continue
                    
                # Get data up to current date
                ticker_data = data[ticker][data[ticker].index <= current_date].copy()
                
                if len(ticker_data) < self.lookback_window:
                    continue
                
                # Get recent data for prediction
                recent_data = ticker_data.tail(1)
                
                if ticker in self.featured_data:
                    # Use featured data if available
                    featured_df = self.featured_data[ticker]
                    feature_data = featured_df[featured_df.index <= current_date].tail(1)
                    
                    if not feature_data.empty and ticker in self.models:
                        model_dict = self.models[ticker]
                        
                        if 'best_model' in model_dict:
                            model = model_dict['best_model']
                            
                            # Prepare features (exclude target columns)
                            feature_cols = [col for col in feature_data.columns 
                                          if not col.startswith('next_') and 
                                          col not in ['symbol', 'date']]
                            
                            if feature_cols:
                                X = feature_data[feature_cols].fillna(0)
                                
                                # Generate prediction
                                try:
                                    if hasattr(model, 'predict'):
                                        prediction = model.predict(X)[0]
                                        # Convert prediction to signal strength (-1 to 1)
                                        signal_strength = np.tanh(prediction / 100)  # Normalize
                                        signals[ticker] = float(signal_strength)
                                    else:
                                        signals[ticker] = 0.0
                                except Exception as e:
                                    logging.warning(f"Prediction failed for {ticker}: {e}")
                                    signals[ticker] = 0.0
                            else:
                                signals[ticker] = 0.0
                        else:
                            signals[ticker] = 0.0
                    else:
                        signals[ticker] = 0.0
                else:
                    # Simple momentum strategy as fallback
                    if len(ticker_data) >= 20:
                        short_ma = ticker_data['close'].tail(5).mean()
                        long_ma = ticker_data['close'].tail(20).mean()
                        signal_strength = (short_ma - long_ma) / long_ma
                        signals[ticker] = float(np.clip(signal_strength, -1, 1))
                    else:
                        signals[ticker] = 0.0
                        
        except Exception as e:
            logging.error(f"Signal generation failed: {e}")
            
        return signals
    
    def get_exit_signal(self, ticker: str, entry_date: datetime, current_date: datetime, 
                       entry_price: float, current_price: float, current_return: float) -> Tuple[bool, str]:
        """
        Determine if position should be exited
        
        Args:
            ticker: Stock ticker
            entry_date: Position entry date
            current_date: Current date
            entry_price: Entry price
            current_price: Current price
            current_return: Current return percentage
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Simple exit rules - can be enhanced
        holding_days = (current_date - entry_date).days
        
        # Exit conditions
        if current_return > 0.20:  # 20% profit target
            return True, "profit_target"
        elif current_return < -0.10:  # 10% stop loss
            return True, "stop_loss"
        elif holding_days > 60:  # Maximum holding period
            return True, "max_holding_period"
        
        return False, "hold"
    
    def get_strategy_name(self) -> str:
        """Return strategy name"""
        return f"ML_Strategy_{self.horizon}"
    
    def get_required_data_columns(self) -> List[str]:
        """Return list of required data columns"""
        return ['open', 'high', 'low', 'close', 'volume']

class EnhancedBacktestConfig:
    """Enhanced configuration with comprehensive risk management"""
    # Original backtesting parameters
    initial_capital: float = 1000000  # 10 lakh
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    min_position_size: float = 10000  # Minimum 10k per position
    max_position_size: float = 200000  # Maximum 2 lakh per position
    max_positions: int = 20  # Maximum concurrent positions
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    benchmark: str = 'NIFTY50'
    risk_free_rate: float = 0.06  # 6% annual
    
    # Enhanced risk management parameters
    max_portfolio_drawdown: float = 0.15  # Stop trading if drawdown > 15%
    max_position_correlation: float = 0.7  # Maximum correlation between positions
    var_confidence_level: float = 0.95  # VaR confidence level
    stress_test_frequency: int = 5  # Run stress tests every N days
    risk_budget_limit: float = 0.20  # Maximum risk budget utilization
    kelly_fraction_cap: float = 0.25  # Cap Kelly criterion
    
    # Position sizing method
    position_sizing_method: str = 'risk_parity'  # 'equal_weight', 'risk_parity', 'kelly', 'erc'
    
    # Risk monitoring
    risk_monitoring_enabled: bool = True
    correlation_monitoring: bool = True
    drawdown_monitoring: bool = True
    stress_testing_enabled: bool = True
    
    # Advanced settings
    lookback_window: int = 252  # Days for rolling calculations
    confidence_level: float = 0.95  # For VaR calculations
    rebalance_threshold: float = 0.05  # Drift threshold for rebalancing

@dataclass
class EnhancedTrade:
    """Enhanced trade record with risk metrics"""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    direction: str  # 'long' or 'short'
    entry_signal: str
    exit_signal: str
    gross_pnl: float
    transaction_costs: float
    net_pnl: float
    return_pct: float
    holding_period: int
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    # Enhanced risk metrics
    entry_var: float = 0.0  # VaR at entry
    max_drawdown_during_hold: float = 0.0
    correlation_with_portfolio: float = 0.0
    kelly_fraction_used: float = 0.0
    risk_contribution: float = 0.0

@dataclass
class EnhancedPortfolioState:
    """Enhanced portfolio state with risk tracking"""
    cash: float
    positions: Dict[str, Dict] = field(default_factory=dict)
    portfolio_value: float = 0.0
    leverage: float = 1.0
    drawdown: float = 0.0
    peak_value: float = 0.0
    
    # Enhanced risk metrics
    current_var: float = 0.0
    portfolio_correlation: float = 0.0
    risk_budget_used: float = 0.0
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    position_correlations: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, float] = field(default_factory=dict)

class EnhancedRiskManager:
    """Enhanced risk manager integrating comprehensive risk management"""
    
    def __init__(self, config: EnhancedBacktestConfig):
        self.config = config
        
        # Initialize comprehensive risk management
        risk_config = RiskConfig(
            max_portfolio_drawdown=config.max_portfolio_drawdown,
            max_position_size=config.max_position_size / config.initial_capital,  # Convert to fraction
            min_position_size=config.min_position_size / config.initial_capital,
            max_correlation_threshold=config.max_position_correlation,
            var_confidence_level=config.var_confidence_level,
            kelly_fraction_cap=config.kelly_fraction_cap,
            risk_free_rate=config.risk_free_rate
        )
        
        self.comprehensive_manager = ComprehensiveRiskManager(risk_config)
        self.correlation_analyzer = CorrelationAnalyzer(risk_config)
        self.drawdown_tracker = DrawdownTracker(risk_config)
        self.position_sizer = PositionSizer(risk_config)
        self.stress_tester = StressTester(risk_config)
        
        # Risk monitoring history
        self.risk_history = []
        self.correlation_history = []
        self.drawdown_history = []
        
    def calculate_position_size(self, signal_strength: float, portfolio_state: EnhancedPortfolioState, 
                              stock_price: float, returns_data: pd.DataFrame = None,
                              ticker: str = None) -> int:
        """Enhanced position sizing with multiple methods"""
        
        portfolio_value = portfolio_state.portfolio_value
        
        if self.config.position_sizing_method == 'equal_weight':
            target_value = portfolio_value * 0.05  # 5% per position
            return int(target_value / stock_price)
            
        elif self.config.position_sizing_method == 'kelly' and returns_data is not None:
            # Enhanced Kelly criterion with historical data
            if ticker in returns_data.columns:
                ticker_returns = returns_data[ticker].dropna()
                if len(ticker_returns) > 30:
                    # Calculate win probability and average win/loss
                    positive_returns = ticker_returns[ticker_returns > 0]
                    negative_returns = ticker_returns[ticker_returns < 0]
                    
                    if len(positive_returns) > 0 and len(negative_returns) > 0:
                        win_prob = len(positive_returns) / len(ticker_returns)
                        avg_win = positive_returns.mean()
                        avg_loss = abs(negative_returns.mean())
                        
                        kelly_size = self.position_sizer.kelly_criterion_sizing(
                            win_prob, avg_win, avg_loss, portfolio_value
                        )
                        return int(kelly_size / stock_price)
            
            # Fallback to signal-based Kelly
            win_prob = (signal_strength + 1) / 2  # Convert to probability
            kelly_size = self.position_sizer.kelly_criterion_sizing(
                win_prob, 0.05, 0.03, portfolio_value
            )
            return int(kelly_size / stock_price)
            
        elif self.config.position_sizing_method == 'risk_parity' and returns_data is not None:
            # Risk parity position sizing
            current_tickers = list(portfolio_state.positions.keys()) + [ticker]
            if len(current_tickers) > 1:
                available_tickers = [t for t in current_tickers if t in returns_data.columns]
                if len(available_tickers) > 1:
                    risk_parity_sizes = self.position_sizer.risk_parity_sizing(
                        returns_data[available_tickers], portfolio_value
                    )
                    if ticker in risk_parity_sizes:
                        return int(risk_parity_sizes[ticker] / stock_price)
            
            # Fallback to volatility-adjusted
            if ticker in returns_data.columns:
                vol = returns_data[ticker].std()
                vol_adjusted_size = portfolio_value * 0.05 / (1 + vol)
                return int(vol_adjusted_size / stock_price)
        
        elif self.config.position_sizing_method == 'erc' and returns_data is not None:
            # Equal Risk Contribution
            current_tickers = list(portfolio_state.positions.keys()) + [ticker]
            available_tickers = [t for t in current_tickers if t in returns_data.columns]
            if len(available_tickers) > 1:
                erc_sizes = self.position_sizer.equal_risk_contribution_sizing(
                    returns_data[available_tickers], portfolio_value
                )
                if ticker in erc_sizes:
                    return int(erc_sizes[ticker] / stock_price)
        
        # Default fallback
        target_value = portfolio_value * 0.05
        return int(target_value / stock_price)
    
    def check_risk_constraints(self, portfolio_state: EnhancedPortfolioState, 
                             returns_data: pd.DataFrame = None,
                             new_position: Dict = None) -> Dict[str, bool]:
        """Comprehensive risk constraint checking"""
        
        checks = {
            'drawdown_ok': True,
            'correlation_ok': True,
            'position_size_ok': True,
            'var_ok': True,
            'concentration_ok': True,
            'leverage_ok': True
        }
        
        # 1. Drawdown check
        drawdown_result = self.drawdown_tracker.check_drawdown_limits(portfolio_state.drawdown)
        checks['drawdown_ok'] = drawdown_result['within_limits']
        
        # 2. Position size check
        if new_position:
            position_value = new_position.get('value', 0)
            position_weight = position_value / portfolio_state.portfolio_value
            checks['position_size_ok'] = position_weight <= self.config.max_position_size / portfolio_state.portfolio_value
        
        # 3. Correlation check
        if returns_data is not None and len(portfolio_state.positions) > 0:
            portfolio_tickers = list(portfolio_state.positions.keys())
            if new_position and 'ticker' in new_position:
                portfolio_tickers.append(new_position['ticker'])
            
            available_tickers = [t for t in portfolio_tickers if t in returns_data.columns]
            if len(available_tickers) > 1:
                corr_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                    returns_data[available_tickers]
                )
                high_correlations = self.correlation_analyzer.find_high_correlations(
                    corr_matrix, self.config.max_position_correlation
                )
                checks['correlation_ok'] = len(high_correlations) == 0
        
        # 4. Leverage check
        checks['leverage_ok'] = portfolio_state.leverage <= 1.0
        
        # 5. Maximum positions check
        max_positions_ok = len(portfolio_state.positions) < self.config.max_positions
        checks['concentration_ok'] = max_positions_ok
        
        return checks
    
    def run_portfolio_stress_test(self, portfolio_state: EnhancedPortfolioState,
                                 returns_data: pd.DataFrame) -> Dict[str, float]:
        """Run stress test on current portfolio"""
        
        if len(portfolio_state.positions) == 0:
            return {'stress_test_passed': True}
        
        # Calculate portfolio weights
        portfolio_weights = {}
        total_value = portfolio_state.portfolio_value
        
        for ticker, position in portfolio_state.positions.items():
            position_value = position.get('quantity', 0) * position.get('current_price', position.get('entry_price', 0))
            portfolio_weights[ticker] = position_value / total_value
        
        # Run stress tests
        try:
            stress_results = self.stress_tester.run_historical_stress_tests(
                portfolio_weights, returns_data
            )
            
            monte_carlo_results = self.stress_tester.monte_carlo_stress_test(
                portfolio_weights, returns_data, n_simulations=1000
            )
            
            # Check if stress test results are acceptable
            stress_test_passed = True
            if 'monte_carlo' in monte_carlo_results:
                var_95 = monte_carlo_results.get('var_95', 0)
                if var_95 < -0.15:  # 15% daily VaR threshold
                    stress_test_passed = False
            
            return {
                'stress_test_passed': stress_test_passed,
                'var_95': monte_carlo_results.get('var_95', 0),
                'expected_shortfall': monte_carlo_results.get('expected_shortfall_95', 0),
                'worst_case': monte_carlo_results.get('worst_case_return', 0),
                'historical_stress': stress_results
            }
            
        except Exception as e:
            logging.warning(f"Stress test failed: {e}")
            return {'stress_test_passed': True, 'error': str(e)}
    
    def update_risk_metrics(self, portfolio_state: EnhancedPortfolioState,
                           returns_data: pd.DataFrame) -> EnhancedPortfolioState:
        """Update portfolio risk metrics"""
        
        if len(portfolio_state.positions) == 0:
            return portfolio_state
        
        try:
            # Calculate portfolio weights for risk analysis
            portfolio_data = {}
            for ticker, position in portfolio_state.positions.items():
                position_value = position.get('quantity', 0) * position.get('current_price', position.get('entry_price', 0))
                portfolio_data[ticker] = {
                    'weight': position_value / portfolio_state.portfolio_value,
                    'value': position_value
                }
            
            # Run comprehensive risk assessment
            risk_assessment = self.comprehensive_manager.comprehensive_risk_assessment(
                portfolio_data, returns_data
            )
            
            # Update portfolio state with risk metrics
            if 'stress_testing' in risk_assessment:
                monte_carlo = risk_assessment['stress_testing'].get('monte_carlo', {})
                portfolio_state.current_var = monte_carlo.get('var_95', 0)
                portfolio_state.stress_test_results = monte_carlo
            
            if 'correlation_analysis' in risk_assessment:
                portfolio_state.portfolio_correlation = risk_assessment['correlation_analysis'].get('max_correlation', 0)
            
            # Log risk metrics
            self.risk_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_state.portfolio_value,
                'var_95': portfolio_state.current_var,
                'max_correlation': portfolio_state.portfolio_correlation,
                'drawdown': portfolio_state.drawdown
            })
            
        except Exception as e:
            logging.warning(f"Risk metrics update failed: {e}")
        
        return portfolio_state

class EnhancedBacktestEngine:
    """Enhanced backtesting engine with comprehensive risk management"""
    
    def __init__(self, config: EnhancedBacktestConfig):
        self.config = config
        self.risk_manager = EnhancedRiskManager(config)
        self.portfolio_state = EnhancedPortfolioState(cash=config.initial_capital)
        self.trades: List[EnhancedTrade] = []
        self.portfolio_history = []
        self.risk_events = []
        self.benchmark_data = None
        
    def run_enhanced_backtest(self, strategy, data: Dict[str, pd.DataFrame], 
                            start_date: datetime, end_date: datetime) -> Dict:
        """Run enhanced backtest with comprehensive risk management"""
        
        logging.info(f"Starting enhanced backtest from {start_date} to {end_date}")
        logging.info(f"Risk management features: {self.config.risk_monitoring_enabled}")
        
        # Initialize enhanced portfolio state
        self.portfolio_state = EnhancedPortfolioState(cash=self.config.initial_capital)
        self.portfolio_state.peak_value = self.config.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.risk_events = []
        
        # Prepare returns data for risk analysis
        returns_data = self._prepare_returns_data(data)
        
        # Get all available dates
        all_dates = set()
        for ticker_data in data.values():
            all_dates.update(ticker_data.index)
        
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        # Enhanced rebalancing logic
        last_rebalance = None
        last_stress_test = None
        rebalance_freq_map = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90
        }
        rebalance_days = rebalance_freq_map.get(self.config.rebalance_frequency, 30)
        
        for current_date in trading_dates:
            try:
                # Get current prices
                current_prices = {}
                for ticker, ticker_data in data.items():
                    if current_date in ticker_data.index:
                        current_prices[ticker] = ticker_data.loc[current_date, 'Close']
                
                if not current_prices:
                    continue
                
                # Update portfolio value with current prices
                self._update_enhanced_portfolio_value(current_prices)
                
                # Enhanced risk monitoring
                if self.config.risk_monitoring_enabled:
                    # Update risk metrics
                    self.portfolio_state = self.risk_manager.update_risk_metrics(
                        self.portfolio_state, returns_data
                    )
                    
                    # Check risk constraints
                    risk_checks = self.risk_manager.check_risk_constraints(
                        self.portfolio_state, returns_data
                    )
                    
                    # Handle risk violations
                    if not all(risk_checks.values()):
                        self._handle_risk_violations(risk_checks, current_prices, current_date)
                
                # Run periodic stress tests
                if (self.config.stress_testing_enabled and 
                    (last_stress_test is None or 
                     (current_date - last_stress_test).days >= self.config.stress_test_frequency)):
                    
                    stress_results = self.risk_manager.run_portfolio_stress_test(
                        self.portfolio_state, returns_data
                    )
                    
                    if not stress_results.get('stress_test_passed', True):
                        self._handle_stress_test_failure(stress_results, current_prices, current_date)
                    
                    last_stress_test = current_date
                
                # Enhanced exit signal checking
                self._check_enhanced_exit_signals(strategy, current_prices, current_date, returns_data)
                
                # Enhanced rebalancing with risk considerations
                if self._should_rebalance(current_date, last_rebalance, rebalance_days):
                    
                    # Generate signals
                    signals = strategy.generate_signals(data, current_date)
                    
                    # Enhanced signal filtering with risk considerations
                    filtered_signals = self._filter_signals_with_risk(
                        signals, current_prices, returns_data
                    )
                    
                    # Execute new positions with enhanced risk management
                    self._execute_enhanced_positions(
                        filtered_signals, current_prices, current_date, returns_data
                    )
                    
                    last_rebalance = current_date
                
                # Record enhanced portfolio state
                self._record_enhanced_portfolio_state(current_date)
                
            except Exception as e:
                logging.error(f"Error processing date {current_date}: {e}")
                continue
        
        # Close all remaining positions
        if self.portfolio_state.positions:
            final_prices = {ticker: data[ticker].iloc[-1]['Close'] 
                          for ticker in self.portfolio_state.positions.keys() 
                          if ticker in data}
            self._liquidate_all_positions(final_prices, end_date, "End of enhanced backtest")
        
        return self._generate_enhanced_results()
    
    def _prepare_returns_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare returns data for risk analysis"""
        
        returns_dict = {}
        
        for ticker, df in data.items():
            if 'Close' in df.columns and len(df) > 1:
                returns = df['Close'].pct_change().dropna()
                returns_dict[ticker] = returns
        
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            return returns_df
        
        return pd.DataFrame()
    
    def _update_enhanced_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value with enhanced tracking"""
        
        position_value = 0.0
        
        for ticker, position in self.portfolio_state.positions.items():
            if ticker in current_prices:
                current_price = current_prices[ticker]
                position['current_price'] = current_price
                pos_value = position['quantity'] * current_price
                position_value += pos_value
                
                # Update position-level metrics
                entry_price = position.get('entry_price', current_price)
                position['unrealized_pnl'] = (current_price - entry_price) * position['quantity']
                position['return_pct'] = (current_price / entry_price - 1) if entry_price > 0 else 0
        
        self.portfolio_state.portfolio_value = self.portfolio_state.cash + position_value
        
        # Update drawdown tracking
        if self.portfolio_state.portfolio_value > self.portfolio_state.peak_value:
            self.portfolio_state.peak_value = self.portfolio_state.portfolio_value
            self.portfolio_state.drawdown = 0.0
        else:
            self.portfolio_state.drawdown = (
                (self.portfolio_state.portfolio_value - self.portfolio_state.peak_value) / 
                self.portfolio_state.peak_value
            )
    
    def _handle_risk_violations(self, risk_checks: Dict[str, bool], 
                               current_prices: Dict[str, float], current_date: datetime):
        """Handle risk constraint violations"""
        
        risk_event = {
            'date': current_date,
            'violations': [k for k, v in risk_checks.items() if not v],
            'actions_taken': []
        }
        
        # Handle drawdown violation
        if not risk_checks['drawdown_ok']:
            # Reduce position sizes by 25%
            positions_to_reduce = list(self.portfolio_state.positions.keys())[:3]  # Reduce largest positions
            for ticker in positions_to_reduce:
                if ticker in current_prices:
                    position = self.portfolio_state.positions[ticker]
                    reduce_quantity = int(position['quantity'] * 0.25)
                    if reduce_quantity > 0:
                        self._execute_enhanced_trade(
                            ticker, reduce_quantity, current_prices[ticker], 
                            'sell', current_date, "Drawdown risk reduction"
                        )
                        risk_event['actions_taken'].append(f"Reduced {ticker} by 25%")
        
        # Handle correlation violation
        if not risk_checks['correlation_ok']:
            # Could implement correlation-based position reduction
            risk_event['actions_taken'].append("Correlation violation detected")
        
        self.risk_events.append(risk_event)
        logging.warning(f"Risk violations on {current_date}: {risk_event['violations']}")
    
    def _handle_stress_test_failure(self, stress_results: Dict, 
                                   current_prices: Dict[str, float], current_date: datetime):
        """Handle stress test failures"""
        
        var_95 = stress_results.get('var_95', 0)
        
        risk_event = {
            'date': current_date,
            'type': 'stress_test_failure',
            'var_95': var_95,
            'actions_taken': []
        }
        
        if var_95 < -0.20:  # Critical threshold
            # Emergency position reduction
            positions_to_reduce = list(self.portfolio_state.positions.keys())
            for ticker in positions_to_reduce:
                if ticker in current_prices:
                    position = self.portfolio_state.positions[ticker]
                    reduce_quantity = int(position['quantity'] * 0.4)  # 40% reduction
                    if reduce_quantity > 0:
                        self._execute_enhanced_trade(
                            ticker, reduce_quantity, current_prices[ticker], 
                            'sell', current_date, "Emergency stress test response"
                        )
                        risk_event['actions_taken'].append(f"Emergency reduction {ticker} by 40%")
        
        elif var_95 < -0.15:  # High risk threshold
            # Moderate position reduction
            largest_positions = sorted(
                self.portfolio_state.positions.items(),
                key=lambda x: x[1]['quantity'] * current_prices.get(x[0], 0),
                reverse=True
            )[:2]
            
            for ticker, position in largest_positions:
                if ticker in current_prices:
                    reduce_quantity = int(position['quantity'] * 0.25)  # 25% reduction
                    if reduce_quantity > 0:
                        self._execute_enhanced_trade(
                            ticker, reduce_quantity, current_prices[ticker], 
                            'sell', current_date, "Stress test risk reduction"
                        )
                        risk_event['actions_taken'].append(f"Reduced {ticker} by 25%")
        
        self.risk_events.append(risk_event)
        logging.warning(f"Stress test failure on {current_date}: VaR 95% = {var_95:.2%}")
    
    def _check_enhanced_exit_signals(self, strategy, current_prices: Dict[str, float], 
                                   current_date: datetime, returns_data: pd.DataFrame):
        """Enhanced exit signal checking with risk considerations"""
        
        positions_to_exit = []
        
        for ticker in list(self.portfolio_state.positions.keys()):
            if ticker in current_prices:
                position = self.portfolio_state.positions[ticker]
                
                # Original strategy exit signal
                should_exit = strategy.should_exit(
                    ticker, position['entry_date'], current_date,
                    current_prices[ticker], position['entry_price']
                )
                
                # Enhanced risk-based exit signals
                risk_exit = self._check_risk_based_exit(ticker, position, current_prices[ticker], returns_data)
                
                if should_exit or risk_exit['should_exit']:
                    exit_reason = risk_exit['reason'] if risk_exit['should_exit'] else "Strategy signal"
                    positions_to_exit.append((ticker, exit_reason))
        
        # Execute exits
        for ticker, exit_reason in positions_to_exit:
            position = self.portfolio_state.positions[ticker]
            self._execute_enhanced_trade(
                ticker, position['quantity'], current_prices[ticker],
                'sell', current_date, exit_reason
            )
    
    def _check_risk_based_exit(self, ticker: str, position: Dict, 
                              current_price: float, returns_data: pd.DataFrame) -> Dict:
        """Check for risk-based exit signals"""
        
        entry_price = position['entry_price']
        return_pct = (current_price / entry_price - 1) if entry_price > 0 else 0
        
        # Stop loss based on portfolio drawdown
        if self.portfolio_state.drawdown < -0.10:  # If portfolio down 10%
            if return_pct < -0.05:  # And this position down 5%
                return {'should_exit': True, 'reason': 'Portfolio drawdown stop loss'}
        
        # Correlation-based exit
        if ticker in returns_data.columns and len(self.portfolio_state.positions) > 1:
            try:
                other_tickers = [t for t in self.portfolio_state.positions.keys() if t != ticker]
                available_others = [t for t in other_tickers if t in returns_data.columns]
                
                if available_others:
                    ticker_returns = returns_data[ticker].tail(30)  # Last 30 days
                    for other_ticker in available_others:
                        other_returns = returns_data[other_ticker].tail(30)
                        if len(ticker_returns) > 10 and len(other_returns) > 10:
                            correlation = ticker_returns.corr(other_returns)
                            if abs(correlation) > self.config.max_position_correlation:
                                return {'should_exit': True, 'reason': f'High correlation with {other_ticker}'}
            except:
                pass
        
        # Volatility-based exit
        if ticker in returns_data.columns:
            try:
                recent_vol = returns_data[ticker].tail(20).std()
                long_vol = returns_data[ticker].tail(60).std()
                
                if recent_vol > long_vol * 2:  # Recent volatility 2x normal
                    return {'should_exit': True, 'reason': 'Excessive volatility'}
            except:
                pass
        
        return {'should_exit': False, 'reason': 'No risk exit signal'}
    
    def _filter_signals_with_risk(self, signals: Dict[str, float], 
                                 current_prices: Dict[str, float],
                                 returns_data: pd.DataFrame) -> Dict[str, float]:
        """Filter signals based on risk considerations"""
        
        filtered_signals = {}
        
        for ticker, signal_strength in signals.items():
            if ticker not in current_prices:
                continue
            
            # Basic signal strength filter
            if signal_strength < 0.6:
                continue
            
            # Check correlation with existing positions
            if len(self.portfolio_state.positions) > 0 and ticker in returns_data.columns:
                max_correlation = 0
                for existing_ticker in self.portfolio_state.positions.keys():
                    if existing_ticker in returns_data.columns:
                        try:
                            corr = returns_data[ticker].tail(60).corr(returns_data[existing_ticker].tail(60))
                            max_correlation = max(max_correlation, abs(corr))
                        except:
                            continue
                
                if max_correlation > self.config.max_position_correlation:
                    continue
            
            # Check if we're at position limit
            if len(self.portfolio_state.positions) >= self.config.max_positions:
                continue
            
            # Check portfolio risk budget
            if self.portfolio_state.risk_budget_used > self.config.risk_budget_limit:
                continue
            
            filtered_signals[ticker] = signal_strength
        
        return filtered_signals
    
    def _execute_enhanced_positions(self, signals: Dict[str, float], 
                                   current_prices: Dict[str, float],
                                   current_date: datetime, returns_data: pd.DataFrame):
        """Execute new positions with enhanced risk management"""
        
        # Sort signals by strength
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        
        for ticker, signal_strength in sorted_signals:
            if ticker not in self.portfolio_state.positions and ticker in current_prices:
                
                stock_price = current_prices[ticker]
                
                # Enhanced position sizing
                quantity = self.risk_manager.calculate_position_size(
                    signal_strength, self.portfolio_state, stock_price, 
                    returns_data, ticker
                )
                
                # Apply risk limits
                quantity = self.risk_manager.position_sizer._apply_position_constraints(
                    np.array([quantity / self.portfolio_state.portfolio_value])
                )[0] * self.portfolio_state.portfolio_value / stock_price
                quantity = int(quantity)
                
                if quantity > 0:
                    # Check if adding this position violates risk constraints
                    mock_position = {
                        'ticker': ticker,
                        'value': quantity * stock_price
                    }
                    
                    risk_checks = self.risk_manager.check_risk_constraints(
                        self.portfolio_state, returns_data, mock_position
                    )
                    
                    if all(risk_checks.values()):
                        success = self._execute_enhanced_trade(
                            ticker, quantity, stock_price, 'buy', 
                            current_date, f"Enhanced ML Signal: {signal_strength:.3f}"
                        )
                        
                        if success:
                            logging.info(f"Opened enhanced position: {ticker} x {quantity} @ {stock_price}")
                    else:
                        logging.info(f"Skipped {ticker} due to risk constraints: {risk_checks}")
    
    def _execute_enhanced_trade(self, ticker: str, quantity: int, price: float, 
                               direction: str, date: datetime, signal: str) -> bool:
        """Execute trade with enhanced tracking"""
        
        # Calculate slippage and costs (same as original)
        slippage = self._calculate_slippage(price, quantity, direction)
        execution_price = price + slippage
        trade_value = quantity * execution_price
        transaction_cost = trade_value * self.config.transaction_cost_pct
        total_cost = trade_value + transaction_cost
        
        if direction == 'buy':
            if total_cost > self.portfolio_state.cash:
                return False
            
            # Execute buy order
            self.portfolio_state.cash -= total_cost
            self.portfolio_state.positions[ticker] = {
                'quantity': quantity,
                'entry_price': execution_price,
                'current_price': execution_price,
                'entry_date': date,
                'entry_signal': signal,
                'transaction_cost': transaction_cost,
                'unrealized_pnl': 0.0,
                'return_pct': 0.0
            }
            
        elif direction == 'sell':
            if ticker not in self.portfolio_state.positions:
                return False
            
            position = self.portfolio_state.positions[ticker]
            
            # Create enhanced trade record
            holding_period = (date - position['entry_date']).days
            gross_pnl = quantity * (execution_price - position['entry_price'])
            total_transaction_cost = position['transaction_cost'] + transaction_cost
            net_pnl = gross_pnl - total_transaction_cost
            return_pct = net_pnl / (quantity * position['entry_price'])
            
            # Calculate enhanced metrics
            entry_var = 0.0  # Could be calculated from historical data
            max_dd_during_hold = 0.0  # Could track this during holding period
            
            enhanced_trade = EnhancedTrade(
                ticker=ticker,
                entry_date=position['entry_date'],
                exit_date=date,
                entry_price=position['entry_price'],
                exit_price=execution_price,
                quantity=quantity,
                direction='long',
                entry_signal=position['entry_signal'],
                exit_signal=signal,
                gross_pnl=gross_pnl,
                transaction_costs=total_transaction_cost,
                net_pnl=net_pnl,
                return_pct=return_pct,
                holding_period=holding_period,
                entry_var=entry_var,
                max_drawdown_during_hold=max_dd_during_hold
            )
            
            self.trades.append(enhanced_trade)
            
            # Add cash back
            self.portfolio_state.cash += trade_value - transaction_cost
            
            # Remove or reduce position
            if quantity >= position['quantity']:
                del self.portfolio_state.positions[ticker]
            else:
                position['quantity'] -= quantity
        
        return True
    
    def _calculate_slippage(self, price: float, quantity: int, direction: str) -> float:
        """Calculate realistic slippage (same as original)"""
        base_slippage = price * self.config.slippage_pct
        size_factor = min(2.0, quantity / 1000)
        slippage_direction = 1 if direction == 'buy' else -1
        return base_slippage * size_factor * slippage_direction
    
    def _should_rebalance(self, current_date: datetime, last_rebalance: Optional[datetime], 
                         rebalance_days: int) -> bool:
        """Check if rebalancing is needed"""
        if last_rebalance is None:
            return True
        return (current_date - last_rebalance).days >= rebalance_days
    
    def _record_enhanced_portfolio_state(self, current_date: datetime):
        """Record enhanced portfolio state"""
        
        state_record = {
            'date': current_date,
            'portfolio_value': self.portfolio_state.portfolio_value,
            'cash': self.portfolio_state.cash,
            'positions': len(self.portfolio_state.positions),
            'drawdown': self.portfolio_state.drawdown,
            'current_var': self.portfolio_state.current_var,
            'portfolio_correlation': self.portfolio_state.portfolio_correlation,
            'risk_budget_used': self.portfolio_state.risk_budget_used
        }
        
        self.portfolio_history.append(state_record)
    
    def _liquidate_all_positions(self, prices: Dict[str, float], date: datetime, reason: str):
        """Liquidate all positions (same as original)"""
        for ticker in list(self.portfolio_state.positions.keys()):
            if ticker in prices:
                position = self.portfolio_state.positions[ticker]
                self._execute_enhanced_trade(
                    ticker, position['quantity'], prices[ticker],
                    'sell', date, reason
                )
    
    def _generate_enhanced_results(self) -> Dict:
        """Generate enhanced backtest results"""
        
        if not self.portfolio_history:
            return {'error': 'No portfolio history generated'}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Enhanced performance metrics
        enhanced_metrics = self._calculate_enhanced_metrics(portfolio_df, returns)
        
        # Risk analysis results
        risk_analysis = self._analyze_risk_performance()
        
        return {
            'enhanced_metrics': enhanced_metrics,
            'risk_analysis': risk_analysis,
            'portfolio_history': portfolio_df,
            'enhanced_trades': self.trades,
            'risk_events': self.risk_events,
            'returns': returns,
            'config': self.config,
            'risk_manager_history': self.risk_manager.risk_history
        }
    
    def _calculate_enhanced_metrics(self, portfolio_df: pd.DataFrame, returns: pd.Series) -> Dict:
        """Calculate enhanced performance metrics"""
        
        from utils.backtesting import PerformanceMetrics  # Import original metrics
        
        # Original metrics
        original_metrics = {
            'total_return': (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1,
            'annual_return': (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) ** (252 / len(portfolio_df)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, self.config.risk_free_rate),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, self.config.risk_free_rate),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, portfolio_df['portfolio_value']),
            'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_df['portfolio_value']),
            'var_95': PerformanceMetrics.value_at_risk(returns, 0.95),
            'cvar_95': PerformanceMetrics.conditional_var(returns, 0.95)
        }
        
        # Enhanced risk-adjusted metrics
        enhanced_metrics = {}
        
        # Risk-adjusted return metrics
        if 'current_var' in portfolio_df.columns:
            avg_var = portfolio_df['current_var'].mean()
            enhanced_metrics['var_adjusted_return'] = original_metrics['annual_return'] / abs(avg_var) if avg_var < 0 else 0
        
        # Correlation-adjusted metrics
        if 'portfolio_correlation' in portfolio_df.columns:
            avg_correlation = portfolio_df['portfolio_correlation'].mean()
            enhanced_metrics['correlation_penalty'] = avg_correlation
            enhanced_metrics['diversification_ratio'] = 1 - avg_correlation
        
        # Drawdown frequency and duration
        drawdown_series = portfolio_df['drawdown']
        drawdown_periods = (drawdown_series < -0.05).astype(int)  # Periods with >5% drawdown
        enhanced_metrics['drawdown_frequency'] = drawdown_periods.sum() / len(drawdown_periods)
        
        # Maximum adverse excursion (MAE) for trades
        if self.trades:
            mae_values = []
            for trade in self.trades:
                if hasattr(trade, 'max_drawdown_during_hold'):
                    mae_values.append(trade.max_drawdown_during_hold)
            if mae_values:
                enhanced_metrics['avg_mae'] = np.mean(mae_values)
                enhanced_metrics['max_mae'] = np.max(mae_values)
        
        # Risk event analysis
        enhanced_metrics['risk_events_count'] = len(self.risk_events)
        enhanced_metrics['risk_events_per_year'] = len(self.risk_events) / (len(portfolio_df) / 252)
        
        # Combine original and enhanced metrics
        all_metrics = {**original_metrics, **enhanced_metrics}
        
        return all_metrics
    
    def _analyze_risk_performance(self) -> Dict:
        """Analyze risk management performance"""
        
        analysis = {
            'drawdown_violations': 0,
            'correlation_violations': 0,
            'stress_test_failures': 0,
            'risk_adjusted_trades': 0,
            'avg_position_correlation': 0.0,
            'max_var_reached': 0.0
        }
        
        # Analyze risk events
        for event in self.risk_events:
            if 'drawdown' in event.get('violations', []):
                analysis['drawdown_violations'] += 1
            if 'correlation' in event.get('violations', []):
                analysis['correlation_violations'] += 1
            if event.get('type') == 'stress_test_failure':
                analysis['stress_test_failures'] += 1
        
        # Analyze trades with risk adjustments
        for trade in self.trades:
            if 'risk' in trade.exit_signal.lower():
                analysis['risk_adjusted_trades'] += 1
        
        # Portfolio correlation analysis
        if self.risk_manager.risk_history:
            correlations = [r['max_correlation'] for r in self.risk_manager.risk_history if 'max_correlation' in r]
            if correlations:
                analysis['avg_position_correlation'] = np.mean(correlations)
            
            vars = [r['var_95'] for r in self.risk_manager.risk_history if 'var_95' in r]
            if vars:
                analysis['max_var_reached'] = min(vars)  # Most negative VaR
        
        return analysis

# Keep the original classes for backward compatibility
BacktestConfig = EnhancedBacktestConfig
Trade = EnhancedTrade
PortfolioState = EnhancedPortfolioState
BacktestEngine = EnhancedBacktestEngine

# Export the enhanced classes
__all__ = [
    'EnhancedBacktestConfig', 'EnhancedTrade', 'EnhancedPortfolioState', 
    'EnhancedRiskManager', 'EnhancedBacktestEngine',
    'BacktestConfig', 'Trade', 'PortfolioState', 'BacktestEngine'  # Backward compatibility
]