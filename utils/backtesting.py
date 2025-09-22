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
        self.name = f"ML_Strategy_{horizon}"
        
    def generate_signals(self, current_date: datetime, prices: Dict[str, float]) -> Dict[str, Dict]:
        """Generate trading signals for all available stocks"""
        signals = {}
        
        try:
            for ticker in self.models.keys():
                if ticker in self.featured_data and ticker in prices:
                    # Get latest features for this stock
                    ticker_features = self.featured_data[ticker]
                    
                    # Filter features up to current date
                    date_mask = ticker_features.index <= current_date
                    if not date_mask.any():
                        continue
                        
                    latest_features = ticker_features[date_mask].iloc[-1:]
                    
                    if len(latest_features) > 0:
                        # Get model predictions
                        ticker_models = self.models[ticker]
                        
                        # Ensemble prediction
                        predictions = []
                        confidences = []
                        
                        for model_name, model in ticker_models.items():
                            if hasattr(model, 'predict_proba'):
                                pred_proba = model.predict_proba(latest_features)[0]
                                prediction = 1 if pred_proba[1] > 0.5 else 0
                                confidence = max(pred_proba)
                            else:
                                prediction = model.predict(latest_features)[0]
                                confidence = 0.6  # Default confidence
                            
                            predictions.append(prediction)
                            confidences.append(confidence)
                        
                        # Ensemble decision
                        avg_prediction = np.mean(predictions)
                        avg_confidence = np.mean(confidences)
                        
                        signal_strength = avg_confidence * (1 if avg_prediction > 0.5 else -1)
                        
                        signals[ticker] = {
                            'signal': 1 if avg_prediction > 0.5 else 0,
                            'strength': abs(signal_strength),
                            'confidence': avg_confidence,
                            'price': prices[ticker],
                            'date': current_date
                        }
                        
        except Exception as e:
            logging.warning(f"Error generating signals: {e}")
            
        return signals

@dataclass
class EnhancedBacktestConfig:
    """Enhanced configuration for backtesting with risk management"""
    
    # Core parameters
    initial_capital: float = 1000000
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    
    # Position management
    max_positions: int = 10
    position_sizing_method: str = 'risk_parity'
    rebalance_frequency: str = 'monthly'
    
    # Risk management
    max_drawdown_limit: float = 0.15
    max_correlation: float = 0.7
    risk_free_rate: float = 0.06
    enable_risk_management: bool = True
    
    # Strategy parameters
    min_signal_strength: float = 0.3
    profit_target: float = 0.20
    stop_loss: float = 0.10
    max_holding_days: int = 60
    
    # Advanced risk settings
    var_confidence: float = 0.95
    kelly_cap: float = 0.25
    stress_test_frequency: int = 5
    risk_budget_limit: float = 0.2
    rebalance_threshold: float = 0.05

@dataclass
class PortfolioState:
    """Track portfolio state during backtesting"""
    
    def __init__(self, initial_capital: float):
        self.portfolio_value: float = initial_capital
        self.cash: float = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.last_rebalance: Optional[datetime] = None
        
        # Risk metrics
        self.drawdown: float = 0.0
        self.current_var: float = 0.0
        self.portfolio_correlation: float = 0.0
        self.risk_budget_used: float = 0.0
        self.peak_value: float = initial_capital

@dataclass
class EnhancedTrade:
    """Enhanced trade record with risk metrics"""
    
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    return_pct: float
    holding_period: int
    exit_signal: str
    net_pnl: float
    
    # Risk metrics
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    var_at_entry: float = 0.0
    correlation_at_entry: float = 0.0

class EnhancedBacktestEngine:
    """Enhanced backtesting engine with comprehensive risk management"""
    
    def __init__(self, config: EnhancedBacktestConfig):
        self.config = config
        self.portfolio_state = PortfolioState(config.initial_capital)
        self.portfolio_history: List[Dict] = []
        self.trades: List[EnhancedTrade] = []
        self.risk_events: List[Dict] = []
        
        # Initialize risk manager if enabled
        if config.enable_risk_management:
            risk_config = RiskConfig(
                var_confidence=config.var_confidence,
                max_correlation=config.max_correlation,
                max_drawdown=config.max_drawdown_limit,
                kelly_cap=config.kelly_cap
            )
            self.risk_manager = ComprehensiveRiskManager(risk_config)
        else:
            self.risk_manager = None
    
    def run_backtest(self, strategy: MLStrategy, data: Dict[str, pd.DataFrame], 
                    start_date: datetime, end_date: datetime) -> Dict:
        """Run enhanced backtest with risk management"""
        
        logging.info(f"Starting enhanced backtest from {start_date} to {end_date}")
        
        # Prepare data
        all_dates = self._get_trading_dates(data, start_date, end_date)
        rebalance_days = self._get_rebalance_days(self.config.rebalance_frequency)
        
        for current_date in all_dates:
            try:
                # Get current prices
                prices = self._get_prices_for_date(data, current_date)
                if not prices:
                    continue
                
                # Update portfolio value and risk metrics
                self._update_portfolio_value(prices, current_date)
                
                # Risk management checks
                if self.risk_manager:
                    risk_violations = self._check_risk_violations(prices, current_date)
                    if risk_violations:
                        self._handle_risk_violations(risk_violations, prices, current_date)
                
                # Generate signals
                signals = strategy.generate_signals(current_date, prices)
                
                # Portfolio rebalancing
                if self._should_rebalance(current_date, self.portfolio_state.last_rebalance, rebalance_days):
                    self._rebalance_portfolio(signals, prices, current_date)
                    self.portfolio_state.last_rebalance = current_date
                
                # Exit management
                self._manage_exits(prices, current_date)
                
                # Record portfolio state
                self._record_enhanced_portfolio_state(current_date)
                
            except Exception as e:
                logging.warning(f"Error on {current_date}: {e}")
                continue
        
        # Generate final results
        return self._generate_enhanced_results()
    
    def _get_trading_dates(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get available trading dates from data"""
        all_dates = set()
        
        for ticker_data in data.values():
            if not ticker_data.empty:
                ticker_dates = ticker_data.index.to_pydatetime()
                date_mask = (ticker_dates >= start_date) & (ticker_dates <= end_date)
                all_dates.update(ticker_dates[date_mask])
        
        return sorted(list(all_dates))
    
    def _get_rebalance_days(self, frequency: str) -> int:
        """Convert rebalance frequency to days"""
        freq_map = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90
        }
        return freq_map.get(frequency, 30)
    
    def _get_prices_for_date(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        """Get closing prices for all stocks on given date"""
        prices = {}
        
        for ticker, ticker_data in data.items():
            if not ticker_data.empty:
                # Find the closest date
                available_dates = ticker_data.index
                if date in available_dates:
                    prices[ticker] = ticker_data.loc[date, 'Close']
                else:
                    # Find closest previous date
                    previous_dates = available_dates[available_dates <= date]
                    if len(previous_dates) > 0:
                        closest_date = previous_dates[-1]
                        prices[ticker] = ticker_data.loc[closest_date, 'Close']
        
        return prices
    
    def _update_portfolio_value(self, prices: Dict[str, float], current_date: datetime):
        """Update portfolio value and risk metrics"""
        
        # Calculate position values
        position_value = 0
        position_weights = {}
        
        for ticker, position in self.portfolio_state.positions.items():
            if ticker in prices:
                current_value = position['quantity'] * prices[ticker]
                position_value += current_value
                position_weights[ticker] = current_value
        
        # Total portfolio value
        self.portfolio_state.portfolio_value = self.portfolio_state.cash + position_value
        
        # Update peak value for drawdown calculation
        if self.portfolio_state.portfolio_value > self.portfolio_state.peak_value:
            self.portfolio_state.peak_value = self.portfolio_state.portfolio_value
        
        # Calculate current drawdown
        self.portfolio_state.drawdown = (self.portfolio_state.peak_value - self.portfolio_state.portfolio_value) / self.portfolio_state.peak_value
        
        # Update risk metrics if risk manager is enabled
        if self.risk_manager and position_weights:
            # Normalize weights
            total_value = sum(position_weights.values())
            if total_value > 0:
                for ticker in position_weights:
                    position_weights[ticker] /= total_value
                
                # Calculate portfolio metrics
                try:
                    returns_data = self._get_returns_data(prices, current_date)
                    if returns_data is not None and len(returns_data) > 20:
                        portfolio_returns = self._calculate_portfolio_returns(returns_data, position_weights)
                        
                        # VaR calculation
                        if len(portfolio_returns) > 0:
                            self.portfolio_state.current_var = np.percentile(portfolio_returns, (1 - self.config.var_confidence) * 100)
                        
                        # Correlation calculation
                        if len(position_weights) > 1:
                            corr_matrix = returns_data[list(position_weights.keys())].corr()
                            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                            self.portfolio_state.portfolio_correlation = avg_corr
                        
                except Exception as e:
                    logging.warning(f"Error calculating risk metrics: {e}")
    
    def _get_returns_data(self, prices: Dict[str, float], current_date: datetime, lookback_days: int = 60) -> Optional[pd.DataFrame]:
        """Get returns data for risk calculations"""
        # This would need access to historical data
        # For now, return None to avoid errors
        return None
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """Calculate portfolio returns"""
        portfolio_returns = []
        
        for date_idx in returns_data.index:
            daily_return = 0
            for ticker, weight in weights.items():
                if ticker in returns_data.columns:
                    daily_return += weight * returns_data.loc[date_idx, ticker]
            portfolio_returns.append(daily_return)
        
        return np.array(portfolio_returns)
    
    def _check_risk_violations(self, prices: Dict[str, float], current_date: datetime) -> List[str]:
        """Check for risk violations"""
        violations = []
        
        # Drawdown check
        if self.portfolio_state.drawdown > self.config.max_drawdown_limit:
            violations.append(f"Max drawdown exceeded: {self.portfolio_state.drawdown:.2%}")
        
        # Correlation check
        if self.portfolio_state.portfolio_correlation > self.config.max_correlation:
            violations.append(f"Portfolio correlation too high: {self.portfolio_state.portfolio_correlation:.2%}")
        
        # VaR check (if available)
        if hasattr(self.portfolio_state, 'current_var') and self.portfolio_state.current_var < -0.05:
            violations.append(f"High VaR risk: {self.portfolio_state.current_var:.2%}")
        
        return violations
    
    def _handle_risk_violations(self, violations: List[str], prices: Dict[str, float], current_date: datetime):
        """Handle risk violations"""
        
        # Record risk event
        risk_event = {
            'date': current_date,
            'type': 'Risk Violation',
            'violations': violations,
            'actions_taken': []
        }
        
        # Take action based on violations
        if any("drawdown" in v for v in violations):
            # Reduce position sizes
            self._reduce_position_sizes(0.5, prices, current_date)
            risk_event['actions_taken'].append("Reduced position sizes by 50%")
        
        if any("correlation" in v for v in violations):
            # Liquidate highly correlated positions
            self._liquidate_correlated_positions(prices, current_date)
            risk_event['actions_taken'].append("Liquidated highly correlated positions")
        
        self.risk_events.append(risk_event)
    
    def _reduce_position_sizes(self, reduction_factor: float, prices: Dict[str, float], current_date: datetime):
        """Reduce all position sizes by given factor"""
        for ticker in list(self.portfolio_state.positions.keys()):
            if ticker in prices:
                position = self.portfolio_state.positions[ticker]
                reduce_quantity = int(position['quantity'] * reduction_factor)
                
                if reduce_quantity > 0:
                    self._execute_enhanced_trade(
                        ticker, reduce_quantity, prices[ticker], 
                        'sell', current_date, 'Risk Management'
                    )
    
    def _liquidate_correlated_positions(self, prices: Dict[str, float], current_date: datetime):
        """Liquidate positions with high correlation"""
        # Simple implementation - liquidate positions with correlation > threshold
        positions_to_liquidate = []
        
        for ticker in self.portfolio_state.positions.keys():
            if self.portfolio_state.portfolio_correlation > self.config.max_correlation:
                positions_to_liquidate.append(ticker)
        
        # Liquidate half of highly correlated positions
        for i, ticker in enumerate(positions_to_liquidate):
            if i % 2 == 0 and ticker in prices:  # Liquidate every other position
                position = self.portfolio_state.positions[ticker]
                self._execute_enhanced_trade(
                    ticker, position['quantity'], prices[ticker],
                    'sell', current_date, 'Correlation Risk'
                )
    
    def _rebalance_portfolio(self, signals: Dict[str, Dict], prices: Dict[str, float], current_date: datetime):
        """Rebalance portfolio based on signals and risk management"""
        
        # Filter signals by strength
        valid_signals = {
            ticker: signal for ticker, signal in signals.items() 
            if signal['strength'] >= self.config.min_signal_strength
        }
        
        if not valid_signals:
            return
        
        # Calculate position sizes
        target_positions = self._calculate_position_sizes(valid_signals, prices)
        
        # Execute trades to reach target positions
        for ticker, target_size in target_positions.items():
            if ticker in prices:
                current_size = self.portfolio_state.positions.get(ticker, {}).get('quantity', 0)
                size_diff = target_size - current_size
                
                if abs(size_diff) > 0:
                    direction = 'buy' if size_diff > 0 else 'sell'
                    self._execute_enhanced_trade(
                        ticker, abs(size_diff), prices[ticker],
                        direction, current_date, 'Rebalance'
                    )
    
    def _calculate_position_sizes(self, signals: Dict[str, Dict], prices: Dict[str, float]) -> Dict[str, int]:
        """Calculate optimal position sizes"""
        
        if self.config.position_sizing_method == 'equal_weight':
            return self._equal_weight_sizing(signals, prices)
        elif self.config.position_sizing_method == 'risk_parity':
            return self._risk_parity_sizing(signals, prices)
        elif self.config.position_sizing_method == 'kelly_criterion':
            return self._kelly_criterion_sizing(signals, prices)
        else:
            return self._equal_weight_sizing(signals, prices)
    
    def _equal_weight_sizing(self, signals: Dict[str, Dict], prices: Dict[str, float]) -> Dict[str, int]:
        """Equal weight position sizing"""
        target_positions = {}
        
        if not signals:
            return target_positions
        
        # Allocate equal weight to each position
        available_capital = min(self.portfolio_state.portfolio_value * 0.95, self.portfolio_state.cash)
        position_value = available_capital / len(signals)
        
        for ticker in signals.keys():
            if ticker in prices and prices[ticker] > 0:
                quantity = int(position_value / prices[ticker])
                target_positions[ticker] = max(0, quantity)
        
        return target_positions
    
    def _risk_parity_sizing(self, signals: Dict[str, Dict], prices: Dict[str, float]) -> Dict[str, int]:
        """Risk parity position sizing (simplified)"""
        target_positions = {}
        
        if not signals:
            return target_positions
        
        # Simplified risk parity - weight by inverse confidence
        total_inv_confidence = sum(1/signal['confidence'] for signal in signals.values())
        available_capital = min(self.portfolio_state.portfolio_value * 0.95, self.portfolio_state.cash)
        
        for ticker, signal in signals.items():
            if ticker in prices and prices[ticker] > 0:
                weight = (1/signal['confidence']) / total_inv_confidence
                position_value = available_capital * weight
                quantity = int(position_value / prices[ticker])
                target_positions[ticker] = max(0, quantity)
        
        return target_positions
    
    def _kelly_criterion_sizing(self, signals: Dict[str, Dict], prices: Dict[str, float]) -> Dict[str, int]:
        """Kelly criterion position sizing (simplified)"""
        target_positions = {}
        
        if not signals:
            return target_positions
        
        available_capital = min(self.portfolio_state.portfolio_value * 0.95, self.portfolio_state.cash)
        
        for ticker, signal in signals.items():
            if ticker in prices and prices[ticker] > 0:
                # Simplified Kelly formula
                win_prob = signal['confidence']
                avg_win = 0.15  # Assume 15% average win
                avg_loss = 0.10  # Assume 10% average loss
                
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, self.config.kelly_cap))
                
                position_value = available_capital * kelly_fraction
                quantity = int(position_value / prices[ticker])
                target_positions[ticker] = max(0, quantity)
        
        return target_positions
    
    def _manage_exits(self, prices: Dict[str, float], current_date: datetime):
        """Manage position exits based on various criteria"""
        
        positions_to_exit = []
        
        for ticker, position in self.portfolio_state.positions.items():
            if ticker in prices:
                current_price = prices[ticker]
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                
                # Calculate returns
                return_pct = (current_price - entry_price) / entry_price
                holding_days = (current_date - entry_date).days
                
                exit_reason = None
                
                # Profit target
                if return_pct >= self.config.profit_target:
                    exit_reason = 'Profit Target'
                
                # Stop loss
                elif return_pct <= -self.config.stop_loss:
                    exit_reason = 'Stop Loss'
                
                # Max holding period
                elif holding_days >= self.config.max_holding_days:
                    exit_reason = 'Max Holding Period'
                
                if exit_reason:
                    positions_to_exit.append((ticker, exit_reason))
        
        # Execute exits
        for ticker, reason in positions_to_exit:
            if ticker in prices:
                position = self.portfolio_state.positions[ticker]
                self._execute_enhanced_trade(
                    ticker, position['quantity'], prices[ticker],
                    'sell', current_date, reason
                )
    
    def _execute_enhanced_trade(self, ticker: str, quantity: int, price: float, 
                              direction: str, date: datetime, reason: str) -> bool:
        """Execute enhanced trade with comprehensive tracking"""
        
        if quantity <= 0:
            return False
        
        # Calculate costs
        slippage = self._calculate_slippage(price, quantity, direction)
        execution_price = price + slippage
        gross_value = quantity * execution_price
        transaction_cost = gross_value * self.config.transaction_cost_pct
        net_value = gross_value + transaction_cost
        
        if direction == 'buy':
            # Check if we have enough cash
            if net_value > self.portfolio_state.cash:
                return False
            
            # Execute buy
            self.portfolio_state.cash -= net_value
            
            if ticker in self.portfolio_state.positions:
                # Add to existing position (average price)
                existing = self.portfolio_state.positions[ticker]
                total_quantity = existing['quantity'] + quantity
                avg_price = ((existing['quantity'] * existing['entry_price']) + 
                           (quantity * execution_price)) / total_quantity
                
                self.portfolio_state.positions[ticker].update({
                    'quantity': total_quantity,
                    'entry_price': avg_price
                })
            else:
                # New position
                self.portfolio_state.positions[ticker] = {
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'entry_date': date
                }
        
        else:  # sell
            # Check if we have enough shares
            if ticker not in self.portfolio_state.positions:
                return False
            
            position = self.portfolio_state.positions[ticker]
            if quantity > position['quantity']:
                return False
            
            # Execute sell
            self.portfolio_state.cash += (gross_value - transaction_cost)
            
            # Record trade
            if 'entry_date' in position:
                holding_period = (date - position['entry_date']).days
                return_pct = (execution_price - position['entry_price']) / position['entry_price']
                net_pnl = quantity * (execution_price - position['entry_price']) - transaction_cost
                
                trade = EnhancedTrade(
                    ticker=ticker,
                    entry_date=position['entry_date'],
                    exit_date=date,
                    entry_price=position['entry_price'],
                    exit_price=execution_price,
                    quantity=quantity,
                    return_pct=return_pct,
                    holding_period=holding_period,
                    exit_signal=reason,
                    net_pnl=net_pnl
                )
                
                self.trades.append(trade)
            
            # Update position
            if quantity == position['quantity']:
                # Complete exit
                del self.portfolio_state.positions[ticker]
            else:
                # Partial exit
                position['quantity'] -= quantity
        
        return True
    
    def _calculate_slippage(self, price: float, quantity: int, direction: str) -> float:
        """Calculate realistic slippage"""
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
        """Liquidate all positions"""
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
            'risk_manager_history': self.risk_manager.risk_history if self.risk_manager else []
        }
    
    def _calculate_enhanced_metrics(self, portfolio_df: pd.DataFrame, returns: pd.Series) -> Dict:
        """Calculate enhanced performance metrics"""
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1
        annual_return = (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) ** (252 / len(portfolio_df)) - 1
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Enhanced risk-adjusted metrics
        enhanced_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': portfolio_df['drawdown'].max(),
            'avg_drawdown': portfolio_df['drawdown'].mean(),
            'total_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t.return_pct > 0]) / len(self.trades) if self.trades else 0,
            'avg_trade_return': np.mean([t.return_pct for t in self.trades]) if self.trades else 0,
            'profit_factor': abs(sum(t.net_pnl for t in self.trades if t.net_pnl > 0) / 
                                sum(t.net_pnl for t in self.trades if t.net_pnl < 0)) if any(t.net_pnl < 0 for t in self.trades) else float('inf'),
            'risk_events': len(self.risk_events),
            'final_value': portfolio_df['portfolio_value'].iloc[-1]
        }
        
        # Risk-adjusted return metrics
        if 'current_var' in portfolio_df.columns:
            avg_var = portfolio_df['current_var'].mean()
            enhanced_metrics['var_adjusted_return'] = annual_return / abs(avg_var) if avg_var < 0 else 0
        
        # Correlation-adjusted metrics
        if 'portfolio_correlation' in portfolio_df.columns:
            avg_correlation = portfolio_df['portfolio_correlation'].mean()
            enhanced_metrics['correlation_penalty'] = avg_correlation
            enhanced_metrics['diversification_ratio'] = 1 - avg_correlation
        
        return enhanced_metrics
    
    def _analyze_risk_performance(self) -> Dict:
        """Analyze risk management performance"""
        
        risk_analysis = {
            'total_risk_events': len(self.risk_events),
            'risk_event_types': {},
            'avg_drawdown_duration': 0,
            'max_consecutive_losses': 0,
            'risk_adjusted_metrics': {}
        }
        
        # Analyze risk events
        for event in self.risk_events:
            event_type = event.get('type', 'Unknown')
            risk_analysis['risk_event_types'][event_type] = risk_analysis['risk_event_types'].get(event_type, 0) + 1
        
        # Analyze trade sequences
        if self.trades:
            consecutive_losses = 0
            max_consecutive = 0
            
            for trade in self.trades:
                if trade.return_pct < 0:
                    consecutive_losses += 1
                    max_consecutive = max(max_consecutive, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            risk_analysis['max_consecutive_losses'] = max_consecutive
        
        return risk_analysis


class BacktestAnalyzer:
    """Analyzer for backtest results with comprehensive reporting"""
    
    def __init__(self, results: Dict):
        self.results = results
        self.portfolio_history = results.get('portfolio_history', pd.DataFrame())
        self.trades = results.get('enhanced_trades', [])
        self.risk_events = results.get('risk_events', [])
        self.metrics = results.get('enhanced_metrics', {})
        self.risk_analysis = results.get('risk_analysis', {})
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        if self.portfolio_history.empty:
            return {'error': 'No portfolio history available'}
        
        # Performance summary
        performance_summary = {
            'period': {
                'start_date': self.portfolio_history.index[0].strftime('%Y-%m-%d'),
                'end_date': self.portfolio_history.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(self.portfolio_history)
            },
            'returns': {
                'total_return': self.metrics.get('total_return', 0),
                'annual_return': self.metrics.get('annual_return', 0),
                'volatility': self.metrics.get('volatility', 0),
                'sharpe_ratio': self.metrics.get('sharpe_ratio', 0)
            },
            'risk_metrics': {
                'max_drawdown': self.metrics.get('max_drawdown', 0),
                'avg_drawdown': self.metrics.get('avg_drawdown', 0),
                'var_adjusted_return': self.metrics.get('var_adjusted_return', 0),
                'diversification_ratio': self.metrics.get('diversification_ratio', 0)
            },
            'trading_activity': {
                'total_trades': len(self.trades),
                'win_rate': self.metrics.get('win_rate', 0),
                'avg_trade_return': self.metrics.get('avg_trade_return', 0),
                'profit_factor': self.metrics.get('profit_factor', 0)
            },
            'risk_management': {
                'total_risk_events': len(self.risk_events),
                'risk_event_types': self.risk_analysis.get('risk_event_types', {}),
                'max_consecutive_losses': self.risk_analysis.get('max_consecutive_losses', 0)
            }
        }
        
        return performance_summary
    
    def analyze_trades(self) -> Dict:
        """Analyze trading performance in detail"""
        
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        # Convert trades to DataFrame for analysis
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'ticker': trade.ticker,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'holding_period': trade.holding_period,
                'return_pct': trade.return_pct,
                'net_pnl': trade.net_pnl,
                'exit_reason': trade.exit_signal
            })
        
        trades_df = pd.DataFrame(trade_data)
        
        # Trading statistics
        trade_analysis = {
            'by_ticker': trades_df.groupby('ticker').agg({
                'return_pct': ['count', 'mean', 'std'],
                'net_pnl': 'sum',
                'holding_period': 'mean'
            }).round(4).to_dict(),
            
            'by_exit_reason': trades_df.groupby('exit_reason').agg({
                'return_pct': ['count', 'mean'],
                'net_pnl': 'sum'
            }).round(4).to_dict(),
            
            'monthly_performance': trades_df.set_index('exit_date').resample('M').agg({
                'return_pct': ['count', 'mean'],
                'net_pnl': 'sum'
            }).round(4).to_dict(),
            
            'holding_period_analysis': {
                'avg_holding_days': trades_df['holding_period'].mean(),
                'median_holding_days': trades_df['holding_period'].median(),
                'max_holding_days': trades_df['holding_period'].max(),
                'min_holding_days': trades_df['holding_period'].min()
            }
        }
        
        return trade_analysis
    
    def create_performance_charts(self) -> Dict:
        """Create performance visualization data"""
        
        if self.portfolio_history.empty:
            return {'error': 'No data for charts'}
        
        # Portfolio value over time
        portfolio_chart = {
            'dates': self.portfolio_history.index.strftime('%Y-%m-%d').tolist(),
            'values': self.portfolio_history['portfolio_value'].tolist(),
            'drawdown': self.portfolio_history['drawdown'].tolist()
        }
        
        # Trade performance
        if self.trades:
            trade_returns = [t.return_pct for t in self.trades]
            trade_dates = [t.exit_date.strftime('%Y-%m-%d') for t in self.trades]
            
            trades_chart = {
                'dates': trade_dates,
                'returns': trade_returns,
                'cumulative_pnl': np.cumsum([t.net_pnl for t in self.trades]).tolist()
            }
        else:
            trades_chart = {'dates': [], 'returns': [], 'cumulative_pnl': []}
        
        # Risk metrics over time
        risk_chart = {
            'dates': self.portfolio_history.index.strftime('%Y-%m-%d').tolist(),
            'var': self.portfolio_history.get('current_var', []).tolist(),
            'correlation': self.portfolio_history.get('portfolio_correlation', []).tolist()
        }
        
        return {
            'portfolio_performance': portfolio_chart,
            'trade_performance': trades_chart,
            'risk_metrics': risk_chart
        }


class BacktestDB:
    """Database for storing backtest results"""
    
    def __init__(self, db_path: str = "backtests.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for backtest storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Backtests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_date TEXT,
                    config TEXT,
                    results TEXT,
                    metrics TEXT
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT,
                    ticker TEXT,
                    entry_date TEXT,
                    exit_date TEXT,
                    return_pct REAL,
                    net_pnl REAL,
                    exit_reason TEXT,
                    FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                )
            """)
            
            conn.commit()
    
    def save_backtest(self, backtest_id: str, name: str, results: Dict) -> bool:
        """Save backtest results to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save main backtest record
                cursor.execute("""
                    INSERT OR REPLACE INTO backtests 
                    (id, name, created_date, config, results, metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    name,
                    datetime.now().isoformat(),
                    pickle.dumps(results.get('config')).hex(),
                    pickle.dumps(results).hex(),
                    pickle.dumps(results.get('enhanced_metrics', {})).hex()
                ))
                
                # Save individual trades
                trades = results.get('enhanced_trades', [])
                for trade in trades:
                    cursor.execute("""
                        INSERT INTO trades 
                        (backtest_id, ticker, entry_date, exit_date, return_pct, net_pnl, exit_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        backtest_id,
                        trade.ticker,
                        trade.entry_date.isoformat(),
                        trade.exit_date.isoformat(),
                        trade.return_pct,
                        trade.net_pnl,
                        trade.exit_signal
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Error saving backtest: {e}")
            return False
    
    def get_backtest(self, backtest_id: str) -> Optional[Dict]:
        """Retrieve backtest results from database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT name, created_date, config, results, metrics
                    FROM backtests WHERE id = ?
                """, (backtest_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                name, created_date, config_hex, results_hex, metrics_hex = row
                
                return {
                    'id': backtest_id,
                    'name': name,
                    'created_date': created_date,
                    'config': pickle.loads(bytes.fromhex(config_hex)),
                    'results': pickle.loads(bytes.fromhex(results_hex)),
                    'metrics': pickle.loads(bytes.fromhex(metrics_hex))
                }
                
        except Exception as e:
            logging.error(f"Error retrieving backtest: {e}")
            return None
    
    def list_backtests(self) -> List[Dict]:
        """List all saved backtests"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, created_date FROM backtests
                    ORDER BY created_date DESC
                """)
                
                rows = cursor.fetchall()
                return [
                    {'id': row[0], 'name': row[1], 'created_date': row[2]}
                    for row in rows
                ]
                
        except Exception as e:
            logging.error(f"Error listing backtests: {e}")
            return []
    
    def delete_backtest(self, backtest_id: str) -> bool:
        """Delete a backtest and its trades"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete trades first (foreign key constraint)
                cursor.execute("DELETE FROM trades WHERE backtest_id = ?", (backtest_id,))
                
                # Delete backtest
                cursor.execute("DELETE FROM backtests WHERE id = ?", (backtest_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            logging.error(f"Error deleting backtest: {e}")
            return False


class PerformanceMetrics:
    """Static methods for calculating performance metrics"""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns.mean() - risk_free_rate/252
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns.mean() - risk_free_rate/252
        downside_std = returns[returns < 0].std()
        if downside_std == 0:
            return float('inf') if excess_returns > 0 else 0
        return (excess_returns / downside_std) * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1
        max_dd = PerformanceMetrics.max_drawdown(portfolio_values)
        return annual_return / abs(max_dd) if max_dd != 0 else 0
    
    @staticmethod
    def max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = PerformanceMetrics.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        if tracking_error == 0:
            return 0
        return excess_returns.mean() / tracking_error * np.sqrt(252)
    
    @staticmethod
    def beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Beta"""
        covariance = np.cov(returns.dropna(), market_returns.dropna())[0][1]
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 0
    
    @staticmethod
    def alpha(returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Alpha"""
        beta = PerformanceMetrics.beta(returns, market_returns)
        portfolio_return = returns.mean() * 252
        market_return = market_returns.mean() * 252
        return portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))


# Utility functions for enhanced backtesting
def create_backtest_config(**kwargs) -> EnhancedBacktestConfig:
    """Create backtest configuration with defaults"""
    return EnhancedBacktestConfig(**kwargs)


def run_enhanced_backtest(strategy: MLStrategy, data: Dict[str, pd.DataFrame], 
                         config: EnhancedBacktestConfig, 
                         start_date: datetime, end_date: datetime) -> Dict:
    """Run enhanced backtest with full configuration"""
    
    engine = EnhancedBacktestEngine(config)
    return engine.run_backtest(strategy, data, start_date, end_date)