# utils/backtesting.py
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 1000000  # 10 lakh
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    min_position_size: float = 10000  # Minimum 10k per position
    max_position_size: float = 200000  # Maximum 2 lakh per position
    max_positions: int = 20  # Maximum concurrent positions
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    benchmark: str = 'NIFTY50'
    risk_free_rate: float = 0.06  # 6% annual
    max_drawdown_limit: float = 0.15  # Stop trading if drawdown > 15%
    position_sizing_method: str = 'equal_weight'  # 'equal_weight', 'risk_parity', 'kelly'
    lookback_window: int = 252  # Days for rolling calculations
    confidence_level: float = 0.95  # For VaR calculations

@dataclass
class Trade:
    """Individual trade record"""
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

@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: Dict[str, Dict] = field(default_factory=dict)
    portfolio_value: float = 0.0
    leverage: float = 1.0
    drawdown: float = 0.0
    peak_value: float = 0.0

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_returns(portfolio_values: pd.Series) -> pd.Series:
        """Calculate portfolio returns"""
        return portfolio_values.pct_change().dropna()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio (return/max drawdown)"""
        annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1
        max_drawdown = PerformanceMetrics.max_drawdown(portfolio_values)
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = PerformanceMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def win_rate(trades: List[Trade]) -> float:
        """Calculate percentage of winning trades"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade.net_pnl > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not trades:
            return 0.0
        gross_profit = sum(trade.net_pnl for trade in trades if trade.net_pnl > 0)
        gross_loss = abs(sum(trade.net_pnl for trade in trades if trade.net_pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @staticmethod
    def average_trade_duration(trades: List[Trade]) -> float:
        """Calculate average holding period in days"""
        if not trades:
            return 0.0
        return np.mean([trade.holding_period for trade in trades])

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def calculate_position_size(self, signal_strength: float, portfolio_value: float, 
                              stock_price: float, volatility: float = None) -> int:
        """Calculate position size based on method"""
        
        if self.config.position_sizing_method == 'equal_weight':
            target_value = portfolio_value * 0.05  # 5% per position
            return int(target_value / stock_price)
            
        elif self.config.position_sizing_method == 'risk_parity':
            if volatility is None:
                volatility = 0.02  # Default 2% daily volatility
            risk_budget = portfolio_value * 0.01  # 1% risk per position
            position_value = risk_budget / volatility
            return int(position_value / stock_price)
            
        elif self.config.position_sizing_method == 'kelly':
            # Simplified Kelly criterion
            if signal_strength <= 0.5:
                return 0
            win_prob = signal_strength
            avg_win = 0.05  # Assume 5% average win
            avg_loss = 0.03  # Assume 3% average loss
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            target_value = portfolio_value * kelly_fraction
            return int(target_value / stock_price)
        
        else:
            # Default equal weight
            target_value = portfolio_value * 0.05
            return int(target_value / stock_price)
    
    def apply_risk_limits(self, quantity: int, stock_price: float) -> int:
        """Apply position size limits"""
        position_value = quantity * stock_price
        
        # Apply minimum position size
        if position_value < self.config.min_position_size:
            return 0
            
        # Apply maximum position size
        if position_value > self.config.max_position_size:
            return int(self.config.max_position_size / stock_price)
            
        return quantity
    
    def check_portfolio_risk(self, portfolio_state: PortfolioState) -> Dict[str, bool]:
        """Check various risk constraints"""
        checks = {
            'max_drawdown_ok': portfolio_state.drawdown > -self.config.max_drawdown_limit,
            'max_positions_ok': len(portfolio_state.positions) <= self.config.max_positions,
            'leverage_ok': portfolio_state.leverage <= 1.0,  # No leverage for now
        }
        return checks

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """Generate trading signals for given date"""
        pass
    
    @abstractmethod
    def should_exit(self, ticker: str, entry_date: datetime, current_date: datetime, 
                   current_price: float, entry_price: float) -> bool:
        """Determine if position should be exited"""
        pass

class MLStrategy(Strategy):
    """Machine Learning based strategy using your existing models"""
    
    def __init__(self, models: Dict, featured_data: Dict, horizon: str = 'next_month'):
        self.models = models
        self.featured_data = featured_data
        self.horizon = horizon
        
    def generate_signals(self, data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """Generate signals from ML models"""
        signals = {}
        
        for ticker in self.models.keys():
            if ticker not in self.featured_data:
                continue
                
            ticker_data = self.featured_data[ticker]
            # Get data up to current date
            historical_data = ticker_data[ticker_data.index <= date]
            
            if len(historical_data) < 50:  # Need minimum history
                continue
                
            try:
                # Get the latest data point
                latest_data = historical_data.iloc[[-1]]
                
                # Get model predictions
                ticker_models = self.models[ticker]
                predictions = []
                confidences = []
                
                for model_key, model in ticker_models.items():
                    if self.horizon in model_key:
                        try:
                            prob = model.predict_proba(latest_data, ticker)[0][1]
                            pred = model.predict(latest_data, ticker)[0]
                            confidence = model.validation_score if hasattr(model, 'validation_score') else 0.5
                            
                            predictions.append(prob)
                            confidences.append(confidence)
                        except Exception as e:
                            logging.warning(f"Model prediction failed for {ticker}: {e}")
                            continue
                
                if predictions:
                    # Weight predictions by confidence
                    if sum(confidences) > 0:
                        weighted_prob = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
                    else:
                        weighted_prob = np.mean(predictions)
                    
                    signals[ticker] = weighted_prob
                    
            except Exception as e:
                logging.warning(f"Signal generation failed for {ticker}: {e}")
                continue
                
        return signals
    
    def should_exit(self, ticker: str, entry_date: datetime, current_date: datetime, 
                   current_price: float, entry_price: float) -> bool:
        """Exit logic based on time and profit/loss"""
        
        # Time-based exit
        holding_period = (current_date - entry_date).days
        if holding_period > 30:  # Exit after 30 days for monthly horizon
            return True
            
        # Profit/loss based exit
        return_pct = (current_price - entry_price) / entry_price
        
        # Take profit at 10%
        if return_pct > 0.10:
            return True
            
        # Stop loss at -5%
        if return_pct < -0.05:
            return True
            
        return False

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.portfolio_state = PortfolioState(cash=config.initial_capital)
        self.trades: List[Trade] = []
        self.portfolio_history = []
        self.benchmark_data = None
        
    def load_benchmark_data(self, benchmark_file: str = None):
        """Load benchmark data for comparison"""
        # Placeholder - you would load NIFTY50 data here
        # For now, simulate benchmark returns
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        benchmark_returns = np.random.normal(0.0005, 0.015, len(dates))  # Simulate market returns
        self.benchmark_data = pd.Series(benchmark_returns, index=dates)
        
    def calculate_slippage(self, price: float, quantity: int, direction: str) -> float:
        """Calculate realistic slippage based on order size"""
        base_slippage = price * self.config.slippage_pct
        
        # Increase slippage for larger orders
        size_factor = min(2.0, quantity / 1000)  # Cap at 2x slippage
        
        # Direction matters - buying increases price, selling decreases
        slippage_direction = 1 if direction == 'buy' else -1
        
        return base_slippage * size_factor * slippage_direction
        
    def execute_trade(self, ticker: str, quantity: int, price: float, 
                     direction: str, date: datetime, signal: str) -> bool:
        """Execute a trade with realistic costs"""
        
        if quantity <= 0:
            return False
            
        # Calculate slippage
        slippage = self.calculate_slippage(price, quantity, direction)
        execution_price = price + slippage
        
        # Calculate transaction costs
        trade_value = quantity * execution_price
        transaction_cost = trade_value * self.config.transaction_cost_pct
        
        total_cost = trade_value + transaction_cost
        
        if direction == 'buy':
            # Check if we have enough cash
            if total_cost > self.portfolio_state.cash:
                return False
                
            # Execute buy order
            self.portfolio_state.cash -= total_cost
            self.portfolio_state.positions[ticker] = {
                'quantity': quantity,
                'entry_price': execution_price,
                'entry_date': date,
                'entry_signal': signal,
                'transaction_cost': transaction_cost
            }
            
        elif direction == 'sell':
            # Check if we have the position
            if ticker not in self.portfolio_state.positions:
                return False
                
            position = self.portfolio_state.positions[ticker]
            
            # Create trade record
            holding_period = (date - position['entry_date']).days
            gross_pnl = quantity * (execution_price - position['entry_price'])
            total_transaction_cost = position['transaction_cost'] + transaction_cost
            net_pnl = gross_pnl - total_transaction_cost
            return_pct = net_pnl / (quantity * position['entry_price'])
            
            trade = Trade(
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
                holding_period=holding_period
            )
            
            self.trades.append(trade)
            
            # Add cash back
            self.portfolio_state.cash += trade_value - transaction_cost
            
            # Remove position
            del self.portfolio_state.positions[ticker]
            
        return True
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update current portfolio value"""
        position_value = 0.0
        
        for ticker, position in self.portfolio_state.positions.items():
            if ticker in current_prices:
                position_value += position['quantity'] * current_prices[ticker]
                
        self.portfolio_state.portfolio_value = self.portfolio_state.cash + position_value
        
        # Update drawdown tracking
        if self.portfolio_state.portfolio_value > self.portfolio_state.peak_value:
            self.portfolio_state.peak_value = self.portfolio_state.portfolio_value
            self.portfolio_state.drawdown = 0.0
        else:
            self.portfolio_state.drawdown = (self.portfolio_state.portfolio_value - self.portfolio_state.peak_value) / self.portfolio_state.peak_value
            
    def run_backtest(self, strategy: Strategy, data: Dict[str, pd.DataFrame], 
                    start_date: datetime, end_date: datetime) -> Dict:
        """Run the complete backtest"""
        
        logging.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize
        self.portfolio_state = PortfolioState(cash=self.config.initial_capital)
        self.portfolio_state.peak_value = self.config.initial_capital
        self.trades = []
        self.portfolio_history = []
        
        # Get all available dates from data
        all_dates = set()
        for ticker_data in data.values():
            all_dates.update(ticker_data.index)
        
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        # Rebalancing logic
        last_rebalance = None
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
                
                # Update portfolio value
                self.update_portfolio_value(current_prices)
                
                # Check risk constraints
                risk_checks = self.risk_manager.check_portfolio_risk(self.portfolio_state)
                if not all(risk_checks.values()):
                    logging.warning(f"Risk constraint violated on {current_date}: {risk_checks}")
                    # In severe cases, liquidate positions
                    if not risk_checks['max_drawdown_ok']:
                        self._liquidate_all_positions(current_prices, current_date, "Risk limit breach")
                
                # Check for exit signals on existing positions
                positions_to_exit = []
                for ticker in list(self.portfolio_state.positions.keys()):
                    if ticker in current_prices:
                        position = self.portfolio_state.positions[ticker]
                        should_exit = strategy.should_exit(
                            ticker, position['entry_date'], current_date,
                            current_prices[ticker], position['entry_price']
                        )
                        
                        if should_exit:
                            positions_to_exit.append(ticker)
                
                # Execute exits
                for ticker in positions_to_exit:
                    position = self.portfolio_state.positions[ticker]
                    self.execute_trade(
                        ticker, position['quantity'], current_prices[ticker],
                        'sell', current_date, "Exit signal"
                    )
                
                # Check if it's time to rebalance
                should_rebalance = False
                if last_rebalance is None:
                    should_rebalance = True
                elif (current_date - last_rebalance).days >= rebalance_days:
                    should_rebalance = True
                
                if should_rebalance:
                    # Generate new signals
                    signals = strategy.generate_signals(data, current_date)
                    
                    # Filter and rank signals
                    valid_signals = {k: v for k, v in signals.items() 
                                   if v > 0.6 and k in current_prices}  # Only strong buy signals
                    
                    # Sort by signal strength
                    sorted_signals = sorted(valid_signals.items(), key=lambda x: x[1], reverse=True)
                    
                    # Enter new positions
                    max_new_positions = self.config.max_positions - len(self.portfolio_state.positions)
                    for ticker, signal_strength in sorted_signals[:max_new_positions]:
                        if ticker not in self.portfolio_state.positions:
                            
                            # Calculate position size
                            stock_price = current_prices[ticker]
                            quantity = self.risk_manager.calculate_position_size(
                                signal_strength, self.portfolio_state.portfolio_value, stock_price
                            )
                            
                            # Apply risk limits
                            quantity = self.risk_manager.apply_risk_limits(quantity, stock_price)
                            
                            if quantity > 0:
                                success = self.execute_trade(
                                    ticker, quantity, stock_price, 'buy', 
                                    current_date, f"ML Signal: {signal_strength:.3f}"
                                )
                                
                                if success:
                                    logging.info(f"Opened position: {ticker} x {quantity} @ {stock_price}")
                    
                    last_rebalance = current_date
                
                # Record portfolio state
                self.portfolio_history.append({
                    'date': current_date,
                    'portfolio_value': self.portfolio_state.portfolio_value,
                    'cash': self.portfolio_state.cash,
                    'positions': len(self.portfolio_state.positions),
                    'drawdown': self.portfolio_state.drawdown
                })
                
            except Exception as e:
                logging.error(f"Error processing date {current_date}: {e}")
                continue
        
        # Close all remaining positions at end
        if self.portfolio_state.positions:
            final_prices = {ticker: data[ticker].iloc[-1]['Close'] 
                          for ticker in self.portfolio_state.positions.keys() 
                          if ticker in data}
            self._liquidate_all_positions(final_prices, end_date, "End of backtest")
        
        return self._generate_results()
    
    def _liquidate_all_positions(self, prices: Dict[str, float], date: datetime, reason: str):
        """Liquidate all positions"""
        for ticker in list(self.portfolio_state.positions.keys()):
            if ticker in prices:
                position = self.portfolio_state.positions[ticker]
                self.execute_trade(
                    ticker, position['quantity'], prices[ticker],
                    'sell', date, reason
                )
    
    def _generate_results(self) -> Dict:
        """Generate comprehensive backtest results"""
        
        if not self.portfolio_history:
            return {'error': 'No portfolio history generated'}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        returns = PerformanceMetrics.calculate_returns(portfolio_df['portfolio_value'])
        
        # Performance metrics
        metrics = {
            'total_return': (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1,
            'annual_return': (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) ** (252 / len(portfolio_df)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, self.config.risk_free_rate),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, self.config.risk_free_rate),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, portfolio_df['portfolio_value']),
            'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_df['portfolio_value']),
            'var_95': PerformanceMetrics.value_at_risk(returns, 0.95),
            'cvar_95': PerformanceMetrics.conditional_var(returns, 0.95),
            'win_rate': PerformanceMetrics.win_rate(self.trades),
            'profit_factor': PerformanceMetrics.profit_factor(self.trades),
            'avg_trade_duration': PerformanceMetrics.average_trade_duration(self.trades),
            'total_trades': len(self.trades),
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1]
        }
        
        # Trade analysis
        if self.trades:
            trade_returns = [trade.return_pct for trade in self.trades]
            trade_pnl = [trade.net_pnl for trade in self.trades]
            
            metrics.update({
                'avg_trade_return': np.mean(trade_returns),
                'avg_trade_pnl': np.mean(trade_pnl),
                'best_trade': max(trade_pnl),
                'worst_trade': min(trade_pnl),
                'trade_return_std': np.std(trade_returns)
            })
        
        return {
            'metrics': metrics,
            'portfolio_history': portfolio_df,
            'trades': self.trades,
            'returns': returns,
            'config': self.config
        }

class BacktestAnalyzer:
    """Analyze and visualize backtest results"""
    
    @staticmethod
    def create_performance_report(results: Dict) -> str:
        """Generate a comprehensive performance report"""
        
        metrics = results['metrics']
        
        report = f"""
BACKTEST PERFORMANCE REPORT
{'='*50}

RETURN METRICS:
Total Return: {metrics['total_return']:.2%}
Annual Return: {metrics['annual_return']:.2%}
Volatility: {metrics['volatility']:.2%}

RISK METRICS:
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Sortino Ratio: {metrics['sortino_ratio']:.3f}
Calmar Ratio: {metrics['calmar_ratio']:.3f}
Maximum Drawdown: {metrics['max_drawdown']:.2%}
Value at Risk (95%): {metrics['var_95']:.2%}
Conditional VaR (95%): {metrics['cvar_95']:.2%}

TRADING METRICS:
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2%}
Profit Factor: {metrics['profit_factor']:.3f}
Average Trade Duration: {metrics['avg_trade_duration']:.1f} days
Average Trade Return: {metrics.get('avg_trade_return', 0):.2%}
Best Trade: ₹{metrics.get('best_trade', 0):,.0f}
Worst Trade: ₹{metrics.get('worst_trade', 0):,.0f}

FINAL VALUES:
Final Portfolio Value: ₹{metrics['final_portfolio_value']:,.0f}
"""
        return report
    
    @staticmethod
    def plot_performance(results: Dict, save_path: str = None):
        """Create performance visualization plots"""
        
        portfolio_df = results['portfolio_history']
        metrics = results['metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value (₹)')
        axes[0, 0].grid(True)
        
        # Drawdown
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak
        axes[0, 1].fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True)
        
        # Returns distribution
        if 'returns' in results:
            returns = results['returns']
            axes[1, 0].hist(returns, bins=50, alpha=0.7)
            axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', label='Mean')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Rolling metrics
        if len(portfolio_df) > 60:
            rolling_sharpe = []
            rolling_window = 60
            
            for i in range(rolling_window, len(portfolio_df)):
                window_returns = portfolio_df['portfolio_value'].iloc[i-rolling_window:i].pct_change().dropna()
                if len(window_returns) > 0:
                    sharpe = PerformanceMetrics.sharpe_ratio(window_returns)
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
            
            rolling_dates = portfolio_df.index[rolling_window:]
            axes[1, 1].plot(rolling_dates, rolling_sharpe)
            axes[1, 1].set_title('Rolling 60-Day Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Database integration for storing backtest results
class BacktestDB:
    """Database operations for backtest results"""
    
    def __init__(self, db_path: str = "data/backtests.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    initial_capital REAL,
                    final_value REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT,
                    results_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id INTEGER,
                    ticker TEXT,
                    entry_date TEXT,
                    exit_date TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    net_pnl REAL,
                    return_pct REAL,
                    holding_period INTEGER,
                    FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                )
            """)
    
    def save_backtest(self, name: str, results: Dict) -> int:
        """Save backtest results to database"""
        
        metrics = results['metrics']
        config = results['config']
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO backtests 
                (name, initial_capital, final_value, total_return, sharpe_ratio, 
                 max_drawdown, total_trades, win_rate, config_json, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                config.initial_capital,
                metrics['final_portfolio_value'],
                metrics['total_return'],
                metrics['sharpe_ratio'],
                metrics['max_drawdown'],
                metrics['total_trades'],
                metrics['win_rate'],
                pickle.dumps(config).hex(),
                pickle.dumps(results).hex()
            ))
            
            backtest_id = cursor.lastrowid
            
            # Save trades
            for trade in results['trades']:
                conn.execute("""
                    INSERT INTO trades 
                    (backtest_id, ticker, entry_date, exit_date, entry_price, 
                     exit_price, quantity, net_pnl, return_pct, holding_period)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id, trade.ticker, trade.entry_date.isoformat(),
                    trade.exit_date.isoformat(), trade.entry_price, trade.exit_price,
                    trade.quantity, trade.net_pnl, trade.return_pct, trade.holding_period
                ))
            
            return backtest_id
    
    def load_backtest(self, backtest_id: int) -> Dict:
        """Load backtest results from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT results_json FROM backtests WHERE id = ?
            """, (backtest_id,))
            
            row = cursor.fetchone()
            if row:
                return pickle.loads(bytes.fromhex(row[0]))
            
        return None
    
    def list_backtests(self) -> pd.DataFrame:
        """List all saved backtests"""
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("""
                SELECT id, name, initial_capital, final_value, total_return, 
                       sharpe_ratio, max_drawdown, total_trades, win_rate, created_at
                FROM backtests
                ORDER BY created_at DESC
            """, conn)
            
        return df

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the backtesting framework
    
    # Configuration
    config = BacktestConfig(
        initial_capital=1000000,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
        max_positions=10,
        rebalance_frequency='monthly'
    )
    
    # Mock data for testing
    def create_mock_data():
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        data = {}
        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)  # Deterministic but different for each ticker
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
            
            df = pd.DataFrame({
                'Open': prices * np.random.uniform(0.99, 1.01, len(dates)),
                'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
                'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
                'Close': prices,
                'Volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
            
            data[ticker] = df
            
        return data
    
    # Create mock strategy
    class MockStrategy(Strategy):
        def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
            # Simple momentum strategy
            signals = {}
            for ticker, df in data.items():
                if date in df.index:
                    # Get recent data
                    recent_data = df[df.index <= date].tail(20)
                    if len(recent_data) >= 20:
                        short_ma = recent_data['Close'].tail(5).mean()
                        long_ma = recent_data['Close'].tail(20).mean()
                        
                        if short_ma > long_ma:
                            signals[ticker] = 0.7  # Buy signal
                        else:
                            signals[ticker] = 0.3  # Weak signal
            
            return signals
        
        def should_exit(self, ticker: str, entry_date: datetime, current_date: datetime, 
                       current_price: float, entry_price: float) -> bool:
            # Simple exit rules
            days_held = (current_date - entry_date).days
            return_pct = (current_price - entry_price) / entry_price
            
            # Exit after 30 days or 10% profit or -5% loss
            return days_held > 30 or return_pct > 0.10 or return_pct < -0.05
    
    # Run test
    print("Testing Backtesting Framework...")
    
    # Create test data
    test_data = create_mock_data()
    
    # Initialize engine and strategy
    engine = BacktestEngine(config)
    strategy = MockStrategy()
    
    # Run backtest
    results = engine.run_backtest(
        strategy=strategy,
        data=test_data,
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2022, 12, 31)
    )
    
    # Display results
    if 'error' not in results:
        analyzer = BacktestAnalyzer()
        report = analyzer.create_performance_report(results)
        print(report)
        
        # Save to database
        db = BacktestDB()
        backtest_id = db.save_backtest("Test Backtest", results)
        print(f"\nBacktest saved with ID: {backtest_id}")
        
        # List all backtests
        print("\nAll backtests:")
        print(db.list_backtests())
        
    else:
        print(f"Backtest failed: {results['error']}")
    
    print("\nBacktesting framework test completed!")