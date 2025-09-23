# utils/backtesting.py - Complete Enhanced Backtesting System
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
class EnhancedBacktestConfig(BacktestConfig):
    """Enhanced configuration with risk management features"""
    # Risk management parameters
    max_correlation: float = 0.7
    max_drawdown_limit: float = 0.15
    var_confidence: float = 0.95
    stress_test_frequency: int = 20  # days
    position_sizing_method: str = 'kelly'  # 'kelly', 'risk_parity', 'equal_weight'
    
    # Enhanced features
    enable_dynamic_hedging: bool = True
    enable_correlation_monitoring: bool = True
    enable_stress_testing: bool = True
    rebalance_on_risk_breach: bool = True
    
    # ML Strategy parameters
    prediction_confidence_threshold: float = 0.6
    ensemble_weight_decay: float = 0.1
    signal_aggregation_method: str = 'weighted_average'

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
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        downside_deviation = negative_returns.std()
        if downside_deviation == 0:
            return 0
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calmar_ratio(portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(portfolio_values) < 2:
            return 0.0
        
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        max_dd = abs(PerformanceMetrics.max_drawdown(portfolio_values))
        
        if max_dd == 0:
            return np.inf
        
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        return annual_return / max_dd

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        """Generate trading signals for given date and data"""
        pass
    
    @abstractmethod
    def should_exit(self, ticker: str, entry_date: datetime, current_date: datetime, 
                   current_price: float, entry_price: float) -> bool:
        """Determine if position should be exited"""
        pass

class MLStrategy(Strategy):
    """Machine Learning Strategy for backtesting"""
    
    def __init__(self, models: Dict, config: Dict = None):
        self.models = models
        self.config = config or {}
        self.prediction_horizon = self.config.get('investment_horizon', 'next_month')
        self.confidence_threshold = self.config.get('prediction_confidence_threshold', 0.6)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        """Generate ML-based trading signals"""
        signals = {}
        
        for ticker in data.keys():
            if ticker not in self.models:
                continue
                
            try:
                # Get the data up to current date
                df = data[ticker]
                current_data = df[df.index <= date]
                
                if len(current_data) < 50:  # Need minimum data
                    continue
                
                # Basic feature engineering for prediction
                features = self._prepare_features(current_data)
                
                if features is None or len(features) == 0:
                    continue
                
                # Get model prediction
                model = self.models[ticker]
                if hasattr(model, 'predict_proba'):
                    prediction = model.predict_proba([features])[-1]
                    confidence = max(prediction) if len(prediction) > 0 else 0.5
                    signal = 1.0 if prediction[1] > prediction[0] else 0.0
                else:
                    prediction = model.predict([features])[0]
                    confidence = abs(prediction)
                    signal = 1.0 if prediction > 0 else 0.0
                
                # Only generate signal if confidence is above threshold
                if confidence >= self.confidence_threshold:
                    signals[ticker] = signal * confidence
                
            except Exception as e:
                logging.warning(f"Signal generation failed for {ticker}: {e}")
                continue
        
        return signals
    
    def _prepare_features(self, df: pd.DataFrame) -> List[float]:
        """Prepare features for ML model prediction"""
        try:
            if len(df) < 20:
                return None
            
            # Calculate basic technical features
            features = []
            
            # Price features
            features.append(df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1)  # 20-day return
            features.append(df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1)   # 5-day return
            
            # Moving averages
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_5 = df['Close'].rolling(5).mean().iloc[-1]
            features.append((sma_5 / sma_20) - 1)  # MA ratio
            
            # Volatility
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 1:
                features.append(returns.std())  # Historical volatility
            else:
                features.append(0.02)  # Default volatility
            
            # Volume features
            if 'Volume' in df.columns:
                vol_ma = df['Volume'].rolling(20).mean().iloc[-1]
                current_vol = df['Volume'].iloc[-1]
                features.append(current_vol / vol_ma if vol_ma > 0 else 1.0)
            else:
                features.append(1.0)  # Default volume ratio
            
            return features
            
        except Exception as e:
            logging.warning(f"Feature preparation failed: {e}")
            return None
    
    def should_exit(self, ticker: str, entry_date: datetime, current_date: datetime, 
                   current_price: float, entry_price: float) -> bool:
        """Enhanced exit logic"""
        days_held = (current_date - entry_date).days
        return_pct = (current_price - entry_price) / entry_price
        
        # Exit conditions
        max_holding_period = self.config.get('max_holding_period', 60)
        profit_target = self.config.get('profit_target', 0.20)
        stop_loss = self.config.get('stop_loss', -0.10)
        
        return (days_held >= max_holding_period or 
                return_pct >= profit_target or 
                return_pct <= stop_loss)

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = PortfolioState(cash=config.initial_capital)
        self.trades = []
        self.portfolio_history = []
        
    def run_backtest(self, strategy: Strategy, data: Dict[str, pd.DataFrame], 
                    start_date: datetime, end_date: datetime) -> Dict:
        """Run backtest for given strategy and data"""
        
        try:
            # Initialize
            self.portfolio = PortfolioState(cash=self.config.initial_capital)
            self.portfolio.peak_value = self.config.initial_capital
            self.trades = []
            self.portfolio_history = []
            
            # Get all dates in the range
            all_dates = set()
            for df in data.values():
                dates_in_range = df[(df.index >= start_date) & (df.index <= end_date)].index
                all_dates.update(dates_in_range)
            
            all_dates = sorted(list(all_dates))
            
            if len(all_dates) == 0:
                return {'error': 'No data available for the specified date range'}
            
            # Run backtest day by day
            for date in all_dates:
                self._process_day(strategy, data, date)
                
                # Record portfolio state
                portfolio_value = self._calculate_portfolio_value(data, date)
                self.portfolio.portfolio_value = portfolio_value
                
                # Update drawdown
                if portfolio_value > self.portfolio.peak_value:
                    self.portfolio.peak_value = portfolio_value
                    self.portfolio.drawdown = 0
                else:
                    self.portfolio.drawdown = (portfolio_value - self.portfolio.peak_value) / self.portfolio.peak_value
                
                # Check drawdown limit
                if abs(self.portfolio.drawdown) > self.config.max_drawdown_limit:
                    logging.warning(f"Drawdown limit exceeded on {date}: {self.portfolio.drawdown:.2%}")
                    # Close all positions
                    self._close_all_positions(data, date)
                
                self.portfolio_history.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': self.portfolio.cash,
                    'positions': len(self.portfolio.positions),
                    'drawdown': self.portfolio.drawdown
                })
            
            # Prepare results
            results = self._prepare_results()
            return results
            
        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}")
            return {'error': f'Backtest failed: {str(e)}'}
    
    def _process_day(self, strategy: Strategy, data: Dict[str, pd.DataFrame], date: datetime):
        """Process a single day of backtesting"""
        
        # Check exit conditions for existing positions
        positions_to_close = []
        for ticker, position in self.portfolio.positions.items():
            if ticker in data and date in data[ticker].index:
                current_price = data[ticker].loc[date, 'Close']
                
                if strategy.should_exit(ticker, position['entry_date'], date, 
                                      current_price, position['entry_price']):
                    positions_to_close.append(ticker)
        
        # Close positions
        for ticker in positions_to_close:
            self._close_position(ticker, data, date)
        
        # Generate new signals
        signals = strategy.generate_signals(data, date)
        
        # Process signals for new positions
        for ticker, signal in signals.items():
            if (ticker not in self.portfolio.positions and 
                ticker in data and date in data[ticker].index and
                signal > 0.5):  # Buy signal threshold
                
                self._open_position(ticker, data, date, signal)
    
    def _open_position(self, ticker: str, data: Dict[str, pd.DataFrame], 
                      date: datetime, signal: float):
        """Open a new position"""
        
        if len(self.portfolio.positions) >= self.config.max_positions:
            return
        
        try:
            current_price = data[ticker].loc[date, 'Close']
            
            # Calculate position size
            position_value = self._calculate_position_size(signal)
            
            if position_value < self.config.min_position_size or position_value > self.portfolio.cash:
                return
            
            # Calculate quantity and costs
            quantity = int(position_value / current_price)
            actual_cost = quantity * current_price
            transaction_cost = actual_cost * self.config.transaction_cost_pct
            total_cost = actual_cost + transaction_cost
            
            if total_cost > self.portfolio.cash:
                return
            
            # Update portfolio
            self.portfolio.cash -= total_cost
            self.portfolio.positions[ticker] = {
                'quantity': quantity,
                'entry_price': current_price,
                'entry_date': date,
                'entry_signal': f'ML_Signal_{signal:.3f}',
                'transaction_cost': transaction_cost
            }
            
        except Exception as e:
            logging.warning(f"Failed to open position for {ticker}: {e}")
    
    def _close_position(self, ticker: str, data: Dict[str, pd.DataFrame], date: datetime):
        """Close an existing position"""
        
        if ticker not in self.portfolio.positions:
            return
        
        try:
            position = self.portfolio.positions[ticker]
            current_price = data[ticker].loc[date, 'Close']
            
            # Calculate proceeds and costs
            gross_proceeds = position['quantity'] * current_price
            transaction_cost = gross_proceeds * self.config.transaction_cost_pct
            net_proceeds = gross_proceeds - transaction_cost
            
            # Calculate P&L
            entry_cost = position['quantity'] * position['entry_price']
            gross_pnl = gross_proceeds - entry_cost
            net_pnl = gross_pnl - position['transaction_cost'] - transaction_cost
            return_pct = net_pnl / entry_cost
            
            # Create trade record
            trade = Trade(
                ticker=ticker,
                entry_date=position['entry_date'],
                exit_date=date,
                entry_price=position['entry_price'],
                exit_price=current_price,
                quantity=position['quantity'],
                direction='long',
                entry_signal=position['entry_signal'],
                exit_signal='Exit_Strategy',
                gross_pnl=gross_pnl,
                transaction_costs=position['transaction_cost'] + transaction_cost,
                net_pnl=net_pnl,
                return_pct=return_pct,
                holding_period=(date - position['entry_date']).days
            )
            
            self.trades.append(trade)
            
            # Update portfolio
            self.portfolio.cash += net_proceeds
            del self.portfolio.positions[ticker]
            
        except Exception as e:
            logging.warning(f"Failed to close position for {ticker}: {e}")
    
    def _close_all_positions(self, data: Dict[str, pd.DataFrame], date: datetime):
        """Close all open positions"""
        tickers_to_close = list(self.portfolio.positions.keys())
        for ticker in tickers_to_close:
            if ticker in data and date in data[ticker].index:
                self._close_position(ticker, data, date)
    
    def _calculate_position_size(self, signal: float) -> float:
        """Calculate position size based on configuration"""
        
        if self.config.position_sizing_method == 'equal_weight':
            # Equal weight among maximum positions
            return self.portfolio.cash / self.config.max_positions
        
        elif self.config.position_sizing_method == 'signal_weighted':
            # Weight by signal strength
            base_size = self.portfolio.cash / self.config.max_positions
            return base_size * signal
        
        else:
            return min(self.config.max_position_size, 
                      self.portfolio.cash / self.config.max_positions)
    
    def _calculate_portfolio_value(self, data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total portfolio value"""
        
        total_value = self.portfolio.cash
        
        for ticker, position in self.portfolio.positions.items():
            if ticker in data and date in data[ticker].index:
                current_price = data[ticker].loc[date, 'Close']
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        return total_value
    
    def _prepare_results(self) -> Dict:
        """Prepare backtest results"""
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        if len(portfolio_df) == 0:
            return {'error': 'No portfolio data generated'}
        
        # Calculate performance metrics
        returns = PerformanceMetrics.calculate_returns(portfolio_df['portfolio_value'])
        
        results = {
            'portfolio_values': portfolio_df['portfolio_value'],
            'portfolio_history': portfolio_df,
            'trades': self.trades,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.net_pnl > 0]),
            'losing_trades': len([t for t in self.trades if t.net_pnl < 0]),
            'total_return': (portfolio_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1,
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_df['portfolio_value']),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(portfolio_df['portfolio_value']),
            'var_95': PerformanceMetrics.value_at_risk(returns),
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
            'total_pnl': sum(t.net_pnl for t in self.trades)
        }
        
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades']
            results['avg_win'] = np.mean([t.net_pnl for t in self.trades if t.net_pnl > 0])
            results['avg_loss'] = np.mean([t.net_pnl for t in self.trades if t.net_pnl < 0])
            results['profit_factor'] = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else np.inf
        
        return results

class EnhancedBacktestEngine(BacktestEngine):
    """Enhanced backtesting engine with risk management"""
    
    def __init__(self, config: EnhancedBacktestConfig):
        super().__init__(config)
        self.config = config
        self.risk_events = []
        self.correlation_matrix = None
        
    def run_enhanced_backtest(self, strategy: MLStrategy, data: Dict[str, pd.DataFrame], 
                            start_date: datetime, end_date: datetime) -> Dict:
        """Run enhanced backtest with risk management"""
        try:
            # Run the base backtest
            results = self.run_backtest(strategy, data, start_date, end_date)
            
            if 'error' in results:
                return results
            
            # Add enhanced analytics
            enhanced_results = self._add_enhanced_analytics(results, data)
            enhanced_results['risk_events'] = self.risk_events
            enhanced_results['config'] = self.config
            
            return enhanced_results
            
        except Exception as e:
            return {'error': f'Enhanced backtest failed: {str(e)}'}
    
    def _add_enhanced_analytics(self, results: Dict, data: Dict[str, pd.DataFrame]) -> Dict:
        """Add enhanced analytics to backtest results"""
        enhanced_results = results.copy()
        
        try:
            # Calculate additional risk metrics
            portfolio_values = results.get('portfolio_values', pd.Series())
            
            if not portfolio_values.empty:
                returns = portfolio_values.pct_change().dropna()
                
                # Enhanced metrics
                enhanced_results['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)
                enhanced_results['var_95'] = np.percentile(returns, 5) if len(returns) > 0 else 0
                enhanced_results['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(returns)
                enhanced_results['sortino_ratio'] = self._calculate_sortino_ratio(returns)
                enhanced_results['calmar_ratio'] = self._calculate_calmar_ratio(portfolio_values)
                
                # Risk-adjusted returns
                annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252/len(portfolio_values)) - 1
                enhanced_results['annual_return'] = annual_return
                enhanced_results['volatility'] = returns.std() * np.sqrt(252)
                
        except Exception as e:
            logging.warning(f"Enhanced analytics failed: {e}")
        
        return enhanced_results
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        downside_deviation = negative_returns.std()
        if downside_deviation == 0:
            return 0
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252/len(portfolio_values)) - 1
        max_drawdown = abs(self._calculate_max_drawdown(portfolio_values))
        
        if max_drawdown == 0:
            return np.inf
        
        return annual_return / max_drawdown

class BacktestAnalyzer:
    """Analyze and report backtest results"""
    
    def __init__(self):
        self.results = None
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze backtest results"""
        
        if 'error' in results:
            return results
        
        analysis = {
            'performance_summary': self._create_performance_summary(results),
            'trade_analysis': self._analyze_trades(results.get('trades', [])),
            'risk_analysis': self._analyze_risk(results),
            'monthly_returns': self._calculate_monthly_returns(results)
        }
        
        return analysis
    
    def _create_performance_summary(self, results: Dict) -> Dict:
        """Create performance summary"""
        
        return {
            'Total Return': f"{results.get('total_return', 0):.2%}",
            'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.3f}",
            'Max Drawdown': f"{results.get('max_drawdown', 0):.2%}",
            'Win Rate': f"{results.get('win_rate', 0):.2%}",
            'Total Trades': results.get('total_trades', 0),
            'Profit Factor': f"{results.get('profit_factor', 0):.2f}"
        }
    
    def _analyze_trades(self, trades: List[Trade]) -> Dict:
        """Analyze individual trades"""
        
        if not trades:
            return {'error': 'No trades to analyze'}
        
        returns = [t.return_pct for t in trades]
        holding_periods = [t.holding_period for t in trades]
        
        return {
            'total_trades': len(trades),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_holding_period': np.mean(holding_periods),
            'best_trade': max(returns),
            'worst_trade': min(returns)
        }
    
    def _analyze_risk(self, results: Dict) -> Dict:
        """Analyze risk metrics"""
        
        return {
            'max_drawdown': results.get('max_drawdown', 0),
            'var_95': results.get('var_95', 0),
            'sortino_ratio': results.get('sortino_ratio', 0),
            'calmar_ratio': results.get('calmar_ratio', 0)
        }
    
    def _calculate_monthly_returns(self, results: Dict) -> pd.Series:
        """Calculate monthly returns"""
        
        portfolio_values = results.get('portfolio_values', pd.Series())
        
        if portfolio_values.empty:
            return pd.Series()
        
        monthly_values = portfolio_values.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        return monthly_returns
    
    def create_performance_report(self, results: Dict) -> str:
        """Create a comprehensive performance report"""
        
        if 'error' in results:
            return f"Error in backtest: {results['error']}"
        
        analysis = self.analyze_results(results)
        
        report = f"""
BACKTEST PERFORMANCE REPORT
{'='*50}

PERFORMANCE SUMMARY:
{'-'*20}
"""
        
        for key, value in analysis['performance_summary'].items():
            report += f"{key}: {value}\n"
        
        report += f"""
TRADE ANALYSIS:
{'-'*20}
Total Trades: {analysis['trade_analysis'].get('total_trades', 0)}
Average Return: {analysis['trade_analysis'].get('avg_return', 0):.2%}
Best Trade: {analysis['trade_analysis'].get('best_trade', 0):.2%}
Worst Trade: {analysis['trade_analysis'].get('worst_trade', 0):.2%}
Average Holding Period: {analysis['trade_analysis'].get('avg_holding_period', 0):.1f} days

RISK ANALYSIS:
{'-'*20}
Maximum Drawdown: {analysis['risk_analysis'].get('max_drawdown', 0):.2%}
Value at Risk (95%): {analysis['risk_analysis'].get('var_95', 0):.2%}
Sortino Ratio: {analysis['risk_analysis'].get('sortino_ratio', 0):.3f}
Calmar Ratio: {analysis['risk_analysis'].get('calmar_ratio', 0):.3f}

{'='*50}
"""
        
        return report

class BacktestDB:
    """Database for storing backtest results"""
    
    def __init__(self, db_path: str = 'data/backtests.db'):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure database and tables exist"""
        
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config TEXT,
                    results TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
    
    def save_backtest(self, name: str, results: Dict) -> int:
        """Save backtest results to database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute('''
                INSERT INTO backtests (name, results)
                VALUES (?, ?)
            ''', (name, pickle.dumps(results)))
            
            backtest_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return backtest_id
            
        except Exception as e:
            logging.error(f"Failed to save backtest: {e}")
            return -1
    
    def load_backtest(self, backtest_id: int) -> Dict:
        """Load backtest results from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute('''
                SELECT results FROM backtests WHERE id = ?
            ''', (backtest_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return pickle.loads(row[0])
            else:
                return {'error': 'Backtest not found'}
                
        except Exception as e:
            logging.error(f"Failed to load backtest: {e}")
            return {'error': f'Failed to load backtest: {str(e)}'}
    
    def list_backtests(self) -> pd.DataFrame:
        """List all backtests"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT id, name, created_at FROM backtests
                ORDER BY created_at DESC
            ''', conn)
            
            conn.close()
            return df
            
        except Exception as e:
            logging.error(f"Failed to list backtests: {e}")
            return pd.DataFrame()

# Export all classes
__all__ = [
    'BacktestConfig', 'EnhancedBacktestConfig', 'Trade', 'PortfolioState', 
    'PerformanceMetrics', 'Strategy', 'MLStrategy', 'BacktestEngine', 
    'EnhancedBacktestEngine', 'BacktestAnalyzer', 'BacktestDB'
]

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Backtesting Framework")
    print("="*50)
    
    # Test with mock data
    def create_mock_data():
        """Create mock stock data for testing"""
        dates = pd.date_range('2021-01-01', '2022-12-31', freq='D')
        tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        
        data = {}
        
        for ticker in tickers:
            # Generate realistic stock price data
            np.random.seed(hash(ticker) % 1000)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
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
    config = BacktestConfig()
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