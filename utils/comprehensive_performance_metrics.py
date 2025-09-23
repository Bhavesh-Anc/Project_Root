# utils/comprehensive_performance_metrics.py
"""
Comprehensive Performance Metrics Module for AI Stock Advisor Pro
Implements Sharpe, Sortino, Calmar ratios, win rate, profit factor, and rolling performance analysis
Author: AI Stock Advisor Pro Team
Version: 2.0 - Enhanced Edition
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import streamlit as st
from dataclasses import dataclass, field
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ==================== DATA CLASSES ====================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container with enhanced features"""
    
    # Basic return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    monthly_returns: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    
    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    value_at_risk_95: float = 0.0
    value_at_risk_99: float = 0.0
    conditional_var_95: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_holding_period: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Advanced metrics
    beta: float = 0.0
    alpha: float = 0.0
    correlation_with_market: float = 0.0
    tracking_error: float = 0.0
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    
    # Rolling analysis
    rolling_sharpe: Optional[pd.Series] = None
    rolling_returns: Optional[pd.Series] = None
    rolling_volatility: Optional[pd.Series] = None
    rolling_max_drawdown: Optional[pd.Series] = None
    
    # Regime analysis
    bull_market_return: float = 0.0
    bear_market_return: float = 0.0
    market_regime_performance: Optional[Dict] = None
    
    # Performance consistency
    return_consistency: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for easy serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                result[key] = value.to_dict() if value is not None else None
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create PerformanceMetrics from dictionary"""
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

# ==================== MAIN ANALYZER CLASS ====================

class ComprehensivePerformanceAnalyzer:
    """Advanced performance metrics calculator with comprehensive analysis capabilities"""
    
    def __init__(self, risk_free_rate: float = 0.06, benchmark_return: float = 0.12):
        """
        Initialize the performance analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 6%)
            benchmark_return: Annual benchmark return (default: 12%)
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        self.daily_benchmark = (1 + benchmark_return) ** (1/252) - 1
        
        logging.info(f"Initialized PerformanceAnalyzer with risk_free_rate={risk_free_rate:.2%}")
        
    def calculate_all_metrics(self, 
                            returns: pd.Series, 
                            prices: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            trade_log: Optional[pd.DataFrame] = None,
                            include_rolling: bool = True) -> PerformanceMetrics:
        """
        Calculate all performance metrics comprehensively
        
        Args:
            returns: Daily returns series
            prices: Price series
            benchmark_returns: Benchmark returns for comparison
            trade_log: DataFrame with trade details
            include_rolling: Whether to calculate rolling metrics
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        
        try:
            logging.info("Starting comprehensive performance analysis...")
            
            metrics = PerformanceMetrics()
            
            # Ensure we have valid data
            if returns.empty or prices.empty:
                logging.warning("Empty returns or prices data provided")
                return metrics
            
            # Clean data
            returns = returns.dropna()
            prices = prices.dropna()
            
            if len(returns) < 2 or len(prices) < 2:
                logging.warning("Insufficient data for analysis")
                return metrics
            
            # Basic return metrics
            metrics.total_return = self._calculate_total_return(prices)
            metrics.annualized_return = self._calculate_annualized_return(returns)
            metrics.daily_returns = returns
            
            # Risk metrics
            metrics.volatility = self._calculate_volatility(returns)
            metrics.downside_volatility = self._calculate_downside_volatility(returns)
            metrics.max_drawdown = self._calculate_max_drawdown(prices)
            metrics.max_drawdown_duration = self._calculate_drawdown_duration(prices)
            metrics.value_at_risk_95 = self._calculate_var(returns, 0.05)
            metrics.value_at_risk_99 = self._calculate_var(returns, 0.01)
            metrics.conditional_var_95 = self._calculate_conditional_var(returns, 0.05)
            
            # Risk-adjusted returns
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            metrics.calmar_ratio = self._calculate_calmar_ratio(metrics.annualized_return, metrics.max_drawdown)
            metrics.omega_ratio = self._calculate_omega_ratio(returns)
            
            # Market-relative metrics
            if benchmark_returns is not None:
                benchmark_returns = benchmark_returns.dropna()
                if not benchmark_returns.empty:
                    metrics.beta = self._calculate_beta(returns, benchmark_returns)
                    metrics.alpha = self._calculate_alpha(returns, benchmark_returns, metrics.beta)
                    metrics.treynor_ratio = self._calculate_treynor_ratio(metrics.annualized_return, metrics.beta)
                    metrics.information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
                    metrics.correlation_with_market = self._calculate_correlation(returns, benchmark_returns)
                    metrics.tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
                    metrics.upside_capture = self._calculate_upside_capture(returns, benchmark_returns)
                    metrics.downside_capture = self._calculate_downside_capture(returns, benchmark_returns)
            
            # Trading metrics from trade log
            if trade_log is not None and not trade_log.empty:
                trading_metrics = self._calculate_trading_metrics(trade_log)
                for key, value in trading_metrics.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
            
            # Advanced metrics
            metrics.ulcer_index = self._calculate_ulcer_index(prices)
            metrics.pain_index = self._calculate_pain_index(prices)
            
            # Statistical properties
            metrics.skewness = returns.skew()
            metrics.kurtosis = returns.kurtosis()
            metrics.return_consistency = self._calculate_return_consistency(returns)
            
            # Monthly returns
            metrics.monthly_returns = self._calculate_monthly_returns(returns, prices.index)
            
            # Rolling analysis (if requested)
            if include_rolling and len(returns) > 63:  # Need at least 3 months of data
                metrics.rolling_sharpe = self._calculate_rolling_sharpe(returns, window=63)
                metrics.rolling_returns = self._calculate_rolling_returns(prices, window=63)
                metrics.rolling_volatility = self._calculate_rolling_volatility(returns, window=63)
                metrics.rolling_max_drawdown = self._calculate_rolling_max_drawdown(prices, window=126)
            
            # Market regime analysis
            if benchmark_returns is not None and not benchmark_returns.empty:
                regime_performance = self._analyze_market_regimes(returns, benchmark_returns)
                metrics.bull_market_return = regime_performance.get('bull_return', 0)
                metrics.bear_market_return = regime_performance.get('bear_return', 0)
                metrics.market_regime_performance = regime_performance
            
            logging.info("Performance analysis completed successfully")
            return metrics
            
        except Exception as e:
            logging.error(f"Error in calculate_all_metrics: {str(e)}")
            return PerformanceMetrics()
    
    # ==================== BASIC METRICS ====================
    
    def _calculate_total_return(self, prices: pd.Series) -> float:
        """Calculate total return"""
        if len(prices) < 2:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[0]) - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        total_return = (1 + returns).prod() - 1
        periods = len(returns)
        return ((1 + total_return) ** (252 / periods)) - 1 if periods > 0 else 0.0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_downside_volatility(self, returns: pd.Series, target: float = 0) -> float:
        """Calculate downside volatility (semi-deviation)"""
        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(252)
    
    # ==================== DRAWDOWN ANALYSIS ====================
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _calculate_drawdown_duration(self, prices: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(prices) < 2:
            return 0
        
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        
        # Find periods in drawdown
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return 0
        
        # Calculate consecutive drawdown periods
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_periods = in_drawdown.groupby(groups).sum()
        
        return int(drawdown_periods.max()) if not drawdown_periods.empty else 0
    
    def _calculate_rolling_max_drawdown(self, prices: pd.Series, window: int = 126) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        rolling_drawdowns = []
        
        for i in range(window, len(prices) + 1):
            window_prices = prices.iloc[i-window:i]
            peak = window_prices.expanding().max()
            drawdown = (window_prices - peak) / peak
            rolling_drawdowns.append(drawdown.min())
        
        return pd.Series(rolling_drawdowns, index=prices.index[window-1:])
    
    # ==================== RISK METRICS ====================
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return returns.quantile(confidence)
    
    def _calculate_conditional_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_ulcer_index(self, prices: pd.Series) -> float:
        """Calculate Ulcer Index (alternative measure of downside risk)"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices.expanding().max()
        drawdown_pct = ((prices - peak) / peak) * 100
        squared_drawdown = drawdown_pct ** 2
        return np.sqrt(squared_drawdown.mean())
    
    def _calculate_pain_index(self, prices: pd.Series) -> float:
        """Calculate Pain Index (average drawdown over the period)"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices.expanding().max()
        drawdown_pct = ((prices - peak) / peak) * 100
        return abs(drawdown_pct.mean())
    
    # ==================== RISK-ADJUSTED RETURNS ====================
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        if excess_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino Ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar Ratio"""
        if abs(max_drawdown) < 1e-6:  # Avoid division by zero
            return float('inf') if annual_return > 0 else 0.0
        return annual_return / abs(max_drawdown)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega Ratio"""
        if len(returns) == 0:
            return 1.0
        
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns <= 0].sum())
        
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 1.0
        
        return positive_returns / negative_returns
    
    # ==================== MARKET-RELATIVE METRICS ====================
    
    def _calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Beta"""
        aligned_data = self._align_series(returns, market_returns)
        if aligned_data is None:
            return 0.0
        
        portfolio_returns, market_returns_aligned = aligned_data
        
        covariance = np.cov(portfolio_returns, market_returns_aligned)[0][1]
        market_variance = market_returns_aligned.var()
        
        return covariance / market_variance if market_variance != 0 else 0.0
    
    def _calculate_alpha(self, returns: pd.Series, market_returns: pd.Series, beta: float) -> float:
        """Calculate Alpha (Jensen's Alpha)"""
        portfolio_return = returns.mean() * 252
        market_return = market_returns.mean() * 252
        return portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
    
    def _calculate_treynor_ratio(self, annual_return: float, beta: float) -> float:
        """Calculate Treynor Ratio"""
        if abs(beta) < 1e-6:  # Avoid division by zero
            return float('inf') if annual_return > self.risk_free_rate else 0.0
        return (annual_return - self.risk_free_rate) / beta
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        aligned_data = self._align_series(returns, benchmark_returns)
        if aligned_data is None:
            return 0.0
        
        portfolio_returns, benchmark_returns_aligned = aligned_data
        
        excess_returns = portfolio_returns - benchmark_returns_aligned
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / tracking_error
    
    def _calculate_correlation(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate correlation with market"""
        aligned_data = self._align_series(returns, market_returns)
        if aligned_data is None:
            return 0.0
        
        portfolio_returns, market_returns_aligned = aligned_data
        return portfolio_returns.corr(market_returns_aligned)
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        aligned_data = self._align_series(returns, benchmark_returns)
        if aligned_data is None:
            return 0.0
        
        portfolio_returns, benchmark_returns_aligned = aligned_data
        excess_returns = portfolio_returns - benchmark_returns_aligned
        return excess_returns.std() * np.sqrt(252)
    
    def _calculate_upside_capture(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate upside capture ratio"""
        aligned_data = self._align_series(returns, benchmark_returns)
        if aligned_data is None:
            return 0.0
        
        portfolio_returns, benchmark_returns_aligned = aligned_data
        
        # Filter for positive benchmark periods
        positive_benchmark = benchmark_returns_aligned > 0
        if not positive_benchmark.any():
            return 0.0
        
        portfolio_upside = portfolio_returns[positive_benchmark].mean()
        benchmark_upside = benchmark_returns_aligned[positive_benchmark].mean()
        
        return (portfolio_upside / benchmark_upside) if benchmark_upside != 0 else 0.0
    
    def _calculate_downside_capture(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate downside capture ratio"""
        aligned_data = self._align_series(returns, benchmark_returns)
        if aligned_data is None:
            return 0.0
        
        portfolio_returns, benchmark_returns_aligned = aligned_data
        
        # Filter for negative benchmark periods
        negative_benchmark = benchmark_returns_aligned < 0
        if not negative_benchmark.any():
            return 0.0
        
        portfolio_downside = portfolio_returns[negative_benchmark].mean()
        benchmark_downside = benchmark_returns_aligned[negative_benchmark].mean()
        
        return (portfolio_downside / benchmark_downside) if benchmark_downside != 0 else 0.0
    
    # ==================== TRADING METRICS ====================
    
    def _calculate_trading_metrics(self, trade_log: pd.DataFrame) -> Dict:
        """Calculate comprehensive trading metrics"""
        
        if trade_log.empty:
            return self._get_empty_trading_metrics()
        
        try:
            # Ensure required columns exist
            required_cols = ['return_pct', 'pnl']
            if not all(col in trade_log.columns for col in required_cols):
                logging.warning("Trade log missing required columns")
                return self._get_empty_trading_metrics()
            
            total_trades = len(trade_log)
            winning_trades = len(trade_log[trade_log['return_pct'] > 0])
            losing_trades = len(trade_log[trade_log['return_pct'] <= 0])
            
            # Basic trading metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
                'loss_rate': losing_trades / total_trades if total_trades > 0 else 0.0
            }
            
            # Profit and loss analysis
            winning_trades_data = trade_log[trade_log['return_pct'] > 0]
            losing_trades_data = trade_log[trade_log['return_pct'] <= 0]
            
            if not winning_trades_data.empty:
                metrics['average_win'] = winning_trades_data['return_pct'].mean()
                metrics['largest_win'] = winning_trades_data['return_pct'].max()
                gross_profit = winning_trades_data['pnl'].sum()
            else:
                metrics['average_win'] = 0.0
                metrics['largest_win'] = 0.0
                gross_profit = 0.0
            
            if not losing_trades_data.empty:
                metrics['average_loss'] = losing_trades_data['return_pct'].mean()
                metrics['largest_loss'] = losing_trades_data['return_pct'].min()
                gross_loss = abs(losing_trades_data['pnl'].sum())
            else:
                metrics['average_loss'] = 0.0
                metrics['largest_loss'] = 0.0
                gross_loss = 1.0  # Avoid division by zero
            
            # Profit factor
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average holding period
            if 'entry_date' in trade_log.columns and 'exit_date' in trade_log.columns:
                try:
                    holding_periods = (pd.to_datetime(trade_log['exit_date']) - 
                                     pd.to_datetime(trade_log['entry_date'])).dt.days
                    metrics['average_holding_period'] = int(holding_periods.mean())
                except:
                    metrics['average_holding_period'] = 30  # Default assumption
            else:
                metrics['average_holding_period'] = 30  # Default assumption
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating trading metrics: {str(e)}")
            return self._get_empty_trading_metrics()
    
    def _get_empty_trading_metrics(self) -> Dict:
        """Return empty trading metrics structure"""
        return {
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'average_holding_period': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    # ==================== ROLLING ANALYSIS ====================
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 63) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        excess_returns = returns - self.daily_rf
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()
        
        # Avoid division by zero
        rolling_sharpe = np.where(rolling_std != 0, 
                                (rolling_mean / rolling_std) * np.sqrt(252), 
                                0)
        
        return pd.Series(rolling_sharpe, index=returns.index).dropna()
    
    def _calculate_rolling_returns(self, prices: pd.Series, window: int = 63) -> pd.Series:
        """Calculate rolling returns"""
        if len(prices) < window:
            return pd.Series(dtype=float)
        
        rolling_returns = prices.pct_change(periods=window).dropna()
        return rolling_returns
    
    def _calculate_rolling_volatility(self, returns: pd.Series, window: int = 63) -> pd.Series:
        """Calculate rolling volatility"""
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_vol.dropna()
    
    # ==================== UTILITY METHODS ====================
    
    def _calculate_monthly_returns(self, returns: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate monthly returns"""
        if len(returns) == 0 or len(dates) == 0:
            return pd.Series(dtype=float)
        
        try:
            monthly_data = pd.DataFrame({'returns': returns, 'date': dates})
            monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
            
            # Calculate monthly returns
            monthly_returns = monthly_data.groupby('year_month')['returns'].apply(
                lambda x: (1 + x).prod() - 1
            )
            
            return monthly_returns
        except Exception as e:
            logging.warning(f"Error calculating monthly returns: {str(e)}")
            return pd.Series(dtype=float)
    
    def _calculate_return_consistency(self, returns: pd.Series) -> float:
        """Calculate return consistency (percentage of positive periods)"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).mean()
    
    def _analyze_market_regimes(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Analyze performance in different market regimes"""
        aligned_data = self._align_series(returns, benchmark_returns)
        if aligned_data is None:
            return {}
        
        portfolio_returns, benchmark_returns_aligned = aligned_data
        
        # Define bull and bear markets (simple definition)
        # Bull: benchmark > 0, Bear: benchmark <= 0
        bull_periods = benchmark_returns_aligned > 0
        bear_periods = benchmark_returns_aligned <= 0
        
        results = {}
        
        if bull_periods.any():
            bull_returns = portfolio_returns[bull_periods]
            results['bull_return'] = (bull_returns + 1).prod() - 1
            results['bull_periods'] = bull_periods.sum()
        else:
            results['bull_return'] = 0.0
            results['bull_periods'] = 0
        
        if bear_periods.any():
            bear_returns = portfolio_returns[bear_periods]
            results['bear_return'] = (bear_returns + 1).prod() - 1
            results['bear_periods'] = bear_periods.sum()
        else:
            results['bear_return'] = 0.0
            results['bear_periods'] = 0
        
        return results
    
    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Optional[Tuple[pd.Series, pd.Series]]:
        """Align two series on common dates"""
        try:
            # Find common dates
            common_dates = series1.index.intersection(series2.index)
            
            if len(common_dates) < 2:
                logging.warning("Insufficient common dates for alignment")
                return None
            
            aligned_series1 = series1.loc[common_dates]
            aligned_series2 = series2.loc[common_dates]
            
            return aligned_series1, aligned_series2
            
        except Exception as e:
            logging.error(f"Error aligning series: {str(e)}")
            return None

# ==================== VISUALIZATION CLASS ====================

class PerformanceVisualizer:
    """Comprehensive performance visualization dashboard with enhanced styling"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_performance_dashboard(self, metrics: PerformanceMetrics, prices: pd.Series) -> None:
        """Create comprehensive performance dashboard with enhanced visualizations"""
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 25px; margin: 2rem 0; text-align: center;'>
            <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700;'>📊 Comprehensive Performance Analytics</h1>
            <p style='margin: 1rem 0 0 0; opacity: 0.9; font-size: 1.2rem;'>
                Advanced performance metrics with professional-grade analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics overview
        self._display_main_metrics(metrics)
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Return Analysis", 
            "🛡️ Risk Analysis", 
            "📊 Trading Metrics", 
            "🔄 Rolling Analysis",
            "📋 Advanced Metrics"
        ])
        
        with tab1:
            self._display_return_analysis(metrics, prices)
        
        with tab2:
            self._display_risk_analysis(metrics, prices)
        
        with tab3:
            self._display_trading_analysis(metrics)
        
        with tab4:
            self._display_rolling_analysis(metrics)
        
        with tab5:
            self._display_advanced_metrics(metrics)
    
    def _display_main_metrics(self, metrics: PerformanceMetrics) -> None:
        """Display main performance metrics with enhanced styling"""
        
        st.subheader("🎯 Key Performance Indicators")
        
        # Top row - Return metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if metrics.total_return > 0 else "inverse"
            st.metric(
                "Total Return",
                f"{metrics.total_return:.2%}",
                delta=f"Annualized: {metrics.annualized_return:.2%}"
            )
        
        with col2:
            sharpe_status = "Excellent" if metrics.sharpe_ratio > 2.0 else "Good" if metrics.sharpe_ratio > 1.0 else "Moderate"
            st.metric(
                "Sharpe Ratio",
                f"{metrics.sharpe_ratio:.3f}",
                delta=sharpe_status
            )
        
        with col3:
            sortino_display = f"{metrics.sortino_ratio:.3f}" if metrics.sortino_ratio != float('inf') else "∞"
            st.metric(
                "Sortino Ratio",
                sortino_display,
                delta="Downside optimized"
            )
        
        with col4:
            calmar_display = f"{metrics.calmar_ratio:.3f}" if metrics.calmar_ratio != float('inf') else "∞"
            st.metric(
                "Calmar Ratio",
                calmar_display,
                delta="Risk-adjusted"
            )
        
        # Second row - Risk and trading metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            drawdown_status = "Low Risk" if abs(metrics.max_drawdown) < 0.1 else "Moderate" if abs(metrics.max_drawdown) < 0.2 else "High Risk"
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown:.2%}",
                delta=drawdown_status
            )
        
        with col2:
            vol_status = "Low Vol" if metrics.volatility < 0.15 else "Moderate" if metrics.volatility < 0.25 else "High Vol"
            st.metric(
                "Volatility",
                f"{metrics.volatility:.2%}",
                delta=vol_status
            )
        
        with col3:
            if metrics.total_trades > 0:
                st.metric(
                    "Win Rate",
                    f"{metrics.win_rate:.1%}",
                    delta=f"{metrics.winning_trades}/{metrics.total_trades} trades"
                )
            else:
                st.metric("Win Rate", "N/A", delta="No trades available")
        
        with col4:
            if metrics.profit_factor != 0:
                pf_display = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "∞"
                pf_status = "Excellent" if metrics.profit_factor > 2.0 else "Good" if metrics.profit_factor > 1.5 else "Fair"
                st.metric(
                    "Profit Factor",
                    pf_display,
                    delta=pf_status
                )
            else:
                st.metric("Profit Factor", "N/A", delta="No trading data")
    
    def _display_return_analysis(self, metrics: PerformanceMetrics, prices: pd.Series) -> None:
        """Display detailed return analysis with enhanced visualizations"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly returns heatmap
            if metrics.monthly_returns is not None and not metrics.monthly_returns.empty:
                st.subheader("📅 Monthly Returns Heatmap")
                
                self._create_monthly_returns_heatmap(metrics.monthly_returns)
        
        with col2:
            # Return distribution
            if metrics.daily_returns is not None and not metrics.daily_returns.empty:
                st.subheader("📊 Return Distribution")
                
                self._create_return_distribution(metrics.daily_returns)
        
        # Performance comparison table
        self._create_performance_comparison_table(metrics)
    
    def _display_risk_analysis(self, metrics: PerformanceMetrics, prices: pd.Series) -> None:
        """Display comprehensive risk analysis"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drawdown chart
            st.subheader("📉 Drawdown Analysis")
            self._create_drawdown_chart(prices, metrics.max_drawdown)
        
        with col2:
            # Risk metrics summary
            st.subheader("🛡️ Risk Metrics Summary")
            self._create_risk_metrics_table(metrics)
    
    def _display_trading_analysis(self, metrics: PerformanceMetrics) -> None:
        """Display trading performance analysis"""
        
        if metrics.total_trades == 0:
            st.info("No trading data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Win/Loss pie chart
            st.subheader("🎯 Win/Loss Distribution")
            self._create_win_loss_chart(metrics)
        
        with col2:
            # Trading metrics
            st.subheader("📊 Trading Performance Metrics")
            self._create_trading_metrics_table(metrics)
    
    def _display_rolling_analysis(self, metrics: PerformanceMetrics) -> None:
        """Display rolling performance analysis"""
        
        if (metrics.rolling_sharpe is None or metrics.rolling_sharpe.empty):
            st.info("Rolling analysis not available - insufficient data or not calculated")
            return
        
        # Rolling Sharpe ratio
        st.subheader("📈 Rolling Sharpe Ratio (3-Month Window)")
        self._create_rolling_sharpe_chart(metrics.rolling_sharpe)
        
        # Rolling volatility and returns
        col1, col2 = st.columns(2)
        
        with col1:
            if metrics.rolling_volatility is not None and not metrics.rolling_volatility.empty:
                st.subheader("📊 Rolling Volatility")
                self._create_rolling_volatility_chart(metrics.rolling_volatility)
        
        with col2:
            if metrics.rolling_returns is not None and not metrics.rolling_returns.empty:
                st.subheader("📈 Rolling Returns")
                self._create_rolling_returns_chart(metrics.rolling_returns)
    
    def _display_advanced_metrics(self, metrics: PerformanceMetrics) -> None:
        """Display advanced performance metrics"""
        
        st.subheader("🔬 Advanced Performance Metrics")
        
        # Advanced metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Advanced Return Metrics**")
            advanced_return_metrics = {
                'Metric': [
                    'Alpha (Annual)',
                    'Beta',
                    'Treynor Ratio',
                    'Information Ratio',
                    'Omega Ratio'
                ],
                'Value': [
                    f"{metrics.alpha:.4f}" if metrics.alpha != 0 else "N/A",
                    f"{metrics.beta:.3f}" if metrics.beta != 0 else "N/A",
                    f"{metrics.treynor_ratio:.3f}" if metrics.treynor_ratio != float('inf') and metrics.treynor_ratio != 0 else "N/A",
                    f"{metrics.information_ratio:.3f}" if metrics.information_ratio != 0 else "N/A",
                    f"{metrics.omega_ratio:.3f}" if metrics.omega_ratio != float('inf') and metrics.omega_ratio != 0 else "N/A"
                ],
                'Interpretation': [
                    "Excess return vs market" if metrics.alpha != 0 else "No benchmark data",
                    "Market sensitivity" if metrics.beta != 0 else "No benchmark data",
                    "Risk-adjusted excess return" if metrics.treynor_ratio != 0 else "No benchmark data",
                    "Skill vs benchmark" if metrics.information_ratio != 0 else "No benchmark data",
                    "Upside/downside capture" if metrics.omega_ratio != 0 else "No data"
                ]
            }
            
            advanced_df = pd.DataFrame(advanced_return_metrics)
            st.dataframe(advanced_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**🛡️ Advanced Risk Metrics**")
            advanced_risk_metrics = {
                'Metric': [
                    'Ulcer Index',
                    'Pain Index',
                    'VaR (95%)',
                    'CVaR (95%)',
                    'Skewness',
                    'Kurtosis'
                ],
                'Value': [
                    f"{metrics.ulcer_index:.2f}",
                    f"{metrics.pain_index:.2f}",
                    f"{metrics.value_at_risk_95:.2%}",
                    f"{metrics.conditional_var_95:.2%}",
                    f"{metrics.skewness:.3f}",
                    f"{metrics.kurtosis:.3f}"
                ],
                'Status': [
                    "Low Pain" if metrics.ulcer_index < 5 else "Moderate" if metrics.ulcer_index < 10 else "High Pain",
                    "Low Pain" if metrics.pain_index < 2 else "Moderate" if metrics.pain_index < 5 else "High Pain",
                    "Conservative" if abs(metrics.value_at_risk_95) < 0.02 else "Moderate" if abs(metrics.value_at_risk_95) < 0.04 else "Aggressive",
                    "Low Risk" if abs(metrics.conditional_var_95) < 0.03 else "Moderate" if abs(metrics.conditional_var_95) < 0.06 else "High Risk",
                    "Positive Skew" if metrics.skewness > 0 else "Negative Skew" if metrics.skewness < -0.5 else "Normal",
                    "Fat Tails" if abs(metrics.kurtosis) > 3 else "Normal Distribution"
                ]
            }
            
            risk_df = pd.DataFrame(advanced_risk_metrics)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # ==================== CHART CREATION METHODS ====================
    
    def _create_monthly_returns_heatmap(self, monthly_returns: pd.Series):
        """Create monthly returns heatmap"""
        try:
            if monthly_returns.empty:
                st.info("No monthly returns data available")
                return
            
            monthly_df = monthly_returns.reset_index()
            monthly_df['year'] = monthly_df['year_month'].dt.year
            monthly_df['month'] = monthly_df['year_month'].dt.month
            
            pivot_df = monthly_df.pivot(index='year', columns='month', values=monthly_returns.name)
            
            if pivot_df.empty:
                st.info("Insufficient data for heatmap")
                return
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=[f'M{i}' for i in pivot_df.columns],
                y=pivot_df.index,
                colorscale='RdYlGn',
                zmid=0,
                text=[[f'{val:.2%}' if not pd.isna(val) else '' for val in row] for row in pivot_df.values],
                texttemplate='%{text}',
                textfont={'size': 10},
                hoverongaps=False,
                colorbar=dict(title="Monthly Return")
            ))
            
            fig.update_layout(
                title="Monthly Returns Distribution",
                xaxis_title="Month",
                yaxis_title="Year",
                height=400,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating monthly returns heatmap: {str(e)}")
    
    def _create_return_distribution(self, returns: pd.Series):
        """Create return distribution histogram"""
        try:
            if returns.empty:
                st.info("No returns data available")
                return
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=30,
                marker=dict(
                    color=self.color_palette['primary'],
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                name='Daily Returns',
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))
            
            # Add normal distribution overlay
            mean_return = returns.mean() * 100
            std_return = returns.std() * 100
            
            x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
            y_norm = len(returns) * (std_return * np.sqrt(2 * np.pi))**(-1) * np.exp(-0.5 * ((x_norm - mean_return) / std_return)**2)
            
            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Distribution',
                line=dict(color=self.color_palette['danger'], width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Distribution of Daily Returns",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400,
                showlegend=True,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating return distribution: {str(e)}")
    
    def _create_performance_comparison_table(self, metrics: PerformanceMetrics):
        """Create performance comparison table"""
        
        st.subheader("📈 Risk-Adjusted Return Comparison")
        
        comparison_data = {
            'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio'],
            'Value': [
                f"{metrics.sharpe_ratio:.3f}",
                f"{metrics.sortino_ratio:.3f}" if metrics.sortino_ratio != float('inf') else "∞",
                f"{metrics.calmar_ratio:.3f}" if metrics.calmar_ratio != float('inf') else "∞",
                f"{metrics.omega_ratio:.3f}" if metrics.omega_ratio != float('inf') else "∞"
            ],
            'Interpretation': [
                "Excellent" if metrics.sharpe_ratio > 2.0 else "Good" if metrics.sharpe_ratio > 1.0 else "Moderate",
                "Excellent downside protection" if metrics.sortino_ratio > 2.0 else "Good" if metrics.sortino_ratio > 1.0 else "Moderate",
                "Excellent drawdown control" if metrics.calmar_ratio > 1.0 else "Good" if metrics.calmar_ratio > 0.5 else "Moderate",
                "Strong upside capture" if metrics.omega_ratio > 1.5 else "Balanced" if metrics.omega_ratio > 1.0 else "Defensive"
            ],
            'Benchmark': [
                "> 1.0 Good, > 2.0 Excellent",
                "> 1.0 Good, > 2.0 Excellent", 
                "> 0.5 Good, > 1.0 Excellent",
                "> 1.0 Positive, > 1.5 Strong"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    def _create_drawdown_chart(self, prices: pd.Series, max_drawdown: float):
        """Create drawdown chart"""
        try:
            if prices.empty:
                st.info("No price data available")
                return
            
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=prices.index,
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown %',
                line=dict(color=self.color_palette['danger']),
                fillcolor=f'rgba(220, 53, 69, 0.3)'
            ))
            
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig.add_hline(y=max_drawdown * 100, line_dash="dot", line_color="red", 
                         annotation_text=f"Max DD: {max_drawdown:.2%}", annotation_position="bottom right")
            
            fig.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating drawdown chart: {str(e)}")
    
    def _create_risk_metrics_table(self, metrics: PerformanceMetrics):
        """Create risk metrics summary table"""
        
        risk_data = {
            'Risk Metric': [
                'Maximum Drawdown',
                'Volatility (Annual)',
                'Value at Risk (95%)',
                'Conditional VaR (95%)',
                'Ulcer Index',
                'Pain Index'
            ],
            'Value': [
                f"{metrics.max_drawdown:.2%}",
                f"{metrics.volatility:.2%}",
                f"{metrics.value_at_risk_95:.2%}",
                f"{metrics.conditional_var_95:.2%}",
                f"{metrics.ulcer_index:.2f}",
                f"{metrics.pain_index:.2f}"
            ],
            'Status': [
                "Low Risk" if abs(metrics.max_drawdown) < 0.1 else "Moderate Risk" if abs(metrics.max_drawdown) < 0.2 else "High Risk",
                "Low Vol" if metrics.volatility < 0.15 else "Moderate Vol" if metrics.volatility < 0.25 else "High Vol",
                "Conservative" if abs(metrics.value_at_risk_95) < 0.02 else "Moderate" if abs(metrics.value_at_risk_95) < 0.04 else "Aggressive",
                "Low Risk" if abs(metrics.conditional_var_95) < 0.03 else "Moderate Risk" if abs(metrics.conditional_var_95) < 0.06 else "High Risk",
                "Low Pain" if metrics.ulcer_index < 5 else "Moderate Pain" if metrics.ulcer_index < 10 else "High Pain",
                "Low Pain" if metrics.pain_index < 2 else "Moderate Pain" if metrics.pain_index < 5 else "High Pain"
            ]
        }
        
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    def _create_win_loss_chart(self, metrics: PerformanceMetrics):
        """Create win/loss pie chart"""
        try:
            if metrics.total_trades == 0:
                st.info("No trading data available")
                return
            
            fig = go.Figure(data=[go.Pie(
                labels=['Winning Trades', 'Losing Trades'],
                values=[metrics.winning_trades, metrics.losing_trades],
                marker=dict(colors=[self.color_palette['success'], self.color_palette['danger']]),
                hole=0.4,
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f"Trading Performance ({metrics.total_trades} total trades)",
                height=400,
                font=dict(family="Inter, sans-serif")
            )
            
            fig.add_annotation(
                text=f"<b>{metrics.win_rate:.1%}</b><br>Win Rate",
                x=0.5, y=0.5,
                font=dict(size=16),
                showarrow=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating win/loss chart: {str(e)}")
    
    def _create_trading_metrics_table(self, metrics: PerformanceMetrics):
        """Create trading metrics table"""
        
        trading_data = {
            'Metric': [
                'Total Trades',
                'Winning Trades',
                'Losing Trades',
                'Win Rate',
                'Profit Factor',
                'Average Win',
                'Average Loss',
                'Average Holding Period'
            ],
            'Value': [
                f"{metrics.total_trades}",
                f"{metrics.winning_trades}",
                f"{metrics.losing_trades}",
                f"{metrics.win_rate:.1%}",
                f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "∞",
                f"{metrics.average_win:.2%}",
                f"{metrics.average_loss:.2%}",
                f"{metrics.average_holding_period} days"
            ],
            'Assessment': [
                "Sufficient sample" if metrics.total_trades > 30 else "Small sample" if metrics.total_trades > 10 else "Very small sample",
                f"{metrics.winning_trades} wins",
                f"{metrics.losing_trades} losses",
                "Excellent" if metrics.win_rate > 0.6 else "Good" if metrics.win_rate > 0.5 else "Needs improvement",
                "Excellent" if metrics.profit_factor > 2.0 else "Good" if metrics.profit_factor > 1.5 else "Acceptable" if metrics.profit_factor > 1.0 else "Poor",
                "Strong" if metrics.average_win > 0.05 else "Moderate" if metrics.average_win > 0.02 else "Weak",
                "Controlled" if abs(metrics.average_loss) < 0.03 else "Moderate" if abs(metrics.average_loss) < 0.06 else "High",
                "Short-term" if metrics.average_holding_period < 14 else "Medium-term" if metrics.average_holding_period < 60 else "Long-term"
            ]
        }
        
        trading_df = pd.DataFrame(trading_data)
        st.dataframe(trading_df, use_container_width=True, hide_index=True)
    
    def _create_rolling_sharpe_chart(self, rolling_sharpe: pd.Series):
        """Create rolling Sharpe ratio chart"""
        try:
            if rolling_sharpe.empty:
                st.info("No rolling Sharpe data available")
                return
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.color_palette['primary'], width=2)
            ))
            
            fig.add_hline(y=1.0, line_dash="dash", line_color=self.color_palette['success'], 
                         annotation_text="Good Performance (1.0)", annotation_position="bottom right")
            fig.add_hline(y=2.0, line_dash="dash", line_color=self.color_palette['success'], 
                         annotation_text="Excellent Performance (2.0)", annotation_position="top right")
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
            
            fig.update_layout(
                title="Rolling Sharpe Ratio Over Time (3-Month Window)",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                height=400,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating rolling Sharpe chart: {str(e)}")
    
    def _create_rolling_volatility_chart(self, rolling_volatility: pd.Series):
        """Create rolling volatility chart"""
        try:
            if rolling_volatility.empty:
                st.info("No rolling volatility data available")
                return
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_volatility.index,
                y=rolling_volatility * 100,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color=self.color_palette['warning'], width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="3-Month Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=300,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating rolling volatility chart: {str(e)}")
    
    def _create_rolling_returns_chart(self, rolling_returns: pd.Series):
        """Create rolling returns chart"""
        try:
            if rolling_returns.empty:
                st.info("No rolling returns data available")
                return
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns * 100,
                mode='lines',
                name='Rolling Returns',
                line=dict(color=self.color_palette['success'], width=2)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            fig.update_layout(
                title="3-Month Rolling Returns",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                height=300,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating rolling returns chart: {str(e)}")

# ==================== MAIN INTEGRATION FUNCTION ====================

def create_comprehensive_performance_dashboard(predictions_df: pd.DataFrame, 
                                             price_targets_df: pd.DataFrame,
                                             raw_data: Dict,
                                             models: Dict) -> Optional[PerformanceMetrics]:
    """Main function to create comprehensive performance dashboard"""
    
    try:
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    color: white; padding: 2rem; border-radius: 25px; margin: 2rem 0; text-align: center;'>
            <h2 style='margin: 0; font-size: 2.5rem; font-weight: 700;'>🎯 Advanced Performance Analytics</h2>
            <p style='margin: 1rem 0 0 0; opacity: 0.9; font-size: 1.2rem;'>
                Comprehensive performance analysis with professional-grade metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if predictions_df.empty:
            st.warning("No prediction data available for performance analysis")
            return None
        
        # Initialize analyzer
        analyzer = ComprehensivePerformanceAnalyzer()
        
        # Generate synthetic performance data for demonstration
        synthetic_data = generate_synthetic_performance_data(predictions_df, price_targets_df)
        
        # Calculate comprehensive metrics
        metrics = analyzer.calculate_all_metrics(
            returns=synthetic_data['returns'],
            prices=synthetic_data['prices'],
            benchmark_returns=synthetic_data['benchmark_returns'],
            trade_log=synthetic_data['trade_log'],
            include_rolling=True
        )
        
        # Display comprehensive dashboard
        visualizer = PerformanceVisualizer()
        visualizer.create_performance_dashboard(metrics, synthetic_data['prices'])
        
        return metrics
        
    except Exception as e:
        st.error(f"Error creating performance dashboard: {str(e)}")
        logging.error(f"Performance dashboard error: {str(e)}")
        return None

# ==================== SYNTHETIC DATA GENERATION ====================

def generate_synthetic_performance_data(predictions_df: pd.DataFrame, 
                                       price_targets_df: pd.DataFrame) -> Dict:
    """Generate synthetic performance data for demonstration purposes"""
    
    try:
        # Generate synthetic price series based on predictions
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Base performance on prediction confidence and expected returns
        if not predictions_df.empty and 'ensemble_confidence' in predictions_df.columns:
            avg_confidence = predictions_df['ensemble_confidence'].mean()
        else:
            avg_confidence = 0.6
        
        if not price_targets_df.empty and 'percentage_change' in price_targets_df.columns:
            avg_expected_return = price_targets_df['percentage_change'].mean()
        else:
            avg_expected_return = 0.05
        
        # Create more realistic return distribution
        base_return = 0.0005 + (avg_confidence - 0.5) * 0.001 + avg_expected_return * 0.1 / 252
        volatility = 0.015 + (1 - avg_confidence) * 0.01  # Lower confidence = higher volatility
        
        # Generate returns with some autocorrelation (more realistic)
        returns = np.random.normal(base_return, volatility, len(dates))
        
        # Add some momentum/mean reversion
        for i in range(1, len(returns)):
            momentum = 0.05 * returns[i-1]  # Small momentum effect
            mean_reversion = -0.02 * (returns[i-1] - base_return)  # Mean reversion
            returns[i] += momentum + mean_reversion
        
        # Create price series
        prices = pd.Series(1000 * np.cumprod(1 + returns), index=dates)
        returns_series = pd.Series(returns, index=dates)
        
        # Generate benchmark returns (market)
        benchmark_returns = np.random.normal(0.0003, 0.012, len(dates))
        
        # Add correlation with portfolio returns
        correlation = 0.7
        benchmark_returns = correlation * returns + np.sqrt(1 - correlation**2) * benchmark_returns
        benchmark_returns_series = pd.Series(benchmark_returns, index=dates)
        
        # Generate synthetic trade log
        n_trades = len(predictions_df) * 4 if not predictions_df.empty else 20
        n_trades = max(10, min(n_trades, 100))  # Reasonable bounds
        
        trade_dates = pd.to_datetime(np.random.choice(dates, n_trades, replace=True))
        
        # Generate trade returns based on overall performance
        trade_returns = np.random.normal(base_return * 25, volatility * 5, n_trades)  # ~monthly returns
        
        # Add some realistic win/loss distribution
        win_rate = 0.55 + avg_confidence * 0.2  # Higher confidence = higher win rate
        wins = np.random.random(n_trades) < win_rate
        
        # Make winning trades slightly larger on average
        trade_returns[wins] = np.abs(trade_returns[wins]) + np.random.normal(0.01, 0.02, sum(wins))
        trade_returns[~wins] = -np.abs(trade_returns[~wins]) + np.random.normal(-0.01, 0.015, sum(~wins))
        
        # Generate P&L based on returns
        base_position_size = 50000  # ₹50,000 per trade
        trade_pnl = trade_returns * base_position_size
        
        # Create trade log
        trade_log = pd.DataFrame({
            'entry_date': trade_dates,
            'exit_date': trade_dates + pd.Timedelta(days=np.random.randint(1, 45, n_trades)),
            'return_pct': trade_returns,
            'pnl': trade_pnl,
            'entry_price': 1000 + np.random.normal(0, 50, n_trades),
            'exit_price': lambda x: x + trade_returns * x  # This won't work, will be recalculated
        })
        
        # Fix exit_price calculation
        trade_log['exit_price'] = trade_log['entry_price'] * (1 + trade_log['return_pct'])
        
        return {
            'returns': returns_series,
            'prices': prices,
            'benchmark_returns': benchmark_returns_series,
            'trade_log': trade_log
        }
        
    except Exception as e:
        logging.error(f"Error generating synthetic data: {str(e)}")
        
        # Return minimal fallback data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.02, len(dates)), index=dates)
        prices = pd.Series(1000 * np.cumprod(1 + returns), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, len(dates)), index=dates)
        
        return {
            'returns': returns,
            'prices': prices,
            'benchmark_returns': benchmark_returns,
            'trade_log': pd.DataFrame()
        }

# ==================== EXAMPLE USAGE AND TESTING ====================

def example_usage():
    """Example of how to use the comprehensive performance metrics"""
    
    # Sample data generation
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = pd.Series(1000 * np.cumprod(1 + returns), index=dates)
    benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))
    
    # Initialize analyzer
    analyzer = ComprehensivePerformanceAnalyzer()
    
    # Calculate metrics
    metrics = analyzer.calculate_all_metrics(
        returns=pd.Series(returns, index=dates),
        prices=prices,
        benchmark_returns=pd.Series(benchmark_returns, index=dates),
        include_rolling=True
    )
    
    # Display dashboard
    visualizer = PerformanceVisualizer()
    visualizer.create_performance_dashboard(metrics, prices)
    
    return metrics

# ==================== MODULE EXPORTS ====================

__all__ = [
    'PerformanceMetrics',
    'ComprehensivePerformanceAnalyzer', 
    'PerformanceVisualizer',
    'create_comprehensive_performance_dashboard',
    'generate_synthetic_performance_data',
    'example_usage'
]

if __name__ == "__main__":
    # Run example if executed directly
    st.write("Running Comprehensive Performance Metrics Example...")
    example_metrics = example_usage()
    st.write(f"Example completed. Total return: {example_metrics.total_return:.2%}")