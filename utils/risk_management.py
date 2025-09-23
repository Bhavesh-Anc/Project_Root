# utils/risk_management.py - Complete Risk Management System
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import sqlite3
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_correlation: float = 0.7      # Maximum correlation between positions
    max_drawdown: float = 0.15        # Maximum portfolio drawdown
    var_confidence: float = 0.95      # VaR confidence level
    max_concentration: float = 0.20   # Maximum position concentration
    
    # Position sizing parameters
    kelly_fraction: float = 0.25      # Kelly criterion fraction
    risk_parity_lookback: int = 252   # Risk parity lookback period
    volatility_target: float = 0.15   # Target portfolio volatility
    
    # Stress testing parameters
    stress_scenarios: List[str] = field(default_factory=lambda: [
        'market_crash_2008', 'covid_2020', 'dotcom_2000', 'custom_stress'
    ])
    monte_carlo_simulations: int = 1000
    confidence_intervals: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Dynamic hedging parameters
    hedge_threshold: float = 0.05     # Hedge when portfolio correlation > threshold
    rebalance_frequency: int = 5      # Rebalance every N days
    correlation_window: int = 60      # Rolling correlation window

@dataclass
class RiskEvent:
    """Risk event record"""
    timestamp: datetime
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_positions: List[str]
    risk_metrics: Dict[str, float]
    actions_taken: List[str]
    resolved: bool = False

class ComprehensiveRiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.risk_events = []
        self.portfolio_history = []
        self.correlation_analyzer = CorrelationAnalyzer()
        self.drawdown_tracker = DrawdownTracker()
        self.position_sizer = PositionSizer()
        self.stress_tester = StressTester()
        
    def assess_portfolio_risk(self, positions: Dict, prices: Dict, 
                            portfolio_value: float) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment"""
        
        try:
            risk_assessment = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'position_count': len(positions),
                'risk_events': [],
                'risk_score': 0.0,
                'recommendations': []
            }
            
            if not positions:
                risk_assessment['risk_score'] = 0.0
                risk_assessment['status'] = 'No positions'
                return risk_assessment
            
            # 1. Concentration Risk Assessment
            concentration_risk = self._assess_concentration_risk(positions, portfolio_value)
            risk_assessment['concentration_risk'] = concentration_risk
            
            # 2. Correlation Risk Assessment
            correlation_risk = self._assess_correlation_risk(positions, prices)
            risk_assessment['correlation_risk'] = correlation_risk
            
            # 3. Drawdown Assessment
            drawdown_risk = self._assess_drawdown_risk(portfolio_value)
            risk_assessment['drawdown_risk'] = drawdown_risk
            
            # 4. Volatility Assessment
            volatility_risk = self._assess_volatility_risk(positions, prices)
            risk_assessment['volatility_risk'] = volatility_risk
            
            # 5. Liquidity Assessment
            liquidity_risk = self._assess_liquidity_risk(positions, prices)
            risk_assessment['liquidity_risk'] = liquidity_risk
            
            # 6. Calculate Overall Risk Score
            risk_score = self._calculate_overall_risk_score(
                concentration_risk, correlation_risk, drawdown_risk, 
                volatility_risk, liquidity_risk
            )
            risk_assessment['risk_score'] = risk_score
            
            # 7. Generate Risk Events and Recommendations
            events, recommendations = self._generate_risk_alerts(risk_assessment)
            risk_assessment['risk_events'] = events
            risk_assessment['recommendations'] = recommendations
            
            # 8. Update Risk History
            self._update_risk_history(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            logging.error(f"Risk assessment failed: {str(e)}")
            return {
                'error': f'Risk assessment failed: {str(e)}',
                'timestamp': datetime.now(),
                'risk_score': 1.0  # Maximum risk on error
            }
    
    def _assess_concentration_risk(self, positions: Dict, portfolio_value: float) -> Dict:
        """Assess portfolio concentration risk"""
        
        if not positions or portfolio_value <= 0:
            return {'score': 0.0, 'largest_position': 0.0, 'top_5_concentration': 0.0}
        
        # Calculate position weights
        position_weights = []
        for ticker, position in positions.items():
            weight = position.get('value', 0) / portfolio_value
            position_weights.append((ticker, weight))
        
        # Sort by weight
        position_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate metrics
        largest_position = position_weights[0][1] if position_weights else 0.0
        top_5_concentration = sum(w for _, w in position_weights[:5])
        
        # Calculate concentration score (0-1, higher is more risky)
        concentration_score = min(largest_position / self.config.max_concentration, 1.0)
        
        return {
            'score': concentration_score,
            'largest_position': largest_position,
            'top_5_concentration': top_5_concentration,
            'herfindahl_index': sum(w**2 for _, w in position_weights),
            'effective_positions': 1 / sum(w**2 for _, w in position_weights) if position_weights else 0
        }
    
    def _assess_correlation_risk(self, positions: Dict, prices: Dict) -> Dict:
        """Assess correlation risk between positions"""
        
        try:
            tickers = list(positions.keys())
            
            if len(tickers) < 2:
                return {'score': 0.0, 'max_correlation': 0.0, 'avg_correlation': 0.0}
            
            # Calculate correlation matrix
            correlation_matrix = self.correlation_analyzer.calculate_correlations(
                {ticker: prices.get(ticker, pd.DataFrame()) for ticker in tickers}
            )
            
            if correlation_matrix.empty:
                return {'score': 0.5, 'max_correlation': 0.0, 'avg_correlation': 0.0}
            
            # Extract correlation values (excluding diagonal)
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    corr = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if not correlations:
                return {'score': 0.5, 'max_correlation': 0.0, 'avg_correlation': 0.0}
            
            max_correlation = max(correlations)
            avg_correlation = np.mean(correlations)
            
            # Calculate correlation risk score
            correlation_score = min(max_correlation / self.config.max_correlation, 1.0)
            
            return {
                'score': correlation_score,
                'max_correlation': max_correlation,
                'avg_correlation': avg_correlation,
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
        except Exception as e:
            logging.warning(f"Correlation risk assessment failed: {e}")
            return {'score': 0.5, 'max_correlation': 0.0, 'avg_correlation': 0.0}
    
    def _assess_drawdown_risk(self, current_portfolio_value: float) -> Dict:
        """Assess current drawdown risk"""
        
        drawdown_metrics = self.drawdown_tracker.update_drawdown(current_portfolio_value)
        
        # Calculate drawdown risk score
        current_drawdown = abs(drawdown_metrics['current_drawdown'])
        drawdown_score = min(current_drawdown / self.config.max_drawdown, 1.0)
        
        return {
            'score': drawdown_score,
            'current_drawdown': current_drawdown,
            'max_drawdown': abs(drawdown_metrics['max_drawdown']),
            'peak_value': drawdown_metrics['peak_value'],
            'underwater_days': self.drawdown_tracker.underwater_days
        }
    
    def _assess_volatility_risk(self, positions: Dict, prices: Dict) -> Dict:
        """Assess portfolio volatility risk"""
        
        try:
            if not positions:
                return {'score': 0.0, 'portfolio_volatility': 0.0}
            
            # Calculate portfolio volatility
            returns_data = {}
            for ticker in positions.keys():
                if ticker in prices and not prices[ticker].empty:
                    returns = prices[ticker]['Close'].pct_change().dropna()
                    if len(returns) > 20:
                        returns_data[ticker] = returns.tail(252)  # Last year
            
            if len(returns_data) < 2:
                return {'score': 0.3, 'portfolio_volatility': 0.0}
            
            # Calculate portfolio volatility (simplified)
            avg_volatility = np.mean([returns.std() * np.sqrt(252) for returns in returns_data.values()])
            
            # Calculate volatility risk score
            volatility_score = min(avg_volatility / self.config.volatility_target, 1.0)
            
            return {
                'score': volatility_score,
                'portfolio_volatility': avg_volatility,
                'individual_volatilities': {
                    ticker: returns.std() * np.sqrt(252) 
                    for ticker, returns in returns_data.items()
                }
            }
            
        except Exception as e:
            logging.warning(f"Volatility risk assessment failed: {e}")
            return {'score': 0.3, 'portfolio_volatility': 0.0}
    
    def _assess_liquidity_risk(self, positions: Dict, prices: Dict) -> Dict:
        """Assess portfolio liquidity risk"""
        
        try:
            liquidity_scores = []
            
            for ticker, position in positions.items():
                if ticker in prices and not prices[ticker].empty:
                    df = prices[ticker]
                    
                    # Simple liquidity proxy using volume
                    if 'Volume' in df.columns:
                        avg_volume = df['Volume'].tail(20).mean()
                        position_size = position.get('value', 0)
                        
                        # Rough liquidity score based on volume
                        if avg_volume > 1000000:  # High volume
                            liquidity_score = 0.1
                        elif avg_volume > 500000:  # Medium volume
                            liquidity_score = 0.3
                        else:  # Low volume
                            liquidity_score = 0.7
                        
                        liquidity_scores.append(liquidity_score)
                    else:
                        liquidity_scores.append(0.5)  # Default medium risk
                else:
                    liquidity_scores.append(0.8)  # High risk for missing data
            
            overall_liquidity_score = np.mean(liquidity_scores) if liquidity_scores else 0.5
            
            return {
                'score': overall_liquidity_score,
                'individual_scores': dict(zip(positions.keys(), liquidity_scores))
            }
            
        except Exception as e:
            logging.warning(f"Liquidity risk assessment failed: {e}")
            return {'score': 0.5, 'individual_scores': {}}
    
    def _calculate_overall_risk_score(self, concentration_risk: Dict, correlation_risk: Dict,
                                    drawdown_risk: Dict, volatility_risk: Dict, 
                                    liquidity_risk: Dict) -> float:
        """Calculate overall portfolio risk score"""
        
        # Weighted combination of risk scores
        weights = {
            'concentration': 0.25,
            'correlation': 0.20,
            'drawdown': 0.25,
            'volatility': 0.20,
            'liquidity': 0.10
        }
        
        overall_score = (
            weights['concentration'] * concentration_risk['score'] +
            weights['correlation'] * correlation_risk['score'] +
            weights['drawdown'] * drawdown_risk['score'] +
            weights['volatility'] * volatility_risk['score'] +
            weights['liquidity'] * liquidity_risk['score']
        )
        
        return min(overall_score, 1.0)
    
    def _generate_risk_alerts(self, risk_assessment: Dict) -> Tuple[List[RiskEvent], List[str]]:
        """Generate risk events and recommendations"""
        
        events = []
        recommendations = []
        
        # Concentration risk alerts
        concentration = risk_assessment.get('concentration_risk', {})
        if concentration.get('score', 0) > 0.8:
            events.append(RiskEvent(
                timestamp=datetime.now(),
                event_type='concentration_risk',
                severity='high',
                description=f"High concentration risk: largest position is {concentration.get('largest_position', 0):.1%}",
                affected_positions=[],
                risk_metrics=concentration,
                actions_taken=[]
            ))
            recommendations.append("Consider reducing largest position size")
        
        # Correlation risk alerts
        correlation = risk_assessment.get('correlation_risk', {})
        if correlation.get('score', 0) > 0.8:
            events.append(RiskEvent(
                timestamp=datetime.now(),
                event_type='correlation_risk',
                severity='high',
                description=f"High correlation risk: max correlation is {correlation.get('max_correlation', 0):.2f}",
                affected_positions=[],
                risk_metrics=correlation,
                actions_taken=[]
            ))
            recommendations.append("Consider diversifying into uncorrelated assets")
        
        # Drawdown risk alerts
        drawdown = risk_assessment.get('drawdown_risk', {})
        if drawdown.get('score', 0) > 0.8:
            events.append(RiskEvent(
                timestamp=datetime.now(),
                event_type='drawdown_risk',
                severity='critical',
                description=f"High drawdown risk: current drawdown is {drawdown.get('current_drawdown', 0):.1%}",
                affected_positions=[],
                risk_metrics=drawdown,
                actions_taken=[]
            ))
            recommendations.append("Consider reducing position sizes or implementing stop losses")
        
        return events, recommendations
    
    def _update_risk_history(self, risk_assessment: Dict):
        """Update risk history for trend analysis"""
        
        self.portfolio_history.append({
            'timestamp': risk_assessment['timestamp'],
            'risk_score': risk_assessment['risk_score'],
            'portfolio_value': risk_assessment['portfolio_value'],
            'concentration_score': risk_assessment.get('concentration_risk', {}).get('score', 0),
            'correlation_score': risk_assessment.get('correlation_risk', {}).get('score', 0),
            'drawdown_score': risk_assessment.get('drawdown_risk', {}).get('score', 0)
        })
        
        # Keep only last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def get_risk_dashboard_data(self) -> Dict:
        """Get data for risk dashboard visualization"""
        
        if not self.portfolio_history:
            return {'error': 'No risk history available'}
        
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            'risk_history': df,
            'current_risk': df.iloc[-1] if not df.empty else {},
            'risk_trends': self._calculate_risk_trends(df),
            'risk_events': self.risk_events[-10:]  # Last 10 events
        }
    
    def _calculate_risk_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate risk trends"""
        
        if len(df) < 2:
            return {}
        
        # Calculate trends over different periods
        periods = [7, 30, 90]  # days
        trends = {}
        
        for period in periods:
            if len(df) >= period:
                recent = df.tail(period)
                trend = (recent['risk_score'].iloc[-1] - recent['risk_score'].iloc[0]) / period
                trends[f'{period}d_trend'] = trend
        
        return trends

class CorrelationAnalyzer:
    """Analyze correlations between assets"""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.correlation_matrix = None
        self.correlation_history = []
    
    def calculate_correlations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for given assets"""
        
        try:
            returns_data = {}
            
            # Extract returns for each asset
            for ticker, df in data.items():
                if not df.empty and 'Close' in df.columns:
                    returns = df['Close'].pct_change().dropna()
                    if len(returns) > 20:  # Minimum data requirement
                        returns_data[ticker] = returns.tail(self.lookback_window)
            
            if len(returns_data) < 2:
                return pd.DataFrame()
            
            # Align data by index
            aligned_data = pd.DataFrame(returns_data)
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 20:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            self.correlation_matrix = aligned_data.corr()
            
            # Store correlation history
            self.correlation_history.append({
                'timestamp': datetime.now(),
                'correlations': self.correlation_matrix.to_dict()
            })
            
            # Keep only last 100 entries
            if len(self.correlation_history) > 100:
                self.correlation_history = self.correlation_history[-100:]
            
            return self.correlation_matrix
            
        except Exception as e:
            logging.error(f"Correlation calculation failed: {str(e)}")
            return pd.DataFrame()
    
    def get_correlation_clusters(self, threshold: float = 0.7) -> List[List[str]]:
        """Identify highly correlated asset clusters"""
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return []
        
        clusters = []
        processed = set()
        
        for i, asset1 in enumerate(self.correlation_matrix.columns):
            if asset1 in processed:
                continue
            
            cluster = [asset1]
            processed.add(asset1)
            
            for j, asset2 in enumerate(self.correlation_matrix.columns):
                if i != j and asset2 not in processed:
                    correlation = abs(self.correlation_matrix.iloc[i, j])
                    if correlation >= threshold:
                        cluster.append(asset2)
                        processed.add(asset2)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def calculate_rolling_correlations(self, data: Dict[str, pd.DataFrame], 
                                     window: int = 60) -> Dict[str, pd.Series]:
        """Calculate rolling correlations between assets"""
        
        rolling_correlations = {}
        
        try:
            tickers = list(data.keys())
            
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    ticker1, ticker2 = tickers[i], tickers[j]
                    
                    if (not data[ticker1].empty and not data[ticker2].empty and
                        'Close' in data[ticker1].columns and 'Close' in data[ticker2].columns):
                        
                        returns1 = data[ticker1]['Close'].pct_change().dropna()
                        returns2 = data[ticker2]['Close'].pct_change().dropna()
                        
                        # Align data
                        aligned = pd.DataFrame({
                            ticker1: returns1,
                            ticker2: returns2
                        }).dropna()
                        
                        if len(aligned) > window:
                            rolling_corr = aligned[ticker1].rolling(window).corr(aligned[ticker2])
                            rolling_correlations[f'{ticker1}_{ticker2}'] = rolling_corr
            
            return rolling_correlations
            
        except Exception as e:
            logging.error(f"Rolling correlation calculation failed: {str(e)}")
            return {}

class DrawdownTracker:
    """Track portfolio drawdowns"""
    
    def __init__(self):
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.underwater_days = 0
        self.drawdown_history = []
        self.last_peak_date = None
    
    def update_drawdown(self, current_value: float, date: datetime = None) -> Dict:
        """Update drawdown metrics"""
        
        if date is None:
            date = datetime.now()
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
            self.underwater_days = 0
            self.last_peak_date = date
        else:
            # Calculate current drawdown
            self.current_drawdown = (current_value - self.peak_value) / self.peak_value
            self.underwater_days += 1
            
            # Update maximum drawdown
            if self.current_drawdown < self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Record drawdown history
        self.drawdown_history.append({
            'date': date,
            'value': current_value,
            'peak_value': self.peak_value,
            'drawdown': self.current_drawdown,
            'underwater_days': self.underwater_days
        })
        
        # Keep only last 1000 entries
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-1000:]
        
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_value': self.peak_value,
            'underwater_days': self.underwater_days,
            'last_peak_date': self.last_peak_date
        }
    
    def get_drawdown_periods(self) -> List[Dict]:
        """Get all significant drawdown periods"""
        
        drawdown_periods = []
        current_period = None
        
        for entry in self.drawdown_history:
            if entry['drawdown'] < -0.05:  # 5% threshold
                if current_period is None:
                    current_period = {
                        'start_date': entry['date'],
                        'start_value': entry['peak_value'],
                        'max_drawdown': entry['drawdown'],
                        'duration': 0
                    }
                else:
                    current_period['max_drawdown'] = min(
                        current_period['max_drawdown'], 
                        entry['drawdown']
                    )
                    current_period['duration'] = entry['underwater_days']
            else:
                if current_period is not None:
                    current_period['end_date'] = entry['date']
                    current_period['recovery_value'] = entry['value']
                    drawdown_periods.append(current_period)
                    current_period = None
        
        return drawdown_periods

class PositionSizer:
    """Calculate optimal position sizes using various methods"""
    
    def __init__(self, method: str = 'equal_weight'):
        self.method = method
        self.sizing_history = []
    
    def calculate_position_sizes(self, signals: Dict[str, float], portfolio_value: float,
                               risk_data: Dict = None, historical_returns: Dict = None) -> Dict[str, float]:
        """Calculate position sizes for given signals"""
        
        if not signals:
            return {}
        
        try:
            if self.method == 'equal_weight':
                return self._equal_weight_sizing(signals, portfolio_value)
            
            elif self.method == 'risk_parity':
                return self._risk_parity_sizing(signals, portfolio_value, historical_returns)
            
            elif self.method == 'kelly':
                return self._kelly_criterion_sizing(signals, portfolio_value, historical_returns)
            
            elif self.method == 'volatility_target':
                return self._volatility_target_sizing(signals, portfolio_value, historical_returns)
            
            else:
                return self._equal_weight_sizing(signals, portfolio_value)
                
        except Exception as e:
            logging.error(f"Position sizing failed: {str(e)}")
            return self._equal_weight_sizing(signals, portfolio_value)
    
    def _equal_weight_sizing(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, float]:
        """Equal weight position sizing"""
        
        num_positions = len(signals)
        if num_positions == 0:
            return {}
        
        equal_weight = portfolio_value / num_positions
        
        return {ticker: equal_weight * signal for ticker, signal in signals.items()}
    
    def _risk_parity_sizing(self, signals: Dict[str, float], portfolio_value: float,
                           historical_returns: Dict = None) -> Dict[str, float]:
        """Risk parity position sizing"""
        
        if not historical_returns:
            return self._equal_weight_sizing(signals, portfolio_value)
        
        # Calculate volatilities
        volatilities = {}
        for ticker in signals.keys():
            if ticker in historical_returns:
                returns = historical_returns[ticker]
                if len(returns) > 20:
                    vol = returns.std() * np.sqrt(252)
                    volatilities[ticker] = vol
                else:
                    volatilities[ticker] = 0.2  # Default volatility
            else:
                volatilities[ticker] = 0.2
        
        # Calculate inverse volatility weights
        inv_vol_weights = {ticker: 1/vol for ticker, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol_weights.values())
        
        # Normalize weights
        normalized_weights = {
            ticker: (weight / total_inv_vol) * portfolio_value * signals[ticker]
            for ticker, weight in inv_vol_weights.items()
        }
        
        return normalized_weights
    
    def _kelly_criterion_sizing(self, signals: Dict[str, float], portfolio_value: float,
                               historical_returns: Dict = None) -> Dict[str, float]:
        """Kelly criterion position sizing"""
        
        if not historical_returns:
            return self._equal_weight_sizing(signals, portfolio_value)
        
        kelly_fractions = {}
        
        for ticker in signals.keys():
            if ticker in historical_returns:
                returns = historical_returns[ticker]
                if len(returns) > 50:
                    # Simplified Kelly calculation
                    avg_return = returns.mean()
                    variance = returns.var()
                    
                    if variance > 0:
                        kelly_fraction = avg_return / variance
                        kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
                    else:
                        kelly_fraction = 0.1
                else:
                    kelly_fraction = 0.1
            else:
                kelly_fraction = 0.1
            
            kelly_fractions[ticker] = kelly_fraction
        
        # Calculate position sizes
        position_sizes = {
            ticker: portfolio_value * kelly_fractions[ticker] * signals[ticker]
            for ticker in signals.keys()
        }
        
        return position_sizes
    
    def _volatility_target_sizing(self, signals: Dict[str, float], portfolio_value: float,
                                 historical_returns: Dict = None, target_vol: float = 0.15) -> Dict[str, float]:
        """Volatility target position sizing"""
        
        if not historical_returns:
            return self._equal_weight_sizing(signals, portfolio_value)
        
        # Calculate portfolio volatility and scale positions
        portfolio_vol = self._estimate_portfolio_volatility(signals, historical_returns)
        
        if portfolio_vol > 0:
            scale_factor = target_vol / portfolio_vol
            scale_factor = np.clip(scale_factor, 0.1, 2.0)  # Reasonable bounds
        else:
            scale_factor = 1.0
        
        base_sizes = self._equal_weight_sizing(signals, portfolio_value)
        
        return {ticker: size * scale_factor for ticker, size in base_sizes.items()}
    
    def _estimate_portfolio_volatility(self, signals: Dict[str, float], 
                                     historical_returns: Dict) -> float:
        """Estimate portfolio volatility (simplified)"""
        
        try:
            # Calculate average individual volatility (simplified)
            volatilities = []
            
            for ticker in signals.keys():
                if ticker in historical_returns:
                    returns = historical_returns[ticker]
                    if len(returns) > 20:
                        vol = returns.std() * np.sqrt(252)
                        volatilities.append(vol)
            
            if volatilities:
                # Simple average (ignoring correlations for simplicity)
                return np.mean(volatilities)
            else:
                return 0.2  # Default
                
        except Exception:
            return 0.2

class StressTester:
    """Perform stress testing on portfolio"""
    
    def __init__(self):
        self.stress_scenarios = {
            'market_crash_2008': {'factor': -0.35, 'correlation_increase': 0.3},
            'covid_2020': {'factor': -0.30, 'correlation_increase': 0.4},
            'dotcom_2000': {'factor': -0.45, 'correlation_increase': 0.2},
            'flash_crash': {'factor': -0.15, 'correlation_increase': 0.5},
            'currency_crisis': {'factor': -0.20, 'correlation_increase': 0.15}
        }
    
    def run_comprehensive_stress_test(self, portfolio_value: float, positions: Dict,
                                    historical_data: Dict = None) -> Dict:
        """Run comprehensive stress test on portfolio"""
        
        stress_results = {
            'original_value': portfolio_value,
            'scenario_results': {},
            'monte_carlo_results': {},
            'summary': {}
        }
        
        try:
            # Run historical scenario stress tests
            for scenario_name, scenario_params in self.stress_scenarios.items():
                result = self._run_scenario_stress_test(
                    portfolio_value, positions, scenario_params
                )
                stress_results['scenario_results'][scenario_name] = result
            
            # Run Monte Carlo stress test
            if historical_data:
                mc_results = self._run_monte_carlo_stress_test(
                    portfolio_value, positions, historical_data
                )
                stress_results['monte_carlo_results'] = mc_results
            
            # Calculate summary statistics
            stress_results['summary'] = self._calculate_stress_summary(stress_results)
            
            return stress_results
            
        except Exception as e:
            logging.error(f"Stress testing failed: {str(e)}")
            return {
                'error': f'Stress testing failed: {str(e)}',
                'original_value': portfolio_value
            }
    
    def _run_scenario_stress_test(self, portfolio_value: float, positions: Dict,
                                 scenario_params: Dict) -> Dict:
        """Run stress test for specific scenario"""
        
        stress_factor = scenario_params.get('factor', -0.2)
        stressed_value = portfolio_value * (1 + stress_factor)
        
        potential_loss = portfolio_value - stressed_value
        loss_percentage = abs(stress_factor) * 100
        
        return {
            'stressed_value': stressed_value,
            'potential_loss': potential_loss,
            'loss_percentage': loss_percentage,
            'scenario_params': scenario_params
        }
    
    def _run_monte_carlo_stress_test(self, portfolio_value: float, positions: Dict,
                                   historical_data: Dict, simulations: int = 1000) -> Dict:
        """Run Monte Carlo stress test"""
        
        try:
            # Extract historical returns
            returns_data = []
            for ticker, df in historical_data.items():
                if not df.empty and 'Close' in df.columns:
                    returns = df['Close'].pct_change().dropna()
                    if len(returns) > 50:
                        returns_data.append(returns.tail(252))  # Last year
            
            if not returns_data:
                return {'error': 'Insufficient historical data'}
            
            # Run simulations
            simulated_returns = []
            
            for _ in range(simulations):
                # Random sampling from historical returns
                portfolio_return = 0
                for returns in returns_data:
                    random_return = np.random.choice(returns)
                    portfolio_return += random_return / len(returns_data)  # Equal weight
                
                simulated_returns.append(portfolio_return)
            
            simulated_returns = np.array(simulated_returns)
            simulated_values = portfolio_value * (1 + simulated_returns)
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(simulated_values, percentiles)
            
            return {
                'simulated_values': simulated_values.tolist(),
                'percentiles': dict(zip(percentiles, percentile_values)),
                'worst_case': simulated_values.min(),
                'best_case': simulated_values.max(),
                'expected_value': simulated_values.mean(),
                'value_at_risk_95': np.percentile(simulated_values, 5),
                'conditional_var_95': simulated_values[simulated_values <= np.percentile(simulated_values, 5)].mean()
            }
            
        except Exception as e:
            logging.error(f"Monte Carlo stress test failed: {str(e)}")
            return {'error': f'Monte Carlo stress test failed: {str(e)}'}
    
    def _calculate_stress_summary(self, stress_results: Dict) -> Dict:
        """Calculate summary of stress test results"""
        
        summary = {
            'worst_scenario': None,
            'best_scenario': None,
            'average_loss': 0,
            'scenarios_analyzed': 0
        }
        
        scenario_results = stress_results.get('scenario_results', {})
        
        if scenario_results:
            losses = []
            worst_loss = 0
            best_loss = float('inf')
            worst_scenario = None
            best_scenario = None
            
            for scenario_name, result in scenario_results.items():
                loss_pct = result.get('loss_percentage', 0)
                losses.append(loss_pct)
                
                if loss_pct > worst_loss:
                    worst_loss = loss_pct
                    worst_scenario = scenario_name
                
                if loss_pct < best_loss:
                    best_loss = loss_pct
                    best_scenario = scenario_name
            
            summary.update({
                'worst_scenario': worst_scenario,
                'best_scenario': best_scenario,
                'average_loss': np.mean(losses),
                'scenarios_analyzed': len(scenario_results),
                'worst_loss_percentage': worst_loss,
                'best_loss_percentage': best_loss
            })
        
        return summary

def create_risk_dashboard_plots(risk_data: Dict) -> Dict[str, go.Figure]:
    """Create comprehensive risk dashboard plots"""
    
    plots = {}
    
    try:
        # 1. Risk Gauge Plot
        overall_risk = risk_data.get('risk_score', 0.5)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_risk * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Risk Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig_gauge.update_layout(
            title="Portfolio Risk Assessment",
            height=300
        )
        
        plots['risk_gauge'] = fig_gauge
        
        # 2. Risk Breakdown Pie Chart
        risk_components = {
            'Concentration': risk_data.get('concentration_risk', {}).get('score', 0) * 100,
            'Correlation': risk_data.get('correlation_risk', {}).get('score', 0) * 100,
            'Drawdown': risk_data.get('drawdown_risk', {}).get('score', 0) * 100,
            'Volatility': risk_data.get('volatility_risk', {}).get('score', 0) * 100,
            'Liquidity': risk_data.get('liquidity_risk', {}).get('score', 0) * 100
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(risk_components.keys()),
            values=list(risk_components.values()),
            hole=0.3
        )])
        
        fig_pie.update_layout(
            title="Risk Component Breakdown",
            height=400
        )
        
        plots['risk_breakdown'] = fig_pie
        
        # 3. Correlation Heatmap
        correlation_matrix = risk_data.get('correlation_risk', {}).get('correlation_matrix', {})
        
        if correlation_matrix:
            corr_df = pd.DataFrame(correlation_matrix)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdYlBu',
                zmid=0
            ))
            
            fig_heatmap.update_layout(
                title="Asset Correlation Matrix",
                height=400
            )
            
            plots['correlation_heatmap'] = fig_heatmap
        
        # 4. Risk History Timeline (if available)
        if 'risk_history' in risk_data and not risk_data['risk_history'].empty:
            df = risk_data['risk_history']
            
            fig_timeline = go.Figure()
            
            fig_timeline.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['risk_score'],
                mode='lines',
                name='Risk Score',
                line=dict(color='red', width=2)
            ))
            
            fig_timeline.update_layout(
                title="Risk Score Over Time",
                xaxis_title="Date",
                yaxis_title="Risk Score",
                height=300
            )
            
            plots['risk_timeline'] = fig_timeline
        
        return plots
        
    except Exception as e:
        logging.error(f"Risk plot creation failed: {str(e)}")
        return {}

# Export all classes and functions
__all__ = [
    'RiskConfig', 'RiskEvent', 'ComprehensiveRiskManager',
    'CorrelationAnalyzer', 'DrawdownTracker', 'PositionSizer', 'StressTester',
    'create_risk_dashboard_plots'
]

# Example usage
if __name__ == "__main__":
    print("Comprehensive Risk Management System")
    print("="*50)
    
    # Test risk manager
    config = RiskConfig()
    risk_manager = ComprehensiveRiskManager(config)
    
    # Mock portfolio data
    mock_positions = {
        'RELIANCE.NS': {'value': 500000},
        'TCS.NS': {'value': 300000},
        'HDFCBANK.NS': {'value': 200000}
    }
    
    mock_prices = {
        'RELIANCE.NS': pd.DataFrame({
            'Close': np.random.normal(2500, 50, 100)
        }),
        'TCS.NS': pd.DataFrame({
            'Close': np.random.normal(3500, 70, 100)
        }),
        'HDFCBANK.NS': pd.DataFrame({
            'Close': np.random.normal(1600, 40, 100)
        })
    }
    
    # Run risk assessment
    risk_assessment = risk_manager.assess_portfolio_risk(
        positions=mock_positions,
        prices=mock_prices,
        portfolio_value=1000000
    )
    
    print(f"Overall Risk Score: {risk_assessment.get('risk_score', 0):.3f}")
    print(f"Risk Events: {len(risk_assessment.get('risk_events', []))}")
    print(f"Recommendations: {len(risk_assessment.get('recommendations', []))}")
    
    # Test stress testing
    stress_tester = StressTester()
    stress_results = stress_tester.run_comprehensive_stress_test(
        portfolio_value=1000000,
        positions=mock_positions,
        historical_data=mock_prices
    )
    
    print(f"\nStress Test Results:")
    print(f"Scenarios Analyzed: {stress_results.get('summary', {}).get('scenarios_analyzed', 0)}")
    print(f"Worst Case Loss: {stress_results.get('summary', {}).get('worst_loss_percentage', 0):.1f}%")
    
    print("\nRisk Management System Test Completed!")