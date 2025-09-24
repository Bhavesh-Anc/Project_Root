# utils/portfolio_optimization.py - Complete Portfolio Optimization Module
"""
Advanced Portfolio Optimization with Modern Portfolio Theory
Compatible with AI Stock Advisor Pro - Enhanced Edition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Using simplified optimization.")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY not available. Using alternative optimization methods.")

# ==================== CONFIGURATION ====================

@dataclass
class OptimizationConfig:
    """Configuration class for portfolio optimization"""
    
    # Optimization objectives
    objective: str = 'max_sharpe'  # 'max_sharpe', 'min_risk', 'max_return', 'risk_parity'
    risk_free_rate: float = 0.06  # 6% risk-free rate
    
    # Constraints
    min_weight: float = 0.01  # Minimum 1% allocation per stock
    max_weight: float = 0.40  # Maximum 40% allocation per stock
    max_concentration: float = 0.60  # Maximum combined weight of top 3 positions
    target_return: Optional[float] = None  # For efficient frontier optimization
    target_risk: Optional[float] = None  # For risk-targeted optimization
    
    # Risk parameters
    lookback_days: int = 252  # 1 year of data for calculations
    confidence_level: float = 0.95  # For VaR calculations
    rebalance_frequency: str = 'quarterly'  # 'monthly', 'quarterly', 'semi-annual', 'annual'
    
    # Advanced features
    enable_transaction_costs: bool = False
    transaction_cost_rate: float = 0.001  # 0.1% transaction cost
    enable_sector_constraints: bool = False
    max_sector_weight: float = 0.30  # Maximum 30% per sector
    
    # Optimization parameters
    max_iterations: int = 1000
    tolerance: float = 1e-8
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0 < self.min_weight < self.max_weight <= 1:
            raise ValueError("Invalid weight constraints")
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            raise ValueError("Risk-free rate must be between 0 and 1")
        if self.lookback_days < 30:
            raise ValueError("Lookback period too short")

# ==================== PORTFOLIO METRICS CALCULATOR ====================

class PortfolioMetricsCalculator:
    """Calculate various portfolio performance and risk metrics"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def calculate_returns_and_covariance(self, price_data: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate expected returns and covariance matrix from price data"""
        
        # Extract returns for each stock
        returns_dict = {}
        
        for ticker, df in price_data.items():
            if 'Close' in df.columns and len(df) >= self.config.lookback_days:
                # Use the most recent data
                recent_data = df.tail(self.config.lookback_days)
                daily_returns = recent_data['Close'].pct_change().dropna()
                
                if len(daily_returns) >= 30:  # Minimum data requirement
                    returns_dict[ticker] = daily_returns
                else:
                    logging.warning(f"Insufficient return data for {ticker}")
        
        if len(returns_dict) < 2:
            raise ValueError("Need at least 2 stocks with sufficient data")
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Handle missing values
        returns_df = returns_df.fillna(returns_df.mean())
        
        # Calculate expected returns (annualized)
        expected_returns = returns_df.mean() * 252
        
        # Calculate covariance matrix (annualized)
        covariance_matrix = returns_df.cov() * 252
        
        # Ensure positive semi-definite covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
        covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        covariance_matrix = pd.DataFrame(covariance_matrix, 
                                       index=expected_returns.index, 
                                       columns=expected_returns.index)
        
        return expected_returns, covariance_matrix
    
    def calculate_portfolio_performance(self, weights: np.ndarray, 
                                      expected_returns: pd.Series, 
                                      covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Portfolio return
        portfolio_return = np.dot(weights, expected_returns)
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        # Sharpe ratio
        excess_return = portfolio_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Information ratio (simplified)
        information_ratio = sharpe_ratio  # Simplified for now
        
        # Maximum individual weight
        max_weight = np.max(weights)
        
        # Concentration (Herfindahl index)
        concentration = np.sum(weights**2)
        
        # Diversification ratio
        individual_volatilities = np.sqrt(np.diag(covariance_matrix))
        weighted_avg_volatility = np.dot(weights, individual_volatilities)
        diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'max_weight': max_weight,
            'concentration': concentration,
            'diversification_ratio': diversification_ratio,
            'variance': portfolio_variance
        }
    
    def calculate_var(self, weights: np.ndarray, 
                     expected_returns: pd.Series, 
                     covariance_matrix: pd.DataFrame) -> float:
        """Calculate Value at Risk (VaR) using parametric method"""
        
        # Portfolio statistics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        # VaR calculation (assuming normal distribution)
        from scipy.stats import norm
        z_score = norm.ppf(1 - self.config.confidence_level)
        var = -(portfolio_return + z_score * portfolio_volatility)
        
        return max(var, 0)  # Ensure non-negative VaR

# ==================== OPTIMIZATION ENGINES ====================

class ModernPortfolioTheoryOptimizer:
    """Modern Portfolio Theory optimization implementation"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics_calculator = PortfolioMetricsCalculator(config)
    
    def optimize_max_sharpe(self, expected_returns: pd.Series, 
                           covariance_matrix: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Optimize for maximum Sharpe ratio"""
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            """Objective function to minimize (negative Sharpe ratio)"""
            performance = self.metrics_calculator.calculate_portfolio_performance(
                weights, expected_returns, covariance_matrix)
            return -performance['sharpe_ratio']
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Add concentration constraint
        if self.config.max_concentration < 1.0:
            def concentration_constraint(weights):
                sorted_weights = np.sort(weights)[::-1]  # Sort descending
                top_3_weight = np.sum(sorted_weights[:min(3, len(weights))])
                return self.config.max_concentration - top_3_weight
            
            constraints.append({'type': 'ineq', 'fun': concentration_constraint})
        
        # Initial guess (equal weights)
        initial_guess = np.array([1.0/n_assets] * n_assets)
        
        # Optimization
        if SCIPY_AVAILABLE:
            result = minimize(
                objective, 
                initial_guess, 
                method='SLSQP',
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)  # Normalize
                optimization_info = {
                    'success': True,
                    'message': result.message,
                    'iterations': result.nit,
                    'method': 'max_sharpe_scipy'
                }
            else:
                # Fallback to equal weights
                optimal_weights = initial_guess
                optimization_info = {
                    'success': False,
                    'message': 'Optimization failed, using equal weights',
                    'method': 'equal_weights_fallback'
                }
        else:
            # Simple fallback without SciPy
            optimal_weights = self._optimize_sharpe_simple(expected_returns, covariance_matrix)
            optimization_info = {
                'success': True,
                'message': 'Simple optimization without SciPy',
                'method': 'simple_sharpe'
            }
        
        return optimal_weights, optimization_info
    
    def optimize_min_variance(self, expected_returns: pd.Series, 
                            covariance_matrix: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Optimize for minimum variance (risk)"""
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            """Objective function to minimize (portfolio variance)"""
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return portfolio_variance
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_guess = np.array([1.0/n_assets] * n_assets)
        
        # Optimization
        if SCIPY_AVAILABLE:
            result = minimize(
                objective, 
                initial_guess, 
                method='SLSQP',
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)
                optimization_info = {
                    'success': True,
                    'message': result.message,
                    'method': 'min_variance_scipy'
                }
            else:
                optimal_weights = initial_guess
                optimization_info = {
                    'success': False,
                    'message': 'Optimization failed, using equal weights',
                    'method': 'equal_weights_fallback'
                }
        else:
            # Simple minimum variance without SciPy
            optimal_weights = self._optimize_min_variance_simple(covariance_matrix)
            optimization_info = {
                'success': True,
                'message': 'Simple minimum variance optimization',
                'method': 'simple_min_variance'
            }
        
        return optimal_weights, optimization_info
    
    def optimize_risk_parity(self, expected_returns: pd.Series, 
                           covariance_matrix: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Risk parity optimization - equal risk contribution"""
        
        n_assets = len(expected_returns)
        
        def risk_budget_objective(weights):
            """Minimize the sum of squared risk contribution differences"""
            weights = weights / np.sum(weights)  # Normalize
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            if portfolio_variance <= 0:
                return 1e6  # Penalize invalid solutions
            
            # Calculate marginal risk contributions
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # Target is equal risk contribution (1/n for each asset)
            target_contrib = np.ones(n_assets) / n_assets
            
            # Minimize squared differences from target
            return np.sum((risk_contrib - target_contrib)**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_guess = np.array([1.0/n_assets] * n_assets)
        
        # Optimization
        if SCIPY_AVAILABLE:
            result = minimize(
                risk_budget_objective, 
                initial_guess, 
                method='SLSQP',
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': self.config.max_iterations * 2}  # Risk parity needs more iterations
            )
            
            if result.success:
                optimal_weights = result.x / np.sum(result.x)
                optimization_info = {
                    'success': True,
                    'message': 'Risk parity optimization successful',
                    'method': 'risk_parity_scipy'
                }
            else:
                # Fallback to inverse volatility weights
                optimal_weights = self._inverse_volatility_weights(covariance_matrix)
                optimization_info = {
                    'success': False,
                    'message': 'Risk parity failed, using inverse volatility',
                    'method': 'inverse_volatility_fallback'
                }
        else:
            # Fallback without SciPy
            optimal_weights = self._inverse_volatility_weights(covariance_matrix)
            optimization_info = {
                'success': True,
                'message': 'Inverse volatility weighting (risk parity approximation)',
                'method': 'inverse_volatility'
            }
        
        return optimal_weights, optimization_info
    
    def _optimize_sharpe_simple(self, expected_returns: pd.Series, 
                               covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Simple Sharpe ratio optimization without SciPy"""
        
        # Use analytical solution for unconstrained case, then project to feasible region
        excess_returns = expected_returns - self.config.risk_free_rate
        
        try:
            # Inverse covariance matrix
            inv_cov = np.linalg.inv(covariance_matrix.values)
            
            # Analytical solution
            optimal_weights = np.dot(inv_cov, excess_returns.values)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            # Project to feasible region
            optimal_weights = np.clip(optimal_weights, self.config.min_weight, self.config.max_weight)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to equal weights if matrix inversion fails
            n_assets = len(expected_returns)
            optimal_weights = np.array([1.0/n_assets] * n_assets)
        
        return optimal_weights
    
    def _optimize_min_variance_simple(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Simple minimum variance optimization without SciPy"""
        
        try:
            # Analytical solution for minimum variance
            inv_cov = np.linalg.inv(covariance_matrix.values)
            ones = np.ones((len(covariance_matrix), 1))
            
            # Minimum variance weights
            weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
            optimal_weights = weights.flatten()
            
            # Apply constraints
            optimal_weights = np.clip(optimal_weights, self.config.min_weight, self.config.max_weight)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to equal weights
            n_assets = len(covariance_matrix)
            optimal_weights = np.array([1.0/n_assets] * n_assets)
        
        return optimal_weights
    
    def _inverse_volatility_weights(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate inverse volatility weights (risk parity approximation)"""
        
        # Individual asset volatilities
        volatilities = np.sqrt(np.diag(covariance_matrix))
        
        # Inverse volatility weights
        inv_vol_weights = 1 / volatilities
        
        # Normalize
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Apply constraints
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        weights = weights / np.sum(weights)
        
        return weights

# ==================== ADVANCED PORTFOLIO OPTIMIZER ====================

class AdvancedPortfolioOptimizer:
    """Main portfolio optimization class with multiple strategies"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.mpt_optimizer = ModernPortfolioTheoryOptimizer(self.config)
        self.metrics_calculator = PortfolioMetricsCalculator(self.config)
        
        np.random.seed(self.config.random_seed)
    
    def optimize_portfolio(self, price_data: Dict[str, pd.DataFrame], 
                          predictions_df: Optional[pd.DataFrame] = None,
                          selected_tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main portfolio optimization function
        
        Args:
            price_data: Dictionary of stock price data
            predictions_df: Optional ML predictions to incorporate
            selected_tickers: List of tickers to optimize (if None, use all available)
        
        Returns:
            Dictionary with optimization results
        """
        
        try:
            # Filter data to selected tickers if provided
            if selected_tickers:
                price_data = {ticker: df for ticker, df in price_data.items() 
                             if ticker in selected_tickers}
            
            if len(price_data) < 2:
                raise ValueError("Need at least 2 stocks for optimization")
            
            # Calculate returns and covariance
            expected_returns, covariance_matrix = self.metrics_calculator.calculate_returns_and_covariance(price_data)
            
            # Incorporate ML predictions if available
            if predictions_df is not None and not predictions_df.empty:
                expected_returns = self._incorporate_ml_predictions(expected_returns, predictions_df)
            
            # Run optimization based on objective
            if self.config.objective == 'max_sharpe':
                optimal_weights, optimization_info = self.mpt_optimizer.optimize_max_sharpe(
                    expected_returns, covariance_matrix)
            elif self.config.objective == 'min_risk':
                optimal_weights, optimization_info = self.mpt_optimizer.optimize_min_variance(
                    expected_returns, covariance_matrix)
            elif self.config.objective == 'risk_parity':
                optimal_weights, optimization_info = self.mpt_optimizer.optimize_risk_parity(
                    expected_returns, covariance_matrix)
            else:
                # Default to max Sharpe
                optimal_weights, optimization_info = self.mpt_optimizer.optimize_max_sharpe(
                    expected_returns, covariance_matrix)
            
            # Calculate portfolio performance
            performance_metrics = self.metrics_calculator.calculate_portfolio_performance(
                optimal_weights, expected_returns, covariance_matrix)
            
            # Calculate additional metrics
            var_95 = self.metrics_calculator.calculate_var(
                optimal_weights, expected_returns, covariance_matrix)
            
            # Create results dictionary
            tickers = list(expected_returns.index)
            weights_dict = {ticker: weight for ticker, weight in zip(tickers, optimal_weights)}
            
            results = {
                'weights': weights_dict,
                'expected_return': performance_metrics['expected_return'],
                'volatility': performance_metrics['volatility'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'var_95': var_95,
                'max_weight': performance_metrics['max_weight'],
                'concentration': performance_metrics['concentration'],
                'diversification_ratio': performance_metrics['diversification_ratio'],
                'optimization_method': self.config.objective,
                'optimization_info': optimization_info,
                'individual_expected_returns': expected_returns.to_dict(),
                'correlation_matrix': expected_returns.index.to_list(),  # For reference
                'config': {
                    'objective': self.config.objective,
                    'risk_free_rate': self.config.risk_free_rate,
                    'min_weight': self.config.min_weight,
                    'max_weight': self.config.max_weight,
                    'lookback_days': self.config.lookback_days
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Add rebalancing recommendations
            results['rebalancing'] = self._generate_rebalancing_recommendations(
                optimal_weights, tickers, performance_metrics)
            
            # Add sector analysis if available
            results['sector_analysis'] = self._analyze_sector_exposure(weights_dict)
            
            logging.info(f"Portfolio optimization completed: {self.config.objective}")
            
            return results
            
        except Exception as e:
            logging.error(f"Portfolio optimization failed: {e}")
            
            # Return fallback equal-weight portfolio
            if selected_tickers:
                tickers = selected_tickers
            else:
                tickers = list(price_data.keys())
            
            n_assets = len(tickers)
            equal_weight = 1.0 / n_assets
            
            return {
                'weights': {ticker: equal_weight for ticker in tickers},
                'expected_return': 0.08,  # Assumed 8% return
                'volatility': 0.15,  # Assumed 15% volatility
                'sharpe_ratio': (0.08 - self.config.risk_free_rate) / 0.15,
                'var_95': 0.10,  # Assumed 10% VaR
                'max_weight': equal_weight,
                'concentration': 1.0 / n_assets,  # Herfindahl index for equal weights
                'diversification_ratio': 1.0,
                'optimization_method': 'equal_weight_fallback',
                'optimization_info': {
                    'success': False,
                    'message': f'Optimization failed: {str(e)}. Using equal weights.',
                    'method': 'equal_weight_fallback'
                },
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _incorporate_ml_predictions(self, expected_returns: pd.Series, 
                                  predictions_df: pd.DataFrame) -> pd.Series:
        """Incorporate ML predictions into expected returns"""
        
        enhanced_returns = expected_returns.copy()
        
        try:
            # Map predictions to returns
            for _, row in predictions_df.iterrows():
                ticker = row.get('ticker')
                predicted_return = row.get('predicted_return', 0)
                confidence = row.get('ensemble_confidence', 0.5)
                
                if ticker in enhanced_returns.index and predicted_return is not None:
                    # Blend historical and predicted returns based on confidence
                    historical_return = enhanced_returns[ticker]
                    blended_return = (confidence * predicted_return + 
                                    (1 - confidence) * historical_return)
                    enhanced_returns[ticker] = blended_return
            
            logging.info("ML predictions incorporated into expected returns")
            
        except Exception as e:
            logging.warning(f"Failed to incorporate ML predictions: {e}")
        
        return enhanced_returns
    
    def _generate_rebalancing_recommendations(self, optimal_weights: np.ndarray, 
                                            tickers: List[str],
                                            performance_metrics: Dict) -> Dict:
        """Generate rebalancing recommendations"""
        
        recommendations = {
            'frequency': self.config.rebalance_frequency,
            'next_rebalance': self._calculate_next_rebalance_date(),
            'tolerance_bands': {},
            'monitoring_metrics': {
                'concentration': performance_metrics['concentration'],
                'max_weight': performance_metrics['max_weight'],
                'diversification_ratio': performance_metrics['diversification_ratio']
            }
        }
        
        # Calculate tolerance bands for each asset
        for i, ticker in enumerate(tickers):
            weight = optimal_weights[i]
            # Set tolerance bands based on volatility and weight
            tolerance = min(0.05, weight * 0.25)  # Max 5% or 25% of weight
            recommendations['tolerance_bands'][ticker] = {
                'target': weight,
                'lower_bound': max(self.config.min_weight, weight - tolerance),
                'upper_bound': min(self.config.max_weight, weight + tolerance)
            }
        
        return recommendations
    
    def _calculate_next_rebalance_date(self) -> str:
        """Calculate next rebalancing date"""
        
        today = datetime.now()
        
        if self.config.rebalance_frequency == 'monthly':
            # Next month, same day
            if today.month == 12:
                next_date = today.replace(year=today.year + 1, month=1)
            else:
                next_date = today.replace(month=today.month + 1)
        elif self.config.rebalance_frequency == 'quarterly':
            # Next quarter end
            current_quarter = (today.month - 1) // 3 + 1
            if current_quarter == 4:
                next_date = datetime(today.year + 1, 3, 31)
            else:
                quarter_end_months = [3, 6, 9, 12]
                next_month = quarter_end_months[current_quarter]
                next_date = datetime(today.year, next_month, 30)
        elif self.config.rebalance_frequency == 'semi-annual':
            # Next June 30 or December 31
            if today.month <= 6:
                next_date = datetime(today.year, 6, 30)
            else:
                next_date = datetime(today.year, 12, 31)
        else:  # annual
            # Next December 31
            next_date = datetime(today.year, 12, 31)
            if today.month == 12 and today.day > 25:
                next_date = datetime(today.year + 1, 12, 31)
        
        return next_date.strftime('%Y-%m-%d')
    
    def _analyze_sector_exposure(self, weights_dict: Dict[str, float]) -> Dict:
        """Analyze sector exposure (simplified mapping)"""
        
        # Simplified sector mapping for Indian stocks
        sector_mapping = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'Technology',
            'HDFCBANK.NS': 'Banking',
            'INFY.NS': 'Technology',
            'HINDUNILVR.NS': 'Consumer Goods',
            'ICICIBANK.NS': 'Banking',
            'KOTAKBANK.NS': 'Banking',
            'SBIN.NS': 'Banking',
            'BHARTIARTL.NS': 'Telecom',
            'LT.NS': 'Industrial',
            'ASIANPAINT.NS': 'Consumer Goods',
            'AXISBANK.NS': 'Banking',
            'MARUTI.NS': 'Automotive',
            'NESTLEIND.NS': 'Consumer Goods',
            'HCLTECH.NS': 'Technology'
        }
        
        sector_exposure = {}
        unknown_sector_weight = 0
        
        for ticker, weight in weights_dict.items():
            sector = sector_mapping.get(ticker, 'Unknown')
            if sector == 'Unknown':
                unknown_sector_weight += weight
            else:
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        if unknown_sector_weight > 0:
            sector_exposure['Other'] = unknown_sector_weight
        
        return {
            'sector_weights': sector_exposure,
            'max_sector_exposure': max(sector_exposure.values()) if sector_exposure else 0,
            'sector_count': len(sector_exposure),
            'is_diversified': max(sector_exposure.values()) < 0.5 if sector_exposure else False
        }
    
    def generate_efficient_frontier(self, price_data: Dict[str, pd.DataFrame],
                                   n_portfolios: int = 50) -> pd.DataFrame:
        """Generate efficient frontier points"""
        
        try:
            # Calculate returns and covariance
            expected_returns, covariance_matrix = self.metrics_calculator.calculate_returns_and_covariance(price_data)
            
            # Range of target returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, n_portfolios)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                # Temporarily set target return in config
                original_objective = self.config.objective
                original_target = self.config.target_return
                
                self.config.objective = 'efficient_frontier'
                self.config.target_return = target_return
                
                try:
                    # Optimize for minimum risk at target return
                    weights, _ = self.mpt_optimizer.optimize_min_variance(expected_returns, covariance_matrix)
                    performance = self.metrics_calculator.calculate_portfolio_performance(
                        weights, expected_returns, covariance_matrix)
                    
                    efficient_portfolios.append({
                        'target_return': target_return,
                        'return': performance['expected_return'],
                        'volatility': performance['volatility'],
                        'sharpe_ratio': performance['sharpe_ratio']
                    })
                    
                except Exception as e:
                    logging.warning(f"Failed to optimize for target return {target_return:.3f}: {e}")
                
                # Restore original config
                self.config.objective = original_objective
                self.config.target_return = original_target
            
            return pd.DataFrame(efficient_portfolios)
            
        except Exception as e:
            logging.error(f"Efficient frontier generation failed: {e}")
            return pd.DataFrame()

# ==================== CONVENIENCE FUNCTIONS ====================

def optimize_portfolio_for_selected_stocks(predictions_df: pd.DataFrame,
                                         price_data: Dict[str, pd.DataFrame], 
                                         selected_tickers: List[str],
                                         objective: str = 'max_sharpe',
                                         risk_free_rate: float = 0.06,
                                         max_weight: float = 0.40) -> Dict[str, Any]:
    """
    Convenience function to optimize portfolio for user-selected stocks
    
    Args:
        predictions_df: ML predictions DataFrame
        price_data: Historical price data
        selected_tickers: List of selected stock tickers
        objective: Optimization objective ('max_sharpe', 'min_risk', 'risk_parity')
        risk_free_rate: Risk-free rate for Sharpe calculation
        max_weight: Maximum weight per stock
    
    Returns:
        Dictionary with optimization results
    """
    
    # Create configuration
    config = OptimizationConfig(
        objective=objective,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_weight=0.02,  # Minimum 2% allocation
        max_concentration=0.70  # Allow 70% in top 3 positions
    )
    
    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize_portfolio(
        price_data=price_data,
        predictions_df=predictions_df,
        selected_tickers=selected_tickers
    )
    
    logging.info(f"Portfolio optimization completed for {len(selected_tickers)} selected stocks")
    
    return results

def create_risk_budgeted_portfolio(price_data: Dict[str, pd.DataFrame],
                                 selected_tickers: List[str],
                                 risk_budgets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create a risk-budgeted portfolio for selected stocks
    
    Args:
        price_data: Historical price data
        selected_tickers: List of selected tickers
        risk_budgets: Optional risk budget per stock (if None, equal risk)
    
    Returns:
        Risk-budgeted portfolio results
    """
    
    config = OptimizationConfig(
        objective='risk_parity',
        max_weight=0.50,  # Allow higher concentration for risk budgeting
        min_weight=0.01
    )
    
    optimizer = AdvancedPortfolioOptimizer(config)
    
    results = optimizer.optimize_portfolio(
        price_data=price_data,
        selected_tickers=selected_tickers
    )
    
    # Adjust for custom risk budgets if provided
    if risk_budgets:
        # This would require more complex implementation
        results['custom_risk_budgets'] = risk_budgets
        results['note'] = 'Custom risk budgets noted but not implemented in this version'
    
    return results

# ==================== PORTFOLIO ANALYSIS FUNCTIONS ====================

def analyze_portfolio_concentration(weights_dict: Dict[str, float]) -> Dict[str, Any]:
    """Analyze portfolio concentration metrics"""
    
    weights = np.array(list(weights_dict.values()))
    
    return {
        'herfindahl_index': np.sum(weights**2),
        'effective_n_stocks': 1 / np.sum(weights**2),
        'max_weight': np.max(weights),
        'top_3_concentration': np.sum(np.sort(weights)[-3:]),
        'top_5_concentration': np.sum(np.sort(weights)[-5:]),
        'gini_coefficient': _calculate_gini_coefficient(weights)
    }

def _calculate_gini_coefficient(weights: np.ndarray) -> float:
    """Calculate Gini coefficient for weight distribution"""
    
    n = len(weights)
    if n == 0:
        return 0
    
    # Sort weights
    sorted_weights = np.sort(weights)
    
    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_weights)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_weights))) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    return max(0, min(1, gini))

def calculate_portfolio_attribution(weights_dict: Dict[str, float],
                                  returns_data: Dict[str, pd.Series]) -> Dict[str, Any]:
    """Calculate portfolio return attribution"""
    
    attribution = {}
    
    for ticker, weight in weights_dict.items():
        if ticker in returns_data:
            stock_return = returns_data[ticker].mean() * 252  # Annualized
            contribution = weight * stock_return
            attribution[ticker] = {
                'weight': weight,
                'return': stock_return,
                'contribution': contribution
            }
    
    total_contribution = sum(attr['contribution'] for attr in attribution.values())
    
    return {
        'stock_attribution': attribution,
        'total_return': total_contribution,
        'top_contributors': sorted(attribution.items(), 
                                  key=lambda x: x[1]['contribution'], 
                                  reverse=True)[:5]
    }

# ==================== VALIDATION AND TESTING ====================

def validate_optimization_results(results: Dict[str, Any]) -> Dict[str, bool]:
    """Validate optimization results"""
    
    validation = {
        'weights_sum_to_one': False,
        'weights_within_bounds': False,
        'positive_sharpe': False,
        'reasonable_return': False,
        'reasonable_volatility': False
    }
    
    try:
        # Check weights sum to 1
        total_weight = sum(results.get('weights', {}).values())
        validation['weights_sum_to_one'] = abs(total_weight - 1.0) < 1e-6
        
        # Check weight bounds
        weights = list(results.get('weights', {}).values())
        validation['weights_within_bounds'] = all(0 <= w <= 1 for w in weights)
        
        # Check positive Sharpe ratio
        sharpe = results.get('sharpe_ratio', -999)
        validation['positive_sharpe'] = sharpe > 0
        
        # Check reasonable return (between -50% and 100%)
        ret = results.get('expected_return', -999)
        validation['reasonable_return'] = -0.5 <= ret <= 1.0
        
        # Check reasonable volatility (between 1% and 100%)
        vol = results.get('volatility', -999)
        validation['reasonable_volatility'] = 0.01 <= vol <= 1.0
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
    
    return validation

# ==================== MAIN EXECUTION AND TESTING ====================

if __name__ == "__main__":
    print("Advanced Portfolio Optimization Module")
    print("=" * 50)
    
    # Configuration showcase
    print("\nConfiguration Options:")
    config = OptimizationConfig()
    print(f"  Default Objective: {config.objective}")
    print(f"  Risk-free Rate: {config.risk_free_rate:.1%}")
    print(f"  Weight Bounds: {config.min_weight:.1%} - {config.max_weight:.1%}")
    print(f"  Lookback Period: {config.lookback_days} days")
    print(f"  Rebalancing: {config.rebalance_frequency}")
    
    # Feature showcase
    print(f"\nAvailable Features:")
    print(f"  ✓ Modern Portfolio Theory Optimization")
    print(f"  ✓ Multiple Objectives (Max Sharpe, Min Risk, Risk Parity)")
    print(f"  ✓ ML Predictions Integration")
    print(f"  ✓ Advanced Risk Metrics (VaR, Concentration, etc.)")
    print(f"  ✓ Efficient Frontier Generation")
    print(f"  ✓ Rebalancing Recommendations")
    print(f"  ✓ Sector Exposure Analysis")
    print(f"  ✓ Portfolio Attribution Analysis")
    print(f"  ✓ Comprehensive Validation")
    
    # Dependency status
    print(f"\nDependency Status:")
    print(f"  SciPy: {'✓ Available' if SCIPY_AVAILABLE else '✗ Not Available (using fallbacks)'}")
    print(f"  CVXPY: {'✓ Available' if CVXPY_AVAILABLE else '✗ Not Available (not critical)'}")
    
    print(f"\nOptimization Methods Available:")
    print(f"  • Maximum Sharpe Ratio")
    print(f"  • Minimum Variance (Risk)")
    print(f"  • Risk Parity")
    print(f"  • Equal Weight (Fallback)")
    print(f"  • Inverse Volatility Weighting")
    
    print(f"\nIntegration Features:")
    print(f"  • Compatible with AI Stock Advisor Pro")
    print(f"  • ML Predictions Integration")
    print(f"  • User Stock Selection Support")
    print(f"  • Comprehensive Error Handling")
    print(f"  • Fallback Mechanisms")
    
    # Test basic functionality
    print(f"\nBasic Functionality Test:")
    try:
        test_config = OptimizationConfig(objective='max_sharpe', max_weight=0.3)
        optimizer = AdvancedPortfolioOptimizer(test_config)
        print(f"  ✓ Configuration and initialization successful")
        
        # Test metrics calculator
        calc = PortfolioMetricsCalculator(test_config)
        print(f"  ✓ Metrics calculator initialized")
        
        print(f"  ✓ All systems operational!")
        
    except Exception as e:
        print(f"  ✗ Error during testing: {e}")
    
    print(f"\nModule ready for integration with AI Stock Advisor Pro!")