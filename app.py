# app.py - Complete Full Version - Fixed Investment Horizon Error Only
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import gc
import warnings
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import sys
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

# ==================== ROBUST IMPORT HANDLING ====================

# Configure logging first
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('model_cache', exist_ok=True)
os.makedirs('feature_cache_v2', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Enhanced Streamlit page configuration
st.set_page_config(
    page_title="AI Stock Advisor Pro - Enhanced Edition",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FIXED MODULE LOADER ====================

class FixedModuleLoader:
    """Fixed module loader compatible with existing codebase"""
    
    def __init__(self):
        self.modules_status = {
            'data_loader': False,
            'feature_engineer': False, 
            'model': False,
            'backtesting': False,
            'risk_management': False
        }
    
    def load_data_module(self):
        """Load data loader module with correct function signature"""
        try:
            from utils.data_loader import get_comprehensive_stock_data, DATA_CONFIG
            self.modules_status['data_loader'] = True
            st.success("✅ Data loader module loaded successfully")
            return get_comprehensive_stock_data, DATA_CONFIG
        except ImportError as e:
            st.warning(f"⚠️ Using fallback data loader: {e}")
            return self._create_fallback_data_loader()
    
    def load_feature_module(self):
        """Load feature engineering module"""
        try:
            from utils.feature_engineer import create_features_enhanced, FEATURE_CONFIG
            self.modules_status['feature_engineer'] = True
            st.success("✅ Feature engineering module loaded successfully")
            return create_features_enhanced, FEATURE_CONFIG
        except ImportError as e:
            st.warning(f"⚠️ Using fallback feature engineering: {e}")
            return self._create_fallback_feature_engineer()
    
    def load_model_module(self):
        """Load model module with FIXED wrapper function usage"""
        try:
            # FIXED: Import the wrapper function that has the correct signature
            from utils.model import (
                train_models_for_selected_stocks,  # FIXED: Use wrapper function
                predict_with_ensemble_and_targets,
                save_models_optimized, 
                load_models_optimized
            )
            self.modules_status['model'] = True
            st.success("✅ Model module loaded successfully")
            return {
                'train_models_enhanced_parallel': train_models_for_selected_stocks,  # FIXED: Map to wrapper
                'predict_with_ensemble_and_targets': predict_with_ensemble_and_targets,
                'save_models_optimized': save_models_optimized,
                'load_models_optimized': load_models_optimized
            }
        except ImportError as e:
            st.warning(f"⚠️ Using fallback modeling: {e}")
            return self._create_fallback_model_functions()
    
    def load_risk_management(self):
        """Load risk management with fallbacks"""
        try:
            from utils.risk_management import (
                ComprehensiveRiskManager, RiskConfig, CorrelationAnalyzer, 
                DrawdownTracker, PositionSizer, StressTester
            )
            self.modules_status['risk_management'] = True
            st.success("✅ Risk management module loaded successfully")
            
            return {
                'ComprehensiveRiskManager': ComprehensiveRiskManager,
                'RiskConfig': RiskConfig,
                'CorrelationAnalyzer': CorrelationAnalyzer,
                'available': True
            }
            
        except ImportError as e:
            st.warning(f"⚠️ Risk management loading with fallback: {e}")
            return self._create_risk_management_fallback()
    
    def load_backtesting(self):
        """Load backtesting with fallbacks"""
        try:
            from utils.backtesting import (
                EnhancedBacktestEngine, EnhancedBacktestConfig, MLStrategy
            )
            self.modules_status['backtesting'] = True
            st.success("✅ Backtesting module loaded successfully")
            
            return {
                'EnhancedBacktestEngine': EnhancedBacktestEngine,
                'EnhancedBacktestConfig': EnhancedBacktestConfig,
                'MLStrategy': MLStrategy,
                'available': True
            }
            
        except ImportError as e:
            st.warning(f"⚠️ Backtesting loading with fallback: {e}")
            return self._create_backtesting_fallback()
    
    def load_sentiment_analysis(self):
        """Load sentiment analysis with fallbacks"""
        try:
            from utils.news_sentiment import (
                AdvancedSentimentAnalyzer, get_sentiment_for_selected_stocks,
                get_sentiment_insights
            )
            self.modules_status['sentiment'] = True
            st.success("✅ Sentiment analysis module loaded successfully")
            
            return {
                'AdvancedSentimentAnalyzer': AdvancedSentimentAnalyzer,
                'get_sentiment_for_selected_stocks': get_sentiment_for_selected_stocks,
                'get_sentiment_insights': get_sentiment_insights,
                'available': True
            }
            
        except ImportError as e:
            st.warning(f"⚠️ Sentiment analysis loading with fallback: {e}")
            return self._create_sentiment_fallback()
    
    def load_portfolio_optimization(self):
        """Load portfolio optimization with fallbacks"""
        try:
            from utils.portfolio_optimization import (
                AdvancedPortfolioOptimizer, OptimizationConfig,
                optimize_portfolio_for_selected_stocks
            )
            self.modules_status['portfolio'] = True
            st.success("✅ Portfolio optimization module loaded successfully")
            
            return {
                'AdvancedPortfolioOptimizer': AdvancedPortfolioOptimizer,
                'OptimizationConfig': OptimizationConfig,
                'optimize_portfolio_for_selected_stocks': optimize_portfolio_for_selected_stocks,
                'available': True
            }
            
        except ImportError as e:
            st.warning(f"⚠️ Portfolio optimization loading with fallback: {e}")
            return self._create_portfolio_fallback()
    
    def _create_fallback_data_loader(self):
        """Create fallback data loader"""
        import yfinance as yf
        
        def get_comprehensive_stock_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
            """Fallback data loader using yfinance"""
            stock_data = {}
            for ticker in tickers:
                try:
                    data = yf.download(ticker, period='5y', auto_adjust=True, threads=True)
                    if not data.empty:
                        stock_data[ticker] = data
                except Exception as e:
                    st.warning(f"Failed to load {ticker}: {e}")
            return stock_data
        
        DATA_CONFIG = {'fallback': True, 'cache_duration_hours': 24}
        return get_comprehensive_stock_data, DATA_CONFIG
    
    def _create_fallback_feature_engineer(self):
        """Create fallback feature engineering"""
        
        def create_features_enhanced(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
            """Fallback feature engineering"""
            try:
                features_df = df.copy()
                
                # Price-based features
                features_df['price_change'] = features_df['Close'].pct_change()
                features_df['sma_20'] = features_df['Close'].rolling(20).mean()
                features_df['sma_50'] = features_df['Close'].rolling(50).mean()
                features_df['price_to_sma20'] = features_df['Close'] / features_df['sma_20']
                
                # Momentum indicators
                delta = features_df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                features_df['rsi'] = 100 - (100 / (1 + rs))
                
                # Volume features
                if 'Volume' in df.columns:
                    features_df['volume_sma'] = df['Volume'].rolling(20).mean()
                    features_df['volume_ratio'] = df['Volume'] / features_df['volume_sma']
                
                # Volatility
                features_df['volatility_20'] = features_df['price_change'].rolling(20).std()
                
                # FIXED: Create targets for different horizons
                for horizon in ['next_week', 'next_month', 'next_quarter']:
                    if horizon == 'next_week':
                        periods = 5
                    elif horizon == 'next_month':
                        periods = 22
                    else:  # next_quarter
                        periods = 66
                    
                    future_returns = features_df['Close'].pct_change(periods).shift(-periods)
                    features_df[f'Target_{horizon}'] = (future_returns > 0.05).astype(int)
                
                # Fill NaN values
                features_df = features_df.fillna(method='forward').fillna(0)
                
                return features_df
                
            except Exception as e:
                st.warning(f"Feature engineering failed: {e}")
                return df.fillna(0)
        
        FEATURE_CONFIG = {'fallback': True}
        return create_features_enhanced, FEATURE_CONFIG
    
    def _create_fallback_model_functions(self):
        """Create fallback model functions matching FIXED signatures"""
        
        def train_models_for_selected_stocks(featured_data: Dict, selected_tickers: List[str], 
                                           investment_horizon: str = 'next_month',
                                           model_types: List[str] = None, **kwargs):
            """FIXED: Fallback model training with correct signature"""
            models = {}
            for ticker in selected_tickers:
                if ticker in featured_data:
                    # Create dummy model with proper structure
                    models[ticker] = {
                        f'{investment_horizon}_xgboost': type('DummyModel', (), {
                            'predict': lambda self, X: np.random.uniform(0, 1, len(X)),
                            'predict_proba': lambda self, X: np.column_stack([
                                np.random.uniform(0.2, 0.6, len(X)),
                                np.random.uniform(0.4, 0.8, len(X))
                            ]),
                            'model_type': 'fallback'
                        })()
                    }
            return models
        
        def predict_with_ensemble_and_targets(models: Dict, current_data: Dict, 
                                            investment_horizon: str,
                                            model_types: List[str] = None,
                                            ensemble_method: str = 'weighted_average',
                                            selected_tickers: List[str] = None, **kwargs):
            """Fallback prediction function matching original signature"""
            predictions = []
            targets = []
            
            selected_tickers = selected_tickers or list(models.keys())
            
            for ticker in selected_tickers:
                if ticker in current_data and ticker in models:
                    df = current_data[ticker]
                    current_price = df['Close'].iloc[-1] if not df.empty else 100
                    
                    # Generate reasonable predictions
                    predicted_return = np.random.uniform(-0.1, 0.15)  # -10% to 15%
                    confidence = np.random.uniform(0.5, 0.8)
                    
                    predictions.append({
                        'ticker': ticker,
                        'predicted_return': predicted_return,
                        'ensemble_confidence': confidence,
                        'signal_strength': confidence,
                        'horizon': investment_horizon
                    })
                    
                    target_price = current_price * (1 + predicted_return)
                    targets.append({
                        'ticker': ticker,
                        'current_price': current_price,
                        'target_price': target_price,
                        'percentage_change': predicted_return * 100,
                        'horizon': investment_horizon,
                        'confidence': confidence
                    })
            
            return pd.DataFrame(predictions), pd.DataFrame(targets)
        
        def save_models_optimized(models: Dict, filename: str) -> bool:
            """Fallback model saving"""
            try:
                import joblib
                joblib.dump(models, filename)
                return True
            except:
                return False
        
        def load_models_optimized(filename: str) -> Dict:
            """Fallback model loading"""
            try:
                import joblib
                return joblib.load(filename)
            except:
                return {}
        
        return {
            'train_models_enhanced_parallel': train_models_for_selected_stocks,  # FIXED: Correct mapping
            'predict_with_ensemble_and_targets': predict_with_ensemble_and_targets,
            'save_models_optimized': save_models_optimized,
            'load_models_optimized': load_models_optimized
        }
    
    def _create_risk_management_fallback(self):
        """Create fallback risk management functionality"""
        
        class FallbackRiskConfig:
            def __init__(self):
                self.max_position_size = 0.1
                self.stop_loss_pct = 0.1
                self.max_correlation = 0.8
        
        class FallbackRiskManager:
            def __init__(self, config):
                self.config = config
            
            def analyze_portfolio_risk(self, predictions_df, current_data):
                return {
                    'portfolio_var': 0.05,
                    'max_drawdown': 0.15,
                    'sharpe_ratio': 1.2,
                    'risk_score': 'Medium'
                }
        
        return {
            'ComprehensiveRiskManager': FallbackRiskManager,
            'RiskConfig': FallbackRiskConfig,
            'CorrelationAnalyzer': type('FallbackCorrelationAnalyzer', (), {}),
            'available': False
        }
    
    def _create_backtesting_fallback(self):
        """Create fallback backtesting functionality"""
        
        class FallbackBacktestConfig:
            def __init__(self):
                self.initial_capital = 500000
                self.rebalance_frequency = 'monthly'
        
        class FallbackMLStrategy:
            def __init__(self, models, config):
                self.models = models
                self.config = config
        
        class FallbackBacktestEngine:
            def __init__(self,config):
                self.config = config
            
            def run_backtest(self, data_dict, selected_tickers):
                return {
                    'total_return': 0.15,
                    'annualized_return': 0.12,
                    'max_drawdown': 0.08,
                    'sharpe_ratio': 1.5,
                    'win_rate': 0.65
                }
        
        return {
            'EnhancedBacktestEngine': FallbackBacktestEngine,
            'EnhancedBacktestConfig': FallbackBacktestConfig,
            'MLStrategy': FallbackMLStrategy,
            'available': False
        }
    
    def _create_sentiment_fallback(self):
        """Create fallback sentiment analysis"""
        
        def get_sentiment_for_selected_stocks(tickers, api_key=None, days=7):
            return {ticker: np.random.uniform(-0.5, 0.5) for ticker in tickers}
        
        def get_sentiment_insights(tickers, api_key=None):
            sentiment_scores = get_sentiment_for_selected_stocks(tickers)
            return {
                'overall_sentiment': np.mean(list(sentiment_scores.values())),
                'positive_stocks': len([s for s in sentiment_scores.values() if s > 0.1]),
                'negative_stocks': len([s for s in sentiment_scores.values() if s < -0.1]),
                'sentiment_distribution': sentiment_scores
            }
        
        return {
            'AdvancedSentimentAnalyzer': type('FallbackSentimentAnalyzer', (), {}),
            'get_sentiment_for_selected_stocks': get_sentiment_for_selected_stocks,
            'get_sentiment_insights': get_sentiment_insights,
            'available': False
        }
    
    def _create_portfolio_fallback(self):
        """Create fallback portfolio optimization"""
        
        def optimize_portfolio_for_selected_stocks(predictions_df, returns_data, selected_tickers):
            # Equal weight fallback
            n_stocks = len(selected_tickers)
            weights = [1/n_stocks] * n_stocks
            
            return {
                'weights': dict(zip(selected_tickers, weights)),
                'expected_return': 0.08,
                'volatility': 0.15,
                'sharpe_ratio': 0.53,
                'optimization_method': 'equal_weight_fallback'
            }
        
        return {
            'AdvancedPortfolioOptimizer': type('FallbackOptimizer', (), {}),
            'OptimizationConfig': type('FallbackConfig', (), {}),
            'optimize_portfolio_for_selected_stocks': optimize_portfolio_for_selected_stocks,
            'available': False
        }

# Initialize fixed module loader
module_loader = FixedModuleLoader()

# Load all modules with proper error handling
get_comprehensive_stock_data, DATA_CONFIG = module_loader.load_data_module()
create_features_enhanced, FEATURE_CONFIG = module_loader.load_feature_module()
model_functions = module_loader.load_model_module()
risk_components = module_loader.load_risk_management()
backtest_components = module_loader.load_backtesting()
sentiment_components = module_loader.load_sentiment_analysis()
portfolio_components = module_loader.load_portfolio_optimization()

# Extract functions - FIXED MAPPING
train_models_enhanced_parallel = model_functions['train_models_enhanced_parallel']  # Now correctly mapped to wrapper
predict_with_ensemble_and_targets = model_functions['predict_with_ensemble_and_targets']
save_models_optimized = model_functions['save_models_optimized']
load_models_optimized = model_functions['load_models_optimized']

# Store availability flags
MODULES_STATUS = module_loader.modules_status
RISK_MANAGEMENT_AVAILABLE = risk_components['available']
BACKTESTING_AVAILABLE = backtest_components['available']
SENTIMENT_AVAILABLE = sentiment_components['available']
PORTFOLIO_OPTIMIZATION_AVAILABLE = portfolio_components['available']

# ==================== ENHANCED CSS STYLING ====================

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .prediction-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .risk-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .portfolio-summary {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ENHANCED PRICE FORECASTING COMPONENTS ====================

class EnhancedPriceForecaster:
    """Enhanced price forecasting with confidence intervals and scenarios"""
    
    def __init__(self, horizon_days: int = 30):
        self.horizon_days = horizon_days
        
    def generate_enhanced_forecast(self, ticker: str, data: pd.DataFrame) -> Dict:
        """Generate comprehensive price forecast"""
        try:
            current_price = data['Close'].iloc[-1]
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Monte Carlo simulation for different scenarios
            scenarios = {}
            scenario_configs = {
                'bullish': {'drift': 0.12, 'vol_mult': 0.8},
                'base_case': {'drift': 0.08, 'vol_mult': 1.0},
                'bearish': {'drift': 0.02, 'vol_mult': 1.2}
            }
            
            for scenario_name, config in scenario_configs.items():
                drift = config['drift'] / 252  # Daily drift
                vol = volatility * config['vol_mult'] / np.sqrt(252)  # Daily volatility
                
                # Simple geometric Brownian motion
                random_shocks = np.random.normal(0, 1, self.horizon_days)
                daily_returns = drift + vol * random_shocks
                price_path = current_price * np.cumprod(1 + daily_returns)
                target_price = price_path[-1]
                
                scenarios[scenario_name] = {
                    'target_price': target_price,
                    'return_pct': ((target_price / current_price) - 1) * 100,
                    'probability': 0.6 if scenario_name == 'base_case' else 0.2,
                    'price_path': price_path
                }
            
            # Generate recommendation
            base_return = scenarios['base_case']['return_pct']
            recommendation = {
                'action': 'BUY' if base_return > 5 else 'HOLD' if base_return > -5 else 'SELL',
                'confidence': min(abs(base_return) / 10, 1.0),
                'reasoning': f"Expected {base_return:.1f}% return over {self.horizon_days} days"
            }
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'scenarios': scenarios,
                'recommendation': recommendation,
                'forecast_horizon_days': self.horizon_days,
                'volatility': volatility
            }
            
        except Exception as e:
            # Fallback forecast
            current_price = data['Close'].iloc[-1] if not data.empty else 100
            return {
                'ticker': ticker,
                'current_price': current_price,
                'scenarios': {
                    'base_case': {
                        'target_price': current_price * 1.05,
                        'return_pct': 5.0,
                        'probability': 0.6
                    }
                },
                'recommendation': {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reasoning': 'Fallback forecast due to insufficient data'
                },
                'forecast_horizon_days': self.horizon_days,
                'volatility': 0.20
            }

# ==================== COMPREHENSIVE INTERFACE COMPONENTS ====================

def create_enhanced_stock_selection_interface() -> List[str]:
    """Create comprehensive stock selection interface with extensive categories"""
    with st.sidebar:
        st.header("📊 Advanced Stock Selection Hub")
        
        # COMPREHENSIVE STOCK CATEGORIES - INDIAN MARKET
        stock_categories = {
            # ==================== FINANCIAL SERVICES ====================
            "🏦 Banking - Large Cap": [
                "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS",
                "INDUSINDBK.NS", "BANDHANBNK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "PNB.NS"
            ],
            
            "🏦 Banking - Mid & Small Cap": [
                "RBLBANK.NS", "SOUTHBANK.NS", "CITYUNIONBK.NS", "DCBBANK.NS", "KARURBANK.NS",
                "TMVIL.NS", "SURYODAY.NS", "EQUITASBNK.NS", "UJJIVANSFB.NS", "AUBANK.NS"
            ],
            
            "💰 NBFCs & Financial Services": [
                "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCAMC.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
                "CHOLAFIN.NS", "PFC.NS", "RECLTD.NS", "LICHSGFIN.NS", "IIFL.NS"
            ],
            
            "📈 Insurance": [
                "HDFCLIFE.NS", "ICICIPRULI.NS", "SBILIFE.NS", "MAXLIFEINS.NS", "LICI.NS",
                "ICICIGI.NS", "HDFCERGO.NS", "NIACL.NS", "UIICL.NS", "ORIENTALINS.NS"
            ],
            
            "🏛️ Capital Markets": [
                "HDFCAMC.NS", "NIFTYBEES.NS", "CDSL.NS", "BSE.NS", "MCX.NS",
                "ANGELONE.NS", "MOTILALOFS.NS", "EDELWEISS.NS", "IIFLSEC.NS", "SHAREKHAN.NS"
            ],
            
            # ==================== TECHNOLOGY ====================
            "💻 IT Services - Large Cap": [
                "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
                "LTI.NS", "MINDTREE.NS", "MPHASIS.NS", "PERSISTENT.NS", "LTTS.NS"
            ],
            
            "💻 IT Services - Mid Cap": [
                "COFORGE.NS", "SONATSOFTW.NS", "CYIENT.NS", "HEXAWARE.NS", "KPITTECH.NS",
                "INTELLECT.NS", "RAMCOCEM.NS", "NIITTECH.NS", "ECLERX.NS", "ZENSAR.NS"
            ],
            
            "📱 Software Products": [
                "TATAELXSI.NS", "DATAPATTNS.NS", "NEWGEN.NS", "SUBEXLTD.NS", "QUICKHEAL.NS",
                "SAKSOFT.NS", "MASTEK.NS", "MINDACORP.NS", "NETWEB.NS", "RATEGAIN.NS"
            ],
            
            "🌐 Telecom & Internet": [
                "BHARTIARTL.NS", "RJIO.NS", "VODAIDEA.NS", "TTML.NS", "RAILTEL.NS",
                "GTLINFRA.NS", "RCOM.NS", "TATACOMM.NS", "MAHANAGAR.NS", "TEJAS.NS"
            ],
            
            # ==================== CONSUMER SECTORS ====================
            "🛍️ FMCG - Large Cap": [
                "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS",
                "COLPAL.NS", "MARICO.NS", "EMAMILTD.NS", "VBL.NS", "TATACONSUM.NS"
            ],
            
            "🛍️ FMCG - Mid Cap": [
                "RELAXO.NS", "VGUARD.NS", "BAJAJCON.NS", "GILLETTE.NS", "HONASA.NS",
                "PRATAAP.NS", "JYOTHYLAB.NS", "RADICO.NS", "CCL.NS", "BAJAJHIND.NS"
            ],
            
            "🏠 Consumer Durables": [
                "WHIRLPOOL.NS", "VOLTAS.NS", "BLUESTAR.NS", "CROMPTON.NS", "HAVELLS.NS",
                "ORIENTELEC.NS", "AMBER.NS", "DIXON.NS", "RAJESHEXPO.NS", "TTK.NS"
            ],
            
            "👗 Textiles & Apparel": [
                "ADITIABIRLA.NS", "WELCORP.NS", "VARDHMAN.NS", "ARVIND.NS", "RAYMOND.NS",
                "SPANDANA.NS", "TRIDENT.NS", "WELSPUN.NS", "INDHOTEL.NS", "KPR.NS"
            ],
            
            "🍕 Food & Beverages": [
                "JUBLFOOD.NS", "DEVYANI.NS", "WESTLIFE.NS", "SAPPHIRE.NS", "ZOMATO.NS",
                "SWIGGY.NS", "VARUN.NS", "USLIND.NS", "UFLEX.NS", "BALRAMCHIN.NS"
            ],
            
            # ==================== HEALTHCARE & PHARMA ====================
            "💊 Pharmaceuticals - Large Cap": [
                "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS",
                "BIOCON.NS", "AUROPHARM.NS", "CADILAHC.NS", "TORNTPHARM.NS", "AUROPHARMA.NS"
            ],
            
            "💊 Pharmaceuticals - Mid Cap": [
                "ALKEM.NS", "GLENMARK.NS", "LALPATHLAB.NS", "METROPOLIS.NS", "PFIZER.NS",
                "ABBOTINDIA.NS", "SANOFI.NS", "GLAXO.NS", "NOVARTIS.NS", "MERCK.NS"
            ],
            
            "🏥 Hospitals & Diagnostics": [
                "APOLLOHOSP.NS", "FORTIS.NS", "MAXHEALTHC.NS", "NARAYANHRT.NS", "RHIM.NS",
                "LALPATHLAB.NS", "METROPOLIS.NS", "THYROCARE.NS", "HESTERBIO.NS", "ISHAN.NS"
            ],
            
            "🧬 Biotechnology": [
                "BIOCON.NS", "BIOCORP.NS", "SYNGENE.NS", "SEQUENT.NS", "ASTER.NS",
                "KRSNAA.NS", "VIJAYA.NS", "STRIDES.NS", "GRANULES.NS", "REDDY.NS"
            ],
            
            # ==================== INDUSTRIAL & MANUFACTURING ====================
            "🏭 Engineering - Large Cap": [
                "LT.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS", "CUMMINSIND.NS",
                "THERMAX.NS", "KECL.NS", "POWERGRID.NS", "SAIL.NS", "GAIL.NS"
            ],
            
            "🏭 Engineering - Mid Cap": [
                "CARBORUNIV.NS", "SCHAEFFLER.NS", "TIMKEN.NS", "SKCIL.NS", "GRINDWELL.NS",
                "IFBIND.NS", "RATNAMANI.NS", "APCOTEXIND.NS", "SKFINDIA.NS", "AKZOINDIA.NS"
            ],
            
            "⚡ Power & Utilities": [
                "POWERGRID.NS", "NTPC.NS", "NHPC.NS", "SJVN.NS", "TATAPOWER.NS",
                "ADANIGR.NS", "ADANIGREEN.NS", "SUZLON.NS", "RPOWER.NS", "TORNTPOWER.NS"
            ],
            
            "🏗️ Infrastructure": [
                "LTTS.NS", "HCC.NS", "IRCON.NS", "RVNL.NS", "NBCC.NS",
                "GMRINFRA.NS", "GICRE.NS", "CONCOR.NS", "ADANIPORTS.NS", "JSWSTEEL.NS"
            ],
            
            # ==================== MATERIALS & COMMODITIES ====================
            "🏗️ Cement": [
                "ULTRACEMCO.NS", "GRASIM.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS",
                "DALMIACONT.NS", "JKCEMENT.NS", "RAMCOCEM.NS", "HEIDELBERG.NS", "JK.NS"
            ],
            
            "⚙️ Steel & Metals": [
                "TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS", "JINDALSTEL.NS", "NMDC.NS",
                "VEDL.NS", "HINDALCO.NS", "NALCO.NS", "RATNAMANI.NS", "MOIL.NS"
            ],
            
            "🧪 Chemicals": [
                "PIDILITIND.NS", "ASIANPAINT.NS", "BERGER.NS", "AKZOINDIA.NS", "KANSAINER.NS",
                "BALRAMCHIN.NS", "TATACHEM.NS", "GHCL.NS", "ALKYLAMINE.NS", "DEEPAKFERT.NS"
            ],
            
            "🌾 Fertilizers & Agri": [
                "UPL.NS", "PIIND.NS", "COROMANDEL.NS", "CHAMBLFERT.NS", "MADRASFER.NS",
                "GSFC.NS", "ZUARI.NS", "RCF.NS", "MANGALAM.NS", "NFL.NS"
            ],
            
            # ==================== ENERGY & RESOURCES ====================
            "⛽ Oil & Gas - Large Cap": [
                "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HPCL.NS",
                "GAIL.NS", "PETRONET.NS", "OIL.NS", "MRF.NS", "CASTROLIND.NS"
            ],
            
            "⛽ Oil & Gas - Mid Cap": [
                "AEGISCHEM.NS", "DEEPAKFERT.NS", "GSPL.NS", "HINDPETRO.NS", "MRPL.NS",
                "CHENNPETRO.NS", "TIDEWATER.NS", "ABAN.NS", "SELAN.NS", "BGRENERGY.NS"
            ],
            
            "🔋 Renewable Energy": [
                "ADANIGREEN.NS", "SUZLON.NS", "ORIENTGREEN.NS", "WEBSOL.NS", "BOROSIL.NS",
                "CLEANSCIEN.NS", "VOLTAMP.NS", "RATTAN.NS", "INOXWIND.NS", "JPPOWER.NS"
            ],
            
            "⚡ Power Equipment": [
                "BHEL.NS", "CROMPTON.NS", "HAVELLS.NS", "POLYCAB.NS", "ORIENTELEC.NS",
                "KECL.NS", "SIEMENS.NS", "ABB.NS", "SCHNEIDER.NS", "CUMMINSIND.NS"
            ],
            
            # ==================== AUTOMOBILE SECTOR ====================
            "🚗 Automobiles - Large Cap": [
                "MARUTI.NS", "HYUNDAI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS",
                "HEROMOTOCO.NS", "TVSMOTORS.NS", "ASHOKLEY.NS", "EICHERMOT.NS", "FORCEMOT.NS"
            ],
            
            "🚗 Auto Components": [
                "BOSCHLTD.NS", "MOTHERSON.NS", "BALKRISIND.NS", "MRF.NS", "APOLLOTYRE.NS",
                "CEATLTD.NS", "BHARAT.NS", "EXIDEIND.NS", "AMARA.NS", "SUNDRMFAST.NS"
            ],
            
            "🏍️ Two Wheelers": [
                "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "TVSMOTORS.NS", "EICHERMOT.NS", "RAJAJIAUTO.NS",
                "MINDACORP.NS", "SUPRAJIT.NS", "MUNJALSHOW.NS", "SCHAEFFLER.NS", "ENDURANCE.NS"
            ],
            
            # ==================== REAL ESTATE & CONSTRUCTION ====================
            "🏘️ Real Estate - Large Cap": [
                "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "BRIGADE.NS", "PRESTIGE.NS",
                "SOBHA.NS", "PHOENIXLTD.NS", "KOLTE.NS", "MAHLIFE.NS", "SUNTECK.NS"
            ],
            
            "🏘️ Real Estate - Mid Cap": [
                "LODHA.NS", "MAHINDRA.NS", "RAYMOND.NS", "PURAVANKARA.NS", "ASHOKA.NS",
                "MAHLOG.NS", "ANANTRAJ.NS", "OMAXE.NS", "UNITECH.NS", "PARSVNATH.NS"
            ],
            
            "🏗️ Construction": [
                "LT.NS", "HCC.NS", "IRCON.NS", "RVNL.NS", "NBCC.NS",
                "KNR.NS", "PNC.NS", "SADBHAV.NS", "JAIBALAJI.NS", "WELCORP.NS"
            ],
            
            # ==================== MEDIA & ENTERTAINMENT ====================
            "📺 Media & Entertainment": [
                "ZEEL.NS", "PVR.NS", "INOXLEISUR.NS", "BALAJITELE.NS", "SAREGAMA.NS",
                "TIPS.NS", "EROS.NS", "NETWORK18.NS", "JAINIRRIG.NS", "NAZARA.NS"
            ],
            
            "📰 Publishing & Digital": [
                "HT.NS", "DBCORP.NS", "SANDESH.NS", "JAGRAN.NS", "NAVNEET.NS",
                "JUSTDIAL.NS", "INDIAMART.NS", "NAUKRI.NS", "MATRIMONY.NS", "ZOMATO.NS"
            ],
            
            # ==================== AVIATION & LOGISTICS ====================
            "✈️ Aviation": [
                "SPICEJET.NS", "INDIGO.NS", "JETAIRWAYS.NS", "GMR.NS", "GVK.NS",
                "AAI.NS", "MAHINDRA.NS", "BLUEDART.NS", "PATEL.NS", "SPANDANA.NS"
            ],
            
            "🚛 Logistics & Transportation": [
                "CONCOR.NS", "GATI.NS", "BLUEDART.NS", "VRL.NS", "TCI.NS",
                "MAHLOG.NS", "SICAL.NS", "ALLCARGO.NS", "SNOWMAN.NS", "MAHINDRA.NS"
            ],
            
            # ==================== RETAIL & E-COMMERCE ====================
            "🛒 Retail - Large Cap": [
                "RELIANCE.NS", "AVENUELSX.NS", "SHOPSSTOP.NS", "TRENTLTD.NS", "VMART.NS",
                "FRETAIL.NS", "SPENCERS.NS", "LANDMARK.NS", "TITAN.NS", "KALYAN.NS"
            ],
            
            "🛒 E-Commerce & Digital": [
                "ZOMATO.NS", "NYKAA.NS", "POLICYBZR.NS", "CARDUNIA.NS", "PAYTM.NS",
                "EASEMYTRIP.NS", "MATRIMONY.NS", "JUSTDIAL.NS", "INDIAMART.NS", "NAUKRI.NS"
            ],
            
            # ==================== INTERNATIONAL MARKETS ====================
            "🌍 US Tech Giants": [
                "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NFLX", "NVDA", "CRM", "ORCL"
            ],
            
            "🌍 US Blue Chips": [
                "JPM", "JNJ", "PG", "KO", "PEP", "WMT", "V", "MA", "HD", "DIS"
            ],
            
            "🌍 Global Banks": [
                "JPM", "BAC", "WFC", "GS", "MS", "C", "BCS", "DB", "CS", "UBS"
            ],
            
            "🌍 European Stocks": [
                "ASML", "SAP", "NESN.SW", "NOVN.SW", "ROG.SW", "MC.PA", "LVMH.PA", "OR.PA", "SAN.PA", "BNP.PA"
            ],
            
            # ==================== CRYPTOCURRENCY & FINTECH ====================
            "₿ Cryptocurrency ETFs": [
                "BITO", "BTCC.TO", "ETHE", "COIN", "MARA", "RIOT", "HUT.TO", "BITF.TO", "SQ", "PYPL"
            ],
            
            "💳 Fintech & Payments": [
                "PAYTM.NS", "POLICYBZR.NS", "V", "MA", "PYPL", "SQ", "ADYEN.AS", "TEAM", "CRM", "NOW"
            ],
            
            # ==================== COMMODITIES & PRECIOUS METALS ====================
            "🥇 Gold & Precious Metals": [
                "GOLDBEES.NS", "GOLDGUINEA.NS", "TITAN.NS", "KALYAN.NS", "THANGAMAY.NS",
                "GLD", "SLV", "PALL", "PPLT", "IAU"
            ],
            
            "🛢️ Commodity ETFs": [
                "USO", "UNG", "DBA", "JJG", "JJU", "JJT", "JJS", "JJN", "JJC", "JO"
            ],
            
            # ==================== THEMATIC INVESTMENTS ====================
            "🤖 AI & Technology": [
                "TCS.NS", "INFY.NS", "WIPRO.NS", "NVDA", "AMD", "INTC", "TSM", "QCOM", "AVGO", "MU"
            ],
            
            "🌱 ESG & Sustainability": [
                "ADANIGREEN.NS", "SUZLON.NS", "TATAPOWER.NS", "ICLN", "QCLN", "PBW", "FAN", "SMOG", "CNRG", "ACES"
            ],
            
            "🏥 Healthcare Innovation": [
                "SUNPHARMA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "JNJ", "PFE", "MRNA", "BNTX", "GILD", "AMGN", "BIIB"
            ],
            
            "🚀 Space & Defense": [
                "HAL.NS", "BEL.NS", "BEML.NS", "BA", "LMT", "RTX", "NOC", "GD", "SPCE", "UFO"
            ],
            
            # ==================== DIVIDEND ARISTOCRATS ====================
            "💰 High Dividend Yield - Indian": [
                "SBIN.NS", "COALINDIA.NS", "NMDC.NS", "IOC.NS", "ONGC.NS",
                "VEDL.NS", "HINDALCO.NS", "ITC.NS", "POWERGRID.NS", "NTPC.NS"
            ],
            
            "💰 Dividend Aristocrats - Global": [
                "JNJ", "PG", "KO", "PEP", "WMT", "MMM", "ABT", "CL", "GD", "SYY"
            ],
            
            # ==================== SMALL CAP GEMS ====================
            "💎 Small Cap - High Growth": [
                "DIXON.NS", "CLEANSCIEN.NS", "ROSSARI.NS", "HAPPSTMNDS.NS", "ROUTE.NS",
                "LAXMIMACH.NS", "TIINDIA.NS", "KFINTECH.NS", "DEVYANI.NS", "EASEMYTRIP.NS"
            ],
            
            "💎 Micro Cap - Emerging": [
                "AMBER.NS", "WEBELSOLAR.NS", "MISHTANN.NS", "RTNINDIA.NS", "INVENTURE.NS",
                "SHREYAS.NS", "UGAR.NS", "CENTUM.NS", "DPSCLTD.NS", "VSSL.NS"
            ],
            
            # ==================== SECTOR ETFs ====================
            "📊 Indian Sector ETFs": [
                "BANKBEES.NS", "ITBEES.NS", "PHARMBEES.NS", "AUTOBEES.NS", "PSUBNKBEES.NS",
                "PVTBNKBEES.NS", "INFRABEES.NS", "FMCGBEES.NS", "METALBEES.NS", "ENERGYBEES.NS"
            ],
            
            "📊 Global Sector ETFs": [
                "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE"
            ],
            
            # ==================== REGIONAL & COUNTRY ETFs ====================
            "🌏 Emerging Markets": [
                "EEM", "VWO", "IEMG", "EWZ", "INDA", "FXI", "EWT", "EWY", "RSX", "EPOL"
            ],
            
            "🌍 Developed Markets": [
                "EFA", "VEA", "IEFA", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWS", "EWP"
            ]
        }
        
        # Enhanced selection interface
        st.subheader("🎯 Selection Method")
        
        selection_method = st.radio(
            "Choose your selection approach:",
            [
                "📂 Browse by Categories", 
                "🔍 Search by Keywords", 
                "📝 Manual Entry", 
                "📁 Upload CSV File",
                "⭐ Popular Picks",
                "🎲 Random Portfolio Generator"
            ],
            help="Different ways to build your stock portfolio"
        )
        
        selected_tickers = []
        
        # ==================== CATEGORY BROWSING ====================
        if selection_method == "📂 Browse by Categories":
            st.subheader("📊 Stock Categories")
            
            # Market segment filter
            market_segments = st.multiselect(
                "🏢 Market Segments:",
                ["🇮🇳 Indian Stocks", "🇺🇸 US Stocks", "🌍 International", "📊 ETFs", "₿ Crypto-related"],
                default=["🇮🇳 Indian Stocks"],
                help="Filter categories by market segment"
            )
            
            # Filter categories based on market segment
            filtered_categories = {}
            
            for category, stocks in stock_categories.items():
                include_category = False
                
                if "🇮🇳 Indian Stocks" in market_segments:
                    if any(stock.endswith('.NS') for stock in stocks):
                        include_category = True
                
                if "🇺🇸 US Stocks" in market_segments:
                    if any(not stock.endswith('.NS') and not any(ext in stock for ext in ['.PA', '.SW', '.TO', '.AS']) for stock in stocks):
                        include_category = True
                
                if "🌍 International" in market_segments:
                    if any(any(ext in stock for ext in ['.PA', '.SW', '.TO', '.AS']) for stock in stocks):
                        include_category = True
                
                if "📊 ETFs" in market_segments:
                    if "ETF" in category or "BEES" in category or any("BEES" in stock for stock in stocks):
                        include_category = True
                
                if "₿ Crypto-related" in market_segments:
                    if "Cryptocurrency" in category or "Fintech" in category:
                        include_category = True
                
                if include_category:
                    filtered_categories[category] = stocks
            
            # Category selection
            selected_categories = st.multiselect(
                "📋 Select Categories:",
                list(filtered_categories.keys()),
                help="Choose one or more categories to explore"
            )
            
            # Stock selection within categories
            for category in selected_categories:
                with st.expander(f"📈 {category} ({len(filtered_categories[category])} stocks)"):
                    
                    # Select all/none buttons
                    col1, col2 = st.columns(2)
                    select_all_key = f"select_all_{category}"
                    select_none_key = f"select_none_{category}"
                    
                    with col1:
                        if st.button(f"✅ Select All", key=select_all_key):
                            st.session_state[f"stocks_{category}"] = filtered_categories[category]
                    
                    with col2:
                        if st.button(f"❌ Clear All", key=select_none_key):
                            st.session_state[f"stocks_{category}"] = []
                    
                    # Individual stock selection
                    selected_stocks = st.multiselect(
                        f"Stocks from {category}:",
                        filtered_categories[category],
                        default=st.session_state.get(f"stocks_{category}", filtered_categories[category][:3]),
                        key=f"multiselect_{category}",
                        help=f"Select individual stocks from {category}"
                    )
                    
                    selected_tickers.extend(selected_stocks)
            
            # Remove duplicates while preserving order
            selected_tickers = list(dict.fromkeys(selected_tickers))
        
        # ==================== KEYWORD SEARCH ====================
        elif selection_method == "🔍 Search by Keywords":
            st.subheader("🔎 Smart Stock Search")
            
            # Create searchable stock database
            all_stocks = []
            for category, stocks in stock_categories.items():
                for stock in stocks:
                    all_stocks.append({
                        'ticker': stock,
                        'category': category,
                        'market': 'Indian' if stock.endswith('.NS') else 'International',
                        'searchable': f"{stock} {category}".lower()
                    })
            
            # Search input
            search_query = st.text_input(
                "🔍 Search stocks, sectors, or companies:",
                placeholder="e.g., 'banking', 'tech', 'RELIANCE', 'dividend'",
                help="Search by ticker, company name, or sector"
            )
            
            # Additional filters
            col1, col2 = st.columns(2)
            with col1:
                market_filter = st.selectbox(
                    "🌍 Market:",
                    ["All Markets", "Indian Stocks (.NS)", "International Stocks"],
                    help="Filter by market"
                )
            
            with col2:
                max_results = st.slider(
                    "📊 Max Results:",
                    min_value=10,
                    max_value=100,
                    value=20,
                    help="Maximum number of search results"
                )
            
            # Perform search
            if search_query:
                search_terms = search_query.lower().split()
                matched_stocks = []
                
                for stock_info in all_stocks:
                    # Check if all search terms match
                    if all(term in stock_info['searchable'] for term in search_terms):
                        # Apply market filter
                        if market_filter == "All Markets":
                            matched_stocks.append(stock_info)
                        elif market_filter == "Indian Stocks (.NS)" and stock_info['ticker'].endswith('.NS'):
                            matched_stocks.append(stock_info)
                        elif market_filter == "International Stocks" and not stock_info['ticker'].endswith('.NS'):
                            matched_stocks.append(stock_info)
                
                # Display results
                if matched_stocks:
                    st.success(f"Found {len(matched_stocks)} matching stocks")
                    
                    # Limit results
                    display_stocks = matched_stocks[:max_results]
                    
                    # Create selection interface
                    search_results = []
                    for stock_info in display_stocks:
                        search_results.append(f"{stock_info['ticker']} - {stock_info['category']}")
                    
                    selected_results = st.multiselect(
                        "📋 Select from search results:",
                        search_results,
                        help="Choose stocks from search results"
                    )
                    
                    # Extract tickers
                    selected_tickers = [result.split(' - ')[0] for result in selected_results]
                    
                else:
                    st.info(f"No stocks found matching '{search_query}'. Try different keywords.")
        
        # ==================== MANUAL ENTRY ====================
        elif selection_method == "📝 Manual Entry":
            st.subheader("✍️ Manual Stock Entry")
            
            # Text area for manual input
            manual_input = st.text_area(
                "📝 Enter stock symbols (one per line):",
                value="RELIANCE.NS\nTCS.NS\nHDFCBANK.NS\nINFY.NS\nHINDUNILVR.NS",
                height=200,
                help="Enter stock symbols, one per line. Use .NS suffix for Indian stocks."
            )
            
            # Format options
            col1, col2 = st.columns(2)
            with col1:
                auto_format = st.checkbox(
                    "🔧 Auto-format Indian stocks",
                    value=True,
                    help="Automatically add .NS suffix to Indian stock symbols"
                )
            
            with col2:
                validate_symbols = st.checkbox(
                    "✅ Validate symbols",
                    value=True,
                    help="Check if symbols are in our database"
                )
            
            # Process manual input
            if manual_input:
                raw_tickers = [ticker.strip().upper() for ticker in manual_input.split('\n') if ticker.strip()]
                
                # Auto-format Indian stocks
                if auto_format:
                    formatted_tickers = []
                    for ticker in raw_tickers:
                        if not any(suffix in ticker for suffix in ['.NS', '.BO', '.PA', '.SW', '.TO', '.AS']):
                            # Check if it's likely an Indian stock
                            indian_indicators = ['RELIANCE', 'TCS', 'HDFC', 'ICICI', 'INFY', 'WIPRO', 'SBI', 'BHARTI']
                            if any(indicator in ticker for indicator in indian_indicators) or len(ticker) <= 12:
                                formatted_tickers.append(f"{ticker}.NS")
                            else:
                                formatted_tickers.append(ticker)
                        else:
                            formatted_tickers.append(ticker)
                    selected_tickers = formatted_tickers
                else:
                    selected_tickers = raw_tickers
                
                # Validate symbols
                if validate_symbols:
                    all_available_stocks = []
                    for stocks in stock_categories.values():
                        all_available_stocks.extend(stocks)
                    
                    valid_tickers = []
                    invalid_tickers = []
                    
                    for ticker in selected_tickers:
                        if ticker in all_available_stocks:
                            valid_tickers.append(ticker)
                        else:
                            invalid_tickers.append(ticker)
                    
                    if invalid_tickers:
                        st.warning(f"⚠️ Unrecognized symbols: {', '.join(invalid_tickers)}")
                        st.info("💡 These stocks will still be included but may not have all features available")
                    
                    if valid_tickers:
                        st.success(f"✅ Recognized {len(valid_tickers)} symbols from our database")
        
        # ==================== CSV UPLOAD ====================
        elif selection_method == "📁 Upload CSV File":
            st.subheader("📁 CSV File Upload")
            
            uploaded_file = st.file_uploader(
                "📤 Upload CSV file with stock symbols:",
                type=['csv', 'txt'],
                help="CSV should have symbols in first column or a column named 'symbol', 'ticker', or 'stock'"
            )
            
            if uploaded_file is not None:
                try:
                    # Read CSV
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
                    
                    # Show preview
                    with st.expander("👀 File Preview"):
                        st.dataframe(df.head(10))
                    
                    # Column selection
                    symbol_column = st.selectbox(
                        "📊 Select column containing stock symbols:",
                        df.columns.tolist(),
                        index=0,
                        help="Choose which column contains the stock symbols"
                    )
                    
                    # Extract symbols
                    if symbol_column in df.columns:
                        uploaded_tickers = df[symbol_column].dropna().astype(str).str.strip().str.upper().tolist()
                        
                        # Remove duplicates
                        uploaded_tickers = list(dict.fromkeys(uploaded_tickers))
                        
                        st.info(f"📈 Found {len(uploaded_tickers)} unique symbols in '{symbol_column}' column")
                        
                        # Option to filter/modify
                        col1, col2 = st.columns(2)
                        with col1:
                            max_symbols = st.slider(
                                "📊 Maximum symbols to use:",
                                min_value=1,
                                max_value=min(100, len(uploaded_tickers)),
                                value=min(20, len(uploaded_tickers)),
                                help="Limit number of symbols for performance"
                            )
                        
                        with col2:
                            add_ns_suffix = st.checkbox(
                                "🇮🇳 Add .NS for Indian stocks",
                                value=True,
                                help="Automatically add .NS suffix"
                            )
                        
                        # Process tickers
                        selected_tickers = uploaded_tickers[:max_symbols]
                        
                        if add_ns_suffix:
                            processed_tickers = []
                            for ticker in selected_tickers:
                                if not any(suffix in ticker for suffix in ['.NS', '.BO', '.PA', '.SW', '.TO', '.AS']):
                                    processed_tickers.append(f"{ticker}.NS")
                                else:
                                    processed_tickers.append(ticker)
                            selected_tickers = processed_tickers
                        
                        # Show final selection
                        with st.expander("📋 Final Stock Selection"):
                            st.write(selected_tickers[:20])  # Show first 20
                            if len(selected_tickers) > 20:
                                st.info(f"... and {len(selected_tickers) - 20} more stocks")
                    
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
                    st.info("💡 Please ensure your CSV file is properly formatted")
        
        # ==================== POPULAR PICKS ====================
        elif selection_method == "⭐ Popular Picks":
            st.subheader("⭐ Popular Investment Themes")
            
            popular_themes = {
                "🚀 Top Performers 2024": [
                    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                    "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS"
                ],
                "💎 Value Investing": [
                    "SBIN.NS", "COALINDIA.NS", "NMDC.NS", "IOC.NS", "ONGC.NS",
                    "VEDL.NS", "HINDALCO.NS", "POWERGRID.NS", "NTPC.NS", "SAIL.NS"
                ],
                "📈 Growth Stocks": [
                    "TCS.NS", "INFY.NS", "HDFCBANK.NS", "KOTAKBANK.NS", "NESTLEIND.NS",
                    "DIXON.NS", "CLEANSCIEN.NS", "HAPPSTMNDS.NS", "ZOMATO.NS", "NYKAA.NS"
                ],
                "💰 Dividend Champions": [
                    "COALINDIA.NS", "SBIN.NS", "NMDC.NS", "IOC.NS", "ONGC.NS",
                    "VEDL.NS", "POWERGRID.NS", "NTPC.NS", "ITC.NS", "GAIL.NS"
                ],
                "🏦 Banking Powerhouse": [
                    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS",
                    "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCAMC.NS", "PNB.NS"
                ],
                "💻 Tech Titans": [
                    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
                    "LTI.NS", "MINDTREE.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS"
                ],
                "🌐 Global Diversified": [
                    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "RELIANCE.NS", "TCS.NS",
                    "HDFCBANK.NS", "ASML", "TSM"
                ],
                "🌱 ESG Leaders": [
                    "ADANIGREEN.NS", "TATAPOWER.NS", "SUZLON.NS", "NESTLEIND.NS", "UNILEVER.NS",
                    "ICLN", "QCLN", "PBW", "HDFCBANK.NS", "TCS.NS"
                ],
                "🏥 Healthcare Heroes": [
                    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "BIOCON.NS",
                    "JNJ", "PFE", "MRNA", "ABBOTT.NS", "FORTIS.NS"
                ],
                "🔮 Future Tech": [
                    "TCS.NS", "INFY.NS", "NVDA", "AMD", "TSLA", "ZOMATO.NS",
                    "PAYTM.NS", "NYKAA.NS", "HAPPSTMNDS.NS", "COFORGE.NS"
                ]
            }
            
            selected_theme = st.selectbox(
                "🎯 Choose an investment theme:",
                list(popular_themes.keys()),
                help="Pre-curated portfolios based on popular investment strategies"
            )
            
            # Display theme details
            theme_stocks = popular_themes[selected_theme]
            
            st.info(f"📊 **{selected_theme}** contains {len(theme_stocks)} carefully selected stocks")
            
            # Show stocks in theme
            with st.expander(f"👀 View all stocks in {selected_theme}"):
                cols = st.columns(3)
                for i, stock in enumerate(theme_stocks):
                    with cols[i % 3]:
                        st.write(f"• {stock}")
            
            # Selection options
            col1, col2 = st.columns(2)
            with col1:
                use_all = st.checkbox(
                    f"✅ Use all {len(theme_stocks)} stocks",
                    value=True,
                    help="Include all stocks from this theme"
                )
            
            with col2:
                if not use_all:
                    max_stocks = st.slider(
                        "📊 Number of stocks:",
                        min_value=1,
                        max_value=len(theme_stocks),
                        value=min(10, len(theme_stocks)),
                        help="Choose how many stocks from this theme"
                    )
            
            # Set selected tickers
            if use_all:
                selected_tickers = theme_stocks.copy()
            else:
                selected_tickers = theme_stocks[:max_stocks]
        
        # ==================== RANDOM PORTFOLIO GENERATOR ====================
        elif selection_method == "🎲 Random Portfolio Generator":
            st.subheader("🎲 Random Portfolio Generator")
            
            # Generator options
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_size = st.slider(
                    "📊 Portfolio Size:",
                    min_value=3,
                    max_value=30,
                    value=10,
                    help="Number of stocks in random portfolio"
                )
            
            with col2:
                market_focus = st.selectbox(
                    "🌍 Market Focus:",
                    ["Indian Stocks Only", "Global Mix", "US Stocks Only", "Mixed with ETFs"],
                    help="Geographic focus for random selection"
                )
            
            # Sector diversification
            diversification = st.selectbox(
                "🎯 Diversification Level:",
                ["High Diversification", "Moderate Diversification", "Sector Focused", "Completely Random"],
                index=1,
                help="How diversified should the random portfolio be"
            )
            
            # Generate button
            if st.button("🎲 Generate Random Portfolio", use_container_width=True):
                
                # Filter stocks based on market focus
                available_stocks = []
                
                if market_focus == "Indian Stocks Only":
                    for stocks in stock_categories.values():
                        available_stocks.extend([s for s in stocks if s.endswith('.NS')])
                elif market_focus == "US Stocks Only":
                    for stocks in stock_categories.values():
                        available_stocks.extend([s for s in stocks if not s.endswith('.NS') and not any(ext in s for ext in ['.PA', '.SW', '.TO', '.AS'])])
                elif market_focus == "Mixed with ETFs":
                    for category, stocks in stock_categories.items():
                        if "ETF" in category or "BEES" in category:
                            available_stocks.extend(stocks)
                        else:
                            available_stocks.extend(stocks[:5])  # Limit non-ETF stocks
                else:  # Global Mix
                    for stocks in stock_categories.values():
                        available_stocks.extend(stocks)
                
                # Remove duplicates
                available_stocks = list(set(available_stocks))
                
                # Generate random portfolio with diversification logic
                if diversification == "High Diversification":
                    # Try to pick from different sectors
                    selected_tickers = []
                    used_categories = set()
                    
                    attempts = 0
                    while len(selected_tickers) < portfolio_size and attempts < 1000:
                        # Pick random stock
                        random_stock = np.random.choice(available_stocks)
                        
                        # Find its category
                        stock_category = None
                        for category, stocks in stock_categories.items():
                            if random_stock in stocks:
                                stock_category = category.split(' - ')[0] if ' - ' in category else category.split(' ')[0]
                                break
                        
                        # Add if category not overused
                        if stock_category not in used_categories or len(used_categories) >= 5:
                            selected_tickers.append(random_stock)
                            used_categories.add(stock_category)
                        
                        attempts += 1
                
                else:
                    # Simple random selection
                    selected_tickers = list(np.random.choice(
                        available_stocks, 
                        size=min(portfolio_size, len(available_stocks)), 
                        replace=False
                    ))
                
                st.success(f"🎉 Generated random portfolio with {len(selected_tickers)} stocks!")
                
                # Show generated portfolio
                with st.expander("🎲 Your Random Portfolio"):
                    cols = st.columns(3)
                    for i, stock in enumerate(selected_tickers):
                        with cols[i % 3]:
                            st.write(f"• {stock}")
        
        # ==================== PORTFOLIO SUMMARY AND VALIDATION ====================
        
        # Remove duplicates while preserving order
        selected_tickers = list(dict.fromkeys(selected_tickers))
        
        # Portfolio summary
        if selected_tickers:
            st.markdown("---")
            st.subheader("📋 Portfolio Summary")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Stocks", len(selected_tickers))
            
            with col2:
                indian_count = sum(1 for ticker in selected_tickers if ticker.endswith('.NS'))
                st.metric("🇮🇳 Indian Stocks", indian_count)
            
            with col3:
                international_count = len(selected_tickers) - indian_count
                st.metric("🌍 International", international_count)
            
            with col4:
                # Estimate diversification score
                if len(selected_tickers) >= 10:
                    diversification = "High 🌟"
                elif len(selected_tickers) >= 5:
                    diversification = "Medium 📊"
                else:
                    diversification = "Low ⚠️"
                st.info(f"**Diversification:** {diversification}")
            
            # Portfolio composition analysis
            with st.expander("📊 Detailed Portfolio Analysis"):
                
                # Market distribution
                market_dist = {"Indian (.NS)": 0, "US Stocks": 0, "Other International": 0, "ETFs": 0}
                
                for ticker in selected_tickers:
                    if ticker.endswith('.NS'):
                        market_dist["Indian (.NS)"] += 1
                    elif any(ext in ticker for ext in ['.PA', '.SW', '.TO', '.AS']):
                        market_dist["Other International"] += 1
                    elif "BEES" in ticker or ticker in ["GLD", "SLV", "USO", "EEM", "VTI"]:
                        market_dist["ETFs"] += 1
                    else:
                        market_dist["US Stocks"] += 1
                
                st.write("🌍 **Market Distribution:**")
                for market, count in market_dist.items():
                    if count > 0:
                        percentage = (count / len(selected_tickers)) * 100
                        st.write(f"• {market}: {count} stocks ({percentage:.1f}%)")
                
                # Sector analysis (simplified)
                sector_count = {}
                for ticker in selected_tickers:
                    for category, stocks in stock_categories.items():
                        if ticker in stocks:
                            sector = category.split(' - ')[0] if ' - ' in category else category.split()[0]
                            sector_count[sector] = sector_count.get(sector, 0) + 1
                            break
                
                if sector_count:
                    st.write("🏭 **Sector Distribution:**")
                    for sector, count in sorted(sector_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                        percentage = (count / len(selected_tickers)) * 100
                        st.write(f"• {sector}: {count} stocks ({percentage:.1f}%)")
            
            # Validation and recommendations
            st.subheader("💡 Portfolio Recommendations")
            
            recommendations = []
            
            # Size recommendations
            if len(selected_tickers) < 3:
                recommendations.append("⚠️ **Portfolio too small** - Consider adding more stocks for diversification")
            elif len(selected_tickers) > 25:
                recommendations.append("ℹ️ **Large portfolio** - Consider reducing to 15-20 stocks for easier management")
            
            # Market concentration
            if indian_count / len(selected_tickers) > 0.8:
                recommendations.append("🌍 **High India concentration** - Consider adding international stocks for global diversification")
            elif indian_count == 0:
                recommendations.append("🇮🇳 **No Indian exposure** - Consider adding some Indian stocks for emerging market exposure")
            
            # Sector concentration
            if sector_count:
                max_sector_count = max(sector_count.values())
                if max_sector_count > len(selected_tickers) * 0.5:
                    recommendations.append("⚖️ **Sector concentration risk** - Consider diversifying across more sectors")
            
            # Performance recommendations
            if len(selected_tickers) >= 5 and len(selected_tickers) <= 20:
                recommendations.append("✅ **Good portfolio size** - Optimal balance of diversification and manageability")
            
            if not recommendations:
                recommendations.append("🌟 **Well-balanced portfolio** - Good diversification across markets and sectors")
            
            for rec in recommendations:
                if rec.startswith("⚠️") or rec.startswith("ℹ️"):
                    st.warning(rec)
                elif rec.startswith("✅") or rec.startswith("🌟"):
                    st.success(rec)
                else:
                    st.info(rec)
            
            # Final portfolio display
            with st.expander("📋 Final Stock Selection"):
                cols = st.columns(4)
                for i, ticker in enumerate(selected_tickers):
                    with cols[i % 4]:
                        st.code(ticker)
        
        else:
            st.info("👆 Select stocks using one of the methods above to build your portfolio")
        
        return selected_tickers

def create_enhanced_configuration_interface() -> Dict[str, Any]:
    """Create enhanced configuration interface with advanced options"""
    with st.sidebar:
        st.header("⚙️ Enhanced Analysis Configuration")
        
        config = {}
        
        # Time horizon with enhanced options
        st.subheader("📅 Investment Timeline")
        config['investment_horizon'] = st.selectbox(
            "Primary Investment Horizon:",
            ['next_week', 'next_month', 'next_quarter', 'next_year'],
            index=1,
            help="Primary time horizon for predictions and strategy"
        )
        
        config['secondary_horizons'] = st.multiselect(
            "Additional Horizons (Optional):",
            ['next_week', 'next_month', 'next_quarter', 'next_year'],
            default=[],
            help="Additional time horizons for comprehensive analysis"
        )
        
        # Enhanced model selection
        st.subheader("🤖 Machine Learning Configuration")
        all_model_types = ['xgboost', 'lightgbm', 'random_forest', 'neural_network', 'catboost']
        config['model_types'] = st.multiselect(
            "ML Model Types:",
            all_model_types,
            default=['xgboost', 'random_forest'],
            help="Select machine learning models to use in ensemble"
        )
        
        # Ensemble configuration
        config['ensemble_method'] = st.selectbox(
            "Ensemble Strategy:",
            ['weighted_average', 'voting', 'stacking', 'dynamic_weighting'],
            index=0,
            help="Method for combining predictions from multiple models"
        )
        
        # Advanced model tuning
        with st.expander("🔬 Advanced Model Settings"):
            config['hyperparameter_tuning'] = st.checkbox(
                "Hyperparameter Optimization", 
                value=False,
                help="Enable automatic hyperparameter tuning (slower but better performance)"
            )
            
            config['cross_validation_folds'] = st.slider(
                "Cross-Validation Folds:",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
            
            config['feature_selection'] = st.checkbox(
                "Automatic Feature Selection",
                value=True,
                help="Enable automatic selection of most important features"
            )
        
        # Investment and risk parameters
        st.subheader("💰 Investment Parameters")
        config['investment_amount'] = st.number_input(
            "Total Investment Amount (₹):",
            min_value=10000,
            max_value=100000000,
            value=500000,
            step=50000,
            help="Total amount to be invested across selected stocks"
        )
        
        config['risk_tolerance'] = st.selectbox(
            "Risk Tolerance Level:",
            ['Conservative', 'Moderate', 'Aggressive'],
            index=1,
            help="Your risk tolerance level affects position sizing and recommendations"
        )
        
        # Advanced features configuration
        st.subheader("🚀 Enhanced Features")
        
        config['enable_enhanced_forecasting'] = st.checkbox(
            "Monte Carlo Price Forecasting", 
            value=True,
            help="Enable advanced price forecasting with multiple scenarios"
        )
        
        config['enable_sentiment_analysis'] = st.checkbox(
            "News Sentiment Analysis", 
            value=SENTIMENT_AVAILABLE,
            disabled=not SENTIMENT_AVAILABLE,
            help="Include news sentiment in analysis (requires API key)"
        )
        
        config['enable_risk_management'] = st.checkbox(
            "Advanced Risk Management", 
            value=RISK_MANAGEMENT_AVAILABLE,
            disabled=not RISK_MANAGEMENT_AVAILABLE,
            help="Enable comprehensive risk analysis and monitoring"
        )
        
        config['enable_portfolio_optimization'] = st.checkbox(
            "Portfolio Optimization", 
            value=PORTFOLIO_OPTIMIZATION_AVAILABLE,
            disabled=not PORTFOLIO_OPTIMIZATION_AVAILABLE,
            help="Optimize portfolio weights using modern portfolio theory"
        )
        
        config['enable_backtesting'] = st.checkbox(
            "Historical Backtesting", 
            value=BACKTESTING_AVAILABLE,
            disabled=not BACKTESTING_AVAILABLE,
            help="Test strategy performance on historical data"
        )
        
        # Performance and system settings
        with st.expander("⚡ Performance Settings"):
            config['parallel_processing'] = st.checkbox(
                "Parallel Processing",
                value=True,
                help="Use parallel processing for faster analysis"
            )
            
            config['cache_results'] = st.checkbox(
                "Cache Intermediate Results",
                value=True,
                help="Cache results to speed up repeated analysis"
            )
            
            config['max_data_points'] = st.slider(
                "Historical Data Points:",
                min_value=252,  # 1 year
                max_value=1260,  # 5 years
                value=756,  # 3 years
                help="Number of historical data points to use"
            )
        
        return config

# ==================== ENHANCED ANALYSIS EXECUTION ====================

def run_enhanced_comprehensive_analysis(selected_tickers: List[str], full_config: Dict[str, Any]):
    """Run the complete enhanced analysis pipeline with all features"""
    if not selected_tickers:
        st.error("❌ Please select at least one stock from the sidebar")
        return
    
    # Enhanced progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Progress stages with detailed steps
        stages = [
            ("📥 Loading Market Data", 0.15),
            ("🔧 Feature Engineering", 0.30),
            ("🧠 Training ML Models", 0.50),
            ("🔮 Generating Predictions", 0.65),
            ("📊 Price Forecasting", 0.75),
            ("🛡️ Risk Analysis", 0.85),
            ("📈 Portfolio Optimization", 0.95),
            ("✅ Finalizing Results", 1.0)
        ]
    
    try:
        with st.spinner("🚀 Running Comprehensive AI Analysis..."):
            
            # Stage 1: Enhanced Data Loading
            status_text.text(stages[0][0] + "...")
            progress_bar.progress(stages[0][1])
            
            st.info(f"📊 Loading comprehensive market data for {len(selected_tickers)} stocks...")
            raw_data = get_comprehensive_stock_data(selected_tickers)
            
            if not raw_data:
                st.error("❌ Failed to load market data. Please check your internet connection and try again.")
                return
            
            st.success(f"✅ Successfully loaded data for {len(raw_data)} stocks")
            
            # Data quality assessment
            total_data_points = sum(len(df) for df in raw_data.values())
            avg_data_points = total_data_points / len(raw_data)
            st.info(f"📈 Data Quality: {total_data_points:,} total data points, {avg_data_points:.0f} average per stock")
            
            # Stage 2: Advanced Feature Engineering
            status_text.text(stages[1][0] + "...")
            progress_bar.progress(stages[1][1])
            
            st.info("🔧 Creating advanced technical indicators and features...")
            featured_data = {}
            feature_stats = {'success': 0, 'total': len(raw_data)}
            
            for ticker, df in raw_data.items():
                if not df.empty:
                    try:
                        featured_df = create_features_enhanced(df, FEATURE_CONFIG)
                        if not featured_df.empty:
                            featured_data[ticker] = featured_df
                            feature_stats['success'] += 1
                    except Exception as e:
                        st.warning(f"Feature engineering failed for {ticker}: {e}")
            
            if not featured_data:
                st.error("❌ Feature engineering failed for all stocks")
                return
            
            st.success(f"✅ Feature engineering completed: {feature_stats['success']}/{feature_stats['total']} stocks processed")
            
            # Stage 3: Advanced ML Model Training
            status_text.text(stages[2][0] + "...")
            progress_bar.progress(stages[2][1])
            
            st.info(f"🤖 Training {len(full_config['model_types'])} ML model types using {full_config['ensemble_method']} ensemble...")
            
            # FIXED: Use correct function call with proper parameters
            models = train_models_enhanced_parallel(
                featured_data=featured_data,
                selected_tickers=selected_tickers,
                investment_horizon=full_config['investment_horizon'],
                model_types=full_config['model_types']
            )
            
            model_count = sum(len(model_dict) for model_dict in models.values()) if models else 0
            st.success(f"✅ Model training completed: {model_count} models trained across {len(models)} stocks")
            
            # Stage 4: Enhanced Prediction Generation
            status_text.text(stages[3][0] + "...")
            progress_bar.progress(stages[3][1])
            
            st.info("🔮 Generating ensemble predictions with confidence intervals...")
            
            # FIXED: Use correct function call with proper parameters
            predictions_df, price_targets_df = predict_with_ensemble_and_targets(
                models=models,
                current_data=featured_data,
                investment_horizon=full_config['investment_horizon'],
                model_types=full_config['model_types'],
                ensemble_method=full_config['ensemble_method'],
                selected_tickers=selected_tickers
            )
            
            prediction_count = len(predictions_df) if not predictions_df.empty else 0
            st.success(f"✅ Predictions generated for {prediction_count} stocks")
            
            # Stage 5: Advanced Price Forecasting
            status_text.text(stages[4][0] + "...")
            progress_bar.progress(stages[4][1])
            
            forecast_results = {}
            if full_config.get('enable_enhanced_forecasting', True):
                st.info("📊 Running Monte Carlo price simulations...")
                forecaster = EnhancedPriceForecaster(horizon_days=30)
                
                for ticker in selected_tickers[:8]:  # Limit for performance
                    if ticker in raw_data:
                        try:
                            forecast_results[ticker] = forecaster.generate_enhanced_forecast(ticker, raw_data[ticker])
                        except Exception as e:
                            st.warning(f"Price forecasting failed for {ticker}: {e}")
                
                st.success(f"✅ Price forecasting completed for {len(forecast_results)} stocks")
            
            # Stage 6: Advanced Risk Analysis
            status_text.text(stages[5][0] + "...")
            progress_bar.progress(stages[5][1])
            
            risk_analysis = None
            if full_config.get('enable_risk_management', False) and RISK_MANAGEMENT_AVAILABLE:
                st.info("🛡️ Conducting comprehensive risk analysis...")
                try:
                    risk_config = risk_components['RiskConfig']()
                    risk_manager = risk_components['ComprehensiveRiskManager'](risk_config)
                    risk_analysis = risk_manager.analyze_portfolio_risk(predictions_df, raw_data)
                    st.success("✅ Risk analysis completed")
                except Exception as e:
                    st.warning(f"Risk analysis failed: {e}")
            
            # Stage 7: Portfolio Optimization
            status_text.text(stages[6][0] + "...")
            progress_bar.progress(stages[6][1])
            
            portfolio_optimization = None
            if full_config.get('enable_portfolio_optimization', False) and PORTFOLIO_OPTIMIZATION_AVAILABLE:
                st.info("📈 Optimizing portfolio allocations...")
                try:
                    portfolio_optimization = portfolio_components['optimize_portfolio_for_selected_stocks'](
                        predictions_df, raw_data, selected_tickers
                    )
                    st.success("✅ Portfolio optimization completed")
                except Exception as e:
                    st.warning(f"Portfolio optimization failed: {e}")
            
            # Stage 8: Sentiment Analysis (if enabled)
            sentiment_analysis = None
            if full_config.get('enable_sentiment_analysis', False) and SENTIMENT_AVAILABLE:
                st.info("📰 Analyzing market sentiment...")
                try:
                    sentiment_analysis = sentiment_components['get_sentiment_insights'](selected_tickers)
                    st.success("✅ Sentiment analysis completed")
                except Exception as e:
                    st.warning(f"Sentiment analysis failed: {e}")
            
            # Stage 9: Backtesting (if enabled)
            backtest_results = None
            if full_config.get('enable_backtesting', False) and BACKTESTING_AVAILABLE:
                st.info("📈 Running historical backtesting...")
                try:
                    config_obj = backtest_components['EnhancedBacktestConfig']()
                    strategy = backtest_components['MLStrategy'](models, config_obj)
                    engine = backtest_components['EnhancedBacktestEngine'](config_obj)
                    backtest_results = engine.run_backtest(raw_data, selected_tickers)
                    st.success("✅ Backtesting completed")
                except Exception as e:
                    st.warning(f"Backtesting failed: {e}")
            
            # Final stage: Results compilation
            status_text.text(stages[7][0] + "...")
            progress_bar.progress(stages[7][1])
            
            # Generate comprehensive report data
            report_data = generate_comprehensive_report_data(
                selected_tickers, models, predictions_df, price_targets_df,
                forecast_results, risk_analysis, portfolio_optimization,
                sentiment_analysis, backtest_results
            )
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 **Enhanced Analysis Completed Successfully!**")
            
            # Display comprehensive results
            display_comprehensive_results(
                predictions_df=predictions_df,
                price_targets_df=price_targets_df,
                raw_data=raw_data,
                featured_data=featured_data,
                selected_tickers=selected_tickers,
                models=models,
                config=full_config,
                forecast_results=forecast_results,
                risk_analysis=risk_analysis,
                portfolio_optimization=portfolio_optimization,
                sentiment_analysis=sentiment_analysis,
                backtest_results=backtest_results,
                report_data=report_data
            )
            
            # Store results in session state for persistence
            st.session_state['comprehensive_analysis_results'] = {
                'predictions': predictions_df,
                'price_targets': price_targets_df,
                'models': models,
                'raw_data': raw_data,
                'featured_data': featured_data,
                'forecast_results': forecast_results,
                'risk_analysis': risk_analysis,
                'portfolio_optimization': portfolio_optimization,
                'sentiment_analysis': sentiment_analysis,
                'backtest_results': backtest_results,
                'report_data': report_data,
                'config': full_config,
                'selected_tickers': selected_tickers,
                'timestamp': datetime.now()
            }
            
    except Exception as e:
        st.error(f"❌ **Analysis Failed:** {str(e)}")
        st.error("**Troubleshooting Steps:**")
        st.error("1. 🔄 Try selecting fewer stocks (3-5 recommended)")
        st.error("2. 🌐 Check your internet connection")
        st.error("3. 💾 Clear browser cache and refresh")
        st.error("4. ⚙️ Disable some advanced features and retry")
        
        # Detailed error information for debugging
        with st.expander("🔧 Technical Error Details", expanded=False):
            st.code(f"Error: {str(e)}")
            st.code(f"Selected tickers: {selected_tickers}")
            st.code(f"Configuration: {full_config}")
            st.code(f"Module status: {MODULES_STATUS}")
            st.exception(e)
    
    finally:
        # Cleanup
        progress_container.empty()
        gc.collect()

def generate_comprehensive_report_data(selected_tickers: List[str], models: Dict, predictions_df: pd.DataFrame,
                                     price_targets_df: pd.DataFrame, forecast_results: Dict,
                                     risk_analysis: Optional[Dict], portfolio_optimization: Optional[Dict],
                                     sentiment_analysis: Optional[Dict], backtest_results: Optional[Dict]) -> Dict:
    """Generate comprehensive report data for display"""
    
    report_data = {
        # Basic metrics
        'total_stocks_selected': len(selected_tickers),
        'total_stocks_analyzed': len(models) if models else 0,
        'models_trained': sum(len(model_dict) for model_dict in models.values()) if models else 0,
        'predictions_generated': len(predictions_df) if not predictions_df.empty else 0,
        
        # Performance metrics
        'analysis_success_rate': len(models) / len(selected_tickers) if selected_tickers else 0,
        'avg_prediction_confidence': predictions_df['ensemble_confidence'].mean() if not predictions_df.empty and 'ensemble_confidence' in predictions_df.columns else 0,
        'bullish_predictions': (predictions_df['predicted_return'] > 0).sum() if not predictions_df.empty and 'predicted_return' in predictions_df.columns else 0,
        'bearish_predictions': (predictions_df['predicted_return'] < 0).sum() if not predictions_df.empty and 'predicted_return' in predictions_df.columns else 0,
        'high_confidence_predictions': (predictions_df['ensemble_confidence'] > 0.7).sum() if not predictions_df.empty and 'ensemble_confidence' in predictions_df.columns else 0,
        
        # Financial metrics
        'avg_expected_return': price_targets_df['percentage_change'].mean() / 100 if not price_targets_df.empty and 'percentage_change' in price_targets_df.columns else 0,
        'max_expected_return': price_targets_df['percentage_change'].max() / 100 if not price_targets_df.empty and 'percentage_change' in price_targets_df.columns else 0,
        'min_expected_return': price_targets_df['percentage_change'].min() / 100 if not price_targets_df.empty and 'percentage_change' in price_targets_df.columns else 0,
        
        # Advanced metrics
        'forecasts_generated': len(forecast_results),
        'risk_analysis_available': risk_analysis is not None,
        'portfolio_optimized': portfolio_optimization is not None,
        'sentiment_analyzed': sentiment_analysis is not None,
        'backtesting_performed': backtest_results is not None,
        
        # Risk metrics (if available)
        'portfolio_risk_score': risk_analysis.get('risk_score', 'N/A') if risk_analysis else 'N/A',
        'portfolio_var': risk_analysis.get('portfolio_var', 0) if risk_analysis else 0,
        'max_drawdown': risk_analysis.get('max_drawdown', 0) if risk_analysis else 0,
        'sharpe_ratio': risk_analysis.get('sharpe_ratio', 0) if risk_analysis else 0,
        
        # Sentiment metrics (if available)
        'overall_sentiment': sentiment_analysis.get('overall_sentiment', 0) if sentiment_analysis else 0,
        'positive_sentiment_stocks': sentiment_analysis.get('positive_stocks', 0) if sentiment_analysis else 0,
        
        # Backtest metrics (if available)
        'backtest_total_return': backtest_results.get('total_return', 0) if backtest_results else 0,
        'backtest_sharpe_ratio': backtest_results.get('sharpe_ratio', 0) if backtest_results else 0,
        'backtest_win_rate': backtest_results.get('win_rate', 0) if backtest_results else 0,
        
        # Timestamp
        'generated_at': datetime.now().isoformat()
    }
    
    return report_data

def display_comprehensive_performance_report(report_data: Dict):
    """Display comprehensive performance report with enhanced metrics"""
    
    st.header("📊 **Comprehensive Analysis Performance Report**")
    
    # Executive Summary Cards
    st.subheader("📈 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Stocks Analyzed", 
            f"{report_data.get('total_stocks_analyzed', 0)}/{report_data.get('total_stocks_selected', 0)}",
            help="Successfully analyzed stocks out of selected stocks"
        )
    
    with col2:
        success_rate = report_data.get('analysis_success_rate', 0)
        st.metric(
            "Success Rate", 
            f"{success_rate:.1%}",
            help="Percentage of successful stock analysis"
        )
    
    with col3:
        models_trained = report_data.get('models_trained', 0)
        st.metric(
            "Models Trained", 
            models_trained,
            help="Total ML models successfully trained"
        )
    
    with col4:
        avg_confidence = report_data.get('avg_prediction_confidence', 0)
        st.metric(
            "Avg Confidence", 
            f"{avg_confidence:.1%}",
            help="Average prediction confidence across all stocks"
        )
    
    with col5:
        predictions = report_data.get('predictions_generated', 0)
        st.metric(
            "Predictions", 
            predictions,
            help="Number of stock predictions generated"
        )
    
    # Prediction Summary
    st.subheader("🎯 Prediction Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bullish = report_data.get('bullish_predictions', 0)
        st.metric(
            "Bullish Signals", 
            bullish,
            help="Number of BUY recommendations",
            delta="Positive Outlook" if bullish > 0 else None
        )
    
    with col2:
        bearish = report_data.get('bearish_predictions', 0)
        st.metric(
            "Bearish Signals", 
            bearish,
            help="Number of SELL recommendations",
            delta="Caution Advised" if bearish > 0 else None
        )
    
    with col3:
        high_confidence = report_data.get('high_confidence_predictions', 0)
        st.metric(
            "High Confidence", 
            high_confidence,
            help="Predictions with >70% confidence",
            delta="Strong Signals" if high_confidence > 0 else None
        )
    
    with col4:
        avg_return = report_data.get('avg_expected_return', 0)
        st.metric(
            "Avg Expected Return", 
            f"{avg_return:.1%}",
            help="Average expected return across all price targets"
        )
    
    # Return Distribution
    if report_data.get('max_expected_return', 0) != 0:
        st.subheader("📊 Return Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_return = report_data.get('max_expected_return', 0)
            st.metric("Max Expected Return", f"{max_return:.1%}", delta="Best Opportunity")
        
        with col2:
            min_return = report_data.get('min_expected_return', 0)
            st.metric("Min Expected Return", f"{min_return:.1%}", delta="Highest Risk")
        
        with col3:
            return_spread = max_return - min_return
            st.metric("Return Spread", f"{return_spread:.1%}", help="Difference between max and min expected returns")
    
    # Advanced Features Summary
    st.subheader("🚀 Advanced Features Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk Management Status
        if report_data.get('risk_analysis_available', False):
            risk_score = report_data.get('portfolio_risk_score', 'N/A')
            sharpe = report_data.get('sharpe_ratio', 0)
            st.success(f"✅ **Risk Management Active**")
            st.write(f"**Risk Level:** {risk_score}")
            st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
        else:
            st.warning("⚠️ **Risk Management:** Fallback Mode")
    
    with col2:
        # Portfolio Optimization Status
        if report_data.get('portfolio_optimized', False):
            st.success("✅ **Portfolio Optimization Completed**")
            st.write("**Method:** Modern Portfolio Theory")
            st.write("**Weights:** Optimally Allocated")
        else:
            st.warning("⚠️ **Portfolio Optimization:** Not Available")
    
    with col3:
        # Sentiment Analysis Status
        if report_data.get('sentiment_analyzed', False):
            sentiment = report_data.get('overall_sentiment', 0)
            positive_count = report_data.get('positive_sentiment_stocks', 0)
            sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
            st.success("✅ **Sentiment Analysis Active**")
            st.write(f"**Market Sentiment:** {sentiment_label}")
            st.write(f"**Positive Stocks:** {positive_count}")
        else:
            st.warning("⚠️ **Sentiment Analysis:** Not Available")
    
    # Backtesting Results (if available)
    if report_data.get('backtesting_performed', False):
        st.subheader("📈 Historical Backtesting Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_return = report_data.get('backtest_total_return', 0)
            st.metric("Backtest Total Return", f"{backtest_return:.1%}")
        
        with col2:
            backtest_sharpe = report_data.get('backtest_sharpe_ratio', 0)
            st.metric("Backtest Sharpe Ratio", f"{backtest_sharpe:.2f}")
        
        with col3:
            win_rate = report_data.get('backtest_win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1%}")
    
    # System Capabilities Overview
    st.subheader("🛠️ System Capabilities")
    
    capabilities = {
        "Data Loading": MODULES_STATUS.get('data_loader', False),
        "Feature Engineering": MODULES_STATUS.get('feature_engineer', False),
        "ML Models": MODULES_STATUS.get('model', False),
        "Risk Management": RISK_MANAGEMENT_AVAILABLE,
        "Backtesting": BACKTESTING_AVAILABLE,
        "Sentiment Analysis": SENTIMENT_AVAILABLE,
        "Portfolio Optimization": PORTFOLIO_OPTIMIZATION_AVAILABLE,
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (capability, available) in enumerate(capabilities.items()):
        col_index = i % 3
        if col_index == 0:
            col = col1
        elif col_index == 1:
            col = col2
        else:
            col = col3
        
        with col:
            if available:
                st.markdown(f'<span class="status-indicator status-success">✅ {capability}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-indicator status-warning">⚠️ {capability} (Fallback)</span>', unsafe_allow_html=True)
    
    # Performance Recommendations
    st.subheader("💡 Performance Recommendations")
    
    recommendations = []
    
    if report_data.get('analysis_success_rate', 0) < 0.8:
        recommendations.append("⚠️ Consider selecting stocks with more historical data for better analysis")
    
    if report_data.get('avg_prediction_confidence', 0) < 0.6:
        recommendations.append("🔧 Enable hyperparameter tuning for improved model performance")
    
    if not report_data.get('risk_analysis_available', False):
        recommendations.append("🛡️ Enable advanced risk management for better risk assessment")
    
    if report_data.get('bullish_predictions', 0) == 0 and report_data.get('bearish_predictions', 0) == 0:
        recommendations.append("📊 Review stock selection - consider adding more volatile stocks for clearer signals")
    
    if len(recommendations) == 0:
        st.success("🎉 **Excellent!** All analysis components are performing optimally.")
    else:
        for rec in recommendations:
            st.warning(rec)

def create_enhanced_analysis_tabs(predictions_df: pd.DataFrame, 
                                price_targets_df: pd.DataFrame,
                                raw_data: Dict,
                                featured_data: Dict,
                                selected_tickers: List[str],
                                models: Dict,
                                config: Dict,
                                forecast_results: Optional[Dict] = None,
                                risk_analysis: Optional[Dict] = None,
                                portfolio_optimization: Optional[Dict] = None,
                                sentiment_analysis: Optional[Dict] = None,
                                backtest_results: Optional[Dict] = None,
                                report_data: Optional[Dict] = None):
    """Create comprehensive analysis tabs with all features"""
    
    # Create tabs with enhanced features
    tab_names = ["📊 Predictions & Targets", "📈 Interactive Charts", "🔮 Price Forecasting", "📰 Sentiment Analysis"]
    
    if portfolio_optimization:
        tab_names.append("💼 Portfolio Optimization")
    
    if risk_analysis:
        tab_names.append("🛡️ Risk Management")
    
    if backtest_results:
        tab_names.append("📈 Backtesting Results")
    
    tab_names.extend(["📋 Data Quality", "⚙️ Model Details"])
    
    tabs = st.tabs(tab_names)
    
    # Tab 1: Predictions & Targets
    with tabs[0]:
        create_predictions_and_targets_tab(predictions_df, price_targets_df, selected_tickers)
    
    # Tab 2: Interactive Charts
    with tabs[1]:
        create_interactive_charts_tab(raw_data, predictions_df, price_targets_df, selected_tickers)
    
    # Tab 3: Price Forecasting
    with tabs[2]:
        create_price_forecasting_tab(forecast_results, raw_data, selected_tickers)
    
    # Tab 4: Sentiment Analysis
    with tabs[3]:
        create_sentiment_analysis_tab(sentiment_analysis, selected_tickers)
    
    tab_index = 4
    
    # Tab 5: Portfolio Optimization (conditional)
    if portfolio_optimization:
        with tabs[tab_index]:
            create_portfolio_optimization_tab(portfolio_optimization, selected_tickers, config)
        tab_index += 1
    
    # Tab 6: Risk Management (conditional)
    if risk_analysis:
        with tabs[tab_index]:
            create_risk_management_tab(risk_analysis, predictions_df, raw_data, selected_tickers)
        tab_index += 1
    
    # Tab 7: Backtesting Results (conditional)
    if backtest_results:
        with tabs[tab_index]:
            create_backtesting_results_tab(backtest_results, selected_tickers)
        tab_index += 1
    
    # Tab: Data Quality
    with tabs[tab_index]:
        create_data_quality_tab(raw_data, featured_data, selected_tickers)
    tab_index += 1
    
    # Tab: Model Details
    with tabs[tab_index]:
        create_model_details_tab(models, config, selected_tickers)

def create_predictions_and_targets_tab(predictions_df: pd.DataFrame, price_targets_df: pd.DataFrame, selected_tickers: List[str]):
    """Enhanced predictions and targets display"""
    
    st.header("🎯 ML Model Predictions & Price Targets")
    
    if predictions_df.empty and price_targets_df.empty:
        st.warning("No prediction data available. Please run the analysis first.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if not predictions_df.empty:
        with col1:
            bullish_count = (predictions_df['predicted_return'] > 0).sum()
            st.metric("Bullish Predictions", bullish_count, delta=f"Out of {len(predictions_df)}")
        
        with col2:
            avg_confidence = predictions_df['ensemble_confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            high_confidence = (predictions_df['ensemble_confidence'] > 0.7).sum()
            st.metric("High Confidence (>70%)", high_confidence)
        
        with col4:
            max_return = predictions_df['predicted_return'].max()
            st.metric("Best Predicted Return", f"{max_return:.1%}")
    
    # Enhanced predictions table
    if not predictions_df.empty:
        st.subheader("📊 Detailed Prediction Results")
        
        # Format the DataFrame for better display
        display_predictions = predictions_df.copy()
        display_predictions['predicted_return'] = display_predictions['predicted_return'].apply(lambda x: f"{x:.2%}")
        display_predictions['ensemble_confidence'] = display_predictions['ensemble_confidence'].apply(lambda x: f"{x:.1%}")
        display_predictions['signal_strength'] = display_predictions['signal_strength'].apply(lambda x: f"{x:.2f}")
        
        # Add recommendation column
        def get_recommendation(row):
            if row['predicted_return'] > 0.05:
                return "🟢 STRONG BUY"
            elif row['predicted_return'] > 0.02:
                return "🟡 BUY"
            elif row['predicted_return'] > -0.02:
                return "⚪ HOLD"
            elif row['predicted_return'] > -0.05:
                return "🟠 SELL"
            else:
                return "🔴 STRONG SELL"
        
        display_predictions['recommendation'] = predictions_df.apply(get_recommendation, axis=1)
        
        # Display with styling
        st.dataframe(
            display_predictions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Stock", width="small"),
                "predicted_return": st.column_config.TextColumn("Expected Return", width="small"),
                "ensemble_confidence": st.column_config.TextColumn("Confidence", width="small"),
                "signal_strength": st.column_config.TextColumn("Signal Strength", width="small"),
                "recommendation": st.column_config.TextColumn("Recommendation", width="medium")
            }
        )
        
        # Prediction visualization
        if len(display_predictions) > 1:
            st.subheader("📈 Prediction Visualization")
            
            fig = px.scatter(
                predictions_df,
                x='predicted_return',
                y='ensemble_confidence',
                size='signal_strength',
                color='ticker',
                hover_name='ticker',
                title='Prediction Confidence vs Expected Return',
                labels={
                    'predicted_return': 'Expected Return (%)',
                    'ensemble_confidence': 'Model Confidence',
                    'ticker': 'Stock',
                    'signal_strength': 'Signal Strength'
                }
            )
            
            # Add quadrant lines
            fig.add_hline(y=0.7, line_dash="dash", line_color="gray", annotation_text="High Confidence Threshold")
            fig.add_vline(x=0.05, line_dash="dash", line_color="gray", annotation_text="Buy Threshold")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced price targets
    if not price_targets_df.empty:
        st.subheader("💰 Price Target Analysis")
        
        # Price target summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            percentage_col = None
            for col in ['percentage_change', 'expected_return', 'target_return']:
                if col in price_targets_df.columns:
                    percentage_col = col
                    break

            if percentage_col:
                avg_target_return = price_targets_df[percentage_col].mean()
            else:
                avg_target_return = 0
            st.metric("Avg Target Return", f"{avg_target_return:.1f}%")
        
        with col2:
            def get_percentage_column(df):
                """Safely get the percentage change column name"""
                for col in ['percentage_change', 'expected_return', 'target_return']:
                    if col in df.columns:
                        return col
                return None
    
            percentage_col = get_percentage_column(price_targets_df)
            if percentage_col:
                max_upside = price_targets_df[percentage_col].max()
                min_downside = price_targets_df[percentage_col].min()
            else:
                max_upside = 0
                min_downside = 0
            st.metric("Max Upside Potential", f"{max_upside:.1f}%")

        with col3:
            st.metric("Max Downside Risk", f"{min_downside:.1f}%")
        
        with col4:
            high_confidence_targets = (price_targets_df['confidence'] > 0.7).sum()
            st.metric("High Confidence Targets", high_confidence_targets)
        
        # Enhanced price targets table
        display_targets = price_targets_df.copy()
        display_targets['current_price'] = display_targets['current_price'].apply(lambda x: f"₹{x:.2f}")
        display_targets['target_price'] = display_targets['target_price'].apply(lambda x: f"₹{x:.2f}")
        if percentage_col:
            display_targets[percentage_col] = display_targets[percentage_col].apply(lambda x: f"{x:.1f}%")
        display_targets['confidence'] = display_targets['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_targets,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "Stock",
                "current_price": "Current Price",
                "target_price": "Target Price", 
                "percentage_change": "Expected Change",
                "confidence": "Confidence",
                "horizon": "Time Horizon"
            }
        )
        
        # Price target visualization
        if len(display_targets) > 1:
            fig = px.bar(
                price_targets_df,
                x='ticker',
                y='percentage_change',
                color='confidence',
                title='Price Target Analysis by Stock',
                labels={
                    'ticker': 'Stock',
                    'percentage_change': 'Expected Price Change (%)',
                    'confidence': 'Confidence Level'
                },
                color_continuous_scale='RdYlGn'
            )
            
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            st.plotly_chart(fig, use_container_width=True)

def create_interactive_charts_tab(raw_data: Dict, predictions_df: pd.DataFrame, price_targets_df: pd.DataFrame, selected_tickers: List[str]):
    """Create interactive charts and analysis"""
    
    st.header("📈 Interactive Charts & Technical Analysis")
    
    if not raw_data:
        st.warning("No data available for charting.")
        return
    
    # Stock selector for individual analysis
    st.subheader("📊 Individual Stock Analysis")
    
    chart_ticker = st.selectbox(
        "Select stock for detailed chart:",
        selected_tickers,
        help="Choose a stock to view detailed technical analysis"
    )
    
    if chart_ticker in raw_data:
        df = raw_data[chart_ticker]
        
        if not df.empty:
            # Create comprehensive chart
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=['Price & Volume', 'Technical Indicators', 'Prediction Signals'],
                row_width=[0.2, 0.2, 0.6]
            )
            
            # Price and volume
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.3),
                row=2, col=1
            )
            
            # Moving averages
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(20).mean()
                sma_50 = df['Close'].rolling(50).mean()
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=sma_20, name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
                
                if len(df) >= 50:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=sma_50, name='SMA 50', line=dict(color='red')),
                        row=1, col=1
                    )
            
            # Prediction signals
            if not predictions_df.empty and chart_ticker in predictions_df['ticker'].values:
                pred_row = predictions_df[predictions_df['ticker'] == chart_ticker].iloc[0]
                signal_strength = pred_row['signal_strength']
                predicted_return = pred_row['predicted_return']
                
                # Add prediction signal
                latest_date = df.index[-1]
                latest_price = df['Close'].iloc[-1]
                target_price = latest_price * (1 + predicted_return)
                
                fig.add_trace(
                    go.Scatter(
                        x=[latest_date], y=[target_price],
                        mode='markers',
                        marker=dict(size=signal_strength*20, color='red' if predicted_return < 0 else 'green'),
                        name=f'Price Target (₹{target_price:.2f})',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            fig.update_layout(
                title=f'{chart_ticker} - Comprehensive Technical Analysis',
                xaxis_title='Date',
                height=800,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Recent Performance")
                recent_data = df.tail(30)  # Last 30 days
                
                if len(recent_data) > 1:
                    recent_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1)
                    recent_volatility = recent_data['Close'].pct_change().std() * np.sqrt(252)
                    recent_volume = recent_data['Volume'].mean()
                    
                    st.metric("30-Day Return", f"{recent_return:.1%}")
                    st.metric("Annualized Volatility", f"{recent_volatility:.1%}")
                    st.metric("Avg Daily Volume", f"{recent_volume:,.0f}")
            
            with col2:
                st.subheader("🎯 ML Prediction Details")
                if not predictions_df.empty and chart_ticker in predictions_df['ticker'].values:
                    pred_row = predictions_df[predictions_df['ticker'] == chart_ticker].iloc[0]
                    
                    st.metric("Expected Return", f"{pred_row['predicted_return']:.1%}")
                    st.metric("Model Confidence", f"{pred_row['ensemble_confidence']:.1%}")
                    st.metric("Signal Strength", f"{pred_row['signal_strength']:.2f}")
                    
                    # Recommendation badge
                    predicted_return = pred_row['predicted_return']
                    if predicted_return > 0.05:
                        st.success("🟢 **STRONG BUY** - High upside potential")
                    elif predicted_return > 0.02:
                        st.success("🟡 **BUY** - Moderate upside potential")
                    elif predicted_return > -0.02:
                        st.info("⚪ **HOLD** - Limited price movement expected")
                    elif predicted_return > -0.05:
                        st.warning("🟠 **SELL** - Potential downside risk")
                    else:
                        st.error("🔴 **STRONG SELL** - High downside risk")
                else:
                    st.info("No predictions available for this stock")
    
    # Portfolio overview
    st.subheader("💼 Portfolio Overview")
    
    if len(selected_tickers) > 1:
        # Create portfolio performance comparison
        portfolio_data = []
        
        for ticker in selected_tickers:
            if ticker in raw_data:
                df = raw_data[ticker]
                if len(df) > 30:  # Ensure sufficient data
                    recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1)
                    portfolio_data.append({
                        'Ticker': ticker,
                        '30-Day Return': recent_return,
                        'Current Price': df['Close'].iloc[-1],
                        'Volume': df['Volume'].iloc[-1]
                    })
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            
            # Performance comparison chart
            fig = px.bar(
                portfolio_df,
                x='Ticker',
                y='30-Day Return',
                title='30-Day Performance Comparison',
                labels={'30-Day Return': '30-Day Return (%)'},
                color='30-Day Return',
                color_continuous_scale='RdYlGn'
            )
            
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            display_df = portfolio_df.copy()
            display_df['30-Day Return'] = display_df['30-Day Return'].apply(lambda x: f"{x:.1%}")
            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.2f}")
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

def create_price_forecasting_tab(forecast_results: Optional[Dict], raw_data: Dict, selected_tickers: List[str]):
    """Enhanced price forecasting tab with scenarios"""
    
    st.header("🔮 Advanced Price Forecasting & Scenarios")
    
    if not forecast_results:
        st.info("Enhanced price forecasting not available. Run analysis with 'Monte Carlo Price Forecasting' enabled.")
        
        # Provide basic forecasting using available data
        st.subheader("📊 Basic Price Analysis")
        
        if raw_data:
            forecaster = EnhancedPriceForecaster()
            
            basic_forecasts = {}
            for ticker in selected_tickers[:5]:  # Limit for performance
                if ticker in raw_data:
                    try:
                        basic_forecasts[ticker] = forecaster.generate_enhanced_forecast(ticker, raw_data[ticker])
                    except Exception as e:
                        st.warning(f"Basic forecast failed for {ticker}: {e}")
            
            if basic_forecasts:
                display_forecast_results(basic_forecasts, selected_tickers)
        
        return
    
    st.success(f"✅ Enhanced forecasting available for {len(forecast_results)} stocks")
    
    display_forecast_results(forecast_results, selected_tickers)

def display_forecast_results(forecast_results: Dict, selected_tickers: List[str]):
    """Display comprehensive forecasting results"""
    
    # Forecast summary table
    st.subheader("📊 Forecast Summary")
    
    summary_data = []
    for ticker, forecast in forecast_results.items():
        base_case = forecast['scenarios']['base_case']
        recommendation = forecast['recommendation']
        
        summary_data.append({
            'Stock': ticker,
            'Current Price': f"₹{forecast['current_price']:.2f}",
            'Target Price': f"₹{base_case['target_price']:.2f}",
            'Expected Return': f"{base_case['return_pct']:.1f}%",
            'Recommendation': f"{recommendation['action']}",
            'Confidence': f"{recommendation['confidence']:.1%}",
            'Volatility': f"{forecast.get('volatility', 0):.1%}"
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Forecast visualization
        st.subheader("📈 Return Expectations vs Risk")
        
        # Extract data for plotting
        plot_data = []
        for ticker, forecast in forecast_results.items():
            base_case = forecast['scenarios']['base_case']
            plot_data.append({
                'ticker': ticker,
                'expected_return': base_case['return_pct'],
                'volatility': forecast.get('volatility', 0) * 100,
                'confidence': forecast['recommendation']['confidence']
            })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            fig = px.scatter(
                plot_df,
                x='volatility',
                y='expected_return',
                size='confidence',
                color='ticker',
                hover_name='ticker',
                title='Risk-Return Profile of Selected Stocks',
                labels={
                    'volatility': 'Risk (Volatility %)',
                    'expected_return': 'Expected Return (%)',
                    'confidence': 'Forecast Confidence'
                }
            )
            
            # Add quadrant lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=plot_df['volatility'].mean(), line_dash="dash", line_color="gray", 
                         annotation_text="Avg Portfolio Risk")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed scenarios for top stocks
    st.subheader("🎯 Detailed Scenario Analysis")
    
    # Allow user to select stock for detailed analysis
    selected_stock = st.selectbox(
        "Select stock for detailed scenario analysis:",
        list(forecast_results.keys()),
        help="Choose a stock to view detailed bullish, bearish, and base case scenarios"
    )
    
    if selected_stock in forecast_results:
        forecast = forecast_results[selected_stock]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scenario comparison
            scenarios_data = []
            for scenario_name, scenario_data in forecast['scenarios'].items():
                scenarios_data.append({
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Target Price': f"₹{scenario_data['target_price']:.2f}",
                    'Expected Return': f"{scenario_data['return_pct']:.1f}%",
                    'Probability': f"{scenario_data['probability']:.1%}",
                    'Return_Value': scenario_data['return_pct']  # For coloring
                })
            
            scenarios_df = pd.DataFrame(scenarios_data)
            
            st.dataframe(
                scenarios_df[['Scenario', 'Target Price', 'Expected Return', 'Probability']],
                hide_index=True,
                use_container_width=True
            )
            
            # Scenario visualization
            fig = px.bar(
                scenarios_df,
                x='Scenario',
                y='Return_Value',
                color='Return_Value',
                title=f'{selected_stock} - Scenario Analysis',
                labels={'Return_Value': 'Expected Return (%)'},
                color_continuous_scale='RdYlGn'
            )
            
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recommendation box
            rec = forecast['recommendation']
            current_price = forecast['current_price']
            
            # Color code the recommendation
            action_colors = {
                'BUY': '#28a745',
                'STRONG BUY': '#155724', 
                'HOLD': '#ffc107',
                'SELL': '#dc3545',
                'STRONG SELL': '#721c24'
            }
            
            action_color = action_colors.get(rec['action'], '#6c757d')
            
            st.markdown(f"""
            <div class="prediction-card" style="border-left-color: {action_color};">
                <h3 style="color: {action_color};">{rec['action']}</h3>
                <p><strong>Confidence:</strong> {rec['confidence']:.1%}</p>
                <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
                <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
                <p><strong>Forecast Horizon:</strong> {forecast['forecast_horizon_days']} days</p>
                <p><strong>Risk Level:</strong> {forecast.get('volatility', 0):.1%} volatility</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            volatility = forecast.get('volatility', 0)
            if volatility > 0.30:
                risk_level = "🔴 High Risk"
            elif volatility > 0.20:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🟢 Low Risk"
            
            st.info(f"**Risk Assessment:** {risk_level}")

def create_sentiment_analysis_tab(sentiment_analysis: Optional[Dict], selected_tickers: List[str]):
    """Enhanced sentiment analysis display"""
    
    st.header("📰 Market Sentiment Analysis")
    
    if not sentiment_analysis:
        st.info("Sentiment analysis not available. This feature requires news API access.")
        
        # Provide fallback sentiment analysis
        if SENTIMENT_AVAILABLE:
            st.info("Attempting to fetch sentiment data...")
            try:
                fallback_sentiment = sentiment_components['get_sentiment_insights'](selected_tickers)
                if fallback_sentiment:
                    display_sentiment_results(fallback_sentiment, selected_tickers)
                else:
                    st.warning("Unable to fetch sentiment data at this time.")
            except Exception as e:
                st.warning(f"Sentiment analysis error: {e}")
        else:
            st.warning("Sentiment analysis module not available.")
        
        return
    
    st.success("✅ Market sentiment analysis completed")
    display_sentiment_results(sentiment_analysis, selected_tickers)

def display_sentiment_results(sentiment_analysis: Dict, selected_tickers: List[str]):
    """Display comprehensive sentiment analysis results"""
    
    # Overall sentiment metrics
    st.subheader("📊 Market Sentiment Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_sentiment = sentiment_analysis.get('overall_sentiment', 0)
        sentiment_label = "Positive" if overall_sentiment > 0.1 else "Negative" if overall_sentiment < -0.1 else "Neutral"
        sentiment_color = "normal" if overall_sentiment > 0.1 else "inverse" if overall_sentiment < -0.1 else "off"
        st.metric("Overall Sentiment", sentiment_label, f"{overall_sentiment:.3f}", delta_color=sentiment_color)
    
    with col2:
        positive_stocks = sentiment_analysis.get('positive_stocks', 0)
        st.metric("Positive Sentiment", positive_stocks, f"out of {len(selected_tickers)}")
    
    with col3:
        negative_stocks = sentiment_analysis.get('negative_stocks', 0)
        st.metric("Negative Sentiment", negative_stocks, f"out of {len(selected_tickers)}")
    
    with col4:
        neutral_stocks = sentiment_analysis.get('neutral_stocks', 0)
        st.metric("Neutral Sentiment", neutral_stocks, f"out of {len(selected_tickers)}")
    
    # Sentiment distribution
    sentiment_distribution = sentiment_analysis.get('sentiment_distribution', {})
    
    if sentiment_distribution:
        st.subheader("📈 Individual Stock Sentiment")
        
        # Create DataFrame for display
        sentiment_data = []
        for ticker, sentiment_score in sentiment_distribution.items():
            sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
            
            sentiment_data.append({
                'Stock': ticker,
                'Sentiment Score': f"{sentiment_score:.3f}",
                'Sentiment': sentiment_label,
                'Score_Value': sentiment_score  # For visualization
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Display table
        st.dataframe(
            sentiment_df[['Stock', 'Sentiment Score', 'Sentiment']],
            use_container_width=True,
            hide_index=True
        )
        
        # Sentiment visualization
        fig = px.bar(
            sentiment_df,
            x='Stock',
            y='Score_Value',
            color='Score_Value',
            title='News Sentiment by Stock',
            labels={'Score_Value': 'Sentiment Score'},
            color_continuous_scale='RdYlGn'
        )
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Positive Threshold")
        fig.add_hline(y=-0.1, line_dash="dash", line_color="red", annotation_text="Negative Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top positive and negative sentiments
    col1, col2 = st.columns(2)
    
    with col1:
        top_positive = sentiment_analysis.get('top_positive', [])
        if top_positive:
            st.subheader("🟢 Most Positive Sentiment")
            for ticker, score in top_positive[:3]:
                st.success(f"**{ticker}**: {score:.3f}")
    
    with col2:
        top_negative = sentiment_analysis.get('top_negative', [])
        if top_negative:
            st.subheader("🔴 Most Negative Sentiment")
            for ticker, score in top_negative[:3]:
                st.error(f"**{ticker}**: {score:.3f}")
    
    # Sentiment insights and recommendations
    st.subheader("💡 Sentiment-Based Insights")
    
    if overall_sentiment > 0.2:
        st.success("🟢 **Strong Positive Market Sentiment** - Market conditions appear favorable for the selected stocks.")
    elif overall_sentiment > 0.1:
        st.info("🟡 **Moderate Positive Sentiment** - Generally positive market conditions.")
    elif overall_sentiment < -0.2:
        st.error("🔴 **Strong Negative Market Sentiment** - Exercise caution, negative news flow detected.")
    elif overall_sentiment < -0.1:
        st.warning("🟠 **Moderate Negative Sentiment** - Some negative market sentiment present.")
    else:
        st.info("⚪ **Neutral Market Sentiment** - Balanced news sentiment across selected stocks.")
    
    # Sentiment-based recommendations
    recommendations = []
    
    if positive_stocks > negative_stocks * 2:
        recommendations.append("📈 Consider increasing exposure to stocks with positive sentiment")
    
    if negative_stocks > positive_stocks:
        recommendations.append("⚠️ Consider reducing positions in stocks with negative sentiment")
    
    if neutral_stocks > len(selected_tickers) * 0.5:
        recommendations.append("📊 Monitor neutral-sentiment stocks for sentiment shifts")
    
    if recommendations:
        st.markdown("**Recommendations based on sentiment analysis:**")
        for rec in recommendations:
            st.info(rec)

def create_portfolio_optimization_tab(portfolio_optimization: Dict, selected_tickers: List[str], config: Dict):
    """Enhanced portfolio optimization display"""
    
    st.header("💼 Advanced Portfolio Optimization")
    
    if not portfolio_optimization:
        st.warning("Portfolio optimization not available.")
        return
    
    st.success("✅ Portfolio optimization completed using Modern Portfolio Theory")
    
    # Optimization results summary
    st.subheader("📊 Optimization Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        expected_return = portfolio_optimization.get('expected_return', 0)
        st.metric("Expected Annual Return", f"{expected_return:.1%}")
    
    with col2:
        volatility = portfolio_optimization.get('volatility', 0)
        st.metric("Portfolio Volatility", f"{volatility:.1%}")
    
    with col3:
        sharpe_ratio = portfolio_optimization.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        optimization_method = portfolio_optimization.get('optimization_method', 'Unknown')
        st.info(f"**Method:** {optimization_method.replace('_', ' ').title()}")
    
    # Optimal portfolio weights
    st.subheader("⚖️ Optimal Portfolio Allocation")
    
    weights = portfolio_optimization.get('weights', {})
    investment_amount = config.get('investment_amount', 500000)
    
    if weights:
        # Create allocation DataFrame
        allocation_data = []
        for ticker, weight in weights.items():
            allocation_amount = investment_amount * weight
            allocation_data.append({
                'Stock': ticker,
                'Weight': f"{weight:.1%}",
                'Allocation': f"₹{allocation_amount:,.0f}",
                'Weight_Value': weight  # For visualization
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        
        # Display allocation table
        st.dataframe(
            allocation_df[['Stock', 'Weight', 'Allocation']],
            use_container_width=True,
            hide_index=True
        )
        
        # Portfolio allocation pie chart
        fig = px.pie(
            allocation_df,
            values='Weight_Value',
            names='Stock',
            title='Optimal Portfolio Allocation',
            hover_data=['Allocation']
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Allocation recommendations
        st.subheader("💡 Allocation Insights")
        
        max_weight_stock = max(weights.items(), key=lambda x: x[1])
        min_weight_stock = min(weights.items(), key=lambda x: x[1])
        
        st.info(f"**Largest Position:** {max_weight_stock[0]} ({max_weight_stock[1]:.1%})")
        st.info(f"**Smallest Position:** {min_weight_stock[0]} ({min_weight_stock[1]:.1%})")
        
        # Diversification analysis
        max_weight = max(weights.values())
        if max_weight > 0.4:
            st.warning("⚠️ **High Concentration Risk** - Consider diversifying further")
        elif max_weight < 0.15:
            st.info("✅ **Well Diversified** - Risk is well distributed")
        else:
            st.success("✅ **Balanced Portfolio** - Good risk-return balance")

def create_risk_management_tab(risk_analysis: Dict, predictions_df: pd.DataFrame, raw_data: Dict, selected_tickers: List[str]):
    """Enhanced risk management and analysis display"""
    
    st.header("🛡️ Comprehensive Risk Management Analysis")
    
    if not risk_analysis:
        st.warning("Advanced risk analysis not available.")
        return
    
    st.success("✅ Comprehensive risk analysis completed")
    
    # Risk metrics overview
    st.subheader("📊 Portfolio Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        portfolio_var = risk_analysis.get('portfolio_var', 0)
        st.metric("Value at Risk (95%)", f"{portfolio_var:.1%}")
    
    with col2:
        max_drawdown = risk_analysis.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    with col3:
        sharpe_ratio = risk_analysis.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        risk_score = risk_analysis.get('risk_score', 'Medium')
        risk_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
        st.metric("Risk Level", f"{risk_colors.get(risk_score, '⚪')} {risk_score}")
    
    # Risk assessment and recommendations
    st.subheader("🎯 Risk Assessment")
    
    if risk_score == 'High':
        st.error("🔴 **High Risk Portfolio** - Consider reducing position sizes or diversifying further")
    elif risk_score == 'Medium':
        st.warning("🟡 **Moderate Risk Portfolio** - Balanced risk profile with room for optimization")
    else:
        st.success("🟢 **Low Risk Portfolio** - Conservative risk profile with stable expected returns")
    
    # Correlation analysis
    if len(selected_tickers) > 1:
        st.subheader("🔗 Portfolio Correlation Analysis")
        
        try:
            # Calculate correlation matrix
            returns_data = {}
            for ticker in selected_tickers:
                if ticker in raw_data and not raw_data[ticker].empty:
                    returns_data[ticker] = raw_data[ticker]['Close'].pct_change().dropna()
            
            if len(returns_data) > 1:
                correlation_df = pd.DataFrame(returns_data).corr()
                
                # Create correlation heatmap
                fig = px.imshow(
                    correlation_df,
                    title="Stock Return Correlations",
                    color_continuous_scale='RdYlGn_r',
                    aspect="auto"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation insights
                high_correlations = []
                for i in range(len(correlation_df.columns)):
                    for j in range(i+1, len(correlation_df.columns)):
                        corr_value = correlation_df.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            stock1 = correlation_df.columns[i]
                            stock2 = correlation_df.columns[j]
                            high_correlations.append((stock1, stock2, corr_value))
                
                if high_correlations:
                    st.warning("⚠️ **High Correlations Detected:**")
                    for stock1, stock2, corr in high_correlations:
                        st.write(f"• {stock1} ↔ {stock2}: {corr:.2f}")
                    st.info("Consider diversifying into different sectors to reduce correlation risk.")
                else:
                    st.success("✅ **Good Diversification** - Low correlations between selected stocks")
        
        except Exception as e:
            st.warning(f"Correlation analysis failed: {e}")
    
    # Risk recommendations
    st.subheader("💡 Risk Management Recommendations")
    
    recommendations = []
    
    if portfolio_var > 0.15:
        recommendations.append("⚠️ **High VaR** - Consider reducing position sizes or adding defensive stocks")
    
    if max_drawdown > 0.20:
        recommendations.append("🛡️ **High Drawdown Risk** - Implement stop-loss strategies")
    
    if sharpe_ratio < 0.5:
        recommendations.append("📈 **Low Risk-Adjusted Returns** - Consider rebalancing portfolio")
    
    if len(recommendations) == 0:
        recommendations.append("✅ **Risk Profile Acceptable** - Current risk levels are within acceptable ranges")
    
    for rec in recommendations:
        if rec.startswith("⚠️") or rec.startswith("🛡️"):
            st.warning(rec)
        elif rec.startswith("📈"):
            st.info(rec)
        else:
            st.success(rec)

def create_backtesting_results_tab(backtest_results: Dict, selected_tickers: List[str]):
    """Enhanced backtesting results display"""
    
    st.header("📈 Historical Backtesting Results")
    
    if not backtest_results:
        st.warning("Backtesting results not available.")
        return
    
    st.success("✅ Historical backtesting completed")
    
    # Performance metrics
    st.subheader("📊 Strategy Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = backtest_results.get('total_return', 0)
        st.metric("Total Return", f"{total_return:.1%}")
    
    with col2:
        annualized_return = backtest_results.get('annualized_return', 0)
        st.metric("Annualized Return", f"{annualized_return:.1%}")
    
    with col3:
        max_drawdown = backtest_results.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    with col4:
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        win_rate = backtest_results.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    with col2:
        total_trades = backtest_results.get('total_trades', 0)
        st.metric("Total Trades", total_trades)
    
    with col3:
        avg_trade = backtest_results.get('avg_trade_return', 0)
        st.metric("Avg Trade Return", f"{avg_trade:.1%}")
    
    with col4:
        volatility = backtest_results.get('volatility', 0)
        st.metric("Strategy Volatility", f"{volatility:.1%}")
    
    # Performance assessment
    st.subheader("🎯 Performance Assessment")
    
    if sharpe_ratio > 1.5:
        st.success("🌟 **Excellent Performance** - Strategy shows outstanding risk-adjusted returns")
    elif sharpe_ratio > 1.0:
        st.success("✅ **Good Performance** - Strategy demonstrates solid risk-adjusted returns")
    elif sharpe_ratio > 0.5:
        st.info("⚠️ **Fair Performance** - Strategy shows moderate risk-adjusted returns")
    else:
        st.warning("❌ **Poor Performance** - Strategy underperforms risk-free investments")
    
    # Detailed analysis
    if win_rate > 0.6:
        st.info(f"💪 **High Win Rate** - {win_rate:.1%} of trades were profitable")
    elif win_rate < 0.4:
        st.warning(f"⚠️ **Low Win Rate** - Only {win_rate:.1%} of trades were profitable")
    
    if max_drawdown > 0.20:
        st.error("🔴 **High Drawdown** - Strategy experienced significant losses during worst period")
    elif max_drawdown < 0.10:
        st.success("✅ **Low Drawdown** - Strategy maintained stable performance")

def create_data_quality_tab(raw_data: Dict, featured_data: Dict, selected_tickers: List[str]):
    """Enhanced data quality assessment"""
    
    st.header("📋 Data Quality & Coverage Report")
    
    # Data quality metrics
    quality_data = []
    
    for ticker in selected_tickers:
        quality_metrics = {
            'Stock': ticker,
            'Raw Data Available': ticker in raw_data,
            'Featured Data Available': ticker in featured_data,
            'Data Points': 0,
            'Date Range': 'N/A',
            'Missing Data': 0,
            'Data Quality': 'No Data'
        }
        
        if ticker in raw_data:
            df = raw_data[ticker]
            
            if not df.empty:
                quality_metrics.update({
                    'Data Points': len(df),
                    'Date Range': f"{(df.index.max() - df.index.min()).days} days",
                    'Missing Data': df.isnull().sum().sum(),
                })
                
                # Assess quality
                if len(df) >= 500 and df.isnull().sum().sum() < len(df) * 0.1:
                    quality_metrics['Data Quality'] = 'Excellent'
                elif len(df) >= 250 and df.isnull().sum().sum() < len(df) * 0.2:
                    quality_metrics['Data Quality'] = 'Good'
                elif len(df) >= 100:
                    quality_metrics['Data Quality'] = 'Fair'
                else:
                    quality_metrics['Data Quality'] = 'Poor'
        
        quality_data.append(quality_metrics)
    
    quality_df = pd.DataFrame(quality_data)
    
    # Summary metrics
    st.subheader("📊 Data Quality Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = len(selected_tickers)
        st.metric("Total Stocks", total_stocks)
    
    with col2:
        data_available = sum(quality_df['Raw Data Available'])
        st.metric("Data Available", f"{data_available}/{total_stocks}")
    
    with col3:
        excellent_quality = (quality_df['Data Quality'] == 'Excellent').sum()
        st.metric("Excellent Quality", f"{excellent_quality}/{total_stocks}")
    
    with col4:
        total_data_points = quality_df['Data Points'].sum()
        st.metric("Total Data Points", f"{total_data_points:,}")
    
    # Detailed quality table
    st.subheader("📋 Detailed Quality Assessment")
    
    # Format the display DataFrame
    display_quality = quality_df.copy()
    display_quality['Raw Data Available'] = display_quality['Raw Data Available'].apply(lambda x: '✅' if x else '❌')
    display_quality['Featured Data Available'] = display_quality['Featured Data Available'].apply(lambda x: '✅' if x else '❌')
    display_quality['Data Points'] = display_quality['Data Points'].apply(lambda x: f"{x:,}" if x > 0 else "0")
    
    st.dataframe(display_quality, use_container_width=True, hide_index=True)
    
    # Data quality insights
    st.subheader("💡 Data Quality Insights")
    
    excellent_stocks = (quality_df['Data Quality'] == 'Excellent').sum()
    good_stocks = (quality_df['Data Quality'] == 'Good').sum()
    fair_stocks = (quality_df['Data Quality'] == 'Fair').sum()
    poor_stocks = (quality_df['Data Quality'] == 'Poor').sum()
    
    if excellent_stocks >= len(selected_tickers) * 0.8:
        st.success("🌟 **Excellent Data Quality** - Most stocks have comprehensive historical data")
    elif good_stocks + excellent_stocks >= len(selected_tickers) * 0.6:
        st.info("✅ **Good Data Quality** - Sufficient data for reliable analysis")
    else:
        st.warning("⚠️ **Limited Data Quality** - Consider selecting stocks with more historical data")
    
    # Recommendations
    if poor_stocks > 0:
        poor_stock_list = quality_df[quality_df['Data Quality'] == 'Poor']['Stock'].tolist()
        st.warning(f"📊 **Data Limited Stocks:** {', '.join(poor_stock_list)} - Consider replacing with stocks having more historical data")

def create_model_details_tab(models: Dict, config: Dict, selected_tickers: List[str]):
    """Enhanced model details and configuration display"""
    
    st.header("🤖 ML Model Details & Configuration")
    
    if not models:
        st.warning("No model details available.")
        return
    
    # Model overview
    st.subheader("📊 Model Training Summary")
    
    total_models = sum(len(model_dict) for model_dict in models.values())
    model_types = config.get('model_types', [])
    ensemble_method = config.get('ensemble_method', 'weighted_average')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models Trained", total_models)
    
    with col2:
        st.metric("Stocks with Models", len(models))
    
    with col3:
        st.metric("Model Types Used", len(model_types))
    
    with col4:
        st.info(f"**Ensemble:** {ensemble_method.replace('_', ' ').title()}")
    
    # Model configuration details
    st.subheader("⚙️ Configuration Details")
    
    config_details = {
        'Investment Horizon': config.get('investment_horizon', 'N/A'),
        'Model Types': ', '.join(model_types),
        'Ensemble Method': ensemble_method.replace('_', ' ').title(),
        'Hyperparameter Tuning': 'Enabled' if config.get('hyperparameter_tuning', False) else 'Disabled',
        'Cross Validation': f"{config.get('cross_validation_folds', 5)} folds",
        'Feature Selection': 'Enabled' if config.get('feature_selection', True) else 'Disabled',
        'Parallel Processing': 'Enabled' if config.get('parallel_processing', True) else 'Disabled'
    }
    
    for key, value in config_details.items():
        st.info(f"**{key}:** {value}")
    
    # Per-stock model details
    st.subheader("📋 Stock-wise Model Details")
    
    model_details = []
    for ticker, ticker_models in models.items():
        model_count = len(ticker_models)
        model_list = list(ticker_models.keys())
        
        model_details.append({
            'Stock': ticker,
            'Models Trained': model_count,
            'Model Keys': ', '.join(model_list[:3]) + ('...' if len(model_list) > 3 else ''),
            'Success Rate': '100%' if model_count > 0 else '0%'
        })
    
    model_details_df = pd.DataFrame(model_details)
    st.dataframe(model_details_df, use_container_width=True, hide_index=True)
    
    # Model performance insights
    st.subheader("🎯 Model Training Insights")
    
    successful_stocks = len(models)
    total_stocks = len(selected_tickers)
    success_rate = successful_stocks / total_stocks if total_stocks > 0 else 0
    
    if success_rate >= 0.9:
        st.success("🌟 **Excellent Training Success** - Models trained successfully for almost all stocks")
    elif success_rate >= 0.7:
        st.success("✅ **Good Training Success** - Models trained for most stocks")
    elif success_rate >= 0.5:
        st.warning("⚠️ **Moderate Training Success** - Some stocks may have insufficient data")
    else:
        st.error("❌ **Low Training Success** - Consider selecting different stocks or adjusting parameters")
    
    # Configuration recommendations
    st.subheader("💡 Configuration Recommendations")
    
    recommendations = []
    
    if not config.get('hyperparameter_tuning', False):
        recommendations.append("🔧 Consider enabling hyperparameter tuning for better model performance")
    
    if len(model_types) < 2:
        recommendations.append("🤖 Consider using multiple model types for better ensemble performance")
    
    if config.get('cross_validation_folds', 5) < 5:
        recommendations.append("📊 Consider increasing cross-validation folds for more robust validation")
    
    if len(recommendations) == 0:
        recommendations.append("✅ **Configuration Optimal** - Current settings are well-configured for analysis")
    
    for rec in recommendations:
        if rec.startswith("✅"):
            st.success(rec)
        else:
            st.info(rec)

def display_comprehensive_results(predictions_df: pd.DataFrame,
                                price_targets_df: pd.DataFrame,
                                raw_data: Dict,
                                featured_data: Dict,
                                selected_tickers: List[str],
                                models: Dict,
                                config: Dict,
                                forecast_results: Optional[Dict] = None,
                                risk_analysis: Optional[Dict] = None,
                                portfolio_optimization: Optional[Dict] = None,
                                sentiment_analysis: Optional[Dict] = None,
                                backtest_results: Optional[Dict] = None,
                                report_data: Optional[Dict] = None):
    """Display comprehensive analysis results with all features"""
    
    # Display performance report first
    if report_data:
        display_comprehensive_performance_report(report_data)
    
    st.markdown("---")
    
    # Create enhanced analysis tabs
    create_enhanced_analysis_tabs(
        predictions_df=predictions_df,
        price_targets_df=price_targets_df,
        raw_data=raw_data,
        featured_data=featured_data,
        selected_tickers=selected_tickers,
        models=models,
        config=config,
        forecast_results=forecast_results,
        risk_analysis=risk_analysis,
        portfolio_optimization=portfolio_optimization,
        sentiment_analysis=sentiment_analysis,
        backtest_results=backtest_results,
        report_data=report_data
    )
    
    # Action buttons
    st.markdown("---")
    st.subheader("🔄 Actions & Export")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Results", use_container_width=True):
            # Create downloadable results
            results_summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'selected_stocks': selected_tickers,
                'configuration': config,
                'performance_metrics': report_data,
                'predictions_summary': {
                    'total_predictions': len(predictions_df),
                    'avg_confidence': predictions_df['ensemble_confidence'].mean() if not predictions_df.empty and 'ensemble_confidence' in predictions_df.columns else 0,
                    'bullish_count': (predictions_df['predicted_return'] > 0).sum() if not predictions_df.empty and 'predicted_return' in predictions_df.columns else 0
                }
            }
            
            st.download_button(
                "⬇️ Download Analysis Report",
                data=pd.DataFrame([results_summary]).to_json(indent=2),
                file_name=f"ai_stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Export Predictions", use_container_width=True):
            if not predictions_df.empty:
                csv_data = predictions_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_data,
                    file_name=f"stock_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No predictions to export")
    
    with col3:
        if st.button("🔄 Re-run Analysis", use_container_width=True):
            # Clear session state and rerun
            for key in list(st.session_state.keys()):
                if 'analysis_results' in key:
                    del st.session_state[key]
            st.experimental_rerun()
    
    with col4:
        if st.button("⚙️ Modify Configuration", use_container_width=True):
            st.info("💡 Modify settings in the sidebar and re-run analysis")

# ==================== SYSTEM STATUS DISPLAY ====================

def display_system_status():
    """Display comprehensive system module status"""
    with st.expander("🔧 System Status & Module Availability", expanded=False):
        st.markdown("**📊 Module Status Overview:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Modules:**")
            core_modules = ['data_loader', 'feature_engineer', 'model']
            for module in core_modules:
                status = MODULES_STATUS.get(module, False)
                if status:
                    st.markdown(f'<span class="status-indicator status-success">✅ {module.replace("_", " ").title()}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-indicator status-error">❌ {module.replace("_", " ").title()} (Fallback)</span>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Advanced Features:**")
            advanced_features = {
                'Risk Management': RISK_MANAGEMENT_AVAILABLE,
                'Backtesting Engine': BACKTESTING_AVAILABLE,
                'Sentiment Analysis': SENTIMENT_AVAILABLE,
                'Portfolio Optimization': PORTFOLIO_OPTIMIZATION_AVAILABLE
            }
            
            for feature, available in advanced_features.items():
                if available:
                    st.markdown(f'<span class="status-indicator status-success">✅ {feature}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-indicator status-warning">⚠️ {feature} (Fallback)</span>', unsafe_allow_html=True)
        
        # System configuration summary
        st.markdown("---")
        st.markdown("**⚙️ System Configuration:**")
        
        config_status = {
            'Data Source': 'Enhanced Multi-Source' if MODULES_STATUS.get('data_loader') else 'Yahoo Finance (Fallback)',
            'ML Pipeline': 'Advanced Ensemble' if MODULES_STATUS.get('model') else 'Basic Models',
            'Feature Engineering': 'Comprehensive' if MODULES_STATUS.get('feature_engineer') else 'Technical Indicators Only',
            'Risk Analysis': 'Full Portfolio Risk Management' if RISK_MANAGEMENT_AVAILABLE else 'Basic Risk Metrics',
            'Performance Testing': 'Advanced Backtesting' if BACKTESTING_AVAILABLE else 'Simple Performance Analysis',
            'Market Sentiment': 'Real-time News Analysis' if SENTIMENT_AVAILABLE else 'Not Available',
            'Portfolio Optimization': 'Modern Portfolio Theory' if PORTFOLIO_OPTIMIZATION_AVAILABLE else 'Equal Weight Fallback'
        }
        
        for component, status in config_status.items():
            st.info(f"**{component}:** {status}")
        
        # Performance recommendations
        st.markdown("**💡 System Recommendations:**")
        
        missing_features = []
        if not RISK_MANAGEMENT_AVAILABLE:
            missing_features.append("Risk Management")
        if not BACKTESTING_AVAILABLE:
            missing_features.append("Backtesting")
        if not SENTIMENT_AVAILABLE:
            missing_features.append("Sentiment Analysis")
        if not PORTFOLIO_OPTIMIZATION_AVAILABLE:
            missing_features.append("Portfolio Optimization")
        
        if missing_features:
            st.warning(f"⚠️ **Missing Advanced Features:** {', '.join(missing_features)}")
            st.info("💡 Install missing dependencies to enable full functionality")
        else:
            st.success("🌟 **All Systems Operational** - Running with complete functionality!")
        
        # System health check
        total_modules = len(MODULES_STATUS)
        working_modules = sum(MODULES_STATUS.values())
        health_score = working_modules / total_modules * 100
        
        if health_score >= 80:
            st.success(f"🟢 **System Health: {health_score:.0f}%** - Excellent")
        elif health_score >= 60:
            st.warning(f"🟡 **System Health: {health_score:.0f}%** - Good")
        else:
            st.error(f"🔴 **System Health: {health_score:.0f}%** - Needs Attention")

# ==================== MAIN APPLICATION ====================

def main():
    """Main application function with comprehensive features"""
    
    # Enhanced header
    st.markdown('<h1 class="main-header">🤖 AI Stock Advisor Pro - Complete Enhanced Edition</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='color: #2c3e50; margin: 0; font-weight: 600;'>Advanced Stock Analysis with AI-Powered Predictions & Comprehensive Risk Management</h3>
        <p style='color: #7f8c8d; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>Enhanced with Monte Carlo forecasting, portfolio optimization, sentiment analysis, and advanced backtesting</p>
        <div style='margin-top: 1rem;'>
            <span style='background: #28a745; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem; margin: 0.2rem;'>✓ ML Ensemble</span>
            <span style='background: #17a2b8; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem; margin: 0.2rem;'>✓ Risk Management</span>
            <span style='background: #ffc107; color: black; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem; margin: 0.2rem;'>✓ Portfolio Optimization</span>
            <span style='background: #6f42c1; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem; margin: 0.2rem;'>✓ Sentiment Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System status display
    display_system_status()
    
    # Enhanced sidebar configuration
    selected_tickers = create_enhanced_stock_selection_interface()
    full_config = create_enhanced_configuration_interface()
    
    # Main analysis section with enhanced features
    st.markdown("---")
    
    if not selected_tickers:
        # Enhanced welcome section
        st.markdown("""
        <div class="portfolio-summary">
            <h2 style="margin-top: 0;">🚀 Welcome to AI Stock Advisor Pro</h2>
            <p>Experience the most comprehensive AI-powered stock analysis platform with advanced risk management and portfolio optimization.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ **Get Started:** Please select stocks from the sidebar to begin comprehensive analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 **How to Get Started:**
            1. **📊 Select Stocks** - Choose 3-10 stocks from the sidebar using categories or custom entry
            2. **⚙️ Configure Analysis** - Set your investment horizon, risk tolerance, and enable advanced features
            3. **🚀 Run Analysis** - Click the analysis button to generate comprehensive insights
            4. **📈 Review Results** - Explore predictions, risk analysis, and portfolio optimization
            5. **💾 Export & Act** - Download reports and implement your investment strategy
            """)
        
        with col2:
            st.markdown("""
            ### 🌟 **Advanced Features Available:**
            - **🤖 ML Ensemble Models** - XGBoost, LightGBM, Random Forest combinations
            - **🔮 Monte Carlo Forecasting** - Multiple scenario price predictions
            - **🛡️ Risk Management** - VaR, correlation analysis, drawdown tracking
            - **💼 Portfolio Optimization** - Modern Portfolio Theory allocation
            - **📰 Sentiment Analysis** - Real-time news sentiment integration
            - **📈 Advanced Backtesting** - Historical strategy performance validation
            - **📊 Interactive Visualizations** - Dynamic charts and technical analysis
            """)
        
        # Feature showcase
        st.markdown("### 🎨 Feature Showcase")
        
        feature_tabs = st.tabs(["🤖 AI Models", "🔮 Forecasting", "🛡️ Risk Management", "💼 Portfolio"])
        
        with feature_tabs[0]:
            st.markdown("""
            #### Advanced Machine Learning Pipeline
            - **Ensemble Methods:** Combines multiple AI models for superior accuracy
            - **Feature Engineering:** 50+ technical and fundamental indicators
            - **Hyperparameter Optimization:** Automatic model tuning for best performance
            - **Cross-Validation:** Robust model validation to prevent overfitting
            """)
            
            # Mock performance chart
            sample_performance = pd.DataFrame({
                'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Neural Network', 'Ensemble'],
                'Accuracy': [0.78, 0.76, 0.73, 0.75, 0.82],
                'Confidence': [0.85, 0.83, 0.80, 0.77, 0.88]
            })
            
            fig = px.bar(sample_performance, x='Model', y='Accuracy', color='Confidence',
                        title='AI Model Performance Comparison', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with feature_tabs[1]:
            st.markdown("""
            #### Monte Carlo Price Forecasting
            - **Scenario Analysis:** Bullish, bearish, and base case predictions
            - **Confidence Intervals:** Statistical probability ranges
            - **Risk-Adjusted Targets:** Expected returns with volatility consideration
            - **Time-Series Modeling:** Advanced mathematical forecasting methods
            """)
        
        with feature_tabs[2]:
            st.markdown("""
            #### Comprehensive Risk Management
            - **Value at Risk (VaR):** Maximum expected loss calculations
            - **Correlation Analysis:** Portfolio diversification assessment
            - **Drawdown Tracking:** Maximum portfolio decline monitoring
            - **Stress Testing:** Performance under adverse market conditions
            """)
        
        with feature_tabs[3]:
            st.markdown("""
            #### Modern Portfolio Optimization
            - **Efficient Frontier:** Optimal risk-return combinations
            - **Weight Allocation:** Mathematically optimized position sizing
            - **Constraint Optimization:** Custom investment restrictions
            - **Rebalancing Strategies:** Dynamic portfolio adjustment recommendations
            """)
        
        return
    
    # Display enhanced configuration summary
    st.markdown("### 📋 **Analysis Configuration Summary**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        **📊 Portfolio:**
        - **Stocks:** {len(selected_tickers)}
        - **Investment:** ₹{full_config.get('investment_amount', 500000):,}
        - **Risk Level:** {full_config.get('risk_tolerance', 'Moderate')}
        """)
    
    with col2:
        st.markdown(f"""
        **🤖 Models:**
        - **Types:** {len(full_config.get('model_types', []))}
        - **Horizon:** {full_config.get('investment_horizon', 'N/A')}
        - **Ensemble:** {full_config.get('ensemble_method', 'N/A').replace('_', ' ').title()}
        """)
    
    with col3:
        advanced_features = sum([
            full_config.get('enable_enhanced_forecasting', False),
            full_config.get('enable_risk_management', False),
            full_config.get('enable_portfolio_optimization', False),
            full_config.get('enable_sentiment_analysis', False),
            full_config.get('enable_backtesting', False)
        ])
        
        st.markdown(f"""
        **🚀 Features:**
        - **Advanced:** {advanced_features}/5 enabled
        - **Forecasting:** {'✅' if full_config.get('enable_enhanced_forecasting', False) else '❌'}
        - **Risk Mgmt:** {'✅' if full_config.get('enable_risk_management', False) else '❌'}
        """)
    
    with col4:
        st.markdown(f"""
        **⚙️ Performance:**
        - **Parallel:** {'✅' if full_config.get('parallel_processing', True) else '❌'}
        - **Caching:** {'✅' if full_config.get('cache_results', True) else '❌'}
        - **Tuning:** {'✅' if full_config.get('hyperparameter_tuning', False) else '❌'}
        """)
    
    # Selected stocks overview
    with st.expander("📋 Selected Stocks Overview", expanded=False):
        cols = st.columns(min(5, len(selected_tickers)))
        for i, ticker in enumerate(selected_tickers):
            with cols[i % 5]:
                st.info(f"**{ticker}**\nSelected for analysis")
    
    # Main analysis execution button with enhanced styling
    st.markdown("---")
    
    analysis_container = st.container()
    with analysis_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 **Run Comprehensive AI Analysis**", 
                type="primary", 
                use_container_width=True,
                help="Execute complete analysis pipeline with all enabled features"
            ):
                # Run the comprehensive analysis
                run_enhanced_comprehensive_analysis(selected_tickers, full_config)
    
    # Load and display previous results if available
    if 'comprehensive_analysis_results' in st.session_state:
        st.markdown("---")
        st.info("📋 **Displaying Previous Analysis Results** - Use the button above to run a fresh analysis")
        
        results = st.session_state['comprehensive_analysis_results']
        
        # Check if results are recent (within last hour)
        result_time = results.get('timestamp', datetime.now() - timedelta(hours=2))
        if isinstance(result_time, str):
            result_time = datetime.fromisoformat(result_time)
        
        time_diff = datetime.now() - result_time
        if time_diff.total_seconds() > 3600:  # 1 hour
            st.warning("⏰ **Results are over 1 hour old** - Consider running a fresh analysis for current market conditions")
        
        # Display comprehensive results
        display_comprehensive_results(
            predictions_df=results.get('predictions', pd.DataFrame()),
            price_targets_df=results.get('price_targets', pd.DataFrame()),
            raw_data=results.get('raw_data', {}),
            featured_data=results.get('featured_data', {}),
            selected_tickers=results.get('selected_tickers', selected_tickers),
            models=results.get('models', {}),
            config=results.get('config', full_config),
            forecast_results=results.get('forecast_results', {}),
            risk_analysis=results.get('risk_analysis', None),
            portfolio_optimization=results.get('portfolio_optimization', None),
            sentiment_analysis=results.get('sentiment_analysis', None),
            backtest_results=results.get('backtest_results', None),
            report_data=results.get('report_data', {})
        )
    
    # Enhanced footer with system info
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 30px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;'>
        <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;'><strong>🤖 AI Stock Advisor Pro - Complete Enhanced Edition</strong></p>
        <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;'>
            <div>📊 <strong>Portfolio:</strong> {len(selected_tickers)} stocks selected</div>
            <div>🎯 <strong>Horizon:</strong> {full_config.get('investment_horizon', 'N/A')}</div>
            <div>🛡️ <strong>Risk Mgmt:</strong> {'Advanced' if RISK_MANAGEMENT_AVAILABLE else 'Basic'}</div>
            <div>📈 <strong>Backtesting:</strong> {'Enhanced' if BACKTESTING_AVAILABLE else 'Simple'}</div>
        </div>
        <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;'>
            <div>🚀 <strong>ML Models:</strong> {'Advanced Pipeline' if MODULES_STATUS.get('model', False) else 'Fallback'}</div>
            <div>📰 <strong>Sentiment:</strong> {'Real-time' if SENTIMENT_AVAILABLE else 'Not Available'}</div>
            <div>💼 <strong>Optimization:</strong> {'Modern Portfolio Theory' if PORTFOLIO_OPTIMIZATION_AVAILABLE else 'Equal Weight'}</div>
        </div>
        <div style='border-top: 1px solid #dee2e6; padding-top: 1rem; margin-top: 1rem;'>
            <p style='margin: 0.5rem 0; font-style: italic;'>⚠️ <strong>Important Disclaimer:</strong> This tool provides AI-powered analysis for educational and informational purposes only.</p>
            <p style='margin: 0.5rem 0; font-style: italic;'>📞 <strong>Always consult with qualified financial advisors before making investment decisions.</strong></p>
            <p style='margin: 0.5rem 0; color: #28a745;'>💡 <strong>Pro Tip:</strong> Use this analysis as a starting point for your investment research, not as the sole basis for investment decisions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"💥 **Critical Application Error:** {str(e)}")
        st.error("**Recovery Options:**")
        st.error("1. 🔄 **Refresh the page** and try again")
        st.error("2. 🧹 **Clear browser cache** and restart")
        st.error("3. 📉 **Select fewer stocks** (3-5 recommended)")
        st.error("4. ⚙️ **Disable advanced features** temporarily")
        st.error("5. 🌐 **Check internet connection** and retry")
        
        # Comprehensive error information for debugging
        with st.expander("🔧 **Detailed Technical Error Information**", expanded=False):
            st.code(f"Error Type: {type(e).__name__}")
            st.code(f"Error Message: {str(e)}")
            st.code(f"Module Status: {MODULES_STATUS}")
            st.code(f"Advanced Features: Risk={RISK_MANAGEMENT_AVAILABLE}, Backtest={BACKTESTING_AVAILABLE}, Sentiment={SENTIMENT_AVAILABLE}, Portfolio={PORTFOLIO_OPTIMIZATION_AVAILABLE}")
            st.exception(e)
        
        # Emergency fallback interface
        st.markdown("---")
        st.markdown("### 🚨 **Emergency Mode - Basic Analysis**")
        st.warning("The full application encountered an error. You can try basic analysis below:")
        
        emergency_tickers = st.multiselect(
            "Select up to 3 stocks for basic analysis:",
            ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"],
            default=["RELIANCE.NS"],
            max_selections=3
        )
        
        if emergency_tickers and st.button("🔧 **Run Emergency Analysis**", type="secondary"):
            try:
                st.info("🔄 Running basic emergency analysis...")
                
                import yfinance as yf
                
                emergency_data = {}
                for ticker in emergency_tickers:
                    try:
                        data = yf.download(ticker, period="1y", progress=False)
                        if not data.empty:
                            emergency_data[ticker] = data
                    except Exception as download_error:
                        st.warning(f"Failed to load {ticker}: {download_error}")
                
                if emergency_data:
                    st.success(f"✅ Emergency data loaded for {len(emergency_data)} stocks")
                    
                    for ticker, df in emergency_data.items():
                        with st.expander(f"📊 {ticker} - Basic Analysis"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                current_price = df['Close'].iloc[-1]
                                st.metric("Current Price", f"₹{current_price:.2f}")
                            
                            with col2:
                                month_return = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) if len(df) > 22 else 0
                                st.metric("1-Month Return", f"{month_return:.1%}")
                            
                            with col3:
                                volatility = df['Close'].pct_change().std() * np.sqrt(252)
                                st.metric("Volatility", f"{volatility:.1%}")
                            
                            # Simple price chart
                            fig = px.line(
                                df.reset_index(), 
                                x='Date', 
                                y='Close', 
                                title=f'{ticker} - Price Chart (1 Year)'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Emergency analysis failed - no data could be loaded")
                    
            except Exception as emergency_error:
                st.error(f"❌ Emergency analysis also failed: {emergency_error}")
                st.info("💡 **Final Suggestion:** Please refresh the page and try again with a stable internet connection")
        
        # Contact and support information
        st.markdown("---")
        st.info("""
        🆘 **Need Help?**
        
        If you continue experiencing issues:
        1. **Check System Requirements:** Ensure you have a stable internet connection
        2. **Browser Compatibility:** Use a modern browser (Chrome, Firefox, Safari, Edge)
        3. **Clear Cache:** Clear your browser cache and cookies
        4. **Restart Application:** Close and reopen your browser tab
        
        **System Health Check:**
        - Data Loading: {'✅ Working' if MODULES_STATUS.get('data_loader', False) else '❌ Issues Detected'}
        - ML Models: {'✅ Working' if MODULES_STATUS.get('model', False) else '❌ Issues Detected'}
        - Feature Engineering: {'✅ Working' if MODULES_STATUS.get('feature_engineer', False) else '❌ Issues Detected'}
        """)

    finally:
        # Cleanup and memory management
        try:
            gc.collect()  # Force garbage collection
        except:
            pass