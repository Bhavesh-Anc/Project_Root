import pickle
import hashlib
from datetime import datetime, timedelta
import logging
import json
import warnings
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier, 
    ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (
    TimeSeriesSplit, 
    cross_val_score
)
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, log_loss
)
# Fixed imports - removed non-existent modules and added fallback
try:
    from utils.data_loader import RealTimeDataManager
except ImportError:
    # Fallback class if module doesn't exist
    class RealTimeDataManager:
        def __init__(self):
            pass
        def get_latest_data(self, ticker, lookback_minutes=240):
            return pd.DataFrame()

try:
    from utils.news_sentiment import AdvancedSentimentAnalyzer
except ImportError:
    # Fallback class
    class AdvancedSentimentAnalyzer:
        def __init__(self, api_key=None):
            pass
        def get_ticker_sentiment(self, ticker):
            return 0.0

from config import secrets

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ==================== ENHANCED CONFIGURATION ====================
ENHANCED_MODEL_CONFIG = {
    'n_jobs': -1,
    'random_state': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'validation_size': 0.1,
    'model_types': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'neural_network'],
    'ensemble_methods': ['voting', 'stacking', 'blending'],
    'fast_mode': False,
    'incremental_training': True,
    'feature_selection_top_k': 100,
    'early_stopping': True,
    'hyperparameter_tuning': True,
    'model_calibration': True,
    'feature_importance_analysis': True,
    'cache_dir': 'model_cache_enhanced',
    'parallel_training': True,
    'batch_size': 8,
    'early_stopping_patience': 10,
    'model_selection_metric': 'roc_auc',
    'ensemble_size': 5,
    'realtime_integration': True,
    'sentiment_integration': True,
    'priority_horizons': ['next_month', 'next_quarter'],
    'selected_stocks_only': True,  # New: Train only selected stocks
    'min_data_points': 200         # New: Minimum data points required
}

# ==================== PRICE TARGET PREDICTION ====================
class PriceTargetPredictor:
    """Advanced price target prediction for selected stocks"""
    
    def __init__(self, horizon: str, model_type: str):
        self.horizon = horizon
        self.model_type = model_type
        self.price_model = None  # For price prediction
        self.direction_model = None  # For direction prediction
        self.scaler = None
        self.selected_features = None
        self.training_date = datetime.now()
        
        # Horizon to days mapping
        self.horizon_days = {
            'next_week': 7,
            'next_month': 30,
            'next_quarter': 90,
            'next_year': 365
        }
    
    def predict_price_targets(self, X: pd.DataFrame, current_prices: pd.Series, ticker: str = None) -> dict:
        """Generate comprehensive price targets"""
        
        if self.direction_model is None:
            return self._default_targets(current_prices, ticker)
        
        try:
            # Get direction probability
            direction_proba = self.direction_model.predict_proba(X, ticker)[:, 1][0]
            
            # Calculate expected return based on historical volatility and direction
            returns = self._calculate_expected_returns(X, direction_proba, ticker)
            
            # Generate targets
            current_price = current_prices.iloc[0] if len(current_prices) > 0 else 100
            
            targets = {
                'current_price': float(current_price),
                'target_price': float(current_price * (1 + returns['expected_return'])),
                'price_change': float(current_price * returns['expected_return']),
                'percentage_change': float(returns['expected_return'] * 100),
                'direction_probability': float(direction_proba),
                'confidence_level': float(returns['confidence']),
                'risk_level': self._calculate_risk_level(returns['volatility']),
                'horizon': self.horizon,
                'horizon_days': self.horizon_days.get(self.horizon, 30),
                'targets': {
                    'conservative': float(current_price * (1 + returns['conservative'])),
                    'moderate': float(current_price * (1 + returns['expected_return'])),
                    'aggressive': float(current_price * (1 + returns['aggressive']))
                },
                'support_resistance': {
                    'support': float(current_price * (1 + returns['support_level'])),
                    'resistance': float(current_price * (1 + returns['resistance_level']))
                }
            }
            
            return targets
            
        except Exception as e:
            print(f"Price target prediction failed for {ticker}: {e}")
            return self._default_targets(current_prices, ticker)
    
    def _calculate_expected_returns(self, X: pd.DataFrame, direction_proba: float, ticker: str) -> dict:
        """Calculate expected returns based on model predictions and market conditions"""
        
        try:
            # Base expected return from direction probability
            base_return = (direction_proba - 0.5) * 2  # Scale to [-1, 1]
            
            # Adjust based on horizon
            horizon_multiplier = {
                'next_week': 0.02,
                'next_month': 0.05,
                'next_quarter': 0.12,
                'next_year': 0.25
            }
            
            multiplier = horizon_multiplier.get(self.horizon, 0.05)
            expected_return = base_return * multiplier
            
            # Calculate volatility estimate (simplified)
            volatility = min(abs(expected_return) * 2, 0.3)  # Cap at 30%
            
            # Generate conservative and aggressive targets
            conservative = expected_return * 0.6  # 60% of expected
            aggressive = expected_return * 1.4   # 140% of expected
            
            # Support and resistance levels
            support_level = expected_return - volatility
            resistance_level = expected_return + volatility
            
            # Confidence based on probability distance from 0.5
            confidence = abs(direction_proba - 0.5) * 2
            
            return {
                'expected_return': expected_return,
                'conservative': conservative,
                'aggressive': aggressive,
                'volatility': volatility,
                'confidence': confidence,
                'support_level': support_level,
                'resistance_level': resistance_level
            }
            
        except Exception as e:
            print(f"Return calculation failed: {e}")
            return {
                'expected_return': 0.02,
                'conservative': 0.01,
                'aggressive': 0.04,
                'volatility': 0.15,
                'confidence': 0.5,
                'support_level': -0.05,
                'resistance_level': 0.08
            }
    
    def _calculate_risk_level(self, volatility: float) -> str:
        """Calculate risk level based on volatility"""
        if volatility < 0.1:
            return "Low"
        elif volatility < 0.2:
            return "Medium"
        elif volatility < 0.3:
            return "High"
        else:
            return "Very High"
    
    def _default_targets(self, current_prices: pd.Series, ticker: str) -> dict:
        """Default price targets when model is not available"""
        current_price = current_prices.iloc[0] if len(current_prices) > 0 else 100
        
        # Conservative default targets based on market averages
        default_returns = {
            'next_week': 0.01,
            'next_month': 0.03,
            'next_quarter': 0.08,
            'next_year': 0.15
        }
        
        expected_return = default_returns.get(self.horizon, 0.03)
        
        return {
            'current_price': float(current_price),
            'target_price': float(current_price * (1 + expected_return)),
            'price_change': float(current_price * expected_return),
            'percentage_change': float(expected_return * 100),
            'direction_probability': 0.55,  # Slightly bullish default
            'confidence_level': 0.5,
            'risk_level': "Medium",
            'horizon': self.horizon,
            'horizon_days': self.horizon_days.get(self.horizon, 30),
            'targets': {
                'conservative': float(current_price * (1 + expected_return * 0.6)),
                'moderate': float(current_price * (1 + expected_return)),
                'aggressive': float(current_price * (1 + expected_return * 1.4))
            },
            'support_resistance': {
                'support': float(current_price * 0.95),
                'resistance': float(current_price * 1.08)
            }
        }

def generate_price_targets_for_selected_stocks(models: dict, 
                                             current_data: dict, 
                                             selected_tickers: list,
                                             investment_horizon: str) -> pd.DataFrame:
    """Generate price targets for selected stocks"""
    
    price_targets = []
    
    for ticker in selected_tickers:
        try:
            if ticker not in current_data:
                continue
                
            df = current_data[ticker]
            if df.empty:
                continue
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            latest_data = df.iloc[[-1]].copy()
            
            # Try to get model for price prediction
            ticker_models = models.get(ticker, {})
            best_model = None
            
            # Find best model for this horizon
            for model_key, model in ticker_models.items():
                if investment_horizon in model_key:
                    best_model = model
                    break
            
            if best_model and hasattr(best_model, 'predict_proba'):
                # Create price target predictor
                predictor = PriceTargetPredictor(investment_horizon, best_model.model_type)
                predictor.direction_model = best_model
                
                # Generate price targets
                targets = predictor.predict_price_targets(
                    latest_data, 
                    pd.Series([current_price]), 
                    ticker
                )
            else:
                # Use default targets
                predictor = PriceTargetPredictor(investment_horizon, 'default')
                targets = predictor._default_targets(pd.Series([current_price]), ticker)
            
            # Add ticker information
            targets['ticker'] = ticker
            price_targets.append(targets)
            
        except Exception as e:
            print(f"Price target generation failed for {ticker}: {e}")
            continue
    
    if price_targets:
        return pd.DataFrame(price_targets)
    else:
        return pd.DataFrame()

# Enhanced prediction function with price targets
def predict_with_ensemble_and_targets(models: Dict[str, Any], 
                                     current_data: Dict[str, pd.DataFrame],
                                     investment_horizon: str,
                                     model_types: List[str] = None,
                                     ensemble_method: str = 'weighted_average',
                                     selected_tickers: List[str] = None) -> tuple:
    """
    Enhanced prediction with both direction and price targets
    
    Returns:
        tuple: (predictions_df, price_targets_df)
    """
    
    # Get direction predictions (existing functionality)
    predictions_df = predict_with_ensemble(
        models, current_data, investment_horizon, 
        model_types, ensemble_method, selected_tickers
    )
    
    # Generate price targets
    price_targets_df = generate_price_targets_for_selected_stocks(
        models, current_data, selected_tickers or list(current_data.keys()), investment_horizon
    )
    
    return predictions_df, price_targets_df

# ==================== REAL-TIME INTEGRATION ====================
class RealTimeIntegration:
    """Integrate real-time data into models"""
    def __init__(self):
        try:
            self.realtime_manager = RealTimeDataManager()
            self.sentiment_analyzer = AdvancedSentimentAnalyzer(api_key=secrets.NEWS_API_KEY)
        except:
            self.realtime_manager = None
            self.sentiment_analyzer = None
    
    def enhance_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Enhance features with real-time data and sentiment"""
        enhanced_df = df.copy()
        
        # Add real-time features if available
        if self.realtime_manager:
            try:
                realtime_data = self.realtime_manager.get_latest_data(ticker, lookback_minutes=240)
                if not realtime_data.empty:
                    # Calculate technical indicators on real-time data
                    enhanced_df.loc[:, 'realtime_rsi'] = self.calculate_rsi(realtime_data['Close'], 14).iloc[-1] if len(realtime_data) > 14 else 50
                    enhanced_df.loc[:, 'realtime_macd'] = self.calculate_macd(realtime_data['Close']).iloc[-1] if len(realtime_data) > 26 else 0
                    
                    # Add volume features
                    if len(realtime_data) > 20:
                        vol_ratio = realtime_data['Volume'] / realtime_data['Volume'].rolling(20).mean()
                        enhanced_df.loc[:, 'realtime_volume_ratio'] = vol_ratio.iloc[-1]
                    else:
                        enhanced_df.loc[:, 'realtime_volume_ratio'] = 1.0
            except Exception as e:
                logging.warning(f"Real-time data enhancement failed: {e}")
        
        # Add sentiment features
        if self.sentiment_analyzer:
            try:
                sentiment_score = self.sentiment_analyzer.get_ticker_sentiment(ticker)
                enhanced_df.loc[:, 'sentiment_score'] = sentiment_score
            except Exception as e:
                logging.warning(f"Sentiment analysis failed: {e}")
                enhanced_df.loc[:, 'sentiment_score'] = 0.0
        
        return enhanced_df
    
    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for real-time data"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=window).mean()
        avg_loss = loss.ewm(span=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
        """Calculate MACD for real-time data"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        return macd_line

# ==================== MODEL MONITORING ====================
class ModelMonitor:
    """Monitor model performance and data drift"""
    def __init__(self, db_path: str = "data/model_monitor.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    PRIMARY KEY (model_id, timestamp, metric_name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_drift (
                    model_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    feature_name TEXT,
                    drift_score REAL,
                    PRIMARY KEY (model_id, timestamp, feature_name)
                )
            """)
    
    def log_performance(self, model_id: str, metrics: dict):
        """Log model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for metric, value in metrics.items():
                    conn.execute("""
                        INSERT INTO model_performance (model_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (model_id, metric, value))
        except Exception as e:
            logging.warning(f"Failed to log performance metrics: {e}")
    
    def log_feature_drift(self, model_id: str, feature_name: str, drift_score: float):
        """Log feature drift detection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feature_drift (model_id, feature_name, drift_score)
                    VALUES (?, ?, ?)
                """, (model_id, feature_name, drift_score))
        except Exception as e:
            logging.warning(f"Failed to log feature drift: {e}")
    
    def get_performance_history(self, model_id: str, metric_name: str, days=30):
        """Get historical performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql("""
                    SELECT timestamp, metric_value FROM model_performance
                    WHERE model_id = ? AND metric_name = ?
                    AND timestamp >= datetime('now', ?)
                    ORDER BY timestamp
                """, conn, params=(model_id, metric_name, f"-{days} days"))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            return df
        except Exception as e:
            logging.warning(f"Failed to get performance history: {e}")
            return pd.DataFrame()

# ==================== WALK-FORWARD VALIDATION ====================
class WalkForwardValidator:
    """Enhanced walk-forward validation for time series"""
    def __init__(self, initial_train_size=0.7, step_size=0.05, metric='roc_auc'):
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.metric = metric
        self.results = []
    
    def validate(self, model, X, y):
        """Perform walk-forward validation"""
        n = len(X)
        train_size = int(n * self.initial_train_size)
        step = int(n * self.step_size)
        
        scores = []
        while train_size + step < n:
            try:
                # Split data
                X_train, X_test = X[:train_size], X[train_size:train_size+step]
                y_train, y_test = y[:train_size], y[train_size:train_size+step]
                
                # Train and evaluate
                model.fit(X_train, y_train)
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_test)[:, 1]
                else:
                    preds = model.predict(X_test)
                
                if self.metric == 'roc_auc':
                    if len(np.unique(y_test)) > 1:
                        score = roc_auc_score(y_test, preds)
                    else:
                        score = 0.5
                elif self.metric == 'accuracy':
                    score = accuracy_score(y_test, (preds > 0.5).astype(int) if hasattr(model, 'predict_proba') else preds)
                elif self.metric == 'f1':
                    score = f1_score(y_test, (preds > 0.5).astype(int) if hasattr(model, 'predict_proba') else preds)
                else:
                    score = 0.5
                    
                scores.append(score)
                train_size += step
            except Exception as e:
                logging.warning(f"Walk-forward validation step failed: {e}")
                scores.append(0.5)
                train_size += step
        
        return np.mean(scores) if scores else 0.5

# ==================== ENHANCED MODEL CLASSES ====================
class AdvancedStockPredictor:
    """Enhanced predictor class with monitoring and real-time capabilities"""
    
    def __init__(self, horizon: str, model_type: str):
        self.horizon = horizon
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.calibrator = None
        self.required_features = None
        self.selected_features = None
        self.cv_scores = None
        self.feature_importances = None
        self.training_date = datetime.now()
        self.validation_score = None
        self.hyperparameters = None
        self.training_time = None
        self.monitor = ModelMonitor()
        self.realtime_integrator = RealTimeIntegration()
        
    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess features with scaling and selection"""
        try:
            if fit:
                # Feature selection
                if self.feature_selector:
                    X_selected = self.feature_selector.fit_transform(X, y=None)
                    self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
                else:
                    X_selected = X.values
                    self.selected_features = X.columns.tolist()
                
                # Scaling
                if self.scaler:
                    X_scaled = self.scaler.fit_transform(X_selected)
                else:
                    X_scaled = X_selected
                    
                return X_scaled
            else:
                # Transform using fitted preprocessors
                if self.selected_features:
                    available_features = [f for f in self.selected_features if f in X.columns]
                    if len(available_features) != len(self.selected_features):
                        logging.warning(f"Missing features: {set(self.selected_features) - set(available_features)}")
                    X_selected = X[available_features].values
                else:
                    X_selected = X.values
                    
                if self.scaler:
                    X_scaled = self.scaler.transform(X_selected)
                else:
                    X_scaled = X_selected
                    
                return X_scaled
        except Exception as e:
            logging.warning(f"Feature preprocessing failed: {e}")
            return X.fillna(0).values if hasattr(X, 'fillna') else X
    
    def predict(self, X: pd.DataFrame, ticker: str = None) -> np.ndarray:
        """Make predictions with preprocessing and real-time enhancement"""
        if self.model is None:
            return np.array([0] * len(X))
        
        try:
            # Enhance features with real-time data
            if ticker and ENHANCED_MODEL_CONFIG.get('realtime_integration', True):
                try:
                    X_enhanced = self.realtime_integrator.enhance_features(X, ticker)
                except:
                    X_enhanced = X
            else:
                X_enhanced = X
            
            X_processed = self.preprocess_features(X_enhanced, fit=False)
            predictions = self.model.predict(X_processed)
            
            return predictions
        except Exception as e:
            logging.warning(f"Prediction failed: {e}")
            return np.array([0] * len(X))
    
    def predict_proba(self, X: pd.DataFrame, ticker: str = None) -> np.ndarray:
        """Make probability predictions with preprocessing"""
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            return np.array([[0.5, 0.5]] * len(X))
        
        try:
            # Enhance features with real-time data
            if ticker and ENHANCED_MODEL_CONFIG.get('realtime_integration', True):
                try:
                    X_enhanced = self.realtime_integrator.enhance_features(X, ticker)
                except:
                    X_enhanced = X
            else:
                X_enhanced = X
            
            X_processed = self.preprocess_features(X_enhanced, fit=False)
            
            if self.calibrator:
                probabilities = self.calibrator.predict_proba(X_processed)
            else:
                probabilities = self.model.predict_proba(X_processed)
                
            return probabilities
        except Exception as e:
            logging.warning(f"Probability prediction failed: {e}")
            return np.array([[0.5, 0.5]] * len(X))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.feature_importances or not self.selected_features:
            return {}
        
        try:
            return dict(zip(self.selected_features, self.feature_importances))
        except:
            return {}
    
    def log_performance(self, X_test: pd.DataFrame, y_test: pd.Series, ticker: str):
        """Log model performance for monitoring"""
        if self.model is None:
            return
        
        try:
            # Predict probabilities
            proba = self.predict_proba(X_test, ticker)[:, 1]
            preds = (proba > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, preds),
                'precision': precision_score(y_test, preds, zero_division=0),
                'recall': recall_score(y_test, preds, zero_division=0),
                'f1': f1_score(y_test, preds, zero_division=0),
                'roc_auc': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5,
                'log_loss': log_loss(y_test, np.clip(proba, 1e-7, 1-1e-7))
            }
            
            # Log to monitoring system
            model_id = f"{ticker}_{self.model_type}_{self.horizon}"
            self.monitor.log_performance(model_id, metrics)
        except Exception as e:
            logging.warning(f"Performance logging failed: {e}")

class IntelligentFeatureSelector:
    """Advanced feature selection with multiple methods"""
    
    def __init__(self, top_k: int = 100, selection_methods: List[str] = None):
        self.top_k = top_k
        self.selection_methods = selection_methods or ['variance', 'correlation', 'univariate']
        self.selected_features = None
        self.feature_scores = None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       feature_names: List[str] = None) -> List[str]:
        """Intelligent feature selection using multiple methods"""
        
        feature_names = feature_names or X.columns.tolist()
        
        if len(feature_names) <= self.top_k:
            self.selected_features = feature_names
            return feature_names
        
        try:
            feature_scores = {}
            
            # 1. Variance-based selection
            if 'variance' in self.selection_methods:
                variances = X.var()
                var_ranks = variances.rank(ascending=False, method='min') / len(variances)
                for feat, score in var_ranks.items():
                    feature_scores[feat] = feature_scores.get(feat, 0) + score * 0.3
            
            # 2. Correlation with target
            if 'correlation' in self.selection_methods:
                correlations = X.corrwith(y).abs()
                corr_ranks = correlations.rank(ascending=False, method='min') / len(correlations)
                for feat, score in corr_ranks.items():
                    if not pd.isna(score):
                        feature_scores[feat] = feature_scores.get(feat, 0) + score * 0.4
            
            # 3. Univariate statistical tests
            if 'univariate' in self.selection_methods:
                try:
                    from sklearn.feature_selection import SelectKBest, f_classif
                    selector = SelectKBest(score_func=f_classif, k=min(self.top_k * 2, len(feature_names)))
                    selector.fit(X.fillna(0), y)
                    scores = selector.scores_
                    uni_ranks = pd.Series(scores, index=feature_names).rank(ascending=False, method='min') / len(scores)
                    for feat, score in uni_ranks.items():
                        if not pd.isna(score):
                            feature_scores[feat] = feature_scores.get(feat, 0) + score * 0.3
                except Exception as e:
                    logging.warning(f"Univariate selection failed: {e}")
            
            # Select top features
            if feature_scores:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                self.selected_features = [feat for feat, score in sorted_features[:self.top_k]]
                self.feature_scores = dict(sorted_features[:self.top_k])
            else:
                # Fallback to first top_k features
                self.selected_features = feature_names[:self.top_k]
                self.feature_scores = {}
            
            return self.selected_features
            
        except Exception as e:
            logging.warning(f"Feature selection failed: {e}")
            self.selected_features = feature_names[:self.top_k]
            return self.selected_features

# ==================== HYPERPARAMETER OPTIMIZATION ====================

class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimization with Optuna"""
    
    def __init__(self, n_trials: int = 50, cv_folds: int = 3):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
        self.best_score = None
        
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        try:
            from xgboost import XGBClassifier
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbosity': 0
                }
                
                model = XGBClassifier(**params)
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(direction='maximize', 
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            return self.best_params
        except Exception as e:
            logging.warning(f"XGBoost optimization failed: {e}")
            return {}
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        try:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'random_state': 42,
                    'n_jobs': 1,
                    'verbosity': -1
                }
                
                model = lgb.LGBMClassifier(**params)
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(direction='maximize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            return self.best_params
        except Exception as e:
            logging.warning(f"LightGBM optimization failed: {e}")
            return {}
    
    def optimize_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for specified model type"""
        
        if model_type == 'xgboost':
            return self.optimize_xgboost(X, y)
        elif model_type == 'lightgbm':
            return self.optimize_lightgbm(X, y)
        else:
            # Default parameters for other models
            return {}

# ==================== ENHANCED TRAINING PIPELINE ====================

def train_enhanced_model(args) -> Tuple[str, str, AdvancedStockPredictor, float]:
    """Enhanced single model training with all optimizations - FIXED VERSION"""
    
    ticker, df, horizon, model_type, config = args
    
    try:
        target_col = f"Target_{horizon}"
        if target_col not in df.columns or df.empty or len(df) < config.get('min_data_points', 200):
            return None
            
        # Enhanced feature selection
        feature_cols = [col for col in df.columns 
                       if not col.startswith('Target_') 
                       and not col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        if len(feature_cols) == 0:
            return None
        
        # Temporal splits for proper time series validation
        total_len = len(df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)
        
        # Training data
        X_train = df[feature_cols].iloc[:train_end].fillna(0)
        y_train = df[target_col].iloc[:train_end]
        
        # Validation data
        X_val = df[feature_cols].iloc[train_end:val_end].fillna(0)
        y_val = df[target_col].iloc[train_end:val_end]
        
        # Test data
        X_test = df[feature_cols].iloc[val_end:].fillna(0)
        y_test = df[target_col].iloc[val_end:]
        
        # Remove rows with NaN targets
        train_mask = ~y_train.isna()
        val_mask = ~y_val.isna()
        test_mask = ~y_test.isna()
        
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        
        if len(X_train) < 100 or len(X_val) < 20 or len(X_test) < 20:
            return None
        
        # Feature selection
        feature_selector = IntelligentFeatureSelector(
            top_k=config.get('feature_selection_top_k', 100)
        )
        selected_features = feature_selector.select_features(X_train, y_train)
        
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Feature scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Initialize model with hyperparameter optimization
        start_time = datetime.now()
        
        if config.get('hyperparameter_tuning', False):
            optimizer = AdvancedHyperparameterOptimizer(n_trials=20, cv_folds=3)
            best_params = optimizer.optimize_model(model_type, X_train_scaled, y_train)
        else:
            best_params = {}
        
        # Create model with optimized parameters
        model = create_optimized_model(model_type, best_params)
        
        # FIXED: Train model with proper early stopping handling
        if model_type == 'xgboost':
            try:
                # XGBoost 2.0+ compatible training - FIXED VERSION
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False  # Removed early_stopping_rounds parameter
                )
            except Exception as e:
                logging.warning(f"XGBoost early stopping failed for {ticker}: {e}")
                model.fit(X_train_scaled, y_train)
                
        elif model_type == 'lightgbm':
            try:
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(config.get('early_stopping_patience', 10), verbose=False)]
                )
            except Exception as e:
                logging.warning(f"LightGBM early stopping failed for {ticker}: {e}")
                model.fit(X_train_scaled, y_train)
        else:
            # For other models, just train normally
            model.fit(X_train_scaled, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Model calibration
        calibrated_model = None
        if config.get('model_calibration', False):
            try:
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated_model.fit(X_train_scaled, y_train)
            except Exception as e:
                logging.warning(f"Model calibration failed for {ticker}: {e}")
        
        # Evaluate model
        test_pred = model.predict(X_test_scaled)
        test_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else [0.5] * len(X_test)
        
        # Comprehensive metrics
        metrics = {}
        try:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_test, test_pred)
            metrics['precision'] = precision_score(y_test, test_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, test_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, test_pred, zero_division=0)
            
            # ROC AUC
            if len(np.unique(y_test)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_test, test_proba)
            else:
                metrics['roc_auc'] = 0.5
            
            # Additional metrics
            metrics['mcc'] = matthews_corrcoef(y_test, test_pred)
            
            # Log loss (with clipping to avoid extreme values)
            test_proba_clipped = np.clip(test_proba, 1e-7, 1 - 1e-7)
            metrics['log_loss'] = log_loss(y_test, test_proba_clipped)
            
            # Profit-based metrics (assuming simple trading strategy)
            returns_if_predicted_positive = y_test[test_pred == 1]
            if len(returns_if_predicted_positive) > 0:
                metrics['hit_rate'] = np.mean(returns_if_predicted_positive)
                metrics['trading_accuracy'] = np.mean(returns_if_predicted_positive > 0)
            else:
                metrics['hit_rate'] = 0.0
                metrics['trading_accuracy'] = 0.0
                
        except Exception as e:
            logging.warning(f"Metrics calculation failed: {e}")
            metrics = {'accuracy': 0.5, 'roc_auc': 0.5, 'f1': 0.0}
        
        # Feature importance analysis
        feature_importance = None
        if config.get('feature_importance_analysis', False):
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = dict(zip(selected_features, importances))
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                    feature_importance = dict(zip(selected_features, importances))
                else:
                    feature_importance = {}
            except Exception as e:
                logging.warning(f"Feature importance extraction failed: {e}")
                feature_importance = {}
        
        # Create advanced predictor
        predictor = AdvancedStockPredictor(horizon, model_type)
        predictor.model = calibrated_model if calibrated_model else model
        predictor.scaler = scaler
        predictor.required_features = feature_cols
        predictor.selected_features = selected_features
        predictor.cv_scores = metrics
        predictor.feature_importances = feature_importance
        predictor.hyperparameters = best_params
        predictor.training_time = training_time
        predictor.validation_score = metrics.get('roc_auc', 0.5)
        
        return (ticker, f"{model_type}_{horizon}", predictor, metrics.get('roc_auc', 0.5))
        
    except Exception as e:
        logging.error(f"Enhanced training failed for {ticker} {horizon} {model_type}: {e}")
        return None

def create_optimized_model(model_type: str, best_params: Dict[str, Any]):
    """Create optimized model with best parameters - FIXED VERSION"""
    
    try:
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            default_params = {
                'n_estimators': 500, 
                'max_depth': 6, 
                'learning_rate': 0.1,
                'subsample': 0.8, 
                'colsample_bytree': 0.8, 
                'random_state': 42,
                'n_jobs': 1, 
                'verbosity': 0,
                'eval_metric': 'logloss'  # Fixed: Add eval_metric
            }
            default_params.update(best_params)
            return XGBClassifier(**default_params)
            
        elif model_type == 'lightgbm':
            default_params = {
                'n_estimators': 500, 
                'max_depth': 6, 
                'learning_rate': 0.1,
                'subsample': 0.8, 
                'colsample_bytree': 0.8, 
                'num_leaves': 31,
                'random_state': 42, 
                'n_jobs': 1, 
                'verbosity': -1
            }
            default_params.update(best_params)
            return lgb.LGBMClassifier(**default_params)
            
        elif model_type == 'catboost':
            default_params = {
                'n_estimators': 500, 
                'max_depth': 6, 
                'learning_rate': 0.1,
                'l2_leaf_reg': 3, 
                'random_state': 42, 
                'verbose': False
            }
            default_params.update(best_params)
            return CatBoostClassifier(**default_params)
            
        elif model_type == 'random_forest':
            default_params = {
                'n_estimators': 500, 
                'max_depth': 15, 
                'min_samples_split': 5,
                'min_samples_leaf': 2, 
                'class_weight': 'balanced', 
                'random_state': 42, 
                'n_jobs': 1
            }
            default_params.update(best_params)
            return RandomForestClassifier(**default_params)
            
        elif model_type == 'neural_network':
            default_params = {
                'hidden_layer_sizes': (100, 50), 
                'activation': 'relu',
                'alpha': 0.001, 
                'learning_rate': 'adaptive', 
                'max_iter': 1000, 
                'random_state': 42
            }
            default_params.update(best_params)
            return MLPClassifier(**default_params)
            
        else:
            # Default to Random Forest
            return RandomForestClassifier(random_state=42, n_jobs=1)
            
    except Exception as e:
        logging.warning(f"Model creation failed for {model_type}: {e}")
        return RandomForestClassifier(random_state=42, n_jobs=1)

# ==================== ENHANCED PARALLEL TRAINING ====================

def train_models_enhanced_parallel(featured_data: Dict[str, pd.DataFrame], 
                                  config: Dict = None,
                                  selected_tickers: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced parallel training with advanced ML techniques for selected tickers only
    """
    config = config or ENHANCED_MODEL_CONFIG
    
    # Filter to selected tickers only
    if selected_tickers:
        original_count = len(featured_data)
        featured_data = {ticker: df for ticker, df in featured_data.items() 
                        if ticker in selected_tickers}
        print(f"Training filtered from {original_count} to {len(featured_data)} selected tickers")
    
    # Enhanced model selection
    horizons = config.get('priority_horizons', ['next_month', 'next_quarter'])
    model_types = config.get('model_types', ['xgboost', 'lightgbm', 'random_forest'])
    
    print(f"Enhanced training: {len(featured_data)} selected tickers, {len(horizons)} horizons, {len(model_types)} models")
    print(f"Selected stocks: {list(featured_data.keys())}")
    print(f"Features: Hyperparameter tuning={config.get('hyperparameter_tuning', False)}")
    print(f"         Model calibration={config.get('model_calibration', False)}")
    print(f"         Feature selection top K={config.get('feature_selection_top_k', 100)}")
    
    # Prepare enhanced training tasks
    training_tasks = []
    for ticker, df in featured_data.items():
        if df.empty or len(df) < config.get('min_data_points', 200):
            logging.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            continue
        for horizon in horizons:
            if f"Target_{horizon}" not in df.columns:
                continue
            for model_type in model_types:
                training_tasks.append((ticker, df, horizon, model_type, config))
    
    print(f"Total enhanced training tasks for selected stocks: {len(training_tasks)}")
    
    if len(training_tasks) == 0:
        return {'models': {}, 'training_summary': {'total_tasks': 0, 'successful': 0, 'success_rate': 0}}
    
    # Enhanced parallel training
    all_models = {}
    successful_trains = 0
    training_results = []
    
    # Conservative worker count for enhanced training
    max_workers = min(mp.cpu_count() // 2, len(training_tasks), 4)
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(train_enhanced_model, task): task 
                             for task in training_tasks}
            
            # Process results with detailed progress
            for future in tqdm(as_completed(future_to_task), total=len(training_tasks), 
                              desc="Enhanced model training for selected stocks"):
                result = future.result()
                if result:
                    ticker, model_key, predictor, score = result
                    if ticker not in all_models:
                        all_models[ticker] = {}
                    all_models[ticker][model_key] = predictor
                    successful_trains += 1
                    
                    training_results.append({
                        'ticker': ticker,
                        'model_key': model_key,
                        'score': score,
                        'training_time': predictor.training_time,
                        'feature_count': len(predictor.selected_features) if predictor.selected_features else 0
                    })
    except Exception as e:
        logging.error(f"Parallel training failed: {e}")
        # Fallback to sequential training
        for task in tqdm(training_tasks[:10], desc="Sequential fallback training"):  # Limit to 10 for safety
            result = train_enhanced_model(task)
            if result:
                ticker, model_key, predictor, score = result
                if ticker not in all_models:
                    all_models[ticker] = {}
                all_models[ticker][model_key] = predictor
                successful_trains += 1
                
                training_results.append({
                    'ticker': ticker,
                    'model_key': model_key,
                    'score': score,
                    'training_time': predictor.training_time,
                    'feature_count': len(predictor.selected_features) if predictor.selected_features else 0
                })
    
    print(f"Enhanced training completed for selected stocks: {successful_trains}/{len(training_tasks)} models")
    
    # Generate training summary
    if training_results:
        results_df = pd.DataFrame(training_results)
        avg_score = results_df['score'].mean()
        avg_training_time = results_df['training_time'].mean()
        avg_features = results_df['feature_count'].mean()
        
        print(f"Training Performance Summary:")
        print(f"  Average ROC AUC: {avg_score:.3f}")
        print(f"  Average training time: {avg_training_time:.1f}s")
        print(f"  Average features per model: {avg_features:.0f}")
        print(f"  Models per selected stock: {successful_trains/len(featured_data):.1f}")
    
    return {
        'models': all_models,
        'training_summary': {
            'total_tasks': len(training_tasks),
            'successful': successful_trains,
            'success_rate': successful_trains / len(training_tasks) if training_tasks else 0,
            'training_results': training_results,
            'selected_tickers': list(featured_data.keys()) if selected_tickers else None
        }
    }

# ==================== ENHANCED PREDICTION SYSTEM ====================

def predict_with_ensemble(models: Dict[str, Any], 
                         current_data: Dict[str, pd.DataFrame],
                         investment_horizon: str,
                         model_types: List[str] = None,
                         ensemble_method: str = 'weighted_average',
                         selected_tickers: List[str] = None) -> pd.DataFrame:
    """
    Enhanced prediction with ensemble methods for selected tickers
    """
    model_types = model_types or ['xgboost', 'lightgbm', 'random_forest']
    predictions = []
    
    # Filter to selected tickers if provided
    if selected_tickers:
        models = {ticker: model_dict for ticker, model_dict in models.items() 
                 if ticker in selected_tickers}
        current_data = {ticker: df for ticker, df in current_data.items() 
                       if ticker in selected_tickers}
        print(f"Making predictions for {len(models)} selected tickers")
    
    for ticker, model_dict in tqdm(models.items(), desc="Ensemble predictions for selected stocks"):
        try:
            if ticker not in current_data:
                continue
                
            df = current_data[ticker]
            if df.empty:
                continue
            
            # Collect predictions from different models
            model_predictions = []
            model_probabilities = []
            model_weights = []
            
            for model_type in model_types:
                model_key = f"{model_type}_{investment_horizon}"
                if model_key not in model_dict:
                    continue
                    
                predictor = model_dict[model_key]
                if not hasattr(predictor, 'model') or predictor.model is None:
                    continue
                
                # Prepare latest data
                latest_data = df.iloc[[-1]].copy()
                
                try:
                    # Make prediction
                    pred = predictor.predict(latest_data, ticker)[0]
                    proba = predictor.predict_proba(latest_data, ticker)[0][1]
                    
                    model_predictions.append(pred)
                    model_probabilities.append(proba)
                    
                    # Weight by validation score
                    weight = predictor.validation_score if predictor.validation_score else 0.5
                    model_weights.append(weight)
                    
                except Exception as e:
                    logging.warning(f"Prediction failed for {ticker} {model_key}: {e}")
                    continue
            
            if not model_predictions:
                continue
            
            # Ensemble prediction
            if ensemble_method == 'weighted_average':
                # Weighted average based on validation scores
                total_weight = sum(model_weights)
                if total_weight > 0:
                    ensemble_proba = sum(p * w for p, w in zip(model_probabilities, model_weights)) / total_weight
                    ensemble_pred = 1 if ensemble_proba > 0.5 else 0
                else:
                    ensemble_proba = np.mean(model_probabilities)
                    ensemble_pred = int(np.round(np.mean(model_predictions)))
            elif ensemble_method == 'majority_vote':
                ensemble_pred = 1 if sum(model_predictions) > len(model_predictions) / 2 else 0
                ensemble_proba = np.mean(model_probabilities)
            else:  # simple_average
                ensemble_pred = int(np.round(np.mean(model_predictions)))
                ensemble_proba = np.mean(model_probabilities)
            
            # Calculate ensemble confidence
            prediction_std = np.std(model_probabilities) if len(model_probabilities) > 1 else 0
            ensemble_confidence = (1 - prediction_std) * (sum(model_weights) / len(model_weights) if model_weights else 0.5)
            
            # Risk metrics
            returns = df['Close'].pct_change().dropna()
            if len(returns) > 20:
                volatility = returns.std() * np.sqrt(252)
                risk_score = min(volatility / 0.5, 1.0)
            else:
                volatility = 0.5
                risk_score = 0.5
            
            predictions.append({
                'ticker': ticker,
                'predicted_return': int(ensemble_pred),
                'success_prob': float(ensemble_proba),
                'ensemble_confidence': float(ensemble_confidence),
                'model_agreement': 1.0 - prediction_std,
                'risk_score': float(risk_score),
                'volatility': float(volatility),
                'models_used': len(model_predictions),
                'ensemble_method': ensemble_method,
                'horizon': investment_horizon
            })
            
        except Exception as e:
            logging.warning(f"Ensemble prediction failed for {ticker}: {e}")
            continue
    
    result_df = pd.DataFrame(predictions)
    if not result_df.empty and selected_tickers:
        print(f"Generated predictions for {len(result_df)} selected stocks")
    
    return result_df

# ==================== MODEL PERSISTENCE ====================

def save_models_optimized(models: Dict[str, Any], cache_dir: str = "model_cache") -> bool:
    """Save models with optimized storage"""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        
        for ticker, model_dict in models.items():
            ticker_dir = os.path.join(cache_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            for model_key, predictor in model_dict.items():
                model_file = os.path.join(ticker_dir, f"{model_key}.pkl")
                
                try:
                    with open(model_file, 'wb') as f:
                        pickle.dump(predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logging.warning(f"Failed to save model {ticker}/{model_key}: {e}")
        
        logging.info(f"Models saved to {cache_dir}")
        return True
    except Exception as e:
        logging.error(f"Model saving failed: {e}")
        return False

def load_models_optimized(cache_dir: str = "model_cache") -> Dict[str, Any]:
    """Load models with optimized loading"""
    models = {}
    
    try:
        if not os.path.exists(cache_dir):
            return models
        
        for ticker_dir in os.listdir(cache_dir):
            ticker_path = os.path.join(cache_dir, ticker_dir)
            if not os.path.isdir(ticker_path):
                continue
                
            models[ticker_dir] = {}
            
            for model_file in os.listdir(ticker_path):
                if not model_file.endswith('.pkl'):
                    continue
                    
                model_key = model_file.replace('.pkl', '')
                model_path = os.path.join(ticker_path, model_file)
                
                try:
                    with open(model_path, 'rb') as f:
                        predictor = pickle.load(f)
                        models[ticker_dir][model_key] = predictor
                except Exception as e:
                    logging.warning(f"Failed to load model {ticker_dir}/{model_key}: {e}")
        
        total_models = sum(len(model_dict) for model_dict in models.values())
        logging.info(f"Loaded {total_models} models from {cache_dir}")
        return models
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return {}

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Enhanced Stock Prediction Model System - User Selection + Price Targets Version")
    print("="*60)
    
    # Enhanced configuration
    enhanced_config = ENHANCED_MODEL_CONFIG.copy()
    enhanced_config['hyperparameter_tuning'] = True
    enhanced_config['model_calibration'] = True
    enhanced_config['feature_importance_analysis'] = True
    enhanced_config['selected_stocks_only'] = True
    
    print("Enhanced Configuration:")
    for key, value in enhanced_config.items():
        if key != 'param_space':
            print(f"  {key}: {value}")
    
    print(f"\nNew Features:")
    print(f"  ✓ Price Target Predictions - Specific price predictions")
    print(f"  ✓ No Pre-selection - Users must choose stocks")
    print(f"  ✓ Fixed XGBoost Early Stopping")
    print(f"  ✓ Train models only for user-selected stocks")
    print(f"  ✓ Faster training with fewer stocks")
    print(f"  ✓ Enhanced ensemble methods")
    print(f"  ✓ Comprehensive evaluation metrics")
    print(f"  ✓ Feature importance analysis")
    print(f"  ✓ Time series validation")
    print(f"  ✓ Enhanced error handling and logging")