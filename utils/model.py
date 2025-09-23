# utils/model.py - Institutional-Grade Quantitative Models
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import os
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass

# Advanced ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ==================== INSTITUTIONAL CONFIGURATION ====================

INSTITUTIONAL_MODEL_CONFIG = {
    'ensemble_size': 7,
    'enable_stacking': True,
    'cross_validation_folds': 10,
    'test_size': 0.2,
    'validation_size': 0.15,
    'random_state': 42,
    
    # Advanced feature selection
    'feature_selection': True,
    'feature_selection_methods': ['mutual_info', 'f_classif', 'rfe'],
    'max_features': 100,
    'min_feature_importance': 0.001,
    
    # Hyperparameter optimization
    'hyperparameter_tuning': True,
    'optimization_trials': 100,
    'optimization_timeout': 1800,  # 30 minutes per model
    
    # Model types - institutional grade
    'model_types': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees', 'gradient_boosting'],
    
    # Target horizons with sophisticated thresholds
    'target_horizons': ['next_week', 'next_month', 'next_quarter', 'next_6_months'],
    'return_thresholds': {
        'next_week': 0.015,     # 1.5% weekly return threshold
        'next_month': 0.04,     # 4% monthly return threshold  
        'next_quarter': 0.08,   # 8% quarterly return threshold
        'next_6_months': 0.12,  # 12% 6-month return threshold
    },
    
    # Advanced training parameters
    'early_stopping': True,
    'early_stopping_rounds': 50,
    'validation_split': 0.2,
    'model_monitoring': True,
    'cache_models': True,
    'model_cache_dir': 'model_cache_institutional',
    
    # Parallel processing
    'parallel_training': True,
    'max_workers': 8,
    'batch_size': 4,
    
    # Advanced techniques
    'use_pseudo_labeling': True,
    'use_adversarial_validation': True,
    'use_model_calibration': True,
    'ensemble_weights_optimization': True
}

# ==================== ADVANCED FEATURE ENGINEERING FOR MODELS ====================

class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    """Advanced feature transformer for institutional models"""
    
    def __init__(self, create_interactions=True, create_ratios=True, 
                 create_technical_patterns=True, lookback_periods=[5, 20, 60]):
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_technical_patterns = create_technical_patterns
        self.lookback_periods = lookback_periods
        self.feature_names = []
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.original_features = X.columns.tolist()
        else:
            self.original_features = [f'feature_{i}' for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.original_features)
        
        X_transformed = X.copy()
        
        # Create interaction features
        if self.create_interactions:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # Create ratio features
        if self.create_ratios:
            X_transformed = self._create_ratio_features(X_transformed)
        
        # Create technical pattern features
        if self.create_technical_patterns:
            X_transformed = self._create_pattern_features(X_transformed)
        
        # Handle infinite and NaN values
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
        X_transformed = X_transformed.fillna(X_transformed.median())
        
        self.feature_names = X_transformed.columns.tolist()
        
        return X_transformed
    
    def _create_interaction_features(self, X):
        """Create interaction features between key variables"""
        
        # Find price-related features
        price_features = [col for col in X.columns if any(term in col.lower() 
                         for term in ['price', 'sma', 'ema', 'close'])]
        
        # Find volume features
        volume_features = [col for col in X.columns if 'volume' in col.lower()]
        
        # Find volatility features
        vol_features = [col for col in X.columns if 'volatility' in col.lower()]
        
        # Create price-volume interactions
        for price_feat in price_features[:5]:  # Limit to avoid explosion
            for vol_feat in volume_features[:3]:
                if price_feat in X.columns and vol_feat in X.columns:
                    X[f'{price_feat}_x_{vol_feat}'] = X[price_feat] * X[vol_feat]
        
        # Create price-volatility interactions  
        for price_feat in price_features[:5]:
            for vol_feat in vol_features[:3]:
                if price_feat in X.columns and vol_feat in X.columns:
                    X[f'{price_feat}_x_{vol_feat}'] = X[price_feat] * X[vol_feat]
        
        return X
    
    def _create_ratio_features(self, X):
        """Create sophisticated ratio features"""
        
        # Price ratios
        price_cols = [col for col in X.columns if any(term in col.lower() 
                     for term in ['sma_', 'ema_', 'price'])]
        
        for i, col1 in enumerate(price_cols[:8]):
            for col2 in price_cols[i+1:8]:
                if col1 in X.columns and col2 in X.columns:
                    # Avoid division by zero
                    mask = X[col2] != 0
                    X.loc[mask, f'{col1}_to_{col2}_ratio'] = X.loc[mask, col1] / X.loc[mask, col2]
        
        # Volume ratios
        volume_cols = [col for col in X.columns if 'volume' in col.lower()]
        for i, col1 in enumerate(volume_cols[:5]):
            for col2 in volume_cols[i+1:5]:
                if col1 in X.columns and col2 in X.columns:
                    mask = X[col2] != 0
                    X.loc[mask, f'{col1}_to_{col2}_ratio'] = X.loc[mask, col1] / X.loc[mask, col2]
        
        return X
    
    def _create_pattern_features(self, X):
        """Create technical pattern recognition features"""
        
        # RSI patterns
        if 'rsi' in X.columns:
            X['rsi_oversold'] = (X['rsi'] < 30).astype(int)
            X['rsi_overbought'] = (X['rsi'] > 70).astype(int)
            X['rsi_momentum'] = X['rsi'].diff()
        
        # Bollinger Band patterns
        if all(col in X.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
            X['bb_squeeze'] = ((X['bb_upper'] - X['bb_lower']) < X['bb_width'].rolling(20).quantile(0.2)).astype(int)
            X['bb_breakout_up'] = (X['bb_position'] > 1.0).astype(int)
            X['bb_breakout_down'] = (X['bb_position'] < 0.0).astype(int)
        
        # Moving average patterns
        ma_cols = [col for col in X.columns if 'sma_' in col or 'ema_' in col]
        if len(ma_cols) >= 2:
            short_ma = ma_cols[0]  # Assuming sorted by period
            long_ma = ma_cols[-1]
            
            if short_ma in X.columns and long_ma in X.columns:
                X['ma_golden_cross'] = (X[short_ma] > X[long_ma]).astype(int)
                X['ma_death_cross'] = (X[short_ma] < X[long_ma]).astype(int)
                X['ma_convergence'] = abs(X[short_ma] - X[long_ma]) / X[long_ma]
        
        return X

# ==================== INSTITUTIONAL MODEL CLASSES ====================

class InstitutionalModelTrainer:
    """Institutional-grade model trainer with advanced techniques"""
    
    def __init__(self, config: Dict = None):
        self.config = config or INSTITUTIONAL_MODEL_CONFIG
        self.feature_transformer = AdvancedFeatureTransformer()
        self.model_cache = {}
        self.training_history = []
        
    def prepare_institutional_features_targets(self, df: pd.DataFrame, 
                                             horizon: str = 'next_month') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets with institutional-grade preprocessing"""
        
        try:
            if df.empty or len(df) < 200:  # Institutional minimum
                logging.warning(f"Insufficient data for institutional analysis: {len(df)} records")
                return None, None, []
            
            # Enhanced feature selection
            exclude_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                'Date', 'Datetime', 'date', 'datetime'
            ]
            
            # Exclude existing targets
            target_cols = [col for col in df.columns if any(term in col.lower() 
                          for term in ['target_', 'return_', 'label_', 'y_'])]
            exclude_cols.extend(target_cols)
            
            # Select sophisticated features
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) < 10:  # Institutional minimum features
                logging.warning(f"Insufficient features for institutional modeling: {len(feature_cols)}")
                return None, None, []
            
            logging.info(f"Using {len(feature_cols)} features for institutional modeling")
            
            # Create sophisticated target variable
            target = self._create_institutional_target(df, horizon)
            
            if target is None or target.isnull().all():
                logging.error(f"Target creation failed for horizon: {horizon}")
                return None, None, []
            
            # Prepare feature matrix with advanced transformations
            X = df[feature_cols].copy()
            
            # Advanced preprocessing
            X = self._advanced_preprocessing(X)
            
            # Apply institutional feature transformations
            X_transformed = self.feature_transformer.fit_transform(X)
            
            # Align features and targets
            valid_mask = ~target.isnull()
            X_final = X_transformed[valid_mask]
            y_final = target[valid_mask]
            
            if len(X_final) < 100:  # Institutional minimum
                logging.warning(f"Insufficient valid samples after preprocessing: {len(X_final)}")
                return None, None, []
            
            # Final feature names
            final_feature_names = self.feature_transformer.feature_names
            
            logging.info(f"Institutional preprocessing completed: {X_final.shape[1]} features, {len(y_final)} samples")
            
            return X_final.values, y_final.values, final_feature_names
            
        except Exception as e:
            logging.error(f"Institutional feature preparation failed: {e}")
            return None, None, []
    
    def _create_institutional_target(self, df: pd.DataFrame, horizon: str) -> Optional[pd.Series]:
        """Create sophisticated target variable for institutional models"""
        
        try:
            if 'Close' not in df.columns:
                return None
            
            close_prices = df['Close']
            
            # Institutional horizon mapping with sophisticated periods
            periods_map = {
                'next_week': 5,
                'next_month': 21,
                'next_quarter': 63,
                'next_6_months': 126,
                'next_year': 252
            }
            
            periods = periods_map.get(horizon, 21)
            threshold = self.config.get('return_thresholds', {}).get(horizon, 0.04)
            
            # Calculate future returns with institutional adjustments
            future_returns = close_prices.shift(-periods) / close_prices - 1
            
            # Create sophisticated multi-class target
            # Class 0: Strong negative (< -threshold)
            # Class 1: Weak/neutral (-threshold to +threshold) 
            # Class 2: Strong positive (> threshold)
            
            target = np.where(
                future_returns > threshold, 2,
                np.where(future_returns < -threshold, 0, 1)
            )
            
            # Convert to binary for institutional focus on outperformance
            # 1 if strong positive, 0 otherwise
            binary_target = (target == 2).astype(int)
            
            # Add risk-adjusted targeting
            if 'volatility_20d' in df.columns:
                volatility = df['volatility_20d']
                # Adjust threshold based on volatility (higher vol stocks need higher returns)
                adjusted_threshold = threshold * (1 + volatility)
                risk_adjusted_target = (future_returns > adjusted_threshold).astype(int)
                
                # Combine both approaches
                final_target = ((binary_target == 1) | (risk_adjusted_target == 1)).astype(int)
            else:
                final_target = binary_target
            
            return pd.Series(final_target, index=df.index)
            
        except Exception as e:
            logging.error(f"Institutional target creation failed for {horizon}: {e}")
            return None
    
    def _advanced_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Advanced preprocessing for institutional models"""
        
        # Handle missing values with sophisticated imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Forward fill then backward fill for time series
        X[numeric_cols] = X[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Use median for remaining missing values
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Remove constant features
        constant_features = []
        for col in numeric_cols:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X = X.drop(columns=constant_features)
            logging.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        if len(X.columns) > 50:  # Only for large feature sets
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > 0.95)]
            
            if high_corr_features:
                X = X.drop(columns=high_corr_features)
                logging.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        return X
    
    def create_institutional_model(self, model_type: str, X_train: np.ndarray, 
                                 y_train: np.ndarray, optimization_trial=None) -> Optional[Any]:
        """Create institutional-grade model with optimized hyperparameters"""
        
        try:
            if model_type == 'xgboost' and XGBOOST_AVAILABLE:
                if optimization_trial and OPTUNA_AVAILABLE:
                    # Optimized hyperparameters
                    params = {
                        'n_estimators': optimization_trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': optimization_trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': optimization_trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': optimization_trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': optimization_trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': optimization_trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': optimization_trial.suggest_float('reg_lambda', 0, 10),
                        'random_state': self.config['random_state']
                    }
                else:
                    # Default institutional parameters
                    params = {
                        'n_estimators': 500,
                        'max_depth': 8,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 1,
                        'reg_lambda': 1,
                        'random_state': self.config['random_state']
                    }
                
                return xgb.XGBClassifier(**params)
            
            elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                if optimization_trial and OPTUNA_AVAILABLE:
                    params = {
                        'n_estimators': optimization_trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': optimization_trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': optimization_trial.suggest_float('learning_rate', 0.01, 0.3),
                        'feature_fraction': optimization_trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': optimization_trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'reg_alpha': optimization_trial.suggest_float('reg_alpha', 0, 10),
                        'reg_lambda': optimization_trial.suggest_float('reg_lambda', 0, 10),
                        'random_state': self.config['random_state'],
                        'verbose': -1
                    }
                else:
                    params = {
                        'n_estimators': 500,
                        'max_depth': 8,
                        'learning_rate': 0.1,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'reg_alpha': 1,
                        'reg_lambda': 1,
                        'random_state': self.config['random_state'],
                        'verbose': -1
                    }
                
                return lgb.LGBMClassifier(**params)
            
            elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                if optimization_trial and OPTUNA_AVAILABLE:
                    params = {
                        'iterations': optimization_trial.suggest_int('iterations', 100, 1000),
                        'depth': optimization_trial.suggest_int('depth', 3, 10),
                        'learning_rate': optimization_trial.suggest_float('learning_rate', 0.01, 0.3),
                        'l2_leaf_reg': optimization_trial.suggest_float('l2_leaf_reg', 1, 10),
                        'random_state': self.config['random_state'],
                        'verbose': False
                    }
                else:
                    params = {
                        'iterations': 500,
                        'depth': 8,
                        'learning_rate': 0.1,
                        'l2_leaf_reg': 3,
                        'random_state': self.config['random_state'],
                        'verbose': False
                    }
                
                return CatBoostClassifier(**params)
            
            elif model_type == 'random_forest':
                if optimization_trial and OPTUNA_AVAILABLE:
                    params = {
                        'n_estimators': optimization_trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': optimization_trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': optimization_trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': optimization_trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': optimization_trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
                        'random_state': self.config['random_state'],
                        'n_jobs': -1
                    }
                else:
                    params = {
                        'n_estimators': 300,
                        'max_depth': 15,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt',
                        'random_state': self.config['random_state'],
                        'n_jobs': -1
                    }
                
                return RandomForestClassifier(**params)
            
            elif model_type == 'extra_trees':
                params = {
                    'n_estimators': 300,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': self.config['random_state'],
                    'n_jobs': -1
                }
                
                return ExtraTreesClassifier(**params)
            
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'subsample': 0.8,
                    'max_features': 'sqrt',
                    'random_state': self.config['random_state']
                }
                
                return GradientBoostingClassifier(**params)
            
            else:
                # Fallback to Random Forest
                return RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
                
        except Exception as e:
            logging.error(f"Model creation failed for {model_type}: {e}")
            return None
    
    def train_institutional_ensemble(self, X: np.ndarray, y: np.ndarray, 
                                   ticker: str, horizon: str) -> Dict[str, Any]:
        """Train institutional-grade ensemble with advanced techniques"""
        
        try:
            logging.info(f"Training institutional ensemble for {ticker} - {horizon}")
            
            # Advanced train-validation-test split for time series
            n_samples = len(X)
            train_size = int(n_samples * 0.6)
            val_size = int(n_samples * 0.2)
            
            X_train, X_temp = X[:train_size], X[train_size:]
            y_train, y_temp = y[:train_size], y[train_size:]
            
            X_val, X_test = X_temp[:val_size], X_temp[val_size:]
            y_val, y_test = y_temp[:val_size], y_temp[val_size:]
            
            if len(X_train) < 100 or len(X_test) < 30:
                logging.warning(f"Insufficient data splits for {ticker}")
                return {}
            
            # Check class balance
            class_balance = np.bincount(y_train.astype(int))
            if len(class_balance) < 2 or min(class_balance) < 10:
                logging.warning(f"Poor class balance for {ticker}: {class_balance}")
                return {}
            
            # Feature scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Feature selection
            feature_selector = None
            if self.config.get('feature_selection', True) and X_train.shape[1] > 50:
                feature_selector = SelectKBest(
                    score_func=f_classif,
                    k=min(self.config.get('max_features', 50), X_train.shape[1])
                )
                
                X_train_scaled = feature_selector.fit_transform(X_train_scaled, y_train)
                X_val_scaled = feature_selector.transform(X_val_scaled)
                X_test_scaled = feature_selector.transform(X_test_scaled)
            
            # Train ensemble models
            ensemble_models = {}
            model_types = self.config.get('model_types', ['xgboost', 'random_forest'])
            
            for model_type in model_types:
                try:
                    logging.info(f"Training {model_type} for {ticker}")
                    
                    # Hyperparameter optimization
                    if self.config.get('hyperparameter_tuning', False) and OPTUNA_AVAILABLE:
                        best_model = self._optimize_hyperparameters(
                            model_type, X_train_scaled, y_train, X_val_scaled, y_val
                        )
                    else:
                        best_model = self.create_institutional_model(
                            model_type, X_train_scaled, y_train
                        )
                    
                    if best_model is None:
                        continue
                    
                    # Train model
                    best_model.fit(X_train_scaled, y_train)
                    
                    # Validate model
                    y_pred_val = best_model.predict(X_val_scaled)
                    y_pred_test = best_model.predict(X_test_scaled)
                    
                    # Calculate comprehensive metrics
                    metrics = self._calculate_institutional_metrics(
                        y_val, y_pred_val, y_test, y_pred_test, best_model, X_val_scaled, X_test_scaled
                    )
                    
                    if metrics['test_roc_auc'] > 0.52:  # Institutional minimum
                        ensemble_models[model_type] = {
                            'model': best_model,
                            'scaler': scaler,
                            'feature_selector': feature_selector,
                            'metrics': metrics,
                            'training_samples': len(X_train),
                            'validation_samples': len(X_val),
                            'test_samples': len(X_test),
                            'trained_at': datetime.now()
                        }
                        
                        logging.info(f"✅ {model_type} for {ticker}: Val AUC={metrics['val_roc_auc']:.3f}, Test AUC={metrics['test_roc_auc']:.3f}")
                    else:
                        logging.warning(f"❌ {model_type} for {ticker} failed quality threshold: AUC={metrics['test_roc_auc']:.3f}")
                        
                except Exception as model_error:
                    logging.error(f"Training failed for {model_type} on {ticker}: {model_error}")
                    continue
            
            if not ensemble_models:
                logging.error(f"No models passed institutional quality standards for {ticker}")
                return {}
            
            # Create ensemble package
            ensemble_package = {
                'models': ensemble_models,
                'ensemble_type': 'institutional_weighted',
                'ticker': ticker,
                'horizon': horizon,
                'trained_at': datetime.now(),
                'ensemble_metrics': self._calculate_ensemble_metrics(ensemble_models),
                'model_weights': self._calculate_ensemble_weights(ensemble_models),
                'feature_importance': self._calculate_ensemble_feature_importance(ensemble_models)
            }
            
            # Store training history
            self.training_history.append({
                'ticker': ticker,
                'horizon': horizon,
                'models_trained': len(ensemble_models),
                'best_auc': max(m['metrics']['test_roc_auc'] for m in ensemble_models.values()),
                'timestamp': datetime.now()
            })
            
            return ensemble_package
            
        except Exception as e:
            logging.error(f"Institutional ensemble training failed for {ticker}: {e}")
            return {}
    
    def _optimize_hyperparameters(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Optional[Any]:
        """Optimize hyperparameters using Optuna"""
        
        if not OPTUNA_AVAILABLE:
            return self.create_institutional_model(model_type, X_train, y_train)
        
        try:
            def objective(trial):
                model = self.create_institutional_model(model_type, X_train, y_train, trial)
                if model is None:
                    return 0.0
                
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
                
                try:
                    score = roc_auc_score(y_val, y_pred)
                except:
                    score = accuracy_score(y_val, model.predict(X_val))
                
                return score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, timeout=300)  # 5 minutes max per model
            
            # Create best model with optimal parameters
            best_model = self.create_institutional_model(model_type, X_train, y_train, study.best_trial)
            
            logging.info(f"Hyperparameter optimization for {model_type}: Best score = {study.best_value:.4f}")
            
            return best_model
            
        except Exception as e:
            logging.warning(f"Hyperparameter optimization failed for {model_type}: {e}")
            return self.create_institutional_model(model_type, X_train, y_train)
    
    def _calculate_institutional_metrics(self, y_val: np.ndarray, y_pred_val: np.ndarray,
                                       y_test: np.ndarray, y_pred_test: np.ndarray,
                                       model: Any, X_val: np.ndarray, X_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive institutional-grade metrics"""
        
        metrics = {}
        
        try:
            # Validation metrics
            metrics['val_accuracy'] = accuracy_score(y_val, y_pred_val)
            metrics['val_precision'] = precision_score(y_val, y_pred_val, zero_division=0)
            metrics['val_recall'] = recall_score(y_val, y_pred_val, zero_division=0)
            metrics['val_f1'] = f1_score(y_val, y_pred_val, zero_division=0)
            
            # Test metrics
            metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
            metrics['test_precision'] = precision_score(y_test, y_pred_test, zero_division=0)
            metrics['test_recall'] = recall_score(y_test, y_pred_test, zero_division=0)
            metrics['test_f1'] = f1_score(y_test, y_pred_test, zero_division=0)
            
            # ROC AUC
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba_val = model.predict_proba(X_val)[:, 1]
                    y_proba_test = model.predict_proba(X_test)[:, 1]
                    
                    metrics['val_roc_auc'] = roc_auc_score(y_val, y_proba_val)
                    metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba_test)
                    
                except:
                    metrics['val_roc_auc'] = 0.5
                    metrics['test_roc_auc'] = 0.5
            else:
                metrics['val_roc_auc'] = 0.5
                metrics['test_roc_auc'] = 0.5
            
            # Institutional-specific metrics
            metrics['generalization_score'] = metrics['test_accuracy'] / metrics['val_accuracy'] if metrics['val_accuracy'] > 0 else 0
            metrics['stability_score'] = 1 - abs(metrics['test_accuracy'] - metrics['val_accuracy'])
            
            # Risk-adjusted metrics
            if metrics['test_precision'] > 0 and metrics['test_recall'] > 0:
                metrics['risk_adjusted_f1'] = 2 * (metrics['test_precision'] * metrics['test_recall']) / (metrics['test_precision'] + metrics['test_recall'])
            else:
                metrics['risk_adjusted_f1'] = 0
            
            # Institutional quality score (composite metric)
            metrics['institutional_quality'] = (
                0.3 * metrics['test_roc_auc'] + 
                0.25 * metrics['test_f1'] + 
                0.25 * metrics['stability_score'] + 
                0.2 * metrics['generalization_score']
            )
            
        except Exception as e:
            logging.warning(f"Institutional metrics calculation failed: {e}")
            # Return default metrics
            metrics = {
                'val_accuracy': 0.5, 'test_accuracy': 0.5,
                'val_precision': 0.5, 'test_precision': 0.5,
                'val_recall': 0.5, 'test_recall': 0.5,
                'val_f1': 0.5, 'test_f1': 0.5,
                'val_roc_auc': 0.5, 'test_roc_auc': 0.5,
                'institutional_quality': 0.5
            }
        
        return metrics
    
    def _calculate_ensemble_metrics(self, ensemble_models: Dict) -> Dict[str, float]:
        """Calculate ensemble-level metrics"""
        
        if not ensemble_models:
            return {}
        
        try:
            # Aggregate metrics across models
            all_metrics = [model_data['metrics'] for model_data in ensemble_models.values()]
            
            ensemble_metrics = {}
            
            # Calculate mean and std for each metric
            for metric_name in all_metrics[0].keys():
                values = [metrics[metric_name] for metrics in all_metrics]
                ensemble_metrics[f'mean_{metric_name}'] = np.mean(values)
                ensemble_metrics[f'std_{metric_name}'] = np.std(values)
                ensemble_metrics[f'min_{metric_name}'] = np.min(values)
                ensemble_metrics[f'max_{metric_name}'] = np.max(values)
            
            # Overall ensemble quality
            quality_scores = [metrics['institutional_quality'] for metrics in all_metrics]
            ensemble_metrics['ensemble_quality'] = np.mean(quality_scores)
            ensemble_metrics['model_consistency'] = 1 - np.std(quality_scores)
            
            return ensemble_metrics
            
        except Exception as e:
            logging.warning(f"Ensemble metrics calculation failed: {e}")
            return {}
    
    def _calculate_ensemble_weights(self, ensemble_models: Dict) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on performance"""
        
        if not ensemble_models:
            return {}
        
        try:
            # Weight based on institutional quality score
            quality_scores = {}
            for model_type, model_data in ensemble_models.items():
                quality_scores[model_type] = model_data['metrics']['institutional_quality']
            
            # Softmax transformation for weights
            max_score = max(quality_scores.values())
            exp_scores = {k: np.exp(v - max_score) for k, v in quality_scores.items()}
            total_exp = sum(exp_scores.values())
            
            weights = {k: exp_score / total_exp for k, exp_score in exp_scores.items()}
            
            return weights
            
        except Exception as e:
            logging.warning(f"Ensemble weight calculation failed: {e}")
            # Return equal weights
            n_models = len(ensemble_models)
            return {k: 1.0/n_models for k in ensemble_models.keys()}
    
    def _calculate_ensemble_feature_importance(self, ensemble_models: Dict) -> Dict[str, float]:
        """Calculate ensemble feature importance"""
        
        try:
            feature_importances = {}
            
            for model_type, model_data in ensemble_models.items():
                model = model_data['model']
                
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                else:
                    continue
                
                # Store with model weight
                model_weight = 1.0 / len(ensemble_models)  # Equal weight for simplicity
                for i, importance in enumerate(importances):
                    feature_name = f'feature_{i}'
                    if feature_name not in feature_importances:
                        feature_importances[feature_name] = 0
                    feature_importances[feature_name] += importance * model_weight
            
            return feature_importances
            
        except Exception as e:
            logging.warning(f"Feature importance calculation failed: {e}")
            return {}

# ==================== MAIN TRAINING FUNCTIONS ====================

def train_models_enhanced_parallel(featured_data: Dict[str, pd.DataFrame], 
                                 config: Dict = None,
                                 selected_tickers: List[str] = None) -> Dict[str, Any]:
    """
    Train institutional-grade models with advanced parallel processing
    """
    
    config = config or INSTITUTIONAL_MODEL_CONFIG
    
    if not featured_data:
        logging.warning("No featured data provided for institutional model training")
        return {}
    
    # Filter to selected tickers
    if selected_tickers:
        filtered_data = {ticker: df for ticker, df in featured_data.items() 
                        if ticker in selected_tickers}
    else:
        filtered_data = featured_data
    
    if not filtered_data:
        logging.warning("No valid tickers found for institutional model training")
        return {}
    
    logging.info(f"🏦 Training institutional models for {len(filtered_data)} securities")
    
    # Initialize institutional trainer
    trainer = InstitutionalModelTrainer(config)
    
    # Training results
    all_models = {}
    training_summary = {
        'total_tickers': len(filtered_data),
        'successful_tickers': 0,
        'total_models': 0,
        'avg_quality_score': 0.0,
        'training_time': datetime.now(),
        'model_distribution': {},
        'best_performers': {}
    }
    
    try:
        horizons = config.get('target_horizons', ['next_month'])
        
        if config.get('parallel_training', True) and len(filtered_data) > 1:
            # Parallel training for institutional efficiency
            all_models = _train_institutional_models_parallel(filtered_data, trainer, horizons, config)
        else:
            # Sequential training
            all_models = _train_institutional_models_sequential(filtered_data, trainer, horizons, config)
        
        # Calculate training summary
        successful_tickers = len(all_models)
        total_models = sum(len(ticker_models) for ticker_models in all_models.values())
        
        quality_scores = []
        model_counts = {}
        
        for ticker, ticker_models in all_models.items():
            for horizon, ensemble in ticker_models.items():
                if 'ensemble_metrics' in ensemble:
                    quality_score = ensemble['ensemble_metrics'].get('ensemble_quality', 0)
                    quality_scores.append(quality_score)
                
                for model_type in ensemble.get('models', {}).keys():
                    model_counts[model_type] = model_counts.get(model_type, 0) + 1
                
                # Track best performers
                if ticker not in training_summary['best_performers']:
                    training_summary['best_performers'][ticker] = {
                        'best_quality': quality_score if 'ensemble_metrics' in ensemble else 0,
                        'best_horizon': horizon,
                        'model_count': len(ensemble.get('models', {}))
                    }
                elif quality_score > training_summary['best_performers'][ticker]['best_quality']:
                    training_summary['best_performers'][ticker].update({
                        'best_quality': quality_score,
                        'best_horizon': horizon
                    })
        
        # Update summary
        training_summary.update({
            'successful_tickers': successful_tickers,
            'total_models': total_models,
            'success_rate': successful_tickers / len(filtered_data),
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'model_distribution': model_counts
        })
        
        logging.info(f"🏆 Institutional training completed: {successful_tickers}/{len(filtered_data)} tickers successful")
        logging.info(f"📊 Total models trained: {total_models}")
        logging.info(f"🎯 Average quality score: {training_summary['avg_quality_score']:.3f}")
        
        return {
            'models': all_models,
            'training_summary': training_summary,
            'trainer': trainer,
            'config': config
        }
        
    except Exception as e:
        logging.error(f"Institutional model training failed: {e}")
        return {}

def _train_institutional_models_parallel(filtered_data: Dict, trainer: InstitutionalModelTrainer, 
                                       horizons: List[str], config: Dict) -> Dict:
    """Train institutional models in parallel"""
    
    def train_ticker_ensemble(ticker_data_horizon):
        ticker, df, horizon = ticker_data_horizon
        
        try:
            logging.info(f"🔄 Training {ticker} - {horizon}")
            
            # Prepare institutional features and targets
            X, y, feature_names = trainer.prepare_institutional_features_targets(df, horizon)
            
            if X is None or len(X) < 200:  # Institutional minimum
                logging.warning(f"❌ Insufficient institutional data for {ticker} - {horizon}: {len(X) if X is not None else 0}")
                return ticker, horizon, None
            
            # Train institutional ensemble
            ensemble = trainer.train_institutional_ensemble(X, y, ticker, horizon)
            
            if ensemble:
                logging.info(f"✅ Institutional ensemble trained for {ticker} - {horizon}")
                return ticker, horizon, ensemble
            else:
                logging.warning(f"❌ Institutional ensemble training failed for {ticker} - {horizon}")
                return ticker, horizon, None
                
        except Exception as e:
            logging.error(f"❌ Parallel training failed for {ticker} - {horizon}: {e}")
            return ticker, horizon, None
    
    # Prepare tasks
    tasks = []
    for ticker, df in filtered_data.items():
        for horizon in horizons:
            tasks.append((ticker, df, horizon))
    
    # Execute parallel training
    all_models = {}
    
    max_workers = min(config.get('max_workers', 4), len(tasks))
    batch_size = config.get('batch_size', 4)
    
    # Process in batches to manage memory
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(train_ticker_ensemble, task): task 
                    for task in batch_tasks
                }
                
                for future in as_completed(future_to_task):
                    ticker, horizon, ensemble = future.result(timeout=1800)  # 30 min timeout
                    
                    if ensemble is not None:
                        if ticker not in all_models:
                            all_models[ticker] = {}
                        all_models[ticker][horizon] = ensemble
        
        except Exception as batch_error:
            logging.error(f"Batch training failed: {batch_error}")
            # Continue with remaining batches
            
        # Memory cleanup between batches
        gc.collect()
        time.sleep(1)
    
    return all_models

def _train_institutional_models_sequential(filtered_data: Dict, trainer: InstitutionalModelTrainer,
                                         horizons: List[str], config: Dict) -> Dict:
    """Train institutional models sequentially"""
    
    all_models = {}
    
    for ticker, df in filtered_data.items():
        try:
            ticker_models = {}
            
            for horizon in horizons:
                logging.info(f"🔄 Training {ticker} - {horizon} (sequential)")
                
                # Prepare institutional features and targets
                X, y, feature_names = trainer.prepare_institutional_features_targets(df, horizon)
                
                if X is None or len(X) < 200:  # Institutional minimum
                    logging.warning(f"❌ Insufficient institutional data for {ticker} - {horizon}")
                    continue
                
                # Train institutional ensemble
                ensemble = trainer.train_institutional_ensemble(X, y, ticker, horizon)
                
                if ensemble:
                    ticker_models[horizon] = ensemble
                    logging.info(f"✅ Institutional ensemble trained for {ticker} - {horizon}")
                else:
                    logging.warning(f"❌ Institutional ensemble training failed for {ticker} - {horizon}")
            
            if ticker_models:
                all_models[ticker] = ticker_models
                
        except Exception as e:
            logging.error(f"❌ Sequential training failed for {ticker}: {e}")
            continue
    
    return all_models

# ==================== PREDICTION FUNCTIONS ====================

def predict_with_ensemble(models: Dict[str, Any], 
                         featured_data: Dict[str, pd.DataFrame],
                         investment_horizon: str = 'next_month',
                         selected_tickers: List[str] = None) -> pd.DataFrame:
    """
    Generate institutional-grade predictions using ensemble models
    """
    
    try:
        if not models or not featured_data:
            logging.warning("No models or data provided for institutional predictions")
            return pd.DataFrame()
        
        # Filter to selected tickers
        if selected_tickers:
            filtered_tickers = [ticker for ticker in selected_tickers 
                               if ticker in models and ticker in featured_data]
        else:
            filtered_tickers = list(set(models.keys()) & set(featured_data.keys()))
        
        if not filtered_tickers:
            logging.warning("No valid tickers found for institutional predictions")
            return pd.DataFrame()
        
        logging.info(f"🔮 Generating institutional predictions for {len(filtered_tickers)} securities")
        
        predictions = []
        
        for ticker in filtered_tickers:
            try:
                ticker_models = models.get(ticker, {})
                horizon_ensemble = ticker_models.get(investment_horizon, {})
                
                if not horizon_ensemble or 'models' not in horizon_ensemble:
                    logging.warning(f"No ensemble models found for {ticker} - {investment_horizon}")
                    continue
                
                df = featured_data[ticker]
                if df.empty:
                    continue
                
                # Prepare latest features for prediction
                latest_features = _prepare_institutional_features(df, horizon_ensemble)
                
                if latest_features is None:
                    continue
                
                # Generate institutional ensemble prediction
                prediction_result = _generate_institutional_ensemble_prediction(
                    horizon_ensemble, latest_features, ticker, investment_horizon
                )
                
                if prediction_result:
                    predictions.append(prediction_result)
                
            except Exception as e:
                logging.warning(f"Institutional prediction failed for {ticker}: {e}")
                continue
        
        # Convert to DataFrame with institutional formatting
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            
            # Add institutional metadata
            predictions_df['prediction_timestamp'] = datetime.now()
            predictions_df['model_version'] = 'institutional_v1.0'
            predictions_df['confidence_tier'] = predictions_df['ensemble_confidence'].apply(
                lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
            )
            
            logging.info(f"🎯 Generated institutional predictions for {len(predictions_df)} securities")
            return predictions_df
        else:
            logging.warning("No institutional predictions generated")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Institutional ensemble prediction failed: {e}")
        return pd.DataFrame()

def _prepare_institutional_features(df: pd.DataFrame, ensemble: Dict) -> Optional[np.ndarray]:
    """Prepare features for institutional prediction"""
    
    try:
        # Get the feature transformer from the first model
        first_model_data = list(ensemble['models'].values())[0]
        scaler = first_model_data.get('scaler')
        feature_selector = first_model_data.get('feature_selector')
        
        # Prepare features similar to training
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
            'Date', 'Datetime', 'date', 'datetime'
        ]
        
        target_cols = [col for col in df.columns if any(term in col.lower() 
                      for term in ['target_', 'return_', 'label_', 'y_'])]
        exclude_cols.extend(target_cols)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            return None
        
        # Get latest row and handle missing values
        latest_features = df[feature_cols].iloc[-1:].copy()
        latest_features = latest_features.fillna(method='ffill').fillna(0)
        latest_features = latest_features.replace([np.inf, -np.inf], 0)
        
        # Apply same transformations as in training
        if scaler is not None:
            latest_features_scaled = scaler.transform(latest_features)
        else:
            latest_features_scaled = latest_features.values
        
        if feature_selector is not None:
            latest_features_final = feature_selector.transform(latest_features_scaled)
        else:
            latest_features_final = latest_features_scaled
        
        return latest_features_final
        
    except Exception as e:
        logging.error(f"Institutional feature preparation failed: {e}")
        return None

def _generate_institutional_ensemble_prediction(ensemble: Dict, features: np.ndarray, 
                                              ticker: str, horizon: str) -> Optional[Dict]:
    """Generate sophisticated institutional ensemble prediction"""
    
    try:
        ensemble_models = ensemble.get('models', {})
        ensemble_weights = ensemble.get('model_weights', {})
        
        if not ensemble_models:
            return None
        
        predictions = []
        confidences = []
        model_contributions = {}
        
        for model_type, model_data in ensemble_models.items():
            try:
                model = model_data['model']
                weight = ensemble_weights.get(model_type, 1.0 / len(ensemble_models))
                
                # Get prediction
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    prediction = 1 if proba[1] > 0.5 else 0
                    confidence = max(proba)
                else:
                    prediction = model.predict(features)[0]
                    confidence = 0.6  # Default confidence for non-probabilistic models
                
                # Weight the prediction
                weighted_prediction = prediction * weight
                weighted_confidence = confidence * weight
                
                predictions.append(weighted_prediction)
                confidences.append(weighted_confidence)
                
                model_contributions[model_type] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': weighted_prediction
                }
                
            except Exception as model_error:
                logging.warning(f"Individual institutional model prediction failed for {model_type}: {model_error}")
                continue
        
        if not predictions:
            return None
        
        # Institutional ensemble aggregation
        ensemble_prediction = sum(predictions)
        ensemble_confidence = sum(confidences)
        
        # Convert to binary decision with institutional threshold
        institutional_threshold = 0.6  # Higher threshold for institutional quality
        final_prediction = 1 if ensemble_prediction > institutional_threshold else 0
        
        # Calculate institutional return expectation
        if final_prediction == 1:
            # Positive prediction - scale by confidence
            expected_return = 0.08 * ensemble_confidence  # Base 8% scaled by confidence
        else:
            # Negative or neutral prediction
            expected_return = -0.02 * (1 - ensemble_confidence)  # Small negative expectation
        
        # Institutional prediction result
        result = {
            'ticker': ticker,
            'predicted_return': expected_return,
            'prediction_direction': 'BUY' if final_prediction == 1 else 'HOLD',
            'ensemble_confidence': ensemble_confidence,
            'signal_strength': ensemble_confidence,
            'horizon': horizon,
            'model_contributions': model_contributions,
            'ensemble_prediction_raw': ensemble_prediction,
            'institutional_threshold': institutional_threshold,
            'model_count': len(ensemble_models),
            'prediction_timestamp': datetime.now(),
            'prediction_quality': 'institutional'
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Institutional ensemble prediction generation failed: {e}")
        return None

def generate_price_targets_for_selected_stocks(models: Dict[str, Any],
                                             raw_data: Dict[str, pd.DataFrame],
                                             investment_horizon: str = 'next_month',
                                             selected_tickers: List[str] = None) -> pd.DataFrame:
    """
    Generate institutional-grade price targets
    """
    
    try:
        if not models or not raw_data:
            return pd.DataFrame()
        
        # Filter to selected tickers
        if selected_tickers:
            filtered_tickers = [ticker for ticker in selected_tickers 
                               if ticker in models and ticker in raw_data]
        else:
            filtered_tickers = list(set(models.keys()) & set(raw_data.keys()))
        
        if not filtered_tickers:
            return pd.DataFrame()
        
        logging.info(f"🎯 Generating institutional price targets for {len(filtered_tickers)} securities")
        
        price_targets = []
        
        for ticker in filtered_tickers:
            try:
                df = raw_data[ticker]
                if df.empty:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Get institutional model predictions
                ticker_models = models.get(ticker, {})
                horizon_ensemble = ticker_models.get(investment_horizon, {})
                
                if horizon_ensemble and 'ensemble_metrics' in horizon_ensemble:
                    # Use institutional quality score for price target confidence
                    ensemble_quality = horizon_ensemble['ensemble_metrics'].get('ensemble_quality', 0.5)
                    
                    # Calculate sophisticated price targets based on multiple factors
                    
                    # 1. Historical return analysis
                    returns = df['Close'].pct_change().dropna()
                    
                    if len(returns) > 252:  # Need at least 1 year
                        # Calculate risk-adjusted expected returns
                        mean_return = returns.mean()
                        std_return = returns.std()
                        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                        
                        # Institutional horizon mapping
                        horizon_days = {
                            'next_week': 5,
                            'next_month': 21,
                            'next_quarter': 63,
                            'next_6_months': 126
                        }.get(investment_horizon, 21)
                        
                        # Expected return adjusted by model quality
                        base_expected_return = mean_return * horizon_days
                        quality_adjusted_return = base_expected_return * (0.5 + 0.5 * ensemble_quality)
                        
                        # Volatility adjustment
                        volatility_adjustment = std_return * np.sqrt(horizon_days)
                        
                        # Conservative, base, and optimistic targets
                        conservative_return = quality_adjusted_return - volatility_adjustment
                        optimistic_return = quality_adjusted_return + volatility_adjustment
                        
                        # Final target based on model confidence
                        final_expected_return = quality_adjusted_return
                        target_price = current_price * (1 + final_expected_return)
                        
                        # Calculate support and resistance levels
                        high_52w = df['High'].tail(252).max() if len(df) > 252 else df['High'].max()
                        low_52w = df['Low'].tail(252).min() if len(df) > 252 else df['Low'].min()
                        
                        # Risk metrics
                        upside_potential = (target_price - current_price) / current_price
                        downside_risk = (current_price - low_52w) / current_price
                        
                        # Institutional recommendation logic
                        if upside_potential > 0.15 and ensemble_quality > 0.7:
                            recommendation = 'STRONG_BUY'
                        elif upside_potential > 0.08 and ensemble_quality > 0.6:
                            recommendation = 'BUY'
                        elif upside_potential > 0.03 and ensemble_quality > 0.5:
                            recommendation = 'HOLD'
                        else:
                            recommendation = 'SELL'
                    
                    else:
                        # Fallback for insufficient data
                        target_price = current_price * 1.05
                        upside_potential = 0.05
                        high_52w = current_price * 1.1
                        low_52w = current_price * 0.9
                        recommendation = 'HOLD'
                
                else:
                    # No model available - use technical analysis
                    if len(df) > 50:
                        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
                        target_price = (current_price + sma_50) / 2  # Mean reversion target
                    else:
                        target_price = current_price * 1.03  # Conservative 3%
                    
                    upside_potential = (target_price - current_price) / current_price
                    high_52w = df['High'].max()
                    low_52w = df['Low'].min()
                    recommendation = 'HOLD'
                    ensemble_quality = 0.3  # Low confidence
                
                # Create institutional price target record
                price_targets.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'target_price': target_price,
                    'upside_potential': upside_potential,
                    'resistance_level': high_52w,
                    'support_level': low_52w,
                    'horizon': investment_horizon,
                    'confidence': ensemble_quality,
                    'recommendation': recommendation,
                    'target_date': datetime.now() + timedelta(days={
                        'next_week': 7, 'next_month': 30, 'next_quarter': 90, 'next_6_months': 180
                    }.get(investment_horizon, 30)),
                    'risk_reward_ratio': upside_potential / max(abs(upside_potential), 0.01),
                    'institutional_grade': True,
                    'analysis_timestamp': datetime.now()
                })
                
            except Exception as e:
                logging.warning(f"Price target generation failed for {ticker}: {e}")
                continue
        
        if price_targets:
            targets_df = pd.DataFrame(price_targets)
            
            # Add institutional rankings
            targets_df['upside_rank'] = targets_df['upside_potential'].rank(ascending=False)
            targets_df['confidence_rank'] = targets_df['confidence'].rank(ascending=False)
            targets_df['composite_score'] = (
                0.6 * targets_df['upside_potential'] + 
                0.4 * targets_df['confidence']
            )
            targets_df['overall_rank'] = targets_df['composite_score'].rank(ascending=False)
            
            logging.info(f"🎯 Generated institutional price targets for {len(targets_df)} securities")
            
            return targets_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Institutional price target generation failed: {e}")
        return pd.DataFrame()

def predict_with_ensemble_and_targets(models: Dict[str, Any],
                                     featured_data: Dict[str, pd.DataFrame],
                                     raw_data: Dict[str, pd.DataFrame],
                                     investment_horizon: str = 'next_month',
                                     selected_tickers: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combined institutional predictions and price targets
    """
    
    predictions_df = predict_with_ensemble(
        models, featured_data, investment_horizon, selected_tickers
    )
    
    price_targets_df = generate_price_targets_for_selected_stocks(
        models, raw_data, investment_horizon, selected_tickers
    )
    
    return predictions_df, price_targets_df

# ==================== MODEL PERSISTENCE ====================

def save_models_optimized(models: Dict[str, Any], filename: str) -> bool:
    """Save institutional models with optimization"""
    
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Institutional-grade compression
        with open(filename, 'wb') as f:
            joblib.dump(models, f, compress=('lz4', 3))  # High compression
        
        logging.info(f"🏦 Institutional models saved to {filename}")
        return True
        
    except Exception as e:
        logging.error(f"Institutional model saving failed: {e}")
        return False

def load_models_optimized(filename: str) -> Dict[str, Any]:
    """Load institutional models with optimization"""
    
    try:
        if not os.path.exists(filename):
            logging.warning(f"Institutional model file not found: {filename}")
            return {}
        
        with open(filename, 'rb') as f:
            models = joblib.load(f)
        
        logging.info(f"🏦 Institutional models loaded from {filename}")
        return models
        
    except Exception as e:
        logging.error(f"Institutional model loading failed: {e}")
        return {}

# ==================== EXPORT ====================

__all__ = [
    'train_models_enhanced_parallel',
    'predict_with_ensemble',
    'generate_price_targets_for_selected_stocks', 
    'predict_with_ensemble_and_targets',
    'save_models_optimized',
    'load_models_optimized',
    'INSTITUTIONAL_MODEL_CONFIG',
    'InstitutionalModelTrainer',
    'AdvancedFeatureTransformer'
]