# utils/model.py
from .feature_engineer import calculate_risk_score
import numpy as np
import pandas as pd
import joblib
import os
import gc
import optuna
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import warnings
from xgboost import XGBClassifier
from sklearn.base import clone

# Enhanced Configuration
MODEL_CONFIG = {
    'n_jobs': -1,
    'random_state': 42,
    'cv_folds': 5,  # Increased from 3 for more robust validation
    'test_size': 0.2,
    'model_types': ['random_forest', 'xgboost', 'gradient_boosting'],  # Added model diversity
    'param_space': {
        'random_forest': {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None],
            'max_features': ['sqrt', 'log2', None]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'horizons': {
        'short_term': ['next_day', 'next_week'],
        'medium_term': ['next_month', 'next_quarter'],
        'long_term': ['next_year', 'next_3_years', 'next_5_years']
    }
}

class StockPredictor:
    """Enhanced stock prediction model handler with multiple model types"""
    
    def __init__(self, horizon: str, model_type: str = 'random_forest'):
        self.horizon = horizon
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.feature_importances = None
        self.required_features = None
        self.uncertainty_estimates = None  # For prediction intervals
        self.cv_scores = None  # Store cross-validation results

    def train(self, X: pd.DataFrame, y: pd.Series, tune: bool = True):
        """Enhanced training with model diversity and uncertainty estimation"""
        try:
            # Validate feature columns
            self._validate_features(X)
            
            if tune:
                self._hyperparameter_tuning(X, y)
            
            # Initialize model with best or default parameters
            model = self._init_model()
            
            # Enhanced time-series cross-validation
            self._cross_validate(model, X, y)
            
            # Final training on full data
            self.model = model.fit(X, y)
            
            # Feature importance and uncertainty estimation
            self._post_training_analysis(X, y)
            
        except Exception as e:
            warnings.warn(f"Training failed: {str(e)}")
            self.model = None

    def _init_model(self):
        """Initialize appropriate model based on type"""
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                **self.best_params if self.best_params else {
                    'class_weight': 'balanced',
                    'n_jobs': MODEL_CONFIG['n_jobs'],
                    'random_state': MODEL_CONFIG['random_state']
                }
            )
        elif self.model_type == 'xgboost':
            model = XGBClassifier(
                **self.best_params if self.best_params else {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'random_state': MODEL_CONFIG['random_state']
                }
            )
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                **self.best_params if self.best_params else {
                    'random_state': MODEL_CONFIG['random_state']
                }
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return model

    def _validate_features(self, X: pd.DataFrame):
        """Validate that required features exist"""
        if X.empty:
            raise ValueError("Empty feature set provided for training")
        self.required_features = X.columns.tolist()

    def _hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series):
        """Enhanced Optuna-based hyperparameter optimization with pruning"""
        def objective(trial):
            params = {}
            model_type = self.model_type
            
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', 
                                 MODEL_CONFIG['param_space']['random_forest']['n_estimators']),
                    'max_depth': trial.suggest_categorical('max_depth', 
                               MODEL_CONFIG['param_space']['random_forest']['max_depth']),
                    'min_samples_split': trial.suggest_int('min_samples_split', 
                                       min(MODEL_CONFIG['param_space']['random_forest']['min_samples_split']),
                                       max(MODEL_CONFIG['param_space']['random_forest']['min_samples_split'])),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 
                                      min(MODEL_CONFIG['param_space']['random_forest']['min_samples_leaf']),
                                      max(MODEL_CONFIG['param_space']['random_forest']['min_samples_leaf'])),
                    'class_weight': trial.suggest_categorical('class_weight', 
                                 MODEL_CONFIG['param_space']['random_forest']['class_weight']),
                    'max_features': trial.suggest_categorical('max_features',
                                 MODEL_CONFIG['param_space']['random_forest']['max_features'])
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators',
                                 MODEL_CONFIG['param_space']['xgboost']['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth',
                                      min(MODEL_CONFIG['param_space']['xgboost']['max_depth']),
                                      max(MODEL_CONFIG['param_space']['xgboost']['max_depth'])),
                    'learning_rate': trial.suggest_float('learning_rate',
                                      min(MODEL_CONFIG['param_space']['xgboost']['learning_rate']),
                                      max(MODEL_CONFIG['param_space']['xgboost']['learning_rate'])),
                    'subsample': trial.suggest_float('subsample',
                                    min(MODEL_CONFIG['param_space']['xgboost']['subsample']),
                                    max(MODEL_CONFIG['param_space']['xgboost']['subsample'])),
                    'colsample_bytree': trial.suggest_float('colsample_bytree',
                                          min(MODEL_CONFIG['param_space']['xgboost']['colsample_bytree']),
                                          max(MODEL_CONFIG['param_space']['xgboost']['colsample_bytree']))
                }
                model = XGBClassifier(**params)
                
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators',
                                 MODEL_CONFIG['param_space']['gradient_boosting']['n_estimators']),
                    'learning_rate': trial.suggest_float('learning_rate',
                                      min(MODEL_CONFIG['param_space']['gradient_boosting']['learning_rate']),
                                      max(MODEL_CONFIG['param_space']['gradient_boosting']['learning_rate'])),
                    'max_depth': trial.suggest_int('max_depth',
                               min(MODEL_CONFIG['param_space']['gradient_boosting']['max_depth']),
                               max(MODEL_CONFIG['param_space']['gradient_boosting']['max_depth'])),
                    'min_samples_split': trial.suggest_int('min_samples_split',
                                       min(MODEL_CONFIG['param_space']['gradient_boosting']['min_samples_split']),
                                       max(MODEL_CONFIG['param_space']['gradient_boosting']['min_samples_split'])),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                                      min(MODEL_CONFIG['param_space']['gradient_boosting']['min_samples_leaf']),
                                      max(MODEL_CONFIG['param_space']['gradient_boosting']['min_samples_leaf']))
                }
                model = GradientBoostingClassifier(**params)
            
            cv = TimeSeriesSplit(MODEL_CONFIG['cv_folds'])
            scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                scores.append(roc_auc_score(y_val, preds))  # Using AUC for better class imbalance handling
                
                # Intermediate pruning for efficiency
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=20, timeout=600)  # Increased trials with timeout
        self.best_params = study.best_params
        self.cv_scores = study.best_value

    def _cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series):
        """Enhanced time-series cross-validation with metrics tracking"""
        cv = TimeSeriesSplit(MODEL_CONFIG['cv_folds'])
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            preds = cloned_model.predict(X_val)
            probas = cloned_model.predict_proba(X_val)[:, 1] if hasattr(cloned_model, "predict_proba") else [0.5]*len(X_val)
            
            metrics['precision'].append(precision_score(y_val, preds, zero_division=0))
            metrics['recall'].append(recall_score(y_val, preds, zero_division=0))
            metrics['f1'].append(f1_score(y_val, preds, zero_division=0))
            try:
                metrics['roc_auc'].append(roc_auc_score(y_val, probas) if len(np.unique(y_val)) > 1 else 0.5)
            except:
                metrics['roc_auc'].append(0.5)
        
        self.cv_scores = {k: np.mean(v) for k, v in metrics.items()}

    def _post_training_analysis(self, X: pd.DataFrame, y: pd.Series):
        """Calculate feature importance and uncertainty estimates"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
        self.required_features = X.columns.tolist()
        
        # Uncertainty estimation using bootstrap approach
        if self.model_type == 'random_forest' and hasattr(self.model, 'estimators_'):
            self.uncertainty_estimates = {
                'std': np.std([tree.predict_proba(X)[:, 1] for tree in self.model.estimators_], axis=0),
                'min': np.min([tree.predict_proba(X)[:, 1] for tree in self.model.estimators_], axis=0),
                'max': np.max([tree.predict_proba(X)[:, 1] for tree in self.model.estimators_], axis=0)
            }

def train_all_models(featured_data: Dict[str, pd.DataFrame],
                    horizons: List[str] = None,
                    model_types: List[str] = None) -> Dict[str, Any]:
    """
    Enhanced training function with model diversity and parallelization
    """
    all_horizons = [h for group in MODEL_CONFIG['horizons'].values() for h in group]
    horizons = horizons or all_horizons
    model_types = model_types or MODEL_CONFIG['model_types']
    
    all_models = {}
    metrics = []
    
    for ticker, original_df in tqdm(featured_data.items(), desc="Training models"):
        try:
            if original_df.empty:
                continue
                
            # Get pure feature columns (no targets)
            pure_features = [col for col in original_df.columns 
                           if not col.startswith('Target_')]
            
            ticker_models = {}
            for horizon in horizons:
                for model_type in model_types:
                    try:
                        target_col = f"Target_{horizon}"
                        if target_col not in original_df.columns:
                            continue
                            
                        # Create isolated dataset
                        df = original_df[pure_features + [target_col]].copy()
                        
                        # Temporal split
                        split_idx = int(len(df) * (1 - MODEL_CONFIG['test_size']))
                        train = df.iloc[:split_idx]
                        test = df.iloc[split_idx:]
                        
                        # Prepare data
                        X_train = train[pure_features]
                        y_train = train[target_col]
                        X_test = test[pure_features]
                        y_test = test[target_col]
                        
                        # Skip if insufficient data
                        if X_train.empty or len(X_train) < 50:
                            continue
                            
                        # Train and validate
                        predictor = StockPredictor(horizon, model_type)
                        predictor.train(X_train, y_train, tune=True)
                        
                        # Store results
                        model_key = f"{model_type}_{horizon}"
                        ticker_models[model_key] = predictor
                        
                        # Evaluate on test set
                        if not X_test.empty:
                            if not hasattr(predictor.model, "predict"):
                                continue
                                
                            test_preds = predictor.model.predict(X_test)
                            test_probas = predictor.model.predict_proba(X_test)[:, 1] if hasattr(predictor.model, "predict_proba") else [0.5]*len(X_test)
                            
                            metrics.append({
                                'ticker': ticker,
                                'horizon': horizon,
                                'model_type': model_type,
                                'precision': precision_score(y_test, test_preds, zero_division=0),
                                'recall': recall_score(y_test, test_preds, zero_division=0),
                                'f1': f1_score(y_test, test_preds, zero_division=0),
                                'roc_auc': roc_auc_score(y_test, test_probas) if len(np.unique(y_test)) > 1 else 0.5,
                                'cv_score': predictor.cv_scores['roc_auc'] if predictor.cv_scores else 0.5
                            })
                            
                    except Exception as e:
                        warnings.warn(f"{ticker} {horizon} {model_type} failed: {str(e)}")
                        continue
                        
            if ticker_models:
                all_models[ticker] = ticker_models
            gc.collect()
            
        except Exception as e:
            warnings.warn(f"{ticker} failed: {str(e)}")
            continue
            
    return {
        'models': all_models,
        'metrics': pd.DataFrame(metrics) if metrics else pd.DataFrame()
    }

def save_models(models: Dict[str, Any], directory: str = "models"):
    """Enhanced model saving with versioning"""
    try:
        os.makedirs(directory, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(directory, f"v_{timestamp}")
        os.makedirs(version_dir, exist_ok=True)
        
        for ticker, horizons in models.items():
            for model_key, model in horizons.items():
                try:
                    clean_ticker = ticker.replace(".NS", "")
                    filename = f"{clean_ticker}_{model_key}.joblib"
                    filepath = os.path.join(version_dir, filename)
                    joblib.dump(model, filepath)
                    
                    # Save metadata
                    metadata = {
                        'ticker': ticker,
                        'model_type': model.model_type,
                        'horizon': model.horizon,
                        'cv_scores': model.cv_scores,
                        'feature_importances': model.feature_importances,
                        'training_date': timestamp
                    }
                    joblib.dump(metadata, os.path.join(version_dir, f"{clean_ticker}_{model_key}_meta.joblib"))
                except Exception as e:
                    warnings.warn(f"Failed to save {ticker} {model_key}: {str(e)}")
                    
        # Create symlink to latest version
        latest_path = os.path.join(directory, "latest")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(version_dir, latest_path)
        print(f"Models saved to {version_dir} and symlinked to {latest_path}")
        
    except Exception as e:
        warnings.warn(f"Failed to save models: {str(e)}")

def load_models(directory: str = "models", version: str = "latest") -> Dict[str, Any]:
    """Enhanced model loading with version support"""
    loaded_models = {}
    version_dir = os.path.join(directory, version)
    
    try:
        if not os.path.exists(version_dir):
            # Don't warn here - it's expected behavior for first run
            return loaded_models
            
        for filename in os.listdir(version_dir):
            if filename.endswith(".joblib") and not filename.endswith("_meta.joblib"):
                try:
                    base_name = filename[:-7]  # Remove .joblib
                    parts = base_name.split("_")
                    ticker_part = parts[0]
                    ticker = f"{ticker_part}.NS"
                    model_key = "_".join(parts[1:])
                    
                    path = os.path.join(version_dir, filename)
                    model = joblib.load(path)
                    
                    # Load metadata
                    meta_path = os.path.join(version_dir, f"{base_name}_meta.joblib")
                    if os.path.exists(meta_path):
                        metadata = joblib.load(meta_path)
                        model.cv_scores = metadata.get('cv_scores')
                        model.feature_importances = metadata.get('feature_importances')
                    
                    if ticker not in loaded_models:
                        loaded_models[ticker] = {}
                    loaded_models[ticker][model_key] = model
                except Exception as e:
                    warnings.warn(f"Failed to load {filename}: {str(e)}")
    except Exception as e:
        warnings.warn(f"Critical error loading models: {str(e)}")
    
    return loaded_models

def predict_returns(models: Dict[str, Any], 
                   current_data: Dict[str, pd.DataFrame],
                   investment_horizon: str,
                   model_type: str = 'random_forest') -> pd.DataFrame:
    """
    Enhanced prediction with model selection and uncertainty estimation
    """
    predictions = []
    model_key = f"{model_type}_{investment_horizon}"
    
    for ticker, model_dict in models.items():
        try:
            if ticker not in current_data:
                continue
                
            # Prepare prediction data
            df = current_data[ticker]
            if df.empty:
                continue
                
            latest_data = df.iloc[[-1]].copy()
            latest_data = latest_data[[col for col in latest_data.columns 
                                     if not col.startswith('Target_')]]
            
            horizon_model = model_dict.get(model_key)
            
            if not horizon_model or not hasattr(horizon_model, 'model') or not horizon_model.model:
                continue

            # Validate feature compatibility
            if not hasattr(horizon_model, 'required_features'):
                continue
                
            missing_features = set(horizon_model.required_features) - set(latest_data.columns)
            if missing_features:
                # Try to impute missing features with mean
                for feature in missing_features:
                    if feature in df.columns:
                        latest_data[feature] = df[feature].mean()
                remaining = set(horizon_model.required_features) - set(latest_data.columns)
                if remaining:
                    warnings.warn(f"Skipping {ticker} - missing: {remaining}")
                    continue

            # Generate predictions
            try:
                if hasattr(horizon_model.model, "predict_proba"):
                    proba = horizon_model.model.predict_proba(latest_data[horizon_model.required_features])[0][1]
                else:
                    proba = 0.5
                prediction = horizon_model.model.predict(latest_data[horizon_model.required_features])[0]
            except Exception as e:
                warnings.warn(f"Prediction failed for {ticker}: {str(e)}")
                continue
                
            risk = calculate_risk_score(df)
            
            pred_dict = {
                'ticker': ticker,
                'predicted_return': prediction,
                'success_prob': proba,
                'risk_score': risk,
                'model_type': model_type,
                'horizon': investment_horizon
            }
            
            # Add uncertainty estimates if available
            if hasattr(horizon_model, 'uncertainty_estimates') and horizon_model.uncertainty_estimates:
                try:
                    pred_dict.update({
                        'prob_std': horizon_model.uncertainty_estimates['std'][0],
                        'prob_range': (horizon_model.uncertainty_estimates['min'][0],
                                      horizon_model.uncertainty_estimates['max'][0])
                    })
                except:
                    pass
            
            predictions.append(pred_dict)
        except Exception as e:
            warnings.warn(f"Prediction failed for {ticker}: {str(e)}")
            continue
            
    return pd.DataFrame(predictions) if predictions else pd.DataFrame()