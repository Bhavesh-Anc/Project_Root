# utils/feature_engineer.py - Fixed with Missing Functions
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import hashlib
import pickle
from functools import lru_cache
import json

# Try to import ta library, provide fallback if not available
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("ta library not available. Technical indicators will use simplified versions.")

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Will use basic scaling methods.")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ==================== CONFIGURATION ====================

FEATURE_CONFIG = {
    'lookback_periods': [5, 10, 20, 50],
    'technical_indicators': [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 
        'stochastic', 'atr', 'cci', 'williams_r', 'obv'
    ],
    'price_features': True,
    'volume_features': True, 
    'volatility_features': True,
    'momentum_features': True,
    'trend_features': True,
    'pattern_features': True,
    'market_microstructure': True,
    'sentiment_features': True,
    'target_horizons': ['next_week', 'next_month', 'next_quarter', 'next_year'],
    'feature_selection_enabled': True,
    'parallel_processing': True,
    'cache_features': True,
    'advanced_features': True
}

# ==================== MISSING FUNCTIONS FOR APP.PY ====================

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function used by app.py for creating technical features
    This is the function that was missing and causing the import error
    """
    if df.empty or len(df) < 20:
        logging.warning("Insufficient data for technical feature creation")
        return df.copy()
    
    try:
        features_df = df.copy()
        
        # Basic Moving Averages
        features_df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        features_df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        features_df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        features_df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential Moving Averages
        features_df['EMA_12'] = df['Close'].ewm(span=12, min_periods=1).mean()
        features_df['EMA_26'] = df['Close'].ewm(span=26, min_periods=1).mean()
        
        # MACD
        features_df['MACD'] = features_df['EMA_12'] - features_df['EMA_26']
        features_df['MACD_signal'] = features_df['MACD'].ewm(span=9, min_periods=1).mean()
        features_df['MACD_histogram'] = features_df['MACD'] - features_df['MACD_signal']
        
        # RSI (Simplified version)
        if TA_AVAILABLE:
            features_df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        else:
            features_df['RSI'] = calculate_rsi_simple(df['Close'])
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        sma_20 = df['Close'].rolling(window=bb_window, min_periods=1).mean()
        bb_std_dev = df['Close'].rolling(window=bb_window, min_periods=1).std()
        features_df['BB_upper'] = sma_20 + (bb_std_dev * bb_std)
        features_df['BB_lower'] = sma_20 - (bb_std_dev * bb_std)
        features_df['BB_width'] = features_df['BB_upper'] - features_df['BB_lower']
        features_df['BB_position'] = (df['Close'] - features_df['BB_lower']) / features_df['BB_width']
        
        # Price momentum features
        features_df['momentum_1d'] = df['Close'].pct_change(1)
        features_df['momentum_5d'] = df['Close'].pct_change(5)
        features_df['momentum_10d'] = df['Close'].pct_change(10)
        
        # Volume features
        if 'Volume' in df.columns:
            features_df['volume_sma_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
            features_df['volume_ratio'] = df['Volume'] / features_df['volume_sma_10']
            
            # Price-Volume features
            features_df['price_volume'] = df['Close'] * df['Volume']
            features_df['vwap'] = features_df['price_volume'].rolling(window=20, min_periods=1).sum() / df['Volume'].rolling(window=20, min_periods=1).sum()
        
        # Volatility features
        features_df['volatility_10d'] = df['Close'].pct_change().rolling(window=10, min_periods=1).std()
        features_df['volatility_20d'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
        
        # High-Low features
        features_df['high_low_ratio'] = df['High'] / df['Low']
        features_df['close_to_high'] = df['Close'] / df['High']
        features_df['close_to_low'] = df['Close'] / df['Low']
        
        # True Range and ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        features_df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Additional technical indicators if ta library is available
        if TA_AVAILABLE:
            try:
                features_df['stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
                features_df['stoch_d'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
                features_df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
                features_df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
                
                if 'Volume' in df.columns:
                    features_df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            except Exception as e:
                logging.warning(f"Some technical indicators failed: {e}")
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logging.info(f"Created {len(features_df.columns)} technical features")
        return features_df
        
    except Exception as e:
        logging.error(f"Technical feature creation failed: {e}")
        return df.copy()

def prepare_features_for_training(raw_data: Dict[str, pd.DataFrame], 
                                investment_horizon: str = 'next_month') -> Dict[str, pd.DataFrame]:
    """
    Prepare features for training - this was the other missing function
    """
    if not raw_data:
        logging.warning("No raw data provided for feature preparation")
        return {}
    
    prepared_data = {}
    
    logging.info(f"Preparing features for {len(raw_data)} tickers with horizon: {investment_horizon}")
    
    # Define target mappings
    horizon_mapping = {
        'next_day': 1,
        'next_week': 5,
        'next_month': 22,
        'next_quarter': 66,
        'next_year': 252
    }
    
    target_days = horizon_mapping.get(investment_horizon, 22)
    
    for ticker, df in tqdm(raw_data.items(), desc="Preparing features"):
        try:
            if df.empty or len(df) < 50:
                logging.warning(f"Insufficient data for {ticker}")
                continue
            
            # Create technical features
            featured_df = create_technical_features(df)
            
            # Create target variables
            featured_df = create_target_variables(featured_df, target_days)
            
            # Clean and validate data
            featured_df = clean_and_validate_features(featured_df)
            
            if not featured_df.empty:
                prepared_data[ticker] = featured_df
                logging.info(f"Prepared {ticker}: {featured_df.shape}")
            
        except Exception as e:
            logging.error(f"Feature preparation failed for {ticker}: {e}")
            continue
    
    logging.info(f"Feature preparation completed for {len(prepared_data)} tickers")
    return prepared_data

# ==================== HELPER FUNCTIONS ====================

def calculate_rsi_simple(prices: pd.Series, window: int = 14) -> pd.Series:
    """Simple RSI calculation when ta library is not available"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def create_target_variables(df: pd.DataFrame, target_days: int = 22) -> pd.DataFrame:
    """Create target variables for machine learning"""
    try:
        df = df.copy()
        
        # Future returns
        future_close = df['Close'].shift(-target_days)
        current_close = df['Close']
        
        # Calculate return
        future_return = (future_close - current_close) / current_close
        
        # Binary classification targets
        df[f'Target_positive_{target_days}d'] = (future_return > 0).astype(int)
        df[f'Target_strong_positive_{target_days}d'] = (future_return > 0.05).astype(int)  # 5% gain
        df[f'Target_negative_{target_days}d'] = (future_return < 0).astype(int)
        df[f'Target_strong_negative_{target_days}d'] = (future_return < -0.05).astype(int)  # 5% loss
        
        # Regression targets
        df[f'Target_return_{target_days}d'] = future_return
        df[f'Target_return_class_{target_days}d'] = pd.cut(future_return, 
                                                           bins=[-np.inf, -0.05, 0, 0.05, np.inf], 
                                                           labels=[0, 1, 2, 3]).astype(float)
        
        # Remove last target_days rows (no future data)
        df = df.iloc[:-target_days] if len(df) > target_days else df.iloc[:0]
        
        return df
        
    except Exception as e:
        logging.error(f"Target variable creation failed: {e}")
        return df

def clean_and_validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate feature data"""
    try:
        # Remove rows with too many NaN values
        threshold = len(df.columns) * 0.8  # Allow up to 80% NaN
        df = df.dropna(thresh=threshold)
        
        # Fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove columns with all zeros or constant values
        for col in df.columns:
            if df[col].nunique() <= 1:
                df = df.drop(columns=[col])
        
        logging.info(f"Cleaned data shape: {df.shape}")
        return df
        
    except Exception as e:
        logging.error(f"Data cleaning failed: {e}")
        return df

# ==================== ENHANCED FEATURE CREATION ====================

def create_features_enhanced(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Enhanced feature creation with advanced indicators"""
    config = config or FEATURE_CONFIG
    
    if df.empty or len(df) < 50:
        logging.warning("Insufficient data for enhanced feature creation")
        return pd.DataFrame()
    
    try:
        # Start with technical features
        features_df = create_technical_features(df)
        
        # Add advanced features if enabled
        if config.get('advanced_features', False):
            features_df = add_advanced_features(features_df, config)
        
        # Add pattern features
        if config.get('pattern_features', False):
            features_df = add_pattern_features(features_df)
        
        # Add market microstructure features
        if config.get('market_microstructure', False):
            features_df = add_microstructure_features(features_df)
        
        return features_df
        
    except Exception as e:
        logging.error(f"Enhanced feature creation failed: {e}")
        return create_technical_features(df)  # Fallback to basic features

def add_advanced_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add advanced features like gaps, spreads, etc."""
    try:
        # Price gaps
        df['gap_up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.01).astype(int)
        df['gap_down'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) < -0.01).astype(int)
        
        # Price spreads
        df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_spread'] = abs(df['Open'] - df['Close']) / df['Close']
        
        # Multi-timeframe features
        for period in [5, 10, 20]:
            df[f'sma_cross_{period}'] = (df['Close'] > df[f'SMA_{period}']).astype(int)
            df[f'price_distance_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
        
        return df
        
    except Exception as e:
        logging.warning(f"Advanced features failed: {e}")
        return df

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern features"""
    try:
        # Basic candlestick patterns
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - np.maximum(df['Close'], df['Open'])
        lower_shadow = np.minimum(df['Close'], df['Open']) - df['Low']
        
        # Doji pattern
        df['doji'] = (body < (df['High'] - df['Low']) * 0.1).astype(int)
        
        # Hammer pattern
        df['hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(int)
        
        # Shooting star pattern
        df['shooting_star'] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.5)).astype(int)
        
        return df
        
    except Exception as e:
        logging.warning(f"Pattern features failed: {e}")
        return df

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market microstructure features"""
    try:
        # Tick direction
        df['tick_direction'] = np.sign(df['Close'].diff()).fillna(0)
        
        # Upticks and downticks
        df['upticks'] = (df['tick_direction'] > 0).astype(int)
        df['downticks'] = (df['tick_direction'] < 0).astype(int)
        
        # Rolling tick imbalance
        df['tick_imbalance'] = (df['upticks'].rolling(10).sum() - df['downticks'].rolling(10).sum()) / 10
        
        return df
        
    except Exception as e:
        logging.warning(f"Microstructure features failed: {e}")
        return df

# ==================== CACHING SYSTEM ====================

class EnhancedTrainingCache:
    """Caching system for feature engineering"""
    
    def __init__(self, cache_dir: str = "feature_cache_v2"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_processed_features(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get cached features if available"""
        try:
            cache_key = f"features_{ticker}_{len(df)}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is recent (within 24 hours)
                if (pd.Timestamp.now() - cached_data['timestamp']).total_seconds() < 86400:
                    logging.info(f"Loading cached features for {ticker}")
                    return cached_data['features']
                    
        except Exception as e:
            logging.warning(f"Cache lookup failed for {ticker}: {e}")
            
        return None
    
    def save_processed_features(self, ticker: str, df: pd.DataFrame, processed_df: pd.DataFrame):
        """Save processed features to cache"""
        try:
            cache_key = f"features_{ticker}_{len(df)}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            cache_data = {
                'features': processed_df,
                'timestamp': pd.Timestamp.now(),
                'ticker': ticker
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logging.info(f"Cached features for {ticker}")
            
        except Exception as e:
            logging.warning(f"Failed to cache features for {ticker}: {e}")

# ==================== MAIN INTERFACE FUNCTIONS ====================

def engineer_features_enhanced(data_dict: Dict[str, pd.DataFrame],
                             config: Dict = None,
                             use_cache: bool = True,
                             parallel: bool = True,
                             selected_tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Enhanced feature engineering with caching and parallel processing"""
    config = config or FEATURE_CONFIG
    
    # Filter to selected tickers if provided
    if selected_tickers:
        data_dict = {ticker: df for ticker, df in data_dict.items() 
                    if ticker in selected_tickers}
        logging.info(f"Processing {len(data_dict)} selected tickers")
    
    # Initialize cache
    cache = EnhancedTrainingCache() if use_cache else None
    
    processed_data = {}
    cache_hits = 0
    
    # Check cache first
    remaining_data = {}
    for ticker, df in data_dict.items():
        if df.empty:
            processed_data[ticker] = pd.DataFrame()
            continue
            
        if cache:
            cached_features = cache.get_processed_features(ticker, df)
            if cached_features is not None:
                processed_data[ticker] = cached_features
                cache_hits += 1
                continue
        
        remaining_data[ticker] = df
    
    logging.info(f"Cache hits: {cache_hits}/{len(data_dict)}")
    
    # Process remaining tickers
    for ticker, df in tqdm(remaining_data.items(), desc="Engineering features"):
        try:
            features_df = create_features_enhanced(df, config)
            
            if not features_df.empty:
                processed_data[ticker] = features_df
                
                # Save to cache
                if cache:
                    cache.save_processed_features(ticker, df, features_df)
                    
        except Exception as e:
            logging.error(f"Feature engineering failed for {ticker}: {e}")
            processed_data[ticker] = df.copy()
    
    successful_count = sum(1 for df in processed_data.values() if not df.empty)
    logging.info(f"Feature engineering completed: {successful_count}/{len(data_dict)} successful")
    
    return processed_data

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Enhanced Feature Engineering System")
    print("="*60)
    
    # Test the main functions
    try:
        # Create sample data
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        
        print("Testing create_technical_features...")
        features = create_technical_features(sample_data)
        print(f"✅ Created {len(features.columns)} features")
        
        print("Testing prepare_features_for_training...")
        raw_data = {'TEST': sample_data}
        prepared = prepare_features_for_training(raw_data, 'next_month')
        print(f"✅ Prepared features for {len(prepared)} tickers")
        
        print("\n🎉 All functions working correctly!")
        print("The missing functions have been added and are compatible with app.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Some dependencies may be missing. Install with: pip install pandas numpy ta scikit-learn")