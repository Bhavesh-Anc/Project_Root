# utils/feature_engineer.py - Complete Feature Engineering System
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import pickle
import hashlib
from functools import partial

# Enhanced import handling for technical analysis
try:
    import ta
    TA_AVAILABLE = True
    logging.info("Technical Analysis library (ta) loaded successfully")
except ImportError:
    TA_AVAILABLE = False
    logging.warning("Technical Analysis library not available. Using simplified versions. " +
                   "Technical indicators will use simplified versions.")

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.decomposition import PCA
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

# ==================== CORE FEATURE ENGINEERING FUNCTIONS ====================

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical features - Main function used by app.py
    This is the critical missing function that was causing import errors
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
            features_df['RSI'] = calculate_rsi_simple(df['Close'], window=14)
        
        # Bollinger Bands
        if TA_AVAILABLE:
            bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
            features_df['BB_upper'] = bollinger.bollinger_hband()
            features_df['BB_lower'] = bollinger.bollinger_lband()
            features_df['BB_middle'] = bollinger.bollinger_mavg()
            features_df['BB_width'] = features_df['BB_upper'] - features_df['BB_lower']
            features_df['BB_position'] = (df['Close'] - features_df['BB_lower']) / features_df['BB_width']
        else:
            bb_data = calculate_bollinger_bands_simple(df['Close'])
            features_df = pd.concat([features_df, bb_data], axis=1)
        
        # Price-based features
        features_df['price_change'] = df['Close'].pct_change()
        features_df['price_change_5d'] = df['Close'].pct_change(periods=5)
        features_df['price_change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility features
        features_df['volatility_5d'] = features_df['price_change'].rolling(5).std()
        features_df['volatility_20d'] = features_df['price_change'].rolling(20).std()
        
        # High-Low features
        features_df['high_low_pct'] = (df['High'] - df['Low']) / df['Close']
        features_df['close_to_high'] = (df['High'] - df['Close']) / df['Close']
        features_df['close_to_low'] = (df['Close'] - df['Low']) / df['Close']
        
        # Volume features (if available)
        if 'Volume' in df.columns:
            features_df['volume_ma_5'] = df['Volume'].rolling(5).mean()
            features_df['volume_ma_20'] = df['Volume'].rolling(20).mean()
            features_df['volume_ratio'] = df['Volume'] / features_df['volume_ma_20']
            features_df['volume_price_trend'] = features_df['volume_ratio'] * features_df['price_change']
            
            # On Balance Volume (simplified)
            features_df['OBV'] = calculate_obv_simple(df['Close'], df['Volume'])
        
        # Average True Range (ATR)
        if TA_AVAILABLE:
            features_df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        else:
            features_df['ATR'] = calculate_atr_simple(df['High'], df['Low'], df['Close'])
        
        # Stochastic Oscillator
        if TA_AVAILABLE:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            features_df['Stoch_K'] = stoch.stoch()
            features_df['Stoch_D'] = stoch.stoch_signal()
        else:
            stoch_data = calculate_stochastic_simple(df['High'], df['Low'], df['Close'])
            features_df = pd.concat([features_df, stoch_data], axis=1)
        
        # Williams %R
        if TA_AVAILABLE:
            features_df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        else:
            features_df['Williams_R'] = calculate_williams_r_simple(df['High'], df['Low'], df['Close'])
        
        # Commodity Channel Index (CCI)
        if TA_AVAILABLE:
            features_df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        else:
            features_df['CCI'] = calculate_cci_simple(df['High'], df['Low'], df['Close'])
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logging.info(f"Created {len(features_df.columns) - len(df.columns)} technical features")
        
        return features_df
        
    except Exception as e:
        logging.error(f"Technical feature creation failed: {e}")
        return df.copy()

def engineer_features_enhanced(data_dict: Dict[str, pd.DataFrame], 
                             config: Dict = None, 
                             use_cache: bool = True,
                             parallel: bool = False,
                             selected_tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Enhanced feature engineering for selected tickers with caching and parallel processing
    Main function called by app.py
    """
    config = config or FEATURE_CONFIG
    
    if not data_dict:
        return {}
    
    # Filter to selected tickers if provided
    if selected_tickers:
        filtered_data = {ticker: df for ticker, df in data_dict.items() 
                        if ticker in selected_tickers}
    else:
        filtered_data = data_dict
    
    logging.info(f"Engineering features for {len(filtered_data)} stocks")
    
    enhanced_data = {}
    
    try:
        if parallel and len(filtered_data) > 1:
            # Parallel processing
            enhanced_data = _engineer_features_parallel(filtered_data, config, use_cache)
        else:
            # Sequential processing
            for ticker, df in filtered_data.items():
                if df.empty:
                    enhanced_data[ticker] = df
                    continue
                
                try:
                    # Check cache first
                    if use_cache:
                        cached_features = _load_cached_features(ticker, df, config)
                        if cached_features is not None:
                            enhanced_data[ticker] = cached_features
                            logging.info(f"Loaded cached features for {ticker}")
                            continue
                    
                    # Create features
                    features_df = _create_comprehensive_features(df, config)
                    enhanced_data[ticker] = features_df
                    
                    # Cache features
                    if use_cache:
                        _cache_features(ticker, df, features_df, config)
                    
                    logging.info(f"Created {len(features_df.columns)} features for {ticker}")
                    
                except Exception as e:
                    logging.warning(f"Feature engineering failed for {ticker}: {e}")
                    enhanced_data[ticker] = df
                    continue
        
        logging.info(f"Feature engineering completed for {len(enhanced_data)} stocks")
        
        return enhanced_data
        
    except Exception as e:
        logging.error(f"Enhanced feature engineering failed: {e}")
        # Return original data on failure
        return data_dict

def _create_comprehensive_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Create comprehensive feature set"""
    
    if df.empty or len(df) < 20:
        return df.copy()
    
    features_df = df.copy()
    
    try:
        # 1. Basic Technical Features (Always included)
        features_df = create_technical_features(features_df)
        
        # 2. Price Features
        if config.get('price_features', True):
            features_df = _add_price_features(features_df, config)
        
        # 3. Volume Features
        if config.get('volume_features', True) and 'Volume' in df.columns:
            features_df = _add_volume_features(features_df, config)
        
        # 4. Volatility Features
        if config.get('volatility_features', True):
            features_df = _add_volatility_features(features_df, config)
        
        # 5. Momentum Features
        if config.get('momentum_features', True):
            features_df = _add_momentum_features(features_df, config)
        
        # 6. Trend Features
        if config.get('trend_features', True):
            features_df = _add_trend_features(features_df, config)
        
        # 7. Pattern Features
        if config.get('pattern_features', True):
            features_df = _add_pattern_features(features_df, config)
        
        # 8. Market Microstructure Features
        if config.get('market_microstructure', True):
            features_df = _add_microstructure_features(features_df, config)
        
        # 9. Advanced Features
        if config.get('advanced_features', True):
            features_df = _add_advanced_features(features_df, config)
        
        # Clean up features
        features_df = _clean_features(features_df)
        
        return features_df
        
    except Exception as e:
        logging.error(f"Comprehensive feature creation failed: {e}")
        return df.copy()

# ==================== SIMPLIFIED TECHNICAL INDICATORS ====================

def calculate_rsi_simple(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using simple method"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
    except Exception:
        return pd.Series(50, index=prices.index)

def calculate_bollinger_bands_simple(prices: pd.Series, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands using simple method"""
    try:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = upper - lower
        position = (prices - lower) / width
        
        return pd.DataFrame({
            'BB_upper': upper,
            'BB_lower': lower,
            'BB_middle': sma,
            'BB_width': width,
            'BB_position': position
        })
    except Exception:
        return pd.DataFrame({
            'BB_upper': prices,
            'BB_lower': prices,
            'BB_middle': prices,
            'BB_width': 0,
            'BB_position': 0.5
        })

def calculate_atr_simple(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate ATR using simple method"""
    try:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr.fillna(0)
    except Exception:
        return pd.Series(0, index=high.index)

def calculate_stochastic_simple(high: pd.Series, low: pd.Series, close: pd.Series, 
                               k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator using simple method"""
    try:
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent.fillna(50),
            'Stoch_D': d_percent.fillna(50)
        })
    except Exception:
        return pd.DataFrame({
            'Stoch_K': pd.Series(50, index=high.index),
            'Stoch_D': pd.Series(50, index=high.index)
        })

def calculate_williams_r_simple(high: pd.Series, low: pd.Series, close: pd.Series, 
                               window: int = 14) -> pd.Series:
    """Calculate Williams %R using simple method"""
    try:
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r.fillna(-50)
    except Exception:
        return pd.Series(-50, index=high.index)

def calculate_cci_simple(high: pd.Series, low: pd.Series, close: pd.Series, 
                        window: int = 20) -> pd.Series:
    """Calculate CCI using simple method"""
    try:
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return cci.fillna(0)
    except Exception:
        return pd.Series(0, index=high.index)

def calculate_obv_simple(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume using simple method"""
    try:
        price_change = close.diff()
        obv = volume.copy()
        
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        
        return obv.cumsum().fillna(0)
    except Exception:
        return pd.Series(0, index=close.index)

# ==================== FEATURE CATEGORY FUNCTIONS ====================

def _add_price_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add price-based features"""
    try:
        # Multiple timeframe returns
        for period in config.get('lookback_periods', [5, 10, 20]):
            df[f'return_{period}d'] = df['Close'].pct_change(periods=period)
            df[f'log_return_{period}d'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Price ratios
        df['open_close_ratio'] = df['Open'] / df['Close']
        df['high_close_ratio'] = df['High'] / df['Close']
        df['low_close_ratio'] = df['Low'] / df['Close']
        
        # Gap features
        df['gap'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
        df['gap_filled'] = (df['gap'] > 0) & (df['Low'] <= df['Close'].shift())
        
        return df
    except Exception as e:
        logging.warning(f"Price feature creation failed: {e}")
        return df

def _add_volume_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add volume-based features"""
    try:
        # Volume moving averages
        for period in config.get('lookback_periods', [5, 10, 20]):
            df[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']
        
        # Volume-price features
        df['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
        df['volume_weighted_price'] = (df['Volume'] * df['Close']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # Price Volume Trend (PVT)
        df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        return df
    except Exception as e:
        logging.warning(f"Volume feature creation failed: {e}")
        return df

def _add_volatility_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add volatility-based features"""
    try:
        returns = df['Close'].pct_change()
        
        # Rolling volatilities
        for period in config.get('lookback_periods', [5, 10, 20]):
            df[f'volatility_{period}d'] = returns.rolling(period).std()
            df[f'volatility_ann_{period}d'] = df[f'volatility_{period}d'] * np.sqrt(252)
        
        # GARCH-like features
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
        df['volatility_rank'] = df['volatility_20d'].rolling(252).rank(pct=True)
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['High'] / df['Low']) ** 2
        )
        
        return df
    except Exception as e:
        logging.warning(f"Volatility feature creation failed: {e}")
        return df

def _add_momentum_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add momentum-based features"""
    try:
        # Rate of Change (ROC)
        for period in config.get('lookback_periods', [10, 20]):
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Momentum
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_20'] = df['Close'] - df['Close'].shift(20)
        
        # Price acceleration
        df['price_acceleration'] = df['Close'].pct_change().diff()
        
        return df
    except Exception as e:
        logging.warning(f"Momentum feature creation failed: {e}")
        return df

def _add_trend_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add trend-based features"""
    try:
        # Moving average trends
        df['MA_trend_5_20'] = df['SMA_5'] > df['SMA_20']
        df['MA_trend_10_50'] = df['SMA_10'] > df['SMA_50']
        
        # ADX (simplified)
        df['ADX'] = calculate_adx_simple(df['High'], df['Low'], df['Close'])
        
        # Trend strength
        df['trend_strength'] = abs(df['Close'] - df['SMA_20']) / df['SMA_20']
        
        return df
    except Exception as e:
        logging.warning(f"Trend feature creation failed: {e}")
        return df

def _add_pattern_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add pattern recognition features"""
    try:
        # Doji patterns
        body_size = abs(df['Close'] - df['Open'])
        range_size = df['High'] - df['Low']
        df['doji'] = body_size < (range_size * 0.1)
        
        # Hammer patterns
        lower_shadow = df['Open'].combine(df['Close'], min) - df['Low']
        upper_shadow = df['High'] - df['Open'].combine(df['Close'], max)
        df['hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
        
        # Inside/Outside bars
        df['inside_bar'] = (df['High'] < df['High'].shift()) & (df['Low'] > df['Low'].shift())
        df['outside_bar'] = (df['High'] > df['High'].shift()) & (df['Low'] < df['Low'].shift())
        
        return df
    except Exception as e:
        logging.warning(f"Pattern feature creation failed: {e}")
        return df

def _add_microstructure_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add market microstructure features"""
    try:
        # Spread proxies
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['oc_spread'] = abs(df['Open'] - df['Close']) / df['Close']
        
        # Intraday returns
        df['overnight_return'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
        df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        
        # Price efficiency
        returns = df['Close'].pct_change()
        df['return_autocorr'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
        
        return df
    except Exception as e:
        logging.warning(f"Microstructure feature creation failed: {e}")
        return df

def _add_advanced_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Add advanced mathematical features"""
    try:
        # Fractal dimension
        df['fractal_dim'] = calculate_fractal_dimension_simple(df['Close'])
        
        # Hurst exponent (simplified)
        df['hurst_exp'] = calculate_hurst_simple(df['Close'])
        
        # Entropy
        returns = df['Close'].pct_change().dropna()
        df['entropy'] = returns.rolling(50).apply(calculate_entropy_simple)
        
        return df
    except Exception as e:
        logging.warning(f"Advanced feature creation failed: {e}")
        return df

# ==================== ADVANCED CALCULATION FUNCTIONS ====================

def calculate_adx_simple(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate ADX using simplified method"""
    try:
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        
        # Smooth the values
        tr_smooth = tr.rolling(window).mean()
        dm_plus_smooth = dm_plus.rolling(window).mean()
        dm_minus_smooth = dm_minus.rolling(window).mean()
        
        # Calculate DI
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window).mean()
        
        return adx.fillna(25)
    except Exception:
        return pd.Series(25, index=high.index)

def calculate_fractal_dimension_simple(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate simplified fractal dimension"""
    try:
        def _fractal_dim(series):
            if len(series) < 2:
                return 1.5
            
            # Simplified calculation
            returns = series.pct_change().dropna()
            if len(returns) == 0:
                return 1.5
            
            # Approximate using volatility
            vol = returns.std()
            return 1.5 + 0.5 * np.tanh(vol * 100 - 1)
        
        fractal_dim = prices.rolling(window).apply(_fractal_dim)
        return fractal_dim.fillna(1.5)
    except Exception:
        return pd.Series(1.5, index=prices.index)

def calculate_hurst_simple(prices: pd.Series, window: int = 50) -> pd.Series:
    """Calculate simplified Hurst exponent"""
    try:
        def _hurst_exp(series):
            if len(series) < 10:
                return 0.5
            
            # Simplified R/S analysis
            returns = series.pct_change().dropna()
            if len(returns) < 2:
                return 0.5
            
            # Approximate using autocorrelation
            autocorr = returns.autocorr(lag=1)
            if np.isnan(autocorr):
                return 0.5
            
            return 0.5 + 0.3 * autocorr
        
        hurst = prices.rolling(window).apply(_hurst_exp)
        return hurst.fillna(0.5)
    except Exception:
        return pd.Series(0.5, index=prices.index)

def calculate_entropy_simple(returns: pd.Series) -> float:
    """Calculate simplified entropy"""
    try:
        if len(returns) < 5:
            return 0.0
        
        # Discretize returns into bins
        bins = np.histogram_bin_edges(returns, bins=10)
        hist, _ = np.histogram(returns, bins=bins)
        
        # Calculate probabilities
        probs = hist / len(returns)
        probs = probs[probs > 0]  # Remove zero probabilities
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    except Exception:
        return 0.0

# ==================== CACHING AND PARALLEL PROCESSING ====================

def _generate_cache_key(ticker: str, df: pd.DataFrame, config: Dict) -> str:
    """Generate unique cache key"""
    try:
        # Use DataFrame hash and config hash
        df_hash = hashlib.md5(str(df.shape).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8]
        return f"{ticker}_{df_hash}_{config_hash}"
    except:
        return f"{ticker}_default"

def _load_cached_features(ticker: str, df: pd.DataFrame, config: Dict) -> Optional[pd.DataFrame]:
    """Load cached features if available"""
    try:
        cache_dir = config.get('feature_cache_dir', 'feature_cache_v2')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = _generate_cache_key(ticker, df, config)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            # Check if cache is recent (within 24 hours)
            cache_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - cache_time < 86400:  # 24 hours
                with open(cache_file, 'rb') as f:
                    cached_df = pickle.load(f)
                
                # Verify cache integrity
                if len(cached_df) == len(df) and 'Close' in cached_df.columns:
                    return cached_df
        
        return None
    except Exception as e:
        logging.warning(f"Cache loading failed for {ticker}: {e}")
        return None

def _cache_features(ticker: str, original_df: pd.DataFrame, features_df: pd.DataFrame, config: Dict):
    """Cache engineered features"""
    try:
        cache_dir = config.get('feature_cache_dir', 'feature_cache_v2')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = _generate_cache_key(ticker, original_df, config)
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(features_df, f)
        
        logging.debug(f"Cached features for {ticker}")
    except Exception as e:
        logging.warning(f"Cache saving failed for {ticker}: {e}")

def _engineer_features_parallel(data_dict: Dict[str, pd.DataFrame], 
                               config: Dict, use_cache: bool) -> Dict[str, pd.DataFrame]:
    """Engineer features in parallel"""
    try:
        def process_ticker(ticker_data):
            ticker, df = ticker_data
            if df.empty:
                return ticker, df
            
            try:
                # Check cache
                if use_cache:
                    cached_features = _load_cached_features(ticker, df, config)
                    if cached_features is not None:
                        return ticker, cached_features
                
                # Create features
                features_df = _create_comprehensive_features(df, config)
                
                # Cache features
                if use_cache:
                    _cache_features(ticker, df, features_df, config)
                
                return ticker, features_df
            except Exception as e:
                logging.warning(f"Parallel feature engineering failed for {ticker}: {e}")
                return ticker, df
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_ticker, data_dict.items()))
        
        return dict(results)
        
    except Exception as e:
        logging.error(f"Parallel feature engineering failed: {e}")
        # Fall back to sequential processing
        return {ticker: _create_comprehensive_features(df, config) 
                for ticker, df in data_dict.items()}

def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate features"""
    try:
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            logging.info(f"Removed {len(constant_cols)} constant columns")
        
        return df
    except Exception as e:
        logging.warning(f"Feature cleaning failed: {e}")
        return df

# ==================== FEATURE SELECTION ====================

def select_features(df: pd.DataFrame, target_col: str = None, method: str = 'variance', 
                   k: int = 50) -> pd.DataFrame:
    """Select most important features"""
    
    if not SKLEARN_AVAILABLE or target_col is None or target_col not in df.columns:
        return df
    
    try:
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Remove NaN and infinite values
        mask = ~(X.isnull().any(axis=1) | y.isnull() | np.isinf(X).any(axis=1))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 50:
            return df
        
        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
            X_selected = selector.fit_transform(X_clean, y_clean)
            
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
        elif method == 'pca':
            # PCA feature reduction
            pca = PCA(n_components=min(k, len(feature_cols)))
            X_selected = pca.fit_transform(X_clean)
            
            # Create new column names
            selected_features = [f'PC_{i+1}' for i in range(X_selected.shape[1])]
            
        else:
            # Variance-based selection
            variances = X_clean.var()
            selected_features = variances.nlargest(min(k, len(feature_cols))).index.tolist()
        
        # Return dataframe with selected features
        result_df = df[[target_col] + selected_features].copy()
        
        logging.info(f"Selected {len(selected_features)} features using {method} method")
        
        return result_df
        
    except Exception as e:
        logging.warning(f"Feature selection failed: {e}")
        return df

# ==================== EXPORT AND MAIN EXECUTION ====================

# Export main functions
__all__ = [
    'create_technical_features',
    'engineer_features_enhanced', 
    'FEATURE_CONFIG',
    'select_features'
]

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Feature Engineering System - User Selection Version")
    print("="*60)
    
    # Test with mock data
    def create_test_data():
        dates = pd.date_range('2022-01-01', periods=1000, freq='D')
        
        # Generate realistic stock data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        volume = np.random.lognormal(15, 0.5, len(dates))
        
        df = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.01, len(dates)),
            'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
            'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
            'Close': prices,
            'Volume': volume
        }, index=dates)
        
        return df
    
    # Test feature engineering
    test_df = create_test_data()
    print(f"Original data shape: {test_df.shape}")
    
    # Test basic technical features
    tech_features = create_technical_features(test_df)
    print(f"Technical features shape: {tech_features.shape}")
    print(f"Added {tech_features.shape[1] - test_df.shape[1]} technical features")
    
    # Test enhanced feature engineering
    test_data = {'TEST_STOCK': test_df}
    selected_tickers = ['TEST_STOCK']
    
    enhanced_features = engineer_features_enhanced(
        test_data, 
        config=FEATURE_CONFIG,
        selected_tickers=selected_tickers,
        use_cache=False,
        parallel=False
    )
    
    if 'TEST_STOCK' in enhanced_features:
        enhanced_df = enhanced_features['TEST_STOCK']
        print(f"Enhanced features shape: {enhanced_df.shape}")
        print(f"Total features created: {enhanced_df.shape[1] - test_df.shape[1]}")
        
        # Show feature categories
        feature_names = enhanced_df.columns.tolist()
        print(f"\nFeature categories created:")
        print(f"  - SMA features: {len([f for f in feature_names if 'SMA' in f])}")
        print(f"  - EMA features: {len([f for f in feature_names if 'EMA' in f])}")
        print(f"  - RSI features: {len([f for f in feature_names if 'RSI' in f])}")
        print(f"  - Bollinger features: {len([f for f in feature_names if 'BB_' in f])}")
        print(f"  - Volume features: {len([f for f in feature_names if 'volume' in f.lower()])}")
        print(f"  - Volatility features: {len([f for f in feature_names if 'volatility' in f.lower()])}")
    
    print(f"\nUser Selection Features:")
    print(f"  ✓ Optimized for selected tickers only")
    print(f"  ✓ Caching for improved performance")
    print(f"  ✓ Parallel processing support")
    print(f"  ✓ Comprehensive technical indicators")
    print(f"  ✓ Advanced mathematical features")
    print(f"  ✓ Robust error handling")
    
    print(f"\nFeature Engineering System Test Completed!")