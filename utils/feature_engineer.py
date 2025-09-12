# utils/feature_engineer_fixed.py
import pandas as pd
import numpy as np
import ta
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import hashlib
import pickle
from functools import lru_cache
import json

warnings.filterwarnings('ignore')

# ==================== FIXED CONFIGURATION ====================

FEATURE_CONFIG = {
    'lookback_periods': [5, 10, 20, 50],  # Restored full periods
    'technical_indicators': [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 
        'stochastic', 'atr', 'cci', 'williams_r', 'obv'
    ],
    'price_features': True,
    'volume_features': True, 
    'volatility_features': True,
    'momentum_features': True,
    'trend_features': True,
    'pattern_features': True,    # Re-enabled
    'market_microstructure': True,  # Re-enabled  
    'sentiment_features': True,  # New
    'target_horizons': ['next_week', 'next_month', 'next_quarter', 'next_year'],
    'feature_selection_enabled': True,
    'parallel_processing': True,
    'cache_features': True,
    'advanced_features': True    # New advanced feature flag
}

# ==================== FIXED CACHING SYSTEM ====================

class EnhancedTrainingCache:
    """Fixed caching system that handles pandas Series properly"""
    
    def __init__(self, cache_dir: str = "feature_cache_v2"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _create_dataframe_signature(self, df: pd.DataFrame) -> str:
        """Create a reliable signature for DataFrame caching"""
        try:
            # Create signature from key properties instead of hashing Series directly
            signature_data = {
                'shape': df.shape,
                'columns': list(df.columns),
                'index_start': str(df.index[0]) if len(df) > 0 else '',
                'index_end': str(df.index[-1]) if len(df) > 0 else '',
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'null_counts': df.isnull().sum().to_dict()
            }
            
            # Convert to JSON string and hash
            signature_str = json.dumps(signature_data, sort_keys=True, default=str)
            return hashlib.md5(signature_str.encode()).hexdigest()
            
        except Exception as e:
            logging.warning(f"Failed to create DataFrame signature: {e}")
            # Fallback: use shape and columns only
            fallback_sig = f"{df.shape}_{hash(tuple(df.columns))}"
            return hashlib.md5(str(fallback_sig).encode()).hexdigest()
    
    def get_processed_features(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get cached processed features if available"""
        try:
            data_signature = self._create_dataframe_signature(df)
            cache_key = f"features_{ticker}_{data_signature[:12]}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        
                    # Validate cache age (24 hours)
                    cache_age_hours = (pd.Timestamp.now() - cached_data['timestamp']).total_seconds() / 3600
                    if cache_age_hours < 24:
                        logging.info(f"Loading cached features for {ticker}")
                        return cached_data['features']
                    else:
                        # Remove expired cache
                        os.remove(cache_path)
                        
                except Exception as e:
                    logging.warning(f"Failed to load cached features for {ticker}: {e}")
                    
        except Exception as e:
            logging.warning(f"Cache lookup failed for {ticker}: {e}")
            
        return None
    
    def save_processed_features(self, ticker: str, df: pd.DataFrame, processed_df: pd.DataFrame):
        """Save processed features to cache"""
        try:
            data_signature = self._create_dataframe_signature(df)
            cache_key = f"features_{ticker}_{data_signature[:12]}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            cache_data = {
                'features': processed_df,
                'timestamp': pd.Timestamp.now(),
                'ticker': ticker,
                'original_shape': df.shape,
                'processed_shape': processed_df.shape
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logging.info(f"Cached features for {ticker}: {processed_df.shape}")
            
        except Exception as e:
            logging.warning(f"Failed to cache features for {ticker}: {e}")

# ==================== ENHANCED TECHNICAL INDICATORS ====================

class AdvancedTechnicalIndicators:
    """Enhanced technical indicators with additional features"""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=window).mean()
        avg_loss = loss.ewm(span=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD with optimized calculations"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line, 
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        return {
            'bb_upper': sma + (std * std_dev),
            'bb_middle': sma,
            'bb_lower': sma - (std * std_dev),
            'bb_width': (std * std_dev * 2) / (sma + 1e-10),
            'bb_position': (series - sma) / (std * std_dev + 1e-10)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        range_hl = highest_high - lowest_low
        k_percent = 100 * ((close - lowest_low) / (range_hl + 1e-10))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'stoch_k': k_percent.fillna(50),
            'stoch_d': d_percent.fillna(50)
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=window).mean()
        return atr.fillna(atr.mean())
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        price_change = close.diff()
        volume_direction = np.where(price_change > 0, volume, 
                                  np.where(price_change < 0, -volume, 0))
        obv = pd.Series(volume_direction, index=close.index).cumsum()
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        # Calculate directional movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        atr = tr.ewm(span=window).mean()
        di_plus = 100 * (dm_plus.ewm(span=window).mean() / atr)
        di_minus = 100 * (dm_minus.ewm(span=window).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = dx.ewm(span=window).mean()
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }

# ==================== ENHANCED FEATURE CREATION ====================

def create_advanced_price_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create advanced price-based features"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    indicators = AdvancedTechnicalIndicators()
    
    # Basic price features
    features_df['price_change'] = df['Close'].pct_change()
    features_df['price_change_abs'] = abs(features_df['price_change'])
    features_df['log_return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # Price spreads and ratios
    features_df['hl_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    features_df['oc_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
    features_df['ho_spread'] = (df['High'] - df['Open']) / (df['Open'] + 1e-10)
    features_df['lo_spread'] = (df['Low'] - df['Open']) / (df['Open'] + 1e-10)
    
    # Price position within day's range
    range_hl = df['High'] - df['Low']
    features_df['price_position'] = (df['Close'] - df['Low']) / (range_hl + 1e-10)
    features_df['price_position'] = features_df['price_position'].fillna(0.5)
    
    # Gap analysis
    features_df['gap_up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).fillna(0)
    features_df['gap_up_binary'] = (features_df['gap_up'] > 0.02).astype(int)
    
    # Moving averages and crossovers
    for period in config['lookback_periods']:
        sma = indicators.sma(df['Close'], period)
        ema = indicators.ema(df['Close'], period)
        
        features_df[f'sma_{period}'] = sma
        features_df[f'ema_{period}'] = ema
        features_df[f'price_vs_sma_{period}'] = (df['Close'] / sma - 1).fillna(0)
        features_df[f'price_vs_ema_{period}'] = (df['Close'] / ema - 1).fillna(0)
        
        # Moving average slopes
        features_df[f'sma_slope_{period}'] = (sma - sma.shift(5)) / sma.shift(5)
        features_df[f'ema_slope_{period}'] = (ema - ema.shift(5)) / ema.shift(5)
    
    # Cross-timeframe analysis
    sma_5 = indicators.sma(df['Close'], 5)
    sma_20 = indicators.sma(df['Close'], 20)
    sma_50 = indicators.sma(df['Close'], 50)
    
    features_df['golden_cross'] = ((sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))).astype(int)
    features_df['death_cross'] = ((sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))).astype(int)
    features_df['ma_alignment'] = ((sma_5 > sma_20) & (sma_20 > sma_50)).astype(int)
    
    return features_df

def create_advanced_volume_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create advanced volume-based features"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    indicators = AdvancedTechnicalIndicators()
    
    # Basic volume features
    features_df['volume_change'] = df['Volume'].pct_change().fillna(0)
    features_df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    features_df['volume_ratio'] = df['Volume'] / (features_df['volume_ma_20'] + 1)
    
    # Volume price trend
    price_change = df['Close'].pct_change()
    features_df['pv_trend'] = price_change * np.sign(features_df['volume_change'])
    features_df['volume_price_corr'] = price_change.rolling(20).corr(features_df['volume_change'])
    
    # On Balance Volume
    features_df['obv'] = indicators.obv(df['Close'], df['Volume'])
    features_df['obv_ma'] = features_df['obv'].rolling(20).mean()
    features_df['obv_signal'] = (features_df['obv'] > features_df['obv_ma']).astype(int)
    
    # Volume Weighted Average Price
    features_df['vwap'] = indicators.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    features_df['price_vs_vwap'] = (df['Close'] / features_df['vwap'] - 1).fillna(0)
    
    # Volume patterns
    for period in [5, 10, 20]:
        vol_ma = df['Volume'].rolling(period).mean()
        features_df[f'volume_spike_{period}'] = (df['Volume'] > vol_ma * 2).astype(int)
        features_df[f'volume_dry_{period}'] = (df['Volume'] < vol_ma * 0.5).astype(int)
    
    # Accumulation/Distribution Line
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    features_df['ad_line'] = (clv * df['Volume']).cumsum()
    features_df['ad_line_ma'] = features_df['ad_line'].rolling(20).mean()
    
    return features_df

def create_pattern_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create candlestick pattern features"""
    features_df = df.copy()
    
    # Body and shadow analysis
    body = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - np.maximum(df['Close'], df['Open'])
    lower_shadow = np.minimum(df['Close'], df['Open']) - df['Low']
    range_hl = df['High'] - df['Low']
    
    features_df['body_ratio'] = body / (range_hl + 1e-10)
    features_df['upper_shadow_ratio'] = upper_shadow / (range_hl + 1e-10)
    features_df['lower_shadow_ratio'] = lower_shadow / (range_hl + 1e-10)
    
    # Candlestick patterns
    features_df['doji'] = (body < range_hl * 0.1).astype(int)
    features_df['hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body * 0.5)).astype(int)
    features_df['shooting_star'] = ((upper_shadow > body * 2) & (lower_shadow < body * 0.5)).astype(int)
    features_df['engulfing_bull'] = ((df['Close'] > df['Open']) & 
                                    (df['Close'] > df['Open'].shift(1)) & 
                                    (df['Open'] < df['Close'].shift(1))).astype(int)
    features_df['engulfing_bear'] = ((df['Close'] < df['Open']) & 
                                    (df['Close'] < df['Open'].shift(1)) & 
                                    (df['Open'] > df['Close'].shift(1))).astype(int)
    
    # Multi-day patterns
    features_df['three_white_soldiers'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'] > df['Close'].shift(1)) &
        (df['Close'].shift(1) > df['Close'].shift(2))
    ).astype(int)
    
    features_df['three_black_crows'] = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'] < df['Close'].shift(1)) &
        (df['Close'].shift(1) < df['Close'].shift(2))
    ).astype(int)
    
    return features_df

def create_sentiment_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create market sentiment features"""
    features_df = df.copy()
    
    # Price momentum sentiment
    returns = df['Close'].pct_change()
    
    # Rolling sentiment scores
    for window in [5, 10, 20]:
        positive_days = (returns > 0).rolling(window).sum()
        features_df[f'bull_sentiment_{window}'] = positive_days / window
        features_df[f'bear_sentiment_{window}'] = 1 - features_df[f'bull_sentiment_{window}']
        
        # Volatility-adjusted sentiment
        vol = returns.rolling(window).std()
        features_df[f'vol_adj_sentiment_{window}'] = returns.rolling(window).mean() / (vol + 1e-10)
    
    # Consecutive patterns
    consecutive_up = (returns > 0).astype(int)
    consecutive_down = (returns <= 0).astype(int)
    
    features_df['consecutive_up'] = consecutive_up.groupby((consecutive_up != consecutive_up.shift()).cumsum()).cumcount() + 1
    features_df['consecutive_down'] = consecutive_down.groupby((consecutive_down != consecutive_down.shift()).cumsum()).cumcount() + 1
    
    features_df['consecutive_up'] = features_df['consecutive_up'] * consecutive_up
    features_df['consecutive_down'] = features_df['consecutive_down'] * consecutive_down
    
    # Fear and greed proxies
    rsi_14 = AdvancedTechnicalIndicators.rsi(df['Close'], 14)
    features_df['fear_index'] = (rsi_14 < 30).astype(int)
    features_df['greed_index'] = (rsi_14 > 70).astype(int)
    features_df['neutral_sentiment'] = ((rsi_14 >= 40) & (rsi_14 <= 60)).astype(int)
    
    return features_df

def create_market_microstructure_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create market microstructure features"""
    features_df = df.copy()
    
    # Intraday patterns
    features_df['opening_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    features_df['closing_strength'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    features_df['opening_strength'] = (df['Open'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    
    # Bid-ask spread proxy (using high-low)
    features_df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
    features_df['spread_ma'] = features_df['spread_proxy'].rolling(20).mean()
    features_df['spread_normalized'] = features_df['spread_proxy'] / features_df['spread_ma']
    
    # Tick analysis proxy
    price_changes = df['Close'].diff()
    features_df['upticks'] = (price_changes > 0).rolling(10).sum()
    features_df['downticks'] = (price_changes < 0).rolling(10).sum()
    features_df['tick_imbalance'] = (features_df['upticks'] - features_df['downticks']) / 10
    
    # Volume distribution
    volume_ma = df['Volume'].rolling(20).mean()
    features_df['volume_imbalance'] = df['Volume'] / volume_ma
    
    # Price efficiency measures
    returns = df['Close'].pct_change()
    features_df['return_reversal'] = -(returns * returns.shift(1))
    features_df['momentum_1d'] = returns
    features_df['momentum_2d'] = returns + returns.shift(1)
    
    return features_df

def create_features_enhanced(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Enhanced main feature creation function with error handling"""
    config = config or FEATURE_CONFIG
    
    if df.empty or len(df) < 50:
        logging.warning("Insufficient data for feature creation")
        return pd.DataFrame()
    
    try:
        # Start with base dataframe
        features_df = df.copy()
        
        # Create different feature groups with error handling
        if config.get('price_features', True):
            try:
                price_features = create_advanced_price_features(df, config)
                # Merge only new columns
                new_cols = [col for col in price_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, price_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Price features failed: {e}")
        
        if config.get('volume_features', True) and 'Volume' in df.columns:
            try:
                volume_features = create_advanced_volume_features(df, config)
                new_cols = [col for col in volume_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, volume_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Volume features failed: {e}")
        
        if config.get('technical_indicators', True):
            try:
                technical_features = create_technical_features_enhanced(df, config)
                new_cols = [col for col in technical_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, technical_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Technical features failed: {e}")
        
        if config.get('volatility_features', True):
            try:
                volatility_features = create_volatility_features_optimized(df, config)
                new_cols = [col for col in volatility_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, volatility_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Volatility features failed: {e}")
        
        if config.get('momentum_features', True):
            try:
                momentum_features = create_momentum_features_optimized(df, config)
                new_cols = [col for col in momentum_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, momentum_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Momentum features failed: {e}")
        
        if config.get('trend_features', True):
            try:
                trend_features = create_trend_features_optimized(df, config)
                new_cols = [col for col in trend_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, trend_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Trend features failed: {e}")
        
        # New advanced features
        if config.get('pattern_features', True):
            try:
                pattern_features = create_pattern_features(df, config)
                new_cols = [col for col in pattern_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, pattern_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Pattern features failed: {e}")
        
        if config.get('sentiment_features', True):
            try:
                sentiment_features = create_sentiment_features(df, config)
                new_cols = [col for col in sentiment_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, sentiment_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Sentiment features failed: {e}")
        
        if config.get('market_microstructure', True):
            try:
                micro_features = create_market_microstructure_features(df, config)
                new_cols = [col for col in micro_features.columns if col not in features_df.columns]
                features_df = pd.concat([features_df, micro_features[new_cols]], axis=1)
            except Exception as e:
                logging.warning(f"Microstructure features failed: {e}")
        
        # Create target variables
        try:
            target_features = create_target_variables_enhanced(df, config)
            target_cols = [col for col in target_features.columns if col.startswith('Target_')]
            features_df = pd.concat([features_df, target_features[target_cols]], axis=1)
        except Exception as e:
            logging.warning(f"Target creation failed: {e}")
        
        # Clean up data
        features_df = clean_features_data(features_df)
        
        logging.info(f"Feature creation completed: {len(features_df.columns)} features, {len(features_df)} rows")
        return features_df
        
    except Exception as e:
        logging.error(f"Feature creation failed: {e}")
        return df

def create_technical_features_enhanced(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Enhanced technical indicator features"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    indicators = AdvancedTechnicalIndicators()
    
    # RSI family
    if 'rsi' in config['technical_indicators']:
        features_df['rsi'] = indicators.rsi(df['Close'])
        features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
        features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
        features_df['rsi_normalized'] = (features_df['rsi'] - 50) / 50
        
        # Multi-timeframe RSI
        features_df['rsi_9'] = indicators.rsi(df['Close'], 9)
        features_df['rsi_21'] = indicators.rsi(df['Close'], 21)
    
    # MACD family
    if 'macd' in config['technical_indicators']:
        macd_data = indicators.macd(df['Close'])
        features_df['macd'] = macd_data['macd']
        features_df['macd_signal'] = macd_data['signal']
        features_df['macd_histogram'] = macd_data['histogram']
        features_df['macd_bullish'] = (features_df['macd'] > features_df['macd_signal']).astype(int)
        features_df['macd_crossover'] = ((features_df['macd'] > features_df['macd_signal']) & 
                                        (features_df['macd'].shift(1) <= features_df['macd_signal'].shift(1))).astype(int)
    
    # Bollinger Bands family
    if 'bollinger' in config['technical_indicators']:
        bb_data = indicators.bollinger_bands(df['Close'])
        features_df['bb_upper'] = bb_data['bb_upper']
        features_df['bb_lower'] = bb_data['bb_lower']
        features_df['bb_width'] = bb_data['bb_width']
        features_df['bb_position'] = bb_data['bb_position']
        features_df['bb_squeeze'] = (bb_data['bb_width'] < bb_data['bb_width'].rolling(20).mean()).astype(int)
        features_df['bb_breakout_up'] = (df['Close'] > bb_data['bb_upper']).astype(int)
        features_df['bb_breakout_down'] = (df['Close'] < bb_data['bb_lower']).astype(int)
    
    # Stochastic family
    if 'stochastic' in config['technical_indicators']:
        stoch_data = indicators.stochastic(df['High'], df['Low'], df['Close'])
        features_df['stoch_k'] = stoch_data['stoch_k']
        features_df['stoch_d'] = stoch_data['stoch_d']
        features_df['stoch_overbought'] = (features_df['stoch_k'] > 80).astype(int)
        features_df['stoch_oversold'] = (features_df['stoch_k'] < 20).astype(int)
        features_df['stoch_crossover'] = ((features_df['stoch_k'] > features_df['stoch_d']) & 
                                         (features_df['stoch_k'].shift(1) <= features_df['stoch_d'].shift(1))).astype(int)
    
    # ATR and volatility
    if 'atr' in config['technical_indicators']:
        features_df['atr'] = indicators.atr(df['High'], df['Low'], df['Close'])
        features_df['atr_normalized'] = features_df['atr'] / df['Close']
        features_df['atr_percentile'] = features_df['atr'].rolling(50).rank(pct=True)
    
    # ADX trend strength
    if 'adx' in config.get('technical_indicators', []):
        adx_data = indicators.adx(df['High'], df['Low'], df['Close'])
        features_df['adx'] = adx_data['adx']
        features_df['di_plus'] = adx_data['di_plus']
        features_df['di_minus'] = adx_data['di_minus']
        features_df['trend_strength'] = (features_df['adx'] > 25).astype(int)
    
    return features_df

def create_volatility_features_optimized(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create optimized volatility features"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    
    # Basic volatility measures
    returns = df['Close'].pct_change().fillna(0)
    
    for period in config['lookback_periods']:
        # Historical volatility
        features_df[f'volatility_{period}'] = returns.rolling(period).std()
        
        # Realized volatility (sum of squared returns)
        features_df[f'realized_vol_{period}'] = (returns ** 2).rolling(period).sum()
        
        # Volatility of volatility
        vol_series = returns.rolling(period).std()
        features_df[f'vol_of_vol_{period}'] = vol_series.rolling(period).std()
        
        # Parkinson estimator (high-low volatility)
        parkinson_vol = np.sqrt((np.log(df['High'] / df['Low']) ** 2).rolling(period).mean())
        features_df[f'parkinson_vol_{period}'] = parkinson_vol
    
    # Volatility regime detection
    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()
    features_df['vol_regime'] = (vol_20 > vol_60).astype(int)
    
    # GARCH-like features (simplified)
    features_df['vol_mean_reversion'] = (vol_20 / vol_20.rolling(60).mean() - 1).fillna(0)
    
    return features_df

def create_momentum_features_optimized(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create optimized momentum features"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    
    # Price momentum over different periods
    for period in [5, 10, 20, 50]:
        momentum = (df['Close'] / df['Close'].shift(period) - 1).fillna(0)
        features_df[f'momentum_{period}'] = momentum
        features_df[f'momentum_{period}_rank'] = momentum.rolling(60).rank(pct=True)
    
    # Rate of change
    for period in config['lookback_periods']:
        roc = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)).fillna(0)
        features_df[f'roc_{period}'] = roc
    
    # Momentum oscillators
    ma_20 = df['Close'].rolling(20).mean()
    features_df['price_momentum'] = (df['Close'] / ma_20 - 1).fillna(0)
    
    vol_ma_20 = df['Volume'].rolling(20).mean()
    features_df['volume_momentum'] = (df['Volume'] / vol_ma_20 - 1).fillna(0)
    
    # Acceleration (momentum of momentum)
    mom_10 = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
    features_df['momentum_acceleration'] = (mom_10 - mom_10.shift(5)).fillna(0)
    
    return features_df

def create_trend_features_optimized(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create optimized trend features"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    
    # Trend direction using multiple timeframes
    for period in config['lookback_periods']:
        # Linear regression slope approximation
        def calculate_slope(series):
            if len(series) < period:
                return 0
            x = np.arange(len(series))
            try:
                slope = np.polyfit(x, series, 1)[0]
                return slope
            except:
                return 0
        
        slope = df['Close'].rolling(period).apply(calculate_slope, raw=False)
        features_df[f'trend_slope_{period}'] = slope.fillna(0)
        
        # Trend strength
        sma = df['Close'].rolling(period).mean()
        features_df[f'trend_strength_{period}'] = (abs(df['Close'] - sma) / sma).fillna(0)
    
    # Moving average convergence/divergence
    sma_5 = df['Close'].rolling(5).mean()
    sma_20 = df['Close'].rolling(20).mean()
    features_df['ma_convergence'] = ((sma_5 - sma_20) / sma_20).fillna(0)
    
    # Trend consistency
    returns = df['Close'].pct_change().fillna(0)
    for period in [10, 20]:
        # Percentage of positive returns in period
        features_df[f'positive_returns_pct_{period}'] = (returns > 0).rolling(period).mean()
        
        # Trend persistence
        trend_up = (df['Close'] > df['Close'].shift(1)).astype(int)
        features_df[f'trend_persistence_{period}'] = trend_up.rolling(period).mean()
    
    return features_df

def create_target_variables_enhanced(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Create enhanced target variables for different investment horizons"""
    config = config or FEATURE_CONFIG
    features_df = df.copy()
    
    target_definitions = {
        'next_week': 5,
        'next_month': 21,
        'next_quarter': 63,
        'next_year': 252
    }
    
    for horizon, days in target_definitions.items():
        if horizon in config.get('target_horizons', []):
            # Future returns
            future_returns = (df['Close'].shift(-days) / df['Close'] - 1)
            
            # Binary classification target (positive returns)
            features_df[f'Target_{horizon}'] = (future_returns > 0).astype(int)
            
            # Enhanced targets
            features_df[f'Target_{horizon}_returns'] = future_returns.fillna(0)
            
            # Multi-class target with better thresholds
            if not future_returns.isna().all():
                returns_25 = future_returns.quantile(0.25)
                returns_75 = future_returns.quantile(0.75)
                
                conditions = [
                    future_returns <= returns_25,
                    (future_returns > returns_25) & (future_returns <= 0),
                    (future_returns > 0) & (future_returns <= returns_75),
                    future_returns > returns_75
                ]
                choices = [0, 1, 2, 3]  # Strong sell, weak sell, weak buy, strong buy
                
                features_df[f'Target_{horizon}_multiclass'] = np.select(conditions, choices, default=1)
            else:
                features_df[f'Target_{horizon}_multiclass'] = 1
    
    return features_df

def clean_features_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate features data"""
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Handle NaN values intelligently
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df[col].isnull().any():
            # For different feature types, use different fill strategies
            if 'ratio' in col.lower() or 'normalized' in col.lower():
                # Ratios and normalized features: fill with 0
                df[col] = df[col].fillna(0)
            elif 'ma' in col.lower() or 'sma' in col.lower() or 'ema' in col.lower():
                # Moving averages: forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            elif col.startswith('Target_'):
                # Targets: don't fill, keep NaN for proper handling
                pass
            else:
                # Other features: fill with median
                median_val = df[col].median()
                if pd.isna(median_val):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(median_val)
    
    # Cap extreme values (outliers)
    for col in numeric_columns:
        if not col.startswith('Target_'):
            Q1 = df[col].quantile(0.01)
            Q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=Q1, upper=Q99)
    
    return df

# ==================== PARALLEL PROCESSING FIXED ====================

def process_ticker_features_safe(args):
    """Process features for a single ticker - thread-safe version"""
    ticker, df, config = args
    try:
        if df.empty or len(df) < 50:
            logging.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return ticker, pd.DataFrame()
        
        processed_df = create_features_enhanced(df, config)
        
        if processed_df.empty:
            logging.warning(f"Feature creation failed for {ticker}")
            return ticker, pd.DataFrame()
        
        logging.info(f"Successfully processed {ticker}: {processed_df.shape}")
        return ticker, processed_df
        
    except Exception as e:
        logging.error(f"Feature processing failed for {ticker}: {e}")
        return ticker, pd.DataFrame()

def create_features_parallel_safe(data_dict: Dict[str, pd.DataFrame], 
                                config: Dict = None,
                                max_workers: int = None) -> Dict[str, pd.DataFrame]:
    """Thread-safe parallel feature creation"""
    config = config or FEATURE_CONFIG
    
    if not config.get('parallel_processing', False) or len(data_dict) <= 2:
        # Sequential processing for small datasets or when parallel is disabled
        result = {}
        for ticker, df in tqdm(data_dict.items(), desc="Processing features"):
            try:
                result[ticker] = create_features_enhanced(df, config)
            except Exception as e:
                logging.error(f"Sequential processing failed for {ticker}: {e}")
                result[ticker] = pd.DataFrame()
        return result
    
    # Parallel processing
    max_workers = max_workers or min(4, len(data_dict))  # Conservative worker count
    
    tasks = [(ticker, df, config) for ticker, df in data_dict.items()]
    results = {}
    
    print(f"Processing {len(tasks)} tickers with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker_features_safe, task): task[0] 
                           for task in tasks}
        
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker), 
                          desc="Feature processing"):
            ticker = future_to_ticker[future]
            try:
                ticker_result, processed_df = future.result()
                results[ticker_result] = processed_df
            except Exception as e:
                logging.error(f"Failed to get result for {ticker}: {e}")
                results[ticker] = pd.DataFrame()
    
    successful_count = sum(1 for df in results.values() if not df.empty)
    print(f"Feature processing completed: {successful_count}/{len(tasks)} successful")
    
    return results

# ==================== MAIN INTERFACE FUNCTIONS ====================

def engineer_features(data_dict: Dict[str, pd.DataFrame],
                             config: Dict = None,
                             use_cache: bool = True,
                             parallel: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Enhanced main interface for feature engineering
    """
    config = config or FEATURE_CONFIG
    config['parallel_processing'] = parallel
    
    # Initialize enhanced cache
    cache = EnhancedTrainingCache() if use_cache else None
    
    processed_data = {}
    cache_hits = 0
    
    print(f"Starting enhanced feature engineering for {len(data_dict)} tickers...")
    
    # First pass: check cache
    remaining_data = {}
    for ticker, df in data_dict.items():
        if df.empty:
            processed_data[ticker] = pd.DataFrame()
            continue
            
        # Check cache first
        if cache:
            cached_features = cache.get_processed_features(ticker, df)
            if cached_features is not None and not cached_features.empty:
                processed_data[ticker] = cached_features
                cache_hits += 1
                continue
        
        remaining_data[ticker] = df
    
    print(f"Cache hits: {cache_hits}/{len(data_dict)}")
    print(f"Processing remaining: {len(remaining_data)} tickers")
    
    # Process remaining tickers
    if remaining_data:
        if parallel and len(remaining_data) > 1:
            new_results = create_features_parallel_safe(remaining_data, config)
        else:
            new_results = {}
            for ticker, df in tqdm(remaining_data.items(), desc="Sequential processing"):
                new_results[ticker] = create_features_enhanced(df, config)
        
        # Cache new results and add to processed data
        for ticker, processed_df in new_results.items():
            if not processed_df.empty and cache:
                cache.save_processed_features(ticker, remaining_data[ticker], processed_df)
            processed_data[ticker] = processed_df
    
    # Generate summary
    successful_features = sum(1 for df in processed_data.values() if not df.empty)
    print(f"Enhanced feature engineering completed: {successful_features}/{len(data_dict)} successful")
    
    if successful_features > 0:
        sample_df = next(df for df in processed_data.values() if not df.empty)
        feature_count = len([col for col in sample_df.columns if not col.startswith('Target_')])
        target_count = len([col for col in sample_df.columns if col.startswith('Target_')])
        print(f"Features per ticker: {feature_count} features, {target_count} targets")
    
    return processed_data

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Enhanced Feature Engineering System - Fixed Version")
    print("="*60)
    
    # Enhanced configuration
    enhanced_config = FEATURE_CONFIG.copy()
    enhanced_config['advanced_features'] = True
    enhanced_config['pattern_features'] = True
    enhanced_config['sentiment_features'] = True
    enhanced_config['market_microstructure'] = True
    
    print("Enhanced Configuration:")
    for key, value in enhanced_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nNew features added:")
    print(f"  ✓ Advanced price features (gaps, spreads, crossovers)")
    print(f"  ✓ Enhanced volume features (OBV, VWAP, accumulation)")
    print(f"  ✓ Candlestick patterns (doji, hammer, engulfing)")
    print(f"  ✓ Market sentiment indicators")
    print(f"  ✓ Market microstructure features")
    print(f"  ✓ Multi-timeframe analysis")
    print(f"  ✓ Fixed caching system (no Series hash error)")
    print(f"  ✓ Enhanced error handling and logging")
    print(f"  ✓ Comprehensive data cleaning")