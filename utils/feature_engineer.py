# utils/feature_engineer.py
import pandas as pd
import numpy as np
import warnings
import joblib
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import gc
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Define HORIZONS locally to avoid circular import
HORIZONS = {
    'next_day': 1,
    'next_week': 5,
    'next_month': 22,
    'next_quarter': 66,
    'next_year': 252,
    'next_3_years': 756,
    'next_5_years': 1260
}

# Configuration
FEATURE_CONFIG = {
    'technical_indicators': {
        'sma_periods': [5, 10, 20, 50, 100, 200],
        'ema_periods': [12, 26, 50],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14,
        'stoch_k': 14,
        'stoch_d': 3,
        'williams_r': 14
    },
    'risk_metrics': {
        'volatility_window': 20,
        'var_confidence': 0.05,
        'sharpe_window': 252,
        'max_drawdown_window': 252
    },
    'market_regime': {
        'trend_window': 50,
        'volatility_regime_window': 20,
        'correlation_window': 60
    },
    'feature_selection': {
        'correlation_threshold': 0.95,
        'variance_threshold': 0.01,
        'missing_threshold': 0.3
    }
}

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators using pandas-ta"""
    if df.empty or len(df) < 50:
        return df
    
    try:
        # Required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # Create a copy to avoid modifying original data
        df = df.copy()
        n = len(df)
        
        # ===== Moving Averages =====
        for period in FEATURE_CONFIG['technical_indicators']['sma_periods']:
            if n >= period:
                # Calculate SMA
                sma = df.ta.sma(length=period)
                if sma is not None:
                    df[f'SMA_{period}'] = sma
                    # Calculate ratio safely
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = df['Close'] / df[f'SMA_{period}']
                        ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df[f'Price_SMA_{period}_Ratio'] = ratio
        
        for period in FEATURE_CONFIG['technical_indicators']['ema_periods']:
            if n >= period:
                # Calculate EMA
                ema = df.ta.ema(length=period)
                if ema is not None:
                    df[f'EMA_{period}'] = ema
                    # Calculate ratio safely
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = df['Close'] / df[f'EMA_{period}']
                        ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df[f'Price_EMA_{period}_Ratio'] = ratio
        
        # ===== RSI =====
        rsi_period = FEATURE_CONFIG['technical_indicators']['rsi_period']
        if n >= rsi_period:
            rsi = df.ta.rsi(length=rsi_period)
            if rsi is not None:
                df['RSI'] = rsi
                df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
                df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        
        # ===== MACD =====
        macd_fast = FEATURE_CONFIG['technical_indicators']['macd_fast']
        macd_slow = FEATURE_CONFIG['technical_indicators']['macd_slow']
        macd_signal = FEATURE_CONFIG['technical_indicators']['macd_signal']
        if n >= macd_slow:
            macd = df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
                df.rename(columns={
                    f'MACD_{macd_fast}_{macd_slow}_{macd_signal}': 'MACD',
                    f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}': 'MACD_Signal',
                    f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}': 'MACD_Histogram'
                }, inplace=True)
                df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        # ===== Bollinger Bands =====
        bb_period = FEATURE_CONFIG['technical_indicators']['bb_period']
        bb_std = FEATURE_CONFIG['technical_indicators']['bb_std']
        if n >= bb_period:
            bb = df.ta.bbands(length=bb_period, std=bb_std)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
                df.rename(columns={
                    f'BBU_{bb_period}_{bb_std}': 'BB_Upper',
                    f'BBM_{bb_period}_{bb_std}': 'BB_Middle',
                    f'BBL_{bb_period}_{bb_std}': 'BB_Lower',
                    f'BBB_{bb_period}_{bb_std}': 'BB_Width',
                    f'BBP_{bb_period}_{bb_std}': 'BB_Position'
                }, inplace=True)
        
        # ===== ATR =====
        atr_period = FEATURE_CONFIG['technical_indicators']['atr_period']
        if n >= atr_period:
            atr = df.ta.atr(length=atr_period)
            if atr is not None:
                df['ATR'] = atr
                # Calculate ratio safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = df['ATR'] / df['Close']
                    ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
                df['ATR_Ratio'] = ratio
        
        # ===== Stochastic Oscillator =====
        stoch_k = FEATURE_CONFIG['technical_indicators']['stoch_k']
        stoch_d = FEATURE_CONFIG['technical_indicators']['stoch_d']
        if n >= stoch_k:
            stoch = df.ta.stoch(k=stoch_k, d=stoch_d)
            if stoch is not None:
                df = pd.concat([df, stoch], axis=1)
                df.rename(columns={
                    f'STOCHk_{stoch_k}_{stoch_d}_{stoch_d}': 'Stoch_K',
                    f'STOCHd_{stoch_k}_{stoch_d}_{stoch_d}': 'Stoch_D'
                }, inplace=True)
        
        # ===== Williams %R =====
        williams_period = FEATURE_CONFIG['technical_indicators']['williams_r']
        if n >= williams_period:
            willr = df.ta.willr(length=williams_period)
            if willr is not None:
                df['Williams_R'] = willr
        
        # ===== Volume indicators =====
        if n >= 20:
            # Volume SMA
            vol_sma = df.ta.sma(length=20, close='Volume')
            if vol_sma is not None:
                df['Volume_SMA_20'] = vol_sma
                # Calculate ratio safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = df['Volume'] / df['Volume_SMA_20']
                    ratio.replace([np.inf, -np.inf], 1, inplace=True)
                    ratio.fillna(1, inplace=True)
                df['Volume_Ratio'] = ratio
                df['Price_Volume'] = df['Close'] * df['Volume']
        
        # ===== Momentum indicators =====
        if n >= 10:
            mom = df.ta.mom(length=10)
            roc = df.ta.roc(length=10)
            if mom is not None:
                df['MOM_10'] = mom
            if roc is not None:
                df['ROC_10'] = roc
        
        # ===== CCI =====
        if n >= 20:
            cci = df.ta.cci(length=20)
            if cci is not None:
                df['CCI'] = cci
        
        # ===== MFI =====
        if n >= 14:
            mfi = df.ta.mfi(length=14)
            if mfi is not None:
                df['MFI'] = mfi
            
    except Exception as e:
        warnings.warn(f"Technical indicators calculation failed: {str(e)}")
    
    return df

def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price-based features with safe log calculations"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        new_features = {}
        
        # Create a working copy
        df = df.copy()
        
        # Handle zero/negative prices
        df['Close'] = df['Close'].replace(0, np.nan).ffill().bfill()
        if 'Open' in df.columns:
            df['Open'] = df['Open'].replace(0, np.nan).ffill().bfill()
        
        # Returns - safe calculation
        returns = df['Close'].pct_change()
        returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_features['Returns'] = returns
        
        # Log Returns - handle non-positive values safely
        with np.errstate(divide='ignore', invalid='ignore'):
            price_ratio = df['Close'] / df['Close'].shift(1)
            # Create mask for valid positive values
            valid_mask = (price_ratio > 0) & (df['Close'] > 0) & (df['Close'].shift(1) > 0)
            log_returns = np.full(len(df), np.nan)
            log_returns[valid_mask] = np.log(price_ratio[valid_mask])
        new_features['Log_Returns'] = log_returns
        
        # Price gaps
        if 'Open' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                gap = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
                gap.replace([np.inf, -np.inf], 0, inplace=True)
                gap.fillna(0, inplace=True)
            new_features['Gap'] = gap
            new_features['Gap_Up'] = (new_features['Gap'] > 0.02).astype(int)
            new_features['Gap_Down'] = (new_features['Gap'] < -0.02).astype(int)
        
        # Intraday patterns
        if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                intraday_return = (df['Close'] - df['Open']) / np.where(df['Open'] > 0, df['Open'], 1)
                intraday_return.replace([np.inf, -np.inf], 0, inplace=True)
                intraday_return.fillna(0, inplace=True)
                new_features['Intraday_Return'] = intraday_return
                
                hl_ratio = df['High'] / np.where(df['Low'] > 0, df['Low'], 1)
                hl_ratio.replace([np.inf, -np.inf], 1, inplace=True)
                hl_ratio.fillna(1, inplace=True)
                new_features['High_Low_Ratio'] = hl_ratio
                
                body_size = abs(df['Close'] - df['Open']) / np.where(df['Open'] > 0, df['Open'], 1)
                body_size.replace([np.inf, -np.inf], 0, inplace=True)
                body_size.fillna(0, inplace=True)
                new_features['Body_Size'] = body_size
            
            # Calculate shadows safely
            upper_shadow = (df['High'] - np.maximum(df['Open'], df['Close'])) 
            lower_shadow = (np.minimum(df['Open'], df['Close']) - df['Low'])
            
            # Avoid division by zero
            max_open_close = np.maximum(df['Open'], df['Close'])
            min_open_close = np.minimum(df['Open'], df['Close'])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                upper_shadow_ratio = upper_shadow / np.where(max_open_close > 0, max_open_close, 1)
                lower_shadow_ratio = lower_shadow / np.where(min_open_close > 0, min_open_close, 1)
                upper_shadow_ratio.replace([np.inf, -np.inf], 0, inplace=True)
                lower_shadow_ratio.replace([np.inf, -np.inf], 0, inplace=True)
                upper_shadow_ratio.fillna(0, inplace=True)
                lower_shadow_ratio.fillna(0, inplace=True)
            
            new_features['Upper_Shadow'] = upper_shadow_ratio
            new_features['Lower_Shadow'] = lower_shadow_ratio
            
            # Price patterns
            new_features['Doji'] = (abs(df['Close'] - df['Open']) < 0.001).astype(int)
            new_features['Hammer'] = ((new_features['Lower_Shadow'] > 2 * new_features['Body_Size']) & 
                                   (new_features['Upper_Shadow'] < new_features['Body_Size'])).astype(int)
            new_features['Shooting_Star'] = ((new_features['Upper_Shadow'] > 2 * new_features['Body_Size']) & 
                                          (new_features['Lower_Shadow'] < new_features['Body_Size'])).astype(int)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            if len(df) > window:
                new_features[f'Returns_Mean_{window}'] = new_features['Returns'].rolling(window).mean()
                new_features[f'Returns_Std_{window}'] = new_features['Returns'].rolling(window).std()
                new_features[f'Returns_Skew_{window}'] = new_features['Returns'].rolling(window).skew()
                new_features[f'Returns_Kurt_{window}'] = new_features['Returns'].rolling(window).kurt()
                new_features[f'Price_Min_{window}'] = df['Close'].rolling(window).min()
                new_features[f'Price_Max_{window}'] = df['Close'].rolling(window).max()
                
                # Calculate position safely
                price_range = new_features[f'Price_Max_{window}'] - new_features[f'Price_Min_{window}']
                with np.errstate(divide='ignore', invalid='ignore'):
                    position = (df['Close'] - new_features[f'Price_Min_{window}']) / np.where(price_range > 0, price_range, 1)
                    position.replace([np.inf, -np.inf], 0.5, inplace=True)
                    position.fillna(0.5, inplace=True)
                new_features[f'Price_Position_{window}'] = position
        
        # Volatility features
        if 'Returns' in new_features:
            new_features['Volatility_5'] = new_features['Returns'].rolling(5).std() * np.sqrt(252)
            new_features['Volatility_20'] = new_features['Returns'].rolling(20).std() * np.sqrt(252)
            
            # Calculate volatility ratio safely
            if 'Volatility_5' in new_features and 'Volatility_20' in new_features:
                with np.errstate(divide='ignore', invalid='ignore'):
                    vol_ratio = new_features['Volatility_5'] / new_features['Volatility_20']
                    vol_ratio.replace([np.inf, -np.inf], 1, inplace=True)
                    vol_ratio.fillna(1, inplace=True)
                new_features['Volatility_Ratio'] = vol_ratio
        
        # Merge new features
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    except Exception as e:
        warnings.warn(f"Price features calculation failed: {str(e)}")
    
    return df

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk-based features"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        new_features = {}
        returns = df['Returns'].dropna() if 'Returns' in df.columns else df['Close'].pct_change().dropna()
        
        # Value at Risk
        window = FEATURE_CONFIG['risk_metrics']['volatility_window']
        confidence = FEATURE_CONFIG['risk_metrics']['var_confidence']
        
        if len(returns) > window:
            new_features['VaR'] = returns.rolling(window).quantile(confidence)
            new_features['CVaR'] = returns.rolling(window).apply(lambda x: x[x <= x.quantile(confidence)].mean())
        
        # Maximum Drawdown
        rolling_max = df['Close'].rolling(window=252, min_periods=1).max()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        new_features['Drawdown'] = drawdown
        new_features['Max_Drawdown'] = drawdown.rolling(window=252, min_periods=1).min()
        
        # Sharpe Ratio (rolling)
        if len(returns) > 252:
            mean_returns = returns.rolling(252).mean()
            std_returns = returns.rolling(252).std()
            with np.errstate(divide='ignore', invalid='ignore'):
                sharpe = mean_returns / std_returns * np.sqrt(252)
                sharpe.replace([np.inf, -np.inf], 0, inplace=True)
                sharpe.fillna(0, inplace=True)
            new_features['Sharpe_Ratio'] = sharpe
        
        # Beta (relative to market - using own returns as proxy)
        if len(returns) > 60:
            market_returns = returns.rolling(60).mean()
            covariance = returns.rolling(60).cov(market_returns)
            market_variance = market_returns.rolling(60).var()
            with np.errstate(divide='ignore', invalid='ignore'):
                beta = covariance / market_variance
                beta.replace([np.inf, -np.inf], 1, inplace=True)
                beta.fillna(1, inplace=True)
            new_features['Beta'] = beta
        
        # Sortino Ratio
        if len(returns) > 252:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.rolling(252).std()
                mean_returns = returns.rolling(252).mean()
                with np.errstate(divide='ignore', invalid='ignore'):
                    sortino = mean_returns / downside_std * np.sqrt(252)
                    sortino.replace([np.inf, -np.inf], 0, inplace=True)
                    sortino.fillna(0, inplace=True)
                new_features['Sortino_Ratio'] = sortino
        
        # Merge new features
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    except Exception as e:
        warnings.warn(f"Risk metrics calculation failed: {str(e)}")
    
    return df

def calculate_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market regime indicators"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        new_features = {}
        
        # Trend identification
        trend_window = FEATURE_CONFIG['market_regime']['trend_window']
        if len(df) > trend_window:
            new_features['Trend_SMA'] = df['Close'].rolling(trend_window).mean()
            new_features['Trend_Direction'] = (df['Close'] > new_features['Trend_SMA']).astype(int)
            
            # Calculate trend strength safely
            with np.errstate(divide='ignore', invalid='ignore'):
                strength = abs(df['Close'] - new_features['Trend_SMA']) / new_features['Trend_SMA']
                strength.replace([np.inf, -np.inf], 0, inplace=True)
                strength.fillna(0, inplace=True)
            new_features['Trend_Strength'] = strength
        
        # Volatility regime
        vol_window = FEATURE_CONFIG['market_regime']['volatility_regime_window']
        if len(df) > vol_window:
            if 'Returns' in df.columns:
                vol_rolling = df['Returns'].rolling(vol_window).std()
            else:
                vol_rolling = df['Close'].pct_change().rolling(vol_window).std()
            
            vol_threshold = vol_rolling.quantile(0.7)
            new_features['High_Vol_Regime'] = (vol_rolling > vol_threshold).astype(int)
        
        # Price momentum
        for period in [5, 10, 20]:
            if len(df) > period:
                # Calculate momentum safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    momentum = df['Close'] / df['Close'].shift(period) - 1
                    momentum.replace([np.inf, -np.inf], 0, inplace=True)
                    momentum.fillna(0, inplace=True)
                new_features[f'Momentum_{period}'] = momentum
                new_features[f'Momentum_Rank_{period}'] = new_features[f'Momentum_{period}'].rolling(60).rank(pct=True)
        
        # Support and resistance levels
        if len(df) > 20:
            new_features['Support_20'] = df['Low'].rolling(20).min()
            new_features['Resistance_20'] = df['High'].rolling(20).max()
            
            # Calculate distances safely
            with np.errstate(divide='ignore', invalid='ignore'):
                support_dist = (df['Close'] - new_features['Support_20']) / df['Close']
                resistance_dist = (new_features['Resistance_20'] - df['Close']) / df['Close']
                support_dist.replace([np.inf, -np.inf], 0, inplace=True)
                resistance_dist.replace([np.inf, -np.inf], 0, inplace=True)
                support_dist.fillna(0, inplace=True)
                resistance_dist.fillna(0, inplace=True)
                
            new_features['Support_Distance'] = support_dist
            new_features['Resistance_Distance'] = resistance_dist
        
        # Merge new features
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    except Exception as e:
        warnings.warn(f"Market regime features calculation failed: {str(e)}")
    
    return df

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features"""
    if df.empty or 'Volume' not in df.columns:
        return df
    
    try:
        new_features = {}
        
        # Volume trends
        for period in [5, 10, 20]:
            if len(df) > period:
                vol_sma = df['Volume'].rolling(period).mean()
                new_features[f'Volume_SMA_{period}'] = vol_sma
                
                # Calculate volume ratio safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    vol_ratio = df['Volume'] / vol_sma
                    vol_ratio.replace([np.inf, -np.inf], 1, inplace=True)
                    vol_ratio.fillna(1, inplace=True)
                new_features[f'Volume_Ratio_{period}'] = vol_ratio
        
        # Volume patterns
        if 'Volume_SMA_20' in new_features:
            new_features['Volume_Spike'] = (df['Volume'] > new_features['Volume_SMA_20'] * 2).astype(int)
            new_features['Volume_Dry'] = (df['Volume'] < new_features['Volume_SMA_20'] * 0.5).astype(int)
        
        # Price-Volume relationship
        if 'Close' in df.columns:
            new_features['PV_Trend'] = ((df['Close'] > df['Close'].shift(1)) & 
                                     (df['Volume'] > df['Volume'].shift(1))).astype(int)
            new_features['Volume_Price_Correlation'] = df['Volume'].rolling(20).corr(df['Close'])
        
        # On-Balance Volume
        if 'Close' in df.columns:
            obv = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
            new_features['OBV'] = obv
            new_features['OBV_Slope'] = obv.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Merge new features
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    except Exception as e:
        warnings.warn(f"Volume features calculation failed: {str(e)}")
    
    return df

def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based features"""
    if df.empty:
        return df
    
    try:
        new_features = {}
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Basic time features
        new_features['DayOfWeek'] = df.index.dayofweek
        new_features['Month'] = df.index.month
        new_features['Quarter'] = df.index.quarter
        new_features['DayOfYear'] = df.index.dayofyear
        
        # Market timing features
        new_features['Monday'] = (new_features['DayOfWeek'] == 0).astype(int)
        new_features['Friday'] = (new_features['DayOfWeek'] == 4).astype(int)
        new_features['MonthEnd'] = df.index.is_month_end.astype(int)
        new_features['MonthStart'] = df.index.is_month_start.astype(int)
        new_features['QuarterEnd'] = df.index.is_quarter_end.astype(int)
        new_features['YearEnd'] = df.index.is_year_end.astype(int)
        
        # Seasonal patterns
        new_features['Sin_DayOfYear'] = np.sin(2 * np.pi * new_features['DayOfYear'] / 365.25)
        new_features['Cos_DayOfYear'] = np.cos(2 * np.pi * new_features['DayOfYear'] / 365.25)
        new_features['Sin_DayOfWeek'] = np.sin(2 * np.pi * new_features['DayOfWeek'] / 7)
        new_features['Cos_DayOfWeek'] = np.cos(2 * np.pi * new_features['DayOfWeek'] / 7)
        
        # Merge new features
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    except Exception as e:
        warnings.warn(f"Time features calculation failed: {str(e)}")
    
    return df

def create_target_variables(df: pd.DataFrame, horizons: Dict[str, int] = None) -> pd.DataFrame:
    """Create target variables for different investment horizons"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    horizons = horizons or HORIZONS
    
    try:
        for horizon_name, days in horizons.items():
            if len(df) > days:
                # Calculate future returns
                future_price = df['Close'].shift(-days)
                returns = (future_price - df['Close']) / df['Close']
                
                # Create binary target (1 if positive return, 0 otherwise)
                df[f'Target_{horizon_name}'] = (returns > 0).astype(int)
                
                # Also create continuous target for regression
                df[f'Target_Continuous_{horizon_name}'] = returns
                
                # Risk-adjusted target
                if 'Returns' in df.columns:
                    volatility = df['Returns'].rolling(20).std()
                else:
                    volatility = df['Close'].pct_change().rolling(20).std()
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    risk_adj_return = returns / (volatility * np.sqrt(days))
                    risk_adj_return.replace([np.inf, -np.inf], 0, inplace=True)
                    risk_adj_return.fillna(0, inplace=True)
                df[f'Target_RiskAdj_{horizon_name}'] = (risk_adj_return > 0).astype(int)
        
    except Exception as e:
        warnings.warn(f"Target variable creation failed: {str(e)}")
    
    return df

def calculate_risk_score(df: pd.DataFrame) -> float:
    """Calculate composite risk score for a stock"""
    if df.empty or 'Close' not in df.columns:
        return 0.5
    
    try:
        # Calculate individual risk components
        returns = df['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            return 0.5
        
        # Volatility (higher = more risky)
        volatility = returns.std() * np.sqrt(252)
        vol_score = min(volatility / 0.5, 1.0)  # Normalize to 0-1
        
        # Maximum drawdown
        rolling_max = df['Close'].expanding().max()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        dd_score = min(max_drawdown / 0.5, 1.0) if max_drawdown > 0 else 0
        
        # Volume volatility
        if 'Volume' in df.columns:
            volume_returns = df['Volume'].pct_change().dropna()
            if len(volume_returns) > 0:
                volume_vol = volume_returns.std()
                vol_vol_score = min(volume_vol / 2.0, 1.0) if volume_vol > 0 else 0
            else:
                vol_vol_score = 0.0
        else:
            vol_vol_score = 0.0
        
        # Composite risk score
        risk_score = 0.4 * vol_score + 0.3 * dd_score + 0.3 * vol_vol_score
        
        return float(np.clip(risk_score, 0, 1))
        
    except Exception as e:
        warnings.warn(f"Risk score calculation failed: {str(e)}")
        return 0.5

def clean_and_validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate feature data"""
    if df.empty:
        return df
    
    try:
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many missing values
        missing_threshold = FEATURE_CONFIG['feature_selection']['missing_threshold']
        missing_pct = df.isnull().sum() / len(df)
        cols_to_keep = missing_pct[missing_pct <= missing_threshold].index
        df = df[cols_to_keep]
        
        # Remove highly correlated features
        corr_threshold = FEATURE_CONFIG['feature_selection']['correlation_threshold']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
            df = df.drop(columns=to_drop)
        
        # Remove low variance features
        variance_threshold = FEATURE_CONFIG['feature_selection']['variance_threshold']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].var() < variance_threshold:
                df = df.drop(columns=[col])
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Final check for any remaining NaN values
        df = df.dropna()
        
    except Exception as e:
        warnings.warn(f"Feature cleaning failed: {str(e)}")
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering pipeline"""
    if df.empty:
        return df
    
    try:
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Apply feature engineering steps
        df = calculate_technical_indicators(df)
        df = calculate_price_features(df)
        df = calculate_risk_metrics(df)
        df = calculate_market_regime_features(df)
        df = calculate_volume_features(df)
        df = calculate_time_features(df)
        df = create_target_variables(df)
        df = clean_and_validate_features(df)
        
        # Force garbage collection
        gc.collect()
        
        return df
        
    except Exception as e:
        warnings.warn(f"Feature engineering failed: {str(e)}")
        return df
def process_all_data(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Process all ticker data with feature engineering"""
    processed_data = {}
    
    for ticker, df in tqdm(raw_data.items(), desc="Engineering features"):
        try:
            if df.empty or len(df) < 100:  # Skip if insufficient data
                continue
                
            processed_df = create_features(df)
            
            if not processed_df.empty:
                processed_data[ticker] = processed_df
                
        except Exception as e:
            warnings.warn(f"Processing failed for {ticker}: {str(e)}")
            continue
            
    return processed_data

def save_processed_data(data: Dict[str, pd.DataFrame], directory: str = "processed_data"):
    """Save processed data to disk"""
    try:
        os.makedirs(directory, exist_ok=True)
        
        for ticker, df in data.items():
            try:
                clean_ticker = ticker.replace(".NS", "")
                filename = f"{clean_ticker}_processed.joblib"
                filepath = os.path.join(directory, filename)
                joblib.dump(df, filepath)
            except Exception as e:
                warnings.warn(f"Failed to save {ticker}: {str(e)}")
                
    except Exception as e:
        warnings.warn(f"Save operation failed: {str(e)}")

def load_processed_data(directory: str = "processed_data") -> Dict[str, pd.DataFrame]:
    """Load processed data from disk"""
    loaded_data = {}
    
    try:
        if not os.path.exists(directory):
            warnings.warn(f"Directory {directory} not found")
            return loaded_data
            
        for filename in os.listdir(directory):
            if filename.endswith("_processed.joblib"):
                try:
                    ticker_name = filename.replace("_processed.joblib", "")
                    ticker = f"{ticker_name}.NS"
                    filepath = os.path.join(directory, filename)
                    df = joblib.load(filepath)
                    loaded_data[ticker] = df
                except Exception as e:
                    warnings.warn(f"Failed to load {filename}: {str(e)}")
                    
    except Exception as e:
        warnings.warn(f"Load operation failed: {str(e)}")
    
    return loaded_data

# Utility functions
def get_feature_importance_summary(processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Get summary of feature importance across all tickers"""
    all_features = set()
    
    for df in processed_data.values():
        feature_cols = [col for col in df.columns if not col.startswith('Target_')]
        all_features.update(feature_cols)
    
    feature_summary = []
    for feature in all_features:
        count = sum(1 for df in processed_data.values() if feature in df.columns)
        coverage = count / len(processed_data)
        
        feature_summary.append({
            'feature': feature,
            'ticker_count': count,
            'coverage': coverage
        })
    
    return pd.DataFrame(feature_summary).sort_values('coverage', ascending=False)

def validate_data_quality(processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Validate quality of processed data"""
    quality_report = {
        'total_tickers': len(processed_data),
        'avg_features': 0,
        'avg_samples': 0,
        'missing_data_pct': 0,
        'feature_coverage': {}
    }
    
    if not processed_data:
        return quality_report
    
    total_features = sum(len(df.columns) for df in processed_data.values())
    total_samples = sum(len(df) for df in processed_data.values())
    total_missing = sum(df.isnull().sum().sum() for df in processed_data.values())
    total_cells = sum(df.size for df in processed_data.values())
    
    quality_report['avg_features'] = total_features / len(processed_data)
    quality_report['avg_samples'] = total_samples / len(processed_data)
    quality_report['missing_data_pct'] = (total_missing / total_cells) * 100
    
    # Feature coverage analysis
    feature_summary = get_feature_importance_summary(processed_data)
    quality_report['feature_coverage'] = {
        'high_coverage_features': len(feature_summary[feature_summary['coverage'] > 0.8]),
        'medium_coverage_features': len(feature_summary[
            (feature_summary['coverage'] > 0.5) & (feature_summary['coverage'] <= 0.8)
        ]),
        'low_coverage_features': len(feature_summary[feature_summary['coverage'] <= 0.5])
    }
    
    return quality_report

def monte_carlo_forecast(df: pd.DataFrame, horizon_days: int, n_simulations: int = 1000) -> np.ndarray:
    """Monte Carlo simulation for price forecasting"""
    if df.empty or 'Close' not in df.columns or len(df) < 2:
        return np.array([])
    
    # Calculate log returns
    log_returns = np.log(1 + df['Close'].pct_change().dropna())
    
    if len(log_returns) < 5:  # Require at least 5 observations
        return np.array([])
    
    # Compute mean and standard deviation of log returns
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    current_price = df['Close'].iloc[-1]
    
    # Vectorized simulation
    daily_returns = np.random.normal(mu, sigma, (n_simulations, horizon_days))
    cumulative_returns = np.exp(np.cumsum(daily_returns, axis=1))
    final_prices = current_price * cumulative_returns[:, -1]
    
    return final_prices

if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully")
    print(f"Available horizons: {list(HORIZONS.keys())}")
    print(f"Configuration: {FEATURE_CONFIG}")