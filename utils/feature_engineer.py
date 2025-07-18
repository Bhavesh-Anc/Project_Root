# utils/feature_engineer.py
import pandas as pd
import numpy as np
import warnings
import joblib
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import gc

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
    """Calculate comprehensive technical indicators without TA-Lib"""
    if df.empty or len(df) < 50:
        return df
    
    try:
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return df
            
        # ===== Moving Averages =====
        # Simple Moving Averages
        for period in FEATURE_CONFIG['technical_indicators']['sma_periods']:
            if len(df) > period:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'Price_SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
        
        # Exponential Moving Averages
        for period in FEATURE_CONFIG['technical_indicators']['ema_periods']:
            if len(df) > period:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                df[f'Price_EMA_{period}_Ratio'] = df['Close'] / df[f'EMA_{period}']
        
        # ===== RSI (Relative Strength Index) =====
        rsi_period = FEATURE_CONFIG['technical_indicators']['rsi_period']
        if len(df) > rsi_period:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            for i in range(rsi_period, len(df)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
            df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        
        # ===== MACD (Moving Average Convergence Divergence) =====
        fast = FEATURE_CONFIG['technical_indicators']['macd_fast']
        slow = FEATURE_CONFIG['technical_indicators']['macd_slow']
        signal = FEATURE_CONFIG['technical_indicators']['macd_signal']
        
        if len(df) > slow:
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            
            df['MACD'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        # ===== Bollinger Bands =====
        bb_period = FEATURE_CONFIG['technical_indicators']['bb_period']
        bb_std = FEATURE_CONFIG['technical_indicators']['bb_std']
        if len(df) > bb_period:
            sma = df['Close'].rolling(window=bb_period).mean()
            std = df['Close'].rolling(window=bb_period).std()
            
            df['BB_Upper'] = sma + (std * bb_std)
            df['BB_Middle'] = sma
            df['BB_Lower'] = sma - (std * bb_std)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ===== ATR (Average True Range) =====
        atr_period = FEATURE_CONFIG['technical_indicators']['atr_period']
        if len(df) > atr_period:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=atr_period).mean()
            df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # ===== Stochastic Oscillator =====
        k_period = FEATURE_CONFIG['technical_indicators']['stoch_k']
        d_period = FEATURE_CONFIG['technical_indicators']['stoch_d']
        if len(df) > k_period:
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            
            df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        # ===== Williams %R =====
        williams_period = FEATURE_CONFIG['technical_indicators']['williams_r']
        if len(df) > williams_period:
            highest_high = df['High'].rolling(window=williams_period).max()
            lowest_low = df['Low'].rolling(window=williams_period).min()
            
            df['Williams_R'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        
        # ===== Volume indicators =====
        if len(df) > 20:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            df['Price_Volume'] = df['Close'] * df['Volume']
        
        # ===== Momentum indicators =====
        if len(df) > 10:
            df['MOM_10'] = df['Close'].diff(10)
            df['ROC_10'] = (df['Close'].diff(10) / df['Close'].shift(10)) * 100
        
        # ===== CCI (Commodity Channel Index) =====
        if len(df) > 20:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            
            df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # ===== MFI (Money Flow Index) =====
        if len(df) > 14:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            rmf = tp * df['Volume']
            
            pmf = np.where(tp > tp.shift(), rmf, 0)
            nmf = np.where(tp < tp.shift(), rmf, 0)
            
            pmf_sum = pmf.rolling(window=14).sum()
            nmf_sum = nmf.rolling(window=14).sum()
            
            mfr = pmf_sum / nmf_sum
            df['MFI'] = 100 - (100 / (1 + mfr))
        
    except Exception as e:
        warnings.warn(f"Technical indicators calculation failed: {str(e)}")
    
    return df

def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price-based features"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price gaps
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Up'] = (df['Gap'] > 0.02).astype(int)
        df['Gap_Down'] = (df['Gap'] < -0.02).astype(int)
        
        # Intraday patterns
        df['Intraday_Return'] = (df['Close'] - df['Open']) / df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / np.minimum(df['Open'], df['Close'])
        
        # Price patterns
        df['Doji'] = (abs(df['Close'] - df['Open']) < 0.001).astype(int)
        df['Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body_Size']) & 
                       (df['Upper_Shadow'] < df['Body_Size'])).astype(int)
        df['Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body_Size']) & 
                              (df['Lower_Shadow'] < df['Body_Size'])).astype(int)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            if len(df) > window:
                df[f'Returns_Mean_{window}'] = df['Returns'].rolling(window).mean()
                df[f'Returns_Std_{window}'] = df['Returns'].rolling(window).std()
                df[f'Returns_Skew_{window}'] = df['Returns'].rolling(window).skew()
                df[f'Returns_Kurt_{window}'] = df['Returns'].rolling(window).kurt()
                df[f'Price_Min_{window}'] = df['Close'].rolling(window).min()
                df[f'Price_Max_{window}'] = df['Close'].rolling(window).max()
                df[f'Price_Position_{window}'] = (df['Close'] - df[f'Price_Min_{window}']) / (df[f'Price_Max_{window}'] - df[f'Price_Min_{window}'])
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(5).std() * np.sqrt(252)
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['Volatility_Ratio'] = df['Volatility_5'] / df['Volatility_20']
        
    except Exception as e:
        warnings.warn(f"Price features calculation failed: {str(e)}")
    
    return df

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk-based features"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        returns = df['Returns'].dropna()
        
        # Value at Risk
        window = FEATURE_CONFIG['risk_metrics']['volatility_window']
        confidence = FEATURE_CONFIG['risk_metrics']['var_confidence']
        
        if len(returns) > window:
            df['VaR'] = returns.rolling(window).quantile(confidence)
            df['CVaR'] = returns.rolling(window).apply(lambda x: x[x <= x.quantile(confidence)].mean())
        
        # Maximum Drawdown
        rolling_max = df['Close'].rolling(window=252, min_periods=1).max()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        df['Drawdown'] = drawdown
        df['Max_Drawdown'] = drawdown.rolling(window=252, min_periods=1).min()
        
        # Sharpe Ratio (rolling)
        if len(returns) > 252:
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
            df['Sharpe_Ratio'] = rolling_sharpe
        
        # Beta (relative to market - using own returns as proxy)
        if len(returns) > 60:
            market_returns = returns.rolling(60).mean()
            covariance = returns.rolling(60).cov(market_returns)
            market_variance = market_returns.rolling(60).var()
            df['Beta'] = covariance / market_variance
        
        # Sortino Ratio
        if len(returns) > 252:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.rolling(252).std()
                df['Sortino_Ratio'] = returns.rolling(252).mean() / downside_std * np.sqrt(252)
        
    except Exception as e:
        warnings.warn(f"Risk metrics calculation failed: {str(e)}")
    
    return df

def calculate_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market regime indicators"""
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        # Trend identification
        trend_window = FEATURE_CONFIG['market_regime']['trend_window']
        if len(df) > trend_window:
            df['Trend_SMA'] = df['Close'].rolling(trend_window).mean()
            df['Trend_Direction'] = (df['Close'] > df['Trend_SMA']).astype(int)
            df['Trend_Strength'] = abs(df['Close'] - df['Trend_SMA']) / df['Trend_SMA']
        
        # Volatility regime
        vol_window = FEATURE_CONFIG['market_regime']['volatility_regime_window']
        if len(df) > vol_window:
            vol_rolling = df['Returns'].rolling(vol_window).std()
            vol_threshold = vol_rolling.quantile(0.7)
            df['High_Vol_Regime'] = (vol_rolling > vol_threshold).astype(int)
        
        # Price momentum
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
                df[f'Momentum_Rank_{period}'] = df[f'Momentum_{period}'].rolling(60).rank(pct=True)
        
        # Support and resistance levels
        if len(df) > 20:
            df['Support_20'] = df['Low'].rolling(20).min()
            df['Resistance_20'] = df['High'].rolling(20).max()
            df['Support_Distance'] = (df['Close'] - df['Support_20']) / df['Close']
            df['Resistance_Distance'] = (df['Resistance_20'] - df['Close']) / df['Close']
        
    except Exception as e:
        warnings.warn(f"Market regime features calculation failed: {str(e)}")
    
    return df

def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features"""
    if df.empty or 'Volume' not in df.columns:
        return df
    
    try:
        # Volume trends
        for period in [5, 10, 20]:
            if len(df) > period:
                df[f'Volume_SMA_{period}'] = df['Volume'].rolling(period).mean()
                df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_SMA_{period}']
        
        # Volume patterns
        df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)
        df['Volume_Dry'] = (df['Volume'] < df['Volume'].rolling(20).mean() * 0.5).astype(int)
        
        # Price-Volume relationship
        df['PV_Trend'] = ((df['Close'] > df['Close'].shift(1)) & 
                         (df['Volume'] > df['Volume'].shift(1))).astype(int)
        df['Volume_Price_Correlation'] = df['Volume'].rolling(20).corr(df['Close'])
        
        # On-Balance Volume
        df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
        df['OBV_Slope'] = df['OBV'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
    except Exception as e:
        warnings.warn(f"Volume features calculation failed: {str(e)}")
    
    return df

def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based features"""
    if df.empty:
        return df
    
    try:
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Basic time features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfYear'] = df.index.dayofyear
        
        # Market timing features
        df['Monday'] = (df['DayOfWeek'] == 0).astype(int)
        df['Friday'] = (df['DayOfWeek'] == 4).astype(int)
        df['MonthEnd'] = df.index.is_month_end.astype(int)
        df['MonthStart'] = df.index.is_month_start.astype(int)
        df['QuarterEnd'] = df.index.is_quarter_end.astype(int)
        df['YearEnd'] = df.index.is_year_end.astype(int)
        
        # Seasonal patterns
        df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
        df['Sin_DayOfWeek'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['Cos_DayOfWeek'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
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
                volatility = df['Returns'].rolling(20).std()
                risk_adj_return = returns / (volatility * np.sqrt(days))
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
        max_drawdown = abs(drawdown.min())
        dd_score = min(max_drawdown / 0.5, 1.0)
        
        # Volume volatility
        if 'Volume' in df.columns:
            volume_returns = df['Volume'].pct_change().dropna()
            if len(volume_returns) > 0:
                volume_vol = volume_returns.std()
                vol_vol_score = min(volume_vol / 2.0, 1.0)
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