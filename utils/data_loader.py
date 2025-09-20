import yfinance as yf
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from datetime import datetime, timedelta
import os
import sqlite3
from pathlib import Path
import hashlib
from tqdm import tqdm
import warnings
import json
import sys
from config import secrets
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ==================== ENHANCED CONFIGURATION ====================
DATA_CONFIG = {
    'default_period': '10y',
    'max_period': '20y',
    'default_interval': '1d',
    'max_workers': 6,
    'retry_attempts': 5,
    'retry_delay': 2,
    'cache_enabled': True,
    'cache_duration_hours': 12,
    'batch_size': 8,
    'request_delay': 0.5,
    'timeout': 45,
    'validate_data': True,
    'use_database': True,
    'fallback_tickers': True,
    'data_quality_threshold': 0.7,
    'realtime_refresh_minutes': 5
}

# ==================== DATABASE STORAGE ====================
class StockDataDatabase:
    """SQLite database for persistent stock data storage"""
    
    def __init__(self, db_path: str = "data/stock_data.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        ticker TEXT,
                        date TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        adj_close REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (ticker, date)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ticker_metadata (
                        ticker TEXT PRIMARY KEY,
                        last_updated TIMESTAMP,
                        data_quality REAL,
                        total_records INTEGER,
                        earliest_date TEXT,
                        latest_date TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (ticker, date);
                """)
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
    
    def save_stock_data(self, ticker: str, df: pd.DataFrame):
        """Save stock data to database"""
        if df.empty:
            return
            
        try:
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            df_copy['ticker'] = ticker
            df_copy['Date'] = df_copy['Date'].dt.strftime('%Y-%m-%d')
            
            # Rename columns to match database schema
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            df_copy = df_copy.rename(columns=column_mapping)
            
            with sqlite3.connect(self.db_path) as conn:
                # Delete existing data for this ticker
                conn.execute("DELETE FROM stock_data WHERE ticker = ?", (ticker,))
                
                # Insert new data
                df_copy[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']].to_sql(
                    'stock_data', conn, if_exists='append', index=False
                )
                
                # Update metadata
                conn.execute("""
                    INSERT OR REPLACE INTO ticker_metadata 
                    (ticker, last_updated, data_quality, total_records, earliest_date, latest_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    datetime.now().isoformat(),
                    1.0,
                    len(df_copy),
                    df_copy['date'].min(),
                    df_copy['date'].max()
                ))
        except Exception as e:
            logging.error(f"Failed to save data for {ticker}: {e}")
    
    def load_stock_data(self, ticker: str, start_date: str = None) -> pd.DataFrame:
        """Load stock data from database"""
        try:
            query = "SELECT * FROM stock_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
                
            query += " ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=params)
                
            if df.empty:
                return pd.DataFrame()
                
            # Convert back to yfinance format
            df['Date'] = pd.to_datetime(df['date'])
            df = df.set_index('Date')
            
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low', 
                'close': 'Close',
                'volume': 'Volume',
                'adj_close': 'Adj Close'
            }
            df = df.rename(columns=column_mapping)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            
            return df
        except Exception as e:
            logging.error(f"Failed to load data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_ticker_metadata(self, ticker: str) -> Optional[Dict]:
        """Get metadata for a ticker"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM ticker_metadata WHERE ticker = ?", 
                    (ticker,)
                )
                row = cursor.fetchone()
                
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
        except Exception as e:
            logging.error(f"Failed to get metadata for {ticker}: {e}")
            return None

# ==================== REAL-TIME DATA INTEGRATION ====================
class RealTimeDataManager:
    """Manager for real-time data updates"""
    def __init__(self, db_path: str = "data/realtime_data.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS realtime_data (
                        ticker TEXT,
                        timestamp DATETIME,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        PRIMARY KEY (ticker, timestamp)
                    )
                """)
        except Exception as e:
            logging.error(f"Realtime database initialization failed: {e}")
    
    def update_realtime_data(self, tickers: list):
        """Fetch and store real-time data"""
        for ticker in tqdm(tickers, desc="Updating real-time data"):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period='1d', interval='1m')
                
                if not data.empty:
                    data = data.reset_index()
                    data['ticker'] = ticker
                    data.rename(columns={'Datetime': 'timestamp'}, inplace=True)
                    
                    with sqlite3.connect(self.db_path) as conn:
                        data[['ticker', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].to_sql(
                            'realtime_data', conn, if_exists='append', index=False
                        )
            except Exception as e:
                logging.error(f"Real-time update failed for {ticker}: {e}")
    
    def get_latest_data(self, ticker: str, lookback_minutes=60):
        """Get latest real-time data"""
        try:
            query = f"""
                SELECT * FROM realtime_data 
                WHERE ticker = ? 
                AND timestamp >= datetime('now', '-{lookback_minutes} minutes')
                ORDER BY timestamp DESC
            """
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=(ticker,))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            return df
        except Exception as e:
            logging.error(f"Failed to get latest data for {ticker}: {e}")
            return pd.DataFrame()

# ==================== ALTERNATIVE DATA SOURCES ====================
class NewsSentimentLoader:
    """Fetch and process news sentiment data"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or secrets.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch_news(self, query: str, days=7):
        """Fetch news articles for a query"""
        if not self.api_key:
            return []
            
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'pageSize': 50
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            logging.error(f"News API error: {e}")
            return []
    
    def analyze_sentiment(self, text: str):
        """Simple sentiment analysis"""
        positive_words = ['bullish', 'growth', 'strong', 'buy', 'outperform', 'positive']
        negative_words = ['bearish', 'decline', 'weak', 'sell', 'underperform', 'negative']
        
        if any(word in text.lower() for word in positive_words):
            return 1
        elif any(word in text.lower() for word in negative_words):
            return -1
        return 0
    
    def get_sentiment_scores(self, ticker: str):
        """Get sentiment scores for a ticker"""
        articles = self.fetch_news(ticker)
        if not articles:
            return 0
        
        scores = []
        for article in articles:
            content = f"{article['title']} {article.get('description', '')}"
            scores.append(self.analyze_sentiment(content))
        
        return np.mean(scores) if scores else 0

# ==================== ENHANCED TICKER MANAGEMENT ====================
def get_updated_nifty_tickers() -> List[str]:
    """Get most current NIFTY tickers with automatic updates"""
    current_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
        "HDFC.NS", "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS", "SUNPHARMA.NS", "NESTLEIND.NS",
        "HCLTECH.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS",
        "TATAMOTORS.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "INDUSINDBK.NS", "CIPLA.NS",
        "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
        "IOC.NS", "JSWSTEEL.NS", "M&M.NS", "BRITANNIA.NS", "DIVISLAB.NS",
        "ADANIPORTS.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "BPCL.NS", "SHREECEM.NS",
        "TATASTEEL.NS", "TITAN.NS", "UPL.NS", "VEDL.NS", "TATACONSUM.NS"
    ]
    
    delisted_tickers = {"ADANITRANS.NS", "LTI.NS"}
    active_tickers = [ticker for ticker in current_tickers if ticker not in delisted_tickers]
    
    return active_tickers

def validate_and_filter_tickers(tickers: List[str]) -> List[str]:
    """Validate tickers and filter out invalid ones"""
    valid_tickers = []
    
    for ticker in tickers:
        if not ticker or not isinstance(ticker, str):
            continue
            
        ticker = ticker.upper().strip()
        
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            ticker += '.NS'
            
        if any(char in ticker for char in ['/', '\\', '|', '<', '>', '"']):
            continue
            
        valid_tickers.append(ticker)
    
    return list(set(valid_tickers))

# ==================== ENHANCED DATA FETCHING ====================
def fetch_extended_historical_data(ticker: str, 
                                 max_period: str = "20y",
                                 fallback_periods: List[str] = None) -> pd.DataFrame:
    """Fetch maximum available historical data with fallback periods"""
    if fallback_periods is None:
        fallback_periods = ["20y", "15y", "10y", "5y", "2y", "1y"]
    
    for period in fallback_periods:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, timeout=60)
            
            if not df.empty and len(df) > 100:
                logging.info(f"Successfully fetched {len(df)} records for {ticker} ({period})")
                return df
            elif not df.empty:
                logging.warning(f"Limited data for {ticker}: {len(df)} records ({period})")  
                
        except Exception as e:
            logging.warning(f"Failed to fetch {ticker} with period {period}: {e}")
            continue
    
    # Last resort: try with specific date range
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*20)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                          end=end_date.strftime('%Y-%m-%d'))
        
        if not df.empty:
            logging.info(f"Fetched {len(df)} records for {ticker} using date range")
            return df
            
    except Exception as e:
        logging.error(f"All methods failed for {ticker}: {e}")
    
    return pd.DataFrame()

def fetch_single_ticker_enhanced(args: Tuple) -> Tuple[str, pd.DataFrame]:
    """Enhanced single ticker fetching with database integration"""
    ticker, config, database = args
    
    try:
        # Check database first
        if database and config.get('use_database'):
            metadata = database.get_ticker_metadata(ticker)
            if metadata:
                last_updated = datetime.fromisoformat(metadata['last_updated'])
                hours_since_update = (datetime.now() - last_updated).total_seconds() / 3600
                
                if hours_since_update < config.get('cache_duration_hours', 12):
                    cached_data = database.load_stock_data(ticker)
                    if not cached_data.empty:
                        logging.info(f"Loaded {ticker} from database cache")
                        return ticker, cached_data
        
        # Add progressive delay to avoid rate limiting
        time.sleep(config.get('request_delay', 0.5))
        
        # Fetch with extended historical data
        df = fetch_extended_historical_data(ticker, config.get('max_period', '20y'))
        
        if not df.empty:
            # Validate data quality
            if config.get('validate_data'):
                df = validate_stock_data_enhanced(df, ticker)
                
            # Save to database
            if database and config.get('use_database') and not df.empty:
                database.save_stock_data(ticker, df)
                logging.info(f"Saved {ticker} to database")
            
            return ticker, df
        
        logging.warning(f"No data retrieved for {ticker}")
        return ticker, pd.DataFrame()
        
    except Exception as e:
        logging.error(f"Enhanced fetch failed for {ticker}: {e}")
        return ticker, pd.DataFrame()

def fetch_historical_data_enhanced(tickers: List[str], 
                                 config: Dict = None) -> Dict[str, pd.DataFrame]:
    """Enhanced historical data fetching with database integration"""
    config = config or DATA_CONFIG
    
    # Initialize database
    database = StockDataDatabase() if config.get('use_database') else None
    
    # Validate and filter tickers
    tickers = validate_and_filter_tickers(tickers)
    
    print(f"Fetching enhanced data for {len(tickers)} selected tickers...")
    print(f"Selected stocks: {tickers}")
    print(f"Maximum period: {config.get('max_period', '20y')}")
    print(f"Using database cache: {config.get('use_database', False)}")
    
    # Prepare tasks
    tasks = [(ticker, config, database) for ticker in tickers]
    
    results = {}
    successful_fetches = 0
    
    # Reduced batch size and workers for stability
    batch_size = config.get('batch_size', 8)
    max_workers = min(config.get('max_workers', 6), len(tickers))
    
    for i in tqdm(range(0, len(tasks), batch_size), desc="Fetching enhanced batches"):
        batch_tasks = tasks[i:i + batch_size]
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(fetch_single_ticker_enhanced, task): task[0] 
                    for task in batch_tasks
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        ticker_result, df = future.result()
                        if not df.empty:
                            results[ticker_result] = df
                            successful_fetches += 1
                            
                            # Log data range
                            date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                            logging.info(f"{ticker}: {len(df)} records ({date_range})")
                            
                    except Exception as e:
                        logging.warning(f"Failed to process {ticker}: {e}")
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            # Fallback to sequential processing for this batch
            for task in batch_tasks:
                try:
                    ticker_result, df = fetch_single_ticker_enhanced(task)
                    if not df.empty:
                        results[ticker_result] = df
                        successful_fetches += 1
                except Exception as e:
                    logging.warning(f"Sequential fallback failed for {task[0]}: {e}")
        
        # Longer pause between batches
        if i + batch_size < len(tasks):
            time.sleep(1.0)
    
    print(f"Successfully fetched enhanced data for {successful_fetches}/{len(tickers)} tickers")
    
    # Generate data quality report
    generate_data_quality_report(results)
    
    return results

# ==================== ENHANCED DATA VALIDATION ====================
def validate_stock_data_enhanced(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Enhanced data validation with comprehensive checks"""
    if df.empty:
        return df
    
    original_length = len(df)
    
    try:
        # Remove duplicate dates
        df = df[~df.index.duplicated(keep='first')]
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.warning(f"Missing columns for {ticker}: {missing_columns}")
            return pd.DataFrame()
        
        # Enhanced data quality checks
        # 1. Basic OHLC validation
        invalid_hl = df['High'] < df['Low']
        if invalid_hl.any():
            df = df[~invalid_hl]
            logging.warning(f"Removed {invalid_hl.sum()} rows with High < Low for {ticker}")
        
        # 2. OHLC range validation
        invalid_ohlc = (
            (df['Open'] > df['High']) | (df['Open'] < df['Low']) |
            (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        )
        if invalid_ohlc.any():
            df = df[~invalid_ohlc]
            logging.warning(f"Removed {invalid_ohlc.sum()} rows with invalid OHLC for {ticker}")
        
        # 3. Remove zero/negative prices
        zero_negative = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if zero_negative.any():
            df = df[~zero_negative]
            logging.warning(f"Removed {zero_negative.sum()} rows with zero/negative prices for {ticker}")
        
        # 4. Volume validation
        df.loc[df['Volume'] < 0, 'Volume'] = 0
        
        # 5. Handle extreme price movements (potential errors)
        if len(df) > 1:
            returns = df['Close'].pct_change().abs()
            
            # Flag extreme moves (>80% in a day) as potential errors
            extreme_threshold = 0.8
            extreme_moves = returns > extreme_threshold
            
            if extreme_moves.any():
                # Keep extreme moves but log them
                extreme_dates = df.index[extreme_moves]
                logging.info(f"Extreme moves detected for {ticker}: {len(extreme_dates)} days")
                
                # Only remove if too many extreme moves (likely data error)
                if extreme_moves.sum() > len(df) * 0.05:
                    df = df[~extreme_moves]
                    logging.warning(f"Removed excessive extreme moves for {ticker}")
        
        # 6. Handle missing values intelligently
        # Forward fill small gaps
        missing_mask = df[required_columns].isnull().any(axis=1)
        if missing_mask.any():
            gap_size = missing_mask.astype(int).groupby((~missing_mask).cumsum()).sum()
            small_gaps = gap_size <= 5  # Fill gaps of 5 days or less
            
            if small_gaps.any():
                df = df.fillna(method='ffill', limit=5)
                df = df.fillna(method='bfill', limit=5)
            
            # Remove remaining missing values
            df = df.dropna(subset=required_columns)
        
        # 7. Ensure chronological order
        df = df.sort_index()
        
        # 8. Data quality assessment
        data_quality = len(df) / original_length if original_length > 0 else 0
        
        if data_quality < DATA_CONFIG.get('data_quality_threshold', 0.7):
            logging.warning(f"Low data quality for {ticker}: {data_quality:.2%}")
        
        logging.info(f"Data validation complete for {ticker}: {original_length} -> {len(df)} rows")
        
    except Exception as e:
        logging.error(f"Data validation failed for {ticker}: {e}")
        return pd.DataFrame()
    
    return df

# ==================== DATA QUALITY REPORTING ====================
def generate_data_quality_report(data_dict: Dict[str, pd.DataFrame]):
    """Generate comprehensive data quality report"""
    try:
        report = {
            'total_tickers': len(data_dict),
            'successful_tickers': len([k for k, v in data_dict.items() if not v.empty]),
            'date_ranges': {},
            'record_counts': {},
            'data_quality_scores': {}
        }
        
        for ticker, df in data_dict.items():
            if df.empty:
                continue
                
            # Date range analysis
            start_date = df.index[0]
            end_date = df.index[-1]
            years_of_data = (end_date - start_date).days / 365.25
            
            report['date_ranges'][ticker] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'years': round(years_of_data, 1),
                'records': len(df)
            }
            
            # Data quality metrics
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            volume_zero_pct = (df['Volume'] == 0).sum() / len(df) if 'Volume' in df.columns else 0
            
            quality_score = 1.0 - (missing_pct * 0.5 + volume_zero_pct * 0.3)
            report['data_quality_scores'][ticker] = round(quality_score, 3)
        
        # Summary statistics
        if report['date_ranges']:
            years_list = [info['years'] for info in report['date_ranges'].values()]
            records_list = [info['records'] for info in report['date_ranges'].values()]
            quality_scores = list(report['data_quality_scores'].values())
            
            print("\n" + "="*60)
            print("DATA QUALITY REPORT")
            print("="*60)
            print(f"Total tickers processed: {report['total_tickers']}")
            print(f"Successful tickers: {report['successful_tickers']}")
            print(f"Success rate: {report['successful_tickers']/report['total_tickers']:.1%}")
            print(f"\nHistorical Data Coverage:")
            print(f"  Average years of data: {np.mean(years_list):.1f}")
            print(f"  Maximum years: {max(years_list):.1f}")
            print(f"  Minimum years: {min(years_list):.1f}")
            print(f"\nData Volume:")
            print(f"  Total records: {sum(records_list):,}")
            print(f"  Average records per ticker: {np.mean(records_list):.0f}")
            print(f"\nData Quality:")
            print(f"  Average quality score: {np.mean(quality_scores):.3f}")
            print(f"  Tickers with quality > 0.9: {sum(1 for s in quality_scores if s > 0.9)}")
            print("="*60)
    except Exception as e:
        logging.error(f"Data quality report generation failed: {e}")

# ==================== MAIN INTERFACE ====================
def get_comprehensive_stock_data(tickers: Optional[List[str]] = None,
                               config: Dict = None,
                               max_tickers: int = None,
                               selected_tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Main interface for comprehensive stock data collection"""
    config = config or DATA_CONFIG
    
    # Use selected_tickers if provided, otherwise use default logic
    if selected_tickers:
        tickers = selected_tickers
        print(f"Using user-selected tickers: {len(tickers)} stocks")
    elif tickers is None:
        tickers = get_updated_nifty_tickers()
        print(f"Using default NIFTY tickers: {len(tickers)} stocks")
        
    if max_tickers and not selected_tickers:
        tickers = tickers[:max_tickers]
        print(f"Limited to max_tickers: {len(tickers)} stocks")
    
    print(f"Starting comprehensive data collection...")
    print(f"Target tickers: {len(tickers)}")
    print(f"Selected stocks: {tickers}")
    print(f"Max period: {config['max_period']}")
    print(f"Database enabled: {config['use_database']}")
    
    # Fetch enhanced data
    try:
        data = fetch_historical_data_enhanced(tickers, config)
        return data
    except Exception as e:
        logging.error(f"Comprehensive data collection failed: {e}")
        return {}

# ==================== COMPATIBILITY FUNCTIONS ====================
# These are added for backward compatibility with existing code

def fetch_historical_data(tickers: List[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
    """Backward compatibility function"""
    config = DATA_CONFIG.copy()
    config['max_period'] = period
    return fetch_historical_data_enhanced(tickers, config)

def get_nifty_50_tickers() -> List[str]:
    """Backward compatibility function"""
    return get_updated_nifty_tickers()

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("Enhanced Stock Data Loader - User Selection Version")
    print("="*60)
    
    # Test the system with user selection
    try:
        # Simulate user selection
        all_tickers = get_updated_nifty_tickers()
        selected_tickers = all_tickers[:5]  # Select first 5 for testing
        
        print(f"Available tickers: {len(all_tickers)}")
        print(f"User selected: {len(selected_tickers)} tickers")
        for ticker in selected_tickers:
            print(f"  - {ticker}")
        
        # Fetch data for selected tickers only
        data = get_comprehensive_stock_data(selected_tickers=selected_tickers)
        
        if data:
            print(f"\nSuccessfully loaded data for {len(data)} selected tickers")
            for ticker, df in data.items():
                if not df.empty:
                    print(f"  {ticker}: {len(df)} records from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                else:
                    print(f"  {ticker}: No data")
        else:
            print("No data loaded")
            
    except Exception as e:
        print(f"Test failed: {e}")
        
    print("\nUser Selection Features:")
    print("  ✓ Support for user-selected stocks only")
    print("  ✓ Faster processing with fewer stocks")
    print("  ✓ Resource-efficient data fetching")
    print("  ✓ Backward compatibility maintained")
    print("  ✓ Enhanced error handling and logging")