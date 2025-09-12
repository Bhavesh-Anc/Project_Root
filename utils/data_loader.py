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
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
import threading
from queue import Queue
import pickle
from config import secrets, DATABASE_CONFIG

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== COMPREHENSIVE CONFIGURATION ====================
@dataclass
class DataSourceConfig:
    """Configuration for different data sources"""
    primary_source: str = "yfinance"
    fallback_sources: List[str] = None
    max_retries: int = 5
    retry_delay: float = 2.0
    timeout: int = 60
    rate_limit: float = 0.1
    
    def __post_init__(self):
        if self.fallback_sources is None:
            self.fallback_sources = ["alpha_vantage", "financial_modeling_prep", "polygon"]

COMPREHENSIVE_DATA_CONFIG = {
    'default_period': '20y',
    'max_period': '25y',
    'intervals': ['1d', '1wk', '1mo'],
    'default_interval': '1d',
    'max_workers': 8,
    'retry_attempts': 7,
    'retry_delay': 2,
    'exponential_backoff': True,
    'cache_enabled': True,
    'cache_duration_hours': 24,
    'batch_size': 12,
    'request_delay': 0.3,
    'timeout': 90,
    'validate_data': True,
    'use_database': True,
    'use_async': True,
    'fallback_tickers': True,
    'data_quality_threshold': 0.75,
    'realtime_refresh_minutes': 1,
    'enable_multi_timeframe': True,
    'enable_options_data': True,
    'enable_insider_data': True,
    'enable_earnings_data': True,
    'enable_analyst_data': True,
    'enable_sector_data': True,
    'enable_macro_data': True,
    'data_compression': True,
    'backup_enabled': True,
    'data_encryption': False,
    'auto_cleanup': True,
    'max_cache_size_gb': 5.0
}

# ==================== ADVANCED DATABASE SYSTEM ====================
class AdvancedStockDatabase:
    """Advanced SQLite database with comprehensive features"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DATABASE_CONFIG['stock_data_db']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.connection_pool = Queue(maxsize=5)
        self.init_database()
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup connection pool for better performance"""
        for _ in range(5):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety and speed
            conn.execute("PRAGMA cache_size=10000")  # 10MB cache
            conn.execute("PRAGMA temp_store=memory")  # Store temp tables in memory
            self.connection_pool.put(conn)
    
    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.get()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.put(conn)
    
    def init_database(self):
        """Initialize comprehensive database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Main stock data table
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
                    dividends REAL,
                    stock_splits REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT DEFAULT 'yfinance',
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # Multi-timeframe data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data_weekly (
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
                CREATE TABLE IF NOT EXISTS stock_data_monthly (
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
            
            # Options data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_data (
                    ticker TEXT,
                    expiration_date TEXT,
                    strike REAL,
                    option_type TEXT,
                    last_price REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility REAL,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, expiration_date, strike, option_type)
                )
            """)
            
            # Earnings data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings_data (
                    ticker TEXT,
                    quarter TEXT,
                    year INTEGER,
                    earnings_date TEXT,
                    reported_eps REAL,
                    estimated_eps REAL,
                    surprise_percent REAL,
                    revenue REAL,
                    estimated_revenue REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, quarter, year)
                )
            """)
            
            # Analyst recommendations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_data (
                    ticker TEXT,
                    date TEXT,
                    analyst_firm TEXT,
                    rating TEXT,
                    price_target REAL,
                    previous_rating TEXT,
                    previous_target REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, date, analyst_firm)
                )
            """)
            
            # Insider trading data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insider_data (
                    ticker TEXT,
                    date TEXT,
                    insider_name TEXT,
                    title TEXT,
                    transaction_type TEXT,
                    shares_traded INTEGER,
                    price REAL,
                    total_value REAL,
                    shares_owned_after INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sector and industry data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sector_data (
                    ticker TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    enterprise_value REAL,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    debt_to_equity REAL,
                    roe REAL,
                    roa REAL,
                    current_ratio REAL,
                    quick_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker)
                )
            """)
            
            # News and sentiment data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    ticker TEXT,
                    date TEXT,
                    headline TEXT,
                    summary TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    relevance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Economic indicators
            conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    indicator_name TEXT,
                    date TEXT,
                    value REAL,
                    frequency TEXT,
                    unit TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (indicator_name, date)
                )
            """)
            
            # Enhanced metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_metadata (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    exchange TEXT,
                    currency TEXT,
                    country TEXT,
                    last_updated TIMESTAMP,
                    data_quality REAL,
                    total_records INTEGER,
                    earliest_date TEXT,
                    latest_date TEXT,
                    data_completeness REAL,
                    volatility_30d REAL,
                    volume_avg_30d REAL,
                    market_cap REAL,
                    is_active BOOLEAN DEFAULT TRUE,
                    data_sources TEXT,
                    last_earnings_date TEXT,
                    next_earnings_date TEXT
                )
            """)
            
            # Create comprehensive indices
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (ticker, date)",
                "CREATE INDEX IF NOT EXISTS idx_date ON stock_data (date)",
                "CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data (ticker)",
                "CREATE INDEX IF NOT EXISTS idx_volume ON stock_data (volume)",
                "CREATE INDEX IF NOT EXISTS idx_ticker_weekly ON stock_data_weekly (ticker, date)",
                "CREATE INDEX IF NOT EXISTS idx_ticker_monthly ON stock_data_monthly (ticker, date)",
                "CREATE INDEX IF NOT EXISTS idx_options_ticker ON options_data (ticker)",
                "CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_data (ticker)",
                "CREATE INDEX IF NOT EXISTS idx_analyst_ticker ON analyst_data (ticker)",
                "CREATE INDEX IF NOT EXISTS idx_insider_ticker ON insider_data (ticker)",
                "CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_sentiment (ticker, date)",
                "CREATE INDEX IF NOT EXISTS idx_economic_date ON economic_indicators (date)"
            ]
            
            for index in indices:
                conn.execute(index)
    
    def save_stock_data_comprehensive(self, ticker: str, df: pd.DataFrame, 
                                    interval: str = '1d', source: str = 'yfinance'):
        """Save comprehensive stock data with metadata"""
        if df.empty:
            return
        
        conn = self.get_connection()
        try:
            df_copy = df.copy()
            df_copy.reset_index(inplace=True)
            df_copy['ticker'] = ticker
            df_copy['data_source'] = source
            
            # Handle different intervals
            table_name = {
                '1d': 'stock_data',
                '1wk': 'stock_data_weekly', 
                '1mo': 'stock_data_monthly'
            }.get(interval, 'stock_data')
            
            # Standardize column names
            column_mapping = {
                'Date': 'date',
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }
            
            # Handle date formatting
            if 'Date' in df_copy.columns:
                df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.strftime('%Y-%m-%d')
            elif 'Datetime' in df_copy.columns:
                df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime']).dt.strftime('%Y-%m-%d')
            
            df_copy = df_copy.rename(columns=column_mapping)
            
            # Fill missing optional columns
            optional_columns = ['dividends', 'stock_splits']
            for col in optional_columns:
                if col not in df_copy.columns:
                    df_copy[col] = 0.0
            
            # Insert data with conflict resolution
            df_copy.to_sql(table_name, conn, if_exists='replace', index=False, method='multi')
            
            # Update comprehensive metadata
            self._update_ticker_metadata_comprehensive(conn, ticker, df_copy)
            
            conn.commit()
            logging.info(f"Saved {len(df_copy)} records for {ticker} ({interval}) from {source}")
            
        except Exception as e:
            logging.error(f"Failed to save data for {ticker}: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def _update_ticker_metadata_comprehensive(self, conn, ticker: str, df: pd.DataFrame):
        """Update comprehensive ticker metadata"""
        try:
            # Calculate advanced metrics
            data_quality = self._calculate_data_quality(df)
            data_completeness = (len(df.dropna()) / len(df)) if len(df) > 0 else 0
            
            if 'close' in df.columns and len(df) > 30:
                returns = df['close'].pct_change().dropna()
                volatility_30d = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else 0
            else:
                volatility_30d = 0
            
            if 'volume' in df.columns and len(df) > 30:
                volume_avg_30d = df['volume'].tail(30).mean() if len(df) >= 30 else 0
            else:
                volume_avg_30d = 0
            
            # Get company info if available
            try:
                stock_info = yf.Ticker(ticker).info
                company_name = stock_info.get('longName', ticker)
                sector = stock_info.get('sector', 'Unknown')
                industry = stock_info.get('industry', 'Unknown')
                market_cap = stock_info.get('marketCap', 0)
                exchange = stock_info.get('exchange', 'NSE')
                currency = stock_info.get('currency', 'INR')
                country = stock_info.get('country', 'India')
            except:
                company_name = ticker
                sector = industry = 'Unknown'
                market_cap = 0
                exchange = 'NSE'
                currency = 'INR'
                country = 'India'
            
            conn.execute("""
                INSERT OR REPLACE INTO ticker_metadata 
                (ticker, company_name, sector, industry, exchange, currency, country,
                 last_updated, data_quality, total_records, earliest_date, latest_date,
                 data_completeness, volatility_30d, volume_avg_30d, market_cap, 
                 is_active, data_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, company_name, sector, industry, exchange, currency, country,
                datetime.now().isoformat(), data_quality, len(df),
                df['date'].min(), df['date'].max(), data_completeness,
                volatility_30d, volume_avg_30d, market_cap, True, 'yfinance'
            ))
            
        except Exception as e:
            logging.warning(f"Failed to update metadata for {ticker}: {e}")
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate comprehensive data quality score"""
        if df.empty:
            return 0.0
        
        quality_factors = []
        
        # Completeness check
        completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        quality_factors.append(completeness * 0.4)
        
        # OHLC consistency check
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_valid = (
                (df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close'])
            ).mean()
            quality_factors.append(ohlc_valid * 0.3)
        
        # Volume consistency check
        if 'volume' in df.columns:
            volume_valid = (df['volume'] >= 0).mean()
            quality_factors.append(volume_valid * 0.2)
        
        # Time continuity check
        if 'date' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('date')
            date_diff = pd.to_datetime(df_sorted['date']).diff().dt.days.dropna()
            continuity_score = (date_diff <= 7).mean()  # Allow for weekends
            quality_factors.append(continuity_score * 0.1)
        
        return sum(quality_factors)
    
    def load_stock_data_comprehensive(self, ticker: str, start_date: str = None,
                                    end_date: str = None, interval: str = '1d') -> pd.DataFrame:
        """Load comprehensive stock data with advanced filtering"""
        table_name = {
            '1d': 'stock_data',
            '1wk': 'stock_data_weekly',
            '1mo': 'stock_data_monthly'
        }.get(interval, 'stock_data')
        
        query = f"SELECT * FROM {table_name} WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        conn = self.get_connection()
        try:
            df = pd.read_sql(query, conn, params=params)
        except Exception as e:
            logging.warning(f"Failed to load data for {ticker}: {e}")
            return pd.DataFrame()
        finally:
            self.return_connection(conn)
            
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
            'adj_close': 'Adj Close',
            'dividends': 'Dividends',
            'stock_splits': 'Stock Splits'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Select and order columns properly
        base_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        optional_columns = ['Dividends', 'Stock Splits']
        
        available_columns = [col for col in base_columns if col in df.columns]
        available_columns.extend([col for col in optional_columns if col in df.columns])
        
        return df[available_columns]
    
    def save_options_data(self, ticker: str, options_df: pd.DataFrame):
        """Save options chain data"""
        if options_df.empty:
            return
        
        conn = self.get_connection()
        try:
            options_df['ticker'] = ticker
            options_df.to_sql('options_data', conn, if_exists='append', index=False)
            conn.commit()
            logging.info(f"Saved {len(options_df)} options records for {ticker}")
        except Exception as e:
            logging.error(f"Failed to save options data for {ticker}: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def save_earnings_data(self, ticker: str, earnings_df: pd.DataFrame):
        """Save earnings data"""
        if earnings_df.empty:
            return
        
        conn = self.get_connection()
        try:
            earnings_df['ticker'] = ticker
            earnings_df.to_sql('earnings_data', conn, if_exists='replace', index=False)
            conn.commit()
            logging.info(f"Saved earnings data for {ticker}")
        except Exception as e:
            logging.error(f"Failed to save earnings data for {ticker}: {e}")
            conn.rollback()
        finally:
            self.return_connection(conn)
    
    def get_ticker_metadata_comprehensive(self, ticker: str) -> Optional[Dict]:
        """Get comprehensive metadata for a ticker"""
        conn = self.get_connection()
        try:
            cursor = conn.execute("SELECT * FROM ticker_metadata WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
        except Exception as e:
            logging.warning(f"Failed to get metadata for {ticker}: {e}")
        finally:
            self.return_connection(conn)
        return None
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to manage database size"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        
        conn = self.get_connection()
        try:
            # Clean up old news sentiment data
            conn.execute("DELETE FROM news_sentiment WHERE date < ?", (cutoff_date,))
            
            # Clean up old insider trading data
            conn.execute("DELETE FROM insider_data WHERE date < ?", (cutoff_date,))
            
            # Vacuum database to reclaim space
            conn.execute("VACUUM")
            conn.commit()
            logging.info(f"Cleaned up data older than {cutoff_date}")
        except Exception as e:
            logging.error(f"Database cleanup failed: {e}")
        finally:
            self.return_connection(conn)

# ==================== MULTI-SOURCE DATA FETCHER ====================
class MultiSourceDataFetcher:
    """Fetch data from multiple sources with intelligent fallback"""
    
    def __init__(self):
        self.sources = {
            'yfinance': self._fetch_yfinance,
            'alpha_vantage': self._fetch_alpha_vantage,
            'financial_modeling_prep': self._fetch_fmp,
            'polygon': self._fetch_polygon
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Stock-Advisor-Pro/2.0'
        })
    
    def fetch_with_fallback(self, ticker: str, period: str = '10y', 
                          interval: str = '1d') -> pd.DataFrame:
        """Fetch data with intelligent fallback between sources"""
        
        for source_name, fetch_func in self.sources.items():
            try:
                logging.info(f"Attempting to fetch {ticker} from {source_name}")
                df = fetch_func(ticker, period, interval)
                
                if not df.empty and len(df) > 50:
                    logging.info(f"Successfully fetched {ticker} from {source_name}: {len(df)} records")
                    df.attrs['source'] = source_name
                    return df
                else:
                    logging.warning(f"Insufficient data from {source_name} for {ticker}")
                    
            except Exception as e:
                logging.warning(f"Failed to fetch {ticker} from {source_name}: {e}")
                continue
        
        logging.error(f"All sources failed for {ticker}")
        return pd.DataFrame()
    
    def _fetch_yfinance(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, timeout=60)
            
            if not df.empty:
                # Get additional data
                try:
                    info = stock.info
                    df.attrs['info'] = info
                except:
                    pass
            
            return df
        except Exception as e:
            raise Exception(f"YFinance error: {e}")
    
    def _fetch_alpha_vantage(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage"""
        if not secrets.ALPHA_VANTAGE_API_KEY:
            raise Exception("Alpha Vantage API key not available")
        
        # Convert Yahoo Finance ticker format to Alpha Vantage format
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'apikey': secrets.ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise Exception(f"Alpha Vantage API error: {data}")
        
        # Convert to DataFrame
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Standardize column names
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
        df = df.astype(float)
        
        # Apply period filter
        if period.endswith('y'):
            years = int(period[:-1])
            cutoff_date = datetime.now() - timedelta(days=years * 365)
            df = df[df.index >= cutoff_date]
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    def _fetch_fmp(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Financial Modeling Prep"""
        # Placeholder for FMP implementation
        raise Exception("FMP implementation not available")
    
    def _fetch_polygon(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data from Polygon.io"""
        # Placeholder for Polygon implementation
        raise Exception("Polygon implementation not available")

# ==================== ENHANCED REAL-TIME DATA MANAGER ====================
class EnhancedRealTimeDataManager:
    """Advanced real-time data manager with WebSocket support"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DATABASE_CONFIG['realtime_data_db']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_database()
        self.active_subscriptions = set()
        self.data_queue = Queue()
    
    def init_database(self):
        """Initialize real-time database with enhanced schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enhanced real-time data table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS realtime_data (
                        ticker TEXT,
                        timestamp DATETIME,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        bid REAL,
                        ask REAL,
                        bid_size INTEGER,
                        ask_size INTEGER,
                        last_trade_time DATETIME,
                        change_percent REAL,
                        data_source TEXT DEFAULT 'yfinance',
                        PRIMARY KEY (ticker, timestamp)
                    )
                """)
                
                # Real-time options data
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS realtime_options (
                        ticker TEXT,
                        timestamp DATETIME,
                        expiration_date TEXT,
                        strike REAL,
                        option_type TEXT,
                        last_price REAL,
                        bid REAL,
                        ask REAL,
                        volume INTEGER,
                        open_interest INTEGER,
                        implied_volatility REAL,
                        PRIMARY KEY (ticker, timestamp, expiration_date, strike, option_type)
                    )
                """)
                
                # Market-wide data
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        timestamp DATETIME,
                        index_name TEXT,
                        value REAL,
                        change_percent REAL,
                        volume BIGINT,
                        PRIMARY KEY (timestamp, index_name)
                    )
                """)
                
                # Sector performance
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sector_performance (
                        timestamp DATETIME,
                        sector_name TEXT,
                        performance REAL,
                        market_cap REAL,
                        volume REAL,
                        PRIMARY KEY (timestamp, sector_name)
                    )
                """)
                
                # Create indices for performance
                indices = [
                    "CREATE INDEX IF NOT EXISTS idx_realtime_ticker_time ON realtime_data (ticker, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_realtime_time ON realtime_data (timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_market_time ON market_data (timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_sector_time ON sector_performance (timestamp)"
                ]
                
                for index in indices:
                    conn.execute(index)
                    
        except Exception as e:
            logging.warning(f"Failed to initialize realtime database: {e}")
    
    def start_realtime_updates(self, tickers: List[str], update_interval: int = 60):
        """Start real-time data collection for specified tickers"""
        def update_worker():
            while True:
                try:
                    for ticker in tickers:
                        if ticker in self.active_subscriptions:
                            self._fetch_realtime_data(ticker)
                    time.sleep(update_interval)
                except Exception as e:
                    logging.error(f"Real-time update error: {e}")
                    time.sleep(10)  # Wait before retrying
        
        # Start background thread
        update_thread = threading.Thread(target=update_worker, daemon=True)
        update_thread.start()
        
        self.active_subscriptions.update(tickers)
        logging.info(f"Started real-time updates for {len(tickers)} tickers")
    
    def _fetch_realtime_data(self, ticker: str):
        """Fetch real-time data for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get current quote
            fast_info = stock.fast_info
            if not fast_info:
                return
            
            current_time = datetime.now()
            
            # Extract real-time data
            data = {
                'ticker': ticker,
                'timestamp': current_time.isoformat(),
                'open': fast_info.get('open', 0),
                'high': fast_info.get('dayHigh', 0),
                'low': fast_info.get('dayLow', 0),
                'close': fast_info.get('lastPrice', 0),
                'volume': fast_info.get('lastVolume', 0),
                'change_percent': ((fast_info.get('lastPrice', 0) - fast_info.get('previousClose', 0)) / fast_info.get('previousClose', 1)) * 100,
                'data_source': 'yfinance'
            }
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO realtime_data 
                    (ticker, timestamp, open, high, low, close, volume, change_percent, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['ticker'], data['timestamp'], data['open'], data['high'],
                    data['low'], data['close'], data['volume'], data['change_percent'],
                    data['data_source']
                ))
                
        except Exception as e:
            logging.warning(f"Failed to fetch real-time data for {ticker}: {e}")
    
    def get_latest_data_enhanced(self, ticker: str, lookback_minutes: int = 60) -> pd.DataFrame:
        """Get enhanced latest real-time data"""
        query = """
            SELECT * FROM realtime_data 
            WHERE ticker = ? 
            AND datetime(timestamp) >= datetime('now', '-{} minutes')
            ORDER BY timestamp DESC
        """.format(lookback_minutes)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(query, conn, params=(ticker,))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
        except Exception as e:
            logging.warning(f"Failed to get latest data for {ticker}: {e}")
            df = pd.DataFrame()
            
        return df
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get latest market indices
                market_query = """
                    SELECT index_name, value, change_percent 
                    FROM market_data 
                    WHERE timestamp = (SELECT MAX(timestamp) FROM market_data)
                """
                market_df = pd.read_sql(market_query, conn)
                
                # Get sector performance
                sector_query = """
                    SELECT sector_name, performance, market_cap 
                    FROM sector_performance 
                    WHERE timestamp = (SELECT MAX(timestamp) FROM sector_performance)
                """
                sector_df = pd.read_sql(sector_query, conn)
                
                # Get most active stocks
                active_query = """
                    SELECT ticker, volume, change_percent 
                    FROM realtime_data 
                    WHERE timestamp >= datetime('now', '-1 hour')
                    ORDER BY volume DESC 
                    LIMIT 20
                """
                active_df = pd.read_sql(active_query, conn)
                
                return {
                    'market_indices': market_df.to_dict('records'),
                    'sector_performance': sector_df.to_dict('records'),
                    'most_active': active_df.to_dict('records'),
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"Failed to get market overview: {e}")
            return {}

# ==================== ADVANCED NEWS AND SENTIMENT ANALYZER ====================
class ComprehensiveNewsAnalyzer:
    """Advanced news and sentiment analysis system"""
    
    def __init__(self):
        self.news_sources = {
            'newsapi': self._fetch_newsapi,
            'alpha_vantage_news': self._fetch_av_news,
            'financial_news_api': self._fetch_financial_news,
            'rss_feeds': self._fetch_rss_news
        }
        
        # Enhanced sentiment lexicon
        self.sentiment_lexicon = {
            'very_positive': ['surge', 'boom', 'breakout', 'rally', 'bullish', 'soar', 'skyrocket'],
            'positive': ['growth', 'strong', 'buy', 'outperform', 'profit', 'gain', 'rise', 'up', 'increase'],
            'neutral': ['stable', 'hold', 'maintain', 'steady', 'unchanged'],
            'negative': ['decline', 'weak', 'sell', 'underperform', 'loss', 'fall', 'down', 'decrease'],
            'very_negative': ['crash', 'plummet', 'collapse', 'bearish', 'panic', 'disaster', 'crisis']
        }
        
        self.sentiment_scores = {
            'very_positive': 1.0,
            'positive': 0.5,
            'neutral': 0.0,
            'negative': -0.5,
            'very_negative': -1.0
        }
    
    def analyze_comprehensive_sentiment(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis"""
        
        all_articles = []
        sentiment_scores = []
        source_breakdown = {}
        
        # Fetch from multiple sources
        for source_name, fetch_func in self.news_sources.items():
            try:
                articles = fetch_func(ticker, days)
                if articles:
                    all_articles.extend(articles)
                    source_breakdown[source_name] = len(articles)
                    logging.info(f"Fetched {len(articles)} articles from {source_name} for {ticker}")
            except Exception as e:
                logging.warning(f"Failed to fetch news from {source_name}: {e}")
        
        if not all_articles:
            return self._get_default_sentiment_result(ticker)
        
        # Analyze sentiment for each article
        for article in all_articles:
            try:
                content = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
                sentiment_data = self._analyze_article_sentiment(content, article)
                sentiment_scores.append(sentiment_data)
            except Exception as e:
                logging.warning(f"Failed to analyze sentiment for article: {e}")
        
        # Calculate comprehensive metrics
        if sentiment_scores:
            overall_sentiment = np.mean([s['score'] for s in sentiment_scores])
            confidence = self._calculate_sentiment_confidence(sentiment_scores)
            trend = self._calculate_sentiment_trend(sentiment_scores)
            
            # Categorize sentiment
            if overall_sentiment > 0.3:
                sentiment_label = 'Very Positive'
            elif overall_sentiment > 0.1:
                sentiment_label = 'Positive'
            elif overall_sentiment > -0.1:
                sentiment_label = 'Neutral'
            elif overall_sentiment > -0.3:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Very Negative'
            
            return {
                'ticker': ticker,
                'overall_sentiment': overall_sentiment,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'trend': trend,
                'article_count': len(all_articles),
                'source_breakdown': source_breakdown,
                'sentiment_distribution': self._get_sentiment_distribution(sentiment_scores),
                'key_topics': self._extract_key_topics(all_articles),
                'analysis_date': datetime.now().isoformat()
            }
        
        return self._get_default_sentiment_result(ticker)
    
    def _analyze_article_sentiment(self, content: str, article: Dict) -> Dict[str, Any]:
        """Analyze sentiment of individual article with advanced techniques"""
        if not content:
            return {'score': 0.0, 'confidence': 0.0, 'factors': []}
        
        content_lower = content.lower()
        sentiment_factors = []
        
        # Lexicon-based analysis
        for category, words in self.sentiment_lexicon.items():
            word_count = sum(1 for word in words if word in content_lower)
            if word_count > 0:
                impact = word_count * self.sentiment_scores[category]
                sentiment_factors.append({
                    'category': category,
                    'impact': impact,
                    'word_count': word_count
                })
        
        # Calculate base sentiment score
        total_impact = sum(factor['impact'] for factor in sentiment_factors)
        word_count = len(content.split())
        
        if word_count > 0:
            base_score = total_impact / max(1, word_count / 10)  # Normalize by content length
        else:
            base_score = 0.0
        
        # Apply modifiers
        # Source credibility modifier
        source = article.get('source', {}).get('name', '').lower()
        credibility_modifier = self._get_source_credibility(source)
        
        # Recency modifier
        published_at = article.get('publishedAt', '')
        recency_modifier = self._get_recency_modifier(published_at)
        
        # Final score calculation
        final_score = base_score * credibility_modifier * recency_modifier
        final_score = max(-1.0, min(1.0, final_score))  # Clamp to [-1, 1]
        
        # Calculate confidence based on content length and factor diversity
        confidence = min(1.0, (word_count / 100) * (len(sentiment_factors) / 5))
        
        return {
            'score': final_score,
            'confidence': confidence,
            'factors': sentiment_factors,
            'base_score': base_score,
            'credibility_modifier': credibility_modifier,
            'recency_modifier': recency_modifier,
            'word_count': word_count,
            'article_url': article.get('url', ''),
            'published_at': published_at
        }
    
    def _get_source_credibility(self, source: str) -> float:
        """Get credibility modifier based on news source"""
        high_credibility = ['reuters', 'bloomberg', 'financial times', 'wall street journal', 
                           'economic times', 'business standard', 'moneycontrol']
        medium_credibility = ['cnbc', 'marketwatch', 'yahoo finance', 'investing.com']
        
        if any(credible in source for credible in high_credibility):
            return 1.2
        elif any(credible in source for credible in medium_credibility):
            return 1.0
        else:
            return 0.8
    
    def _get_recency_modifier(self, published_at: str) -> float:
        """Get recency modifier based on article age"""
        try:
            if not published_at:
                return 0.8
            
            pub_date = pd.to_datetime(published_at)
            hours_ago = (datetime.now() - pub_date.replace(tzinfo=None)).total_seconds() / 3600
            
            if hours_ago <= 24:
                return 1.2  # Recent news has higher impact
            elif hours_ago <= 72:
                return 1.0
            elif hours_ago <= 168:  # 1 week
                return 0.8
            else:
                return 0.6
        except:
            return 0.8
    
    def _calculate_sentiment_confidence(self, sentiment_scores: List[Dict]) -> float:
        """Calculate overall confidence in sentiment analysis"""
        if not sentiment_scores:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of articles
        # 2. Agreement between articles
        # 3. Quality of sources
        # 4. Content depth
        
        article_count_score = min(1.0, len(sentiment_scores) / 20)
        
        # Agreement score (inverse of standard deviation)
        scores = [s['score'] for s in sentiment_scores]
        agreement_score = 1.0 - min(1.0, np.std(scores))
        
        # Average individual confidence
        avg_confidence = np.mean([s['confidence'] for s in sentiment_scores])
        
        return (article_count_score * 0.3 + agreement_score * 0.4 + avg_confidence * 0.3)
    
    def _calculate_sentiment_trend(self, sentiment_scores: List[Dict]) -> str:
        """Calculate sentiment trend over time"""
        if len(sentiment_scores) < 3:
            return 'Insufficient Data'
        
        # Sort by publication date
        sorted_scores = sorted(sentiment_scores, 
                             key=lambda x: x.get('published_at', ''), reverse=True)
        
        recent_scores = [s['score'] for s in sorted_scores[:len(sorted_scores)//2]]
        older_scores = [s['score'] for s in sorted_scores[len(sorted_scores)//2:]]
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        difference = recent_avg - older_avg
        
        if difference > 0.1:
            return 'Improving'
        elif difference < -0.1:
            return 'Deteriorating'
        else:
            return 'Stable'
    
    def _get_sentiment_distribution(self, sentiment_scores: List[Dict]) -> Dict[str, int]:
        """Get distribution of sentiment categories"""
        distribution = {
            'Very Positive': 0,
            'Positive': 0,
            'Neutral': 0,
            'Negative': 0,
            'Very Negative': 0
        }
        
        for score_data in sentiment_scores:
            score = score_data['score']
            if score > 0.3:
                distribution['Very Positive'] += 1
            elif score > 0.1:
                distribution['Positive'] += 1
            elif score > -0.1:
                distribution['Neutral'] += 1
            elif score > -0.3:
                distribution['Negative'] += 1
            else:
                distribution['Very Negative'] += 1
        
        return distribution
    
    def _extract_key_topics(self, articles: List[Dict]) -> List[str]:
        """Extract key topics from articles using simple keyword analysis"""
        all_text = ' '.join([
            f"{article.get('title', '')} {article.get('description', '')}"
            for article in articles
        ])
        
        # Common financial keywords
        keywords = {
            'earnings': ['earnings', 'profit', 'revenue', 'eps'],
            'growth': ['growth', 'expansion', 'increase'],
            'merger': ['merger', 'acquisition', 'takeover'],
            'dividend': ['dividend', 'payout', 'yield'],
            'regulatory': ['regulatory', 'compliance', 'policy'],
            'management': ['ceo', 'management', 'leadership'],
            'market': ['market', 'sector', 'industry'],
            'technology': ['technology', 'innovation', 'digital'],
            'financial': ['debt', 'credit', 'loan', 'investment']
        }
        
        topic_scores = {}
        all_text_lower = all_text.lower()
        
        for topic, words in keywords.items():
            score = sum(1 for word in words if word in all_text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:5]]
    
    def _fetch_newsapi(self, ticker: str, days: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not secrets.NEWS_API_KEY:
            return []
        
        # Convert ticker format
        company_name = ticker.replace('.NS', '').replace('.BO', '')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{company_name} OR stock OR shares",
            'apiKey': secrets.NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'pageSize': 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            logging.warning(f"NewsAPI error for {ticker}: {e}")
            return []
    
    def _fetch_av_news(self, ticker: str, days: int) -> List[Dict]:
        """Fetch news from Alpha Vantage News API"""
        if not secrets.ALPHA_VANTAGE_API_KEY:
            return []
        
        symbol = ticker.replace('.NS', '').replace('.BO', '')
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': secrets.ALPHA_VANTAGE_API_KEY,
            'limit': 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('feed', [])
        except Exception as e:
            logging.warning(f"Alpha Vantage News error for {ticker}: {e}")
            return []
    
    def _fetch_financial_news(self, ticker: str, days: int) -> List[Dict]:
        """Fetch from financial news aggregators"""
        # Placeholder for additional financial news sources
        return []
    
    def _fetch_rss_news(self, ticker: str, days: int) -> List[Dict]:
        """Fetch from RSS feeds"""
        # Placeholder for RSS feed integration
        return []
    
    def _get_default_sentiment_result(self, ticker: str) -> Dict[str, Any]:
        """Return default sentiment result when no data available"""
        return {
            'ticker': ticker,
            'overall_sentiment': 0.0,
            'sentiment_label': 'Neutral',
            'confidence': 0.0,
            'trend': 'No Data',
            'article_count': 0,
            'source_breakdown': {},
            'sentiment_distribution': {'Neutral': 1},
            'key_topics': [],
            'analysis_date': datetime.now().isoformat()
        }

# ==================== COMPREHENSIVE TICKER MANAGEMENT ====================
def get_comprehensive_nifty_tickers() -> Dict[str, List[str]]:
    """Get comprehensive list of Indian stock tickers by category"""
    
    nifty_50 = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
        "ITC.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS", "BAJFINANCE.NS",
        "WIPRO.NS", "ONGC.NS", "SUNPHARMA.NS", "NESTLEIND.NS", "HCLTECH.NS",
        "POWERGRID.NS", "NTPC.NS", "ULTRACEMCO.NS", "TITAN.NS", "DIVISLAB.NS",
        "ADANIENTS.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS", "HINDALCO.NS", "CIPLA.NS",
        "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS",
        "INDUSINDBK.NS", "BRITANNIA.NS", "TATAMOTORS.NS", "TECHM.NS", "TATASTEEL.NS",
        "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "BPCL.NS", "IOC.NS", "TATACONSUM.NS",
        "UPL.NS", "SHRIRAMFIN.NS", "LTIM.NS", "ADANIPORTS.NS", "M&M.NS"
    ]
    
    nifty_next_50 = [
        "BANDHANBNK.NS", "BANKBARODA.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
        "CADILAHC.NS", "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "CONCOR.NS",
        "COROMANDEL.NS", "CUMMINSIND.NS", "DABUR.NS", "DLF.NS", "FRETAIL.NS",
        "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "HDFCLIFE.NS", "ICICIPRULI.NS",
        "IDFCFIRSTB.NS", "IGL.NS", "INDIGO.NS", "INDUSTOWER.NS", "JINDALSTEL.NS",
        "JUBLFOOD.NS", "LICHSGFIN.NS", "LUPIN.NS", "MARICO.NS", "MCDOWELL-N.NS",
        "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NMDC.NS",
        "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "PAGEIND.NS",
        "PERSISTENT.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POONAWALLA.NS",
        "RECLTD.NS", "SAIL.NS", "SBILIFE.NS", "SIEMENS.NS", "TORNTPHARM.NS"
    ]
    
    # Sectoral classifications
    banking_stocks = [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS"
    ]
    
    it_stocks = [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "OFSS.NS"
    ]
    
    pharma_stocks = [
        "SUNPHARMA.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS",
        "BIOCON.NS", "CADILAHC.NS", "TORNTPHARM.NS"
    ]
    
    auto_stocks = [
        "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS", "M&M.NS", "MOTHERSON.NS"
    ]
    
    energy_stocks = [
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
        "BPCL.NS", "IOC.NS", "GAIL.NS", "OIL.NS", "SAIL.NS"
    ]
    
    fmcg_stocks = [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        "DABUR.NS", "MARICO.NS", "COLPAL.NS", "GODREJCP.NS", "MCDOWELL-N.NS"
    ]
    
    return {
        'nifty_50': nifty_50,
        'nifty_next_50': nifty_next_50,
        'banking': banking_stocks,
        'it': it_stocks,
        'pharma': pharma_stocks,
        'auto': auto_stocks,
        'energy': energy_stocks,
        'fmcg': fmcg_stocks,
        'all': list(set(nifty_50 + nifty_next_50))
    }

def validate_and_enhance_tickers(tickers: List[str]) -> Dict[str, Any]:
    """Validate tickers and provide enhancement suggestions"""
    
    validation_result = {
        'valid_tickers': [],
        'invalid_tickers': [],
        'suggestions': {},
        'metadata': {}
    }
    
    # Common ticker corrections
    ticker_corrections = {
        'RELIANCE': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'INFY': 'INFY.NS',
        'HDFCBANK': 'HDFCBANK.NS',
        'ICICIBANK': 'ICICIBANK.NS'
    }
    
    for ticker in tickers:
        if not ticker or not isinstance(ticker, str):
            validation_result['invalid_tickers'].append(ticker)
            continue
        
        ticker = ticker.upper().strip()
        
        # Apply corrections
        if ticker in ticker_corrections:
            ticker = ticker_corrections[ticker]
        
        # Add .NS suffix if missing for Indian stocks
        if not ticker.endswith(('.NS', '.BO', '.NYSE', '.NASDAQ')):
            ticker += '.NS'
        
        # Validate format
        if any(char in ticker for char in ['/', '\\', '|', '<', '>', '"', ' ']):
            validation_result['invalid_tickers'].append(ticker)
            continue
        
        # Quick validation using yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            if info and hasattr(info, 'lastPrice') and info.lastPrice:
                validation_result['valid_tickers'].append(ticker)
                validation_result['metadata'][ticker] = {
                    'lastPrice': info.lastPrice,
                    'currency': getattr(info, 'currency', 'INR'),
                    'exchange': getattr(info, 'exchange', 'NSE')
                }
            else:
                validation_result['invalid_tickers'].append(ticker)
        except:
            validation_result['invalid_tickers'].append(ticker)
    
    return validation_result

# ==================== ENHANCED MAIN DATA COLLECTION INTERFACE ====================
def get_comprehensive_stock_data_enhanced(
    tickers: Optional[List[str]] = None,
    config: Dict = None,
    max_tickers: int = None,
    categories: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Main interface for comprehensive stock data collection with advanced features
    """
    config = config or COMPREHENSIVE_DATA_CONFIG
    
    # Initialize components
    database = AdvancedStockDatabase()
    fetcher = MultiSourceDataFetcher()
    realtime_manager = EnhancedRealTimeDataManager()
    news_analyzer = ComprehensiveNewsAnalyzer()
    
    # Get tickers based on categories or defaults
    if tickers is None:
        ticker_dict = get_comprehensive_nifty_tickers()
        
        if categories:
            selected_tickers = []
            for category in categories:
                if category in ticker_dict:
                    selected_tickers.extend(ticker_dict[category])
        else:
            selected_tickers = ticker_dict['nifty_50']
    else:
        selected_tickers = tickers
    
    # Apply ticker limit
    if max_tickers:
        selected_tickers = selected_tickers[:max_tickers]
    
    # Validate tickers
    validation_result = validate_and_enhance_tickers(selected_tickers)
    valid_tickers = validation_result['valid_tickers']
    
    print(f"=== COMPREHENSIVE STOCK DATA COLLECTION ===")
    print(f"Target tickers: {len(selected_tickers)}")
    print(f"Valid tickers: {len(valid_tickers)}")
    print(f"Invalid tickers: {len(validation_result['invalid_tickers'])}")
    print(f"Max period: {config['max_period']}")
    print(f"Multi-source enabled: True")
    print(f"Real-time updates: {config.get('realtime_refresh_minutes', 0) > 0}")
    print(f"News sentiment: True")
    
    if validation_result['invalid_tickers']:
        print(f"Invalid tickers: {validation_result['invalid_tickers'][:5]}...")
    
    # Prepare comprehensive data collection tasks
    data_collection_results = {}
    successful_fetches = 0
    failed_fetches = 0
    
    # Configure fetching parameters
    periods_to_try = [config['max_period'], '15y', '10y', '5y', '2y']
    intervals = config.get('intervals', ['1d'])
    
    print("\n=== DATA COLLECTION PROGRESS ===")
    
    for ticker in tqdm(valid_tickers, desc="Fetching comprehensive data"):
        try:
            # Check cache first
            cached_data = database.load_stock_data_comprehensive(
                ticker, 
                start_date=(datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            )
            
            metadata = database.get_ticker_metadata_comprehensive(ticker)
            use_cache = False
            
            if metadata and not cached_data.empty:
                last_updated = datetime.fromisoformat(metadata['last_updated'])
                hours_since_update = (datetime.now() - last_updated).total_seconds() / 3600
                
                if hours_since_update < config.get('cache_duration_hours', 24):
                    data_collection_results[ticker] = cached_data
                    successful_fetches += 1
                    use_cache = True
                    logging.info(f"Using cached data for {ticker}")
            
            if not use_cache:
                # Fetch fresh data with multi-source fallback
                df = fetcher.fetch_with_fallback(ticker, config['max_period'])
                
                if not df.empty and len(df) > 100:
                    # Enhanced data validation
                    validated_df = validate_stock_data_comprehensive(df, ticker)
                    
                    if not validated_df.empty:
                        # Save to database
                        database.save_stock_data_comprehensive(
                            ticker, validated_df, '1d', 
                            validated_df.attrs.get('source', 'yfinance')
                        )
                        
                        # Collect multi-timeframe data if enabled
                        if config.get('enable_multi_timeframe', False):
                            for interval in ['1wk', '1mo']:
                                try:
                                    interval_df = fetcher.fetch_with_fallback(ticker, '5y', interval)
                                    if not interval_df.empty:
                                        database.save_stock_data_comprehensive(
                                            ticker, interval_df, interval
                                        )
                                except Exception as e:
                                    logging.warning(f"Failed to fetch {interval} data for {ticker}: {e}")
                        
                        # Collect additional data types if enabled
                        if config.get('enable_options_data', False):
                            try:
                                options_data = fetch_options_data(ticker)
                                if not options_data.empty:
                                    database.save_options_data(ticker, options_data)
                            except Exception as e:
                                logging.warning(f"Options data failed for {ticker}: {e}")
                        
                        if config.get('enable_earnings_data', False):
                            try:
                                earnings_data = fetch_earnings_data(ticker)
                                if not earnings_data.empty:
                                    database.save_earnings_data(ticker, earnings_data)
                            except Exception as e:
                                logging.warning(f"Earnings data failed for {ticker}: {e}")
                        
                        data_collection_results[ticker] = validated_df
                        successful_fetches += 1
                        
                        # Log comprehensive data info
                        date_range = f"{validated_df.index[0].strftime('%Y-%m-%d')} to {validated_df.index[-1].strftime('%Y-%m-%d')}"
                        years_of_data = (validated_df.index[-1] - validated_df.index[0]).days / 365.25
                        logging.info(f"{ticker}: {len(validated_df)} records, {years_of_data:.1f} years ({date_range})")
                        
                    else:
                        failed_fetches += 1
                        logging.warning(f"Data validation failed for {ticker}")
                else:
                    failed_fetches += 1
                    logging.warning(f"Insufficient data for {ticker}")
            
            # Add small delay to respect rate limits
            time.sleep(config.get('request_delay', 0.3))
            
        except Exception as e:
            failed_fetches += 1
            logging.error(f"Comprehensive data collection failed for {ticker}: {e}")
    
    # Start real-time updates if enabled
    if config.get('realtime_refresh_minutes', 0) > 0 and successful_fetches > 0:
        try:
            realtime_manager.start_realtime_updates(
                list(data_collection_results.keys()),
                config['realtime_refresh_minutes'] * 60
            )
            logging.info("Real-time updates started")
        except Exception as e:
            logging.warning(f"Failed to start real-time updates: {e}")
    
    # Collect news sentiment if enabled
    if config.get('enable_sentiment_analysis', False) and successful_fetches > 0:
        print("\n=== NEWS SENTIMENT ANALYSIS ===")
        sentiment_results = {}
        
        for ticker in tqdm(list(data_collection_results.keys())[:20], desc="Analyzing sentiment"):
            try:
                sentiment_data = news_analyzer.analyze_comprehensive_sentiment(ticker)
                sentiment_results[ticker] = sentiment_data
                
                # Save sentiment to database
                database.save_sentiment_data(ticker, sentiment_data)
                
            except Exception as e:
                logging.warning(f"Sentiment analysis failed for {ticker}: {e}")
        
        # Attach sentiment data to results
        for ticker, sentiment in sentiment_results.items():
            if ticker in data_collection_results and not data_collection_results[ticker].empty:
                data_collection_results[ticker].attrs['sentiment'] = sentiment
    
    # Generate comprehensive report
    print(f"\n=== COLLECTION SUMMARY ===")
    print(f"Total tickers processed: {len(valid_tickers)}")
    print(f"Successful fetches: {successful_fetches}")
    print(f"Failed fetches: {failed_fetches}")
    print(f"Success rate: {successful_fetches/(successful_fetches+failed_fetches)*100:.1f}%")
    
    if data_collection_results:
        generate_comprehensive_data_quality_report(data_collection_results)
    
    # Database maintenance
    if config.get('auto_cleanup', False):
        try:
            database.cleanup_old_data(days_to_keep=config.get('data_retention_days', 1095))
            logging.info("Database cleanup completed")
        except Exception as e:
            logging.warning(f"Database cleanup failed: {e}")
    
    return data_collection_results

# ==================== ENHANCED DATA VALIDATION ====================
def validate_stock_data_comprehensive(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Comprehensive data validation with advanced quality checks"""
    if df.empty:
        return df
    
    original_length = len(df)
    validation_issues = []
    
    try:
        # 1. Remove duplicate dates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        # 2. Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Missing critical columns for {ticker}: {missing_columns}")
            return pd.DataFrame()
        
        # 3. Advanced OHLC validation
        # Basic OHLC relationships
        invalid_hl = df['High'] < df['Low']
        if invalid_hl.any():
            df = df[~invalid_hl]
            validation_issues.append(f"Removed {invalid_hl.sum()} rows with High < Low")
        
        # OHLC range validation
        invalid_ohlc = (
            (df['Open'] > df['High']) | (df['Open'] < df['Low']) |
            (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        )
        if invalid_ohlc.any():
            df = df[~invalid_ohlc]
            validation_issues.append(f"Removed {invalid_ohlc.sum()} rows with invalid OHLC ranges")
        
        # 4. Price validation
        # Remove zero/negative prices
        zero_negative = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if zero_negative.any():
            df = df[~zero_negative]
            validation_issues.append(f"Removed {zero_negative.sum()} rows with zero/negative prices")
        
        # Detect and handle stock splits and extreme price movements
        if len(df) > 1:
            returns = df['Close'].pct_change().abs()
            
            # Identify potential stock splits (>90% price change)
            potential_splits = returns > 0.9
            if potential_splits.any():
                split_dates = df.index[potential_splits]
                validation_issues.append(f"Potential stock splits detected on {len(split_dates)} dates")
                
                # Log but don't remove split data as it's legitimate
                for date in split_dates[:3]:  # Log first 3
                    before_price = df.loc[:date, 'Close'].iloc[-2] if len(df.loc[:date]) > 1 else 0
                    after_price = df.loc[date, 'Close']
                    logging.info(f"Potential split on {date}: {before_price:.2f} -> {after_price:.2f}")
            
            # Remove extreme outliers (>50% single day move, excluding splits)
            extreme_moves = (returns > 0.5) & (returns < 0.95)  # Exclude splits
            if extreme_moves.any() and extreme_moves.sum() < len(df) * 0.02:  # Less than 2% of data
                df = df[~extreme_moves]
                validation_issues.append(f"Removed {extreme_moves.sum()} extreme price movement outliers")
        
        # 5. Volume validation
        df.loc[df['Volume'] < 0, 'Volume'] = 0
        
        # Identify and handle volume anomalies
        if len(df) > 30:
            volume_median = df['Volume'].median()
            volume_mad = (df['Volume'] - volume_median).abs().median()  # Median Absolute Deviation
            
            # Flag volumes that are more than 10 MADs from median
            if volume_mad > 0:
                volume_outliers = abs(df['Volume'] - volume_median) > (10 * volume_mad)
                if volume_outliers.any() and volume_outliers.sum() < len(df) * 0.01:
                    # Cap extreme volumes rather than remove
                    max_reasonable_volume = volume_median + (10 * volume_mad)
                    df.loc[volume_outliers, 'Volume'] = max_reasonable_volume
                    validation_issues.append(f"Capped {volume_outliers.sum()} extreme volume values")
        
        # 6. Handle missing values intelligently
        missing_mask = df[required_columns].isnull().any(axis=1)
        if missing_mask.any():
            # For small gaps (≤3 days), use interpolation
            gap_analysis = missing_mask.astype(int).groupby((~missing_mask).cumsum()).sum()
            small_gaps = gap_analysis <= 3
            
            if small_gaps.any():
                df = df.interpolate(method='linear', limit=3, limit_area='inside')
                df = df.fillna(method='ffill', limit=2)
                df = df.fillna(method='bfill', limit=2)
            
            # Remove remaining NaN rows
            remaining_missing = df[required_columns].isnull().any(axis=1)
            if remaining_missing.any():
                df = df[~remaining_missing]
                validation_issues.append(f"Removed {remaining_missing.sum()} rows with missing data")
        
        # 7. Time continuity validation
        if len(df) > 1:
            date_diff = pd.Series(df.index).diff().dt.days
            large_gaps = date_diff > 30  # More than 30 days gap
            
            if large_gaps.any():
                validation_issues.append(f"Found {large_gaps.sum()} large time gaps (>30 days)")
        
        # 8. Data density validation
        if len(df) > 0:
            total_span = (df.index[-1] - df.index[0]).days
            expected_trading_days = total_span * (5/7)  # Approximate trading days
            actual_days = len(df)
            data_density = actual_days / max(expected_trading_days, 1)
            
            if data_density < 0.7:  # Less than 70% expected density
                validation_issues.append(f"Low data density: {data_density:.2%}")
        
        # 9. Price trend validation (detect flat-lining)
        if len(df) > 10:
            price_variance = df['Close'].rolling(10).var()
            flat_periods = (price_variance < 1e-6).sum()  # Near-zero variance
            
            if flat_periods > len(df) * 0.1:  # More than 10% flat periods
                validation_issues.append(f"High flat-lining detected: {flat_periods} periods")
        
        # 10. Calculate final data quality score
        data_quality = len(df) / original_length if original_length > 0 else 0
        
        # Additional quality factors
        completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        consistency = 1.0 if not validation_issues else max(0.5, 1.0 - len(validation_issues) * 0.1)
        
        final_quality = (data_quality * 0.5 + completeness * 0.3 + consistency * 0.2)
        
        # Attach quality metadata
        df.attrs['data_quality'] = final_quality
        df.attrs['validation_issues'] = validation_issues
        df.attrs['original_length'] = original_length
        df.attrs['final_length'] = len(df)
        
        # Log validation summary
        if validation_issues:
            logging.info(f"Validation for {ticker}: {len(validation_issues)} issues addressed")
            for issue in validation_issues[:3]:  # Log first 3 issues
                logging.info(f"  - {issue}")
        
        logging.info(f"Data quality for {ticker}: {final_quality:.3f} ({original_length} -> {len(df)} rows)")
        
        # Return data only if quality is acceptable
        if final_quality >= COMPREHENSIVE_DATA_CONFIG.get('data_quality_threshold', 0.75):
            return df
        else:
            logging.warning(f"Data quality too low for {ticker}: {final_quality:.3f}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Data validation failed for {ticker}: {e}")
        return pd.DataFrame()

# ==================== ADDITIONAL DATA FETCHERS ====================
def fetch_options_data(ticker: str) -> pd.DataFrame:
    """Fetch options chain data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get expiration dates
        exp_dates = stock.options
        if not exp_dates:
            return pd.DataFrame()
        
        all_options = []
        
        # Limit to next 3 expiration dates to avoid too much data
        for exp_date in exp_dates[:3]:
            try:
                # Get calls and puts
                calls = stock.option_chain(exp_date).calls
                puts = stock.option_chain(exp_date).puts
                
                # Add metadata
                calls['expiration_date'] = exp_date
                calls['option_type'] = 'call'
                puts['expiration_date'] = exp_date
                puts['option_type'] = 'put'
                
                all_options.extend([calls, puts])
                
            except Exception as e:
                logging.warning(f"Failed to fetch options for {exp_date}: {e}")
        
        if all_options:
            options_df = pd.concat(all_options, ignore_index=True)
            
            # Standardize column names
            column_mapping = {
                'contractSymbol': 'contract_symbol',
                'lastTradeDate': 'last_trade_date',
                'strike': 'strike',
                'lastPrice': 'last_price',
                'bid': 'bid',
                'ask': 'ask',
                'change': 'change',
                'percentChange': 'percent_change',
                'volume': 'volume',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility'
            }
            
            options_df = options_df.rename(columns=column_mapping)
            return options_df
        
    except Exception as e:
        logging.warning(f"Options data collection failed for {ticker}: {e}")
    
    return pd.DataFrame()

def fetch_earnings_data(ticker: str) -> pd.DataFrame:
    """Fetch earnings data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get earnings history
        earnings = stock.earnings
        quarterly_earnings = stock.quarterly_earnings
        
        if earnings is not None and not earnings.empty:
            earnings_df = earnings.reset_index()
            earnings_df['period_type'] = 'annual'
            earnings_df['ticker'] = ticker
            return earnings_df
        
        if quarterly_earnings is not None and not quarterly_earnings.empty:
            quarterly_df = quarterly_earnings.reset_index()
            quarterly_df['period_type'] = 'quarterly'
            quarterly_df['ticker'] = ticker
            return quarterly_df
        
    except Exception as e:
        logging.warning(f"Earnings data collection failed for {ticker}: {e}")
    
    return pd.DataFrame()

def fetch_analyst_recommendations(ticker: str) -> pd.DataFrame:
    """Fetch analyst recommendations for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get analyst recommendations
        recommendations = stock.recommendations
        
        if recommendations is not None and not recommendations.empty:
            recommendations['ticker'] = ticker
            return recommendations.reset_index()
        
    except Exception as e:
        logging.warning(f"Analyst data collection failed for {ticker}: {e}")
    
    return pd.DataFrame()

# ==================== COMPREHENSIVE DATA QUALITY REPORTING ====================
def generate_comprehensive_data_quality_report(data_dict: Dict[str, pd.DataFrame]):
    """Generate comprehensive data quality report with advanced analytics"""
    
    if not data_dict:
        print("No data available for comprehensive quality report")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA QUALITY REPORT")
    print("="*80)
    
    # Basic statistics
    total_tickers = len(data_dict)
    successful_tickers = len([k for k, v in data_dict.items() if not v.empty])
    total_records = sum(len(df) for df in data_dict.values() if not df.empty)
    
    print(f"Dataset Overview:")
    print(f"  Total tickers processed: {total_tickers}")
    print(f"  Successful tickers: {successful_tickers}")
    print(f"  Success rate: {successful_tickers/total_tickers:.1%}")
    print(f"  Total records collected: {total_records:,}")
    
    # Quality analysis
    quality_scores = []
    date_ranges = []
    validation_issues_summary = {}
    
    for ticker, df in data_dict.items():
        if df.empty:
            continue
        
        # Extract quality metadata
        quality_score = df.attrs.get('data_quality', 1.0)
        quality_scores.append(quality_score)
        
        # Date range analysis
        start_date = df.index[0]
        end_date = df.index[-1]
        years_of_data = (end_date - start_date).days / 365.25
        date_ranges.append(years_of_data)
        
        # Validation issues
        issues = df.attrs.get('validation_issues', [])
        for issue in issues:
            issue_type = issue.split()[0]  # First word as issue type
            validation_issues_summary[issue_type] = validation_issues_summary.get(issue_type, 0) + 1
    
    # Quality statistics
    if quality_scores:
        print(f"\nData Quality Analysis:")
        print(f"  Average quality score: {np.mean(quality_scores):.3f}")
        print(f"  Median quality score: {np.median(quality_scores):.3f}")
        print(f"  Minimum quality score: {np.min(quality_scores):.3f}")
        print(f"  Maximum quality score: {np.max(quality_scores):.3f}")
        print(f"  High quality tickers (>0.9): {sum(1 for s in quality_scores if s > 0.9)}")
        print(f"  Medium quality tickers (0.7-0.9): {sum(1 for s in quality_scores if 0.7 <= s <= 0.9)}")
        print(f"  Low quality tickers (<0.7): {sum(1 for s in quality_scores if s < 0.7)}")
    
    # Historical coverage
    if date_ranges:
        print(f"\nHistorical Data Coverage:")
        print(f"  Average years of data: {np.mean(date_ranges):.1f}")
        print(f"  Median years of data: {np.median(date_ranges):.1f}")
        print(f"  Maximum coverage: {max(date_ranges):.1f} years")
        print(f"  Minimum coverage: {min(date_ranges):.1f} years")
        print(f"  Tickers with 10+ years: {sum(1 for y in date_ranges if y >= 10)}")
        print(f"  Tickers with 5+ years: {sum(1 for y in date_ranges if y >= 5)}")
    
    # Data volume analysis
    record_counts = [len(df) for df in data_dict.values() if not df.empty]
    if record_counts:
        print(f"\nData Volume Analysis:")
        print(f"  Average records per ticker: {np.mean(record_counts):.0f}")
        print(f"  Median records per ticker: {np.median(record_counts):.0f}")
        print(f"  Largest dataset: {max(record_counts):,} records")
        print(f"  Smallest dataset: {min(record_counts):,} records")
    
    # Validation issues summary
    if validation_issues_summary:
        print(f"\nValidation Issues Summary:")
        sorted_issues = sorted(validation_issues_summary.items(), key=lambda x: x[1], reverse=True)
        for issue_type, count in sorted_issues[:10]:  # Top 10 issues
            print(f"  {issue_type}: {count} occurrences")
    
    # Sector/Market analysis if available
    sectors = {}
    for ticker, df in data_dict.items():
        if not df.empty and hasattr(df, 'attrs') and 'sector' in df.attrs:
            sector = df.attrs['sector']
            sectors[sector] = sectors.get(sector, 0) + 1
    
    if sectors:
        print(f"\nSector Distribution:")
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
        for sector, count in sorted_sectors:
            print(f"  {sector}: {count} tickers")
    
    # Data freshness analysis
    current_time = datetime.now()
    freshness_analysis = {}
    
    for ticker, df in data_dict.items():
        if not df.empty:
            latest_date = df.index[-1]
            if hasattr(latest_date, 'to_pydatetime'):
                latest_date = latest_date.to_pydatetime()
            
            days_old = (current_time - latest_date).days
            
            if days_old <= 1:
                freshness_category = 'Very Fresh (≤1 day)'
            elif days_old <= 7:
                freshness_category = 'Fresh (≤1 week)'
            elif days_old <= 30:
                freshness_category = 'Recent (≤1 month)'
            else:
                freshness_category = f'Stale (>{days_old} days)'
            
            freshness_analysis[freshness_category] = freshness_analysis.get(freshness_category, 0) + 1
    
    if freshness_analysis:
        print(f"\nData Freshness Analysis:")
        for category, count in sorted(freshness_analysis.items()):
            print(f"  {category}: {count} tickers")
    
    print("="*80)

# ==================== ADVANCED CACHING SYSTEM ====================
class AdvancedDataCache:
    """Advanced caching system with compression and intelligent invalidation"""
    
    def __init__(self, cache_dir: str = "data_cache_v3", max_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except:
            self.metadata = {}
    
    def save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logging.warning(f"Failed to save cache metadata: {e}")
    
    def get_cache_key(self, ticker: str, params: Dict) -> str:
        """Generate cache key from ticker and parameters"""
        params_str = json.dumps(params, sort_keys=True, default=str)
        combined = f"{ticker}_{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def is_cache_valid(self, cache_key: str, max_age_hours: float = 24) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.metadata:
            return False
        
        cache_info = self.metadata[cache_key]
        created_time = datetime.fromisoformat(cache_info['created_at'])
        age_hours = (datetime.now() - created_time).total_seconds() / 3600
        
        return age_hours < max_age_hours
    
    def get_cached_data(self, ticker: str, params: Dict) -> Optional[pd.DataFrame]:
        """Retrieve cached data if valid"""
        cache_key = self.get_cache_key(ticker, params)
        
        if not self.is_cache_valid(cache_key):
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logging.info(f"Cache hit for {ticker}")
            return data
            
        except Exception as e:
            logging.warning(f"Failed to load cached data for {ticker}: {e}")
            return None
    
    def save_to_cache(self, ticker: str, params: Dict, data: pd.DataFrame):
        """Save data to cache with compression"""
        if data.empty:
            return
        
        cache_key = self.get_cache_key(ticker, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Save compressed data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            self.metadata[cache_key] = {
                'ticker': ticker,
                'created_at': datetime.now().isoformat(),
                'file_size': file_size,
                'record_count': len(data),
                'params': params
            }
            
            self.save_metadata()
            
            # Clean up if cache is too large
            self.cleanup_cache()
            
            logging.info(f"Cached data for {ticker}: {len(data)} records, {file_size/1024:.1f}KB")
            
        except Exception as e:
            logging.warning(f"Failed to cache data for {ticker}: {e}")
    
    def cleanup_cache(self):
        """Clean up cache if it exceeds size limit"""
        try:
            total_size = sum(info.get('file_size', 0) for info in self.metadata.values())
            
            if total_size > self.max_size_bytes:
                # Sort by creation time (oldest first)
                sorted_entries = sorted(
                    self.metadata.items(),
                    key=lambda x: x[1].get('created_at', '1970-01-01')
                )
                
                # Remove oldest entries until under limit
                removed_count = 0
                for cache_key, info in sorted_entries:
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    
                    if cache_file.exists():
                        cache_file.unlink()
                        total_size -= info.get('file_size', 0)
                        removed_count += 1
                    
                    del self.metadata[cache_key]
                    
                    if total_size <= self.max_size_bytes * 0.8:  # Keep 20% buffer
                        break
                
                self.save_metadata()
                logging.info(f"Cache cleanup: removed {removed_count} entries")
                
        except Exception as e:
            logging.warning(f"Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.metadata:
            return {'total_entries': 0, 'total_size_mb': 0}
        
        total_entries = len(self.metadata)
        total_size = sum(info.get('file_size', 0) for info in self.metadata.values())
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_entry': min(info.get('created_at', '') for info in self.metadata.values()),
            'newest_entry': max(info.get('created_at', '') for info in self.metadata.values()),
            'cache_directory': str(self.cache_dir)
        }

# Export main functions
__all__ = [
    'get_comprehensive_stock_data_enhanced',
    'AdvancedStockDatabase',
    'MultiSourceDataFetcher',
    'EnhancedRealTimeDataManager',
    'ComprehensiveNewsAnalyzer',
    'get_comprehensive_nifty_tickers',
    'validate_and_enhance_tickers',
    'validate_stock_data_comprehensive',
    'generate_comprehensive_data_quality_report',
    'AdvancedDataCache',
    'COMPREHENSIVE_DATA_CONFIG'
]