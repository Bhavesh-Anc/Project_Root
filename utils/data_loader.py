# utils/data_loader.py - Complete Data Loading System
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sqlite3
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from functools import lru_cache

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# ==================== CONFIGURATION ====================

DATA_CONFIG = {
    'max_period': '5y',
    'use_database': True,
    'database_path': 'data/stock_data.db',
    'cache_duration_hours': 24,
    'validate_data': True,
    'min_data_points': 100,
    'parallel_downloads': True,
    'max_workers': 4,
    'retry_attempts': 3,
    'timeout_seconds': 30,
    'backup_sources': ['yfinance'],  # Can add more sources
    'data_quality_checks': True,
    'auto_cleanup': True
}

# Indian stock market tickers with sectors
NIFTY_50_TICKERS = {
    # Banking
    'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking'},
    'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking'},
    'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking'},
    'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking'},
    'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking'},
    'INDUSINDBK.NS': {'name': 'IndusInd Bank', 'sector': 'Banking'},
    
    # Technology
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'Technology'},
    'INFY.NS': {'name': 'Infosys', 'sector': 'Technology'},
    'WIPRO.NS': {'name': 'Wipro', 'sector': 'Technology'},
    'HCLTECH.NS': {'name': 'HCL Technologies', 'sector': 'Technology'},
    'TECHM.NS': {'name': 'Tech Mahindra', 'sector': 'Technology'},
    
    # Industrial & Infrastructure
    'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Industrial'},
    'LT.NS': {'name': 'Larsen & Toubro', 'sector': 'Industrial'},
    'ULTRACEMCO.NS': {'name': 'UltraTech Cement', 'sector': 'Industrial'},
    'POWERGRID.NS': {'name': 'Power Grid Corporation', 'sector': 'Industrial'},
    'NTPC.NS': {'name': 'NTPC', 'sector': 'Industrial'},
    'COALINDIA.NS': {'name': 'Coal India', 'sector': 'Industrial'},
    'ONGC.NS': {'name': 'Oil & Natural Gas Corporation', 'sector': 'Industrial'},
    'IOC.NS': {'name': 'Indian Oil Corporation', 'sector': 'Industrial'},
    'BPCL.NS': {'name': 'Bharat Petroleum', 'sector': 'Industrial'},
    'TATASTEEL.NS': {'name': 'Tata Steel', 'sector': 'Industrial'},
    'JSWSTEEL.NS': {'name': 'JSW Steel', 'sector': 'Industrial'},
    'HINDALCO.NS': {'name': 'Hindalco Industries', 'sector': 'Industrial'},
    'VEDL.NS': {'name': 'Vedanta', 'sector': 'Industrial'},
    'ADANIPORTS.NS': {'name': 'Adani Ports', 'sector': 'Industrial'},
    'GRASIM.NS': {'name': 'Grasim Industries', 'sector': 'Industrial'},
    
    # FMCG & Consumer
    'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG'},
    'ITC.NS': {'name': 'ITC', 'sector': 'FMCG'},
    'NESTLEIND.NS': {'name': 'Nestle India', 'sector': 'FMCG'},
    'BRITANNIA.NS': {'name': 'Britannia Industries', 'sector': 'FMCG'},
    'TATACONSUM.NS': {'name': 'Tata Consumer Products', 'sector': 'FMCG'},
    'TITAN.NS': {'name': 'Titan Company', 'sector': 'FMCG'},
    'ASIANPAINT.NS': {'name': 'Asian Paints', 'sector': 'FMCG'},
    
    # Auto & Financial Services
    'MARUTI.NS': {'name': 'Maruti Suzuki', 'sector': 'Auto'},
    'TATAMOTORS.NS': {'name': 'Tata Motors', 'sector': 'Auto'},
    'BAJAJ-AUTO.NS': {'name': 'Bajaj Auto', 'sector': 'Auto'},
    'EICHERMOT.NS': {'name': 'Eicher Motors', 'sector': 'Auto'},
    'HEROMOTOCO.NS': {'name': 'Hero MotoCorp', 'sector': 'Auto'},
    'M&M.NS': {'name': 'Mahindra & Mahindra', 'sector': 'Auto'},
    
    'BAJFINANCE.NS': {'name': 'Bajaj Finance', 'sector': 'Financial Services'},
    'BAJAJFINSV.NS': {'name': 'Bajaj Finserv', 'sector': 'Financial Services'},
    'HDFC.NS': {'name': 'HDFC', 'sector': 'Financial Services'},
    
    # Pharma & Healthcare
    'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical', 'sector': 'Pharma'},
    'DRREDDY.NS': {'name': 'Dr. Reddys Laboratories', 'sector': 'Pharma'},
    'CIPLA.NS': {'name': 'Cipla', 'sector': 'Pharma'},
    'DIVISLAB.NS': {'name': 'Divis Laboratories', 'sector': 'Pharma'},
    'APOLLOHOSP.NS': {'name': 'Apollo Hospitals', 'sector': 'Healthcare'},
    
    # Telecom & Others
    'BHARTIARTL.NS': {'name': 'Bharti Airtel', 'sector': 'Telecom'},
    'UPL.NS': {'name': 'UPL', 'sector': 'Chemicals'},
    'SHREECEM.NS': {'name': 'Shree Cement', 'sector': 'Cement'}
}

# ==================== DATABASE MANAGEMENT ====================

class StockDataDB:
    """Database manager for stock data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATA_CONFIG['database_path']
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Create main stock data table
            conn.execute('''
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
            ''')
            
            # Create metadata table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_metadata (
                    ticker TEXT PRIMARY KEY,
                    last_updated TIMESTAMP,
                    data_points INTEGER,
                    start_date TEXT,
                    end_date TEXT,
                    data_quality_score REAL
                )
            ''')
            
            # Create indices for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data(ticker, date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)')
            
            conn.commit()
            conn.close()
            
            logging.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
    
    def save_stock_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """Save stock data to database"""
        try:
            if df.empty:
                return False
            
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            data_records = []
            for date, row in df.iterrows():
                data_records.append((
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    float(row.get('Open', 0)),
                    float(row.get('High', 0)),
                    float(row.get('Low', 0)),
                    float(row.get('Close', 0)),
                    int(row.get('Volume', 0)),
                    float(row.get('Adj Close', row.get('Close', 0)))
                ))
            
            # Insert data (replace existing)
            conn.execute('DELETE FROM stock_data WHERE ticker = ?', (ticker,))
            
            conn.executemany('''
                INSERT INTO stock_data 
                (ticker, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_records)
            
            # Update metadata
            conn.execute('''
                INSERT OR REPLACE INTO data_metadata 
                (ticker, last_updated, data_points, start_date, end_date, data_quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                datetime.now(),
                len(df),
                df.index[0].strftime('%Y-%m-%d'),
                df.index[-1].strftime('%Y-%m-%d'),
                self._calculate_data_quality_score(df)
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Saved {len(df)} records for {ticker}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save data for {ticker}: {e}")
            return False
    
    def load_stock_data(self, ticker: str, days_old: int = None) -> Optional[pd.DataFrame]:
        """Load stock data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if data exists and is recent enough
            metadata_query = '''
                SELECT last_updated, data_points FROM data_metadata 
                WHERE ticker = ?
            '''
            metadata = conn.execute(metadata_query, (ticker,)).fetchone()
            
            if not metadata:
                conn.close()
                return None
            
            last_updated, data_points = metadata
            last_updated = datetime.fromisoformat(last_updated)
            
            # Check if data is recent enough
            if days_old is not None:
                age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                if age_hours > days_old * 24:
                    conn.close()
                    return None
            
            # Load data
            query = '''
                SELECT date, open, high, low, close, volume, adj_close
                FROM stock_data 
                WHERE ticker = ?
                ORDER BY date
            '''
            
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            
            if df.empty:
                return None
            
            # Process dataframe
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to match yfinance format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            
            logging.info(f"Loaded {len(df)} cached records for {ticker}")
            return df
            
        except Exception as e:
            logging.error(f"Failed to load cached data for {ticker}: {e}")
            return None
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        try:
            if df.empty:
                return 0.0
            
            score = 1.0
            
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            score -= missing_ratio * 0.3
            
            # Check for zero values in price columns
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    zero_ratio = (df[col] == 0).sum() / len(df)
                    score -= zero_ratio * 0.2
            
            # Check for reasonable price movements
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()
                if len(returns) > 0:
                    extreme_moves = (abs(returns) > 0.2).sum() / len(returns)
                    score -= extreme_moves * 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Remove old data
            conn.execute('''
                DELETE FROM stock_data 
                WHERE date < ?
            ''', (cutoff_date.strftime('%Y-%m-%d'),))
            
            # Update metadata
            conn.execute('''
                DELETE FROM data_metadata 
                WHERE last_updated < ?
            ''', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logging.error(f"Database cleanup failed: {e}")

# ==================== DATA DOWNLOAD AND PROCESSING ====================

class EnhancedDataLoader:
    """Enhanced data loader with multiple sources and robust error handling"""
    
    def __init__(self, config: Dict = None):
        self.config = config or DATA_CONFIG
        self.db = StockDataDB(self.config.get('database_path'))
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_single_stock(self, ticker: str, period: str = '5y', 
                             force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Download data for a single stock with robust error handling"""
        
        try:
            # Check cache first (if not forcing refresh)
            if not force_refresh and self.config.get('use_database', True):
                cached_data = self.db.load_stock_data(
                    ticker, 
                    days_old=self.config.get('cache_duration_hours', 24) // 24
                )
                
                if cached_data is not None and len(cached_data) > self.config.get('min_data_points', 100):
                    logging.info(f"Using cached data for {ticker}")
                    return cached_data
            
            # Download fresh data
            logging.info(f"Downloading fresh data for {ticker}")
            
            success = False
            df = None
            
            # Try multiple download attempts
            for attempt in range(self.config.get('retry_attempts', 3)):
                try:
                    # Primary method: yfinance
                    stock = yf.Ticker(ticker)
                    df = stock.history(
                        period=period,
                        timeout=self.config.get('timeout_seconds', 30)
                    )
                    
                    if not df.empty and len(df) > 10:
                        success = True
                        break
                    
                except Exception as e:
                    logging.warning(f"Download attempt {attempt + 1} failed for {ticker}: {e}")
                    time.sleep(1)  # Brief pause between attempts
            
            if not success or df is None or df.empty:
                logging.error(f"Failed to download data for {ticker}")
                return None
            
            # Validate and clean data
            df = self._validate_and_clean_data(df, ticker)
            
            if df is None or len(df) < self.config.get('min_data_points', 100):
                logging.warning(f"Insufficient data for {ticker}: {len(df) if df is not None else 0} points")
                return None
            
            # Save to database
            if self.config.get('use_database', True):
                self.db.save_stock_data(ticker, df)
            
            logging.info(f"Successfully downloaded {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logging.error(f"Download failed for {ticker}: {e}")
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Validate and clean downloaded data"""
        
        try:
            if df.empty:
                return None
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logging.warning(f"Missing columns for {ticker}: {missing_columns}")
                return None
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Remove rows where all price columns are zero or negative
            price_columns = ['Open', 'High', 'Low', 'Close']
            valid_price_mask = (df[price_columns] > 0).any(axis=1)
            df = df[valid_price_mask]
            
            # Forward fill missing values
            df = df.fillna(method='ffill')
            
            # Remove remaining NaN values
            df = df.dropna()
            
            # Validate High >= Low and High >= Close and High >= Open
            df = df[(df['High'] >= df['Low']) & 
                   (df['High'] >= df['Close']) & 
                   (df['High'] >= df['Open']) &
                   (df['Low'] <= df['Close']) &
                   (df['Low'] <= df['Open'])]
            
            # Remove extreme outliers (daily changes > 50%)
            if len(df) > 1:
                returns = df['Close'].pct_change().abs()
                valid_returns = returns < 0.5  # 50% daily change threshold
                df = df[valid_returns.fillna(True)]
            
            # Ensure minimum data points
            if len(df) < 50:
                logging.warning(f"Insufficient valid data for {ticker}: {len(df)} points")
                return None
            
            # Add Adj Close if missing
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            logging.info(f"Data validation passed for {ticker}: {len(df)} valid records")
            return df
            
        except Exception as e:
            logging.error(f"Data validation failed for {ticker}: {e}")
            return None
    
    def download_multiple_stocks(self, tickers: List[str], period: str = '5y',
                                force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Download data for multiple stocks with parallel processing"""
        
        if not tickers:
            return {}
        
        logging.info(f"Downloading data for {len(tickers)} stocks")
        
        results = {}
        
        if self.config.get('parallel_downloads', True) and len(tickers) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
                # Submit all download tasks
                future_to_ticker = {
                    executor.submit(self.download_single_stock, ticker, period, force_refresh): ticker
                    for ticker in tickers
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        df = future.result(timeout=60)  # 60 second timeout
                        if df is not None:
                            results[ticker] = df
                    except Exception as e:
                        logging.error(f"Parallel download failed for {ticker}: {e}")
        else:
            # Sequential processing
            for ticker in tickers:
                df = self.download_single_stock(ticker, period, force_refresh)
                if df is not None:
                    results[ticker] = df
        
        logging.info(f"Successfully downloaded data for {len(results)}/{len(tickers)} stocks")
        return results

# ==================== MAIN DATA LOADER FUNCTIONS ====================

@lru_cache(maxsize=100)
def get_available_tickers() -> Dict[str, Dict[str, str]]:
    """Get available tickers with metadata"""
    return NIFTY_50_TICKERS.copy()

def get_tickers_by_sector(sector: str) -> List[str]:
    """Get tickers filtered by sector"""
    tickers = get_available_tickers()
    return [ticker for ticker, info in tickers.items() 
            if info.get('sector', '').lower() == sector.lower()]

def get_comprehensive_stock_data(selected_tickers: List[str] = None, 
                               period: str = '5y',
                               force_refresh: bool = False,
                               validate_data: bool = True,
                               parallel: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive stock data loader - Main function used by app.py
    
    Args:
        selected_tickers: List of tickers to download. If None, downloads all available
        period: Data period (1y, 2y, 5y, 10y, max)
        force_refresh: Force download even if cached data exists
        validate_data: Perform data quality validation
        parallel: Use parallel processing for downloads
        
    Returns:
        Dictionary mapping ticker symbols to DataFrames
    """
    
    try:
        # Use selected tickers or default to all available
        if selected_tickers is None:
            available_tickers = get_available_tickers()
            selected_tickers = list(available_tickers.keys())
        
        if not selected_tickers:
            logging.warning("No tickers provided for data download")
            return {}
        
        # Filter to valid tickers
        available_tickers = get_available_tickers()
        valid_tickers = [ticker for ticker in selected_tickers 
                        if ticker in available_tickers]
        
        if len(valid_tickers) != len(selected_tickers):
            invalid_tickers = set(selected_tickers) - set(valid_tickers)
            logging.warning(f"Invalid tickers removed: {invalid_tickers}")
        
        if not valid_tickers:
            logging.error("No valid tickers found")
            return {}
        
        logging.info(f"Loading data for {len(valid_tickers)} selected stocks")
        
        # Configure data loader
        config = DATA_CONFIG.copy()
        config['parallel_downloads'] = parallel
        config['validate_data'] = validate_data
        
        # Initialize loader and download data
        loader = EnhancedDataLoader(config)
        
        # Download data
        data_dict = loader.download_multiple_stocks(
            tickers=valid_tickers,
            period=period,
            force_refresh=force_refresh
        )
        
        # Additional validation if requested
        if validate_data:
            validated_data = {}
            for ticker, df in data_dict.items():
                if _perform_quality_checks(df, ticker):
                    validated_data[ticker] = df
                else:
                    logging.warning(f"Data quality check failed for {ticker}")
            
            data_dict = validated_data
        
        # Log summary
        successful_downloads = len(data_dict)
        total_data_points = sum(len(df) for df in data_dict.values())
        
        logging.info(f"Data loading summary:")
        logging.info(f"  - Successful downloads: {successful_downloads}/{len(valid_tickers)}")
        logging.info(f"  - Total data points: {total_data_points:,}")
        logging.info(f"  - Average points per stock: {total_data_points // max(successful_downloads, 1):,}")
        
        # Auto cleanup if enabled
        if config.get('auto_cleanup', True) and successful_downloads > 0:
            try:
                loader.db.cleanup_old_data(days_to_keep=90)
            except Exception as e:
                logging.warning(f"Auto cleanup failed: {e}")
        
        return data_dict
        
    except Exception as e:
        logging.error(f"Comprehensive data loading failed: {e}")
        return {}

def _perform_quality_checks(df: pd.DataFrame, ticker: str) -> bool:
    """Perform additional data quality checks"""
    
    try:
        if df.empty:
            return False
        
        # Check minimum data points
        if len(df) < DATA_CONFIG.get('min_data_points', 100):
            logging.warning(f"Insufficient data points for {ticker}: {len(df)}")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logging.warning(f"Missing required columns for {ticker}")
            return False
        
        # Check for reasonable data distribution
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if df[col].min() <= 0:
                logging.warning(f"Invalid prices found in {col} for {ticker}")
                return False
        
        # Check for excessive missing data
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.1:  # 10% threshold
            logging.warning(f"Excessive missing data for {ticker}: {missing_ratio:.1%}")
            return False
        
        # Check for data recency (should have recent data)
        latest_date = df.index[-1]
        days_old = (datetime.now() - latest_date).days
        if days_old > 7:  # Data should be within last week
            logging.warning(f"Data too old for {ticker}: {days_old} days")
            return False
        
        logging.debug(f"Quality checks passed for {ticker}")
        return True
        
    except Exception as e:
        logging.error(f"Quality check failed for {ticker}: {e}")
        return False

def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Get additional stock information"""
    
    try:
        available_tickers = get_available_tickers()
        
        if ticker not in available_tickers:
            return {'error': f'Ticker {ticker} not found'}
        
        info = available_tickers[ticker].copy()
        
        # Try to get additional info from yfinance
        try:
            stock = yf.Ticker(ticker)
            yf_info = stock.info
            
            # Extract useful information
            if yf_info:
                info.update({
                    'market_cap': yf_info.get('marketCap'),
                    'pe_ratio': yf_info.get('trailingPE'),
                    'dividend_yield': yf_info.get('dividendYield'),
                    'beta': yf_info.get('beta'),
                    'price': yf_info.get('currentPrice'),
                    'currency': yf_info.get('currency', 'INR')
                })
                
        except Exception as e:
            logging.warning(f"Could not fetch additional info for {ticker}: {e}")
        
        return info
        
    except Exception as e:
        logging.error(f"Failed to get stock info for {ticker}: {e}")
        return {'error': str(e)}

def refresh_all_data(force: bool = False) -> Dict[str, bool]:
    """Refresh all available stock data"""
    
    try:
        available_tickers = list(get_available_tickers().keys())
        
        logging.info(f"Refreshing data for {len(available_tickers)} stocks")
        
        # Use comprehensive data loader
        results = get_comprehensive_stock_data(
            selected_tickers=available_tickers,
            force_refresh=force,
            parallel=True
        )
        
        # Return success status for each ticker
        success_status = {}
        for ticker in available_tickers:
            success_status[ticker] = ticker in results
        
        successful_count = sum(success_status.values())
        
        logging.info(f"Data refresh completed: {successful_count}/{len(available_tickers)} successful")
        
        return success_status
        
    except Exception as e:
        logging.error(f"Data refresh failed: {e}")
        return {}

# ==================== UTILITY FUNCTIONS ====================

def get_market_status() -> Dict[str, Any]:
    """Get current market status"""
    
    try:
        now = datetime.now()
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
        is_trading_hours = market_open_time <= now <= market_close_time
        
        market_status = "CLOSED"
        if not is_weekend and is_trading_hours:
            market_status = "OPEN"
        elif not is_weekend and now < market_open_time:
            market_status = "PRE_MARKET"
        elif not is_weekend and now > market_close_time:
            market_status = "AFTER_MARKET"
        
        return {
            'status': market_status,
            'is_trading_day': not is_weekend,
            'current_time': now,
            'market_open': market_open_time,
            'market_close': market_close_time,
            'next_trading_day': _get_next_trading_day(now)
        }
        
    except Exception as e:
        logging.error(f"Failed to get market status: {e}")
        return {'status': 'UNKNOWN', 'error': str(e)}

def _get_next_trading_day(current_time: datetime) -> datetime:
    """Get next trading day"""
    
    next_day = current_time + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    # Set to market open time
    return next_day.replace(hour=9, minute=15, second=0, microsecond=0)

def estimate_download_time(num_tickers: int) -> float:
    """Estimate download time in seconds"""
    
    # Rough estimates based on experience
    base_time_per_ticker = 2.0  # seconds
    parallel_efficiency = 0.4  # 40% efficiency for parallel downloads
    
    if num_tickers <= 1:
        return base_time_per_ticker
    elif num_tickers <= 4:
        return base_time_per_ticker * num_tickers * parallel_efficiency
    else:
        # For larger batches, assume some overhead
        return base_time_per_ticker * num_tickers * parallel_efficiency * 1.2

# ==================== EXPORT AND TESTING ====================

# Export main functions and classes
__all__ = [
    'get_comprehensive_stock_data',
    'get_available_tickers', 
    'get_tickers_by_sector',
    'get_stock_info',
    'refresh_all_data',
    'get_market_status',
    'DATA_CONFIG',
    'NIFTY_50_TICKERS',
    'EnhancedDataLoader',
    'StockDataDB'
]

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Stock Data Loader - User Selection Version")
    print("="*60)
    
    # Test basic functionality
    print("Available sectors:")
    sectors = set(info['sector'] for info in get_available_tickers().values())
    for sector in sorted(sectors):
        tickers = get_tickers_by_sector(sector)
        print(f"  - {sector}: {len(tickers)} stocks")
    
    # Test market status
    market_status = get_market_status()
    print(f"\nMarket Status: {market_status['status']}")
    print(f"Is Trading Day: {market_status['is_trading_day']}")
    
    # Test download for selected stocks
    selected_test_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    print(f"\nTesting download for selected stocks: {selected_test_tickers}")
    
    estimated_time = estimate_download_time(len(selected_test_tickers))
    print(f"Estimated download time: {estimated_time:.1f} seconds")
    
    try:
        start_time = time.time()
        
        # Test comprehensive data loading
        data = get_comprehensive_stock_data(
            selected_tickers=selected_test_tickers,
            period='1y',  # Use shorter period for testing
            force_refresh=False,  # Use cache if available
            parallel=True
        )
        
        actual_time = time.time() - start_time
        
        print(f"Actual download time: {actual_time:.1f} seconds")
        print(f"Successfully downloaded: {len(data)}/{len(selected_test_tickers)} stocks")
        
        for ticker, df in data.items():
            print(f"  - {ticker}: {len(df)} data points ({df.index[0].date()} to {df.index[-1].date()})")
        
        # Test stock info
        if data:
            test_ticker = list(data.keys())[0]
            stock_info = get_stock_info(test_ticker)
            print(f"\nStock info for {test_ticker}:")
            for key, value in stock_info.items():
                if key != 'error':
                    print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"Test failed: {e}")
    
    print(f"\nUser Selection Features:")
    print(f"  ✓ Optimized for user-selected stocks only")
    print(f"  ✓ Intelligent caching and database storage")
    print(f"  ✓ Parallel downloading for performance")
    print(f"  ✓ Comprehensive data validation")
    print(f"  ✓ Robust error handling and retries")
    print(f"  ✓ Sector-based filtering")
    print(f"  ✓ Market status monitoring")
    print(f"  ✓ Automatic data cleanup")
    
    print(f"\nData Loader Test Completed!")