# utils/indian_data_loader.py
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import json
from tqdm import tqdm
import re
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "indian_stock_data"
os.makedirs(DATA_DIR, exist_ok=True)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}
NSE_DATA_URL = "https://www.nseindia.com/api/historical/cm/equity"
MAX_WORKERS = 10
REQUEST_DELAY = 1.5

def get_nifty_indices_tickers():
    """Fetch current constituents for all major Nifty indices"""
    indices = {
        "NIFTY 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
        "NIFTY NEXT 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20NEXT%2050",
        "NIFTY 100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
        "NIFTY 200": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200",
        "NIFTY 500": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
    }
    
    all_tickers = set()
    
    for idx_name, url in indices.items():
        try:
            session = requests.Session()
            session.headers.update(HEADERS)
            session.get("https://www.nseindia.com", timeout=5)  # Get cookies
            response = session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for item in data['data']:
                ticker = item['symbol'] + ".NS"
                all_tickers.add(ticker)
                
            logger.info(f"Fetched {len(data['data'])} tickers from {idx_name}")
            time.sleep(1)  # Be polite to the server
            
        except Exception as e:
            logger.error(f"Error fetching {idx_name}: {str(e)}")
            if idx_name == "NIFTY 50":
                all_tickers.update([
                    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
                    "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "TITAN.NS",
                    "SUNPHARMA.NS", "AXISBANK.NS", "DMART.NS", "ULTRACEMCO.NS", "TATASTEEL.NS"
                ])
    
    # Remove any invalid entries
    all_tickers = {t for t in all_tickers if re.match(r"^[A-Z0-9]+\.NS$", t)}
    return list(all_tickers)

def fetch_via_nsepy(ticker, start_date=None, end_date=None):
    """Fetch data using NSEPy library"""
    try:
        from nsepy import get_history
        
        base_ticker = ticker.replace(".NS", "")
        start_date = start_date or date.today() - timedelta(days=365*5)  # 5 years
        end_date = end_date or date.today()
        
        df = get_history(
            symbol=base_ticker,
            start=start_date,
            end=end_date,
            series="EQ"
        )
        
        if not df.empty:
            # Standardize columns
            df = df.rename(columns={
                'Prev Close': 'Previous Close',
                'Turnover': 'Volume'
            })
            # Select relevant columns
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Deliverable Volume', '%Deliverble']
            df = df[[c for c in cols if c in df.columns]]
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            return df.sort_index()
            
    except ImportError:
        logger.warning("NSEPy not installed. Install with: pip install nsepy")
    except Exception as e:
        logger.error(f"NSEPy failed for {ticker}: {str(e)}")
    return pd.DataFrame()

def fetch_via_alpha_vantage(ticker, api_key=None):
    """Fetch data using Alpha Vantage API"""
    try:
        base_ticker = ticker.replace(".NS", "")
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": base_ticker,
            "outputsize": "full",
            "apikey": api_key or "demo",
            "datatype": "json"
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            # Convert to numeric
            df = df.apply(pd.to_numeric)
            return df.sort_index()
            
    except Exception as e:
        logger.error(f"Alpha Vantage failed for {ticker}: {str(e)}")
    return pd.DataFrame()

def fetch_via_nse_website(ticker, start_date=None, end_date=None):
    """Fetch data by scraping NSE website"""
    try:
        base_ticker = ticker.replace(".NS", "")
        session = requests.Session()
        session.headers.update(HEADERS)
        
        # First request to get cookies
        session.get("https://www.nseindia.com", timeout=5)
        
        # Format dates
        start_date = start_date or date.today() - timedelta(days=365*2)  # 2 years default
        end_date = end_date or date.today()
        
        # Fetch historical data
        params = {
            "symbol": base_ticker,
            "series": ["EQ"],
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        }
        
        response = session.get(NSE_DATA_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data:
            logger.warning(f"No data found for {ticker} on NSE website")
            return pd.DataFrame()
            
        # Parse JSON data
        df = pd.DataFrame(data['data'])
        
        if df.empty:
            return df
            
        # Clean and transform data
        df['CH_TIMESTAMP'] = pd.to_datetime(df['CH_TIMESTAMP'], format='%d-%b-%Y')
        df = df.rename(columns={
            'CH_TIMESTAMP': 'Date',
            'CH_OPENING_PRICE': 'Open',
            'CH_TRADE_HIGH_PRICE': 'High',
            'CH_TRADE_LOW_PRICE': 'Low',
            'CH_CLOSING_PRICE': 'Close',
            'CH_LAST_TRADED_PRICE': 'Last',
            'CH_TOT_TRADED_QTY': 'Volume',
            'CH_PREVIOUS_CLS_PRICE': 'Previous Close'
        })
        df = df.set_index('Date')
        
        # Select relevant columns
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous Close']
        df = df[[c for c in cols if c in df.columns]]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            
        return df.sort_index()
        
    except Exception as e:
        logger.error(f"NSE Website failed for {ticker}: {str(e)}")
    return pd.DataFrame()

def fetch_ticker_data(ticker, api_key=None):
    """Fetch data with fallback mechanism and caching"""
    cache_file = os.path.join(DATA_DIR, f"{ticker.replace('.NS', '')}.csv")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                logger.info(f"Loaded cached data for {ticker}")
                return df
        except Exception as e:
            logger.warning(f"Error loading cached data for {ticker}: {str(e)}")
            
    logger.info(f"Fetching fresh data for {ticker}")
    
    # Try sources in priority order
    df = fetch_via_nsepy(ticker)
    if not df.empty:
        df.to_csv(cache_file)
        return df
        
    df = fetch_via_alpha_vantage(ticker, api_key)
    if not df.empty:
        df.to_csv(cache_file)
        return df
        
    df = fetch_via_nse_website(ticker)
    if not df.empty:
        df.to_csv(cache_file)
        return df
        
    logger.warning(f"All sources failed for {ticker}")
    return pd.DataFrame()

def fetch_historical_data_parallel(tickers, api_key=None, max_workers=MAX_WORKERS):
    """Fetch data for multiple tickers in parallel"""
    historical_data = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ticker in tickers:
            futures[executor.submit(fetch_ticker_data, ticker, api_key)] = ticker
            time.sleep(0.1)  # Stagger requests to avoid rate limiting
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data"):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    historical_data[ticker] = df
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
    
    return historical_data