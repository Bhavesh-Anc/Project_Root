# utils/data_loader.py

import requests
import yfinance as yf
import pandas as pd
import os
import time
import random
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Expanded fallback list: NIFTY 50 + NIFTY Next 50 + some popular midcaps
NIFTY100_FALLBACK = [
    # NIFTY 50
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "AXISBANK.NS", "DMART.NS", "ULTRACEMCO.NS", "TATASTEEL.NS",
    "BAJAJFINSV.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "NESTLEIND.NS",
    "ADANIPORTS.NS", "M&M.NS", "WIPRO.NS", "ITC.NS", "JSWSTEEL.NS", "INDUSINDBK.NS",
    "TECHM.NS", "DRREDDY.NS", "HDFCLIFE.NS", "COALINDIA.NS", "GRASIM.NS",
    "TATAMOTORS.NS", "BRITANNIA.NS", "CIPLA.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS",
    "UPL.NS", "DIVISLAB.NS", "SBILIFE.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BPCL.NS", "IOC.NS", "HDFC.NS", "APOLLOHOSP.NS", "ADANIENT.NS",
    # NIFTY NEXT 50 and other large/midcaps (partial, update as needed)
    "ABB.NS", "ACC.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "AUROPHARMA.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS",
    "COLPAL.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS",
    "HAVELLS.NS", "HINDPETRO.NS", "ICICIGI.NS", "ICICIPRULI.NS", "INDIGO.NS",
    "LTI.NS", "LTIM.NS", "MARICO.NS", "MCDOWELL-N.NS", "MUTHOOTFIN.NS",
    "NAUKRI.NS", "PEL.NS", "PGHH.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POLYCAB.NS", "SBICARD.NS", "SIEMENS.NS", "SRF.NS",
    "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "VBL.NS",
    "VEDL.NS", "ZOMATO.NS", "GLAND.NS", "JUBLFOOD.NS", "ALKEM.NS",
    "APLLTD.NS", "BAJAJHLDNG.NS", "BIOCON.NS", "IDFCFIRSTB.NS",
    "NHPC.NS", "IRFC.NS", "IRCTC.NS", "BHEL.NS", "ADANIPOWER.NS",
    "TATAPOWER.NS", "UNIONBANK.NS", "FEDERALBNK.NS", "CUB.NS", "IEX.NS",
    "MOTHERSON.NS"
]

def fetch_nifty50_tickers(use_fallback: bool = True) -> List[str]:
    """
    Fetch live Nifty 50 tickers from NSE API with fallback to a cached list.
    Returns: List of tickers in Yahoo Finance format (e.g., 'RELIANCE.NS')
    """
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [item['symbol'] + ".NS" for item in data['data']]
    except Exception as e:
        logging.warning(f"Failed to fetch live Nifty 50 tickers: {e}")
        if use_fallback:
            logging.info("Using fallback ticker list (NIFTY100_FALLBACK)")
            return NIFTY100_FALLBACK
        else:
            raise ValueError("Failed to fetch tickers and fallback disabled") from e

def fetch_historical_data_parallel(
    tickers: List[str],
    max_retries: int = 3,
    delay: float = 2,
    max_workers: int = 4,
    period: str = "max",
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data in parallel with progress bar and logging.
    Args:
        tickers: List of ticker symbols.
        max_retries: Number of download attempts per ticker.
        delay: Minimum seconds to wait after each ticker (avoid rate limiting).
        max_workers: Number of parallel threads.
        period: Data period (e.g., "max", "1y").
        start, end: Optional custom date range (YYYY-MM-DD).
    Returns:
        Dictionary of {ticker: DataFrame with historical data}
    """
    def fetch_single(ticker: str):
        for retry in range(max_retries):
            try:
                if start or end:
                    data = yf.Ticker(ticker).history(start=start, end=end, period=period)
                else:
                    data = yf.Ticker(ticker).history(period=period)
                if not data.empty:
                    logging.info(f"Success: {ticker} ({len(data)} rows)")
                    return ticker, data
                else:
                    logging.warning(f"No data for {ticker} (possibly delisted)")
                    return ticker, None
            except Exception as e:
                if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                    logging.error(f"{ticker} rate-limited ({retry+1}/{max_retries}), sleeping...")
                    # Exponential backoff with some random jitter to avoid simultaneous retries
                    sleep_time = delay * (2 ** retry) + random.uniform(1, 5)
                    time.sleep(sleep_time)
                else:
                    logging.error(f"{ticker} failed ({retry+1}/{max_retries}): {e}")
                    time.sleep(delay + random.uniform(1, 3))
        logging.error(f"All retries exhausted for {ticker}")
        return ticker, None

    historical_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single, ticker): ticker for ticker in tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching data"):
            ticker, data = future.result()
            if data is not None:
                historical_data[ticker] = data
            # Short sleep to avoid hammering the API (especially important for Yahoo)
            time.sleep(delay + random.uniform(0, 2))
    return historical_data

def save_to_csv(
    data_dict: Dict[str, pd.DataFrame],
    dir_path: str = "data/historical"
) -> None:
    """
    Save historical data to CSV files.
    Args:
        data_dict: Dictionary of DataFrames from fetch_historical_data_parallel()
        dir_path: Directory to save CSV files.
    """
    os.makedirs(dir_path, exist_ok=True)
    for ticker, data in data_dict.items():
        clean_ticker = ticker.replace(".NS", "")
        file_path = os.path.join(dir_path, f"{clean_ticker}.csv")
        data.to_csv(file_path)
        logging.info(f"Saved {len(data)} rows to {file_path}")

def load_from_csv(
    dir_path: str = "data/historical"
) -> Dict[str, pd.DataFrame]:
    """
    Load historical data from CSV files.
    Returns:
        Dictionary of {ticker: DataFrame with historical data}
    """
    data_dict = {}
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".csv"):
            ticker = file_name.replace(".csv", "") + ".NS"
            file_path = os.path.join(dir_path, file_name)
            try:
                data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
                if not data.empty:
                    data_dict[ticker] = data
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
    logging.info(f"Loaded data for {len(data_dict)} tickers from {dir_path}")
    return data_dict

# Optional: utility to get latest valid tickers from fallback (removes delisted ones)
def filter_valid_tickers(tickers: List[str]) -> List[str]:
    valid = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if info and info.get("regularMarketPrice", None) is not None:
                valid.append(ticker)
        except Exception:
            continue
    return valid

# Example usage:
if __name__ == "__main__":
    tickers = fetch_nifty50_tickers()
    data = fetch_historical_data_parallel(
        tickers, max_retries=3, delay=3, max_workers=3, period="max"
    )
    save_to_csv(data, dir_path="data/historical")