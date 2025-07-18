# utils/data_loader.py
import os
import pandas as pd
from typing import List
from .indian_data_loader import get_nifty_indices_tickers

def fetch_nifty50_tickers(use_fallback=True) -> List[str]:
    """Fetch Nifty 50 tickers with fallback"""
    try:
        all_tickers = get_nifty_indices_tickers()
        # Return all tickers from Nifty indices
        return all_tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        if use_fallback:
            return [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
                "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "TITAN.NS",
                "SUNPHARMA.NS", "AXISBANK.NS", "DMART.NS", "ULTRACEMCO.NS", "TATASTEEL.NS",
                "BAJAJFINSV.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "NESTLEIND.NS",
                "ADANIPORTS.NS", "M&M.NS", "WIPRO.NS", "ITC.NS", "JSWSTEEL.NS", "INDUSINDBK.NS",
                "TECHM.NS", "DRREDDY.NS", "HDFCLIFE.NS", "COALINDIA.NS", "GRASIM.NS",
                "TATAMOTORS.NS", "BRITANNIA.NS", "CIPLA.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS",
                "UPL.NS", "DIVISLAB.NS", "SBILIFE.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
                "BPCL.NS", "IOC.NS", "HDFC.NS", "APOLLOHOSP.NS", "ADANIENT.NS"
            ]
        return []