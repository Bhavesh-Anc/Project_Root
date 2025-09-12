# utils/news_sentiment.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from config import secrets

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer for stock-related news"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or secrets.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        
        # Enhanced sentiment words
        self.positive_words = [
            'bullish', 'growth', 'strong', 'buy', 'outperform', 'positive', 
            'profit', 'gain', 'surge', 'rally', 'boom', 'expansion',
            'breakthrough', 'success', 'earnings', 'revenue', 'upgrade'
        ]
        
        self.negative_words = [
            'bearish', 'decline', 'weak', 'sell', 'underperform', 'negative',
            'loss', 'fall', 'crash', 'recession', 'downgrade', 'concern',
            'risk', 'warning', 'deficit', 'bankruptcy', 'lawsuit'
        ]
    
    def fetch_news(self, query: str, days: int = 7) -> List[Dict]:
        """Fetch news articles with error handling"""
        if not self.api_key:
            logging.warning("No NEWS API key available")
            return []
        
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'pageSize': 20  # Reduced for stability
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logging.warning(f"News API returned status {response.status_code}")
                return []
        except Exception as e:
            logging.warning(f"News API error: {e}")
            return []
    
    def analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Simple scoring
        if positive_count > negative_count:
            return min(1.0, (positive_count - negative_count) / max(1, len(text.split()) / 10))
        elif negative_count > positive_count:
            return max(-1.0, -(negative_count - positive_count) / max(1, len(text.split()) / 10))
        else:
            return 0.0
    
    def get_ticker_sentiment(self, ticker: str, days: int = 7) -> float:
        """Get sentiment score for a specific ticker"""
        # Remove .NS suffix for news search
        search_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        articles = self.fetch_news(search_ticker, days)
        if not articles:
            return 0.0
        
        sentiments = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            
            sentiment = self.analyze_text_sentiment(content)
            sentiments.append(sentiment)
        
        return np.mean(sentiments) if sentiments else 0.0
    
    def get_market_sentiment(self, tickers: List[str]) -> Dict[str, float]:
        """Get sentiment scores for multiple tickers"""
        results = {}
        
        for ticker in tickers:
            try:
                sentiment = self.get_ticker_sentiment(ticker)
                results[ticker] = sentiment
            except Exception as e:
                logging.warning(f"Failed to get sentiment for {ticker}: {e}")
                results[ticker] = 0.0
        
        return results