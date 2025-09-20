# utils/news_sentiment.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from config import secrets

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer for stock-related news with user selection support"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or secrets.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        
        # Enhanced sentiment words
        self.positive_words = [
            'bullish', 'growth', 'strong', 'buy', 'outperform', 'positive', 
            'profit', 'gain', 'surge', 'rally', 'boom', 'expansion',
            'breakthrough', 'success', 'earnings', 'revenue', 'upgrade',
            'robust', 'healthy', 'optimistic', 'confident', 'promising',
            'beat', 'exceed', 'momentum', 'rise', 'climb', 'soar'
        ]
        
        self.negative_words = [
            'bearish', 'decline', 'weak', 'sell', 'underperform', 'negative',
            'loss', 'fall', 'crash', 'recession', 'downgrade', 'concern',
            'risk', 'warning', 'deficit', 'bankruptcy', 'lawsuit',
            'weak', 'poor', 'disappointing', 'miss', 'below', 'struggle',
            'pressure', 'challenge', 'volatility', 'uncertainty', 'drop'
        ]
        
        # Cache for recent sentiment analysis
        self.sentiment_cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    def fetch_news(self, query: str, days: int = 7) -> List[Dict]:
        """Fetch news articles with error handling"""
        if not self.api_key:
            logging.warning("No NEWS API key available")
            return []
        
        # Check cache first
        cache_key = f"{query}_{days}"
        if cache_key in self.sentiment_cache:
            cached_data, cached_time = self.sentiment_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logging.info(f"Using cached news for {query}")
                return cached_data
        
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'pageSize': 30  # Increased for better analysis
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Cache the results
                self.sentiment_cache[cache_key] = (articles, datetime.now())
                
                logging.info(f"Fetched {len(articles)} articles for {query}")
                return articles
            else:
                logging.warning(f"News API returned status {response.status_code} for {query}")
                return []
        except Exception as e:
            logging.warning(f"News API error for {query}: {e}")
            return []
    
    def analyze_text_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis of text"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count positive and negative words with weights
        positive_score = 0
        negative_score = 0
        
        for word in words:
            if word in self.positive_words:
                positive_score += 1
            elif word in self.negative_words:
                negative_score += 1
        
        # Enhanced scoring with context
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Calculate weighted sentiment
        positive_weight = positive_score / total_words
        negative_weight = negative_score / total_words
        
        # Normalize to [-1, 1] range
        net_sentiment = (positive_weight - negative_weight) * 10  # Amplify signal
        return max(-1.0, min(1.0, net_sentiment))
    
    def get_ticker_sentiment(self, ticker: str, days: int = 7) -> float:
        """Get sentiment score for a specific ticker with enhanced analysis"""
        # Remove .NS/.BO suffix for news search and add company context
        search_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        # Try multiple search queries for better coverage
        search_queries = [
            search_ticker,
            f"{search_ticker} stock",
            f"{search_ticker} shares",
            f"{search_ticker} earnings"
        ]
        
        all_sentiments = []
        
        for query in search_queries:
            articles = self.fetch_news(query, days)
            if not articles:
                continue
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                # Filter relevant articles (containing ticker name)
                if search_ticker.lower() in content.lower():
                    sentiment = self.analyze_text_sentiment(content)
                    
                    # Weight recent articles more heavily
                    published_at = article.get('publishedAt', '')
                    try:
                        article_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        days_old = (datetime.now(article_date.tzinfo) - article_date).days
                        weight = max(0.1, 1.0 - (days_old / days))  # Recent articles get higher weight
                        weighted_sentiment = sentiment * weight
                        all_sentiments.append(weighted_sentiment)
                    except:
                        all_sentiments.append(sentiment)
        
        if not all_sentiments:
            logging.info(f"No relevant sentiment data found for {ticker}")
            return 0.0
        
        # Calculate weighted average sentiment
        final_sentiment = np.mean(all_sentiments)
        
        logging.info(f"Sentiment for {ticker}: {final_sentiment:.3f} (based on {len(all_sentiments)} articles)")
        return final_sentiment
    
    def get_market_sentiment(self, tickers: List[str]) -> Dict[str, float]:
        """Get sentiment scores for multiple tickers (optimized for user selection)"""
        results = {}
        
        logging.info(f"Analyzing sentiment for {len(tickers)} selected tickers")
        
        for i, ticker in enumerate(tickers):
            try:
                logging.info(f"Processing sentiment for {ticker} ({i+1}/{len(tickers)})")
                sentiment = self.get_ticker_sentiment(ticker)
                results[ticker] = sentiment
                
                # Add small delay to avoid rate limiting
                if i < len(tickers) - 1:  # Don't delay after last ticker
                    import time
                    time.sleep(0.5)
                    
            except Exception as e:
                logging.warning(f"Failed to get sentiment for {ticker}: {e}")
                results[ticker] = 0.0
        
        # Calculate overall market sentiment for selected stocks
        if results:
            overall_sentiment = np.mean(list(results.values()))
            logging.info(f"Overall market sentiment for selected stocks: {overall_sentiment:.3f}")
        
        return results
    
    def get_sentiment_summary(self, tickers: List[str]) -> Dict[str, any]:
        """Get comprehensive sentiment summary for selected tickers"""
        sentiment_scores = self.get_market_sentiment(tickers)
        
        if not sentiment_scores:
            return {
                'overall_sentiment': 0.0,
                'positive_stocks': 0,
                'negative_stocks': 0,
                'neutral_stocks': 0,
                'top_positive': [],
                'top_negative': [],
                'sentiment_distribution': {}
            }
        
        # Analyze sentiment distribution
        positive_stocks = [ticker for ticker, score in sentiment_scores.items() if score > 0.1]
        negative_stocks = [ticker for ticker, score in sentiment_scores.items() if score < -0.1]
        neutral_stocks = [ticker for ticker, score in sentiment_scores.items() if -0.1 <= score <= 0.1]
        
        # Get top performers
        sorted_sentiments = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
        top_positive = sorted_sentiments[:5] if len(sorted_sentiments) >= 5 else sorted_sentiments
        top_negative = sorted_sentiments[-5:] if len(sorted_sentiments) >= 5 else []
        
        return {
            'overall_sentiment': np.mean(list(sentiment_scores.values())),
            'positive_stocks': len(positive_stocks),
            'negative_stocks': len(negative_stocks),
            'neutral_stocks': len(neutral_stocks),
            'top_positive': top_positive,
            'top_negative': top_negative,
            'sentiment_distribution': sentiment_scores,
            'total_analyzed': len(sentiment_scores)
        }
    
    def clear_cache(self):
        """Clear sentiment cache"""
        self.sentiment_cache = {}
        logging.info("Sentiment cache cleared")

# Enhanced helper functions for integration
def get_sentiment_for_selected_stocks(selected_tickers: List[str], 
                                    api_key: str = None,
                                    days: int = 7) -> Dict[str, float]:
    """
    Convenience function to get sentiment for user-selected stocks
    
    Args:
        selected_tickers: List of user-selected stock tickers
        api_key: News API key (optional, will use config if not provided)
        days: Number of days to look back for news
    
    Returns:
        Dictionary with ticker -> sentiment score mapping
    """
    analyzer = AdvancedSentimentAnalyzer(api_key)
    return analyzer.get_market_sentiment(selected_tickers)

def get_sentiment_insights(selected_tickers: List[str],
                         api_key: str = None) -> Dict[str, any]:
    """
    Get comprehensive sentiment insights for selected stocks
    
    Args:
        selected_tickers: List of user-selected stock tickers
        api_key: News API key (optional)
    
    Returns:
        Comprehensive sentiment analysis results
    """
    analyzer = AdvancedSentimentAnalyzer(api_key)
    return analyzer.get_sentiment_summary(selected_tickers)

# Example usage
if __name__ == "__main__":
    print("Enhanced News Sentiment Analyzer - User Selection Version")
    print("="*60)
    
    # Test with sample selected stocks
    sample_selected_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    
    try:
        analyzer = AdvancedSentimentAnalyzer()
        
        print(f"Analyzing sentiment for {len(sample_selected_stocks)} selected stocks:")
        for ticker in sample_selected_stocks:
            print(f"  - {ticker}")
        
        # Get sentiment analysis
        sentiment_results = analyzer.get_market_sentiment(sample_selected_stocks)
        
        print(f"\nSentiment Analysis Results:")
        for ticker, sentiment in sentiment_results.items():
            sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
            print(f"  {ticker}: {sentiment:.3f} ({sentiment_label})")
        
        # Get comprehensive summary
        summary = analyzer.get_sentiment_summary(sample_selected_stocks)
        
        print(f"\nSentiment Summary:")
        print(f"  Overall Sentiment: {summary['overall_sentiment']:.3f}")
        print(f"  Positive Stocks: {summary['positive_stocks']}")
        print(f"  Negative Stocks: {summary['negative_stocks']}")
        print(f"  Neutral Stocks: {summary['neutral_stocks']}")
        
        if summary['top_positive']:
            print(f"\nTop Positive Sentiments:")
            for ticker, score in summary['top_positive'][:3]:
                print(f"    {ticker}: {score:.3f}")
        
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        print("This might be due to missing API key or network issues")
    
    print(f"\nUser Selection Features:")
    print(f"  ✓ Optimized for user-selected stocks")
    print(f"  ✓ Enhanced sentiment analysis with multiple queries")
    print(f"  ✓ Caching for improved performance")
    print(f"  ✓ Weighted recent news more heavily")
    print(f"  ✓ Comprehensive sentiment summary")
    print(f"  ✓ Rate limiting to avoid API issues")
    print(f"  ✓ Error handling and fallback mechanisms")