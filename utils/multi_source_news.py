"""
Multi-Source News Aggregator for Enhanced Sentiment Analysis
Combines multiple free news APIs for better coverage and reliability
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from textblob import TextBlob
import yfinance as yf

logger = logging.getLogger(__name__)


class MultiSourceNewsAggregator:
    """Aggregate news from multiple free sources"""

    def __init__(self, config: Dict = None):
        """Initialize with API keys from config"""
        from config import secrets

        self.newsapi_key = secrets.NEWS_API_KEY
        self.alphavantage_key = secrets.ALPHA_VANTAGE_API_KEY

        # Free API endpoints
        self.sources = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'enabled': bool(self.newsapi_key),
                'rate_limit': 0.5  # seconds between requests
            },
            'alphavantage': {
                'url': 'https://www.alphavantage.co/query',
                'enabled': bool(self.alphavantage_key),
                'rate_limit': 1.0
            },
            'yahoo': {
                'enabled': True,  # No API key needed
                'rate_limit': 0.3
            },
            'finnhub': {
                'url': 'https://finnhub.io/api/v1/company-news',
                'enabled': False,  # User can add key
                'rate_limit': 1.0
            }
        }

        self.cache = {}
        self.last_request_time = {}

    def get_aggregated_news(self, ticker: str, days: int = 30, max_articles: int = 50) -> List[Dict]:
        """
        Aggregate news from multiple sources

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            max_articles: Maximum articles to return

        Returns:
            List of article dictionaries with sentiment
        """
        all_articles = []

        # Remove .NS suffix for API calls
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')

        # Fetch from each enabled source
        if self.sources['newsapi']['enabled']:
            try:
                newsapi_articles = self._fetch_newsapi(clean_ticker, days)
                all_articles.extend(newsapi_articles)
                logger.info(f"NewsAPI: {len(newsapi_articles)} articles for {ticker}")
            except Exception as e:
                logger.warning(f"NewsAPI failed for {ticker}: {e}")

        if self.sources['alphavantage']['enabled']:
            try:
                av_articles = self._fetch_alphavantage(clean_ticker)
                all_articles.extend(av_articles)
                logger.info(f"AlphaVantage: {len(av_articles)} articles for {ticker}")
            except Exception as e:
                logger.warning(f"AlphaVantage failed for {ticker}: {e}")

        if self.sources['yahoo']['enabled']:
            try:
                yahoo_articles = self._fetch_yahoo_news(ticker, days)
                all_articles.extend(yahoo_articles)
                logger.info(f"Yahoo: {len(yahoo_articles)} articles for {ticker}")
            except Exception as e:
                logger.warning(f"Yahoo News failed for {ticker}: {e}")

        # Remove duplicates based on title similarity
        unique_articles = self._deduplicate_articles(all_articles)

        # Add sentiment analysis
        articles_with_sentiment = []
        for article in unique_articles[:max_articles]:
            article_with_sentiment = self._add_sentiment(article)
            articles_with_sentiment.append(article_with_sentiment)

        # Sort by date (newest first)
        articles_with_sentiment.sort(
            key=lambda x: x.get('published_at', ''),
            reverse=True
        )

        logger.info(f"Total unique articles for {ticker}: {len(articles_with_sentiment)}")
        return articles_with_sentiment

    def _fetch_newsapi(self, ticker: str, days: int) -> List[Dict]:
        """Fetch from NewsAPI.org"""
        self._rate_limit('newsapi')

        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        params = {
            'q': ticker,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.newsapi_key
        }

        try:
            response = requests.get(
                self.sources['newsapi']['url'],
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                articles = []

                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', 'No Title'),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}",
                        'published_at': article.get('publishedAt', ''),
                        'image_url': article.get('urlToImage', ''),
                        'content': article.get('content', ''),
                        'api_source': 'newsapi'
                    })

                return articles
            else:
                logger.warning(f"NewsAPI returned status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

    def _fetch_alphavantage(self, ticker: str) -> List[Dict]:
        """Fetch from Alpha Vantage News Sentiment API"""
        self._rate_limit('alphavantage')

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': self.alphavantage_key,
            'limit': 50
        }

        try:
            response = requests.get(
                self.sources['alphavantage']['url'],
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                articles = []

                for item in data.get('feed', []):
                    articles.append({
                        'title': item.get('title', 'No Title'),
                        'description': item.get('summary', ''),
                        'url': item.get('url', ''),
                        'source': f"Alpha Vantage - {', '.join([s.get('name', 'Unknown') for s in item.get('authors', [])][:2])}",
                        'published_at': item.get('time_published', ''),
                        'image_url': item.get('banner_image', ''),
                        'content': item.get('summary', ''),
                        'api_source': 'alphavantage',
                        # Alpha Vantage provides sentiment
                        'av_sentiment_score': item.get('overall_sentiment_score', 0),
                        'av_sentiment_label': item.get('overall_sentiment_label', 'Neutral')
                    })

                return articles
            else:
                logger.warning(f"AlphaVantage returned status {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"AlphaVantage error: {e}")
            return []

    def _fetch_yahoo_news(self, ticker: str, days: int) -> List[Dict]:
        """Fetch news from Yahoo Finance via yfinance"""
        self._rate_limit('yahoo')

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                return []

            cutoff_date = datetime.now() - timedelta(days=days)
            articles = []

            for item in news:
                # Check if article is within date range
                pub_timestamp = item.get('providerPublishTime', 0)
                if pub_timestamp:
                    pub_date = datetime.fromtimestamp(pub_timestamp)
                    if pub_date < cutoff_date:
                        continue

                    published_at = pub_date.isoformat()
                else:
                    published_at = datetime.now().isoformat()

                articles.append({
                    'title': item.get('title', 'No Title'),
                    'description': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'source': f"Yahoo Finance - {item.get('publisher', 'Unknown')}",
                    'published_at': published_at,
                    'image_url': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', ''),
                    'content': item.get('summary', ''),
                    'api_source': 'yahoo'
                })

            return articles

        except Exception as e:
            logger.error(f"Yahoo News error: {e}")
            return []

    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []

        unique_articles = []
        seen_titles = set()

        for article in articles:
            title = article.get('title', '').lower().strip()

            # Skip if no title
            if not title or title == 'no title':
                continue

            # Simple deduplication - check if title already seen
            # For better deduplication, could use fuzzy matching
            title_key = ''.join(title.split()[:5])  # First 5 words

            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        return unique_articles

    def _add_sentiment(self, article: Dict) -> Dict:
        """Add sentiment analysis to article"""

        # If Alpha Vantage already provided sentiment, convert it
        if article.get('api_source') == 'alphavantage' and 'av_sentiment_score' in article:
            av_score = article['av_sentiment_score']
            sentiment_score = float(av_score)
        else:
            # Use TextBlob for sentiment analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"

            if text.strip():
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
            else:
                sentiment_score = 0.0

        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment_label = "🟢 Positive"
        elif sentiment_score < -0.1:
            sentiment_label = "🔴 Negative"
        else:
            sentiment_label = "🟡 Neutral"

        article['sentiment_score'] = sentiment_score
        article['sentiment_label'] = sentiment_label

        return article

    def _rate_limit(self, source: str):
        """Apply rate limiting for API calls"""
        if source not in self.last_request_time:
            self.last_request_time[source] = 0

        time_since_last = time.time() - self.last_request_time[source]
        rate_limit = self.sources[source]['rate_limit']

        if time_since_last < rate_limit:
            time.sleep(rate_limit - time_since_last)

        self.last_request_time[source] = time.time()

    def get_sentiment_summary(self, articles: List[Dict]) -> Dict:
        """Get aggregate sentiment statistics"""
        if not articles:
            return {
                'avg_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'sources': []
            }

        sentiments = [a.get('sentiment_score', 0) for a in articles]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        positive = sum(1 for s in sentiments if s > 0.1)
        negative = sum(1 for s in sentiments if s < -0.1)
        neutral = len(sentiments) - positive - negative

        sources = list(set(a.get('api_source', 'unknown') for a in articles))

        return {
            'avg_sentiment': avg_sentiment,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'total_articles': len(articles),
            'sources': sources,
            'coverage_quality': 'Excellent' if len(sources) >= 3 else 'Good' if len(sources) >= 2 else 'Basic'
        }


# Convenience function for backward compatibility
def get_multi_source_news(ticker: str, days: int = 30, max_articles: int = 50) -> List[Dict]:
    """Get news from multiple sources"""
    aggregator = MultiSourceNewsAggregator()
    return aggregator.get_aggregated_news(ticker, days, max_articles)


if __name__ == "__main__":
    # Test the aggregator
    print("Testing Multi-Source News Aggregator...")

    aggregator = MultiSourceNewsAggregator()

    # Test with a ticker
    test_ticker = "RELIANCE.NS"
    print(f"\nFetching news for {test_ticker}...")

    articles = aggregator.get_aggregated_news(test_ticker, days=7, max_articles=10)

    print(f"\nFound {len(articles)} articles")

    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Sentiment: {article['sentiment_label']} ({article['sentiment_score']:.3f})")

    # Get summary
    summary = aggregator.get_sentiment_summary(articles)
    print(f"\nSentiment Summary:")
    print(f"  Average: {summary['avg_sentiment']:.3f}")
    print(f"  Positive: {summary['positive_count']}, Negative: {summary['negative_count']}, Neutral: {summary['neutral_count']}")
    print(f"  Sources: {', '.join(summary['sources'])}")
    print(f"  Quality: {summary['coverage_quality']}")
