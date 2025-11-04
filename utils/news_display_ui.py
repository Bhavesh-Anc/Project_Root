"""
News Display UI Components for Streamlit
Provides ready-to-use UI components for displaying news articles with sentiment
"""

import streamlit as st
import pandas as pd
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def display_news_article_card(article: Dict):
    """
    Display a single news article in a nice card format

    Args:
        article: Dictionary containing article data
    """
    try:
        # Create a card-like container
        with st.container():
            # Sentiment indicator
            sentiment_label = article.get('sentiment_label', '🟡 Neutral')
            sentiment_score = article.get('sentiment_score', 0)

            # Color code based on sentiment
            if '🟢' in sentiment_label:
                border_color = "#00C851"  # Green
            elif '🔴' in sentiment_label:
                border_color = "#ff4444"  # Red
            else:
                border_color = "#ffbb33"  # Yellow

            # Article container with custom styling
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {border_color};
                    padding: 15px;
                    margin-bottom: 15px;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 5px;
                ">
                """,
                unsafe_allow_html=True
            )

            # Title and sentiment
            col1, col2 = st.columns([4, 1])

            with col1:
                title = article.get('title', 'No Title')
                url = article.get('url', '')
                if url:
                    st.markdown(f"### [{title}]({url})")
                else:
                    st.markdown(f"### {title}")

            with col2:
                st.markdown(f"**{sentiment_label}**")
                st.caption(f"Score: {sentiment_score:.2f}")

            # Description
            description = article.get('description', 'No description available')
            st.write(description)

            # Footer with source and date
            col1, col2 = st.columns(2)

            with col1:
                source = article.get('source', 'Unknown')
                st.caption(f"📰 **Source:** {source}")

            with col2:
                published_at = article.get('published_at', '')
                if published_at:
                    try:
                        # Parse and format date
                        date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%B %d, %Y at %I:%M %p')
                        st.caption(f"🕒 **Published:** {formatted_date}")
                    except:
                        st.caption(f"🕒 {published_at}")

            # Close the styled div
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error displaying news article: {e}")
        st.error("Error displaying this article")


def display_news_feed(ticker: str, articles: List[Dict]):
    """
    Display a feed of news articles for a stock

    Args:
        ticker: Stock ticker symbol
        articles: List of article dictionaries
    """
    try:
        if not articles:
            st.info(f"📭 No recent news found for {ticker}")
            st.caption("This could be due to:")
            st.caption("• Limited news coverage for this stock")
            st.caption("• API rate limits")
            st.caption("• News API key configuration")
            return

        # Header
        st.subheader(f"📰 Recent News for {ticker}")
        st.caption(f"Showing {len(articles)} articles from the last 30 days")

        # Sentiment summary
        positive = sum(1 for a in articles if '🟢' in a.get('sentiment_label', ''))
        negative = sum(1 for a in articles if '🔴' in a.get('sentiment_label', ''))
        neutral = len(articles) - positive - negative

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Positive", positive)
        with col2:
            st.metric("🟡 Neutral", neutral)
        with col3:
            st.metric("🔴 Negative", negative)

        st.markdown("---")

        # Filter options
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All", "Positive", "Negative", "Neutral"],
            key=f"sentiment_filter_{ticker}"
        )

        # Filter articles
        filtered_articles = articles
        if sentiment_filter == "Positive":
            filtered_articles = [a for a in articles if '🟢' in a.get('sentiment_label', '')]
        elif sentiment_filter == "Negative":
            filtered_articles = [a for a in articles if '🔴' in a.get('sentiment_label', '')]
        elif sentiment_filter == "Neutral":
            filtered_articles = [a for a in articles if '🟡' in a.get('sentiment_label', '')]

        if not filtered_articles:
            st.info(f"No {sentiment_filter.lower()} articles found")
            return

        st.caption(f"Displaying {len(filtered_articles)} articles")

        # Display articles
        for article in filtered_articles:
            display_news_article_card(article)

    except Exception as e:
        logger.error(f"Error displaying news feed for {ticker}: {e}")
        st.error(f"Error displaying news feed: {e}")


def display_sentiment_comparison(tickers: List[str], news_data: Dict[str, List[Dict]]):
    """
    Display sentiment comparison across multiple stocks

    Args:
        tickers: List of stock tickers
        news_data: Dictionary mapping ticker to list of articles
    """
    try:
        st.subheader("📊 News Sentiment Comparison")

        if not news_data:
            st.warning("No news data available for comparison")
            return

        # Calculate sentiment stats for each ticker
        comparison_data = []

        for ticker in tickers:
            articles = news_data.get(ticker, [])
            if not articles:
                continue

            total = len(articles)
            positive = sum(1 for a in articles if '🟢' in a.get('sentiment_label', ''))
            negative = sum(1 for a in articles if '🔴' in a.get('sentiment_label', ''))
            neutral = total - positive - negative

            avg_sentiment = sum(a.get('sentiment_score', 0) for a in articles) / total if total > 0 else 0

            comparison_data.append({
                'Ticker': ticker,
                'Total Articles': total,
                '🟢 Positive': positive,
                '🟡 Neutral': neutral,
                '🔴 Negative': negative,
                'Avg Sentiment': round(avg_sentiment, 3),
                'Positive %': round((positive / total * 100) if total > 0 else 0, 1)
            })

        if not comparison_data:
            st.info("No news articles found for any of the selected stocks")
            return

        df = pd.DataFrame(comparison_data)

        # Display as styled dataframe
        st.dataframe(
            df.style.background_gradient(
                subset=['Avg Sentiment'],
                cmap='RdYlGn',
                vmin=-1,
                vmax=1
            ).background_gradient(
                subset=['Positive %'],
                cmap='Greens'
            ),
            width='stretch',
            hide_index=True
        )

        # Top movers
        if len(comparison_data) > 1:
            st.markdown("#### 📈 Sentiment Leaders")

            col1, col2 = st.columns(2)

            with col1:
                # Most positive
                most_positive = max(comparison_data, key=lambda x: x['Avg Sentiment'])
                st.success(f"**Most Positive:** {most_positive['Ticker']}")
                st.caption(f"Avg Sentiment: {most_positive['Avg Sentiment']:.3f}")
                st.caption(f"{most_positive['Positive %']:.1f}% positive articles")

            with col2:
                # Most negative
                most_negative = min(comparison_data, key=lambda x: x['Avg Sentiment'])
                st.error(f"**Most Negative:** {most_negative['Ticker']}")
                st.caption(f"Avg Sentiment: {most_negative['Avg Sentiment']:.3f}")
                st.caption(f"{most_negative['🔴 Negative']} negative articles")

    except Exception as e:
        logger.error(f"Error displaying sentiment comparison: {e}")
        st.error(f"Error displaying comparison: {e}")


def display_news_timeline(articles: List[Dict], ticker: str):
    """
    Display news articles in a timeline format

    Args:
        articles: List of article dictionaries
        ticker: Stock ticker symbol
    """
    try:
        if not articles:
            st.info(f"No news articles to display for {ticker}")
            return

        st.subheader(f"📅 News Timeline for {ticker}")

        # Group articles by date
        from collections import defaultdict
        articles_by_date = defaultdict(list)

        for article in articles:
            published_at = article.get('published_at', '')
            if published_at:
                try:
                    date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    date_key = date_obj.strftime('%Y-%m-%d')
                    articles_by_date[date_key].append(article)
                except:
                    articles_by_date['Unknown'].append(article)

        # Display by date
        for date_key in sorted(articles_by_date.keys(), reverse=True):
            if date_key != 'Unknown':
                try:
                    date_obj = datetime.strptime(date_key, '%Y-%m-%d')
                    display_date = date_obj.strftime('%B %d, %Y')
                except:
                    display_date = date_key
            else:
                display_date = 'Date Unknown'

            st.markdown(f"### 📆 {display_date}")
            st.caption(f"{len(articles_by_date[date_key])} articles")

            for article in articles_by_date[date_key]:
                display_news_article_card(article)

            st.markdown("---")

    except Exception as e:
        logger.error(f"Error displaying news timeline: {e}")
        st.error(f"Error displaying timeline: {e}")


if __name__ == "__main__":
    print("News Display UI Module - Ready for integration with app.py")
    print("Available functions:")
    print("  - display_news_article_card(article)")
    print("  - display_news_feed(ticker, articles)")
    print("  - display_sentiment_comparison(tickers, news_data)")
    print("  - display_news_timeline(articles, ticker)")
