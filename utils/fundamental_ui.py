"""
Fundamental Analysis UI Components for Streamlit
Provides ready-to-use UI components for displaying fundamental data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def display_fundamental_summary(ticker: str, fundamental_data: Dict[str, Any]):
    """
    Display a comprehensive fundamental analysis summary for a single stock

    Args:
        ticker: Stock ticker symbol
        fundamental_data: Dictionary containing fundamental data from AdvancedDataAggregator
    """
    try:
        st.subheader(f"📊 Fundamental Analysis: {ticker}")

        if not fundamental_data or not fundamental_data.get('fundamentals'):
            st.warning(f"No fundamental data available for {ticker}")
            return

        fundamentals = fundamental_data.get('fundamentals', {})
        fund_score = fundamental_data.get('fundamental_score', 50)
        insider_data = fundamental_data.get('insider_activity', {})
        institutional_data = fundamental_data.get('institutional', {})

        # Create 4 columns for key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Fundamental Score",
                f"{fund_score:.1f}/100",
                delta=f"{fund_score - 50:.1f} vs neutral",
                delta_color="normal"
            )

        with col2:
            insider_sentiment = insider_data.get('sentiment', 'neutral')
            insider_score = insider_data.get('score', 0)
            st.metric(
                "Insider Sentiment",
                insider_sentiment.upper(),
                delta=f"Score: {insider_score:.1f}",
                delta_color="normal" if insider_score >= 0 else "inverse"
            )

        with col3:
            institutional_score = institutional_data.get('ownership_score', 5)
            st.metric(
                "Institutional Score",
                f"{institutional_score:.1f}/10",
                delta=f"{institutional_score - 5:.1f} vs avg",
                delta_color="normal"
            )

        with col4:
            sector = fundamentals.get('sector', 'Unknown')
            st.metric("Sector", sector, delta=fundamentals.get('industry', 'N/A'))

        # Create tabs for different categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Valuation", "💰 Profitability", "📊 Growth", "💪 Financial Health", "👥 Ownership"
        ])

        with tab1:
            _display_valuation_metrics(fundamentals)

        with tab2:
            _display_profitability_metrics(fundamentals)

        with tab3:
            _display_growth_metrics(fundamentals)

        with tab4:
            _display_financial_health_metrics(fundamentals)

        with tab5:
            _display_ownership_data(insider_data, institutional_data)

    except Exception as e:
        logger.error(f"Error displaying fundamental summary for {ticker}: {e}")
        st.error(f"Error displaying fundamental data: {e}")


def _display_valuation_metrics(fundamentals: Dict):
    """Display valuation metrics"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Price Ratios")

        metrics = {
            "P/E Ratio": fundamentals.get('pe_ratio'),
            "Forward P/E": fundamentals.get('forward_pe'),
            "PEG Ratio": fundamentals.get('peg_ratio'),
            "Price to Book": fundamentals.get('price_to_book'),
            "Price to Sales": fundamentals.get('price_to_sales'),
            "EV/EBITDA": fundamentals.get('ev_to_ebitda')
        }

        df_metrics = pd.DataFrame([
            {"Metric": k, "Value": f"{v:.2f}" if v else "N/A"}
            for k, v in metrics.items()
        ])
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 💵 Market Capitalization")

        market_cap = fundamentals.get('market_cap')
        if market_cap:
            market_cap_b = market_cap / 1e9
            st.metric("Market Cap", f"${market_cap_b:.2f}B")

            # Create gauge chart for valuation
            pe_ratio = fundamentals.get('pe_ratio', 20)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pe_ratio,
                title={'text': "P/E Ratio"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 25], 'color': "lightyellow"},
                        {'range': [25, 50], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)


def _display_profitability_metrics(fundamentals: Dict):
    """Display profitability metrics"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💹 Margins")

        margins = {
            "Profit Margin": fundamentals.get('profit_margin'),
            "Operating Margin": fundamentals.get('operating_margin'),
            "Gross Margin": fundamentals.get('gross_margin')
        }

        # Create bar chart
        df_margins = pd.DataFrame([
            {"Margin Type": k, "Percentage": v * 100 if v else 0}
            for k, v in margins.items()
        ])

        fig = px.bar(
            df_margins,
            x="Margin Type",
            y="Percentage",
            title="Profit Margins (%)",
            color="Percentage",
            color_continuous_scale="Greens"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📈 Returns")

        returns = {
            "ROE (Return on Equity)": fundamentals.get('roe'),
            "ROA (Return on Assets)": fundamentals.get('roa')
        }

        for metric, value in returns.items():
            if value:
                st.metric(metric, f"{value * 100:.2f}%")
            else:
                st.metric(metric, "N/A")


def _display_growth_metrics(fundamentals: Dict):
    """Display growth metrics"""
    st.markdown("#### 📊 Growth Rates")

    growth_data = {
        "Revenue Growth": fundamentals.get('revenue_growth'),
        "Earnings Growth": fundamentals.get('earnings_growth'),
        "Quarterly Earnings Growth": fundamentals.get('earnings_quarterly_growth')
    }

    col1, col2, col3 = st.columns(3)

    for i, (metric, value) in enumerate(growth_data.items()):
        with [col1, col2, col3][i]:
            if value is not None:
                st.metric(
                    metric,
                    f"{value * 100:.2f}%",
                    delta=f"{'Growth' if value > 0 else 'Decline'}",
                    delta_color="normal" if value >= 0 else "inverse"
                )
            else:
                st.metric(metric, "N/A")

    # Create growth trend visualization
    df_growth = pd.DataFrame([
        {"Metric": k, "Growth %": v * 100 if v else 0}
        for k, v in growth_data.items()
    ])

    fig = px.bar(
        df_growth,
        x="Metric",
        y="Growth %",
        title="Growth Metrics Comparison",
        color="Growth %",
        color_continuous_scale=["red", "yellow", "green"]
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_financial_health_metrics(fundamentals: Dict):
    """Display financial health metrics"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💪 Liquidity Ratios")

        current_ratio = fundamentals.get('current_ratio')
        quick_ratio = fundamentals.get('quick_ratio')

        if current_ratio:
            st.metric(
                "Current Ratio",
                f"{current_ratio:.2f}",
                delta="Healthy" if current_ratio > 1.5 else "Low",
                delta_color="normal" if current_ratio > 1.5 else "inverse"
            )

        if quick_ratio:
            st.metric(
                "Quick Ratio",
                f"{quick_ratio:.2f}",
                delta="Healthy" if quick_ratio > 1.0 else "Low",
                delta_color="normal" if quick_ratio > 1.0 else "inverse"
            )

    with col2:
        st.markdown("#### 📊 Leverage")

        debt_to_equity = fundamentals.get('debt_to_equity')

        if debt_to_equity is not None:
            st.metric(
                "Debt to Equity",
                f"{debt_to_equity:.2f}",
                delta="Low leverage" if debt_to_equity < 1 else "High leverage",
                delta_color="normal" if debt_to_equity < 1 else "inverse"
            )

            # Create gauge for debt to equity
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=debt_to_equity,
                title={'text': "Debt/Equity Ratio"},
                gauge={
                    'axis': {'range': [None, 3]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1.5], 'color': "lightyellow"},
                        {'range': [1.5, 3], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

    # Additional financial metrics
    st.markdown("#### 💰 Cash & Dividends")
    col3, col4, col5 = st.columns(3)

    with col3:
        total_cash = fundamentals.get('total_cash')
        if total_cash:
            st.metric("Total Cash", f"${total_cash / 1e9:.2f}B")

    with col4:
        dividend_yield = fundamentals.get('dividend_yield')
        if dividend_yield:
            st.metric("Dividend Yield", f"{dividend_yield * 100:.2f}%")

    with col5:
        beta = fundamentals.get('beta')
        if beta:
            st.metric(
                "Beta (Volatility)",
                f"{beta:.2f}",
                delta="More volatile" if beta > 1 else "Less volatile",
                delta_color="inverse" if beta > 1.5 else "normal"
            )


def _display_ownership_data(insider_data: Dict, institutional_data: Dict):
    """Display insider and institutional ownership data"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👨‍💼 Insider Activity")

        insider_sentiment = insider_data.get('sentiment', 'neutral')
        insider_score = insider_data.get('score', 0)
        transaction_count = insider_data.get('transaction_count', 0)
        buy_volume = insider_data.get('buy_volume', 0)
        sell_volume = insider_data.get('sell_volume', 0)

        # Color-code sentiment
        sentiment_color = {
            'bullish': '🟢',
            'bearish': '🔴',
            'neutral': '🟡'
        }.get(insider_sentiment, '⚪')

        st.markdown(f"**Sentiment:** {sentiment_color} {insider_sentiment.upper()}")
        st.metric("Insider Score", f"{insider_score:.1f}/10")
        st.metric("Recent Transactions (90 days)", transaction_count)

        if transaction_count > 0:
            df_insider = pd.DataFrame({
                'Type': ['Buys', 'Sells'],
                'Volume': [buy_volume, sell_volume]
            })

            fig = px.pie(
                df_insider,
                values='Volume',
                names='Type',
                title="Insider Trading Volume",
                color='Type',
                color_discrete_map={'Buys': 'green', 'Sells': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🏢 Institutional Ownership")

        institutional_score = institutional_data.get('ownership_score', 5)
        major_holders = institutional_data.get('major_holders', {})

        st.metric("Institutional Score", f"{institutional_score:.1f}/10")

        if major_holders:
            st.markdown("**Major Holders:**")
            for holder, percentage in major_holders.items():
                st.write(f"- {holder}: {percentage}%")


def display_fundamentals_comparison(tickers: List[str], fundamentals_data: Dict[str, Dict]):
    """
    Display comparative fundamental analysis for multiple stocks

    Args:
        tickers: List of stock tickers
        fundamentals_data: Dictionary mapping ticker to fundamental data
    """
    try:
        st.subheader("📊 Fundamental Comparison")

        if not fundamentals_data:
            st.warning("No fundamental data available for comparison")
            return

        # Create comparison DataFrame
        comparison_data = []

        for ticker in tickers:
            data = fundamentals_data.get(ticker, {})
            if not data or not data.get('fundamentals'):
                continue

            fundamentals = data['fundamentals']

            comparison_data.append({
                'Ticker': ticker,
                'Fundamental Score': data.get('fundamental_score', 0),
                'P/E Ratio': fundamentals.get('pe_ratio', 0),
                'ROE (%)': fundamentals.get('roe', 0) * 100 if fundamentals.get('roe') else 0,
                'Revenue Growth (%)': fundamentals.get('revenue_growth', 0) * 100 if fundamentals.get('revenue_growth') else 0,
                'Debt/Equity': fundamentals.get('debt_to_equity', 0),
                'Dividend Yield (%)': fundamentals.get('dividend_yield', 0) * 100 if fundamentals.get('dividend_yield') else 0,
                'Sector': fundamentals.get('sector', 'Unknown')
            })

        if not comparison_data:
            st.warning("No valid fundamental data to compare")
            return

        df_comparison = pd.DataFrame(comparison_data)

        # Display comparison table
        st.dataframe(
            df_comparison.style.background_gradient(
                subset=['Fundamental Score', 'ROE (%)', 'Revenue Growth (%)'],
                cmap='RdYlGn'
            ).background_gradient(
                subset=['P/E Ratio', 'Debt/Equity'],
                cmap='RdYlGn_r'
            ),
            use_container_width=True,
            hide_index=True
        )

        # Create comparison charts
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                df_comparison,
                x='Ticker',
                y='Fundamental Score',
                title='Fundamental Score Comparison',
                color='Fundamental Score',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                df_comparison,
                x='P/E Ratio',
                y='ROE (%)',
                size='Fundamental Score',
                color='Ticker',
                title='Valuation vs Profitability',
                hover_data=['Revenue Growth (%)']
            )
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        logger.error(f"Error displaying fundamentals comparison: {e}")
        st.error(f"Error displaying comparison: {e}")


def display_market_context(market_context: Dict):
    """
    Display overall market context and economic indicators

    Args:
        market_context: Dictionary containing market context data
    """
    try:
        st.subheader("🌍 Market Context & Economic Indicators")

        col1, col2, col3 = st.columns(3)

        with col1:
            market_regime = market_context.get('market_regime', 'UNKNOWN')
            regime_emoji = {
                'BULL_LOW_VOL': '🟢📈',
                'BULL_NORMAL_VOL': '🟢📊',
                'BEAR_HIGH_VOL': '🔴📉',
                'BEAR_NORMAL_VOL': '🔴📊',
                'NEUTRAL': '🟡➡️'
            }.get(market_regime, '⚪')

            st.metric("Market Regime", f"{regime_emoji} {market_regime}")

        treasury_rates = market_context.get('treasury_rates', {})

        with col2:
            ten_year_rate = treasury_rates.get('10_year', 0)
            st.metric("10Y Treasury", f"{ten_year_rate:.2f}%")

        with col3:
            three_month_rate = treasury_rates.get('3_month', 0)
            st.metric("3M Treasury", f"{three_month_rate:.2f}%")

        # Display market indices
        market_indices = market_context.get('market_indices', {})

        if market_indices:
            st.markdown("#### 📈 Market Indices")

            indices_data = []
            for name, data in market_indices.items():
                indices_data.append({
                    'Index': name,
                    'Price': data.get('current_price', 0),
                    'Monthly Return (%)': data.get('monthly_return', 0),
                    'Volatility (%)': data.get('volatility', 0)
                })

            df_indices = pd.DataFrame(indices_data)
            st.dataframe(
                df_indices.style.background_gradient(
                    subset=['Monthly Return (%)'],
                    cmap='RdYlGn'
                ),
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        logger.error(f"Error displaying market context: {e}")
        st.error(f"Error displaying market context: {e}")


if __name__ == "__main__":
    print("Fundamental UI Module - Ready for integration with app.py")
    print("Available functions:")
    print("  - display_fundamental_summary(ticker, data)")
    print("  - display_fundamentals_comparison(tickers, data)")
    print("  - display_market_context(market_context)")
