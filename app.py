# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# Custom modules
from utils.data_loader import fetch_nifty50_tickers, fetch_historical_data_parallel as fetch_historical_data
from utils.feature_engineer import (
    create_features, 
    monte_carlo_forecast,
    calculate_risk_score,
    load_processed_data,
    HORIZONS
)
from utils.model import load_models, predict_returns, train_all_models, save_models
from utils.evaluator import StockEvaluator, COLORS

# Configuration
st.set_page_config(
    page_title="AI Stock Advisor Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stMetric {background-color: white; border-radius: 10px; padding: 15px;}
    .stProgress > div > div {background-color: #4CAF50;}
    .st-bb {background-color: white;}
    .st-at {background-color: #4CAF50;}
    footer {visibility: hidden;}
    .risk-high {color: #F44336;}
    .risk-medium {color: #FFC107;}
    .risk-low {color: #4CAF50;}
</style>
""", unsafe_allow_html=True)

def get_user_inputs():
    """Collect user inputs through a form"""
    with st.sidebar:
        with st.form("user_inputs"):
            st.header("⚙️ Investment Parameters")
            
            investment = st.number_input(
                "Investment Amount (₹)", 
                min_value=10000, 
                value=100000, 
                step=10000
            )
            
            duration = st.selectbox(
                "Investment Horizon",
                ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years"]
            )
            
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )
            
            submitted = st.form_submit_button("Generate Recommendations")
            
            if submitted:
                st.session_state['submitted'] = True
                
            st.markdown("---")
            st.markdown("📊 **Advanced Analytics:**")
            st.markdown("- Monte Carlo Simulations")
            st.markdown("- Risk-Adjusted Returns")
            st.markdown("- Feature Importance Analysis")
            st.markdown("🔄 Updated Daily at Market Close")
    
    return investment, duration, risk_tolerance, submitted

@st.cache_data(ttl=24*3600, show_spinner="Loading market data...")
def load_and_process_data():
    """Load and process market data with enhanced error handling"""
    try:
        tickers = fetch_nifty50_tickers()
        if not tickers:
            st.error("Failed to fetch tickers")
            return {}
            
        raw_data = fetch_historical_data(tickers)
        if not raw_data:
            st.error("No historical data retrieved")
            return {}

        featured_data = {}
        progress_bar = st.progress(0)
        total_tickers = len(raw_data)
        
        for i, (ticker, df) in enumerate(raw_data.items()):
            try:
                processed_df = create_features(df)
                if not processed_df.empty:
                    featured_data[ticker] = processed_df
            except Exception as e:
                st.warning(f"Error processing {ticker}: {str(e)}")
            progress_bar.progress((i+1)/total_tickers)
            
        st.info(f"Processed {len(featured_data)}/{total_tickers} tickers successfully")
        return featured_data
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return {}

@st.cache_resource(show_spinner=False)
def handle_models(_featured_data):
    """Handle model loading/training with improved feedback"""
    try:
        models = load_models()
        if models:
            st.success("Loaded pre-trained models")
            return models
            
        with st.spinner("Training models (this may take 5-10 minutes)..."):
            training_result = train_all_models(_featured_data)
            if not training_result.get('models'):
                st.error("Model training failed")
                return {}
                
            save_models(training_result['models'])
            
            # Run comprehensive evaluation
            evaluator = StockEvaluator(training_result['models'], _featured_data)
            evaluator.full_evaluation()
            st.success("Models trained and evaluated successfully")
            
            models = load_models()
            return models
            
    except Exception as e:
        st.error(f"Model handling failed: {str(e)}")
        return {}

def display_metric_card(label, value, delta=None, delta_color="normal"):
    """Custom metric card display with validation"""
    col = st.columns(1)[0]
    with col:
        try:
            progress_value = float(delta.strip("%"))/100 if delta and "%" in delta else 0
        except:
            progress_value = 0
            
        st.metric(
            label=label,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
        st.progress(min(max(progress_value, 0), 1))

def get_risk_class(score):
    """Get CSS class for risk score"""
    if score > 0.7:
        return "risk-high"
    elif score > 0.4:
        return "risk-medium"
    return "risk-low"

def main():
    st.title("📈 AI-Powered Stock Recommendation System Pro")
    st.markdown("**Advanced Investing with Machine Learning & Risk Analysis**")
    
    # Get user inputs
    investment, duration, risk_tolerance, submitted = get_user_inputs()
    
    if not submitted and 'submitted' not in st.session_state:
        st.info("Please configure your investment parameters in the sidebar and click 'Generate Recommendations'")
        return

    # Convert duration to model horizon
    horizon_map = {
        "1 Week": "next_week",
        "1 Month": "next_month",
        "3 Months": "next_quarter",
        "6 Months": "next_quarter", 
        "1 Year": "next_year",
        "3 Years": "next_3_years",
        "5 Years": "next_5_years"
    }
    horizon = horizon_map.get(duration, "next_year")
    
    # Load data with progress
    with st.spinner("Loading and processing market data..."):
        featured_data = load_and_process_data()
        if not featured_data:
            return

    # Handle models (load or train)
    models = handle_models(featured_data)
    if not models:
        return

    # Generate predictions with validation
    with st.spinner("Analyzing market trends with advanced models..."):
        try:
            predictions = predict_returns(models, featured_data, horizon)
            if predictions.empty:
                st.error("No predictions available")
                return
                
            # Add additional metrics
            for ticker in predictions['ticker']:
                if ticker in featured_data:
                    df = featured_data[ticker]
                    predictions.loc[predictions['ticker'] == ticker, 'current_price'] = df['Close'].iloc[-1]
                    predictions.loc[predictions['ticker'] == ticker, '52w_high'] = df['Close'].rolling(252).max().iloc[-1]
                    predictions.loc[predictions['ticker'] == ticker, '52w_low'] = df['Close'].rolling(252).min().iloc[-1]
                
            predictions.dropna(subset=['current_price'], inplace=True)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return

    # Enhanced risk-adjusted scoring
    risk_weights = {
        "Conservative": {'success': 0.4, 'upside': 0.2, 'risk': 0.3, 'liquidity': 0.1},
        "Moderate": {'success': 0.5, 'upside': 0.3, 'risk': 0.15, 'liquidity': 0.05},
        "Aggressive": {'success': 0.6, 'upside': 0.35, 'risk': 0.05, 'liquidity': 0.0}
    }
    
    weights = risk_weights[risk_tolerance]
    predictions['upside_potential'] = (predictions['52w_high'] - predictions['current_price']) / predictions['current_price']
    predictions['downside_risk'] = (predictions['current_price'] - predictions['52w_low']) / predictions['current_price']
    
    predictions['score'] = (
        weights['success'] * predictions['success_prob'] +
        weights['upside'] * predictions['upside_potential'] -
        weights['risk'] * predictions['risk_score'] -
        weights['liquidity'] * predictions['downside_risk']
    )
    
    recommendations = predictions.sort_values('score', ascending=False).head(10)

    # ===== Dashboard Display =====
    st.header("🏆 Top Investment Opportunities")
    
    # Top Recommendations Grid
    cols = st.columns(4)
    for idx, (_, row) in enumerate(recommendations.head(4).iterrows()):
        with cols[idx % 4]:
            delta_color = "normal" if row['success_prob'] > 0.6 else "inverse"
            risk_class = get_risk_class(row['risk_score'])
            
            display_metric_card(
                label=row['ticker'],
                value=f"₹{row['current_price']:,.1f}",
                delta=f"{row['success_prob']:.1%} Success",
                delta_color=delta_color
            )
            
            with st.expander("Advanced Metrics"):
                st.write(f"**52W Range:** ₹{row['52w_low']:,.1f} - ₹{row['52w_high']:,.1f}")
                st.write(f"**Upside Potential:** {row['upside_potential']:.1%}")
                st.write(f"**Downside Risk:** {row['downside_risk']:.1%}")
                st.markdown(f"**Risk Score:** <span class='{risk_class}'>{row['risk_score']:.2f}/1.0</span>", unsafe_allow_html=True)
                st.write(f"**AI Confidence Score:** {row['score']:.2f}")

    # ===== Advanced Analysis Section =====
    st.markdown("---")
    st.header("🔍 Advanced Analytics")
    
    selected_ticker = st.selectbox(
        "Select Stock for Detailed Analysis",
        [t for t in recommendations['ticker'] if t in featured_data],
        key="ticker_select"
    )
    
    if selected_ticker not in featured_data:
        st.warning("Selected ticker data not available")
        return
        
    df = featured_data[selected_ticker]
    
    # Enhanced Price Analysis
    st.subheader(f"{selected_ticker} Technical Analysis")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Price and Moving Averages
    df['Close'].plot(ax=ax1, label='Close', color=COLORS['neutral'])
    for ma in ['5D_MA', '20D_MA', '50D_MA']:
        if ma in df.columns:
            df[ma].plot(ax=ax1, label=ma)
    ax1.set_title("Price Trend with Moving Averages")
    ax1.legend()
    
    # Volume Analysis
    df['Volume'].plot(ax=ax2, color=COLORS['good'])
    ax2.set_title("Trading Volume")
    
    st.pyplot(fig)
    plt.close()

    # Monte Carlo Simulation with Horizon Days
    st.subheader("Monte Carlo Price Projections")
    horizon_days = HORIZONS.get(horizon, 252)  # Default to 1 year if not found
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**{duration} Forecast Statistics**")
        mc_prices = monte_carlo_forecast(df, horizon_days)
        if len(mc_prices) > 0:
            current_price = df['Close'].iloc[-1]
            median_price = np.median(mc_prices)
            p5 = np.percentile(mc_prices, 5)
            p95 = np.percentile(mc_prices, 95)
            
            st.write(f"- Current Price: ₹{current_price:,.1f}")
            st.write(f"- Median Projection: ₹{median_price:,.1f}")
            st.write(f"- 5th Percentile: ₹{p5:,.1f}")
            st.write(f"- 95th Percentile: ₹{p95:,.1f}")
            st.write(f"- Potential Range: {((p95-p5)/current_price):.1%}")
        else:
            st.warning("Forecast unavailable for this stock")
    
    with col2:
        if len(mc_prices) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(mc_prices, bins=50, kde=True, color=COLORS['neutral'])
            plt.axvline(current_price, color=COLORS['good'], linestyle='--', label='Current Price')
            plt.axvline(median_price, color=COLORS['bad'], linestyle='--', label='Median Projection')
            plt.title(f"{duration} Price Probability Distribution")
            plt.xlabel("Price (₹)")
            plt.legend()
            st.pyplot(fig)
            plt.close()

    # ===== Portfolio Builder with Risk Analysis =====
    st.markdown("---")
    st.header("💰 Advanced Portfolio Builder")
    
    valid_tickers = [t for t in recommendations['ticker'] if t in featured_data]
    selected_stocks = st.multiselect(
        "Select Stocks for Portfolio",
        valid_tickers,
        default=valid_tickers[:3] if len(valid_tickers) >=3 else valid_tickers,
        key="portfolio_select"
    )
    
    if selected_stocks:
        # Calculate smart default allocations based on scores
        scores = recommendations.set_index('ticker').loc[selected_stocks]['score']
        defaults = (scores / scores.sum() * 100).round().astype(int)
        
        allocations = {}
        cols = st.columns(len(selected_stocks))
        
        for idx, ticker in enumerate(selected_stocks):
            with cols[idx]:
                alloc = st.slider(
                    f"{ticker} Allocation",
                    0, 100, 
                    defaults[ticker],
                    key=f"alloc_{ticker}"
                )
                allocations[ticker] = alloc
        
        # Normalize allocations
        total = sum(allocations.values())
        if total == 0:
            st.warning("Total allocation cannot be 0%")
            return
            
        allocations = {k: v/total for k, v in allocations.items()}
        
        # Calculate portfolio metrics
        portfolio_return = sum(
            allocations[t] * recommendations.set_index('ticker').loc[t]['success_prob']
            for t in allocations
        )
        
        portfolio_risk = sum(
            allocations[t] * recommendations.set_index('ticker').loc[t]['risk_score']
            for t in allocations
        )
        
        # Enhanced Portfolio Summary
        st.subheader("Portfolio Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Allocation Breakdown**")
            alloc_df = pd.DataFrame.from_dict(allocations, orient='index', columns=['Allocation'])
            st.dataframe(alloc_df.style.format("{:.1%}"))
        
        with col2:
            st.write("**Performance Metrics**")
            st.metric("Expected Success Probability", f"{portfolio_return:.1%}")
            st.metric("Composite Risk Score", f"{portfolio_risk:.2f}")
            st.metric("Risk-Adjusted Return", f"{(portfolio_return/portfolio_risk):.2f}" if portfolio_risk > 0 else "N/A")
        
        # Advanced Risk-Return Visualization
        st.subheader("Portfolio Optimization")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for t in allocations:
            ax.scatter(
                recommendations.set_index('ticker').loc[t]['risk_score'],
                recommendations.set_index('ticker').loc[t]['success_prob'],
                s=200 * allocations[t],
                label=t,
                alpha=0.7
            )
        
        ax.scatter(
            [portfolio_risk],
            [portfolio_return],
            s=400,
            marker="X",
            c="red",
            label="Your Portfolio"
        )
        
        ax.set_xlabel("Risk Score (Lower is Better)")
        ax.set_ylabel("Success Probability (Higher is Better)")
        ax.set_title("Portfolio Risk-Return Profile")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        plt.close()

if __name__ == "__main__":
    main()
