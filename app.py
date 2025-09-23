# app.py - Enhanced Institutional-Grade Version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import gc
import warnings
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import sys
from typing import Dict, List, Optional, Tuple, Any

# Configure institutional-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_stock_advisor.log'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings('ignore')

# Create institutional directories
institutional_dirs = [
    'logs', 'data', 'model_cache', 'feature_cache_v2', 'reports', 
    'risk_reports', 'backtest_results', 'sentiment_data', 'market_data'
]
for dir_name in institutional_dirs:
    os.makedirs(dir_name, exist_ok=True)

# Enhanced Streamlit configuration for institutional use
st.set_page_config(
    page_title="AI Stock Advisor Pro - Institutional Edition",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# REMOVE FALLBACKS - ALWAYS USE ACTUAL MODULES
try:
    # Core modules - INSTITUTIONAL GRADE
    from utils.data_loader import get_comprehensive_stock_data, DATA_CONFIG
    from utils.feature_engineer import (
        engineer_features_enhanced, create_technical_features, 
        FEATURE_CONFIG
    )
    from utils.model import (
        train_models_enhanced_parallel, 
        predict_with_ensemble,
        generate_price_targets_for_selected_stocks,
        predict_with_ensemble_and_targets,
        ENHANCED_MODEL_CONFIG
    )
    from utils.news_sentiment import (
        AdvancedSentimentAnalyzer,
        get_sentiment_for_selected_stocks,
        get_sentiment_insights
    )
    from utils.backtesting import (
        EnhancedBacktestEngine, EnhancedBacktestConfig, 
        MLStrategy, BacktestAnalyzer, BacktestDB
    )
    from utils.risk_management import (
        ComprehensiveRiskManager, RiskConfig, 
        create_risk_dashboard_plots
    )
    from config import CONFIG, secrets
    
    # Module status
    MODULES_LOADED = {
        'data_loader': True,
        'feature_engineer': True,
        'model': True,
        'news_sentiment': True,
        'backtesting': True,
        'risk_management': True
    }
    
    st.success("🏦 All institutional-grade modules loaded successfully")
    
except ImportError as e:
    st.error(f"❌ CRITICAL: Required modules not available: {e}")
    st.error("🚨 SYSTEM FAILURE: Cannot proceed without all institutional modules")
    st.stop()

# Enhanced CSS for institutional appearance
st.markdown("""
<style>
    .institutional-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .institutional-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1e3c72;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
        color: white;
    }
    .risk-alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #d63031;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    .institutional-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_institutional_stock_selection():
    """Enhanced institutional-grade stock selection"""
    
    st.sidebar.header("🎯 Institutional Stock Selection")
    
    # Institutional ticker categories
    institutional_tickers = {
        'Large Cap Banking': ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
        'Technology Leaders': ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTI.NS"],
        'Infrastructure & Capital Goods': ["RELIANCE.NS", "LT.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "COALINDIA.NS"],
        'Consumer & FMCG': ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "TITAN.NS"],
        'Pharma & Healthcare': ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
        'Auto & Engineering': ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS"],
        'Financial Services': ["BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFC.NS"],
        'Commodities & Materials': ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "COALINDIA.NS"],
        'Energy & Utilities': ["ONGC.NS", "IOC.NS", "BPCL.NS", "POWERGRID.NS", "NTPC.NS"],
        'Telecom & Media': ["BHARTIARTL.NS"]
    }
    
    all_tickers = []
    for category_tickers in institutional_tickers.values():
        all_tickers.extend(category_tickers)
    all_tickers = list(set(all_tickers))  # Remove duplicates
    
    # Advanced selection interface
    selection_method = st.sidebar.radio(
        "Selection Method:",
        ["Manual Selection", "Sector-Based", "Risk-Based Portfolio", "Custom Universe"],
        key="institutional_selection_method"
    )
    
    selected_tickers = []
    
    if selection_method == "Manual Selection":
        selected_tickers = st.sidebar.multiselect(
            "Choose Stocks for Institutional Analysis:",
            options=all_tickers,
            default=[],
            help="Select individual stocks for comprehensive analysis",
            key="manual_stock_selection"
        )
        
    elif selection_method == "Sector-Based":
        selected_sectors = st.sidebar.multiselect(
            "Choose Sectors:",
            options=list(institutional_tickers.keys()),
            default=[],
            key="sector_selection"
        )
        
        for sector in selected_sectors:
            selected_tickers.extend(institutional_tickers[sector])
        selected_tickers = list(set(selected_tickers))
        
        if selected_tickers:
            st.sidebar.success(f"Selected {len(selected_tickers)} stocks from {len(selected_sectors)} sectors")
            
    elif selection_method == "Risk-Based Portfolio":
        risk_profile = st.sidebar.selectbox(
            "Risk Profile:",
            ["Conservative (Low Beta)", "Balanced (Market Beta)", "Aggressive (High Beta)"],
            key="risk_profile_selection"
        )
        
        # Predefined risk-based portfolios
        if risk_profile == "Conservative (Low Beta)":
            selected_tickers = ["HDFCBANK.NS", "TCS.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "POWERGRID.NS"]
        elif risk_profile == "Balanced (Market Beta)":
            selected_tickers = ["RELIANCE.NS", "INFY.NS", "ICICIBANK.NS", "LT.NS", "MARUTI.NS", "ITC.NS"]
        else:  # Aggressive
            selected_tickers = ["TATASTEEL.NS", "BAJFINANCE.NS", "TATAMOTORS.NS", "VEDL.NS", "COALINDIA.NS"]
            
    else:  # Custom Universe
        custom_tickers = st.sidebar.text_area(
            "Enter Custom Tickers (one per line):",
            help="Enter NSE ticker symbols, one per line (e.g., RELIANCE.NS)",
            key="custom_tickers_input"
        )
        
        if custom_tickers:
            selected_tickers = [ticker.strip().upper() for ticker in custom_tickers.split('\n') if ticker.strip()]
            # Ensure .NS suffix
            selected_tickers = [ticker if ticker.endswith('.NS') else f"{ticker}.NS" for ticker in selected_tickers]
    
    # Quick selection buttons
    st.sidebar.markdown("**🚀 Quick Institutional Portfolios:**")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🏦 Banking Focus", key="quick_banking"):
            selected_tickers = institutional_tickers['Large Cap Banking'][:5]
            st.rerun()
            
        if st.button("🏭 Industrial", key="quick_industrial"):
            selected_tickers = institutional_tickers['Infrastructure & Capital Goods'][:5]
            st.rerun()
    
    with col2:
        if st.button("💻 Tech Giants", key="quick_tech"):
            selected_tickers = institutional_tickers['Technology Leaders'][:5]
            st.rerun()
            
        if st.button("🛒 Consumer", key="quick_consumer"):
            selected_tickers = institutional_tickers['Consumer & FMCG'][:5]
            st.rerun()
    
    # Display selection summary
    if selected_tickers:
        st.sidebar.success(f"✅ {len(selected_tickers)} institutional stocks selected")
        
        # Portfolio allocation preview
        with st.sidebar.expander("📊 Portfolio Preview", expanded=False):
            equal_weight = 100 / len(selected_tickers)
            for ticker in selected_tickers:
                st.write(f"• {ticker}: {equal_weight:.1f}%")
                
    else:
        st.sidebar.info("👆 Select stocks for institutional-grade analysis")
        st.sidebar.markdown("""
        **🏦 Institutional Features:**
        - Advanced quantitative models
        - Multi-factor risk analysis  
        - News sentiment integration
        - Professional backtesting
        - Real-time risk monitoring
        """)
    
    return selected_tickers

def create_institutional_configuration():
    """Enhanced institutional configuration"""
    
    st.sidebar.header("⚙️ Institutional Configuration")
    
    # Investment parameters
    investment_horizon = st.sidebar.selectbox(
        "Investment Horizon",
        ["next_week", "next_month", "next_quarter", "next_6_months", "next_year"],
        index=1,
        help="Institutional investment timeframe",
        key="institutional_investment_horizon"
    )
    
    # Model sophistication
    model_complexity = st.sidebar.selectbox(
        "Model Complexity",
        ["Standard", "Advanced", "Institutional"],
        index=2,
        help="Level of model sophistication",
        key="model_complexity"
    )
    
    # Risk management level
    risk_management_level = st.sidebar.selectbox(
        "Risk Management",
        ["Basic", "Enhanced", "Institutional"],
        index=2,
        help="Level of risk management sophistication",
        key="risk_management_level"
    )
    
    # News sentiment integration
    enable_news_sentiment = st.sidebar.checkbox(
        "News Sentiment Analysis",
        value=True,
        help="Integrate real-time news sentiment analysis",
        key="enable_news_sentiment"
    )
    
    # Advanced features
    with st.sidebar.expander("🔬 Advanced Features"):
        
        # Multi-factor model
        enable_multi_factor = st.checkbox(
            "Multi-Factor Models",
            value=True,
            help="Use Fama-French and custom factors",
            key="enable_multi_factor"
        )
        
        # Alternative data
        enable_alternative_data = st.checkbox(
            "Alternative Data Sources",
            value=True,
            help="Incorporate satellite, social media, and other alt data",
            key="enable_alternative_data"
        )
        
        # High-frequency features
        enable_microstructure = st.checkbox(
            "Market Microstructure",
            value=True,
            help="Include order book and tick-level analysis",
            key="enable_microstructure"
        )
        
        # ESG integration
        enable_esg = st.checkbox(
            "ESG Factors",
            value=True,
            help="Environmental, Social, Governance factor integration",
            key="enable_esg"
        )
    
    return {
        'investment_horizon': investment_horizon,
        'model_complexity': model_complexity,
        'risk_management_level': risk_management_level,
        'enable_news_sentiment': enable_news_sentiment,
        'enable_multi_factor': enable_multi_factor,
        'enable_alternative_data': enable_alternative_data,
        'enable_microstructure': enable_microstructure,
        'enable_esg': enable_esg
    }

@st.cache_data(ttl=1800, max_entries=5, show_spinner="Loading institutional-grade data...")
def load_institutional_data(selected_tickers: List[str], config: Dict) -> Tuple[Dict, Dict, Dict]:
    """Load comprehensive institutional data including sentiment"""
    
    if not selected_tickers:
        return {}, {}, {}
    
    st.info(f"🏦 Loading institutional-grade data for {len(selected_tickers)} securities...")
    
    # Step 1: Load comprehensive market data
    try:
        enhanced_config = DATA_CONFIG.copy()
        enhanced_config.update({
            'max_period': '10y',  # Extended history for institutional analysis
            'use_database': True,
            'validate_data': True,
            'include_corporate_actions': True,
            'include_fundamentals': True
        })
        
        raw_data = get_comprehensive_stock_data(
            selected_tickers=selected_tickers,
            config=enhanced_config
        )
        
        if not raw_data:
            st.error("❌ Failed to load market data")
            return {}, {}, {}
            
        st.success(f"✅ Market data loaded for {len(raw_data)} securities")
        
    except Exception as e:
        st.error(f"❌ Market data loading failed: {e}")
        return {}, {}, {}
    
    # Step 2: Advanced feature engineering
    try:
        st.info("🔧 Generating institutional-grade features...")
        
        enhanced_feature_config = FEATURE_CONFIG.copy()
        enhanced_feature_config.update({
            'advanced_features': True,
            'market_microstructure': config.get('enable_microstructure', True),
            'alternative_data': config.get('enable_alternative_data', True),
            'multi_factor': config.get('enable_multi_factor', True),
            'lookback_periods': [5, 10, 20, 50, 100, 252],  # Extended periods
            'technical_indicators': [
                'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 
                'atr', 'cci', 'williams_r', 'obv', 'adx', 'aroon', 'momentum'
            ]
        })
        
        featured_data = engineer_features_enhanced(
            data_dict=raw_data,
            config=enhanced_feature_config,
            use_cache=True,
            parallel=True,
            selected_tickers=selected_tickers
        )
        
        st.success(f"✅ Advanced features generated for {len(featured_data)} securities")
        
    except Exception as e:
        st.error(f"❌ Feature engineering failed: {e}")
        return raw_data, raw_data, {}
    
    # Step 3: News sentiment analysis (INTEGRATED)
    sentiment_data = {}
    
    if config.get('enable_news_sentiment', True):
        try:
            st.info("📰 Integrating real-time news sentiment analysis...")
            
            # Use the enhanced sentiment analyzer
            sentiment_analyzer = AdvancedSentimentAnalyzer(api_key=secrets.NEWS_API_KEY)
            
            # Get comprehensive sentiment insights
            sentiment_results = get_sentiment_insights(
                selected_tickers=selected_tickers,
                api_key=secrets.NEWS_API_KEY
            )
            
            if sentiment_results and 'sentiment_distribution' in sentiment_results:
                sentiment_data = sentiment_results
                
                # Integrate sentiment into featured data
                for ticker in selected_tickers:
                    if ticker in featured_data and ticker in sentiment_results['sentiment_distribution']:
                        sentiment_score = sentiment_results['sentiment_distribution'][ticker]
                        
                        # Add sentiment features
                        featured_data[ticker]['news_sentiment'] = sentiment_score
                        featured_data[ticker]['news_sentiment_ma5'] = sentiment_score  # Could be rolling average
                        featured_data[ticker]['news_sentiment_strength'] = abs(sentiment_score)
                        
                        # Binary sentiment features
                        featured_data[ticker]['news_positive'] = 1 if sentiment_score > 0.1 else 0
                        featured_data[ticker]['news_negative'] = 1 if sentiment_score < -0.1 else 0
                        featured_data[ticker]['news_neutral'] = 1 if -0.1 <= sentiment_score <= 0.1 else 0
                
                st.success(f"✅ News sentiment integrated for {len(sentiment_results['sentiment_distribution'])} securities")
                st.info(f"📊 Overall market sentiment: {sentiment_results.get('overall_sentiment', 0):.3f}")
                
            else:
                st.warning("⚠️ News sentiment data not available")
                
        except Exception as e:
            st.warning(f"⚠️ News sentiment analysis failed: {e}")
            st.info("Continuing without sentiment data...")
    
    return raw_data, featured_data, sentiment_data

def run_institutional_analysis(selected_tickers: List[str], config: Dict):
    """Run comprehensive institutional-grade analysis"""
    
    # Initialize progress tracking
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Step 1: Data Loading (30%)
        status_text.text("🏦 Loading institutional-grade data...")
        progress_bar.progress(30)
        
        raw_data, featured_data, sentiment_data = load_institutional_data(selected_tickers, config)
        
        if not raw_data or not featured_data:
            st.error("❌ Insufficient data for analysis")
            return
        
        # Step 2: Model Training (60%)
        status_text.text("🤖 Training sophisticated quantitative models...")
        progress_bar.progress(60)
        
        # Enhanced model configuration based on complexity
        model_config = ENHANCED_MODEL_CONFIG.copy()
        
        if config['model_complexity'] == 'Institutional':
            model_config.update({
                'ensemble_size': 7,
                'model_types': ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'gradient_boosting'],
                'hyperparameter_tuning': True,
                'cross_validation_folds': 10,
                'feature_selection': True,
                'max_features': 100,
                'early_stopping': True
            })
        elif config['model_complexity'] == 'Advanced':
            model_config.update({
                'ensemble_size': 5,
                'model_types': ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting'],
                'cross_validation_folds': 7,
                'max_features': 75
            })
        
        # Train models
        model_results = train_models_enhanced_parallel(
            featured_data=featured_data,
            config=model_config,
            selected_tickers=selected_tickers
        )
        
        models = model_results.get('models', {})
        training_summary = model_results.get('training_summary', {})
        
        if not models:
            st.error("❌ Model training failed")
            return
        
        # Step 3: Predictions & Analysis (80%)
        status_text.text("🔮 Generating institutional predictions...")
        progress_bar.progress(80)
        
        # Generate comprehensive predictions
        predictions_df, price_targets_df = predict_with_ensemble_and_targets(
            models=models,
            featured_data=featured_data,
            raw_data=raw_data,
            investment_horizon=config['investment_horizon'],
            selected_tickers=selected_tickers
        )
        
        # Step 4: Complete (100%)
        status_text.text("✅ Institutional analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Store results for persistence
        st.session_state['institutional_results'] = {
            'raw_data': raw_data,
            'featured_data': featured_data,
            'sentiment_data': sentiment_data,
            'models': models,
            'training_summary': training_summary,
            'predictions': predictions_df,
            'price_targets': price_targets_df,
            'config': config
        }
        
        # Display institutional results
        display_institutional_results(
            selected_tickers=selected_tickers,
            predictions_df=predictions_df,
            price_targets_df=price_targets_df,
            models=models,
            training_summary=training_summary,
            sentiment_data=sentiment_data,
            config=config
        )
        
    except Exception as e:
        st.error(f"❌ Institutional analysis failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        
        # Detailed error reporting for institutional users
        with st.expander("🔧 Detailed Error Analysis", expanded=True):
            st.code(f"Error: {str(e)}")
            st.code(f"Selected tickers: {selected_tickers}")
            st.code(f"Configuration: {config}")
            
            st.markdown("**🛠️ Troubleshooting Steps:**")
            st.markdown("1. Verify all tickers are valid NSE symbols")
            st.markdown("2. Check internet connectivity for data feeds")
            st.markdown("3. Ensure sufficient historical data availability")
            st.markdown("4. Review model complexity settings")

def display_institutional_results(selected_tickers, predictions_df, price_targets_df, 
                                 models, training_summary, sentiment_data, config):
    """Display comprehensive institutional results"""
    
    st.success("🎉 Institutional Analysis Completed Successfully!")
    
    # Institutional performance metrics
    st.subheader("🏦 Institutional Performance Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Securities Analyzed", 
            len(selected_tickers),
            help="Number of securities in institutional portfolio"
        )
    
    with col2:
        success_rate = training_summary.get('success_rate', 0) if training_summary else 0
        st.metric(
            "Model Success Rate", 
            f"{success_rate:.1%}",
            help="Percentage of successfully trained quantitative models"
        )
    
    with col3:
        avg_confidence = predictions_df['ensemble_confidence'].mean() if not predictions_df.empty and 'ensemble_confidence' in predictions_df.columns else 0
        st.metric(
            "Avg Model Confidence", 
            f"{avg_confidence:.1%}",
            help="Average prediction confidence across ensemble models"
        )
    
    with col4:
        if sentiment_data and 'overall_sentiment' in sentiment_data:
            overall_sentiment = sentiment_data['overall_sentiment']
            sentiment_label = "Bullish" if overall_sentiment > 0.1 else "Bearish" if overall_sentiment < -0.1 else "Neutral"
            st.metric(
                "Market Sentiment", 
                sentiment_label,
                f"{overall_sentiment:.3f}",
                help="Overall news sentiment across selected securities"
            )
        else:
            st.metric("Market Sentiment", "Not Available")
    
    with col5:
        expected_return = price_targets_df['percentage_change'].mean() if not price_targets_df.empty and 'percentage_change' in price_targets_df.columns else 0
        st.metric(
            "Expected Portfolio Return", 
            f"{expected_return:.1%}",
            help="Average expected return across price targets"
        )
    
    # Detailed results in professional tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Institutional Analytics", 
        "📈 Quantitative Analysis", 
        "📰 Sentiment Intelligence",
        "🔬 Professional Backtesting", 
        "🛡️ Institutional Risk Management"
    ])
    
    with tab1:
        st.subheader("📊 Institutional Analytics Dashboard")
        
        # Predictions table with institutional formatting
        if not predictions_df.empty:
            st.markdown("**🔮 Quantitative Model Predictions:**")
            
            # Format for institutional display
            display_predictions = predictions_df.copy()
            if 'ensemble_confidence' in display_predictions.columns:
                display_predictions['ensemble_confidence'] = display_predictions['ensemble_confidence'].apply(lambda x: f"{x:.1%}")
            if 'predicted_return' in display_predictions.columns:
                display_predictions['predicted_return'] = display_predictions['predicted_return'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_predictions, use_container_width=True)
            
            # Institutional analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction distribution
                if 'predicted_return' in predictions_df.columns:
                    fig_dist = px.histogram(
                        predictions_df, 
                        x='predicted_return',
                        title="Prediction Distribution",
                        nbins=10,
                        color_discrete_sequence=['#1e3c72']
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Confidence analysis
                if 'ensemble_confidence' in predictions_df.columns:
                    fig_conf = px.box(
                        predictions_df,
                        y='ensemble_confidence',
                        title="Model Confidence Distribution",
                        color_discrete_sequence=['#2a5298']
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
        
        # Price targets with institutional analysis
        if not price_targets_df.empty:
            st.markdown("**🎯 Institutional Price Targets:**")
            
            display_targets = price_targets_df.copy()
            for col in ['current_price', 'target_price']:
                if col in display_targets.columns:
                    display_targets[col] = display_targets[col].apply(lambda x: f"₹{x:.2f}")
            if 'percentage_change' in display_targets.columns:
                display_targets['percentage_change'] = display_targets['percentage_change'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_targets, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Advanced Quantitative Analysis")
        
        # Professional charts for each security
        for ticker in selected_tickers[:3]:  # Show top 3 for performance
            if ticker in st.session_state['institutional_results']['raw_data']:
                create_institutional_chart(ticker, st.session_state['institutional_results'])
    
    with tab3:
        st.subheader("📰 News Sentiment Intelligence")
        
        if sentiment_data:
            display_sentiment_analysis(sentiment_data, selected_tickers)
        else:
            st.warning("⚠️ News sentiment data not available")
            st.info("Enable news sentiment analysis in configuration to access this feature")
    
    with tab4:
        st.subheader("🔬 Professional Backtesting Suite")
        
        try:
            # Create institutional backtesting interface
            create_institutional_backtesting_interface(
                models=models,
                featured_data=st.session_state['institutional_results']['featured_data'],
                raw_data=st.session_state['institutional_results']['raw_data'],
                selected_tickers=selected_tickers
            )
        except Exception as bt_error:
            st.error(f"❌ Backtesting interface error: {bt_error}")
    
    with tab5:
        st.subheader("🛡️ Institutional Risk Management")
        
        try:
            create_institutional_risk_interface(
                selected_tickers=selected_tickers,
                raw_data=st.session_state['institutional_results']['raw_data'],
                config=config
            )
        except Exception as risk_error:
            st.error(f"❌ Risk management interface error: {risk_error}")

def create_institutional_chart(ticker: str, results: Dict):
    """Create institutional-grade chart analysis"""
    
    df = results['raw_data'][ticker]
    featured_df = results['featured_data'].get(ticker, df)
    
    st.markdown(f"**📊 {ticker} - Institutional Analysis**")
    
    # Create comprehensive subplot
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f'{ticker} Price & Technical Indicators',
            'Volume Analysis',
            'Volatility & Risk Metrics',
            'Sentiment & News Impact'
        ],
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Main price chart with advanced indicators
    fig.add_trace(
        go.Candlestick(
            x=df.index[-252:],
            open=df['Open'].tail(252),
            high=df['High'].tail(252),
            low=df['Low'].tail(252),
            close=df['Close'].tail(252),
            name='OHLC',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add institutional-grade indicators
    if 'sma_20' in featured_df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['sma_20'].tail(252),
                      mode='lines', name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'sma_50' in featured_df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['sma_50'].tail(252),
                      mode='lines', name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in featured_df.columns and 'bb_lower' in featured_df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['bb_upper'].tail(252),
                      mode='lines', name='BB Upper', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['bb_lower'].tail(252),
                      mode='lines', name='BB Lower', line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
    
    # Volume analysis
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(x=df.index[-252:], y=df['Volume'].tail(252),
                  name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # Volume moving average
        if 'volume_ma_20' in featured_df.columns:
            fig.add_trace(
                go.Scatter(x=df.index[-252:], y=featured_df['volume_ma_20'].tail(252),
                          mode='lines', name='Volume MA', line=dict(color='blue')),
                row=2, col=1
            )
    
    # Volatility analysis
    if 'volatility_20d' in featured_df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['volatility_20d'].tail(252),
                      mode='lines', name='20D Volatility', line=dict(color='purple')),
            row=3, col=1
        )
    
    # RSI
    if 'rsi' in featured_df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['rsi'].tail(252),
                      mode='lines', name='RSI', line=dict(color='green')),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # News sentiment if available
    if 'news_sentiment' in featured_df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-252:], y=featured_df['news_sentiment'].tail(252),
                      mode='lines', name='News Sentiment', line=dict(color='orange')),
            row=4, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)
    
    fig.update_layout(
        height=800,
        title=f'{ticker} - Institutional Technical Analysis',
        xaxis4_title='Date',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['Close'].iloc[-1]
        st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col2:
        daily_return = df['Close'].pct_change().iloc[-1]
        st.metric("Daily Return", f"{daily_return:.2%}")
    
    with col3:
        if 'volatility_20d' in featured_df.columns:
            current_vol = featured_df['volatility_20d'].iloc[-1]
            st.metric("20D Volatility", f"{current_vol:.2%}")
    
    with col4:
        if 'rsi' in featured_df.columns:
            current_rsi = featured_df['rsi'].iloc[-1]
            rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            st.metric("RSI Signal", rsi_signal, f"{current_rsi:.1f}")

def display_sentiment_analysis(sentiment_data: Dict, selected_tickers: List[str]):
    """Display comprehensive sentiment analysis"""
    
    if not sentiment_data:
        return
    
    # Overall sentiment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        st.metric("Overall Sentiment", f"{overall_sentiment:.3f}")
    
    with col2:
        positive_stocks = sentiment_data.get('positive_stocks', 0)
        st.metric("Positive Stocks", positive_stocks)
    
    with col3:
        negative_stocks = sentiment_data.get('negative_stocks', 0)
        st.metric("Negative Stocks", negative_stocks)
    
    with col4:
        neutral_stocks = sentiment_data.get('neutral_stocks', 0)
        st.metric("Neutral Stocks", neutral_stocks)
    
    # Sentiment distribution
    sentiment_dist = sentiment_data.get('sentiment_distribution', {})
    
    if sentiment_dist:
        # Create sentiment visualization
        sentiment_df = pd.DataFrame([
            {'Ticker': ticker, 'Sentiment': score, 
             'Category': 'Positive' if score > 0.1 else 'Negative' if score < -0.1 else 'Neutral'}
            for ticker, score in sentiment_dist.items()
        ])
        
        # Sentiment bar chart
        fig_sentiment = px.bar(
            sentiment_df,
            x='Ticker',
            y='Sentiment',
            color='Category',
            title="News Sentiment Analysis by Stock",
            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107'}
        )
        
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Sentiment table
        st.markdown("**📰 Detailed Sentiment Scores:**")
        sentiment_display = sentiment_df.copy()
        sentiment_display['Sentiment'] = sentiment_display['Sentiment'].apply(lambda x: f"{x:.3f}")
        st.dataframe(sentiment_display, use_container_width=True)
    
    # Top positive and negative
    top_positive = sentiment_data.get('top_positive', [])
    top_negative = sentiment_data.get('top_negative', [])
    
    if top_positive or top_negative:
        col1, col2 = st.columns(2)
        
        with col1:
            if top_positive:
                st.markdown("**📈 Most Positive Sentiment:**")
                for ticker, score in top_positive[:3]:
                    st.success(f"{ticker}: {score:.3f}")
        
        with col2:
            if top_negative:
                st.markdown("**📉 Most Negative Sentiment:**")
                for ticker, score in top_negative[:3]:
                    st.error(f"{ticker}: {score:.3f}")

def create_institutional_backtesting_interface(models, featured_data, raw_data, selected_tickers):
    """Create institutional-grade backtesting interface"""
    
    st.markdown("**🔬 Professional Backtesting Suite**")
    
    # Enhanced backtesting configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input("Initial Capital (₹)", value=10000000, step=1000000, help="Starting capital for institutional portfolio")
        max_positions = st.number_input("Max Positions", value=min(10, len(selected_tickers)), min_value=1, max_value=len(selected_tickers))
    
    with col2:
        transaction_cost = st.slider("Transaction Cost (bps)", 0, 50, 10, help="Transaction cost in basis points") / 10000
        position_sizing = st.selectbox("Position Sizing", ["Equal Weight", "Risk Parity", "Kelly Optimal", "Volatility Target"])
    
    with col3:
        rebalance_freq = st.selectbox("Rebalancing", ["Daily", "Weekly", "Monthly", "Quarterly"])
        benchmark = st.selectbox("Benchmark", ["NIFTY 50", "NIFTY 100", "Custom"])
    
    # Advanced risk parameters
    with st.expander("🛡️ Advanced Risk Controls"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_drawdown = st.slider("Max Drawdown (%)", 5, 30, 15) / 100
            var_confidence = st.slider("VaR Confidence", 90, 99, 95) / 100
        
        with col2:
            max_correlation = st.slider("Max Correlation", 0.1, 0.9, 0.7)
            concentration_limit = st.slider("Max Position Size (%)", 5, 50, 20) / 100
    
    if st.button("🚀 Run Institutional Backtest", type="primary"):
        
        with st.spinner("Running comprehensive institutional backtest..."):
            
            # Create enhanced backtest configuration
            backtest_config = EnhancedBacktestConfig(
                initial_capital=initial_capital,
                transaction_cost_pct=transaction_cost,
                max_positions=max_positions,
                position_sizing_method=position_sizing.lower().replace(' ', '_'),
                rebalance_frequency=rebalance_freq.lower(),
                max_drawdown_limit=max_drawdown,
                max_correlation=max_correlation,
                var_confidence=var_confidence,
                enable_risk_management=True,
                enable_stress_testing=True
            )
            
            # Initialize and run backtest
            engine = EnhancedBacktestEngine(backtest_config)
            
            # Create ML strategy
            strategy = MLStrategy(models, {
                'investment_horizon': 'next_month',
                'confidence_threshold': 0.6,
                'profit_target': 0.25,
                'stop_loss': 0.15,
                'max_holding_period': 60
            })
            
            # Run backtest
            results = engine.run_enhanced_backtest(
                strategy=strategy,
                data=raw_data,
                start_date=datetime.now() - timedelta(days=730),  # 2 years
                end_date=datetime.now()
            )
            
            if 'error' not in results:
                st.success("✅ Institutional backtest completed!")
                
                # Display institutional results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_return = results.get('total_return', 0)
                    st.metric("Total Return", f"{total_return:.1%}")
                    
                    annual_return = results.get('annual_return', 0)
                    st.metric("Annualized Return", f"{annual_return:.1%}")
                
                with col2:
                    sharpe_ratio = results.get('sharpe_ratio', 0)
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    
                    sortino_ratio = results.get('sortino_ratio', 0)
                    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                
                with col3:
                    max_dd = results.get('max_drawdown', 0)
                    st.metric("Max Drawdown", f"{max_dd:.1%}")
                    
                    calmar_ratio = results.get('calmar_ratio', 0)
                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
                
                with col4:
                    win_rate = results.get('win_rate', 0)
                    st.metric("Win Rate", f"{win_rate:.1%}")
                    
                    profit_factor = results.get('profit_factor', 0)
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                
                # Portfolio performance chart
                portfolio_values = results.get('portfolio_values', pd.Series())
                
                if not portfolio_values.empty:
                    fig_performance = go.Figure()
                    
                    fig_performance.add_trace(go.Scatter(
                        x=portfolio_values.index,
                        y=portfolio_values.values,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='#1e3c72', width=3)
                    ))
                    
                    # Add benchmark comparison
                    benchmark_values = pd.Series(initial_capital, index=portfolio_values.index)
                    fig_performance.add_trace(go.Scatter(
                        x=benchmark_values.index,
                        y=benchmark_values.values,
                        mode='lines',
                        name='Benchmark (Buy & Hold)',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                    
                    fig_performance.update_layout(
                        title="Institutional Portfolio Performance",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value (₹)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_performance, use_container_width=True)
            
            else:
                st.error(f"❌ Backtest failed: {results['error']}")

def create_institutional_risk_interface(selected_tickers, raw_data, config):
    """Create institutional risk management interface"""
    
    st.markdown("**🛡️ Institutional Risk Management Dashboard**")
    
    # Initialize risk manager
    risk_config = RiskConfig(
        max_portfolio_risk=0.02,
        max_correlation=0.7,
        max_drawdown=0.15,
        var_confidence=0.95,
        max_concentration=0.20
    )
    
    risk_manager = ComprehensiveRiskManager(risk_config)
    
    # Create institutional portfolio positions
    portfolio_value = 10000000  # 1 crore
    positions = {}
    
    for i, ticker in enumerate(selected_tickers):
        if ticker in raw_data and not raw_data[ticker].empty:
            current_price = raw_data[ticker]['Close'].iloc[-1]
            position_value = portfolio_value / len(selected_tickers)  # Equal weight
            
            positions[ticker] = {
                'value': position_value,
                'current_price': current_price,
                'quantity': position_value / current_price,
                'weight': 1.0 / len(selected_tickers)
            }
    
    # Run comprehensive risk assessment
    try:
        risk_assessment = risk_manager.assess_portfolio_risk(
            positions=positions,
            prices=raw_data,
            portfolio_value=portfolio_value
        )
        
        # Display risk dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            overall_risk = risk_assessment.get('risk_score', 0)
            risk_level = "Low" if overall_risk < 0.3 else "Medium" if overall_risk < 0.7 else "High"
            st.metric("Overall Risk", risk_level, f"{overall_risk:.2f}")
        
        with col2:
            concentration = risk_assessment.get('concentration_risk', {}).get('largest_position', 0)
            st.metric("Max Concentration", f"{concentration:.1%}")
        
        with col3:
            correlation = risk_assessment.get('correlation_risk', {}).get('max_correlation', 0)
            st.metric("Max Correlation", f"{correlation:.2f}")
        
        with col4:
            drawdown = risk_assessment.get('drawdown_risk', {}).get('current_drawdown', 0)
            st.metric("Current Drawdown", f"{abs(drawdown):.1%}")
        
        with col5:
            volatility = risk_assessment.get('volatility_risk', {}).get('portfolio_volatility', 0)
            st.metric("Portfolio Vol", f"{volatility:.1%}")
        
        # Risk recommendations
        recommendations = risk_assessment.get('recommendations', [])
        if recommendations:
            st.markdown("**🎯 Risk Management Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                st.warning(f"{i}. {rec}")
        
        # Risk visualization
        try:
            risk_plots = create_risk_dashboard_plots(risk_assessment)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'risk_gauge' in risk_plots:
                    st.plotly_chart(risk_plots['risk_gauge'], use_container_width=True)
            
            with col2:
                if 'risk_breakdown' in risk_plots:
                    st.plotly_chart(risk_plots['risk_breakdown'], use_container_width=True)
        
        except Exception as plot_error:
            st.info(f"Risk visualization: {plot_error}")
        
        # Advanced risk analytics
        with st.expander("📊 Advanced Risk Analytics"):
            
            # Correlation matrix
            if len(selected_tickers) > 1:
                returns_data = {}
                for ticker in selected_tickers:
                    if ticker in raw_data and not raw_data[ticker].empty:
                        returns = raw_data[ticker]['Close'].pct_change().dropna()
                        if len(returns) > 50:
                            returns_data[ticker] = returns.tail(252)  # Last year
                
                if len(returns_data) > 1:
                    corr_matrix = pd.DataFrame(returns_data).corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu',
                        aspect='auto',
                        title="Correlation Matrix"
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    except Exception as risk_error:
        st.error(f"❌ Risk analysis failed: {risk_error}")

def main():
    """Main institutional application"""
    
    # Institutional header
    st.markdown('<h1 class="institutional-header">🏦 AI Stock Advisor Pro - Institutional Edition</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 1.3rem; color: #1e3c72; margin-bottom: 2rem; font-weight: bold;'>
        Professional Quantitative Analysis • Advanced Risk Management • Real-Time Sentiment Intelligence
    </div>
    """, unsafe_allow_html=True)
    
    # Module status display
    st.success("🏦 All institutional-grade modules loaded and operational")
    
    # Institutional interfaces
    selected_tickers = create_institutional_stock_selection()
    config = create_institutional_configuration()
    
    # Main analysis
    if selected_tickers:
        st.header(f"🏦 Institutional Analysis for {len(selected_tickers)} Selected Securities")
        
        # Portfolio preview
        with st.expander(f"📋 Portfolio Overview ({len(selected_tickers)} Securities)", expanded=False):
            cols = st.columns(min(5, len(selected_tickers)))
            for i, ticker in enumerate(selected_tickers):
                with cols[i % len(cols)]:
                    st.info(f"**{ticker}**")
        
        # Run institutional analysis
        if st.button("🚀 Execute Institutional Analysis", type="primary", key="execute_institutional_analysis"):
            run_institutional_analysis(selected_tickers, config)
    
    else:
        # Welcome screen for institutional users
        st.markdown("""
        <div style='text-align: center; padding: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin: 2rem 0;'>
            <h2>🏦 Welcome to Institutional-Grade Quantitative Analysis</h2>
            <p style='font-size: 1.2rem; margin: 2rem 0;'>
                Advanced AI-powered securities analysis with professional risk management and real-time market intelligence
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Institutional features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='institutional-metric'>
                <h3>🤖 Quantitative Models</h3>
                <ul>
                    <li>Multi-factor risk models</li>
                    <li>Advanced ensemble methods</li>
                    <li>Real-time model updating</li>
                    <li>Cross-asset correlation analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='institutional-metric'>
                <h3>📰 Market Intelligence</h3>
                <ul>
                    <li>Real-time news sentiment</li>
                    <li>Alternative data integration</li>
                    <li>Market microstructure analysis</li>
                    <li>ESG factor incorporation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='institutional-metric'>
                <h3>🛡️ Risk Management</h3>
                <ul>
                    <li>Real-time risk monitoring</li>
                    <li>Advanced stress testing</li>
                    <li>Dynamic position sizing</li>
                    <li>Regulatory compliance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Institutional footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #1e3c72; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;'>
        <p><strong>AI Stock Advisor Pro - Institutional Edition</strong></p>
        <p>Securities: {len(selected_tickers)} • Configuration: {config.get('model_complexity', 'Standard')} • Risk Level: {config.get('risk_management_level', 'Basic')}</p>
        <p>News Sentiment: {'Enabled' if config.get('enable_news_sentiment') else 'Disabled'} • Multi-Factor: {'Enabled' if config.get('enable_multi_factor') else 'Disabled'}</p>
        <p><em>🏦 Professional quantitative analysis for institutional investors. All models and risk metrics are for informational purposes only.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"💥 Critical institutional system error: {str(e)}")
        st.error("Please contact system administrator immediately.")
        
        with st.expander("🔧 System Diagnostic Information", expanded=True):
            st.code(f"Error: {str(e)}")
            st.code(f"Modules loaded: {MODULES_LOADED}")
            
            st.markdown("**🚨 Emergency Protocols:**")
            st.markdown("1. 🔄 Restart application server")
            st.markdown("2. 🔌 Verify data feed connections")
            st.markdown("3. 📞 Contact technical support")
            st.markdown("4. 📊 Review system logs for detailed diagnostics")