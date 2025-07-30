# app_enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import gc
import warnings
from datetime import datetime
import logging
import joblib

# Import enhanced modules from the utils package
from utils.data_loader import get_comprehensive_stock_data, DATA_CONFIG
from utils.feature_engineer import engineer_features_enhanced, FEATURE_CONFIG
from utils.model import (
    train_models_enhanced_parallel,
    predict_with_ensemble,
    ENHANCED_MODEL_CONFIG
)
from utils.evaluator import StockEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# Enhanced Streamlit page configuration
st.set_page_config(
    page_title="AI Stock Advisor Pro - Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    /* Add other CSS styles from your provided code here */
</style>
""", unsafe_allow_html=True)


# --- Model Persistence ---
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "enhanced_models.joblib")

def save_models_optimized(models: dict):
    """Saves the trained models using joblib for efficiency."""
    try:
        joblib.dump(models, MODEL_FILE_PATH)
        logging.info(f"Models saved successfully to {MODEL_FILE_PATH}")
    except Exception as e:
        logging.error(f"Error saving models: {e}")

def load_models_optimized() -> dict:
    """Loads models from the specified path."""
    if os.path.exists(MODEL_FILE_PATH):
        try:
            models = joblib.load(MODEL_FILE_PATH)
            logging.info("Models loaded successfully.")
            return models
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            return {}
    return {}

# Enhanced caching system
@st.cache_data(ttl=3600, max_entries=5, show_spinner="🔄 Loading and processing data...")
def load_and_prepare_data(max_tickers: int, historical_period: str, use_db: bool):
    """Loads, processes, and engineers features for stock data."""
    data_cfg = DATA_CONFIG.copy()
    data_cfg['default_period'] = historical_period
    data_cfg['use_database'] = use_db
    
    raw_data = get_comprehensive_stock_data(max_tickers=max_tickers, config=data_cfg)
    if not raw_data:
        st.error("Failed to load raw data.")
        return {}, {}
    
    feature_cfg = FEATURE_CONFIG.copy()
    featured_data = engineer_features_enhanced(raw_data, config=feature_cfg, use_cache=True, parallel=True)
    
    # Filter out empty dataframes to prevent downstream errors
    valid_tickers = [t for t, df in featured_data.items() if not df.empty]
    raw_data = {t: raw_data[t] for t in valid_tickers if t in raw_data}
    featured_data = {t: featured_data[t] for t in valid_tickers}

    return raw_data, featured_data


@st.cache_data(ttl=86400, show_spinner="🤖 Loading or training ML models...")
def load_or_train_models(featured_data, _force_retrain, model_cfg):
    """Loads pre-trained models or retrains them."""
    if not _force_retrain:
        models = load_models_optimized()
        if models:
            st.success("Loaded pre-trained models from cache.")
            return models, {"status": "loaded_from_cache"}

    st.info("Training new models. This might take a few minutes...")
    training_output = train_models_enhanced_parallel(featured_data, config=model_cfg)
    models = training_output.get('models', {})
    summary = training_output.get('training_summary', {})
    
    if models:
        save_models_optimized(models)
        st.success(f"Model training complete. Success Rate: {summary.get('success_rate', 0):.1%}")
    else:
        st.error("Model training failed.")
        
    return models, summary

def create_enhanced_prediction_dashboard(predictions_df):
    """Creates the main prediction dashboard with metrics and top recommendations."""
    if predictions_df.empty:
        st.warning("⚠️ No predictions available for the selected criteria.")
        return

    st.subheader("📊 Enhanced Market Intelligence")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_preds = len(predictions_df)
    high_conf = (predictions_df['ensemble_confidence'] >= 0.8).sum()
    bullish = (predictions_df['predicted_return'] == 1).sum()
    bearish = total_preds - bullish

    col1.metric("Total Predictions", total_preds, f"{high_conf} High Confidence")
    col2.metric("Avg. Success Prob.", f"{predictions_df['success_prob'].mean():.2%}")
    col3.metric("Market Sentiment", f"{bullish} Bullish", f"{bearish} Bearish")
    col4.metric("Avg. Model Confidence", f"{predictions_df['ensemble_confidence'].mean():.2%}")
    col5.metric("Avg. Ensemble Size", f"{predictions_df['models_used'].mean():.1f} Models")
    
    st.subheader("🎯 Top Investment Opportunities")
    predictions_df['composite_score'] = (
        predictions_df['success_prob'] * 0.4 +
        predictions_df['ensemble_confidence'] * 0.3 +
        predictions_df['model_agreement'] * 0.2 +
        (1 - predictions_df['risk_score']) * 0.1
    )
    
    top_buys = predictions_df[predictions_df['predicted_return'] == 1].nlargest(10, 'composite_score')
    
    st.markdown("#### 📈 Top Buy Recommendations")
    for _, row in top_buys.iterrows():
        st.markdown(
            f"**{row['ticker']}** | Prob: **{row['success_prob']:.1%}** | "
            f"Conf: **{row['ensemble_confidence']:.1%}** | Risk: **{row['risk_score']:.2f}**"
        )
        st.progress(row['composite_score'])

def create_advanced_portfolio_optimizer(predictions_df):
    """Creates the portfolio optimizer interface."""
    st.header("💼 Advanced Portfolio Optimization")
    if predictions_df.empty:
        st.warning("Prediction data not available for optimization.")
        return

    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Investment Amount (₹)", 10000, 10000000, 100000, 10000)
    risk_profile = c2.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
    portfolio_size = c3.slider("Portfolio Size", 5, 25, 10)

    if st.button("🎯 Generate Optimal Portfolio", type="primary"):
        buy_signals = predictions_df[predictions_df.predicted_return == 1].copy()
        risk_map = {"Conservative": 0.4, "Moderate": 0.6, "Aggressive": 0.9}
        candidates = buy_signals[buy_signals.risk_score <= risk_map[risk_profile]]
        
        if len(candidates) < portfolio_size:
            st.warning(f"Found only {len(candidates)} stocks matching criteria. Adjusting portfolio size.")
            portfolio_size = len(candidates)
        
        if portfolio_size == 0:
            st.error("No suitable stocks found for the portfolio.")
            return

        candidates['sharpe_proxy'] = (candidates['success_prob'] - 0.5) / (candidates['risk_score'] + 1e-6)
        portfolio = candidates.nlargest(portfolio_size, 'sharpe_proxy')
        
        # Risk Parity Weighting (simplified)
        inv_risk = 1 / (portfolio['risk_score'] + 1e-6)
        portfolio['weight'] = inv_risk / inv_risk.sum()
        portfolio['investment'] = portfolio['weight'] * amount
        
        st.success("Optimal portfolio generated.")
        st.dataframe(portfolio[['ticker', 'weight', 'investment', 'success_prob', 'risk_score']])

        c1, c2 = st.columns(2)
        pie_fig = px.pie(portfolio, names='ticker', values='weight', title='Portfolio Allocation')
        c1.plotly_chart(pie_fig, use_container_width=True)
        scatter_fig = px.scatter(portfolio, x='risk_score', y='success_prob', size='weight', color='ticker', title='Portfolio Holdings Risk/Reward')
        c2.plotly_chart(scatter_fig, use_container_width=True)

def display_model_analytics(models, featured_data, training_summary):
    """Displays training summaries and evaluation metrics."""
    st.header("📊 Advanced Model Analytics")
    if training_summary and training_summary.get('status') != 'loaded_from_cache':
        st.subheader("Training Performance")
        results = pd.DataFrame(training_summary.get('training_results', []))
        if not results.empty:
            c1, c2 = st.columns(2)
            c1.metric("Models Trained", training_summary['successful'])
            c2.metric("Avg. ROC AUC", f"{results['score'].mean():.3f}")
    else:
        st.info("Models were loaded from cache. Run with 'Force Model Retraining' to see a new training summary.")

    st.subheader("Live Model Evaluation")
    with st.spinner("Running deep model evaluation..."):
        evaluator = StockEvaluator(models, featured_data)
        evaluator.full_evaluation()

    if not evaluator.metrics.empty:
        st.write("#### Performance by Horizon (ROC AUC)")
        st.dataframe(evaluator.metrics.groupby('horizon')[['roc_auc', 'f1']].mean())

        st.write("#### Top 5 Performing Models")
        st.dataframe(evaluator.get_top_performers(n=5))

        st.write("#### Top 5 Risk-Adjusted Models")
        st.dataframe(evaluator.get_risk_adjusted_returns(n=5))

def main():
    """Main Streamlit application function."""
    st.markdown('<div class="main-header">🚀 AI Stock Advisor Pro - Enhanced</div>', unsafe_allow_html=True)
    
    st.sidebar.header("⚙️ System Configuration")
    max_tickers = st.sidebar.slider("Max Tickers", 20, 150, 50, 10)
    h_period = st.sidebar.select_slider("History", ["5y", "10y", "15y"], "10y")
    use_db = st.sidebar.checkbox("Use DB Cache", True)
    
    horizon = st.sidebar.selectbox("Horizon", ["next_week", "next_month", "next_quarter", "next_year"], 1)
    m_types = st.sidebar.multiselect("Models", ['xgboost', 'lightgbm', 'catboost'], ['lightgbm', 'xgboost'])
    e_method = st.sidebar.selectbox("Ensemble Method", ["weighted_average", "majority_vote"], 0)
    force_retrain = st.sidebar.checkbox("Force Model Retraining")

    # Load data
    raw_data, featured_data = load_and_prepare_data(max_tickers, h_period, use_db)
    if not featured_data:
        st.error("Data preparation failed. Cannot proceed.")
        return

    # Load/Train models
    model_cfg = ENHANCED_MODEL_CONFIG.copy()
    model_cfg['model_types'] = m_types
    models, training_summary = load_or_train_models(featured_data, force_retrain, model_cfg)
    if not models:
        st.error("Model loading/training failed. Cannot proceed.")
        return

    # Generate Predictions
    predictions_df = predict_with_ensemble(models, featured_data, horizon, m_types, e_method)
    
    # Setup Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "💼 Portfolio Optimizer", "📊 Model Analytics"])
    
    with tab1:
        create_enhanced_prediction_dashboard(predictions_df)
        
    with tab2:
        create_advanced_portfolio_optimizer(predictions_df)
        
    with tab3:
        display_model_analytics(models, featured_data, training_summary)

    st.sidebar.markdown("---")
    st.sidebar.info(f"Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.markdown(
        "<p><em>Disclaimer: For educational purposes only. Not financial advice.</em></p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()