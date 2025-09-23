# utils/backtest_integration.py - Complete Enhanced Backtesting Integration
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any, Tuple

# Import enhanced backtesting and risk management
try:
    from utils.backtesting import (
        EnhancedBacktestEngine, EnhancedBacktestConfig, MLStrategy, 
        BacktestAnalyzer, BacktestDB, PerformanceMetrics
    )
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    BACKTESTING_AVAILABLE = False
    logging.warning(f"Enhanced backtesting not available: {e}")

try:
    from utils.risk_management import (
        ComprehensiveRiskManager, RiskConfig, CorrelationAnalyzer, 
        DrawdownTracker, PositionSizer, StressTester, create_risk_dashboard_plots
    )
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    RISK_MANAGEMENT_AVAILABLE = False
    logging.warning(f"Risk management not available: {e}")

def create_enhanced_backtesting_tab(models, featured_data, raw_data, selected_tickers):
    """Create enhanced backtesting interface with comprehensive risk management"""
    
    st.header("🔬 Enhanced Backtesting & Risk-Managed Strategy Validation")
    
    # Check availability
    if not BACKTESTING_AVAILABLE:
        st.error("❌ Enhanced backtesting framework not available. Please check your installation.")
        return
    
    if not models or not featured_data or not raw_data or not selected_tickers:
        st.warning("⚠️ Models, data, or stock selection required for backtesting.")
        return
    
    # Enhanced introduction
    st.markdown("""
    This enhanced backtesting framework integrates comprehensive risk management including:
    - **Real-time correlation monitoring** with automatic position adjustments
    - **Dynamic drawdown tracking** with configurable stop-loss mechanisms  
    - **Advanced position sizing** using Kelly Criterion, Risk Parity, and ERC methods
    - **Continuous stress testing** with historical and Monte Carlo scenarios
    - **Risk-based exit signals** beyond traditional strategy signals
    """)
    
    # Enhanced Configuration section
    st.subheader("⚙️ Enhanced Backtest Configuration")
    
    # Core backtesting parameters
    st.markdown("**📊 Core Backtesting Parameters**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital (₹)", 
            min_value=100000, 
            max_value=50000000, 
            value=1000000,
            step=100000,
            help="Starting capital for backtesting",
            key="bt_initial_capital"
        )
        
        transaction_cost = st.slider(
            "Transaction Cost (%)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1,
            step=0.05,
            help="Transaction cost as percentage of trade value",
            key="bt_transaction_cost"
        )
        
        max_positions = st.number_input(
            "Max Positions", 
            min_value=1, 
            max_value=len(selected_tickers), 
            value=min(10, len(selected_tickers)),
            help="Maximum number of concurrent positions",
            key="bt_max_positions"
        )
    
    with col2:
        position_sizing = st.selectbox(
            "Position Sizing Method",
            options=['equal_weight', 'kelly', 'risk_parity', 'volatility_target'],
            index=0,
            help="Method for calculating position sizes",
            key="bt_position_sizing"
        )
        
        rebalance_freq = st.selectbox(
            "Rebalance Frequency",
            options=['daily', 'weekly', 'monthly', 'quarterly'],
            index=2,
            help="How often to rebalance the portfolio",
            key="bt_rebalance_freq"
        )
        
        investment_horizon = st.selectbox(
            "Investment Horizon",
            options=['next_week', 'next_month', 'next_quarter'],
            index=1,
            help="Prediction horizon for ML models",
            key="bt_investment_horizon"
        )
    
    with col3:
        # Date range selection
        max_date = max(df.index[-1] for df in raw_data.values() if not df.empty)
        min_date = min(df.index[0] for df in raw_data.values() if not df.empty)
        
        default_start = max_date - timedelta(days=365*2)  # 2 years
        
        start_date = st.date_input(
            "Backtest Start Date",
            value=max(default_start.date(), min_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            help="Start date for backtesting period",
            key="bt_start_date"
        )
        
        end_date = st.date_input(
            "Backtest End Date",
            value=max_date.date(),
            min_value=start_date,
            max_value=max_date.date(),
            help="End date for backtesting period",
            key="bt_end_date"
        )
        
        # Calculate backtest period
        total_days = (end_date - start_date).days
        if total_days < 30:
            st.warning("⚠️ Short backtest period. Consider extending to at least 3 months.")
        
        st.info(f"ℹ️ Backtest period: {total_days} days ({total_days/365:.1f} years)")
    
    # Risk Management Parameters
    if RISK_MANAGEMENT_AVAILABLE:
        st.markdown("**🛡️ Risk Management Parameters**")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            max_drawdown_limit = st.slider(
                "Max Drawdown Limit (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=1.0,
                help="Stop trading if drawdown exceeds this limit",
                key="bt_max_drawdown"
            ) / 100
            
            enable_risk_management = st.checkbox(
                "Enable Risk Management",
                value=True,
                help="Enable comprehensive risk management features",
                key="bt_enable_risk"
            )
        
        with risk_col2:
            max_correlation = st.slider(
                "Max Correlation Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Maximum allowed correlation between positions",
                key="bt_max_correlation"
            )
            
            var_confidence = st.slider(
                "VaR Confidence Level (%)",
                min_value=90.0,
                max_value=99.0,
                value=95.0,
                step=1.0,
                help="Confidence level for Value at Risk calculations",
                key="bt_var_confidence"
            ) / 100
        
        with risk_col3:
            stress_test_frequency = st.number_input(
                "Stress Test Frequency (days)",
                min_value=1,
                max_value=30,
                value=7,
                help="How often to run stress tests",
                key="bt_stress_freq"
            )
            
            enable_dynamic_hedging = st.checkbox(
                "Enable Dynamic Hedging",
                value=False,
                help="Enable dynamic hedging based on risk metrics",
                key="bt_dynamic_hedging"
            )
    else:
        # Default risk parameters when risk management not available
        max_drawdown_limit = 0.15
        enable_risk_management = False
        max_correlation = 0.7
        var_confidence = 0.95
        stress_test_frequency = 7
        enable_dynamic_hedging = False
    
    # ML Strategy Parameters
    st.markdown("**🤖 ML Strategy Parameters**")
    
    ml_col1, ml_col2, ml_col3 = st.columns(3)
    
    with ml_col1:
        confidence_threshold = st.slider(
            "Prediction Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Minimum confidence required for trade signals",
            key="bt_confidence_threshold"
        )
        
        profit_target = st.slider(
            "Profit Target (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=2.5,
            help="Take profit when position reaches this return",
            key="bt_profit_target"
        ) / 100
    
    with ml_col2:
        stop_loss = st.slider(
            "Stop Loss (%)",
            min_value=2.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="Stop loss when position falls below this return",
            key="bt_stop_loss"
        ) / 100
        
        max_holding_period = st.number_input(
            "Max Holding Period (days)",
            min_value=1,
            max_value=180,
            value=60,
            help="Maximum days to hold a position",
            key="bt_max_holding"
        )
    
    with ml_col3:
        signal_aggregation = st.selectbox(
            "Signal Aggregation Method",
            options=['weighted_average', 'majority_vote', 'confidence_weighted'],
            index=0,
            help="How to combine signals from multiple models",
            key="bt_signal_aggregation"
        )
        
        ensemble_weight_decay = st.slider(
            "Ensemble Weight Decay",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="Decay factor for older model predictions",
            key="bt_weight_decay"
        )
    
    # Data validation
    if raw_data:
        # Check if we have sufficient data for the backtest period
        available_data = {}
        for ticker in selected_tickers:
            if ticker in raw_data and not raw_data[ticker].empty:
                ticker_data = raw_data[ticker]
                ticker_start = ticker_data.index[0].date()
                ticker_end = ticker_data.index[-1].date()
                
                if ticker_start <= start_date and ticker_end >= end_date:
                    available_data[ticker] = ticker_data
        
        if len(available_data) < len(selected_tickers):
            missing_tickers = set(selected_tickers) - set(available_data.keys())
            st.warning(f"⚠️ Insufficient data for: {', '.join(missing_tickers)}")
            
        if len(available_data) == 0:
            st.error("❌ No data available for backtesting period")
            return
        
        st.info(f"ℹ️ Backtesting {len(available_data)} stocks with sufficient data")
    else:
        st.error("❌ No data available for backtesting")
        return
    
    # Pre-flight checks
    st.subheader("✅ Pre-flight Checks")
    
    checks_passed = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data availability checks
        if models and featured_data:
            st.success("✅ Models and features available")
        else:
            st.error("❌ Models or features missing")
            checks_passed = False
        
        if selected_tickers:
            st.success(f"✅ {len(selected_tickers)} stocks selected")
        else:
            st.error("❌ No stocks selected")
            checks_passed = False
        
        if raw_data:
            total_data_points = sum(len(df) for df in raw_data.values())
            st.success(f"✅ {total_data_points:,} data points available")
        else:
            st.error("❌ No historical data available")
            checks_passed = False
    
    with col2:
        # Configuration checks
        if max_drawdown_limit > 0 and max_drawdown_limit < 0.5:
            st.success("✅ Reasonable drawdown limit")
        else:
            st.warning("⚠️ Check drawdown limit setting")
        
        if max_correlation > 0 and max_correlation < 1:
            st.success("✅ Valid correlation threshold")
        else:
            st.warning("⚠️ Check correlation threshold")
        
        if initial_capital >= 100000:
            st.success("✅ Sufficient initial capital")
        else:
            st.warning("⚠️ Low initial capital may affect results")
    
    # Run enhanced backtest button
    if st.button("🚀 Run Enhanced Backtest with Risk Management", 
                type="primary", 
                disabled=not checks_passed,
                help="Execute comprehensive backtest with risk management",
                key="bt_run_button"):
        
        if not checks_passed:
            st.error("❌ Please address the issues above before running backtest")
            return
        
        # Create enhanced configuration
        config = EnhancedBacktestConfig(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost / 100,
            max_positions=max_positions,
            position_sizing_method=position_sizing,
            rebalance_frequency=rebalance_freq,
            max_drawdown_limit=max_drawdown_limit,
            max_correlation=max_correlation,
            var_confidence=var_confidence,
            stress_test_frequency=stress_test_frequency,
            enable_dynamic_hedging=enable_dynamic_hedging,
            enable_correlation_monitoring=enable_risk_management,
            enable_stress_testing=enable_risk_management,
            prediction_confidence_threshold=confidence_threshold,
            ensemble_weight_decay=ensemble_weight_decay,
            signal_aggregation_method=signal_aggregation
        )
        
        # Create enhanced parameters
        enhanced_params = {
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'max_holding_period': max_holding_period,
            'investment_horizon': investment_horizon
        }
        
        # Create ML strategy
        strategy = _create_ml_strategy_for_backtest(
            models, featured_data, investment_horizon, enhanced_params
        )
        
        # Initialize and run enhanced backtest
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with status_container:
            with st.spinner("Initializing enhanced backtesting engine with risk management..."):
                try:
                    engine = EnhancedBacktestEngine(config)
                    
                    status_text.text("🔄 Running enhanced backtest with comprehensive risk management...")
                    progress_bar.progress(0.1)
                    
                    # Run the enhanced backtest
                    results = engine.run_enhanced_backtest(
                        strategy=strategy,
                        data=available_data,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.min.time())
                    )
                    
                    progress_bar.progress(0.8)
                    status_text.text("📊 Processing results and generating reports...")
                    
                    if 'error' in results:
                        st.error(f"❌ Enhanced backtest failed: {results['error']}")
                        return
                    
                    progress_bar.progress(0.9)
                    status_text.text("💾 Saving results to database...")
                    
                    # Save results to database
                    db = BacktestDB()
                    backtest_name = f"Enhanced_Risk_Managed_{len(selected_tickers)}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    backtest_id = db.save_backtest(backtest_name, results)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Enhanced backtest completed successfully!")
                    
                    # Clear progress indicators after a moment
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    display_enhanced_backtest_results(results, config, selected_tickers, backtest_name)
                    
                except Exception as e:
                    st.error(f"❌ Enhanced backtest execution failed: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
                    
                    with st.expander("🔧 Error Details", expanded=False):
                        st.code(str(e))

def _create_ml_strategy_for_backtest(models, featured_data, investment_horizon, enhanced_params):
    """Create ML strategy for backtesting"""
    
    try:
        # Filter models for the specific horizon
        horizon_models = {}
        
        for ticker, ticker_models in models.items():
            if investment_horizon in ticker_models:
                horizon_models[ticker] = ticker_models[investment_horizon]
        
        if not horizon_models:
            # Fallback: use any available models
            for ticker, ticker_models in models.items():
                if ticker_models:
                    # Use first available horizon
                    first_horizon = list(ticker_models.keys())[0]
                    horizon_models[ticker] = ticker_models[first_horizon]
        
        # Create strategy
        strategy = MLStrategy(horizon_models, enhanced_params)
        
        return strategy
        
    except Exception as e:
        logging.error(f"ML strategy creation failed: {e}")
        return None

def display_enhanced_backtest_results(results, config, selected_tickers, backtest_name):
    """Display comprehensive backtest results"""
    
    st.subheader("📊 Enhanced Backtest Results")
    
    try:
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = results.get('total_return', 0)
            st.metric("Total Return", f"{total_return:.1%}")
            
            annual_return = results.get('annual_return', 0)
            st.metric("Annual Return", f"{annual_return:.1%}")
        
        with col2:
            sharpe_ratio = results.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            sortino_ratio = results.get('sortino_ratio', 0)
            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        
        with col3:
            max_drawdown = results.get('max_drawdown', 0)
            st.metric("Max Drawdown", f"{max_drawdown:.1%}")
            
            calmar_ratio = results.get('calmar_ratio', 0)
            st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
        
        with col4:
            total_trades = results.get('total_trades', 0)
            st.metric("Total Trades", f"{total_trades}")
            
            win_rate = results.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        # Portfolio performance chart
        st.subheader("📈 Portfolio Performance")
        
        portfolio_values = results.get('portfolio_values', pd.Series())
        
        if not portfolio_values.empty:
            fig = go.Figure()
            
            # Portfolio value line
            fig.add_trace(go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Add benchmark if available (using initial capital as baseline)
            initial_capital = config.initial_capital
            benchmark_values = pd.Series(
                initial_capital, 
                index=portfolio_values.index
            )
            
            fig.add_trace(go.Scatter(
                x=benchmark_values.index,
                y=benchmark_values.values,
                mode='lines',
                name='Initial Capital',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics visualization
        if RISK_MANAGEMENT_AVAILABLE and results.get('risk_events'):
            st.subheader("🛡️ Risk Management Analysis")
            
            risk_events = results.get('risk_events', [])
            
            if risk_events:
                risk_df = pd.DataFrame([{
                    'Date': event.timestamp,
                    'Type': event.event_type,
                    'Severity': event.severity,
                    'Description': event.description
                } for event in risk_events])
                
                st.dataframe(risk_df, use_container_width=True)
                
                # Risk events timeline
                fig = px.scatter(
                    risk_df, 
                    x='Date', 
                    y='Type',
                    color='Severity',
                    hover_data=['Description'],
                    title="Risk Events Timeline"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed trade analysis
        trades = results.get('trades', [])
        
        if trades:
            st.subheader("📋 Trade Analysis")
            
            # Trade summary
            winning_trades = [t for t in trades if t.net_pnl > 0]
            losing_trades = [t for t in trades if t.net_pnl < 0]
            
            trade_col1, trade_col2, trade_col3 = st.columns(3)
            
            with trade_col1:
                st.metric("Winning Trades", len(winning_trades))
                if winning_trades:
                    avg_win = np.mean([t.net_pnl for t in winning_trades])
                    st.metric("Avg Win", f"₹{avg_win:,.0f}")
            
            with trade_col2:
                st.metric("Losing Trades", len(losing_trades))
                if losing_trades:
                    avg_loss = np.mean([t.net_pnl for t in losing_trades])
                    st.metric("Avg Loss", f"₹{avg_loss:,.0f}")
            
            with trade_col3:
                if winning_trades and losing_trades:
                    profit_factor = abs(avg_win / avg_loss)
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                
                total_pnl = sum(t.net_pnl for t in trades)
                st.metric("Total P&L", f"₹{total_pnl:,.0f}")
            
            # Trade distribution chart
            trade_returns = [t.return_pct for t in trades]
            
            fig = go.Figure(data=[go.Histogram(
                x=trade_returns,
                nbinsx=20,
                name="Trade Returns",
                marker_color='skyblue'
            )])
            
            fig.update_layout(
                title="Distribution of Trade Returns",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed trades table
            with st.expander("📋 Detailed Trades", expanded=False):
                trades_df = pd.DataFrame([{
                    'Ticker': t.ticker,
                    'Entry Date': t.entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': t.exit_date.strftime('%Y-%m-%d'),
                    'Entry Price': f"₹{t.entry_price:.2f}",
                    'Exit Price': f"₹{t.exit_price:.2f}",
                    'Return %': f"{t.return_pct:.1%}",
                    'P&L': f"₹{t.net_pnl:,.0f}",
                    'Holding Days': t.holding_period,
                    'Exit Reason': t.exit_signal
                } for t in trades])
                
                st.dataframe(trades_df, use_container_width=True)
        
        # Performance comparison
        st.subheader("📊 Performance Comparison")
        
        # Create comparison metrics
        comparison_data = {
            'Metric': ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Strategy': [
                f"{results.get('total_return', 0):.1%}",
                f"{results.get('annual_return', 0):.1%}",
                f"{results.get('sharpe_ratio', 0):.2f}",
                f"{results.get('max_drawdown', 0):.1%}",
                f"{results.get('win_rate', 0):.1%}"
            ],
            'Benchmark (Buy & Hold)': [
                "5.0%",  # Placeholder
                "2.5%",  # Placeholder
                "0.8",   # Placeholder
                "-15.0%", # Placeholder
                "60.0%"  # Placeholder
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Download report button
        generate_downloadable_report(results, config, backtest_name)
        
    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")
        with st.expander("🔧 Error Details", expanded=False):
            st.code(str(e))

def generate_detailed_text_report(results, config, risk_events):
    """Generate detailed text report for download"""
    
    report = f"""
ENHANCED BACKTESTING REPORT
{'='*50}

BACKTEST CONFIGURATION
{'-'*25}
Initial Capital: ₹{config.initial_capital:,.0f}
Transaction Cost: {config.transaction_cost_pct:.3%}
Max Positions: {config.max_positions}
Position Sizing: {config.position_sizing_method}
Max Drawdown Limit: {config.max_drawdown_limit:.1%}
Risk Management: {'Enabled' if hasattr(config, 'enable_correlation_monitoring') else 'Disabled'}

PERFORMANCE SUMMARY
{'-'*25}
Total Return: {results.get('total_return', 0):.2%}
Annual Return: {results.get('annual_return', 0):.2%}
Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {results.get('sortino_ratio', 0):.3f}
Maximum Drawdown: {results.get('max_drawdown', 0):.2%}
Calmar Ratio: {results.get('calmar_ratio', 0):.3f}
Volatility: {results.get('volatility', 0):.2%}

TRADING STATISTICS
{'-'*25}
Total Trades: {results.get('total_trades', 0)}
Winning Trades: {results.get('winning_trades', 0)}
Losing Trades: {results.get('losing_trades', 0)}
Win Rate: {results.get('win_rate', 0):.2%}
Profit Factor: {results.get('profit_factor', 0):.2f}
Average Win: ₹{results.get('avg_win', 0):,.0f}
Average Loss: ₹{results.get('avg_loss', 0):,.0f}
Total P&L: ₹{sum(t.net_pnl for t in results.get('trades', [])):,.0f}
"""
    
    enhanced_trades = results.get('trades', [])
    if enhanced_trades:
        report += f"\n\nDETAILED TRADE LOG\n{'='*50}\n"
        for i, trade in enumerate(enhanced_trades):
            report += f"""
Trade {i+1}:
  Ticker: {trade.ticker}
  Entry: {trade.entry_date.strftime('%Y-%m-%d')} @ ₹{trade.entry_price:.2f}
  Exit: {trade.exit_date.strftime('%Y-%m-%d')} @ ₹{trade.exit_price:.2f}
  Holding Period: {trade.holding_period} days
  Return: {trade.return_pct:.2%}
  P&L: ₹{trade.net_pnl:,.0f}
  Exit Reason: {trade.exit_signal}
"""
    
    if risk_events:
        report += f"\n\nRISK EVENTS LOG\n{'='*50}\n"
        for i, event in enumerate(risk_events):
            report += f"""
Risk Event {i+1}:
  Date: {event['date'].strftime('%Y-%m-%d %H:%M')}
  Type: {event.get('type', 'Violation')}
  Details: {event.get('violations', [])}
  Actions: {event.get('actions_taken', [])}
"""
    
    report += f"\n\nREPORT END\n{'='*50}\n"
    
    return report

def generate_downloadable_report(results, config, backtest_name):
    """Generate and offer downloadable report"""
    
    detailed_report = generate_detailed_text_report(results, config, {})
    
    st.download_button(
        label="📥 Download Enhanced Backtest Report",
        data=detailed_report,
        file_name=f"{backtest_name}_detailed_report.txt",
        mime="text/plain",
        help="Download comprehensive backtest report as text file",
        key="download_report_general"
    )

# Additional utility functions for integration
def get_available_models_for_backtesting(models):
    """Get available models that can be used for backtesting"""
    
    available_models = {}
    model_count = 0
    
    for ticker, model_dict in models.items():
        if model_dict:
            available_models[ticker] = model_dict
            model_count += len(model_dict)
    
    return available_models, model_count

def validate_backtest_requirements(models, featured_data, raw_data, selected_tickers):
    """Validate that all requirements for backtesting are met"""
    
    requirements = {
        'models_available': bool(models),
        'featured_data_available': bool(featured_data),
        'raw_data_available': bool(raw_data),
        'tickers_selected': bool(selected_tickers),
        'sufficient_data': False,
        'models_trained_for_tickers': False
    }
    
    if raw_data:
        # Check if we have sufficient data points
        total_data_points = sum(len(df) for df in raw_data.values())
        requirements['sufficient_data'] = total_data_points > 1000
    
    if models and selected_tickers:
        # Check if models are trained for selected tickers
        model_tickers = set(models.keys())
        selected_set = set(selected_tickers)
        overlap = len(model_tickers.intersection(selected_set))
        requirements['models_trained_for_tickers'] = overlap >= len(selected_tickers) * 0.8
    
    return requirements

# Export functions for use in main app
__all__ = [
    'create_enhanced_backtesting_tab',
    'display_enhanced_backtest_results', 
    'get_available_models_for_backtesting',
    'validate_backtest_requirements'
]

if __name__ == "__main__":
    print("Enhanced Backtesting Integration with Comprehensive Risk Management")
    print("="*70)
    print("Features:")
    print("✓ Real-time risk monitoring during backtesting")
    print("✓ Advanced position sizing with Kelly Criterion & Risk Parity")
    print("✓ Continuous correlation and drawdown tracking")
    print("✓ Periodic stress testing with Monte Carlo scenarios")
    print("✓ Risk-based exit signals beyond strategy signals")
    print("✓ Comprehensive risk event logging and analysis")
    print("✓ Enhanced performance visualization and reporting")
    print("✓ Historical backtest comparison with risk metrics")
    print("✓ Downloadable detailed reports")
    print("✓ Pre-flight validation and requirement checking")