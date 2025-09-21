# utils/backtest_integration.py - Enhanced with Comprehensive Risk Management
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import enhanced backtesting and risk management
from utils.backtesting import (
    EnhancedBacktestEngine, EnhancedBacktestConfig, MLStrategy, 
    BacktestAnalyzer, BacktestDB, PerformanceMetrics
)

from utils.risk_management import (
    ComprehensiveRiskManager, RiskConfig, CorrelationAnalyzer, 
    DrawdownTracker, PositionSizer, StressTester
)

def create_enhanced_backtesting_tab(models, featured_data, raw_data, selected_tickers):
    """Create enhanced backtesting interface with comprehensive risk management"""
    
    st.header("🔬 Enhanced Backtesting & Risk-Managed Strategy Validation")
    
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
            min_value=0.01, 
            max_value=0.5, 
            value=0.1,
            step=0.01,
            help="Cost per trade as percentage",
            key="bt_transaction_cost"
        ) / 100
        
        slippage = st.slider(
            "Slippage (%)", 
            min_value=0.01, 
            max_value=0.2, 
            value=0.05,
            step=0.01,
            help="Market impact as percentage",
            key="bt_slippage"
        ) / 100
    
    with col2:
        max_positions = st.slider(
            "Max Positions", 
            min_value=5, 
            max_value=25, 
            value=10,
            help="Maximum concurrent positions",
            key="bt_max_positions"
        )
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["weekly", "monthly", "quarterly"],
            index=1,
            help="How often to rebalance portfolio",
            key="bt_rebalance_freq"
        )
        
        position_sizing = st.selectbox(
            "Position Sizing Method",
            ["risk_parity", "kelly_criterion", "equal_weight", "erc"],
            index=0,
            help="Method for determining position sizes",
            key="bt_position_sizing"
        )
    
    with col3:
        max_drawdown_limit = st.slider(
            "Max Drawdown Limit (%)", 
            min_value=5, 
            max_value=30, 
            value=15,
            help="Stop trading if drawdown exceeds this",
            key="bt_max_drawdown_limit"
        ) / 100
        
        max_correlation = st.slider(
            "Max Position Correlation", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.7,
            step=0.05,
            help="Maximum allowed correlation between positions",
            key="bt_max_correlation"
        )
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)", 
            min_value=3, 
            max_value=10, 
            value=6,
            help="Risk-free rate for Sharpe calculation",
            key="bt_risk_free_rate"
        ) / 100
    
    # Enhanced Risk Management Settings
    with st.expander("🛡️ Advanced Risk Management Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_monitoring = st.checkbox(
                "Enable Risk Monitoring", 
                True, 
                help="Monitor risk metrics during backtest",
                key="bt_risk_monitoring"
            )
            correlation_monitoring = st.checkbox(
                "Correlation Monitoring", 
                True, 
                help="Monitor position correlations",
                key="bt_correlation_monitoring"
            )
            stress_testing = st.checkbox(
                "Periodic Stress Testing", 
                True, 
                help="Run stress tests during backtest",
                key="bt_stress_testing"
            )
        
        with col2:
            var_confidence = st.slider(
                "VaR Confidence Level", 
                min_value=0.90, 
                max_value=0.99, 
                value=0.95, 
                step=0.01,
                help="Value at Risk confidence level",
                key="bt_var_confidence"
            )
            kelly_cap = st.slider(
                "Kelly Fraction Cap", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.25, 
                step=0.05,
                help="Maximum Kelly fraction to use",
                key="bt_kelly_cap"
            )
            stress_test_frequency = st.number_input(
                "Stress Test Frequency (days)", 
                min_value=1, 
                max_value=30, 
                value=5,
                help="How often to run stress tests",
                key="bt_stress_frequency"
            )
        
        with col3:
            risk_budget_limit = st.slider(
                "Risk Budget Limit", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.2, 
                step=0.05,
                help="Maximum risk budget utilization",
                key="bt_risk_budget_limit"
            )
            rebalance_threshold = st.slider(
                "Rebalance Threshold", 
                min_value=0.01, 
                max_value=0.2, 
                value=0.05, 
                step=0.01,
                help="Drift threshold for rebalancing",
                key="bt_rebalance_threshold"
            )
    
    # Strategy Settings
    with st.expander("🎯 Strategy Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_signal_strength = st.slider(
                "Minimum Signal Strength", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.3,
                step=0.05,
                help="Minimum signal strength to trade",
                key="bt_min_signal_strength"
            )
            
            profit_target = st.slider(
                "Profit Target (%)", 
                min_value=5, 
                max_value=50, 
                value=20,
                help="Take profit at this percentage",
                key="bt_profit_target"
            ) / 100
            
            stop_loss = st.slider(
                "Stop Loss (%)", 
                min_value=5, 
                max_value=30, 
                value=10,
                help="Stop loss at this percentage",
                key="bt_stop_loss"
            ) / 100
        
        with col2:
            max_holding_days = st.number_input(
                "Max Holding Days", 
                min_value=5, 
                max_value=365, 
                value=60,
                help="Maximum days to hold a position",
                key="bt_max_holding_days"
            )
            
            risk_based_exits = st.checkbox(
                "Risk-based Exit Signals", 
                True,
                help="Use risk metrics for exit decisions",
                key="bt_risk_based_exits"
            )
            
            volatility_exits = st.checkbox(
                "Volatility-based Exits", 
                True,
                help="Exit on high volatility periods",
                key="bt_volatility_exits"
            )
            
            correlation_exits = st.checkbox(
                "Correlation-based Exits", 
                True,
                help="Exit when correlation exceeds threshold",
                key="bt_correlation_exits"
            )
    
    # Backtest Period Selection
    st.subheader("📅 Backtest Period")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            help="Backtest start date",
            key="bt_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=30),
            help="Backtest end date",
            key="bt_end_date"
        )
    
    with col3:
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ["next_week", "next_month", "next_quarter"],
            index=1,
            help="Model prediction horizon",
            key="bt_investment_horizon"
        )
    
    # Validate backtest period
    if raw_data:
        # Get date range from data
        all_dates = []
        for ticker_data in raw_data.values():
            if not ticker_data.empty:
                all_dates.extend(ticker_data.index.tolist())
        
        if all_dates:
            data_start = min(all_dates)
            data_end = max(all_dates)
            total_days = (end_date - start_date).days
            
            st.info(f"📊 Data available: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")
            
            if total_days < 30:
                st.warning("⚠️ Short backtest period may not provide reliable results. Consider extending to at least 3 months.")
            
            st.info(f"ℹ️ Backtest period: {total_days} days ({total_days/365:.1f} years)")
        
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
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage,
            max_positions=max_positions,
            rebalance_frequency=rebalance_freq,
            position_sizing_method=position_sizing,
            max_portfolio_drawdown=max_drawdown_limit,
            max_position_correlation=max_correlation,
            risk_free_rate=risk_free_rate,
            var_confidence_level=var_confidence,
            kelly_fraction_cap=kelly_cap,
            stress_test_frequency=stress_test_frequency,
            risk_budget_limit=risk_budget_limit,
            rebalance_threshold=rebalance_threshold,
            risk_monitoring_enabled=risk_monitoring,
            correlation_monitoring=correlation_monitoring,
            stress_testing_enabled=stress_testing
        )
        
        # Enhanced ML strategy with risk management
        class EnhancedMLStrategyWithRisk(MLStrategy):
            def __init__(self, models, featured_data, horizon, enhanced_params):
                super().__init__(models, featured_data, horizon)
                self.enhanced_params = enhanced_params
                
            def generate_signals(self, data, current_date):
                """Generate enhanced signals with risk considerations"""
                base_signals = super().generate_signals(data, current_date)
                
                # Apply minimum signal strength filter
                min_strength = self.enhanced_params.get('min_signal_strength', 0.3)
                filtered_signals = {
                    ticker: signal for ticker, signal in base_signals.items()
                    if abs(signal) >= min_strength
                }
                
                return filtered_signals
            
            def get_exit_signal(self, ticker, entry_date, current_date, entry_price, current_price, current_return):
                """Enhanced exit logic with risk-based exits"""
                
                # Get base exit signal
                should_exit, exit_reason = super().get_exit_signal(
                    ticker, entry_date, current_date, entry_price, current_price, current_return
                )
                
                if should_exit:
                    return should_exit, exit_reason
                
                # Enhanced exit conditions
                profit_target = self.enhanced_params.get('profit_target', 0.2)
                stop_loss = self.enhanced_params.get('stop_loss', 0.1)
                max_holding_days = self.enhanced_params.get('max_holding_days', 60)
                
                holding_days = (current_date - entry_date).days
                
                # Enhanced profit target
                if current_return > profit_target:
                    return True, "enhanced_profit_target"
                
                # Enhanced stop loss
                if current_return < -stop_loss:
                    return True, "enhanced_stop_loss"
                
                # Enhanced max holding period
                if holding_days > max_holding_days:
                    return True, "enhanced_max_holding"
                
                # Risk-based exits
                if self.enhanced_params.get('risk_based_exits', True):
                    # Example: Exit if individual position risk is too high
                    if abs(current_return) > 0.15:  # 15% move triggers risk review
                        return True, "risk_based_exit"
                
                return False, "hold"
        
        enhanced_params = {
            'min_signal_strength': min_signal_strength,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'max_holding_days': max_holding_days,
            'risk_based_exits': risk_based_exits,
            'volatility_exits': volatility_exits,
            'correlation_exits': correlation_exits
        }
        
        strategy = EnhancedMLStrategyWithRisk(
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
                        data=raw_data,
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
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    display_enhanced_backtest_results(results, config, enhanced_params)
                    
                    st.success(f"🎉 Enhanced backtest completed and saved with ID: {backtest_id}")
                    
                except Exception as e:
                    st.error(f"❌ Enhanced backtest execution failed: {str(e)}")
                    st.exception(e)
    
    # Historical backtests section
    create_enhanced_historical_backtests_section()

def display_enhanced_backtest_results(results, config, enhanced_params=None):
    """Display enhanced backtest results with risk management metrics"""
    
    st.header("📈 Enhanced Backtest Results with Risk Management")
    
    # Extract results
    portfolio_history = results.get('portfolio_history', [])
    enhanced_trades = results.get('trades', [])
    risk_events = results.get('risk_events', [])
    enhanced_metrics = results.get('metrics', {})
    risk_analysis = results.get('risk_analysis', {})
    
    if not portfolio_history:
        st.error("No backtest results to display")
        return
    
    # Create portfolio value DataFrame
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df.set_index('date', inplace=True)
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = enhanced_metrics.get('total_return', 0)
        st.metric("Total Return", f"{total_return:.2%}")
    
    with col2:
        sharpe_ratio = enhanced_metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
    
    with col3:
        max_drawdown = enhanced_metrics.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    
    with col4:
        total_trades = len(enhanced_trades)
        st.metric("Total Trades", total_trades)
    
    # Enhanced portfolio value chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add drawdown periods
    if 'drawdown' in portfolio_df.columns:
        drawdown_periods = portfolio_df[portfolio_df['drawdown'] < -0.05]  # 5% drawdown threshold
        if not drawdown_periods.empty:
            fig.add_trace(go.Scatter(
                x=drawdown_periods.index,
                y=drawdown_periods['portfolio_value'],
                mode='markers',
                name='Drawdown Periods',
                marker=dict(color='red', size=4, opacity=0.7)
            ))
    
    fig.update_layout(
        title="Enhanced Portfolio Performance with Risk Events",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics visualization
    if risk_analysis:
        st.subheader("🛡️ Risk Management Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_events_count = len(risk_events)
            st.metric("Risk Events", risk_events_count)
        
        with col2:
            drawdown_violations = risk_analysis.get('drawdown_violations', 0)
            st.metric("Drawdown Violations", drawdown_violations)
        
        with col3:
            correlation_violations = risk_analysis.get('correlation_violations', 0)
            st.metric("Correlation Violations", correlation_violations)
    
    # Trading analysis
    if enhanced_trades:
        st.subheader("📊 Trading Analysis")
        
        # Create trades DataFrame
        trades_data = []
        for trade in enhanced_trades:
            trades_data.append({
                'Ticker': trade.ticker,
                'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                'Entry Price': f"₹{trade.entry_price:.2f}",
                'Exit Price': f"₹{trade.exit_price:.2f}",
                'Return': f"{trade.return_pct:.2%}",
                'P&L': f"₹{trade.net_pnl:,.0f}",
                'Holding Days': trade.holding_period,
                'Exit Reason': trade.exit_signal
            })
        
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, use_container_width=True)
        
        # Trading statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            win_rate = sum(1 for t in enhanced_trades if t.net_pnl > 0) / len(enhanced_trades)
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        with col2:
            avg_return = np.mean([t.return_pct for t in enhanced_trades])
            st.metric("Avg Return", f"{avg_return:.2%}")
        
        with col3:
            avg_holding = np.mean([t.holding_period for t in enhanced_trades])
            st.metric("Avg Holding Days", f"{avg_holding:.1f}")
        
        with col4:
            total_pnl = sum(t.net_pnl for t in enhanced_trades)
            st.metric("Total P&L", f"₹{total_pnl:,.0f}")
    
    # Performance summary
    st.subheader("📋 Performance Summary")
    
    st.markdown(f"""
    **Enhanced Performance Metrics**:
    - Total Return: {enhanced_metrics.get('total_return', 0):.2%}
    - Annualized Return: {enhanced_metrics.get('annual_return', 0):.2%}
    - Volatility: {enhanced_metrics.get('volatility', 0):.2%}
    - Sharpe Ratio: {enhanced_metrics.get('sharpe_ratio', 0):.3f}
    - Maximum Drawdown: {enhanced_metrics.get('max_drawdown', 0):.2%}
    
    **Risk Management Summary**:
    - Risk Events: {len(risk_events)}
    - Drawdown Violations: {risk_analysis.get('drawdown_violations', 0)}
    - Correlation Violations: {risk_analysis.get('correlation_violations', 0)}
    - Risk-Adjusted Trades: {risk_analysis.get('risk_adjusted_trades', 0)}
    
    **Trading Summary**:
    - Total Trades: {len(enhanced_trades)}
    - Average Holding Period: {np.mean([t.holding_period for t in enhanced_trades]) if enhanced_trades else 0:.1f} days
    - Win Rate: {(sum(1 for t in enhanced_trades if t.net_pnl > 0) / len(enhanced_trades) if enhanced_trades else 0):.1%}
    """)
    
    # Configuration summary
    st.markdown("### ⚙️ Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Backtest Parameters**:")
        st.markdown(f"- Initial Capital: ₹{config.initial_capital:,}")
        st.markdown(f"- Transaction Cost: {config.transaction_cost_pct:.3%}")
        st.markdown(f"- Max Positions: {config.max_positions}")
        st.markdown(f"- Rebalancing: {config.rebalance_frequency}")
        st.markdown(f"- Position Sizing: {config.position_sizing_method}")
    
    with col2:
        st.markdown("**Risk Management Parameters**:")
        st.markdown(f"- Max Drawdown: {config.max_portfolio_drawdown:.1%}")
        st.markdown(f"- Max Correlation: {config.max_position_correlation:.2f}")
        st.markdown(f"- VaR Confidence: {config.var_confidence_level:.1%}")
        st.markdown(f"- Kelly Cap: {config.kelly_fraction_cap:.1%}")
        st.markdown(f"- Stress Test Frequency: {config.stress_test_frequency} days")
    
    # Downloadable detailed report
    if st.button("📥 Generate Downloadable Report", key="bt_download_report"):
        detailed_report = generate_detailed_text_report(results, config, enhanced_params or {})
        st.download_button(
            label="Download Detailed Report",
            data=detailed_report,
            file_name=f"enhanced_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            key="bt_download_button"
        )

def create_enhanced_historical_backtests_section():
    """Create enhanced historical backtests section"""
    
    st.subheader("📊 Historical Enhanced Backtests")
    
    try:
        db = BacktestDB()
        historical_backtests = db.list_backtests()
        
        if not historical_backtests.empty:
            # Enhanced display with filtering
            col1, col2 = st.columns(2)
            
            with col1:
                min_return = st.slider(
                    "Min Total Return (%)", 
                    min_value=-50, 
                    max_value=100, 
                    value=-50,
                    key="hist_min_return"
                )
                max_drawdown_filter = st.slider(
                    "Max Drawdown Filter (%)", 
                    min_value=0, 
                    max_value=50, 
                    value=50,
                    key="hist_max_drawdown"
                )
            
            with col2:
                min_sharpe = st.slider(
                    "Min Sharpe Ratio", 
                    min_value=-2.0, 
                    max_value=5.0, 
                    value=-2.0, 
                    step=0.1,
                    key="hist_min_sharpe"
                )
                min_trades = st.number_input(
                    "Min Trades", 
                    min_value=0, 
                    max_value=1000, 
                    value=0,
                    key="hist_min_trades"
                )
            
            # Apply filters
            filtered_backtests = historical_backtests[
                (historical_backtests['total_return'] >= min_return/100) &
                (historical_backtests['max_drawdown'] >= -max_drawdown_filter/100) &
                (historical_backtests['sharpe_ratio'] >= min_sharpe) &
                (historical_backtests['total_trades'] >= min_trades)
            ]
            
            if not filtered_backtests.empty:
                # Display enhanced table
                display_cols = ['name', 'initial_capital', 'final_value', 'total_return', 
                              'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate', 'created_at']
                
                formatted_df = filtered_backtests[display_cols].copy()
                formatted_df['total_return'] = formatted_df['total_return'].apply(lambda x: f"{x:.2%}")
                formatted_df['sharpe_ratio'] = formatted_df['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
                formatted_df['max_drawdown'] = formatted_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
                formatted_df['win_rate'] = formatted_df['win_rate'].apply(lambda x: f"{x:.2%}")
                formatted_df['initial_capital'] = formatted_df['initial_capital'].apply(lambda x: f"₹{x:,.0f}")
                formatted_df['final_value'] = formatted_df['final_value'].apply(lambda x: f"₹{x:,.0f}")
                
                st.dataframe(formatted_df, use_container_width=True)
                
                # Enhanced comparison functionality
                selected_backtests = st.multiselect(
                    "Select backtests to compare:",
                    options=filtered_backtests['id'].tolist(),
                    format_func=lambda x: f"ID {x}: {filtered_backtests[filtered_backtests['id']==x]['name'].iloc[0]}",
                    key="hist_compare_select"
                )
                
                if selected_backtests and st.button("🔍 Compare Selected Enhanced Backtests", key="hist_compare_button"):
                    compare_enhanced_backtests(selected_backtests, db)
            
            else:
                st.info("No backtests match the current filters.")
        
        else:
            st.info("No historical enhanced backtests found. Run your first enhanced backtest above!")
    
    except Exception as e:
        st.error(f"Error loading historical backtests: {e}")

def compare_enhanced_backtests(backtest_ids, db):
    """Compare multiple enhanced backtests"""
    
    st.subheader("📊 Enhanced Backtest Comparison")
    
    try:
        comparison_data = []
        
        for backtest_id in backtest_ids:
            backtest_results = db.get_backtest(backtest_id)
            if backtest_results:
                metrics = backtest_results.get('metrics', {})
                comparison_data.append({
                    'ID': backtest_id,
                    'Name': backtest_results.get('name', f'Backtest {backtest_id}'),
                    'Total Return': metrics.get('total_return', 0),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown': metrics.get('max_drawdown', 0),
                    'Total Trades': len(backtest_results.get('trades', [])),
                    'Risk Events': len(backtest_results.get('risk_events', [])),
                    'Final Value': metrics.get('final_value', 0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Return comparison
                fig_return = px.bar(
                    comparison_df, 
                    x='Name', 
                    y='Total Return',
                    title='Total Return Comparison',
                    color='Total Return',
                    color_continuous_scale='RdYlGn'
                )
                fig_return.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_return, use_container_width=True)
            
            with col2:
                # Risk-Return scatter
                fig_scatter = px.scatter(
                    comparison_df,
                    x='Max Drawdown',
                    y='Total Return',
                    size='Total Trades',
                    color='Sharpe Ratio',
                    hover_name='Name',
                    title='Risk-Return Analysis',
                    labels={'Max Drawdown': 'Max Drawdown (%)', 'Total Return': 'Total Return (%)'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Detailed comparison table
            st.markdown("### Detailed Comparison")
            
            formatted_comparison = comparison_df.copy()
            formatted_comparison['Total Return'] = formatted_comparison['Total Return'].apply(lambda x: f"{x:.2%}")
            formatted_comparison['Sharpe Ratio'] = formatted_comparison['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
            formatted_comparison['Max Drawdown'] = formatted_comparison['Max Drawdown'].apply(lambda x: f"{x:.2%}")
            formatted_comparison['Final Value'] = formatted_comparison['Final Value'].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(formatted_comparison, use_container_width=True)
            
        else:
            st.warning("No valid backtest data found for comparison")
    
    except Exception as e:
        st.error(f"Error comparing backtests: {e}")

def generate_detailed_text_report(results, config, enhanced_params):
    """Generate detailed text report for download"""
    
    portfolio_history = results.get('portfolio_history', [])
    enhanced_trades = results.get('trades', [])
    risk_events = results.get('risk_events', [])
    enhanced_metrics = results.get('metrics', {})
    risk_analysis = results.get('risk_analysis', {})
    
    report = f"""
ENHANCED BACKTESTING REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION SUMMARY
{'='*30}
Initial Capital: ₹{config.initial_capital:,}
Transaction Cost: {config.transaction_cost_pct:.3%}
Slippage: {config.slippage_pct:.3%}
Max Positions: {config.max_positions}
Rebalancing Frequency: {config.rebalance_frequency}
Position Sizing Method: {config.position_sizing_method}

RISK MANAGEMENT SETTINGS
{'='*30}
Max Portfolio Drawdown: {config.max_portfolio_drawdown:.1%}
Max Position Correlation: {config.max_position_correlation:.2f}
VaR Confidence Level: {config.var_confidence_level:.1%}
Kelly Fraction Cap: {config.kelly_fraction_cap:.1%}
Stress Test Frequency: {config.stress_test_frequency} days
Risk Budget Limit: {config.risk_budget_limit:.1%}
Rebalance Threshold: {config.rebalance_threshold:.1%}

PERFORMANCE METRICS
{'='*30}
Total Return: {enhanced_metrics.get('total_return', 0):.2%}
Annualized Return: {enhanced_metrics.get('annual_return', 0):.2%}
Volatility: {enhanced_metrics.get('volatility', 0):.2%}
Sharpe Ratio: {enhanced_metrics.get('sharpe_ratio', 0):.3f}
Maximum Drawdown: {enhanced_metrics.get('max_drawdown', 0):.2%}
Calmar Ratio: {enhanced_metrics.get('calmar_ratio', 0):.3f}
Sortino Ratio: {enhanced_metrics.get('sortino_ratio', 0):.3f}

RISK MANAGEMENT SUMMARY
{'='*30}
Total Risk Events: {len(risk_events)}
Drawdown Violations: {risk_analysis.get('drawdown_violations', 0)}
Correlation Violations: {risk_analysis.get('correlation_violations', 0)}
Risk-Adjusted Trades: {risk_analysis.get('risk_adjusted_trades', 0)}
VaR Breaches: {risk_analysis.get('var_breaches', 0)}

TRADING SUMMARY
{'='*30}
Total Trades: {len(enhanced_trades)}
Winning Trades: {sum(1 for t in enhanced_trades if t.net_pnl > 0)}
Losing Trades: {sum(1 for t in enhanced_trades if t.net_pnl <= 0)}
Win Rate: {(sum(1 for t in enhanced_trades if t.net_pnl > 0) / len(enhanced_trades) if enhanced_trades else 0):.1%}
Average Return per Trade: {(np.mean([t.return_pct for t in enhanced_trades]) if enhanced_trades else 0):.2%}
Average Holding Period: {(np.mean([t.holding_period for t in enhanced_trades]) if enhanced_trades else 0):.1f} days
Total P&L: ₹{sum(t.net_pnl for t in enhanced_trades):,.0f}
"""
    
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